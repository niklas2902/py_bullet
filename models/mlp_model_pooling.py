import torch
import torch.nn as nn
import torch.nn.functional as F

HEAD_OUT_DIM = 11  # 1 (collision) + 1 (depth) + 3 (force_residual) + 3 (normal) + 3 (lever)
VEL_SLICE = slice(0, 3)  # [vx, vy, vz] are the first 3 features


class ResBlock(nn.Module):
    def __init__(self, width, expansion=4):
        super().__init__()
        self.act = nn.ReLU(inplace=True)

        self.block = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width * expansion),
            self.act,
            nn.LayerNorm(width * expansion),
            nn.Linear(width * expansion, width),
            self.act,
        )

    def forward(self, x):
        return self.act(x + self.block(x))


class WrenchPredictor(nn.Module):
    """
    Predicts net wrench (force + torque) with a head whose force assembly
    matches the physics sim, plus a learned residual to absorb deviations
    the analytical model cannot express.
    """

    def __init__(self, input_dim=13, width=256, num_blocks=5,
                 baseline_k=1e3, learn_k=True,
                 baseline_bounciness=0.5, learn_bounciness=True,
                 baseline_mass=1.0, learn_mass=True,
                 head_hidden=64):
        super().__init__()

        # --- backbone ---
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, width),
            nn.LayerNorm(width),
            nn.GELU(),
        )

        backbone_layers = [nn.LayerNorm(width)]
        for _ in range(num_blocks, 1, -1):
            backbone_layers.append(ResBlock(width))
        self.backbone = nn.Sequential(*backbone_layers)

        self.head_trunk = nn.Sequential(
            nn.Linear(width, head_hidden),
            nn.LayerNorm(head_hidden),
            nn.GELU(),
        )
        self.head_out = nn.Linear(head_hidden, HEAD_OUT_DIM)

        # Physics parameters.
        self.k = nn.Parameter(
            torch.tensor(baseline_k, dtype=torch.float32),
            requires_grad=learn_k,
        )
        self.bounciness = nn.Parameter(
            torch.tensor(baseline_bounciness, dtype=torch.float32),
            requires_grad=learn_bounciness,
        )
        self.mass = nn.Parameter(
            torch.tensor(baseline_mass, dtype=torch.float32),
            requires_grad=learn_mass,
        )

    def forward(self, x, velocity):
        h = self.input_proj(x)
        h = self.backbone(h)

        raw = self.head_out(self.head_trunk(h))
        collision_logit, depth_raw, force_residual, normal_raw, lever = raw.split(
            [1, 1, 3, 3, 3], dim=-1
        )

        k_pos    = F.softplus(self.k)     if self.k.requires_grad else self.k
        mass_pos = F.softplus(self.mass)  if self.mass.requires_grad else self.mass
        mass_pos = torch.clamp(mass_pos, min=1e-6)
        bounciness = torch.sigmoid(self.bounciness)

        c = 2.0 * torch.sqrt(k_pos * mass_pos) * bounciness

        depth = torch.clamp(depth_raw, max=0.0)
        contact_normal = F.normalize(normal_raw, dim=-1, eps=1e-8)

        F_spring = -k_pos * depth
        vel_normal = (velocity * contact_normal).sum(dim=-1, keepdim=True)
        F_damping = -c * vel_normal
        F_mag = F.relu(F_spring + F_damping)

        force_normal = F_mag * contact_normal
        force_vec    = force_normal + force_residual
        torque_vec   = torch.cross(lever, force_vec, dim=-1)

        return {
            "collision_logit": collision_logit,
            "force":  force_vec,
            "torque": torque_vec,
            "aux": {
                "contact_normal": contact_normal,
                "lever":          lever,
                "depth":          depth,
                "k":              k_pos.detach(),
                "c":              c.detach(),
                "mass":           mass_pos.detach(),
                "F_mag":          F_mag,
                "F_spring":       F_spring,
                "F_damping":      F_damping,
                "force_normal":   force_normal,
                "force_residual": force_residual,
            },
        }


# --------------------------------------------------------------------------
# World-space vertex transform
# --------------------------------------------------------------------------
LIN_VEL_SLICE = slice(0, 3)
ANG_VEL_SLICE = slice(3, 6)
ROLL_SC_SLICE  = slice(7, 9)
PITCH_SC_SLICE = slice(9, 11)
YAW_SC_SLICE   = slice(11, 13)


def _rotmat_from_sincos(features: torch.Tensor) -> torch.Tensor:
    """[B, F] feature batch -> [B, 3, 3] rotation matrix.

    Z-Y-X intrinsic: R = Rz(yaw) Ry(pitch) Rx(roll).
    """
    sr, cr = features[:, 7:8],  features[:, 8:9]
    sp, cp = features[:, 9:10], features[:, 10:11]
    sy, cy = features[:, 11:12], features[:, 12:13]

    row0 = torch.cat([cy * cp,             cy * sp * sr - sy * cr,  cy * sp * cr + sy * sr], dim=-1)
    row1 = torch.cat([sy * cp,             sy * sp * sr + cy * cr,  sy * sp * cr - cy * sr], dim=-1)
    row2 = torch.cat([-sp,                 cp * sr,                 cp * cr               ], dim=-1)
    return torch.stack([row0, row1, row2], dim=1)


class VertexMLPWrench(nn.Module):
    VERT_GEOM_DIM = 16

    def __init__(self, state_dim: int = 13,
                 nodes_per_graph: int = None,
                 width: int = 256, num_blocks: int = 5,
                 encoder_dim: int = 256):
        super().__init__()
        self.N = nodes_per_graph
        self.state_dim = state_dim
        self.encoder_dim = encoder_dim

        # Encoder over the flattened all-vertex feature vector. Takes the
        # [B, N*16] block down to a fixed-size embedding before the head.
        flat_vert_dim = self.N * self.VERT_GEOM_DIM
        self.vertex_encoder = nn.Sequential(
            nn.Linear(flat_vert_dim, encoder_dim),
            nn.LayerNorm(encoder_dim),
            nn.GELU(),
            ResBlock(encoder_dim),
        )

        self.head = WrenchPredictor(
            input_dim=state_dim + encoder_dim,
            width=width, num_blocks=num_blocks,
        )

    def forward(self, features, vertex_pos, body_position=None):
        B = features.size(0)
        N = vertex_pos.size(0)

        R = _rotmat_from_sincos(features)

        # All vertices, transformed into world frame.
        vp_local = vertex_pos.unsqueeze(0).expand(B, N, 3)         # [B, N, 3]
        vp_world = torch.einsum("bij,bnj->bni", R, vp_local)

        if body_position is not None:
            vp_world = vp_world + body_position.unsqueeze(1)

        v_lin = features[:, LIN_VEL_SLICE].unsqueeze(1).expand(B, N, 3)
        v_ang = features[:, ANG_VEL_SLICE].unsqueeze(1).expand(B, N, 3)
        vp_world, vp_local, v_lin, v_ang = select_vertices(vp_world, vp_local, v_lin, v_ang, 502)
        N = vp_world.size(1)
        world_z = vp_world[..., 2:3]                               # [B, N, 1]

        vertex_vel = v_lin + torch.cross(v_ang, vp_world, dim=-1)

        vert_feats = torch.cat([
            vp_local, vp_world, world_z, vertex_vel, v_lin, v_ang
        ], dim=-1)                                                 # [B, N, 16]

        # Flatten all vertices, then encode down to a fixed embedding.
        flat_verts = vert_feats.reshape(B, N * self.VERT_GEOM_DIM)  # [B, N*16]
        graph_embed = self.vertex_encoder(flat_verts)              # [B, encoder_dim]

        x = torch.cat([features, graph_embed], dim=-1)
        return self.head(x, velocity=features[:, LIN_VEL_SLICE])


def select_vertices(vertex_pos_world, vp_local, v_lin, v_ang, num_vertices):
    # vertex_pos_world: [B, N, 3] -> [B, num_vertices, 3]
    order = vertex_pos_world[..., 1].argsort(dim=-1)        # [B, N]
    order = order[..., :num_vertices]                        # [B, num_vertices]
    idx = order.unsqueeze(-1).expand(-1, -1, 3)              # [B, num_vertices, 3]
    return torch.gather(vertex_pos_world, 1, idx),torch.gather(vp_local, 1, idx), torch.gather(v_lin, 1, idx), torch.gather(v_ang, 1, idx)

# --- usage in the module ---

def make_fast_predictor(input_dim=13, width=256, num_blocks=5,
                        encoder_dim=256, num_vertices = None):
    model = VertexMLPWrench(
        state_dim=input_dim,
        nodes_per_graph=num_vertices,
        width=width, num_blocks=num_blocks,
        encoder_dim=encoder_dim,
    )
    return model