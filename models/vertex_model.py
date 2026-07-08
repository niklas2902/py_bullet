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

    Sim-matching part:
        F_spring  = -k * depth                        (depth <= 0)
        F_damping = -c * (v . n),  c = 2*sqrt(k*m)*b
        F_mag     = max(F_spring + F_damping, 0)      (hard clamp, no softplus)
        F_normal  = F_mag * n                         (along contact normal)

    Residual:
        F_vec     = F_normal + force_residual         (3-vec correction, unconstrained)
        T_vec     = lever x F_vec

    NOTE: `input_dim` here is the *full* feature size going into the MLP,
    which in this variant is the per-sample state (13) PLUS the flattened
    per-vertex world-space features. `forward` still slices the linear
    velocity out of the FIRST 3 entries, which we keep as v_lin convention
    so the damping term is well-defined.
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
        """
        Args:
            x:        [B, input_dim]  full feature vector going into the MLP
                                       (sample state + flattened vertex features).
            velocity: [B, 3]           body linear velocity in world frame, used
                                       for the Hooke damping term. Passed in
                                       explicitly so the slicing convention is
                                       robust to whatever `x` actually contains.
        """
        h = self.input_proj(x)
        h = self.backbone(h)

        raw = self.head_out(self.head_trunk(h))
        collision_logit, depth_raw, force_residual, normal_raw, lever = raw.split(
            [1, 1, 3, 3, 3], dim=-1
        )

        # Match sim: mass clamped to >= 1e-6, k positive.
        k_pos    = F.softplus(self.k)     if self.k.requires_grad else self.k
        mass_pos = F.softplus(self.mass)  if self.mass.requires_grad else self.mass
        mass_pos = torch.clamp(mass_pos, min=1e-6)
        bounciness = torch.sigmoid(self.bounciness)

        # Critical damping
        c = 2.0 * torch.sqrt(k_pos * mass_pos) * bounciness

        # Match sim's `min(penetration, 0.0)`: allow exact zero, clamp positives.
        depth = torch.clamp(depth_raw, max=0.0)

        # Normalize the contact normal (sim does the same defensively).
        contact_normal = F.normalize(normal_raw, dim=-1, eps=1e-8)

        F_spring = -k_pos * depth

        # Damping force along the normal.
        vel_normal = (velocity * contact_normal).sum(dim=-1, keepdim=True)
        F_damping = -c * vel_normal

        # Sim uses a hard clamp at 0 — never sucks objects into surfaces.
        F_mag = F.relu(F_spring + F_damping)

        # Sim-matching normal force, plus learned residual correction.
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
# Feature layout (13-D, set by ContactDataset):
#   0:3   linear velocity     v_lin       (world frame)
#   3:6   angular velocity    omega       (world frame)
#   6     rel_pos_z           (height above plane)
#   7:9   (sin roll,  cos roll)
#   9:11  (sin pitch, cos pitch)
#   11:13 (sin yaw,   cos yaw)
#
# Rotation matrix is built directly from sin/cos pairs — no atan2 round-trip
# needed. Convention: intrinsic Z-Y-X (yaw, then pitch, then roll), i.e.
#     R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
# which is what Blender exports for X-Y-Z Euler angles (the most common
# default). If the sim uses a different order, change `_rotmat_from_sincos`.
LIN_VEL_SLICE = slice(0, 3)
ANG_VEL_SLICE = slice(3, 6)
ROLL_SC_SLICE  = slice(7, 9)    # (sin, cos)
PITCH_SC_SLICE = slice(9, 11)
YAW_SC_SLICE   = slice(11, 13)


def _rotmat_from_sincos(features: torch.Tensor) -> torch.Tensor:
    """[B, F] feature batch -> [B, 3, 3] rotation matrix.

    Z-Y-X intrinsic: R = Rz(yaw) Ry(pitch) Rx(roll).
    """
    sr, cr = features[:, 7:8],  features[:, 8:9]
    sp, cp = features[:, 9:10], features[:, 10:11]
    sy, cy = features[:, 11:12], features[:, 12:13]

    # Each row is a [B, 3] tensor; stack along dim=1 -> [B, 3, 3].
    row0 = torch.cat([cy * cp,             cy * sp * sr - sy * cr,  cy * sp * cr + sy * sr], dim=-1)
    row1 = torch.cat([sy * cp,             sy * sp * sr + cy * cr,  sy * sp * cr - cy * sr], dim=-1)
    row2 = torch.cat([-sp,                 cp * sr,                 cp * cr               ], dim=-1)
    return torch.stack([row0, row1, row2], dim=1)

class VertexMLPWrench(nn.Module):
    VERT_GEOM_DIM = 16

    def __init__(self, state_dim: int = 13,
                 nodes_per_graph: int = None,
                 width: int = 256, num_blocks: int = 5,
                 pooled_n: int = 128,
                 vert_embed_dim: int = 256):
        super().__init__()
        self.N = nodes_per_graph
        self.pooled_n = pooled_n
        self.state_dim = state_dim
        self.vert_embed_dim = vert_embed_dim

        # Shared per-vertex encoder (pure MLP, no neighborhood mixing).
        self.vertex_encoder = nn.Sequential(
            nn.Linear(self.VERT_GEOM_DIM, vert_embed_dim),
            nn.LayerNorm(vert_embed_dim),
            nn.GELU(),
            ResBlock(vert_embed_dim),
        )

        # max + penetration-weighted mean -> 2 * vert_embed_dim graph embedding.
        graph_embed_dim = 2 * vert_embed_dim
        self.head = WrenchPredictor(
            input_dim=state_dim + graph_embed_dim,
            width=width, num_blocks=num_blocks,
        )

    def forward(self, features, vertex_pos, body_position=None):
        B = features.size(0)
        K = self.pooled_n

        R = _rotmat_from_sincos(features)

        vp_local_k, vp_world_k, idx = select_contact_vertices(
            vertex_pos, R, K, body_position=body_position, stable_order=True
        )
        Kc = vp_world_k.size(1)

        world_z = vp_world_k[..., 2:3]                              # [B, Kc, 1]

        v_lin = features[:, LIN_VEL_SLICE].unsqueeze(1).expand(B, Kc, 3)
        v_ang = features[:, ANG_VEL_SLICE].unsqueeze(1).expand(B, Kc, 3)
        vertex_vel = v_lin + torch.cross(v_ang, vp_world_k, dim=-1)

        vert_feats = torch.cat([
            vp_local_k, vp_world_k, world_z, vertex_vel, v_lin, v_ang
        ], dim=-1)                                                  # [B, Kc, 16]

        h = self.vertex_encoder(vert_feats)                         # [B, Kc, E]

        # --- penetration weight: only vertices below the plane contribute ---
        penetration = F.relu(-world_z)                              # [B, Kc, 1]
        pen_sum = penetration.sum(dim=1, keepdim=True).clamp_min(1e-6)
        pen_w = penetration / pen_sum                               # [B, Kc, 1]

        # --- two pooling branches ---
        h_max = h.amax(dim=1)                                       # [B, E]
        h_mean = (h * pen_w).sum(dim=1)                             # [B, E]  (penetration-weighted)

        graph_embed = torch.cat([h_max, h_mean], dim=-1)           # [B, 2E]

        x = torch.cat([features, graph_embed], dim=-1)
        return self.head(x, velocity=features[:, LIN_VEL_SLICE])


def select_contact_vertices(vertex_pos, R, K, body_position=None, stable_order=True):
    """Select the K lowest (most penetrating) vertices and return their geometry.

    Args:
        vertex_pos:    [N, 3] mesh vertices in local frame.
        R:             [B, 3, 3] rotation matrices.
        K:             number of vertices to keep.
        body_position: [B, 3] or None. If given, world position offset.
        stable_order:  if True, sort survivors by buffer index. With pooling this
                       no longer matters for correctness (pool is permutation
                       invariant) but it is cheap and harmless.

    Returns:
        vp_local_k: [B, K, 3]  selected vertices in the local body frame
        vp_world_k: [B, K, 3]  same vertices rotated (+translated) into world
        idx:        [B, K]     selected vertex indices (for debugging/masking)
    """
    B = R.size(0)
    N = vertex_pos.size(0)
    vp_local = vertex_pos.unsqueeze(0).expand(B, N, 3)            # [B, N, 3]

    # Cheap height-only pass: world Z = R[2,:] . vp_local
    z_world = torch.einsum("bj,bnj->bn", R[:, 2, :], vp_local)    # [B, N]
    if body_position is not None:
        z_world = z_world + body_position[:, 2:3]

    # Bottom-K: lowest / most penetrating vertices.
    K = min(K, N)
    _, idx = torch.topk(z_world, K, dim=1, largest=False)        # [B, K]
    if stable_order:
        idx, _ = idx.sort(dim=1)

    # Gather survivors, transform only those.
    gather_idx = idx.unsqueeze(-1).expand(B, K, 3)
    vp_local_k = torch.gather(vp_local, 1, gather_idx)           # [B, K, 3]
    vp_world_k = torch.einsum("bij,bnj->bni", R, vp_local_k)
    if body_position is not None:
        vp_world_k = vp_world_k + body_position.unsqueeze(1)

    return vp_local_k, vp_world_k, idx


# --- usage in the module ---

def make_fast_predictor(input_dim=13, width=256, num_blocks=5,
                        pooled_n=256, vert_embed_dim=256, num_vertices=None):
    model = VertexMLPWrench(
        state_dim=input_dim,
        nodes_per_graph=num_vertices,
        width=width, num_blocks=num_blocks,
        pooled_n=pooled_n, vert_embed_dim=vert_embed_dim,
    )
    return model
