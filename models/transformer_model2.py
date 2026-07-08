import math
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
    def __init__(self, input_dim=13, width=256, num_blocks=5,
                 baseline_k=50.0, learn_k=True,
                 baseline_bounciness=0.5, learn_bounciness=True,
                 baseline_mass=1.0, learn_mass=True,
                 head_hidden=64):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, width),
            nn.LayerNorm(width),
            nn.GELU(),
        )
        backbone_layers = [nn.LayerNorm(width)]
        for _ in range(num_blocks):
            backbone_layers.append(ResBlock(width))
        self.backbone = nn.Sequential(*backbone_layers)
        self.head_trunk = nn.Sequential(
            nn.Linear(width, head_hidden),
            nn.LayerNorm(head_hidden),
            nn.GELU(),
        )
        self.head_out = nn.Linear(head_hidden, HEAD_OUT_DIM)
        # Parameterise k in log-space so softplus(k_raw) ~= baseline_k at init
        # AND has a healthy gradient (softplus saturates for large positive
        # inputs, which is why a raw baseline of 1e3 produced a frozen k).
        k_raw0 = math.log(math.expm1(baseline_k))  # inverse-softplus(baseline_k)
        self.k = nn.Parameter(torch.tensor(k_raw0, dtype=torch.float32), requires_grad=learn_k)
        self.bounciness = nn.Parameter(torch.tensor(baseline_bounciness, dtype=torch.float32), requires_grad=learn_bounciness)
        self.mass = nn.Parameter(torch.tensor(baseline_mass, dtype=torch.float32), requires_grad=learn_mass)

    def forward(self, x, velocity):
        h = self.input_proj(x)
        h = self.backbone(h)
        raw = self.head_out(self.head_trunk(h))
        collision_logit, depth_raw, force_residual, normal_raw, lever = raw.split([1, 1, 3, 3, 3], dim=-1)
        k_pos    = F.softplus(self.k)     if self.k.requires_grad else self.k
        mass_pos = F.softplus(self.mass)  if self.mass.requires_grad else self.mass
        mass_pos = torch.clamp(mass_pos, min=1e-6)
        bounciness = torch.sigmoid(self.bounciness)
        c = 2.0 * torch.sqrt(k_pos * mass_pos) * bounciness
        # Bound penetration depth to a physically plausible range so the
        # spring force cannot explode at init (this is what drove the energy
        # log-ratio to ~+9.8 and made the conservation budget unsatisfiable).
        depth = -F.softplus(-depth_raw)          # in (-inf, 0], smooth, bounded slope
        depth = torch.clamp(depth, min=-0.1)     # max 10 cm penetration
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
                "contact_normal": contact_normal, "lever": lever, "depth": depth,
                "k": k_pos.detach(), "c": c.detach(), "mass": mass_pos.detach(),
                "F_mag": F_mag, "F_spring": F_spring, "F_damping": F_damping,
                "force_normal": force_normal, "force_residual": force_residual,
            },
        }


LIN_VEL_SLICE = slice(0, 3)
ANG_VEL_SLICE = slice(3, 6)
ROLL_SC_SLICE  = slice(7, 9)
PITCH_SC_SLICE = slice(9, 11)
YAW_SC_SLICE   = slice(11, 13)


def _rotmat_from_sincos(features: torch.Tensor) -> torch.Tensor:
    sr, cr = features[:, 7:8],  features[:, 8:9]
    sp, cp = features[:, 9:10], features[:, 10:11]
    sy, cy = features[:, 11:12], features[:, 12:13]
    row0 = torch.cat([cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr], dim=-1)
    row1 = torch.cat([sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr], dim=-1)
    row2 = torch.cat([-sp,     cp * sr,                cp * cr               ], dim=-1)
    return torch.stack([row0, row1, row2], dim=1)


class WrenchTransformer(nn.Module):
    VERT_GEOM_DIM = 16

    def __init__(self, state_dim=13, nodes_per_graph=None, d_model=128, nhead=8,
                 num_layers=5, dim_feedforward=256, dropout=0, max_seq_len=64,
                 width=256, num_blocks=5, predict_sequence=False):
        super().__init__()
        if nodes_per_graph is None:
            raise ValueError("nodes_per_graph must be set (number of selected vertices)")
        self.N = nodes_per_graph
        self.state_dim = state_dim
        self.predict_sequence = predict_sequence
        self.in_dim = state_dim + nodes_per_graph * self.VERT_GEOM_DIM
        self.input_proj = nn.Linear(self.in_dim, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = WrenchPredictor(input_dim=d_model, width=width, num_blocks=num_blocks)

    def _vertex_features(self, R, vp_local, v_lin, v_ang, body_position=None):
        B, N = vp_local.shape[:2]
        r_world = torch.einsum("bij,bnj->bni", R, vp_local)
        vp_world = r_world
        if body_position is not None:
            vp_world = vp_world + body_position.unsqueeze(1)
        world_z = vp_world[..., 2:3]
        omega = v_ang.unsqueeze(1).expand(-1, N, -1)
        vertex_vel = v_lin.unsqueeze(1) + torch.cross(omega, r_world, dim=-1)
        v_lin_b = v_lin.unsqueeze(1).expand(-1, N, -1)
        v_ang_b = v_ang.unsqueeze(1).expand(-1, N, -1)
        return torch.cat([vp_local, vp_world, world_z, vertex_vel, v_lin_b, v_ang_b], dim=-1)

    def forward(self, features, vertex_pos, body_position=None,
                key_padding_mask=None, lengths=None):
        B, T = features.shape[:2]
        if vertex_pos.dim() == 2:
            vertex_pos = vertex_pos.unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1)
        elif vertex_pos.dim() == 3:
            vertex_pos = vertex_pos.unsqueeze(1).expand(-1, T, -1, -1)
        N = vertex_pos.size(2)

        feats = []
        for t in range(T):
            bp_t = body_position[:, t] if body_position is not None else None
            R = _rotmat_from_sincos(features[:, t])
            v_lin = features[:, t, LIN_VEL_SLICE]
            v_ang = features[:, t, ANG_VEL_SLICE]
            vert_feats = self._vertex_features(R, vertex_pos[:, t], v_lin, v_ang, bp_t)
            vert_feats_flat = vert_feats.reshape(B, N * self.VERT_GEOM_DIM)
            feats.append(torch.cat([features[:, t], vert_feats_flat], dim=-1))
        seq = torch.stack(feats, dim=1)

        h = self.input_proj(seq) + self.pos_embedding[:, :T]
        h = self.encoder(h, src_key_padding_mask=key_padding_mask)
        h = self.norm(h)

        if self.predict_sequence:
            vel = features[..., LIN_VEL_SLICE]
            return self.head(h, vel)

        if lengths is not None:
            last_idx = (lengths - 1).clamp(min=0)
        else:
            last_idx = torch.full((B,), T - 1, device=h.device, dtype=torch.long)
        bidx = torch.arange(B, device=h.device)
        h_last   = h[bidx, last_idx]
        vel_last = features[bidx, last_idx][:, LIN_VEL_SLICE]
        return self.head(h_last, vel_last)


def make_fast_predictor(num_vertices, input_dim=13, width=256, num_blocks=5):
    return WrenchTransformer(state_dim=input_dim, nodes_per_graph=num_vertices,
                             width=width, num_blocks=num_blocks)