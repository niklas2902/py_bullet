import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_max_pool
from torch_geometric.nn import SGConv

HEAD_OUT_DIM = 11  # 1 (collision) + 1 (depth) + 3 (force_residual) + 3 (normal) + 3 (lever)

def fast_global_max_pool_fixed(h, batch):
    batch_size = int(batch[-1].item()) + 1  # assumes sorted batch
    return h.view(batch_size, -1, h.size(-1)).amax(dim=1)

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
    Predicts net wrench (force + torque). Velocity is passed in explicitly
    (it's no longer part of `x`) so the damping term can be assembled
    against the real linear velocity.
    """

    def __init__(self, input_dim=9, width=256, num_blocks=5,
                 baseline_k=1e3, learn_k=True,
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
        for _ in range(num_blocks, 1, -1):
            backbone_layers.append(ResBlock(width))
        self.backbone = nn.Sequential(*backbone_layers)

        self.head_trunk = nn.Sequential(
            nn.Linear(width, head_hidden),
            nn.LayerNorm(head_hidden),
            nn.GELU(),
        )
        self.head_out = nn.Linear(head_hidden, HEAD_OUT_DIM)

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
            x:        [B, input_dim] fused features (graph embedding + state)
            velocity: [B, 3] linear velocity in world frame
        """
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
# Feature layout (9-D, set by ContactDataset):
#   0:3   relative position to collider  (rel_pos)
#   3:5   (sin roll,  cos roll)
#   5:7   (sin pitch, cos pitch)
#   7:9   (sin yaw,   cos yaw)
#
# Linear and angular velocity are NOT in x — they are passed as separate
# `v_lin` and `v_ang` arguments to GCNWrench.forward.
#
# Convention: intrinsic Z-Y-X (yaw, then pitch, then roll), i.e.
#     R = Rz(yaw) @ Ry(pitch) @ Rx(roll)

ROLL_SC_SLICE  = slice(3, 5)
PITCH_SC_SLICE = slice(5, 7)
YAW_SC_SLICE   = slice(7, 9)


def _rotmat_from_sincos(features: torch.Tensor) -> torch.Tensor:
    """[B, F] feature batch -> [B, 3, 3] rotation matrix.

    Z-Y-X intrinsic: R = Rz(yaw) Ry(pitch) Rx(roll).
    Reads sin/cos pairs from columns 3:5, 5:7, 7:9.
    """
    assert features.shape[-1] >= 9, (
        f"expected at least 9 features, got {features.shape[-1]}"
    )
    sr, cr = features[:, 3:4], features[:, 4:5]
    sp, cp = features[:, 5:6], features[:, 6:7]
    sy, cy = features[:, 7:8], features[:, 8:9]

    row0 = torch.cat([cy * cp,  cy * sp * sr - sy * cr,  cy * sp * cr + sy * sr], dim=-1)
    row1 = torch.cat([sy * cp,  sy * sp * sr + cy * cr,  sy * sp * cr - cy * sr], dim=-1)
    row2 = torch.cat([-sp,      cp * sr,                 cp * cr               ], dim=-1)
    return torch.stack([row0, row1, row2], dim=1)


class GCNWrench(nn.Module):
    """
    Vertex features (per node, fed to conv1):
        local_xyz             3
        world_xyz             3   (R @ v_local + t)
        world_z               1
        body_lin_vel          3
        body_ang_vel          3
        vertex_world_vel      3   (= v_lin + omega x r_world)
                                   ----
                                   total per-node geometric features = 16,
                                   plus broadcast `in_channels` features
                                   (the original 9-D per-sample state).
    """
    VERT_GEOM_DIM = 16

    def __init__(self, in_channels, hidden_channels, out_channels,
                 nodes_per_graph: int = None):
        super().__init__()
        self.conv = SGConv(
            in_channels=in_channels + self.VERT_GEOM_DIM,
            out_channels=out_channels,
            K=5,
            cached=False,
            add_self_loops=True,
            bias=True,
        )
        self.out_channels = out_channels
        self.N = nodes_per_graph
        # Head sees: v_lin(3) + v_ang(3) + graph_embedding(out_channels)
        self.head = WrenchPredictor(
            input_dim=6 + out_channels,
            width=256, num_blocks=5,
        )

    def forward(self, x, edge_index, vertex_pos, v_lin, v_ang, body_position=None):
        N = self.N
        B = x.size(0) // N

        batch = torch.arange(B, device=x.device).repeat_interleave(N)

        # One state row per sample
        x_per_sample = x.view(B, N, -1)[:, 0, :]                # [B, in_channels]
        R = _rotmat_from_sincos(x_per_sample)                   # [B, 3, 3]

        # Local vertices in world frame
        vp_local = vertex_pos.unsqueeze(0).expand(B, N, 3)      # [B, N, 3]
        vp_world = torch.einsum("bij,bnj->bni", R, vp_local)    # [B, N, 3]
        if body_position is not None:
            vp_world = vp_world + body_position.unsqueeze(1)

        world_z = vp_world[..., 2:3]                            # [B, N, 1]

        # Broadcast body-level velocities to every node
        v_lin_b = v_lin.unsqueeze(1).expand(B, N, 3)            # [B, N, 3]
        v_ang_b = v_ang.unsqueeze(1).expand(B, N, 3)            # [B, N, 3]

        # Per-vertex world velocity: v_lin + omega x r_world
        vertex_vel = v_lin_b + torch.cross(v_ang_b, vp_world, dim=-1)  # [B, N, 3]

        # Stack the 16 geometric features: local(3) + world(3) + world_z(1)
        #                                  + lin(3) + ang(3) + vert_vel(3)
        geom = torch.cat([vp_local, vp_world, world_z,
                        v_lin_b, v_ang_b, vertex_vel], dim=-1)        # [B, N, 16]
        geom = geom.reshape(B * N, self.VERT_GEOM_DIM)                  # [B*N, 16]

        x_full = torch.cat([x, geom], dim=-1)                            # [B*N, in_channels + 16]
        h = self.conv(x_full, edge_index)

        h_graph = fast_global_max_pool_fixed(h, batch)                              # [B, out_channels]
        head_input = torch.cat([v_lin, v_ang, h_graph], dim=-1)
        return self.head(head_input, velocity=v_lin)


def make_fast_predictor(num_vertices, input_dim=9, width=128, output_dim=64, num_blocks=5, baseline_k=1e3):
    model = GCNWrench(in_channels=input_dim, hidden_channels=width, out_channels=output_dim,
                      nodes_per_graph=num_vertices)
    return model