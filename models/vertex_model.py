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
    """
    Vertex-transform MLP (no GNN).

    For each sample we:
      1. Build the body's rotation matrix from the sin/cos features.
      2. Rotate each local vertex into world space, optionally translating
         by `body_position`.
      3. Compute the per-vertex world-frame velocity v_lin + omega x r_world
         (the actual physical velocity at that vertex on the rigid body).
      4. Concatenate, per vertex:
           local_xyz (3) + world_xyz (3) + world_z (1) + vel_at_vertex (3)  = 10
         and FLATTEN across all N vertices into a single fixed-size vector
         of length N * 10. This is concatenated with the original per-sample
         state (13-D) and fed into `WrenchPredictor`.

    There is no message passing and no notion of mesh edges in this model —
    the MLP sees the vertices as a fixed-order positional feature bank, and
    learns whatever spatial structure it needs to from training data.
    """
    # local(3) + world(3) + world_z(1) + vel_at_vertex(3) = 10 per vertex
    VERT_GEOM_DIM = 10

    def __init__(self, state_dim: int = 13, nodes_per_graph: int = None,
                 width: int = 256, num_blocks: int = 5):
        super().__init__()
        self.N = nodes_per_graph
        self.state_dim = state_dim
        flat_vert_dim = self.N * self.VERT_GEOM_DIM
        self.head = WrenchPredictor(
            input_dim=state_dim + flat_vert_dim,
            width=width, num_blocks=num_blocks,
        )

    def forward(self, features, vertex_pos, body_position=None):
        """
        Args:
            features:      [B, state_dim] per-sample state vector
                           (the 13-D ContactDataset feature).
            vertex_pos:    [N, 3] local-frame OBJ vertex positions
                           (shared across the batch — the rest pose).
            body_position: [B, 3] world-frame position of each body's
                           origin (= `self_position` in the dataset, the
                           same point the torque target was computed about).
                           If None, no translation is applied (body at origin).

        Returns:
            WrenchPredictor output dict.
        """
        B = features.size(0)
        N = self.N

        # --- Per-sample rotation matrix from the sin/cos features ---
        R = _rotmat_from_sincos(features)               # [B, 3, 3]

        # --- Place local vertices in world space ---
        vp_local = vertex_pos.unsqueeze(0).expand(B, N, 3)  # [B, N, 3]

        # World rotation: for each b, n: world_n_i = sum_j R[b, i, j] * vp_local[b, n, j]
        vp_world = torch.einsum("bij,bnj->bni", R, vp_local)  # [B, N, 3]

        if body_position is not None:
            vp_world = vp_world + body_position.unsqueeze(1)  # broadcast over N

        # --- Velocity at each vertex: v_at_vertex = v_lin + omega x r_world ---
        # r_world here is the lever from the body origin: rotated, NOT translated
        # (so this lives in the same frame as the torque target's lever).
        v_lin   = features[:, LIN_VEL_SLICE].unsqueeze(1)   # [B, 1, 3]
        omega   = features[:, ANG_VEL_SLICE].unsqueeze(1)   # [B, 1, 3]
        r_world = torch.einsum("bij,bnj->bni", R, vp_local) # [B, N, 3]
        vel_at_vertex = v_lin + torch.cross(
            omega.expand_as(r_world), r_world, dim=-1)      # [B, N, 3]

        # --- Per-vertex geometric features, stacked then flattened ---
        # [B, N, 10] -> [B, N*10]
        vert_feats = torch.cat([
            vp_local,                # local xyz                 (3)
            vp_world,                # world xyz                 (3)
            vp_world[..., 2:3],      # explicit height           (1)
            vel_at_vertex,           # per-vertex world velocity (3)
        ], dim=-1)                                            # [B, N, 10]
        vert_flat = vert_feats.reshape(B, N * self.VERT_GEOM_DIM)

        # Concat original state + flattened vertex bank -> single MLP input.
        x = torch.cat([features, vert_flat], dim=-1)          # [B, state_dim + N*10]

        # Pass the body's linear velocity through explicitly so the Hooke
        # damping term in the head doesn't depend on the layout of `x`.
        return self.head(x, velocity=features[:, LIN_VEL_SLICE])


def make_fast_predictor(num_vertices, input_dim=13, width=512, num_blocks=5):
    model = VertexMLPWrench(
        state_dim=input_dim,
        nodes_per_graph=num_vertices,
        width=width, num_blocks=num_blocks,
    )
    return model
