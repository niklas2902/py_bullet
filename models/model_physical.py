import torch
import torch.nn as nn
import torch.nn.functional as F

HEAD_OUT_DIM = 11  # 1 (collision) + 1 (depth) + 3 (force_residual) + 3 (normal) + 3 (lever)
VEL_SLICE = slice(0, 3)  # [vx, vy, vz] are the first 3 features


class ResidualBlock(nn.Module):
    """SiLU-gated FFN with zero-init output."""
    def __init__(self, dim, expansion=4):
        super().__init__()
        hidden = dim * expansion
        self.norm = nn.LayerNorm(dim)
        self.ff1  = nn.Linear(dim, hidden * 2)
        self.ff2  = nn.Linear(hidden, dim)
        nn.init.zeros_(self.ff2.weight)
        nn.init.zeros_(self.ff2.bias)

    def forward(self, x):
        h = self.norm(x)
        gate, val = self.ff1(h).chunk(2, dim=-1)
        h = self.ff2(F.silu(gate) * val)
        return x + h


class WrenchPredictor(nn.Module):
    """
    Predicts net wrench (force + torque) with a physics-inspired head.

    Semantics (note: 'per-contact' names but aggregated meaning because targets
    are net wrench across all contact points):
      - depth:          effective penetration (<= 0), scales the spring force
      - contact_normal: direction of net force, unit vector
      - force_residual: 3-vector correction (NOT constrained to the normal)
      - lever:          effective lever arm for torque

    Force assembly:
      F_spring  = -k * depth          (positive scalar; depth <= 0)
      F_damping = -c * (v . n)        (scalar, opposes normal-component of velocity)
      F_mag     = softplus(F_spring + F_damping)   (keeps positivity, gradient-safe)
      F_vec     = F_mag * n + force_residual       (residual can fix direction errors)
      T_vec     = lever x F_vec
    """

    def __init__(self, input_dim=12, width=256, num_blocks=4,
                 baseline_k=1e3, learn_k=False,
                 baseline_bounciness=0.5, learn_bounciness=True,
                 baseline_mass=1.0, learn_mass=False,
                 head_hidden=64):
        super().__init__()

        # --- backbone ---
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, width),
            nn.LayerNorm(width),
            nn.GELU(),
        )
        self.backbone = nn.Sequential(
            *[ResidualBlock(width) for _ in range(num_blocks)]
        )

        self.head_trunk = nn.Sequential(
            nn.Linear(width, head_hidden),
            nn.LayerNorm(head_hidden),
            nn.GELU(),
        )
        self.head_out = nn.Linear(head_hidden, HEAD_OUT_DIM)

        # Physics parameters.
        # Defaults: freeze k and mass (you know them from the sim), learn bounciness.
        # Flip learn_k=True only if you want to let the model discover it.
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
    def forward(self, x):
        h = self.input_proj(x)
        h = self.backbone(h)

        raw = self.head_out(self.head_trunk(h))
        collision_logit, depth_raw, force_residual, normal_raw, lever = raw.split(
            [1, 1, 3, 3, 3], dim=-1
        )

        # Physical parameters: ensure positivity (only apply softplus if learnable,
        # otherwise use the frozen value directly).
        k_pos    = F.softplus(self.k)     if self.k.requires_grad else self.k
        mass_pos = F.softplus(self.mass)  if self.mass.requires_grad else self.mass
        # Bounciness in [0, 1] via sigmoid — matches your sim's BOUNCINESS_FACTOR=0.5
        bounciness = torch.sigmoid(self.bounciness)

        # Critical damping coefficient: c = 2 * sqrt(k * m) * bounciness
        c = 2.0 * torch.sqrt(k_pos * mass_pos) * bounciness

        # Effective penetration: non-positive. Softplus applied to -depth_raw,
        # then negated. At init (depth_raw = 0), depth = -ln(2) ≈ -0.693.
        depth = -F.softplus(depth_raw)

        # Bias the contact normal toward +z at init so it's not random.
        # The network can move it anywhere, but the starting point is "upward",
        # which is the correct prior for a cube-on-plane scene.
        normal_biased = normal_raw + torch.tensor([0.0, 0.0, 1.0], device=x.device)
        contact_normal = F.normalize(normal_biased, dim=-1, eps=1e-6)

        # Spring force magnitude (scalar, >= 0 since depth <= 0)
        F_spring = -k_pos * depth

        # Damping force along the normal direction
        velocity = x[:, VEL_SLICE]
        vel_normal = (velocity * contact_normal).sum(dim=-1, keepdim=True)
        F_damping = -c * vel_normal

        # Combined magnitude through softplus: smooth, always positive, gradient
        # flows everywhere (unlike ReLU which dies when F_spring + F_damping < 0).
        F_mag = F.softplus(F_spring + F_damping)

        # Residual is a full 3-vector so it can correct direction errors,
        # not just magnitude along the predicted normal.
        force_vec  = F_mag * contact_normal + force_residual
        torque_vec = torch.cross(lever, force_vec, dim=-1)

        return {
            "collision_logit": collision_logit,
            "force":  force_vec,
            "torque": torque_vec,
            "aux": {
                "contact_normal": contact_normal,
                "lever":          lever,
                "depth":          depth,
                "k":              k_pos.detach() if torch.is_tensor(k_pos) else k_pos,
                "c":              c.detach(),
                "mass":           mass_pos.detach() if torch.is_tensor(mass_pos) else mass_pos,
                "F_mag":          F_mag,
                "F_spring":       F_spring,
                "F_damping":      F_damping,
                "force_residual": force_residual,
            },
        }


def make_fast_predictor(input_dim=12, width=128, num_blocks=4, baseline_k=1e3):
    model = WrenchPredictor(input_dim=input_dim,
                            width=width, num_blocks=num_blocks,
                            baseline_k=baseline_k)
    try:
        compiled = torch.compile(model, mode="default")
        print("Model successfully compiled for optimised performance.")
        return compiled
    except Exception as e:
        print(f"torch.compile failed: {e}. Returning standard model.")
        return model