HEAD_OUT_DIM = 9
import torch
import torch.nn as nn
import torch.nn.functional as F
HEAD_OUT_DIM = 9
VEL_SLICE = slice(0, 3)  # TODO: set to wherever velocity lives in your input
HEAD_OUT_DIM = 9
VEL_SLICE = slice(0, 3)  # TODO: set to wherever velocity lives in your input

class ResidualBlock(nn.Module):
    """SiLU-gated FFN with zero-init output."""
    def __init__(self, dim, expansion=4, dropout=0.1):
        super().__init__()
        hidden = dim * expansion
        self.norm = nn.LayerNorm(dim)
        self.ff1  = nn.Linear(dim, hidden * 2)
        self.ff2  = nn.Linear(hidden, dim)
        self.drop_p = dropout
        nn.init.zeros_(self.ff2.weight)
        nn.init.zeros_(self.ff2.bias)

    def forward(self, x):
        h = self.norm(x)
        gate, val = self.ff1(h).chunk(2, dim=-1)
        h = self.ff2(F.dropout(F.silu(gate) * val, self.drop_p, self.training))
        return x + F.dropout(h, self.drop_p, self.training)


def mlp_head(dims, dropout=0.1):
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.LayerNorm(dims[i + 1]))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)



class WrenchPredictor(nn.Module):
    def __init__(self, input_dim=10, width=256, num_blocks=4, dropout=0.1,
                 baseline_k=1e3, learn_k=True, baseline_bounciness = 0.5, learn_bounciness =True,
                 head_hidden=64):
        super().__init__()

        # --- backbone ---
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, width),
            nn.LayerNorm(width),
            nn.GELU(),
        )
        self.backbone = nn.Sequential(
            *[ResidualBlock(width, dropout=dropout) for _ in range(num_blocks)]
        )

        self.head_trunk = nn.Sequential(
            nn.Linear(width, head_hidden),
            nn.LayerNorm(head_hidden),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )
        self.head_out = nn.Linear(head_hidden, HEAD_OUT_DIM)

        self.k= nn.Parameter(
            torch.tensor(baseline_k, dtype=torch.float32),
            requires_grad=learn_k,
        )
        self.bounciness = nn.Parameter(
            torch.tensor(baseline_bounciness, dtype=torch.float32),
            requires_grad=learn_bounciness,
        )
        nn.init.zeros_(self.head_out.weight)
        nn.init.zeros_(self.head_out.bias)


    def forward(self, x):
        h = self.input_proj(x)
        h = self.backbone(h)

        raw = self.head_out(self.head_trunk(h))
        collision_logit, depth, force_residual, contact_normal, lever = raw.split(
            [1, 1, 1, 3, 3], dim=-1
        )

        # Stiffness (always positive) and damping coefficient
        k_pos = F.softplus(self.k)
        c = 2.0 * torch.sqrt(k_pos) * torch.sigmoid(self.bounciness)  # assumes unit mass

        # Penetration proxy, non-negative
        depth = F.softplus(depth)

        # Normal-force magnitude: Hooke baseline + learned residual
        force_physical = k_pos * depth
        force = force_physical + force_residual

        # Unit contact normal in body frame
        contact_normal = F.normalize(contact_normal, dim=-1, eps = 1e-6)  # (B, 3)

        # Critically damp force
        velocity = x[:, VEL_SLICE]                                    # (B, 3)
        vel_normal = (velocity * contact_normal).sum(dim=-1, keepdim=True)  # (B, 1)
        damping_force = -c * vel_normal                               # opposes motion
        force = F.relu(force + damping_force)

        # Assemble force and torque
        force_vec  = force * contact_normal                           # (B, 3)
        torque_vec = torch.cross(lever, force_vec, dim=-1)            # (B, 3)

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
                "force":          force,
                "force_physical": force_physical,
                "force_residual": force_residual,
            },
          }
def make_fast_predictor(input_dim=10, width=128, num_blocks=4, dropout=0.1,
                        baseline_k=1e3):
    model = WrenchPredictor(input_dim=input_dim,
                            width=width, num_blocks=num_blocks, dropout=dropout,
                            baseline_k=baseline_k)
    # NOTE: compile mode changed to "default" — "reduce-overhead" uses CUDA
    # graphs and is picky about dynamic shapes / dict outputs during training.
    try:
        compiled = torch.compile(model, mode="default")
        print("Model successfully compiled for optimised performance.")
        return compiled
    except Exception as e:
        print(f"torch.compile failed: {e}. Returning standard model.")
        return model