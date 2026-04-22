import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

class ResidualBlock(nn.Module):
    """SiLU-GLU gated FFN with zero-init output (residual branch starts as identity)."""
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
    """Black-box contact-wrench predictor.

    Architecture:
        - Input projection + stack of residual SiLU-GLU FFN blocks.
        - Collision head: scalar logit.
        - Force head:     3-D body-frame force.
        - Torque head:    3-D body-frame torque, conditioned on trunk features
                          concatenated with the predicted force (information
                          shortcut, not a physics constraint).

    Inputs (10-D):
        [v_bx, v_by, v_bz, rel_pos_z, R[:,0] (3), R[:,1] (3)]

    Outputs (body frame, physical units):
        collision_logit: (B, 1)
        force:           (B, 3)
        torque:          (B, 3)
    """
    def __init__(self, input_dim=10, width=256, num_blocks=6, dropout=0.1):
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

        # --- heads ---
        # Collision: scalar logit.
        self.collision_head = mlp_head([width, 128, 1], dropout)
        # Force: 3-D regression.
        self.force_head     = mlp_head([width, 128, 3], dropout)
        # Torque: reads trunk features + predicted force (width + 3 inputs).
        self.torque_head    = mlp_head([width + 3, 128, 3], dropout)

        # Regression heads: small init on the final layer so outputs start near
        # zero (so training is stable for samples that are actually non-contact).
        for head in (self.force_head, self.torque_head):
            nn.init.normal_(head[-1].weight, std=0.01)
            nn.init.zeros_(head[-1].bias)
        # Collision head: zero-init so the initial logit is ~0 (p ~= 0.5).
        nn.init.zeros_(self.collision_head[-1].weight)
        nn.init.zeros_(self.collision_head[-1].bias)

    def forward(self, x):
        h = self.input_proj(x)
        h = self.backbone(h)

        collision_logit = self.collision_head(h)                       # (B, 1)
        force_vec       = self.force_head(h)                           # (B, 3)
        torque_vec      = self.torque_head(torch.cat([h, force_vec], dim=-1))  # (B, 3)

        return {
            "collision_logit": collision_logit,
            "force":           force_vec,
            "torque":          torque_vec,
        }


def make_fast_predictor(input_dim=10, width=256, num_blocks=6, dropout=0.1):
    model = WrenchPredictor(input_dim=input_dim,
                            width=width, num_blocks=num_blocks, dropout=dropout)
    try:
        compiled = torch.compile(model, mode="default")
        print("Model successfully compiled for optimised performance.")
        return compiled
    except Exception as e:
        print(f"torch.compile failed: {e}. Returning standard model.")
        return model
