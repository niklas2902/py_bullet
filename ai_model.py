import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────
#  Building blocks
# ─────────────────────────────────────────────

class ResBlock(nn.Module):
    """Residual MLP block with LayerNorm."""
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.net(x)


class ContactHead(nn.Module):
    """
    Predicts 4 contact points or impulses via a small per-slot MLP.
    Shared trunk → 4 independent slot heads.
    """
    def __init__(self, in_dim: int, out_per_slot: int = 3, n_slots: int = 4):
        super().__init__()
        self.n_slots = n_slots
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim, 64),
                nn.GELU(),
                nn.Linear(64, out_per_slot),
            )
            for _ in range(n_slots)
        ])

    def forward(self, x):
        # x: (B, in_dim)  →  (B, n_slots * 3)
        return torch.cat([h(x) for h in self.heads], dim=-1)


# ─────────────────────────────────────────────
#  Main model
# ─────────────────────────────────────────────

class ContactPredictor(nn.Module):
    """
    Multi-task MLP that predicts:
      - num_contacts  : scalar ∈ [0, 1]
      - contact_points: (4 × 3,) = 12-d  (padded slots = -1)
      - impulses      : (4 × 3,) = 12-d  (padded slots = -1)

    Architecture
    ─────────────
    Input (9-d)
      └─ Stem linear → hidden_dim
           └─ N × ResBlock     (shared trunk)
                ├─ num_contacts head  (linear → sigmoid)
                ├─ contact_points head (4 slot heads)
                └─ impulses head       (4 slot heads)
    """

    def __init__(
        self,
        in_dim: int = 9,
        hidden_dim: int = 256,
        n_blocks: int = 6,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Stem
        self.stem = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
        )

        # Shared trunk
        self.trunk = nn.Sequential(
            *[ResBlock(hidden_dim, dropout) for _ in range(n_blocks)]
        )

        # Task heads
        self.num_contacts_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),           # outputs ∈ (0, 1)
        )

        self.cp_head  = ContactHead(hidden_dim, out_per_slot=3, n_slots=4)
        self.imp_head = ContactHead(hidden_dim, out_per_slot=3, n_slots=4)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        h = self.stem(x)
        h = self.trunk(h)
        return {
            "num_contacts":   self.num_contacts_head(h),   # (B, 1)
            "contact_points": self.cp_head(h),              # (B, 12)
            "impulses":       self.imp_head(h),             # (B, 12)
        }