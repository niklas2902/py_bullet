"""
Optimized GNN Collision Predictor — same architecture, faster execution.

Key optimizations:
  1. Pre-expand dst index ONCE, reuse across all GNN layers
  2. Fuse euler trig ops (single cos/sin call on full tensor)
  3. Pre-register slot_idx buffer (no per-call arange)
  4. torch.compile()-friendly (no dynamic shapes, no Python-side branching in hot path)
  5. Optional: use torch.compile(mode="reduce-overhead") for 2-4x on GPU
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------
# Residual Block
# ----------------------
class ResidualBlock(nn.Module):
    def __init__(self, dim: int, expansion: int = 4, dropout: float = 0.0):
        super().__init__()
        hidden = dim * expansion
        self.norm = nn.LayerNorm(dim)
        self.ff1 = nn.Linear(dim, hidden * 2)
        self.ff2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(dropout)
        nn.init.zeros_(self.ff2.weight)
        nn.init.zeros_(self.ff2.bias)

    def forward(self, x):
        h = self.norm(x)
        gate, val = self.ff1(h).chunk(2, dim=-1)
        h = self.ff2(self.drop(F.silu(gate) * val))
        return x + self.drop(h)


# ----------------------
# Transformer Block
# ----------------------
class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4, expansion: int = 4, dropout: float = 0.0):
        super().__init__()
        self.attn_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = ResidualBlock(dim, expansion, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = self.attn_norm(x)
        h, _ = self.attn(h, h, h)
        x = x + self.drop(h)
        x = self.ffn(x)
        return x


# ----------------------
# MLP builder
# ----------------------
def mlp(dims, dropout=0.0):
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.LayerNorm(dims[i + 1]))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


# ==============================================================================
# GNN Layer — same as original, accepts pre-expanded dst index
# ==============================================================================
class GNNLayer(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int, dropout: float = 0.0):
        super().__init__()
        self.msg_mlp = nn.Sequential(
            nn.Linear(node_dim * 2 + edge_dim, node_dim),
            nn.LayerNorm(node_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(node_dim, node_dim),
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(node_dim * 2, node_dim),
            nn.LayerNorm(node_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(node_dim, node_dim),
        )
        nn.init.zeros_(self.update_mlp[-1].weight)
        nn.init.zeros_(self.update_mlp[-1].bias)

    def forward(self, node_features, edge_index, edge_features, dst_exp):
        B, N, D = node_features.shape
        src, dst = edge_index

        h_src = node_features[:, src]
        h_dst = node_features[:, dst]

        msg_input = torch.cat([h_src, h_dst, edge_features], dim=-1)
        messages = self.msg_mlp(msg_input)

        # Use pre-expanded dst index instead of rebuilding each layer
        agg = torch.zeros(B, N, D, device=node_features.device, dtype=node_features.dtype)
        agg.scatter_add_(1, dst_exp, messages)

        update_input = torch.cat([node_features, agg], dim=-1)
        node_features = node_features + self.update_mlp(update_input)
        return node_features


# ==============================================================================
# Cube Edges
# ==============================================================================
def build_cube_edges(fully_connected: bool = False):
    cube_edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    if fully_connected:
        cube_edges = [(i, j) for i in range(8) for j in range(i + 1, 8)]
    bidir = []
    for s, d in cube_edges:
        bidir.append((s, d))
        bidir.append((d, s))
    src = [e[0] for e in bidir]
    dst = [e[1] for e in bidir]
    return torch.tensor([src, dst], dtype=torch.long)


# ==============================================================================
# Euler → Rotation Matrix (fused trig)
# ==============================================================================
def euler_to_rotation_matrix(euler: torch.Tensor) -> torch.Tensor:
    cos = torch.cos(euler)
    sin = torch.sin(euler)
    cx, cy, cz = cos[:, 0], cos[:, 1], cos[:, 2]
    sx, sy, sz = sin[:, 0], sin[:, 1], sin[:, 2]

    sx_sy = sx * sy
    cx_sy = cx * sy

    R = torch.stack([
        torch.stack([cy * cz,  sx_sy * cz - cx * sz,  cx_sy * cz + sx * sz], dim=-1),
        torch.stack([cy * sz,  sx_sy * sz + cx * cz,  cx_sy * sz - sx * cz], dim=-1),
        torch.stack([-sy,      sx * cy,                cx * cy],              dim=-1),
    ], dim=1)
    return R


UNIT_CUBE_VERTICES = torch.tensor([
    [-0.5, -0.5, -0.5], [+0.5, -0.5, -0.5],
    [+0.5, +0.5, -0.5], [-0.5, +0.5, -0.5],
    [-0.5, -0.5, +0.5], [+0.5, -0.5, +0.5],
    [+0.5, +0.5, +0.5], [-0.5, +0.5, +0.5],
], dtype=torch.float32)


# ==============================================================================
# Encoder
# ==============================================================================
class CubePlaneFeatureEncoder(nn.Module):
    RAW_NODE_DIM = 7
    RAW_EDGE_DIM = 4

    def __init__(self, node_dim=64, edge_dim=32, plane_normal=(0., 1., 0.), plane_offset=0.):
        super().__init__()
        self.register_buffer("plane_normal", torch.tensor(plane_normal, dtype=torch.float32).view(1, 1, 3))
        self.register_buffer("plane_offset", torch.tensor([plane_offset]))
        self.register_buffer("local_verts", UNIT_CUBE_VERTICES)

        self.node_proj = nn.Sequential(
            nn.Linear(self.RAW_NODE_DIM, node_dim), nn.LayerNorm(node_dim), nn.GELU(),
        )
        self.edge_proj = nn.Sequential(
            nn.Linear(self.RAW_EDGE_DIM, edge_dim), nn.LayerNorm(edge_dim), nn.GELU(),
        )

    def forward(self, position, euler, edge_index):
        B = position.shape[0]
        R = euler_to_rotation_matrix(euler)
        local = self.local_verts.unsqueeze(0).expand(B, -1, -1)
        world_verts = torch.bmm(local, R.transpose(1, 2)) + position.unsqueeze(1)

        signed_dist = (world_verts * self.plane_normal).sum(-1, keepdim=True) - self.plane_offset

        raw_node = torch.cat([world_verts, local, signed_dist], dim=-1)
        node_features = self.node_proj(raw_node)

        src, dst = edge_index
        rel_pos = world_verts[:, dst] - world_verts[:, src]
        dist = rel_pos.norm(dim=-1, keepdim=True)

        raw_edge = torch.cat([rel_pos, dist], dim=-1)
        edge_features = self.edge_proj(raw_edge)

        return node_features, edge_features


# ==============================================================================
# Prediction Head
# ==============================================================================
class CollisionPredictorHead(nn.Module):
    def __init__(self, width=256, num_heads=4, max_contacts=4, dropout=0.0):
        super().__init__()
        self.max_contacts = max_contacts

        self.count_proj = ResidualBlock(width, dropout=dropout)
        self.count_head = mlp([width, 128, 64, max_contacts + 1], dropout)

        self.slot_embeddings = nn.Parameter(
            nn.init.trunc_normal_(torch.empty(max_contacts, width), std=0.02)
        )
        self.slot_cross_attn = nn.MultiheadAttention(width, num_heads, dropout=dropout, batch_first=True)
        self.slot_cross_norm = nn.LayerNorm(width)
        self.slot_self_attn = TransformerBlock(width, num_heads, dropout=dropout)
        self.slot_mixer = nn.Sequential(
            ResidualBlock(width, dropout=dropout),
            ResidualBlock(width, dropout=dropout),
        )
        self.contact_points_head = mlp([width, 128, 64, 3], dropout)
        self.impulses_head = mlp([width, 128, 64, 3], dropout)

        # Pre-register to avoid per-call allocation
        self.register_buffer("_slot_idx", torch.arange(max_contacts), persistent=False)

    def forward(self, features, gt_num_contacts=None):
        B = features.shape[0]

        pooled = features.mean(dim=1)
        count_features = self.count_proj(pooled)
        num_contacts_logits = self.count_head(count_features)

        queries = self.slot_embeddings.unsqueeze(0).expand(B, -1, -1)
        q_norm = self.slot_cross_norm(queries)
        slots, _ = self.slot_cross_attn(q_norm, features, features)
        slots = queries + slots
        slots = self.slot_self_attn(slots)
        slots = self.slot_mixer(slots)

        contact_points = self.contact_points_head(slots)
        impulses = self.impulses_head(slots)

        if gt_num_contacts is not None:
            n = gt_num_contacts
        else:
            n = num_contacts_logits.argmax(dim=-1)
        slot_mask = self._slot_idx.unsqueeze(0) < n.unsqueeze(1)

        return {
            "num_contacts": num_contacts_logits,
            "contact_points": contact_points,
            "impulses": impulses,
            "slot_mask": slot_mask,
        }


# ==============================================================================
# Full Model
# ==============================================================================
class GNNCollisionPredictor(nn.Module):
    """
    Same architecture as original. Speed gains from:
      1. dst index expanded once, shared across all 4 GNN layers
      2. Fused cos/sin in euler_to_rotation_matrix
      3. Pre-registered buffer for slot_idx
      4. torch.compile()-ready (no dynamic shapes)
    """

    def __init__(
        self,
        node_dim=128, edge_dim=64, gnn_layers=4,
        head_width=128, num_heads=4, max_contacts=4,
        dropout=0.0, fully_connected=True,
        plane_normal=(0., 1., 0.), plane_offset=0.,
    ):
        super().__init__()
        self.fully_connected = fully_connected

        self.encoder = CubePlaneFeatureEncoder(
            node_dim, edge_dim, plane_normal=plane_normal, plane_offset=plane_offset,
        )
        self.gnn_layers = nn.ModuleList([
            GNNLayer(node_dim, edge_dim, dropout=dropout) for _ in range(gnn_layers)
        ])
        self.to_head = (
            nn.Sequential(nn.LayerNorm(node_dim), nn.Linear(node_dim, head_width))
            if node_dim != head_width else nn.Identity()
        )
        self.head = CollisionPredictorHead(
            width=head_width, num_heads=num_heads,
            max_contacts=max_contacts, dropout=dropout,
        )

    def forward(self, position, euler, edge_index, gt_num_contacts=None):
        node_feat, edge_feat = self.encoder(position, euler, edge_index)

        # === KEY OPTIMIZATION: expand dst index once, reuse in all layers ===
        B, N, D = node_feat.shape
        dst = edge_index[1]
        dst_exp = dst.unsqueeze(0).unsqueeze(-1).expand(B, -1, D)

        for layer in self.gnn_layers:
            node_feat = layer(node_feat, edge_index, edge_feat, dst_exp)

        node_feat = self.to_head(node_feat)
        return self.head(node_feat, gt_num_contacts)