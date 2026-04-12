import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------
# Fused SiLU-gated FFN with zero-init output
# ----------------------
class ResidualBlock(nn.Module):
    def __init__(self, dim: int, expansion: int = 4, dropout: float = 0.001):
        super().__init__()
        hidden = dim * expansion
        self.norm = nn.LayerNorm(dim)
        self.ff1 = nn.Linear(dim, hidden * 2)
        self.ff2 = nn.Linear(hidden, dim)
        self.drop_p = dropout

        nn.init.zeros_(self.ff2.weight)
        nn.init.zeros_(self.ff2.bias)

    def forward(self, x):
        h = self.norm(x)
        # Fused SiLU gate: split + silu + mul in one go
        gate_val = self.ff1(h)
        gate, val = gate_val.chunk(2, dim=-1)
        h = self.ff2(F.dropout(F.silu(gate) * val, self.drop_p, self.training))
        return x + F.dropout(h, self.drop_p, self.training)


# ----------------------
# Fast Multi-Head Attention using F.scaled_dot_product_attention
# (auto-dispatches to FlashAttention-2, memory-efficient, or math backend)
# ----------------------
class FastSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.001):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.dropout_p = dropout

        # Single fused QKV projection — one matmul instead of three
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, T, hd]
        q, k, v = qkv.unbind(0)

        # This auto-selects flash/mem-efficient/math backend
        h = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=False,
        )
        h = h.transpose(1, 2).reshape(B, T, D)
        return self.out_proj(h)


# ----------------------
# Fast Cross-Attention (queries from slots, KV from encoder)
# ----------------------
class FastCrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.001):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dropout_p = dropout

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.kv_proj = nn.Linear(dim, 2 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, queries, kv_source):
        B, Tq, D = queries.shape
        Tkv = kv_source.shape[1]

        q = self.q_proj(queries).reshape(B, Tq, self.num_heads, self.head_dim).transpose(1, 2)
        kv = self.kv_proj(kv_source).reshape(B, Tkv, 2, self.num_heads, self.head_dim)
        kv = kv.permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)

        h = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=False,
        )
        h = h.transpose(1, 2).reshape(B, Tq, D)
        return self.out_proj(h)


# ----------------------
# Transformer Block with fast self-attention
# ----------------------
class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4, expansion: int = 4, dropout: float = 0.001):
        super().__init__()
        self.attn_norm = nn.LayerNorm(dim)
        self.attn = FastSelfAttention(dim, num_heads, dropout)
        self.ffn = ResidualBlock(dim, expansion, dropout)
        self.drop_p = dropout

    def forward(self, x):
        h = self.attn(self.attn_norm(x))
        x = x + F.dropout(h, self.drop_p, self.training)
        x = self.ffn(x)
        return x


# ----------------------
# Simple MLP builder (streamlined)
# ----------------------
def mlp(dims, dropout=0.001):
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.LayerNorm(dims[i + 1]))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


# ----------------------
# Collision Predictor (optimized)
# ----------------------
class CollisionPredictorTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int = 15,
        num_input_tokens: int = 3,
        width: int = 256,
        num_encoder_layers: int = 4,
        num_heads: int = 4,
        max_contacts: int = 4,
        dropout: float = 0.001,
    ):
        super().__init__()

        self.max_contacts = max_contacts
        self.num_input_tokens = num_input_tokens
        self.width = width

        # --- Tokenized input projection ---
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, width * num_input_tokens),
            nn.LayerNorm(width * num_input_tokens),
            nn.GELU(),
        )
        self.token_norm = nn.LayerNorm(width)

        # --- Transformer encoder ---
        self.encoder = nn.Sequential(
            *[TransformerBlock(width, num_heads, dropout=dropout)
              for _ in range(num_encoder_layers)]
        )

        # --- Count head ---
        self.count_proj = ResidualBlock(width, dropout=dropout)
        self.count_head = mlp([width, 128, 64, max_contacts + 1], dropout)

        # --- Slot cross-attention and regression ---
        self.slot_embeddings = nn.Parameter(
            nn.init.trunc_normal_(torch.empty(max_contacts, width), std=0.02)
        )

        # Pre-register the index tensor so it's on the right device
        self.register_buffer(
            "slot_idx", torch.arange(max_contacts), persistent=False
        )

        self.slot_cross_norm = nn.LayerNorm(width)
        self.slot_cross_attn = FastCrossAttention(width, num_heads, dropout)

        self.slot_self_attn = TransformerBlock(width, num_heads, dropout=dropout)

        self.slot_mixer = nn.Sequential(
            ResidualBlock(width, dropout=dropout),
            ResidualBlock(width, dropout=dropout),
        )
        self.contact_points_head = mlp([width, 128, 64, 3], dropout)
        self.impulses_head = mlp([width, 128, 64, 3], dropout)

    def forward(
        self, x: torch.Tensor, gt_num_contacts: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        B = x.shape[0]

        # --- Tokenize and encode ---
        tokens = self.input_proj(x)
        tokens = tokens.view(B, self.num_input_tokens, self.width)
        tokens = self.token_norm(tokens)
        features = self.encoder(tokens)

        # --- Count head (pooled) ---
        pooled = features.mean(dim=1)
        count_features = self.count_proj(pooled)
        num_contacts_logits = self.count_head(count_features)

        # --- Slot cross-attention over encoder features ---
        queries = self.slot_embeddings.unsqueeze(0).expand(B, -1, -1)
        q_norm = self.slot_cross_norm(queries)
        slots = queries + self.slot_cross_attn(q_norm, features)  # residual

        # Slot self-attention + mixing
        slots = self.slot_self_attn(slots)
        slots = self.slot_mixer(slots)

        contact_points = self.contact_points_head(slots)
        impulses = self.impulses_head(slots)

        # --- Slot mask ---
        if gt_num_contacts is not None:
            n = gt_num_contacts
        else:
            n = num_contacts_logits.argmax(dim=-1)

        slot_mask = self.slot_idx.unsqueeze(0) < n.unsqueeze(1)

        return {
            "num_contacts": num_contacts_logits,
            "contact_points": contact_points,
            "impulses": impulses,
            "slot_mask": slot_mask,
        }


# ----------------------
# Convenience: compile the model for maximum speed
# ----------------------
def make_fast_predictor(**kwargs) -> CollisionPredictor:
    """Create a CollisionPredictor and wrap it with torch.compile for best perf."""
    model = CollisionPredictor(**kwargs)
    # fullgraph=True enables maximum fusion; mode="reduce-overhead" uses CUDA graphs
    compiled = torch.compile(model, fullgraph=True, mode="reduce-overhead")
    return compiled