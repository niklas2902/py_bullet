import torch
import torch.nn as nn
import torch.nn.functional as F

class NumberContactPointsPredictor(nn.Module):
    def __init__(self, input_dim=6, output_dim=1,
                 hidden_dims=[256, 128, 64, 48], dropout=0.1):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.SiLU())
            prev_dim = h

        # output layer
        layers.append(nn.Linear(prev_dim, prev_dim))
        layers.append(nn.Linear(prev_dim, prev_dim))
        layers.append(nn.Linear(prev_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        out = self.net(x)
        return out

class ContactPointsPredictor(nn.Module):
    def __init__(self, input_dim=6, output_dim=12,
                 hidden_dims=[256,128, 64], dropout=0.1):
        super().__init__()

        layers = []


        prev_dim = input_dim

        for h in [512, 256, 128, 64]:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.GELU())
            prev_dim = h


        # output layer
        layers.append(nn.Linear(prev_dim, prev_dim))
        layers.append(nn.Linear(prev_dim, prev_dim))
        layers.append(nn.Linear(prev_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class ImpulesePredictor(nn.Module):
    def __init__(self, input_dim=9, output_dim=12,
                 hidden_dims=[512, 256,128, 64, 48], dropout=0.1):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.SiLU())
            prev_dim = h

        # output layer
        layers.append(nn.Linear(prev_dim, prev_dim))
        layers.append(nn.Linear(prev_dim, prev_dim))
        layers.append(nn.Linear(prev_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class CollisionPredictor(nn.Module):
    def __init__(
        self,
        input_dim=15,          # 6 + 9
        shared_dims=(640, 512, 384,  256),
        dropout=0.1,
    ):
        super().__init__()

        # ---------------------------
        # Shared encoder
        # ---------------------------
        layers = []
        prev_dim = input_dim

        for h in shared_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.SiLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h

        self.encoder = nn.Sequential(*layers)

        # ---------------------------
        # Fused prediction head
        # ---------------------------
        # Output: 1 (num_contacts) + 12 (contact_points) + 12 (impulses) = 25
        self.prediction_head = nn.Sequential(
            nn.Linear(prev_dim, 128),
            nn.Dropout(dropout),
            nn.Linear(128, 25)
        )

    def forward(self, x):
        """
        x: (B, 15)
        """
        features = self.encoder(x)
        predictions = self.prediction_head(features)

        # Split the fused output into separate components
        num_contacts = predictions[:, :1]
        contact_points = predictions[:, 1:13]
        impulses = predictions[:, 13:25]

        return {
            "num_contacts": num_contacts,
            "contact_points": contact_points,
            "impulses": impulses,
        }


class EdgeConvBlock(nn.Module):
    def __init__(self, hidden_dim, dropout):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        B, N, H = x.shape
        src, dst = edge_index

        x_src = x[:, src]
        x_dst = x[:, dst]

        edge_feat = torch.cat([x_src, x_dst], dim=-1)
        msg = self.mlp(edge_feat)

        # ----- aggregate (mean + max) -----
        out = torch.zeros_like(x)

        out.scatter_reduce_(
            dim=1,
            index=dst.view(1, -1, 1).expand(B, -1, H),
            src=msg,
            reduce="mean",
            include_self=False,
        )

        out_max = torch.zeros_like(x)
        out_max.scatter_reduce_(
            dim=1,
            index=dst.view(1, -1, 1).expand(B, -1, H),
            src=msg,
            reduce="amax",
            include_self=False,
        )

        out = out + out_max

        return self.dropout(out)


class EdgeConvGNN(nn.Module):
    """
    Stronger + better regularized EdgeConv GNN for a cube
    """

    def __init__(self, input_dim, hidden_dim=96, num_layers=3, dropout=0.15):
        super().__init__()

        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        ]

        edge_index = []
        for i, j in edges:
            edge_index += [(i, j), (j, i)]

        self.register_buffer(
            "edge_index",
            torch.tensor(edge_index, dtype=torch.long).t()
        )

        # --- identity prior ---
        self.vertex_embed = nn.Embedding(8, hidden_dim)

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
        )

        self.blocks = nn.ModuleList([
            EdgeConvBlock(hidden_dim, dropout)
            for _ in range(num_layers)
        ])

        self.output_proj = nn.Linear(hidden_dim, 4)

        self.graph_norm = nn.LayerNorm(hidden_dim)

    def forward(self, global_features):
        B = global_features.size(0)

        # ----- initial node features -----
        x = self.input_proj(global_features)        # (B, H)
        x = x.unsqueeze(1).expand(B, 8, -1)

        # add vertex identity
        vid = torch.arange(8, device=x.device)
        x = x + self.vertex_embed(vid)[None, :, :]

        # ----- message passing -----
        for block in self.blocks:
            x = x + block(x, self.edge_index)  # residual

        # ----- graph embedding -----
        graph_emb = self.graph_norm(x.mean(dim=1))

        # ----- contact points -----
        vertex_out = self.output_proj(x)  # (B, 8, 4)
        confidence = vertex_out[..., 3]

        _, top_idx = torch.topk(confidence, k=4, dim=1)
        batch_idx = torch.arange(B, device=x.device)[:, None]

        contact_points = vertex_out[batch_idx, top_idx, :3]

        return contact_points.reshape(B, 12), graph_emb


class CollisionPredictorGNN(nn.Module):
    def __init__(
        self,
        input_dim=9,
        shared_dims=(640, 512, 384, 256),
        gnn_hidden_dim=64,
        dropout=0.1,
    ):
        super().__init__()

        # ---------------------------
        # Shared encoder
        # ---------------------------
        layers = []
        prev_dim = input_dim
        for h in shared_dims:
            layers += [
                nn.Linear(prev_dim, h),
                nn.SiLU(),
                nn.Dropout(dropout),
            ]
            prev_dim = h

        self.encoder = nn.Sequential(*layers)

        # ---------------------------
        # GNN
        # ---------------------------
        self.contact_gnn = EdgeConvGNN(
            input_dim=prev_dim,
            hidden_dim=gnn_hidden_dim,
            dropout=dropout
        )

        fused_dim = prev_dim + gnn_hidden_dim

        # ---------------------------
        # Heads (NOW GNN-AWARE)
        # ---------------------------
        self.num_contacts_head = nn.Sequential(
            nn.Linear(fused_dim, 128),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

        self.impulse_head = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, 12)
        )

    def forward(self, x):
        features = self.encoder(x)  # (B, D)

        contact_points, graph_emb = self.contact_gnn(features)

        fused = torch.cat([features, graph_emb], dim=-1)

        num_contacts = self.num_contacts_head(fused)
        impulses = self.impulse_head(fused)

        return {
            "num_contacts": num_contacts,
            "contact_points": contact_points,
            "impulses": impulses,
        }
