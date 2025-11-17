import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """Simple residual block for MLP"""

    def __init__(self, dim):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, x):
        return x + self.layer(x)


class ImpulsePredictor(nn.Module):
    """Physics-aware impulse predictor"""

    def __init__(self, input_dim=21, hidden_dim=128, dropout=0.1):
        super().__init__()

        self.input_ln = nn.LayerNorm(input_dim)

        # Separate encoders for different physical quantities
        # Adjust these dimensions based on your actual input structure
        self.velocity_encoder = nn.Sequential(
            nn.Linear(6, 64),  # Assuming 6D velocity (linear + angular)
            nn.SiLU(),
            nn.Dropout(dropout)
        )

        self.geometry_encoder = nn.Sequential(
            nn.Linear(9, 64),  # Collision geometry features
            nn.SiLU(),
            nn.Dropout(dropout)
        )

        self.material_encoder = nn.Sequential(
            nn.Linear(6, 32),  # Material/mass properties
            nn.SiLU(),
            nn.Dropout(dropout)
        )

        # Fusion network
        combined_dim = 64 + 64 + 32
        self.fusion = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            ResidualBlock(hidden_dim),
            ResidualBlock(hidden_dim),
        )

        # Output heads with skip connections
        self.linear_head = nn.Sequential(
            nn.Linear(hidden_dim + combined_dim, 128),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 3)
        )

        self.angular_head = nn.Sequential(
            nn.Linear(hidden_dim + combined_dim, 128),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 3)
        )

        # Auxiliary predictions for multi-task learning
        self.contact_force_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Softplus()  # Ensure positive force
        )

        self.restitution_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Bound between 0 and 1
        )

    def forward(self, x, return_aux=False):
        x = self.input_ln(x)

        # Split input based on your feature structure
        # Adjust indices based on your actual data
        vel_features = x[:, :6]
        geom_features = x[:, 6:15]
        mat_features = x[:, 15:21]

        # Encode different physics components
        vel_enc = self.velocity_encoder(vel_features)
        geom_enc = self.geometry_encoder(geom_features)
        mat_enc = self.material_encoder(mat_features)

        # Combine encodings
        combined = torch.cat([vel_enc, geom_enc, mat_enc], dim=1)

        # Process through fusion network
        h = self.fusion(combined)

        # Generate outputs with skip connections
        features_skip = torch.cat([h, combined], dim=1)
        linear_out = self.linear_head(features_skip)
        angular_out = self.angular_head(features_skip)

        impulse = torch.cat([linear_out, angular_out], dim=1)

        if return_aux:
            aux = {
                'contact_force': self.contact_force_head(h),
                'restitution': self.restitution_head(h)
            }
            return impulse, aux

        return impulse
