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

class ComplexImpulsePredictor(nn.Module):
    """Complex MLP for collision impulse prediction"""
    def __init__(self, input_dim=18, hidden_dims=[128, 64, 32], dropout=0.05):
        super().__init__()

        self.input_bn = nn.BatchNorm1d(input_dim)

        # Build deep residual MLP
        self.layers = nn.ModuleList()
        prev_dim = input_dim
        for h in hidden_dims:
            self.layers.append(nn.Linear(prev_dim, h))
            self.layers.append(nn.BatchNorm1d(h))
            self.layers.append(nn.SiLU())
            self.layers.append(nn.Dropout(dropout))
            self.layers.append(ResidualBlock(h))
            prev_dim = h

        # Separate heads for linear and angular impulses
        self.linear_head = nn.Sequential(
            nn.Linear(prev_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 3)
        )
        self.angular_head = nn.Sequential(
            nn.Linear(prev_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 3)
        )

    def forward(self, x):
        x = self.input_bn(x)
        for layer in self.layers:
            if isinstance(layer, ResidualBlock):
                x = layer(x)
            else:
                x = layer(x)
        linear_out = self.linear_head(x)
        angular_out = self.angular_head(x)
        return torch.cat([linear_out, angular_out], dim=1)