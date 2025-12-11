import torch
import torch.nn as nn
import torch.nn.functional as F

class ImpulsePredictor(nn.Module):
    def __init__(self,
                 input_size=67,  # 6+6+6+1+3+2+44+1 = 69, but your data shows 67
                 max_contacts=4,
                 mass=1.0,
                 gravity=9.81,
                 timestep=1/240.0):
        super().__init__()

        self.max_contacts = max_contacts
        self.mass = mass
        self.gravity = gravity
        self.timestep = timestep

        # Shared encoder
        self.shared_encoder = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
        )

        # Linear impulse branch
        self.linear_branch = nn.Sequential(
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Dropout(0.1),
        )

        # Output head - predicting 3 values per contact point
        # Linear impulse: 3 components (x, y, z)
        self.head_linear = nn.Linear(128, max_contacts * 3)

        self._initialize_physics_biases()

    def _initialize_physics_biases(self):
        """Initialize biases with physics-informed values"""
        with torch.no_grad():
            # Start with small random values
            self.head_linear.bias.normal_(0.0, 0.01)

    def forward(self, x):
        batch_size = x.shape[0]

        # Shared encoding
        features = self.shared_encoder(x)

        # Branch processing
        linear_feat = self.linear_branch(features)

        # Predictions
        linear_impulses = self.head_linear(linear_feat)  # [Batch, Contacts*3]

        # Reshape to [Batch, Contacts, 3]
        linear_impulses = linear_impulses.view(batch_size, self.max_contacts, 3)

        # Output: for each contact, output [lin_x, lin_y, lin_z]
        output = linear_impulses.view(batch_size, self.max_contacts * 3)  # [Batch, 12] (4 contacts * 3 values)

        return output
