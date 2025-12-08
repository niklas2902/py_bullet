import torch
import torch.nn as nn
import torch.nn.functional as F


class ImpulsePredictor(nn.Module):
    def __init__(self,
                 input_size=30,
                 max_contacts=4,
                 mass=1.0,
                 gravity=9.81,
                 timestep=1 / 240.0):
        super().__init__()

        self.max_contacts = max_contacts
        self.mass = mass
        self.gravity = gravity
        self.timestep = timestep

        # ---------------------------------------------------------
        # 1. SHARED ENCODER
        # Process raw inputs into a high-level physics latent state
        # ---------------------------------------------------------
        self.shared_encoder = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.LayerNorm(128),  # Stabilizes training
            nn.SiLU(),  # Swish activation (often better for physics than ReLU)
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
        )

        # ---------------------------------------------------------
        # 2. SEPARATE BRANCHES
        # Split processing for Linear (Force) vs Angular (Torque)
        # ---------------------------------------------------------

        # Branch for Linear Impulses
        self.linear_branch = nn.Sequential(
            nn.Linear(256, 128),
            nn.SiLU()
        )

        # Branch for Angular Impulses
        self.angular_branch = nn.Sequential(
            nn.Linear(256, 128),
            nn.SiLU()
        )

        # ---------------------------------------------------------
        # 3. SPECIALIZED OUTPUT HEADS
        # ---------------------------------------------------------

        # Head A: Tangential Linear Impulse (X, Y)
        # No constraints, can be positive or negative
        self.head_tangential = nn.Linear(128, max_contacts * 2)

        # Head B: Normal Linear Impulse (Z)
        # MUST be non-negative (objects don't stick/pull)
        self.head_normal = nn.Linear(128, max_contacts * 1)

        # Head C: Angular Impulse (X, Y, Z)
        # No constraints
        self.head_angular = nn.Linear(128, max_contacts * 3)

        # Initialize biases to start with valid physics
        self._initialize_physics_biases()

    def _initialize_physics_biases(self):
        """
        Sets the bias of the normal head so the network starts by
        predicting a force that exactly counteracts gravity.
        """
        # Impulse J = F * dt. Force F = m * g.
        expected_normal_impulse = self.mass * self.gravity * self.timestep

        with torch.no_grad():
            # Tangential and Angular start at 0 (neutral)
            self.head_tangential.bias.fill_(0.0)
            self.head_angular.bias.fill_(0.0)

            # Normal starts at gravity compensation
            # We use inverse softplus (approx log(exp(y)-1)) or just a direct value
            # if we expect the pre-activation to be passed to softplus.
            # Since softplus(x) ≈ x for large x, setting bias to expected_val is safe.
            self.head_normal.bias.fill_(expected_normal_impulse)

    def forward(self, x):
        batch_size = x.shape[0]

        # --- Shared Processing ---
        features = self.shared_encoder(x)

        # --- Branch Processing ---
        linear_feat = self.linear_branch(features)
        angular_feat = self.angular_branch(features)

        # --- Predictions ---

        # 1. Tangential (X, Y) - Unconstrained
        tangential = self.head_tangential(linear_feat)  # [Batch, Contacts*2]
        tangential = tangential.view(batch_size, self.max_contacts, 2)

        # 2. Normal (Z) - Constrained to be Positive
        normal_raw = self.head_normal(linear_feat)  # [Batch, Contacts*1]
        # Softplus ensures Normal Impulse is always > 0 (Physics constraint)
        normal = F.softplus(normal_raw)
        normal = normal.view(batch_size, self.max_contacts, 1)

        # 3. Angular - Unconstrained
        angular = self.head_angular(angular_feat)  # [Batch, Contacts*3]
        angular = angular.view(batch_size, -1)  # Flatten

        # --- Assembly ---

        # Combine X,Y (tangential) and Z (normal)
        linear = torch.cat([tangential, normal], dim=2)  # [Batch, Contacts, 3]
        linear = linear.view(batch_size, -1)  # Flatten

        # Final concat: [Linear_All_Contacts, Angular_All_Contacts]
        output = torch.cat([linear, angular], dim=1)

        return output