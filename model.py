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