import torch
import torch.nn as nn
import torch.nn.functional as F

class ContactPointsPredictor(nn.Module):
    def __init__(self, input_dim=18, output_dim=13,
                 hidden_dims=[1024, 512,256,128,64,32], dropout=0.1):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h

        # output layer
        layers.append(nn.Linear(prev_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ImpulsePredictor(nn.Module):
    def __init__(self, input_dim=18, output_dim=12,
                 hidden_dims=[64,32], dropout=0.1):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h

        # output layer
        layers.append(nn.Linear(prev_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)