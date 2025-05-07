import torch
import torch.nn as nn

class Financial4DEncoder(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64, output_dim=4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        # x: (batch_size, input_dim) where input_dim = [price, volume, time]
        return self.encoder(x)