import torch
from torch.nn import Module, Linear

class MemoryEncoder(Module):
    def __init__(self, input_dim, encoding_dim=128):
        super().__init__()
        self.linear = Linear(input_dim, encoding_dim)
        self.relu = torch.nn.ReLU()

    def forward(self, experience):
        # Ensure input is on the same device as the encoder's weights
        experience = experience.to(self.linear.weight.device)
        encoded = self.linear(experience)
        encoded = self.relu(encoded)
        return encoded
