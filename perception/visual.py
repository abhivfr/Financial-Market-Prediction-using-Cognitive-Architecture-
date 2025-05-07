import torch.nn as nn
import torch

class VisualStream(nn.Module):
    def __init__(self):
        super().__init__()
        # Separate the convolutional backbone from the final linear layer
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1,1))
        )
        # Linear layer as a separate component
        self.fc = nn.Linear(32, 256)

    def forward(self, x):
        # Process through convolutional layers
        x = self.conv_layers(x)
        print(f"Shape after conv_layers: {x.shape}")
        # Flatten the tensor properly before passing to linear layer
        x = x.view(x.size(0), -1)
        print(f"Shape after flatten: {x.shape}")
        # Now apply the linear layer
        x = self.fc(x)
        print(f"Shape after fc: {x.shape}")
        return x
