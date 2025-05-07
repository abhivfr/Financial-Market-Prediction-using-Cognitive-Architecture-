import torch
import torch.nn as nn # Import torch.nn
import torch.nn.functional as F

# Inherit from nn.Module
class MemoryConfidence(nn.Module):
    # Add input_dim and memory_dim parameters
    def __init__(self, input_dim=4, memory_dim=256):
        super().__init__() # Call the parent constructor
        self.input_dim = input_dim
        self.memory_dim = memory_dim

        # Add a linear layer to project input features to memory dimension
        self.input_projection = nn.Linear(input_dim, memory_dim)

    def compute(self, retrieved, current):
        # retrieved shape: (batch_size, memory_dim) -> (batch_size, 256)
        # current shape: (batch_size, input_dim) -> (batch_size, 4)

        # Project the current input to the memory dimension
        projected_current = self.input_projection(current) # Shape: (batch_size, memory_dim)

        # Normalize both the projected current input and the retrieved memory
        projected_current_normalized = F.normalize(projected_current, dim=-1)
        retrieved_normalized = F.normalize(retrieved, dim=-1)

        # Calculate confidence using the dot product (cosine similarity)
        confidence = torch.sum(projected_current_normalized * retrieved_normalized, dim=-1) # Shape: (batch_size,)

        return confidence # Return confidence scores
