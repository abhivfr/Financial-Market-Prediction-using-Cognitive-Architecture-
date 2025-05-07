import torch
import torch.nn as nn # Import nn

class MemoryReplay(nn.Module): # Inherit from nn.Module
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(128, 256) # Example projection layer

    def replay(self, memory):
        if memory is not None:
            return memory.clone().detach().requires_grad_(True)
        return None
