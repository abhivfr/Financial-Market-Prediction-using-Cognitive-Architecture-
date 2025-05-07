import torch
import torch.nn as nn
import torch.nn.functional as F

class VolumeProcessor(nn.Module):
    """Base class for volume-specific processing"""
    def __init__(self, input_dim=1, hidden_dim=64):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        self.processor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, volume_data):
        """
        Process raw volume data
        Args:
            volume_data: Raw volume features (batch_size, seq_len?, input_dim)
        Returns:
            Processed volume features
        """
        if volume_data.dim() == 3:
            # Handle sequence data
            batch_size, seq_len, _ = volume_data.size()
            flat_volume = volume_data.view(-1, self.input_dim)
            processed = self.processor(flat_volume)
            return processed.view(batch_size, seq_len, self.hidden_dim)
        else:
            # Handle single-step data
            return self.processor(volume_data)

class VolumeNormalizer(nn.Module):
    """Normalizes volume data using adaptive statistics"""
    def __init__(self, window_size=100):
        super().__init__()
        self.window_size = window_size
        self.register_buffer('running_mean', torch.zeros(1))
        self.register_buffer('running_var', torch.ones(1))
        self.register_buffer('count', torch.zeros(1))
        
    def forward(self, volume_data):
        """
        Normalize volume data
        Args:
            volume_data: Raw volume data
        Returns:
            Normalized volume data
        """
        if self.training:
            # Update running statistics
            current_mean = volume_data.mean()
            current_var = volume_data.var()
            
            # Exponential moving average
            momentum = min(1.0, 1.0 / (self.count + 1))
            self.running_mean = (1 - momentum) * self.running_mean + momentum * current_mean
            self.running_var = (1 - momentum) * self.running_var + momentum * current_var
            self.count += 1
            
        # Normalize using current statistics
        return (volume_data - self.running_mean) / (torch.sqrt(self.running_var) + 1e-8)
