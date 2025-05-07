import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils.volume_processor import VolumeProcessor, VolumeNormalizer

class VolumeStabilizer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.ema_alpha = 0.99
        self.register_buffer('running_mean', torch.zeros(1))
        self.register_buffer('running_var', torch.ones(1))
        self.register_buffer('running_median', torch.zeros(1))
        self.register_buffer('running_mad', torch.ones(1))
        
        self.volume_proj = nn.Sequential(
            nn.Linear(1, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, volume):
        if self.training:
            # Update running statistics
            curr_mean = volume.mean()
            curr_var = volume.var()
            # Update mean and variance with EMA
            self.running_mean = self.ema_alpha * self.running_mean + (1 - self.ema_alpha) * curr_mean
            self.running_var = self.ema_alpha * self.running_var + (1 - self.ema_alpha) * curr_var
        
            # Update median and MAD for robust normalization
            curr_median = volume.median()
            curr_mad = (volume - curr_median).abs().median()
            self.running_median = self.ema_alpha * self.running_median + (1 - self.ema_alpha) * curr_median
            self.running_mad = self.ema_alpha * self.running_mad + (1 - self.ema_alpha) * curr_mad
        
        # Robust normalization with median and MAD
        volume_centered = volume - self.running_median
        volume_norm = volume_centered / (self.running_mad + 1e-5)
        volume_norm = torch.clamp(volume_norm, -3, 3)  # Clip outliers
        
        # Project and stabilize
        volume_feat = self.volume_proj(volume_norm)
        return volume_feat

class VolumeAttention(nn.Module):
    """
    Enhanced volume-based attention mechanism with pattern detection
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Add layer normalization for stability
        self.input_norm = nn.LayerNorm(4)
        self.volume_norm = nn.LayerNorm(1)
        
        # Volume preprocessing with improved robustness
        self.volume_normalizer = VolumeNormalizer()
        self.volume_stabilizer = VolumeStabilizer(hidden_dim)
        
        # Volume pattern detection
        self.volume_pattern = nn.Sequential(
            nn.Conv1d(1, hidden_dim // 4, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.ReLU(),
            nn.Conv1d(hidden_dim // 4, hidden_dim // 4, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.ReLU()
        )
        
        # Sequence encoder with residual connections
        self.sequence_encoder = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Volume-specific encoder with residual connections
        self.volume_encoder = nn.Sequential(
            nn.Linear(1 + hidden_dim // 4, hidden_dim),  # Raw + pattern features
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Multi-head attention with more heads for better volume processing
        self.volume_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Add temperature parameter for sharper attention
        self.attention_temperature = nn.Parameter(torch.ones(1) * 1.5)
        
        # Output projection with skip connection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 4)
        )
        
        # Final normalization
        self.final_norm = nn.LayerNorm(4)
        
    def normalize_volume(self, volume):
        """
        Robust volume normalization
        Args:
            volume: Volume tensor (batch_size, seq_len, 1)
        Returns:
            Normalized volume (batch_size, seq_len, 1)
        """
        # Calculate median and MAD along sequence dimension
        volume_median = torch.median(volume, dim=1, keepdim=True)[0]
        volume_mad = torch.median(torch.abs(volume - volume_median), dim=1, keepdim=True)[0]
        normalized = (volume - volume_median) / (volume_mad + 1e-6)
        # Clip outliers
        return torch.clamp(normalized, -3, 3)
        
    def calculate_attention_variance(self, attn_weights):
        """
        Calculate the variance of attention weights
        Args:
            attn_weights: Attention weights (batch_size, num_heads, seq_len, seq_len)
            or (batch_size, seq_len, seq_len) if averaged
        Returns:
            Attention variance (batch_size,)
        """
        # If we get averaged weights (no head dimension)
        if len(attn_weights.shape) == 3:
            mean_attn = attn_weights
        else:
            # Mean across heads first
            mean_attn = attn_weights.mean(dim=1)  # (batch_size, seq_len, seq_len)
        
        # Calculate variance across sequence positions
        var_seq = mean_attn.var(dim=(1,2))  # (batch_size,)
        
        return var_seq
        
    def forward(self, x, volume=None):
        """
        Enhanced volume processing with pattern detection
        Args:
            x: Input features (batch_size, seq_len, 4)
            volume: Optional volume data (batch_size, seq_len, 1)
        Returns:
            Processed features with volume attention (batch_size, seq_len, 4)
        """
        batch_size, seq_len, _ = x.size()
        
        # Add this code for dimension checking
        # Ensure volume has correct dimensions
        if volume is not None:
            if volume.dim() == 2:
                # [batch_size, seq_len] -> [batch_size, seq_len, 1]
                volume = volume.unsqueeze(-1)
            elif volume.dim() == 3 and volume.size(-1) != 1:
                # [batch_size, seq_len, n] -> [batch_size, seq_len, 1]
                volume = volume[..., :1]
        
        # Input normalization
        x = self.input_norm(x)
        
        if volume is None:
            volume = x[:, :, 1:2]
        
        # Apply robust normalization
        volume_norm = self.normalize_volume(volume)
        
        # Detect volume patterns using 1D convolution
        volume_conv = volume_norm.transpose(1, 2)  # [batch_size, 1, seq_len]
        volume_patterns = self.volume_pattern(volume_conv)  # [batch_size, hidden_dim//4, seq_len]
        volume_patterns = volume_patterns.transpose(1, 2)  # [batch_size, seq_len, hidden_dim//4]
        
        # Stabilize volume
        volume_stabilized = self.volume_stabilizer(volume_norm)
        
        # Combine raw volume with pattern features
        volume_features = torch.cat([volume_stabilized, volume_patterns], dim=-1)
        
        # Encode the full sequence and volume
        encoded_seq = self.sequence_encoder(x)
        volume_encoded = self.volume_encoder(volume_features)
        
        # Scale queries by temperature for more dynamic attention
        query_scaling = self.attention_temperature.unsqueeze(0).unsqueeze(1)
        scaled_query = encoded_seq * query_scaling
        
        # Apply volume-based attention and capture attention weights
        attended_features, attn_weights = self.volume_attention(
            query=scaled_query,
            key=volume_encoded,
            value=encoded_seq,
            need_weights=True,
            average_attn_weights=False
        )
        
        # Calculate attention variance
        attn_variance = self.calculate_attention_variance(attn_weights)
        
        # Project back to 4D space with residual connection
        output = self.output_proj(attended_features)
        output = self.final_norm(output + x)
        
        return output, attn_variance
