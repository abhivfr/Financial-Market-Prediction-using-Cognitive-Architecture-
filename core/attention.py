from torch import nn
import torch
import math

class TemporalAttention(nn.Module):
    def __init__(self, dim=1024, heads=4):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.window_size = 12  # Increased window size
        self.scale = self.head_dim ** -0.5

        # Input projection
        self.q_proj = nn.Linear(256, self.dim)
        self.k_proj = nn.Linear(self.dim, self.dim)
        self.v_proj = nn.Linear(self.dim, self.dim)
        
        # Output projection
        self.out_proj = nn.Linear(dim, dim)
        
        # Positional encoding
        self.pos_embedding = nn.Parameter(torch.randn(self.window_size, self.dim))
        
        # Initialize window with zeros
        self.register_buffer('window', torch.zeros(self.window_size, self.dim))
        self.register_buffer('attention_weights', torch.zeros(heads, self.window_size))

    def forward(self, x):
        # Project input
        q_projected = self.q_proj(x)
        
        # Update attention window
        self.window = torch.roll(self.window, -1, 0)
        self.window[-1] = q_projected.detach().squeeze(0)
        
        # Add positional embeddings
        window_with_pos = self.window + self.pos_embedding

        # Split into heads
        q = q_projected.view(-1, self.heads, self.head_dim)
        k = self.k_proj(window_with_pos).view(self.window_size, self.heads, self.head_dim)
        v = self.v_proj(window_with_pos).view(self.window_size, self.heads, self.head_dim)

        # Scaled dot-product attention
        attn = torch.einsum('bhd,thd->bth', q, k) * self.scale
        attn = attn.softmax(dim=1)
        
        # Store attention weights for monitoring
        self.attention_weights = attn.mean(dim=0)

        # Apply attention to values
        out = torch.einsum('bth,thd->bhd', attn, v)
        out = out.view(-1, self.heads * self.head_dim)
        
        # Final projection
        return self.out_proj(out)

    def get_attention_weights(self):
        return self.attention_weights