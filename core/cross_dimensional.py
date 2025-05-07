import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.jit

class CrossDimensionalAttention(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Enhanced dimensional embeddings with non-linear transformations
        self.dim_embed = nn.Parameter(torch.randn(input_dim, hidden_dim))
        self.dim_bias = nn.Parameter(torch.zeros(input_dim))
        
        # Cross-dimension attention with multi-head
        self.query_proj = nn.Linear(input_dim, hidden_dim)
        self.key_proj = nn.Linear(input_dim, hidden_dim)
        self.value_proj = nn.Linear(input_dim, hidden_dim)
        
        # Multi-head attention for dimensions
        self.dim_attention = nn.MultiheadAttention(hidden_dim, num_heads=2, batch_first=True)
        
        # Dimension importance weighting
        self.dim_importance = nn.Linear(hidden_dim, input_dim)
        self.dim_scale = nn.Parameter(torch.ones(input_dim))
        
        # Output projection with residual connection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
        # Temperature parameter for attention
        self.attention_temp = nn.Parameter(torch.ones(1) * 1.5)
        
    def forward(self, x):
        """
        Apply enhanced cross-dimensional attention
        Args:
            x: Input tensor (batch_size, seq_len, input_dim)
        Returns:
            Tensor with cross-dimensional relationships emphasized
        """
        # Handle empty input gracefully
        if x.size(0) == 0 or x.size(1) == 0:
            return x
        
        # Check for NaN or inf values in input
        if torch.isnan(x).any() or torch.isinf(x).any():
            # Log warning and apply normalization
            print("Warning: NaN or inf values detected in input to CrossDimensionalAttention")
            x = torch.where(torch.isnan(x) | torch.isinf(x), torch.zeros_like(x), x)
        
        batch_size, seq_len, _ = x.shape
        
        # Project to queries, keys, values
        q = self.query_proj(x)  # (batch_size, seq_len, hidden_dim)
        k = self.key_proj(x)    # (batch_size, seq_len, hidden_dim)
        v = self.value_proj(x)  # (batch_size, seq_len, hidden_dim)
        
        # Reshape for attention over dimensions
        q = q.view(batch_size * seq_len, 1, self.hidden_dim)
        k = k.view(batch_size * seq_len, 1, self.hidden_dim)
        v = v.view(batch_size * seq_len, 1, self.hidden_dim)
        
        # Expand dimension embeddings with bias
        dim_k = self.dim_embed.unsqueeze(0).expand(batch_size * seq_len, self.input_dim, self.hidden_dim)
        
        # Use the JIT compiled function for attention scores
        attn_scores = compute_attention_scores(q, dim_k, self.attention_temp)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        
        # Calculate dimension importance based on current features
        dim_feat = F.softmax(self.dim_importance(q.squeeze(1)), dim=-1)
        
        # Scale attention weights by dimension importance
        scaled_weights = attn_weights * dim_feat.unsqueeze(1) * self.dim_scale
        
        # Apply attention
        dim_context = torch.bmm(scaled_weights, dim_k)  # (batch_size*seq_len, 1, hidden_dim)
        dim_context = dim_context.squeeze(1)  # (batch_size*seq_len, hidden_dim)
        
        # Reshape back and project to output space
        dim_context = dim_context.view(batch_size, seq_len, self.hidden_dim)
        
        # Multi-head self-attention across dimensions
        dim_context, _ = self.dim_attention(dim_context, dim_context, dim_context)
        
        # Project to output space
        output = self.output_proj(dim_context)
        
        # Enhanced residual connection with scaling
        return x + output * self.dim_scale.unsqueeze(0).unsqueeze(0)

# Add this function outside the class
@torch.jit.script
def compute_attention_scores(q, dim_k, temp):
    # Separate computation for JIT optimization
    attn_scores = torch.bmm(q, dim_k.transpose(1, 2))
    attn_scores = attn_scores / temp
    return attn_scores
