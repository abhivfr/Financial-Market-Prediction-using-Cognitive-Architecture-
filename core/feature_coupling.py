import torch
import torch.nn as nn

class FinancialFeatureCoupling(nn.Module):
    """Models explicit relationships between financial features (P,V,R,V)"""
    def __init__(self, input_dim=4):
        super().__init__()
        self.input_dim = input_dim
        
        # Pairwise relationship modeling
        self.pairwise = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2, 16),
                nn.LayerNorm(16),
                nn.ReLU(),
                nn.Linear(16, 1)
            )
            for _ in range(input_dim * (input_dim - 1) // 2)  # Number of unique pairs
        ])
        
        # Feature fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(input_dim + input_dim * (input_dim - 1) // 2, input_dim),
            nn.LayerNorm(input_dim),
            nn.Tanh()  # Tanh to keep values in reasonable range
        )
        
    def forward(self, x):
        """
        Model interdependencies between financial features
        Args:
            x: Input tensor (batch_size, seq_len, input_dim)
        Returns:
            Enhanced features with cross-feature relationships
        """
        batch_size, seq_len, _ = x.shape
        
        # Compute all pairwise relationships
        pair_idx = 0
        pair_outputs = []
        
        for i in range(self.input_dim):
            for j in range(i+1, self.input_dim):
                # Extract the pair of features
                pair = torch.cat([x[:, :, i:i+1], x[:, :, j:j+1]], dim=-1)
                # Apply pair-specific network
                pair_out = self.pairwise[pair_idx](pair)
                pair_outputs.append(pair_out)
                pair_idx += 1
        
        # Concatenate original features with pair relationships
        pair_tensor = torch.cat(pair_outputs, dim=-1)
        combined = torch.cat([x, pair_tensor], dim=-1)
        
        # Apply fusion layer
        enhanced = self.fusion(combined)
        
        # Return enhanced features that preserve 4D structure
        return enhanced
