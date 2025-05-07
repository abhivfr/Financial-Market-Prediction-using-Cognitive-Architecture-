import torch
import torch.nn as nn
import torch.nn.functional as F

class MetaController(nn.Module):
    def __init__(self, input_dim=256, param_dim=8, market_importance_dim=4):
        super().__init__()
        self.market_importance_dim = market_importance_dim
        
        self.controller = nn.Sequential(
            nn.Linear(input_dim, param_dim * 4),
            nn.LayerNorm(param_dim * 4),
            nn.ReLU(),
            nn.Linear(param_dim * 4, param_dim * 2)
        )
        
        # Separate heads for different parameters
        self.memory_head = nn.Linear(param_dim * 2, 1)
        self.attention_head = nn.Linear(param_dim * 2, 1)
        self.market_importance_head = nn.Linear(param_dim * 2, market_importance_dim)
        
    def forward(self, system_state):
        features = self.controller(system_state)
        
        return {
            'memory_retention': torch.sigmoid(self.memory_head(features)),
            'attention_temperature': F.softplus(self.attention_head(features)) + 1.0,
            'market_importance': F.softplus(self.market_importance_head(features))
        }

class EnhancedMetaController(nn.Module):
    def __init__(self, input_dim=256, param_dim=8, market_importance_dim=4, attention_dim=2):
        super().__init__()
        self.market_importance_dim = market_importance_dim
        
        # Enhanced controller with attention awareness
        self.controller = nn.Sequential(
            nn.Linear(input_dim + attention_dim, param_dim * 4),  # Note added attention_dim
            nn.LayerNorm(param_dim * 4),
            nn.ReLU(),
            nn.Linear(param_dim * 4, param_dim * 2)
        )
        
        # Specialized heads
        self.memory_head = nn.Linear(param_dim * 2, 1)
        self.attention_head = nn.Linear(param_dim * 2, 1)
        self.market_importance_head = nn.Linear(param_dim * 2, market_importance_dim)
        self.volume_weight_head = nn.Linear(param_dim * 2, 1)  # New head for volume importance
        
    def forward(self, system_state, attention_stats):
        # Combine system state with attention statistics
        attention_tensor = torch.tensor([
            attention_stats.get('base_attention_variance', 0.0),
            attention_stats.get('volume_attention_variance', 0.0)
        ], device=system_state.device).unsqueeze(0)
        
        # Concatenate along feature dimension
        combined_input = torch.cat([system_state, attention_tensor], dim=-1)
        features = self.controller(combined_input)
        
        # Generate output parameters with appropriate activation functions
        return {
            'memory_retention': torch.sigmoid(self.memory_head(features)),
            'attention_temperature': F.softplus(self.attention_head(features)) + 1.0,
            'market_importance': F.softplus(self.market_importance_head(features)),
            'volume_weight': torch.sigmoid(self.volume_weight_head(features)) * 2.0  # Scale 0-2
        }
