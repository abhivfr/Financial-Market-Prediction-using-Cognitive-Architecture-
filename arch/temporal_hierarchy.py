import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalHierarchy(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=256, num_levels=3, max_seq_len=100):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_levels = num_levels
        self.max_seq_len = max_seq_len
        self.regime_detection_enabled = True
        
        # Create temporal encoders at different time scales
        self.encoders = nn.ModuleList([
            nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, 
                   num_layers=1, batch_first=True)
            for _ in range(num_levels)
        ])
        
        # Replace fixed thresholds with learnable boundaries
        self.vol_boundaries = nn.Parameter(torch.tensor([0.01, 0.03]))
        
        # Add regime detectors with improved architecture
        self.regime_detectors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, 3)  # 3 regime types: low/med/high volatility
            )
            for _ in range(num_levels)
        ])
        
        # Add scale importance for adaptive fusion
        self.scale_importance = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid()
            )
            for _ in range(num_levels)
        ])
        
        # Add pattern recognition with improved feature extraction
        self.pattern_detectors = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(input_dim, hidden_dim // 2, kernel_size=3, padding=1),
                nn.BatchNorm1d(hidden_dim // 2),
                nn.ReLU(),
                nn.Conv1d(hidden_dim // 2, hidden_dim // 2, kernel_size=5, padding=2),
                nn.BatchNorm1d(hidden_dim // 2),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(hidden_dim // 2, hidden_dim // 2)
            )
            for _ in range(num_levels)
        ])
        
        # Enhanced cross-scale attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=4, batch_first=True
        )
        
        # Regime-aware integration
        self.regime_integration = nn.Sequential(
            nn.Linear(num_levels * hidden_dim + 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Volatility tracking with improved smoothing
        self.register_buffer('volatility_history', torch.zeros(num_levels, 100))
        self.volatility_ptr = 0
        self.ema_factor = 0.95
        
    def detect_regime(self, vol):
        """Detect market regime with proper error handling"""
        # Ensure non-negative volatility
        vol = torch.abs(vol)
        
        # Sort boundaries for consistency
        boundaries = torch.sort(self.vol_boundaries)[0]
        
        # Calculate regime probabilities with smooth transitions
        regime_probs = torch.zeros(vol.size(0), 3, device=vol.device)
        
        # Low volatility regime: vol < boundaries[0]
        regime_probs[:, 0] = torch.sigmoid(-(vol - boundaries[0]) * 10)
        
        # Medium volatility: boundaries[0] < vol < boundaries[1]
        regime_probs[:, 1] = torch.sigmoid((vol - boundaries[0]) * 10) * torch.sigmoid(-(vol - boundaries[1]) * 10)
        
        # High volatility: vol > boundaries[1]
        regime_probs[:, 2] = torch.sigmoid((vol - boundaries[1]) * 10)
        
        # Handle NaN or inf values
        regime_probs = torch.where(torch.isnan(regime_probs) | torch.isinf(regime_probs), 
                                  torch.ones_like(regime_probs) / 3, 
                                  regime_probs)
        
        # Normalize to ensure sum is 1
        regime_probs = regime_probs / (regime_probs.sum(dim=1, keepdim=True) + 1e-8)
        
        return regime_probs
        
    def detect_patterns(self, x, level):
        """Detect temporal patterns at a specific time scale."""
        # Transpose for conv1d which expects [batch, channels, length]
        x_conv = x.transpose(1, 2)
        patterns = self.pattern_detectors[level](x_conv)
        return patterns.mean(dim=2)  # Average over time
        
    def update_volatility(self, x):
        """Track volatility at different time scales."""
        for level in range(self.num_levels):
            stride = max(1, 2 ** level)
            if x.size(1) // stride >= 2:
                downsampled = x[:, ::stride, 0]  # Assume first feature is price
                returns = (downsampled[:, 1:] - downsampled[:, :-1]) / (downsampled[:, :-1] + 1e-8)
                vol = returns.std(dim=1).mean().item()
                self.volatility_history[level, self.volatility_ptr] = vol
                
        self.volatility_ptr = (self.volatility_ptr + 1) % 100
        
    def get_regime_context(self):
        """Get historical regime context."""
        recent_vol = self.volatility_history[:, max(0, self.volatility_ptr-10):self.volatility_ptr].mean(dim=1)
        return recent_vol
        
    def forward(self, x):
        """Enhanced forward pass with improved regime detection and sequence handling"""
        batch_size, seq_len, _ = x.shape
        
        # Create padding mask for variable length sequences
        padding_mask = None
        if seq_len < self.max_seq_len:
            # Pad sequence to max_seq_len
            padding = x[:, :1].expand(-1, self.max_seq_len - seq_len, -1)
            x_padded = torch.cat([x, padding], dim=1)
            padding_mask = torch.ones(batch_size, self.max_seq_len, device=x.device)
            padding_mask[:, :seq_len] = 0
        else:
            # Truncate to max sequence length
            x_padded = x[:, -self.max_seq_len:]
            padding_mask = torch.zeros(batch_size, self.max_seq_len, device=x.device)
        
        # Calculate volatility for regime detection
        returns = torch.diff(x_padded[..., 0], dim=1) / (x_padded[..., 0][:, :-1] + 1e-8)
        vol = returns.std(dim=1, keepdim=True)  # Shape: [batch_size, 1]
        
        # Detect regime with learned boundaries
        regime_probs = self.detect_regime(vol) if self.regime_detection_enabled else torch.ones(batch_size, 3, device=x.device) / 3
        
        # Process at different time scales with importance weighting
        multi_scale_features = []
        scale_importances = []
        
        for level in range(self.num_levels):
            stride = max(1, 2 ** level)
            min_seq_needed = 2 * stride  # Need at least 2 points after downsampling
            
            if seq_len < min_seq_needed:
                continue
            
            downsampled = x_padded[:, ::stride, :]
            outputs, _ = self.encoders[level](downsampled)  # outputs: [batch_size, downsampled_len, hidden_dim]
            
            # Apply attention mask for padding if needed
            if padding_mask is not None:
                downsampled_mask = padding_mask[:, ::stride]
                outputs = outputs * (1 - downsampled_mask.unsqueeze(-1))
            
            # Get the last meaningful output
            last_idx = min(outputs.size(1) - 1, (seq_len - 1) // stride)
            last_hidden = outputs[:, last_idx, :]  # shape: [batch_size, hidden_dim]
            
            # Calculate scale importance
            importance = self.scale_importance[level](last_hidden)
            scale_importances.append(importance)
            
            # Process patterns from this scale
            pattern_input = downsampled.transpose(1, 2)  # [batch_size, input_dim, downsampled_len]
            pattern_features = self.pattern_detectors[level](pattern_input)
            
            # Combine LSTM and pattern features
            combined_features = torch.cat([last_hidden, pattern_features], dim=1)
            
            # Add to multi-scale features
            multi_scale_features.append(combined_features)
        
        if not multi_scale_features:
            # If no scales could be processed, return zeros
            return torch.zeros(batch_size, self.hidden_dim, device=x.device), regime_probs
        
        # Normalize importances
        importances = torch.cat(scale_importances, dim=1)
        scale_weights = F.softmax(importances, dim=1)
        
        # Stack features and apply importance weights
        stacked = torch.stack(multi_scale_features, dim=1)  # [batch_size, num_levels, hidden_dim+hidden_dim//2]
        
        # Apply scale importance weighting
        weighted_stacked = stacked * scale_weights.unsqueeze(-1)
        
        # Cross-scale attention for feature integration
        attn_output, _ = self.cross_attention(weighted_stacked, weighted_stacked, weighted_stacked)
        
        # Flatten attention output
        flat = attn_output.reshape(batch_size, -1)  # [batch_size, num_levels * (hidden_dim+hidden_dim//2)]
        
        # Concatenate with regime probabilities for final integration
        combined = torch.cat([flat, regime_probs], dim=1)
        
        # Final integration
        integrated = self.regime_integration(combined)
        
        return integrated, regime_probs
