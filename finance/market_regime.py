import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional

class MarketRegimeDetector(nn.Module):
    def __init__(self, 
                 input_dim: int = 4,
                 hidden_dim: int = 256,
                 num_regimes: int = 3,
                 lookback_window: int = 50):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_regimes = num_regimes
        self.lookback_window = lookback_window
        
        # Enhanced volatility analysis network
        self.vol_encoder = nn.Sequential(
            nn.Linear(input_dim * 3, hidden_dim),  # Input + enhanced volatility features
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Enhanced trend analysis network
        self.trend_encoder = nn.Sequential(
            nn.Linear(input_dim * 3, hidden_dim),  # Input + trend features
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Enhanced regime classifier with uncertainty
        self.regime_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),  # Increased dropout for better generalization
            nn.Linear(hidden_dim, num_regimes * 2)  # Mean and variance for each regime
        )
        
        # Correlation structure analyzer with improved architecture
        self.correlation_encoder = nn.GRU(
            input_size=input_dim * input_dim,  # Correlation matrix features
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        
        # Liquidity analyzer with market microstructure features
        self.liquidity_encoder = nn.Sequential(
            nn.Linear(input_dim * 3, hidden_dim),  # Price + volume + microstructure
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Initialize with Xavier/Glorot
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
    def compute_volatility_features(self, x: torch.Tensor) -> torch.Tensor:
        """Enhanced volatility features with multiple time horizons"""
        price = x[..., 0]
        
        # Log returns for better numerical stability
        log_returns = torch.diff(torch.log(price + 1e-8), dim=-1)
        
        # Multiple volatility measures
        vol_features = []
        
        # Short-term volatility (5 steps)
        if log_returns.size(-1) >= 5:
            vol_st = log_returns[..., -5:].std(dim=-1, keepdim=True)
            vol_features.append(vol_st)
        else:
            vol_features.append(torch.zeros_like(price[..., :1]))
            
        # Medium-term volatility (10 steps)
        if log_returns.size(-1) >= 10:
            vol_mt = log_returns[..., -10:].std(dim=-1, keepdim=True)
            vol_features.append(vol_mt)
        else:
            vol_features.append(torch.zeros_like(price[..., :1]))
            
        # Realized range (High-Low)
        if price.size(-1) >= 5:
            high_5d = price[..., -5:].max(dim=-1, keepdim=True)[0]
            low_5d = price[..., -5:].min(dim=-1, keepdim=True)[0]
            realized_range = (high_5d - low_5d) / (low_5d + 1e-8)
            vol_features.append(realized_range)
        else:
            vol_features.append(torch.zeros_like(price[..., :1]))
        
        # Realized volatility (absolute return)
        if log_returns.size(-1) >= 5:
            realized_vol = log_returns[..., -5:].abs().mean(dim=-1, keepdim=True)
            vol_features.append(realized_vol)
        else:
            vol_features.append(torch.zeros_like(price[..., :1]))
            
        return torch.cat(vol_features, dim=-1)
        
    def compute_trend_features(self, x: torch.Tensor) -> torch.Tensor:
        """Enhanced trend indicators with multiple timeframes"""
        price = x[..., 0]
        
        trend_features = []
        
        # Calculate multiple moving averages if enough data
        if price.size(-1) >= 10:
            ma_short = price[..., -10:].mean(dim=-1, keepdim=True)
            trend_features.append(ma_short)
        else:
            trend_features.append(price[..., -1:])
            
        if price.size(-1) >= 20:
            ma_medium = price[..., -20:].mean(dim=-1, keepdim=True)
            trend_features.append(ma_medium)
        else:
            trend_features.append(price[..., -1:])
            
        if price.size(-1) >= 5:
            # Current price / MA ratio
            price_ma_ratio = price[..., -1:] / (ma_short + 1e-8)
            trend_features.append(price_ma_ratio)
        else:
            trend_features.append(torch.ones_like(price[..., -1:]))
        
        # Linear regression slope if enough data
        if price.size(-1) >= 10:
            x_vals = torch.arange(10, device=price.device).float()
            x_vals = (x_vals - x_vals.mean()) / x_vals.std()
            
            price_window = price[..., -10:]
            # Normalize price for numerical stability
            norm_price = (price_window - price_window.mean(dim=-1, keepdim=True)) / (price_window.std(dim=-1, keepdim=True) + 1e-8)
            
            # Calculate slope using covariance
            slope = (norm_price * x_vals).mean(dim=-1, keepdim=True) / (x_vals * x_vals).mean()
            trend_features.append(slope)
        else:
            trend_features.append(torch.zeros_like(price[..., -1:]))
        
        return torch.cat(trend_features, dim=-1)
        
    def compute_correlation_features(self, x: torch.Tensor) -> torch.Tensor:
        """Enhanced correlation features with exponential weighting"""
        # Ensure we have enough data
        if x.size(-2) < 5:
            # Return zeros if not enough data
            return torch.zeros(*x.shape[:-2], self.input_dim * self.input_dim, device=x.device)
            
        # Use exponential weighting for correlation
        weights = torch.exp(torch.arange(x.size(-2), device=x.device) / 5 - 1)
        weights = weights / weights.sum()
        
        # Center the data
        mean = (x * weights.unsqueeze(-1)).sum(dim=-2, keepdim=True)
        centered = x - mean
        
        # Calculate weighted covariance
        weighted_centered = centered * weights.unsqueeze(-1)
        cov = torch.matmul(weighted_centered.transpose(-2, -1), centered)
        
        # Calculate correlation matrix
        std = torch.sqrt(torch.diagonal(cov, dim1=-2, dim2=-1) + 1e-8)
        corr = cov / (std.unsqueeze(-1) * std.unsqueeze(-2) + 1e-8)
        
        return corr.view(*x.shape[:-2], -1)
        
    def compute_microstructure(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculate market microstructure features for liquidity assessment
        Args:
            x: Input features (batch_size, seq_len, input_dim)
        Returns:
            Microstructure features (batch_size, seq_len, 2)
        """
        # Extract features
        price = x[..., 0]
        volume = x[..., 1]
        volatility = x[..., 3]
        
        # Calculate basic microstructure features
        # Proxy for liquidity: turnover
        turnover = volume / (price + 1e-8)
        
        # Proxy for market impact: volatility / turnover
        impact = volatility / (turnover + 1e-8)
        
        # Volume volatility: std of volume over time
        if volume.size(-1) >= 5:
            vol_volatility = volume[..., -5:].std(dim=-1) / (volume[..., -5:].mean() + 1e-8)
        else:
            vol_volatility = torch.zeros_like(impact[..., -1])
            
        # Stack features
        microstructure = torch.stack([turnover[..., -1], impact[..., -1], vol_volatility], dim=-1)
        
        return microstructure
        
    def forward(self, x: torch.Tensor, batch_process: bool = True) -> Dict[str, torch.Tensor]:
        """Enhanced forward with batch processing option"""
        batch_size, seq_len, _ = x.shape
        
        # For large batches, process in chunks to reduce memory usage
        if batch_process and batch_size > 16:
            chunk_size = 16
            num_chunks = (batch_size + chunk_size - 1) // chunk_size
            
            # Initialize result containers
            regime_probs_list = []
            regime_uncertainty_list = []
            volatility_list = []
            trend_strength_list = []
            liquidity_list = []
            
            for i in range(num_chunks):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, batch_size)
                chunk_x = x[start_idx:end_idx]
                
                # Process chunk
                chunk_result = self.forward_impl(chunk_x)
                
                # Collect results
                regime_probs_list.append(chunk_result['regime_probs'])
                regime_uncertainty_list.append(chunk_result['regime_uncertainty'])
                volatility_list.append(chunk_result['volatility'])
                trend_strength_list.append(chunk_result['trend_strength'])
                liquidity_list.append(chunk_result['liquidity'])
            
            # Combine results
            return {
                'regime_probs': torch.cat(regime_probs_list, dim=0),
                'regime_uncertainty': torch.cat(regime_uncertainty_list, dim=0),
                'volatility': torch.cat(volatility_list, dim=0),
                'trend_strength': torch.cat(trend_strength_list, dim=0),
                'liquidity': torch.cat(liquidity_list, dim=0),
            }
        else:
            # Process entire batch at once
            return self.forward_impl(x)

    def forward_impl(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Enhanced regime detection with microstructure awareness
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_dim)
            
        Returns:
            Dict containing:
                - regime_probs: Probability distribution over regimes
                - regime_uncertainty: Uncertainty for each regime
                - volatility: Volatility level
                - trend_strength: Trend strength indicator
                - liquidity: Liquidity indicator
                - correlation_structure: Correlation structure features
                - microstructure: Market microstructure features
        """
        batch_size, seq_len, _ = x.shape
        
        # Compute enhanced features
        vol_features = self.compute_volatility_features(x)
        trend_features = self.compute_trend_features(x)
        corr_features = self.compute_correlation_features(x)
        micro_features = self.compute_microstructure(x)
        
        # Last time step for pointwise features
        x_last = x[:, -1, :]
        
        # Encode features
        vol_input = torch.cat([x_last, vol_features], dim=-1)
        vol_encoded = self.vol_encoder(vol_input)
        
        trend_input = torch.cat([x_last, trend_features], dim=-1)
        trend_encoded = self.trend_encoder(trend_input)
        
        # Process correlation structure
        corr_encoded, _ = self.correlation_encoder(corr_features.unsqueeze(1))
        corr_encoded = corr_encoded.squeeze(1)
        
        # Combine features for regime classification
        combined_features = torch.cat([vol_encoded, trend_encoded], dim=-1)
        regime_output = self.regime_classifier(combined_features)
        
        # Split into means and log variances
        regime_means, regime_logvars = regime_output.chunk(2, dim=-1)
        regime_probs = F.softmax(regime_means, dim=-1)
        regime_uncertainty = torch.exp(regime_logvars)
        
        # Compute liquidity score with microstructure features
        liquidity_input = torch.cat([x_last[:, 0:1], x_last[:, 1:2], micro_features], dim=-1)
        liquidity = self.liquidity_encoder(liquidity_input)
        
        return {
            'regime_probs': regime_probs,
            'regime_uncertainty': regime_uncertainty,
            'volatility': vol_features[:, 0],  # Short-term volatility
            'trend_strength': trend_features[:, -1],  # Regression slope
            'liquidity': liquidity.squeeze(-1),
            'correlation_structure': corr_encoded,
            'microstructure': micro_features
        }
