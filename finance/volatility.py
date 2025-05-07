import torch
import numpy as np

class VolatilityRegimeDetector:
    def __init__(self, window_size=20, low_threshold=0.15, high_threshold=0.30):
        self.window_size = window_size
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.history = []
        self.current_regime = 0.5  # Start in neutral regime
        
    def update(self, price_data):
        """
        Update volatility regime using new price data
        Args:
            price_data: tensor of shape (batch_size, features) or (batch_size, seq_len, features)
        Returns:
            float: Current regime value between 0 (low vol) and 1 (high vol)
        """
        if isinstance(price_data, torch.Tensor):
            price_data = price_data.detach().cpu().numpy()
            
        # Extract price from features (first dimension)
        if price_data.ndim == 3:
            prices = price_data[:, :, 0].mean(axis=0)  # Average across batch
        else:
            prices = price_data[:, 0]
            
        # Calculate returns
        returns = np.diff(np.log(prices + 1e-6))
        
        # Update history
        self.history.extend(returns.tolist())
        if len(self.history) > self.window_size:
            self.history = self.history[-self.window_size:]
            
        # Calculate current volatility
        vol = self.get_current_volatility()
        
        # Update regime with smoothing
        target_regime = self._calculate_regime(vol)
        self.current_regime = 0.9 * self.current_regime + 0.1 * target_regime
        
        return self.current_regime
            
    def get_current_volatility(self):
        """Return current annualized volatility estimate"""
        if len(self.history) < 2:
            return 0.0
        return np.std(self.history) * np.sqrt(252)  # Annualized
        
    def _calculate_regime(self, volatility):
        """
        Calculate regime value based on current volatility
        Args:
            volatility: Current volatility estimate
        Returns:
            float: Regime value between 0 (low vol) and 1 (high vol)
        """
        if volatility < self.low_threshold:
            return 0.0
        elif volatility > self.high_threshold:
            return 1.0
        else:
            # Linear interpolation between thresholds
            return (volatility - self.low_threshold) / (self.high_threshold - self.low_threshold)
            
    def get_regime_weights(self):
        """
        Get weight adjustments based on current regime
        Returns:
            tuple: (price_weight, volume_weight, returns_weight, volatility_weight)
        """
        # Low vol regime weights
        low_weights = np.array([0.3, 0.4, 0.2, 0.1])
        # High vol regime weights
        high_weights = np.array([0.2, 0.3, 0.3, 0.2])
        
        # Interpolate between regimes
        weights = low_weights * (1 - self.current_regime) + high_weights * self.current_regime
        return tuple(weights)
