import torch
import numpy as np
from collections import deque

class AdaptiveLearning:
    def __init__(self, base_lr=1e-4, min_lr=1e-6, max_lr=1e-3):
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.max_lr = max_lr
        
        # Enhanced metric tracking
        self.gradient_history = deque(maxlen=50)
        self.activation_history = deque(maxlen=50)
        self.attention_var_history = deque(maxlen=50)
        self.regime_history = deque(maxlen=50)
        
        # Learning rate multipliers for different components
        self.lr_multipliers = {
            'memory': 1.0,
            'attention': 1.0,
            'core': 1.0
        }
        
        # Add regime-specific learning rates
        self.regime_lr_multipliers = {
            'low_vol': 1.2,    # Higher in low volatility
            'medium_vol': 1.0, # Baseline in medium volatility
            'high_vol': 0.7    # Lower in high volatility
        }
    
    def update(self, monitoring_stats):
        """
        Enhanced learning rate adaptation with regime awareness
        Args:
            monitoring_stats: Dictionary with monitoring statistics
        Returns:
            Dictionary of learning rate multipliers
        """
        # Extract relevant metrics
        try:
            gradient_norm = monitoring_stats['core_metrics']['layer_gradients']['mean']
            self.gradient_history.append(gradient_norm)
            
            activation_mean = monitoring_stats.get('core_metrics', {}).get('layer_activations', {}).get('mean', 0)
            self.activation_history.append(activation_mean)
            
            attention_var = monitoring_stats.get('attention_variance', 0)
            if attention_var:
                # Handle both dictionary and scalar formats
                if isinstance(attention_var, dict):
                    avg_var = sum(attention_var.values()) / len(attention_var)
                    self.attention_var_history.append(avg_var)
                else:
                    self.attention_var_history.append(attention_var)
                    
            # Extract regime information
            if 'regime_probabilities' in monitoring_stats:
                regime_probs = monitoring_stats['regime_probabilities']
                # Track dominant regime
                if isinstance(regime_probs, torch.Tensor):
                    dominant_regime = regime_probs.argmax().item()
                    self.regime_history.append(dominant_regime)
                elif isinstance(regime_probs, (list, np.ndarray)):
                    dominant_regime = np.argmax(regime_probs)
                    self.regime_history.append(dominant_regime)
                    
        except (KeyError, TypeError):
            return self.lr_multipliers  # Return current multipliers if monitoring data is incomplete
            
        # Only adjust if we have enough history
        if len(self.gradient_history) < 10:
            return self.lr_multipliers
            
        # Calculate gradient ratio with enhanced stability
        recent_grads = list(self.gradient_history)[-5:]
        older_grads = list(self.gradient_history)[-10:-5]
        
        if older_grads:
            recent_grad_mean = np.mean(recent_grads)
            older_grad_mean = np.mean(older_grads)
            
            if older_grad_mean > 1e-8:  # Avoid division by very small values
                grad_ratio = recent_grad_mean / older_grad_mean
            else:
                grad_ratio = 1.0
        else:
            grad_ratio = 1.0
        
        # Adjust memory learning rate based on attention variance trend
        if self.attention_var_history:
            recent_att_var = np.mean(list(self.attention_var_history)[-5:])
            older_att_var = np.mean(list(self.attention_var_history)[-10:-5] or [recent_att_var])
            
            # More nuanced adjustment based on attention variance trends
            if recent_att_var > older_att_var * 1.3:
                # Sharp increase - memory needs more rapid adaptation
                self.lr_multipliers['memory'] = min(self.lr_multipliers['memory'] * 1.1, 2.0)
            elif recent_att_var > older_att_var * 1.1:
                # Moderate increase - gentle boost
                self.lr_multipliers['memory'] = min(self.lr_multipliers['memory'] * 1.03, 1.5)
            elif recent_att_var < older_att_var * 0.7:
                # Sharp decrease - memory might be overshooting
                self.lr_multipliers['memory'] = max(self.lr_multipliers['memory'] * 0.9, 0.5)
            else:
                # Gradual normalization toward baseline
                self.lr_multipliers['memory'] = self.lr_multipliers['memory'] * 0.95 + 1.0 * 0.05
                
        # Adjust core learning rate based on gradient behavior
        if grad_ratio > 2.0:  # Gradient increased sharply - potential instability
            self.lr_multipliers['core'] = max(self.lr_multipliers['core'] * 0.8, 0.5)
        elif grad_ratio > 1.5:  # Gradient increased moderately
            self.lr_multipliers['core'] = max(self.lr_multipliers['core'] * 0.95, 0.7)
        elif grad_ratio < 0.5:  # Gradient decreased sharply - might be converging or stuck
            self.lr_multipliers['core'] = min(self.lr_multipliers['core'] * 1.1, 1.5)
        elif grad_ratio < 0.8:  # Gradient decreased moderately
            self.lr_multipliers['core'] = min(self.lr_multipliers['core'] * 1.05, 1.3)
        else:  # Stable gradients - normalize toward baseline
            self.lr_multipliers['core'] = self.lr_multipliers['core'] * 0.9 + 1.0 * 0.1
            
        # Adjust attention learning rate based on activation statistics
        if len(self.activation_history) > 10:
            recent_act = np.mean(list(self.activation_history)[-5:])
            
            # Check for vanishing or exploding activations
            if recent_act < 0.1:  # Vanishing activations
                self.lr_multipliers['attention'] = min(self.lr_multipliers['attention'] * 1.1, 1.5)
            elif recent_act > 5.0:  # Potentially exploding activations
                self.lr_multipliers['attention'] = max(self.lr_multipliers['attention'] * 0.9, 0.7)
            else:  # Healthy activation range
                self.lr_multipliers['attention'] = self.lr_multipliers['attention'] * 0.95 + 1.0 * 0.05
                
        # Apply regime-based adjustments if we have regime history
        if self.regime_history:
            # Count regime frequencies
            regime_counts = np.bincount(list(self.regime_history)[-10:], minlength=3)
            regime_freqs = regime_counts / regime_counts.sum()
            
            # Calculate weighted regime multiplier
            regime_multiplier = (
                self.regime_lr_multipliers['low_vol'] * regime_freqs[0] +
                self.regime_lr_multipliers['medium_vol'] * regime_freqs[1] +
                self.regime_lr_multipliers['high_vol'] * regime_freqs[2]
            )
            
            # Apply to all components
            for key in self.lr_multipliers:
                self.lr_multipliers[key] *= regime_multiplier
            
        # Final safety limits
        for key in self.lr_multipliers:
            self.lr_multipliers[key] = max(min(self.lr_multipliers[key], 2.0), 0.5)
            
        return self.lr_multipliers
        
    def get_lr_for_param_group(self, param_group_name):
        """Get learning rate for a specific parameter group with enhanced bounds"""
        multiplier = self.lr_multipliers.get(param_group_name, 1.0)
        lr = self.base_lr * multiplier
        
        # Apply dynamic bounds based on training progress
        if hasattr(self, 'global_step'):
            # Gradually reduce max_lr as training progresses
            progress = min(self.global_step / 10000, 1.0)
            adjusted_max_lr = self.max_lr * (1.0 - 0.5 * progress)
            return max(min(lr, adjusted_max_lr), self.min_lr)
        else:
            return max(min(lr, self.max_lr), self.min_lr)
