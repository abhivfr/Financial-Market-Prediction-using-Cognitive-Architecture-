import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import os
from datetime import datetime
import json
from collections import deque

class EarlyWarningSystem:
    """
    Monitors model predictions and activations to detect unusual patterns
    that may indicate regime changes or model deterioration
    """
    
    def __init__(self, model, lookback_window=20, threshold_multiplier=2.0, 
                 output_dir="monitoring/early_warning"):
        """
        Initialize the early warning system
        
        Args:
            model: The cognitive model to monitor
            lookback_window: Window size for detecting anomalies
            threshold_multiplier: Multiplier for setting anomaly thresholds
            output_dir: Directory to save warning logs and visualizations
        """
        self.model = model
        self.lookback_window = lookback_window
        self.threshold_multiplier = threshold_multiplier
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize monitoring buffers
        self.error_history = deque(maxlen=lookback_window)
        self.prediction_history = deque(maxlen=lookback_window)
        self.activation_history = {}
        self.attention_history = deque(maxlen=lookback_window)
        self.confidence_history = deque(maxlen=lookback_window)
        
        # Warning log
        self.warnings = []
        
        # Register hooks for internal monitoring
        self._register_monitoring_hooks()
        
    def _register_monitoring_hooks(self):
        """Register hooks to monitor internal model activations"""
        self.hooks = []
        self.current_activations = {}
        
        # Monitor attention module
        if hasattr(self.model, 'attention'):
            def attention_hook(module, input, output):
                self.current_activations['attention'] = output.detach().mean().item()
                
                # Track attention weights if available
                if hasattr(module, 'attention_weights'):
                    weights = module.attention_weights.detach()
                    if weights is not None:
                        # Calculate entropy of attention weights as a measure of focus
                        entropy = -torch.sum(weights * torch.log(weights + 1e-10), dim=-1)
                        self.current_activations['attention_entropy'] = entropy.mean().item()
            
            hook = self.model.attention.register_forward_hook(attention_hook)
            self.hooks.append(hook)
            self.activation_history['attention'] = deque(maxlen=self.lookback_window)
            self.activation_history['attention_entropy'] = deque(maxlen=self.lookback_window)
        
        # Monitor memory module
        if hasattr(self.model, 'memory_module'):
            def memory_hook(module, input, output):
                self.current_activations['memory'] = output.detach().mean().item()
                
                # Track memory diversity if available
                if hasattr(module, 'get_memory_stats'):
                    stats = module.get_memory_stats()
                    if 'diversity' in stats:
                        self.current_activations['memory_diversity'] = stats['diversity']
            
            hook = self.model.memory_module.register_forward_hook(memory_hook)
            self.hooks.append(hook)
            self.activation_history['memory'] = deque(maxlen=self.lookback_window)
            self.activation_history['memory_diversity'] = deque(maxlen=self.lookback_window)
        
        # Monitor regime detector if available
        if hasattr(self.model, 'regime_detector'):
            def regime_hook(module, input, output):
                # Track regime probabilities
                if isinstance(output, torch.Tensor):
                    probs = output.detach()
                    
                    # Max probability as a confidence measure
                    max_prob = probs.max(dim=1)[0].mean().item()
                    self.current_activations['regime_confidence'] = max_prob
                    
                    # Regime entropy as a measure of uncertainty
                    entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
                    self.current_activations['regime_entropy'] = entropy.mean().item()
            
            hook = self.model.regime_detector.register_forward_hook(regime_hook)
            self.hooks.append(hook)
            self.activation_history['regime_confidence'] = deque(maxlen=self.lookback_window)
            self.activation_history['regime_entropy'] = deque(maxlen=self.lookback_window)
    
    def process_prediction(self, inputs, prediction, target, timestamp=None):
        """
        Process a new prediction and check for warning signals
        
        Args:
            inputs: Model input data
            prediction: Model prediction
            target: Actual target value
            timestamp: Optional timestamp for the observation
        
        Returns:
            Dict of detected warnings
        """
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Calculate prediction error
        error = torch.abs(prediction - target).mean().item()
        
        # Record history
        self.error_history.append(error)
        self.prediction_history.append(prediction.mean().item())
        
        # Add confidence if available
        if hasattr(self.model, 'get_confidence'):
            confidence = self.model.get_confidence(inputs)
            self.confidence_history.append(confidence)
        
        # Record activation history
        for key, value in self.current_activations.items():
            if key in self.activation_history:
                self.activation_history[key].append(value)
        
        # Check for warnings
        return self._check_warning_signals(timestamp)
    
    def _check_warning_signals(self, timestamp):
        """Check for potential warning signals in recent history"""
        warnings = {}
        
        # Only check if we have sufficient history
        if len(self.error_history) < self.lookback_window:
            return warnings
        
        # Calculate error statistics
        mean_error = np.mean(list(self.error_history)[:-1])  # Exclude latest
        std_error = np.std(list(self.error_history)[:-1])
        latest_error = self.error_history[-1]
        
        # Error spike detection
        error_threshold = mean_error + (std_error * self.threshold_multiplier)
        if latest_error > error_threshold:
            warnings['error_spike'] = {
                'current': latest_error,
                'threshold': error_threshold,
                'deviation': (latest_error - mean_error) / std_error
            }
        
        # Check activation anomalies
        for key, history in self.activation_history.items():
            if len(history) < self.lookback_window:
                continue
                
            values = list(history)
            mean_val = np.mean(values[:-1])  # Exclude latest
            std_val = np.std(values[:-1])
            latest_val = values[-1]
            
            # Check for significant deviations
            if std_val > 0:  # Avoid division by zero
                deviation = abs(latest_val - mean_val) / std_val
                if deviation > self.threshold_multiplier:
                    warnings[f'{key}_anomaly'] = {
                        'current': latest_val,
                        'mean': mean_val,
                        'deviation': deviation
                    }
        
        # Confidence-error correlation (if confidence available)
        if len(self.confidence_history) == self.lookback_window:
            confidences = np.array(list(self.confidence_history))
            errors = np.array(list(self.error_history))
            
            # Check for overconfidence with high error
            if confidences[-1] > np.mean(confidences) and errors[-1] > np.mean(errors):
                warnings['overconfidence'] = {
                    'confidence': confidences[-1],
                    'error': errors[-1]
                }
        
        # Log any warnings
        if warnings:
            warning_entry = {
                'timestamp': timestamp,
                'warnings': warnings
            }
            self.warnings.append(warning_entry)
            
            # Save to log file
            self._save_warnings()
            
            # Generate visualization for significant warnings
            if 'error_spike' in warnings or len(warnings) > 2:
                self._generate_warning_visualization()
        
        return warnings
    
    def _save_warnings(self):
        """Save warning log to file"""
        with open(os.path.join(self.output_dir, 'warning_log.json'), 'w') as f:
            json.dump(self.warnings, f, indent=2)
    
    def _generate_warning_visualization(self):
        """Generate visualization of recent data with warning indicators"""
        # Create figure with multiple subplots
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        
        # Time indices
        t = list(range(len(self.error_history)))
        
        # Plot 1: Prediction error
        axes[0].plot(t, list(self.error_history), 'b-', label='Prediction Error')
        
        # Add error threshold if calculated
        if len(self.error_history) >= self.lookback_window:
            mean_error = np.mean(list(self.error_history)[:-1])
            std_error = np.std(list(self.error_history)[:-1])
            threshold = mean_error + (std_error * self.threshold_multiplier)
            axes[0].axhline(y=threshold, color='r', linestyle='--', label='Error Threshold')
        
        axes[0].set_title('Prediction Error with Threshold')
        axes[0].set_ylabel('Error')
        axes[0].legend()
        
        # Plot 2: Key activations
        key_activations = ['attention', 'memory', 'regime_confidence'] 
        for key in key_activations:
            if key in self.activation_history and len(self.activation_history[key]) > 0:
                axes[1].plot(range(len(self.activation_history[key])), 
                             list(self.activation_history[key]), 
                             label=key)
        
        axes[1].set_title('Key Component Activations')
        axes[1].set_ylabel('Activation')
        axes[1].legend()
        
        # Plot 3: Prediction vs Confidence
        axes[2].plot(t, list(self.prediction_history), 'g-', label='Prediction')
        
        if len(self.confidence_history) > 0:
            conf_t = list(range(len(self.confidence_history)))
            axes[2].plot(conf_t, list(self.confidence_history), 'r-', label='Confidence')
        
        axes[2].set_title('Predictions and Confidence')
        axes[2].set_xlabel('Time Step')
        axes[2].legend()
        
        # Highlight the most recent point in all plots
        for ax in axes:
            if len(t) > 0:
                ax.plot(t[-1], ax.get_lines()[0].get_ydata()[-1], 'ro', markersize=8)
        
        plt.tight_layout()
        
        # Save visualization with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(os.path.join(self.output_dir, f'warning_{timestamp}.png'))
        plt.close()
    
    def get_summary_report(self):
        """Generate a summary report of recent warning patterns"""
        if not self.warnings:
            return {"status": "No warnings detected", "warnings_count": 0}
        
        # Count warning types
        warning_types = {}
        for entry in self.warnings:
            for warning_type in entry['warnings'].keys():
                warning_types[warning_type] = warning_types.get(warning_type, 0) + 1
        
        # Get most recent warnings
        recent_warnings = self.warnings[-5:] if len(self.warnings) >= 5 else self.warnings
        
        return {
            "status": "Warnings detected",
            "warnings_count": len(self.warnings),
            "warning_types": warning_types,
            "recent_warnings": recent_warnings
        }
    
    def reset(self):
        """Reset the warning system"""
        self.error_history.clear()
        self.prediction_history.clear()
        self.confidence_history.clear()
        
        for key in self.activation_history:
            self.activation_history[key].clear()
        
        self.warnings = []
        self._save_warnings()
