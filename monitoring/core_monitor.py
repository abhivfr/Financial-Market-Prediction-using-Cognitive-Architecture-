import torch
import numpy as np
from collections import deque
import time
from src.visualization.plot_engine import CognitiveVisualizer, VisualizationEngine

class CoreMonitor:
    def __init__(self, window_size=100):
        self.window_size = window_size
        self.metrics = {
            'layer_activations': deque(maxlen=window_size),
            'attention_patterns': deque(maxlen=window_size),
            'layer_gradients': deque(maxlen=window_size),
            'forward_time': deque(maxlen=window_size),
            'backward_time': deque(maxlen=window_size),
            'layer_sparsity': deque(maxlen=window_size)
        }
        self.start_time = None
        self.visualizer = CognitiveVisualizer(None)
        
    def start_forward(self):
        self.start_time = time.time()
        
    def end_forward(self, activations=None):
        if self.start_time:
            self.metrics['forward_time'].append(time.time() - self.start_time)
            if activations is not None:
                self._process_activations(activations)
                
    def start_backward(self):
        self.start_time = time.time()
        
    def end_backward(self, gradients=None):
        if self.start_time:
            self.metrics['backward_time'].append(time.time() - self.start_time)
            if gradients is not None:
                self._process_gradients(gradients)
                
    def _process_activations(self, activations):
        if isinstance(activations, torch.Tensor):
            sparsity = (activations == 0).float().mean().item()
            self.metrics['layer_sparsity'].append(sparsity)
            self.metrics['layer_activations'].append(
                activations.detach().float().mean().item()
            )
            
    def _process_gradients(self, gradients):
        if isinstance(gradients, torch.Tensor):
            self.metrics['layer_gradients'].append(
                torch.norm(gradients.detach()).item()
            )
            
    def log_attention(self, attention_weights):
        if isinstance(attention_weights, torch.Tensor):
            # Capture 4D attention patterns
            if attention_weights.dim() == 4:
                self.visualizer.plot_4d_attention(attention_weights.detach())
                
            pattern = attention_weights.detach().float().mean().item()
            self.metrics['attention_patterns'].append(pattern)
            
    def get_statistics(self):
        stats = {}
        for key, values in self.metrics.items():
            if values:
                stats[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
        return stats
        
    def reset(self):
        for key in self.metrics:
            self.metrics[key].clear()
        self.start_time = None

    def add_visualization_hooks(self, model):
        """Add visualization hooks to attention layers"""
        def attention_hook(module, inputs, outputs):
            if outputs.shape[-1] == 4:  # 4D attention heads
                self.log_attention(outputs)
                
        for name, layer in model.named_children():
            if 'attention' in name.lower():
                layer.register_forward_hook(attention_hook)