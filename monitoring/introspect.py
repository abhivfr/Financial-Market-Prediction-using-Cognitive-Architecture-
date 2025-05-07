#!/usr/bin/env python
# introspect.py - Model introspection utilities

import torch
import numpy as np
import time
from collections import defaultdict
import GPUtil
import psutil
from src.visualization.flow_visualizer import InformationFlowVisualizer
import os

class Introspection:
    """
    Tool for monitoring internal states and dynamics of the cognitive architecture
    """
    def __init__(self, window_size=50):
        """
        Initialize introspection tool
        
        Args:
            window_size: Window size for moving averages
        """
        self.window_size = window_size
        self.metrics = {}
        self.history = defaultdict(list)
        self.hooks = []
        self.start_time = time.time()
        self.gpu_util = GPUtil.getGPUs()[0] if GPUtil.getGPUs() else None # Get the first GPU if available
        self.flow_visualizer = InformationFlowVisualizer(None, "evaluation/information_flow")

    def register_hooks(self, model):
        """
        Register hooks to monitor model activations
        
        Args:
            model: Model to monitor
        """
        # Remove existing hooks
        self.remove_hooks()
        
        # Activation monitoring function
        def hook_fn(name):
            def fn(module, input, output):
                # Skip if not in training mode
                if not module.training:
                    return
                
                # Calculate activation statistics
                if isinstance(output, torch.Tensor):
                    # Calculate statistics
                    act_mean = output.mean().item()
                    act_std = output.std().item()
                    act_norm = output.norm().item()
                    
                    # Track activation statistics
                    self.metrics[f"{name}_mean"] = act_mean
                    self.metrics[f"{name}_std"] = act_std
                    self.metrics[f"{name}_norm"] = act_norm
                    
                    # Check for issues
                    if np.isnan(act_mean) or np.isnan(act_std):
                        print(f"Warning: NaN detected in {name} activations")
                    
                    if act_std < 1e-6:
                        print(f"Warning: Low variance in {name} activations")
                        
            return fn
        
        # Register hooks for all modules
        for name, module in model.named_modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.LSTM, torch.nn.Conv1d)):
                hook = module.register_forward_hook(hook_fn(name))
                self.hooks.append(hook)
    
    def remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def log(self, model, gpu_memory_allocated=0):
        """
        Log model metrics
        
        Args:
            model: Model to log metrics for
        """
        # Collect gradient statistics
        grad_norms = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_norms[name] = grad_norm
                
                # Track in metrics
                self.metrics[f"grad_{name}"] = grad_norm
        
        # Average gradient norm
        if grad_norms:
            self.metrics["grad_norm_avg"] = np.mean(list(grad_norms.values()))
        
        # Get memory usage (for models with memory components)
        if hasattr(model, "financial_memory") and hasattr(model.financial_memory, "get_memory_usage"):
            self.metrics["memory_usage"] = model.financial_memory.get_memory_usage()
        
        # Get attention statistics (for models with attention)
        if hasattr(model, "attention") and hasattr(model.attention, "get_attention_weights"):
            attention_weights = model.attention.get_attention_weights()
            if attention_weights is not None:
                self.metrics["attention_entropy"] = self._calculate_attention_entropy(attention_weights)
        
        # Update history
        for key, value in self.metrics.items():
            self.history[key].append(value)
            # Keep only the recent history
            if len(self.history[key]) > self.window_size:
                self.history[key] = self.history[key][-self.window_size:]
        
        # Add elapsed time
        self.metrics["elapsed_time"] = time.time() - self.start_time

        current_time = time.time()
        runtime = current_time - self.start_time
        self._add_metric('runtime', runtime)
        self._add_metric('gpu_memory', gpu_memory_allocated)

        # Log GPU utilization if available
        if self.gpu_util:
            self._add_metric('gpu_util', self.gpu_util.load)
            self._add_metric('gpu_temp', self.gpu_util.temperature)


        # Log model gradient norms
        total_grad_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                total_grad_norm += p.grad.norm().item()**2
        self._add_metric('model_grad_norm', total_grad_norm**0.5)

        # Log memory stats from NeuralDictionary
        # Assuming 'model' has a 'financial_memory' attribute
        if hasattr(model, 'financial_memory') and hasattr(model.financial_memory, 'get_memory_stats'):
            memory_stats = model.financial_memory.get_memory_stats()
            for key, value in memory_stats.items():
                # Handle the 'market_importance' list specifically, or ensure all logged values are scalar
                # If get_memory_stats returns a list for market_importance, we need to handle it in report()
                # For now, append the value as is, and handle different types in report()
                self._add_metric(key, value)

    def _add_metric(self, key, value):
        if key not in self.metrics:
            self.metrics[key] = []
        self.metrics[key].append(value)

    def report(self):
        print("\n=== Introspection Report ===")

        # Print Runtime and Memory Usage (assuming these are scalar metrics)
        runtime = self.metrics.get('runtime', [0])[-1]
        gpu_memory = self.metrics.get('gpu_memory', [0])[-1]
        print(f"Runtime: {runtime:.2f}s")
        print("Memory Usage:")
        print(f"Current: {gpu_memory / (1024**3):.2f}GB") # Convert bytes to GB

        # Print GPU stats if available
        if 'gpu_util' in self.metrics and 'gpu_temp' in self.metrics:
            gpu_util = self.metrics['gpu_util'][-1]
            gpu_temp = self.metrics['gpu_temp'][-1]
            print(f"GPU Utilization: {gpu_util:.2%}") # Format as percentage
            print(f"GPU Temperature: {gpu_temp:.2f}°C")


        # Print Model Gradient Norm (assuming scalar)
        if 'model_grad_norm' in self.metrics:
            grad_norm = self.metrics['model_grad_norm'][-1]
            print(f"Model Grad Norm: {grad_norm:.4f}")


        # Print 4D Market Metrics (Placeholder based on previous structure)
        print("\n4D Market Metrics:")
        # Add logic here if you have specific 4D market metrics to report


        # Print Memory Stats from NeuralDictionary
        print("\nMemory Stats:")
        # Iterate through metrics collected from memory_stats
        for key, values_list in self.metrics.items():
            # Exclude metrics already printed above or handled differently
            if key in ['runtime', 'gpu_memory', 'gpu_util', 'gpu_temp', 'model_grad_norm']:
                continue

            # Ensure values_list is not empty before accessing elements
            if not values_list:
                print(f"{key}: No data recorded")
                continue

            # Get the last recorded value (which could be a scalar or a list)
            last_recorded_value = values_list[-1]

            if key == 'market_importance':
                # Handle the list of market importances specifically
                if isinstance(last_recorded_value, list):
                     # Format and join the elements of the list
                     formatted_elements = []
                     for item in last_recorded_value:
                         # Ensure item is a number before formatting
                         if isinstance(item, (int, float, torch.Tensor)):
                             formatted_elements.append(f'{float(item):.4f}')
                         else:
                             formatted_elements.append(str(item)) # Convert non-numeric items to string
                     print(f"{key}: [{', '.join(formatted_elements)}]")
                else:
                    # Should not happen for market_importance if log is correct, but print as is
                    print(f"{key}: {last_recorded_value}")
            else:
                # For other memory stats (scalars like usage_mean, std, norms, filled_capacity)
                # Ensure it's a scalar before formatting as float
                if isinstance(last_recorded_value, (int, float, torch.Tensor)) and (isinstance(last_recorded_value, torch.Tensor) and last_recorded_value.ndim == 0 or not isinstance(last_recorded_value, torch.Tensor)):
                     print(f"{key}: {float(last_recorded_value):.4f}")
                else:
                     # If it's not a scalar (e.g., a tensor with dimensions, or another type), print as is
                     print(f"{key}: {last_recorded_value}")


        print("============================")


    def plot_metrics(self):
        # This method is not fully implemented in the provided code snippet,
        # but would typically involve using matplotlib or other plotting libraries
        # to visualize the historical data stored in self.metrics.
        print("Plotting functionality not fully implemented.")
        pass # Placeholder for plotting logic

    def analyze_regime_transitions(self, regime_history):
        """
        Analyze regime transitions for stability assessment
        Args:
            regime_history: List of regime probabilities over time
        Returns:
            Dictionary of transition metrics
        """
        if len(regime_history) < 10:
            return {}
        
        # Extract dominant regimes
        dominant_regimes = [probs.argmax().item() if isinstance(probs, torch.Tensor) else np.argmax(probs) 
                            for probs in regime_history]
        
        # Count transitions
        transitions = 0
        for i in range(1, len(dominant_regimes)):
            if dominant_regimes[i] != dominant_regimes[i-1]:
                transitions += 1
        
        # Calculate stability metrics
        stability = 1.0 - (transitions / (len(dominant_regimes) - 1))
        
        # Count occurrences of each regime
        regime_counts = {}
        for i in range(3):  # 3 regimes: low/med/high volatility
            regime_counts[f'regime_{i}'] = dominant_regimes.count(i) / len(dominant_regimes)
        
        return {
            'regime_stability': stability,
            'regime_transitions': transitions,
            'regime_distribution': regime_counts
        }

    def analyze_memory_performance(self, memory):
        """
        Analyze memory performance and usage patterns
        Args:
            memory: NeuralDictionary instance
        Returns:
            Dictionary of memory performance metrics
        """
        if not hasattr(memory, 'keys') or not hasattr(memory, 'values'):
            return {}
        
        # Calculate memory utilization
        capacity = memory.capacity
        filled = memory.next_index
        utilization = filled / capacity
        
        # Calculate key diversity (cosine similarity between keys)
        if filled > 1:
            keys = memory.keys[:filled].detach()
            key_norms = torch.norm(keys, dim=1, keepdim=True)
            normalized_keys = keys / (key_norms + 1e-8)
            similarity_matrix = torch.matmul(normalized_keys, normalized_keys.t())
            # Exclude self-similarity (diagonal)
            mask = torch.ones_like(similarity_matrix) - torch.eye(filled, device=keys.device)
            mean_similarity = (similarity_matrix * mask).sum() / (filled * (filled - 1))
            diversity = 1.0 - mean_similarity.item()
        else:
            diversity = 0.0
        
        # Calculate importance statistics
        if hasattr(memory, 'importance_scores') and filled > 0:
            importance = memory.importance_scores[:filled].detach()
            importance_mean = importance.mean().item()
            importance_std = importance.std().item() if filled > 1 else 0.0
            importance_max = importance.max().item()
            importance_min = importance.min().item()
        else:
            importance_mean = importance_std = importance_max = importance_min = 0.0
        
        # Calculate access patterns
        if hasattr(memory, 'access_counts') and filled > 0:
            access_counts = memory.access_counts[:filled].detach()
            access_mean = access_counts.float().mean().item()
            access_std = access_counts.float().std().item() if filled > 1 else 0.0
            access_max = access_counts.max().item()
            hot_memory_ratio = (access_counts > access_mean).sum().item() / filled
        else:
            access_mean = access_std = access_max = hot_memory_ratio = 0.0
        
        return {
            'memory_utilization': utilization,
            'memory_diversity': diversity,
            'importance_mean': importance_mean,
            'importance_std': importance_std,
            'importance_range': importance_max - importance_min,
            'access_mean': access_mean,
            'access_std': access_std,
            'access_max': access_max,
            'hot_memory_ratio': hot_memory_ratio
        }

    def analyze_attention_patterns(self, attn_weights, attn_variance):
        """
        Analyze attention patterns for better interpretability
        Args:
            attn_weights: Attention weights from model
            attn_variance: Attention variance metrics
        Returns:
            Dictionary of attention pattern metrics
        """
        metrics = {}
        
        # Process attention weights if available
        if isinstance(attn_weights, torch.Tensor):
            # Convert to numpy for easier analysis
            weights = attn_weights.detach().cpu().numpy()
            
            # Handle different attention weight formats
            if len(weights.shape) == 4:  # [batch, heads, seq_len, seq_len]
                batch_size, num_heads, seq_len, _ = weights.shape
                
                # Calculate entropy per head (higher = more uniform attention)
                entropy = 0
                for h in range(num_heads):
                    head_weights = weights[:, h, :, :]
                    # Add small epsilon to avoid log(0)
                    head_weights = head_weights + 1e-10
                    head_weights = head_weights / head_weights.sum(axis=-1, keepdims=True)
                    head_entropy = -np.sum(head_weights * np.log(head_weights), axis=-1).mean()
                    entropy += head_entropy
                entropy /= num_heads
                
                # Calculate attention concentration (higher = more focused)
                concentration = 0
                for h in range(num_heads):
                    head_weights = weights[:, h, :, :]
                    # Calculate max attention weight for each query
                    max_weights = head_weights.max(axis=-1).mean()
                    concentration += max_weights
                concentration /= num_heads
                
                # Calculate head diversity (how different the heads are from each other)
                head_diversity = 0
                if num_heads > 1:
                    for i in range(num_heads):
                        for j in range(i+1, num_heads):
                            head_i = weights[:, i, :, :].reshape(batch_size, -1)
                            head_j = weights[:, j, :, :].reshape(batch_size, -1)
                            # Calculate cosine distance between heads
                            similarity = np.sum(head_i * head_j, axis=-1) / (
                                np.linalg.norm(head_i, axis=-1) * np.linalg.norm(head_j, axis=-1) + 1e-10
                            )
                            head_diversity += 1.0 - similarity.mean()
                    head_diversity /= (num_heads * (num_heads - 1) / 2)
                
                metrics['attention_entropy'] = float(entropy)
                metrics['attention_concentration'] = float(concentration)
                metrics['head_diversity'] = float(head_diversity)
            
            elif len(weights.shape) == 3:  # [batch, seq_len, seq_len]
                # Calculate entropy (higher = more uniform attention)
                weights = weights + 1e-10
                weights = weights / weights.sum(axis=-1, keepdims=True)
                entropy = -np.sum(weights * np.log(weights), axis=-1).mean()
                
                # Calculate attention concentration
                concentration = weights.max(axis=-1).mean()
                
                metrics['attention_entropy'] = float(entropy)
                metrics['attention_concentration'] = float(concentration)
        
        # Process attention variance if available
        if isinstance(attn_variance, torch.Tensor):
            variance = attn_variance.detach().cpu().numpy()
            if isinstance(variance, np.ndarray):
                metrics['attention_variance_mean'] = float(variance.mean())
                if variance.size > 1:
                    metrics['attention_variance_std'] = float(variance.std())
            else:
                metrics['attention_variance_mean'] = float(variance)
        
        return metrics

    def analyze_gradient_health(self, model):
        """
        Analyze gradient health across model components
        Args:
            model: The model instance
        Returns:
            Dictionary of gradient health metrics
        """
        metrics = {}
        
        # Check if model has gradients
        has_grads = False
        for param in model.parameters():
            if param.grad is not None:
                has_grads = True
                break
        
        if not has_grads:
            return {'has_gradients': False}
        
        # Calculate gradient statistics for different model components
        components = {
            'encoder': [],
            'memory': [],
            'attention': [],
            'temporal': [],
            'prediction': []
        }
        
        # Collect gradients by component
        for name, param in model.named_parameters():
            if param.grad is not None:
                if 'encoder' in name:
                    components['encoder'].append(param.grad.detach().norm().item())
                elif 'memory' in name:
                    components['memory'].append(param.grad.detach().norm().item())
                elif 'attention' in name:
                    components['attention'].append(param.grad.detach().norm().item())
                elif 'temporal' in name or 'lstm' in name:
                    components['temporal'].append(param.grad.detach().norm().item())
                elif 'predict' in name or 'market_state' in name:
                    components['prediction'].append(param.grad.detach().norm().item())
        
        # Calculate statistics for each component
        for component, grads in components.items():
            if grads:
                metrics[f'{component}_grad_mean'] = np.mean(grads)
                metrics[f'{component}_grad_std'] = np.std(grads)
                metrics[f'{component}_grad_max'] = np.max(grads)
        
        # Calculate overall gradient health score
        component_means = [metrics.get(f'{c}_grad_mean', 0) for c in components.keys()]
        if component_means:
            # Calculate gradient balance - how evenly distributed are gradients across components
            if len(component_means) > 1:
                component_means = np.array(component_means)
                component_means = component_means[component_means > 0]  # Only consider components with gradients
                if len(component_means) > 1:
                    # Lower std/mean ratio indicates more balanced gradients
                    balance_score = 1.0 - min(1.0, np.std(component_means) / (np.mean(component_means) + 1e-10))
                    metrics['gradient_balance'] = float(balance_score)
            
            # Calculate overall gradient magnitude
            metrics['gradient_magnitude'] = float(np.mean(component_means))
        
        return metrics

    def enhanced_report(self, model=None):
        """
        Generate enhanced monitoring report with component analysis
        Args:
            model: Optional model instance for gradient analysis
        """
        print("\n=== Enhanced Monitoring Report ===")
        
        # Runtime and resource usage
        runtime = self.metrics.get('runtime', [0])[-1] if 'runtime' in self.metrics else 0
        gpu_memory = self.metrics.get('gpu_memory', [0])[-1] if 'gpu_memory' in self.metrics else 0
        print(f"Runtime: {runtime:.2f}s")
        print(f"Memory Usage: {gpu_memory / (1024**3):.2f}GB")
        
        # GPU stats if available
        if 'gpu_util' in self.metrics and 'gpu_temp' in self.metrics:
            gpu_util = self.metrics['gpu_util'][-1]
            gpu_temp = self.metrics['gpu_temp'][-1]
            print(f"GPU Utilization: {gpu_util:.2%}")
            print(f"GPU Temperature: {gpu_temp:.2f}°C")
        
        # Model gradient analysis if model provided
        if model is not None:
            grad_metrics = self.analyze_gradient_health(model)
            if grad_metrics.get('has_gradients', True):
                print("\nGradient Health:")
                for key in ['encoder_grad_mean', 'memory_grad_mean', 'attention_grad_mean', 
                            'temporal_grad_mean', 'prediction_grad_mean']:
                    if key in grad_metrics:
                        component = key.split('_')[0]
                        print(f"  {component.capitalize()}: {grad_metrics[key]:.4f}")
                
                if 'gradient_balance' in grad_metrics:
                    balance = grad_metrics['gradient_balance']
                    balance_status = "Good" if balance > 0.7 else "Moderate" if balance > 0.4 else "Poor"
                    print(f"  Balance: {balance:.4f} ({balance_status})")
        
        # Memory statistics if available
        memory_stats = {}
        for key, values in self.metrics.items():
            if key.startswith('memory_') or key.startswith('importance_') or key.startswith('access_'):
                memory_stats[key] = values[-1] if values else 0
        
        if memory_stats:
            print("\nMemory Performance:")
            if 'memory_utilization' in memory_stats:
                util = memory_stats['memory_utilization']
                print(f"  Utilization: {util:.2%}")
            if 'memory_diversity' in memory_stats:
                div = memory_stats['memory_diversity']
                print(f"  Key Diversity: {div:.4f}")
            if 'hot_memory_ratio' in memory_stats:
                hot_ratio = memory_stats['hot_memory_ratio']
                print(f"  Hot Memory Ratio: {hot_ratio:.2%}")
        
        # Attention analysis if available
        attn_stats = {}
        for key, values in self.metrics.items():
            if key.startswith('attention_') or key == 'head_diversity':
                attn_stats[key] = values[-1] if values else 0
        
        if attn_stats:
            print("\nAttention Analysis:")
            if 'attention_entropy' in attn_stats:
                entropy = attn_stats['attention_entropy']
                print(f"  Entropy: {entropy:.4f}")
            if 'attention_concentration' in attn_stats:
                conc = attn_stats['attention_concentration']
                print(f"  Concentration: {conc:.4f}")
            if 'head_diversity' in attn_stats:
                div = attn_stats['head_diversity']
                print(f"  Head Diversity: {div:.4f}")
        
        # Regime analysis if available
        regime_stats = {}
        for key, values in self.metrics.items():
            if key.startswith('regime_'):
                regime_stats[key] = values[-1] if values else 0
        
        if regime_stats:
            print("\nRegime Analysis:")
            if 'regime_stability' in regime_stats:
                stability = regime_stats['regime_stability']
                print(f"  Stability: {stability:.4f}")
            if 'regime_transitions' in regime_stats:
                transitions = regime_stats['regime_transitions']
                print(f"  Transitions: {transitions}")
            
            regime_dist = {k: v for k, v in regime_stats.items() if k.startswith('regime_distribution')}
            if regime_dist:
                print("  Distribution:")
                for regime, ratio in regime_dist.items():
                    regime_name = regime.split('_')[-1]
                    print(f"    Regime {regime_name}: {ratio:.2%}")
        
        print("=============================")

    def _calculate_attention_entropy(self, attention_weights):
        """
        Calculate entropy of attention weights
        
        Args:
            attention_weights: Attention weights tensor
            
        Returns:
            Entropy value
        """
        if isinstance(attention_weights, torch.Tensor):
            attention_weights = attention_weights.detach().cpu().numpy()
        
        # Ensure weights sum to 1
        if attention_weights.ndim > 1:
            attention_weights = attention_weights / (attention_weights.sum(axis=-1, keepdims=True) + 1e-10)
            
            # Calculate entropy
            entropy = -np.sum(attention_weights * np.log(attention_weights + 1e-10), axis=-1)
            return float(np.mean(entropy))
        
        return 0.0

    def visualize_information_flow(self, financial_data, financial_seq, output_dir=None):
        """Generate visualization of information flow through the model"""
        return self.flow_visualizer.create_comprehensive_flow_report(
            financial_data, financial_seq, output_dir
        )

def analyze_model(model, data_loader, num_samples=10, output_dir='monitoring'):
    # Existing code...
    
    # Add information flow analysis
    flow_viz = InformationFlowVisualizer(model, output_dir=os.path.join(output_dir, 'flow'))
    
    # Generate flow report for a batch of samples
    sample_batch = []
    for i, (data, _, _) in enumerate(data_loader):
        if i >= num_samples:
            break
        sample_batch.append(data[0].unsqueeze(0))
    
    flow_report = flow_viz.generate_flow_report(sample_batch)
    
    # Add to the report
    report['information_flow'] = {
        'visualization_path': os.path.join(output_dir, 'flow', 'average_information_flow.png'),
        'component_activations': {k: np.mean(v) for k, v in flow_report['component_activations'].items()}
    }
    
    # Continue with existing code...
