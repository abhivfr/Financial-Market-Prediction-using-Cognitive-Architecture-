#!/usr/bin/env python
# plot_engine.py - Visualization engine for cognitive architecture

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import os
from datetime import datetime

class CognitiveVisualizer:
    """
    Visualization engine for cognitive architecture, providing insight into internal processes
    """
    def __init__(self, model):
        """
        Initialize visualizer
        
        Args:
            model: Cognitive architecture model to visualize
        """
        self.model = model
        self.hooks = []
        self.activation_data = {}
        self.register_hooks()
    
    def register_hooks(self):
        """Register hooks to capture internal activations"""
        # Remove existing hooks
        self.remove_hooks()
        
        # Capture activations
        def hook_fn(name):
            def fn(module, input, output):
                # Store activation
                if isinstance(output, torch.Tensor):
                    self.activation_data[name] = output.detach().clone()
                elif isinstance(output, tuple) and isinstance(output[0], torch.Tensor):
                    self.activation_data[name] = output[0].detach().clone()
                    
            return fn
        
        # Register hooks for key components
        for name, module in self.model.named_modules():
            # Only attach hooks to specific module types/names
            if any(key in name for key in [
                'attention', 'memory', 'financial_encoder', 'temporal', 
                'fusion', 'market_state', 'meta_controller'
            ]):
                hook = module.register_forward_hook(hook_fn(name))
                self.hooks.append(hook)
    
    def remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.activation_data = {}
    
    def run_with_visualization(self, financial_data, financial_seq, volume=None):
        """
        Run model with visualization hooks
        
        Args:
            financial_data: Financial feature data
            financial_seq: Financial sequence data
            volume: Optional volume data
            
        Returns:
            Tuple of (model outputs, visualization data)
        """
        # Clear previous activation data
        self.activation_data = {}
        
        # Set model to eval mode for visualization
        self.model.eval()
        
        # Forward pass
        outputs = self.model(financial_data=financial_data, financial_seq=financial_seq, volume=volume)
        
        # Extract visualization data
        visual_data = self._process_activations()
        
        # Return both outputs and visualization data
        return outputs, visual_data
    
    def _process_activations(self):
        """
        Process captured activations for visualization
        
        Returns:
            Dictionary of processed visualization data
        """
        visual_data = {}
        
        # Extract attention weights if available
        for name, activation in self.activation_data.items():
            if 'attention' in name and activation.ndim >= 3:
                # For attention modules, extract weights
                if activation.ndim == 4:  # [batch, heads, seq, seq]
                    # Multi-head attention, average across heads
                    weights = activation.mean(dim=1)
                else:  # [batch, seq, seq]
                    weights = activation
                
                # Average across batch dimension
                if weights.ndim >= 3:
                    weights = weights.mean(dim=0)
                
                visual_data['attention_weights'] = weights.cpu().numpy()
                break
        
        # Extract memory activities if available
        memory_activations = []
        for name, activation in self.activation_data.items():
            if 'memory' in name:
                # For memory modules, extract activations
                if activation.ndim >= 2:
                    # Get the memory cell activities
                    # Reshape if needed to [cells, time]
                    if activation.ndim > 2:
                        # Reshape based on context
                        if 'bank' in name or 'cell' in name:
                            # Likely [batch, cells, features]
                            mem_act = activation.mean(dim=0)
                        else:
                            # Try to infer the right reshape
                            mem_act = activation.view(-1, activation.size(-1)).t()
                    else:
                        mem_act = activation.t()  # Transpose to [cells, time]
                    
                    memory_activations.append(mem_act.cpu().numpy())
        
        # Combine memory activations if multiple found
        if memory_activations:
            # Use the one with most variance for visualization
            mem_variances = [np.var(mem) for mem in memory_activations]
            visual_data['memory_activity'] = memory_activations[np.argmax(mem_variances)]
        
        return visual_data
    
    def visualize_attention(self, attention_weights, feature_names=None, output_path=None):
        """
        Visualize attention weights
        
        Args:
            attention_weights: Attention weights matrix
            feature_names: Optional list of feature names
            output_path: Path to save visualization
            
        Returns:
            Matplotlib figure
        """
        # Create figure
        plt.figure(figsize=(12, 10))
        
        # Create custom colormap
        colors = ["#f7fbff", "#4393c3", "#2166ac"]
        cmap = LinearSegmentedColormap.from_list("custom_blue", colors)
        
        # Create default feature names if not provided
        if feature_names is None:
            feature_names = [f"Feature {i+1}" for i in range(attention_weights.shape[0])]
        
        # Create heatmap
        sns.heatmap(
            attention_weights,
            cmap=cmap,
            xticklabels=range(attention_weights.shape[1]),
            yticklabels=feature_names[:attention_weights.shape[0]],
            annot=False,
            cbar_kws={"label": "Attention Weight"}
        )
        
        # Set title and labels
        plt.title("Attention Weights Heatmap")
        plt.xlabel("Sequence Position")
        plt.ylabel("Feature")
        plt.tight_layout()
        
        # Save if output path provided
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        fig = plt.gcf()
        plt.close()
        
        return fig

    def visualize_predictions(self, predictions, targets, timestamps=None, output_path=None):
        """
        Visualize model predictions against targets
        
        Args:
            predictions: Model predictions
            targets: Target values
            timestamps: Optional timestamps for x-axis
            output_path: Path to save visualization
            
        Returns:
            Matplotlib figure
        """
        # Convert to numpy if needed
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()
        
        # Ensure 2D arrays
        if predictions.ndim > 2:
            predictions = predictions.reshape(predictions.shape[0], -1)
        if targets.ndim > 2:
            targets = targets.reshape(targets.shape[0], -1)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 7))
        
        # Create x-axis values
        x = range(len(predictions)) if timestamps is None else timestamps
        
        # Plot predictions and targets (first feature, usually price)
        ax.plot(x, predictions[:, 0], 'b-', label='Predicted', linewidth=2)
        ax.plot(x, targets[:, 0], 'r-', label='Actual', linewidth=2)
        
        # Calculate metrics
        mse = np.mean((predictions[:, 0] - targets[:, 0])**2)
        corr = np.corrcoef(predictions[:, 0], targets[:, 0])[0, 1]
        
        # Add metrics to title
        plt.title(f"Prediction vs Actual (MSE: {mse:.4f}, Correlation: {corr:.4f})")
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save if output path provided
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        return fig

    def visualize_memory_usage(self, memory_activity, output_path=None):
        """
        Visualize memory cell usage over time
        
        Args:
            memory_activity: Memory activity matrix [cells, time]
            output_path: Path to save visualization
            
        Returns:
            Matplotlib figure
        """
        # Create figure
        plt.figure(figsize=(14, 6))
        
        # Create heatmap
        sns.heatmap(
            memory_activity,
            cmap="viridis",
            xticklabels=range(0, memory_activity.shape[1], max(1, memory_activity.shape[1] // 10)),
            yticklabels=[f"Cell {i+1}" for i in range(memory_activity.shape[0])],
            annot=False,
            cbar_kws={"label": "Activity Level"}
        )
        
        # Set title and labels
        plt.title("Memory Cell Activity Over Time")
        plt.xlabel('Time Step')
        plt.ylabel('Memory Cell')
        plt.tight_layout()
        
        # Save if output path provided
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        fig = plt.gcf()
        plt.close()
        
        return fig

    def create_component_activation_summary(self, output_dir):
        """
        Create summary visualizations of component activations
        
        Args:
            output_dir: Directory to save visualizations
        """
        # Create directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Process each recorded activation
        for name, activation in self.activation_data.items():
            # Skip non-tensor activations
            if not isinstance(activation, torch.Tensor):
                continue
            
            # Convert to numpy
            act_data = activation.detach().cpu().numpy()
            
            # Skip if empty
            if act_data.size == 0:
                continue
            
            # Create activation summary based on tensor shape
            if act_data.ndim == 2:  # [batch, features] or [features, time]
                # Create heatmap
                plt.figure(figsize=(12, 6))
                sns.heatmap(
                    act_data,
                    cmap="viridis",
                    annot=False,
                    cbar_kws={"label": "Activation"}
                )
                plt.title(f"{name} Activation Pattern")
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"{name}_heatmap.png"), dpi=300)
                plt.close()
                
                # Create distribution plot
                plt.figure(figsize=(10, 6))
                sns.histplot(act_data.flatten(), bins=30, kde=True)
                plt.title(f"{name} Activation Distribution")
                plt.xlabel("Activation Value")
                plt.ylabel("Frequency")
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"{name}_distribution.png"), dpi=300)
                plt.close()
            
            elif act_data.ndim >= 3:  # Higher dimensional tensors
                # Create summary statistics
                act_mean = np.mean(act_data, axis=tuple(range(act_data.ndim - 1)))
                act_std = np.std(act_data, axis=tuple(range(act_data.ndim - 1)))
                
                # Create line plot of mean activations
                plt.figure(figsize=(12, 6))
                plt.plot(act_mean, 'b-', linewidth=2)
                plt.fill_between(
                    range(len(act_mean)),
                    act_mean - act_std,
                    act_mean + act_std,
                    alpha=0.3
                )
                plt.title(f"{name} Mean Activation (±σ)")
                plt.xlabel("Feature Index")
                plt.ylabel("Activation Value")
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"{name}_profile.png"), dpi=300)
                plt.close()

    def plot_4d_attention(self, attention_weights, output_path=None):
        """
        Visualize 4D attention weights (batch, heads, seq, seq)
        
        Args:
            attention_weights: 4D attention weights tensor
            output_path: Optional path to save visualization
            
        Returns:
            Matplotlib figure
        """
        # Convert to numpy if needed
        if isinstance(attention_weights, torch.Tensor):
            attention_weights = attention_weights.detach().cpu().numpy()
        
        # Get dimensions
        batch_size, num_heads, seq_len, _ = attention_weights.shape
        
        # Create figure
        fig, axes = plt.subplots(nrows=min(batch_size, 2), ncols=min(num_heads, 4), 
                                 figsize=(12, 8))
        
        # Handle single axes case
        if batch_size == 1 and num_heads == 1:
            axes = np.array([[axes]])
        elif batch_size == 1:
            axes = axes.reshape(1, -1)
        elif num_heads == 1:
            axes = axes.reshape(-1, 1)
        
        # Plot attention for each head and batch (limited to avoid too many plots)
        for b in range(min(batch_size, 2)):
            for h in range(min(num_heads, 4)):
                ax = axes[b, h]
                im = ax.imshow(attention_weights[b, h], cmap='viridis', aspect='auto')
                ax.set_title(f'Batch {b}, Head {h}')
                ax.set_xticks([])
                ax.set_yticks([])
        
        # Add colorbar
        plt.colorbar(im, ax=axes.ravel().tolist())
        
        plt.tight_layout()
        
        # Save if output path provided
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        fig = plt.gcf()
        plt.close()
        
        return fig

# Alias for backward compatibility
VisualizationEngine = CognitiveVisualizer