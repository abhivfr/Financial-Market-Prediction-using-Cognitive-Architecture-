#!/usr/bin/env python
# interpretability.py - Tools for model interpretability

import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import os

class FeatureAttributionAnalyzer:
    """
    Analyze feature attribution for model predictions
    """
    def __init__(self, model: torch.nn.Module, baseline_value: float = 0.0):
        """
        Initialize feature attribution analyzer
        
        Args:
            model: PyTorch model
            baseline_value: Baseline value for integrated gradients
        """
        self.model = model
        self.baseline_value = baseline_value
        self.hooks = []
        self.gradients = {}
        self.activations = {}
    
    def _register_hooks(self) -> None:
        """Register hooks for gradient and activation collection"""
        # Clear any existing hooks
        self._remove_hooks()
        
        # Activation hook
        def activation_hook(name):
            def hook(module, input, output):
                self.activations[name] = output
            return hook
        
        # Gradient hook
        def gradient_hook(name):
            def hook(module, grad_in, grad_out):
                self.gradients[name] = grad_out[0]
            return hook
        
        # Register hooks for each module
        for name, module in self.model.named_modules():
            if any(x in name for x in ['encoder', 'attention', 'temporal', 'memory']):
                self.hooks.append(module.register_forward_hook(activation_hook(name)))
                self.hooks.append(module.register_backward_hook(gradient_hook(name)))
    
    def _remove_hooks(self) -> None:
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.gradients = {}
        self.activations = {}
    
    def compute_integrated_gradients(
        self, 
        inputs: Union[torch.Tensor, Dict[str, torch.Tensor]], 
        target_idx: int = 0,
        steps: int = 20
    ) -> Dict[str, np.ndarray]:
        """
        Compute integrated gradients for feature attribution
        
        Args:
            inputs: Model inputs (tensor or dictionary of tensors)
            target_idx: Index of the target output to explain
            steps: Number of steps for path integral
            
        Returns:
            Dictionary of integrated gradients for inputs
        """
        self._register_hooks()
        
        # Handle dictionary inputs
        if isinstance(inputs, dict):
            # Extract key input tensors
            sequence = inputs.get('sequence', None)
            features = inputs.get('features', None)
            
            # Create baseline inputs
            baseline_inputs = {}
            path_inputs = {}
            
            for key, tensor in inputs.items():
                baseline = torch.ones_like(tensor) * self.baseline_value
                baseline_inputs[key] = baseline
                
                # Create path inputs
                path_inputs[key] = [
                    baseline + (tensor - baseline) * float(i) / steps
                    for i in range(steps + 1)
                ]
            
            # Compute integrated gradients along the path
            integrated_grads = {
                key: torch.zeros_like(tensor).float()
                for key, tensor in inputs.items()
            }
            
            for i in range(steps):
                # Create current inputs
                current_inputs = {
                    key: path_inputs[key][i]
                    for key in inputs.keys()
                }
                
                # Forward pass
                current_inputs = {
                    key: tensor.requires_grad_(True)
                    for key, tensor in current_inputs.items()
                }
                
                outputs = self.model(**current_inputs)
                
                # Ensure outputs is usable for backprop
                if isinstance(outputs, dict):
                    if 'market_state' in outputs:
                        pred = outputs['market_state']
                    else:
                        # Use first tensor in the dict
                        pred = next(iter(outputs.values()))
                else:
                    pred = outputs
                
                # Zero gradients
                self.model.zero_grad()
                
                # Get target prediction and backward
                target = pred[:, target_idx]
                target.sum().backward()
                
                # Accumulate gradients
                for key, tensor in inputs.items():
                    if key in current_inputs and current_inputs[key].grad is not None:
                        # Get step size
                        step_size = (tensor - baseline_inputs[key]) / steps
                        # Accumulate gradients * step_size
                        integrated_grads[key] += current_inputs[key].grad * step_size
        
        else:
            # Handle single tensor input
            tensor = inputs
            baseline = torch.ones_like(tensor) * self.baseline_value
            
            # Create interpolation path
            path_inputs = [
                baseline + (tensor - baseline) * float(i) / steps
                for i in range(steps + 1)
            ]
            
            # Initialize integrated gradients
            integrated_grads = torch.zeros_like(tensor).float()
            
            for i in range(steps):
                # Get current input
                current_input = path_inputs[i].requires_grad_(True)
                
                # Forward pass
                outputs = self.model(current_input)
                
                # Ensure outputs is usable for backprop
                if isinstance(outputs, dict):
                    if 'market_state' in outputs:
                        pred = outputs['market_state']
                    else:
                        # Use first tensor in the dict
                        pred = next(iter(outputs.values()))
                else:
                    pred = outputs
                
                # Zero gradients
                self.model.zero_grad()
                
                # Get target prediction and backward
                target = pred[:, target_idx]
                target.sum().backward()
                
                # Accumulate gradients
                if current_input.grad is not None:
                    step_size = (tensor - baseline) / steps
                    integrated_grads += current_input.grad * step_size
        
        # Clean up
        self._remove_hooks()
        
        # Convert to numpy for easier handling
        if isinstance(integrated_grads, dict):
            return {
                key: grads.detach().cpu().numpy()
                for key, grads in integrated_grads.items()
            }
        else:
            return integrated_grads.detach().cpu().numpy()
    
    def compute_gradient_times_input(
        self, 
        inputs: Union[torch.Tensor, Dict[str, torch.Tensor]], 
        target_idx: int = 0
    ) -> Dict[str, np.ndarray]:
        """
        Compute gradient times input for feature attribution
        
        Args:
            inputs: Model inputs (tensor or dictionary of tensors)
            target_idx: Index of the target output to explain
            
        Returns:
            Dictionary of gradient * input for feature attribution
        """
        self._register_hooks()
        
        # Handle dictionary inputs
        if isinstance(inputs, dict):
            # Ensure inputs require gradients
            inputs = {
                key: tensor.requires_grad_(True) if tensor.requires_grad == False else tensor
                for key, tensor in inputs.items()
            }
            
            # Forward pass
            outputs = self.model(**inputs)
            
            # Ensure outputs is usable for backprop
            if isinstance(outputs, dict):
                if 'market_state' in outputs:
                    pred = outputs['market_state']
                else:
                    # Use first tensor in the dict
                    pred = next(iter(outputs.values()))
            else:
                pred = outputs
            
            # Zero gradients
            self.model.zero_grad()
            
            # Get target prediction and backward
            target = pred[:, target_idx]
            target.sum().backward()
            
            # Compute gradient * input
            grad_times_input = {
                key: (tensor * tensor.grad).detach().cpu().numpy()
                if tensor.grad is not None else np.zeros_like(tensor.detach().cpu().numpy())
                for key, tensor in inputs.items()
            }
            
        else:
            # Handle single tensor input
            inputs = inputs.requires_grad_(True) if inputs.requires_grad == False else inputs
            
            # Forward pass
            outputs = self.model(inputs)
            
            # Ensure outputs is usable for backprop
            if isinstance(outputs, dict):
                if 'market_state' in outputs:
                    pred = outputs['market_state']
                else:
                    # Use first tensor in the dict
                    pred = next(iter(outputs.values()))
            else:
                pred = outputs
            
            # Zero gradients
            self.model.zero_grad()
            
            # Get target prediction and backward
            target = pred[:, target_idx]
            target.sum().backward()
            
            # Compute gradient * input
            if inputs.grad is not None:
                grad_times_input = (inputs * inputs.grad).detach().cpu().numpy()
            else:
                grad_times_input = np.zeros_like(inputs.detach().cpu().numpy())
        
        # Clean up
        self._remove_hooks()
        
        return grad_times_input
    
    def generate_feature_attribution_plot(
        self,
        attributions: np.ndarray,
        feature_names: List[str],
        title: str = "Feature Attribution",
        output_path: Optional[str] = None,
        top_k: int = None
    ) -> plt.Figure:
        """
        Generate feature attribution plot
        
        Args:
            attributions: Feature attributions (numpy array)
            feature_names: Names of features
            title: Plot title
            output_path: Optional path to save the plot
            top_k: Show only top k features by absolute attribution
            
        Returns:
            Matplotlib figure
        """
        # Ensure attributions is 1D
        if attributions.ndim > 1:
            attributions = attributions.mean(axis=tuple(range(attributions.ndim - 1)))
        
        # Ensure feature_names matches attributions length
        if len(feature_names) != len(attributions):
            feature_names = [f"Feature {i}" for i in range(len(attributions))]
        
        # Get top k features by absolute attribution if specified
        if top_k is not None and top_k < len(attributions):
            indices = np.argsort(np.abs(attributions))[-top_k:]
            attributions = attributions[indices]
            feature_names = [feature_names[i] for i in indices]
        
        # Create DataFrame for plotting
        df = pd.DataFrame({
            'Feature': feature_names,
            'Attribution': attributions
        })
        
        # Sort by absolute attribution
        df = df.reindex(df['Attribution'].abs().sort_values().index)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot horizontal bars
        bars = ax.barh(
            df['Feature'],
            df['Attribution'],
            color=df['Attribution'].apply(lambda x: 'green' if x > 0 else 'red')
        )
        
        # Add values as text labels
        for bar in bars:
            width = bar.get_width()
            label_x_pos = width if width > 0 else width - 0.05
            ax.text(
                label_x_pos, 
                bar.get_y() + bar.get_height()/2, 
                f'{width:.4f}', 
                va='center'
            )
        
        # Set title and labels
        ax.set_title(title)
        ax.set_xlabel('Attribution Value')
        ax.set_ylabel('Feature')
        
        # Set grid
        ax.grid(axis='x', linestyle='--', alpha=0.6)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save if output path is provided
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def visualize_attention_attributions(
        self,
        attention_weights: torch.Tensor,
        input_features: List[str],
        title: str = "Attention Weight Attribution",
        output_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Visualize attention weights as feature attributions
        
        Args:
            attention_weights: Attention weights tensor [batch, seq_len, seq_len]
            input_features: Names of input features
            title: Plot title
            output_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure
        """
        # Convert to numpy
        if isinstance(attention_weights, torch.Tensor):
            attention_weights = attention_weights.detach().cpu().numpy()
        
        # Average across batch dimension if needed
        if attention_weights.ndim > 2:
            attention_weights = attention_weights.mean(axis=0)
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Create custom colormap
        colors = ["#f7fbff", "#4393c3", "#2166ac"]
        cmap = plt.cm.get_cmap('Blues')
        
        # Create feature labels if needed
        if input_features is None or len(input_features) != attention_weights.shape[0]:
            input_features = [f"Feature {i+1}" for i in range(attention_weights.shape[0])]
        
        # Plot heatmap
        sns.heatmap(
            attention_weights,
            cmap=cmap,
            annot=True,
            fmt=".2f",
            linewidths=0.5,
            ax=ax,
            xticklabels=range(1, attention_weights.shape[1] + 1),
            yticklabels=input_features
        )
        
        # Set title and labels
        ax.set_title(title)
        ax.set_xlabel("Sequence Position")
        ax.set_ylabel("Feature")
        
        # Adjust layout
        plt.tight_layout()
        
        # Save if output path is provided
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def visualize_embedding_space(
        self,
        activations: torch.Tensor,
        labels: Optional[Union[torch.Tensor, np.ndarray]] = None,
        method: str = 'tsne',
        title: str = "Embedding Space Visualization",
        output_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Visualize embedding space using dimensionality reduction
        
        Args:
            activations: Model activations/embeddings
            labels: Optional labels for coloring points
            method: Dimensionality reduction method ('tsne' or 'pca')
            title: Plot title
            output_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure
        """
        # Convert to numpy
        if isinstance(activations, torch.Tensor):
            activations = activations.detach().cpu().numpy()
        
        # Reshape if needed
        if activations.ndim > 2:
            activations = activations.reshape(activations.shape[0], -1)
        
        # Apply dimensionality reduction
        if method.lower() == 'tsne':
            reducer = TSNE(n_components=2, random_state=42)
        else:  # PCA
            reducer = PCA(n_components=2, random_state=42)
        
        # Reduce dimensionality
        reduced_data = reducer.fit_transform(activations)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot with or without labels
        if labels is not None:
            if isinstance(labels, torch.Tensor):
                labels = labels.detach().cpu().numpy()
            
            # Get unique labels
            unique_labels = np.unique(labels)
            
            # Plot each label with different color
            for label in unique_labels:
                mask = labels == label
                ax.scatter(
                    reduced_data[mask, 0],
                    reduced_data[mask, 1],
                    label=f"Class {label}",
                    alpha=0.7,
                    edgecolors='w',
                    linewidth=0.5
                )
            
            ax.legend()
        else:
            ax.scatter(
                reduced_data[:, 0],
                reduced_data[:, 1],
                alpha=0.7,
                edgecolors='w',
                linewidth=0.5
            )
        
        # Set title and labels
        ax.set_title(title)
        ax.set_xlabel(f"Dimension 1 ({method.upper()})")
        ax.set_ylabel(f"Dimension 2 ({method.upper()})")
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save if output path is provided
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        return fig

class InterpretabilityReport:
    """Generate comprehensive interpretability report for model predictions"""
    
    def __init__(self, model: torch.nn.Module, output_dir: str = "interpretability_reports"):
        """
        Initialize interpretability report generator
        
        Args:
            model: PyTorch model
            output_dir: Directory to save reports
        """
        self.model = model
        self.output_dir = output_dir
        self.feature_attributor = FeatureAttributionAnalyzer(model)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
    
    def generate_report(
        self,
        inputs: Union[torch.Tensor, Dict[str, torch.Tensor]],
        feature_names: List[str],
        timestamp: Optional[str] = None,
        target_idx: int = 0
    ) -> Dict[str, Any]:
        """
        Generate comprehensive interpretability report
        
        Args:
            inputs: Model inputs
            feature_names: Names of features
            timestamp: Optional timestamp for the report (default: current time)
            target_idx: Index of target to explain
            
        Returns:
            Dictionary of report elements and paths
        """
        # Create timestamp if not provided
        if timestamp is None:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create report directory
        report_dir = os.path.join(self.output_dir, f"report_{timestamp}")
        os.makedirs(report_dir, exist_ok=True)
        
        # Report elements
        report = {
            "timestamp": timestamp,
            "report_dir": report_dir,
            "plots": {},
            "data": {}
        }
        
        # Run model to get prediction
        with torch.no_grad():
            outputs = self.model(**inputs) if isinstance(inputs, dict) else self.model(inputs)
        
        # Extract prediction
        if isinstance(outputs, dict):
            if 'market_state' in outputs:
                prediction = outputs['market_state']
            else:
                prediction = next(iter(outputs.values()))
        else:
            prediction = outputs
        
        # Store prediction
        report["data"]["prediction"] = prediction.detach().cpu().numpy()
        
        # 1. Compute integrated gradients
        ig_attributions = self.feature_attributor.compute_integrated_gradients(
            inputs, target_idx=target_idx, steps=20
        )
        
        # Store attributions
        report["data"]["integrated_gradients"] = ig_attributions
        
        # Generate plots for integrated gradients
        if isinstance(ig_attributions, dict):
            # For dictionary inputs
            for key, attrs in ig_attributions.items():
                if attrs.ndim <= 2:  # Skip tensors with more than 2 dims
                    plot_path = os.path.join(report_dir, f"ig_attribution_{key}.png")
                    fig = self.feature_attributor.generate_feature_attribution_plot(
                        attrs, feature_names, 
                        title=f"Integrated Gradients: {key}",
                        output_path=plot_path
                    )
                    report["plots"][f"ig_{key}"] = plot_path
        else:
            # For single tensor input
            plot_path = os.path.join(report_dir, "ig_attribution.png")
            fig = self.feature_attributor.generate_feature_attribution_plot(
                ig_attributions, feature_names, 
                title="Integrated Gradients",
                output_path=plot_path
            )
            report["plots"]["ig"] = plot_path
        
        # 2. Compute gradient * input
        grad_input_attributions = self.feature_attributor.compute_gradient_times_input(
            inputs, target_idx=target_idx
        )
        
        # Store attributions
        report["data"]["gradient_input"] = grad_input_attributions
        
        # Generate plots for gradient * input
        if isinstance(grad_input_attributions, dict):
            # For dictionary inputs
            for key, attrs in grad_input_attributions.items():
                if attrs.ndim <= 2:  # Skip tensors with more than 2 dims
                    plot_path = os.path.join(report_dir, f"grad_input_{key}.png")
                    fig = self.feature_attributor.generate_feature_attribution_plot(
                        attrs, feature_names, 
                        title=f"Gradient * Input: {key}",
                        output_path=plot_path
                    )
                    report["plots"][f"grad_input_{key}"] = plot_path
        else:
            # For single tensor input
            plot_path = os.path.join(report_dir, "grad_input_attribution.png")
            fig = self.feature_attributor.generate_feature_attribution_plot(
                grad_input_attributions, feature_names, 
                title="Gradient * Input",
                output_path=plot_path
            )
            report["plots"]["grad_input"] = plot_path
        
        # 3. Visualize attention weights if available
        if isinstance(outputs, dict) and 'attention_weights' in outputs:
            attention_weights = outputs['attention_weights']
            
            plot_path = os.path.join(report_dir, "attention_weights.png")
            fig = self.feature_attributor.visualize_attention_attributions(
                attention_weights, feature_names,
                title="Attention Weights",
                output_path=plot_path
            )
            report["plots"]["attention"] = plot_path
        
        # 4. Generate embedding visualization if applicable
        if isinstance(outputs, dict) and 'embeddings' in outputs:
            embeddings = outputs['embeddings']
            
            plot_path = os.path.join(report_dir, "embedding_space.png")
            fig = self.feature_attributor.visualize_embedding_space(
                embeddings, 
                title="Embedding Space (t-SNE)",
                output_path=plot_path
            )
            report["plots"]["embedding"] = plot_path
        
        # 5. Generate HTML report
        html_path = os.path.join(report_dir, "report.html")
        self._generate_html_report(report, html_path)
        report["html"] = html_path
        
        return report
    
    def _generate_html_report(self, report: Dict[str, Any], output_path: str) -> None:
        """
        Generate HTML report
        
        Args:
            report: Report data
            output_path: Output HTML file path
        """
        # Start HTML
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Model Interpretability Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1, h2, h3 {{ color: #333; }}
        .report-section {{ margin-bottom: 30px; }}
        .plot-container {{ display: flex; flex-wrap: wrap; gap: 20px; }}
        .plot-item {{ flex: 1; min-width: 300px; margin-bottom: 20px; }}
        .plot-item img {{ max-width: 100%; border: 1px solid #ddd; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
    </style>
</head>
<body>
    <h1>Model Interpretability Report</h1>
    <p>Generated on: {report["timestamp"]}</p>
    
    <div class="report-section">
        <h2>Feature Attribution</h2>
        <div class="plot-container">
"""
        
        # Add feature attribution plots
        for key, path in report["plots"].items():
            if "ig_" in key or "grad_input_" in key:
                rel_path = os.path.basename(path)
                html += f"""
            <div class="plot-item">
                <h3>{key.replace('_', ' ').title()}</h3>
                <img src="{rel_path}" alt="{key}">
            </div>
"""
        
        html += """
        </div>
    </div>
"""
        
        # Add attention visualization if available
        if "attention" in report["plots"]:
            rel_path = os.path.basename(report["plots"]["attention"])
            html += f"""
    <div class="report-section">
        <h2>Attention Visualization</h2>
        <div class="plot-container">
            <div class="plot-item">
                <img src="{rel_path}" alt="Attention Weights">
            </div>
        </div>
    </div>
"""
        
        # Add embedding visualization if available
        if "embedding" in report["plots"]:
            rel_path = os.path.basename(report["plots"]["embedding"])
            html += f"""
    <div class="report-section">
        <h2>Embedding Space Visualization</h2>
        <div class="plot-container">
            <div class="plot-item">
                <img src="{rel_path}" alt="Embedding Space">
            </div>
        </div>
    </div>
"""
        
        # Close HTML
        html += """
</body>
</html>
"""
        
        # Write HTML to file
        with open(output_path, 'w') as f:
            f.write(html)
