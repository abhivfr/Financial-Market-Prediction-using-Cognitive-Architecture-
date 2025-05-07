#!/usr/bin/env python
# setup_monitoring.py - Setup monitoring infrastructure for cognitive models

import os
import sys
import argparse
import torch
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging
from tqdm import tqdm

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, root_dir)

from src.arch.cognitive import CognitiveArchitecture
from src.monitoring.introspect import ModelIntrospector

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('monitoring.log')
    ]
)
logger = logging.getLogger('monitoring')

class ModelMonitor:
    """Monitoring infrastructure for cognitive models"""
    
    def __init__(
        self,
        model: torch.nn.Module,
        model_type: str,
        output_dir: str,
        device: str = "cpu"
    ):
        """
        Initialize model monitor
        
        Args:
            model: PyTorch model
            model_type: Model type ('cognitive' or 'baseline')
            output_dir: Directory to save monitoring data
            device: Computation device
        """
        self.model = model
        self.model_type = model_type
        self.output_dir = output_dir
        self.device = device
        
        # Create introspector if cognitive model
        self.introspector = None
        if model_type.lower() == "cognitive":
            self.introspector = ModelIntrospector(model)
        
        # Create output directories
        self.logs_dir = os.path.join(output_dir, "logs")
        self.viz_dir = os.path.join(output_dir, "visualizations")
        self.data_dir = os.path.join(output_dir, "data")
        
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.viz_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Initialize records
        self.prediction_records = []
        self.activation_records = []
        self.attention_records = []
        self.memory_records = []
        
        logger.info(f"Initialized model monitoring in {output_dir}")
    
    def setup_component_monitoring(self):
        """Setup component-specific monitoring"""
        if self.model_type.lower() != "cognitive":
            logger.info("Component monitoring only available for cognitive models")
            return
        
        # Register hooks for activation monitoring
        self.activation_hooks = []
        
        def save_activation(name):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    self.activation_records.append({
                        'name': name,
                        'timestamp': datetime.now().isoformat(),
                        'mean': output.mean().item(),
                        'std': output.std().item(),
                        'min': output.min().item(),
                        'max': output.max().item(),
                        'shape': list(output.shape)
                    })
            return hook
        
        # Register hooks for interesting modules
        for name, module in self.model.named_modules():
            if any(key in name for key in ['attention', 'memory', 'introspection', 'regime']):
                hook = module.register_forward_hook(save_activation(name))
                self.activation_hooks.append(hook)
        
        logger.info(f"Registered activation hooks for {len(self.activation_hooks)} modules")
    
    def record_prediction(
        self,
        financial_data: torch.Tensor,
        financial_seq: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Record a prediction
        
        Args:
            financial_data: Financial data tensor
            financial_seq: Financial sequence tensor
            target: Target tensor (optional)
            metadata: Additional metadata (optional)
            
        Returns:
            Prediction data
        """
        # Ensure model is in eval mode
        self.model.eval()
        
        # Move tensors to device
        financial_data = financial_data.to(self.device)
        financial_seq = financial_seq.to(self.device)
        if target is not None:
            target = target.to(self.device)
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(financial_data=financial_data, financial_seq=financial_seq)
        
        # Extract prediction
        if isinstance(outputs, dict) and 'market_state' in outputs:
            prediction = outputs['market_state'].cpu().detach().numpy()
            confidence = outputs.get('confidence', torch.ones_like(outputs['market_state'])).cpu().detach().numpy()
        else:
            prediction = outputs.cpu().detach().numpy()
            confidence = np.ones_like(prediction)
        
        # Calculate error if target is provided
        error = None
        if target is not None:
            if isinstance(outputs, dict) and 'market_state' in outputs:
                error = torch.nn.functional.mse_loss(outputs['market_state'], target).item()
            else:
                error = torch.nn.functional.mse_loss(outputs, target).item()
        
        # Record prediction
        timestamp = datetime.now().isoformat()
        
        record = {
            'timestamp': timestamp,
            'prediction': prediction.tolist() if hasattr(prediction, 'tolist') else prediction,
            'confidence': confidence.tolist() if hasattr(confidence, 'tolist') else confidence,
            'error': error
        }
        
        # Add metadata if provided
        if metadata:
            record['metadata'] = metadata
        
        # Add to records
        self.prediction_records.append(record)
        
        # Record introspection data if available
        if self.introspector:
            introspection = self.introspector.introspect_model(
                financial_data=financial_data,
                financial_seq=financial_seq
            )
            
            # Record attention information
            if 'attention_weights' in introspection:
                attention_weights = introspection['attention_weights'].cpu().detach().numpy()
                
                self.attention_records.append({
                    'timestamp': timestamp,
                    'weights': attention_weights.tolist() if hasattr(attention_weights, 'tolist') else attention_weights,
                    'entropy': (-np.sum(attention_weights * np.log(attention_weights + 1e-10), axis=-1)).tolist()
                })
            
            # Record memory information
            if 'memory_usage' in introspection:
                memory_usage = introspection['memory_usage'].cpu().detach().numpy()
                
                self.memory_records.append({
                    'timestamp': timestamp,
                    'usage': memory_usage.tolist() if hasattr(memory_usage, 'tolist') else memory_usage,
                    'diversity': np.std(memory_usage).item(),
                    'max_usage': np.max(memory_usage).item()
                })
        
        return record
    
    def analyze_predictions(self, save: bool = True) -> Dict[str, Any]:
        """
        Analyze prediction records
        
        Args:
            save: Whether to save analysis
            
        Returns:
            Analysis results
        """
        if not self.prediction_records:
            logger.warning("No prediction records to analyze")
            return {}
        
        # Extract data
        predictions = [r['prediction'] for r in self.prediction_records]
        confidences = [r['confidence'] for r in self.prediction_records]
        errors = [r['error'] for r in self.prediction_records if r['error'] is not None]
        
        # Flatten predictions and confidences if they're lists/arrays
        flat_predictions = []
        flat_confidences = []
        
        for p, c in zip(predictions, confidences):
            if isinstance(p, list):
                if isinstance(p[0], list):
                    # 2D list
                    flat_predictions.extend(p[0])
                    if isinstance(c, list) and isinstance(c[0], list):
                        flat_confidences.extend(c[0])
                    else:
                        flat_confidences.extend([c] * len(p[0]))
                else:
                    # 1D list
                    flat_predictions.append(p[0] if len(p) > 0 else 0)
                    flat_confidences.append(c[0] if isinstance(c, list) and len(c) > 0 else c)
            else:
                flat_predictions.append(p)
                flat_confidences.append(c)
        
        # Calculate statistics
        prediction_mean = np.mean(flat_predictions) if flat_predictions else np.nan
        prediction_std = np.std(flat_predictions) if flat_predictions else np.nan
        confidence_mean = np.mean(flat_confidences) if flat_confidences else np.nan
        error_mean = np.mean(errors) if errors else np.nan
        error_std = np.std(errors) if errors else np.nan
        
        # Calculate confidence calibration
        confidence_error_correlation = np.nan
        if errors and len(flat_confidences) == len(errors):
            confidence_error_correlation = np.corrcoef(flat_confidences, errors)[0, 1]
        
        # Create analysis results
        analysis = {
            'prediction_stats': {
                'mean': float(prediction_mean),
                'std': float(prediction_std),
                'min': float(np.min(flat_predictions)) if flat_predictions else np.nan,
                'max': float(np.max(flat_predictions)) if flat_predictions else np.nan
            },
            'confidence_stats': {
                'mean': float(confidence_mean),
                'min': float(np.min(flat_confidences)) if flat_confidences else np.nan,
                'max': float(np.max(flat_confidences)) if flat_confidences else np.nan
            },
            'error_stats': {
                'mean': float(error_mean),
                'std': float(error_std),
                'min': float(np.min(errors)) if errors else np.nan,
                'max': float(np.max(errors)) if errors else np.nan
            },
            'calibration': {
                'confidence_error_correlation': float(confidence_error_correlation)
            },
            'record_count': len(self.prediction_records)
        }
        
        # Save analysis if requested
        if save:
            analysis_path = os.path.join(self.data_dir, "prediction_analysis.json")
            with open(analysis_path, 'w') as f:
                json.dump(analysis, f, indent=2)
            
            # Create visualization
            self._create_prediction_visualization()
        
        return analysis
    
    def analyze_activations(self, save: bool = True) -> Dict[str, Any]:
        """
        Analyze activation records
        
        Args:
            save: Whether to save analysis
            
        Returns:
            Analysis results
        """
        if not self.activation_records:
            logger.warning("No activation records to analyze")
            return {}
        
        # Group by module name
        module_groups = {}
        for record in self.activation_records:
            name = record['name']
            if name not in module_groups:
                module_groups[name] = []
            module_groups[name].append(record)
        
        # Analyze each module
        module_analyses = {}
        for name, records in module_groups.items():
            means = [r['mean'] for r in records]
            stds = [r['std'] for r in records]
            
            module_analyses[name] = {
                'mean': float(np.mean(means)),
                'std': float(np.mean(stds)),
                'stability': float(1.0 - np.std(means) / (np.mean(means) + 1e-10)),
                'record_count': len(records)
            }
        
        # Create analysis results
        analysis = {
            'module_analyses': module_analyses,
            'total_records': len(self.activation_records),
            'module_count': len(module_groups)
        }
        
        # Save analysis if requested
        if save:
            analysis_path = os.path.join(self.data_dir, "activation_analysis.json")
            with open(analysis_path, 'w') as f:
                json.dump(analysis, f, indent=2)
            
            # Create visualization
            self._create_activation_visualization(module_groups)
        
        return analysis
    
    def analyze_attention(self, save: bool = True) -> Dict[str, Any]:
        """
        Analyze attention records
        
        Args:
            save: Whether to save analysis
            
        Returns:
            Analysis results
        """
        if not self.attention_records:
            logger.warning("No attention records to analyze")
            return {}
        
        # Extract entropy values
        entropies = []
        for record in self.attention_records:
            if isinstance(record['entropy'], list):
                entropies.extend(record['entropy'])
            else:
                entropies.append(record['entropy'])
        
        # Calculate statistics
        entropy_mean = np.mean(entropies) if entropies else np.nan
        entropy_std = np.std(entropies) if entropies else np.nan
        
        # Create analysis results
        analysis = {
            'entropy_stats': {
                'mean': float(entropy_mean),
                'std': float(entropy_std),
                'min': float(np.min(entropies)) if entropies else np.nan,
                'max': float(np.max(entropies)) if entropies else np.nan
            },
            'record_count': len(self.attention_records)
        }
        
        # Save analysis if requested
        if save:
            analysis_path = os.path.join(self.data_dir, "attention_analysis.json")
            with open(analysis_path, 'w') as f:
                json.dump(analysis, f, indent=2)
            
            # Create visualization
            self._create_attention_visualization()
        
        return analysis
    
    def analyze_memory(self, save: bool = True) -> Dict[str, Any]:
        """
        Analyze memory records
        
        Args:
            save: Whether to save analysis
            
        Returns:
            Analysis results
        """
        if not self.memory_records:
            logger.warning("No memory records to analyze")
            return {}
        
        # Extract metrics
        diversities = [r['diversity'] for r in self.memory_records]
        max_usages = [r['max_usage'] for r in self.memory_records]
        
        # Calculate statistics
        diversity_mean = np.mean(diversities) if diversities else np.nan
        max_usage_mean = np.mean(max_usages) if max_usages else np.nan
        
        # Create analysis results
        analysis = {
            'diversity_stats': {
                'mean': float(diversity_mean),
                'min': float(np.min(diversities)) if diversities else np.nan,
                'max': float(np.max(diversities)) if diversities else np.nan
            },
            'usage_stats': {
                'mean_max_usage': float(max_usage_mean),
                'min_max_usage': float(np.min(max_usages)) if max_usages else np.nan,
                'max_max_usage': float(np.max(max_usages)) if max_usages else np.nan
            },
            'record_count': len(self.memory_records)
        }
        
        # Save analysis if requested
        if save:
            analysis_path = os.path.join(self.data_dir, "memory_analysis.json")
            with open(analysis_path, 'w') as f:
                json.dump(analysis, f, indent=2)
            
            # Create visualization
            self._create_memory_visualization()
        
        return analysis
    
    def create_monitoring_dashboard(self) -> str:
        """
        Create comprehensive monitoring dashboard
        
        Returns:
            Path to dashboard HTML
        """
        # Analyze all data
        prediction_analysis = self.analyze_predictions(save=True)
        
        if self.model_type.lower() == "cognitive":
            # Add cognitive model-specific analysis
            activation_analysis = self.analyze_activations(save=True)
            attention_analysis = self.analyze_attention(save=True)
            memory_analysis = self.analyze_memory(save=True)
            
            # Combine all analyses
            analysis = {
                'cognitive_model': {
                    'prediction_analysis': prediction_analysis,
                    'activation_analysis': activation_analysis,
                    'attention_analysis': attention_analysis,
                    'memory_analysis': memory_analysis
                }
            }
        else:
            # For baseline model, only prediction analysis is available
            analysis = {
                'baseline_model': {
                    'prediction_analysis': prediction_analysis
                }
            }
        
        # Create HTML dashboard
        dashboard_path = self._create_dashboard_html(analysis)
        
        # Save dashboard
        dashboard_path = os.path.join(self.viz_dir, "monitoring_dashboard.html")
        with open(dashboard_path, 'w') as f:
            f.write(f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Model Monitoring Dashboard</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    .container {{ display: flex; flex-wrap: wrap; }}
                    .card {{ border: 1px solid #ddd; border-radius: 5px; margin: 10px; padding: 15px; width: 45%; }}
                    .full-width {{ width: 95%; }}
                    h1, h2, h3 {{ color: #333; }}
                    table {{ border-collapse: collapse; width: 100%; margin-bottom: 15px; }}
                    th, td {{ text-align: left; padding: 8px; border-bottom: 1px solid #ddd; }}
                    th {{ background-color: #f2f2f2; }}
                    img {{ max-width: 100%; height: auto; }}
                </style>
            </head>
            <body>
                <h1>Model Monitoring Dashboard</h1>
                <p>Model Type: {self.model_type}</p>
                <p>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                
                <div class="container">
                    <div class="card">
                        <h2>Prediction Analysis</h2>
                        <table>
                            <tr>
                                <th>Metric</th>
                                <th>Value</th>
                            </tr>
                            <tr>
                                <td>Prediction Mean</td>
                                <td>{prediction_analysis.get('prediction_stats', {}).get('mean', 'N/A')}</td>
                            </tr>
                            <tr>
                                <td>Prediction Std</td>
                                <td>{prediction_analysis.get('prediction_stats', {}).get('std', 'N/A')}</td>
                            </tr>
                            <tr>
                                <td>Confidence Mean</td>
                                <td>{prediction_analysis.get('confidence_stats', {}).get('mean', 'N/A')}</td>
                            </tr>
                            <tr>
                                <td>Error Mean</td>
                                <td>{prediction_analysis.get('error_stats', {}).get('mean', 'N/A')}</td>
                            </tr>
                            <tr>
                                <td>Confidence-Error Correlation</td>
                                <td>{prediction_analysis.get('calibration', {}).get('confidence_error_correlation', 'N/A')}</td>
                            </tr>
                        </table>
                        <img src="visualizations/prediction_visualization.png" alt="Prediction Visualization">
                    </div>
            """)
            
            # Add cognitive-specific sections if applicable
            if self.model_type.lower() == "cognitive":
                f.write(f"""
                    <div class="card">
                        <h2>Activation Analysis</h2>
                        <table>
                            <tr>
                                <th>Module</th>
                                <th>Mean</th>
                                <th>Stability</th>
                            </tr>
                """)
                
                for module, stats in activation_analysis.get('module_analyses', {}).items():
                    f.write(f"""
                            <tr>
                                <td>{module}</td>
                                <td>{stats.get('mean', 'N/A')}</td>
                                <td>{stats.get('stability', 'N/A')}</td>
                            </tr>
                    """)
                
                f.write(f"""
                        </table>
                        <img src="visualizations/activation_visualization.png" alt="Activation Visualization">
                    </div>
                    
                    <div class="card">
                        <h2>Attention Analysis</h2>
                        <table>
                            <tr>
                                <th>Metric</th>
                                <th>Value</th>
                            </tr>
                            <tr>
                                <td>Entropy Mean</td>
                                <td>{attention_analysis.get('entropy_stats', {}).get('mean', 'N/A')}</td>
                            </tr>
                            <tr>
                                <td>Entropy Std</td>
                                <td>{attention_analysis.get('entropy_stats', {}).get('std', 'N/A')}</td>
                            </tr>
                        </table>
                        <img src="visualizations/attention_visualization.png" alt="Attention Visualization">
                    </div>
                    
                    <div class="card">
                        <h2>Memory Analysis</h2>
                        <table>
                            <tr>
                                <th>Metric</th>
                                <th>Value</th>
                            </tr>
                            <tr>
                                <td>Diversity Mean</td>
                                <td>{memory_analysis.get('diversity_stats', {}).get('mean', 'N/A')}</td>
                            </tr>
                            <tr>
                                <td>Max Usage Mean</td>
                                <td>{memory_analysis.get('usage_stats', {}).get('mean_max_usage', 'N/A')}</td>
                            </tr>
                        </table>
                        <img src="visualizations/memory_visualization.png" alt="Memory Visualization">
                    </div>
                """)
            
            # Close HTML
            f.write("""
                </div>
            </body>
            </html>
            """)
        
        logger.info(f"Created monitoring dashboard at {dashboard_path}")
        
        return dashboard_path
    
    def _create_prediction_visualization(self):
        """Create visualization of prediction data"""
        if not self.prediction_records:
            logger.warning("No prediction records for visualization")
            return
        
        plt.figure(figsize=(12, 10))
        
        # Extract data
        predictions = []
        confidences = []
        errors = []
        
        for record in self.prediction_records:
            pred = record['prediction']
            conf = record['confidence']
            err = record['error']
            
            # Handle different data shapes
            if isinstance(pred, list):
                if isinstance(pred[0], list):
                    # 2D list
                    pred = pred[0][0] if len(pred[0]) > 0 else 0
                else:
                    # 1D list
                    pred = pred[0] if len(pred) > 0 else 0
            
            if isinstance(conf, list):
                if isinstance(conf[0], list):
                    # 2D list
                    conf = conf[0][0] if len(conf[0]) > 0 else 1
                else:
                    # 1D list
                    conf = conf[0] if len(conf) > 0 else 1
            
            predictions.append(pred)
            confidences.append(conf)
            if err is not None:
                errors.append(err)
        
        # Plot predictions
        plt.subplot(3, 1, 1)
        plt.plot(predictions, label='Predictions')
        plt.title('Predictions Over Time')
        plt.xlabel('Sample Index')
        plt.ylabel('Prediction Value')
        plt.grid(True, alpha=0.3)
        
        # Plot confidences
        plt.subplot(3, 1, 2)
        plt.plot(confidences, label='Confidence', color='green')
        plt.title('Confidence Over Time')
        plt.xlabel('Sample Index')
        plt.ylabel('Confidence Value')
        plt.grid(True, alpha=0.3)
        
        # Plot errors if available
        if errors:
            plt.subplot(3, 1, 3)
            plt.plot(errors, label='Error', color='red')
            plt.title('Error Over Time')
            plt.xlabel('Sample Index')
            plt.ylabel('Error Value')
            plt.grid(True, alpha=0.3)
            
            # Add correlation between confidence and error
            if len(confidences) == len(errors):
                corr = np.corrcoef(confidences, errors)[0, 1]
                plt.annotate(f'Confidence-Error Correlation: {corr:.4f}', 
                           xy=(0.5, 0.9), xycoords='axes fraction', 
                           ha='center', va='center',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.viz_dir, 'prediction_visualization.png'), dpi=300)
        plt.close()
    
    def _create_activation_visualization(self, module_groups):
        """Create visualization of activation data"""
        if not module_groups:
            logger.warning("No activation records for visualization")
            return
        
        # Count the number of modules
        n_modules = len(module_groups)
        
        # Determine subplot grid
        n_cols = min(3, n_modules)
        n_rows = (n_modules + n_cols - 1) // n_cols
        
        plt.figure(figsize=(15, 5 * n_rows))
        
        # Plot each module's activations
        for i, (name, records) in enumerate(module_groups.items()):
            plt.subplot(n_rows, n_cols, i + 1)
            
            means = [r['mean'] for r in records]
            stds = [r['std'] for r in records]
            
            plt.plot(means, label='Mean Activation')
            plt.fill_between(range(len(means)), 
                           [m - s for m, s in zip(means, stds)],
                           [m + s for m, s in zip(means, stds)],
                           alpha=0.3)
            
            plt.title(f'Module: {name}')
            plt.xlabel('Sample Index')
            plt.ylabel('Activation')
            plt.grid(True, alpha=0.3)
            
            # Calculate stability
            stability = 1.0 - np.std(means) / (np.mean(means) + 1e-10)
            plt.annotate(f'Stability: {stability:.4f}', 
                       xy=(0.5, 0.9), xycoords='axes fraction', 
                       ha='center', va='center',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.viz_dir, 'activation_visualization.png'), dpi=300)
        plt.close()
    
    def _create_attention_visualization(self):
        """Create visualization of attention data"""
        if not self.attention_records:
            logger.warning("No attention records for visualization")
            return
        
        plt.figure(figsize=(15, 10))
        
        # Plot attention entropy
        plt.subplot(2, 1, 1)
        
        entropies = []
        for record in self.attention_records:
            if isinstance(record['entropy'], list):
                # Use mean entropy if there are multiple values
                entropies.append(np.mean(record['entropy']))
            else:
                entropies.append(record['entropy'])
        
        plt.plot(entropies, label='Attention Entropy')
        plt.title('Attention Entropy Over Time')
        plt.xlabel('Sample Index')
        plt.ylabel('Entropy')
        plt.grid(True, alpha=0.3)
        
        # Plot attention heat map for last record
        plt.subplot(2, 1, 2)
        
        if self.attention_records:
            last_record = self.attention_records[-1]
            weights = last_record['weights']
            
            # Handle different data shapes
            if isinstance(weights, list):
                if isinstance(weights[0], list):
                    # For 2D attention weights
                    weights_array = np.array(weights)
                    if weights_array.ndim == 2:
                        plt.imshow(weights_array, cmap='viridis', aspect='auto')
                        plt.colorbar(label='Attention Weight')
                        plt.title('Last Attention Weight Matrix')
                        plt.xlabel('Key Position')
                        plt.ylabel('Query Position')
                    else:
                        # For higher dimensions, flatten to 2D
                        flat_weights = weights_array.reshape(weights_array.shape[0], -1)
                        plt.imshow(flat_weights, cmap='viridis', aspect='auto')
                        plt.colorbar(label='Attention Weight')
                        plt.title('Last Attention Weight Matrix (Flattened)')
                        plt.xlabel('Key Position (Flattened)')
                        plt.ylabel('Query Position')
                else:
                    # For 1D attention weights
                    plt.bar(range(len(weights)), weights)
                    plt.title('Last Attention Weights')
                    plt.xlabel('Position')
                    plt.ylabel('Weight')
            else:
                plt.text(0.5, 0.5, 'No valid attention weights to visualize', 
                       ha='center', va='center', transform=plt.gca().transAxes)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.viz_dir, 'attention_visualization.png'), dpi=300)
        plt.close()
    
    def _create_memory_visualization(self):
        """Create visualization of memory data"""
        if not self.memory_records:
            logger.warning("No memory records for visualization")
            return
        
        plt.figure(figsize=(15, 10))
        
        # Plot memory diversity
        plt.subplot(2, 1, 1)
        
        diversities = [r['diversity'] for r in self.memory_records]
        
        plt.plot(diversities, label='Memory Diversity')
        plt.title('Memory Diversity Over Time')
        plt.xlabel('Sample Index')
        plt.ylabel('Diversity (Std)')
        plt.grid(True, alpha=0.3)
        
        # Plot maximum memory usage
        plt.subplot(2, 1, 2)
        
        max_usages = [r['max_usage'] for r in self.memory_records]
        
        plt.plot(max_usages, label='Max Memory Usage', color='orange')
        plt.title('Maximum Memory Usage Over Time')
        plt.xlabel('Sample Index')
        plt.ylabel('Max Usage')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.viz_dir, 'memory_visualization.png'), dpi=300)
        plt.close()
    
    def save_records(self):
        """Save all monitoring records"""
        # Save prediction records
        if self.prediction_records:
            with open(os.path.join(self.data_dir, 'prediction_records.json'), 'w') as f:
                json.dump(self.prediction_records, f, indent=2)
        
        # Save activation records
        if self.activation_records:
            with open(os.path.join(self.data_dir, 'activation_records.json'), 'w') as f:
                json.dump(self.activation_records, f, indent=2)
        
        # Save attention records
        if self.attention_records:
            with open(os.path.join(self.data_dir, 'attention_records.json'), 'w') as f:
                json.dump(self.attention_records, f, indent=2)
        
        # Save memory records
        if self.memory_records:
            with open(os.path.join(self.data_dir, 'memory_records.json'), 'w') as f:
                json.dump(self.memory_records, f, indent=2)
        
        logger.info(f"Saved all monitoring records to {self.data_dir}")
    
    def cleanup(self):
        """Clean up resources"""
        # Remove hooks
        if hasattr(self, 'activation_hooks'):
            for hook in self.activation_hooks:
                hook.remove()

def setup_monitoring(
    model_path: str,
    test_data_path: Optional[str] = None,
    output_dir: str = "monitoring",
    model_type: str = "cognitive",
    device: str = "cpu",
    num_samples: int = 100
) -> str:
    """
    Setup monitoring infrastructure for a model
    
    Args:
        model_path: Path to model weights
        test_data_path: Path to test data (optional)
        output_dir: Directory to save monitoring data
        model_type: Model type ('cognitive' or 'baseline')
        device: Computation device
        num_samples: Number of samples to process
        
    Returns:
        Path to dashboard HTML
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    if model_type.lower() == "cognitive":
        model = CognitiveArchitecture()
    else:
        # Assume baseline LSTM
        from src.arch.baseline_lstm import FinancialLSTMBaseline
        model = FinancialLSTMBaseline()
    
    # Load weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Create monitor
    monitor = ModelMonitor(
        model=model,
        model_type=model_type,
        output_dir=output_dir,
        device=device
    )
    
    # Setup component monitoring for cognitive models
    if model_type.lower() == "cognitive":
        monitor.setup_component_monitoring()
    
    # Process test data if provided
    if test_data_path:
        logger.info(f"Processing test data from {test_data_path}")
        
        # Load test data
        test_data = pd.read_csv(test_data_path)
        
        # Extract features and target
        feature_cols = [col for col in test_data.columns if col not in ['date', 'target']]
        
        if 'target' in test_data.columns:
            target_col = 'target'
        elif 'return_1d' in test_data.columns:
            # Use next day's return as target
            test_data['target'] = test_data['return_1d'].shift(-1)
            target_col = 'target'
        else:
            target_col = None
        
        # Normalize features
        features = test_data[feature_cols].values
        feature_means = np.nanmean(features, axis=0)
        feature_stds = np.nanstd(features, axis=0)
        feature_stds[feature_stds == 0] = 1.0
        normalized_features = (features - feature_means) / feature_stds
        normalized_features = np.nan_to_num(normalized_features)
        
        # Get targets if available
        targets = test_data[target_col].values if target_col else None
        
        # Define sequence length (assuming default is 20)
        sequence_length = 20
        
        # Process samples
        for i in tqdm(range(sequence_length, min(len(normalized_features), sequence_length + num_samples)), desc="Processing samples"):
            # Get sequence and current features
            current_sequence = normalized_features[i-sequence_length:i]
            current_features = normalized_features[i]
            
            # Create tensors
            feature_tensor = torch.tensor(current_features, dtype=torch.float32).unsqueeze(0)
            sequence_tensor = torch.tensor(current_sequence, dtype=torch.float32).unsqueeze(0)
            
            # Create target tensor if available
            target_tensor = torch.tensor([targets[i]], dtype=torch.float32).unsqueeze(0) if targets is not None else None
            
            # Record prediction
            metadata = {
                'index': i,
                'date': test_data['date'].iloc[i] if 'date' in test_data.columns else None
            }
            
            monitor.record_prediction(
                financial_data=feature_tensor,
                financial_seq=sequence_tensor,
                target=target_tensor,
                metadata=metadata
            )
    
    # Analyze data and create dashboard
    dashboard_path = monitor.create_monitoring_dashboard()
    
    # Save records
    monitor.save_records()
    
    # Clean up
    monitor.cleanup()
    
    return dashboard_path

def main():
    parser = argparse.ArgumentParser(description="Setup monitoring infrastructure for cognitive models")
    
    # Required arguments
    parser.add_argument("--model_path", required=True, help="Path to model weights")
    
    # Optional arguments
    parser.add_argument("--test_data", help="Path to test data")
    parser.add_argument("--output_dir", default="monitoring", help="Directory to save monitoring data")
    parser.add_argument("--model_type", choices=["cognitive", "baseline"], default="cognitive", help="Model type")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Computation device")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of samples to process")
    
    args = parser.parse_args()
    
    # Setup monitoring
    dashboard_path = setup_monitoring(
        model_path=args.model_path,
        test_data_path=args.test_data,
        output_dir=args.output_dir,
        model_type=args.model_type,
        device=args.device,
        num_samples=args.num_samples
    )
    
    print(f"Monitoring dashboard created at: {dashboard_path}")

if __name__ == "__main__":
    main()
