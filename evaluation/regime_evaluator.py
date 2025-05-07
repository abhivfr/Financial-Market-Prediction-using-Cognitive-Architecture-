#!/usr/bin/env python
# regime_evaluator.py - Regime-specific evaluation for cognitive architecture

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional
import json
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score
from collections import defaultdict
from tqdm import tqdm

class RegimeEvaluator:
    """
    Performs regime-specific evaluation of model performance
    Isolates metrics by market regime and during regime transitions
    """
    def __init__(self, model, data_loader, regime_labels, regime_names=None, output_dir="evaluation/regime"):
        """
        Initialize the regime-specific evaluator
        
        Args:
            model: The trained model to evaluate
            data_loader: DataLoader containing test data
            regime_labels: Array of regime labels corresponding to test data points
            regime_names: Dictionary mapping regime ids to readable names
            output_dir: Directory to save evaluation results
        """
        self.model = model
        self.data_loader = data_loader
        self.regime_labels = regime_labels
        self.output_dir = output_dir
        
        # Create default regime names if not provided
        if regime_names is None:
            unique_regimes = np.unique(regime_labels)
            self.regime_names = {r: f"Regime {r}" for r in unique_regimes}
        else:
            self.regime_names = regime_names
            
        os.makedirs(output_dir, exist_ok=True)
        
        # Storage for predictions and targets
        self.predictions = []
        self.targets = []
        self.dates = []
        self.sample_regimes = []
    
    def evaluate(self, device="cpu"):
        """
        Evaluate the model on test data, recording predictions and targets
        
        Args:
            device: Device to run the model on
        """
        self.model.eval()
        self.model.to(device)
        
        all_preds = []
        all_targets = []
        all_dates = []
        
        sequence_length = 20  # Default sequence length
        
        try:
            # Try standard iteration through data loader
            with torch.no_grad():
                for batch_idx, batch in enumerate(tqdm(self.data_loader, desc="Evaluating")):
                    # Handle different data formats
                    if isinstance(batch, tuple) and len(batch) >= 2:
                        # Unpack batch
                        if len(batch) == 3:
                            data, target, date_indices = batch
                        elif len(batch) == 2:
                            data, target = batch
                            date_indices = torch.arange(batch_idx * len(data), (batch_idx + 1) * len(data))
                        else:
                            raise ValueError(f"Unexpected batch format with {len(batch)} elements")
                        
                        # Ensure data is a tensor and on the correct device
                        if isinstance(data, str):
                            print(f"Warning: Data is a string - {data[:30]}... Skipping batch")
                            continue
                        
                        if isinstance(data, torch.Tensor):
                            data = data.to(device)
                        else:
                            # Try to convert to tensor
                            try:
                                data = torch.tensor(data, dtype=torch.float32).to(device)
                            except:
                                print(f"Warning: Could not convert data to tensor. Type: {type(data)}. Skipping batch.")
                                continue
                        
                        # Forward pass - handle different model interfaces
                        try:
                            # Check model type
                            if hasattr(self.model, '__class__') and self.model.__class__.__name__ == 'FinancialLSTMBaseline':
                                # Baseline model only accepts a single input
                                output = self.model(data)
                            else:
                                # Try standard forward pass with more arguments
                                output = self.model(data)
                            
                            # Check if output is a dictionary
                            if isinstance(output, dict) and 'market_state' in output:
                                output = output['market_state']
                        except Exception as e:
                            print(f"Error during model forward pass: {str(e)}")
                            continue
                        
                        # Store predictions, targets and dates
                        all_preds.append(output.cpu().numpy())
                        all_targets.append(target.numpy() if isinstance(target, torch.Tensor) else np.array(target))
                        all_dates.extend(date_indices.numpy() if isinstance(date_indices, torch.Tensor) else np.array(date_indices))
                    elif isinstance(batch, dict):
                        # Handle dictionary-style batches
                        data = batch.get('features', None)
                        sequence = batch.get('sequence', None)
                        target = batch.get('target', None)
                        date_indices = batch.get('dates', torch.arange(batch_idx * len(data), (batch_idx + 1) * len(data)))
                        
                        if data is None or target is None:
                            print("Warning: Missing data or target in batch dictionary. Skipping.")
                            continue
                        
                        # Move tensors to device
                        data = data.to(device)
                        if sequence is not None:
                            sequence = sequence.to(device)
                            
                        # Forward pass - handle different model interfaces
                        try:
                            if sequence is not None:
                                # Check model type
                                if hasattr(self.model, '__class__') and self.model.__class__.__name__ == 'FinancialLSTMBaseline':
                                    # Baseline model only accepts sequence
                                    output = self.model(sequence)
                                else:
                                    # Try cognitive architecture interface
                                    output = self.model(financial_data=data, financial_seq=sequence)
                            else:
                                # Check model type
                                if hasattr(self.model, '__class__') and self.model.__class__.__name__ == 'FinancialLSTMBaseline':
                                    # Baseline model only accepts a single input
                                    output = self.model(data)
                                else:
                                    # Try standard forward pass with more arguments
                                    output = self.model(data)
                            
                            # Check if output is a dictionary
                            if isinstance(output, dict) and 'market_state' in output:
                                output = output['market_state']
                        except Exception as e:
                            print(f"Error during model forward pass: {str(e)}")
                            continue
                        
                        # Store predictions, targets and dates
                        all_preds.append(output.cpu().numpy())
                        all_targets.append(target.numpy() if isinstance(target, torch.Tensor) else np.array(target))
                        all_dates.extend(date_indices.numpy() if isinstance(date_indices, torch.Tensor) else np.array(date_indices))
                    else:
                        print(f"Warning: Unexpected batch type: {type(batch)}. Skipping.")
                        continue
        except Exception as e:
            print(f"Error during evaluation loop: {str(e)}")
            print("Trying alternative evaluation approach...")
            
            # Fallback approach: manually iterate through the dataset
            with torch.no_grad():
                # Get features and targets from DataFrame
                if hasattr(self.data_loader, 'dataset') and hasattr(self.data_loader.dataset, 'data'):
                    df = self.data_loader.dataset.data
                    
                    # Extract features and targets
                    features = df.drop(columns=['target', 'date', 'regime'] if 'regime' in df.columns else ['target', 'date']).values
                    targets = df['target'].values if 'target' in df.columns else None
                    
                    # Process data in batches
                    batch_size = 32
                    for i in range(0, len(features) - sequence_length, batch_size):
                        batch_end = min(i + batch_size, len(features) - sequence_length)
                        batch_preds = []
                        
                        for j in range(i, batch_end):
                            # Create sequence
                            seq = features[j:j+sequence_length]
                            seq_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)
                            
                            # Forward pass
                            try:
                                # Check model type
                                if hasattr(self.model, '__class__') and self.model.__class__.__name__ == 'FinancialLSTMBaseline':
                                    # Baseline model only accepts sequence tensor
                                    output = self.model(seq_tensor)
                                else:
                                    # Try more complex interface
                                    output = self.model(financial_data=None, financial_seq=seq_tensor)
                                
                                # Handle dictionary output
                                if isinstance(output, dict) and 'market_state' in output:
                                    output = output['market_state']
                                    
                                batch_preds.append(output.cpu().numpy()[0])
                            except Exception as e:
                                print(f"Error with sample {j}: {str(e)}")
                                try:
                                    # Last fallback - simple forward pass
                                    output = self.model(seq_tensor)
                                    batch_preds.append(output.cpu().numpy()[0])
                                except Exception as e2:
                                    print(f"Secondary error with sample {j}: {str(e2)}")
                        
                        if batch_preds:
                            all_preds.append(np.array(batch_preds))
                            if targets is not None:
                                all_targets.append(targets[i+sequence_length:batch_end+sequence_length])
                            all_dates.extend(range(i+sequence_length, batch_end+sequence_length))
        
        if not all_preds:
            raise ValueError("No predictions were generated. Check data loader and model compatibility.")
        
        # Concatenate results
        try:
            self.predictions = np.concatenate(all_preds)
            if all_targets:
                self.targets = np.concatenate(all_targets)
            else:
                self.targets = np.zeros_like(self.predictions)  # Placeholder if no targets
            self.dates = np.array(all_dates)
            
            # Get regime for each sample
            self.sample_regimes = np.array([self.regime_labels[i % len(self.regime_labels)] for i in self.dates])
            
            return self.calculate_metrics()
        except Exception as e:
            print(f"Error concatenating results: {str(e)}")
            raise
    
    def calculate_metrics(self):
        """Calculate performance metrics for each regime"""
        unique_regimes = np.unique(self.sample_regimes)
        metrics = {}
        
        # Handle 3D predictions (batch, seq_len, features)
        if len(self.predictions.shape) == 3:
            print(f"Found 3D predictions with shape {self.predictions.shape}. Using last timestep only.")
            self.predictions = self.predictions[:, -1, :]
        
        # Overall metrics
        metrics["overall"] = self._calculate_regime_metrics(
            self.predictions, self.targets
        )
        
        # Per-regime metrics
        for regime in unique_regimes:
            regime_mask = self.sample_regimes == regime
            regime_preds = self.predictions[regime_mask]
            regime_targets = self.targets[regime_mask]
            
            regime_name = self.regime_names.get(regime, f"Regime {regime}")
            metrics[regime_name] = self._calculate_regime_metrics(
                regime_preds, regime_targets
            )
            metrics[regime_name]["samples"] = int(np.sum(regime_mask))
            metrics[regime_name]["percentage"] = float(np.mean(regime_mask) * 100)
        
        return metrics
    
    def _calculate_regime_metrics(self, predictions, targets):
        """Calculate metrics for a specific regime"""
        if len(predictions) == 0 or len(targets) == 0:
            return {
                "mse": None,
                "rmse": None,
                "mae": None,
                "direction_accuracy": None
            }
        
        # Handle dimension mismatches
        if predictions.shape != targets.shape:
            print(f"Warning: Shape mismatch between predictions {predictions.shape} and targets {targets.shape}")
            
            # Get minimum dimension sizes
            min_batch = min(predictions.shape[0], targets.shape[0])
            
            # Get feature dimensions
            if len(predictions.shape) > 1:
                pred_features = predictions.shape[1]
            else:
                pred_features = 1
                predictions = predictions.reshape(-1, 1)
                
            if len(targets.shape) > 1:
                target_features = targets.shape[1]
            else:
                target_features = 1
                targets = targets.reshape(-1, 1)
                
            min_features = min(pred_features, target_features)
            
            # Truncate to match dimensions
            predictions = predictions[:min_batch, :min_features]
            targets = targets[:min_batch, :min_features]
            
            print(f"Adjusted shapes - predictions: {predictions.shape}, targets: {targets.shape}")
        
        try:
            mse = mean_squared_error(targets, predictions)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(targets, predictions)
            
            # Direction accuracy
            pred_direction = np.sign(predictions)
            target_direction = np.sign(targets)
            
            # Handle multi-dimensional arrays correctly
            if len(predictions.shape) > 1:
                # For each sample, count a match only if all dimensions match
                direction_matches = np.sum(np.all(pred_direction == target_direction, axis=1))
            else:
                direction_matches = np.sum(pred_direction == target_direction)
                
            # Calculate accuracy as a proportion between 0 and 1
            total_samples = len(predictions)
            direction_accuracy = direction_matches / total_samples if total_samples > 0 else 0
            
            return {
                "mse": float(mse),
                "rmse": float(rmse),
                "mae": float(mae),
                "direction_accuracy": float(direction_accuracy)
            }
        except Exception as e:
            print(f"Error calculating metrics: {e}")
            return {
                "mse": None,
                "rmse": None,
                "mae": None,
                "direction_accuracy": None
            }
    
    def visualize_regime_performance(self, metrics=None):
        """Visualize model performance across different regimes"""
        if metrics is None:
            metrics = self.calculate_metrics()
        
        regime_names = [name for name in metrics.keys() if name != "overall"]
        
        # Extract metrics for plotting
        mse_values = [metrics[regime]["mse"] for regime in regime_names]
        rmse_values = [metrics[regime]["rmse"] for regime in regime_names]
        direction_acc = [metrics[regime]["direction_accuracy"] * 100 for regime in regime_names]
        
        # Plot MSE and RMSE
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # MSE by regime
        bars = ax1.bar(regime_names, mse_values)
        ax1.set_title("Mean Squared Error by Market Regime")
        ax1.set_ylabel("MSE")
        ax1.set_xlabel("Market Regime")
        ax1.axhline(y=metrics["overall"]["mse"], color='r', linestyle='--', 
                   label=f"Overall MSE: {metrics['overall']['mse']:.4f}")
        ax1.legend()
        
        # Add sample counts as annotations
        for i, bar in enumerate(bars):
            regime = regime_names[i]
            sample_count = metrics[regime]["samples"]
            percentage = metrics[regime]["percentage"]
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                    f"n={sample_count}\n({percentage:.1f}%)", 
                    ha='center', va='bottom', fontsize=8)
        
        # Direction accuracy by regime
        bars = ax2.bar(regime_names, direction_acc)
        ax2.set_title("Direction Accuracy by Market Regime")
        ax2.set_ylabel("Direction Accuracy (%)")
        ax2.set_xlabel("Market Regime")
        ax2.axhline(y=metrics["overall"]["direction_accuracy"] * 100, color='r', linestyle='--',
                   label=f"Overall: {metrics['overall']['direction_accuracy'] * 100:.1f}%")
        ax2.legend()
        
        # Add sample counts as annotations
        for i, bar in enumerate(bars):
            regime = regime_names[i]
            sample_count = metrics[regime]["samples"]
            percentage = metrics[regime]["percentage"]
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    f"n={sample_count}\n({percentage:.1f}%)", 
                    ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "regime_performance.png"))
        plt.close()
        
        # Create error distribution plots by regime
        self._plot_error_distribution_by_regime()
        
        # Create prediction vs actual scatter plots by regime
        self._plot_predictions_by_regime()
    
    def _plot_error_distribution_by_regime(self):
        """Plot error distribution for each regime"""
        unique_regimes = np.unique(self.sample_regimes)
        
        # Handle dimension mismatches for visualization
        preds = self.predictions
        targs = self.targets
        
        # Reshape or slice if necessary
        if preds.shape != targs.shape:
            print(f"Reshaping for error visualization: predictions {preds.shape}, targets {targs.shape}")
            # Use minimum dimensions
            min_features = min(preds.shape[1] if len(preds.shape) > 1 else 1, 
                              targs.shape[1] if len(targs.shape) > 1 else 1)
            
            # Reshape if 1D
            if len(preds.shape) == 1:
                preds = preds.reshape(-1, 1)
            if len(targs.shape) == 1:
                targs = targs.reshape(-1, 1)
                
            # Slice to match
            preds = preds[:, :min_features]
            targs = targs[:, :min_features]
        
        # Calculate error only for the first feature
        errors = preds[:, 0] - targs[:, 0]
        
        n_regimes = len(unique_regimes)
        fig, axes = plt.subplots(1, n_regimes, figsize=(n_regimes * 5, 5))
        
        if n_regimes == 1:
            axes = [axes]
        
        for i, regime in enumerate(unique_regimes):
            regime_mask = self.sample_regimes == regime
            regime_errors = errors[regime_mask]
            
            regime_name = self.regime_names.get(regime, f"Regime {regime}")
            
            # Plot histogram
            axes[i].hist(regime_errors, bins=30, alpha=0.7)
            axes[i].set_title(f"{regime_name} Error Distribution")
            axes[i].set_xlabel("Prediction Error")
            axes[i].set_ylabel("Frequency")
            
            # Add statistics
            mean_error = np.mean(regime_errors)
            std_error = np.std(regime_errors)
            axes[i].axvline(mean_error, color='r', linestyle='--', 
                           label=f"Mean: {mean_error:.4f}")
            axes[i].axvline(mean_error + std_error, color='g', linestyle=':', 
                           label=f"Std: {std_error:.4f}")
            axes[i].axvline(mean_error - std_error, color='g', linestyle=':')
            axes[i].legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "error_distribution_by_regime.png"))
        plt.close()
    
    def _plot_predictions_by_regime(self):
        """Create scatter plots of predictions vs actual values by regime"""
        unique_regimes = np.unique(self.sample_regimes)
        
        # Handle dimension mismatches for visualization
        preds = self.predictions
        targs = self.targets
        
        # Reshape or slice if necessary
        if preds.shape != targs.shape:
            print(f"Reshaping for scatter visualization: predictions {preds.shape}, targets {targs.shape}")
            # Use minimum dimensions
            min_features = min(preds.shape[1] if len(preds.shape) > 1 else 1, 
                              targs.shape[1] if len(targs.shape) > 1 else 1)
            
            # Reshape if 1D
            if len(preds.shape) == 1:
                preds = preds.reshape(-1, 1)
            if len(targs.shape) == 1:
                targs = targs.reshape(-1, 1)
                
            # Slice to match
            preds = preds[:, :min_features]
            targs = targs[:, :min_features]
        
        n_regimes = len(unique_regimes)
        fig, axes = plt.subplots(1, n_regimes, figsize=(n_regimes * 5, 5))
        
        if n_regimes == 1:
            axes = [axes]
        
        for i, regime in enumerate(unique_regimes):
            regime_mask = self.sample_regimes == regime
            regime_preds = preds[regime_mask, 0]  # Use first feature
            regime_targets = targs[regime_mask, 0]  # Use first feature
            
            regime_name = self.regime_names.get(regime, f"Regime {regime}")
            
            # Plot scatter
            axes[i].scatter(regime_targets, regime_preds, alpha=0.5)
            
            # Add ideal prediction line
            min_val = min(np.min(regime_targets), np.min(regime_preds))
            max_val = max(np.max(regime_targets), np.max(regime_preds))
            axes[i].plot([min_val, max_val], [min_val, max_val], 'r--')
            
            axes[i].set_title(f"{regime_name} Predictions vs Actual")
            axes[i].set_xlabel("Actual Values")
            axes[i].set_ylabel("Predicted Values")
            
            # Calculate correlation
            try:
                correlation = np.corrcoef(regime_targets, regime_preds)[0, 1]
                axes[i].text(0.05, 0.95, f"Correlation: {correlation:.4f}", 
                            transform=axes[i].transAxes, fontsize=10,
                            verticalalignment='top')
            except Exception as e:
                print(f"Error calculating correlation: {e}")
                axes[i].text(0.05, 0.95, "Correlation: N/A", 
                            transform=axes[i].transAxes, fontsize=10,
                            verticalalignment='top')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "predictions_vs_actual_by_regime.png"))
        plt.close()
    
    def save_results(self, metrics=None):
        """Save evaluation results to output directory"""
        if metrics is None:
            metrics = self.calculate_metrics()
        
        # Save metrics as JSON
        metrics_path = os.path.join(self.output_dir, "regime_metrics.json")
        with open(metrics_path, 'w') as f:
            # Convert numpy types to Python native types for JSON serialization
            json_compatible_metrics = json.loads(
                json.dumps(metrics, default=lambda x: float(x) if isinstance(x, (np.float32, np.float64)) else x)
            )
            json.dump(json_compatible_metrics, f, indent=4)
        
        # Create visualizations
        self.visualize_regime_performance(metrics)
        
        # Save predictions and targets as CSV for further analysis
        try:
            # Create copies of predictions and targets for processing
            preds = self.predictions.copy()
            targs = self.targets.copy()
            dates = self.dates.copy()
            
            # Check for length mismatches
            min_length = min(len(preds), len(targs), len(dates))
            if min_length == 0:
                print("Warning: No valid predictions to save.")
                return
                
            if min_length < len(preds) or min_length < len(targs) or min_length < len(dates):
                print(f"Warning: Length mismatch - predictions:{len(preds)}, targets:{len(targs)}, dates:{len(dates)}")
                print(f"Truncating to minimum length: {min_length}")
                
                preds = preds[:min_length]
                targs = targs[:min_length]
                dates = dates[:min_length]
            
            # Build DataFrame
            try:
                # Handle 3D predictions (batch, seq_len, features)
                if len(preds.shape) == 3:
                    print(f"Found 3D predictions with shape {preds.shape}. Using last timestep for CSV.")
                    preds = preds[:, -1, :]
                
                # Get the first column of predictions and targets if they're multi-dimensional
                if len(preds.shape) > 1:
                    if preds.shape[1] > 1:
                        print(f"Using only first column of predictions (shape: {preds.shape})")
                    pred_values = preds[:, 0]
                else:
                    pred_values = preds
                    
                if len(targs.shape) > 1:
                    if targs.shape[1] > 1:
                        print(f"Using only first column of targets (shape: {targs.shape})")
                    target_values = targs[:, 0]
                else:
                    target_values = targs
                
                # If prediction and target values still don't match in length, truncate
                min_values_length = min(len(pred_values), len(target_values))
                if len(pred_values) != len(target_values):
                    print(f"Truncating values to match: {min_values_length}")
                    pred_values = pred_values[:min_values_length]
                    target_values = target_values[:min_values_length]
                    dates = dates[:min_values_length]
                
                # Create the results DataFrame
                results_df = pd.DataFrame({
                    "date_index": dates,
                    "regime": [self.regime_labels[i % len(self.regime_labels)] for i in dates],
                    "regime_name": [self.regime_names.get(self.regime_labels[i % len(self.regime_labels)], 
                                   f"Regime {self.regime_labels[i % len(self.regime_labels)]}") for i in dates],
                    "prediction": pred_values,
                    "target": target_values,
                    "error": pred_values - target_values
                })
                
                # Ensure all columns have the same length
                assert len(results_df) == min(min_length, min_values_length), "DataFrame shape mismatch after creation"
                
                # Save to CSV
                results_path = os.path.join(self.output_dir, "predictions_by_regime.csv")
                results_df.to_csv(results_path, index=False)
            except Exception as e:
                print(f"Error creating results DataFrame: {e}")
                print(f"Predictions shape: {preds.shape}")
                print(f"Targets shape: {targs.shape}")
                print(f"Dates shape: {np.array(dates).shape}")
        except Exception as e:
            print(f"Error saving results: {e}")
        
        print(f"Results saved to {self.output_dir}")

    def compare_models(self, other_model, model_name="Cognitive", other_model_name="Baseline"):
        """
        Compare performance between two models across different regimes
        
        Args:
            other_model: Another trained model to compare against
            model_name: Name of this model
            other_model_name: Name of the other model
        """
        # Evaluate the other model
        other_evaluator = RegimeEvaluator(
            other_model, 
            self.data_loader,
            self.regime_labels,
            self.regime_names,
            output_dir=os.path.join(self.output_dir, "comparison")
        )
        other_metrics = other_evaluator.evaluate()
        
        # Get metrics for this model
        this_metrics = self.calculate_metrics()
        
        # Prepare comparison plot
        unique_regimes = [name for name in this_metrics.keys() if name != "overall"]
        
        comparison_data = {
            "regime": unique_regimes + ["overall"],
            f"{model_name}_mse": [this_metrics[r]["mse"] for r in unique_regimes] + [this_metrics["overall"]["mse"]],
            f"{other_model_name}_mse": [other_metrics[r]["mse"] for r in unique_regimes] + [other_metrics["overall"]["mse"]],
            f"{model_name}_dir_acc": [this_metrics[r]["direction_accuracy"] for r in unique_regimes] + [this_metrics["overall"]["direction_accuracy"]],
            f"{other_model_name}_dir_acc": [other_metrics[r]["direction_accuracy"] for r in unique_regimes] + [other_metrics["overall"]["direction_accuracy"]]
        }
        
        # Calculate improvement percentages
        comparison_data["mse_improvement"] = [
            ((other_metrics[r]["mse"] - this_metrics[r]["mse"]) / other_metrics[r]["mse"] * 100) 
            for r in unique_regimes + ["overall"]
        ]
        
        comparison_data["dir_acc_improvement"] = [
            ((this_metrics[r]["direction_accuracy"] - other_metrics[r]["direction_accuracy"]) / other_metrics[r]["direction_accuracy"] * 100) 
            for r in unique_regimes + ["overall"]
        ]
        
        # Create a DataFrame
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df.to_csv(os.path.join(self.output_dir, "model_comparison.csv"), index=False)
        
        # Create comparison visualizations
        self._plot_model_comparison(comparison_data, model_name, other_model_name)
        
        return comparison_df
    
    def _plot_model_comparison(self, comparison_data, model_name, other_model_name):
        """Plot performance comparison between models"""
        regimes = comparison_data["regime"]
        
        # Set up plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # MSE comparison
        x = np.arange(len(regimes))
        width = 0.35
        
        ax1.bar(x - width/2, comparison_data[f"{model_name}_mse"], width, label=model_name)
        ax1.bar(x + width/2, comparison_data[f"{other_model_name}_mse"], width, label=other_model_name)
        
        ax1.set_ylabel('Mean Squared Error')
        ax1.set_title('MSE Comparison by Market Regime')
        ax1.set_xticks(x)
        ax1.set_xticklabels(regimes)
        ax1.legend()
        
        # Add improvement percentages
        for i, v in enumerate(comparison_data["mse_improvement"]):
            color = 'green' if v > 0 else 'red'
            ax1.text(i, max(comparison_data[f"{model_name}_mse"][i], 
                           comparison_data[f"{other_model_name}_mse"][i]) + 0.001,
                    f"{v:.1f}%", color=color, ha='center')
        
        # Direction accuracy comparison
        ax2.bar(x - width/2, np.array(comparison_data[f"{model_name}_dir_acc"]) * 100, width, label=model_name)
        ax2.bar(x + width/2, np.array(comparison_data[f"{other_model_name}_dir_acc"]) * 100, width, label=other_model_name)
        
        ax2.set_ylabel('Direction Accuracy (%)')
        ax2.set_title('Direction Accuracy Comparison by Market Regime')
        ax2.set_xticks(x)
        ax2.set_xticklabels(regimes)
        ax2.legend()
        
        # Add improvement percentages
        for i, v in enumerate(comparison_data["dir_acc_improvement"]):
            color = 'green' if v > 0 else 'red'
            ax2.text(i, max(comparison_data[f"{model_name}_dir_acc"][i], 
                           comparison_data[f"{other_model_name}_dir_acc"][i]) * 100 + 1,
                    f"{v:.1f}%", color=color, ha='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "model_comparison.png"))
        plt.close()

def evaluate_regime_performance(model_path, test_data_path, output_dir="evaluation/regime_evaluation", device=None):
    """
    Evaluate model performance by market regime
    
    Args:
        model_path: Path to model weights
        test_data_path: Path to test data
        output_dir: Directory to save evaluation results
        device: Computation device
        
    Returns:
        Dictionary with regime-specific evaluation results
    """
    # Set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Import required modules
    from src.arch.cognitive import CognitiveArchitecture
    from src.data.financial_loader import EnhancedFinancialDataLoader
    
    # Load model
    model = CognitiveArchitecture()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Create data loader
    test_data_loader = EnhancedFinancialDataLoader(test_data_path, batch_size=64, device=device)
    
    # Create regime evaluator
    evaluator = RegimeEvaluator(model, test_data_loader, model.regime_labels, model.regime_names, output_dir)
    
    # Evaluate model
    results = evaluator.evaluate(device)
    
    return results
