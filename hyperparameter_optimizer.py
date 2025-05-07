#!/usr/bin/env python
# hyperparameter_optimizer.py - Hyperparameter optimization framework

import os
import sys
import argparse
import json
import time
import numpy as np
import torch
from datetime import datetime
from typing import Dict, List, Any, Callable, Optional, Union
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import matplotlib.pyplot as plt
import pandas as pd

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from src.arch.cognitive import CognitiveArchitecture
from src.arch.baseline_lstm import FinancialLSTMBaseline
from src.data.financial_loader import EnhancedFinancialDataLoader
from train_cognitive import create_component_optimizer, train
from src.utils.regularization import RegularizationManager

class HyperparameterOptimizer:
    """Hyperparameter optimization framework for cognitive models"""
    
    def __init__(
        self, 
        train_path: str,
        val_path: str,
        model_type: str = "cognitive",
        output_dir: str = "hyperopt_results",
        n_trials: int = 50,
        timeout: Optional[int] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize hyperparameter optimizer
        
        Args:
            train_path: Path to training data
            val_path: Path to validation data
            model_type: Model type ('cognitive' or 'baseline')
            output_dir: Directory to save results
            n_trials: Number of optimization trials
            timeout: Timeout in seconds (optional)
            device: Computation device
        """
        self.train_path = train_path
        self.val_path = val_path
        self.model_type = model_type
        self.output_dir = output_dir
        self.n_trials = n_trials
        self.timeout = timeout
        self.device = device
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Study name based on model type and timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.study_name = f"{model_type}_hyperopt_{timestamp}"
        
        # Initialize study
        self.study = optuna.create_study(
            study_name=self.study_name,
            direction="maximize",
            sampler=TPESampler(seed=42),
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        )
        
        # Store best parameters and metrics
        self.best_params = None
        self.best_val_metrics = None
        self.trials_history = []
        
    def suggest_cognitive_hyperparams(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Suggest hyperparameters for cognitive architecture
        
        Args:
            trial: Optuna trial
            
        Returns:
            Dictionary of hyperparameters
        """
        params = {
            # Model architecture
            "hidden_dim": trial.suggest_categorical("hidden_dim", [32, 64, 96, 128]),
            "memory_size": trial.suggest_int("memory_size", 20, 100, step=10),
            "num_attention_heads": trial.suggest_int("num_attention_heads", 1, 4),
            
            # Training parameters
            "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
            "sequence_length": trial.suggest_categorical("sequence_length", [10, 20, 40, 60]),
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
            
            # Regularization
            "dropout": trial.suggest_float("dropout", 0.1, 0.5),
            "l2_lambda": trial.suggest_float("l2_lambda", 1e-6, 1e-4, log=True),
            "gradient_clip": trial.suggest_float("gradient_clip", 0.1, 1.0),
            
            # Component-specific parameters
            "memory_lr_multiplier": trial.suggest_float("memory_lr_multiplier", 0.5, 2.0),
            "attention_entropy_lambda": trial.suggest_float("attention_entropy_lambda", 0.0, 0.1),
            
            # Training approach
            "progressive_training": trial.suggest_categorical("progressive_training", [True, False]),
            "regime_aware": trial.suggest_categorical("regime_aware", [True, False])
        }
        
        return params
    
    def suggest_baseline_hyperparams(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Suggest hyperparameters for baseline LSTM
        
        Args:
            trial: Optuna trial
            
        Returns:
            Dictionary of hyperparameters
        """
        params = {
            # Model architecture
            "hidden_dim": trial.suggest_categorical("hidden_dim", [32, 64, 128, 256]),
            "num_layers": trial.suggest_int("num_layers", 1, 3),
            
            # Training parameters
            "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
            "sequence_length": trial.suggest_categorical("sequence_length", [10, 20, 40, 60]),
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
            
            # Regularization
            "dropout": trial.suggest_float("dropout", 0.1, 0.5),
            "l2_lambda": trial.suggest_float("l2_lambda", 1e-6, 1e-4, log=True),
            "gradient_clip": trial.suggest_float("gradient_clip", 0.1, 1.0)
        }
        
        return params
    
    def objective(self, trial: optuna.Trial) -> float:
        """
        Optuna objective function for hyperparameter optimization
        
        Args:
            trial: Optuna trial
            
        Returns:
            Validation metric to maximize
        """
        # Suggest hyperparameters based on model type
        if self.model_type.lower() == "cognitive":
            params = self.suggest_cognitive_hyperparams(trial)
        else:
            params = self.suggest_baseline_hyperparams(trial)
        
        # Create model
        if self.model_type.lower() == "cognitive":
            model = CognitiveArchitecture(
                hidden_dim=params["hidden_dim"],
                memory_size=params["memory_size"]
            )
        else:
            model = FinancialLSTMBaseline(
                hidden_dim=params["hidden_dim"],
                num_layers=params.get("num_layers", 2),
                dropout=params["dropout"]
            )
        
        # Move model to device
        model.to(self.device)
        
        # Create data loaders
        train_loader = EnhancedFinancialDataLoader(
            data_path=self.train_path,
            sequence_length=params["sequence_length"],
            batch_size=params["batch_size"],
            regime_aware=params.get("regime_aware", False),
            augmentation=True
        )
        
        val_loader = EnhancedFinancialDataLoader(
            data_path=self.val_path,
            sequence_length=params["sequence_length"],
            batch_size=params["batch_size"],
            regime_aware=params.get("regime_aware", False),
            augmentation=False
        )
        
        # Create optimizers
        if self.model_type.lower() == "cognitive":
            # Adjust component-specific learning rates
            base_lr = params["learning_rate"]
            optimizers = create_component_optimizer(
                model=model, 
                base_lr=base_lr
            )
            
            # Adjust memory learning rate if specified
            if "memory_lr_multiplier" in params and "memory" in optimizers:
                for param_group in optimizers["memory"].param_groups:
                    param_group["lr"] = base_lr * params["memory_lr_multiplier"]
        else:
            # Single optimizer for baseline model
            optimizers = {
                "all": torch.optim.Adam(
                    model.parameters(), 
                    lr=params["learning_rate"],
                    weight_decay=params["l2_lambda"]
                )
            }
        
        # Create regularization manager
        regularization = RegularizationManager(
            l2_lambda=params["l2_lambda"],
            gradient_clip_threshold=params["gradient_clip"],
            attention_entropy_lambda=params.get("attention_entropy_lambda", 0.0)
        )
        
        # Register regularization hooks
        regularization.register_hooks(model)
        
        # Create checkpoint and log directories for this trial
        trial_dir = os.path.join(self.output_dir, f"trial_{trial.number}")
        checkpoint_dir = os.path.join(trial_dir, "checkpoints")
        log_dir = os.path.join(trial_dir, "logs")
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        # Train model
        start_time = time.time()
        
        try:
            # Progressive or standard training
            if self.model_type.lower() == "cognitive" and params.get("progressive_training", False):
                from train_cognitive import train_with_progressive_focus
                
                training_result = train_with_progressive_focus(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    device=self.device,
                    base_lr=params["learning_rate"],
                    total_epochs=20,  # Reduced epochs for hyperopt
                    checkpoint_dir=checkpoint_dir,
                    log_dir=log_dir,
                    early_stopping_patience=5
                )
            else:
                training_result = train(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    optimizers=optimizers,
                    device=self.device,
                    num_epochs=20,  # Reduced epochs for hyperopt
                    checkpoint_dir=checkpoint_dir,
                    log_dir=log_dir,
                    early_stopping_patience=5
                )
        
        except Exception as e:
            print(f"Error during training: {e}")
            # Return a very low score to indicate failure
            return -100.0
        
        training_time = time.time() - start_time
        
        # Get validation metrics
        val_metrics = training_result.get('best_val_metrics', {})
        
        # Calculate combined score (prioritize price and direction accuracy)
        combined_score = (
            0.4 * val_metrics.get('price_accuracy', 0.0) + 
            0.4 * val_metrics.get('direction_accuracy', 0.0) + 
            0.2 * (1.0 - min(1.0, val_metrics.get('val_loss', 1.0)))
        )
        
        # Store trial results
        trial_result = {
            'trial_number': trial.number,
            'params': params,
            'val_metrics': val_metrics,
            'combined_score': combined_score,
            'training_time': training_time
        }
        self.trials_history.append(trial_result)
        
        # Save trial results
        with open(os.path.join(trial_dir, 'result.json'), 'w') as f:
            json.dump(trial_result, f, indent=2)
        
        # Report intermediate values for pruning
        trial.report(combined_score, step=1)
        
        # Check for pruning
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        
        # Update best parameters
        if self.best_val_metrics is None or combined_score > self.best_params.get('combined_score', -float('inf')):
            self.best_params = params
            self.best_val_metrics = val_metrics
        
        return combined_score
    
    def run_optimization(self) -> Dict[str, Any]:
        """
        Run hyperparameter optimization
        
        Returns:
            Best parameters and metrics
        """
        # Start optimization
        print(f"Starting hyperparameter optimization for {self.model_type} model")
        print(f"Number of trials: {self.n_trials}")
        print(f"Timeout: {self.timeout} seconds" if self.timeout else "No timeout")
        print(f"Output directory: {self.output_dir}")
        print(f"Study name: {self.study_name}")
        
        try:
            # Run optimization
            self.study.optimize(
                self.objective, 
                n_trials=self.n_trials,
                timeout=self.timeout,
                show_progress_bar=True
            )
            
            # Get best trial
            best_trial = self.study.best_trial
            
            print("\nOptimization finished!")
            print(f"Best trial: #{best_trial.number}")
            print(f"Best value: {best_trial.value:.4f}")
            print("Best hyperparameters:")
            for param, value in best_trial.params.items():
                print(f"  {param}: {value}")
            
            # Save study results
            self.save_results()
            
            # Generate plots
            self.generate_plots()
            
            # Return best parameters and metrics
            return {
                'best_params': best_trial.params,
                'best_value': best_trial.value,
                'best_trial': best_trial.number,
                'n_trials': len(self.study.trials)
            }
            
        except KeyboardInterrupt:
            print("\nOptimization interrupted!")
            print("Saving current results...")
            self.save_results()
            self.generate_plots()
            
            # Return best results so far
            best_trial = self.study.best_trial
            return {
                'best_params': best_trial.params,
                'best_value': best_trial.value,
                'best_trial': best_trial.number,
                'n_trials': len(self.study.trials),
                'interrupted': True
            }
    
    def save_results(self) -> None:
        """Save optimization results"""
        # Save study statistics
        study_info = {
            'study_name': self.study_name,
            'model_type': self.model_type,
            'direction': self.study.direction.name,
            'n_trials': len(self.study.trials),
            'datetime': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'best_trial': self.study.best_trial.number,
            'best_value': self.study.best_trial.value,
            'best_params': self.study.best_trial.params
        }
        
        with open(os.path.join(self.output_dir, 'study_info.json'), 'w') as f:
            json.dump(study_info, f, indent=2)
        
        # Save all trials data
        trials_data = []
        for trial in self.study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                trials_data.append({
                    'number': trial.number,
                    'value': trial.value,
                    'params': trial.params,
                    'datetime_start': trial.datetime_start.strftime("%Y-%m-%d %H:%M:%S"),
                    'datetime_complete': trial.datetime_complete.strftime("%Y-%m-%d %H:%M:%S")
                })
        
        with open(os.path.join(self.output_dir, 'trials_data.json'), 'w') as f:
            json.dump(trials_data, f, indent=2)
        
        # Save trials history (including validation metrics)
        with open(os.path.join(self.output_dir, 'trials_history.json'), 'w') as f:
            json.dump(self.trials_history, f, indent=2)
        
        print(f"Results saved to {self.output_dir}")
    
    def generate_plots(self) -> None:
        """Generate optimization visualization plots"""
        # Create plots directory
        plots_dir = os.path.join(self.output_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # 1. Optimization history plot
        plt.figure(figsize=(10, 6))
        optuna.visualization.matplotlib.plot_optimization_history(self.study)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'optimization_history.png'), dpi=300)
        plt.close()
        
        # 2. Parallel coordinate plot
        plt.figure(figsize=(12, 8))
        optuna.visualization.matplotlib.plot_parallel_coordinate(
            self.study, 
            params=list(self.study.best_trial.params.keys())
        )
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'parallel_coordinate.png'), dpi=300)
        plt.close()
        
        # 3. Parameter importances
        try:
            plt.figure(figsize=(10, 6))
            optuna.visualization.matplotlib.plot_param_importances(self.study)
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, 'param_importances.png'), dpi=300)
            plt.close()
        except:
            print("Could not generate parameter importances plot")
        
        # 4. Slice plot for key parameters
        try:
            for param in ['learning_rate', 'hidden_dim', 'batch_size', 'sequence_length']:
                if param in self.study.best_trial.params:
                    plt.figure(figsize=(10, 6))
                    optuna.visualization.matplotlib.plot_slice(self.study, params=[param])
                    plt.tight_layout()
                    plt.savefig(os.path.join(plots_dir, f'slice_{param}.png'), dpi=300)
                    plt.close()
        except:
            print("Could not generate slice plots")
        
        # 5. Custom metric comparison (if we have metrics in trials history)
        if self.trials_history and 'val_metrics' in self.trials_history[0]:
            # Extract trials with metrics
            valid_trials = [
                trial for trial in self.trials_history 
                if 'val_metrics' in trial and trial['val_metrics'] is not None
            ]
            
            if valid_trials:
                # Extract key metrics
                trial_nums = [trial['trial_number'] for trial in valid_trials]
                
                metrics_to_plot = ['price_accuracy', 'direction_accuracy', 'val_loss']
                for metric_name in metrics_to_plot:
                    metric_values = [
                        trial['val_metrics'].get(metric_name, 0) 
                        for trial in valid_trials
                    ]
                    
                    # Skip if no values
                    if not any(metric_values):
                        continue
                    
                    # Create plot
                    plt.figure(figsize=(10, 6))
                    plt.plot(trial_nums, metric_values, 'o-', linewidth=2)
                    plt.axhline(y=max(metric_values) if 'accuracy' in metric_name else min(metric_values), 
                              color='r', linestyle='--')
                    
                    plt.title(f"{metric_name.replace('_', ' ').title()} Across Trials")
                    plt.xlabel("Trial Number")
                    plt.ylabel(metric_name.replace('_', ' ').title())
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    
                    plt.savefig(os.path.join(plots_dir, f'{metric_name}_history.png'), dpi=300)
                    plt.close()
        
        print(f"Plots saved to {plots_dir}")
    
    def train_best_model(self, epochs=50) -> Dict[str, Any]:
        """
        Train a model with the best hyperparameters
        
        Args:
            epochs: Number of training epochs
            
        Returns:
            Training results
        """
        if self.best_params is None:
            print("No best parameters found. Run optimization first.")
            return {}
        
        print(f"Training model with best hyperparameters for {epochs} epochs")
        
        # Create model
        if self.model_type.lower() == "cognitive":
            model = CognitiveArchitecture(
                hidden_dim=self.best_params["hidden_dim"],
                memory_size=self.best_params["memory_size"]
            )
        else:
            model = FinancialLSTMBaseline(
                hidden_dim=self.best_params["hidden_dim"],
                num_layers=self.best_params.get("num_layers", 2),
                dropout=self.best_params["dropout"]
            )
        
        # Move model to device
        model.to(self.device)
        
        # Create data loaders
        train_loader = EnhancedFinancialDataLoader(
            data_path=self.train_path,
            sequence_length=self.best_params["sequence_length"],
            batch_size=self.best_params["batch_size"],
            regime_aware=self.best_params.get("regime_aware", False),
            augmentation=True
        )
        
        val_loader = EnhancedFinancialDataLoader(
            data_path=self.val_path,
            sequence_length=self.best_params["sequence_length"],
            batch_size=self.best_params["batch_size"],
            regime_aware=self.best_params.get("regime_aware", False),
            augmentation=False
        )
        
        # Create optimizers
        if self.model_type.lower() == "cognitive":
            # Adjust component-specific learning rates
            base_lr = self.best_params["learning_rate"]
            optimizers = create_component_optimizer(
                model=model, 
                base_lr=base_lr
            )
            
            # Adjust memory learning rate if specified
            if "memory_lr_multiplier" in self.best_params and "memory" in optimizers:
                for param_group in optimizers["memory"].param_groups:
                    param_group["lr"] = base_lr * self.best_params["memory_lr_multiplier"]
        else:
            # Single optimizer for baseline model
            optimizers = {
                "all": torch.optim.Adam(
                    model.parameters(), 
                    lr=self.best_params["learning_rate"],
                    weight_decay=self.best_params["l2_lambda"]
                )
            }
        
        # Create regularization manager
        regularization = RegularizationManager(
            l2_lambda=self.best_params["l2_lambda"],
            gradient_clip_threshold=self.best_params["gradient_clip"],
            attention_entropy_lambda=self.best_params.get("attention_entropy_lambda", 0.0)
        )
        
        # Register regularization hooks
        regularization.register_hooks(model)
        
        # Create checkpoint and log directories
        best_model_dir = os.path.join(self.output_dir, "best_model")
        checkpoint_dir = os.path.join(best_model_dir, "checkpoints")
        log_dir = os.path.join(best_model_dir, "logs")
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        # Train model
        start_time = time.time()
        
        try:
            # Progressive or standard training
            if self.model_type.lower() == "cognitive" and self.best_params.get("progressive_training", False):
                from train_cognitive import train_with_progressive_focus
                
                training_result = train_with_progressive_focus(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    device=self.device,
                    base_lr=self.best_params["learning_rate"],
                    total_epochs=epochs,
                    checkpoint_dir=checkpoint_dir,
                    log_dir=log_dir,
                    early_stopping_patience=10
                )
            else:
                training_result = train(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    optimizers=optimizers,
                    device=self.device,
                    num_epochs=epochs,
                    checkpoint_dir=checkpoint_dir,
                    log_dir=log_dir,
                    early_stopping_patience=10
                )
        
        except Exception as e:
            print(f"Error during best model training: {e}")
            return {'error': str(e)}
        
        training_time = time.time() - start_time
        
        # Save best parameters with the model
        with open(os.path.join(best_model_dir, 'best_params.json'), 'w') as f:
            json.dump(self.best_params, f, indent=2)
        
        print(f"Best model trained and saved to {best_model_dir}")
        
        return {
            'best_params': self.best_params,
            'training_result': training_result,
            'training_time': training_time
        }

def main():
    parser = argparse.ArgumentParser(description="Hyperparameter Optimization for Cognitive Models")
    
    parser.add_argument("--train", required=True, help="Path to training data")
    parser.add_argument("--val", required=True, help="Path to validation data")
    parser.add_argument("--model_type", choices=["cognitive", "baseline"], default="cognitive", 
                      help="Model type to optimize")
    parser.add_argument("--output", default="hyperopt_results", help="Output directory")
    parser.add_argument("--trials", type=int, default=50, help="Number of optimization trials")
    parser.add_argument("--timeout", type=int, help="Timeout in seconds (optional)")
    parser.add_argument("--train_best", action="store_true", 
                      help="Train a full model with the best hyperparameters")
    parser.add_argument("--epochs", type=int, default=50, 
                      help="Number of epochs for training the best model")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                      help="Computation device")
    
    args = parser.parse_args()
    
    # Create optimizer
    optimizer = HyperparameterOptimizer(
        train_path=args.train,
        val_path=args.val,
        model_type=args.model_type,
        output_dir=args.output,
        n_trials=args.trials,
        timeout=args.timeout,
        device=args.device
    )
    
    # Run optimization
    results = optimizer.run_optimization()
    
    # Train best model if requested
    if args.train_best:
        best_model_results = optimizer.train_best_model(epochs=args.epochs)
        print("Best model training completed")

if __name__ == "__main__":
    main()
