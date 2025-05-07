#!/usr/bin/env python
# evaluation_utils.py - Evaluation utilities for cognitive architecture

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score

def calculate_financial_metrics(predictions, targets, return_all=False):
    """
    Calculate financial performance metrics
    
    Args:
        predictions: Model predictions (numpy array)
        targets: Actual target values (numpy array)
        return_all: Whether to return detailed metrics
        
    Returns:
        Dictionary of financial performance metrics
    """
    # Initialize metrics
    metrics = {}
    
    # Ensure predictions and targets are numpy arrays
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    
    # Ensure we have 2D arrays (samples × features)
    if len(predictions.shape) > 2:
        predictions = predictions.reshape(predictions.shape[0], -1)
    if len(targets.shape) > 2:
        targets = targets.reshape(targets.shape[0], -1)
    
    # Price prediction metrics (assuming first column is price)
    price_pred = predictions[:, 0]
    price_true = targets[:, 0]
    
    # Price correlation
    metrics['price_correlation'] = float(np.corrcoef(price_pred, price_true)[0, 1])
    
    # Mean Squared Error
    metrics['price_mse'] = float(np.mean((price_pred - price_true)**2))
    
    # Mean Absolute Error
    metrics['price_mae'] = float(np.mean(np.abs(price_pred - price_true)))
    
    # Direction accuracy (next day up/down)
    if len(price_pred) > 1:
        pred_dir = np.sign(price_pred[1:] - price_pred[:-1])
        true_dir = np.sign(price_true[1:] - price_true[:-1])
        
        # Handle zeros (flat)
        pred_dir = np.where(pred_dir == 0, 0.01, pred_dir)
        true_dir = np.where(true_dir == 0, 0.01, true_dir)
        
        # Calculate direction accuracy
        metrics['direction_accuracy'] = float(np.mean(pred_dir == true_dir))
        
        # Calculate up/down precision and recall
        # Precision: Of all predicted ups, how many were actually up
        # Recall: Of all actual ups, how many were predicted as up
        actual_ups = (true_dir > 0)
        predicted_ups = (pred_dir > 0)
        
        # Avoid division by zero
        if np.sum(predicted_ups) > 0:
            metrics['up_precision'] = float(np.sum(actual_ups & predicted_ups) / np.sum(predicted_ups))
        else:
            metrics['up_precision'] = 0.0
            
        if np.sum(actual_ups) > 0:
            metrics['up_recall'] = float(np.sum(actual_ups & predicted_ups) / np.sum(actual_ups))
        else:
            metrics['up_recall'] = 0.0
    
    # Volatility prediction (assuming fourth column is volatility)
    if predictions.shape[1] >= 4 and targets.shape[1] >= 4:
        vol_pred = predictions[:, 3]
        vol_true = targets[:, 3]
        
        # Volatility correlation
        metrics['volatility_correlation'] = float(np.corrcoef(vol_pred, vol_true)[0, 1])
        
        # Volatility MSE
        metrics['volatility_mse'] = float(np.mean((vol_pred - vol_true)**2))
    
    # Return only key metrics unless detailed metrics requested
    if not return_all:
        return {
            'price_correlation': metrics['price_correlation'],
            'direction_accuracy': metrics.get('direction_accuracy', 0.0),
            'volatility_correlation': metrics.get('volatility_correlation', 0.0)
        }
    
    return metrics

def calculate_cognitive_metrics(outputs, targets, memory_outputs=None):
    """
    Calculate cognitive-specific performance metrics
    
    Args:
        outputs: Model outputs (dictionary or tensor)
        targets: Actual target values
        memory_outputs: Optional memory access patterns
        
    Returns:
        Dictionary of cognitive performance metrics
    """
    # Initialize metrics
    metrics = {}
    
    # Check if outputs is a dictionary
    if isinstance(outputs, dict):
        # Extract uncertainty if available
        if 'uncertainty' in outputs and outputs['uncertainty'] is not None:
            uncertainty = outputs['uncertainty']
            if isinstance(uncertainty, torch.Tensor):
                uncertainty = uncertainty.detach().cpu().numpy()
            
            # Average uncertainty
            metrics['avg_uncertainty'] = float(np.mean(uncertainty))
            
            # Uncertainty calibration
            if 'market_state' in outputs:
                predictions = outputs['market_state']
                if isinstance(predictions, torch.Tensor):
                    predictions = predictions.detach().cpu().numpy()
                
                if isinstance(targets, torch.Tensor):
                    targets = targets.detach().cpu().numpy()
                
                # Calculate prediction errors
                errors = np.abs(predictions - targets)
                
                # Correlation between uncertainty and error
                error_mean = np.mean(errors, axis=1)
                uncertainty_mean = np.mean(uncertainty, axis=1) if uncertainty.ndim > 1 else uncertainty
                
                # Calculate uncertainty calibration (correlation)
                try:
                    metrics['uncertainty_calibration'] = float(np.corrcoef(uncertainty_mean, error_mean)[0, 1])
                except:
                    metrics['uncertainty_calibration'] = 0.0
        
        # Extract regime information if available
        if 'regime_probabilities' in outputs and outputs['regime_probabilities'] is not None:
            regime_probs = outputs['regime_probabilities']
            if isinstance(regime_probs, torch.Tensor):
                regime_probs = regime_probs.detach().cpu().numpy()
            
            # Calculate regime certainty (max probability)
            metrics['regime_certainty'] = float(np.mean(np.max(regime_probs, axis=-1)))
            
            # Calculate regime entropy
            regime_entropy = -np.sum(regime_probs * np.log(regime_probs + 1e-10), axis=-1)
            metrics['regime_entropy'] = float(np.mean(regime_entropy))
    
    # Calculate memory efficiency if memory outputs provided
    if memory_outputs is not None:
        if 'access_count' in memory_outputs and 'total_cells' in memory_outputs:
            access_count = memory_outputs['access_count']
            total_cells = memory_outputs['total_cells']
            
            # Calculate memory utilization (percentage of accessed cells)
            metrics['memory_utilization'] = float(access_count / total_cells) if total_cells > 0 else 0.0
        
        if 'retrieval_accuracy' in memory_outputs:
            metrics['memory_retrieval_accuracy'] = float(memory_outputs['retrieval_accuracy'])
    
    return metrics

def calculate_regime_detection_accuracy(predicted_regimes, true_regimes):
    """
    Calculate regime detection accuracy metrics
    
    Args:
        predicted_regimes: Model's regime predictions (class indices or probabilities)
        true_regimes: True regime labels
        
    Returns:
        Dictionary of regime detection metrics
    """
    # Initialize metrics
    metrics = {}
    
    # Convert to numpy arrays
    if isinstance(predicted_regimes, torch.Tensor):
        predicted_regimes = predicted_regimes.detach().cpu().numpy()
    if isinstance(true_regimes, torch.Tensor):
        true_regimes = true_regimes.detach().cpu().numpy()
    
    # If predicted_regimes are probabilities, convert to class indices
    if predicted_regimes.ndim > 1 and predicted_regimes.shape[1] > 1:
        predicted_classes = np.argmax(predicted_regimes, axis=1)
    else:
        predicted_classes = predicted_regimes
    
    # Ensure true_regimes are class indices
    if true_regimes.ndim > 1 and true_regimes.shape[1] > 1:
        true_classes = np.argmax(true_regimes, axis=1)
    else:
        true_classes = true_regimes
    
    # Calculate accuracy
    metrics['regime_accuracy'] = float(np.mean(predicted_classes == true_classes))
    
    # Calculate confusion matrix
    try:
        cm = confusion_matrix(true_classes, predicted_classes)
        
        # Normalize by row (true labels)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Store normalized confusion matrix
        metrics['regime_confusion'] = cm_normalized.tolist()
        
        # Calculate per-class precision
        precision = np.diag(cm) / np.sum(cm, axis=0)
        
        # Handle division by zero
        precision = np.nan_to_num(precision)
        
        # Calculate per-class recall
        recall = np.diag(cm) / np.sum(cm, axis=1)
        
        # Average precision and recall
        metrics['regime_precision'] = float(np.mean(precision))
        metrics['regime_recall'] = float(np.mean(recall))
        
        # Calculate F1 score
        f1 = 2 * (precision * recall) / (precision + recall)
        f1 = np.nan_to_num(f1)
        metrics['regime_f1'] = float(np.mean(f1))
    except:
        # Handle cases where confusion matrix calculation fails
        metrics['regime_precision'] = 0.0
        metrics['regime_recall'] = 0.0
        metrics['regime_f1'] = 0.0
    
    return metrics

def plot_prediction_vs_actual(predictions, targets, timestamps=None, title="Prediction vs Actual", 
                            output_path=None):
    """
    Create plot of predictions vs actual values
    
    Args:
        predictions: Model predictions (numpy array)
        targets: Actual target values (numpy array)
        timestamps: Optional list of timestamps for x-axis
        title: Plot title
        output_path: Path to save the plot
        
    Returns:
        Matplotlib figure object
    """
    # Convert to numpy arrays
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    
    # Ensure we have 2D arrays (samples × features)
    if len(predictions.shape) > 2:
        predictions = predictions.reshape(predictions.shape[0], -1)
    if len(targets.shape) > 2:
        targets = targets.reshape(targets.shape[0], -1)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create x-axis values
    x = range(len(predictions)) if timestamps is None else timestamps
    
    # Plot predictions vs actual
    ax.plot(x, predictions[:, 0], 'b-', label='Predicted', linewidth=2)
    ax.plot(x, targets[:, 0], 'r-', label='Actual', linewidth=2)
    
    # Calculate metrics
    mse = np.mean((predictions[:, 0] - targets[:, 0])**2)
    corr = np.corrcoef(predictions[:, 0], targets[:, 0])[0, 1]
    
    # Add metrics to title
    title = f"{title} (MSE: {mse:.4f}, Corr: {corr:.4f})"
    ax.set_title(title)
    
    # Set labels
    ax.set_xlabel('Time')
    ax.set_ylabel('Price')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Save if output path provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_uncertainty(predictions, targets, uncertainty, timestamps=None, output_path=None):
    """
    Create plot of predictions with uncertainty intervals
    
    Args:
        predictions: Model predictions (numpy array)
        targets: Actual target values (numpy array)
        uncertainty: Prediction uncertainty (numpy array)
        timestamps: Optional list of timestamps for x-axis
        output_path: Path to save the plot
        
    Returns:
        Matplotlib figure object
    """
    # Convert to numpy arrays
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    if isinstance(uncertainty, torch.Tensor):
        uncertainty = uncertainty.detach().cpu().numpy()
    
    # Ensure we have 2D arrays (samples × features)
    if len(predictions.shape) > 2:
        predictions = predictions.reshape(predictions.shape[0], -1)
    if len(targets.shape) > 2:
        targets = targets.reshape(targets.shape[0], -1)
    
    # Handle uncertainty shapes
    if len(uncertainty.shape) == 1:
        uncertainty = uncertainty.reshape(-1, 1)
    elif len(uncertainty.shape) > 2:
        uncertainty = uncertainty.reshape(uncertainty.shape[0], -1)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create x-axis values
    x = range(len(predictions)) if timestamps is None else timestamps
    
    # Plot predictions vs actual
    ax.plot(x, predictions[:, 0], 'b-', label='Predicted', linewidth=2)
    ax.plot(x, targets[:, 0], 'r-', label='Actual', linewidth=2)
    
    # Add uncertainty bands (assume uncertainty is standard deviation)
    uncertainty_column = min(uncertainty.shape[1] - 1, 0)
    lower_bound = predictions[:, 0] - 2 * uncertainty[:, uncertainty_column]
    upper_bound = predictions[:, 0] + 2 * uncertainty[:, uncertainty_column]
    
    ax.fill_between(x, lower_bound, upper_bound, color='blue', alpha=0.2, label='95% Confidence')
    
    # Calculate metrics
    mse = np.mean((predictions[:, 0] - targets[:, 0])**2)
    corr = np.corrcoef(predictions[:, 0], targets[:, 0])[0, 1]
    
    # Calculate uncertainty calibration (percentage of true values within bounds)
    within_bounds = np.mean((targets[:, 0] >= lower_bound) & (targets[:, 0] <= upper_bound))
    
    # Add metrics to title
    title = f"Prediction with Uncertainty (MSE: {mse:.4f}, Calibration: {within_bounds:.2%})"
    ax.set_title(title)
    
    # Set labels
    ax.set_xlabel('Time')
    ax.set_ylabel('Price')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Save if output path provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_regime_probabilities(regime_probs, timestamps=None, regime_names=None, output_path=None):
    """
    Create plot of regime probability evolution
    
    Args:
        regime_probs: Regime probabilities over time (samples × regimes)
        timestamps: Optional list of timestamps for x-axis
        regime_names: Optional list of regime names
        output_path: Path to save the plot
        
    Returns:
        Matplotlib figure object
    """
    # Convert to numpy array
    if isinstance(regime_probs, torch.Tensor):
        regime_probs = regime_probs.detach().cpu().numpy()
    
    # Ensure we have a 2D array (samples × regimes)
    if len(regime_probs.shape) != 2:
        print(f"Unexpected regime probability shape: {regime_probs.shape}")
        return None
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Create x-axis values
    x = range(len(regime_probs)) if timestamps is None else timestamps
    
    # Create default regime names if not provided
    if regime_names is None:
        regime_names = [f"Regime {i+1}" for i in range(regime_probs.shape[1])]
    
    # Plot each regime probability
    for i in range(regime_probs.shape[1]):
        ax.plot(x, regime_probs[:, i], linewidth=2, label=regime_names[i])
    
    # Add regime transitions (where max probability regime changes)
    max_regime_indices = np.argmax(regime_probs, axis=1)
    transitions = []
    
    for i in range(1, len(max_regime_indices)):
        if max_regime_indices[i] != max_regime_indices[i-1]:
            transitions.append(i)
    
    # Add vertical lines at transitions
    for transition in transitions:
        if timestamps is not None:
            ax.axvline(x=timestamps[transition], color='r', linestyle='--', alpha=0.5)
        else:
            ax.axvline(x=transition, color='r', linestyle='--', alpha=0.5)
    
    # Set title and labels
    ax.set_title(f"Regime Probability Evolution (Transitions: {len(transitions)})")
    ax.set_xlabel('Time')
    ax.set_ylabel('Probability')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Save if output path provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_memory_usage(memory_usage, timestamps=None, output_path=None):
    """
    Create plot of memory usage over time
    
    Args:
        memory_usage: Memory usage metrics over time
        timestamps: Optional list of timestamps for x-axis
        output_path: Path to save the plot
        
    Returns:
        Matplotlib figure object
    """
    # Convert to numpy array
    if isinstance(memory_usage, torch.Tensor):
        memory_usage = memory_usage.detach().cpu().numpy()
    
    # Ensure we have a 1D array
    if memory_usage.ndim > 1:
        memory_usage = memory_usage.mean(axis=1)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create x-axis values
    x = range(len(memory_usage)) if timestamps is None else timestamps
    
    # Plot memory usage
    ax.plot(x, memory_usage, 'g-', linewidth=2)
    
    # Add rolling average
    window = min(20, len(memory_usage) // 5)
    if window > 1:
        rolling_avg = np.convolve(memory_usage, np.ones(window)/window, mode='valid')
        
        # Plot rolling average
        x_rolling = x[window-1:] if timestamps is None else timestamps[window-1:]
        ax.plot(x_rolling, rolling_avg, 'b--', linewidth=2, label=f'{window}-Period Moving Avg')
    
    # Set title and labels
    ax.set_title("Memory Usage Over Time")
    ax.set_xlabel('Time')
    ax.set_ylabel('Memory Usage')
    if 'label' in locals():
        ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Save if output path provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_attention_heatmap(attention_weights, feature_names=None, output_path=None):
    """
    Create heatmap of attention weights
    
    Args:
        attention_weights: Attention weights matrix
        feature_names: Optional list of feature names
        output_path: Path to save the plot
        
    Returns:
        Matplotlib figure object
    """
    # Convert to numpy array
    if isinstance(attention_weights, torch.Tensor):
        attention_weights = attention_weights.detach().cpu().numpy()
    
    # Average across batches if needed
    if len(attention_weights.shape) > 2:
        attention_weights = attention_weights.mean(axis=0)
    
    # Create figure
    plt.figure(figsize=(12, 10))
    
    # Create default feature names if not provided
    if feature_names is None:
        feature_names = [f"Feature {i+1}" for i in range(attention_weights.shape[0])]
    
    # Create heatmap
    sns.heatmap(
        attention_weights,
        cmap="viridis",
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

def compare_models(cognitive_metrics, baseline_metrics, output_path=None):
    """
    Create comparison plot between cognitive and baseline models
    
    Args:
        cognitive_metrics: Dictionary of cognitive model metrics
        baseline_metrics: Dictionary of baseline model metrics
        output_path: Path to save the plot
        
    Returns:
        Matplotlib figure object
    """
    # Find common metrics
    common_metrics = set(cognitive_metrics.keys()).intersection(set(baseline_metrics.keys()))
    
    # Filter to numerical metrics only
    numeric_metrics = []
    for metric in common_metrics:
        try:
            # Check if both values are numeric
            cognitive_value = float(cognitive_metrics[metric])
            baseline_value = float(baseline_metrics[metric])
            
            # Ensure the metric makes sense for comparison
            if not np.isnan(cognitive_value) and not np.isnan(baseline_value):
                numeric_metrics.append(metric)
        except (ValueError, TypeError):
            continue
    
    # Skip if no common numeric metrics
    if not numeric_metrics:
        print("No common numeric metrics found for comparison")
        return None
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Set width of bars
    bar_width = 0.35
    index = np.arange(len(numeric_metrics))
    
    # Create bars
    cognitive_bars = ax.bar(
        index - bar_width/2, 
        [cognitive_metrics[m] for m in numeric_metrics],
        bar_width,
        label='Cognitive Model'
    )
    
    baseline_bars = ax.bar(
        index + bar_width/2, 
        [baseline_metrics[m] for m in numeric_metrics],
        bar_width,
        label='Baseline Model'
    )
    
    # Add labels and title
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Value')
    ax.set_title('Model Comparison')
    ax.set_xticks(index)
    ax.set_xticklabels([m.replace('_', ' ').title() for m in numeric_metrics], rotation=45, ha='right')
    ax.legend()
    
    # Add value labels on bars
    for bars in [cognitive_bars, baseline_bars]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f'{height:.4f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center',
                va='bottom',
                fontsize=8
            )
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if output path provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig

def save_evaluation_results(results, output_path):
    """
    Save evaluation results to JSON file
    
    Args:
        results: Dictionary of evaluation results
        output_path: Path to save results
    """
    # Create directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Convert complex objects to JSON-serializable format
    def convert_to_serializable(obj):
        if isinstance(obj, (np.ndarray, np.generic)):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        else:
            return obj
    
    # Convert results to serializable format
    serializable_results = {}
    for key, value in results.items():
        if isinstance(value, dict):
            serializable_results[key] = {k: convert_to_serializable(v) for k, v in value.items()}
        else:
            serializable_results[key] = convert_to_serializable(value)
    
    # Save to JSON file
    with open(output_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"Evaluation results saved to {output_path}")

def comprehensive_model_evaluation(model, data_loader, output_dir, device='cpu'):
    """
    Run comprehensive evaluation on model
    
    Args:
        model: Model to evaluate
        data_loader: Data loader for evaluation
        output_dir: Directory to save evaluation results
        device: Computation device
        
    Returns:
        Dictionary of evaluation results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize results
    results = {
        'financial_metrics': {},
        'cognitive_metrics': {},
        'predictions': [],
        'targets': [],
        'timestamps': []
    }
    
    # Set model to evaluation mode
    model.eval()
    
    # Run evaluation
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            # Move data to device
            features = batch['features'].to(device)
            sequence = batch['sequence'].to(device)
            targets = batch['target'].to(device)
            
            # Store timestamps if available
            if 'timestamps' in batch:
                results['timestamps'].extend(batch['timestamps'])
            
            # Forward pass
            outputs = model(financial_data=features, financial_seq=sequence)
            
            # Extract predictions
            if isinstance(outputs, dict) and 'market_state' in outputs:
                predictions = outputs['market_state']
            else:
                predictions = outputs
            
            # Store predictions and targets
            results['predictions'].append(predictions.cpu().numpy())
            results['targets'].append(targets.cpu().numpy())
            
            # Print progress
            if batch_idx % 10 == 0:
                print(f"Evaluated batch {batch_idx}/{len(data_loader)}")
    
    # Concatenate predictions and targets
    results['predictions'] = np.vstack(results['predictions'])
    results['targets'] = np.vstack(results['targets'])
    
    # Calculate financial metrics
    financial_metrics = calculate_financial_metrics(
        results['predictions'],
        results['targets'],
        return_all=True
    )
    results['financial_metrics'] = financial_metrics
    
    # Create directory for plots
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Create prediction plot
    plot_prediction_vs_actual(
        results['predictions'],
        results['targets'],
        timestamps=results['timestamps'] if results['timestamps'] else None,
        output_path=os.path.join(plots_dir, 'prediction_vs_actual.png')
    )
    
    # Save evaluation results
    save_evaluation_results(
        results,
        os.path.join(output_dir, 'evaluation_results.json')
    )
    
    return results
