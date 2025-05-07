#!/usr/bin/env python
# eval_compare.py - Compare models using eval_plus functionality

import os
import sys
import torch
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import argparse  # Add this explicitly

# Prevent argparse from being imported from other modules
# by modifying sys.argv temporarily during imports
original_argv = sys.argv
sys.argv = [sys.argv[0]]  # Keep only the script name during imports

# Import from eval_plus avoiding argparse conflicts
from eval_plus import run_eval, run_full_evaluation, compute_metrics

# Import models and loaders
from src.arch.cognitive import CognitiveArchitecture
from src.arch.baseline_lstm import FinancialLSTMBaseline
from train import FinancialDataLoader

# Restore original command line arguments
sys.argv = original_argv

def load_cognitive_model(model_path):
    """Load the cognitive architecture model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CognitiveArchitecture()
    
    if model_path and os.path.exists(model_path):
        print(f"Loading cognitive model from {model_path}")
        try:
            model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
            print("Cognitive model loaded successfully")
        except Exception as e:
            print(f"Error loading cognitive model: {e}")
            sys.exit(1)
    
    return model.to(device)

def load_baseline_model(model_path):
    """Load the baseline LSTM model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FinancialLSTMBaseline()
    
    if model_path and os.path.exists(model_path):
        print(f"Loading baseline model from {model_path}")
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            print("Baseline model loaded successfully")
        except Exception as e:
            print(f"Error loading baseline model: {e}")
            sys.exit(1)
    
    return model.to(device)

def create_comparison_chart(baseline_metrics, cognitive_metrics, output_dir):
    """Create a comparison chart for the metrics"""
    # Find common metrics between the two models
    common_metrics = ['price_accuracy', 'volume_correlation', 'returns_stability', 'volatility_prediction']
    
    # Map from eval_plus metrics to our standard metric names
    metric_mapping = {
        'corrs': 'price_accuracy',  # Using correlation for price accuracy
        'mae': 'returns_stability', # Lower MAE = higher stability
        'rmse': 'volatility_prediction' # Using RMSE for volatility prediction
    }
    
    # Extract and prepare metrics
    baseline_values = []
    cognitive_values = []
    metric_labels = []
    
    for metric in common_metrics:
        if metric in baseline_metrics and metric in cognitive_metrics:
            baseline_values.append(baseline_metrics[metric])
            cognitive_values.append(cognitive_metrics[metric])
            metric_labels.append(metric.replace('_', ' ').title())
    
    # Create the chart
    fig, ax = plt.subplots(figsize=(12, 7))
    x = np.arange(len(metric_labels))
    width = 0.35
    
    baseline_bars = ax.bar(x - width/2, baseline_values, width, label='Baseline', color='skyblue')
    cognitive_bars = ax.bar(x + width/2, cognitive_values, width, label='Cognitive', color='salmon')
    
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.legend()
    
    # Add value labels on top of bars
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    add_labels(baseline_bars)
    add_labels(cognitive_bars)
    
    plt.tight_layout()
    os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'plots', 'model_comparison.png'))
    plt.close()

def convert_eval_to_comparison_metrics(eval_metrics):
    """Convert eval_plus metrics to comparison format"""
    # Extract relevant metrics
    price_accuracy = eval_metrics.get('corrs', [0])[0]  # First element is price
    volume_correlation = eval_metrics.get('corrs', [0, 0])[1] if len(eval_metrics.get('corrs', [])) > 1 else 0 # Second element is volume
    
    # For returns stability, we invert MAE (lower is better, but we want higher = better)
    returns_error = eval_metrics.get('mae', [0, 0, 0])[2] if len(eval_metrics.get('mae', [])) > 2 else 1
    returns_stability = 1 / (1 + returns_error) if returns_error > 0 else 0
    
    # Volatility prediction
    volatility_prediction = eval_metrics.get('corrs', [0, 0, 0, 0])[3] if len(eval_metrics.get('corrs', [])) > 3 else 0

    return {
        'price_accuracy': float(price_accuracy),
        'volume_correlation': float(volume_correlation),
        'returns_stability': float(returns_stability),
        'volatility_prediction': float(volatility_prediction)
    }

def compute_metrics(pred, target):
    """
    Computes evaluation metrics: correlations, MAE, and RMSE.
    """
    # Flatten the last step of predictions and targets for metric computation
    p = pred[:, -1, :].view(-1, pred.size(-1)).cpu().detach().numpy()
    t = target[:, -1, :].view(-1, target.size(-1)).cpu().numpy()

    corrs = []
    for i in range(p.shape[1]):
        # Ensure there's enough data points and variance to compute correlation
        if p.shape[0] > 1 and np.std(p[:, i]) > 1e-8 and np.std(t[:, i]) > 1e-8:
            cor = np.corrcoef(p[:, i], t[:, i])[0, 1]
            corrs.append(cor if not np.isnan(cor) else 0.0)
        else:
            corrs.append(0.0)

    mae = np.mean(np.abs(p - t), axis=0)
    rmse = np.sqrt(np.mean((p - t)**2, axis=0))

    return corrs, mae, rmse

def main():
    # Define our own argument parser
    parser = argparse.ArgumentParser(description="Compare models using eval_plus functionality")
    parser.add_argument("--cognitive", required=True, help="Path to cognitive model")
    parser.add_argument("--baseline", required=True, help="Path to baseline model")
    parser.add_argument("--test_data", required=True, help="Path to test data")
    parser.add_argument("--seq_length", type=int, default=20, help="Sequence length")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--output", default="validation/results", help="Output directory")
    parser.add_argument("--full", action="store_true", help="Run full evaluation including backtest")
    
    args = parser.parse_args()
    
    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load models
    cognitive_model = load_cognitive_model(args.cognitive)
    baseline_model = load_baseline_model(args.baseline)
    
    # Create DataLoader
    print(f"Loading test data: {args.test_data}")
    data_loader = FinancialDataLoader(
        path=args.test_data,
        seq_length=args.seq_length,
        batch_size=args.batch
    )
    
    # Create output directories
    os.makedirs(args.output, exist_ok=True)
    cognitive_output = os.path.join(args.output, "cognitive")
    baseline_output = os.path.join(args.output, "baseline")
    os.makedirs(cognitive_output, exist_ok=True)
    os.makedirs(baseline_output, exist_ok=True)
    
    # Evaluate both models
    print("\n=== Evaluating Cognitive Model ===")
    cognitive_eval_metrics = {}
    
    # Create a wrapper to convert the batch format
    def run_eval_wrapper(model, data_loader, device, skip_memory=False):
        all_pred = []
        all_tgt = []
        
        model.eval()
        with torch.no_grad():
            for batch in data_loader:
                # FinancialDataLoader returns a dict with 'features', 'sequence', 'target'
                features = batch['features'].to(device)
                sequence = batch['sequence'].to(device)
                targets = batch['target'].to(device)
                
                # Get model outputs based on whether it's baseline or cognitive
                if isinstance(model, FinancialLSTMBaseline):
                    outputs = model(sequence)
                else:
                    # Cognitive model takes both features and sequence
                    outputs = model(financial_data=features, financial_seq=sequence)
                    # Extract market_state_sequence if it's a dictionary return
                    if isinstance(outputs, dict) and 'market_state_sequence' in outputs:
                        outputs = outputs['market_state_sequence']
                
                # Reshape to match expected format for metric computation
                outputs = outputs.unsqueeze(1) if outputs.dim() == 2 else outputs
                targets = targets.unsqueeze(1) if targets.dim() == 2 else targets
                
                all_pred.append(outputs.cpu())
                all_tgt.append(targets.cpu())
        
        # Concatenate all predictions and targets
        if all_pred and all_tgt:
            all_pred = torch.cat(all_pred, dim=0)
            all_tgt = torch.cat(all_tgt, dim=0)
            
            # Calculate metrics directly
            corrs, mae, rmse = compute_metrics(all_pred, all_tgt)
            
            return {
                'corrs': corrs,
                'mae': mae,
                'rmse': rmse,
                'conflict_rate': 0.0,  # Placeholder
                'avg_confidence': 0.0  # Placeholder
            }
        else:
            return {}
    
    # Run evaluation on cognitive model
    if args.full:
        # We need to implement a simpler version for full evaluation
        pass
    else:
        cognitive_eval_metrics = run_eval_wrapper(cognitive_model, data_loader, device)
    
    print("\n=== Evaluating Baseline Model ===")
    # Reset data loader
    data_loader.pointer = 0
    
    # Run evaluation on baseline model
    baseline_eval_metrics = run_eval_wrapper(baseline_model, data_loader, device)
    
    # Convert eval metrics to comparison format
    cognitive_metrics = convert_eval_to_comparison_metrics(cognitive_eval_metrics)
    baseline_metrics = convert_eval_to_comparison_metrics(baseline_eval_metrics)
    
    # Print evaluation results
    print("\n=== Evaluation Results ===")
    print("\nBaseline model metrics:")
    for k, v in baseline_metrics.items():
        print(f"  {k}: {v:.4f}")
    
    print("\nCognitive model metrics:")
    for k, v in cognitive_metrics.items():
        print(f"  {k}: {v:.4f}")
    
    # Calculate improvements
    improvements = {}
    for metric in baseline_metrics:
        base_val = baseline_metrics[metric]
        cog_val = cognitive_metrics[metric]
        
        if base_val != 0:
            imp = ((cog_val - base_val) / abs(base_val)) * 100
        else:
            imp = 0 if cog_val == 0 else float('inf')
        
        improvements[metric] = float(imp)
    
    print("\nImprovements:")
    for k, v in improvements.items():
        print(f"  {k}: {v:+.2f}%")
    
    # Save results
    results = {
        'baseline_metrics': baseline_metrics,
        'cognitive_metrics': cognitive_metrics,
        'improvements': improvements,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(os.path.join(args.output, 'comparison.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    with open(os.path.join(args.output, 'baseline_stats.json'), 'w') as f:
        json.dump(baseline_metrics, f, indent=4)
    
    with open(os.path.join(args.output, 'cognitive_stats.json'), 'w') as f:
        json.dump(cognitive_metrics, f, indent=4)
    
    # Create visualization
    create_comparison_chart(baseline_metrics, cognitive_metrics, args.output)
    
    print(f"\nResults saved to {args.output}")
    print(f"Visualization saved to {os.path.join(args.output, 'plots', 'model_comparison.png')}")

if __name__ == "__main__":
    main()
