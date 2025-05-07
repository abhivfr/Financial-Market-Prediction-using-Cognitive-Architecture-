#!/usr/bin/env python
# model_compare_fixed.py - A fixed version of compare_models_simple

import os
import sys
import argparse
import torch
import json
import numpy as np

# Fix import paths
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Import the correct modules
from train import FinancialDataLoader
from src.arch.baseline_lstm import FinancialLSTMBaseline
from src.arch.cognitive import CognitiveArchitecture

def main():
    parser = argparse.ArgumentParser(description="Compare baseline and cognitive models")
    parser.add_argument("--cognitive", required=True, help="Path to cognitive model")
    parser.add_argument("--baseline", required=True, help="Path to baseline model")
    parser.add_argument("--test_data", required=True, help="Path to test data")
    parser.add_argument("--seq_length", type=int, default=20, help="Sequence length")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--output", default="validation/results", help="Output directory")
    
    args = parser.parse_args()
    
    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load models
    print(f"Loading cognitive model: {args.cognitive}")
    cognitive_model = CognitiveArchitecture()
    cognitive_model.load_state_dict(torch.load(args.cognitive, map_location=device), strict=False)
    cognitive_model.to(device)
    cognitive_model.eval()
    print("Cognitive model loaded successfully")
    
    print(f"Loading baseline model: {args.baseline}")
    baseline_model = FinancialLSTMBaseline()
    baseline_model.load_state_dict(torch.load(args.baseline, map_location=device))
    baseline_model.to(device)
    baseline_model.eval()
    print("Baseline model loaded successfully")
    
    # Load test data
    print(f"Loading test data: {args.test_data}")
    test_loader = FinancialDataLoader(
        path=args.test_data,
        seq_length=args.seq_length,
        batch_size=args.batch
    )
    
    # Process baseline model predictions
    baseline_preds = []
    all_targets = []
    
    # Use the iterator pattern for test_loader
    test_iterator = iter(test_loader)
    while True:
        try:
            batch = next(test_iterator)
            features = batch['features'].to(device)
            sequence = batch['sequence'].to(device)
            targets = batch['target'].to(device)
            
            outputs = baseline_model(sequence)
            baseline_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
        except StopIteration:
            break
    
    baseline_preds = np.concatenate(baseline_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    
    # Reset data loader
    test_iterator = iter(test_loader)
    
    # Process cognitive model predictions
    cognitive_preds = []
    
    while True:
        try:
            batch = next(test_iterator)
            features = batch['features'].to(device)
            sequence = batch['sequence'].to(device)
            
            outputs = cognitive_model(financial_data=features, financial_seq=sequence)
            # Extract the market_state from the output dictionary
            market_state = outputs['market_state']
            cognitive_preds.append(market_state.cpu().numpy())
        except StopIteration:
            break
    
    cognitive_preds = np.concatenate(cognitive_preds, axis=0)
    
    # Ensure predictions and targets have the same shape
    min_len = min(baseline_preds.shape[0], cognitive_preds.shape[0], targets.shape[0])
    baseline_preds = baseline_preds[:min_len]
    cognitive_preds = cognitive_preds[:min_len]
    targets = targets[:min_len]
    
    # Calculate metrics
    def calculate_metrics(preds, targets):
        price_acc = np.corrcoef(preds[:, 0], targets[:, 0])[0, 1]
        vol_corr = np.corrcoef(preds[:, 1], targets[:, 1])[0, 1]
        returns_err = np.mean(np.abs(preds[:, 2] - targets[:, 2]))
        returns_stability = 1 / (1 + returns_err)
        vol_pred = np.corrcoef(preds[:, 3], targets[:, 3])[0, 1]
        
        return {
            'price_accuracy': float(price_acc),
            'volume_correlation': float(vol_corr),
            'returns_stability': float(returns_stability),
            'volatility_prediction': float(vol_pred)
        }
    
    baseline_metrics = calculate_metrics(baseline_preds, targets)
    cognitive_metrics = calculate_metrics(cognitive_preds, targets)
    
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
    
    # Print results
    print("\nBaseline model metrics:")
    for k, v in baseline_metrics.items():
        print(f"  {k}: {v:.4f}")
    
    print("\nCognitive model metrics:")
    for k, v in cognitive_metrics.items():
        print(f"  {k}: {v:.4f}")
    
    print("\nImprovements:")
    for k, v in improvements.items():
        print(f"  {k}: {v:+.2f}%")
    
    # Save results
    results = {
        'baseline_metrics': baseline_metrics,
        'cognitive_metrics': cognitive_metrics,
        'improvements': improvements
    }
    
    os.makedirs(args.output, exist_ok=True)
    
    with open(os.path.join(args.output, 'comparison.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    with open(os.path.join(args.output, 'baseline_stats.json'), 'w') as f:
        json.dump(baseline_metrics, f, indent=4)
    
    with open(os.path.join(args.output, 'cognitive_stats.json'), 'w') as f:
        json.dump(cognitive_metrics, f, indent=4)
    
    print(f"\nResults saved to {args.output}")

if __name__ == "__main__":
    main()
