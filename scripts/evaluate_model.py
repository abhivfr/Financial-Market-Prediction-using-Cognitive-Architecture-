#!/usr/bin/env python
# evaluate_model.py - Evaluate a trained model on test data

import sys
import os
import torch
import torch.nn as nn
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd

# Ensure src is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.arch.baseline_lstm import FinancialLSTMBaseline
from src.data.financial_loader import EnhancedFinancialDataLoader

def evaluate_model(model_path, test_data_path, seq_length=20, batch_size=32, output_dir="reports/baseline_eval"):
    """
    Evaluate a trained model on test data
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Running evaluation on {device}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load test data
    test_loader = EnhancedFinancialDataLoader(
        data_path=test_data_path,
        sequence_length=seq_length,
        batch_size=batch_size
    )
    
    # Determine input/output dimensions
    try:
        sample_batch = next(iter(test_loader))
        input_dim = sample_batch['sequence'].shape[2]
        output_dim = sample_batch['target'].shape[1]
        print(f"✅ Automatically determined input_dim={input_dim}, output_dim={output_dim} from data")
    except:
        # Fallback to default dimensions if we can't get a batch
        input_dim = 7
        output_dim = 7
        print(f"⚠️ Using default input_dim={input_dim}, output_dim={output_dim}")
    
    # Initialize model
    model = FinancialLSTMBaseline(
        input_dim=input_dim,
        hidden_dim=64,
        num_layers=2,
        output_dim=output_dim,
        dropout=0.2
    ).to(device)
    
    # Load model weights
    print(f"Loading model from {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Initialize metrics
    all_losses = []
    all_preds = []
    all_targets = []
    all_features = []
    
    # Evaluation loop
    print("Starting evaluation...")
    with torch.no_grad():
        for i in range(min(50, len(test_loader))):  # Process up to 50 batches
            try:
                batch = next(iter(test_loader))
                features = batch['features'].to(device)
                sequence = batch['sequence'].to(device)
                target = batch['target'].to(device)
                
                # Skip batches with NaN/Inf values
                if torch.isnan(sequence).any() or torch.isinf(sequence).any() or \
                   torch.isnan(target).any() or torch.isinf(target).any():
                    continue
                
                # Forward pass
                predictions = model(sequence)
                
                # Make sure shapes match
                if predictions.shape != target.shape:
                    if predictions.shape[0] == target.shape[0]:
                        min_features = min(predictions.shape[1], target.shape[1])
                        predictions = predictions[:, :min_features]
                        target = target[:, :min_features]
                    else:
                        continue
                
                # Calculate loss
                loss = nn.MSELoss()(predictions, target)
                
                # Skip if loss is NaN or infinite
                if torch.isnan(loss) or torch.isinf(loss):
                    continue
                
                all_losses.append(loss.item())
                all_preds.append(predictions.cpu().numpy())
                all_targets.append(target.cpu().numpy())
                all_features.append(features.cpu().numpy())
                
                print(f"Batch {i+1}: Loss = {loss.item():.4f}")
            
            except StopIteration:
                break
    
    # Calculate overall metrics
    if all_preds:
        all_preds = np.vstack(all_preds)
        all_targets = np.vstack(all_targets)
        
        mse = np.mean((all_preds - all_targets) ** 2, axis=0)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(all_preds - all_targets), axis=0)
        
        print("\n=== Model Evaluation Results ===")
        print(f"Average Loss: {np.mean(all_losses):.4f}")
        print(f"MSE per feature: {mse}")
        print(f"RMSE per feature: {rmse}")
        print(f"MAE per feature: {mae}")
        
        # Calculate correlations for each feature
        correlations = []
        for i in range(all_preds.shape[1]):
            p = all_preds[:, i]
            t = all_targets[:, i]
            if len(p) > 1 and not np.isnan(p).any() and not np.isnan(t).any():
                corr = np.corrcoef(p, t)[0, 1]
                corr_val = corr if not np.isnan(corr) else 0.0
            else:
                corr_val = 0.0
            correlations.append(corr_val)
        
        print(f"Correlations per feature: {correlations}")
        
        # Save results to CSV
        results = {
            "Feature": [f"Feature_{i+1}" for i in range(len(mse))],
            "MSE": mse,
            "RMSE": rmse,
            "MAE": mae,
            "Correlation": correlations
        }
        
        results_df = pd.DataFrame(results)
        results_path = os.path.join(output_dir, "evaluation_results.csv")
        results_df.to_csv(results_path, index=False)
        print(f"Results saved to {results_path}")
        
        # Create visualizations
        plt.figure(figsize=(12, 8))
        
        # Plot feature-wise metrics
        plt.subplot(2, 2, 1)
        plt.bar([f"F{i+1}" for i in range(len(mse))], mse)
        plt.title("MSE per Feature")
        plt.ylabel("Mean Squared Error")
        
        plt.subplot(2, 2, 2)
        plt.bar([f"F{i+1}" for i in range(len(correlations))], correlations)
        plt.title("Correlation per Feature")
        plt.ylabel("Correlation Coefficient")
        
        # Plot loss distribution
        plt.subplot(2, 2, 3)
        plt.hist(all_losses, bins=20)
        plt.title("Loss Distribution")
        plt.xlabel("Loss Value")
        plt.ylabel("Frequency")
        
        # Plot predictions vs targets for first feature
        plt.subplot(2, 2, 4)
        plt.scatter(all_targets[:, 0], all_preds[:, 0], alpha=0.5)
        plt.title(f"Predictions vs Targets (Feature 1)")
        plt.xlabel("Target")
        plt.ylabel("Prediction")
        
        # Add line of perfect correlation
        min_val = min(all_targets[:, 0].min(), all_preds[:, 0].min())
        max_val = max(all_targets[:, 0].max(), all_preds[:, 0].max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        plt.tight_layout()
        
        # Save visualization
        viz_path = os.path.join(output_dir, "evaluation_visualizations.png")
        plt.savefig(viz_path)
        print(f"Visualizations saved to {viz_path}")
        
        return np.mean(all_losses), correlations
    else:
        print("❌ No valid evaluation batches found")
        return None, None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained model")
    parser.add_argument("--model", type=str, required=True, help="Path to model file")
    parser.add_argument("--data", type=str, required=True, help="Path to test data")
    parser.add_argument("--seq_length", type=int, default=20, help="Sequence length")
    parser.add_argument("--batch", type=int, default=32, help="Batch size")
    parser.add_argument("--output_dir", type=str, default="reports/baseline_eval", help="Output directory")
    
    args = parser.parse_args()
    
    evaluate_model(
        model_path=args.model,
        test_data_path=args.data,
        seq_length=args.seq_length,
        batch_size=args.batch,
        output_dir=args.output_dir
    ) 