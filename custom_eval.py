#!/usr/bin/env python
import torch
import pandas as pd
import numpy as np
import argparse
import json
import os
from src.arch.cognitive import CognitiveArchitecture
from torch.utils.data import DataLoader, TensorDataset

def custom_evaluate(model_path, data_path, seq_length=20, batch_size=8, output_path=None):
    """Evaluate model with robust error handling"""
    print(f"Evaluating model {model_path} on {data_path}")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    try:
        model = CognitiveArchitecture()
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        print(f"Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return {}
    
    # Load data
    try:
        df = pd.read_csv(data_path)
        print(f"Data loaded with shape {df.shape}")
        
        required_cols = ['price', 'volume', 'returns', 'volatility']
        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            print(f"Error: Missing columns in data: {missing}")
            return {}
            
        # Select only required columns
        df = df[required_cols]
    except Exception as e:
        print(f"Error loading data: {e}")
        return {}
    
    # Create sequences
    try:
        sequences = []
        targets = []
        
        for i in range(len(df) - seq_length * 2):
            seq = df.iloc[i:i+seq_length].values
            target = df.iloc[i+seq_length:i+seq_length*2].values
            sequences.append(seq)
            targets.append(target)
        
        print(f"Created {len(sequences)} evaluation sequences")
        
        # Convert to tensors
        seq_tensor = torch.tensor(np.array(sequences), dtype=torch.float32)
        target_tensor = torch.tensor(np.array(targets), dtype=torch.float32)
        feat_tensor = torch.tensor(np.array([seq[-1] for seq in sequences]), dtype=torch.float32)
        
        # Create dataset and loader
        dataset = TensorDataset(feat_tensor, seq_tensor, target_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    except Exception as e:
        print(f"Error preparing sequences: {e}")
        return {}
    
    # Run evaluation
    metrics = {}
    try:
        with torch.no_grad():
            all_preds = []
            all_targets = []
            
            # Import ModelWrapper directly from file
            sys.path.append(os.getcwd())
            from model_wrapper import ModelWrapper
            wrapped_model = ModelWrapper(model)
            
            for feats, seqs, tgts in loader:
                feats = feats.to(device)
                seqs = seqs.to(device) 
                tgts = tgts.to(device)
                
                # Forward pass through wrapped model
                predictions = wrapped_model(feats, seqs)
                
                # Store predictions and targets
                all_preds.append(predictions.cpu())
                all_targets.append(tgts.cpu())
            
            # Concatenate results
            all_preds = torch.cat(all_preds, dim=0)
            all_targets = torch.cat(all_targets, dim=0)
            
            # Calculate metrics
            metrics = calculate_metrics(all_preds, all_targets)
            
            print("Metrics calculated successfully:")
            for k, v in metrics.items():
                print(f"  {k}: {v:.4f}")
            
            # Save metrics if output path provided
            if output_path:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with open(output_path, 'w') as f:
                    json.dump(metrics, f, indent=2)
                print(f"Metrics saved to {output_path}")
                
    except Exception as e:
        print(f"Error during evaluation: {e}")
        return {}
        
    return metrics

def calculate_metrics(pred, target):
    """Calculate standard metrics used in project"""
    # Use last step predictions
    p = pred[:, -1, :].numpy()
    t = target[:, -1, :].numpy()
    
    # Calculate price accuracy (correlation for price)
    price_accuracy = np.corrcoef(p[:, 0], t[:, 0])[0, 1] if p.shape[0] > 1 else 0
    price_accuracy = 0 if np.isnan(price_accuracy) else price_accuracy
    
    # Calculate volume correlation
    volume_corr = np.corrcoef(p[:, 1], t[:, 1])[0, 1] if p.shape[0] > 1 else 0
    volume_corr = 0 if np.isnan(volume_corr) else volume_corr
    
    # Calculate returns stability (1 - MAE for returns)
    returns_mae = np.mean(np.abs(p[:, 2] - t[:, 2]))
    returns_stability = 1.0 - min(returns_mae, 1.0)
    
    # Calculate volatility prediction (correlation)
    volatility_pred = np.corrcoef(p[:, 3], t[:, 3])[0, 1] if p.shape[0] > 1 else 0
    volatility_pred = 0 if np.isnan(volatility_pred) else volatility_pred
    
    return {
        'price_accuracy': float(price_accuracy),
        'volume_correlation': float(volume_corr),
        'returns_stability': float(returns_stability),
        'volatility_prediction': float(volatility_pred)
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Custom model evaluation")
    parser.add_argument("--model", required=True, help="Path to model file")
    parser.add_argument("--data", required=True, help="Path to data file")
    parser.add_argument("--seq_length", type=int, default=20, help="Sequence length")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--output", help="Output path for metrics JSON")
    args = parser.parse_args()
    
    custom_evaluate(args.model, args.data, args.seq_length, args.batch_size, args.output)
