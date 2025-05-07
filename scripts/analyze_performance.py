#!/usr/bin/env python
# scripts/model_comparison.py

import argparse
import json
import os
import numpy as np
import torch
import sys

# Add the parent directory to the path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Updated imports to match current project structure
from src.data.financial_loader import EnhancedFinancialDataLoader
from src.arch.baseline_lstm import FinancialLSTMBaseline as LSTMBaseline
from src.arch.cognitive import CognitiveArchitecture

def load_model(model_path, device="cpu"):
    """Load a cognitive model from checkpoint"""
    # Use the new from_checkpoint method for safer loading
    try:
        # First try using the new from_checkpoint method
        model = CognitiveArchitecture.from_checkpoint(model_path, device)
        return model
    except Exception as e:
        print(f"Warning: Failed to load model with from_checkpoint method: {e}")
        print("Falling back to legacy loading method...")
        
        # Legacy loading code as fallback
        checkpoint = torch.load(model_path, map_location=device)
        
        # Extract configuration based on actual dimensions in the checkpoint
        config = {}
        
        # Case 1: If 'config' exists in checkpoint, use it
        if isinstance(checkpoint, dict) and 'config' in checkpoint:
            config = checkpoint['config']
        
        # Case 2: If checkpoint is a state dict (most likely)
        else:
            # Try to determine dimensions directly from the weights
            if 'temporal_encoder.lstm.weight_ih_l0' in checkpoint:
                # Extract input dimension from the LSTM input weights
                input_dim = checkpoint['temporal_encoder.lstm.weight_ih_l0'].shape[1]
                config['input_dim'] = input_dim
                
                # Extract hidden dimension from LSTM weights (LSTM has 4 gates)
                hidden_dim = checkpoint['temporal_encoder.lstm.weight_ih_l0'].shape[0] // 4
                config['hidden_dim'] = hidden_dim
            
            # Extract memory size if available
            if 'financial_memory.memory' in checkpoint:
                memory_size = checkpoint['financial_memory.memory'].shape[0]
                config['memory_size'] = memory_size
            
            # Extract output dimension if available 
            if 'market_state_sequence_predictor.predictor.weight' in checkpoint:
                output_dim = checkpoint['market_state_sequence_predictor.predictor.weight'].shape[0]
                config['output_dim'] = output_dim
        
            # Default values if not found
            if 'input_dim' not in config:
                config['input_dim'] = 5  # Default
            if 'hidden_dim' not in config:
                config['hidden_dim'] = 64  # Default
            if 'memory_size' not in config:
                config['memory_size'] = 50  # Default
            if 'output_dim' not in config:
                config['output_dim'] = 4  # Default
            if 'seq_length' not in config:
                config['seq_length'] = 20  # Default
        
        print(f"Using model configuration: input_dim={config['input_dim']}, hidden_dim={config['hidden_dim']}, memory_size={config['memory_size']}")
        
        # Create model with extracted configuration
        model = CognitiveArchitecture(
            input_dim=config['input_dim'],
            hidden_dim=config['hidden_dim'],
            memory_size=config['memory_size'],
            output_dim=config.get('output_dim', 4),
            seq_length=config.get('seq_length', 20)
        )
        
        # Load state dict
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Handle case where checkpoint is just the state dict
            model.load_state_dict(checkpoint)
        
        return model

def calculate_metrics(predictions, actuals):
    """Calculate performance metrics for model predictions vs actual values"""
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(actuals, torch.Tensor):
        actuals = actuals.detach().cpu().numpy()
    
    # Handle dimension mismatches
    if predictions.shape != actuals.shape:
        print(f"Warning: Shape mismatch between predictions {predictions.shape} and actuals {actuals.shape}")
        
        # Get minimum dimension sizes
        min_batch = min(predictions.shape[0], actuals.shape[0])
        pred_features = predictions.shape[1] if len(predictions.shape) > 1 else 1
        actual_features = actuals.shape[1] if len(actuals.shape) > 1 else 1
        min_features = min(pred_features, actual_features)
        
        # Reshape if needed
        if len(predictions.shape) == 1:
            predictions = predictions.reshape(-1, 1)
        if len(actuals.shape) == 1:
            actuals = actuals.reshape(-1, 1)
            
        # Truncate to match dimensions
        predictions = predictions[:min_batch, :min_features]
        actuals = actuals[:min_batch, :min_features]
        
        print(f"Adjusted shapes - predictions: {predictions.shape}, actuals: {actuals.shape}")
    
    # Calculate metrics with error handling
    metrics = {}
    
    try:
        # Price accuracy (correlation for first feature)
        if predictions.shape[1] > 0 and actuals.shape[1] > 0:
            price_corr = np.corrcoef(predictions[:, 0], actuals[:, 0])
            if price_corr.shape == (2, 2):
                price_accuracy = price_corr[0, 1]
            else:
                # Handle case where corrcoef returns singularity
                price_accuracy = 0.0
        else:
            price_accuracy = 0.0
            
        metrics['price_accuracy'] = float(price_accuracy)
    except Exception as e:
        print(f"Error calculating price accuracy: {e}")
        metrics['price_accuracy'] = 0.0
    
    try:
        # Volume correlation (for second feature if available)
        if predictions.shape[1] > 1 and actuals.shape[1] > 1:
            vol_corr = np.corrcoef(predictions[:, 1], actuals[:, 1])
            if vol_corr.shape == (2, 2):
                volume_correlation = vol_corr[0, 1]
            else:
                volume_correlation = 0.0
        else:
            volume_correlation = 0.0
            
        metrics['volume_correlation'] = float(volume_correlation)
    except Exception as e:
        print(f"Error calculating volume correlation: {e}")
        metrics['volume_correlation'] = 0.0
    
    try:
        # Returns error (for third feature if available)
        if predictions.shape[1] > 2 and actuals.shape[1] > 2:
            returns_error = np.mean(np.abs(predictions[:, 2] - actuals[:, 2]))
            returns_stability = 1 / (1 + returns_error) if returns_error > 0 else 0
        else:
            returns_stability = 0.0
            
        metrics['returns_stability'] = float(returns_stability)
    except Exception as e:
        print(f"Error calculating returns stability: {e}")
        metrics['returns_stability'] = 0.0
    
    try:
        # Volatility accuracy (for fourth feature if available)
        if predictions.shape[1] > 3 and actuals.shape[1] > 3:
            vol_acc_corr = np.corrcoef(predictions[:, 3], actuals[:, 3])
            if vol_acc_corr.shape == (2, 2):
                volatility_accuracy = vol_acc_corr[0, 1]
            else:
                volatility_accuracy = 0.0
        else:
            volatility_accuracy = 0.0
            
        metrics['volatility_prediction'] = float(volatility_accuracy)
    except Exception as e:
        print(f"Error calculating volatility prediction: {e}")
        metrics['volatility_prediction'] = 0.0
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Compare models")
    parser.add_argument("--cognitive", required=True, help="Cognitive model path")
    parser.add_argument("--baseline", required=True, help="Baseline model path")
    parser.add_argument("--test_data", required=True, help="Test data path")
    parser.add_argument("--seq_length", type=int, default=20, help="Sequence length")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--output", default="validation/results", help="Output directory")
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load models
    print(f"Loading cognitive model from {args.cognitive}")
    try:
        cognitive_model = load_model(args.cognitive)
        cognitive_model.to(device)
    except Exception as e:
        print(f"Error loading cognitive model: {e}")
        sys.exit(1)
    
    print(f"Loading baseline model from {args.baseline}")
    # Check if baseline model file exists
    if not os.path.exists(args.baseline):
        print(f"Error: Baseline model file '{args.baseline}' does not exist.")
        print(f"Please ensure the correct path is provided or train a baseline model first.")
        sys.exit(2)
        
    try:
        baseline_model = LSTMBaseline(input_dim=7, hidden_dim=64, num_layers=2, output_dim=7)
        baseline_state = torch.load(args.baseline, map_location=device)
        baseline_model.load_state_dict(baseline_state)
        baseline_model.to(device)
    except Exception as e:
        print(f"Error loading baseline model: {e}")
        sys.exit(1)
    
    # Load test data
    print(f"Loading test data from {args.test_data}")
    try:
        test_loader = EnhancedFinancialDataLoader(
            data_path=args.test_data,
            sequence_length=args.seq_length,
            batch_size=args.batch
        )
    except Exception as e:
        print(f"Error loading test data: {e}")
        sys.exit(1)
    
    # Evaluate models
    cognitive_model.eval()
    baseline_model.eval()
    
    # Process baseline model
    baseline_preds = []
    all_targets = []
    
    # Handle the case where test_loader might use a different attribute for data iteration
    test_loader_obj = test_loader.test_loader if hasattr(test_loader, 'test_loader') else test_loader
    
    with torch.no_grad():
        for batch in test_loader_obj:
            # Handle different data formats - could be a tuple or a dict
            if isinstance(batch, tuple) and len(batch) == 3:
                features, sequence, targets = batch
            elif isinstance(batch, dict):
                features = batch.get('features', None)
                sequence = batch.get('sequence', None)
                targets = batch.get('target', None)
            else:
                print(f"Unknown batch format: {type(batch)}")
                continue
                
            features = features.to(device)
            sequence = sequence.to(device)
            targets = targets.to(device)
            
            outputs = baseline_model(sequence)
            baseline_preds.append(outputs.cpu())
            all_targets.append(targets.cpu())
    
    baseline_preds = torch.cat(baseline_preds, dim=0).numpy()
    
    # Reset loader and process cognitive model
    if hasattr(test_loader, 'pointer'):
        test_loader.pointer = 0
    
    cognitive_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in test_loader_obj:
            # Handle different data formats - could be a tuple or a dict
            if isinstance(batch, tuple) and len(batch) == 3:
                features, sequence, targets = batch
            elif isinstance(batch, dict):
                features = batch.get('features', None)
                sequence = batch.get('sequence', None)
                targets = batch.get('target', None)
            else:
                print(f"Unknown batch format: {type(batch)}")
                continue
                
            features = features.to(device)
            sequence = sequence.to(device)
            targets = targets.to(device)
            
            # Handle different model interfaces
            try:
                outputs = cognitive_model(financial_data=features, financial_seq=sequence)
                # Check if outputs is a dict, extract the relevant prediction
                if isinstance(outputs, dict):
                    if 'prediction' in outputs:
                        outputs = outputs['prediction']
                    elif 'market_state' in outputs:
                        outputs = outputs['market_state']
                        
                # Convert 3D outputs [batch, seq_len, features] to 2D [batch, features]
                if len(outputs.shape) == 3:
                    # Use only the last timestep prediction
                    outputs = outputs[:, -1, :]
            except TypeError:
                try:
                    # Fallback to basic interface
                    outputs = cognitive_model(sequence)
                    
                    # Convert 3D outputs if needed
                    if len(outputs.shape) == 3:
                        outputs = outputs[:, -1, :]
                except Exception as e:
                    print(f"Error during cognitive model inference: {e}")
                    continue
                    
            cognitive_preds.append(outputs.cpu())
            all_targets.append(targets.cpu())
    
    cognitive_preds = torch.cat(cognitive_preds, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()
    
    # Calculate metrics
    baseline_metrics = calculate_metrics(baseline_preds, all_targets)
    cognitive_metrics = calculate_metrics(cognitive_preds, all_targets)
    
    # Calculate improvements
    improvements = {}
    for metric in baseline_metrics:
        baseline_val = baseline_metrics[metric]
        cognitive_val = cognitive_metrics[metric]
        
        # Handle division by zero
        if baseline_val != 0:
            improvement = ((cognitive_val - baseline_val) / abs(baseline_val)) * 100
        else:
            improvement = 0 if cognitive_val == 0 else float('inf')
        
        improvements[metric] = float(improvement)
    
    # Print results
    print("\nBaseline Metrics:")
    for k, v in baseline_metrics.items():
        print(f"  {k}: {v:.4f}")
    
    print("\nCognitive Metrics:")
    for k, v in cognitive_metrics.items():
        print(f"  {k}: {v:.4f}")
    
    print("\nImprovements (%):")
    for k, v in improvements.items():
        print(f"  {k}: {v:.2f}%")
    
    # Save results
    os.makedirs(args.output, exist_ok=True)
    
    results = {
        'baseline_metrics': baseline_metrics,
        'cognitive_metrics': cognitive_metrics,
        'improvements': improvements
    }
    
    with open(os.path.join(args.output, 'comparison.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    with open(os.path.join(args.output, 'baseline_stats.json'), 'w') as f:
        json.dump(baseline_metrics, f, indent=4)
    
    with open(os.path.join(args.output, 'cognitive_stats.json'), 'w') as f:
        json.dump(cognitive_metrics, f, indent=4)
    
    print(f"\nResults saved to {args.output}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
