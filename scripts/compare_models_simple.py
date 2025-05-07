#!/usr/bin/env python
# Create a new file: scripts/compare_models_simple.py

import sys
import os
import argparse
import torch
import json
import numpy as np

# Fix import paths for nested directory structure
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.dirname(parent_dir))
sys.path.insert(0, os.path.dirname(os.path.dirname(parent_dir)))

# Updated imports to use the correct paths
from train import FinancialDataLoader
from src.arch.baseline_lstm import FinancialLSTMBaseline
from src.arch.cognitive import CognitiveArchitecture

def load_model(model_path):
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

def calculate_metrics(preds, targets, prev_preds=None, prev_targets=None):
    """
    Enhanced metrics calculation with directional accuracy and technical indicators
    Args:
        preds: Model predictions
        targets: Ground truth values
        prev_preds: Previous predictions (for directional metrics)
        prev_targets: Previous targets (for directional metrics)
    Returns:
        Dictionary of performance metrics
    """
    # Basic correlation metrics with improved robustness
    metrics = {}
    
    # Price accuracy - use correlation coefficient with error handling
    try:
        price_corr = np.corrcoef(preds[:, 0], targets[:, 0])[0, 1]
        if np.isnan(price_corr):
            price_corr = 0.0
    except (ValueError, IndexError):
        price_corr = 0.0
    metrics['price_accuracy'] = float(price_corr)
    
    # Volume correlation with robustness
    try:
        vol_corr = np.corrcoef(preds[:, 1], targets[:, 1])[0, 1]
        if np.isnan(vol_corr):
            vol_corr = 0.0
    except (ValueError, IndexError):
        vol_corr = 0.0
    metrics['volume_correlation'] = float(vol_corr)
    
    # Returns stability with outlier handling
    returns_err = np.mean(np.clip(np.abs(preds[:, 2] - targets[:, 2]), 0, 5))
    metrics['returns_stability'] = float(1 / (1 + returns_err))
    
    # Volatility prediction with correlation
    try:
        vol_pred_corr = np.corrcoef(preds[:, 3], targets[:, 3])[0, 1]
        if np.isnan(vol_pred_corr):
            vol_pred_corr = 0.0
    except (ValueError, IndexError):
        vol_pred_corr = 0.0
    metrics['volatility_prediction'] = float(vol_pred_corr)
    
    # Add directional accuracy if previous data available
    if prev_preds is not None and prev_targets is not None:
        try:
            # Price direction
            pred_dir = np.sign(preds[:, 0] - prev_preds[:, 0])
            true_dir = np.sign(targets[:, 0] - prev_targets[:, 0])
            
            # Handle zero changes (neutral)
            pred_dir = np.where(pred_dir == 0, 0.01, pred_dir)
            true_dir = np.where(true_dir == 0, 0.01, true_dir)
            
            # Direction match ratio
            dir_accuracy = np.mean(pred_dir == true_dir)
            metrics['directional_accuracy'] = float(dir_accuracy)
            
            # Trend alignment score
            trend_match = np.mean(pred_dir * true_dir > 0)
            metrics['trend_alignment'] = float(trend_match)
        except (ValueError, IndexError):
            metrics['directional_accuracy'] = 0.5
            metrics['trend_alignment'] = 0.5
    
    # Add peak detection metrics
    try:
        from scipy.signal import find_peaks
        # Find peaks in price series
        true_peaks, _ = find_peaks(targets[:, 0])
        pred_peaks, _ = find_peaks(preds[:, 0])
        
        # Calculate peak match score (within 2 steps tolerance)
        peak_match_score = 0.0
        if len(true_peaks) > 0 and len(pred_peaks) > 0:
            tolerance = 2
            matches = 0
            
            for tp in true_peaks:
                if any(abs(tp - pp) <= tolerance for pp in pred_peaks):
                    matches += 1
                    
            peak_match_score = matches / max(len(true_peaks), len(pred_peaks))
        
        metrics['peak_detection'] = float(peak_match_score)
    except (ImportError, ValueError, IndexError):
        metrics['peak_detection'] = 0.0
    
    # Add combined score (weighted average of all metrics)
    weights = {
        'price_accuracy': 0.3,
        'directional_accuracy': 0.2,
        'volume_correlation': 0.15,
        'returns_stability': 0.15,
        'volatility_prediction': 0.1,
        'trend_alignment': 0.05,
        'peak_detection': 0.05
    }
    
    # Calculate weighted score for available metrics
    total_weight = 0.0
    weighted_sum = 0.0
    
    for metric, weight in weights.items():
        if metric in metrics:
            weighted_sum += metrics[metric] * weight
            total_weight += weight
    
    if total_weight > 0:
        metrics['combined_score'] = float(weighted_sum / total_weight)
    else:
        metrics['combined_score'] = 0.0
    
    return metrics

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Enhanced model comparison with improved metrics")
    parser.add_argument("--cognitive", required=True, help="Path to cognitive model")
    parser.add_argument("--baseline", required=True, help="Path to baseline model")
    parser.add_argument("--test_data", required=True, help="Path to test data")
    parser.add_argument("--seq_length", type=int, default=20, help="Sequence length")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--output", default="validation/results", help="Output directory")
    parser.add_argument("--ablation", action="store_true", help="Run ablation study")
    
    args = parser.parse_args()
    
    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load models with improved error handling
    print(f"Loading cognitive model: {args.cognitive}")
    try:
        cognitive_model = load_model(args.cognitive)
        cognitive_model.to(device)
        cognitive_model.eval()
    except Exception as e:
        print(f"Error loading cognitive model: {e}")
        return
    
    print(f"Loading baseline model: {args.baseline}")
    try:
        baseline_model = FinancialLSTMBaseline()
        baseline_model.load_state_dict(torch.load(args.baseline, map_location=device))
        baseline_model.to(device)
        baseline_model.eval()
    except Exception as e:
        print(f"Error loading baseline model: {e}")
        return
    
    # Load test data with error handling
    print(f"Loading test data: {args.test_data}")
    try:
        test_loader = FinancialDataLoader(
            path=args.test_data,
            seq_length=args.seq_length,
            batch_size=args.batch
        )
    except Exception as e:
        print(f"Error loading test data: {e}")
        return
    
    # Process baseline model predictions with enhanced tracking
    baseline_preds = []
    targets_list = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            features, sequence, targets = batch
            features = features.to(device)
            sequence = sequence.to(device)
            targets = targets.to(device)
            
            outputs = baseline_model(sequence)
            
            # Track predictions and targets
            baseline_preds.append(outputs[:, -1].cpu().numpy())
            targets_list.append(targets.cpu().numpy())
            
            # Progress indicator
            if batch_idx % 10 == 0:
                print(f"Processed {batch_idx * args.batch} samples for baseline model")
    
    baseline_preds = np.concatenate(baseline_preds, axis=0)
    targets = np.concatenate(targets_list, axis=0)
    
    # Reset data loader
    test_loader.pointer = 0
    
    # Process cognitive model predictions with enhanced tracking
    cognitive_preds = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            features, sequence, targets = batch
            features = features.to(device)
            sequence = sequence.to(device)
            
            # Forward pass with error handling
            try:
                outputs = cognitive_model(
                    seq=None, 
                    financial_data=features, 
                    financial_seq=sequence,
                    skip_memory=False  # Use memory for evaluation
                )
                
                # Extract market state
                if isinstance(outputs, dict) and 'market_state' in outputs:
                    market_state = outputs['market_state']
                else:
                    # Fallback for non-standard outputs
                    market_state = outputs
                
                cognitive_preds.append(market_state.cpu().numpy())
            except Exception as e:
                print(f"Error during cognitive model forward pass (batch {batch_idx}): {e}")
                # Use zeros as fallback
                cognitive_preds.append(np.zeros((features.size(0), 4)))
                
            # Progress indicator
            if batch_idx % 10 == 0:
                print(f"Processed {batch_idx * args.batch} samples for cognitive model")
    
    cognitive_preds = np.concatenate(cognitive_preds, axis=0)
    
    # Create shifted versions for directional metrics
    baseline_preds_prev = np.roll(baseline_preds, 1, axis=0)
    cognitive_preds_prev = np.roll(cognitive_preds, 1, axis=0)
    targets_prev = np.roll(targets, 1, axis=0)
    
    # Calculate enhanced metrics
    baseline_metrics = calculate_metrics(
        baseline_preds, targets, baseline_preds_prev, targets_prev
    )
    
    cognitive_metrics = calculate_metrics(
        cognitive_preds, targets, cognitive_preds_prev, targets_prev
    )
    
    # Calculate improvements with better precision handling
    improvements = {}
    for metric in baseline_metrics:
        base_val = baseline_metrics[metric]
        cog_val = cognitive_metrics[metric]
        
        if abs(base_val) > 1e-6:
            imp = ((cog_val - base_val) / abs(base_val)) * 100
        else:
            # Handle division by very small values
            if abs(cog_val) < 1e-6:
                imp = 0.0  # Both values essentially zero
            else:
                imp = float('inf') if cog_val > 0 else float('-inf')
        
        improvements[metric] = float(imp)
    
    # Run ablation study if requested
    if args.ablation:
        ablation_results = run_ablation_study(
            cognitive_model, test_loader, device, args.batch
        )
    else:
        ablation_results = {}
    
    # Print enhanced results
    print("\nBaseline model metrics:")
    for k, v in baseline_metrics.items():
        print(f"  {k}: {v:.4f}")
    
    print("\nCognitive model metrics:")
    for k, v in cognitive_metrics.items():
        print(f"  {k}: {v:.4f}")
    
    print("\nImprovements:")
    for k, v in improvements.items():
        print(f"  {k}: {v:+.2f}%")
    
    # Print ablation results if available
    if ablation_results:
        print("\nAblation Study Results:")
        for component, score in ablation_results.items():
            print(f"  Without {component}: {score:.4f}")
    
    # Save enhanced results
    results = {
        'baseline_metrics': baseline_metrics,
        'cognitive_metrics': cognitive_metrics,
        'improvements': improvements,
        'ablation_results': ablation_results
    }
    
    os.makedirs(args.output, exist_ok=True)
    
    with open(os.path.join(args.output, 'enhanced_comparison.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\nResults saved to {args.output}")

def run_ablation_study(model, test_loader, device, batch_size):
    """
    Run ablation study to measure the impact of different components
    Args:
        model: CognitiveArchitecture model
        test_loader: Data loader for test set
        device: Computation device
        batch_size: Batch size
    Returns:
        Dictionary of component scores
    """
    results = {}
    
    # Test without memory
    print("\nRunning ablation: without memory")
    memory_preds = []
    targets_list = []
    
    with torch.no_grad():
        test_loader.pointer = 0  # Reset loader
        for batch_idx, batch in enumerate(test_loader):
            features, sequence, targets = batch
            features = features.to(device)
            sequence = sequence.to(device)
            targets = targets.to(device)
            
            try:
                outputs = model(
                    seq=None, 
                    financial_data=features, 
                    financial_seq=sequence,
                    skip_memory=True  # Skip memory for this test
                )
                
                # Extract market state
                if isinstance(outputs, dict) and 'market_state' in outputs:
                    market_state = outputs['market_state']
                else:
                    market_state = outputs
                
                memory_preds.append(market_state.cpu().numpy())
                targets_list.append(targets.cpu().numpy())
            except Exception as e:
                print(f"Error during memory ablation (batch {batch_idx}): {e}")
                memory_preds.append(np.zeros((features.size(0), 4)))
                targets_list.append(targets.cpu().numpy())
                
            # Progress indicator
            if batch_idx % 10 == 0:
                print(f"Processed {batch_idx * batch_size} samples for memory ablation")
    
    memory_preds = np.concatenate(memory_preds, axis=0)
    targets = np.concatenate(targets_list, axis=0)
    
    # Calculate metrics for memory ablation
    memory_metrics = calculate_metrics(memory_preds, targets)
    results['memory'] = memory_metrics['combined_score']
    
    # Test without cross-dimensional attention
    print("\nRunning ablation: without cross-dimensional attention")
    
    # Temporarily replace cross-dimensional attention with identity function
    original_cd_forward = model.cross_dimensional_attention.forward
    model.cross_dimensional_attention.forward = lambda x: x
    
    cd_preds = []
    test_loader.pointer = 0  # Reset loader
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            features, sequence, targets = batch
            features = features.to(device)
            sequence = sequence.to(device)
            
            try:
                outputs = model(
                    seq=None, 
                    financial_data=features, 
                    financial_seq=sequence
                )
                
                # Extract market state
                if isinstance(outputs, dict) and 'market_state' in outputs:
                    market_state = outputs['market_state']
                else:
                    market_state = outputs
                
                cd_preds.append(market_state.cpu().numpy())
            except Exception as e:
                print(f"Error during CD attention ablation (batch {batch_idx}): {e}")
                cd_preds.append(np.zeros((features.size(0), 4)))
                
            # Progress indicator
            if batch_idx % 10 == 0:
                print(f"Processed {batch_idx * batch_size} samples for CD attention ablation")
    
    # Restore original forward method
    model.cross_dimensional_attention.forward = original_cd_forward
    
    cd_preds = np.concatenate(cd_preds, axis=0)
    
    # Calculate metrics for CD attention ablation
    cd_metrics = calculate_metrics(cd_preds, targets)
    results['cross_dim_attention'] = cd_metrics['combined_score']
    
    # Test without regime detection
    print("\nRunning ablation: without regime detection")
    
    # Temporarily disable regime detection
    if hasattr(model, 'regime_detection_enabled'):
        original_regime_enabled = model.regime_detection_enabled
        model.regime_detection_enabled = False
    else:
        # Add attribute if not present
        model.regime_detection_enabled = False
        original_regime_enabled = True
    
    regime_preds = []
    test_loader.pointer = 0  # Reset loader
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            features, sequence, targets = batch
            features = features.to(device)
            sequence = sequence.to(device)
            
            try:
                outputs = model(
                    seq=None, 
                    financial_data=features, 
                    financial_seq=sequence
                )
                
                # Extract market state
                if isinstance(outputs, dict) and 'market_state' in outputs:
                    market_state = outputs['market_state']
                else:
                    market_state = outputs
                
                regime_preds.append(market_state.cpu().numpy())
            except Exception as e:
                print(f"Error during regime detection ablation (batch {batch_idx}): {e}")
                regime_preds.append(np.zeros((features.size(0), 4)))
                
            # Progress indicator
            if batch_idx % 10 == 0:
                print(f"Processed {batch_idx * batch_size} samples for regime detection ablation")
    
    # Restore original regime detection setting
    model.regime_detection_enabled = original_regime_enabled
    
    regime_preds = np.concatenate(regime_preds, axis=0)
    
    # Calculate metrics for regime detection ablation
    regime_metrics = calculate_metrics(regime_preds, targets)
    results['regime_detection'] = regime_metrics['combined_score']
    
    return results

if __name__ == "__main__":
    main()
