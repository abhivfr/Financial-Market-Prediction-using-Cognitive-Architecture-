#!/usr/bin/env python
# stress_test_fix.py - Fixed version of stress test for cognitive models

import os
import sys
import argparse
import torch
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from src.arch.cognitive import CognitiveArchitecture
from src.arch.baseline_lstm import FinancialLSTMBaseline
from scripts.stress_test import StressTestScenario, MarketCrashScenario, HighVolatilityScenario, RegimeChangeScenario, NovelPatternScenario

def fix_multidim_prediction(prediction):
    """
    Fix multi-dimensional predictions to extract the return value
    
    Args:
        prediction: Model prediction, which could be multi-dimensional
        
    Returns:
        Single-dimensional prediction (return value)
    """
    # If prediction is multi-dimensional, use first dimension as return value
    if isinstance(prediction, np.ndarray) and len(prediction.shape) > 0 and prediction.shape[0] > 1:
        return prediction[0]
    return prediction

def run_stress_tests_fixed(
    model_path: str,
    test_data_path: str,
    output_dir: str,
    scenarios: List[str] = None,
    model_type: str = "cognitive",
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> Dict[str, Any]:
    """
    Run stress tests on model with fixes for multi-dimensional predictions
    
    Args:
        model_path: Path to model checkpoint
        test_data_path: Path to test data
        output_dir: Directory to save results
        scenarios: List of scenarios to run
        model_type: Model type ('cognitive' or 'baseline')
        device: Computation device
        
    Returns:
        Dictionary with stress test results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load test data
    test_data = pd.read_csv(test_data_path)
    print(f"Loaded test data with {len(test_data)} samples")
    
    # Load model
    print(f"Loading {model_type} model from {model_path}")
    
    try:
        # For cognitive models, use the new from_checkpoint method
        if model_type.lower() == "cognitive":
            try:
                model = CognitiveArchitecture.from_checkpoint(model_path, device)
                # Get configuration from model attributes
                config = {
                    'input_dim': model.input_dim,
                    'hidden_dim': model.hidden_dim,
                    'memory_size': getattr(model.financial_memory, 'num_slots', 50),
                    'output_dim': model.output_dim,
                    'seq_length': getattr(model, 'seq_length', 20)
                }
                print(f"Using model configuration: input_dim={config['input_dim']}, hidden_dim={config['hidden_dim']}, memory_size={config['memory_size']}")
            except Exception as e:
                print(f"Error loading cognitive model with from_checkpoint: {e}")
                raise
        else:
            # Load baseline model with legacy method
            checkpoint = torch.load(model_path, map_location=device)
            
            # Extract configuration based on actual dimensions in the checkpoint
            config = {}
            
            # Case 1: If 'config' exists in checkpoint, use it
            if isinstance(checkpoint, dict) and 'config' in checkpoint:
                config = checkpoint['config']
            
            # Case 2: If checkpoint is a state dict (most likely)
            else:
                # Try to determine dimensions directly from the weights
                if 'lstm.weight_ih_l0' in checkpoint:
                    # Extract input dimension from the LSTM input weights
                    input_dim = checkpoint['lstm.weight_ih_l0'].shape[1]
                    config['input_dim'] = input_dim
                    
                    # Extract hidden dimension from LSTM weights (LSTM has 4 gates)
                    hidden_dim = checkpoint['lstm.weight_ih_l0'].shape[0] // 4
                    config['hidden_dim'] = hidden_dim
                    
                    # Extract output dimension from final layer if available
                    if 'fc.weight' in checkpoint:
                        output_dim = checkpoint['fc.weight'].shape[0]
                        config['output_dim'] = output_dim
                
                # Default values if not found
                if 'input_dim' not in config:
                    config['input_dim'] = 7  # Default for baseline
                if 'hidden_dim' not in config:
                    config['hidden_dim'] = 64  # Default
                if 'output_dim' not in config:
                    config['output_dim'] = 7  # Default for baseline
            
            print(f"Using baseline model configuration: input_dim={config['input_dim']}, hidden_dim={config['hidden_dim']}")
            
            # Create model with extracted configuration
            model = FinancialLSTMBaseline(
                input_dim=config.get('input_dim', 7),
                hidden_dim=config.get('hidden_dim', 64),
                num_layers=config.get('num_layers', 2),
                output_dim=config.get('output_dim', 7)
            )
            
            # Load state dict
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
        
    model.to(device)
    model.eval()
    
    # Define available scenarios
    available_scenarios = {
        "market_crash": MarketCrashScenario(crash_magnitude=-0.1, crash_duration=5),
        "high_volatility": HighVolatilityScenario(volatility_multiplier=2.5, duration=10),
        "regime_change": RegimeChangeScenario(from_regime=1, to_regime=2, transition_duration=5),
        "novel_pattern": NovelPatternScenario(pattern_duration=15, pattern_type="oscillation")
    }
    
    # Monkey patch the evaluate methods to handle multi-dimensional predictions
    for scenario_name, scenario in available_scenarios.items():
        original_evaluate = scenario.evaluate
        
        def create_patched_evaluate(original_evaluate):
            def patched_evaluate(self, model, data, device):
                # Store original prediction extraction
                original_model_forward = model.forward
                
                # Patch model.forward to handle multi-dimensional outputs
                def patched_forward(*args, **kwargs):
                    outputs = original_model_forward(*args, **kwargs)
                    
                    # Handle dictionary outputs from cognitive model
                    if isinstance(outputs, dict) and 'market_state' in outputs:
                        # Extract market_state and ensure first element in sequence is returned
                        market_state = outputs['market_state']
                        if hasattr(market_state, 'shape') and len(market_state.shape) > 2:
                            # If it's a sequence prediction, get the first step prediction
                            outputs['market_state'] = market_state[:, 0, :]
                    
                    return outputs
                
                # Apply patch
                model.forward = patched_forward
                
                # Call original evaluate
                result = original_evaluate(self, model, data, device)
                
                # Restore original forward method
                model.forward = original_model_forward
                
                return result
            
            return patched_evaluate
        
        # Apply the patched evaluate method
        scenario.evaluate = create_patched_evaluate(original_evaluate).__get__(scenario, type(scenario))
    
    # Select scenarios to run
    if scenarios is None:
        scenarios = list(available_scenarios.keys())
    else:
        # Handle comma separated string input for scenarios
        if isinstance(scenarios, str):
            scenarios = scenarios.split(',')
    
    selected_scenarios = [available_scenarios[name] for name in scenarios if name in available_scenarios]
    
    if not selected_scenarios:
        raise ValueError(f"No valid scenarios selected. Available scenarios: {list(available_scenarios.keys())}")
    
    # Run scenarios
    results = {}
    for scenario in selected_scenarios:
        print(f"Running stress test: {scenario.name} - {scenario.description}")
        
        # Run scenario
        try:
            metrics, figure = scenario.evaluate(model, test_data, device)
            
            # Save visualization
            figure_path = os.path.join(output_dir, f"{scenario.name}_visualization.png")
            figure.savefig(figure_path, dpi=300)
            plt.close(figure)
            
            # Save metrics
            metrics_path = os.path.join(output_dir, f"{scenario.name}_metrics.json")
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            
            # Store results
            results[scenario.name] = {
                'description': scenario.description,
                'metrics': metrics,
                'figure_path': figure_path
            }
            
            print(f"  Results saved to {figure_path} and {metrics_path}")
        except Exception as e:
            print(f"Error running scenario {scenario.name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save overall results
    summary_path = os.path.join(output_dir, "stress_test_results.json")
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Stress testing completed. Results saved to {output_dir}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Fixed stress testing for cognitive models")
    parser.add_argument("--model_path", required=True, help="Path to model checkpoint")
    parser.add_argument("--test_data", required=True, help="Path to test data")
    parser.add_argument("--output_dir", default="evaluation/stress_test", help="Directory to save results")
    parser.add_argument("--model_type", default="cognitive", choices=["cognitive", "baseline"], help="Model type")
    parser.add_argument("--scenarios", default="market_crash,high_volatility", 
                       help="Comma-separated list of scenarios to run")
    parser.add_argument("--device", default=None, help="Computation device")
    parser.add_argument("--compare_with", default=None, help="Path to model to compare with")
    parser.add_argument("--compare_type", default="baseline", help="Type of model to compare with")
    
    args = parser.parse_args()
    
    # Set device
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    try:
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Run stress tests on primary model
        primary_results = run_stress_tests_fixed(
            model_path=args.model_path,
            test_data_path=args.test_data,
            output_dir=os.path.join(args.output_dir, args.model_type),
            scenarios=args.scenarios,
            model_type=args.model_type,
            device=args.device
        )
        
        # Run comparison model if provided
        if args.compare_with:
            print(f"\nRunning comparison with {args.compare_type} model: {args.compare_with}")
            compare_results = run_stress_tests_fixed(
                model_path=args.compare_with,
                test_data_path=args.test_data,
                output_dir=os.path.join(args.output_dir, args.compare_type),
                scenarios=args.scenarios,
                model_type=args.compare_type,
                device=args.device
            )
            
            # Create comparison summary
            print("\nGenerating comparison summary")
            comparison = {
                'primary': {
                    'model_path': args.model_path,
                    'model_type': args.model_type,
                    'results': primary_results
                },
                'comparison': {
                    'model_path': args.compare_with,
                    'model_type': args.compare_type,
                    'results': compare_results
                }
            }
            
            # Save comparison results
            comparison_path = os.path.join(args.output_dir, "model_comparison.json")
            with open(comparison_path, 'w') as f:
                json.dump(comparison, f, indent=2)
            
            print(f"Comparison summary saved to {comparison_path}")
        
        print(f"Stress test complete. Results saved to {args.output_dir}")
        return 0
    
    except Exception as e:
        print(f"Error during stress testing: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main()) 