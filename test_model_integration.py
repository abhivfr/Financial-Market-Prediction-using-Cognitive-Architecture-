#!/usr/bin/env python
import torch
import numpy as np
import os
import sys
import argparse
from contextlib import redirect_stdout

# Fix import paths for nested directory structure
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from src.arch.cognitive import CognitiveArchitecture
from src.arch.baseline_lstm import FinancialLSTMBaseline
from train import FinancialDataLoader

def run_integration_test(model_path, test_data_path, seq_length=10, batch_size=4):
    """
    Run integration tests on the model to verify all components work
    Args:
        model_path: Path to saved model
        test_data_path: Path to test data
        seq_length: Sequence length for temporal features
        batch_size: Batch size for testing
    """
    print(f"Running integration tests on model: {model_path}")
    
    # Create output directory for test results
    os.makedirs("test_results", exist_ok=True)
    
    # Use CPU for reproducibility
    device = torch.device("cpu")
    
    # Load model
    try:
        with redirect_stdout(open("test_results/model_load.log", "w")):
            model = CognitiveArchitecture()
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return False
    
    # Load test data
    try:
        with redirect_stdout(open("test_results/data_load.log", "w")):
            test_loader = FinancialDataLoader(
                path=test_data_path,
                seq_length=seq_length,
                batch_size=batch_size
            )
        print("✓ Test data loaded successfully")
    except Exception as e:
        print(f"✗ Error loading test data: {e}")
        return False
    
    # Test individual components
    components = [
        "cross_dimensional_attention",
        "temporal_hierarchy",
        "financial_memory",
        "volume_processor",
        "market_state_sequence_predictor"
    ]
    
    component_status = {}
    
    # Get a batch for testing
    try:
        features, sequence, targets = next(iter(test_loader))
        print(f"✓ Batch retrieved: features={features.shape}, sequence={sequence.shape}, targets={targets.shape}")
    except Exception as e:
        print(f"✗ Error getting batch: {e}")
        return False
    
    # Test each component
    for component in components:
        print(f"Testing component: {component}")
        try:
            with redirect_stdout(open(f"test_results/{component}.log", "w")):
                if component == "cross_dimensional_attention":
                    output = model.cross_dimensional_attention(sequence)
                    assert output.shape == sequence.shape
                    
                elif component == "temporal_hierarchy":
                    output, regime_probs = model.temporal_hierarchy(sequence)
                    assert output.shape[0] == batch_size
                    assert regime_probs.shape == (batch_size, 3)
                    
                elif component == "financial_memory":
                    if hasattr(model, "financial_memory"):
                        encoded = model.financial_encoder(features)
                        values, variance = model.financial_memory.retrieve(encoded)
                        assert values.shape[0] == batch_size
                    else:
                        raise AttributeError("Model does not have financial_memory")
                        
                elif component == "volume_processor":
                    if hasattr(model, "volume_processor"):
                        volume = sequence[:, :, 1:2]
                        output, variance = model.volume_processor(sequence, volume)
                        assert output.shape == sequence.shape
                    else:
                        raise AttributeError("Model does not have volume_processor")
                        
                elif component == "market_state_sequence_predictor":
                    if hasattr(model, "market_state_sequence_predictor"):
                        temporal_feat, _ = model.temporal_encoder(sequence)
                        output = model.market_state_sequence_predictor['base'](temporal_feat)
                        assert output.shape == (batch_size, sequence.shape[1], 4)
                    else:
                        raise AttributeError("Model does not have market_state_sequence_predictor")
                
            component_status[component] = "✓"
            print(f"✓ {component} test passed")
        except Exception as e:
            component_status[component] = f"✗ {str(e)}"
            print(f"✗ {component} test failed: {e}")
    
    # Test full forward pass
    print("Testing full forward pass")
    try:
        with redirect_stdout(open("test_results/forward_pass.log", "w")):
            outputs = model(financial_data=features, financial_seq=sequence)
            
            # Verify outputs structure
            assert isinstance(outputs, dict)
            assert 'market_state' in outputs
            assert outputs['market_state'].shape == (batch_size, 4)
            
            # Verify market_state_sequence
            assert 'market_state_sequence' in outputs
            assert outputs['market_state_sequence'].shape == (batch_size, sequence.shape[1], 4)
            
            # Verify regime probabilities
            assert 'regime_probabilities' in outputs
            assert outputs['regime_probabilities'].shape == (batch_size, 3)
        
        print("✓ Full forward pass successful")
    except Exception as e:
        print(f"✗ Full forward pass failed: {e}")
        
    # Print summary
    print("\nIntegration Test Summary:")
    print("-----------------------")
    for component, status in component_status.items():
        print(f"{component}: {status}")
    print(f"Full Forward Pass: {'✓' if 'market_state' in outputs else '✗'}")
    print("\nComplete test logs saved in test_results/ directory")
    
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run integration tests on the cognitive model")
    parser.add_argument("--model", required=True, help="Path to saved model")
    parser.add_argument("--data", required=True, help="Path to test data")
    parser.add_argument("--seq_length", type=int, default=10, help="Sequence length")
    parser.add_argument("--batch", type=int, default=4, help="Batch size")
    
    args = parser.parse_args()
    run_integration_test(args.model, args.data, args.seq_length, args.batch)
