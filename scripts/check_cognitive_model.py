#!/usr/bin/env python
# check_cognitive_model.py - Verify cognitive model fixes

import os
import sys
import torch

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.arch.cognitive import CognitiveArchitecture
from src.data.financial_loader import EnhancedFinancialDataLoader

def main():
    print("Testing CognitiveArchitecture model dimensions...")
    
    # Create a model with various input_dim values to verify fix
    test_input_dims = [5, 194, 324]  # Various input dimensions
    
    for dim in test_input_dims:
        print(f"\nTesting with input_dim={dim}:")
        model = CognitiveArchitecture(input_dim=dim, hidden_dim=64, output_dim=7)
        
        # Check financial_encoder input dimension
        # Get the first layer of the encoder which should be a Linear layer
        first_layer = None
        for module in model.financial_encoder.encoder:
            if isinstance(module, torch.nn.Linear):
                first_layer = module
                break
        
        if first_layer:
            fin_encoder_input_dim = first_layer.in_features
            print(f"  financial_encoder input dimension: {fin_encoder_input_dim}")
            assert fin_encoder_input_dim == 7, "Financial encoder input dimension should be 7"
        
        # Check temporal_encoder input dimension
        temporal_input_dim = model.temporal_encoder.lstm.input_size
        print(f"  temporal_encoder input dimension: {temporal_input_dim}")
        assert temporal_input_dim == 7, "Temporal encoder input dimension should be 7"
        
        # Check temporal_hierarchy input dimension
        if hasattr(model.temporal_hierarchy, 'input_dim'):
            hierarchy_input_dim = model.temporal_hierarchy.input_dim
            print(f"  temporal_hierarchy input dimension: {hierarchy_input_dim}")
            assert hierarchy_input_dim == 7, "Temporal hierarchy input dimension should be 7"
    
    print("\nSimulating a forward pass with sample data...")
    
    # Create dummy input data matching the expected 7 features
    batch_size = 2
    seq_length = 20
    financial_seq = torch.randn(batch_size, seq_length, 7)  # 7 features
    financial_data = torch.randn(batch_size, 7)  # 7 features
    
    # Try a forward pass
    try:
        model = CognitiveArchitecture(input_dim=324, hidden_dim=64, output_dim=7)
        model.eval()
        
        with torch.no_grad():
            outputs = model(financial_data=financial_data, financial_seq=financial_seq)
            
        print("✓ Forward pass successful!")
        print(f"  Output shape: {outputs['market_state'].shape}")
    except Exception as e:
        print(f"✗ Forward pass failed: {str(e)}")
    
    print("\nVerification complete. The model should now correctly handle 7-dimensional input features.")

if __name__ == "__main__":
    main() 