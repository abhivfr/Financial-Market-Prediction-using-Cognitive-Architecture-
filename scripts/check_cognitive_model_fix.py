#!/usr/bin/env python
# check_cognitive_model_fix.py - Verify cognitive model dimension fixes

import os
import sys
import torch
import pprint

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.arch.cognitive import CognitiveArchitecture, FinancialEncoder, TemporalEncoder

def print_layer_shapes(model):
    """Print the shapes of key layers in the model"""
    print("\nModel internal dimensions:")
    
    # Check projection layer
    if hasattr(model, 'feature_projection'):
        proj_in = model.feature_projection.in_features
        proj_out = model.feature_projection.out_features
        print(f"  feature_projection: in={proj_in}, out={proj_out}")
    
    # Check financial encoder first layer
    first_layer = None
    for module in model.financial_encoder.encoder:
        if isinstance(module, torch.nn.Linear):
            first_layer = module
            break
    
    if first_layer:
        fin_in = first_layer.in_features
        fin_out = first_layer.out_features
        print(f"  financial_encoder first layer: in={fin_in}, out={fin_out} [weight={list(first_layer.weight.shape)}]")
    
    # Check temporal encoder LSTM
    temp_in = model.temporal_encoder.lstm.input_size
    temp_hidden = model.temporal_encoder.lstm.hidden_size
    print(f"  temporal_encoder LSTM: in={temp_in}, hidden={temp_hidden}")
    
    # Check temporal hierarchy inputs
    if hasattr(model.temporal_hierarchy, 'scale_encoders'):
        hier_in = model.temporal_hierarchy.scale_encoders[0].lstm.input_size
        print(f"  temporal_hierarchy first encoder: in={hier_in}")
    
    # Check market state predictor
    market_in = model.market_state_sequence_predictor.lstm.input_size
    market_out = model.market_state_sequence_predictor.predictor.out_features
    print(f"  market_state_sequence_predictor: lstm_in={market_in}, out={market_out}")

def load_checkpoint_and_check(checkpoint_path):
    """Load a checkpoint and check its weight shapes"""
    print(f"\nAnalyzing checkpoint: {checkpoint_path}")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Extract key shapes from checkpoint
        key_shapes = {}
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
            
        # Get shapes of important weights
        key_weights = [
            'financial_encoder.encoder.0.weight',
            'temporal_encoder.lstm.weight_ih_l0',
            'temporal_hierarchy.scale_encoders.0.lstm.weight_ih_l0',
            'market_state_sequence_predictor.predictor.weight'
        ]
        
        for key in key_weights:
            if key in state_dict:
                key_shapes[key] = list(state_dict[key].shape)
        
        print("Key weight shapes from checkpoint:")
        pprint.pprint(key_shapes)
        
        # Extract config
        config = {}
        if isinstance(checkpoint, dict) and 'config' in checkpoint:
            config = checkpoint['config']
        else:
            # Try to infer from weights
            if 'financial_encoder.encoder.0.weight' in state_dict:
                config['input_dim'] = state_dict['financial_encoder.encoder.0.weight'].shape[1]
            elif 'temporal_encoder.lstm.weight_ih_l0' in state_dict:
                config['input_dim'] = state_dict['temporal_encoder.lstm.weight_ih_l0'].shape[1]
                
            if 'market_state_sequence_predictor.predictor.weight' in state_dict:
                config['output_dim'] = state_dict['market_state_sequence_predictor.predictor.weight'].shape[0]
                
        print("\nInferred config:")
        pprint.pprint(config)
        
        return config, key_shapes
        
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return {}, {}

def test_model_with_config(config):
    """Test the model with the given config"""
    print("\nInstantiating model with config:")
    pprint.pprint(config)
    
    model = CognitiveArchitecture(**config)
    print_layer_shapes(model)
    
    # Test forward pass with 7-feature data
    print("\nTesting forward pass with 7-feature data...")
    batch_size = 2
    seq_length = 20
    
    # Create sample 7-feature data
    financial_seq = torch.randn(batch_size, seq_length, 7)
    financial_data = torch.randn(batch_size, 7)
    
    try:
        model.eval()
        with torch.no_grad():
            outputs = model(financial_data=financial_data, financial_seq=financial_seq)
        
        print("✓ Forward pass successful!")
        market_state_shape = outputs['market_state'].shape
        print(f"  market_state shape: {list(market_state_shape)}")
    except Exception as e:
        print(f"✗ Forward pass failed: {str(e)}")
    
    return model

def main():
    print("COGNITIVE MODEL DIMENSION VERIFICATION")
    print("======================================")
    
    # Path to checkpoint
    checkpoint_path = os.path.join("models", "latest", "best_model.pt")
    
    # Check if the checkpoint exists
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        alternative_paths = ["models/cognitive/best_model.pt", "models/cognitive.pt"]
        
        for alt_path in alternative_paths:
            if os.path.exists(alt_path):
                print(f"Found alternative checkpoint at {alt_path}")
                checkpoint_path = alt_path
                break
        else:
            print("No checkpoint found. Skipping checkpoint analysis.")
            checkpoint_path = None
    
    if checkpoint_path:
        # Load and analyze checkpoint
        config, key_shapes = load_checkpoint_and_check(checkpoint_path)
        
        if config:
            # Test model with the config from checkpoint
            model = test_model_with_config(config)
            
            # Try to load state dict
            print("\nTesting state_dict loading...")
            try:
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                else:
                    state_dict = checkpoint
                
                # Filter out projection layer keys (will be initialized separately)
                filtered_state_dict = {k: v for k, v in state_dict.items() 
                                      if 'feature_projection' not in k}
                
                model.load_state_dict(filtered_state_dict, strict=False)
                print("✓ State dictionary loaded successfully with strict=False!")
            except Exception as e:
                print(f"✗ State dictionary loading failed: {str(e)}")
            
            # Test from_checkpoint method
            print("\nTesting CognitiveArchitecture.from_checkpoint method...")
            try:
                checkpoint_model = CognitiveArchitecture.from_checkpoint(checkpoint_path)
                print("✓ Model loaded successfully from checkpoint!")
                print_layer_shapes(checkpoint_model)
                
                # Test forward pass
                print("\nTesting forward pass with loaded checkpoint model...")
                batch_size = 2
                seq_length = 20
                financial_seq = torch.randn(batch_size, seq_length, 7)
                financial_data = torch.randn(batch_size, 7)
                
                checkpoint_model.eval()
                with torch.no_grad():
                    outputs = checkpoint_model(financial_data=financial_data, financial_seq=financial_seq)
                
                print("✓ Forward pass successful with checkpoint-loaded model!")
                print(f"  market_state shape: {list(outputs['market_state'].shape)}")
                
            except Exception as e:
                print(f"✗ Loading model from checkpoint failed: {str(e)}")
    
    # Test with hardcoded config
    print("\n\nTesting with hardcoded config (input_dim=324, output_dim=7):")
    hardcoded_config = {
        'input_dim': 324,
        'hidden_dim': 64,
        'memory_size': 50,
        'output_dim': 7,
        'seq_length': 20
    }
    
    test_model_with_config(hardcoded_config)
    
    print("\nVerification complete!")

if __name__ == "__main__":
    main() 