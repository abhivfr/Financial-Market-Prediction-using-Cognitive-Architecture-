#!/usr/bin/env python
import torch
import sys

class ModelWrapper(torch.nn.Module):
    """Wrapper to handle different model output formats"""
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, financial_data, financial_seq, skip_memory=False):
        outputs = self.model(financial_data=financial_data, financial_seq=financial_seq, skip_memory=skip_memory)
        
        if isinstance(outputs, dict):
            # If model returns dictionary, extract market_state_sequence
            return outputs['market_state_sequence']
        elif isinstance(outputs, (list, tuple)) and len(outputs) > 0:
            # If model returns list/tuple, find tensors
            tensor_outputs = [o for o in outputs if isinstance(o, torch.Tensor)]
            if tensor_outputs:
                return tensor_outputs[-1]  # Return last tensor
            else:
                # Fallback - use input as prediction
                return financial_seq
        else:
            # Fallback - return unchanged
            return outputs

def patch_compare_models():
    """Function to patch compare_models_simple.py to use ModelWrapper"""
    import torch
    
    # Only import CognitiveArchitecture if we can find it
    try:
        from src.arch.cognitive import CognitiveArchitecture
    except ImportError:
        print("Warning: Could not import CognitiveArchitecture")
        return
    
    # Original load function
    orig_load = torch.load
    
    # Patched load function that wraps CognitiveArchitecture models
    def patched_load(path, *args, **kwargs):
        model_data = orig_load(path, *args, **kwargs)
        if isinstance(path, str) and 'financial_consciousness' in path:
            print(f"Wrapping CognitiveArchitecture model from {path}")
            model = CognitiveArchitecture()
            model.load_state_dict(model_data)
            return ModelWrapper(model)
        return model_data
        
    # Apply monkey patch
    torch.load = patched_load
    print("Model loading patched to use ModelWrapper for CognitiveArchitecture models")


import torch
import torch.nn as nn

class ModelWrapper(nn.Module):
    """Wrapper for the cognitive architecture to standardize outputs"""
    
    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model
        
    def forward(self, features, sequence):
        """
        Standardize model output format
        
        Args:
            features: Current financial features
            sequence: Historical sequence data
            
        Returns:
            Tensor: Standardized market state prediction
        """
        # Forward pass through the model
        outputs = self.model(financial_data=features, financial_seq=sequence)
        
        # Handle different output formats
        if isinstance(outputs, dict):
            # If model returns a dictionary with 'market_state'
            return outputs['market_state']
        elif isinstance(outputs, tuple) and len(outputs) == 6:
            # If model returns a tuple of (fused, financial_feat, recalled, attended, attn_weights, market_state)
            _, _, _, _, _, market_state = outputs
            return market_state
        else:
            # Fallback for other output formats
            raise ValueError(f"Unexpected model output format: {type(outputs)}")
