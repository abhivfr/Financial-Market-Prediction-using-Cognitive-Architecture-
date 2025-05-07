import torch
import json
import sys
from src.arch.cognitive import CognitiveArchitecture

# Load the checkpoint
checkpoint = torch.load('models/latest/best_model.pt', map_location='cpu')
print("Checkpoint is a state dict (not dict with 'model_state_dict')")
print(f"Keys in checkpoint: {len(checkpoint.keys())} keys")
print(f"First few keys: {list(checkpoint.keys())[:5]}")

# Load config from config.json
try:
    with open('models/latest/config.json', 'r') as f:
        config = json.load(f)
    print("\nConfig from config.json:")
    for k, v in config.items():
        print(f"{k}: {v}")
    
    # Create model with this config
    print("\nCreating model with config from config.json")
    model = CognitiveArchitecture(
        input_dim=config.get('input_dim', 5),
        hidden_dim=config.get('hidden_dim', 64),
        memory_size=config.get('memory_size', 50),
        output_dim=config.get('output_dim', 4),
        seq_length=config.get('sequence_length', 20)
    )
    
    # Print model parameters before loading checkpoint
    print("\nModel parameters before loading checkpoint:")
    param_count = 0
    for name, param in model.named_parameters():
        param_count += 1
        if param_count <= 5 or param_count > len(list(model.parameters())) - 5:
            print(f"{name}: {param.shape}")
    print(f"Total parameters: {param_count}")
    
    # Try to load the checkpoint
    try:
        print("\nAttempting to load the checkpoint...")
        model.load_state_dict(checkpoint)
        print("Successfully loaded checkpoint!")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        
        # Get the actual missing/mismatched keys
        unexpected, missing = model.load_state_dict(checkpoint, strict=False)
        print(f"\nMissing keys: {missing}")
        print(f"Unexpected keys: {unexpected}")
        
        # Check for size mismatches
        for name, param in model.named_parameters():
            if name in checkpoint:
                if param.shape != checkpoint[name].shape:
                    print(f"Size mismatch for {name}: model={param.shape}, checkpoint={checkpoint[name].shape}")
    
except Exception as e:
    print(f"Error in script: {e}") 