import torch
import json

# Load checkpoint
checkpoint = torch.load('models/latest/best_model.pt', map_location='cpu')

# Extract dimensions from the checkpoint
# First, get the input dimension from the temporal encoder weights 
input_dim = checkpoint['temporal_encoder.lstm.weight_ih_l0'].shape[1]  # This is input_dim
hidden_dim = checkpoint['temporal_encoder.lstm.weight_ih_l0'].shape[0] // 4  # LSTM has 4 gates
memory_size = checkpoint['financial_memory.memory'].shape[0]  # Number of memory slots

# Also extract from the financial encoder
financial_input_dim = checkpoint['financial_encoder.encoder.0.weight'].shape[1]

print(f"Extracted dimensions from checkpoint:")
print(f"input_dim: {input_dim}")
print(f"financial_input_dim: {financial_input_dim}")
print(f"hidden_dim: {hidden_dim}")
print(f"memory_size: {memory_size}")

# Load config.json for comparison
with open('models/latest/config.json', 'r') as f:
    config = json.load(f)

print("\nConfig from config.json:")
print(f"input_dim: {config.get('input_dim', 'Not found')}")
print(f"hidden_dim: {config.get('hidden_dim', 'Not found')}")
print(f"memory_size: {config.get('memory_size', 'Not found')}")

# Show discrepancies
print("\nDiscrepancies:")
if input_dim != config.get('input_dim', input_dim):
    print(f"input_dim mismatch: checkpoint={input_dim}, config={config.get('input_dim', 'Not found')}")
if hidden_dim != config.get('hidden_dim', hidden_dim):
    print(f"hidden_dim mismatch: checkpoint={hidden_dim}, config={config.get('hidden_dim', 'Not found')}")
if memory_size != config.get('memory_size', memory_size):
    print(f"memory_size mismatch: checkpoint={memory_size}, config={config.get('memory_size', 'Not found')}")

print("\nImportant shapes from checkpoint:")
print(f"financial_encoder.encoder.0.weight: {checkpoint['financial_encoder.encoder.0.weight'].shape}")
print(f"temporal_encoder.lstm.weight_ih_l0: {checkpoint['temporal_encoder.lstm.weight_ih_l0'].shape}")
print(f"financial_memory.memory: {checkpoint['financial_memory.memory'].shape}")
print(f"market_state_sequence_predictor.predictor.weight: {checkpoint['market_state_sequence_predictor.predictor.weight'].shape}") 