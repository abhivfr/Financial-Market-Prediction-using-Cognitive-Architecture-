import sys
import inspect
from src.arch.cognitive import CognitiveArchitecture

# Print the source code of the forward method
print("CognitiveArchitecture.forward method source:")
print(inspect.getsource(CognitiveArchitecture.forward))

# Create a minimal model to check return values
import torch

model = CognitiveArchitecture()
# Create dummy inputs
batch_size = 2
seq_len = 10
feature_dim = 4

dummy_financial_data = torch.randn(batch_size, feature_dim)
dummy_financial_seq = torch.randn(batch_size, seq_len, feature_dim)

# Try to print output info
try:
    outputs = model(financial_data=dummy_financial_data, financial_seq=dummy_financial_seq)
    print(f"Model returns {len(outputs)} values")
    for i, val in enumerate(outputs):
        if isinstance(val, torch.Tensor):
            print(f"  Output {i}: Tensor shape {val.shape}")
        else:
            print(f"  Output {i}: {type(val)}")
except Exception as e:
    print(f"Error during model inspection: {e}")
