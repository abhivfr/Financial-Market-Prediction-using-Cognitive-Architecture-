import torch
from src.arch.cognitive import CognitiveArchitecture

def main():
    model = CognitiveArchitecture()
    batch_size = 2
    seq_len = 5

    # Dummy image and sequence data
    img = torch.randn(batch_size, 3, 64, 64)
    seq = torch.randn(batch_size, seq_len, 128)

    # Dummy financial data (current and sequence)
    financial_data = torch.randn(batch_size, 3)  # [price, volume, time]
    financial_seq = torch.randn(batch_size, seq_len, 4)  # sequence of 4D vectors

    outputs = model(img, seq, financial_data=financial_data, financial_seq=financial_seq)
    print("Fusion output shape:", outputs[0].shape)
    print("Financial feature shape:", outputs[1].shape)
    print("Recalled memory shape:", outputs[2].shape)
    print("Attended vector shape:", outputs[3].shape)
    print("Attention weights shape:", outputs[4].shape)
    print("Market state shape:", None if outputs[5] is None else outputs[5].shape)

if __name__ == "__main__":
    main()