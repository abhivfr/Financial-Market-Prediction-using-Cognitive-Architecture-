import torch
import torch.nn as nn

class FourDSSM(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=32, state_dim=4, num_layers=1):
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.state_proj = nn.Linear(hidden_dim, state_dim)

    def forward(self, x, h0=None):
        # x: (batch, seq_len, input_dim)
        out, hn = self.rnn(x, h0)
        state = self.state_proj(out)  # (batch, seq_len, state_dim)
        return state, hn