import torch
import torch.nn as nn
import torch.nn.functional as F

class FinancialCrossAttention(nn.Module):
    def __init__(self, dim=4, heads=1):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.heads = heads
        self.scale = dim ** -0.5

    def forward(self, x, context=None):
        # x: (batch, dim) - the current 4D financial vector
        # context: (batch, num_context, dim) - optional, e.g., recalled memory
        if context is None:
            context = x.unsqueeze(1)  # (batch, 1, dim)
        q = self.query(x).unsqueeze(1)  # (batch, 1, dim)
        k = self.key(context)           # (batch, num_context, dim)
        v = self.value(context)         # (batch, num_context, dim)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (batch, 1, num_context)
        attn_weights = F.softmax(attn_scores, dim=-1)                    # (batch, 1, num_context)
        attended = torch.matmul(attn_weights, v)                         # (batch, 1, dim)
        return attended.squeeze(1), attn_weights.squeeze(1)