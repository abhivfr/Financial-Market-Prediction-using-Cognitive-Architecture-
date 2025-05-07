from einops import rearrange
import torch.nn as nn
import torch

class CrossModalAttention(nn.Module):
    def __init__(self, dim=256, financial_dim=4):
        super().__init__()
        # Visual-temporal attention
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, 2*dim)
        
        # Financial attention
        self.financial_q = nn.Linear(financial_dim, dim)
        self.financial_kv = nn.Linear(financial_dim, 2*dim)
        
        # Fusion layer
        self.fc = nn.Linear(dim * 2, dim)

    def forward(self, visual, temporal, financial=None):
        # Original fusion
        q = self.q(temporal)
        visual = visual.unsqueeze(1)
        k, v = self.kv(visual).chunk(2, dim=-1)
        attn = torch.einsum('bd,bkd->bk', q, k).softmax(dim=-1)
        vis_temp = torch.einsum('bk,bkd->bd', attn, v)
        
        # Financial fusion
        if financial is not None:
            q_fin = self.financial_q(financial)
            financial = financial.unsqueeze(1)
            k_fin, v_fin = self.financial_kv(financial).chunk(2, dim=-1)
            attn_fin = torch.einsum('bd,bkd->bk', q_fin, k_fin).softmax(dim=-1)
            fin_attn = torch.einsum('bk,bkd->bd', attn_fin, v_fin)
            
            # Combine modalities
            combined = torch.cat([vis_temp, fin_attn], dim=-1)
            return self.fc(combined)
            
        return vis_temp