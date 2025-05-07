from src.core import ConsciousnessCore
from src.perception import VisualStream, CrossModalAttention
from src.core.nas import NASEnforcer  # Added import
import torch.nn as nn

class CognitiveArchitecture(nn.Module):
    def __init__(self):
        super().__init__()
        self.nas = NASEnforcer(input_dim=1024)  # Added NAS instantiation
        self.core = ConsciousnessCore()
        print(f"VisualStream class: {VisualStream}") # Keep this for debugging
        self.vision = VisualStream()
        self.fusion = CrossModalAttention()

    def forward(self, img, seq):
        visual_feat = self.nas(self.vision(img))  # Modified line with NAS
        temporal_feat = self.core(seq)
        return self.fusion(visual_feat, temporal_feat)
