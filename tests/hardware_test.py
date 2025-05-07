import torch
from src.arch.cognitive import CognitiveArchitecture
from src.monitoring.introspect import Introspection
from src.utils.vram_guard import VRAMGuard

def test_memory_pressure():
    VRAMGuard().check("test setup")
    arch = CognitiveArchitecture().cuda()
    introspect = Introspection()

    for _ in range(100):
        VRAMGuard().check("loop iteration")
        img = torch.randn(1,3,224,224).cuda()
        seq = torch.randn(1,256, requires_grad=True).cuda()
        out = arch(img, seq)
        introspect.log(arch, torch.cuda.memory_allocated())

        assert allocated() < 1.8e9 * 0.8, "MX250 VRAM limit exceeded"
