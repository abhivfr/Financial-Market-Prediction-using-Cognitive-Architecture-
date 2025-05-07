import torch
from src.core.core import ConsciousnessCore

def test_recursion_depth():
    core = ConsciousnessCore().cuda() # Move ConsciousnessCore to GPU
    input = torch.randn(1, 256, requires_grad=True).cuda() # Move input tensor to GPU
    output = core(input)
    assert core.depth_counter == 5, "Depth constraint failed!"
