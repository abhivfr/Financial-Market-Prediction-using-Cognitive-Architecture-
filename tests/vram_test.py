import torch
from tqdm import tqdm
from src.utils.memory_guard import MemoryGuard

def test_memory_guard():
    guard = MemoryGuard()
    tensor = torch.randn(512, 512, device='cuda')
    try:
        for _ in tqdm(range(1000)):
            tensor = tensor @ tensor
            tensor = guard.check(tensor)
    except RuntimeError as e:
        assert "CUDA out of memory" in str(e)

if __name__ == "__main__":
    test_memory_guard()
