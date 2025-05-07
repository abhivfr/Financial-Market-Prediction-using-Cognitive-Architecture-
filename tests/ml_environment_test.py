import torch
from src.utils.memory_guard import MemoryGuard

def test_pytorch_gpu():
    print("Checking PyTorch GPU availability:")
    if torch.cuda.is_available():
        gpu_index = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(gpu_index)
        total_memory = torch.cuda.get_device_properties(gpu_index).total_memory / (1024**3)
        print(f"PyTorch is using GPU: {gpu_name}")
        print(f"Total VRAM: {total_memory:.2f} GB")
        assert True
    else:
        print("PyTorch is NOT using GPU.")
        assert not torch.cuda.is_available()

def basic_memory_guard_test():
    if torch.cuda.is_available():
        memory_guard = MemoryGuard(max_memory=1.8e9) # Example limit
        tensor = torch.randn((500, 500, 500), device='cuda') # Smaller tensor
        print("Initial tensor memory allocated:", torch.cuda.memory_allocated())
        tensor = memory_guard.check(tensor)
        print("Memory allocated after check:", torch.cuda.memory_allocated())
        if tensor.device.type == 'cpu':
            print("MemoryGuard triggered and moved tensor to CPU.")
        else:
            print("MemoryGuard did not trigger (or VRAM not exceeded).")

if __name__ == "__main__":
    pytorch_gpu_available = test_pytorch_gpu()
    if pytorch_gpu_available:
        basic_memory_guard_test()
