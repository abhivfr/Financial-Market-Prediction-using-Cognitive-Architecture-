import torch
class TypeSafety:
    def __init__(self):
        self.fp16_fallbacks = 0

    def check(self, tensor):
        if tensor.dtype == torch.float16 and torch.isnan(tensor).any():
            self.fp16_fallbacks += 1
            return tensor.to(torch.float32)
        return tensor

    def report(self):
        print(f"FP16 fallbacks detected: {self.fp16_fallbacks}")
