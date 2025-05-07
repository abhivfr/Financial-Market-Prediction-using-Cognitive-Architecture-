import torch
from typing import Optional

class VRAMGuard:
    def __init__(self, max_usage: float = 1.8):
        """Monitor VRAM usage for MX250 (values in GB)"""
        self.max_bytes = max_usage * 1024**3  # Convert GB to bytes
        self.safety_margin = 0.1 * 1024**3     # 100MB safety buffer
        
    def check(self, context: Optional[str] = None) -> bool:
        """Raise error if VRAM usage exceeds safe threshold"""
        if not torch.cuda.is_available():
            return True
            
        allocated = torch.cuda.memory_allocated()
        if allocated > (self.max_bytes - self.safety_margin):
            error_msg = f"VRAM limit exceeded: {allocated/1024**3:.2f}GB > {self.max_bytes/1024**3:.2f}GB"
            if context:
                error_msg += f" during {context}"
            raise MemoryError(error_msg)
        return True
        
    @staticmethod
    def clear_cache():
        """Aggressive memory cleanup for MX250"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()