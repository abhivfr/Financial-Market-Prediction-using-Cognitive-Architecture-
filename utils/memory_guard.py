import torch
import time
import psutil
import numpy as np
from collections import deque

class MemoryGuard:
    def __init__(self, max_memory=None, warning_threshold=0.7, critical_threshold=0.9):
        if torch.cuda.is_available():
            total_gpu_memory = torch.cuda.get_device_properties(0).total_memory
            self.max_memory = max_memory or total_gpu_memory
            self.warning_threshold = int(total_gpu_memory * warning_threshold)
            self.critical_threshold = int(total_gpu_memory * critical_threshold)
        else:
            self.max_memory = max_memory or 8e9  # 8GB default
            self.warning_threshold = int(self.max_memory * warning_threshold)
            self.critical_threshold = int(self.max_memory * critical_threshold)
            
        self.memory_history = deque(maxlen=1000)
        self.alert_history = deque(maxlen=100)
        self.last_cleanup = time.time()
        self.cleanup_interval = 300  # 5 minutes
        
    def check_tensor(self, tensor):
        current_usage = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        self.memory_history.append(current_usage)
        
        # Calculate memory trend
        if len(self.memory_history) > 10:
            recent_usage = np.array(list(self.memory_history)[-10:])
            trend = np.mean(np.diff(recent_usage))
        else:
            trend = 0
            
        # Check system memory
        system_memory = psutil.virtual_memory()
        if system_memory.percent > 90:
            self._handle_critical_memory(tensor)
            return False
            
        # Handle different memory scenarios
        if current_usage > self.critical_threshold:
            self._handle_critical_memory(tensor)
            return False
        elif current_usage > self.warning_threshold or trend > 1e8:  # Rapid increase
            self._handle_warning_memory()
            return True
            
        return True
        
    def _handle_critical_memory(self, tensor):
        self.alert_history.append({
            'timestamp': time.time(),
            'usage': torch.cuda.memory_allocated() if torch.cuda.is_available() else 0,
            'type': 'critical'
        })
        
        # Emergency cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # Compress tensor if possible
        if hasattr(tensor, 'half'):
            return tensor.half()
        return tensor
        
    def _handle_warning_memory(self):
        self.alert_history.append({
            'timestamp': time.time(),
            'usage': torch.cuda.memory_allocated() if torch.cuda.is_available() else 0,
            'type': 'warning'
        })
        
        current_time = time.time()
        if current_time - self.last_cleanup > self.cleanup_interval:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.last_cleanup = current_time
            
    def get_status(self):
        current_usage = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        return {
            'current_usage': current_usage,
            'warning_threshold': self.warning_threshold,
            'critical_threshold': self.critical_threshold,
            'alert_count': len(self.alert_history),
            'memory_trend': np.mean(np.diff(list(self.memory_history)[-10:])) if len(self.memory_history) > 10 else 0
        }
        
    def reset(self):
        self.memory_history.clear()
        self.alert_history.clear()
        self.last_cleanup = time.time()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()