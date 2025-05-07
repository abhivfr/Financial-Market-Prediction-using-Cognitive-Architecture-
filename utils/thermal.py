import torch
import psutil
import time
import numpy as np
from collections import deque

class ThermalGovernor:
    def __init__(self, temp_threshold=80, memory_threshold=0.9):
        super().__init__()  # Added call to parent constructor for robustness
        self.temp_threshold = temp_threshold
        self.memory_threshold = memory_threshold
        # Using deque with maxlen is good for history
        self.temperature_history = deque(maxlen=100)
        self.memory_history = deque(maxlen=100)
        self.throttle_count = 0
        self.last_check = time.time()
        self.cool_down_period = 30  # seconds

    def get_gpu_temp(self):
        # Check if CUDA is available AND if get_device_properties exists (for older torch versions)
        if torch.cuda.is_available():
            try:
                # Use the documented way to get temperature if available
                # Note: torch.cuda.temperature() is not a standard PyTorch function.
                # You might need to use a library like 'pynvml' for reliable temp reading.
                # Assuming 'torch.cuda.temperature()' is a custom function you added or mocked:
                if hasattr(torch.cuda, 'temperature'):
                    return torch.cuda.temperature()
                else:
                    # Fallback or indicate not implemented
                    # print("Warning: torch.cuda.temperature() not found. Cannot get GPU temp.")
                    return None
            except Exception as e:
                # print(f"Error getting GPU temperature: {e}")
                return None
        return None

    def get_memory_usage(self):
        if torch.cuda.is_available():
            try:
                allocated_bytes = torch.cuda.memory_allocated(0)
                total_bytes = torch.cuda.get_device_properties(0).total_memory
                return allocated_bytes / total_bytes  # Returns a ratio (0.0 to 1.0)
            except Exception as e:
                # print(f"Error getting GPU memory usage: {e}")
                return psutil.virtual_memory().percent / 100  # Fallback to CPU memory
        # If CUDA not available at all
        return psutil.virtual_memory().percent / 100  # CPU memory usage ratio

    def check(self):
        current_time = time.time()

        # Enforce minimum check interval
        if current_time - self.last_check < 1:  # Check every second at most
            return "continue"

        self.last_check = current_time

        # Get current metrics
        temp = self.get_gpu_temp()
        memory_usage = self.get_memory_usage()

        # Update history
        # Only append if temperature is successfully read (not None)
        if temp is not None:
            self.temperature_history.append(temp)
        # Always append memory usage (CPU or GPU)
        self.memory_history.append(memory_usage)

        # --- FIX APPLIED HERE ---
        # ✅ Guard against insufficient history before calculating trends or slicing
        if len(self.temperature_history) < 10:
            temp_trend = 0
            if temp is not None and temp > self.temp_threshold:
                self.throttle_count += 1
                return "throttle"
            if memory_usage > self.memory_threshold:
                self.throttle_count += 1
                return "throttle"
            return "continue"

        # --- Original trend calculation (now only runs if history >= 10) ---
        # Temperature trend calculation
        temp_history_slice = list(self.temperature_history)[-10:]  # Convert to list before slicing
        temp_trend = np.polyfit(
            range(len(temp_history_slice)),  # Ensure integer sequence index for polyfit
            temp_history_slice,
            1
        )[0]

        # Memory trend calculation (fixed with list conversion)
        if len(self.memory_history) > 10:
            mem_history_slice = list(self.memory_history)[-10:]  # Convert to list before slicing
            mem_trend = np.polyfit(
                range(len(mem_history_slice)),
                mem_history_slice,
                1
            )[0]
        else:
            mem_trend = 0

        # --- Original Decision Logic ---
        if temp is not None and temp > self.temp_threshold:
            self.throttle_count += 1
            return "throttle"
        if memory_usage > self.memory_threshold:
            self.throttle_count += 1
            return "throttle"
        if temp_trend > 2 or mem_trend > 0.05:
            self.throttle_count += 1
            return "warning"
        return "continue"

    # Rest of the ThermalGovernor class (get_status, reset) remains unchanged
    def get_status(self):
        return {
            'temperature': self.temperature_history[-1] if self.temperature_history else None,
            'memory_usage': self.memory_history[-1] if self.memory_history else None,
            'throttle_count': self.throttle_count,
            'status': self.check()
        }

    def reset(self):
        self.temperature_history.clear()
        self.memory_history.clear()
        self.throttle_count = 0
        self.last_check = time.time()
