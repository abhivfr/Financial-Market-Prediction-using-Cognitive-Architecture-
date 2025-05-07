import mmap
import numpy as np
import torch

class TieredLoader:
    def __init__(self, ssd_path: str, hdd_path: str):
        self.ssd_path = ssd_path
        self.hdd_path = hdd_path

    def load_mmap(self, path: str) -> torch.Tensor:
        dtype_map = {
            'torch.float32': np.float32,
            'torch.float64': np.float64,
            'torch.float16': np.float16,
            'torch.uint8': np.uint8,
            'torch.int8': np.int8,
            'torch.int16': np.int16,
            'torch.int32': np.int32,
            'torch.int64': np.int64,
            'torch.bool': np.bool_
        }
        torch_dtype_map = {
            'torch.float32': torch.float32,
            'torch.float64': torch.float64,
            'torch.float16': torch.float16,
            'torch.uint8': torch.uint8,
            'torch.int8': torch.int8,
            'torch.int16': torch.int16,
            'torch.int32': torch.int32,
            'torch.int64': torch.int64,
            'torch.bool': torch.bool
        }
        with open(path, 'rb') as f:
            # Read header
            header_bytes = b''
            while True:
                byte = f.read(1)
                if byte == b'\x00':
                    break
                header_bytes += byte
            header_str = header_bytes.decode()
            print(f"Header: {header_str}")
            shape_str, dtype_str = header_str.split('|')
            shape = tuple(map(int, shape_str.split(',')))
            np_dtype = dtype_map.get(dtype_str)
            torch_dtype = torch_dtype_map.get(dtype_str)
            if np_dtype is None:
                raise ValueError(f"Unknown dtype: {dtype_str}")

            header_size = len(header_bytes) + 1 # +1 for the null terminator

            # Memory-map the data
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            array = np.frombuffer(mm, dtype=np_dtype, offset=header_size).reshape(shape)
            return torch.from_numpy(array.copy())

    def fetch(self, idx: int) -> torch.Tensor:
        if idx < 5:  # First 5 batches on SSD
            return self.load_mmap(f"{self.ssd_path}/batch_{idx}.mmap")
        else:  # Rest on HDD
            return self.load_mmap(f"{self.hdd_path}/batch_{idx}.mmap")
