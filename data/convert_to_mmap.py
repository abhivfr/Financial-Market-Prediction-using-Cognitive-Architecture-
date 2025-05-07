import os
import torch
import mmap
from pathlib import Path

def convert_to_mmap(input_path: str, output_path: str):
    """Convert PyTorch tensor to memory-mapped file"""
    tensor = torch.load(input_path)
    shape = tensor.shape
    dtype = tensor.dtype
    data_bytes = tensor.numpy().tobytes()
    data_size = len(data_bytes)
     
    # Create mmap file
    with open(output_path, 'wb+') as f:
        # Write header: shape + dtype
        header = f"{','.join(map(str, shape))}|{str(dtype)}"
        header_bytes = header.encode('utf-8') + b'\x00'
        f.write(header_bytes)
        header_size = f.tell()

        # Set the file size to accommodate header and data
        total_size = header_size + data_size
        f.truncate(total_size)

        # Memory-map the data
        mm = mmap.mmap(f.fileno(), 0)

        # Write data
        mm[header_size:total_size] = data_bytes
        mm.close()

def process_directory(input_dir: str, output_dir: str):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
     
    for fname in os.listdir(input_dir):
        if fname.endswith('.pt'):
            input_path = os.path.join(input_dir, fname)
            output_path = os.path.join(output_dir, fname.replace('.pt', '.mmap'))
            convert_to_mmap(input_path, output_path)
            print(f"Converted {fname} → {os.path.basename(output_path)}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    args = parser.parse_args()
     
    process_directory(args.input_dir, args.output_dir)
