#!/usr/bin/env bash
#
# scripts/fix_cuda.sh — configure CUDA_VISIBLE_DEVICES and memory splits

# 1. Ensure the right GPU is visible
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

# 2. Reduce fragmentation for 2 GB VRAM
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# 3. Ensure device nodes exist (requires sudo once)
if [ ! -e /dev/nvidia0 ]; then
  sudo mknod -m 666 /dev/nvidia0 c 195 0
fi
if [ ! -e /dev/nvidiactl ]; then
  sudo mknod -m 666 /dev/nvidiactl c 195 255
fi

# 4. Quick sanity check
echo "Driver Version: $(nvidia-smi --query-gpu=driver_version --format=tsv,noheader)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"
