#!/bin/bash

# Create a report file with timestamp
REPORT_FILE="dependency_report_$(date +%Y%m%d_%H%M%S).txt"
echo "Generating dependency report: $REPORT_FILE"

# 1. Get Python package versions from tf_venv
echo -e "\n===== Python Packages in tf_venv =====\n" > $REPORT_FILE
source /home/ai-dev/tf_venv/bin/activate
pip freeze >> $REPORT_FILE
deactivate

# 2. Get system package versions
echo -e "\n===== System Packages =====\n" >> $REPORT_FILE
{
    # Python-related
    rpm -qa | grep -i python3
    
    # Development tools
    rpm -qa | grep -i gcc
    rpm -qa | grep -i make
    rpm -qa | grep -i cmake
    
    # Math/ML libraries
    rpm -qa | grep -i openblas
    rpm -qa | grep -i lapack
} >> $REPORT_FILE

# 3. Get NVIDIA/CUDA information
echo -e "\n===== GPU/CUDA Information =====\n" >> $REPORT_FILE
{
    # NVIDIA drivers
    nvidia-smi 2>/dev/null || echo "NVIDIA drivers not detected"
    
    # CUDA version
    nvcc --version 2>/dev/null || echo "CUDA not detected"
    
    # cuDNN version
    cat /usr/include/cudnn_version.h 2>/dev/null || \
    cat /usr/local/cuda/include/cudnn_version.h 2>/dev/null || \
    echo "cuDNN not detected"
} >> $REPORT_FILE

# 4. Get environment info
echo -e "\n===== System Information =====\n" >> $REPORT_FILE
{
    cat /etc/fedora-release
    uname -a
    lscpu
    free -h
} >> $REPORT_FILE

echo "Report generated successfully: $REPORT_FILE"
