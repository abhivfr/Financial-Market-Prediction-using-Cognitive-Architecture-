#!/usr/bin/env python
# debug_train.py - Debugging script to identify issues with train_progressive.py

import os
import sys
import subprocess

def main():
    # Hardcode the paths for debugging
    train_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "enhanced_features_train.csv")
    val_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "enhanced_features_val.csv")
    output_dir = "models/debug_out"
    
    # Print some debug info
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version}")
    print(f"Current directory: {os.getcwd()}")
    print(f"Script location: {os.path.abspath(__file__)}")
    
    # Check if files exist and print their absolute paths
    print(f"\nChecking file existence:")
    for path in [train_path, val_path]:
        exists = os.path.exists(path)
        abs_path = os.path.abspath(path)
        print(f"  {abs_path}: {'EXISTS' if exists else 'MISSING'}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"Created output directory: {output_dir}")
    
    # Build command with explicit paths
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts", "train_progressive.py")
    
    # Make sure paths are absolute
    train_path = os.path.abspath(train_path)
    val_path = os.path.abspath(val_path)
    output_dir = os.path.abspath(output_dir)
    script_path = os.path.abspath(script_path)
    
    # Check if script exists
    if not os.path.exists(script_path):
        print(f"ERROR: Script not found at {script_path}")
        return 1
    
    cmd = [
        sys.executable,
        script_path,
        "--train_data", train_path,
        "--val_data", val_path,
        "--hidden_dim", "64",
        "--memory_size", "50",
        "--sequence_length", "20",
        "--batch_size", "32",
        "--learning_rate", "0.001",
        "--epochs", "2",
        "--output_dir", output_dir
    ]
    
    # Print the exact command for debugging
    print("\nExecuting command:")
    print(" ".join(f'"{arg}"' if ' ' in arg else arg for arg in cmd))
    
    # Try different methods of calling the script
    print("\nMethod 1: Using subprocess.run directly")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print("STDOUT:")
    print(result.stdout)
    print("STDERR:")
    print(result.stderr)
    print(f"Return code: {result.returncode}")
    
    # Try with subprocess.call for better error handling
    print("\nMethod 2: Using subprocess.call")
    return_code = subprocess.call(cmd)
    print(f"Return code: {return_code}")
    
    # Try with os.system as a fallback
    print("\nMethod 3: Using os.system")
    # Escape spaces in paths for shell
    cmd_str = " ".join(f'"{arg}"' if ' ' in arg else arg for arg in cmd)
    ret = os.system(cmd_str)
    print(f"Return code: {ret}")
    
    return result.returncode

if __name__ == "__main__":
    sys.exit(main()) 