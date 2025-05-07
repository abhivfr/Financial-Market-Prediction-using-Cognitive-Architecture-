#!/usr/bin/env python
# Direct call to train_progressive.py with hardcoded paths

import os
import sys
import subprocess

def main():
    # Hardcode the paths for testing
    train_path = "/home/ai-dev/AI-Consciousness1-main/data/enhanced_features_train.csv"
    val_path = "/home/ai-dev/AI-Consciousness1-main/data/enhanced_features_val.csv" 
    output_dir = "models/test_direct"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Build command with direct paths
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts", "train_progressive.py")
    
    # Use the same Python interpreter that was used to run this script
    python_exe = sys.executable
    print(f"Using Python interpreter: {python_exe}")
    
    # Create command with every argument on separate lines for clarity
    cmd = [
        python_exe,
        script_path,
        "--train_data", train_path,
        "--val_data", val_path,
        "--hidden_dim", "64",
        "--memory_size", "50",
        "--sequence_length", "20",
        "--batch_size", "32",
        "--learning_rate", "0.001",
        "--epochs", "2",  # Just use 2 epochs for quick testing
        "--output_dir", output_dir
    ]
    
    # Print the command for debugging
    print("Executing command:")
    print(" ".join(cmd))
    
    # Write command to a shell script for manual testing
    with open("run_training.sh", "w") as f:
        f.write("#!/bin/bash\n")
        f.write(" ".join(cmd))
    os.chmod("run_training.sh", 0o755)
    print("Wrote command to run_training.sh for manual testing")
    
    # Run the command and forward all output
    result = subprocess.run(cmd, capture_output=True, text=True)
    print("STDOUT:")
    print(result.stdout)
    print("STDERR:")
    print(result.stderr)
    print(f"Return code: {result.returncode}")
    
    # Try with subprocess.call for better error handling
    print("\nAttempting with subprocess.call:")
    return_code = subprocess.call(cmd)
    print(f"Return code: {return_code}")
    
    return result.returncode

if __name__ == "__main__":
    sys.exit(main()) 