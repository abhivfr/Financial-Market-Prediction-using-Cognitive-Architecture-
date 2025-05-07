#!/usr/bin/env python
# Simple wrapper for train_progressive.py to ensure arguments are passed correctly

import os
import sys
import subprocess

def main():
    # Default paths for when neither environment variables nor arguments are provided
    default_train_path = "data/enhanced_features_train.csv"
    default_val_path = "data/enhanced_features_val.csv"
    default_output_dir = "models/cognitive"

    # Get the paths from environment variables or command line arguments
    # Make sure to provide fallbacks to defaults
    try:
        train_path = os.environ.get('TRAIN_PATH') or (sys.argv[1] if len(sys.argv) > 1 else default_train_path)
        val_path = os.environ.get('VAL_PATH') or (sys.argv[2] if len(sys.argv) > 2 else default_val_path)
        output_dir = os.environ.get('OUTPUT_DIR') or (sys.argv[3] if len(sys.argv) > 3 else default_output_dir)
    except IndexError:
        # If arguments aren't provided and no environment variables, use defaults
        train_path = default_train_path
        val_path = default_val_path
        output_dir = default_output_dir
    
    # Other arguments with defaults
    hidden_dim = os.environ.get('HIDDEN_DIM') or "64"
    memory_size = os.environ.get('MEMORY_SIZE') or "50"
    sequence_length = os.environ.get('SEQ_LENGTH') or "20"
    batch_size = os.environ.get('BATCH_SIZE') or "32"
    learning_rate = os.environ.get('LEARNING_RATE') or "0.001"
    epochs = os.environ.get('EPOCHS') or "50"
    
    # Print for debugging
    print(f"Train path: {train_path}")
    print(f"Val path: {val_path}")
    print(f"Output dir: {output_dir}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if files exist
    if not os.path.exists(train_path):
        print(f"ERROR: Training file does not exist at {train_path}")
        sys.exit(1)
    
    if not os.path.exists(val_path):
        print(f"ERROR: Validation file does not exist at {val_path}")
        sys.exit(1)
    
    # Build command
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts", "train_progressive.py")
    
    # Use the same Python interpreter that was used to run this script
    # This ensures we're using the interpreter from the virtual environment
    python_exe = sys.executable
    print(f"Using Python interpreter: {python_exe}")
    
    cmd = [
        python_exe,
        script_path,
        "--train_data", train_path,
        "--val_data", val_path,
        "--hidden_dim", hidden_dim,
        "--memory_size", memory_size,
        "--sequence_length", sequence_length,
        "--batch_size", batch_size,
        "--learning_rate", learning_rate,
        "--epochs", epochs,
        "--output_dir", output_dir
    ]
    
    # Print the command for debugging
    print("Executing command:")
    print(" ".join(cmd))
    
    # Run the command and forward all output
    result = subprocess.run(cmd)
    sys.exit(result.returncode)

if __name__ == "__main__":
    main() 