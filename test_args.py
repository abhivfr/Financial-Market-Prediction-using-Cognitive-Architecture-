#!/usr/bin/env python
# test_args.py - Simple script to verify argument passing

import os
import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description="Test argument passing")
    
    # Required arguments
    parser.add_argument("--train_data", required=True, help="Path to training data")
    parser.add_argument("--val_data", required=True, help="Path to validation data")
    
    # Optional arguments
    parser.add_argument("--output_dir", default="models/test", help="Output directory")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden dimension")
    
    # Parse arguments
    print("Parsing arguments...")
    args = parser.parse_args()
    
    # Print all arguments
    print("\nReceived Arguments:")
    print(f"  --train_data: {args.train_data}")
    print(f"  --val_data: {args.val_data}")
    print(f"  --output_dir: {args.output_dir}")
    print(f"  --hidden_dim: {args.hidden_dim}")
    
    # Check if files exist
    print("\nFile Existence:")
    for name, path in [("Train data", args.train_data), ("Val data", args.val_data)]:
        exists = os.path.exists(path)
        print(f"  {name} ({path}): {'EXISTS' if exists else 'MISSING'}")
    
    # Print environment info
    print("\nEnvironment Info:")
    print(f"  Python executable: {sys.executable}")
    print(f"  Python version: {sys.version}")
    print(f"  Working directory: {os.getcwd()}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 