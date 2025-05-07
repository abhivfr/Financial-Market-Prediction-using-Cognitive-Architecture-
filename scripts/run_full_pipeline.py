#!/usr/bin/env python
# scripts/run_full_pipeline.py

import os
import subprocess
import argparse
from datetime import datetime

def run_command(cmd, description):
    """Run a command with proper logging"""
    print(f"\n==== {description} ====")
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=True)
    print(f"Completed with exit code: {result.returncode}")
    return result

def main():
    parser = argparse.ArgumentParser(description="Run full model training and evaluation pipeline")
    parser.add_argument("--iters", type=int, default=5000, help="Number of training iterations")
    parser.add_argument("--batch", type=int, default=32, help="Batch size")
    parser.add_argument("--seq_length", type=int, default=20, help="Sequence length")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    args = parser.parse_args()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("results", exist_ok=True)
    
    # Step 1: Train the baseline LSTM
    run_command([
        "python", "scripts/train_baseline.py",
        "--financial_train", "data/financial/train_data.csv",
        "--financial_val", "data/financial/validation_data.csv",
        "--iters", str(args.iters),
        "--batch", str(args.batch),
        "--seq_length", str(args.seq_length),
        "--lr", str(args.lr)
    ], "Training Baseline LSTM")
    
    # Step 2: Train the full cognitive model
    run_command([
        "python", "train.py",
        "--financial_train", "data/financial/train_data.csv",
        "--financial_val", "data/financial/validation_data.csv",
        "--iters", str(args.iters),
        "--batch", str(args.batch),
        "--seq_length", str(args.seq_length),
        "--lr", str(args.lr),
        "--verbose"
    ], "Training Cognitive Model")
    
    # Step 3: Run comparative evaluation
    baseline_model = f"models/baseline_lstm_{timestamp}.pth"
    cognitive_model = f"models/financial_consciousness_{timestamp}.pth"
    
    # Fallback to most recent models if exact timestamp matches not found
    if not os.path.exists(baseline_model):
        baseline_models = [f for f in os.listdir("models") if f.startswith("baseline_lstm_")]
        if baseline_models:
            baseline_model = os.path.join("models", sorted(baseline_models)[-1])
    
    if not os.path.exists(cognitive_model):
        cognitive_models = [f for f in os.listdir("models") if f.startswith("financial_consciousness_")]
        if cognitive_models:
            cognitive_model = os.path.join("models", sorted(cognitive_models)[-1])
    
    run_command([
        "python", "scripts/analyze_performance.py",
        "--cognitive", cognitive_model,
        "--baseline", baseline_model,
        "--test_data", "data/financial/test_data.csv",
        "--seq_length", str(args.seq_length),
        "--batch", str(args.batch),
        "--output", f"results/comparison_{timestamp}"
    ], "Comparative Analysis")
    
    print("\n==== Pipeline Complete ====")
    print(f"Results saved to: results/comparison_{timestamp}")

if __name__ == "__main__":
    main()
