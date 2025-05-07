import subprocess
import os
import time
import shutil
import json
import argparse
from pathlib import Path

def log(message):
    """Log a message with timestamp"""
    print(f"[{time.strftime('%H:%M:%S')}] {message}")

def run_command(cmd, description):
    """Run a shell command and log output"""
    log(f"Running: {description}")
    log(f"Command: {' '.join(cmd)}")
    
    process = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           text=True, universal_newlines=True)
    
    if process.returncode == 0:
        log(f"✓ Success: {description}")
        print(f"Output: {process.stdout[:500]}..." if len(process.stdout) > 500 else f"Output: {process.stdout}")
        return True
    else:
        log(f"✗ Failed: {description}")
        print(f"Error: {process.stdout}")
        return False

def create_test_directories():
    """Create necessary directories for testing"""
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/splits", exist_ok=True)
    os.makedirs("models/baseline", exist_ok=True)
    os.makedirs("models/cognitive", exist_ok=True)
    os.makedirs("evaluation/latest", exist_ok=True)
    os.makedirs("monitoring/introspect", exist_ok=True)
    os.makedirs("monitoring/early_warning", exist_ok=True)
    os.makedirs("visualizations/flow", exist_ok=True)

def test_data_functionality():
    """Test data download and processing functions"""
    log("Testing data functionality...")
    
    # Download data
    success = run_command(
        ["python", "scripts/download_data.py",
         "--tickers", "SPY,QQQ", 
         "--start_date", "2020-01-01",
         "--end_date", "2023-01-01",
         "--output_dir", "data/raw",
         "--include_features",
         "--detect_regimes"],
        "Download financial data"
    )
    
    if not success:
        return False
    
    # Build dataset
    success = run_command(
        ["python", "scripts/build_dataset.py",
         "--input_dir", "data/raw",
         "--output_path", "data/combined_financial.csv",
         "--min_date", "2020-01-01",
         "--add_market_context",
         "--spy_file", "data/raw/SPY.csv"],
        "Build combined dataset"
    )
    
    if not success:
        return False
    
    # Split dataset
    success = run_command(
        ["python", "scripts/split_data.py",
         "--input_path", "data/combined_financial.csv",
         "--output_dir", "data/splits",
         "--split_mode", "regime",
         "--train_ratio", "0.7",
         "--val_ratio", "0.15",
         "--ensure_regime_coverage"],
        "Split dataset"
    )
    
    if not success:
        return False
    
    # Enhance features
    success = run_command(
        ["python", "scripts/enhance_features.py",
         "--input_path", "data/splits/test.csv",
         "--output_path", "data/enhanced_features.csv",
         "--add_technical",
         "--add_lagged",
         "--lag_periods", "3"],
        "Enhance features"
    )
    
    return success

def test_training_functionality():
    """Test model training functions"""
    log("Testing training functionality...")
    
    # Create simple config
    config = {
        "input_dim": 10,
        "hidden_dim": 32,
        "num_layers": 2,
        "dropout": 0.2,
        "memory_size": 50,
        "num_regimes": 3,
        "learning_rate": 0.001
    }
    
    config_path = "models/test_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f)
    
    # Train baseline model (with fewer epochs for testing)
    success = run_command(
        ["python", "scripts/train.py",
         "--model", "baseline",
         "--config", config_path,
         "--data_path", "data/splits/train.csv",
         "--output_dir", "models/baseline",
         "--epochs", "3",
         "--batch_size", "32"],
        "Train baseline model (mini)"
    )
    
    if not success:
        return False
    
    # Train cognitive model (with fewer epochs for testing)
    success = run_command(
        ["python", "scripts/train.py",
         "--model", "cognitive",
         "--config", config_path,
         "--data_path", "data/splits/train.csv",
         "--output_dir", "models/cognitive",
         "--epochs", "3",
         "--batch_size", "32"],
        "Train cognitive model (mini)"
    )
    
    return success

def test_evaluation_functionality():
    """Test model evaluation functions"""
    log("Testing evaluation functionality...")
    
    # Find model files
    baseline_models = list(Path("models/baseline").glob("*.pth"))
    cognitive_models = list(Path("models/cognitive").glob("*.pth"))
    
    if not baseline_models or not cognitive_models:
        log("✗ Cannot test evaluation - no model files found")
        return False
    
    baseline_model = str(baseline_models[0])
    cognitive_model = str(cognitive_models[0])
    
    # Standard evaluation
    success = run_command(
        ["python", "scripts/evaluate.py",
         "--model_path", cognitive_model,
         "--model_type", "cognitive",
         "--data_path", "data/splits/test.csv",
         "--output_dir", "evaluation/cognitive"],
        "Evaluate cognitive model"
    )
    
    if not success:
        return False
    
    # Regime evaluation
    success = run_command(
        ["python", "scripts/evaluate_by_regime.py",
         "--model_path", cognitive_model,
         "--model_type", "cognitive",
         "--data_path", "data/splits/test.csv",
         "--output_dir", "evaluation/regime",
         "--compare_with", baseline_model,
         "--compare_type", "baseline"],
        "Evaluate by regime with comparison"
    )
    
    if not success:
        return False
    
    # Stress test
    success = run_command(
        ["python", "scripts/stress_test.py",
         "--model_path", cognitive_model,
         "--model_type", "cognitive",
         "--data_path", "data/splits/test.csv",
         "--output_dir", "evaluation/stress",
         "--scenarios", "Market Crash,High Volatility"],
        "Run stress tests"
    )
    
    return success

def test_monitoring_functionality():
    """Test model monitoring functions"""
    log("Testing monitoring functionality...")
    
    # Find model files
    cognitive_models = list(Path("models/cognitive").glob("*.pth"))
    
    if not cognitive_models:
        log("✗ Cannot test monitoring - no model files found")
        return False
    
    cognitive_model = str(cognitive_models[0])
    
    # Introspection
    success = run_command(
        ["python", "scripts/introspect.py",
         "--model_path", cognitive_model,
         "--data_path", "data/splits/test.csv",
         "--output_dir", "monitoring/introspect",
         "--samples", "5"],
        "Run model introspection"
    )
    
    if not success:
        return False
    
    # Flow visualization
    success = run_command(
        ["python", "scripts/visualize_flow.py",
         "--model_path", cognitive_model,
         "--data_path", "data/splits/test.csv",
         "--output_dir", "visualizations/flow",
         "--samples", "3"],
        "Generate flow visualization"
    )
    
    return success

def main():
    parser = argparse.ArgumentParser(description="Test Cognitive Architecture App functionality")
    parser.add_argument("--data", action="store_true", help="Test data functionality")
    parser.add_argument("--train", action="store_true", help="Test training functionality")
    parser.add_argument("--evaluate", action="store_true", help="Test evaluation functionality")
    parser.add_argument("--monitor", action="store_true", help="Test monitoring functionality")
    parser.add_argument("--all", action="store_true", help="Test all functionality")
    
    args = parser.parse_args()
    
    # If no args, test all
    if not (args.data or args.train or args.evaluate or args.monitor or args.all):
        args.all = True
    
    log("Starting Cognitive Architecture App functionality tests")
    create_test_directories()
    
    results = {}
    
    if args.data or args.all:
        results["Data"] = test_data_functionality()
    
    if args.train or args.all:
        results["Training"] = test_training_functionality()
    
    if args.evaluate or args.all:
        results["Evaluation"] = test_evaluation_functionality()
    
    if args.monitor or args.all:
        results["Monitoring"] = test_monitoring_functionality()
    
    # Print summary
    log("Test Results Summary:")
    for test, result in results.items():
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{test}: {status}")
    
    overall = all(results.values())
    print(f"\nOverall: {'✓ ALL TESTS PASSED' if overall else '✗ SOME TESTS FAILED'}")
    
    # Test app launch (brief)
    if overall:
        log("All tests passed. Testing app launch (will exit after 5 seconds)...")
        app_process = subprocess.Popen(["python", "app.py"])
        time.sleep(5)  # Let app start
        app_process.terminate()  # Kill the app
        log("App launch test complete")
    
    return 0 if overall else 1

if __name__ == "__main__":
    exit(main())
