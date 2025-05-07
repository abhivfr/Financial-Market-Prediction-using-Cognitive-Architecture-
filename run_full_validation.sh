#!/bin/bash
# Add environment setup for Fedora
export PYTHONPATH="/home/ai-dev/AI-Consciousness1-main:$PYTHONPATH"
# run_full_validation.sh - Complete workflow for training and validating the AI Consciousness model

# Stop on errors
set -e

echo "===== AI Consciousness Model - Full Validation Workflow ====="
echo "Started at $(date)"
echo

# Set Python path to find custom modules
export PYTHONPATH=$(pwd):$PYTHONPATH
echo "Set PYTHONPATH: $PYTHONPATH"
echo

# Step 1: Create prepare_environment.py if it doesn't exist
if [ ! -f "scripts/prepare_environment.py" ]; then
  echo "Step 1: Creating prepare_environment.py script..."
  cat > scripts/prepare_environment.py << 'EOF'
#!/usr/bin/env python
# scripts/prepare_environment.py

import os
import shutil
from datetime import datetime
import argparse
import glob

def ensure_dir(directory):
    """Ensure a directory exists, create if not"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")
    else:
        print(f"Directory already exists: {directory}")

def archive_files(source_dir, target_dir, file_pattern="*.pth"):
    """Move files from source_dir to target_dir if they match pattern"""
    # Ensure source and target directories exist
    if not os.path.exists(source_dir):
        print(f"Source directory does not exist: {source_dir}")
        return 0

    ensure_dir(target_dir)

    # Find files matching pattern
    pattern = os.path.join(source_dir, file_pattern)
    files = glob.glob(pattern)

    # Archive files
    count = 0
    for file in files:
        filename = os.path.basename(file)
        target_file = os.path.join(target_dir, filename)
        try:
            shutil.move(file, target_file)
            print(f"Archived: {filename}")
            count += 1
        except Exception as e:
            print(f"Error archiving {filename}: {e}")

    return count

def prepare_environment(archive=True):
    """Prepare environment for training and validation"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create archive directories
    archive_base = "archive"
    ensure_dir(archive_base)
    ensure_dir(os.path.join(archive_base, "models"))
    ensure_dir(os.path.join(archive_base, "checkpoints"))
    ensure_dir(os.path.join(archive_base, "logs"))

    # Create validation directories
    validation_base = "validation"
    ensure_dir(validation_base)
    ensure_dir(os.path.join(validation_base, "baselines"))
    ensure_dir(os.path.join(validation_base, "results"))
    ensure_dir(os.path.join(validation_base, "plots"))

    # Archive existing models and logs if requested
    archived_count = 0
    if archive:
        print("\nArchiving existing models and logs...")
        archived_count += archive_files("models", os.path.join(archive_base, "models"))
        archived_count += archive_files("checkpoints", os.path.join(archive_base, "checkpoints"))
        archived_count += archive_files(".", os.path.join(archive_base, "logs"), "train_output*.txt")
        print(f"Archived {archived_count} files in total")

    # Ensure model directories exist (may have been created above)
    ensure_dir("models")
    ensure_dir("checkpoints")
    ensure_dir("logs")

    print(f"\nEnvironment prepared at {timestamp}")
    print(f"Archived {archived_count} files")
    print("Ready for training and validation")

    return {
        "timestamp": timestamp,
        "archived_count": archived_count
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare environment for training and validation")
    parser.add_argument("--no-archive", action="store_true", help="Skip archiving existing models and logs")
    args = parser.parse_args()

    prepare_environment(archive=not args.no_archive)
EOF
  chmod +x scripts/prepare_environment.py
  echo "prepare_environment.py created."
fi

# Step 2: Create generate_report.py if it doesn't exist
if [ ! -f "scripts/generate_report.py" ]; then
  echo "Step 2: Creating generate_report.py script..."
  cat > scripts/generate_report.py << 'EOF'
#!/usr/bin/env python
# scripts/generate_report.py

import argparse
import json
import os
import numpy as np
from datetime import datetime

def load_json(path):
    """Load JSON file safely with error handling"""
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: File not found - {path}")
        return {}
    except json.JSONDecodeError:
        print(f"Warning: Invalid JSON format in file - {path}")
        return {}
    except Exception as e:
        print(f"Warning: Error loading JSON from {path}: {e}")
        return {}

def generate_report(walk_forward_path, backtest_path, comparison_path, output_path):
    """Generate comprehensive validation report in markdown format"""
    # Load result data
    walk_forward = load_json(walk_forward_path)
    backtest = load_json(backtest_path)
    comparison = load_json(comparison_path) if os.path.exists(comparison_path) else {}

    # Start markdown report
    report = f"""# Financial Consciousness Model Validation Report
Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 1. Model Performance Comparison

"""

    # Add comparison results if available
    if comparison:
        cognitive_metrics = comparison.get('cognitive_metrics', {})
        baseline_metrics = comparison.get('baseline_metrics', {})

        report += "### Comparison with Baseline LSTM\n\n"
        report += "| Metric | Baseline | Consciousness | Improvement |\n"
        report += "|--------|----------|---------------|-------------|\n"

        for metric in ['price_accuracy', 'volume_correlation', 'returns_stability', 'volatility_prediction']:
            baseline_val = baseline_metrics.get(metric, 0)
            cognitive_val = cognitive_metrics.get(metric, 0)
            # Avoid division by zero or small numbers for percentage calculation
            improvement = ((cognitive_val - baseline_val) / (abs(baseline_val) + 1e-8)) * 100

            report += f"| {metric.replace('_', ' ').title()} | {baseline_val:.4f} | {cognitive_val:.4f} | {improvement:.2f}% |\n"
    else:
        report += "### Comparison Data Not Available\n\n"
        report += "No comparison data was found. Please ensure the comparative analysis has been run.\n\n"

    # Add walk-forward validation results
    report += "\n\n## 2. Walk-Forward Validation\n\n"

    if isinstance(walk_forward, list) and len(walk_forward) > 0:
        # Calculate average metrics
        avg_metrics = {
            'price_accuracy': np.mean([r.get('metrics', {}).get('price_accuracy', 0) for r in walk_forward]),
            'volume_correlation': np.mean([r.get('metrics', {}).get('volume_correlation', 0) for r in walk_forward]),
            'returns_stability': np.mean([r.get('metrics', {}).get('returns_stability', 0) for r in walk_forward]),
            'volatility_prediction': np.mean([r.get('metrics', {}).get('volatility_prediction', 0) for r in walk_forward])
        }

        report += "### Overall Metrics\n\n"
        for metric, value in avg_metrics.items():
            report += f"- **{metric.replace('_', ' ').title()}**: {value:.4f}\n"

        report += "\n### Window Results\n\n"
        report += "| Window | Price Accuracy | Volume Correlation | Returns Stability | Volatility Prediction |\n"
        report += "|--------|---------------|-------------------|------------------|----------------------|\n"

        for i, window in enumerate(walk_forward):
            metrics = window.get('metrics', {})
            # Safely get window start/end, use index if not available
            window_start = window.get('window_start', i * 41) # Assuming step 41 if not found
            window_end = window.get('window_end', window_start + 252) # Assuming window 252 if not found
            window_desc = f"{window_start}-{window_end}"


            report += f"| {window_desc} | {metrics.get('price_accuracy', 0):.4f} | {metrics.get('volume_correlation', 0):.4f} | "
            report += f"{metrics.get('returns_stability', 0):.4f} | {metrics.get('volatility_prediction', 0):.4f} |\n"
    else:
        report += "### Walk-Forward Data Not Available\n\n"
        report += "No walk-forward validation data was found. Please ensure the walk-forward validation has been run.\n\n"
    
    # Add backtest results
    report += "\n\n## 3. Trading Strategy Backtest\n\n"
    
    if backtest and 'metrics' in backtest:
        metrics = backtest['metrics']
        report += f"- **Total Return**: {metrics.get('total_return', 0):.2%}\n"
        report += f"- **Sharpe Ratio**: {metrics.get('sharpe_ratio', 0):.4f}\n"
        report += f"- **Maximum Drawdown**: {metrics.get('max_drawdown', 0):.2%}\n"
        report += f"- **Win Rate**: {metrics.get('win_rate', 0):.2%}\n"
        
        # Add signal summary if available
        if 'signals_summary' in backtest:
            signal_summary = backtest['signals_summary']
            report += "\n### Signal Statistics\n\n"
            report += "| Stat | Signal | Confidence | Risk Score |\n"
            report += "|------|--------|------------|------------|\n"
            
            try:
                for stat in ['mean', 'std', 'min', '25%', '50%', '75%', 'max']:
                    # Safely get values, default to 0 or handle non-numeric
                    signal_val = signal_summary.get('signal', {}).get(stat, 0)
                    conf_val = signal_summary.get('confidence', {}).get(stat, 0)
                    risk_val = signal_summary.get('risk_score', {}).get(stat, 0)
                    
                    # Ensure values are numeric before formatting
                    signal_val = signal_val if isinstance(signal_val, (int, float)) else 0
                    conf_val = conf_val if isinstance(conf_val, (int, float)) else 0
                    risk_val = risk_val if isinstance(risk_val, (int, float)) else 0
                    
                    report += f"| {stat} | {signal_val:.4f} | {conf_val:.4f} | {risk_val:.4f} |\n"
            except Exception as e:
                report += f"\nError processing signal summary: {e}\n"
    else:
        report += "### Backtest Data Not Available\n\n"
        report += "No backtest data was found. Please ensure the trading backtest has been run.\n\n"
    
    # Add conclusion
    report += "\n\n## 4. Conclusion\n\n"
    
    # Calculate overall assessment
    comparison_improvement = 0
    assessment = "Unable to determine comparative performance" # Default assessment
    if comparison and 'cognitive_metrics' in comparison and 'baseline_metrics' in comparison:
        baseline_metrics = comparison.get('baseline_metrics', {})
        cognitive_metrics = comparison.get('cognitive_metrics', {})
        
        improvements = []
        metrics_to_compare = ['price_accuracy', 'volume_correlation', 'returns_stability', 'volatility_prediction']
        valid_improvements = 0
        for metric in metrics_to_compare:
            baseline_val = baseline_metrics.get(metric, None) # Use None to check if key exists
            cognitive_val = cognitive_metrics.get(metric, None)

            if baseline_val is not None and cognitive_val is not None:
                 # Avoid division by zero, use relative diff if baseline is zero or near zero
                 if abs(baseline_val) < 1e-8:
                      # If baseline is zero, improvement is the cognitive value itself (if positive)
                      # Or infinite if cognitive is non-zero
                      # Let's use absolute difference or a large number to avoid confusion
                      if abs(cognitive_val) > 1e-8:
                            improvements.append(1.0) # Represents significant improvement if baseline was zero
                      else:
                            improvements.append(0.0) # No improvement if both are zero/near zero
                 else:
                    improvements.append((cognitive_val - baseline_val) / abs(baseline_val))
                 valid_improvements += 1


        if valid_improvements > 0:
            comparison_improvement = np.mean(improvements) * 100

             # Qualitative assessment based on average relative improvement
            if comparison_improvement > 20:
                assessment = "significantly outperforms"
            elif comparison_improvement > 5:
                assessment = "moderately outperforms"
            elif comparison_improvement > -5:
                assessment = "performs similarly to"
            else:
                assessment = "underperforms compared to"
        else:
             assessment = "Unable to determine comparative performance (no valid metrics)"


        report += f"The Financial Consciousness model {assessment} baseline LSTM models "
        if valid_improvements > 0:
             report += f"with an average improvement of {comparison_improvement:.1f}% across compared metrics.\n\n"
        else:
             report += "due to missing or invalid baseline/cognitive metrics.\n\n"


    else:
        report += "Unable to determine comparative performance due to missing data.\n\n"

    if backtest and 'metrics' in backtest:
        sharpe_ratio = backtest['metrics'].get('sharpe_ratio', -np.inf) # Default to -inf if not found
        if sharpe_ratio > 1.0:
            report += "The trading strategy shows promising results with a Sharpe ratio > 1.0.\n\n"
        elif sharpe_ratio > 0:
             report += "The trading strategy shows a positive Sharpe ratio, indicating some risk-adjusted return.\n\n"
        else:
            report += "The trading strategy requires further refinement as the Sharpe ratio is not positive.\n\n"

        if 'max_drawdown' in backtest['metrics']:
            max_drawdown = backtest['metrics']['max_drawdown']
            report += f"The maximum drawdown was {max_drawdown:.2%}.\n\n"


    report += "### Next Steps\n\n"
    report += "- Consider additional feature engineering.\n"
    report += "- Further optimize hyperparameters, especially learning rates and memory parameters.\n"
    report += "- Test the model on additional financial instruments.\n"
    report += "- Implement position sizing based on model confidence or risk scores.\n"
    report += "- Explore reinforcement learning for strategy enhancement.\n"
    report += "- Evaluate performance over different market regimes.\n"

    # Save report
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(report)
        print(f"Report generated and saved to {output_path}")
    except Exception as e:
        print(f"Error saving report to {output_path}: {e}")
        print("\nHere's the report contents:")
        print(report)

    return report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate validation report")
    parser.add_argument("--walk_forward", required=True, help="Path to walk-forward results JSON")
    parser.add_argument("--backtest", required=True, help="Path to backtest results JSON")
    parser.add_argument("--comparison", required=False, help="Path to model comparison results JSON")
    parser.add_argument("--output", default="validation/final_report.md", help="Output report path")
    args = parser.parse_args()

    generate_report(args.walk_forward, args.backtest, args.comparison, args.output)
EOF
  chmod +x scripts/generate_report.py
  echo "generate_report.py created."
fi


# Step 3: Prepare Environment
echo "Step 1: Preparing environment..."
python scripts/prepare_environment.py
# Check if prepare_environment.py ran successfully
if [ $? -ne 0 ]; then
    echo "Error during environment preparation. Aborting."
    exit 1
fi
echo "Environment preparation complete."
echo

# Step 4: Data Preparation
echo "Step 2: Preparing data..."
# This script splits the data and creates validation_data.csv and test_data.csv
python split_data.py \
  --input data/financial/AAPL_historical_data_20200420_to_20250419.csv \
  --train 0.7 --val 0.15 --test 0.15 --mode time \
  --output data/financial
# Check if split_data.py ran successfully
if [ $? -ne 0 ]; then
    echo "Error during data splitting. Aborting."
    exit 1
fi
echo "Data preparation complete."
echo

# --- NEW Step: Create 6-column validation data for training/evaluation ---
# The internal validation step in train.py and train_baseline.py seems to expect exactly 6 columns.
# We create a dedicated 6-column file from the validation_data.csv generated above.
echo "Step 2.5: Creating 6-column validation data..."
# Using awk to select the first 6 columns including the header
# Columns: 1-timestamp, 2-price, 3-volume, 4-returns, 5-log_returns, 6-volatility
awk -F, '{print $1","$2","$3","$4","$5","$6}' OFS=, data/financial/validation_data.csv > data/financial/validation_data_6cols.csv
# Check if awk command ran successfully
if [ $? -ne 0 ]; then
    echo "Error creating 6-column validation data. Aborting."
    exit 1
fi
echo "6-column validation data created: data/financial/validation_data_6cols.csv"
echo

# Step 5: Train Baseline Model
echo "Step 3: Training baseline model..."
# Use the 6-column validation data for internal validation during training
python scripts/train_baseline.py \
  --financial_train data/financial/train_data.csv \
  --financial_val data/financial/validation_data_6cols.csv \
  --iters 2000 --batch 16 --seq_length 20 --lr 1e-5 --save_interval 200
# Check if train_baseline.py ran successfully
if [ $? -ne 0 ]; then
    echo "Error during baseline model training. Aborting."
    exit 1
fi
echo "Baseline model training complete."
echo

# Step 6: Train Cognitive Model
echo "Step 4: Training cognitive architecture..."
# Use the 6-column validation data for internal validation during training
python train.py \
  --financial_train data/financial/train_data.csv \
  --financial_val data/financial/validation_data_6cols.csv \
  --iters 2000 --batch 8 --seq_length 20 --lr 5e-6 \
  --eval_interval 100 --save_interval 200 --verbose
# Check if train.py ran successfully
if [ $? -ne 0 ]; then
    echo "Error during cognitive model training. Aborting."
    exit 1
fi
echo "Cognitive model training complete."
echo

# Step 7: Get latest model paths
echo "Step 5: Locating latest models..."
# Ensure models directory exists before listing
mkdir -p models # Use mkdir -p for safety

# Use 2>/dev/null to suppress errors if no files match before check
COGNITIVE_MODEL=$(ls -t models/financial_consciousness_*.pth 2>/dev/null | head -1 || echo "")
BASELINE_MODEL=$(ls -t models/baseline_lstm_*.pth 2>/dev/null | head -1 || echo "")

# Check if models exist and set flags
COGNITIVE_MODEL_FOUND=false
if [ -n "$COGNITIVE_MODEL" ]; then
    echo "Found cognitive model: $COGNITIVE_MODEL"
    COGNITIVE_MODEL_FOUND=true
else
    echo "Error: No cognitive model found (looked for models/financial_consciousness_*.pth). Aborting."
    exit 1 # Abort if cognitive model is missing as it's needed for all evaluation steps
fi

BASELINE_MODEL_FOUND=false
if [ -n "$BASELINE_MODEL" ]; then
     echo "Found baseline model: $BASELINE_MODEL"
     BASELINE_MODEL_FOUND=true
else
    echo "Warning: No baseline model found (looked for models/baseline_lstm_*.pth). Comparative analysis will be skipped."
    # Create empty placeholder for baseline stats to avoid errors later
    mkdir -p validation/results # Ensure output dir exists
    echo "{}" > validation/results/baseline_stats.json # Create an empty JSON file
fi
echo

# Step 8: Run Comparative Analysis
echo "Step 6: Running comparative analysis..."
# Based on scripts/compare_models_simple.py content, it accepts:
# --cognitive, --baseline, --test_data, --seq_length, --batch, --output
# It expects test_data to be loadable by FinancialDataLoader.
# Let's call it with the original test_data.csv.
# The previous argument error was strange, but the script source shows it accepts these.

if [ "$BASELINE_MODEL_FOUND" = true ]; then
    python scripts/compare_models_simple.py \
      --cognitive "$COGNITIVE_MODEL" \
      --baseline "$BASELINE_MODEL" \
      --test_data data/financial/test_data.csv \
      --seq_length 20 --batch 8 \
      --output validation/results
    # Check if compare_models_simple.py ran successfully
    if [ $? -ne 0 ]; then
        echo "Error during comparative analysis. Aborting."
        exit 1
    fi
else
    echo "Skipping compare_models_simple.py since baseline model is missing."
    # Ensure the comparison.json file is created if this step is skipped
    mkdir -p validation/results
    echo "{\"cognitive_metrics\": {}, \"baseline_metrics\": {}, \"improvements\": {}}" > validation/results/comparison.json
fi
echo "Comparative analysis complete."
echo

# Step 9: Run Walk-Forward Validation
echo "Step 7: Running walk-forward validation..."
# Based on scripts/walk_forward.py content, it accepts:
# --model, --data, --window, --step, --seq_length, --batch, --output
# It loads --data using pd.read_csv and then temp files using FinancialDataLoader.
# It is designed for the Cognitive model only.
# It needs AAPL_processed_data.csv. Let's use the original path.

python scripts/walk_forward.py \
  --model "$COGNITIVE_MODEL" \
  --data data/financial/AAPL_processed_data.csv \
  --window 252 --step 41 --seq_length 20 --batch 16 \
  --output validation/walk_forward_results.json
# Check if walk_forward.py ran successfully
if [ $? -ne 0 ]; then
    echo "Error during walk-forward validation. Aborting."
    exit 1
fi
echo "Walk-forward validation complete."
echo

# Step 10: Run Trading Backtest
echo "Step 8: Running trading backtest..."
# Based on scripts/trading_backtest.py content, it accepts:
# --model, --data, --seq_length, --threshold, --output, --initial_capital, --position_size, --transaction_cost
# It loads --data using pd.read_csv expecting 'timestamp' index and then extracts columns.
# It is designed for the Cognitive model only.
# It needs test_data.csv with a timestamp column. Using the original test_data.csv.

python scripts/trading_backtest.py \
  --model "$COGNITIVE_MODEL" \
  --data data/financial/test_data.csv \
  --seq_length 20 \
  --threshold 0.005 \
  --output validation/trading_backtest.json \
  --initial_capital 10000.0 \
  --position_size 0.1 \
  --transaction_cost 0.001
# Check if trading_backtest.py ran successfully
if [ $? -ne 0 ]; then
    echo "Error during trading backtest. Aborting."
    exit 1
fi
echo "Trading backtest complete."
echo

# Step 11: Generate Visualization
echo "Step 9: Generating enhanced visualizations..."
python scripts/enhanced_visualization.py \
  --cognitive_results validation/results/cognitive_stats.json \
  --baseline_results validation/results/baseline_stats.json \
  --output_dir validation/plots
# Check if enhanced_visualization.py ran successfully
if [ $? -ne 0 ]; then
    echo "Error during visualization generation. Aborting."
    exit 1
fi
echo "Visualization generation complete."
echo

# Step 12: Generate Report
echo "Step 10: Generating final report..."
python scripts/generate_report.py \
  --walk_forward validation/walk_forward_results.json \
  --backtest validation/trading_backtest.json \
  --comparison validation/results/comparison.json \
  --output validation/final_report.md
# Check if generate_report.py ran successfully
if [ $? -ne 0 ]; then
    echo "Error during report generation. Aborting."
    exit 1
fi
echo "Report generation complete."
echo

echo "===== Full Validation Workflow Completed ====="
echo "Finished at $(date)"
echo "Final report available at: validation/final_report.md"
EOF
  chmod +x scripts/generate_report.py
  echo "generate_report.py created."
fi


# Step 3: Prepare Environment
echo "Step 1: Preparing environment..."
python scripts/prepare_environment.py
# Check if prepare_environment.py ran successfully
if [ $? -ne 0 ]; then
    echo "Error during environment preparation. Aborting."
    exit 1
fi
echo "Environment preparation complete."
echo

# Step 4: Data Preparation
echo "Step 2: Preparing data..."
# This script splits the data and creates validation_data.csv and test_data.csv
python split_data.py \
  --input data/financial/AAPL_historical_data_20200420_to_20250419.csv \
  --train 0.7 --val 0.15 --test 0.15 --mode time \
  --output data/financial
# Check if split_data.py ran successfully
if [ $? -ne 0 ]; then
    echo "Error during data splitting. Aborting."
    exit 1
fi
echo "Data preparation complete."
echo

# --- NEW Step: Create 6-column validation data for training/evaluation ---
# The internal validation step in train.py and train_baseline.py seems to expect exactly 6 columns.
# We create a dedicated 6-column file from the validation_data.csv generated above.
echo "Step 2.5: Creating 6-column validation data..."
# Using awk to select the first 6 columns including the header
# Columns: 1-timestamp, 2-price, 3-volume, 4-returns, 5-log_returns, 6-volatility
awk -F, '{print $1","$2","$3","$4","$5","$6}' OFS=, data/financial/validation_data.csv > data/financial/validation_data_6cols.csv
# Check if awk command ran successfully
if [ $? -ne 0 ]; then
    echo "Error creating 6-column validation data. Aborting."
    exit 1
fi
echo "6-column validation data created: data/financial/validation_data_6cols.csv"
echo

# Step 5: Train Baseline Model
echo "Step 3: Training baseline model..."
# Use the 6-column validation data for internal validation during training
python scripts/train_baseline.py \
  --financial_train data/financial/train_data.csv \
  --financial_val data/financial/validation_data_6cols.csv \
  --iters 2000 --batch 16 --seq_length 20 --lr 1e-5 --save_interval 200
# Check if train_baseline.py ran successfully
if [ $? -ne 0 ]; then
    echo "Error during baseline model training. Aborting."
    exit 1
fi
echo "Baseline model training complete."
echo

# Step 6: Train Cognitive Model
echo "Step 4: Training cognitive architecture..."
# Use the 6-column validation data for internal validation during training
python train.py \
  --financial_train data/financial/train_data.csv \
  --financial_val data/financial/validation_data_6cols.csv \
  --iters 2000 --batch 8 --seq_length 20 --lr 5e-6 \
  --eval_interval 100 --save_interval 200 --verbose
# Check if train.py ran successfully
if [ $? -ne 0 ]; then
    echo "Error during cognitive model training. Aborting."
    exit 1
fi
echo "Cognitive model training complete."
echo

# Step 7: Get latest model paths
echo "Step 5: Locating latest models..."
# Ensure models directory exists before listing
mkdir -p models # Use mkdir -p for safety

# Use 2>/dev/null to suppress errors if no files match before check
COGNITIVE_MODEL=$(ls -t models/financial_consciousness_*.pth 2>/dev/null | head -1 || echo "")
BASELINE_MODEL=$(ls -t models/baseline_lstm_*.pth 2>/dev/null | head -1 || echo "")

# Check if models exist and set flags
COGNITIVE_MODEL_FOUND=false
if [ -n "$COGNITIVE_MODEL" ]; then
    echo "Found cognitive model: $COGNITIVE_MODEL"
    COGNITIVE_MODEL_FOUND=true
else
    echo "Error: No cognitive model found (looked for models/financial_consciousness_*.pth). Aborting."
    exit 1 # Abort if cognitive model is missing as it's needed for all evaluation steps
fi

BASELINE_MODEL_FOUND=false
if [ -n "$BASELINE_MODEL" ]; then
     echo "Found baseline model: $BASELINE_MODEL"
     BASELINE_MODEL_FOUND=true
else
    echo "Warning: No baseline model found (looked for models/baseline_lstm_*.pth). Comparative analysis will be skipped."
    # Create empty placeholder for baseline stats to avoid errors later
    mkdir -p validation/results # Ensure output dir exists
    echo "{}" > validation/results/baseline_stats.json # Create an empty JSON file
fi
echo

# Step 8: Run Comparative Analysis
echo "Step 6: Running comparative analysis..."
# Based on scripts/compare_models_simple.py content, it accepts:
# --cognitive, --baseline, --test_data, --seq_length, --batch, --output
# It expects test_data to be loadable by FinancialDataLoader.
# Let's call it with the original test_data.csv.
# The previous argument error was strange, but the script source shows it accepts these.

if [ "$BASELINE_MODEL_FOUND" = true ]; then
    python scripts/compare_models_simple.py \
      --cognitive "$COGNITIVE_MODEL" \
      --baseline "$BASELINE_MODEL" \
      --test_data data/financial/test_data.csv \
      --seq_length 20 --batch 8 \
      --output validation/results
    # Check if compare_models_simple.py ran successfully
    if [ $? -ne 0 ]; then
        echo "Error during comparative analysis. Aborting."
        exit 1
    fi
else
    echo "Skipping compare_models_simple.py since baseline model is missing."
    # Ensure the comparison.json file is created if this step is skipped
    mkdir -p validation/results
    echo "{\"cognitive_metrics\": {}, \"baseline_metrics\": {}, \"improvements\": {}}" > validation/results/comparison.json
fi
echo "Comparative analysis complete."
echo

# Step 9: Run Walk-Forward Validation
echo "Step 7: Running walk-forward validation..."
# Based on scripts/walk_forward.py content, it accepts:
# --model, --data, --window, --step, --seq_length, --batch, --output
# It loads --data using pd.read_csv and then temp files using FinancialDataLoader.
# It is designed for the Cognitive model only.
# It needs AAPL_processed_data.csv. Let's use the original path.

python scripts/walk_forward.py \
  --model "$COGNITIVE_MODEL" \
  --data data/financial/AAPL_processed_data.csv \
  --window 252 --step 41 --seq_length 20 --batch 16 \
  --output validation/walk_forward_results.json
# Check if walk_forward.py ran successfully
if [ $? -ne 0 ]; then
    echo "Error during walk-forward validation. Aborting."
    exit 1
fi
echo "Walk-forward validation complete."
echo

# Step 10: Run Trading Backtest
echo "Step 8: Running trading backtest..."
# Based on scripts/trading_backtest.py content, it accepts:
# --model, --data, --seq_length, --threshold, --output, --initial_capital, --position_size, --transaction_cost
# It loads --data using pd.read_csv expecting 'timestamp' index and then extracts columns.
# It is designed for the Cognitive model only.
# It needs test_data.csv with a timestamp column. Using the original test_data.csv.

python scripts/trading_backtest.py \
  --model "$COGNITIVE_MODEL" \
  --data data/financial/test_data.csv \
  --seq_length 20 \
  --threshold 0.005 \
  --output validation/trading_backtest.json \
  --initial_capital 10000.0 \
  --position_size 0.1 \
  --transaction_cost 0.001
# Check if trading_backtest.py ran successfully
if [ $? -ne 0 ]; then
    echo "Error during trading backtest. Aborting."
    exit 1
fi
echo "Trading backtest complete."
echo

# Step 11: Generate Visualization
echo "Step 9: Generating enhanced visualizations..."
python scripts/enhanced_visualization.py \
  --cognitive_results validation/results/cognitive_stats.json \
  --baseline_results validation/results/baseline_stats.json \
  --output_dir validation/plots
# Check if enhanced_visualization.py ran successfully
if [ $? -ne 0 ]; then
    echo "Error during visualization generation. Aborting."
    exit 1
fi
echo "Visualization generation complete."
echo

# Step 12: Generate Report
echo "Step 10: Generating final report..."
python scripts/generate_report.py \
  --walk_forward validation/walk_forward_results.json \
  --backtest validation/trading_backtest.json \
  --comparison validation/results/comparison.json \
  --output validation/final_report.md
# Check if generate_report.py ran successfully
if [ $? -ne 0 ]; then
    echo "Error during report generation. Aborting."
    exit 1
fi
echo "Report generation complete."
echo

echo "===== Full Validation Workflow Completed ====="
echo "Finished at $(date)"
echo "Final report available at: validation/final_report.md"
EOF
  chmod +x scripts/generate_report.py
  echo "generate_report.py created."
fi


# Step 3: Prepare Environment
echo "Step 1: Preparing environment..."
python scripts/prepare_environment.py
# Check if prepare_environment.py ran successfully
if [ $? -ne 0 ]; then
    echo "Error during environment preparation. Aborting."
    exit 1
fi
echo "Environment preparation complete."
echo

# Step 4: Data Preparation
echo "Step 2: Preparing data..."
# This script splits the data and creates validation_data.csv and test_data.csv
python split_data.py \
  --input data/financial/AAPL_historical_data_20200420_to_20250419.csv \
  --train 0.7 --val 0.15 --test 0.15 --mode time \
  --output data/financial
# Check if split_data.py ran successfully
if [ $? -ne 0 ]; then
    echo "Error during data splitting. Aborting."
    exit 1
fi
echo "Data preparation complete."
echo

# --- NEW Step: Create 6-column validation data for training/evaluation ---
# The internal validation step in train.py and train_baseline.py seems to expect exactly 6 columns.
# We create a dedicated 6-column file from the validation_data.csv generated above.
echo "Step 2.5: Creating 6-column validation data..."
# Using awk to select the first 6 columns including the header
# Columns: 1-timestamp, 2-price, 3-volume, 4-returns, 5-log_returns, 6-volatility
awk -F, '{print $1","$2","$3","$4","$5","$6}' OFS=, data/financial/validation_data.csv > data/financial/validation_data_6cols.csv
# Check if awk command ran successfully
if [ $? -ne 0 ]; then
    echo "Error creating 6-column validation data. Aborting."
    exit 1
fi
echo "6-column validation data created: data/financial/validation_data_6cols.csv"
echo

# Step 5: Train Baseline Model
echo "Step 3: Training baseline model..."
# Use the 6-column validation data for internal validation during training
python scripts/train_baseline.py \
  --financial_train data/financial/train_data.csv \
  --financial_val data/financial/validation_data_6cols.csv \
  --iters 2000 --batch 16 --seq_length 20 --lr 1e-5 --save_interval 200
# Check if train_baseline.py ran successfully
if [ $? -ne 0 ]; then
    echo "Error during baseline model training. Aborting."
    exit 1
fi
echo "Baseline model training complete."
echo

# Step 6: Train Cognitive Model
echo "Step 4: Training cognitive architecture..."
# Use the 6-column validation data for internal validation during training
python train.py \
  --financial_train data/financial/train_data.csv \
  --financial_val data/financial/validation_data_6cols.csv \
  --iters 2000 --batch 8 --seq_length 20 --lr 5e-6 \
  --eval_interval 100 --save_interval 200 --verbose
# Check if train.py ran successfully
if [ $? -ne 0 ]; then
    echo "Error during cognitive model training. Aborting."
    exit 1
fi
echo "Cognitive model training complete."
echo

# Step 7: Get latest model paths
echo "Step 5: Locating latest models..."
# Ensure models directory exists before listing
mkdir -p models # Use mkdir -p for safety

# Use 2>/dev/null to suppress errors if no files match before check
COGNITIVE_MODEL=$(ls -t models/financial_consciousness_*.pth 2>/dev/null | head -1 || echo "")
BASELINE_MODEL=$(ls -t models/baseline_lstm_*.pth 2>/dev/null | head -1 || echo "")

# Check if models exist and set flags
COGNITIVE_MODEL_FOUND=false
if [ -n "$COGNITIVE_MODEL" ]; then
    echo "Found cognitive model: $COGNITIVE_MODEL"
    COGNITIVE_MODEL_FOUND=true
else
    echo "Error: No cognitive model found (looked for models/financial_consciousness_*.pth). Aborting."
    exit 1 # Abort if cognitive model is missing as it's needed for all evaluation steps
fi

BASELINE_MODEL_FOUND=false
if [ -n "$BASELINE_MODEL" ]; then
     echo "Found baseline model: $BASELINE_MODEL"
     BASELINE_MODEL_FOUND=true
else
    echo "Warning: No baseline model found (looked for models/baseline_lstm_*.pth). Comparative analysis will be skipped."
    # Create empty placeholder for baseline stats to avoid errors later
    mkdir -p validation/results # Ensure output dir exists
    echo "{}" > validation/results/baseline_stats.json # Create an empty JSON file
fi
echo

# Step 8: Run Comparative Analysis
echo "Step 6: Running comparative analysis..."
# Based on scripts/compare_models_simple.py content, it accepts:
# --cognitive, --baseline, --test_data, --seq_length, --batch, --output
# It expects test_data to be loadable by FinancialDataLoader.
# Let's call it with the original test_data.csv.
# The previous argument error was strange, but the script source shows it accepts these.

if [ "$BASELINE_MODEL_FOUND" = true ]; then
    python scripts/compare_models_simple.py \
      --cognitive "$COGNITIVE_MODEL" \
      --baseline "$BASELINE_MODEL" \
      --test_data data/financial/test_data.csv \
      --seq_length 20 --batch 8 \
      --output validation/results
    # Check if compare_models_simple.py ran successfully
    if [ $? -ne 0 ]; then
        echo "Error during comparative analysis. Aborting."
        exit 1
    fi
else
    echo "Skipping compare_models_simple.py since baseline model is missing."
    # Ensure the comparison.json file is created if this step is skipped
    mkdir -p validation/results
    echo "{\"cognitive_metrics\": {}, \"baseline_metrics\": {}, \"improvements\": {}}" > validation/results/comparison.json
fi
echo "Comparative analysis complete."
echo

# Step 9: Run Walk-Forward Validation
echo "Step 7: Running walk-forward validation..."
# Based on scripts/walk_forward.py content, it accepts:
# --model, --data, --window, --step, --seq_length, --batch, --output
# It loads --data using pd.read_csv and then temp files using FinancialDataLoader.
# It is designed for the Cognitive model only.
# It needs AAPL_processed_data.csv. Let's use the original path.

python scripts/walk_forward.py \
  --model "$COGNITIVE_MODEL" \
  --data data/financial/AAPL_processed_data.csv \
  --window 252 --step 41 --seq_length 20 --batch 16 \
  --output validation/walk_forward_results.json
# Check if walk_forward.py ran successfully
if [ $? -ne 0 ]; then
    echo "Error during walk-forward validation. Aborting."
    exit 1
fi
echo "Walk-forward validation complete."
echo

# Step 10: Run Trading Backtest
echo "Step 8: Running trading backtest..."
# Based on scripts/trading_backtest.py content, it accepts:
# --model, --data, --seq_length, --threshold, --output, --initial_capital, --position_size, --transaction_cost
# It loads --data using pd.read_csv expecting 'timestamp' index and then extracts columns.
# It is designed for the Cognitive model only.
# It needs test_data.csv with a timestamp column. Using the original test_data.csv.

python scripts/trading_backtest.py \
  --model "$COGNITIVE_MODEL" \
  --data data/financial/test_data.csv \
  --seq_length 20 \
  --threshold 0.005 \
  --output validation/trading_backtest.json \
  --initial_capital 10000.0 \
  --position_size 0.1 \
  --transaction_cost 0.001
# Check if trading_backtest.py ran successfully
if [ $? -ne 0 ]; then
    echo "Error during trading backtest. Aborting."
    exit 1
fi
echo "Trading backtest complete."
echo

# Step 11: Generate Visualization
echo "Step 9: Generating enhanced visualizations..."
python scripts/enhanced_visualization.py \
  --cognitive_results validation/results/cognitive_stats.json \
  --baseline_results validation/results/baseline_stats.json \
  --output_dir validation/plots
# Check if enhanced_visualization.py ran successfully
if [ $? -ne 0 ]; then
    echo "Error during visualization generation. Aborting."
    exit 1
fi
echo "Visualization generation complete."
echo

# Step 12: Generate Report
echo "Step 10: Generating final report..."
python scripts/generate_report.py \
  --walk_forward validation/walk_forward_results.json \
  --backtest validation/trading_backtest.json \
  --comparison validation/results/comparison.json \
  --output validation/final_report.md
# Check if generate_report.py ran successfully
if [ $? -ne 0 ]; then
    echo "Error during report generation. Aborting."
    exit 1
fi
echo "Report generation complete."
echo

echo "===== Full Validation Workflow Completed ====="
echo "Finished at $(date)"
echo "Final report available at: validation/final_report.md"
EOF
  chmod +x scripts/generate_report.py
  echo "generate_report.py created."
fi


# Step 3: Prepare Environment
echo "Step 1: Preparing environment..."
python scripts/prepare_environment.py
# Check if prepare_environment.py ran successfully
if [ $? -ne 0 ]; then
    echo "Error during environment preparation. Aborting."
    exit 1
fi
echo "Environment preparation complete."
echo

# Step 4: Data Preparation
echo "Step 2: Preparing data..."
# This script splits the data and creates validation_data.csv and test_data.csv
python split_data.py \
  --input data/financial/AAPL_historical_data_20200420_to_20250419.csv \
  --train 0.7 --val 0.15 --test 0.15 --mode time \
  --output data/financial
# Check if split_data.py ran successfully
if [ $? -ne 0 ]; then
    echo "Error during data splitting. Aborting."
    exit 1
fi
echo "Data preparation complete."
echo

# --- NEW Step: Create 6-column validation data for training/evaluation ---
# The internal validation step in train.py and train_baseline.py seems to expect exactly 6 columns.
# We create a dedicated 6-column file from the validation_data.csv generated above.
echo "Step 2.5: Creating 6-column validation data..."
# Using awk to select the first 6 columns including the header
# Columns: 1-timestamp, 2-price, 3-volume, 4-returns, 5-log_returns, 6-volatility
awk -F, '{print $1","$2","$3","$4","$5","$6}' OFS=, data/financial/validation_data.csv > data/financial/validation_data_6cols.csv
# Check if awk command ran successfully
if [ $? -ne 0 ]; then
    echo "Error creating 6-column validation data. Aborting."
    exit 1
fi
echo "6-column validation data created: data/financial/validation_data_6cols.csv"
echo

# Step 5: Train Baseline Model
echo "Step 3: Training baseline model..."
# Use the 6-column validation data for internal validation during training
python scripts/train_baseline.py \
  --financial_train data/financial/train_data.csv \
  --financial_val data/financial/validation_data_6cols.csv \
  --iters 2000 --batch 16 --seq_length 20 --lr 1e-5 --save_interval 200
# Check if train_baseline.py ran successfully
if [ $? -ne 0 ]; then
    echo "Error during baseline model training. Aborting."
    exit 1
fi
echo "Baseline model training complete."
echo

# Step 6: Train Cognitive Model
echo "Step 4: Training cognitive architecture..."
# Use the 6-column validation data for internal validation during training
python train.py \
  --financial_train data/financial/train_data.csv \
  --financial_val data/financial/validation_data_6cols.csv \
  --iters 2000 --batch 8 --seq_length 20 --lr 5e-6 \
  --eval_interval 100 --save_interval 200 --verbose
# Check if train.py ran successfully
if [ $? -ne 0 ]; then
    echo "Error during cognitive model training. Aborting."
    exit 1
fi
echo "Cognitive model training complete."
echo

# Step 7: Get latest model paths
echo "Step 5: Locating latest models..."
# Ensure models directory exists before listing
mkdir -p models # Use mkdir -p for safety

# Use 2>/dev/null to suppress errors if no files match before check
COGNITIVE_MODEL=$(ls -t models/financial_consciousness_*.pth 2>/dev/null | head -1 || echo "")
BASELINE_MODEL=$(ls -t models/baseline_lstm_*.pth 2>/dev/null | head -1 || echo "")

# Check if models exist and set flags
COGNITIVE_MODEL_FOUND=false
if [ -n "$COGNITIVE_MODEL" ]; then
    echo "Found cognitive model: $COGNITIVE_MODEL"
    COGNITIVE_MODEL_FOUND=true
else
    echo "Error: No cognitive model found (looked for models/financial_consciousness_*.pth). Aborting."
    exit 1 # Abort if cognitive model is missing as it's needed for all evaluation steps
fi

BASELINE_MODEL_FOUND=false
if [ -n "$BASELINE_MODEL" ]; then
     echo "Found baseline model: $BASELINE_MODEL"
     BASELINE_MODEL_FOUND=true
else
    echo "Warning: No baseline model found (looked for models/baseline_lstm_*.pth). Comparative analysis will be skipped."
    # Create empty placeholder for baseline stats to avoid errors later
    mkdir -p validation/results # Ensure output dir exists
    echo "{}" > validation/results/baseline_stats.json # Create an empty JSON file
fi
echo

# Step 8: Run Comparative Analysis
echo "Step 6: Running comparative analysis..."
# Based on scripts/compare_models_simple.py content, it accepts:
# --cognitive, --baseline, --test_data, --seq_length, --batch, --output
# It expects test_data to be loadable by FinancialDataLoader.
# Let's call it with the original test_data.csv.
# The previous argument error was strange, but the script source shows it accepts these.

if [ "$BASELINE_MODEL_FOUND" = true ]; then
    python scripts/compare_models_simple.py \
      --cognitive "$COGNITIVE_MODEL" \
      --baseline "$BASELINE_MODEL" \
      --test_data data/financial/test_data.csv \
      --seq_length 20 --batch 8 \
      --output validation/results
    # Check if compare_models_simple.py ran successfully
    if [ $? -ne 0 ]; then
        echo "Error during comparative analysis. Aborting."
        exit 1
    fi
else
    echo "Skipping compare_models_simple.py since baseline model is missing."
    # Ensure the comparison.json file is created if this step is skipped
    mkdir -p validation/results
    echo "{\"cognitive_metrics\": {}, \"baseline_metrics\": {}, \"improvements\": {}}" > validation/results/comparison.json
fi
echo "Comparative analysis complete."
echo

# Step 9: Run Walk-Forward Validation
echo "Step 7: Running walk-forward validation..."
# Based on scripts/walk_forward.py content, it accepts:
# --model, --data, --window, --step, --seq_length, --batch, --output
# It loads --data using pd.read_csv and then temp files using FinancialDataLoader.
# It is designed for the Cognitive model only.
# It needs AAPL_processed_data.csv. Let's use the original path.

python scripts/walk_forward.py \
  --model "$COGNITIVE_MODEL" \
  --data data/financial/AAPL_processed_data.csv \
  --window 252 --step 41 --seq_length 20 --batch 16 \
  --output validation/walk_forward_results.json
# Check if walk_forward.py ran successfully
if [ $? -ne 0 ]; then
    echo "Error during walk-forward validation. Aborting."
    exit 1
fi
echo "Walk-forward validation complete."
echo

# Step 10: Run Trading Backtest
echo "Step 8: Running trading backtest..."
# Based on scripts/trading_backtest.py content, it accepts:
# --model, --data, --seq_length, --threshold, --output, --initial_capital, --position_size, --transaction_cost
# It loads --data using pd.read_csv expecting 'timestamp' index and then extracts columns.
# It is designed for the Cognitive model only.
# It needs test_data.csv with a timestamp column. Using the original test_data.csv.

python scripts/trading_backtest.py \
  --model "$COGNITIVE_MODEL" \
  --data data/financial/test_data.csv \
  --seq_length 20 \
  --threshold 0.005 \
  --output validation/trading_backtest.json \
  --initial_capital 10000.0 \
  --position_size 0.1 \
  --transaction_cost 0.001
# Check if trading_backtest.py ran successfully
if [ $? -ne 0 ]; then
    echo "Error during trading backtest. Aborting."
    exit 1
fi
echo "Trading backtest complete."
echo

# Step 11: Generate Visualization
echo "Step 9: Generating enhanced visualizations..."
python scripts/enhanced_visualization.py \
  --cognitive_results validation/results/cognitive_stats.json \
  --baseline_results validation/results/baseline_stats.json \
  --output_dir validation/plots
# Check if enhanced_visualization.py ran successfully
if [ $? -ne 0 ]; then
    echo "Error during visualization generation. Aborting."
    exit 1
fi
echo "Visualization generation complete."
echo

# Step 12: Generate Report
echo "Step 10: Generating final report..."
python scripts/generate_report.py \
  --walk_forward validation/walk_forward_results.json \
  --backtest validation/trading_backtest.json \
  --comparison validation/results/comparison.json \
  --output validation/final_report.md
# Check if generate_report.py ran successfully
if [ $? -ne 0 ]; then
    echo "Error during report generation. Aborting."
    exit 1
fi
echo "Report generation complete."
echo

echo "===== Full Validation Workflow Completed ====="
echo "Finished at $(date)"
echo "Final report available at: validation/final_report.md"
EOF
  chmod +x scripts/generate_report.py
  echo "generate_report.py created."
fi


# Step 3: Prepare Environment
echo "Step 1: Preparing environment..."
python scripts/prepare_environment.py
# Check if prepare_environment.py ran successfully
if [ $? -ne 0 ]; then
    echo "Error during environment preparation. Aborting."
    exit 1
fi
echo "Environment preparation complete."
echo

# Step 4: Data Preparation
echo "Step 2: Preparing data..."
# This script splits the data and creates validation_data.csv and test_data.csv
python split_data.py \
  --input data/financial/AAPL_historical_data_20200420_to_20250419.csv \
  --train 0.7 --val 0.15 --test 0.15 --mode time \
  --output data/financial
# Check if split_data.py ran successfully
if [ $? -ne 0 ]; then
    echo "Error during data splitting. Aborting."
    exit 1
fi
echo "Data preparation complete."
echo

# --- NEW Step: Create 6-column validation data for training/evaluation ---
# The internal validation step in train.py and train_baseline.py seems to expect exactly 6 columns.
# We create a dedicated 6-column file from the validation_data.csv generated above.
echo "Step 2.5: Creating 6-column validation data..."
# Using awk to select the first 6 columns including the header
# Columns: 1-timestamp, 2-price, 3-volume, 4-returns, 5-log_returns, 6-volatility
awk -F, '{print $1","$2","$3","$4","$5","$6}' OFS=, data/financial/validation_data.csv > data/financial/validation_data_6cols.csv
# Check if awk command ran successfully
if [ $? -ne 0 ]; then
    echo "Error creating 6-column validation data. Aborting."
    exit 1
fi
echo "6-column validation data created: data/financial/validation_data_6cols.csv"
echo

# Step 5: Train Baseline Model
echo "Step 3: Training baseline model..."
# Use the 6-column validation data for internal validation during training
python scripts/train_baseline.py \
  --financial_train data/financial/train_data.csv \
  --financial_val data/financial/validation_data_6cols.csv \
  --iters 2000 --batch 16 --seq_length 20 --lr 1e-5 --save_interval 200
# Check if train_baseline.py ran successfully
if [ $? -ne 0 ]; then
    echo "Error during baseline model training. Aborting."
    exit 1
fi
echo "Baseline model training complete."
echo

# Step 6: Train Cognitive Model
echo "Step 4: Training cognitive architecture..."
# Use the 6-column validation data for internal validation during training
python train.py \
  --financial_train data/financial/train_data.csv \
  --financial_val data/financial/validation_data_6cols.csv \
  --iters 2000 --batch 8 --seq_length 20 --lr 5e-6 \
  --eval_interval 100 --save_interval 200 --verbose
# Check if train.py ran successfully
if [ $? -ne 0 ]; then
    echo "Error during cognitive model training. Aborting."
    exit 1
fi
echo "Cognitive model training complete."
echo

# Step 7: Get latest model paths
echo "Step 5: Locating latest models..."
# Ensure models directory exists before listing
mkdir -p models # Use mkdir -p for safety

# Use 2>/dev/null to suppress errors if no files match before check
COGNITIVE_MODEL=$(ls -t models/financial_consciousness_*.pth 2>/dev/null | head -1 || echo "")
BASELINE_MODEL=$(ls -t models/baseline_lstm_*.pth 2>/dev/null | head -1 || echo "")

# Check if models exist and set flags
COGNITIVE_MODEL_FOUND=false
if [ -n "$COGNITIVE_MODEL" ]; then
    echo "Found cognitive model: $COGNITIVE_MODEL"
    COGNITIVE_MODEL_FOUND=true
else
    echo "Error: No cognitive model found (looked for models/financial_consciousness_*.pth). Aborting."
    exit 1 # Abort if cognitive model is missing as it's needed for all evaluation steps
fi

BASELINE_MODEL_FOUND=false
if [ -n "$BASELINE_MODEL" ]; then
     echo "Found baseline model: $BASELINE_MODEL"
     BASELINE_MODEL_FOUND=true
else
    echo "Warning: No baseline model found (looked for models/baseline_lstm_*.pth). Comparative analysis will be skipped."
    # Create empty placeholder for baseline stats to avoid errors later
    mkdir -p validation/results # Ensure output dir exists
    echo "{}" > validation/results/baseline_stats.json # Create an empty JSON file
fi
echo

# Step 8: Run Comparative Analysis
echo "Step 6: Running comparative analysis..."
# Based on scripts/compare_models_simple.py content, it accepts:
# --cognitive, --baseline, --test_data, --seq_length, --batch, --output
# It expects test_data to be loadable by FinancialDataLoader.
# Let's call it with the original test_data.csv.
# The previous argument error was strange, but the script source shows it accepts these.

if [ "$BASELINE_MODEL_FOUND" = true ]; then
    python scripts/compare_models_simple.py \
      --cognitive "$COGNITIVE_MODEL" \
      --baseline "$BASELINE_MODEL" \
      --test_data data/financial/test_data.csv \
      --seq_length 20 --batch 8 \
      --output validation/results
    # Check if compare_models_simple.py ran successfully
    if [ $? -ne 0 ]; then
        echo "Error during comparative analysis. Aborting."
        exit 1
    fi
else
    echo "Skipping compare_models_simple.py since baseline model is missing."
    # Ensure the comparison.json file is created if this step is skipped
    mkdir -p validation/results
    echo "{\"cognitive_metrics\": {}, \"baseline_metrics\": {}, \"improvements\": {}}" > validation/results/comparison.json
fi
echo "Comparative analysis complete."
echo

# Step 9: Run Walk-Forward Validation
echo "Step 7: Running walk-forward validation..."
# Based on scripts/walk_forward.py content, it accepts:
# --model, --data, --window, --step, --seq_length, --batch, --output
# It loads --data using pd.read_csv and then temp files using FinancialDataLoader.
# It is designed for the Cognitive model only.
# It needs AAPL_processed_data.csv. Let's use the original path.

python scripts/walk_forward.py \
  --model "$COGNITIVE_MODEL" \
  --data data/financial/AAPL_processed_data.csv \
  --window 252 --step 41 --seq_length 20 --batch 16 \
  --output validation/walk_forward_results.json
# Check if walk_forward.py ran successfully
if [ $? -ne 0 ]; then
    echo "Error during walk-forward validation. Aborting."
    exit 1
fi
echo "Walk-forward validation complete."
echo

# Step 10: Run Trading Backtest
echo "Step 8: Running trading backtest..."
# Based on scripts/trading_backtest.py content, it accepts:
# --model, --data, --seq_length, --threshold, --output, --initial_capital, --position_size, --transaction_cost
# It loads --data using pd.read_csv expecting 'timestamp' index and then extracts columns.
# It is designed for the Cognitive model only.
# It needs test_data.csv with a timestamp column. Using the original test_data.csv.

python scripts/trading_backtest.py \
  --model "$COGNITIVE_MODEL" \
  --data data/financial/test_data.csv \
  --seq_length 20 \
  --threshold 0.005 \
  --output validation/trading_backtest.json \
  --initial_capital 10000.0 \
  --position_size 0.1 \
  --transaction_cost 0.001
# Check if trading_backtest.py ran successfully
if [ $? -ne 0 ]; then
    echo "Error during trading backtest. Aborting."
    exit 1
fi
echo "Trading backtest complete."
echo

# Step 11: Generate Visualization
echo "Step 9: Generating enhanced visualizations..."
python scripts/enhanced_visualization.py \
  --cognitive_results validation/results/cognitive_stats.json \
  --baseline_results validation/results/baseline_stats.json \
  --output_dir validation/plots
# Check if enhanced_visualization.py ran successfully
if [ $? -ne 0 ]; then
    echo "Error during visualization generation. Aborting."
    exit 1
fi
echo "Visualization generation complete."
echo

# Step 12: Generate Report
echo "Step 10: Generating final report..."
python scripts/generate_report.py \
  --walk_forward validation/walk_forward_results.json \
  --backtest validation/trading_backtest.json \
  --comparison validation/results/comparison.json \
  --output validation/final_report.md
# Check if generate_report.py ran successfully
if [ $? -ne 0 ]; then
    echo "Error during report generation. Aborting."
    exit 1
fi
echo "Report generation complete."
echo

echo "===== Full Validation Workflow Completed ====="
echo "Finished at $(date)"
echo "Final report available at: validation/final_report.md"
EOF
  chmod +x scripts/generate_report.py
  echo "generate_report.py created."
fi


# Step 3: Prepare Environment
echo "Step 1: Preparing environment..."
python scripts/prepare_environment.py
# Check if prepare_environment.py ran successfully
if [ $? -ne 0 ]; then
    echo "Error during environment preparation. Aborting."
    exit 1
fi
echo "Environment preparation complete."
echo

# Step 4: Data Preparation
echo "Step 2: Preparing data..."
# This script splits the data and creates validation_data.csv and test_data.csv
python split_data.py \
  --input data/financial/AAPL_historical_data_20200420_to_20250419.csv \
  --train 0.7 --val 0.15 --test 0.15 --mode time \
  --output data/financial
# Check if split_data.py ran successfully
if [ $? -ne 0 ]; then
    echo "Error during data splitting. Aborting."
    exit 1
fi
echo "Data preparation complete."
echo

# --- NEW Step: Create 6-column validation data for training/evaluation ---
# The internal validation step in train.py and train_baseline.py seems to expect exactly 6 columns.
# We create a dedicated 6-column file from the validation_data.csv generated above.
echo "Step 2.5: Creating 6-column validation data..."
# Using awk to select the first 6 columns including the header
# Columns: 1-timestamp, 2-price, 3-volume, 4-returns, 5-log_returns, 6-volatility
awk -F, '{print $1","$2","$3","$4","$5","$6}' OFS=, data/financial/validation_data.csv > data/financial/validation_data_6cols.csv
# Check if awk command ran successfully
if [ $? -ne 0 ]; then
    echo "Error creating 6-column validation data. Aborting."
    exit 1
fi
echo "6-column validation data created: data/financial/validation_data_6cols.csv"
echo

# Step 5: Train Baseline Model
echo "Step 3: Training baseline model..."
# Use the 6-column validation data for internal validation during training
python scripts/train_baseline.py \
  --financial_train data/financial/train_data.csv \
  --financial_val data/financial/validation_data_6cols.csv \
  --iters 2000 --batch 16 --seq_length 20 --lr 1e-5 --save_interval 200
# Check if train_baseline.py ran successfully
if [ $? -ne 0 ]; then
    echo "Error during baseline model training. Aborting."
    exit 1
fi
echo "Baseline model training complete."
echo

# Step 6: Train Cognitive Model
echo "Step 4: Training cognitive architecture..."
# Use the 6-column validation data for internal validation during training
python train.py \
  --financial_train data/financial/train_data.csv \
  --financial_val data/financial/validation_data_6cols.csv \
  --iters 2000 --batch 8 --seq_length 20 --lr 5e-6 \
  --eval_interval 100 --save_interval 200 --verbose
# Check if train.py ran successfully
if [ $? -ne 0 ]; then
    echo "Error during cognitive model training. Aborting."
    exit 1
fi
echo "Cognitive model training complete."
echo

# Step 7: Get latest model paths
echo "Step 5: Locating latest models..."
# Ensure models directory exists before listing
mkdir -p models # Use mkdir -p for safety

# Use 2>/dev/null to suppress errors if no files match before check
COGNITIVE_MODEL=$(ls -t models/financial_consciousness_*.pth 2>/dev/null | head -1 || echo "")
BASELINE_MODEL=$(ls -t models/baseline_lstm_*.pth 2>/dev/null | head -1 || echo "")

# Check if models exist and set flags
COGNITIVE_MODEL_FOUND=false
if [ -n "$COGNITIVE_MODEL" ]; then
    echo "Found cognitive model: $COGNITIVE_MODEL"
    COGNITIVE_MODEL_FOUND=true
else
    echo "Error: No cognitive model found (looked for models/financial_consciousness_*.pth). Aborting."
    exit 1 # Abort if cognitive model is missing as it's needed for all evaluation steps
fi

BASELINE_MODEL_FOUND=false
if [ -n "$BASELINE_MODEL" ]; then
     echo "Found baseline model: $BASELINE_MODEL"
     BASELINE_MODEL_FOUND=true
else
    echo "Warning: No baseline model found (looked for models/baseline_lstm_*.pth). Comparative analysis will be skipped."
    # Create empty placeholder for baseline stats to avoid errors later
    mkdir -p validation/results # Ensure output dir exists
    echo "{}" > validation/results/baseline_stats.json # Create an empty JSON file
fi
echo

# Step 8: Run Comparative Analysis
echo "Step 6: Running comparative analysis..."
# Based on scripts/compare_models_simple.py content, it accepts:
# --cognitive, --baseline, --test_data, --seq_length, --batch, --output
# It expects test_data to be loadable by FinancialDataLoader.
# Let's call it with the original test_data.csv.
# The previous argument error was strange, but the script source shows it accepts these.

if [ "$BASELINE_MODEL_FOUND" = true ]; then
    python scripts/compare_models_simple.py \
      --cognitive "$COGNITIVE_MODEL" \
      --baseline "$BASELINE_MODEL" \
      --test_data data/financial/test_data.csv \
      --seq_length 20 --batch 8 \
      --output validation/results
    # Check if compare_models_simple.py ran successfully
    if [ $? -ne 0 ]; then
        echo "Error during comparative analysis. Aborting."
        exit 1
    fi
else
    echo "Skipping compare_models_simple.py since baseline model is missing."
    # Ensure the comparison.json file is created if this step is skipped
    mkdir -p validation/results
    echo "{\"cognitive_metrics\": {}, \"baseline_metrics\": {}, \"improvements\": {}}" > validation/results/comparison.json
fi
echo "Comparative analysis complete."
echo

# Step 9: Run Walk-Forward Validation
echo "Step 7: Running walk-forward validation..."
# Based on scripts/walk_forward.py content, it accepts:
# --model, --data, --window, --step, --seq_length, --batch, --output
# It loads --data using pd.read_csv and then temp files using FinancialDataLoader.
# It is designed for the Cognitive model only.
# It needs AAPL_processed_data.csv. Let's use the original path.

python scripts/walk_forward.py \
  --model "$COGNITIVE_MODEL" \
  --data data/financial/AAPL_processed_data.csv \
  --window 252 --step 41 --seq_length 20 --batch 16 \
  --output validation/walk_forward_results.json
# Check if walk_forward.py ran successfully
if [ $? -ne 0 ]; then
    echo "Error during walk-forward validation. Aborting."
    exit 1
fi
echo "Walk-forward validation complete."
echo

# Step 10: Run Trading Backtest
echo "Step 8: Running trading backtest..."
# Based on scripts/trading_backtest.py content, it accepts:
# --model, --data, --seq_length, --threshold, --output, --initial_capital, --position_size, --transaction_cost
# It loads --data using pd.read_csv expecting 'timestamp' index and then extracts columns.
# It is designed for the Cognitive model only.
# It needs test_data.csv with a timestamp column. Using the original test_data.csv.

python scripts/trading_backtest.py \
  --model "$COGNITIVE_MODEL" \
  --data data/financial/test_data.csv \
  --seq_length 20 \
  --threshold 0.005 \
  --output validation/trading_backtest.json \
  --initial_capital 10000.0 \
  --position_size 0.1 \
  --transaction_cost 0.001
# Check if trading_backtest.py ran successfully
if [ $? -ne 0 ]; then
    echo "Error during trading backtest. Aborting."
    exit 1
fi
echo "Trading backtest complete."
echo

# Step 11: Generate Visualization
echo "Step 9: Generating enhanced visualizations..."
python scripts/enhanced_visualization.py \
  --cognitive_results validation/results/cognitive_stats.json \
  --baseline_results validation/results/baseline_stats.json \
  --output_dir validation/plots
# Check if enhanced_visualization.py ran successfully
if [ $? -ne 0 ]; then
    echo "Error during visualization generation. Aborting."
    exit 1
fi
echo "Visualization generation complete."
echo

# Step 12: Generate Report
echo "Step 10: Generating final report..."
python scripts/generate_report.py \
  --walk