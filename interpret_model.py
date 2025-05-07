#!/usr/bin/env python
# interpret_model.py - Command-line tool for model interpretation

import argparse
import os
import sys
import json
import torch
import numpy as np
import pandas as pd
from datetime import datetime

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from src.arch.cognitive import CognitiveArchitecture
from src.arch.baseline_lstm import FinancialLSTMBaseline
from src.data.financial_loader import EnhancedFinancialDataLoader
from src.utils.interpretability import InterpretabilityReport, FeatureAttributionAnalyzer

def load_model(model_path, model_type="cognitive", device="cpu"):
    """
    Load trained model
    
    Args:
        model_path: Path to model checkpoint
        model_type: Model type ('cognitive' or 'baseline')
        device: Computation device
        
    Returns:
        Loaded model
    """
    if model_type.lower() == "cognitive":
        model = CognitiveArchitecture()
    else:
        model = FinancialLSTMBaseline()
    
    # Load model state
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    model.to(device)
    model.eval()
    
    return model

def get_feature_names(data_path):
    """
    Extract feature names from data file
    
    Args:
        data_path: Path to data file
        
    Returns:
        List of feature names
    """
    try:
        # Read first row of CSV to get column names
        df = pd.read_csv(data_path, nrows=1)
        feature_names = df.columns.tolist()
        
        # Filter out non-feature columns
        non_features = ['date', 'timestamp', 'Date', 'Timestamp', 'Time']
        feature_names = [name for name in feature_names if name not in non_features]
        
        return feature_names
    except Exception as e:
        print(f"Error extracting feature names: {e}")
        # Return generic feature names as fallback
        return [f"Feature_{i}" for i in range(10)]

def run_interpretability_analysis(
    model_path, 
    data_path, 
    output_dir, 
    model_type="cognitive",
    sequence_length=20, 
    target_idx=0,
    batch_size=1,
    device="cpu"
):
    """
    Run interpretability analysis and generate report
    
    Args:
        model_path: Path to model checkpoint
        data_path: Path to test data
        output_dir: Directory to save reports
        model_type: Model type ('cognitive' or 'baseline')
        sequence_length: Input sequence length
        target_idx: Target index to explain (default: price)
        batch_size: Batch size (should be 1 for interpretability)
        device: Computation device
        
    Returns:
        Path to generated report
    """
    # Load model
    print(f"Loading {model_type} model from {model_path}")
    model = load_model(model_path, model_type, device)
    
    # Load data
    print(f"Loading data from {data_path}")
    data_loader = EnhancedFinancialDataLoader(
        data_path=data_path,
        sequence_length=sequence_length,
        batch_size=batch_size,
        regime_aware=True if model_type.lower() == "cognitive" else False,
        augmentation=False
    )
    
    # Get feature names
    feature_names = get_feature_names(data_path)
    print(f"Found {len(feature_names)} features: {', '.join(feature_names[:5])}...")
    
    # Initialize interpretability report generator
    interpreter = InterpretabilityReport(model, output_dir)
    
    # Generate reports for multiple samples
    reports = []
    
    print("Generating interpretability reports...")
    for batch_idx, batch in enumerate(data_loader):
        if batch_idx >= 5:  # Limit to a few samples
            break
        
        # Move batch to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        # Generate timestamp
        timestamp = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{batch_idx}"
        
        # Generate report
        print(f"  Generating report for sample {batch_idx+1}")
        report = interpreter.generate_report(
            inputs=batch,
            feature_names=feature_names,
            timestamp=timestamp,
            target_idx=target_idx
        )
        
        reports.append(report)
        
        print(f"  Report saved to {report['html']}")
    
    # Create summary report
    summary_path = os.path.join(output_dir, "summary.html")
    
    with open(summary_path, "w") as f:
        f.write("""<!DOCTYPE html>
<html>
<head>
    <title>Interpretability Reports Summary</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        h1, h2 { color: #333; }
        .report-list { margin-top: 20px; }
        .report-item { margin-bottom: 10px; padding: 10px; border: 1px solid #ddd; }
        .report-item a { color: #0066cc; text-decoration: none; }
        .report-item a:hover { text-decoration: underline; }
    </style>
</head>
<body>
    <h1>Interpretability Reports Summary</h1>
    <p>Model: {}</p>
    <p>Data: {}</p>
    <p>Generated on: {}</p>
    
    <h2>Individual Reports</h2>
    <div class="report-list">
""".format(model_path, data_path, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        
        for idx, report in enumerate(reports):
            rel_path = os.path.relpath(report["html"], output_dir)
            f.write(f"""
        <div class="report-item">
            <strong>Report {idx+1}</strong>: <a href="{rel_path}" target="_blank">{os.path.basename(report["html"])}</a>
        </div>""")
        
        f.write("""
    </div>
</body>
</html>""")
    
    print(f"Summary report saved to {summary_path}")
    return summary_path

def compare_feature_attributions(
    cognitive_model_path, 
    baseline_model_path, 
    data_path, 
    output_dir, 
    sequence_length=20, 
    target_idx=0,
    device="cpu"
):
    """
    Compare feature attributions between cognitive and baseline models
    
    Args:
        cognitive_model_path: Path to cognitive model checkpoint
        baseline_model_path: Path to baseline model checkpoint
        data_path: Path to test data
        output_dir: Directory to save comparison results
        sequence_length: Input sequence length
        target_idx: Target index to explain (default: price)
        device: Computation device
        
    Returns:
        Path to comparison report
    """
    # Load models
    print("Loading cognitive model")
    cognitive_model = load_model(cognitive_model_path, "cognitive", device)
    
    print("Loading baseline model")
    baseline_model = load_model(baseline_model_path, "baseline", device)
    
    # Load data
    print(f"Loading data from {data_path}")
    data_loader = EnhancedFinancialDataLoader(
        data_path=data_path,
        sequence_length=sequence_length,
        batch_size=1,
        regime_aware=False,  # Use same settings for both models
        augmentation=False
    )
    
    # Get feature names
    feature_names = get_feature_names(data_path)
    
    # Initialize feature attribution analyzers
    cognitive_analyzer = FeatureAttributionAnalyzer(cognitive_model)
    baseline_analyzer = FeatureAttributionAnalyzer(baseline_model)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate comparisons for multiple samples
    comparisons = []
    
    print("Generating comparison reports...")
    for batch_idx, batch in enumerate(data_loader):
        if batch_idx >= 3:  # Limit to a few samples
            break
        
        # Move batch to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        # Create sample directory
        sample_dir = os.path.join(output_dir, f"sample_{batch_idx}")
        os.makedirs(sample_dir, exist_ok=True)
        
        # Compute attributions for cognitive model
        print(f"  Computing attributions for cognitive model (sample {batch_idx+1})")
        cognitive_ig = cognitive_analyzer.compute_integrated_gradients(
            batch, target_idx=target_idx
        )
        
        # For baseline, we need to extract the sequence tensor
        sequence_batch = {'sequence': batch['sequence']} if isinstance(batch, dict) else batch
        
        # Compute attributions for baseline model
        print(f"  Computing attributions for baseline model (sample {batch_idx+1})")
        baseline_ig = baseline_analyzer.compute_integrated_gradients(
            sequence_batch, target_idx=target_idx
        )
        
        # Extract sequence attributions for comparison
        if isinstance(cognitive_ig, dict) and 'sequence' in cognitive_ig:
            cognitive_seq_ig = cognitive_ig['sequence']
        else:
            cognitive_seq_ig = cognitive_ig
        
        if isinstance(baseline_ig, dict) and 'sequence' in baseline_ig:
            baseline_seq_ig = baseline_ig['sequence']
        else:
            baseline_seq_ig = baseline_ig
        
        # Generate comparison plots
        print(f"  Generating comparison plots for sample {batch_idx+1}")
        
        # 1. Cognitive model attribution plot
        cognitive_path = os.path.join(sample_dir, "cognitive_attribution.png")
        cognitive_analyzer.generate_feature_attribution_plot(
            cognitive_seq_ig, feature_names, 
            title="Cognitive Model: Feature Attribution",
            output_path=cognitive_path,
            top_k=10
        )
        
        # 2. Baseline model attribution plot
        baseline_path = os.path.join(sample_dir, "baseline_attribution.png")
        baseline_analyzer.generate_feature_attribution_plot(
            baseline_seq_ig, feature_names, 
            title="Baseline Model: Feature Attribution",
            output_path=baseline_path,
            top_k=10
        )
        
        # 3. Direct comparison (difference in attribution)
        # Ensure attributions have same shape
        if cognitive_seq_ig.shape != baseline_seq_ig.shape:
            print(f"Warning: Attribution shapes don't match: {cognitive_seq_ig.shape} vs {baseline_seq_ig.shape}")
            # Try to make them comparable
            if len(cognitive_seq_ig.shape) > len(baseline_seq_ig.shape):
                cognitive_seq_ig = cognitive_seq_ig.mean(axis=tuple(range(cognitive_seq_ig.ndim - baseline_seq_ig.ndim)))
            elif len(baseline_seq_ig.shape) > len(cognitive_seq_ig.shape):
                baseline_seq_ig = baseline_seq_ig.mean(axis=tuple(range(baseline_seq_ig.ndim - cognitive_seq_ig.ndim)))
        
        # Calculate difference
        attribution_diff = cognitive_seq_ig - baseline_seq_ig
        
        # Generate difference plot
        diff_path = os.path.join(sample_dir, "attribution_difference.png")
        
        # Create custom plot for difference
        plt.figure(figsize=(12, 8))
        
        # Ensure attribution_diff is 1D
        if attribution_diff.ndim > 1:
            attribution_diff = attribution_diff.mean(axis=tuple(range(attribution_diff.ndim - 1)))
        
        # Get top features by absolute difference
        top_k = min(10, len(attribution_diff))
        top_indices = np.argsort(np.abs(attribution_diff))[-top_k:]
        top_diffs = attribution_diff[top_indices]
        top_features = [feature_names[i] if i < len(feature_names) else f"Feature {i}" for i in top_indices]
        
        # Plot differences
        bars = plt.barh(
            range(len(top_features)),
            top_diffs,
            color=[('green' if x > 0 else 'red') for x in top_diffs]
        )
        
        # Add values as text labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            label_x_pos = width if width > 0 else width - 0.05
            plt.text(
                label_x_pos, 
                bar.get_y() + bar.get_height()/2, 
                f'{width:.4f}', 
                va='center'
            )
        
        plt.yticks(range(len(top_features)), top_features)
        plt.title("Difference in Feature Attribution (Cognitive - Baseline)")
        plt.xlabel("Attribution Difference")
        plt.tight_layout()
        plt.savefig(diff_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Add to comparisons
        comparisons.append({
            "sample_idx": batch_idx,
            "sample_dir": sample_dir,
            "cognitive_attribution": cognitive_path,
            "baseline_attribution": baseline_path,
            "difference": diff_path
        })
        
        print(f"  Comparison saved to {sample_dir}")
    
    # Create summary HTML
    summary_path = os.path.join(output_dir, "comparison_summary.html")
    
    with open(summary_path, "w") as f:
        f.write("""<!DOCTYPE html>
<html>
<head>
    <title>Model Attribution Comparison</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        h1, h2, h3 { color: #333; }
        .comparison-section { margin-bottom: 30px; }
        .comparison-container { display: flex; flex-wrap: wrap; gap: 20px; }
        .comparison-item { flex: 1; min-width: 300px; margin-bottom: 20px; }
        .comparison-item img { max-width: 100%; border: 1px solid #ddd; }
    </style>
</head>
<body>
    <h1>Model Attribution Comparison</h1>
    <p>Cognitive Model: {}</p>
    <p>Baseline Model: {}</p>
    <p>Generated on: {}</p>
""".format(
    os.path.basename(cognitive_model_path), 
    os.path.basename(baseline_model_path),
    datetime.now().strftime("%Y-%m-%d %H:%M:%S")
))
        
        for comp in comparisons:
            f.write(f"""
    <div class="comparison-section">
        <h2>Sample {comp['sample_idx'] + 1}</h2>
        <div class="comparison-container">
            <div class="comparison-item">
                <h3>Cognitive Model</h3>
                <img src="{os.path.relpath(comp['cognitive_attribution'], output_dir)}" alt="Cognitive Attribution">
            </div>
            <div class="comparison-item">
                <h3>Baseline Model</h3>
                <img src="{os.path.relpath(comp['baseline_attribution'], output_dir)}" alt="Baseline Attribution">
            </div>
            <div class="comparison-item">
                <h3>Difference</h3>
                <img src="{os.path.relpath(comp['difference'], output_dir)}" alt="Attribution Difference">
            </div>
        </div>
    </div>""")
        
        f.write("""
</body>
</html>""")
    
    print(f"Comparison summary saved to {summary_path}")
    return summary_path

def main():
    parser = argparse.ArgumentParser(description="Generate model interpretability reports")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Single model analysis
    analyze_parser = subparsers.add_parser("analyze", help="Analyze a single model")
    analyze_parser.add_argument("--model", required=True, help="Path to model checkpoint")
    analyze_parser.add_argument("--data", required=True, help="Path to test data")
    analyze_parser.add_argument("--output", default="interpretability_reports", help="Output directory")
    analyze_parser.add_argument("--model_type", choices=["cognitive", "baseline"], default="cognitive", 
                              help="Model type")
    analyze_parser.add_argument("--seq_length", type=int, default=20, help="Sequence length")
    analyze_parser.add_argument("--target", type=int, default=0, help="Target index to explain")
    analyze_parser.add_argument("--device", default="cpu", help="Computation device")
    
    # Compare models
    compare_parser = subparsers.add_parser("compare", help="Compare cognitive and baseline models")
    compare_parser.add_argument("--cognitive", required=True, help="Path to cognitive model checkpoint")
    compare_parser.add_argument("--baseline", required=True, help="Path to baseline model checkpoint")
    compare_parser.add_argument("--data", required=True, help="Path to test data")
    compare_parser.add_argument("--output", default="model_comparisons", help="Output directory")
    compare_parser.add_argument("--seq_length", type=int, default=20, help="Sequence length")
    compare_parser.add_argument("--target", type=int, default=0, help="Target index to explain")
    compare_parser.add_argument("--device", default="cpu", help="Computation device")
    
    args = parser.parse_args()
    
    if args.command == "analyze":
        run_interpretability_analysis(
            model_path=args.model,
            data_path=args.data,
            output_dir=args.output,
            model_type=args.model_type,
            sequence_length=args.seq_length,
            target_idx=args.target,
            device=args.device
        )
    
    elif args.command == "compare":
        compare_feature_attributions(
            cognitive_model_path=args.cognitive,
            baseline_model_path=args.baseline,
            data_path=args.data,
            output_dir=args.output,
            sequence_length=args.seq_length,
            target_idx=args.target,
            device=args.device
        )
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
