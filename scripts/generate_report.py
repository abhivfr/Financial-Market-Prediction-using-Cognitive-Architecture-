#!/usr/bin/env python
# scripts/generate_report.py

import argparse
import json
import os
import pandas as pd
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
            window_desc = f"{window.get('window_start', '?')}-{window.get('window_end', '?')}"
            
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
                    signal_val = signal_summary.get('signal', {}).get(stat, 0)
                    conf_val = signal_summary.get('confidence', {}).get(stat, 0)
                    risk_val = signal_summary.get('risk_score', {}).get(stat, 0)
                    
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
    if comparison and 'cognitive_metrics' in comparison and 'baseline_metrics' in comparison:
        baseline_metrics = comparison.get('baseline_metrics', {})
        cognitive_metrics = comparison.get('cognitive_metrics', {})
        
        improvements = []
        for metric in ['price_accuracy', 'volume_correlation', 'returns_stability', 'volatility_prediction']:
            baseline_val = baseline_metrics.get(metric, 0)
            cognitive_val = cognitive_metrics.get(metric, 0)
            if baseline_val != 0:
                improvements.append((cognitive_val - baseline_val) / abs(baseline_val))
        
        if improvements:
            comparison_improvement = np.mean(improvements) * 100
    
        # Qualitative assessment
        if comparison_improvement > 20:
            assessment = "significantly outperforms"
        elif comparison_improvement > 5:
            assessment = "moderately outperforms"
        elif comparison_improvement > -5:
            assessment = "performs similarly to"
        else:
            assessment = "underperforms compared to"
        
        report += f"The Financial Consciousness model {assessment} baseline LSTM models "
        report += f"with an average improvement of {comparison_improvement:.1f}% across all metrics.\n\n"
    else:
        report += "Unable to determine comparative performance due to missing data.\n\n"
    
    if backtest and 'metrics' in backtest:
        sharpe_ratio = backtest['metrics'].get('sharpe_ratio', 0)
        if sharpe_ratio > 1.0:
            report += "The trading strategy shows promising results with a positive Sharpe ratio and acceptable drawdown levels.\n\n"
        else:
            report += "The trading strategy requires further refinement as the Sharpe ratio is below optimal levels.\n\n"
    
    report += "### Next Steps\n\n"
    report += "1. Consider additional feature engineering to improve volatility prediction\n"
    report += "2. Further optimize the memory consolidation parameters for more efficient learning\n"
    report += "3. Test the model on additional financial instruments to validate robustness\n"
    report += "4. Implement position sizing based on the model's uncertainty metrics\n"
    report += "5. Explore reinforcement learning techniques to enhance trading strategy performance\n"
    
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
    parser.add_argument("--walk_forward", required=True, help="Path to walk-forward results")
    parser.add_argument("--backtest", required=True, help="Path to backtest results")
    parser.add_argument("--comparison", required=False, help="Path to model comparison results")
    parser.add_argument("--output", default="validation/final_report.md", help="Output report path")
    args = parser.parse_args()
    
    generate_report(args.walk_forward, args.backtest, args.comparison, args.output)
