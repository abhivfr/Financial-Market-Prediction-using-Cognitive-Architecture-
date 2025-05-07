# save as scripts/enhanced_visualization.py
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import argparse
import numpy as np
import json # Import json here

def create_model_comparison(cognitive_results, baseline_results, output_dir):
    """Creates a comparative plot of model metrics using Plotly."""
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create subplots for 4 metrics
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=["Price Prediction Accuracy", "Volume Correlation",
                        "Returns Stability", "Volatility Prediction"] # Updated titles for clarity
    )

    # Define metrics and their corresponding titles for the plot
    metrics_map = {
        "price_accuracy": "Price Prediction Accuracy",
        "volume_correlation": "Volume Correlation",
        "returns_stability": "Returns Stability",
        "volatility_prediction": "Volatility Prediction"
    }

    positions = [(1,1), (1,2), (2,1), (2,2)]

    for (metric_key, metric_title), pos in zip(metrics_map.items(), positions):
        baseline_val = baseline_results.get(metric_key, 0) # Use .get() for safety
        cognitive_val = cognitive_results.get(metric_key, 0) # Use .get() for safety

        fig.add_trace(
            go.Bar(
                x=["Baseline", "Consciousness Model"],
                y=[baseline_val, cognitive_val],
                text=[f"{baseline_val:.4f}", f"{cognitive_val:.4f}"],
                textposition="auto",
                name=metric_title, # Use descriptive title
                marker_color=["royalblue", "crimson"],
                showlegend=False # Hide individual bar legends
            ),
            row=pos[0], col=pos[1]
        )

        # Optional: Update y-axis title for clarity
        fig.update_yaxes(title_text=metric_title, row=pos[0], col=pos[1])


    fig.update_layout(
        title_text="Model Performance Comparison", # Overall title
        height=800,
        width=1000,
        #showlegend=False # Keep showlegend=False as handled per trace
    )

    # Save the figure
    html_path = os.path.join(output_dir, "enhanced_comparison.html")
    png_path = os.path.join(output_dir, "enhanced_comparison.png")

    fig.write_html(html_path)
    print(f"Saved interactive plot to {html_path}")

    try:
        # Requires the 'kaleido' package for image export
        fig.write_image(png_path)
        print(f"Saved static plot to {png_path}")
    except ImportError:
        print("Warning: Install 'kaleido' (pip install kaleido) for static image export.")
    except Exception as e:
        print(f"Warning: Could not save static image: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create enhanced visualizations")
    parser.add_argument("--cognitive_results", type=str, required=True, help="Path to cognitive results JSON")
    parser.add_argument("--baseline_results", type=str, required=True, help="Path to baseline results JSON")
    parser.add_argument("--output_dir", type=str, default="validation/plots", help="Output directory")
    args = parser.parse_args()

    # Load results
    try:
        with open(args.cognitive_results, 'r') as f:
            cognitive_results = json.load(f)
        print(f"Loaded cognitive results from {args.cognitive_results}")
    except FileNotFoundError:
        print(f"Error: Cognitive results file not found at {args.cognitive_results}")
        exit()
    except json.JSONDecodeError:
         print(f"Error: Could not decode JSON from {args.cognitive_results}")
         exit()


    try:
        with open(args.baseline_results, 'r') as f:
            baseline_results = json.load(f)
        print(f"Loaded baseline results from {args.baseline_results}")
    except FileNotFoundError:
        print(f"Error: Baseline results file not found at {args.baseline_results}")
        exit()
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {args.baseline_results}")
        exit()


    create_model_comparison(cognitive_results, baseline_results, args.output_dir)
