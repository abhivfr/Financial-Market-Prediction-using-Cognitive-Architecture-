import torch
from src.visualization.plot_engine import VisualizationEngine
from src.utils import load_checkpoint

def main(checkpoint_path):
    # Load model and data
    model, metrics = load_checkpoint(checkpoint_path)
    vis_engine = VisualizationEngine()
    
    # Generate plots
    price_fig = vis_engine.plot_4d_timeseries(metrics['predictions'], metrics['actuals'])
    attention_fig = vis_engine.plot_consciousness_heatmap(metrics['attention_patterns'][0])
    memory_fig = vis_engine.plot_memory_landscape(metrics['memory_states'])
    
    # Save outputs
    price_fig.savefig("results/4d_timeseries.png")
    attention_fig.write_html("results/attention_patterns.html")
    memory_fig.savefig("results/memory_landscape.png")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/final_model.pth")
    args = parser.parse_args()
    main(args.checkpoint)