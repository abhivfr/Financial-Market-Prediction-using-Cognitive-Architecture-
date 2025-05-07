import argparse
import os
import torch
import pandas as pd
import numpy as np
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Updated imports to match current project structure
from src.arch.cognitive import CognitiveArchitecture
from src.arch.baseline_lstm import FinancialLSTMBaseline as BaselineModel
from src.data.financial_loader import EnhancedFinancialDataLoader
from src.evaluation.regime_evaluator import RegimeEvaluator

def load_model(model_path, model_type="cognitive", device="cpu"):
    """Load a trained model from checkpoint"""
    # Special handling for cognitive models with our new from_checkpoint method
    if model_type == "cognitive":
        try:
            # Use the new from_checkpoint method
            model = CognitiveArchitecture.from_checkpoint(model_path, device)
            
            # Get model configuration
            checkpoint = torch.load(model_path, map_location=device)
            
            if isinstance(checkpoint, dict) and 'config' in checkpoint:
                config = checkpoint['config']
            else:
                config = {
                    'input_dim': model.input_dim,
                    'hidden_dim': model.hidden_dim,
                    'memory_size': model.financial_memory.num_slots,
                    'output_dim': model.output_dim,
                    'seq_length': model.seq_length
                }
            
            print(f"Using cognitive model configuration: input_dim={config['input_dim']}, hidden_dim={config['hidden_dim']}, memory_size={config.get('memory_size', 'unknown')}")
            
            model.eval()
            return model, config
        except Exception as e:
            print(f"Warning: Failed to load cognitive model with from_checkpoint method: {e}")
            print("Falling back to legacy loading...")
    
    # Legacy loading method for baseline models or as fallback for cognitive models
    checkpoint = torch.load(model_path, map_location=device)
    
    # Extract configuration based on actual dimensions in the checkpoint
    config = {}
    
    # Case 1: If 'config' exists in checkpoint, use it
    if isinstance(checkpoint, dict) and 'config' in checkpoint:
        config = checkpoint['config']
    
    # Case 2: If checkpoint is a state dict (most likely)
    else:
        # Try to determine dimensions directly from the weights
        if 'temporal_encoder.lstm.weight_ih_l0' in checkpoint:
            # Extract input dimension from the LSTM input weights
            input_dim = checkpoint['temporal_encoder.lstm.weight_ih_l0'].shape[1]
            config['input_dim'] = input_dim
            
            # Extract hidden dimension from LSTM weights (LSTM has 4 gates)
            hidden_dim = checkpoint['temporal_encoder.lstm.weight_ih_l0'].shape[0] // 4
            config['hidden_dim'] = hidden_dim
        # For baseline models, look for lstm.weight_ih_l0
        elif 'lstm.weight_ih_l0' in checkpoint:
            # Extract input dimension from the LSTM input weights
            input_dim = checkpoint['lstm.weight_ih_l0'].shape[1]
            config['input_dim'] = input_dim
            
            # Extract hidden dimension from LSTM weights (LSTM has 4 gates)
            hidden_dim = checkpoint['lstm.weight_ih_l0'].shape[0] // 4
            config['hidden_dim'] = hidden_dim
            
            # Extract output dimension from final layer if available
            if 'fc.weight' in checkpoint:
                output_dim = checkpoint['fc.weight'].shape[0]
                config['output_dim'] = output_dim
        
        # Extract memory size if available
        if 'financial_memory.memory' in checkpoint:
            memory_size = checkpoint['financial_memory.memory'].shape[0]
            config['memory_size'] = memory_size
        
        # Extract output dimension if available 
        if 'market_state_sequence_predictor.predictor.weight' in checkpoint:
            output_dim = checkpoint['market_state_sequence_predictor.predictor.weight'].shape[0]
            config['output_dim'] = output_dim
    
        # Default values if not found
        if 'input_dim' not in config:
            config['input_dim'] = 5  # Default
        if 'hidden_dim' not in config:
            config['hidden_dim'] = 64  # Default
        if 'memory_size' not in config:
            config['memory_size'] = 50  # Default
        if 'output_dim' not in config:
            config['output_dim'] = 4  # Default
        if 'seq_length' not in config:
            config['seq_length'] = 20  # Default
    
    print(f"Using model configuration: input_dim={config['input_dim']}, hidden_dim={config['hidden_dim']}, memory_size={config['memory_size']}")
    
    # Create model with extracted configuration
    if model_type == "cognitive":
        model = CognitiveArchitecture(
            input_dim=config['input_dim'],
            hidden_dim=config['hidden_dim'],
            memory_size=config['memory_size'],
            output_dim=config.get('output_dim', 4),
            seq_length=config.get('seq_length', 20)
        )
    else:
        model = BaselineModel(
            input_dim=config.get('input_dim', 5),
            hidden_dim=config.get('hidden_dim', 64),
            num_layers=config.get('num_layers', 2),
            output_dim=config.get('output_dim', 4)
        )
    
    # Handle different checkpoint structures
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model.eval()
    
    return model, config

def main():
    parser = argparse.ArgumentParser(description="Evaluate model by market regime")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model checkpoint")
    parser.add_argument("--model_type", type=str, default="cognitive", choices=["cognitive", "baseline"],
                        help="Type of model to evaluate")
    parser.add_argument("--data_path", type=str, required=True, help="Path to test data CSV")
    parser.add_argument("--output_dir", type=str, default="evaluation/regime", help="Output directory for reports")
    parser.add_argument("--sequence_length", type=int, default=20, help="Sequence length for time series")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation")
    parser.add_argument("--compare_with", type=str, default=None, help="Optional path to compare with another model")
    parser.add_argument("--compare_type", type=str, default="baseline", choices=["cognitive", "baseline"],
                        help="Type of model to compare with")
    parser.add_argument("--device", type=str, default=None, help="Device to run evaluation on (cpu or cuda)")
    
    args = parser.parse_args()
    
    # Set device
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # Load data
        print(f"Loading data from {args.data_path}")
        df = pd.read_csv(args.data_path)
        
        # Extract regime labels if they exist in the data
        if "regime" in df.columns:
            regime_labels = df["regime"].values
            
            # Create regime names
            unique_regimes = np.unique(regime_labels)
            regime_names = {}
            
            for regime in unique_regimes:
                # Get a subset of data for this regime
                regime_data = df[df["regime"] == regime]
                
                # Calculate some statistics for this regime
                # Ensure column names exist and have correct case
                price_col = None
                for col in df.columns:
                    if col.lower() in ['close', 'price']:
                        price_col = col
                        break
                
                if price_col is None:
                    # Fallback regime names
                    name = f"Regime {regime}"
                else:
                    # Calculate regime characteristics
                    volatility = regime_data[price_col].pct_change().std() * np.sqrt(252)  # Annualized volatility
                    returns = regime_data[price_col].pct_change().mean() * 252  # Annualized returns
                    
                    # Determine regime characteristics based on statistics
                    if returns > 0.1:  # 10% annual return threshold
                        if volatility < 0.15:  # 15% annual volatility threshold
                            name = "Low Vol. Bull"
                        else:
                            name = "High Vol. Bull"
                    elif returns < -0.1:  # -10% annual return threshold
                        if volatility < 0.15:
                            name = "Low Vol. Bear"
                        else:
                            name = "High Vol. Bear"
                    else:
                        if volatility < 0.1:
                            name = "Sideways/Calm"
                        else:
                            name = "Choppy/Neutral"
                
                regime_names[regime] = f"Regime {regime}: {name}"
        else:
            # If no regime column exists in the data, create a simple one
            print("No regime column found in data. Using single regime for all data.")
            regime_labels = np.zeros(len(df))
            regime_names = {0: "All Data"}
        
        # Create data loader
        print(f"Creating data loader with sequence_length={args.sequence_length}, batch_size={args.batch_size}")
        try:
            data_loader = EnhancedFinancialDataLoader(
                data_path=args.data_path,
                sequence_length=args.sequence_length,
                batch_size=args.batch_size,
                shuffle=False,  # Don't shuffle for evaluation
                train=False  # Evaluation mode
            )
            # Use the data loader or test_loader attribute based on what's available
            loader_to_use = getattr(data_loader, 'test_loader', data_loader)
        except TypeError as e:
            # Try alternative constructor if the above fails
            print(f"First loader initialization failed: {e}. Trying alternative...")
            data_loader = EnhancedFinancialDataLoader(
                args.data_path,
                sequence_length=args.sequence_length,
                batch_size=args.batch_size
            )
            loader_to_use = data_loader
        
        # Load model
        print(f"Loading model from {args.model_path}")
        model, config = load_model(args.model_path, args.model_type, args.device)
        model = model.to(device)
        
        # Create evaluator
        evaluator = RegimeEvaluator(
            model=model,
            data_loader=loader_to_use,
            regime_labels=regime_labels,
            regime_names=regime_names,
            output_dir=args.output_dir
        )
        
        # Run evaluation
        print("Starting regime evaluation...")
        metrics = evaluator.evaluate(device=args.device)
        evaluator.save_results(metrics)
        
        print(f"Regime evaluation complete. Results saved to {args.output_dir}")
        
        # Compare with another model if specified
        if args.compare_with:
            print(f"Comparing with model at {args.compare_with}")
            compare_model, _ = load_model(args.compare_with, args.compare_type, args.device)
            compare_model = compare_model.to(device)
            
            # Run comparison
            comparison = evaluator.compare_models(
                compare_model, 
                model_name=args.model_type.capitalize(),
                other_model_name=args.compare_type.capitalize()
            )
            
            print(f"Model comparison complete. Results saved to {args.output_dir}")
            
    except Exception as e:
        print(f"Error during regime evaluation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
