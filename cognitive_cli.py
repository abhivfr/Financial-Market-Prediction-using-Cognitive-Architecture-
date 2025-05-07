#!/usr/bin/env python
# cognitive_cli.py - Command-line interface for cognitive architecture

import argparse
import os
import sys
import json
from datetime import datetime

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

def setup_data_parser(subparsers):
    """Setup argument parser for data commands"""
    # Data download parser
    download_parser = subparsers.add_parser('download', help='Download financial data')
    download_parser.add_argument('--tickers', nargs='+', default=['SPY'], help='Ticker symbols to download')
    download_parser.add_argument('--start', default='2015-01-01', help='Start date (YYYY-MM-DD)')
    download_parser.add_argument('--end', default=None, help='End date (YYYY-MM-DD), defaults to today')
    download_parser.add_argument('--output', default='data/raw', help='Output directory')
    download_parser.add_argument('--features', action='store_true', help='Calculate additional features')
    
    # Data split parser
    split_parser = subparsers.add_parser('split', help='Split data into train/val/test sets')
    split_parser.add_argument('--input', required=True, help='Input data file')
    split_parser.add_argument('--output', default='data/processed', help='Output directory')
    split_parser.add_argument('--train', type=float, default=0.7, help='Training data ratio')
    split_parser.add_argument('--val', type=float, default=0.15, help='Validation data ratio')
    split_parser.add_argument('--test', type=float, default=0.15, help='Test data ratio')
    split_parser.add_argument('--mode', choices=['random', 'time'], default='time', 
                             help='Split mode: random or time-based')
    split_parser.add_argument('--visualize', action='store_true', help='Visualize data splits')

def setup_train_parser(subparsers):
    """Setup argument parser for training commands"""
    # Training parser
    train_parser = subparsers.add_parser('train', help='Train cognitive architecture')
    train_parser.add_argument('--train', required=True, help='Path to training data')
    train_parser.add_argument('--val', required=True, help='Path to validation data')
    train_parser.add_argument('--output', default='models', help='Output directory for models')
    train_parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    train_parser.add_argument('--seq_length', type=int, default=20, help='Sequence length')
    train_parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    train_parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    train_parser.add_argument('--progressive', action='store_true', help='Use progressive training')
    train_parser.add_argument('--checkpoint', help='Path to checkpoint to resume training')
    train_parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # Pre-training parser
    pretrain_parser = subparsers.add_parser('pretrain', help='Pre-train individual components')
    pretrain_parser.add_argument('--train', required=True, help='Path to training data')
    pretrain_parser.add_argument('--val', required=True, help='Path to validation data')
    pretrain_parser.add_argument('--component', choices=['memory', 'attention', 'temporal', 'core'],
                                required=True, help='Component to pre-train')
    pretrain_parser.add_argument('--output', default='models/components', help='Output directory')
    pretrain_parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    pretrain_parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    pretrain_parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')

def setup_evaluate_parser(subparsers):
    """Setup argument parser for evaluation commands"""
    # Evaluation parser
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate trained model')
    eval_parser.add_argument('--model', required=True, help='Path to trained model')
    eval_parser.add_argument('--data', required=True, help='Path to evaluation data')
    eval_parser.add_argument('--output', default='evaluation_results', help='Output directory')
    eval_parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    eval_parser.add_argument('--seq_length', type=int, default=20, help='Sequence length')
    
    # Benchmark parser
    benchmark_parser = subparsers.add_parser('benchmark', help='Run progressive benchmark')
    benchmark_parser.add_argument('--cognitive', required=True, help='Path to cognitive model')
    benchmark_parser.add_argument('--baseline', required=True, help='Path to baseline model')
    benchmark_parser.add_argument('--data', required=True, help='Path to test data')
    benchmark_parser.add_argument('--output', default='benchmark_results', help='Output directory')
    benchmark_parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    benchmark_parser.add_argument('--seq_length', type=int, default=20, help='Sequence length')
    
    # Cognitive evaluation parser
    cognitive_parser = subparsers.add_parser('cognitive-eval', help='Evaluate cognitive capabilities')
    cognitive_parser.add_argument('--model', required=True, help='Path to trained model')
    cognitive_parser.add_argument('--data', required=True, help='Path to evaluation data')
    cognitive_parser.add_argument('--output', default='cognitive_evaluation', help='Output directory')
    cognitive_parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    cognitive_parser.add_argument('--seq_length', type=int, default=20, help='Sequence length')

def setup_visualize_parser(subparsers):
    """Setup argument parser for visualization commands"""
    # Architecture visualization parser
    viz_parser = subparsers.add_parser('visualize', help='Visualize model architecture')
    viz_parser.add_argument('--model', required=True, help='Path to trained model')
    viz_parser.add_argument('--data', required=True, help='Path to visualization data')
    viz_parser.add_argument('--output', default='visualization_results', help='Output directory')
    viz_parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    viz_parser.add_argument('--seq_length', type=int, default=20, help='Sequence length')
    
    # Results visualization parser
    results_parser = subparsers.add_parser('visualize-results', help='Visualize evaluation results')
    results_parser.add_argument('--cognitive', required=True, help='Path to cognitive evaluation results')
    results_parser.add_argument('--baseline', help='Path to baseline evaluation results for comparison')
    results_parser.add_argument('--output', default='comparison_results', help='Output directory')

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(description='Cognitive Architecture CLI')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Setup parsers for different command categories
    setup_data_parser(subparsers)
    setup_train_parser(subparsers)
    setup_evaluate_parser(subparsers)
    setup_visualize_parser(subparsers)
    
    # Version command
    version_parser = subparsers.add_parser('version', help='Display version information')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Execute command
    if args.command == 'download':
        from src.data.download_data import download_financial_data
        download_financial_data(
            tickers=args.tickers,
            start_date=args.start,
            end_date=args.end,
            output_dir=args.output,
            calculate_features=args.features
        )
    
    elif args.command == 'split':
        from src.data.split_data import split_data, visualize_splits
        split_data(
            input_path=args.input,
            train_ratio=args.train,
            val_ratio=args.val,
            test_ratio=args.test,
            split_mode=args.mode,
            output_dir=args.output
        )
        
        if args.visualize:
            visualize_splits(
                train_path=os.path.join(args.output, 'train.csv'),
                val_path=os.path.join(args.output, 'val.csv'),
                test_path=os.path.join(args.output, 'test.csv'),
                output_dir=args.output
            )
    
    elif args.command == 'train':
        from train_cognitive import train_with_progressive_focus, load_data, CognitiveArchitecture
        import torch
        import numpy as np
        
        # Set random seed
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        
        # Use GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        # Load data
        train_loader, val_loader = load_data(
            train_path=args.train,
            val_path=args.val,
            batch_size=args.batch_size,
            sequence_length=args.seq_length,
            regime_aware=True
        )
        
        # Create model
        model = CognitiveArchitecture()
        model.to(device)
        
        # Load checkpoint if specified
        if args.checkpoint:
            print(f"Loading checkpoint from {args.checkpoint}")
            model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        
        # Create timestamped directory for this run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_dir = os.path.join(args.output, timestamp)
        log_dir = os.path.join(checkpoint_dir, 'logs')
        
        # Train model
        if args.progressive:
            # Progressive training with component-specific focus
            train_with_progressive_focus(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                base_lr=args.lr,
                total_epochs=args.epochs,
                checkpoint_dir=checkpoint_dir,
                log_dir=log_dir
            )
        else:
            # Standard training
            from train_cognitive import create_component_optimizer, train
            optimizers = create_component_optimizer(model, args.lr)
            train(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizers=optimizers,
                device=device,
                num_epochs=args.epochs,
                checkpoint_dir=checkpoint_dir,
                log_dir=log_dir
            )
    
    elif args.command == 'pretrain':
        from train_cognitive import load_data, CognitiveArchitecture, train
        import torch
        import numpy as np
        from src.monitoring.introspect import Introspection
        from src.monitoring.adaptive_learning import AdaptiveLearning
        
        # Use GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        # Load data
        train_loader, val_loader = load_data(
            train_path=args.train,
            val_path=args.val,
            batch_size=args.batch_size,
            sequence_length=20,
            regime_aware=True
        )
        
        # Create model
        model = CognitiveArchitecture()
        model.to(device)
        
        # Create timestamped directory for this run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        component_dir = os.path.join(args.output, args.component)
        checkpoint_dir = os.path.join(component_dir, timestamp)
        log_dir = os.path.join(checkpoint_dir, 'logs')
        
        # Create component-specific optimizers
        from train_cognitive import create_component_optimizer
        optimizers = create_component_optimizer(model, args.lr)
        
        # Check if component optimizer exists
        if args.component not in optimizers:
            print(f"Error: Component '{args.component}' not found in model.")
            sys.exit(1)
        
        # Train component
        train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizers=optimizers,
            device=device,
            num_epochs=args.epochs,
            checkpoint_dir=checkpoint_dir,
            log_dir=log_dir,
            focus_component=args.component
        )
        
        # Save component-specific checkpoint
        component_path = os.path.join(checkpoint_dir, f"{args.component}_pretrained.pth")
        torch.save(model.state_dict(), component_path)
        print(f"Component pre-training complete. Model saved to {component_path}")
    
    elif args.command == 'evaluate':
        from evaluation_utils import comprehensive_model_evaluation
        from src.data.financial_loader import EnhancedFinancialDataLoader
        from src.arch.cognitive import CognitiveArchitecture
        import torch
        
        # Use GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        # Load model
        model = CognitiveArchitecture()
        model.load_state_dict(torch.load(args.model, map_location=device))
        model.to(device)
        model.eval()
        
        # Load data
        data_loader = EnhancedFinancialDataLoader(
            data_path=args.data,
            sequence_length=args.seq_length,
            batch_size=args.batch_size,
            regime_aware=True,
            augmentation=False
        )
        
        # Create timestamped directory for this run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(args.output, timestamp)
        
        # Run evaluation
        results = comprehensive_model_evaluation(
            model=model,
            data_loader=data_loader,
            output_dir=output_dir,
            device=device
        )
        
        print(f"Evaluation complete. Results saved to {output_dir}")
    
    elif args.command == 'benchmark':
        from progressive_benchmark import run_progressive_benchmark
        
        # Run benchmark
        run_progressive_benchmark(
            cognitive_path=args.cognitive,
            baseline_path=args.baseline,
            test_data_path=args.data,
            output_dir=args.output,
            batch_size=args.batch_size,
            seq_length=args.seq_length
        )
    
    elif args.command == 'cognitive-eval':
        from evaluate_cognitive_capabilities import evaluate_cognitive_capabilities
        
        # Run cognitive evaluation
        evaluate_cognitive_capabilities(
            model_path=args.model,
            data_path=args.data,
            output_dir=args.output,
            batch_size=args.batch_size,
            sequence_length=args.seq_length
        )
    
    elif args.command == 'visualize':
        from visualize_architecture import run_architecture_visualization
        
        # Run architecture visualization
        run_architecture_visualization(
            model_path=args.model,
            data_path=args.data,
            output_dir=args.output,
            batch_size=args.batch_size,
            sequence_length=args.seq_length
        )
    
    elif args.command == 'visualize-results':
        from evaluation_utils import compare_models
        import matplotlib.pyplot as plt
        
        # Load results
        with open(args.cognitive, 'r') as f:
            cognitive_results = json.load(f)
        
        # Create output directory
        os.makedirs(args.output, exist_ok=True)
        
        # Compare with baseline if provided
        if args.baseline:
            with open(args.baseline, 'r') as f:
                baseline_results = json.load(f)
            
            # Create comparison plots
            if 'financial_metrics' in cognitive_results and 'financial_metrics' in baseline_results:
                compare_models(
                    cognitive_results['financial_metrics'],
                    baseline_results['financial_metrics'],
                    output_path=os.path.join(args.output, 'financial_metrics_comparison.png')
                )
            
            print(f"Visualization complete. Results saved to {args.output}")
        else:
            # Just visualize cognitive results
            from evaluation_utils import plot_prediction_vs_actual
            
            if 'predictions' in cognitive_results and 'targets' in cognitive_results:
                plot_prediction_vs_actual(
                    cognitive_results['predictions'],
                    cognitive_results['targets'],
                    timestamps=cognitive_results.get('timestamps'),
                    output_path=os.path.join(args.output, 'prediction_vs_actual.png')
                )
            
            print(f"Visualization complete. Results saved to {args.output}")
    
    elif args.command == 'version':
        print("Cognitive Architecture v0.1.0")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
