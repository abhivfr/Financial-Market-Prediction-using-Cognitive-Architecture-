import argparse
import torch
import pandas as pd
import numpy as np
import os
import time
from src.arch.cognitive import CognitiveArchitecture
from src.monitoring.early_warning import EarlyWarningSystem
from src.data.financial_loader import FinancialDataLoader
import logging

logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_model(model_path, device="cpu"):
    """Load a trained model from checkpoint"""
    try:
        # Use the new from_checkpoint method
        model = CognitiveArchitecture.from_checkpoint(model_path, device)
        
        # Extract configuration
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'config' in checkpoint:
            config = checkpoint['config']
        else:
            # Create config from model attributes
            config = {
                'input_dim': model.input_dim,
                'hidden_dim': model.hidden_dim,
                'memory_size': model.financial_memory.num_slots,
                'output_dim': model.output_dim,
                'seq_length': model.seq_length
            }
        
        model.eval()
        return model, config
    except Exception as e:
        logger.warning(f"Failed to load model with from_checkpoint method: {e}")
        logger.info("Falling back to legacy loading method")
        
        # Legacy loading code
        checkpoint = torch.load(model_path, map_location=device)
        config = checkpoint.get('config', {})
        
        model = CognitiveArchitecture(**config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        return model, config

def simulate_live_data(data_loader, delay=1.0):
    """
    Simulate live data feed by iterating through test data with delay
    
    Args:
        data_loader: DataLoader instance
        delay: Delay between samples in seconds
    """
    for batch_idx, (data, target, date_indices) in enumerate(data_loader):
        # Process one sample at a time to simulate live data
        for i in range(len(data)):
            sample = data[i:i+1]
            sample_target = target[i:i+1]
            sample_date = date_indices[i:i+1]
            
            yield sample, sample_target, sample_date
            
            # Simulate delay between live data points
            time.sleep(delay)

def main():
    parser = argparse.ArgumentParser(description="Live monitoring with early warning system")
    parser.add_argument("--model_path", type=str, required=True, 
                        help="Path to the trained model checkpoint")
    parser.add_argument("--data_path", type=str, required=True, 
                        help="Path to test data CSV")
    parser.add_argument("--output_dir", type=str, default="monitoring/live", 
                        help="Output directory for monitoring")
    parser.add_argument("--sequence_length", type=int, default=20, 
                        help="Sequence length for time series")
    parser.add_argument("--lookback_window", type=int, default=30, 
                        help="Lookback window for warning detection")
    parser.add_argument("--threshold", type=float, default=2.5, 
                        help="Threshold multiplier for warnings")
    parser.add_argument("--delay", type=float, default=0.5, 
                        help="Delay between samples in seconds")
    parser.add_argument("--device", type=str, default="cpu", 
                        help="Device to run on (cpu or cuda)")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up logging to file
    file_handler = logging.FileHandler(os.path.join(args.output_dir, 'monitoring.log'))
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    logger.info(f"Starting live monitoring with model: {args.model_path}")
    
    # Load model
    model, config = load_model(args.model_path, args.device)
    model.to(args.device)
    
    # Set up data loader
    data_loader = FinancialDataLoader(
        data_path=args.data_path,
        sequence_length=args.sequence_length,
        batch_size=32,
        train=False  # Use test mode
    )
    
    # Create early warning system
    warning_system = EarlyWarningSystem(
        model=model,
        lookback_window=args.lookback_window,
        threshold_multiplier=args.threshold,
        output_dir=args.output_dir
    )
    
    # Loop through simulated live data
    logger.info("Beginning live data monitoring...")
    
    try:
        for sample_idx, (inputs, target, date_idx) in enumerate(
                simulate_live_data(data_loader.test_loader, args.delay)):
            
            # Get simulated timestamp
            df = pd.read_csv(args.data_path)
            if 'date' in df.columns and date_idx.item() < len(df):
                timestamp = df.iloc[date_idx.item()]['date']
            else:
                timestamp = f"Sample {sample_idx}"
            
            # Forward pass
            inputs = inputs.to(args.device)
            target = target.to(args.device)
            
            with torch.no_grad():
                prediction = model(inputs)
            
            # Process with warning system
            warnings = warning_system.process_prediction(
                inputs=inputs,
                prediction=prediction,
                target=target,
                timestamp=timestamp
            )
            
            # Log any warnings
            if warnings:
                logger.warning(f"Timestamp {timestamp} - Warnings detected: {warnings.keys()}")
                for warn_type, details in warnings.items():
                    logger.warning(f"  {warn_type}: {details}")
            
            # Progress update every 10 samples
            if sample_idx % 10 == 0:
                logger.info(f"Processed {sample_idx} samples. Current time: {timestamp}")
            
            # Stop after 1000 samples to avoid endless loop in demo
            if sample_idx >= 1000:
                logger.info("Reached maximum samples, stopping.")
                break
                
    except KeyboardInterrupt:
        logger.info("Monitoring interrupted by user")
    
    # Generate final report
    summary = warning_system.get_summary_report()
    logger.info(f"Monitoring session complete. Detected {summary['warnings_count']} warnings.")
    
    # Save summary report
    import json
    with open(os.path.join(args.output_dir, 'summary_report.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Summary report saved to {args.output_dir}/summary_report.json")

if __name__ == "__main__":
    main()
