#!/usr/bin/env python
# create_target_datasets.py - Create datasets for different prediction targets

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_return_prediction_dataset(df, horizons=[1, 5, 10], ticker=None):
    """
    Create dataset for return prediction
    
    Args:
        df: Input DataFrame
        horizons: List of prediction horizons (in days)
        ticker: Specific ticker to use (None for all)
        
    Returns:
        DataFrame with return targets
    """
    result = df.copy()
    
    # Filter to specific ticker if provided
    if ticker is not None and 'ticker' in result.columns:
        result = result[result['ticker'] == ticker].copy()
    
    # Create target for each horizon
    for horizon in horizons:
        # Future returns
        result[f'target_return_{horizon}d'] = result.groupby('ticker')['close'].pct_change(horizon).shift(-horizon)
    
    # Drop rows with NaN targets
    result = result.dropna(subset=[f'target_return_{horizon}d' for horizon in horizons])
    
    return result

def create_volatility_prediction_dataset(df, horizons=[5, 10, 20], window_size=20, ticker=None):
    """
    Create dataset for volatility prediction
    
    Args:
        df: Input DataFrame
        horizons: List of prediction horizons (in days)
        window_size: Window size for volatility calculation
        ticker: Specific ticker to use (None for all)
        
    Returns:
        DataFrame with volatility targets
    """
    result = df.copy()
    
    # Filter to specific ticker if provided
    if ticker is not None and 'ticker' in result.columns:
        result = result[result['ticker'] == ticker].copy()
    
    # Create target for each horizon
    for horizon in horizons:
        # Calculate future volatility (std of returns over the next 'horizon' days)
        # This requires using a rolling window on future returns
        future_returns = result.groupby('ticker')['return_1d'].shift(-1).rolling(
            window=horizon, min_periods=1
        ).apply(lambda x: np.std(x) * np.sqrt(252))
        
        result[f'target_volatility_{horizon}d'] = future_returns
    
    # Drop rows with NaN targets
    result = result.dropna(subset=[f'target_volatility_{horizon}d' for horizon in horizons])
    
    return result

def create_regime_prediction_dataset(df, horizon=10, ticker=None):
    """
    Create dataset for regime prediction
    
    Args:
        df: Input DataFrame
        horizon: Prediction horizon (in days)
        ticker: Specific ticker to use (None for all)
        
    Returns:
        DataFrame with regime targets
    """
    result = df.copy()
    
    # Filter to specific ticker if provided
    if ticker is not None and 'ticker' in result.columns:
        result = result[result['ticker'] == ticker].copy()
    
    # Check if market_regime column exists
    if 'market_regime' not in result.columns:
        logger.warning("No 'market_regime' column found, cannot create regime prediction dataset")
        return result
    
    # Create target regime
    result['target_regime'] = result.groupby('ticker')['market_regime'].shift(-horizon)
    
    # Create target regime change (1 if regime will change within horizon, 0 otherwise)
    current_regime = result['market_regime']
    future_regimes = pd.DataFrame()
    
    # Check for regime changes within the horizon
    for i in range(1, horizon + 1):
        future_regimes[f'regime_{i}d'] = result.groupby('ticker')['market_regime'].shift(-i)
    
    # Target is 1 if any future regime is different from current
    result['target_regime_change'] = (future_regimes.values != current_regime.values[:, None]).any(axis=1).astype(int)
    
    # Drop rows with NaN targets
    result = result.dropna(subset=['target_regime', 'target_regime_change'])
    
    return result

def create_multitask_dataset(df, ticker=None):
    """
    Create dataset for multitask learning (return, volatility, and regime)
    
    Args:
        df: Input DataFrame
        ticker: Specific ticker to use (None for all)
        
    Returns:
        DataFrame with multiple targets
    """
    # Start with return prediction
    result = create_return_prediction_dataset(df, horizons=[1, 5], ticker=ticker)
    
    # Add volatility prediction
    vol_df = create_volatility_prediction_dataset(df, horizons=[5, 10], ticker=ticker)
    for col in vol_df.columns:
        if col.startswith('target_volatility'):
            result[col] = vol_df[col]
    
    # Add regime prediction if available
    if 'market_regime' in df.columns:
        regime_df = create_regime_prediction_dataset(df, horizon=5, ticker=ticker)
        result['target_regime'] = regime_df['target_regime']
        result['target_regime_change'] = regime_df['target_regime_change']
    
    # Drop rows with NaN targets
    target_cols = [col for col in result.columns if col.startswith('target_')]
    result = result.dropna(subset=target_cols)
    
    return result

def visualize_targets(df, output_path):
    """Create visualizations of targets"""
    # Get target columns
    target_cols = [col for col in df.columns if col.startswith('target_')]
    
    if not target_cols:
        return
    
    # Create figure directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Set number of rows based on number of target categories
    n_rows = len([col for col in target_cols if 'return' in col]) > 0
    n_rows += len([col for col in target_cols if 'volatility' in col]) > 0
    n_rows += len([col for col in target_cols if 'regime' in col]) > 0
    
    # Create figure
    plt.figure(figsize=(15, 5 * n_rows))
    
    row_idx = 0
    
    # Plot return targets
    return_targets = [col for col in target_cols if 'return' in col]
    if return_targets:
        plt.subplot(n_rows, 1, row_idx + 1)
        
        for col in return_targets:
            plt.plot(df['date'], df[col], label=col)
        
        plt.title('Return Targets')
        plt.xlabel('Date')
        plt.ylabel('Return')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        row_idx += 1
    
    # Plot volatility targets
    vol_targets = [col for col in target_cols if 'volatility' in col]
    if vol_targets:
        plt.subplot(n_rows, 1, row_idx + 1)
        
        for col in vol_targets:
            plt.plot(df['date'], df[col], label=col)
        
        plt.title('Volatility Targets')
        plt.xlabel('Date')
        plt.ylabel('Volatility')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        row_idx += 1
    
    # Plot regime targets
    regime_targets = [col for col in target_cols if 'regime' in col and 'change' not in col]
    if regime_targets:
        plt.subplot(n_rows, 1, row_idx + 1)
        
        for col in regime_targets:
            plt.scatter(df['date'], df[col], label=col, alpha=0.5, s=5)
        
        plt.title('Regime Targets')
        plt.xlabel('Date')
        plt.ylabel('Regime')
        plt.yticks(df[regime_targets[0]].unique())
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Create target-specific datasets")
    
    # Required arguments
    parser.add_argument("--input_path", required=True, help="Path to input data file")
    parser.add_argument("--output_dir", required=True, help="Directory to save datasets")
    
    # Optional arguments
    parser.add_argument("--return_horizons", default="1,5,10", help="Return prediction horizons (comma-separated)")
    parser.add_argument("--volatility_horizons", default="5,10,20", help="Volatility prediction horizons (comma-separated)")
    parser.add_argument("--regime_horizon", type=int, default=10, help="Regime prediction horizon")
    parser.add_argument("--ticker", help="Specific ticker to use (default: all)")
    parser.add_argument("--create_all", action="store_true", help="Create all dataset types")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    df = pd.read_csv(args.input_path)
    logger.info(f"Loaded {len(df)} rows from {args.input_path}")
    
    # Convert date column to datetime
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    
    # Parse horizons
    return_horizons = [int(h) for h in args.return_horizons.split(',')]
    volatility_horizons = [int(h) for h in args.volatility_horizons.split(',')]
    
    # Create visualization directory
    viz_dir = os.path.join(args.output_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    
    # Create return prediction dataset
    if args.create_all or True:
        logger.info("Creating return prediction dataset")
        return_df = create_return_prediction_dataset(df, return_horizons, args.ticker)
        return_path = os.path.join(args.output_dir, 'return_prediction.csv')
        return_df.to_csv(return_path, index=False)
        logger.info(f"Saved return prediction dataset to {return_path} with {len(return_df)} rows")
        
        # Visualize
        if args.ticker:
            visualize_targets(return_df, os.path.join(viz_dir, f'return_targets_{args.ticker}.png'))
        else:
            # Visualize for first ticker
            ticker = return_df['ticker'].iloc[0]
            ticker_df = return_df[return_df['ticker'] == ticker]
            visualize_targets(ticker_df, os.path.join(viz_dir, f'return_targets_{ticker}.png'))
    
    # Create volatility prediction dataset
    if args.create_all or True:
        logger.info("Creating volatility prediction dataset")
        vol_df = create_volatility_prediction_dataset(df, volatility_horizons, ticker=args.ticker)
        vol_path = os.path.join(args.output_dir, 'volatility_prediction.csv')
        vol_df.to_csv(vol_path, index=False)
        logger.info(f"Saved volatility prediction dataset to {vol_path} with {len(vol_df)} rows")
        
        # Visualize
        if args.ticker:
            visualize_targets(vol_df, os.path.join(viz_dir, f'volatility_targets_{args.ticker}.png'))
        else:
            # Visualize for first ticker
            ticker = vol_df['ticker'].iloc[0]
            ticker_df = vol_df[vol_df['ticker'] == ticker]
            visualize_targets(ticker_df, os.path.join(viz_dir, f'volatility_targets_{ticker}.png'))
    
    # Create regime prediction dataset
    if (args.create_all or True) and 'market_regime' in df.columns:
        logger.info("Creating regime prediction dataset")
        regime_df = create_regime_prediction_dataset(df, args.regime_horizon, args.ticker)
        regime_path = os.path.join(args.output_dir, 'regime_prediction.csv')
        regime_df.to_csv(regime_path, index=False)
        logger.info(f"Saved regime prediction dataset to {regime_path} with {len(regime_df)} rows")
        
        # Visualize
        if args.ticker:
            visualize_targets(regime_df, os.path.join(viz_dir, f'regime_targets_{args.ticker}.png'))
        else:
            # Visualize for first ticker
            ticker = regime_df['ticker'].iloc[0]
            ticker_df = regime_df[regime_df['ticker'] == ticker]
            visualize_targets(ticker_df, os.path.join(viz_dir, f'regime_targets_{ticker}.png'))
    
    # Create multitask dataset
    if args.create_all or True:
        logger.info("Creating multitask dataset")
        multi_df = create_multitask_dataset(df, args.ticker)
        multi_path = os.path.join(args.output_dir, 'multitask_prediction.csv')
        multi_df.to_csv(multi_path, index=False)
        logger.info(f"Saved multitask dataset to {multi_path} with {len(multi_df)} rows")
        
        # Visualize
        if args.ticker:
            visualize_targets(multi_df, os.path.join(viz_dir, f'multitask_targets_{args.ticker}.png'))
        else:
            # Visualize for first ticker
            ticker = multi_df['ticker'].iloc[0]
            ticker_df = multi_df[multi_df['ticker'] == ticker]
            visualize_targets(ticker_df, os.path.join(viz_dir, f'multitask_targets_{ticker}.png'))

if __name__ == "__main__":
    main()
