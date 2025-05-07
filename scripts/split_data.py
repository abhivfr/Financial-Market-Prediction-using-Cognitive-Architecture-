#!/usr/bin/env python
# split_data.py - Split financial data into train/val/test sets with regime preservation

import os
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import logging
import sys
import subprocess

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def split_time_series(df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, time_col='date'):
    """Split time series data by time"""
    # Ensure ratios sum to 1
    total_ratio = train_ratio + val_ratio + test_ratio
    train_ratio = train_ratio / total_ratio
    val_ratio = val_ratio / total_ratio
    test_ratio = test_ratio / total_ratio
    
    # Sort by time
    df = df.sort_values(time_col).reset_index(drop=True)
    
    # Get split indices
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    # Split data
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    
    return train_df, val_df, test_df

def split_with_regime_preservation(df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, regime_col='market_regime'):
    """Split data while ensuring all regimes are represented in each split"""
    # Check if regime column exists
    if regime_col not in df.columns:
        logger.warning(f"Regime column '{regime_col}' not found, using time-based split")
        return split_time_series(df, train_ratio, val_ratio, test_ratio)
    
    # Get unique regimes
    regimes = df[regime_col].unique()
    
    # Initialize dataframes
    train_dfs = []
    val_dfs = []
    test_dfs = []
    
    # Split each regime separately
    for regime in regimes:
        regime_df = df[df[regime_col] == regime].copy()
        
        # Skip if too few samples
        if len(regime_df) < 10:
            logger.warning(f"Too few samples for regime {regime}, adding all to training")
            train_dfs.append(regime_df)
            continue
        
        # Split by time within each regime
        regime_train, regime_val, regime_test = split_time_series(
            regime_df, train_ratio, val_ratio, test_ratio
        )
        
        train_dfs.append(regime_train)
        val_dfs.append(regime_val)
        test_dfs.append(regime_test)
    
    # Combine regime splits
    train_df = pd.concat(train_dfs).sort_values('date').reset_index(drop=True)
    val_df = pd.concat(val_dfs).sort_values('date').reset_index(drop=True)
    test_df = pd.concat(test_dfs).sort_values('date').reset_index(drop=True)
    
    return train_df, val_df, test_df

def normalize_features(df, scaler=None, return_scaler=False):
    """Normalize numerical features"""
    # Identify numerical columns
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Exclude certain columns from normalization
    exclude_cols = ['date', 'market_regime', 'regime_change', 'ticker']
    norm_cols = [col for col in num_cols if col not in exclude_cols]
    
    # If no scaler provided, create one
    if scaler is None:
        # Calculate mean and std
        means = df[norm_cols].mean()
        stds = df[norm_cols].std()
        
        # Replace zero std with 1
        stds[stds == 0] = 1.0
        
        # Create scaler dictionary
        scaler = {'means': means, 'stds': stds}
    
    # Normalize features
    df_norm = df.copy()
    df_norm[norm_cols] = (df_norm[norm_cols] - scaler['means']) / scaler['stds']
    
    # Replace NaN values
    df_norm = df_norm.fillna(0)
    
    if return_scaler:
        return df_norm, scaler
    else:
        return df_norm

def visualize_splits(train_df, val_df, test_df, output_path, column='price'):
    """Visualize data splits"""
    plt.figure(figsize=(15, 8))
    
    # Plot data
    if 'ticker' in train_df.columns:
        # Plot for a sample ticker (e.g., SPY)
        ticker = 'SPY' if 'SPY' in train_df['ticker'].unique() else train_df['ticker'].iloc[0]
        
        train_ticker = train_df[train_df['ticker'] == ticker]
        val_ticker = val_df[val_df['ticker'] == ticker]
        test_ticker = test_df[test_df['ticker'] == ticker]
        
        plt.plot(train_ticker['date'], train_ticker[column], label=f'Train ({ticker})')
        plt.plot(val_ticker['date'], val_ticker[column], label=f'Validation ({ticker})')
        plt.plot(test_ticker['date'], test_ticker[column], label=f'Test ({ticker})')
    else:
        # Plot all data
        plt.plot(train_df['date'], train_df[column], label='Train')
        plt.plot(val_df['date'], val_df[column], label='Validation')
        plt.plot(test_df['date'], test_df[column], label='Test')
    
    plt.title(f'Data Splits - {column}')
    plt.xlabel('Date')
    plt.ylabel(column)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def visualize_regime_distribution(train_df, val_df, test_df, output_path, regime_col='market_regime'):
    """Visualize regime distribution in each split"""
    # Check if regime column exists
    if regime_col not in train_df.columns:
        return
    
    # Count regimes in each split
    train_counts = train_df[regime_col].value_counts().sort_index()
    val_counts = val_df[regime_col].value_counts().sort_index()
    test_counts = test_df[regime_col].value_counts().sort_index()
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Get all regimes
    all_regimes = sorted(set(train_counts.index) | set(val_counts.index) | set(test_counts.index))
    
    # Ensure all regimes exist in all counts
    for regime in all_regimes:
        if regime not in train_counts:
            train_counts[regime] = 0
        if regime not in val_counts:
            val_counts[regime] = 0
        if regime not in test_counts:
            test_counts[regime] = 0
    
    # Sort counts
    train_counts = train_counts.sort_index()
    val_counts = val_counts.sort_index()
    test_counts = test_counts.sort_index()
    
    # Calculate percentages
    train_pct = train_counts / train_counts.sum() * 100
    val_pct = val_counts / val_counts.sum() * 100
    test_pct = test_counts / test_counts.sum() * 100
    
    # Bar positions
    x = np.arange(len(all_regimes))
    width = 0.25
    
    # Plot bars
    plt.bar(x - width, train_pct, width, label='Train')
    plt.bar(x, val_pct, width, label='Validation')
    plt.bar(x + width, test_pct, width, label='Test')
    
    # Add labels and legend
    plt.title('Regime Distribution in Data Splits')
    plt.xlabel('Market Regime')
    plt.ylabel('Percentage')
    plt.xticks(x, [f'Regime {r}' for r in all_regimes])
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add count annotations
    for i, regime in enumerate(all_regimes):
        plt.text(i - width, train_pct[regime] + 1, f'{train_counts[regime]}', 
                ha='center', va='bottom', fontsize=8)
        plt.text(i, val_pct[regime] + 1, f'{val_counts[regime]}', 
                ha='center', va='bottom', fontsize=8)
        plt.text(i + width, test_pct[regime] + 1, f'{test_counts[regime]}', 
                ha='center', va='bottom', fontsize=8)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def save_splits(train_df, val_df, test_df, output_dir, normalize=True):
    """Save data splits to CSV files"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Normalize if requested
    if normalize:
        train_df_norm, scaler = normalize_features(train_df, return_scaler=True)
        val_df_norm = normalize_features(val_df, scaler)
        test_df_norm = normalize_features(test_df, scaler)
        
        # Save normalization parameters
        import json
        scaler_path = os.path.join(output_dir, 'normalization_params.json')
        with open(scaler_path, 'w') as f:
            json.dump({
                'means': scaler['means'].to_dict(),
                'stds': scaler['stds'].to_dict()
            }, f, indent=2)
        
        # Save normalized data
        train_df_norm.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
        val_df_norm.to_csv(os.path.join(output_dir, 'val.csv'), index=False)
        test_df_norm.to_csv(os.path.join(output_dir, 'test.csv'), index=False)
        
        logger.info(f"Saved normalized splits to {output_dir}")
    else:
        # Save raw data
        train_df.to_csv(os.path.join(output_dir, 'train_raw.csv'), index=False)
        val_df.to_csv(os.path.join(output_dir, 'val_raw.csv'), index=False)
        test_df.to_csv(os.path.join(output_dir, 'test_raw.csv'), index=False)
        
        logger.info(f"Saved raw splits to {output_dir}")
    
    # Create visualizations
    visuals_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(visuals_dir, exist_ok=True)
    
    # Visualize price splits
    visualize_splits(
        train_df, val_df, test_df,
        os.path.join(visuals_dir, 'price_splits.png'),
        column='price'
    )
    
    # Visualize return splits
    if 'return_1d' in train_df.columns:
        visualize_splits(
            train_df, val_df, test_df,
            os.path.join(visuals_dir, 'return_splits.png'),
            column='return_1d'
        )
    
    # Visualize volatility splits
    if 'volatility_20d' in train_df.columns:
        visualize_splits(
            train_df, val_df, test_df,
            os.path.join(visuals_dir, 'volatility_splits.png'),
            column='volatility_20d'
        )
    
    # Visualize regime distribution
    visualize_regime_distribution(
        train_df, val_df, test_df,
        os.path.join(visuals_dir, 'regime_distribution.png')
    )

    split_viz_path = os.path.join(output_dir, "split_visualization.png")
    visualize_splits(train_df, val_df, test_df, split_viz_path, column='price')
    print("Saved split visualization to", split_viz_path)

def split_data(
    input_path, 
    output_dir, 
    train_ratio=0.7, 
    val_ratio=0.15, 
    test_ratio=0.15,
    split_mode='time',
    normalize=True,
    ensure_regime_coverage=False,
    single_ticker=None
):
    """Split financial data into train/val/test sets"""
    # Load data
    df = pd.read_csv(input_path)
    logger.info(f"Loaded {len(df)} rows from {input_path}")
    
    # Convert date column to datetime
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    
    # Filter to single ticker if specified
    if single_ticker is not None and 'ticker' in df.columns:
        if single_ticker in df['ticker'].unique():
            df = df[df['ticker'] == single_ticker].reset_index(drop=True)
            logger.info(f"Filtered to {len(df)} rows for ticker {single_ticker}")
        else:
            logger.warning(f"Ticker {single_ticker} not found, using all data")
    
    # Split data based on mode
    if split_mode == 'random':
        # Random split (not recommended for time series)
        train_df, temp_df = train_test_split(df, test_size=val_ratio+test_ratio, random_state=42)
        val_df, test_df = train_test_split(temp_df, test_size=test_ratio/(val_ratio+test_ratio), random_state=42)
    elif split_mode == 'time':
        # Time-based split
        train_df, val_df, test_df = split_time_series(df, train_ratio, val_ratio, test_ratio)
    elif split_mode == 'regime':
        # Regime-based split
        train_df, val_df, test_df = split_with_regime_preservation(df, train_ratio, val_ratio, test_ratio)
    else:
        logger.error(f"Invalid split mode: {split_mode}")
        return
    
    # Ensure regime coverage if requested
    if ensure_regime_coverage and 'market_regime' in df.columns:
        # Get all regimes
        all_regimes = df['market_regime'].unique()
        
        # Check if all regimes are in all splits
        for split_name, split_df in [('train', train_df), ('val', val_df), ('test', test_df)]:
            split_regimes = split_df['market_regime'].unique()
            missing_regimes = set(all_regimes) - set(split_regimes)
            
            if missing_regimes:
                logger.warning(f"Regimes {missing_regimes} missing from {split_name} split")
                
                # Ensure at least some samples of each regime in each split
                for regime in missing_regimes:
                    # Get samples from this regime
                    regime_samples = df[df['market_regime'] == regime]
                    
                    # Add a few samples to the split
                    n_samples = min(10, len(regime_samples))
                    
                    if n_samples > 0:
                        # Take samples from beginning, middle, and end
                        indices = [
                            int(i * len(regime_samples) / (n_samples - 1))
                            for i in range(n_samples)
                        ]
                        samples_to_add = regime_samples.iloc[indices].copy()
                        
                        # Add to split
                        if split_name == 'train':
                            train_df = pd.concat([train_df, samples_to_add])
                        elif split_name == 'val':
                            val_df = pd.concat([val_df, samples_to_add])
                        else:  # test
                            test_df = pd.concat([test_df, samples_to_add])
                        
                        logger.info(f"Added {n_samples} samples of regime {regime} to {split_name} split")
    
    # Sort and reset indices
    train_df = train_df.sort_values('date').reset_index(drop=True)
    val_df = val_df.sort_values('date').reset_index(drop=True)
    test_df = test_df.sort_values('date').reset_index(drop=True)
    
    # Save splits
    save_splits(train_df, val_df, test_df, output_dir, normalize)
    
    # Log split sizes
    logger.info(f"Train: {len(train_df)} rows ({len(train_df)/len(df):.1%})")
    logger.info(f"Validation: {len(val_df)} rows ({len(val_df)/len(df):.1%})")
    logger.info(f"Test: {len(test_df)} rows ({len(test_df)/len(df):.1%})")
    
    return train_df, val_df, test_df

def main():
    parser = argparse.ArgumentParser(description="Split financial data into train/val/test sets")
    
    # Required arguments
    parser.add_argument("--input_path", required=True, help="Path to input data file")
    parser.add_argument("--output_dir", required=True, help="Directory to save splits")
    
    # Optional arguments
    parser.add_argument("--train_ratio", type=float, default=0.7, help="Training set ratio")
    parser.add_argument("--val_ratio", type=float, default=0.15, help="Validation set ratio")
    parser.add_argument("--test_ratio", type=float, default=0.15, help="Test set ratio")
    parser.add_argument("--split_mode", choices=['time', 'random', 'regime'], default='time', 
                      help="Split mode (time, random, or regime)")
    parser.add_argument("--normalize", action="store_true", help="Normalize features")
    parser.add_argument("--ensure_regime_coverage", action="store_true", 
                      help="Ensure all regimes are represented in all splits")
    parser.add_argument("--single_ticker", help="Split data for a single ticker")
    
    args = parser.parse_args()
    
    # Check ratios
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 0.01:
        logger.warning(f"Ratios sum to {total_ratio}, normalizing to 1.0")
        args.train_ratio /= total_ratio
        args.val_ratio /= total_ratio
        args.test_ratio /= total_ratio
    
    # Split data
    split_data(
        args.input_path,
        args.output_dir,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        args.split_mode,
        args.normalize,
        args.ensure_regime_coverage,
        args.single_ticker
    )

if __name__ == "__main__":
    main()
