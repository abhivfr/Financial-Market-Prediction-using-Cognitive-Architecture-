#!/usr/bin/env python
# build_dataset.py - Create unified financial dataset from multiple tickers

import os
import argparse
import pandas as pd
import numpy as np
from glob import glob
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_ticker_data(file_path):
    """Load ticker data from CSV file"""
    df = pd.read_csv(file_path)
    
    # Add ticker name from filename
    ticker = os.path.basename(file_path).split('.')[0]
    if 'ticker' not in df.columns:
        df['ticker'] = ticker
        
    return df

def create_unified_dataset(input_dir, output_path, min_date=None, max_date=None):
    """Create unified dataset from multiple ticker files"""
    # Find all CSV files
    csv_files = glob(os.path.join(input_dir, "*.csv"))
    
    if not csv_files:
        logger.error(f"No CSV files found in {input_dir}")
        return None
        
    # Load all ticker data
    all_dfs = []
    for file_path in tqdm(csv_files, desc="Loading data files"):
        try:
            df = load_ticker_data(file_path)
            
            # Apply date filters if specified
            if min_date is not None and 'date' in df.columns:
                df = df[df['date'] >= min_date]
                
            if max_date is not None and 'date' in df.columns:
                df = df[df['date'] <= max_date]
                
            all_dfs.append(df)
        except Exception as e:
            logger.error(f"Error loading {file_path}: {str(e)}")
    
    if not all_dfs:
        logger.error("No valid data files loaded")
        return None
        
    # Combine into unified dataset
    unified_df = pd.concat(all_dfs, ignore_index=True)
    
    # Sort by date and ticker
    if 'date' in unified_df.columns:
        unified_df['date'] = pd.to_datetime(unified_df['date'])
        unified_df = unified_df.sort_values(['date', 'ticker']).reset_index(drop=True)
    
    # Save unified dataset
    unified_df.to_csv(output_path, index=False)
    logger.info(f"Saved unified dataset to {output_path} with {len(unified_df)} rows")
    
    return unified_df

def find_common_dates(df):
    """Find dates that exist for all tickers"""
    date_counts = df.groupby('date')['ticker'].nunique()
    ticker_count = df['ticker'].nunique()
    common_dates = date_counts[date_counts == ticker_count].index
    
    return common_dates

def create_balanced_dataset(df, output_path=None):
    """Create balanced dataset with only dates that exist for all tickers"""
    # Find common dates
    common_dates = find_common_dates(df)
    
    # Filter to common dates
    balanced_df = df[df['date'].isin(common_dates)].copy()
    
    # Save if output path is specified
    if output_path:
        balanced_df.to_csv(output_path, index=False)
        logger.info(f"Saved balanced dataset to {output_path} with {len(balanced_df)} rows")
    
    return balanced_df

def add_market_context_features(df, spy_file=None):
    """Add market-wide context features"""
    result = df.copy()
    
    # If SPY file is provided, use it
    if spy_file and os.path.exists(spy_file):
        spy_df = pd.read_csv(spy_file)
        spy_df['date'] = pd.to_datetime(spy_df['date'])
        
        # Select market context columns
        context_cols = ['date', 'return_1d', 'return_5d', 'return_20d', 
                        'volatility_20d', 'rsi_14', 'market_regime']
        
        spy_context = spy_df[context_cols].copy()
        
        # Rename columns to indicate they're market-wide
        rename_dict = {col: f'market_{col}' for col in context_cols if col != 'date'}
        spy_context = spy_context.rename(columns=rename_dict)
        
        # Merge with main dataframe
        result = pd.merge(result, spy_context, on='date', how='left')
    else:
        # Calculate market averages for each date
        date_groups = result.groupby('date')
        
        # Create new DataFrame with market-wide metrics
        market_metrics = []
        
        for date, group in tqdm(date_groups, desc="Calculating market metrics"):
            metrics = {
                'date': date,
                'market_return_1d': group['return_1d'].mean(),
                'market_return_5d': group['return_5d'].mean(),
                'market_return_20d': group['return_20d'].mean(),
                'market_volatility_20d': group['volatility_20d'].mean()
            }
            
            # Add regime if available
            if 'market_regime' in group.columns:
                # Use most common regime
                metrics['market_regime'] = group['market_regime'].mode()[0]
            
            market_metrics.append(metrics)
        
        market_df = pd.DataFrame(market_metrics)
        
        # Merge with main dataframe
        result = pd.merge(result, market_df, on='date', how='left')
    
    # Calculate relative metrics (stock vs. market)
    if 'market_return_20d' in result.columns:
        result['rel_strength_market'] = result['return_20d'] - result['market_return_20d']
    
    if 'market_volatility_20d' in result.columns:
        result['rel_volatility_market'] = result['volatility_20d'] / result['market_volatility_20d']
    
    return result

def main():
    parser = argparse.ArgumentParser(description="Create unified financial dataset")
    
    # Required arguments
    parser.add_argument("--input_dir", required=True, help="Directory with ticker CSV files")
    parser.add_argument("--output_path", required=True, help="Output path for unified dataset")
    
    # Optional arguments
    parser.add_argument("--min_date", help="Minimum date to include (YYYY-MM-DD)")
    parser.add_argument("--max_date", help="Maximum date to include (YYYY-MM-DD)")
    parser.add_argument("--create_balanced", action="store_true", help="Create balanced dataset with common dates")
    parser.add_argument("--balanced_output", help="Output path for balanced dataset")
    parser.add_argument("--add_market_context", action="store_true", help="Add market-wide context features")
    parser.add_argument("--spy_file", help="Path to SPY CSV file")
    
    args = parser.parse_args()
    
    # Create unified dataset
    df = create_unified_dataset(
        args.input_dir, 
        args.output_path,
        min_date=args.min_date,
        max_date=args.max_date
    )
    
    if df is None:
        return
    
    # Add market context if requested
    if args.add_market_context:
        df = add_market_context_features(df, args.spy_file)
        
        # Save updated dataset
        df.to_csv(args.output_path, index=False)
        logger.info(f"Added market context features to {args.output_path}")
    
    # Create balanced dataset if requested
    if args.create_balanced:
        balanced_output = args.balanced_output or args.output_path.replace('.csv', '_balanced.csv')
        balanced_df = create_balanced_dataset(df, balanced_output)
        logger.info(f"Created balanced dataset with {len(balanced_df)} rows")

if __name__ == "__main__":
    main()
