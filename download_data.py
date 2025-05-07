#!/usr/bin/env python
# Enhanced download_data.py - Download multiple financial assets with additional features

import os
import argparse
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def download_ticker_data(ticker, start_date, end_date, interval='1d'):
    """
    Download data for a specific ticker
    
    Args:
        ticker: Ticker symbol
        start_date: Start date
        end_date: End date
        interval: Data interval ('1d', '1wk', etc.)
        
    Returns:
        DataFrame with ticker data
    """
    logger.info(f"Downloading {ticker} data from {start_date} to {end_date}")
    
    try:
        result = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            interval=interval,
            auto_adjust=True,
            progress=False
        )
        if isinstance(result, tuple):
            data = result[0]
        else:
            data = result

        if not isinstance(data, pd.DataFrame):
            logger.error(f"yfinance did not return a DataFrame for {ticker}, got {type(data)}")
            return pd.DataFrame()

        if data.empty:
            logger.warning(f"No data returned for {ticker}")
            return pd.DataFrame()

        data.reset_index(inplace=True)
        # Flatten MultiIndex columns if present and standardize to lower case
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0].lower() for col in data.columns]
        else:
            data.columns = [col.lower() for col in data.columns]
        # Rename columns to match loader expectations
        rename_map = {
            'close': 'price',
            'volume': 'volume',
            # Add more if needed
        }
        data = data.rename(columns=rename_map)
        data['ticker'] = ticker
        return data

    except Exception as e:
        logger.error(f"Error downloading {ticker}: {str(e)}")
        return pd.DataFrame()

def calculate_features(df):
    """
    Calculate financial features
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        DataFrame with additional features
    """
    # Make a copy
    df_features = df.copy()
    
    # Calculate returns
    for period in [1, 3, 5, 10, 20]:
        df_features[f'return_{period}d'] = df_features['price'].pct_change(period)
    
    # Calculate log returns
    df_features['log_return_1d'] = np.log(df_features['price'] / df_features['price'].shift(1))
    
    # Calculate volatility
    for period in [5, 10, 20, 30]:
        df_features[f'volatility_{period}d'] = df_features['log_return_1d'].rolling(period).std() * np.sqrt(252)
    
    # Calculate moving averages
    for period in [5, 10, 20, 50, 200]:
        df_features[f'ma_{period}'] = df_features['price'].rolling(period).mean()
    
    # Calculate RSI
    delta = df_features['price'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / loss
    df_features['rsi_14'] = 100 - (100 / (1 + rs))
    
    # Calculate MACD
    ema12 = df_features['price'].ewm(span=12, adjust=False).mean()
    ema26 = df_features['price'].ewm(span=26, adjust=False).mean()
    df_features['macd'] = ema12 - ema26
    df_features['macd_signal'] = df_features['macd'].ewm(span=9, adjust=False).mean()
    
    # Calculate ATR
    high_low = df_features['high'] - df_features['low']
    high_close = np.abs(df_features['high'] - df_features['price'].shift())
    low_close = np.abs(df_features['low'] - df_features['price'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df_features['atr_14'] = tr.rolling(14).mean()
    
    # Calculate momentum
    for period in [5, 10, 20, 60]:
        df_features[f'momentum_{period}d'] = df_features['price'] / df_features['price'].shift(period) - 1
    
    # Fill NaN values
    df_features = df_features.bfill().ffill()
    
    return df_features

def detect_basic_regimes(df):
    """
    Detect basic market regimes
    
    Args:
        df: DataFrame with features
        
    Returns:
        DataFrame with regime labels
    """
    # Make a copy
    result = df.copy()
    
    # Calculate regime based on volatility and return
    if 'volatility_20d' in result.columns and 'return_20d' in result.columns:
        # Normalize metrics
        vol = result['volatility_20d']
        returns = result['return_20d']
        
        vol_norm = (vol - vol.mean()) / vol.std()
        returns_norm = (returns - returns.mean()) / returns.std()
        
        # Create regimes
        # 0: Low vol, low returns (range-bound)
        # 1: Low vol, high returns (bull)
        # 2: High vol, low returns (bear)
        # 3: High vol, high returns (volatile bull)
        result['market_regime'] = 0
        result.loc[(vol_norm <= 0) & (returns_norm > 0), 'market_regime'] = 1  # Bull
        result.loc[(vol_norm > 0) & (returns_norm <= 0), 'market_regime'] = 2  # Bear
        result.loc[(vol_norm > 0) & (returns_norm > 0), 'market_regime'] = 3  # Volatile bull
        
        # Add regime change indicator
        result['regime_change'] = result['market_regime'].diff().ne(0)
    
    return result

def add_market_context(df, spy_data):
    """
    Add market context (SPY data) to ticker data
    
    Args:
        df: DataFrame with ticker data
        spy_data: DataFrame with SPY data
        
    Returns:
        DataFrame with market context
    """
    # Make a copy
    result = df.copy()
    
    # Merge with SPY data on date
    spy_cols = [
        'date', 
        'return_1d', 'return_5d', 'return_20d',
        'volatility_20d',
        'market_regime'
    ]
    
    spy_subset = spy_data[spy_cols].copy()
    
    # Rename columns to indicate they're from SPY
    rename_dict = {col: f'spy_{col}' for col in spy_cols if col != 'date'}
    spy_subset = spy_subset.rename(columns=rename_dict)
    
    # Merge
    result = pd.merge(result, spy_subset, on='date', how='left')
    
    # Calculate relative strength vs SPY
    result['rel_strength_spy'] = result['return_20d'] - result['spy_return_20d']
    
    return result

def main():
    parser = argparse.ArgumentParser(description="Download and process financial data")
    
    # Required arguments
    parser.add_argument("--tickers", required=True, help="Comma-separated list of ticker symbols")
    parser.add_argument("--start_date", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end_date", required=True, help="End date (YYYY-MM-DD)")
    
    # Optional arguments
    parser.add_argument("--output_dir", default="data/raw", help="Output directory")
    parser.add_argument("--interval", default="1d", help="Data interval (1d, 1wk, etc.)")
    parser.add_argument("--include_features", action="store_true", help="Calculate additional features")
    parser.add_argument("--detect_regimes", action="store_true", help="Detect market regimes")
    parser.add_argument("--add_market_context", action="store_true", help="Add market context (SPY data)")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Parse tickers
    tickers = [t.strip() for t in args.tickers.split(',')]
    
    # Always include SPY if market context is requested
    if args.add_market_context and 'SPY' not in tickers:
        tickers.append('SPY')
    
    # Download SPY data if needed
    spy_data = None
    if 'SPY' in tickers or args.add_market_context:
        spy_data = download_ticker_data('SPY', args.start_date, args.end_date, args.interval)
        # Standardize columns to lowercase
        spy_data.columns = [col.lower() for col in spy_data.columns]
        if 'price' not in spy_data.columns:
            logger.error("Downloaded SPY data is missing 'price' column.")
            return  # Stop further processing if SPY data is invalid

        if args.include_features:
            spy_data = calculate_features(spy_data)
        if args.detect_regimes:
            spy_data = detect_basic_regimes(spy_data)
        # Save SPY data if it's in the tickers list
        if 'SPY' in tickers:
            spy_output_path = os.path.join(args.output_dir, 'SPY.csv')
            spy_data.to_csv(spy_output_path, index=False)
            logger.info(f"Saved SPY data to {spy_output_path}")

        print("SPY data type:", type(spy_data))

    # Download data for each ticker
    for ticker in tqdm(tickers):
        if ticker == 'SPY' and spy_data is not None:
            # Skip SPY if already downloaded
            continue

        try:
            # Download data
            data = download_ticker_data(ticker, args.start_date, args.end_date, args.interval)
            print(f"{ticker} data type:", type(data))
            # Standardize columns to lowercase
            data.columns = [col.lower() for col in data.columns]
            if 'price' not in data.columns:
                logger.error(f"Downloaded data for {ticker} is missing 'price' column.")
                continue

            if data.empty:
                logger.warning(f"No data found for {ticker}")
                continue

            # Calculate features
            if args.include_features:
                data = calculate_features(data)

            # Detect regimes
            if args.detect_regimes:
                data = detect_basic_regimes(data)

            # Add market context
            if args.add_market_context and spy_data is not None:
                data = add_market_context(data, spy_data)

            # Save data
            output_path = os.path.join(args.output_dir, f'{ticker}.csv')
            data.to_csv(output_path, index=False)

            logger.info(f"Saved {ticker} data to {output_path}")

        except Exception as e:
            logger.error(f"Error processing {ticker}: {str(e)}")

    logger.info("Data download and processing completed")

if __name__ == "__main__":
    main()
