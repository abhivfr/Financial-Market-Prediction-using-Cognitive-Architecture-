#!/usr/bin/env python
# enhance_features.py - Add technical features to financial data

import os
import sys
import argparse
import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any
import glob
from tqdm import tqdm

# Technical analysis libraries
import talib as ta
from hmmlearn.hmm import GaussianHMM

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, root_dir)

# Try to import pandas_ta as a fallback if talib functions are missing
try:
    import pandas_ta as pta
    HAS_PANDAS_TA = True
except ImportError:
    HAS_PANDAS_TA = False

def calculate_technical_features(df):
    """
    Calculate technical indicators for financial data.
    
    This function handles different input formats by checking for required columns
    and only calculating indicators for which data is available.
    
    Args:
        df: DataFrame with price data
        
    Returns:
        DataFrame with added technical indicators
    """
    # Make a copy to avoid SettingWithCopyWarning
    df = df.copy()
    
    # Convert column names to lowercase for consistency
    df.columns = [col.lower() for col in df.columns]
    
    # Check which columns are available
    has_ohlc = all(col in df.columns for col in ['open', 'high', 'low', 'close'])
    has_close = 'close' in df.columns
    has_price = any(col in df.columns for col in ['close', 'price', 'value', 'adj_close'])
    has_volume = 'volume' in df.columns
    
    # Identify the price column to use
    price_column = None
    for col in ['close', 'price', 'adj_close', 'value']:
        if col in df.columns:
            price_column = col
            break
    
    if price_column is None and df.select_dtypes(include=['float64', 'float32', 'int64', 'int32']).columns.any():
        # Find a numeric column that might represent price
        numeric_cols = df.select_dtypes(include=['float64', 'float32', 'int64', 'int32']).columns
        # Exclude columns that are clearly not price
        non_price_keywords = ['volume', 'qty', 'quantity', 'count', 'return', 'pct', 'sentiment', 'score']
        potential_price_cols = [col for col in numeric_cols if not any(kw in col.lower() for kw in non_price_keywords)]
        
        if potential_price_cols:
            price_column = potential_price_cols[0]
            print(f"Using '{price_column}' as price column for technical indicators")
    
    # OHLC-based indicators
    if has_ohlc:
        try:
            # Use pandas_ta if available, otherwise try talib functions
            if HAS_PANDAS_TA:
                # Add trend indicators
                df['sma_20'] = pta.sma(df['close'], length=20)
                df['sma_50'] = pta.sma(df['close'], length=50)
                df['sma_200'] = pta.sma(df['close'], length=200)
                
                # Add momentum indicators
                df['rsi_14'] = pta.rsi(df['close'], length=14)
                macd = pta.macd(df['close'])
                df['macd_line'] = macd['MACD_12_26_9']
                df['macd_signal'] = macd['MACDs_12_26_9']
                df['macd_histogram'] = macd['MACDh_12_26_9']
                
                # Add volatility indicators 
                bbands = pta.bbands(df['close'])
                df['bbands_upper'] = bbands['BBU_5_2.0']
                df['bbands_middle'] = bbands['BBM_5_2.0'] 
                df['bbands_lower'] = bbands['BBL_5_2.0']
                df['atr_14'] = pta.atr(df['high'], df['low'], df['close'])
                
                # Add additional indicators
                stoch = pta.stoch(df['high'], df['low'], df['close'])
                df['stoch_k'] = stoch['STOCHk_14_3_3']
                df['stoch_d'] = stoch['STOCHd_14_3_3']
                df['adx'] = pta.adx(df['high'], df['low'], df['close'])['ADX_14']
            else:
                # Traditional talib approach - check each function call
                try:
                    # Try different talib function conventions
                    # Some versions use ta.function while others use ta.Function
                    if hasattr(ta, 'SMA'):
                        df['sma_20'] = ta.SMA(df['close'], timeperiod=20)
                        df['sma_50'] = ta.SMA(df['close'], timeperiod=50)
                        df['sma_200'] = ta.SMA(df['close'], timeperiod=200)
                        df['rsi_14'] = ta.RSI(df['close'], timeperiod=14)
    
    # MACD
                        macd, signal, hist = ta.MACD(df['close'])
                        df['macd_line'] = macd
                        df['macd_signal'] = signal
                        df['macd_histogram'] = hist
    
    # Bollinger Bands
                        upper, middle, lower = ta.BBANDS(df['close'])
                        df['bbands_upper'] = upper
                        df['bbands_middle'] = middle
                        df['bbands_lower'] = lower
                        
                        # ATR
                        df['atr_14'] = ta.ATR(df['high'], df['low'], df['close'], timeperiod=14)
                        
                        # Stochastic
                        slowk, slowd = ta.STOCH(df['high'], df['low'], df['close'])
                        df['stoch_k'] = slowk
                        df['stoch_d'] = slowd
                        
                        # ADX
                        df['adx'] = ta.ADX(df['high'], df['low'], df['close'], timeperiod=14)
                    else:
                        # Fallback to basic calculations
                        df['sma_20'] = df['close'].rolling(window=20).mean()
                        df['sma_50'] = df['close'].rolling(window=50).mean()
                        df['sma_200'] = df['close'].rolling(window=200).mean()
                except Exception as e:
                    print(f"Error calculating OHLC indicators using talib: {str(e)}")
                    # Fallback to basic calculations
                    df['sma_20'] = df['close'].rolling(window=20).mean()
                    df['sma_50'] = df['close'].rolling(window=50).mean()
                    df['sma_200'] = df['close'].rolling(window=200).mean()
        except Exception as e:
            print(f"Error calculating OHLC-based indicators: {str(e)}")
    
    # Price-based indicators (using identified price column)
    elif price_column is not None:
        try:
            # Basic indicators using pandas
            df['sma_20'] = df[price_column].rolling(window=20).mean()
            df['sma_50'] = df[price_column].rolling(window=50).mean()
            df['sma_200'] = df[price_column].rolling(window=200).mean()
            
            # Try to calculate RSI using pandas
            delta = df[price_column].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss
            df['rsi_14'] = 100 - (100 / (1 + rs))
            
            # Simple volatility proxy
            df['rolling_std_20'] = df[price_column].rolling(window=20).std()
            
            print(f"Added price-based indicators using {price_column} column")
        except Exception as e:
            print(f"Error calculating price-based indicators: {str(e)}")
    else:
        print("Warning: No suitable price column found. Skipping technical indicators.")
    
    # Volume-based indicators
    if has_volume and has_price:
        try:
            # Simple volume indicators using pandas
            df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma_20']
            
            # Try to calculate OBV using pandas
            price_col = 'close' if has_close else price_column
            df['obv'] = (np.sign(df[price_col].diff()) * df['volume']).fillna(0).cumsum()
        except Exception as e:
            print(f"Error calculating volume indicators: {str(e)}")
    
    # Calculate returns if we have a price column
    if price_column is not None:
        try:
            # Calculate returns
            df['daily_return'] = df[price_column].pct_change()
            df['log_return'] = np.log(df[price_column] / df[price_column].shift(1))
            
            # Calculate volatility metrics
            df['rolling_vol_20'] = df['daily_return'].rolling(window=20).std()
            df['rolling_vol_50'] = df['daily_return'].rolling(window=50).std()
        except Exception as e:
            print(f"Error calculating return metrics: {str(e)}")
    
    return df

def add_sentiment_features(df: pd.DataFrame, sentiment_path: Optional[str] = None) -> pd.DataFrame:
    """
    Add sentiment features if available
    
    Args:
        df: DataFrame with financial data
        sentiment_path: Path to sentiment data (optional)
        
    Returns:
        DataFrame with sentiment features
    """
    result = df.copy()
    
    if sentiment_path and os.path.exists(sentiment_path):
        print(f"Adding sentiment features from {sentiment_path}")
        
        # Load sentiment data
        sentiment_df = pd.read_csv(sentiment_path)
        
        # Ensure date column exists
        if 'date' not in sentiment_df.columns:
            print("Warning: Sentiment data missing date column, skipping")
            return result
        
        # Convert date to datetime
        if 'date' in result.columns:
            result['date'] = pd.to_datetime(result['date'])
            sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
            
            # Merge sentiment data
            result = pd.merge(result, sentiment_df, on='date', how='left')
            
            # Fill NaN sentiment values
            sentiment_columns = [col for col in sentiment_df.columns if col != 'date']
            result[sentiment_columns] = result[sentiment_columns].ffill().fillna(0)
    else:
        print("No sentiment data provided or file not found, skipping sentiment features")
    
    return result

def detect_market_regimes(df):
    """
    Detect market regimes using Hidden Markov Models.
    
    This function can work with different data formats by identifying a suitable
    price/return column to use for regime detection.
    
    Args:
        df: DataFrame with financial time series data
    
    Returns:
        DataFrame with market regime labels added
    """
    # Make a copy to avoid SettingWithCopyWarning
    df = df.copy()
    
    # Convert column names to lowercase for consistency
    df.columns = [col.lower() for col in df.columns]
    
    # First, check if regime columns already exist
    if 'market_regime' in df.columns or 'regime' in df.columns:
        existing_regime_col = 'market_regime' if 'market_regime' in df.columns else 'regime'
        print(f"Market regimes already detected in column '{existing_regime_col}'. Skipping detection.")
        return df
    
    # Find the best column to use for regime detection
    ret_column = None
    
    # First priority: existing return columns
    for col in ['daily_return', 'return', 'returns', 'log_return', 'log_returns']:
        if col in df.columns and df[col].notna().sum() > len(df) * 0.7:  # At least 70% non-NA values
            ret_column = col
            break
    
    # Second priority: calculate returns from price columns
    if ret_column is None:
        price_column = None
        for col in ['close', 'price', 'adj_close', 'value']:
            if col in df.columns and df[col].notna().sum() > len(df) * 0.7:
                price_column = col
                break
        
        if price_column is None and df.select_dtypes(include=['float64', 'float32', 'int64', 'int32']).columns.any():
            # Find a numeric column that might represent price
            numeric_cols = df.select_dtypes(include=['float64', 'float32', 'int64', 'int32']).columns
            # Exclude columns that are clearly not price
            non_price_keywords = ['volume', 'qty', 'quantity', 'count', 'regime', 'state', 'sentiment', 'score']
            potential_price_cols = [col for col in numeric_cols if not any(kw in col.lower() for kw in non_price_keywords)]
            
            if potential_price_cols:
                price_column = potential_price_cols[0]
                print(f"Using '{price_column}' as price column for regime detection")
        
        # Calculate returns if we found a suitable price column
        if price_column is not None:
            try:
                df['temp_returns'] = df[price_column].pct_change()
                ret_column = 'temp_returns'
            except Exception as e:
                print(f"Error calculating returns from {price_column}: {str(e)}")
    
    if ret_column is None:
        print("Warning: No suitable return or price column found for regime detection. Skipping.")
        return df
    
    try:
        print(f"Detecting market regimes using '{ret_column}' column...")
        
        # Drop NaN values for model fitting
        returns_data = df[ret_column].dropna().values.reshape(-1, 1)
        
        # Skip if we don't have enough data
        if len(returns_data) < 30:  # Minimum data requirements
            print("Warning: Not enough data for regime detection (minimum 30 data points required)")
            return df
        
        # Fit Hidden Markov Model with 2 regimes
        hmm_model = GaussianHMM(n_components=2, random_state=42, n_iter=1000)
        hmm_model.fit(returns_data)
        
        # Get regime classifications
        regimes = hmm_model.predict(returns_data)
        
        # Identify the bull and bear regimes based on mean returns
        regime_0_mean = np.mean(returns_data[regimes == 0])
        regime_1_mean = np.mean(returns_data[regimes == 1])
        
        bull_regime = 0 if regime_0_mean > regime_1_mean else 1
        bear_regime = 1 if bull_regime == 0 else 0
        
        # Create a mapping dictionary for clarity
        regime_mapping = {
            bull_regime: 'bull',
            bear_regime: 'bear'
        }
        
        # Map numeric regimes to labels
        regime_labels = np.array([regime_mapping[r] for r in regimes])
        
        # Create a DataFrame with the dates from the original returns and the regime labels
        regime_df = pd.DataFrame({
            'date': df.index[df[ret_column].notna()],
            'market_regime': regime_labels
        })
        
        # Set the date as index in the regime DataFrame
        regime_df.set_index('date', inplace=True)
        
        # Merge the regime labels with the original DataFrame
        # First convert string regime labels to numeric for smoother operations
        regime_df['market_regime_num'] = (regime_df['market_regime'] == 'bull').astype(int)
        
        # Add the regimes to the original DataFrame
        if isinstance(df.index, pd.DatetimeIndex):
            # If index is DatetimeIndex, use it directly
            df['market_regime_num'] = regime_df['market_regime_num']
            df['market_regime'] = regime_df['market_regime']
        else:
            # Find date column in original df
            date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
            
            if date_cols and pd.api.types.is_datetime64_any_dtype(df[date_cols[0]]):
                # Set temporary index for merging
                original_index = df.index
                df = df.set_index(date_cols[0])
                
                # Join regimes
                df = df.join(regime_df[['market_regime', 'market_regime_num']], how='left')
                
                # Reset to original index
                df = df.reset_index().set_index(original_index)
            else:
                # If we can't find a date column, resort to position-based assignment
                # Note: This is less precise as it assumes the order is preserved
                valid_indices = df[ret_column].notna()
                df.loc[valid_indices, 'market_regime_num'] = regime_df['market_regime_num'].values
                df.loc[valid_indices, 'market_regime'] = regime_df['market_regime'].values
        
        # Clean up temporary column if we created one
        if ret_column == 'temp_returns':
            df = df.drop('temp_returns', axis=1)
        
        # Fill any NaN values in regime columns
        df['market_regime'].fillna('unknown', inplace=True)
        df['market_regime_num'].fillna(-1, inplace=True)
        
        print(f"Market regimes detected: Bull={np.sum(df['market_regime'] == 'bull')}, Bear={np.sum(df['market_regime'] == 'bear')}")
    except Exception as e:
        print(f"Error in regime detection: {str(e)}")
        # Clean up temporary column if we created one
        if ret_column == 'temp_returns' and 'temp_returns' in df.columns:
            df = df.drop('temp_returns', axis=1)
    
    return df

def add_lagged_features(df: pd.DataFrame, lag_periods: int = 5) -> pd.DataFrame:
    """
    Add lagged features to the dataframe
    
    Args:
        df: DataFrame with financial data
        lag_periods: Number of lag periods to include
        
    Returns:
        DataFrame with lagged features
    """
    result = df.copy()
    
    print(f"Adding lagged features with {lag_periods} periods...")
    
    # Identify numeric columns to lag (excluding date, categorical features, etc.)
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
    # Exclude certain columns that shouldn't be lagged
    exclude_cols = []
    if 'date' in numeric_cols:
        exclude_cols.append('date')
    if 'timestamp' in numeric_cols:
        exclude_cols.append('timestamp')
    
    # Filter columns to lag
    cols_to_lag = [col for col in numeric_cols if col not in exclude_cols]
    
    if not cols_to_lag:
        print("Warning: No numeric columns found for lagging")
        return result
    
    print(f"Creating lagged features for {len(cols_to_lag)} columns...")
    
    # Create all lagged features at once to avoid DataFrame fragmentation
    lagged_features = {}
    
    # For each lag period
    for lag in range(1, lag_periods + 1):
        # For each column
        for col in cols_to_lag:
            lagged_features[f'{col}_lag_{lag}'] = df[col].shift(lag)
    
    # Create a DataFrame with all lagged features
    lagged_df = pd.DataFrame(lagged_features, index=df.index)
    
    # Concatenate with the original DataFrame
    result = pd.concat([result, lagged_df], axis=1)
    
    # Fill NaN values - use more modern methods to avoid warnings
    result = result.ffill().bfill()
    
    print(f"Added {len(lagged_features)} lagged features")
    
    return result

def process_file(
    input_file: str,
    output_dir: str,
    calculate_technical: bool = True,
    add_sentiment: bool = False,
    detect_regimes: bool = True,
    add_lagged: bool = True,
    lag_periods: int = 5,
    use_gpu: bool = False,
    sentiment_dir: Optional[str] = None
) -> str:
    """
    Process a financial data file.
    
    Args:
        input_file: Path to the input file
        output_dir: Directory to save the output or complete output file path
        calculate_technical: Whether to calculate technical indicators
        add_sentiment: Whether to add sentiment features
        detect_regimes: Whether to detect market regimes
        add_lagged: Whether to add lagged features
        lag_periods: Number of lag periods
        use_gpu: Whether to use GPU for calculations
        sentiment_dir: Directory with sentiment data
        
    Returns:
        Path to the processed file
    """
    # Check if the path exists and is a directory (handles the case where a .csv extension might have been mistakenly created as a directory)
    if os.path.exists(output_dir) and os.path.isdir(output_dir):
        # This is definitely a directory
        base_name = os.path.basename(input_file)
        file_name, file_ext = os.path.splitext(base_name)
        output_file = os.path.join(output_dir, f"{file_name}_enhanced{file_ext}")
    elif output_dir.lower().endswith('.csv'):
        # This is a file path with .csv extension
        output_file = output_dir
        # Create parent directory if it doesn't exist
        parent_dir = os.path.dirname(output_file)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
    else:
        # This is a regular directory path that doesn't exist yet
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.basename(input_file)
        file_name, file_ext = os.path.splitext(base_name)
        output_file = os.path.join(output_dir, f"{file_name}_enhanced{file_ext}")
    
    print(f"Processing {input_file}...")
    print(f"Output will be saved to: {output_file}")
    
    try:
        # Read the file
    df = pd.read_csv(input_file)
    
        # Convert date column to datetime if it exists
        date_cols = [col for col in df.columns if 'date' in col.lower()]
        if date_cols:
            df[date_cols[0]] = pd.to_datetime(df[date_cols[0]], errors='coerce')
        
        # Calculate technical indicators if requested
        if calculate_technical:
            print("Calculating technical indicators...")
            try:
        df = calculate_technical_features(df)
            except Exception as e:
                print(f"Warning: Error calculating technical indicators: {str(e)}")
    
        # Add sentiment features if requested
    if add_sentiment:
            print("Adding sentiment features...")
            try:
                df = add_sentiment_features(df, sentiment_dir)
            except Exception as e:
                print(f"Warning: Error adding sentiment features: {str(e)}")
        
        # Detect market regimes if requested
    if detect_regimes:
            print("Detecting market regimes...")
        df = detect_market_regimes(df)
    
        # Add lagged features if requested
        if add_lagged:
            print(f"Adding lagged features with {lag_periods} periods...")
            df = add_lagged_features(df, lag_periods)
        
        # Drop rows with NaN values
        original_len = len(df)
        df = df.dropna()
        
        if len(df) < original_len:
            print(f"Dropped {original_len - len(df)} rows with NaN values")
        
        # Save the processed file
    df.to_csv(output_file, index=False)
        print(f"Saved processed file to {output_file}")
        
        return output_file
    
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        return None

def calculate_cross_asset_correlations(df):
    """Calculate cross-asset correlations"""
    result = df.copy()
    
    # Get unique tickers
    tickers = result['ticker'].unique()
    
    # If only one ticker, nothing to correlate
    if len(tickers) <= 1:
        return result
    
    # Create pivot table for close prices
    price_pivot = result.pivot(index='date', columns='ticker', values='close')
    
    # Calculate returns
    returns_pivot = price_pivot.pct_change(5)
    
    # For each ticker, calculate correlation with others
    for ticker in tickers:
        # Get key reference assets
        reference_assets = ['SPY', 'TLT', 'GLD', 'USO']
        reference_assets = [a for a in reference_assets if a in tickers and a != ticker]
        
        for ref_asset in reference_assets:
            # Calculate rolling correlation
            correlation = returns_pivot[ticker].rolling(30).corr(returns_pivot[ref_asset])
            
            # Create column name
            col_name = f'corr_{ref_asset.lower()}_30d'
            
            # Add to original dataframe
            # We need to merge back using the date
            corr_df = pd.DataFrame({
                'date': correlation.index,
                col_name: correlation.values
            })
            
            # Get only the rows for the current ticker
            ticker_rows = result['ticker'] == ticker
            
            # Merge correlation data
            result_ticker = pd.merge(
                result[ticker_rows], 
                corr_df,
                on='date',
                how='left'
            )
            
            # Update in the main dataframe
            result.loc[ticker_rows, col_name] = result_ticker[col_name].values
    
    return result

def main():
    parser = argparse.ArgumentParser(description="Add technical features to financial data")
    
    # Input/output arguments
    parser.add_argument("--input_dir", help="Directory with input CSV files")
    parser.add_argument("--input_file", "--input_path", dest="input_file", help="Single input CSV file")
    parser.add_argument("--output_dir", "--output_path", dest="output_dir", default=".", help="Directory to save enhanced data or specific output file path")
    
    # Feature arguments
    parser.add_argument("--calculate_technicals", "--add_technical", action="store_true", dest="calculate_technical", help="Calculate technical indicators")
    parser.add_argument("--add_sentiment", "--add_statistical", action="store_true", dest="add_sentiment", help="Add sentiment features")
    parser.add_argument("--detect_regimes", "--add_cyclical", action="store_true", dest="detect_regimes", help="Detect market regimes")
    parser.add_argument("--add_lagged", action="store_true", help="Add lagged features")
    parser.add_argument("--sentiment_dir", help="Directory with sentiment data")
    parser.add_argument("--lag_periods", type=int, default=5, help="Number of lag periods to include")
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU for calculations")
    
    args = parser.parse_args()
    
    # Validate input
    if not args.input_dir and not args.input_file:
        parser.error("Either --input_dir or --input_file (or --input_path) must be specified")
    
    try:
    # Process single file
    if args.input_file:
            print(f"Input file: {args.input_file}")
            print(f"Output: {args.output_dir}")
            
            # We'll let process_file handle the output path logic
        process_file(
            input_file=args.input_file,
            output_dir=args.output_dir,
                calculate_technical=args.calculate_technical,
            add_sentiment=args.add_sentiment,
            detect_regimes=args.detect_regimes,
                sentiment_dir=args.sentiment_dir,
                add_lagged=args.add_lagged,
                lag_periods=args.lag_periods,
                use_gpu=args.use_gpu
        )
    
    # Process directory
    if args.input_dir:
        # Find all CSV files
        input_files = glob.glob(os.path.join(args.input_dir, "*.csv"))
        
        if not input_files:
            print(f"No CSV files found in {args.input_dir}")
            return
            
            # Output directory should be just a directory in this case, make sure it exists
            if not args.output_dir.lower().endswith('.csv'):
                os.makedirs(args.output_dir, exist_ok=True)
        
        # Process each file
        for input_file in tqdm(input_files, desc="Processing files"):
            process_file(
                input_file=input_file,
                output_dir=args.output_dir,
                    calculate_technical=args.calculate_technical,
                add_sentiment=args.add_sentiment,
                detect_regimes=args.detect_regimes,
                    sentiment_dir=args.sentiment_dir,
                    add_lagged=args.add_lagged,
                    lag_periods=args.lag_periods,
                    use_gpu=args.use_gpu
                )
                
        print("\nProcessing completed successfully!")
    except Exception as e:
        import traceback
        print(f"\nERROR: {str(e)}")
        print(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()
