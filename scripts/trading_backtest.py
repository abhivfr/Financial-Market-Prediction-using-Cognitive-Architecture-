# Save as scripts/trading_backtest.py
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns # Used for scatter plot, might require installation
import argparse  # Add argparse import if not already present
import os
import json
from tqdm import tqdm
import sys # Import sys for error handling
from datetime import datetime # Import datetime for handling dates
import pandas._libs.tslibs.offsets as offsets # Import for BDay offset if needed

# Ensure src is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import your model and data loading code
try:
    from src.arch.cognitive import CognitiveArchitecture
    from train import FinancialDataLoader # Although used in generate_signals, we will process dataframes directly
    print("Successfully imported CognitiveArchitecture and FinancialDataLoader.")
except ImportError as e:
    print(f"Error importing necessary modules: {e}")
    print("Please ensure src.arch.cognitive.CognitiveArchitecture and train.FinancialDataLoader are available and importable.")
    sys.exit(1)

# Hardcoded Configuration (based on user's plan arguments)
model_path = "models/financial_consciousness_20250423_181527.pth" # Path to your trained cognitive model - UPDATE THIS IF NEEDED
data_path = "data/financial/test_data.csv" # Path to the PROCESSED test data CSV
seq_length = 20 # Sequence length for the model
threshold = 0.005 # Signal threshold from user's plan
output_path = "validation/trading_backtest.json" # Output JSON file path

# Assuming the model was trained with 4 input features ('price', 'volume', 'returns', 'volatility')
input_dim = 4
PREDICTION_FEATURE_INDEX = 0 # Index of 'price' prediction in the model output (assuming price is the 0th feature)

# Trading Strategy Parameters (from user's original code defaults)
initial_capital = 10000.0 # Ensure float
position_size = 0.1 # This looks like capital allocation percentage (10% of capital)
transaction_cost = 0.001 # 0.1% transaction cost per trade (applied to value traded)

def load_model(model_path, input_dim): # Removed seq_length arg as it's not used here
    """Loads the trained model."""
    # Use CPU for consistency with previous steps
    device = torch.device("cpu")
    # Ensure model architecture parameters match training (input_dim=4, hidden_dim=256, memory_size=1000 assumed)
    # Assuming the model architecture in src.arch.cognitive.py matches what was trained
    model = CognitiveArchitecture(input_dim=input_dim, hidden_dim=256, memory_size=1000)
    try:
        # Load weights onto CPU
        model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=False)
        model.to(device)
        model.eval() # Set to evaluation mode
        print(f"Successfully loaded model state dict from {model_path}")
        return model
    except FileNotFoundError:
        print(f"Error: Model checkpoint not found at {model_path}. Cannot run backtest.")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading model state dict: {e}. Cannot run backtest.")
        sys.exit(1)

# CORRECTED: Added prediction_feature_index to function signature and re-interpreted model output
def generate_trading_signals(model, data_df, seq_length, threshold, prediction_feature_index):
    """
    Generate trading signals from model predictions over a DataFrame.
    The signal for day `i` is based on the prediction made using data up to day `i-1`.
    Returns a DataFrame with date index and columns like 'signal', 'confidence', 'risk_score', 'true_direction'.
    """
    device = torch.device("cpu") # Use CPU for inference
    model.to(device)
    model.eval()

    signals_list = [] # Use a list to collect data for DataFrame

    print(f"\nGenerating trading signals over {len(data_df)} samples...")

    # We need enough data to form the first sequence (seq_length) to predict for the day at index `seq_length`.
    # The first index we can generate a signal FOR is `seq_length`.
    first_signal_idx = seq_length

    if len(data_df) < seq_length + 1: # Need seq_length for seq + target/prediction
         print(f"Not enough data ({len(data_df)}) for signal generation with seq_length {seq_length}. Need at least {seq_length + 1} samples.")
         return pd.DataFrame() # Return empty DataFrame

    data_values = data_df[['price', 'volume', 'returns', 'volatility']].values
    # Ensure data_df has a datetime index for slicing
    if not isinstance(data_df.index, pd.DatetimeIndex):
         print("Warning: data_df index is not DatetimeIndex. Attempting conversion...")
         try:
             data_df.index = pd.to_datetime(data_df.index) # Attempt to convert
         except Exception as e:
             print(f"Error converting index to DatetimeIndex: {e}. Skipping signal generation.")
             return pd.DataFrame()


    # Iterate through the data indices `i` from the first day we can predict FOR (`seq_length`)
    # up to the last day in the dataset.
    # For index `i`, we predict the price *at index i* using data up to index `i-1`.
    for i in tqdm(range(first_signal_idx, len(data_df)), desc="Generating Signals"):
        # Get the sequence ending at i-1
        # Sequence indices: `i - seq_length` to `i - 1`.
        sequence_slice = data_values[i - seq_length : i] # Shape (seq_length, 4)

        # Get features for the last day of the sequence (index i-1)
        features_slice = data_values[i - 1] # Shape (4)

        # Get actual price at the close of the day BEFORE the prediction day (price at index i-1)
        # This is the price used for comparison to make the trade decision FOR day `i`.
        current_price = data_df['price'].iloc[i - 1] # Price at index i-1 from original DataFrame

        # Get actual price on day `i` (for true direction calculation)
        actual_price_day_i = data_df['price'].iloc[i]

        # Convert to torch tensors and add batch dimension (batch size 1)
        financial_seq_tensor = torch.from_numpy(sequence_slice).float().unsqueeze(0).to(device) # Shape (1, seq_length, 4)
        financial_data_tensor = torch.from_numpy(features_slice).float().unsqueeze(0).to(device) # Shape (1, 4)

        # Get prediction for day `i`
        with torch.no_grad():
            try:
                 # Assuming your model takes financial_data (last step features) and financial_seq (history)
                 # Add volume extraction as seen in your train.py code
                 # Check if the volume is the second feature (index 1)
                 if financial_seq_tensor.shape[-1] > 1:
                     volume_input = financial_seq_tensor[:, :, 1:2] # Extract volume, Shape (1, seq_length, 1)
                 else:
                     volume_input = torch.zeros_like(financial_seq_tensor[:, :, :1]) # Provide dummy volume if not enough features


                 outputs = model(
                      financial_data=financial_data_tensor, # Features for index i-1
                      financial_seq=financial_seq_tensor,   # Sequence ending at i-1
                      volume=volume_input
                 )
            except Exception as model_e:
                 print(f"\nError during model forward pass at index {i} (Date: {data_df.index[i]}): {model_e}. Skipping signal generation for this day.")
                 # Append None or default values to signals_list for this index if needed
                 continue # Skip to next day


            # Based on DataLoader and Loss, market_state[:, -1] is the prediction for the NEXT day
            # If sequence ends at i-1, prediction is for index i.
            # market_state shape: (batch_size, seq_len_out, input_dim) or (batch_size, input_dim)
            # Metrics compare market_state[:, -1] to target (index i).
            # So, prediction for day `i` is `outputs['market_state'][:, -1]`.
            predicted_market_state_day_i = None
            if 'market_state' in outputs:
                 if outputs['market_state'].ndim == 3:
                     # Assuming the prediction for the next step is the last step of the output sequence
                     predicted_output_raw = outputs['market_state'][:, -1] # Shape (batch_size, input_dim)
                 elif outputs['market_state'].ndim == 2 and outputs['market_state'].shape[1] == data_values.shape[1]:
                     # Assuming the output is directly the prediction for the next step
                     predicted_output_raw = outputs['market_state'] # Shape (batch_size, input_dim)
                 else:
                      print(f"\nWarning: Unexpected market_state dimensions/shape at index {i} (Date: {data_df.index[i]}): {outputs['market_state'].shape}. Skipping signal generation for this day.")
                      continue # Skip to next day
            else:
                 print(f"\nError: 'market_state' not found in model outputs at index {i} (Date: {data_df.index[i]}). Cannot generate signals. Skipping signal generation for this day.")
                 continue # Skip to next day

            # Ensure predicted_output_raw has the expected size before indexing
            # Use the passed prediction_feature_index
            if predicted_output_raw is None or predicted_output_raw.shape[-1] <= prediction_feature_index:
                 print(f"\nError: Predicted market_state does not have enough features ({predicted_output_raw.shape[-1] if predicted_output_raw is not None else 'None'}) to extract price prediction at index {prediction_feature_index} at index {i} (Date: {data_df.index[i]}). Skipping signal generation for this day.")
                 continue # Skip to next day

            # Extract the raw output for the price feature
            predicted_raw_price_output = predicted_output_raw[0, prediction_feature_index].item() # Scalar raw output

            # INTERPRETATION CHANGE: Interpret the raw output as a PRICE CHANGE from the current price
            predicted_price_change = predicted_raw_price_output # Use the raw output directly as the predicted change

            # The predicted price level is the current price plus the predicted change (for logging/comparison)
            predicted_price_day_i = current_price + predicted_price_change # Calculate predicted price level


            # Extract risk score if available and valid
            risk_score_day_i = 0.0
            if 'risk_scores' in outputs and outputs['risk_scores'] is not None and outputs['risk_scores'].numel() > 0:
                 # Assuming risk_scores is (batch_size, 1) or similar, and the first element is the relevant score
                 # Check dimensions before indexing
                 if outputs['risk_scores'].ndim >= 2:
                     risk_score_val = outputs['risk_scores'][0, 0].item() if outputs['risk_scores'].shape[1] > 0 else 0.0
                 else:
                      risk_score_val = outputs['risk_scores'][0].item() if outputs['risk_scores'].numel() > 0 else 0.0

                 risk_score_day_i = risk_score_val if np.isfinite(risk_score_val) else 0.0 # Use 0 if NaN or Inf
                 # Ensure risk_score is non-negative for threshold adjustment (risk should increase, not decrease threshold)
                 risk_score_day_i = max(0.0, risk_score_day_i)


        # Generate signal for day `i` using the strategy
        # Decision made at end of day `i-1` based on prediction for day `i`.
        # Signal is based on predicted direction and confidence, adjusted by risk.
        # Use the predicted price change directly for direction and confidence calculation
        price_change_pred = predicted_price_change # Use the value interpreted as change

        pred_direction = np.sign(price_change_pred) if abs(price_change_pred) > 1e-9 else 0.0 # +1 for up, -1 for down, 0 for no change
        pred_direction = pred_direction if np.isfinite(pred_direction) else 0.0 # Ensure finite


        # Calculate confidence (absolute predicted change relative to current price)
        # Confidence is based on the magnitude of the predicted CHANGE relative to the current price.
        confidence = np.abs(price_change_pred) / (abs(current_price) + 1e-8) # Use abs(current_price) for robustness
        confidence = confidence if np.isfinite(confidence) else 0.0 # Ensure finite


        # Apply risk-adjusted threshold (higher risk requires higher confidence)
        # threshold * (1 + risk_score). Signal generated only if confidence > adjusted_threshold
        adjusted_threshold = threshold * (1 + risk_score_day_i)
        adjusted_threshold = adjusted_threshold if np.isfinite(adjusted_threshold) else threshold # Use base threshold if adjusted is NaN/Inf


        # Final signal logic based on user's code: direction * valid_signal
        # valid_signal is 1 if confidence > adjusted_threshold, 0 otherwise.
        # This results in -1, 0, or +1 based on direction and whether confidence met the threshold.
        valid_signal_flag = 1.0 if confidence > adjusted_threshold else 0.0
        signal_value = pred_direction * valid_signal_flag # This gives -1, 0, or +1


        # Assuming a simple Long/Flat strategy (position_signal 1 for long, 0 for flat)
        # Go long if the complex signal_value is positive, otherwise flat.
        position_signal = 1.0 if signal_value > 0 else 0.0 # Go long if predicted direction is up AND confidence > threshold


        # Ground truth (actual price movement) for win rate calculation later
        true_direction = np.sign(actual_price_day_i - current_price) if abs(actual_price_day_i - current_price) > 1e-9 else 0.0
        true_direction = true_direction if np.isfinite(true_direction) else 0.0 # Ensure finite


        # Store results for this day
        signals_list.append({
            'date': data_df.index[i], # Use the actual date as index
            'current_price': current_price, # Price at i-1
            'predicted_price': predicted_price_day_i, # Predicted price at i (Re-interpreted level)
            'predicted_price_raw_output': predicted_raw_price_output, # The raw output from the model
            'predicted_price_change': predicted_price_change, # The value interpreted as change
            'predicted_direction': pred_direction, # Predicted direction for price move from i-1 to i
            'confidence': confidence,
            'risk_score': risk_score_day_i,
            'adjusted_threshold': adjusted_threshold,
            'signal_value': signal_value, # The complex signal (-1, 0, +1)
            'position_signal': position_signal, # The simple Long/Flat signal (1 or 0)
            'true_direction': true_direction, # Actual direction for price move from i-1 to i
            'price_at_prediction_day': actual_price_day_i # Actual price on day i (True price level)
        })

    # Convert list of signals to DataFrame, using 'date' as index
    if not signals_list:
         print("No signals generated.")
         return pd.DataFrame() # Return empty if list is empty

    signals_df = pd.DataFrame(signals_list).set_index('date')
    return signals_df


# Renamed position_size to allocation_factor for clarity in backtest function
def run_backtest(signals_df, initial_capital=10000.0, allocation_factor=0.1, transaction_cost_percentage=0.001):
    """
    Runs a simple trading backtest based on signals and actual price data.
    Assumes trades are executed at the close of the day the signal is generated (i.e.,
    position for day `i` is based on signal generated at end of day `i-1`).
    Profit/loss on day `i` is based on the price change from day `i-1`'s close to day `i`'s close.
    allocation_factor is the percentage of current capital allocated (e.g., 0.1 for 10% max).
    transaction_cost_percentage is applied on the value of the trade.
    """
    # Backtest portfolio starts from the first day a signal is generated
    # Need to include the day before the first signal for correct return calculation basis
    if signals_df.empty:
         print("No signals DataFrame provided to run backtest.")
         # Return initial capital in equity curve for plotting consistency
         return {}, pd.DataFrame(), pd.Series(initial_capital, index=[datetime.now()])


    # Get price series needed for calculating daily returns for the asset
    # This needs the price from the day BEFORE the first signal up to the last signal day price.
    # signals_df['current_price'] is the price on day i-1 (closing price before the trading day starts)
    # signals_df['price_at_prediction_day'] is the price on day i (closing price of the trading day)
    # The price series should start with the first 'current_price' and end with the last 'price_at_prediction_day'.
    price_series_for_returns = pd.concat([pd.Series(signals_df['current_price'].iloc[0], index=[signals_df.index[0] - pd.Timedelta(days=1) if isinstance(signals_df.index[0], pd.Timestamp) else signals_df.index[0] - offsets.BDay(1) if len(signals_df.index) > 0 else datetime.now()]), signals_df['price_at_prediction_day']])
    # Ensure the index is sorted for pct_change
    price_series_for_returns = price_series_for_returns.sort_index()

    # Daily returns for the asset during the backtest period (from day i-1 to day i)
    # This percentage return is (price_i - price_{i-1}) / price_{i-1}
    asset_daily_returns = price_series_for_returns.pct_change().loc[signals_df.index] # Align returns with signals_df index


    # --- Simulate trading day by day with transaction costs ---
    equity_with_cost = initial_capital
    # Track equity history for the plot (include the starting point before the first trade)
    equity_history_dates = [price_series_for_returns.index[0]] # Start date for the equity curve
    equity_history_values = [initial_capital]

    prev_position_signal = 0.0 # Assume flat position before the first signal

    # Iterate through days where signals are available (index i of signals_df)
    # Index `i` corresponds to day `i`. Signal for day `i` determines position for day `i`.
    # Return for day `i` is `asset_daily_returns.iloc[i]`.
    for i in range(len(signals_df)):
        current_date = signals_df.index[i]

        current_position_signal = signals_df['position_signal'].iloc[i] # Simple signal (0 or 1) for today (day i)

        # Calculate transaction costs if the position changes from yesterday's signal
        # Trade is executed at the close of day i-1 or open of day i based on signal for day i.
        # Cost is applied to the value *traded*.
        trade_executed_today = False
        if current_position_signal != prev_position_signal:
             trade_executed_today = True

        cost_today = 0.0
        if trade_executed_today:
             # Calculate the value of the capital changing hands
             # Cost is applied to the change in allocated capital percentage relative to current equity
             change_in_allocation_pct = abs(current_position_signal * allocation_factor - prev_position_signal * allocation_factor)
             cost_today = change_in_allocation_pct * equity_with_cost * transaction_cost_percentage
             cost_today = cost_today if np.isfinite(cost_today) else 0.0 # Ensure finite


        # Calculate daily strategy return percentage for day `i`
        # Strategy return = Position held today (%) * Asset return today (%)
        # Position held today (%) = current_position_signal * allocation_factor
        # Asset return today (%) = asset_daily_returns.iloc[i]
        daily_strat_return_pct = current_position_signal * allocation_factor * asset_daily_returns.iloc[i]
        daily_strat_return_pct = daily_strat_return_pct if np.isfinite(daily_strat_return_pct) else 0.0 # Handle potential NaNs from asset returns


        # Update equity
        # Equity at end of day i = Equity at start of day i (before cost) * (1 + daily_strat_return_pct) - cost_today
        equity_after_return = equity_with_cost * (1 + daily_strat_return_pct)
        equity_today = equity_after_return - cost_today
        equity_today = equity_today if np.isfinite(equity_today) else equity_with_cost # Prevent NaN equity

        # Update equity for the next iteration
        equity_with_cost = equity_today

        # Record for history
        equity_history_dates.append(current_date)
        equity_history_values.append(equity_today)

        # Update previous position signal for the next day's cost calculation
        prev_position_signal = current_position_signal


    # Create the final equity curve Series
    equity_curve_with_cost = pd.Series(equity_history_values, index=equity_history_dates)
    equity_curve_with_cost = equity_curve_with_cost.sort_index() # Ensure sorted by date


    # Recalculate metrics using equity_with_cost
    if equity_curve_with_cost.empty:
         print("Equity curve is empty. Cannot calculate backtest metrics.")
         return {}, pd.DataFrame(), equity_curve_with_cost

    final_equity_cost = equity_curve_with_cost.iloc[-1]
    total_return_cost = (final_equity_cost / initial_capital) - 1


    # Calculate Drawdown
    peak_cost = equity_curve_with_cost.expanding(min_periods=1).max()
    drawdown_cost = (equity_curve_with_cost / peak_cost - 1).fillna(0) # Fill NaN drawdown with 0
    max_drawdown_cost = drawdown_cost.min()


    # Calculate Sharpe Ratio using daily strategy returns *percentage*
    # We need the daily percentage returns *including* costs, derived from the equity curve.
    daily_strategy_returns_cost_pct = equity_curve_with_cost.pct_change().dropna()

    trading_days_year = 252 # Approximate number of trading days in a year

    sharpe_ratio_cost = np.nan
    if len(daily_strategy_returns_cost_pct) > 1:
        mean_daily_return_cost = daily_strategy_returns_cost_pct.mean()
        std_daily_return_cost = daily_strategy_returns_cost_pct.std()
        if std_daily_return_cost != 0:
             # Annualized Sharpe = (Annualized Mean - Risk Free Rate) / Annualized Std Dev
             # Annualized Mean = mean_daily_return * trading_days_year
             # Annualized Std Dev = std_daily_return * sqrt(trading_days_year)
             # Assuming Risk Free Rate = 0
             # Sharpe = (mean_daily_return * trading_days_year) / (std_daily_return * sqrt(trading_days_year))
             # Sharpe = mean_daily_return * sqrt(trading_days_year) / std_daily_return
             sharpe_ratio_cost = mean_daily_return_cost * np.sqrt(trading_days_year) / std_daily_return_cost
             sharpe_ratio_cost = sharpe_ratio_cost if np.isfinite(sharpe_ratio_cost) else np.nan # Ensure finite
        else:
             sharpe_ratio_cost = np.nan # Avoid division by zero
    else:
         sharpe_ratio_cost = np.nan # Not enough data for Sharpe


    # Calculate Win Rate from signals_df where a trade signal (position_signal = 1) was issued
    # A 'win' is when position_signal is 1 AND true_direction is 1.
    long_signals_df = signals_df[signals_df['position_signal'] == 1]
    win_rate = np.nan # Default win rate
    if not long_signals_df.empty:
        winning_days = long_signals_df[long_signals_df['true_direction'] > 0]
        # Win rate is number of winning LONG days divided by total number of LONG signal days
        win_rate = len(winning_days) / len(long_signals_df)
        win_rate = win_rate if np.isfinite(win_rate) else np.nan # Ensure finite


    # Count trades: Occurs when position_signal changes (0 to 1 or 1 to 0)
    # Diff will show 1 or -1 when signal changes.
    # Add 1 for the initial trade if starting from 0 and first signal is 1.
    signal_changes = signals_df['position_signal'].diff().abs().fillna(0)
    # Add the first trade if it's not 0
    num_trades = (signal_changes > 0).sum()
    if signals_df['position_signal'].iloc[0] != 0:
         # This counts as one trade to get into the initial position
         num_trades += 1 # Add the initial trade if starting non-flat


    # Summarize results
    backtest_metrics_final = {
        'total_return': total_return_cost,
        'max_drawdown': max_drawdown_cost,
        'sharpe_ratio': sharpe_ratio_cost,
        'num_trades': int(num_trades), # Ensure it's an integer
        'win_rate': win_rate,
        'trading_period_start': equity_curve_with_cost.index[0].strftime('%Y-%m-%d') if isinstance(equity_curve_with_cost.index[0], pd.Timestamp) else str(equity_curve_with_cost.index[0]),
        'trading_period_end': equity_curve_with_cost.index[-1].strftime('%Y-%m-%d') if isinstance(equity_curve_with_cost.index[-1], pd.Timestamp) else str(equity_curve_with_cost.index[-1]),
        'num_trading_days': len(signals_df) # Number of days signals were generated for
    }


    return backtest_metrics_final, signals_df, equity_curve_with_cost # Return metrics, signals_df (useful for plot), and the full equity curve


def main():
    # Add command-line argument support
    parser = argparse.ArgumentParser(description="Run trading backtest on trained model")
    parser.add_argument("--model", type=str, required=True, 
                      help="Path to the trained model checkpoint")
    parser.add_argument("--data", type=str, required=True, 
                      help="Path to test data CSV")
    parser.add_argument("--seq_length", type=int, default=20,
                      help="Sequence length for model")
    parser.add_argument("--threshold", type=float, default=0.005,
                      help="Signal threshold for trading decisions")
    parser.add_argument("--output", type=str, default="validation/trading_backtest.json",
                      help="Output JSON file path")
    parser.add_argument("--initial_capital", type=float, default=10000.0,
                      help="Initial capital for backtest")
    parser.add_argument("--position_size", type=float, default=0.1,
                      help="Position size as fraction of capital")
    parser.add_argument("--transaction_cost", type=float, default=0.001,
                      help="Transaction cost as fraction of trade value")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    os.makedirs(output_dir, exist_ok=True)
    
    print("🚀 Starting Trading Strategy Backtest")
    print(f"Model: {args.model}")
    print(f"Data: {args.data}")
    print(f"Sequence Length: {args.seq_length}")
    print(f"Signal Threshold: {args.threshold}")
    print(f"Output Path: {args.output}")
    print(f"Initial Capital: {args.initial_capital}")
    print(f"Position Size: {args.position_size}")
    print(f"Transaction Cost (%): {args.transaction_cost * 100}")
    
    # Load model
    model = load_model(args.model, input_dim=4)  # Assume 4 input dimensions
    
    # Load the processed data
    try:
        data_df = pd.read_csv(args.data, parse_dates=['timestamp'], index_col='timestamp')
        print(f"Loaded processed data with shape {data_df.shape}")
        if len(data_df) < args.seq_length + 1:
            print(f"Error: Data is too short ({len(data_df)} samples) for signal generation with seq_length {args.seq_length}")
            sys.exit(1)
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)
    
    # Generate trading signals
    signals_df = generate_trading_signals(model, data_df, args.seq_length, args.threshold, PREDICTION_FEATURE_INDEX)
    
    # Run backtest
    backtest_metrics, portfolio_details, equity_curve = run_backtest(
        signals_df,
        initial_capital=args.initial_capital,
        allocation_factor=args.position_size,
        transaction_cost_percentage=args.transaction_cost
    )
    
    # --- Print Backtest Results ---
    print("\n--- Backtest Results ---")
    for metric, value in backtest_metrics.items():
        # Format float values
        if isinstance(value, float):
            print(f"{metric.replace('_', ' ').title()}: {value:.4f}")
        # Format integer values
        elif isinstance(value, int):
             print(f"{metric.replace('_', ' ').title()}: {value}")
        # Print strings as they are
        else:
             print(f"{metric.replace('_', ' ').title()}: {value}")
    print("------------------------")


    # --- Save Results ---
    try:
        # Convert numpy types (like np.nan, np.int64) to Python serializable types
        # Ensure metrics dictionary is fully serializable
        def serialize_value(v):
            if isinstance(v, (np.float64, np.float32)):
                return float(v) if np.isfinite(v) else None # Convert NaN/Inf to None for JSON
            if isinstance(v, (np.int64, np.int32)):
                 return int(v)
            return v # Return other types as is

        serializable_metrics = {k: serialize_value(v) for k, v in backtest_metrics.items()}

        # Ensure all relevant columns are in signals_df for describe()
        signals_summary_cols = ['signal_value', 'confidence', 'risk_score', 'predicted_direction', 'true_direction', 'position_signal']
        # Filter columns that actually exist in signals_df before describing
        existing_summary_cols = [col for col in signals_summary_cols if col in signals_df.columns]

        results = {
            'metrics': serializable_metrics,
            # Convert numpy types in describe() output using .map
            'signals_summary': signals_df[existing_summary_cols].describe().map(serialize_value).to_dict()
        }

        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Backtest results saved to {args.output}")

        # Save equity curve to CSV
        equity_curve_csv_path = args.output.replace('.json', '_equity_curve.csv') # Use output path base name and _equity_curve suffix
        # Ensure header is correct for the single series
        equity_curve.to_csv(equity_curve_csv_path, header=['equity'], index=True) # Save with index (date)
        print(f"Equity curve saved to {equity_curve_csv_path}")

    except Exception as e:
        print(f"Error saving backtest results or equity curve: {e}")
        import traceback
        traceback.print_exc()


    # --- Create Performance Chart ---
    try:
        plt.figure(figsize=(12, 8))

        # Plot Equity Curve
        plt.subplot(2, 1, 1)
        # Calculate percentage cumulative returns for plotting
        # Ensure initial_capital is not zero
        if args.initial_capital > 0:
            percentage_equity_curve = (equity_curve / args.initial_capital - 1) * 100
            plt.plot(percentage_equity_curve.index, percentage_equity_curve, label='Strategy Equity Curve (%)')
            plt.title('Cumulative Returns (%)')
            plt.xlabel('Date')
            plt.ylabel('Cumulative Return (%)')
            plt.grid(True)
            plt.legend()
        else:
             plt.text(0.5, 0.5, "Initial capital is zero. Cannot plot returns.", horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
             print("Warning: Initial capital is zero, skipping equity curve plot.")


        # Plot Signals and Actual Direction
        plt.subplot(2, 1, 2)
        # Filter for days where a LONG signal was issued (position_signal == 1)
        long_signals_for_plot = signals_df[signals_df['position_signal'] == 1].copy()
        if not long_signals_for_plot.empty:
             # Color points by actual direction on that day
             # Map true_direction: >0 positive (Green), <0 negative (Red), 0 (Yellow)
             color_map = {1.0: 'g', -1.0: 'r', 0.0: 'y'} # Use 1.0, -1.0, 0.0 as true_direction is float

             # Use a dummy y-value for scatter plot points, e.g., the signal value (1 in this case)
             y_values = long_signals_for_plot['position_signal'] # Y = 1 for all points here
             colors = long_signals_for_plot['true_direction'].map(color_map)

             plt.scatter(long_signals_for_plot.index, y_values, c=colors, alpha=0.6, label='Long Signals (color=Actual Direction)', s=20) # s is marker size


             # Create a legend for colors
             # Need proxy artists for the legend
             proxy_green = plt.Line2D([0,0],[0,0], color='g', marker='o', linestyle='', label='Actual Up')
             proxy_red = plt.Line2D([0,0],[0,0], color='r', marker='o', linestyle='', label='Actual Down')
             proxy_yellow = plt.Line2D([0,0],[0,0], color='y', marker='o', linestyle='', label='Actual Flat')

             plt.legend(handles=[proxy_green, proxy_red, proxy_yellow], numpoints=1)

             # Set ticks for y-axis if only 0/1 signals
             plt.yticks([0, 1], ['Flat', 'Long'])


        else:
             plt.text(0.5, 0.5, "No active trading signals to plot.", horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
             print("Warning: No active trading signals generated to plot.")


        plt.title('Trading Signals (Long Positions Only) vs Actual Direction')
        plt.xlabel('Date')
        plt.ylabel('Position Signal (Long)')
        plt.grid(True)


        plt.tight_layout()

        # Save plot
        plot_dir = os.path.dirname(args.output) # Save plots in the same validation directory
        os.makedirs(plot_dir, exist_ok=True) # Ensure directory exists
        plot_path = os.path.join(plot_dir, "trading_backtest_performance.png")
        plt.savefig(plot_path)
        print(f"Performance chart saved to {plot_path}")

        # plt.show() # Uncomment to display plot immediately (might block)

    except ImportError:
         print("Warning: matplotlib and/or seaborn not installed. Skipping plot generation.")
         print("Install them with: pip install matplotlib seaborn")
    except Exception as e:
        print(f"Error generating or saving performance chart: {e}")
        import traceback
        traceback.print_exc()


    print("\nBacktest simulation complete.")

if __name__ == "__main__":
    main()
