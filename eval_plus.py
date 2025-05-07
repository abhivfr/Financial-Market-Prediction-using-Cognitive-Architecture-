import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from src.arch.cognitive import CognitiveArchitecture
# Ensure these imports are correct based on your file structure
from src.core.core import MemoryConfidence
from typing import Dict, List, Tuple, Any
from src.memory.conflict import ConflictEvaluator # Import the updated ConflictEvaluator
# Make sure this import works based on your layout.
# You might need to adjust the import path depending on where FinancialDataLoader is defined.
try:
    from train import FinancialDataLoader
except ImportError:
    print("Error: Could not import FinancialDataLoader from train.py.")
    print("Please ensure train.py exists and FinancialDataLoader is defined and importable from it.")
    # Define a dummy class for basic functionality if the import fails
    class FinancialDataLoader:
        def __init__(self, path, seq_length=None, batch_size=None):
            print(f"Using dummy FinancialDataLoader for path: {path}")
            # This dummy implementation assumes the data is a CSV with 4 columns and a header row
            try:
                # Attempt to load data, skipping a header
                self.data = np.genfromtxt(path, delimiter=',', skip_header=1)
                if self.data.ndim == 1: # Handle case where there's only one data row
                     self.data = self.data.reshape(1, -1)
                if self.data.shape[1] != 4:
                     print(f"Warning: Dummy loader expected 4 columns, found {self.data.shape[1]}. Check delimiter or data format.")
                     # Attempt to adjust if more columns, just take first 4 if available
                     if self.data.shape[1] > 4:
                         self.data = self.data[:, :4]
                     else: # If less than 4, pad with NaNs or zeros, or raise error
                         print("Error: Data has fewer than 4 columns.")
                         self.data = np.empty((0, 4)) # Empty data if not enough columns

            except Exception as e:
                print(f"Error loading data in dummy loader: {e}")
                self.data = np.empty((0, 4)) # Provide empty data on failure

            self._max_pointer = len(self.data) -1 if len(self.data) > 0 else 0 # Basic pointer for compatibility
        @property
        def num_features(self):
             return self.data.shape[1] if len(self.data) > 0 else 4


def compute_metrics(pred, target):
    """
    Computes evaluation metrics: correlations, MAE, and RMSE.

    Args:
        pred (torch.Tensor): Model predictions, shape (num_samples, seq_len, num_features).
        target (torch.Tensor): Ground truth targets, shape (num_samples, seq_len, num_features).

    Returns:
        tuple: (corrs, mae, rmse) - Lists of correlation, MAE, and RMSE for each feature.
    """
    # Flatten the last step of predictions and targets for metric computation
    # Assuming we evaluate the prediction for the next single step (last in the predicted sequence)
    p = pred[:, -1, :].view(-1, pred.size(-1)).cpu().detach().numpy()
    t = target[:, -1, :].view(-1, target.size(-1)).cpu().numpy()

    corrs = []
    for i in range(p.shape[1]):
        # Ensure there's enough data points and variance to compute correlation
        if p.shape[0] > 1 and np.std(p[:, i]) > 1e-8 and np.std(t[:, i]) > 1e-8:
            cor = np.corrcoef(p[:, i], t[:, i])[0, 1]
            corrs.append(cor if not np.isnan(cor) else 0.0)  # Handle potential NaN from corrcoef
        else:
            corrs.append(0.0)

    mae = np.mean(np.abs(p - t), axis=0)
    rmse = np.sqrt(np.mean((p - t)**2, axis=0))

    return corrs, mae, rmse


def run_eval(model, loader, device, skip_memory=False, input_dim=4, memory_dim=256):
    """
    Runs the evaluation loop over the dataset.

    Args:
        model (torch.nn.Module): The model to evaluate.
        loader (DataLoader): DataLoader for the evaluation dataset.
        device (torch.device): The device to run evaluation on (cuda or cpu).
        skip_memory (bool): Whether to skip the memory component during evaluation.
        input_dim (int): Dimension of the input features (e.g., 4 for financial data).
        memory_dim (int): Dimension of the memory embeddings (e.g., 256).


    Returns:
        dict: Dictionary containing evaluation metrics.
    """
    model.eval()
    # Initialize ConflictEvaluator and move it to device
    ce = ConflictEvaluator(input_dim=input_dim, memory_dim=memory_dim)
    ce.to(device)

    # Initialize MemoryConfidence and move it to device
    mc = MemoryConfidence(input_dim=input_dim, memory_dim=memory_dim) # Pass dimensions
    mc.to(device) # <--- THIS LINE MOVES mc TO THE DEVICE

    all_pred = []
    all_tgt = []
    total_conflicts = 0
    total_tokens = 0
    total_confidence = []

    with torch.no_grad():
        # Data loader yields feats (current), seqs (historical), tgts (target sequence)
        for feats, seqs, tgts in loader:
            # Move batch tensors to the device
            feats = feats.to(device)  # Current feature: (batch_size, num_features)
            seqs = seqs.to(device)   # Historical sequence: (batch_size, seq_len, num_features)
            tgts = tgts.to(device)   # Target sequence: (batch_size, seq_len, num_features)

            # Model forward pass
            # financial_data is likely the current step (feats)
            # financial_seq is likely the historical sequence (seqs)
            # market_seq is the predicted sequence (hopefully of length seq_len)
            _, _, recalled, _, _, market_seq = model(
                seq=None,  # Assuming text sequence is not used in this evaluation
                financial_data=feats,
                financial_seq=seqs,
                skip_memory=skip_memory
            )

            # Append predictions and targets
            all_pred.append(market_seq)
            all_tgt.append(tgts) # Append the target sequence directly

            if not skip_memory and recalled is not None:
                # current input for conflict and confidence is feats (shape batch_size, input_dim)
                cur = feats
                # Retrieved memory for conflict and confidence is the first item recalled (shape batch_size, memory_dim)
                # Ensure recalled has at least one item in the sequence dimension
                if recalled.size(1) > 0:
                    ret0 = recalled[:, 0, :]
                    # Evaluate conflict using the updated ConflictEvaluator
                    conflict_mask, sim = ce.evaluate(cur, ret0)
                    total_conflicts += conflict_mask.sum().item()
                    total_tokens += conflict_mask.numel()
                    # Compute confidence using the updated MemoryConfidence
                    confidence_scores = mc.compute(ret0, cur) # mc.compute now returns a tensor
                    total_confidence.extend(confidence_scores.cpu().numpy())
                else:
                     # Print a warning if recalled is empty in a batch where memory isn't skipped
                     # This helps in debugging if the model isn't producing recalled items
                     if torch.any(torch.isnan(recalled)).item() or torch.any(torch.isinf(recalled)).item():
                          print("Warning: Recalled tensor contains NaNs or Infs.")
                     print("Warning: recalled tensor has no items in the sequence dimension (size 0). Skipping conflict and confidence calculation for this batch.")


    # Concatenate results from all batches
    # Check if any data was collected before concatenating
    if all_pred and all_tgt:
        all_pred = torch.cat(all_pred, dim=0)
        all_tgt = torch.cat(all_tgt, dim=0)
    else:
        print("Warning: No predictions or targets were collected during evaluation.")
        # Provide default empty tensors or handle gracefully if no data processed
        all_pred = torch.empty(0, 0, input_dim) # Or appropriate shape based on expected output
        all_tgt = torch.empty(0, 0, input_dim) # Or appropriate shape based on expected targets


    # Ensure predictions and targets have the same sequence length for compute_metrics
    # This handles cases where the last batch might not be a full batch_size or sequence length
    min_pred_len = all_pred.size(1) if all_pred.size(0) > 0 else 0
    min_tgt_len = all_tgt.size(1) if all_tgt.size(0) > 0 else 0
    min_len = min(min_pred_len, min_tgt_len)

    if min_len > 0 and all_pred.size(0) > 0 and all_tgt.size(0) > 0:
        all_pred = all_pred[:, :min_len, :]
        all_tgt = all_tgt[:, :min_len, :]

        corrs, mae, rmse = compute_metrics(all_pred, all_tgt)
    else:
        # Handle case where no valid sequences were created or processed
        num_features = input_dim # Default to input_dim if no data processed
        # Try to infer num_features if data was partially processed
        if all_pred.size(0) > 0: num_features = all_pred.size(2)
        elif all_tgt.size(0) > 0: num_features = all_tgt.size(2)

        corrs = [float('nan')] * num_features
        mae = [float('nan')] * num_features
        rmse = [float('nan')] * num_features
        print("Warning: No valid data processed for metric computation.")


    conflict_rate = total_conflicts / total_tokens if total_tokens > 0 else 0.0
    # Check if total_confidence is empty before calculating mean
    avg_confidence = float(np.mean(total_confidence)) if total_confidence else 0.0


    return {
        'corrs': corrs,
        'mae': mae,
        'rmse': rmse,
        'conflict_rate': conflict_rate,
        'avg_confidence': avg_confidence
    }


def main():
    """
    Main function to set up data, model, and run evaluation.
    """
    # === HARDCODED ARGS HERE ===
    # IMPORTANT: Update ckpt_path to your trained model checkpoint
    # This should be the path to the .pth file from your training run
    ckpt_path = "checkpoints/model_iter_500.pth" # <-- UPDATE THIS TO YOUR NEW CHECKPOINT
    batch_size = 16
    seq_len = 10
    input_dim = 4 # Dimension of financial input features
    # IMPORTANT: Match your model's internal memory/hidden dimension
    memory_dim = 256 # Dimension of memory embeddings

    # IMPORTANT: Update test_data_path to your evaluation data file
    test_data_path = "data/financial/validation_data.csv"  # use test_data.csv if available

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Is CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
        # Simple test tensor to check device
        try:
            test_tensor = torch.ones(1).to(device)
            print(f"Test tensor on device: {test_tensor.device}")
        except Exception as e:
            print(f"Error creating test tensor on device: {e}")


    # Prepare test loader
    # Load data using the existing FinancialDataLoader
    # Assumes FinancialDataLoader loads raw data into loader_ds.data (NumPy array or similar)
    try:
        # Pass seq_length and batch_size if FinancialDataLoader's __init__ expects them
        # Check your FinancialDataLoader's __init__ signature
        loader_ds = FinancialDataLoader(path=test_data_path, seq_length=seq_len, batch_size=batch_size)
        print(f"Successfully loaded data from {test_data_path}")
    except FileNotFoundError:
        print(f"Error: Data file not found at {test_data_path}")
        print("Please update test_data_path to the correct location.")
        return
    except Exception as e:
        print(f"Error loading data with FinancialDataLoader: {e}")
        # Check if loader_ds.data was populated in case of partial failure
        if not hasattr(loader_ds, 'data') or loader_ds.data is None or loader_ds.data.size == 0:
             print("FinancialDataLoader failed to load data or loaded empty data.")
             return


    # --- CORRECTED DATA PREPARATION FOR DataLoader ---
    # Create sequences of length seq_len and corresponding targets.
    # Assuming loader_ds.data is a NumPy array of shape (num_total_timesteps, num_features)
    data = loader_ds.data
    num_total_timesteps = data.shape[0]
    num_data_features = data.shape[1] # Verify data features match input_dim

    if num_data_features != input_dim:
         print(f"Warning: Data file has {num_data_features} features, but input_dim is set to {input_dim}.")
         print("Please ensure input_dim matches the number of features in your data.")
         # Optionally update input_dim based on data for robustness, but warning is better.
         # input_dim = num_data_features


    # Create input sequences (X) and corresponding target sequences (Y)
    # Input sequence X: data[i : i + seq_len] (shape seq_len, num_features)
    # Target sequence Y: data[i + 1 : i + seq_len + 1] (shape seq_len, num_features)
    # The loop should go up to num_total_timesteps - seq_len - 1 to have a valid target sequence start index
    # max_i = num_total_timesteps - 2 * seq_len # Corrected index based on target sequence starting seq_len steps *after* current i

    # Let's re-verify the indices for clarity.
    # Input seq: starts at i, ends at i + seq_len - 1
    # Target seq: starts at i + seq_len, ends at i + seq_len + seq_len - 1
    # The last possible start for input seq 'i' is when i + seq_len + seq_len - 1 is the last data point.
    # i + 2*seq_len - 1 = num_total_timesteps - 1
    # i = num_total_timesteps - 2*seq_len
    # So, i ranges from 0 to num_total_timesteps - 2*seq_len (inclusive).
    max_i_inclusive = num_total_timesteps - 2 * seq_len

    sequences_in = []
    sequences_target = []

    # Check if there's enough data to create at least one sequence pair
    # Need at least 2 * seq_len timesteps for one input seq + one target seq
    if num_total_timesteps < 2 * seq_len:
        print(f"Error: Not enough data ({num_total_timesteps} timesteps) to create sequences and targets with seq_len={seq_len}.")
        print(f"Need at least {2 * seq_len} timesteps.")
        print("Please check your data file or reduce seq_len.")
        return


    for i in range(max_i_inclusive + 1):
        seq_in = data[i : i + seq_len]
        seq_target = data[i + seq_len : i + 2 * seq_len] # Target is the sequence of the *next* seq_len steps
        sequences_in.append(seq_in)
        sequences_target.append(seq_target)


    sequences_in = np.array(sequences_in)       # Shape (num_sequences, seq_len, num_features)
    sequences_target = np.array(sequences_target) # Shape (num_sequences, seq_len, num_features)

    if sequences_in.shape[0] == 0:
        print(f"Error: No sequences created with seq_len={seq_len} from {num_total_timesteps} timesteps.")
        print("Check your data size and seq_len. Make sure num_total_timesteps >= 2 * seq_len.")
        return

    print(f"Created {len(sequences_in)} sequences of length {seq_len}")

    # Convert to PyTorch tensors
    # These will be the tensors in the TensorDataset
    # Tensor 1 (feats in run_eval loop): Current features (last step of sequences_in) - Shape (num_sequences, num_features)
    # Tensor 2 (seqs in run_eval loop): Historical sequence (sequences_in) - Shape (num_sequences, seq_len, num_features)
    # Tensor 3 (tgts in run_eval loop): Target sequence (sequences_target) - Shape (num_sequences, seq_len, num_features)

    feats_tensor = torch.tensor(sequences_in[:, -1, :], dtype=torch.float32)
    seqs_tensor = torch.tensor(sequences_in, dtype=torch.float32)
    tgts_tensor = torch.tensor(sequences_target, dtype=torch.float32)


    # Create the DataLoader
    dataset = TensorDataset(feats_tensor, seqs_tensor, tgts_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False) # No shuffle for eval, drop_last=False to include all data


    print(f"Created DataLoader with {len(loader)} batches of size {batch_size}")
    if len(loader) > 0:
        sample_feats, sample_seqs, sample_tgts = next(iter(loader))
        print(f"Sample batch shapes:")
        print(f"  feats: {sample_feats.shape}")
        print(f"  seqs: {sample_seqs.shape}")
        print(f"  tgts: {sample_tgts.shape}")
    else:
        print("DataLoader is empty after creating sequences. Check data, seq_len, and batch_size.")
        return


    # Initialize model
    try:
        model = CognitiveArchitecture().to(device)
        print("CognitiveArchitecture model initialized.")
    except Exception as e:
         print(f"Error initializing CognitiveArchitecture model: {e}")
         return


    # Load checkpoint
    try:
        checkpoint = torch.load(ckpt_path, map_location=device)
        # Handle different checkpoint saving formats
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
            print(f"Loaded model state_dict from {ckpt_path}")
        else:
            # Strict=False allows loading even if some keys don't match, useful if model architecture changed slightly
            model.load_state_dict(checkpoint, strict=False)
            print(f"Loaded model state_dict directly from {ckpt_path} (strict=False)")
        print(f"Model loaded successfully from {ckpt_path}")
    except FileNotFoundError:
        print(f"Error: Checkpoint file not found at {ckpt_path}")
        print("Please update ckpt_path to the correct location of your trained model.")
        return
    except Exception as e:
        print(f"Error loading model checkpoint: {e}")
        print("Ensure the model architecture in src/arch/cognitive.py matches the checkpoint.")
        # You could add more detailed key mismatch debugging here if needed.
        return


    print("\n--- Running Evaluation ---")
    # Run evals
    print("Evaluating with Memory...")
    # Pass input_dim and memory_dim to run_eval
    stats_mem = run_eval(model, loader, device, skip_memory=False, input_dim=input_dim, memory_dim=memory_dim)
    print("Evaluating without Memory...")
    # Pass input_dim and memory_dim to run_eval
    stats_nomem = run_eval(model, loader, device, skip_memory=True, input_dim=input_dim, memory_dim=memory_dim)

    print("\n--- Evaluation Results ---")
    names = ['price', 'volume', 'returns', 'volatility']
    print(f"\nMetric          With Memory     No Memory")
    print("-" * 50)  # Adjusted width for better alignment

    # Print Correlation
    print(f"{'Correlation':<15} ", end="")
    for i, n in enumerate(names):
        # Safely access metrics, default to NaN if index is out of bounds
        corr_mem = stats_mem['corrs'][i] if i < len(stats_mem.get('corrs', [])) else float('nan')
        corr_nomem = stats_nomem['corrs'][i] if i < len(stats_nomem.get('corrs', [])) else float('nan')
        print(f"{corr_mem: .4f} / {corr_nomem: .4f}{' ':8}", end="")
    print() # Newline after correlations

    # Print MAE
    print(f"{'MAE':<15} ", end="")
    for i, n in enumerate(names):
         # Safely access metrics
        mae_mem = stats_mem['mae'][i] if i < len(stats_mem.get('mae', [])) else float('nan')
        mae_nomem = stats_nomem['mae'][i] if i < len(stats_nomem.get('mae', [])) else float('nan')
        print(f"{mae_mem: .4f} / {mae_nomem: .4f}{' ':8}", end="")
    print() # Newline after MAE

    # Print RMSE
    print(f"{'RMSE':<15} ", end="")
    for i, n in enumerate(names):
         # Safely access metrics
        rmse_mem = stats_mem['rmse'][i] if i < len(stats_mem.get('rmse', [])) else float('nan')
        rmse_nomem = stats_nomem['rmse'][i] if i < len(stats_nomem.get('rmse', [])) else float('nan')
        print(f"{rmse_mem: .4f} / {rmse_nomem: .4f}{' ':8}", end="")
    print() # Newline after RMSE

    print(f"\nConflict Rate:   {stats_mem.get('conflict_rate', float('nan')):.4f} / {stats_nomem.get('conflict_rate', float('nan')):.4f}")  # Increased precision, safely access
    print(f"Avg Confidence:  {stats_mem.get('avg_confidence', float('nan')):.4f} / {stats_nomem.get('avg_confidence', float('nan')):.4f}\n") # Increased precision, safely access


def calculate_sharpe_ratio(returns: torch.Tensor, risk_free_rate: float = 0.02) -> float:
    """Calculate the Sharpe ratio of the returns."""
    excess_returns = returns - risk_free_rate / 252  # Assuming daily returns
    if len(excess_returns) < 2:
        return 0.0
    return float(excess_returns.mean() / (returns.std() + 1e-10) * np.sqrt(252))

def calculate_max_drawdown(cumulative_returns: torch.Tensor) -> float:
    """Calculate the maximum drawdown from peak."""
    peak = torch.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_returns - peak) / peak
    return float(drawdown.min())

def calculate_win_rate(returns: torch.Tensor) -> float:
    """Calculate the win rate of trades."""
    wins = (returns > 0).sum()
    total = len(returns)
    return float(wins / total if total > 0 else 0)

def calculate_returns_stability(returns: torch.Tensor) -> float:
    """Calculate the stability of returns using rolling window standard deviation."""
    if len(returns) < 2:
        return 0.0
    rolling_std = returns.unfold(0, 10, 1).std(dim=1)
    return float(1.0 / (rolling_std.mean() + 1e-10))

def calculate_risk_adjusted_returns(returns: torch.Tensor, 
                                 risk_metrics: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """Calculate various risk-adjusted return metrics."""
    volatility = returns.std()
    var = risk_metrics['value_at_risk'].mean()
    es = risk_metrics['expected_shortfall'].mean()
    
    return {
        'sortino_ratio': float(returns.mean() / (returns[returns < 0].std() + 1e-10) * np.sqrt(252)),
        'calmar_ratio': float(returns.mean() * 252 / (calculate_max_drawdown(returns.cumsum()) + 1e-10)),
        'var_adjusted_return': float(returns.mean() / (var + 1e-10)),
        'es_adjusted_return': float(returns.mean() / (es + 1e-10))
    }

def calculate_transaction_costs(position_changes: torch.Tensor,
                             prices: torch.Tensor,
                             base_cost: float = 0.001) -> torch.Tensor:
    """Calculate transaction costs for trades."""
    costs = torch.abs(position_changes) * prices * base_cost
    return costs

def calculate_risk_metrics(predictions, targets, risk_scores):
    """Calculate risk-adjusted performance metrics."""
    returns = (predictions - targets) / targets
    
    metrics = {
        'sharpe_ratio': calculate_sharpe_ratio(returns),
        'max_drawdown': calculate_max_drawdown(returns.cumsum()),
        'win_rate': (returns > 0).float().mean().item(),
        'risk_adjusted_return': (returns / (risk_scores + 1e-8)).mean().item()
    }
    
    return metrics

def evaluate_model(model, eval_loader, device):
    # ... existing evaluation code ...
    
    # Add risk-adjusted metrics
    risk_metrics = model.risk_assessor(fused)
    performance_metrics = calculate_risk_metrics(
        market_state, targets, risk_metrics[:, 0]
    )
    
    metrics.update(performance_metrics)
    return metrics


class BacktestEngine:
    def __init__(self, 
                 model,
                 initial_capital=100000.0,
                 transaction_cost=0.001,
                 risk_free_rate=0.02,
                 max_position=1.0):
        self.model = model
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.risk_free_rate = risk_free_rate
        self.max_position = max_position
        
        # Initialize tracking variables
        self.reset()
        
    def reset(self):
        """Reset backtest state."""
        self.portfolio_value = self.initial_capital
        self.current_position = 0.0
        self.trades = []
        self.portfolio_history = []
        self.regime_history = []
        self.position_history = []
        self.drawdown_history = []
        
    def calculate_position_size(self, prediction, confidence, risk_score):
        """Calculate position size based on prediction and risk."""
        # Direction from prediction
        price_change_pred = prediction[0]  # Assuming first feature is price
        direction = torch.sign(price_change_pred)
        
        # Size from confidence and risk
        size = confidence * (1.0 - risk_score)
        
        # Combine and clip
        position = direction * size * self.max_position
        return position.item()
        
    def execute_trade(self, current_price, new_position, timestamp):
        """Execute trade and calculate costs."""
        position_change = new_position - self.current_position
        
        if abs(position_change) > 0:
            # Calculate transaction cost
            cost = abs(position_change) * current_price * self.transaction_cost
            
            # Record trade
            self.trades.append({
                'timestamp': timestamp,
                'price': current_price,
                'position_change': position_change,
                'cost': cost,
                'portfolio_value': self.portfolio_value
            })
            
            # Update portfolio value
            self.portfolio_value -= cost
            
        # Update position
        self.current_position = new_position
        
    def update_portfolio(self, price_change):
        """Update portfolio value based on position and price change."""
        pnl = self.current_position * price_change * self.portfolio_value
        self.portfolio_value += pnl
        
        # Calculate drawdown
        peak = max(self.portfolio_history) if self.portfolio_history else self.portfolio_value
        drawdown = (peak - self.portfolio_value) / peak
        self.drawdown_history.append(drawdown)
        
    def run_backtest(self, dataloader, device):
        """Run full backtest."""
        self.reset()
        self.model.eval()
        
        results = {
            'timestamps': [],
            'prices': [],
            'positions': [],
            'portfolio_values': [],
            'regimes': [],
            'drawdowns': []
        }
        
        with torch.no_grad():
            for batch_idx, (feats, seqs, targets) in enumerate(dataloader):
                # Move to device
                feats = feats.to(device)
                seqs = seqs.to(device)
                targets = targets.to(device)
                
                # Get model predictions
                outputs = self.model(
                    financial_data=feats,
                    financial_seq=seqs
                )
                
                # Unpack predictions and metadata
                _, _, recalled_values, _, _, market_pred = outputs
                
                # Get current price and next price
                current_price = feats[0, 0].item()  # Assuming first feature is price
                next_price = targets[0, 0].item()
                price_change = (next_price - current_price) / current_price
                
                # Get confidence and risk scores
                if recalled_values is not None:
                    confidence = self.model.financial_memory.get_memory_stats()['usage_mean']
                else:
                    confidence = 0.5  # Default confidence
                
                risk_score = torch.sigmoid(self.model.risk_assessor(outputs[0]))[:, 0]
                
                # Calculate position
                new_position = self.calculate_position_size(
                    market_pred[0],  # Use first prediction
                    confidence,
                    risk_score[0]
                )
                
                # Execute trade
                timestamp = batch_idx  # Or actual timestamp if available
                self.execute_trade(current_price, new_position, timestamp)
                
                # Update portfolio
                self.update_portfolio(price_change)
                
                # Record state
                self.portfolio_history.append(self.portfolio_value)
                self.position_history.append(new_position)
                
                # Get regime probabilities
                _, regime_probs = self.model.temporal_hierarchy(seqs)
                regime_idx = regime_probs.mean(dim=1).argmax(dim=1)[0].item()
                self.regime_history.append(regime_idx)
                
                # Store results
                results['timestamps'].append(timestamp)
                results['prices'].append(current_price)
                results['positions'].append(new_position)
                results['portfolio_values'].append(self.portfolio_value)
                results['regimes'].append(regime_idx)
                results['drawdowns'].append(self.drawdown_history[-1])
        
        # Calculate performance metrics
        metrics = self.calculate_performance_metrics()
        
        return results, metrics
        
    def calculate_performance_metrics(self):
        """Calculate comprehensive performance metrics."""
        returns = np.diff(self.portfolio_history) / self.portfolio_history[:-1]
        
        metrics = {
            'total_return': (self.portfolio_value - self.initial_capital) / self.initial_capital,
            'sharpe_ratio': calculate_sharpe_ratio(torch.tensor(returns), self.risk_free_rate),
            'max_drawdown': max(self.drawdown_history),
            'win_rate': len([r for r in returns if r > 0]) / len(returns),
            'total_trades': len(self.trades),
            'avg_position_size': np.mean(np.abs(self.position_history)),
            'regime_distribution': np.bincount(self.regime_history) / len(self.regime_history)
        }
        
        # Calculate regime-specific performance
        for regime in range(3):  # Assuming 3 regimes
            regime_mask = np.array(self.regime_history[:-1]) == regime
            if regime_mask.any():
                regime_returns = returns[regime_mask]
                metrics[f'regime_{regime}_return'] = np.mean(regime_returns)
                metrics[f'regime_{regime}_sharpe'] = calculate_sharpe_ratio(
                    torch.tensor(regime_returns), 
                    self.risk_free_rate
                )
        
        return metrics


if __name__ == '__main__':
    main()

import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def plot_backtest_results(results, metrics, save_path=None):
    """Create comprehensive visualization of backtest results."""
    fig = plt.figure(figsize=(20, 12))
    
    # Portfolio value and drawdown
    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(results['portfolio_values'], label='Portfolio Value', color='blue')
    ax1.set_title(f'Portfolio Performance (Sharpe: {metrics["sharpe_ratio"]:.2f}, MDD: {metrics["max_drawdown"]:.2%})')
    ax1.grid(True)
    
    ax1_twin = ax1.twinx()
    ax1_twin.fill_between(range(len(results['drawdowns'])), 
                         results['drawdowns'], 
                         alpha=0.3, 
                         color='red', 
                         label='Drawdown')
    ax1_twin.set_ylabel('Drawdown')
    
    # Positions and price
    ax2 = plt.subplot(3, 1, 2)
    ax2.plot(results['prices'], label='Price', color='gray', alpha=0.5)
    ax2_twin = ax2.twinx()
    ax2_twin.plot(results['positions'], label='Position', color='green')
    ax2.set_title(f'Positions and Price (Win Rate: {metrics["win_rate"]:.2%})')
    ax2.grid(True)
    
    # Regime distribution
    ax3 = plt.subplot(3, 1, 3)
    regime_colors = ['green', 'yellow', 'red']
    for regime in range(3):
        mask = np.array(results['regimes']) == regime
        if mask.any():
            ax3.fill_between(range(len(results['regimes'])),
                           regime,
                           regime + 1,
                           where=mask,
                           color=regime_colors[regime],
                           alpha=0.3,
                           label=f'Regime {regime}')
    ax3.set_title('Market Regimes')
    ax3.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

def run_full_evaluation(model, eval_loader, device, save_dir=None):
    """Run comprehensive model evaluation including backtesting."""
    # Standard evaluation
    eval_metrics = run_eval(model, eval_loader, device)
    
    # Backtesting
    backtest = BacktestEngine(model)
    results, backtest_metrics = backtest.run_backtest(eval_loader, device)
    
    # Combine metrics
    all_metrics = {**eval_metrics, **backtest_metrics}
    
    # Save results
    if save_dir:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save metrics
        metrics_path = f"{save_dir}/metrics_{timestamp}.json"
        with open(metrics_path, 'w') as f:
            json.dump(all_metrics, f, indent=4)
        
        # Save visualization
        plot_path = f"{save_dir}/backtest_{timestamp}.png"
        plot_backtest_results(results, all_metrics, plot_path)
    
    return all_metrics, results

def calculate_trading_metrics(predictions, prices, positions):
    """Calculate essential trading metrics."""
    returns = np.diff(prices) / prices[:-1]
    position_returns = positions[:-1] * returns
    
    metrics = {
        'total_return': float(np.prod(1 + position_returns) - 1),
        'sharpe_ratio': float(np.mean(position_returns) / (np.std(position_returns) + 1e-8) * np.sqrt(252)),
        'max_drawdown': float(np.max(np.maximum.accumulate(np.cumprod(1 + position_returns)) - np.cumprod(1 + position_returns))),
        'win_rate': float(np.mean(position_returns > 0)),
        'profit_factor': float(np.sum(position_returns[position_returns > 0]) / (abs(np.sum(position_returns[position_returns < 0])) + 1e-8))
    }
    
    return metrics
