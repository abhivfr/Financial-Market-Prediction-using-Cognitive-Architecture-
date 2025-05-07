import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import traceback
import argparse
from datetime import datetime
import numpy as np
import pandas as pd
import time
import random

torch.autograd.set_detect_anomaly(True)

# Assuming src.utils, src.monitoring, src.arch, src.core are correctly
# structured in your project directory and contain the necessary files.
try:
    from src.utils.thermal import ThermalGovernor
    from src.monitoring.introspect import Introspection
    from src.arch.cognitive import CognitiveArchitecture
    from src.core import NeuralDictionary # Import NeuralDictionary if used directly
    from src.monitoring.adaptive_learning import AdaptiveLearning
    from src.utils.uncertainty_loss import UncertaintyAwareLoss
    from src.utils.market_loss import MarketAwareLoss
    from src.finance.volume_attention import VolumeAttention
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure the src directory structure is correct and contains the required files.")
    sys.exit(1)

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# ============================
# PyTorch and Device Setup
# ============================
print(f"PyTorch version: {torch.__version__}")
# Check if CUDA is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
if torch.cuda.is_available():
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"Current CUDA device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    # Optional: Print a small tensor on the GPU to confirm device assignment
    try:
        test_tensor = torch.randn(2, 2, device='cuda')
        print(f"Test tensor on device: {test_tensor.device}")
    except Exception as e:
        print(f"Error creating test tensor on GPU: {e}")
    torch.backends.cudnn.benchmark = True
else:
    print("CUDA is not available. PyTorch will use CPU.")


# ============================
# Argument Parsing
# ============================
parser = argparse.ArgumentParser(description="Financial Consciousness Training and Evaluation")
parser.add_argument("--financial_train", type=str, default="data/financial/sample_data.csv", help="Path to training market data") # Renamed for clarity
parser.add_argument("--financial_val", type=str, default=None, help="Path to validation market data") # Added validation data arg
parser.add_argument("--financial_test", type=str, default=None, help="Path to test market data") # Added test data arg
parser.add_argument("--iters", type=int, default=1000, help="Training iterations")
parser.add_argument("--batch", type=int, default=16, help="Batch size")
parser.add_argument("--verbose", action="store_true", help="Enable verbose logging during training")
parser.add_argument("--seq_length", type=int, default=10, help="Sequence length for temporal encoder and market state prediction")
parser.add_argument("--eval_interval", type=int, default=0, help="Interval (in iterations) to run validation evaluation (0 for no validation during training)") # Added eval interval
parser.add_argument("--save_interval", type=int, default=100, help="Interval (in iterations) to save model checkpoints") # Added save interval
parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for optimizer") # Added learning rate arg
parser.add_argument("--backtest", action="store_true", help="Enable backtest-driven training")

args = parser.parse_args()


# ============================
# Financial Data Loader
# ============================
class FinancialDataLoader:
    def __init__(self, path, seq_length=10, batch_size=16):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Data file not found at {path}")
        raw = pd.read_csv(path)
        self.data = self._process_raw(raw)
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.pointer = 0
        # Calculate the maximum possible start index for a sequence and target
        self._max_pointer = len(self.data) - self.seq_length - 1
        if self._max_pointer < 0:
             raise ValueError(f"Data file {path} is too short for the specified sequence length ({seq_length}). Needs at least {seq_length + 1} rows after preprocessing.")


    def _process_raw(self, df):
        # Keep only the needed columns if they exist in the dataframe
        required_cols = ['price', 'volume']
        if all(col in df.columns for col in required_cols):
            # If returns and volatility already exist, use them
            if 'returns' in df.columns and 'volatility' in df.columns:
                features = df[['price', 'volume', 'returns', 'volatility']].values
            else:
                # Calculate them if they don't exist
                df['returns'] = df['price'].pct_change()
                df['volatility'] = df['price'].rolling(window=5).std()
                df = df.dropna(subset=['price', 'volume', 'returns', 'volatility'])
                features = df[['price', 'volume', 'returns', 'volatility']].values
        else:
            raise ValueError("Required columns not found in data")
        
        # Ensure required columns exist
        if 'price' not in df.columns or 'volume' not in df.columns:
             raise ValueError("Data file must contain 'price' and 'volume' columns.")

        # --- Added: Explicitly convert price and volume columns to numeric ---
        # Use errors='coerce' to turn any values that cannot be converted into NaNs
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        df['volume'] = pd.to_numeric(df['volume'], errors='coerce')


        df['returns'] = df['price'].pct_change()
        df['volatility'] = df['price'].rolling(window=5).std()
        # Handle potential NaNs created by rolling window and pct_change
        # Drop initial rows where volatility or returns are NaN
        # Also drop rows where price or volume became NaN due to coercion
        df = df.dropna(subset=['price', 'volume', 'returns', 'volatility'])

        # Ensure we still have data after dropping NaNs
        if df.empty:
             raise ValueError("Data is empty after preprocessing (e.g., dropping NaNs). Check raw data for non-numeric values or insufficient length.")


        features = df[['price', 'volume', 'returns', 'volatility']].values

        # Handle potential remaining NaNs (e.g., from missing data points elsewhere)
        # Option 1: Drop rows with NaNs (simplest but loses data)
        # features = features[~np.isnan(features).any(axis=1)]
        # Option 2: Impute NaNs (more complex, requires strategy like mean, median, forward fill)
        # For now, we've dropped rows with NaNs in essential columns and derived features


        means = np.nanmean(features, axis=0)
        stds = np.nanstd(features, axis=0)
        # Avoid division by zero if a column has zero standard deviation
        # Add a small epsilon to stds before division to avoid inf/nan
        normalized = (features - means) / (stds + 1e-8)

        # Final check for NaNs after normalization
        if np.isnan(normalized).any():
             print("🚨 NaNs detected in preprocessed financial data after normalization!")
             # You might want to inspect the data or choose a different normalization/imputation method
             # For now, we'll proceed but be aware of potential issues
             # assert not np.isnan(normalized).any(), "🚨 NaNs detected in preprocessed financial data!" # Keep this assertion if you want to hard fail on NaNs


        assert normalized.shape[1] == 4, f"Expected 4 features after processing, but got {normalized.shape[1]}"
        return normalized

    def __iter__(self):
        self.pointer = 0 # Reset pointer at the start of iteration
        return self

    def __next__(self):
        # Ensure we have enough data for a batch AND the subsequent target
        # plus enough for the sequence
        # Recalculate _max_pointer here based on the potentially reduced size after dropping NaNs in _process_raw
        current_max_pointer = len(self.data) - self.seq_length - 1

        if current_max_pointer < 0:
             raise ValueError(f"Not enough data ({len(self.data)} samples) remaining after preprocessing for sequence length ({self.seq_length}).")


        if self.pointer + self.batch_size > current_max_pointer:
             # print("ℹ️ Data loader reached end of data. Resetting for next epoch.")
             self.pointer = 0 # Reset pointer for next epoch
             raise StopIteration


        batch_indices = np.arange(self.pointer, self.pointer + self.batch_size)
        self.pointer += self.batch_size

        # sequence_data is the historical sequence for the temporal encoder
        sequence_data = np.stack([self.data[i : i + self.seq_length] for i in batch_indices])
        # features_data is the current step's features (the last in the sequence)
        # This is financial_data input to the model's forward pass
        features_data = self.data[batch_indices + self.seq_length - 1]

        # targets_data are the actual values for the *next* timestep after the sequence
        # This is the ground truth for the prediction loss (comparing to the last step of market_state)
        targets_data = self.data[batch_indices + self.seq_length]

        # Debug: verify shapes
        # sequence: (batch_size, seq_length, 4)
        # features: (batch_size, 4) - Current step
        # targets: (batch_size, 4) - Next step
        if args.verbose:
             print(f"Batch shapes → features: {features_data.shape}, sequence: {sequence_data.shape}, targets: {targets_data.shape}")


        return {
            'features': torch.from_numpy(features_data).float(),
            'sequence': torch.from_numpy(sequence_data).float(),
            'target': torch.from_numpy(targets_data).float()
        }

    def __len__(self):
        # Number of possible batches in one pass through the data
        current_max_pointer = len(self.data) - self.seq_length - 1
        if current_max_pointer < 0:
             return 0
        # Calculate the number of full batches
        num_full_batches = (current_max_pointer + 1 - self.batch_size) // self.batch_size
        # Add 1 if there's a partial batch at the end (assuming partial batches are used)
        # If partial batches are NOT used, remove the + (remainder > 0) part
        remainder = (current_max_pointer + 1 - self.batch_size) % self.batch_size
        return max(0, num_full_batches + (1 if remainder > 0 else 0))


# ============================
# Metric Calculation Functions
# ============================

# --- Implementation of the 4D Market Metric calculation functions ---

def calculate_price_accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Calculate accuracy of price predictions
    Args:
        predictions: Model predictions (batch_size, 4)
        targets: Ground truth (batch_size, 4)
    Returns:
        Price accuracy score
    """
    # Extract price dimension (index 0)
    predicted_price = predictions[:, 0].detach()  # Add detach() here
    actual_price = targets[:, 0].detach()  # Add detach() here
    
    # Convert to numpy for calculations
    pred_np = predicted_price.cpu().numpy()
    actual_np = actual_price.cpu().numpy()
    
    # Calculate directional accuracy
    pred_direction = np.diff(pred_np) > 0
    actual_direction = np.diff(actual_np) > 0
    directional_accuracy = np.mean(pred_direction == actual_direction)
    
    # Calculate normalized RMSE
    rmse = np.sqrt(np.mean((pred_np - actual_np) ** 2))
    price_range = np.max(actual_np) - np.min(actual_np)
    normalized_rmse = 1 - (rmse / (price_range + 1e-8))
    
    # Combine metrics
    return 0.7 * directional_accuracy + 0.3 * normalized_rmse

def calculate_volume_correlation(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Calculate correlation between predicted and actual volume changes
    """
    # Extract volume dimension (index 1)
    pred_vol = predictions[:, 1].detach()  # Add detach() here
    target_vol = targets[:, 1].detach()  # Add detach() here
    
    # Calculate changes
    pred_changes = pred_vol[1:] - pred_vol[:-1]
    target_changes = target_vol[1:] - target_vol[:-1]
    
    # Convert to numpy
    pred_np = pred_changes.cpu().numpy()
    target_np = target_changes.cpu().numpy()
    
    # Calculate correlation
    return np.corrcoef(pred_np, target_np)[0, 1]

def calculate_returns_stability(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Calculate stability of returns predictions
    """
    # Extract returns dimension (index 2)
    pred_returns = predictions[:, 2].detach()  # Add detach() here
    target_returns = targets[:, 2].detach()  # Add detach() here
    
    # Convert to numpy
    pred_np = pred_returns.cpu().numpy()
    target_np = target_returns.cpu().numpy()
    
    # Calculate stability score
    pred_volatility = np.std(pred_np)
    target_volatility = np.std(target_np)
    volatility_ratio = min(pred_volatility, target_volatility) / max(pred_volatility, target_volatility)
    
    return volatility_ratio

def calculate_volatility_prediction(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Calculate accuracy of volatility predictions
    """
    # Extract volatility dimension (index 3)
    pred_vol = predictions[:, 3].detach()  # Add detach() here
    target_vol = targets[:, 3].detach()  # Add detach() here
    
    # Convert to numpy
    pred_np = pred_vol.cpu().numpy()
    target_np = target_vol.cpu().numpy()
    
    # Calculate regime accuracy
    pred_regime = pred_np > np.median(pred_np)
    target_regime = target_np > np.median(target_np)
    regime_accuracy = np.mean(pred_regime == target_regime)
    
    # Calculate correlation
    correlation = np.corrcoef(pred_np, target_np)[0, 1]
    
    # Combine metrics
    return 0.6 * regime_accuracy + 0.4 * correlation


# ============================
# Loss Function
# ============================
def calculate_loss(outputs, targets):
    fused_output, fin_feat, recalled, attended, attn_weights, market_state = outputs

    # --- NaN Debug Checks --- (Keep these, they are useful)
    if args.verbose:
        print("\n--- NaN Debug Check - Outputs & Targets ---")
        if torch.isnan(market_state).any():
            print("🚨 NaNs detected in market_state")
        else:
            print("✅ No NaNs in market_state")

        if torch.isnan(targets).any():
            print("🚨 NaNs detected in targets")
        else:
             print("✅ No NaNs in targets")

        if fin_feat is not None and torch.isnan(fin_feat).any():
             print("🚨 NaNs detected in fin_feat")
        elif fin_feat is not None:
             print("✅ No NaNs in fin_feat")
        else:
             print("ℹ️ fin_feat is None")

        if recalled is not None and torch.isnan(recalled).any():
             print("🚨 NaNs detected in recalled")
        elif recalled is not None:
             print("✅ No NaNs in recalled")
        else:
             print("ℹ️ recalled is None")

        if attended is not None and torch.isnan(attended).any():
             print("🚨 NaNs detected in attended")
        elif attended is not None:
             print("✅ No NaNs in attended")
        else:
             print("ℹ️ attended is None")

        if attn_weights is not None and torch.isnan(attn_weights).any():
             print("🚨 NaNs detected in attn_weights")
        elif attn_weights is not None:
             print("✅ No NaNs in attn_weights")
        else:
            print("ℹ️ attn_weights is None")
        print("---------------------------------")


    # --- Loss Calculations ---

    # Pred loss: Compare the LAST step of the predicted market_state sequence
    # with the single-step target for the NEXT timestep.
    # market_state is now (batch_size, seq_len, input_dim)
    # targets is (batch_size, input_dim)
    # Take the last element of the sequence dimension: market_state[:, -1]
    if market_state.ndim < 2 or market_state.shape[1] == 0:
         print("🚨 Error: market_state tensor does not have enough dimensions or sequence length for slicing for pred_loss.")
         return torch.tensor(float('inf'), device=market_state.device)

    # Check if the last dimension of market_state matches the dimension of targets
    if market_state.shape[-1] != targets.shape[-1]:
         print(f"🚨 Error: Last dimension mismatch for pred_loss: market_state.shape[-1]={market_state.shape[-1]}, targets.shape[-1]={targets.shape[-1]}")
         return torch.tensor(float('inf'), device=market_state.device)

    # Ensure targets has an explicit sequence dimension of 1 for consistency with market_state[:, -1]
    # Although MSELoss handles broadcasting, being explicit can help debugging
    # targets_for_pred_loss = targets.unsqueeze(1) # Shape (batch_size, 1, input_dim)
    # pred_loss = nn.MSELoss()(market_state[:, -1].unsqueeze(1), targets_for_pred_loss) # Compare (batch, 1, 4) with (batch, 1, 4)
    # Or simply:
    pred_loss = nn.MSELoss()(market_state[:, -1], targets) # Compare (batch, 4) with (batch, 4)


    # Calculate squared differences for sequence loss
    # Now market_state.shape[1] should be > 1 (e.g., 10)
    seq_loss = torch.tensor(0.0, device=market_state.device) # Initialize seq_loss
    if market_state.shape[1] > 1:
        # Calculate difference between consecutive steps in the predicted sequence
        # This results in a tensor of shape (batch_size, seq_len - 1, input_dim)
        squared_diffs = (market_state[:, 1:] - market_state[:, :-1]) ** 2

        if args.verbose:
            # ✅ <--- CRITICAL DEBUG SECTION - FORCING PRINTS ---
            print("\n--- Seq Loss Debug Check ---")
            print(f"market_state shape for seq_loss: {market_state.shape}") # Added this print
            print(f"squared_diffs shape: {squared_diffs.shape}")
            print(f"squared_diffs device: {squared_diffs.device}")

            if torch.isnan(squared_diffs).any():
                print("🚨 NaNs detected in squared_diffs (seq_loss intermediate)")
                nan_indices = torch.nonzero(torch.isnan(squared_diffs), as_tuple=False)
                if nan_indices.numel() > 0: # Check if there are any NaN indices before trying to print
                    print(f"Sample squared_diffs (NaN) at indices {nan_indices[:min(5, nan_indices.shape[0])].tolist()}: {squared_diffs[nan_indices[:min(5, nan_indices.shape[0])].unbind(1)].tolist()}")
            elif torch.isinf(squared_diffs).any():
                 print("🚨 Infs detected in squared_diffs (seq_loss intermediate) - LIKELY CAUSE OF NaN!")
                 inf_indices = torch.nonzero(torch.isinf(squared_diffs), as_tuple=False)
                 if inf_indices.numel() > 0: # Check before printing
                     print(f"Sample squared_diffs (Inf) at indices {inf_indices[:min(5, inf_indices.shape[0])].tolist()}: {squared_diffs[inf_indices[:min(5, inf_indices.shape[0])].unbind(1)].tolist()}")
            else:
                print("✅ No NaNs or Infs in squared_diffs")

            # Print sample values of squared_diffs unconditionally - check if flatten has enough elements
            if squared_diffs.numel() > 0:
                 print("Sample squared_diffs (first min(10, squared_diffs.numel()) values):", squared_diffs.flatten()[:min(10, squared_diffs.numel())].tolist())
            else:
                 print("Sample squared_diffs: Tensor is empty.")
            print("--------------------------")


        # Mean of squared differences across the sequence dimension and batch dimension
        seq_loss = torch.mean(squared_diffs)

    # else block is implicitly handled by initializing seq_loss to 0.0 above the if


    # Consistency loss (remains the same, comparing fin_feat to mean of recalled_values)
    consistency = torch.tensor(0.0, device=market_state.device) # Initialize
    if fin_feat is not None and fin_feat.numel() > 0 and not torch.isnan(fin_feat).any():
        normalized_fin_feat = F.normalize(fin_feat, dim=-1, eps=1e-8)

        if recalled is not None and recalled.numel() > 0 and not torch.isnan(recalled).any():
            # Ensure recalled has at least 2 dimensions for mean(1)
            if recalled.ndim > 1:
                normalized_recalled = F.normalize(recalled.mean(1), dim=-1, eps=1e-8)
                # Ensure dimensions match for CosineEmbeddingLoss
                if normalized_fin_feat.size() == normalized_recalled.size():
                    # Create target tensor for CosineEmbeddingLoss (should be 1 for similarity)
                    cosine_target = torch.ones(normalized_fin_feat.size(0), device=normalized_fin_feat.device)
                    consistency = nn.CosineEmbeddingLoss()(
                        normalized_fin_feat, normalized_recalled,
                        cosine_target
                    )
                elif args.verbose:
                     print("Warning: Size mismatch between normalized_fin_feat and normalized_recalled for CosineEmbeddingLoss. Setting consistency to 0.")
            elif args.verbose:
                 print("Warning: 'recalled' has insufficient dimensions for mean(1). Setting consistency to 0.")
        elif args.verbose:
            print("Warning: 'recalled' is invalid or None for consistency loss, setting consistency to 0.")
    elif args.verbose:
         print("Warning: 'fin_feat' is invalid or None for consistency loss, setting consistency to 0.")


    # Attention stability and sparsity losses (remain the same)
    attn_stability = torch.tensor(0.0, device=market_state.device) # Initialize
    sparsity = torch.tensor(0.0, device=market_state.device) # Initialize
    if attn_weights is not None and attn_weights.numel() > 0 and not torch.isnan(attn_weights).any():
        # attn_weights shape is (batch_size, num_heads, 1, seq_len_k)
        # std(1) calculates std deviation across the num_heads dimension
        # mean() then averages across batch, 1 (query_seq_len), and key_seq_len dimensions
        if attn_weights.ndim >= 2 and attn_weights.shape[1] > 0: # Needs at least 2 dimensions and non-empty head dim for std(1)
             attn_stability = attn_weights.add(1e-8).std(1).mean() # std across heads, mean across batch and sequence
             sparsity = torch.mean(attn_weights ** 2) # mean across all elements
        elif args.verbose:
             print("Warning: 'attn_weights' has insufficient dimensions or head count for stability/sparsity. Setting to 0.")
    elif args.verbose:
        print("Warning: 'attn_weights' is invalid or None, setting stability and sparsity loss to 0.")


    total = (
        0.4 * pred_loss +
        0.2 * seq_loss +
        0.2 * consistency +
        0.1 * attn_stability +
        0.1 * sparsity
    )

    # Total Loss Check (keep this)
    if torch.isnan(total):
        print("\n🚨🚨💥 TOTAL LOSS IS NaN 💥🚨🚨")
        # Include individual loss components in the error message for easy viewing
        # Use .item() for scalar tensors to print just the number
        # Added checks before .item() to prevent error if a component is NaN
        print(f"Loss components: pred_loss={pred_loss.item() if not torch.isnan(pred_loss) else 'nan'}, seq_loss={seq_loss.item() if not torch.isnan(seq_loss) else 'nan'}, consistency={consistency.item() if not torch.isnan(consistency) else 'nan'}, attn_stability={attn_stability.item() if not torch.isnan(attn_stability) else 'nan'}, sparsity={sparsity.item() if not torch.isnan(sparsity) else 'nan'}")
        raise ValueError("💥 TOTAL LOSS IS NaN - Check debug prints above for origin.")

    return total


# ============================
# Training Loop
# ============================
def compute_risk_adjusted_loss(pred, target, risk_score):
    """Compute risk-weighted loss."""
    base_loss = F.mse_loss(pred, target, reduction='none')
    return (base_loss * (1 + risk_score)).mean()

def train_step(model, optimizer, financial_data, financial_seq, targets, adaptive_learning, introspect):
    """Single training step with enhanced self-regulation"""
    device = financial_data.device
    
    # Initialize loss functions if not already done
    if not hasattr(train_step, 'market_loss_fn'):
        train_step.market_loss_fn = MarketAwareLoss().to(device)
    if not hasattr(train_step, 'uncertainty_loss_fn'):
        train_step.uncertainty_loss_fn = UncertaintyAwareLoss(base_loss='mse', beta=0.1).to(device)
    
    # Extract volume data
    volume_data = financial_seq[:, :, 1:2]
    
    # Forward pass with monitoring
    outputs = model(
        financial_data=financial_data,
        financial_seq=financial_seq,
        volume=volume_data,
        skip_memory=False
    )
    
    # Get monitoring stats
    monitoring_stats = model.get_monitoring_stats()
    
    # Update adaptive learning
    lr_multipliers = adaptive_learning.update(monitoring_stats)
    
    # Apply learning rate updates
    for param_group in optimizer.param_groups:
        component_name = param_group.get('name', 'core')
        param_group['lr'] = adaptive_learning.get_lr_for_param_group(component_name)
    
    # Calculate losses
    market_loss = train_step.market_loss_fn(outputs['market_state'], targets)
    uncertainty_loss = train_step.uncertainty_loss_fn(
        outputs['market_state'],
        targets,
        outputs.get('uncertainty', None)
    )
    
    # Total loss
    total_loss = market_loss + uncertainty_loss
    
    # Backward pass with gradient monitoring
    total_loss.backward()
    
    # Clip gradients based on monitoring
    max_norm = 1.0 if monitoring_stats['core_metrics']['layer_gradients']['mean'] < 10 else 0.5
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
    
    # Update weights
    optimizer.step()
    optimizer.zero_grad()
    
    # Log to introspection
    introspect.log(model, torch.cuda.memory_allocated() if torch.cuda.is_available() else 0)
    
    # Calculate metrics
    with torch.no_grad():
        metrics = {
            'price_accuracy': calculate_price_accuracy(outputs['market_state'], targets),
            'volume_correlation': calculate_volume_correlation(outputs['market_state'], targets),
            'returns_stability': calculate_returns_stability(outputs['market_state'], targets),
            'volatility_prediction': calculate_volatility_prediction(outputs['market_state'], targets)
        }
    
    return total_loss.item(), outputs, metrics, monitoring_stats

def train(model, device, financial_train_path, financial_val_path, num_iters=1000, seq_length=10, eval_interval=0, save_interval=100):
    """Main training loop with enhanced self-regulation"""
    model.train()
    
    # Initialize monitoring components
    adaptive_learning = AdaptiveLearning(base_lr=args.lr)
    introspect = Introspection()
    governor = ThermalGovernor()
    
    # Initialize data loader
    try:
        train_loader = FinancialDataLoader(financial_train_path, seq_length=seq_length, batch_size=args.batch)
        print(f"✅ Loaded training data from {financial_train_path}")
    except (FileNotFoundError, ValueError) as e:
        print(f"Error loading training data from {financial_train_path}: {e}")
        sys.exit(1)
    
    # Create optimizer with parameter groups
    optimizer = optim.Adam([
        {'params': [p for n, p in model.named_parameters() if 'financial_memory' in n], 'lr': args.lr, 'name': 'memory'},
        {'params': [p for n, p in model.named_parameters() if 'attention' in n], 'lr': args.lr, 'name': 'attention'},
        {'params': [p for n, p in model.named_parameters() if 'financial_memory' not in n and 'attention' not in n], 
         'lr': args.lr, 'name': 'core'},
    ])
    
    print(f"Starting training for {num_iters} iterations...")
    
    try:
        train_iterator = iter(train_loader)
        
        for iter_num in range(num_iters):
            # Thermal management
            if governor.check() == "throttle":
                print(f"[Iter {iter_num}] 🔥 Throttling for thermal management")
                time.sleep(1)
                continue
            
            # Get batch
            try:
                batch = next(train_iterator)
            except StopIteration:
                train_iterator = iter(train_loader)
                batch = next(train_iterator)
            
            # Move data to device
            financial_data = batch['features'].to(device)
            financial_seq = batch['sequence'].to(device)
            targets = batch['target'].to(device)
            
            # Training step with monitoring
            loss, outputs, metrics, monitoring_stats = train_step(
                model, optimizer,
                financial_data, financial_seq, targets,
                adaptive_learning, introspect
            )
            
            # Log comprehensive status every 50 iterations
            if iter_num % 50 == 0:
                print(f"\n🧠 Iter {iter_num}/{num_iters} | Loss: {loss:.4f}")
                
                print("\n=== Market Metrics ===")
                for key, value in metrics.items():
                    print(f"{key}: {value:.4f}")
                
                print("\n=== Learning Status ===")
                print(f"Learning Rates: {adaptive_learning.lr_multipliers}")
                if 'core_metrics' in monitoring_stats:
                    print(f"Gradient Norm: {monitoring_stats['core_metrics']['layer_gradients']['mean']:.4f}")
                if 'attention_variance' in monitoring_stats:
                    if isinstance(monitoring_stats['attention_variance'], dict):
                        print("Attention Variance:")
                        for k, v in monitoring_stats['attention_variance'].items():
                            print(f"  - {k}: {v:.4f}")
                    else:
                        print(f"Attention Variance: {monitoring_stats['attention_variance']:.4f}")
                
                print("\n=== System Status ===")
                introspect.report()
                
            # Save checkpoints
            if save_interval > 0 and (iter_num + 1) % save_interval == 0:
                checkpoint_path = f"checkpoints/model_iter_{iter_num+1}.pth"
                os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'adaptive_learning_state': adaptive_learning.lr_multipliers,
                    'iteration': iter_num,
                    'loss': loss,
                    'metrics': metrics
                }, checkpoint_path)
                print(f"✅ Checkpoint saved to {checkpoint_path}")
                
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user.")
        
    finally:
        # Save final model
        final_model_path = f"models/financial_consciousness_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
        os.makedirs(os.path.dirname(final_model_path), exist_ok=True)
        torch.save(model.state_dict(), final_model_path)
        print(f"✅ Model saved to {final_model_path}")


# ============================
# Evaluation Function
# ============================
# --- This is the function for calculating market metrics ---

def evaluate_model(model, data_loader, device):
    model.eval()  # Set the model to evaluation mode
    print("📊 Starting evaluation...")

    # Dictionary to accumulate metrics over the dataset
    total_metrics = {
        'price_accuracy': 0.0,
        'volume_correlation': 0.0,
        'returns_stability': 0.0,
        'volatility_prediction': 0.0,
    }
    num_batches = 0

    # Use torch.no_grad() to disable gradient calculations
    with torch.no_grad():
        # Iterate through batches
        for batch in data_loader:
            financial = {k: v.to(device) for k, v in batch.items()}

            # Get model outputs (predictions)
            outputs = model(
                financial_data=financial['features'],
                financial_seq=financial['sequence']
            )
            
            # Extract market_state from outputs if it's a dictionary
            if isinstance(outputs, dict):
                market_state = outputs['market_state']
            else:
                # Unpack tuple if model returns multiple values
                fused, financial_feat, recalled, attended, attn_weights, market_state = outputs
                
            targets = financial['target']  # The true values for the next step

            # Calculate metrics
            try:
                # Price accuracy: correlation between predicted and actual prices
                price_pred = market_state[:, -1, 0] if market_state.dim() > 2 else market_state[:, 0]
                price_true = targets[:, 0]
                price_corr = torch.corrcoef(torch.stack([price_pred, price_true]))[0, 1]
                total_metrics['price_accuracy'] += float(price_corr) if not torch.isnan(price_corr) else 0.0
                
                # Volume correlation
                vol_pred = market_state[:, -1, 1] if market_state.dim() > 2 else market_state[:, 1]
                vol_true = targets[:, 1]
                vol_corr = torch.corrcoef(torch.stack([vol_pred, vol_true]))[0, 1]
                total_metrics['volume_correlation'] += float(vol_corr) if not torch.isnan(vol_corr) else 0.0
                
                # Returns stability (inverse of mean absolute error)
                returns_pred = market_state[:, -1, 2] if market_state.dim() > 2 else market_state[:, 2]
                returns_true = targets[:, 2]
                returns_mae = torch.mean(torch.abs(returns_pred - returns_true))
                returns_stability = 1.0 / (1.0 + returns_mae)
                total_metrics['returns_stability'] += float(returns_stability)
                
                # Volatility prediction correlation
                vol_pred = market_state[:, -1, 3] if market_state.dim() > 2 else market_state[:, 3]
                vol_true = targets[:, 3]
                vol_pred_corr = torch.corrcoef(torch.stack([vol_pred, vol_true]))[0, 1]
                total_metrics['volatility_prediction'] += float(vol_pred_corr) if not torch.isnan(vol_pred_corr) else 0.0
                
            except Exception as e:
                print(f"Warning: Error calculating batch metrics: {e}")

            num_batches += 1

    # Calculate average metrics
    average_metrics = {}
    if num_batches > 0:
        for metric_name, total_value in total_metrics.items():
            average_metrics[metric_name] = total_value / num_batches
    else:
        print("Warning: No batches processed during evaluation.")
        for metric_name in total_metrics.keys():
            average_metrics[metric_name] = float('nan')

    # Report metrics
    print("\n=== Evaluation Metrics ===")
    if average_metrics:
        for metric_name, value in average_metrics.items():
            print(f"{metric_name.replace('_', ' ').title()}: {value:.4f}")
    else:
        print("No evaluation metrics to report.")
    print("==========================")

    model.train()  # Set the model back to training mode
    return average_metrics  # Return the calculated metrics


# ============================
# Main Execution Block
# ============================
if __name__ == "__main__":
    print(f"🚀 Initializing training on {device}")

    # Instantiate the model
    # Pass input_dim and hidden_dim if they are different from defaults (currently 4 and 256)
    # You might want to make hidden_dim an argparse argument as well
    # Check that input_dim matches the number of features processed by the data loader (should be 4)
    model = CognitiveArchitecture(
        input_dim=4,
        hidden_dim=256,
        memory_size=1000
    ).to(device)
    
    # Initialize volume attention separately if needed
    volume_attention = VolumeAttention(hidden_dim=256).to(device)
    model.volume_processor = volume_attention
    
    print("🧠 Loaded CognitiveArchitecture for 4D Financial Training")

    # --- Start Training ---
    print("\n--- Starting Training ---")
    try:
        # Pass validation data path and intervals to train function
        train(model, device, args.financial_train, args.financial_val, args.iters, args.seq_length, args.eval_interval, args.save_interval)
        print("\n--- Training Complete ---")

    except Exception as e:
        print(f"💥 Training failed: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

    # --- Run Final Evaluation after Training ---
    # You can use the validation data or a separate test data file
    # Ensure args.financial_test is provided if you want a separate test evaluation
    final_eval_data_path = args.financial_test if args.financial_test else args.financial_val

    if final_eval_data_path:
         print("\n--- Running Final Evaluation ---")
         try:
             # Create a data loader for the final evaluation
             # Use a batch size that works for evaluation (can be same as training, or larger/smaller)
             final_eval_loader = FinancialDataLoader(final_eval_data_path, seq_length=args.seq_length, batch_size=args.batch)
             final_metrics = evaluate_model(model, final_eval_loader, device)
             # Optional: You could save these final_metrics to a file (e.g., JSON, CSV)
             # import json
             # with open("final_evaluation_metrics.json", "w") as f:
             #      json.dump(final_metrics, f, indent=4)
             # print("Final metrics saved to final_evaluation_metrics.json")

             print("--- Final Evaluation Complete ---")

         except (FileNotFoundError, ValueError) as e:
             print(f"Error loading data for final evaluation from {final_eval_data_path}: {e}. Skipping final evaluation.")
         except Exception as e:
             print(f"💥 Final Evaluation failed: {str(e)}")
             traceback.print_exc()

    else:
         print("\nNo validation or test data path provided for final evaluation. Skipping final evaluation.")

    if args.backtest:
        print("🔄 Initializing backtest-driven training")
        backtest_engine = BacktestEngine(device=device)
        
        train_with_backtesting(
            model=model,
            train_loader=FinancialDataLoader(args.financial_train, seq_length=args.seq_length, batch_size=args.batch),
            val_loader=FinancialDataLoader(args.financial_val, seq_length=args.seq_length, batch_size=args.batch),
            optimizer=torch.optim.Adam(model.parameters(), lr=args.lr),
            num_epochs=args.iters // 50,  # Convert iterations to epochs
            device=device,
            save_dir="checkpoints/backtest"
        )

def train_with_backtesting(model, train_loader, val_loader, optimizer, num_epochs, device, save_dir):
    best_sharpe = -float('inf')
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_losses = []
        
        for batch in train_loader:
            optimizer.zero_grad()
            financial_data = batch['features'].to(device)
            financial_seq = batch['sequence'].to(device)
            targets = batch['target'].to(device)
            
            loss = train_step(model, optimizer, financial_data, financial_seq, targets)
            train_losses.append(loss.item())
            
        # Validation and backtesting
        if epoch % 5 == 0:  # Run backtest every 5 epochs
            metrics, results = run_full_evaluation(
                model, 
                val_loader,
                device,
                save_dir=f"{save_dir}/epoch_{epoch}"
            )
            
            # Save model if Sharpe ratio improves
            if metrics['sharpe_ratio'] > best_sharpe:
                best_sharpe = metrics['sharpe_ratio']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'metrics': metrics,
                }, f"{save_dir}/best_model.pth")

def adaptive_training(model, device, train_loader, val_loader, num_epochs=10):
    """Adaptive training with dynamic learning rate and batch size."""
    optimizer = torch.optim.Adam([
        {'params': model.financial_memory.parameters(), 'lr': 1e-4},
        {'params': model.attention.parameters(), 'lr': 1e-4},
        {'params': model.temporal_hierarchy.parameters(), 'lr': 1e-4},
        {'params': [p for n, p in model.named_parameters() 
                   if not any(x in n for x in ['financial_memory', 'attention', 'temporal_hierarchy'])],
         'lr': 1e-3}
    ])
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            
            # Forward pass with memory
            outputs = model(financial_data=batch['features'].to(device),
                          financial_seq=batch['sequence'].to(device))
            
            # Calculate loss
            loss = calculate_adaptive_loss(outputs, batch['target'].to(device))
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = validate_model(model, val_loader, device)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Consolidate memory periodically
        if epoch % 5 == 0:
            model.consolidate_memory()
