import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from datetime import datetime
import numpy as np

# Ensure src is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.arch.baseline_lstm import FinancialLSTMBaseline
# Corrected import path
from src.data.financial_loader import EnhancedFinancialDataLoader

# Define metric calculation functions
def calculate_price_accuracy(predictions, targets):
    """Calculate price prediction accuracy (correlation coefficient)"""
    p = predictions[:, 0].detach().cpu().numpy()
    t = targets[:, 0].detach().cpu().numpy()
    if len(p) > 1 and not np.isnan(p).any() and not np.isnan(t).any():
        corr = np.corrcoef(p, t)[0, 1]
        return corr if not np.isnan(corr) else 0.0
    return 0.0

def calculate_volume_correlation(predictions, targets):
    """Calculate volume prediction correlation"""
    p = predictions[:, 1].detach().cpu().numpy()
    t = targets[:, 1].detach().cpu().numpy()
    if len(p) > 1 and not np.isnan(p).any() and not np.isnan(t).any():
        corr = np.corrcoef(p, t)[0, 1]
        return corr if not np.isnan(corr) else 0.0
    return 0.0

def calculate_returns_stability(predictions, targets):
    """Calculate returns prediction stability (1 / (1 + MAE))"""
    if predictions.shape[1] < 3 or targets.shape[1] < 3:
        return 0.0
    p = predictions[:, 2].detach().cpu().numpy()
    t = targets[:, 2].detach().cpu().numpy()
    if np.isnan(p).any() or np.isnan(t).any():
        return 0.0
    mae = np.mean(np.abs(p - t))
    return 1.0 / (1.0 + mae)

def calculate_volatility_prediction(predictions, targets):
    """Calculate volatility prediction accuracy"""
    if predictions.shape[1] < 4 or targets.shape[1] < 4:
        return 0.0
    p = predictions[:, 3].detach().cpu().numpy()
    t = targets[:, 3].detach().cpu().numpy()
    if len(p) > 1 and not np.isnan(p).any() and not np.isnan(t).any():
        corr = np.corrcoef(p, t)[0, 1]
        return corr if not np.isnan(corr) else 0.0
    return 0.0

def train_baseline(financial_train_path, financial_val_path=None, num_iters=1000, 
                  seq_length=10, batch_size=16, lr=1e-4, save_interval=100, 
                  gradient_clip=1.0):
    """
    Train the baseline LSTM model
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Initializing baseline training on {device}")
    
    # Initialize financial data loader - will determine the actual input/output dimensions
    train_loader = EnhancedFinancialDataLoader(
        data_path=financial_train_path,
        sequence_length=seq_length,
        batch_size=batch_size
    )
    
    # Get a batch to determine input dimension
    try:
        sample_batch = next(iter(train_loader))
        input_dim = sample_batch['sequence'].shape[2]
        output_dim = sample_batch['target'].shape[1]
        print(f"✅ Automatically determined input_dim={input_dim}, output_dim={output_dim} from data")
    except:
        # Fallback to default dimensions if we can't get a batch
        input_dim = 7  # Default based on typical features
        output_dim = 7  # Same as input_dim to match target dimensions
        print(f"⚠️ Using default input_dim={input_dim}, output_dim={output_dim}")
    
    # Initialize model with proper input dimensions
    model = FinancialLSTMBaseline(
        input_dim=input_dim, 
        hidden_dim=64,
        num_layers=2,
        output_dim=output_dim,
        dropout=0.2
    ).to(device)
    print(f"🔢 Loaded baseline LSTM model with input_dim={input_dim}, output_dim={output_dim}")
    
    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)  # Added L2 regularization
    
    # Create checkpoint directory if it doesn't exist
    checkpoint_dir = 'checkpoints/baseline'
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Initialize validation data loader if path provided
    val_loader = None
    if financial_val_path:
        val_loader = EnhancedFinancialDataLoader(
            data_path=financial_val_path,
            sequence_length=seq_length,
            batch_size=batch_size
        )
    
    print("--- Starting Baseline Training ---")
    print(f"✅ Loaded training data from {financial_train_path}")
    if val_loader:
        print(f"✅ Loaded validation data from {financial_val_path}")
    print(f"Starting training for {num_iters} iterations...")
    
    # Training loop
    model.train()
    iter_count = 0
    moving_avg_loss = None
    
    while iter_count < num_iters:
        try:
            # Get next batch
            batch = next(iter(train_loader))
            
            # Extract data from batch
            features = batch['features'].to(device)
            sequence = batch['sequence'].to(device)
            target = batch['target'].to(device)
            
            # Check for NaNs or infinites
            if torch.isnan(sequence).any() or torch.isinf(sequence).any() or \
               torch.isnan(target).any() or torch.isinf(target).any():
                print("⚠️ Warning: Skipping batch with NaN or Inf values")
                continue
            
            # Forward pass
            predictions = model(sequence)
            
            # Verify shapes match
            if predictions.shape != target.shape:
                print(f"⚠️ Shape mismatch: predictions {predictions.shape}, target {target.shape}")
                if predictions.shape[0] == target.shape[0]:  # Same batch size
                    min_features = min(predictions.shape[1], target.shape[1])
                    predictions = predictions[:, :min_features]
                    target = target[:, :min_features]
                    print(f"✂️ Trimmed to matching shapes: {predictions.shape}")
                else:
                    continue  # Skip this batch
            
            # Compute loss (MSE on the prediction vs target)
            loss = nn.MSELoss()(predictions, target)
            
            # Skip if loss is NaN or infinite
            if torch.isnan(loss) or torch.isinf(loss):
                print("⚠️ Warning: NaN or Inf loss detected, skipping batch")
                continue
                
            # Update moving average loss
            if moving_avg_loss is None:
                moving_avg_loss = loss.item()
            else:
                moving_avg_loss = 0.9 * moving_avg_loss + 0.1 * loss.item()
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            
            optimizer.step()
            
            # Print progress
            if iter_count % 10 == 0:
                print(f"🧮 Iter {iter_count}/{num_iters} | Loss: {loss.item():.4f} | Moving Avg: {moving_avg_loss:.4f}")
            
            # Save checkpoint
            if save_interval > 0 and iter_count % save_interval == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f"baseline_model_iter_{iter_count}.pth")
                torch.save(model.state_dict(), checkpoint_path)
                print(f"✅ Checkpoint saved to {checkpoint_path}")
            
            # Validation
            if val_loader and iter_count % 100 == 0:
                print(f"--- Running Validation at Iter {iter_count} ---")
                model.eval()
                
                val_losses = []
                all_preds = []
                all_targets = []
                
                with torch.no_grad():
                    for _ in range(10):  # Process 10 validation batches
                        try:
                            val_batch = next(iter(val_loader))
                            val_seq = val_batch['sequence'].to(device)
                            val_target = val_batch['target'].to(device)
                            
                            # Skip batches with NaN/Inf values
                            if torch.isnan(val_seq).any() or torch.isinf(val_seq).any() or \
                               torch.isnan(val_target).any() or torch.isinf(val_target).any():
                                continue
                            
                            val_pred = model(val_seq)
                            
                            # Verify shapes match
                            if val_pred.shape != val_target.shape:
                                if val_pred.shape[0] == val_target.shape[0]:
                                    min_features = min(val_pred.shape[1], val_target.shape[1])
                                    val_pred = val_pred[:, :min_features]
                                    val_target = val_target[:, :min_features]
                                else:
                                    continue
                            
                            val_loss = nn.MSELoss()(val_pred, val_target)
                            
                            # Skip if loss is NaN or infinite
                            if torch.isnan(val_loss) or torch.isinf(val_loss):
                                continue
                                
                            val_losses.append(val_loss.item())
                            
                            all_preds.append(val_pred)
                            all_targets.append(val_target)
                        except StopIteration:
                            break
                
                if all_preds and len(all_preds) > 0:
                    all_preds = torch.cat(all_preds, dim=0)
                    all_targets = torch.cat(all_targets, dim=0)
                    
                    # Calculate metrics using individual metric functions
                    price_acc = calculate_price_accuracy(all_preds, all_targets)
                    volume_corr = calculate_volume_correlation(all_preds, all_targets)
                    returns_stab = calculate_returns_stability(all_preds, all_targets)
                    vol_pred = calculate_volatility_prediction(all_preds, all_targets)
                    
                    print("\n=== Baseline Validation Metrics ===")
                    print(f"Avg Loss: {np.mean(val_losses):.4f}")
                    print(f"Price Accuracy: {price_acc:.4f}")
                    print(f"Volume Correlation: {volume_corr:.4f}")
                    print(f"Returns Stability: {returns_stab:.4f}")
                    print(f"Volatility Prediction: {vol_pred:.4f}")
                    print("===================================\n")
                else:
                    print("❌ No valid validation batches found")
                
                model.train()
            
            iter_count += 1
            
        except StopIteration:
            # Reset data loader if we run out of batches
            train_loader = EnhancedFinancialDataLoader(
                data_path=financial_train_path,
                sequence_length=seq_length,
                batch_size=batch_size
            )
    
    # Save final model
    final_model_path = os.path.join('models', f"baseline_lstm_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"✅ Final model saved to {final_model_path}")
    print("--- Training Complete ---")
    
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Baseline LSTM")
    parser.add_argument("--data_path", type=str, required=True, help="Path to training data")
    parser.add_argument("--val_path", type=str, default=None, help="Path to validation data")
    parser.add_argument("--output_dir", type=str, default="models/baseline", help="Directory to save model")
    parser.add_argument("--iters", type=int, default=1000, help="Training iterations")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--seq_length", type=int, default=20, help="Sequence length")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--save_interval", type=int, default=100, help="Checkpoint save interval")
    parser.add_argument("--gradient_clip", type=float, default=1.0, help="Gradient clipping value")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Train the model
    model = train_baseline(
        financial_train_path=args.data_path,
        financial_val_path=args.val_path,
        num_iters=args.iters,
        seq_length=args.seq_length,
        batch_size=args.batch,
        lr=args.lr,
        save_interval=args.save_interval,
        gradient_clip=args.gradient_clip
    )
    
    # Save final model to specified output directory
    final_model_path = os.path.join(args.output_dir, "baseline.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"✅ Final model saved to {final_model_path}")
