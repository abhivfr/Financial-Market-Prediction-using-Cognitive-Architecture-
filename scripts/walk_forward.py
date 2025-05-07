# Save as scripts/walk_forward.py
import pandas as pd
import numpy as np
import torch
import os
import json
import argparse  # Re-enable argparse
from tqdm import tqdm
import sys # Import sys for error handling

# Ensure src is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import your model and data loading code
try:
    from src.arch.cognitive import CognitiveArchitecture
    from train import FinancialDataLoader
    # Import metric calculation functions - needed for calculating metrics per window
    from train import calculate_price_accuracy, calculate_volume_correlation
    from train import calculate_returns_stability, calculate_volatility_prediction
    print("Successfully imported CognitiveArchitecture, FinancialDataLoader, and metric functions.")
except ImportError as e:
    print(f"Error importing necessary modules: {e}")
    print("Please ensure src.arch.cognitive.CognitiveArchitecture, train.FinancialDataLoader, and the calculate_* functions are available and importable.")
    sys.exit(1)


# Use default batch_size of 1 for the function signature as in original code,
# but override when creating the actual loader in the main loop.
def walk_forward_validation(model, data_path, window_size=252, step=41, seq_length=20, batch_size=1): # Adjusted step size
    """
    Perform walk-forward validation (expanding window approach)
    """
    # Load full dataset (should be the PROCESSED data)
    try:
        df = pd.read_csv(data_path)
        print(f"Loaded full dataset from {data_path} with shape {df.shape}")
    except FileNotFoundError:
        print(f"Error: Full dataset not found at {data_path}")
        return []
    except Exception as e:
        print(f"Error loading full dataset: {e}")
        return []


    # Ensure we have enough data for at least one train/test split
    if len(df) < window_size + step:
        print(f"Error: Dataset too small ({len(df)} samples) for walk-forward validation with initial window_size={window_size} and step={step}. Need at least {window_size + step} samples.")
        return []

    results = []
    # Use CPU for consistency with previous debugged steps
    device = torch.device("cpu") # Forced to CPU
    model.to(device)
    model.eval()

    # Determine validation windows
    # Start with initial window
    start_idx = 0
    end_train_idx = window_size

    print(f"\n--- Starting Walk-Forward Validation ---")

    window_count = 0
    # Loop until the test window extends beyond the end of the dataset
    # Need enough data for at least seq_length + 1 samples to form the first sequence/target
    # and enough total samples after DataLoader processing to form at least one batch.
    # The while condition end_train_idx + step <= len(df) ensures the test slice has 'step' rows.
    while end_train_idx + step <= len(df):
        window_count += 1
        test_start_idx = end_train_idx
        test_end_idx = end_train_idx + step # Test slice from index test_start_idx up to (but not including) test_end_idx


        print(f"\nProcessing Window {window_count}:")
        print(f"  Train Data Indices: {start_idx} to {end_train_idx-1}")
        print(f"  Test Data Indices:  {test_start_idx} to {test_end_idx-1} ({step} samples)")


        # Extract training and test slices
        # train_df is not actually used for evaluation in this script, but kept for clarity
        train_df = df.iloc[start_idx:end_train_idx].copy()
        test_df = df.iloc[test_start_idx:test_end_idx].copy()

        # Save to temporary CSV files
        temp_train_path = "temp_train_wf.csv" # Temp file for train data (not used by test_loader)
        temp_test_path = "temp_test_wf.csv"   # Temp file for test data slice
        try:
             train_df.to_csv(temp_train_path, index=False) # Save train, though it's not loaded for test evaluation
             test_df.to_csv(temp_test_path, index=False) # Save the test slice
        except Exception as e:
             print(f"Error saving temporary data files for window {window_count}: {e}")
             # Clean up potential partial files and continue to next window
             if os.path.exists(temp_train_path): os.remove(temp_train_path)
             if os.path.exists(temp_test_path): os.remove(temp_test_path)
             end_train_idx += step # Move to next window
             continue # Skip evaluation for this window


        # Create data loader for the current test slice
        try:
            # IMPORTANT: Use batch_size > 1 (like 16) for metrics to work correctly
            # We determined batch_size=16 needs step >= 41
            test_loader = FinancialDataLoader(temp_test_path, seq_length=seq_length, batch_size=16) # Changed batch_size to 16
            print(f"  Test DataLoader created with {len(test_loader)} batches.")

            # Check if the DataLoader will yield any batches
            if len(test_loader) == 0:
                 print(f"  Warning: Test DataLoader for window {window_count} yields 0 batches. Skipping evaluation.")
                 # Append zero metrics or NaNs for this window
                 results.append({
                     'window_number': window_count,
                     'train_indices': f"{start_idx}-{end_train_idx-1}",
                     'test_indices': f"{test_start_idx}-{test_end_idx-1}",
                     'metrics': {
                         'price_accuracy': np.nan, 'volume_correlation': np.nan,
                         'returns_stability': np.nan, 'volatility_prediction': np.nan # Report NaN if no batches
                     },
                     'error': "DataLoader yielded 0 batches"
                 })
                 # Clean up temp files and continue
                 if os.path.exists(temp_train_path): os.remove(temp_train_path)
                 if os.path.exists(temp_test_path): os.remove(temp_test_path)
                 end_train_idx += step # Move to next window
                 continue # Skip evaluation for this window


        except (FileNotFoundError, ValueError) as e: # Catch DataLoader specific errors
             print(f"Error creating test data loader for window {window_count}: {e}")
             # Clean up temp files and continue
             if os.path.exists(temp_train_path): os.remove(temp_train_path)
             if os.path.exists(temp_test_path): os.remove(temp_test_path)
             end_train_idx += step # Move to next window
             continue
        except Exception as e: # Catch any other exceptions
             print(f"Unexpected error creating test data loader for window {window_count}: {e}")
             if os.path.exists(temp_train_path): os.remove(temp_train_path)
             if os.path.exists(temp_test_path): os.remove(temp_test_path)
             end_train_idx += step # Move to next window
             continue


        # Make predictions on the current test set slice
        predictions = []
        ground_truth = []

        # Use tqdm for progress bar over batches
        print(f"  Evaluating on test slice ({len(test_df)} samples)...")
        try:
            with torch.no_grad():
                 # Iterate using the DataLoader
                 for batch in tqdm(test_loader, desc=f"  Window {window_count} Evaluation"):
                    features = batch['features'].to(device)
                    sequence = batch['sequence'].to(device)
                    target = batch['target'].to(device)

                    # Assuming your CognitiveArchitecture forward needs these inputs
                    # Add volume extraction as seen in your previous code
                    # Ensure inputs match what the model expects (e.g., volume argument)
                    try:
                        outputs = model(
                            financial_data=features,  # Shape (batch_size, input_dim)
                            financial_seq=sequence,   # Shape (batch_size, seq_length, input_dim)
                            volume=sequence[:, :, 1:2] # Extract volume, Shape (batch_size, seq_length, 1)
                        )
                    except Exception as model_e:
                         print(f"  Error during model forward pass for window {window_count}: {model_e}")
                         # If model forward fails, skip this batch/window
                         continue # Skip to next batch or exit loop if no batches processed


                    # Assuming 'market_state' contains the predictions for metrics
                    # market_state shape: (batch_size, seq_len_out, input_dim) or (batch_size, input_dim)
                    # Metrics compare market_state[:, -1] to target (batch_size, input_dim)
                    # Extract the prediction for the next step (last item of market_state)
                    # Ensure market_state has correct dimensions before slicing
                    if outputs['market_state'].ndim == 3:
                        predicted_output = outputs['market_state'][:, -1] # Shape (batch_size, input_dim)
                    elif outputs['market_state'].ndim == 2:
                        predicted_output = outputs['market_state'] # Shape (batch_size, input_dim) - Assuming this is the prediction for the next step
                    else:
                         print(f"  Warning: Unexpected market_state dimensions for window {window_count}: {outputs['market_state'].shape}. Skipping batch.")
                         continue # Skip to next batch


                    predictions.append(predicted_output.cpu().numpy())
                    ground_truth.append(target.cpu().numpy()) # Assuming target is the ground truth for metrics


            # --- Calculate metrics for this window ---
            # Combine batch predictions and ground truth arrays
            if predictions and ground_truth:
                predictions_np = np.vstack(predictions)
                ground_truth_np = np.vstack(ground_truth)

                # Convert back to torch tensors for metric calculation functions
                # Ensure tensors are on CPU for numpy conversion in metric functions
                predictions_tensor = torch.tensor(predictions_np).cpu()
                ground_truth_tensor = torch.tensor(ground_truth_np).cpu()

                # Call the implemented metric calculation functions
                # These functions expect tensors of shape (Total_Samples_in_Window, input_dim)
                # Pass the combined numpy arrays converted back to tensors
                window_metrics = {
                    'price_accuracy': calculate_price_accuracy(predictions_tensor, ground_truth_tensor).item(),
                    'volume_correlation': calculate_volume_correlation(predictions_tensor, ground_truth_tensor).item(),
                    'returns_stability': calculate_returns_stability(predictions_tensor, ground_truth_tensor).item(),
                    'volatility_prediction': calculate_volatility_prediction(predictions_tensor, ground_truth_tensor).item()
                }
                print(f"  Window {window_count} Metrics: {window_metrics}")

                window_results = {
                    'window_number': window_count,
                    'train_indices': f"{start_idx}-{end_train_idx-1}",
                    'test_indices': f"{test_start_idx}-{test_end_idx-1}",
                    'metrics': window_metrics
                }
                results.append(window_results)
            else:
                # Handle case where evaluation loop didn't yield data
                print(f"  No data batches processed or error occurred during batch processing for window {window_count}.")
                # Append NaN metrics for this window if no valid batches processed
                results.append({
                    'window_number': window_count,
                    'train_indices': f"{start_idx}-{end_train_idx-1}",
                    'test_indices': f"{test_start_idx}-{test_end_idx-1}",
                    'metrics': { # Report NaN metrics if no batches processed successfully
                        'price_accuracy': np.nan, 'volume_correlation': np.nan,
                        'returns_stability': np.nan, 'volatility_prediction': np.nan
                    },
                    'error': "No valid batches processed or error during batch processing"
                })


        except Exception as e:
             print(f"Error during evaluation for window {window_count}: {e}")
             # Append NaN metrics for the failed window
             results.append({
                'window_number': window_count,
                'train_indices': f"{start_idx}-{end_train_idx-1}",
                'test_indices': f"{test_start_idx}-{test_end_idx-1}",
                'metrics': {
                    'price_accuracy': np.nan, 'volume_correlation': np.nan,
                    'returns_stability': np.nan, 'volatility_prediction': np.nan
                },
                'error': str(e)
             })


        finally:
             # Clean up temporary files
             if os.path.exists(temp_train_path):
                 os.remove(temp_train_path)
             if os.path.exists(temp_test_path):
                 os.remove(temp_test_path)


        # Move window forward (expanding window approach)
        # For sliding window: start_idx += step
        end_train_idx += step # Expand training window by 'step'


    print("\n--- Walk-Forward Validation Complete ---")
    print(f"Processed {window_count} windows.")

    # Calculate and print overall metrics from collected results
    # Filter out windows where metrics were reported as NaN due to errors/no batches
    valid_results = [r for r in results if 'metrics' in r and not any(np.isnan(list(r['metrics'].values())))]

    if valid_results: # Check if there are any valid results with non-NaN metrics
        overall_metrics = {
           'price_accuracy': np.mean([r['metrics']['price_accuracy'] for r in valid_results]),
           'volume_correlation': np.mean([r['metrics']['volume_correlation'] for r in valid_results]),
           'returns_stability': np.mean([r['metrics']['returns_stability'] for r in valid_results]),
           'volatility_prediction': np.mean([r['metrics']['volatility_prediction'] for r in valid_results])
        }

        print("\n--- Overall Walk-Forward Validation Results (Average Across Valid Windows) ---")
        for metric, value in overall_metrics.items():
           print(f"{metric.replace('_', ' ').title()}: {value:.4f}")
        print("--------------------------------------------------------------------------------")
    else:
         print("\nNo valid window results with non-NaN metrics collected to calculate overall metrics.")
         overall_metrics = {} # Empty metrics if no valid results


    return results # Return the list of window results (including those with errors/NaNs)


# --- Main Execution Block with Hardcoded Values ---
if __name__ == "__main__":
    # Add command-line argument support
    parser = argparse.ArgumentParser(description="Run walk-forward validation on trained model")
    parser.add_argument("--model", type=str, required=True, 
                      help="Path to the trained model checkpoint")
    parser.add_argument("--data", type=str, required=True, 
                      help="Path to the full dataset CSV")
    parser.add_argument("--window", type=int, default=252,
                      help="Initial training window size (days)")
    parser.add_argument("--step", type=int, default=41, 
                      help="Step size for walk-forward windows (days)")
    parser.add_argument("--seq_length", type=int, default=20,
                      help="Sequence length for model")
    parser.add_argument("--batch", type=int, default=16,
                      help="Batch size for evaluation")
    parser.add_argument("--output", type=str, default="validation/walk_forward_results.json",
                      help="Output JSON file path")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Running walk-forward validation with the following parameters:")
    print(f"Model: {args.model}")
    print(f"Full Data: {args.data}")
    print(f"Initial Window: {args.window} days, Step: {args.step} days")
    print(f"Sequence Length: {args.seq_length}, Batch Size: {args.batch}")
    print(f"Device: cpu (Forced)")
    
    # Load model
    loaded_model = None
    try:
        loaded_model = CognitiveArchitecture(input_dim=4, hidden_dim=256, memory_size=1000)
        loaded_model.load_state_dict(torch.load(args.model, map_location='cpu'), strict=False)
        print(f"Successfully loaded model state dict from {args.model}")
    except FileNotFoundError:
        print(f"Error: Model checkpoint not found at {args.model}. Cannot run walk-forward validation.")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading model state dict from {args.model}: {e}. Cannot run walk-forward validation.")
        sys.exit(1)
    
    # Run walk-forward validation
    results = walk_forward_validation(
        loaded_model,
        args.data,
        window_size=args.window,
        step=args.step,
        seq_length=args.seq_length,
        batch_size=args.batch
    )
    
    # Save results
    try:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nWalk-forward validation results saved to {args.output}")
    except Exception as e:
        print(f"Error saving walk-forward results to JSON: {e}")

    # The overall metrics calculation is now done inside walk_forward_validation before returning results
