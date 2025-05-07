#!/usr/bin/env python
# train_progressive.py - Progressive difficulty training for cognitive architecture

import os
import sys
import argparse
import torch
import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
from tqdm import tqdm

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, root_dir)

from src.arch.cognitive import CognitiveArchitecture
from src.utils.regularization import RegularizationManager

class ProgressiveDataLoader:
    """Data loader with progressive difficulty levels"""
    
    def __init__(
        self,
        data_path: str,
        sequence_length: int = 20,
        batch_size: int = 32,
        difficulty_levels: int = 4,
        shuffle: bool = True,
        augmentation: bool = True
    ):
        """
        Initialize progressive data loader
        
        Args:
            data_path: Path to data file
            sequence_length: Sequence length
            batch_size: Batch size
            difficulty_levels: Number of difficulty levels
            shuffle: Whether to shuffle data
            augmentation: Whether to use data augmentation
        """
        self.data_path = data_path
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.difficulty_levels = difficulty_levels
        self.shuffle = shuffle
        self.augmentation = augmentation
        
        # Load data
        self.data = pd.read_csv(data_path)
        print(f"Loaded data with {len(self.data)} samples")
        
        # Extract features and target
        self.prepare_data()
        
        # Current difficulty level
        self.current_level = 0
        
        # Initialize difficulty settings
        self.setup_difficulty_levels()
    
    def prepare_data(self):
        """Prepare data for training"""
        # Identify target column (assume it's 'target' or last column)
        if 'target' in self.data.columns:
            target_column = 'target'
        else:
            print("Warning: 'target' column not found, using last column as target")
            target_column = self.data.columns[-1]
        
        # First, filter out common non-numeric columns (if present)
        non_numeric_columns = ['date', 'ticker', 'symbol']
        filtered_columns = [col for col in self.data.columns 
                             if col not in non_numeric_columns
                             and col != target_column]
        
        # Now identify only numeric feature columns
        self.feature_columns = []
        for col in filtered_columns:
            # Check if column is numeric
            try:
                # Try to convert the first few values to float
                self.data[col].iloc[:5].astype(float)
                self.feature_columns.append(col)
            except (ValueError, TypeError):
                print(f"Excluding non-numeric column: {col}")
        
        print(f"Using {len(self.feature_columns)} numeric features")
        
        # Extract features and target
        self.features = self.data[self.feature_columns].values.astype(float)
        self.targets = self.data[target_column].values.astype(float)
        
        # Normalize features
        self.normalize_features()
        
        # Create regime labels if available
        if 'market_regime' in self.data.columns:
            self.regimes = self.data['market_regime'].values.astype(float)
            self.has_regimes = True
        else:
            self.regimes = np.zeros(len(self.features))
            self.has_regimes = False
        
        # Create volatility metric if available
        if 'volatility_20d' in self.data.columns:
            self.volatility = self.data['volatility_20d'].values.astype(float)
            self.has_volatility = True
        else:
            self.volatility = np.zeros(len(self.features))
            self.has_volatility = False
    
    def normalize_features(self):
        """Normalize features"""
        # Calculate mean and std for each feature
        self.feature_means = np.nanmean(self.features, axis=0)
        self.feature_stds = np.nanstd(self.features, axis=0)
        
        # Replace zero std with 1 to avoid division by zero
        self.feature_stds[self.feature_stds == 0] = 1.0
        
        # Normalize features
        self.normalized_features = (self.features - self.feature_means) / self.feature_stds
        
        # Replace NaN values with 0
        self.normalized_features = np.nan_to_num(self.normalized_features)
    
    def setup_difficulty_levels(self):
        """Setup difficulty levels"""
        self.difficulty_settings = []
        
        # Level 0: Basic - simple, low volatility, single regime
        self.difficulty_settings.append({
            'feature_noise': 0.0,  # No added noise
            'feature_mask_prob': 0.0,  # No feature masking
            'sequence_mask_prob': 0.0,  # No sequence masking
            'max_volatility': 0.1,  # Only low volatility 
            'regimes': [0, 1] if self.has_regimes else None,  # Only simple regimes
            'augmentation_prob': 0.0  # No augmentation
        })
        
        # Level 1: Intermediate - some complexity
        self.difficulty_settings.append({
            'feature_noise': 0.05,  # Small noise
            'feature_mask_prob': 0.1,  # Some feature masking
            'sequence_mask_prob': 0.1,  # Some sequence masking
            'max_volatility': 0.2,  # Medium volatility
            'regimes': [0, 1, 2] if self.has_regimes else None,  # More regimes
            'augmentation_prob': 0.2  # Some augmentation
        })
        
        # Level 2: Advanced - increased complexity
        self.difficulty_settings.append({
            'feature_noise': 0.1,  # More noise
            'feature_mask_prob': 0.2,  # More feature masking
            'sequence_mask_prob': 0.2,  # More sequence masking
            'max_volatility': 0.3,  # Higher volatility
            'regimes': None,  # All regimes
            'augmentation_prob': 0.4  # More augmentation
        })
        
        # Level 3: Expert - full complexity
        self.difficulty_settings.append({
            'feature_noise': 0.15,  # Significant noise
            'feature_mask_prob': 0.3,  # Significant feature masking
            'sequence_mask_prob': 0.3,  # Significant sequence masking
            'max_volatility': None,  # All volatility levels
            'regimes': None,  # All regimes
            'augmentation_prob': 0.6  # Significant augmentation
        })
        
        print(f"Setup {len(self.difficulty_settings)} difficulty levels")
    
    def set_difficulty(self, level: int):
        """Set difficulty level"""
        if level < 0 or level >= self.difficulty_levels:
            raise ValueError(f"Invalid difficulty level: {level}, should be between 0 and {self.difficulty_levels-1}")
        
        self.current_level = level
        print(f"Set difficulty level to {level}: {self.difficulty_settings[level]}")
    
    def get_indices_for_current_level(self) -> np.ndarray:
        """Get indices for current difficulty level"""
        settings = self.difficulty_settings[self.current_level]
        
        # Start with all indices
        valid_indices = np.arange(self.sequence_length, len(self.features))
        
        # Filter by volatility if specified
        if settings['max_volatility'] is not None and self.has_volatility:
            volatility_mask = self.volatility[valid_indices] <= settings['max_volatility']
            valid_indices = valid_indices[volatility_mask]
        
        # Filter by regime if specified
        if settings['regimes'] is not None and self.has_regimes:
            regime_mask = np.isin(self.regimes[valid_indices], settings['regimes'])
            valid_indices = valid_indices[regime_mask]
        
        # Ensure we have enough samples
        if len(valid_indices) < self.batch_size:
            print(f"Warning: Only {len(valid_indices)} samples for level {self.current_level}, using all samples")
            valid_indices = np.arange(self.sequence_length, len(self.features))
        
        return valid_indices
    
    def __iter__(self):
        """Iterator for data loader"""
        # Get valid indices for current level
        indices = self.get_indices_for_current_level()
        
        # Shuffle if enabled
        if self.shuffle:
            np.random.shuffle(indices)
        
        # Calculate number of batches
        num_batches = len(indices) // self.batch_size
        
        # Generate batches
        for i in range(num_batches):
            # Get batch indices
            batch_indices = indices[i * self.batch_size:(i + 1) * self.batch_size]
            
            # Initialize containers for batch data
            batch_features = []
            batch_sequences = []
            batch_targets = []
            
            # Get data for each sample in batch
            for idx in batch_indices:
                # Get sequence indices
                seq_indices = np.arange(idx - self.sequence_length, idx)
                
                # Get features for current step
                features = self.normalized_features[idx].copy()
                
                # Get sequence for previous steps
                sequence = self.normalized_features[seq_indices].copy()
                
                # Get target
                target = self.targets[idx]
                
                # Apply difficulty-based transformations
                settings = self.difficulty_settings[self.current_level]
                
                # Add feature noise
                if settings['feature_noise'] > 0:
                    noise_scale = settings['feature_noise']
                    features += np.random.normal(0, noise_scale, features.shape)
                    sequence += np.random.normal(0, noise_scale, sequence.shape)
                
                # Mask features
                if settings['feature_mask_prob'] > 0:
                    mask_prob = settings['feature_mask_prob']
                    feature_mask = np.random.random(features.shape) >= mask_prob
                    features *= feature_mask
                    
                    # Apply different mask to each sequence step
                    for j in range(len(sequence)):
                        seq_mask = np.random.random(sequence[j].shape) >= mask_prob
                        sequence[j] *= seq_mask
                
                # Mask sequence steps
                if settings['sequence_mask_prob'] > 0:
                    mask_prob = settings['sequence_mask_prob']
                    seq_mask = np.random.random(len(sequence)) >= mask_prob
                    for j in range(len(sequence)):
                        if not seq_mask[j]:
                            sequence[j] = np.zeros_like(sequence[j])
                
                # Data augmentation
                if self.augmentation and np.random.random() < settings['augmentation_prob']:
                    # Simple augmentation: slight scaling
                    scale_factor = np.random.normal(1.0, 0.05)
                    features *= scale_factor
                    sequence *= scale_factor
                    
                    # Target won't be scaled since it's a future value
                
                # Add to batch
                batch_features.append(features)
                batch_sequences.append(sequence)
                batch_targets.append(target)
            
            # Convert to tensors
            batch_features = torch.tensor(batch_features, dtype=torch.float32)
            batch_sequences = torch.tensor(batch_sequences, dtype=torch.float32)
            batch_targets = torch.tensor(batch_targets, dtype=torch.float32).unsqueeze(1)
            
            yield batch_features, batch_sequences, batch_targets

def train_with_progressive_difficulty(
    model: CognitiveArchitecture,
    train_data_path: str,
    val_data_path: str,
    output_dir: str,
    sequence_length: int = 20,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    total_epochs: int = 200,
    epochs_per_level: Optional[List[int]] = None,
    start_level: int = 0,
    l2_lambda: float = 1e-5,
    gradient_clip: float = 1.0,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> Dict[str, Any]:
    """
    Train model with progressive difficulty
    
    Args:
        model: Cognitive architecture model
        train_data_path: Path to training data
        val_data_path: Path to validation data
        output_dir: Output directory
        sequence_length: Sequence length
        batch_size: Batch size
        learning_rate: Learning rate
        total_epochs: Total number of training epochs
        epochs_per_level: Number of epochs for each difficulty level
        start_level: Starting difficulty level
        l2_lambda: L2 regularization lambda
        gradient_clip: Gradient clipping threshold
        device: Device to train on
        
    Returns:
        Dictionary with training results
    """
    # Setup directories
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Move model to device
    model.to(device)
    
    # Create data loaders
    train_loader = ProgressiveDataLoader(
        data_path=train_data_path,
        sequence_length=sequence_length,
        batch_size=batch_size,
        difficulty_levels=4,
        shuffle=True,
        augmentation=True
    )
    
    val_loader = ProgressiveDataLoader(
        data_path=val_data_path,
        sequence_length=sequence_length,
        batch_size=batch_size,
        difficulty_levels=4,
        shuffle=False,
        augmentation=False
    )
    
    # Set initial difficulty
    train_loader.set_difficulty(start_level)
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_lambda)
    
    # Create regularization manager
    regularization = RegularizationManager(
        l2_lambda=l2_lambda,
        gradient_clip_threshold=gradient_clip
    )
    regularization.register_hooks(model)
    
    # Calculate epochs per level if not provided
    if epochs_per_level is None:
        # Calculate epochs per level based on difficulty (more epochs for harder levels)
        difficulty_weights = [1, 2, 3, 4]  # More epochs for higher difficulty
        total_weight = sum(difficulty_weights)
        
        epochs_per_level = [
            int(total_epochs * weight / total_weight)
            for weight in difficulty_weights
        ]
        
        # Ensure we use all epochs
        epochs_per_level[-1] += total_epochs - sum(epochs_per_level)
    
    # Validate epochs_per_level
    if len(epochs_per_level) != train_loader.difficulty_levels:
        raise ValueError(f"Expected {train_loader.difficulty_levels} values in epochs_per_level, got {len(epochs_per_level)}")
    
    # Calculate total epochs
    actual_total_epochs = sum(epochs_per_level)
    print(f"Training for {actual_total_epochs} epochs across {len(epochs_per_level)} difficulty levels")
    print(f"Epochs per level: {epochs_per_level}")
    
    # Training state
    best_val_loss = float('inf')
    best_model_state = None
    training_history = []
    
    # Function to evaluate model
    def evaluate_model(loader, difficulty_level):
        loader.set_difficulty(difficulty_level)
        model.eval()
        total_loss = 0.0
        prediction_loss = 0.0
        all_targets = []
        all_predictions = []
        num_batches = 0
        
        with torch.no_grad():
            for financial_data, financial_seq, targets in loader:
                # Move to device
                financial_data = financial_data.to(device)
                financial_seq = financial_seq.to(device)
                targets = targets.to(device)
                
                # Forward pass
                outputs = model(financial_data=financial_data, financial_seq=financial_seq)
                
                # Extract predictions
                if isinstance(outputs, dict) and 'market_state' in outputs:
                    predictions = outputs['market_state']
                else:
                    predictions = outputs
                
                # Fix shape mismatch in loss calculation
                # The model outputs market_state with shape [batch_size, seq_len, features]
                # But our targets are [batch_size, 1]
                if predictions.dim() > 2:
                    # Instead of just taking last time step and first feature,
                    # reshape predictions to match targets more intelligently
                    if targets.size(1) == 1:
                        # If target is just one value (e.g., next price),
                        # use the predicted value for the last time step
                        predictions = predictions[:, -1, 0].unsqueeze(1)
                    else:
                        # If target is a vector, match dimensions appropriately
                        predictions = predictions[:, -1, :targets.size(1)]
                elif predictions.dim() == 2 and predictions.size(1) != targets.size(1):
                    # Handle 2D predictions with wrong feature dimension
                    if predictions.size(1) > targets.size(1):
                        predictions = predictions[:, :targets.size(1)]
                    else:
                        # Pad predictions if needed
                        pad_size = targets.size(1) - predictions.size(1)
                        predictions = torch.cat([predictions, torch.zeros(predictions.size(0), pad_size, device=device)], dim=1)
                
                # Calculate loss
                loss = torch.nn.functional.mse_loss(predictions, targets)
                
                # Update metrics
                total_loss += loss.item()
                prediction_loss += loss.item()
                num_batches += 1
                
                # Store predictions and targets
                all_targets.append(targets.cpu().numpy())
                all_predictions.append(predictions.cpu().numpy())
        
        # Calculate average loss
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        avg_pred_loss = prediction_loss / num_batches if num_batches > 0 else float('inf')
        
        # Calculate directional accuracy if targets and predictions exist
        dir_accuracy = 0.0
        price_accuracy = 0.0
        
        if all_targets and all_predictions:
            all_targets = np.concatenate(all_targets)
            all_predictions = np.concatenate(all_predictions)
            
            # Calculate direction accuracy based on price changes between consecutive time steps
            if len(all_predictions) > 1:
                # Calculate directions based on consecutive price changes
                pred_dir = np.sign(all_predictions[1:, 0] - all_predictions[:-1, 0])
                target_dir = np.sign(all_targets[1:, 0] - all_targets[:-1, 0])
                
                # Handle zeros (flat prices)
                pred_dir = np.where(pred_dir == 0, 0.01, pred_dir)
                target_dir = np.where(target_dir == 0, 0.01, target_dir)
                
                # Calculate accuracy
                correct_direction = (pred_dir == target_dir)
                dir_accuracy = np.mean(correct_direction)
            else:
                # Not enough data points to calculate direction accuracy
                dir_accuracy = 0.0
            
            # Price accuracy (1 - normalized error)
            error = np.abs(all_predictions - all_targets)
            normalized_error = error / (np.abs(all_targets) + 1e-8)
            price_accuracy = 1.0 - np.mean(np.minimum(normalized_error, 1.0))
        
        return {
            'val_loss': avg_loss,
            'prediction_loss': avg_pred_loss,
            'direction_accuracy': dir_accuracy,
            'price_accuracy': price_accuracy
        }
    
    # Training loop
    current_level = start_level
    epoch = 0
    global_epoch = 0
    
    # Train through each difficulty level
    for level in range(start_level, train_loader.difficulty_levels):
        # Set difficulty level
        current_level = level
        train_loader.set_difficulty(level)
        val_loader.set_difficulty(level)
        
        print(f"\nTraining at difficulty level {level} for {epochs_per_level[level]} epochs")
        
        # Train for specified epochs at this level
        for level_epoch in range(epochs_per_level[level]):
            epoch = level_epoch
            global_epoch += 1
            
            # Training phase
            model.train()
            train_loss = 0.0
            train_batches = 0
            
            # Progress bar
            progress_bar = tqdm(train_loader, desc=f"Epoch {global_epoch}/{actual_total_epochs} (Level {level})")
            
            for financial_data, financial_seq, targets in progress_bar:
                # Move to device
                financial_data = financial_data.to(device)
                financial_seq = financial_seq.to(device)
                targets = targets.to(device)
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(financial_data=financial_data, financial_seq=financial_seq)
                
                # Extract predictions
                if isinstance(outputs, dict) and 'market_state' in outputs:
                    predictions = outputs['market_state']
                else:
                    predictions = outputs
                
                # Fix shape mismatch in loss calculation
                # The model outputs market_state with shape [batch_size, seq_len, features]
                # But our targets are [batch_size, 1]
                if predictions.dim() > 2:
                    # Instead of just taking last time step and first feature,
                    # reshape predictions to match targets more intelligently
                    if targets.size(1) == 1:
                        # If target is just one value (e.g., next price),
                        # use the predicted value for the last time step
                        predictions = predictions[:, -1, 0].unsqueeze(1)
                    else:
                        # If target is a vector, match dimensions appropriately
                        predictions = predictions[:, -1, :targets.size(1)]
                elif predictions.dim() == 2 and predictions.size(1) != targets.size(1):
                    # Handle 2D predictions with wrong feature dimension
                    if predictions.size(1) > targets.size(1):
                        predictions = predictions[:, :targets.size(1)]
                    else:
                        # Pad predictions if needed
                        pad_size = targets.size(1) - predictions.size(1)
                        predictions = torch.cat([predictions, torch.zeros(predictions.size(0), pad_size, device=device)], dim=1)
                
                # Calculate loss
                loss = torch.nn.functional.mse_loss(predictions, targets)
                
                # Backward pass
                loss.backward()
                
                # Apply gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                
                # Update parameters
                optimizer.step()
                
                # Update metrics
                train_loss += loss.item()
                train_batches += 1
                
                # Update progress bar
                progress_bar.set_postfix({"train_loss": f"{loss.item():.4f}"})
            
            # Calculate average training loss
            train_loss = train_loss / train_batches if train_batches > 0 else float('inf')
            
            # Evaluate on validation set
            val_metrics = evaluate_model(val_loader, level)
            
            # Print metrics
            print(f"Epoch {global_epoch}/{actual_total_epochs} (Level {level}), "
                  f"Train Loss: {train_loss:.4f}, "
                  f"Val Loss: {val_metrics['val_loss']:.4f}, "
                  f"Direction Acc: {val_metrics['direction_accuracy']:.4f}, "
                  f"Price Acc: {val_metrics['price_accuracy']:.4f}")
            
            # Save if best model
            if val_metrics['val_loss'] < best_val_loss:
                best_val_loss = val_metrics['val_loss']
                best_model_state = model.state_dict()
                
                # Save checkpoint
                torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best_model.pt"))
                
                print(f"Saved new best model with validation loss: {best_val_loss:.4f}")
            
            # Save checkpoint
            torch.save({
                'epoch': global_epoch,
                'level': level,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_metrics['val_loss'],
                'best_val_loss': best_val_loss
            }, os.path.join(checkpoint_dir, f"checkpoint_epoch_{global_epoch}.pt"))
            
            # Save metrics
            metrics = {
                'epoch': global_epoch,
                'level': level,
                'train_loss': float(train_loss),
                'val_loss': float(val_metrics['val_loss']),
                'direction_accuracy': float(val_metrics['direction_accuracy']),
                'price_accuracy': float(val_metrics['price_accuracy'])
            }
            
            training_history.append(metrics)
            
            with open(os.path.join(log_dir, f"metrics_epoch_{global_epoch}.json"), 'w') as f:
                json.dump(metrics, f, indent=2)
        
        # Save level-specific model
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"model_level_{level}.pt"))
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Final evaluation on all difficulty levels
    final_metrics = {}
    for level in range(train_loader.difficulty_levels):
        final_metrics[f"level_{level}"] = evaluate_model(val_loader, level)
    
    # Compute overall metrics (averaged across all levels)
    overall_metrics = {
        'val_loss': np.mean([final_metrics[f"level_{level}"]['val_loss'] for level in range(train_loader.difficulty_levels)]),
        'direction_accuracy': np.mean([final_metrics[f"level_{level}"]['direction_accuracy'] for level in range(train_loader.difficulty_levels)]),
        'price_accuracy': np.mean([final_metrics[f"level_{level}"]['price_accuracy'] for level in range(train_loader.difficulty_levels)])
    }
    
    # Convert metrics for JSON serialization
    def convert_for_json(obj):
        if isinstance(obj, (np.integer, np.floating, np.bool_)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(i) for i in obj]
        else:
            return obj
    
    # Save final metrics
    final_results = {
        'best_val_loss': float(best_val_loss),
        'final_metrics': convert_for_json(final_metrics),
        'overall_metrics': convert_for_json(overall_metrics),
        'training_history': convert_for_json(training_history)
    }
    
    with open(os.path.join(log_dir, "final_results.json"), 'w') as f:
        json.dump(final_results, f, indent=2)
    
    # Save best model again
    if best_model_state is not None:
        torch.save(best_model_state, os.path.join(output_dir, "best_model.pt"))
    
    # Create visualization of training progress
    import matplotlib.pyplot as plt
    
    # Extract metrics
    epochs = [m['epoch'] for m in training_history]
    levels = [m['level'] for m in training_history]
    train_losses = [m['train_loss'] for m in training_history]
    val_losses = [m['val_loss'] for m in training_history]
    dir_accuracies = [m['direction_accuracy'] for m in training_history]
    price_accuracies = [m['price_accuracy'] for m in training_history]
    
    # Create figure
    plt.figure(figsize=(15, 10))
    
    # Plot losses
    plt.subplot(2, 2, 1)
    for level in range(train_loader.difficulty_levels):
        level_mask = np.array(levels) == level
        if np.any(level_mask):
            plt.plot(np.array(epochs)[level_mask], np.array(train_losses)[level_mask], 
                    label=f"Train Loss (Level {level})")
    
    for level in range(train_loader.difficulty_levels):
        level_mask = np.array(levels) == level
        if np.any(level_mask):
            plt.plot(np.array(epochs)[level_mask], np.array(val_losses)[level_mask], 
                    '--', label=f"Val Loss (Level {level})")
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    # Plot direction accuracy
    plt.subplot(2, 2, 2)
    for level in range(train_loader.difficulty_levels):
        level_mask = np.array(levels) == level
        if np.any(level_mask):
            plt.plot(np.array(epochs)[level_mask], np.array(dir_accuracies)[level_mask], 
                    label=f"Level {level}")
    
    plt.xlabel('Epoch')
    plt.ylabel('Direction Accuracy')
    plt.legend()
    plt.title('Direction Accuracy')
    
    # Plot price accuracy
    plt.subplot(2, 2, 3)
    for level in range(train_loader.difficulty_levels):
        level_mask = np.array(levels) == level
        if np.any(level_mask):
            plt.plot(np.array(epochs)[level_mask], np.array(price_accuracies)[level_mask], 
                    label=f"Level {level}")
    
    plt.xlabel('Epoch')
    plt.ylabel('Price Accuracy')
    plt.legend()
    plt.title('Price Accuracy')
    
    # Plot final metrics
    plt.subplot(2, 2, 4)
    metrics_names = ['Val Loss', 'Direction Acc', 'Price Acc']
    metrics_values = []
    
    for level in range(train_loader.difficulty_levels):
        metrics_values.append([
            final_metrics[f"level_{level}"]['val_loss'],
            final_metrics[f"level_{level}"]['direction_accuracy'],
            final_metrics[f"level_{level}"]['price_accuracy']
        ])
    
    metrics_values = np.array(metrics_values)
    
    x = np.arange(len(metrics_names))
    width = 0.2
    
    for i in range(train_loader.difficulty_levels):
        plt.bar(x + i*width - (train_loader.difficulty_levels-1)*width/2, 
               metrics_values[i], width, label=f"Level {i}")
    
    plt.xlabel('Metric')
    plt.ylabel('Value')
    plt.xticks(x, metrics_names)
    plt.legend()
    plt.title('Final Metrics by Difficulty Level')
    
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, "training_progress.png"))
    
    print(f"Training completed. Final results:")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Overall metrics: {overall_metrics}")
    
    return final_results

def main():
    parser = argparse.ArgumentParser(description="Progressive difficulty training for cognitive architecture")
    
    # Data arguments
    parser.add_argument("--train_data", required=True, help="Path to training data")
    parser.add_argument("--val_data", required=True, help="Path to validation data")
    
    # Model arguments
    parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden dimension")
    parser.add_argument("--memory_size", type=int, default=50, help="Memory size")
    parser.add_argument("--pretrained_components", help="Directory with pretrained components")
    
    # Training arguments
    parser.add_argument("--sequence_length", type=int, default=20, help="Sequence length")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=200, help="Total number of epochs")
    parser.add_argument("--l2_lambda", type=float, default=1e-5, help="L2 regularization lambda")
    parser.add_argument("--gradient_clip", type=float, default=1.0, help="Gradient clipping threshold")
    
    # Progressive training arguments
    parser.add_argument("--start_easy", action="store_true", help="Start with easy difficulty")
    parser.add_argument("--increase_difficulty", action="store_true", help="Progressively increase difficulty")
    parser.add_argument("--epochs_per_level", help="Epochs per difficulty level (comma-separated)")
    
    # Other arguments
    parser.add_argument("--output_dir", default="models/cognitive_progressive", help="Output directory")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Computation device")
    
    args = parser.parse_args()
    
    # First, load data to determine feature dimension
    temp_loader = ProgressiveDataLoader(
        data_path=args.train_data,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size
    )
    
    # Get the feature dimension from the loaded data
    input_dim = len(temp_loader.feature_columns)
    print(f"Detected input dimension: {input_dim}")
    
    # Create model with the correct input dimension
    model = CognitiveArchitecture(
        input_dim=input_dim,  # Use detected input dimension
        hidden_dim=args.hidden_dim,
        memory_size=args.memory_size
    )
    
    # Load pretrained components if specified
    if args.pretrained_components:
        # Try to load memory component
        memory_path = os.path.join(args.pretrained_components, "memory", "checkpoints", "best_memory_component.pt")
        if os.path.exists(memory_path):
            print(f"Loading pretrained memory component from {memory_path}")
            
            # Load state dict
            state_dict = torch.load(memory_path, map_location=args.device)
            
            # Filter keys related to memory
            memory_state = {k: v for k, v in state_dict.items() if any(x in k for x in ['memory_bank', 'memory_buffer', 'memory_encoder'])}
            
            # Load partial state dict
            model.load_state_dict(memory_state, strict=False)
        
        # Try to load attention component
        attention_path = os.path.join(args.pretrained_components, "attention", "checkpoints", "best_attention_component.pt")
        if os.path.exists(attention_path):
            print(f"Loading pretrained attention component from {attention_path}")
            
            # Load state dict
            state_dict = torch.load(attention_path, map_location=args.device)
            
            # Filter keys related to attention
            attention_state = {k: v for k, v in state_dict.items() if any(x in k for x in ['attention', 'query', 'key', 'value'])}
            
            # Load partial state dict
            model.load_state_dict(attention_state, strict=False)
    
    # Determine start level
    start_level = 0 if args.start_easy else 3
    
    # Parse epochs per level
    epochs_per_level = None
    if args.epochs_per_level:
        epochs_per_level = [int(x) for x in args.epochs_per_level.split(',')]
    
    # Configure progressive difficulty
    if args.increase_difficulty:
        # Start easy and progressively increase difficulty
        if epochs_per_level is None:
            if args.start_easy:
                # Distribute epochs across all levels
                epochs_per_level = [50, 50, 50, 50]  # Equal distribution for 200 epochs
            else:
                # Focus on harder difficulties
                epochs_per_level = [0, 0, 0, 200]  # All epochs at max difficulty
    else:
        # Train at a single difficulty level
        if epochs_per_level is None:
            epochs_per_level = [0, 0, 0, 0]
            epochs_per_level[start_level] = args.epochs
    
    # Train with progressive difficulty
    results = train_with_progressive_difficulty(
        model=model,
        train_data_path=args.train_data,
        val_data_path=args.val_data,
        output_dir=args.output_dir,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        total_epochs=args.epochs,
        epochs_per_level=epochs_per_level,
        start_level=start_level,
        l2_lambda=args.l2_lambda,
        gradient_clip=args.gradient_clip,
        device=args.device
    )
    
    print("Training completed successfully!")

if __name__ == "__main__":
    main()
