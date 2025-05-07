#!/usr/bin/env python
# online_learning.py - Framework for online learning and model adaptation

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import pickle
import json
from typing import Dict, List, Tuple, Any, Optional, Union
from collections import deque
import pandas as pd
from datetime import datetime

class OnlineLearningBuffer:
    """Buffer for storing recent observations for online learning"""
    
    def __init__(self, max_size: int = 1000, min_samples_for_update: int = 50):
        """
        Initialize online learning buffer
        
        Args:
            max_size: Maximum buffer size
            min_samples_for_update: Minimum samples needed before model update
        """
        self.buffer = deque(maxlen=max_size)
        self.min_samples_for_update = min_samples_for_update
    
    def add(self, sample: Dict[str, torch.Tensor]) -> None:
        """
        Add a sample to the buffer
        
        Args:
            sample: Dictionary with keys 'features', 'sequence', 'target'
        """
        # Convert tensors to numpy for storage
        sample_np = {
            k: v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v
            for k, v in sample.items()
        }
        
        # Add timestamp
        sample_np['timestamp'] = datetime.now().timestamp()
        
        # Add to buffer
        self.buffer.append(sample_np)
    
    def is_ready_for_update(self) -> bool:
        """Check if buffer has enough samples for update"""
        return len(self.buffer) >= self.min_samples_for_update
    
    def get_batch(self, batch_size: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        Get a batch of samples from the buffer
        
        Args:
            batch_size: Batch size (None = all samples)
            
        Returns:
            Dictionary of batched tensors
        """
        if not self.buffer:
            return {}
        
        # Use all samples if batch_size is None
        if batch_size is None or batch_size >= len(self.buffer):
            samples = list(self.buffer)
        else:
            # Sample randomly
            indices = np.random.choice(len(self.buffer), batch_size, replace=False)
            samples = [self.buffer[i] for i in indices]
        
        # Convert to batch
        batch = {}
        for key in samples[0].keys():
            if key != 'timestamp':
                # Stack numpy arrays
                batch[key] = np.stack([s[key] for s in samples])
                # Convert to torch tensors
                batch[key] = torch.from_numpy(batch[key]).float()
        
        return batch
    
    def clear(self) -> None:
        """Clear the buffer"""
        self.buffer.clear()
    
    def save(self, path: str) -> None:
        """
        Save buffer to file
        
        Args:
            path: File path
        """
        with open(path, 'wb') as f:
            pickle.dump(list(self.buffer), f)
    
    def load(self, path: str) -> None:
        """
        Load buffer from file
        
        Args:
            path: File path
        """
        with open(path, 'rb') as f:
            buffer_data = pickle.load(f)
            self.buffer = deque(buffer_data, maxlen=self.buffer.maxlen)

class ConceptDriftDetector:
    """Detector for concept drift in data distributions"""
    
    def __init__(
        self, 
        window_size: int = 100, 
        drift_threshold: float = 0.05,
        features_to_monitor: Optional[List[str]] = None
    ):
        """
        Initialize concept drift detector
        
        Args:
            window_size: Size of reference window
            drift_threshold: Threshold for drift detection
            features_to_monitor: List of feature names to monitor (None = all)
        """
        self.window_size = window_size
        self.drift_threshold = drift_threshold
        self.features_to_monitor = features_to_monitor
        
        # Reference statistics
        self.reference_mean = None
        self.reference_std = None
        self.reference_window = deque(maxlen=window_size)
        
        # Drift statistics
        self.drift_detected = False
        self.drift_score = 0.0
        self.drifting_features = []
    
    def update_reference(self, features: np.ndarray) -> None:
        """
        Update reference statistics
        
        Args:
            features: Feature array [batch_size, n_features]
        """
        # Add to reference window
        for sample in features:
            self.reference_window.append(sample)
        
        # Calculate statistics
        if len(self.reference_window) == self.window_size:
            reference_data = np.stack(list(self.reference_window))
            self.reference_mean = np.mean(reference_data, axis=0)
            self.reference_std = np.std(reference_data, axis=0) + 1e-6  # Avoid division by zero
    
    def detect_drift(self, features: np.ndarray) -> bool:
        """
        Detect concept drift
        
        Args:
            features: Feature array [batch_size, n_features]
            
        Returns:
            True if drift detected
        """
        # Check if reference is initialized
        if self.reference_mean is None or self.reference_std is None:
            self.update_reference(features)
            return False
        
        # Calculate statistics of current batch
        current_mean = np.mean(features, axis=0)
        
        # Calculate normalized absolute difference
        diff = np.abs(current_mean - self.reference_mean) / self.reference_std
        
        # Calculate drift score (mean difference across features)
        self.drift_score = np.mean(diff)
        
        # Find drifting features
        self.drifting_features = [
            i for i, d in enumerate(diff) if d > self.drift_threshold
        ]
        
        # Detect drift if score exceeds threshold
        self.drift_detected = self.drift_score > self.drift_threshold
        
        return self.drift_detected
    
    def get_drift_info(self, feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Get drift information
        
        Args:
            feature_names: Optional list of feature names
            
        Returns:
            Dictionary with drift information
        """
        # Format drifting features with names if provided
        if feature_names and len(feature_names) >= len(self.reference_mean):
            drifting_features = [
                (i, feature_names[i]) for i in self.drifting_features
                if i < len(feature_names)
            ]
        else:
            drifting_features = [(i, f"Feature_{i}") for i in self.drifting_features]
        
        return {
            'drift_detected': self.drift_detected,
            'drift_score': float(self.drift_score),
            'drift_threshold': self.drift_threshold,
            'drifting_features': drifting_features,
            'n_drifting_features': len(self.drifting_features)
        }
    
    def reset(self) -> None:
        """Reset detector"""
        self.reference_mean = None
        self.reference_std = None
        self.reference_window.clear()
        self.drift_detected = False
        self.drift_score = 0.0
        self.drifting_features = []

class OnlineLearner:
    """Framework for online learning and model adaptation"""
    
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-4,
        buffer_size: int = 1000,
        min_samples_for_update: int = 50,
        update_frequency: int = 10,
        weight_decay: float = 1e-5,
        drift_threshold: float = 0.05,
        device: str = "cpu"
    ):
        """
        Initialize online learner
        
        Args:
            model: PyTorch model
            learning_rate: Learning rate for online updates
            buffer_size: Maximum buffer size
            min_samples_for_update: Minimum samples needed before model update
            update_frequency: Update model every N samples
            weight_decay: Weight decay for regularization
            drift_threshold: Threshold for concept drift detection
            device: Computation device
        """
        self.model = model
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.update_frequency = update_frequency
        
        # Move model to device
        self.model.to(self.device)
        
        # Create data buffer
        self.buffer = OnlineLearningBuffer(
            max_size=buffer_size,
            min_samples_for_update=min_samples_for_update
        )
        
        # Create drift detector
        self.drift_detector = ConceptDriftDetector(
            window_size=buffer_size // 2,
            drift_threshold=drift_threshold
        )
        
        # Create optimizer for online updates
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning statistics
        self.samples_seen = 0
        self.updates_performed = 0
        self.drift_detected_count = 0
        self.last_update_time = None
        self.update_history = []
    
    def process_sample(
        self, 
        features: torch.Tensor, 
        sequence: torch.Tensor, 
        target: Optional[torch.Tensor] = None,
        update_model: bool = True
    ) -> Dict[str, Any]:
        """
        Process a new sample
        
        Args:
            features: Feature tensor [batch_size, n_features]
            sequence: Sequence tensor [batch_size, seq_len, n_features]
            target: Target tensor [batch_size, n_targets]
            update_model: Whether to update the model
            
        Returns:
            Dictionary with processing results
        """
        # Move tensors to device
        features = features.to(self.device)
        sequence = sequence.to(self.device)
        if target is not None:
            target = target.to(self.device)
        
        # Get model prediction
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(financial_data=features, financial_seq=sequence)
        
        # Extract prediction
        if isinstance(outputs, dict) and 'market_state' in outputs:
            prediction = outputs['market_state']
            uncertainty = outputs.get('uncertainty', None)
        else:
            prediction = outputs
            uncertainty = None
        
        # Check for concept drift
        features_np = features.cpu().numpy()
        drift_detected = self.drift_detector.detect_drift(features_np)
        
        # If drift detected, recommend model update
        should_update = drift_detected
        
        # Add sample to buffer if target is available
        if target is not None:
            sample = {
                'features': features,
                'sequence': sequence,
                'target': target
            }
            self.buffer.add(sample)
            
            # Update counter
            self.samples_seen += 1
            
            # Check if model should be updated based on frequency
            if update_model and self.samples_seen % self.update_frequency == 0:
                should_update = True
        
        # Update model if needed and possible
        update_performed = False
        update_metrics = {}
        
        if should_update and update_model and self.buffer.is_ready_for_update():
            update_metrics = self.update_model()
            update_performed = True
            
            # Record update
            self.updates_performed += 1
            self.last_update_time = datetime.now()
            
            # If drift was detected, record it
            if drift_detected:
                self.drift_detected_count += 1
        
        # Prepare result
        result = {
            'prediction': prediction.cpu().detach(),
            'drift_detected': drift_detected,
            'drift_info': self.drift_detector.get_drift_info(),
            'update_performed': update_performed,
            'buffer_size': len(self.buffer.buffer),
            'samples_seen': self.samples_seen,
            'updates_performed': self.updates_performed
        }
        
        # Add uncertainty if available
        if uncertainty is not None:
            result['uncertainty'] = uncertainty.cpu().detach()
        
        # Add update metrics if available
        if update_metrics:
            result['update_metrics'] = update_metrics
            
            # Add to history
            self.update_history.append({
                'timestamp': datetime.now().timestamp(),
                'samples_seen': self.samples_seen,
                'drift_detected': drift_detected,
                'metrics': update_metrics
            })
        
        return result
    
    def update_model(self, batch_size: Optional[int] = None, epochs: int = 1) -> Dict[str, float]:
        """
        Update model with data from buffer
        
        Args:
            batch_size: Batch size (None = all samples)
            epochs: Number of training epochs
            
        Returns:
            Dictionary with update metrics
        """
        # Ensure buffer has enough samples
        if not self.buffer.is_ready_for_update():
            return {}
        
        # Get batch from buffer
        batch = self.buffer.get_batch(batch_size)
        if not batch:
            return {}
        
        # Move batch to device
        for k, v in batch.items():
            batch[k] = v.to(self.device)
        
        # Set model to training mode
        self.model.train()
        
        # Training metrics
        metrics = {'loss': 0.0}
        
        # Train for specified number of epochs
        for epoch in range(epochs):
            # Reset gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(financial_data=batch['features'], financial_seq=batch['sequence'])
            
            # Extract prediction
            if isinstance(outputs, dict) and 'market_state' in outputs:
                prediction = outputs['market_state']
            else:
                prediction = outputs
            
            # Calculate loss
            if hasattr(self.model, 'calculate_loss'):
                # Use model's loss function if available
                loss = self.model.calculate_loss(outputs, batch['target'])
            else:
                # Default to MSE loss
                loss_fn = nn.MSELoss()
                loss = loss_fn(prediction, batch['target'])
            
            # Backward pass
            loss.backward()
            
            # Update parameters
            self.optimizer.step()
            
            # Update metrics
            metrics['loss'] += loss.item()
        
        # Calculate average over epochs
        for k in metrics:
            metrics[k] /= epochs
        
        return metrics
    
    def reset_buffer(self) -> None:
        """Reset buffer"""
        self.buffer.clear()
    
    def save(self, path: str) -> None:
        """
        Save online learner state
        
        Args:
            path: Directory path
        """
        os.makedirs(path, exist_ok=True)
        
        # Save model
        torch.save(self.model.state_dict(), os.path.join(path, 'model.pt'))
        
        # Save optimizer
        torch.save(self.optimizer.state_dict(), os.path.join(path, 'optimizer.pt'))
        
        # Save buffer
        self.buffer.save(os.path.join(path, 'buffer.pkl'))
        
        # Save metadata
        metadata = {
            'samples_seen': self.samples_seen,
            'updates_performed': self.updates_performed,
            'drift_detected_count': self.drift_detected_count,
            'last_update_time': self.last_update_time.isoformat() if self.last_update_time else None,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'update_frequency': self.update_frequency,
            'update_history': self.update_history
        }
        
        with open(os.path.join(path, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def load(self, path: str) -> None:
        """
        Load online learner state
        
        Args:
            path: Directory path
        """
        # Load model
        self.model.load_state_dict(torch.load(os.path.join(path, 'model.pt'), map_location=self.device))
        
        # Load optimizer
        self.optimizer.load_state_dict(torch.load(os.path.join(path, 'optimizer.pt'), map_location=self.device))
        
        # Load buffer
        self.buffer.load(os.path.join(path, 'buffer.pkl'))
        
        # Load metadata
        with open(os.path.join(path, 'metadata.json'), 'r') as f:
            metadata = json.load(f)
            
            self.samples_seen = metadata['samples_seen']
            self.updates_performed = metadata['updates_performed']
            self.drift_detected_count = metadata['drift_detected_count']
            
            if metadata['last_update_time']:
                self.last_update_time = datetime.fromisoformat(metadata['last_update_time'])
            else:
                self.last_update_time = None
                
            self.learning_rate = metadata['learning_rate']
            self.weight_decay = metadata['weight_decay']
            self.update_frequency = metadata['update_frequency']
            self.update_history = metadata['update_history']

class StreamProcessor:
    """Process a stream of financial data for online learning"""
    
    def __init__(
        self,
        model: nn.Module,
        feature_names: List[str],
        window_size: int = 10,
        update_frequency: int = 5,
        save_dir: str = "online_learning_results",
        device: str = "cpu"
    ):
        """
        Initialize stream processor
        
        Args:
            model: PyTorch model
            feature_names: List of feature names
            window_size: Window size for sequence data
            update_frequency: Update frequency
            save_dir: Directory to save results
            device: Computation device
        """
        self.model = model
        self.feature_names = feature_names
        self.window_size = window_size
        self.save_dir = save_dir
        self.device = device
        
        # Create online learner
        self.learner = OnlineLearner(
            model=model,
            update_frequency=update_frequency,
            device=device
        )
        
        # Create window buffer for sequence data
        self.window_buffer = deque(maxlen=window_size)
        
        # Create results storage
        self.predictions = []
        self.targets = []
        self.timestamps = []
        self.drift_events = []
        
        # Create output directory
        os.makedirs(save_dir, exist_ok=True)
    
    def process_dataframe(
        self, 
        df: pd.DataFrame,
        target_col: str = 'return',
        timestamp_col: Optional[str] = None,
        update_model: bool = True
    ) -> pd.DataFrame:
        """
        Process a DataFrame of financial data
        
        Args:
            df: DataFrame with financial data
            target_col: Target column name
            timestamp_col: Timestamp column name (optional)
            update_model: Whether to update the model
            
        Returns:
            DataFrame with original data and predictions
        """
        # Ensure all feature columns are present
        for feature in self.feature_names:
            if feature not in df.columns:
                raise ValueError(f"Feature column '{feature}' not found in DataFrame")
        
        # Copy DataFrame to avoid modifying original
        result_df = df.copy()
        
        # Initialize prediction columns
        result_df['prediction'] = np.nan
        result_df['drift_detected'] = False
        result_df['update_performed'] = False
        
        # Extract features and target
        features = df[self.feature_names].values
        targets = df[target_col].values if target_col in df.columns else None
        
        # Process each row
        for i in range(len(df)):
            # Get current features and target
            current_features = features[i:i+1]
            current_target = targets[i:i+1] if targets is not None else None
            
            # Convert to tensors
            features_tensor = torch.FloatTensor(current_features)
            target_tensor = torch.FloatTensor(current_target) if current_target is not None else None
            
            # Update window buffer
            self.window_buffer.append(current_features[0])
            
            # If window buffer is full, create sequence tensor
            if len(self.window_buffer) == self.window_size:
                sequence = np.array(list(self.window_buffer))
                sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0)  # [1, window_size, n_features]
                
                # Process sample
                result = self.learner.process_sample(
                    features=features_tensor,
                    sequence=sequence_tensor,
                    target=target_tensor,
                    update_model=update_model
                )
                
                # Store prediction
                result_df.loc[i, 'prediction'] = result['prediction'].numpy()[0]
                result_df.loc[i, 'drift_detected'] = result['drift_detected']
                result_df.loc[i, 'update_performed'] = result['update_performed']
                
                # Store in internal arrays
                self.predictions.append(result['prediction'].numpy()[0])
                if current_target is not None:
                    self.targets.append(current_target[0])
                
                # Add timestamp if available
                if timestamp_col and timestamp_col in df.columns:
                    self.timestamps.append(df[timestamp_col].iloc[i])
                else:
                    self.timestamps.append(i)
                
                # Track drift events
                if result['drift_detected']:
                    self.drift_events.append({
                        'timestamp': self.timestamps[-1],
                        'drift_score': result['drift_info']['drift_score'],
                        'drifting_features': result['drift_info']['drifting_features']
                    })
        
        return result_df
    
    def save_results(self) -> None:
        """Save processing results"""
        # Create plots directory
        plots_dir = os.path.join(self.save_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # Create results DataFrame
        results = {
            'timestamp': self.timestamps,
            'prediction': self.predictions
        }
        
        if self.targets:
            results['target'] = self.targets
        
        results_df = pd.DataFrame(results)
        
        # Save results
        results_df.to_csv(os.path.join(self.save_dir, 'predictions.csv'), index=False)
        
        # Save drift events
        if self.drift_events:
            drift_df = pd.DataFrame(self.drift_events)
            drift_df.to_csv(os.path.join(self.save_dir, 'drift_events.csv'), index=False)
        
        # Save model and learner state
        self.learner.save(os.path.join(self.save_dir, 'learner_state'))
        
        # Create plots if targets are available
        if self.targets:
            self._create_plots(plots_dir)
    
    def _create_plots(self, plots_dir: str) -> None:
        """
        Create evaluation plots
        
        Args:
            plots_dir: Directory to save plots
        """
        import matplotlib.pyplot as plt
        from matplotlib.dates import DateFormatter
        
        # Check if timestamps are datetime
        is_datetime = False
        if len(self.timestamps) > 0 and isinstance(self.timestamps[0], (datetime, pd.Timestamp)):
            is_datetime = True
        
        # 1. Prediction vs target plot
        plt.figure(figsize=(12, 6))
        
        if is_datetime:
            plt.plot(self.timestamps, self.targets, 'b-', label='Actual')
            plt.plot(self.timestamps, self.predictions, 'r-', label='Prediction')
            
            # Mark drift events
            for event in self.drift_events:
                plt.axvline(x=event['timestamp'], color='g', linestyle='--', alpha=0.5)
            
            # Format dates
            plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
            plt.xticks(rotation=45)
        else:
            plt.plot(self.targets, 'b-', label='Actual')
            plt.plot(self.predictions, 'r-', label='Prediction')
            
            # Mark drift events
            for event in self.drift_events:
                plt.axvline(x=event['timestamp'], color='g', linestyle='--', alpha=0.5)
        
        plt.title('Prediction vs Actual')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(os.path.join(plots_dir, 'prediction_vs_actual.png'), dpi=300)
        plt.close()
        
        # 2. Error distribution plot
        errors = np.array(self.predictions) - np.array(self.targets)
        
        plt.figure(figsize=(10, 6))
        plt.hist(errors, bins=50, alpha=0.7)
        plt.axvline(x=0, color='r', linestyle='--')
        plt.title('Prediction Error Distribution')
        plt.xlabel('Error')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(os.path.join(plots_dir, 'error_distribution.png'), dpi=300)
        plt.close()
        
        # 3. Cumulative prediction error plot
        cumulative_error = np.cumsum(np.abs(errors))
        
        plt.figure(figsize=(10, 6))
        if is_datetime:
            plt.plot(self.timestamps, cumulative_error)
            plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
            plt.xticks(rotation=45)
        else:
            plt.plot(cumulative_error)
        
        # Mark drift events
        for event in self.drift_events:
            idx = self.timestamps.index(event['timestamp']) if event['timestamp'] in self.timestamps else -1
            if idx >= 0:
                plt.scatter(event['timestamp'] if is_datetime else idx, 
                          cumulative_error[idx], 
                          color='r', 
                          zorder=5)
        
        plt.title('Cumulative Absolute Error')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(os.path.join(plots_dir, 'cumulative_error.png'), dpi=300)
        plt.close()
        
        # 4. Drift score plot if drift events exist
        if self.drift_events:
            drift_timestamps = [event['timestamp'] for event in self.drift_events]
            drift_scores = [event['drift_score'] for event in self.drift_events]
            
            plt.figure(figsize=(10, 6))
            if is_datetime:
                plt.plot(drift_timestamps, drift_scores, 'o-')
                plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
                plt.xticks(rotation=45)
            else:
                plt.plot(drift_scores, 'o-')
            
            plt.title('Concept Drift Scores')
            plt.ylabel('Drift Score')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            plt.savefig(os.path.join(plots_dir, 'drift_scores.png'), dpi=300)
            plt.close()

def create_stream_processor(
    model_path: str,
    model_type: str = "cognitive",
    feature_names: Optional[List[str]] = None,
    window_size: int = 10,
    device: str = "cpu"
) -> StreamProcessor:
    """
    Create a stream processor with a pre-trained model
    
    Args:
        model_path: Path to model checkpoint
        model_type: Model type ('cognitive' or 'baseline')
        feature_names: List of feature names (optional)
        window_size: Window size for sequence data
        device: Computation device
        
    Returns:
        Initialized stream processor
    """
    # Determine available features if not provided
    if feature_names is None:
        # Basic financial features
        feature_names = [
            'open', 'high', 'low', 'close', 'volume',
            'return_1d', 'return_5d', 'volatility_10d',
            'rsi_14', 'macd', 'ma_20', 'ma_50'
        ]
    
    # Create model based on type
    if model_type.lower() == "cognitive":
        from src.arch.cognitive import CognitiveArchitecture
        model = CognitiveArchitecture()
    else:
        from src.arch.baseline_lstm import FinancialLSTMBaseline
        model = FinancialLSTMBaseline()
    
    # Load model weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # Create stream processor
    processor = StreamProcessor(
        model=model,
        feature_names=feature_names,
        window_size=window_size,
        save_dir=f"{model_type}_online_learning_results",
        device=device
    )
    
    return processor

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Online learning for financial models")
    parser.add_argument("--model_path", required=True, help="Path to trained model checkpoint")
    parser.add_argument("--data_path", required=True, help="Path to input data CSV")
    parser.add_argument("--model_type", choices=["cognitive", "baseline"], default="cognitive", 
                      help="Model type")
    parser.add_argument("--target_col", default="return_1d", 
                      help="Target column name in CSV")
    parser.add_argument("--timestamp_col", default="date", 
                      help="Timestamp column name in CSV")
    parser.add_argument("--window_size", type=int, default=10, 
                      help="Window size for sequence data")
    parser.add_argument("--output_dir", default="online_learning_results", 
                      help="Directory to save results")
    parser.add_argument("--no_update", action="store_true", 
                      help="Disable model updates (evaluation only)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", 
                      help="Computation device")
    
    args = parser.parse_args()
    
    # Read data
    data = pd.read_csv(args.data_path)
    
    # Create stream processor
    processor = create_stream_processor(
        model_path=args.model_path,
        model_type=args.model_type,
        window_size=args.window_size,
        device=args.device
    )
    
    # Override save directory
    processor.save_dir = args.output_dir
    
    # Process data
    result_df = processor.process_dataframe(
        df=data,
        target_col=args.target_col,
        timestamp_col=args.timestamp_col,
        update_model=not args.no_update
    )
    
    # Save results
    result_df.to_csv(os.path.join(args.output_dir, "processed_data.csv"), index=False)
    processor.save_results()
    
    print(f"Online learning completed. Results saved to {args.output_dir}")
