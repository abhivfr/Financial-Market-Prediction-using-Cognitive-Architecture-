import pandas as pd
import numpy as np
import torch
from pathlib import Path
import os
from sklearn.preprocessing import StandardScaler  # Add this import for data scaling

class EnhancedFinancialDataLoader:
    def __init__(self, data_path, sequence_length=20, batch_size=16, regime_aware=True, 
                augmentation=False, regime_path=None):
        """
        Enhanced financial data loader with regime awareness and augmentation
        
        Args:
            data_path: Path to CSV data file
            sequence_length: Length of historical sequences
            batch_size: Number of samples per batch
            regime_aware: Whether to use regime information if available
            augmentation: Whether to apply data augmentation during training
            regime_path: Optional path to regime data (if not in main data file)
        """
        data_path = str(data_path).replace('\\', '/')
        self.data_path = Path(data_path)
        print(f"Loading data from: {self.data_path.absolute()}")
        
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.regime_aware = regime_aware
        self.augmentation = augmentation
        
        # Load and process data
        raw_data = pd.read_csv(data_path)
        self.feature_columns, self.data = self._process_raw(raw_data)
        
        # Initialize scalers for data normalization
        self.scaler = StandardScaler()
        self.data = self.scaler.fit_transform(self.data)
        print(f"Data normalized with mean: {self.scaler.mean_[:5]}... and scale: {self.scaler.scale_[:5]}...")
        
        # Load regime data if separate
        if regime_path and regime_aware:
            try:
                regime_data = pd.read_csv(regime_path)
                # Merge regime data with main data
                if 'timestamp' in regime_data.columns and 'timestamp' in self.data.columns:
                    regime_data['timestamp'] = pd.to_datetime(regime_data['timestamp'])
                    # Convert to DataFrame with datetime index for merging
                    self.data = pd.DataFrame(self.data)
                    self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])
                    self.data = pd.merge(self.data, regime_data[['timestamp', 'volatility_regime', 'trend_regime']], 
                                        on='timestamp', how='left')
                    # Convert back to numpy array
                    self.data = self.data.values
            except Exception as e:
                print(f"Error loading regime data: {e}")
        
        self.pointer = 0
        self.epoch = 0

        print(f"Loaded and processed data shape: {self.data.shape}")
        print(f"Sequence length: {self.sequence_length}")
        print(f"Regime-aware: {self.regime_aware}")
        print(f"Augmentation: {self.augmentation}")

    def _process_raw(self, df):
        """Process raw data with enhanced checks and normalization"""
        print("Processing raw data...")
        
        # Check for essential columns
        essential_cols = ['price', 'volume']
        for col in essential_cols:
            if col not in df.columns:
                raise ValueError(f"Essential column '{col}' not found in data")
        
        # Gather feature columns - more flexible approach
        feature_cols = ['price', 'volume']
        
        # Add returns and volatility if they exist
        if 'returns' in df.columns:
            feature_cols.append('returns')
        else:
            df['returns'] = df['price'].pct_change()
            feature_cols.append('returns')
            
        if 'volatility' in df.columns:
            feature_cols.append('volatility')
        elif 'volatility_20d' in df.columns:
            feature_cols.append('volatility_20d')
            df['volatility'] = df['volatility_20d']
            feature_cols.append('volatility')
        else:
            df['volatility'] = df['returns'].rolling(window=20).std()
            feature_cols.append('volatility')
        
        # Add additional features if they exist
        extended_features = ['momentum_5d', 'momentum_10d', 'rsi', 'ma_cross', 'trend']
        for feature in extended_features:
            if feature in df.columns:
                feature_cols.append(feature)
        
        # Add regime columns if they exist and regime_aware is enabled
        if self.regime_aware:
            if 'volatility_regime' in df.columns:
                feature_cols.append('volatility_regime')
            if 'trend_regime' in df.columns:
                feature_cols.append('trend_regime')
        
        # Ensure all data is numeric
        for col in feature_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Drop rows with NaN values
        df = df.dropna(subset=feature_cols)
        
        # Replace any inf values with NaN and drop those rows
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df = df.dropna(subset=feature_cols)
        
        # Extract features as numpy array
        features = df[feature_cols].values
        
        print(f"Processed data shape: {features.shape}")
        print(f"Feature columns: {feature_cols}")
        
        return feature_cols, features

    def apply_augmentation(self, sequence):
        """Apply data augmentation techniques to improve training"""
        if not self.augmentation or np.random.random() > 0.5:
            return sequence
            
        # Choose a random augmentation technique
        aug_type = np.random.choice(['noise', 'scale', 'shift', 'none'])
        
        if aug_type == 'noise':
            # Add small Gaussian noise
            noise_level = np.random.uniform(0.001, 0.03)
            noise = np.random.normal(0, noise_level, sequence.shape)
            return sequence + noise
            
        elif aug_type == 'scale':
            # Slightly scale the values
            scale_factor = np.random.uniform(0.95, 1.05)
            return sequence * scale_factor
            
        elif aug_type == 'shift':
            # Shift values slightly
            shift_amount = np.random.uniform(-0.05, 0.05)
            return sequence + shift_amount
            
        return sequence  # 'none' case - no augmentation

    def __iter__(self):
        # Reset pointer for a new epoch
        self.pointer = 0
        self.epoch += 1
        return self

    def __next__(self):
        # Check if we have enough data left for a batch
        if self.pointer + self.batch_size > len(self.data) - self.sequence_length:
            # Reset pointer and raise StopIteration
            self.pointer = 0
            raise StopIteration

        batch_indices = np.arange(self.pointer, self.pointer + self.batch_size)
        self.pointer += self.batch_size

        # Construct sequences and targets
        sequences = []
        targets = []
        current_features = []
        
        for i in batch_indices:
            if i + self.sequence_length < len(self.data):
                seq = self.data[i:i + self.sequence_length]
                target = self.data[i + self.sequence_length]
                
                # Skip sequences with NaN or inf values
                if np.isnan(seq).any() or np.isinf(seq).any() or np.isnan(target).any() or np.isinf(target).any():
                    continue
                
                # Apply augmentation if enabled
                if self.augmentation and self.epoch > 1:  # No augmentation on first epoch
                    seq = self.apply_augmentation(seq)
                
                sequences.append(seq)
                targets.append(target)
                current_features.append(self.data[i + self.sequence_length - 1])

        # Create batches
        if len(sequences) == 0:
            # If we couldn't find any valid sequences, try again with new indices
            self.pointer += self.batch_size
            return self.__next__()
        
        batch_sequences = torch.FloatTensor(np.stack(sequences))
        batch_targets = torch.FloatTensor(np.stack(targets))
        batch_features = torch.FloatTensor(np.stack(current_features))

        # Check for NaN or Inf values before returning
        if torch.isnan(batch_sequences).any() or torch.isinf(batch_sequences).any() or \
           torch.isnan(batch_targets).any() or torch.isinf(batch_targets).any():
            print("Warning: NaN or Inf values found in batch. Using new batch...")
            return self.__next__()

        return {
            'features': batch_features,         # Current timestep features
            'sequence': batch_sequences,        # Historical sequence
            'target': batch_targets             # Next timestep features (target)
        }
    
    def __len__(self):
        """Return the number of batches in the dataset"""
        return max(0, (len(self.data) - self.sequence_length) // self.batch_size)

def create_data_loaders(train_path, val_path=None, test_path=None, batch_size=16, 
                       sequence_length=20, regime_aware=True, augmentation=True):
    """
    Create data loaders for training, validation, and testing
    
    Args:
        train_path: Path to training data
        val_path: Path to validation data (optional)
        test_path: Path to test data (optional)
        batch_size: Batch size for data loading
        sequence_length: Sequence length for temporal data
        regime_aware: Whether to use regime information
        augmentation: Whether to apply data augmentation
    
    Returns:
        Dictionary containing data loaders
    """
    loaders = {}
    
    # Create training loader
    train_loader = EnhancedFinancialDataLoader(
        data_path=train_path,
        sequence_length=sequence_length,
        batch_size=batch_size,
        regime_aware=regime_aware,
        augmentation=augmentation
    )
    loaders['train'] = train_loader
    
    # Create validation loader if path provided
    if val_path and os.path.exists(val_path):
        val_loader = EnhancedFinancialDataLoader(
            data_path=val_path,
            sequence_length=sequence_length,
            batch_size=batch_size,
            regime_aware=regime_aware,
            augmentation=False  # No augmentation for validation
        )
        loaders['val'] = val_loader
    
    # Create test loader if path provided
    if test_path and os.path.exists(test_path):
        test_loader = EnhancedFinancialDataLoader(
            data_path=test_path,
            sequence_length=sequence_length,
            batch_size=batch_size,
            regime_aware=regime_aware,
            augmentation=False  # No augmentation for testing
        )
        loaders['test'] = test_loader
    
    return loaders
