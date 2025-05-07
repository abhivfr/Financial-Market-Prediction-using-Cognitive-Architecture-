import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from typing import Tuple, Dict

class FinancialDataProcessor:
    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.scalers = {
            'price': RobustScaler(),
            'volume': RobustScaler(),
            'returns': RobustScaler(),
            'volatility': RobustScaler()
        }
        self.stats = {}
        
    def process_raw_data(self, df: pd.DataFrame) -> Tuple[torch.Tensor, Dict]:
        # Calculate financial metrics
        df['returns'] = df['price'].pct_change()
        df['volatility'] = df['returns'].rolling(window=5).std()
        df['volume_ma'] = df['volume'].rolling(window=5).mean()
        
        # Handle missing values
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        # Scale each dimension independently
        scaled_data = {}
        for feature in ['price', 'volume', 'returns', 'volatility']:
            scaled_data[feature] = self.scalers[feature].fit_transform(
                df[feature].values.reshape(-1, 1)
            )
            
        # Create 4D tensor
        market_tensor = torch.FloatTensor(np.column_stack([
            scaled_data['price'],
            scaled_data['volume'],
            scaled_data['returns'],
            scaled_data['volatility']
        ]))
        
        # Calculate statistics
        self.stats = {
            'mean': market_tensor.mean(dim=0),
            'std': market_tensor.std(dim=0),
            'min': market_tensor.min(dim=0)[0],
            'max': market_tensor.max(dim=0)[0]
        }
        
        return market_tensor, self.stats
        
    def create_sequences(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        sequences = []
        targets = []
        
        for i in range(len(tensor) - self.window_size):
            seq = tensor[i:i + self.window_size]
            target = tensor[i + self.window_size]
            sequences.append(seq)
            targets.append(target)
            
        return (
            torch.stack(sequences),
            torch.stack(targets)
        )
        
    def normalize_sequence(self, seq: torch.Tensor) -> torch.Tensor:
        # Normalize while preserving 4D relationships
        mean = seq.mean(dim=0, keepdim=True)
        std = seq.std(dim=0, keepdim=True)
        return (seq - mean) / (std + 1e-8)
        
    def denormalize(self, tensor: torch.Tensor, feature_idx: int) -> torch.Tensor:
        # Convert back to original scale for a specific feature
        feature_names = ['price', 'volume', 'returns', 'volatility']
        scaler = self.scalers[feature_names[feature_idx]]
        return torch.FloatTensor(
            scaler.inverse_transform(tensor[:, feature_idx].cpu().numpy().reshape(-1, 1))
        )
        
    def get_feature_importance(self) -> Dict[str, float]:
        # Calculate feature importance based on variance
        importance = torch.var(
            torch.stack([self.stats['max'], self.stats['min']]), dim=0
        )
        return {
            'price': importance[0].item(),
            'volume': importance[1].item(),
            'returns': importance[2].item(),
            'volatility': importance[3].item()
        }