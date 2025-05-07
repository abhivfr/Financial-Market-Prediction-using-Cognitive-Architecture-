import os
import torch
import numpy as np
import pandas as pd
from src.data.financial_loader import FinancialDataLoader

def test_financial_loader():
    loader = FinancialDataLoader(
        data_path="data/financial",
        sequence_length=10,
        batch_size=32
    )
    
    # Generate sample data for testing
    sample_data = {
        'timestamp': range(1000),
        'price': np.random.normal(100, 10, 1000),
        'volume': np.random.normal(1000, 100, 1000)
    }
    
    df = pd.DataFrame(sample_data)
    os.makedirs('data/financial', exist_ok=True)
    df.to_csv('data/financial/sample_data.csv', index=False)
    
    # Test loading
    train_seq, val_seq = loader.load_and_prepare('data/financial/sample_data.csv')
    
    print("Data Loading Test Results:")
    print(f"Training sequences shape: {train_seq[0].shape}")
    print(f"Training targets shape: {train_seq[1].shape}")
    print(f"Validation sequences shape: {val_seq[0].shape}")
    print(f"Validation targets shape: {val_seq[1].shape}")
    
    return train_seq, val_seq

if __name__ == "__main__":
    test_financial_loader()
