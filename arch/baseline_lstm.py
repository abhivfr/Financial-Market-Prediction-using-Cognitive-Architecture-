#!/usr/bin/env python
# baseline_lstm.py - Baseline LSTM model for financial prediction

import torch
import torch.nn as nn
import torch.nn.functional as F

class FinancialLSTMBaseline(nn.Module):
    """
    Baseline LSTM model for financial forecasting
    """
    def __init__(self, input_dim=5, hidden_dim=64, num_layers=2, output_dim=4, dropout=0.2):
        """
        Initialize baseline LSTM model
        
        Args:
            input_dim: Number of input features
            hidden_dim: Hidden dimension size
            num_layers: Number of LSTM layers
            output_dim: Number of output features
            dropout: Dropout rate
        """
        super(FinancialLSTMBaseline, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)
        
        # Batch normalization for improved stability
        self.bn_input = nn.BatchNorm1d(input_dim)
        self.bn_hidden = nn.BatchNorm1d(hidden_dim)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for faster convergence"""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                
        # Initialize linear layer
        nn.init.xavier_uniform_(self.fc.weight)
        self.fc.bias.data.fill_(0)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, seq_length, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        batch_size, seq_len, _ = x.size()
        
        # Apply batch normalization to input features
        # Reshape for batch norm, then reshape back
        x_reshaped = x.contiguous().view(batch_size * seq_len, self.input_dim)
        x_normalized = self.bn_input(x_reshaped).view(batch_size, seq_len, self.input_dim)
        
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x_normalized)
        
        # Get last time step output
        last_time_step = lstm_out[:, -1, :]
        
        # Apply batch normalization to hidden state
        last_time_step = self.bn_hidden(last_time_step)
        
        # Apply dropout
        dropout_out = self.dropout(last_time_step)
        
        # Final prediction
        output = self.fc(dropout_out)
        
        return output
