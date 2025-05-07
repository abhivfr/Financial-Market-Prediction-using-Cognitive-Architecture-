#!/usr/bin/env python
# market_loss.py - Market-aware loss function

import torch
import torch.nn as nn
import torch.nn.functional as F

class MarketAwareLoss(nn.Module):
    """
    Market-aware loss function that puts different emphasis on different market conditions
    """
    def __init__(self, price_weight=1.0, direction_weight=0.5, volatility_weight=0.3,
                volume_weight=0.2, high_volatility_multiplier=2.0):
        """
        Initialize market-aware loss function
        
        Args:
            price_weight: Weight for price prediction error
            direction_weight: Weight for direction prediction error
            volatility_weight: Weight for volatility prediction error
            volume_weight: Weight for volume prediction error
            high_volatility_multiplier: Multiplier for loss during high volatility periods
        """
        super(MarketAwareLoss, self).__init__()
        
        self.price_weight = price_weight
        self.direction_weight = direction_weight
        self.volatility_weight = volatility_weight
        self.volume_weight = volume_weight
        self.high_volatility_multiplier = high_volatility_multiplier
        
        # Base loss functions
        self.mse_loss = nn.MSELoss(reduction='none')
    
    def forward(self, predictions, targets):
        """
        Calculate market-aware loss
        
        Args:
            predictions: Model predictions (batch_size, features)
            targets: Target values (batch_size, features)
            
        Returns:
            Loss value
        """
        # Calculate base MSE loss for each feature
        mse_loss = self.mse_loss(predictions, targets)
        
        # Extract individual feature losses
        price_loss = mse_loss[:, 0].mean()
        
        # Extract volatility from targets (assuming it's the 4th feature)
        if targets.shape[1] >= 4:
            volatility = targets[:, 3]
            
            # Calculate high volatility mask
            # Consider periods with volatility > 75th percentile as high volatility
            high_volatility_threshold = torch.quantile(volatility, 0.75)
            high_volatility_mask = (volatility > high_volatility_threshold).float()
            
            # Calculate direction loss if more than one sample in batch
            if predictions.shape[0] > 1:
                # Calculate price directions
                pred_directions = torch.sign(predictions[1:, 0] - predictions[:-1, 0])
                true_directions = torch.sign(targets[1:, 0] - targets[:-1, 0])
                
                # Handle zeros
                pred_directions = torch.where(pred_directions == 0, 
                                             torch.tensor(0.01, device=predictions.device), 
                                             pred_directions)
                true_directions = torch.where(true_directions == 0, 
                                             torch.tensor(0.01, device=targets.device), 
                                             true_directions)
                
                # Direction loss
                direction_loss = torch.mean((pred_directions != true_directions).float())
            else:
                direction_loss = torch.tensor(0.0, device=predictions.device)
            
            # Apply higher weights to high volatility periods for price prediction
            weighted_price_loss = price_loss * (1.0 + high_volatility_mask.mean() * (self.high_volatility_multiplier - 1.0))
        else:
            # Default to standard loss if volatility not available
            weighted_price_loss = price_loss
            direction_loss = torch.tensor(0.0, device=predictions.device)
        
        # Calculate volume loss if available (assuming it's the 2nd feature)
        if targets.shape[1] >= 2:
            volume_loss = mse_loss[:, 1].mean()
        else:
            volume_loss = torch.tensor(0.0, device=predictions.device)
        
        # Calculate volatility loss if available (assuming it's the 4th feature)
        if targets.shape[1] >= 4:
            volatility_loss = mse_loss[:, 3].mean()
        else:
            volatility_loss = torch.tensor(0.0, device=predictions.device)
        
        # Combine losses with weights
        total_loss = (
            self.price_weight * weighted_price_loss + 
            self.direction_weight * direction_loss + 
            self.volatility_weight * volatility_loss + 
            self.volume_weight * volume_loss
        )
        
        return total_loss

def train_step(model, optimizer, financial_data, financial_seq, targets):
    """Enhanced training step with noise injection and gradient stabilization"""
    optimizer.zero_grad()
    
    device = financial_data.device
    
    # Extract volume data for special processing
    volume_data = financial_seq[:, :, 1:2]
    
    # Add feature noise during training for better generalization
    if model.training:
        noise_scale = 0.01
        financial_seq_noised = financial_seq + torch.randn_like(financial_seq) * noise_scale
    else:
        financial_seq_noised = financial_seq
    
    # Forward pass with enhanced volume attention
    outputs = model(
        financial_data=financial_data,
        financial_seq=financial_seq_noised,
        volume=volume_data,
        skip_memory=False
    )
    
    # Initialize loss functions if not already done
    if not hasattr(train_step, 'market_loss_fn'):
        train_step.market_loss_fn = MarketAwareLoss().to(device)
    if not hasattr(train_step, 'uncertainty_loss_fn'):
        from src.utils.uncertainty_loss import UncertaintyAwareLoss
        train_step.uncertainty_loss_fn = UncertaintyAwareLoss(base_loss='mse', beta=0.1).to(device)
    
    # Calculate losses with improved stability
    market_loss = train_step.market_loss_fn(outputs['market_state'], targets)
    
    uncertainty_loss = train_step.uncertainty_loss_fn(
        outputs['market_state'],
        targets,
        outputs.get('uncertainty', None)
    )
    
    # Improved regularization with L2 + gradient noise
    l2_reg = 0.0
    for param in model.parameters():
        if param.requires_grad:
            l2_reg += torch.norm(param)
    
    # Combine losses with adaptive weighting
    # Higher uncertainty loss weight in low volatility, higher market loss in high volatility
    uncertainty_weight = 1.0 - train_step.market_loss_fn.volatility_regime.item() * 0.5
    market_weight = 0.8 + train_step.market_loss_fn.volatility_regime.item() * 0.4
    
    total_loss = market_weight * market_loss + uncertainty_weight * uncertainty_loss + 5e-6 * l2_reg
    
    # Enhanced gradient handling
    # Gradient clipping before backward pass
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
    
    # Backward pass with anomaly detection in high volatility regimes
    with torch.autograd.detect_anomaly():
        total_loss.backward()
    
    # Add gradient noise for better generalization
    for param in model.parameters():
        if param.grad is not None and model.training:
            param.grad += torch.randn_like(param.grad) * 1e-5
    
    # Additional gradient clipping after backward pass
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
    
    optimizer.step()
    
    return total_loss.item(), outputs
