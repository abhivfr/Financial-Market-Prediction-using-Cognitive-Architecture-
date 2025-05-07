#!/usr/bin/env python
# uncertainty_loss.py - Uncertainty-aware loss function

import torch
import torch.nn as nn
import torch.nn.functional as F

class UncertaintyAwareLoss(nn.Module):
    """
    Uncertainty-aware loss function that incorporates model's uncertainty estimates
    """
    def __init__(self, base_loss='mse', beta=0.1, reduction='mean'):
        """
        Initialize uncertainty-aware loss function
        
        Args:
            base_loss: Base loss type ('mse' or 'mae')
            beta: Weight for uncertainty regularization
            reduction: Loss reduction method ('mean' or 'sum')
        """
        super(UncertaintyAwareLoss, self).__init__()
        
        self.base_loss = base_loss
        self.beta = beta
        self.reduction = reduction
        
        # Set base loss function
        if base_loss == 'mse':
            self.loss_fn = nn.MSELoss(reduction='none')
        elif base_loss == 'mae':
            self.loss_fn = nn.L1Loss(reduction='none')
        else:
            raise ValueError(f"Unsupported base loss: {base_loss}")
    
    def forward(self, predictions, targets, uncertainty):
        """
        Calculate uncertainty-aware loss
        
        Args:
            predictions: Model predictions (batch_size, features)
            targets: Target values (batch_size, features)
            uncertainty: Model's uncertainty estimates (batch_size, features)
            
        Returns:
            Loss value
        """
        # Ensure uncertainty is positive
        log_variance = torch.log(torch.clamp(uncertainty, min=1e-6))
        
        # Calculate base loss
        base_loss = self.loss_fn(predictions, targets)
        
        # Weight loss by uncertainty
        precision = torch.exp(-log_variance)
        weighted_loss = precision * base_loss + self.beta * log_variance
        
        # Apply reduction
        if self.reduction == 'mean':
            return weighted_loss.mean()
        elif self.reduction == 'sum':
            return weighted_loss.sum()
        else:
            return weighted_loss
