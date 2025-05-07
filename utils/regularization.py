#!/usr/bin/env python
# regularization.py - Enhanced regularization techniques

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, List, Optional

class RegularizationManager:
    """
    Comprehensive regularization manager for complex neural architectures
    """
    def __init__(
        self, 
        l1_lambda: float = 0.0,
        l2_lambda: float = 1e-5,
        orthogonal_lambda: float = 1e-3,
        gradient_clip_threshold: float = 1.0,
        activation_l1_lambda: float = 0.0,
        attention_entropy_lambda: float = 0.01,
        module_filter: Optional[List[str]] = None
    ):
        """
        Initialize regularization manager
        
        Args:
            l1_lambda: L1 regularization strength
            l2_lambda: L2 regularization strength
            orthogonal_lambda: Orthogonal regularization strength for recurrent layers
            gradient_clip_threshold: Gradient clipping threshold
            activation_l1_lambda: L1 penalty on activations for sparsity
            attention_entropy_lambda: Entropy regularization for attention (higher entropy = more exploration)
            module_filter: List of module name patterns to apply regularization to (None = all)
        """
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda
        self.orthogonal_lambda = orthogonal_lambda
        self.gradient_clip_threshold = gradient_clip_threshold
        self.activation_l1_lambda = activation_l1_lambda
        self.attention_entropy_lambda = attention_entropy_lambda
        self.module_filter = module_filter
        
        # Activation storage for regularization
        self.stored_activations = {}
        self.attention_weights = {}
        self.hooks = []
    
    def register_hooks(self, model: nn.Module) -> None:
        """
        Register hooks to collect activations and attention weights
        
        Args:
            model: PyTorch model
        """
        # Remove existing hooks
        self.remove_hooks()
        
        # Register new activation hooks
        for name, module in model.named_modules():
            # Skip if not in filter (if filter is provided)
            if self.module_filter and not any(pattern in name for pattern in self.module_filter):
                continue
            
            # Hook for attention modules
            if 'attention' in name.lower() and hasattr(module, 'forward'):
                hook = module.register_forward_hook(self._attention_hook(name))
                self.hooks.append(hook)
            
            # Hook for activation modules
            elif isinstance(module, (nn.ReLU, nn.LeakyReLU, nn.Tanh, nn.Sigmoid)):
                hook = module.register_forward_hook(self._activation_hook(name))
                self.hooks.append(hook)
    
    def remove_hooks(self) -> None:
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def _activation_hook(self, name: str):
        """Hook to store activations"""
        def hook(module, input, output):
            self.stored_activations[name] = output.detach()
        return hook
    
    def _attention_hook(self, name: str):
        """Hook to store attention weights"""
        def hook(module, input, output):
            if isinstance(output, tuple) and len(output) > 1:
                # Multi-head attention typically returns (output, attention_weights)
                self.attention_weights[name] = output[1].detach()
            elif hasattr(module, 'attention_weights') and module.attention_weights is not None:
                # For our custom attention that stores weights
                self.attention_weights[name] = module.attention_weights.detach()
        return hook
    
    def get_weight_regularization_loss(self, model: nn.Module) -> torch.Tensor:
        """
        Calculate weight regularization loss
        
        Args:
            model: PyTorch model
            
        Returns:
            Regularization loss
        """
        l1_loss = torch.tensor(0.0, device=next(model.parameters()).device)
        l2_loss = torch.tensor(0.0, device=next(model.parameters()).device)
        orthogonal_loss = torch.tensor(0.0, device=next(model.parameters()).device)
        
        for name, param in model.named_parameters():
            # Skip if not in filter (if filter is provided)
            if self.module_filter and not any(pattern in name for pattern in self.module_filter):
                continue
            
            # Skip bias terms for weight decay
            if 'bias' in name:
                continue
            
            # L1 regularization
            if self.l1_lambda > 0:
                l1_loss += self.l1_lambda * torch.sum(torch.abs(param))
            
            # L2 regularization (weight decay)
            if self.l2_lambda > 0:
                l2_loss += self.l2_lambda * torch.sum(param ** 2)
            
            # Orthogonal regularization for recurrent layers
            if self.orthogonal_lambda > 0 and ('lstm' in name or 'gru' in name) and param.dim() > 1:
                param_flat = param.view(param.size(0), -1)
                sym = torch.mm(param_flat, param_flat.t())
                sym -= torch.eye(param_flat.size(0), device=param.device)
                orthogonal_loss += self.orthogonal_lambda * torch.sum(sym ** 2)
        
        return l1_loss + l2_loss + orthogonal_loss
    
    def get_activation_regularization_loss(self) -> torch.Tensor:
        """
        Calculate activation regularization loss
        
        Returns:
            Activation regularization loss
        """
        if not self.stored_activations:
            return torch.tensor(0.0)
        
        device = next(iter(self.stored_activations.values())).device
        activation_loss = torch.tensor(0.0, device=device)
        
        # Apply L1 penalty on activations for sparsity
        if self.activation_l1_lambda > 0:
            for name, activation in self.stored_activations.items():
                activation_loss += self.activation_l1_lambda * torch.mean(torch.abs(activation))
        
        return activation_loss
    
    def get_attention_entropy_loss(self) -> torch.Tensor:
        """
        Calculate attention entropy regularization loss
        
        Returns:
            Attention entropy loss
        """
        if not self.attention_weights:
            return torch.tensor(0.0)
        
        device = next(iter(self.attention_weights.values())).device
        entropy_loss = torch.tensor(0.0, device=device)
        
        # Encourage more diverse attention (higher entropy)
        if self.attention_entropy_lambda > 0:
            for name, attention in self.attention_weights.items():
                # Ensure proper shape for attention weights
                if attention.dim() > 2:
                    # For multi-head attention: [batch, heads, seq_len, seq_len]
                    # Average across heads
                    if attention.dim() == 4:
                        attention = attention.mean(dim=1)
                    
                    # Calculate entropy along the last dimension
                    log_probs = torch.log(attention + 1e-10)
                    entropy = -torch.sum(attention * log_probs, dim=-1)
                    
                    # Negative entropy loss (maximize entropy)
                    entropy_loss -= self.attention_entropy_lambda * torch.mean(entropy)
        
        return entropy_loss
    
    def get_total_regularization_loss(self, model: nn.Module) -> torch.Tensor:
        """
        Calculate total regularization loss
        
        Args:
            model: PyTorch model
            
        Returns:
            Total regularization loss
        """
        weight_loss = self.get_weight_regularization_loss(model)
        activation_loss = self.get_activation_regularization_loss()
        attention_loss = self.get_attention_entropy_loss()
        
        return weight_loss + activation_loss + attention_loss
    
    def clip_gradients(self, model: nn.Module) -> None:
        """
        Apply gradient clipping
        
        Args:
            model: PyTorch model
        """
        if self.gradient_clip_threshold > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 
                self.gradient_clip_threshold
            )
    
    def get_sparsity_metrics(self) -> Dict[str, float]:
        """
        Calculate sparsity metrics for model activations
        
        Returns:
            Dictionary of sparsity metrics
        """
        metrics = {}
        
        for name, activation in self.stored_activations.items():
            # Calculate percentage of zeros (or near-zeros)
            sparsity = torch.mean((torch.abs(activation) < 1e-6).float()).item()
            metrics[f"sparsity_{name}"] = sparsity
        
        # Overall sparsity
        if self.stored_activations:
            metrics["sparsity_overall"] = np.mean(list(metrics.values()))
        
        return metrics
