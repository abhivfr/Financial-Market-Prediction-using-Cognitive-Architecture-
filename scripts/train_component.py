#!/usr/bin/env python
# train_component.py - Train individual components of the cognitive architecture

import os
import sys
import argparse
import torch
import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, root_dir)

from src.arch.cognitive import CognitiveArchitecture
from src.data.financial_loader import EnhancedFinancialDataLoader
from src.utils.regularization import RegularizationManager

def train_memory_component(
    model: CognitiveArchitecture,
    train_loader: EnhancedFinancialDataLoader,
    val_loader: EnhancedFinancialDataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    num_epochs: int,
    checkpoint_dir: str,
    log_dir: str
) -> Dict[str, Any]:
    """
    Train memory component specifically
    
    Args:
        model: Cognitive architecture model
        train_loader: Training data loader
        val_loader: Validation data loader
        optimizer: Optimizer
        device: Device to train on
        num_epochs: Number of training epochs
        checkpoint_dir: Directory to save checkpoints
        log_dir: Directory to save logs
        
    Returns:
        Dictionary with training results
    """
    print("Training memory component...")
    
    # Create directories
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Initialize tracking variables
    best_val_loss = float('inf')
    best_model_state = None
    train_losses = []
    val_losses = []
    memory_metrics = []
    
    # Freeze non-memory components
    for name, param in model.named_parameters():
        if not any(x in name for x in ['memory_bank', 'memory_buffer', 'memory_encoder']):
            param.requires_grad = False
    
    # Custom memory loss function
    def memory_loss(outputs, targets, memory_batch):
        # Regular prediction loss
        pred_loss = torch.nn.functional.mse_loss(outputs['market_state'], targets)
        
        # Memory retrieval quality
        retrieval_loss = 0.0
        if 'memory_retrieval_score' in outputs:
            retrieval_loss = 1.0 - outputs['memory_retrieval_score'].mean()
        
        # Memory diversity - encourage diverse memory usage
        diversity_loss = 0.0
        if 'memory_usage' in outputs:
            usage = outputs['memory_usage']
            diversity_loss = 1.0 - (usage.std() / (usage.mean() + 1e-6))
        
        # Combined loss
        total_loss = pred_loss + 0.5 * retrieval_loss + 0.2 * diversity_loss
        
        return total_loss, {
            'pred_loss': pred_loss.item(),
            'retrieval_loss': retrieval_loss if isinstance(retrieval_loss, float) else retrieval_loss.item(),
            'diversity_loss': diversity_loss if isinstance(diversity_loss, float) else diversity_loss.item()
        }
    
    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        epoch_train_loss = 0.0
        train_batches = 0
        epoch_loss_components = {'pred_loss': 0.0, 'retrieval_loss': 0.0, 'diversity_loss': 0.0}
        
        for financial_data, financial_seq, targets in train_loader:
            # Move data to device
            financial_data = financial_data.to(device)
            financial_seq = financial_seq.to(device)
            targets = targets.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(financial_data=financial_data, financial_seq=financial_seq)
            
            # Calculate loss
            loss, loss_components = memory_loss(outputs, targets, financial_data)
            
            # Backward pass
            loss.backward()
            
            # Update parameters
            optimizer.step()
            
            # Update metrics
            epoch_train_loss += loss.item()
            train_batches += 1
            
            for k, v in loss_components.items():
                epoch_loss_components[k] += v
        
        # Calculate average training loss
        epoch_train_loss /= train_batches
        for k in epoch_loss_components:
            epoch_loss_components[k] /= train_batches
        
        # Validation phase
        model.eval()
        epoch_val_loss = 0.0
        val_batches = 0
        val_metrics = {'memory_retrieval_score': 0.0, 'memory_utilization': 0.0}
        
        with torch.no_grad():
            for financial_data, financial_seq, targets in val_loader:
                # Move data to device
                financial_data = financial_data.to(device)
                financial_seq = financial_seq.to(device)
                targets = targets.to(device)
                
                # Forward pass
                outputs = model(financial_data=financial_data, financial_seq=financial_seq)
                
                # Calculate loss
                loss, _ = memory_loss(outputs, targets, financial_data)
                
                # Update metrics
                epoch_val_loss += loss.item()
                val_batches += 1
                
                # Track memory-specific metrics
                if 'memory_retrieval_score' in outputs:
                    val_metrics['memory_retrieval_score'] += outputs['memory_retrieval_score'].mean().item()
                
                if 'memory_usage' in outputs:
                    val_metrics['memory_utilization'] += outputs['memory_usage'].mean().item()
        
        # Calculate average validation loss
        epoch_val_loss /= val_batches
        for k in val_metrics:
            val_metrics[k] /= val_batches
        
        # Store metrics
        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        memory_metrics.append(val_metrics)
        
        # Print progress
        print(f"Epoch {epoch+1}/{num_epochs}, "
              f"Train Loss: {epoch_train_loss:.4f}, "
              f"Val Loss: {epoch_val_loss:.4f}, "
              f"Memory Retrieval: {val_metrics['memory_retrieval_score']:.4f}, "
              f"Memory Utilization: {val_metrics['memory_utilization']:.4f}")
        
        # Save if best model
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model_state = model.state_dict()
            
            # Save checkpoint
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best_memory_component.pt"))
            
            print(f"Saved new best model with validation loss: {best_val_loss:.4f}")
        
        # Save logs
        metrics = {
            'epoch': epoch,
            'train_loss': epoch_train_loss,
            'val_loss': epoch_val_loss,
            **epoch_loss_components,
            **val_metrics
        }
        
        with open(os.path.join(log_dir, f"memory_epoch_{epoch}.json"), 'w') as f:
            json.dump(metrics, f, indent=2)
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(checkpoint_dir, "final_memory_component.pt"))
    
    # Save training curves
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot([m['memory_retrieval_score'] for m in memory_metrics], label='Retrieval Score')
    plt.plot([m['memory_utilization'] for m in memory_metrics], label='Memory Utilization')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.title('Memory Metrics')
    
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, "memory_training_curves.png"))
    
    # Return results
    return {
        'best_val_loss': best_val_loss,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'memory_metrics': memory_metrics,
        'best_model_state': best_model_state
    }

def train_attention_component(
    model: CognitiveArchitecture,
    train_loader: EnhancedFinancialDataLoader,
    val_loader: EnhancedFinancialDataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    num_epochs: int,
    checkpoint_dir: str,
    log_dir: str
) -> Dict[str, Any]:
    """
    Train attention component specifically
    
    Args:
        model: Cognitive architecture model
        train_loader: Training data loader
        val_loader: Validation data loader
        optimizer: Optimizer
        device: Device to train on
        num_epochs: Number of training epochs
        checkpoint_dir: Directory to save checkpoints
        log_dir: Directory to save logs
        
    Returns:
        Dictionary with training results
    """
    print("Training attention component...")
    
    # Create directories
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Initialize tracking variables
    best_val_loss = float('inf')
    best_model_state = None
    train_losses = []
    val_losses = []
    attention_metrics = []
    
    # Freeze non-attention components
    for name, param in model.named_parameters():
        if not any(x in name for x in ['attention', 'query', 'key', 'value']):
            param.requires_grad = False
    
    # Custom attention loss function
    def attention_loss(outputs, targets, attention_batch):
        # Regular prediction loss
        pred_loss = torch.nn.functional.mse_loss(outputs['market_state'], targets)
        
        # Attention entropy - encourage focused attention
        entropy_loss = 0.0
        if 'attention_weights' in outputs:
            weights = outputs['attention_weights']
            # Calculate entropy of attention weights
            entropy = -(weights * torch.log(weights + 1e-10)).sum(dim=-1).mean()
            entropy_loss = 0.1 * entropy
        
        # Encourage temporal attention patterns
        temporal_loss = 0.0
        if 'attention_weights' in outputs and outputs['attention_weights'].dim() > 2:
            weights = outputs['attention_weights']
            # Encourage smooth temporal transitions in attention
            if weights.size(1) > 1:  # Need at least 2 time steps
                temp_diff = (weights[:, 1:] - weights[:, :-1]).abs().mean()
                temporal_loss = 0.05 * temp_diff
        
        # Combined loss
        total_loss = pred_loss + entropy_loss + temporal_loss
        
        return total_loss, {
            'pred_loss': pred_loss.item(),
            'entropy_loss': entropy_loss if isinstance(entropy_loss, float) else entropy_loss.item(),
            'temporal_loss': temporal_loss if isinstance(temporal_loss, float) else temporal_loss.item()
        }
    
    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        epoch_train_loss = 0.0
        train_batches = 0
        epoch_loss_components = {'pred_loss': 0.0, 'entropy_loss': 0.0, 'temporal_loss': 0.0}
        
        for financial_data, financial_seq, targets in train_loader:
            # Move data to device
            financial_data = financial_data.to(device)
            financial_seq = financial_seq.to(device)
            targets = targets.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(financial_data=financial_data, financial_seq=financial_seq)
            
            # Calculate loss
            loss, loss_components = attention_loss(outputs, targets, financial_data)
            
            # Backward pass
            loss.backward()
            
            # Update parameters
            optimizer.step()
            
            # Update metrics
            epoch_train_loss += loss.item()
            train_batches += 1
            
            for k, v in loss_components.items():
                epoch_loss_components[k] += v
        
        # Calculate average training loss
        epoch_train_loss /= train_batches
        for k in epoch_loss_components:
            epoch_loss_components[k] /= train_batches
        
        # Validation phase
        model.eval()
        epoch_val_loss = 0.0
        val_batches = 0
        val_metrics = {'attention_entropy': 0.0, 'attention_focus': 0.0}
        
        with torch.no_grad():
            for financial_data, financial_seq, targets in val_loader:
                # Move data to device
                financial_data = financial_data.to(device)
                financial_seq = financial_seq.to(device)
                targets = targets.to(device)
                
                # Forward pass
                outputs = model(financial_data=financial_data, financial_seq=financial_seq)
                
                # Calculate loss
                loss, _ = attention_loss(outputs, targets, financial_data)
                
                # Update metrics
                epoch_val_loss += loss.item()
                val_batches += 1
                
                # Track attention-specific metrics
                if 'attention_weights' in outputs:
                    weights = outputs['attention_weights']
                    # Calculate entropy
                    entropy = -(weights * torch.log(weights + 1e-10)).sum(dim=-1).mean()
                    val_metrics['attention_entropy'] += entropy.item()
                    
                    # Calculate focus (max weight value)
                    val_metrics['attention_focus'] += weights.max(dim=-1)[0].mean().item()
        
        # Calculate average validation loss
        epoch_val_loss /= val_batches
        for k in val_metrics:
            val_metrics[k] /= val_batches
        
        # Store metrics
        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        attention_metrics.append(val_metrics)
        
        # Print progress
        print(f"Epoch {epoch+1}/{num_epochs}, "
              f"Train Loss: {epoch_train_loss:.4f}, "
              f"Val Loss: {epoch_val_loss:.4f}, "
              f"Attention Entropy: {val_metrics['attention_entropy']:.4f}, "
              f"Attention Focus: {val_metrics['attention_focus']:.4f}")
        
        # Save if best model
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model_state = model.state_dict()
            
            # Save checkpoint
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best_attention_component.pt"))
            
            print(f"Saved new best model with validation loss: {best_val_loss:.4f}")
        
        # Save logs
        metrics = {
            'epoch': epoch,
            'train_loss': epoch_train_loss,
            'val_loss': epoch_val_loss,
            **epoch_loss_components,
            **val_metrics
        }
        
        with open(os.path.join(log_dir, f"attention_epoch_{epoch}.json"), 'w') as f:
            json.dump(metrics, f, indent=2)
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(checkpoint_dir, "final_attention_component.pt"))
    
    # Save training curves
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot([m['attention_entropy'] for m in attention_metrics], label='Attention Entropy')
    plt.plot([m['attention_focus'] for m in attention_metrics], label='Attention Focus')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.title('Attention Metrics')
    
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, "attention_training_curves.png"))
    
    # Return results
    return {
        'best_val_loss': best_val_loss,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'attention_metrics': attention_metrics,
        'best_model_state': best_model_state
    }

def main():
    parser = argparse.ArgumentParser(description="Train individual components of cognitive architecture")
    
    # Data arguments
    parser.add_argument("--train_data", required=True, help="Path to training data")
    parser.add_argument("--val_data", required=True, help="Path to validation data")
    
    # Component arguments
    parser.add_argument("--component", required=True, choices=["memory", "attention", "introspection", "regime"],
                      help="Component to train")
    
    # Model arguments
    parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden dimension")
    parser.add_argument("--memory_size", type=int, default=50, help="Memory size")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--sequence_length", type=int, default=20, help="Sequence length")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    
    # Other arguments
    parser.add_argument("--output_dir", default="models/components", help="Output directory")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Computation device")
    
    args = parser.parse_args()
    
    # Create model
    model = CognitiveArchitecture(
        hidden_dim=args.hidden_dim,
        memory_size=args.memory_size
    )
    
    # Move model to device
    device = args.device
    model.to(device)
    
    # Create data loaders
    train_loader = EnhancedFinancialDataLoader(
        data_path=args.train_data,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        regime_aware=True,
        augmentation=True
    )
    
    val_loader = EnhancedFinancialDataLoader(
        data_path=args.val_data,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        regime_aware=True,
        augmentation=False
    )
    
    # Create optimizer based on component
    if args.component == "memory":
        optimizer_params = [p for n, p in model.named_parameters() 
                           if any(x in n for x in ['memory_bank', 'memory_buffer', 'memory_encoder']) 
                           and p.requires_grad]
    elif args.component == "attention":
        optimizer_params = [p for n, p in model.named_parameters() 
                           if any(x in n for x in ['attention', 'query', 'key', 'value']) 
                           and p.requires_grad]
    elif args.component == "introspection":
        optimizer_params = [p for n, p in model.named_parameters() 
                           if any(x in n for x in ['confidence', 'introspection']) 
                           and p.requires_grad]
    elif args.component == "regime":
        optimizer_params = [p for n, p in model.named_parameters() 
                           if any(x in n for x in ['regime', 'market_state']) 
                           and p.requires_grad]
    else:
        raise ValueError(f"Unknown component: {args.component}")
    
    optimizer = torch.optim.Adam(optimizer_params, lr=args.learning_rate)
    
    # Create component directory
    component_dir = os.path.join(args.output_dir, args.component)
    checkpoint_dir = os.path.join(component_dir, "checkpoints")
    log_dir = os.path.join(component_dir, "logs")
    
    os.makedirs(component_dir, exist_ok=True)
    
    # Train component
    if args.component == "memory":
        results = train_memory_component(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            device=device,
            num_epochs=args.epochs,
            checkpoint_dir=checkpoint_dir,
            log_dir=log_dir
        )
    elif args.component == "attention":
        results = train_attention_component(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            device=device,
            num_epochs=args.epochs,
            checkpoint_dir=checkpoint_dir,
            log_dir=log_dir
        )
    else:
        print(f"Training for {args.component} component not yet implemented")
        return
    
    # Save results
    with open(os.path.join(component_dir, "training_results.json"), 'w') as f:
        # Convert tensors to lists
        serializable_results = {}
        for k, v in results.items():
            if k == 'best_model_state':
                continue  # Skip model state dict
            if isinstance(v, list) and all(isinstance(x, dict) for x in v):
                serializable_results[k] = v
            elif isinstance(v, (list, np.ndarray)):
                serializable_results[k] = [float(x) if isinstance(x, (np.number, torch.Tensor)) else x for x in v]
            else:
                serializable_results[k] = float(v) if isinstance(v, (np.number, torch.Tensor)) else v
        
        json.dump(serializable_results, f, indent=2)
    
    print(f"Component training completed. Results saved to {component_dir}")

if __name__ == "__main__":
    main()
