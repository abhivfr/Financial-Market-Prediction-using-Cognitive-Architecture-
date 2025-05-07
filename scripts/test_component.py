#!/usr/bin/env python
# test_component.py - Test and visualize individual components

import os
import sys
import argparse
import torch
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, root_dir)

from src.arch.cognitive import CognitiveArchitecture
from src.data.financial_loader import EnhancedFinancialDataLoader
from src.monitoring.introspect import ModelIntrospector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_test_loader(data_path, sequence_length=20, batch_size=32, regime_aware=True):
    """Create data loader for testing"""
    return EnhancedFinancialDataLoader(
        data_path=data_path,
        sequence_length=sequence_length,
        batch_size=batch_size,
        regime_aware=regime_aware,
        augmentation=False
    )

def test_memory_component(model, test_loader, device, output_dir):
    """Test and analyze memory component"""
    logger.info("Testing memory component")
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize metrics
    memory_metrics = {
        'retrieval_score': [],
        'memory_utilization': [],
        'memory_diversity': []
    }
    
    # Sample data for visualization
    sample_data = []
    
    # Create introspector
    introspector = ModelIntrospector(model)
    
    # Test with different sequence patterns
    model.eval()
    with torch.no_grad():
        for batch_idx, (financial_data, financial_seq, targets) in enumerate(tqdm(test_loader, desc="Testing memory")):
            # Move data to device
            financial_data = financial_data.to(device)
            financial_seq = financial_seq.to(device)
            
            # Get model outputs
            outputs = model(financial_data=financial_data, financial_seq=financial_seq)
            
            # Get introspection data
            introspection = introspector.introspect_model(
                financial_data=financial_data,
                financial_seq=financial_seq
            )
            
            # Extract memory metrics
            if 'memory_retrieval_score' in introspection:
                memory_metrics['retrieval_score'].append(
                    introspection['memory_retrieval_score'].cpu().numpy()
                )
            
            if 'memory_usage' in introspection:
                memory_usage = introspection['memory_usage'].cpu().numpy()
                memory_metrics['memory_utilization'].append(np.mean(memory_usage, axis=1))
                memory_metrics['memory_diversity'].append(np.std(memory_usage, axis=1))
            
            # Store sample data for visualization (maximum 5 batches)
            if batch_idx < 5:
                sample_data.append({
                    'financial_data': financial_data.cpu().numpy(),
                    'financial_seq': financial_seq.cpu().numpy(),
                    'memory_retrieval_score': introspection.get('memory_retrieval_score', torch.zeros(1)).cpu().numpy(),
                    'memory_usage': introspection.get('memory_usage', torch.zeros(1)).cpu().numpy()
                })
    
    # Calculate average metrics
    avg_metrics = {}
    for metric, values in memory_metrics.items():
        if values:
            # Concatenate all batches
            all_values = np.concatenate(values)
            avg_metrics[metric] = float(np.mean(all_values))
    
    # Save metrics
    metrics_path = os.path.join(output_dir, 'memory_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(avg_metrics, f, indent=2)
    
    logger.info(f"Memory metrics: {avg_metrics}")
    
    # Create visualizations
    create_memory_visualizations(sample_data, output_dir)
    
    return avg_metrics

def create_memory_visualizations(sample_data, output_dir):
    """Create visualizations for memory component"""
    viz_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    
    # Create memory usage visualization
    plt.figure(figsize=(15, 10))
    
    # Plot memory usage patterns for each sample
    for i, sample in enumerate(sample_data):
        if 'memory_usage' in sample and sample['memory_usage'].size > 1:
            plt.subplot(len(sample_data), 1, i + 1)
            
            memory_usage = sample['memory_usage'][0]  # First item in batch
            plt.bar(range(len(memory_usage)), memory_usage)
            
            plt.title(f"Sample {i+1} Memory Usage")
            plt.xlabel("Memory Index")
            plt.ylabel("Usage")
            plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'memory_usage.png'))
    plt.close()
    
    # Create memory retrieval visualization
    if all('memory_retrieval_score' in sample for sample in sample_data):
        plt.figure(figsize=(10, 6))
        
        retrieval_scores = [sample['memory_retrieval_score'][0] for sample in sample_data]
        plt.bar(range(len(retrieval_scores)), retrieval_scores)
        
        plt.title("Memory Retrieval Scores")
        plt.xlabel("Sample Index")
        plt.ylabel("Retrieval Score")
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'memory_retrieval.png'))
        plt.close()

def test_attention_component(model, test_loader, device, output_dir):
    """Test and analyze attention component"""
    logger.info("Testing attention component")
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize metrics
    attention_metrics = {
        'attention_entropy': [],
        'attention_focus': []
    }
    
    # Sample data for visualization
    sample_data = []
    
    # Create introspector
    introspector = ModelIntrospector(model)
    
    # Test with different sequence patterns
    model.eval()
    with torch.no_grad():
        for batch_idx, (financial_data, financial_seq, targets) in enumerate(tqdm(test_loader, desc="Testing attention")):
            # Move data to device
            financial_data = financial_data.to(device)
            financial_seq = financial_seq.to(device)
            
            # Get model outputs
            outputs = model(financial_data=financial_data, financial_seq=financial_seq)
            
            # Get introspection data
            introspection = introspector.introspect_model(
                financial_data=financial_data,
                financial_seq=financial_seq
            )
            
            # Extract attention metrics
            if 'attention_weights' in introspection:
                weights = introspection['attention_weights'].cpu().numpy()
                
                # Calculate entropy
                entropy = -np.sum(weights * np.log(weights + 1e-10), axis=-1)
                attention_metrics['attention_entropy'].append(entropy)
                
                # Calculate focus (max weight)
                focus = np.max(weights, axis=-1)
                attention_metrics['attention_focus'].append(focus)
            
            # Store sample data for visualization (maximum 5 batches)
            if batch_idx < 5:
                sample_data.append({
                    'financial_data': financial_data.cpu().numpy(),
                    'financial_seq': financial_seq.cpu().numpy(),
                    'attention_weights': introspection.get('attention_weights', torch.zeros(1)).cpu().numpy()
                })
    
    # Calculate average metrics
    avg_metrics = {}
    for metric, values in attention_metrics.items():
        if values:
            # Concatenate all batches
            all_values = np.concatenate(values)
            avg_metrics[metric] = float(np.mean(all_values))
    
    # Save metrics
    metrics_path = os.path.join(output_dir, 'attention_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(avg_metrics, f, indent=2)
    
    logger.info(f"Attention metrics: {avg_metrics}")
    
    # Create visualizations
    create_attention_visualizations(sample_data, output_dir)
    
    return avg_metrics

def create_attention_visualizations(sample_data, output_dir):
    """Create visualizations for attention component"""
    viz_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    
    # Create attention weights visualization
    for i, sample in enumerate(sample_data):
        if 'attention_weights' in sample and sample['attention_weights'].size > 1:
            plt.figure(figsize=(12, 8))
            
            # Get attention weights for first item in batch
            attention = sample['attention_weights'][0]
            
            # Handle different attention shapes
            if attention.ndim == 1:
                # 1D attention
                plt.bar(range(len(attention)), attention)
                plt.title(f"Sample {i+1} Attention Weights")
                plt.xlabel("Position")
                plt.ylabel("Weight")
                
            elif attention.ndim == 2:
                # 2D attention (e.g., multi-head attention)
                plt.imshow(attention, cmap='viridis', aspect='auto')
                plt.colorbar(label='Attention Weight')
                plt.title(f"Sample {i+1} Attention Matrix")
                plt.xlabel("Key Position")
                plt.ylabel("Query Position")
                
            else:
                # Higher dimensional attention, flatten to 2D
                attention_flat = attention.reshape(attention.shape[0], -1)
                plt.imshow(attention_flat, cmap='viridis', aspect='auto')
                plt.colorbar(label='Attention Weight')
                plt.title(f"Sample {i+1} Attention Matrix (Flattened)")
                plt.xlabel("Key Position (Flattened)")
                plt.ylabel("Query Position")
            
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, f'attention_weights_{i+1}.png'))
            plt.close()

def test_regime_component(model, test_loader, device, output_dir):
    """Test and analyze regime detection component"""
    logger.info("Testing regime detection component")
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize metrics
    regime_metrics = {
        'regime_confidence': [],
        'regime_stability': []
    }
    
    # Sample data for visualization
    sample_data = []
    
    # Create introspector
    introspector = ModelIntrospector(model)
    
    # Test with different sequence patterns
    model.eval()
    with torch.no_grad():
        for batch_idx, (financial_data, financial_seq, targets) in enumerate(tqdm(test_loader, desc="Testing regime detection")):
            # Move data to device
            financial_data = financial_data.to(device)
            financial_seq = financial_seq.to(device)
            
            # Get model outputs
            outputs = model(financial_data=financial_data, financial_seq=financial_seq)
            
            # Get introspection data
            introspection = introspector.introspect_model(
                financial_data=financial_data,
                financial_seq=financial_seq
            )
            
            # Extract regime metrics
            if 'regime_probs' in introspection:
                probs = introspection['regime_probs'].cpu().numpy()
                
                # Calculate confidence (max probability)
                confidence = np.max(probs, axis=-1)
                regime_metrics['regime_confidence'].append(confidence)
                
                # Calculate stability
                if batch_idx > 0 and 'prev_regime_probs' in locals():
                    stability = np.sum(probs * prev_regime_probs, axis=-1)
                    regime_metrics['regime_stability'].append(stability)
                
                prev_regime_probs = probs
            
            # Store sample data for visualization (maximum 5 batches)
            if batch_idx < 5:
                sample_data.append({
                    'financial_data': financial_data.cpu().numpy(),
                    'financial_seq': financial_seq.cpu().numpy(),
                    'regime_probs': introspection.get('regime_probs', torch.zeros(1)).cpu().numpy(),
                    'regime_state': outputs.get('regime_state', torch.zeros(1)).cpu().numpy() if isinstance(outputs, dict) else None
                })
    
    # Calculate average metrics
    avg_metrics = {}
    for metric, values in regime_metrics.items():
        if values:
            # Concatenate all batches
            all_values = np.concatenate(values)
            avg_metrics[metric] = float(np.mean(all_values))
    
    # Save metrics
    metrics_path = os.path.join(output_dir, 'regime_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(avg_metrics, f, indent=2)
    
    logger.info(f"Regime metrics: {avg_metrics}")
    
    # Create visualizations
    create_regime_visualizations(sample_data, output_dir)
    
    return avg_metrics

def create_regime_visualizations(sample_data, output_dir):
    """Create visualizations for regime detection component"""
    viz_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    
    # Create regime probability visualization
    for i, sample in enumerate(sample_data):
        if 'regime_probs' in sample and sample['regime_probs'].size > 1:
            plt.figure(figsize=(12, 8))
            
            # Get regime probabilities for first item in batch
            regime_probs = sample['regime_probs'][0]
            
            # Plot probabilities
            plt.bar(range(len(regime_probs)), regime_probs)
            plt.title(f"Sample {i+1} Regime Probabilities")
            plt.xlabel("Regime")
            plt.ylabel("Probability")
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, f'regime_probs_{i+1}.png'))
            plt.close()

def main():
    parser = argparse.ArgumentParser(description="Test and visualize individual components")
    
    # Required arguments
    parser.add_argument("--component", required=True, choices=["memory", "attention", "regime"],
                      help="Component to test")
    parser.add_argument("--model_path", required=True, help="Path to model weights")
    parser.add_argument("--test_data", required=True, help="Path to test data")
    
    # Optional arguments
    parser.add_argument("--output_dir", default="evaluation/components", help="Output directory")
    parser.add_argument("--sequence_length", type=int, default=20, help="Sequence length")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Computation device")
    
    args = parser.parse_args()
    
    # Create output directory
    component_dir = os.path.join(args.output_dir, args.component)
    os.makedirs(component_dir, exist_ok=True)
    
    # Create model
    model = CognitiveArchitecture()
    
    # Load weights
    model.load_state_dict(torch.load(args.model_path, map_location=args.device))
    model.to(args.device)
    
    # Create test loader
    test_loader = create_test_loader(
        args.test_data,
        args.sequence_length,
        args.batch_size
    )
    
    # Test component
    if args.component == "memory":
        metrics = test_memory_component(model, test_loader, args.device, component_dir)
    elif args.component == "attention":
        metrics = test_attention_component(model, test_loader, args.device, component_dir)
    elif args.component == "regime":
        metrics = test_regime_component(model, test_loader, args.device, component_dir)
    else:
        logger.error(f"Unknown component: {args.component}")
        return
    
    logger.info(f"Component testing completed. Results saved to {component_dir}")

if __name__ == "__main__":
    main()
