#!/usr/bin/env python
# stress_test.py - Stress testing for cognitive models

import os
import sys
import argparse
import torch
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Union, Tuple
from tqdm import tqdm
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from src.arch.cognitive import CognitiveArchitecture
from src.arch.baseline_lstm import FinancialLSTMBaseline
from src.data.financial_loader import EnhancedFinancialDataLoader

class StressTestScenario:
    """Base class for stress test scenarios"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data for the scenario
        
        Args:
            data: Original data
            
        Returns:
            Modified data for the scenario
        """
        return data
    
    def evaluate(self, model: torch.nn.Module, data: pd.DataFrame, device: str) -> Dict[str, Any]:
        """
        Evaluate model on scenario
        
        Args:
            model: PyTorch model
            data: Data for evaluation
            device: Computation device
            
        Returns:
            Evaluation metrics
        """
        raise NotImplementedError("Subclasses must implement evaluate()")

class MarketCrashScenario(StressTestScenario):
    """Simulate a market crash scenario"""
    
    def __init__(self, crash_magnitude: float = -0.1, crash_duration: int = 5):
        super().__init__(
            name="market_crash",
            description=f"Market crash with {crash_magnitude*100:.1f}% drop over {crash_duration} days"
        )
        self.crash_magnitude = crash_magnitude
        self.crash_duration = crash_duration
    
    def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare market crash scenario data"""
        result = data.copy()
        
        # Find suitable location for crash (at least 20% into the data)
        min_index = int(len(result) * 0.2)
        crash_start = min_index + np.random.randint(0, len(result) - min_index - self.crash_duration)
        
        # Apply crash pattern to returns
        if 'return_1d' in result.columns:
            # Original returns
            original_returns = result['return_1d'].values.copy()
            
            # Create crash pattern - negative returns for crash_duration days
            crash_returns = np.linspace(0, self.crash_magnitude, self.crash_duration)
            
            # Apply crash pattern
            result.loc[crash_start:crash_start+self.crash_duration-1, 'return_1d'] = crash_returns
            
            # Adjust price-based columns based on new returns
            self._adjust_prices(result, crash_start, original_returns)
            
            # Mark crash period
            result['crash_period'] = False
            result.loc[crash_start:crash_start+self.crash_duration-1, 'crash_period'] = True
        
        return result
    
    def _adjust_prices(self, data: pd.DataFrame, crash_start: int, original_returns: np.ndarray):
        """Adjust price-based columns"""
        # Adjust close prices based on new returns
        if 'close' in data.columns:
            for i in range(crash_start, crash_start + self.crash_duration):
                if i > 0:
                    data.loc[i, 'close'] = data.loc[i-1, 'close'] * (1 + data.loc[i, 'return_1d'])

    def evaluate(self, model: torch.nn.Module, data: pd.DataFrame, device: str) -> Dict[str, Any]:
        """Evaluate model on market crash scenario"""
        # Prepare data
        stress_data = self.prepare_data(data)
        
        # Extract features and target
        feature_cols = [col for col in stress_data.columns if col not in ['date', 'target', 'crash_period']]
        
        if 'target' in stress_data.columns:
            target_col = 'target'
        elif 'return_1d' in stress_data.columns:
            # Use next day's return as target
            stress_data['target'] = stress_data['return_1d'].shift(-1)
            target_col = 'target'
        else:
            raise ValueError("No target column found in data")
        
        # Create data loader
        sequence_length = 20  # Default sequence length
        
        # Filter out non-numeric columns
        numeric_cols = []
        for col in feature_cols:
            try:
                pd.to_numeric(stress_data[col])
                numeric_cols.append(col)
            except:
                print(f"Skipping non-numeric column: {col}")
                
        feature_cols = numeric_cols
        
        # Select most relevant features if there are too many
        if len(feature_cols) > 20:
            # Prioritize basic price and volatility features
            priority_features = ['price', 'volume', 'returns', 'volatility', 'momentum_5d', 'momentum_10d', 'rsi_14']
            selected_features = []
            
            # First include priority features
            for prefix in priority_features:
                for col in feature_cols:
                    if col.startswith(prefix) and col not in selected_features:
                        selected_features.append(col)
                        if len(selected_features) >= 7:  # Limit to 7 features
                            break
                            
            feature_cols = selected_features[:7]  # Take at most 7 features
            
        if len(feature_cols) == 0:
            raise ValueError("No numeric feature columns found in data")
        
        # Normalize features
        features = stress_data[feature_cols].values
        
        # Convert to numeric safely
        try:
            features = features.astype(np.float32)
        except ValueError:
            # Handle conversion errors by converting each column individually
            numeric_features = np.zeros((features.shape[0], len(feature_cols)), dtype=np.float32)
            for i, col in enumerate(feature_cols):
                try:
                    numeric_features[:, i] = stress_data[col].astype(np.float32)
                except:
                    # Fill with zeros if conversion fails
                    print(f"Failed to convert {col} to numeric, using zeros instead")
            features = numeric_features
        
        # Handle any remaining non-numeric values
        features = np.where(np.isnan(features), 0, features)
        
        feature_means = np.nanmean(features, axis=0)
        feature_stds = np.nanstd(features, axis=0)
        feature_stds[feature_stds == 0] = 1.0
        normalized_features = (features - feature_means) / feature_stds
        normalized_features = np.nan_to_num(normalized_features)
        
        # Get targets
        targets = stress_data[target_col].values
        
        # Prepare model
        model.to(device)
        model.eval()
        
        # Evaluate model
        predictions = []
        actual_values = []
        is_crash_period = []
        
        with torch.no_grad():
            for i in range(sequence_length, len(normalized_features)):
                # Get sequence and current features
                sequence = normalized_features[i-sequence_length:i]
                current_features = normalized_features[i]
                
                # Create tensors
                feature_tensor = torch.tensor(current_features, dtype=torch.float32).unsqueeze(0).to(device)
                sequence_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(device)
                
                # Forward pass - handle different model interfaces
                try:
                    # Try cognitive architecture interface first
                    outputs = model(financial_data=feature_tensor, financial_seq=sequence_tensor)
                    
                    # If outputs is a dict, extract the relevant prediction
                    if isinstance(outputs, dict) and 'market_state' in outputs:
                        # Get the prediction - could be multi-dimensional
                        pred = outputs['market_state'].cpu().numpy()
                        # If prediction has shape [batch, seq, features], take the first timestep
                        if len(pred.shape) > 2:
                            prediction = pred[0, 0, 0]  # Take first batch, first timestep, first feature
                        else:
                            prediction = pred[0, 0]  # Take first batch, first feature
                    else:
                        # Handle tensor output - could be multi-dimensional
                        pred = outputs.cpu().numpy()
                        if len(pred.shape) > 2:
                            prediction = pred[0, 0, 0]  # Take first batch, first timestep, first feature
                        else:
                            prediction = pred[0, 0]  # Take first batch, first feature
                except TypeError:
                    # Fall back to standard interface for baseline model
                    outputs = model(sequence_tensor)
                    prediction = outputs.cpu().numpy()[0, 0]
                
                # Store results
                predictions.append(prediction)
                actual_values.append(targets[i])
                is_crash_period.append(stress_data['crash_period'].iloc[i])
        
        # Convert to numpy arrays
        predictions = np.array(predictions)
        actual_values = np.array(actual_values)
        is_crash_period = np.array(is_crash_period)
        
        # Calculate metrics
        crash_mask = is_crash_period
        non_crash_mask = ~is_crash_period
        
        # Overall metrics
        mse = np.mean((predictions - actual_values) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - actual_values))
        direction_accuracy = np.mean(np.sign(predictions) == np.sign(actual_values))
        
        # Crash period metrics
        crash_mse = np.mean((predictions[crash_mask] - actual_values[crash_mask]) ** 2) if np.any(crash_mask) else np.nan
        crash_rmse = np.sqrt(crash_mse) if not np.isnan(crash_mse) else np.nan
        crash_direction_accuracy = np.mean(np.sign(predictions[crash_mask]) == np.sign(actual_values[crash_mask])) if np.any(crash_mask) else np.nan
        
        # Non-crash period metrics
        non_crash_mse = np.mean((predictions[non_crash_mask] - actual_values[non_crash_mask]) ** 2) if np.any(non_crash_mask) else np.nan
        non_crash_rmse = np.sqrt(non_crash_mse) if not np.isnan(non_crash_mse) else np.nan
        non_crash_direction_accuracy = np.mean(np.sign(predictions[non_crash_mask]) == np.sign(actual_values[non_crash_mask])) if np.any(non_crash_mask) else np.nan
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        
        # Plot returns and predictions
        plt.subplot(2, 1, 1)
        plt.plot(actual_values, label='Actual Returns')
        plt.plot(predictions, label='Predicted Returns')
        
        # Highlight crash period
        if np.any(crash_mask):
            crash_indices = np.where(crash_mask)[0]
            plt.axvspan(min(crash_indices), max(crash_indices), alpha=0.2, color='red', label='Crash Period')
        
        plt.xlabel('Time')
        plt.ylabel('Returns')
        plt.legend()
        plt.title(f'Market Crash Scenario: Actual vs Predicted Returns\nDirection Accuracy: Overall={direction_accuracy:.4f}, During Crash={crash_direction_accuracy:.4f}')
        
        # Plot error
        plt.subplot(2, 1, 2)
        error = np.abs(predictions - actual_values)
        plt.plot(error, label='Absolute Error')
        
        # Highlight crash period
        if np.any(crash_mask):
            crash_indices = np.where(crash_mask)[0]
            plt.axvspan(min(crash_indices), max(crash_indices), alpha=0.2, color='red', label='Crash Period')
        
        plt.xlabel('Time')
        plt.ylabel('Absolute Error')
        plt.legend()
        plt.title(f'Prediction Error: Overall RMSE={rmse:.4f}, During Crash={crash_rmse:.4f}, Non-Crash={non_crash_rmse:.4f}')
        
        plt.tight_layout()
        
        # Return metrics
        metrics = {
            'overall': {
                'mse': float(mse),
                'rmse': float(rmse),
                'mae': float(mae),
                'direction_accuracy': float(direction_accuracy)
            },
            'crash_period': {
                'mse': float(crash_mse) if not np.isnan(crash_mse) else None,
                'rmse': float(crash_rmse) if not np.isnan(crash_rmse) else None,
                'direction_accuracy': float(crash_direction_accuracy) if not np.isnan(crash_direction_accuracy) else None,
                'samples': int(np.sum(crash_mask))
            },
            'non_crash_period': {
                'mse': float(non_crash_mse) if not np.isnan(non_crash_mse) else None,
                'rmse': float(non_crash_rmse) if not np.isnan(non_crash_rmse) else None,
                'direction_accuracy': float(non_crash_direction_accuracy) if not np.isnan(non_crash_direction_accuracy) else None,
                'samples': int(np.sum(non_crash_mask))
            }
        }
        
        return metrics, plt.gcf()

class HighVolatilityScenario(StressTestScenario):
    """Simulate a high volatility scenario"""
    
    def __init__(self, volatility_multiplier: float = 2.5, duration: int = 10):
        super().__init__(
            name="high_volatility",
            description=f"High volatility period with {volatility_multiplier}x multiplier over {duration} days"
        )
        self.volatility_multiplier = volatility_multiplier
        self.duration = duration
    
    def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare high volatility scenario data"""
        result = data.copy()
        
        # Find suitable location for high volatility (at least 20% into the data)
        min_index = int(len(result) * 0.2)
        vol_start = min_index + np.random.randint(0, len(result) - min_index - self.duration)
        
        # Apply high volatility to returns
        if 'return_1d' in result.columns:
            # Original returns
            original_returns = result['return_1d'].values.copy()
            
            # Amplify returns during high volatility period
            for i in range(vol_start, vol_start + self.duration):
                result.loc[i, 'return_1d'] = original_returns[i] * self.volatility_multiplier
            
            # Adjust price-based columns
            self._adjust_prices(result, vol_start, original_returns)
            
            # Mark high volatility period
            result['high_vol_period'] = False
            result.loc[vol_start:vol_start+self.duration-1, 'high_vol_period'] = True
            
            # Recalculate volatility indicators if present
            if 'volatility_10d' in result.columns:
                # Recalculate rolling volatility
                result['volatility_10d'] = result['return_1d'].rolling(10).std() * (252 ** 0.5)
            
            if 'volatility_20d' in result.columns:
                result['volatility_20d'] = result['return_1d'].rolling(20).std() * (252 ** 0.5)
        
        return result
    
    def _adjust_prices(self, data: pd.DataFrame, vol_start: int, original_returns: np.ndarray):
        """Adjust price-based columns"""
        # Adjust close prices based on new returns
        if 'close' in data.columns:
            for i in range(vol_start, vol_start + self.duration):
                if i > 0:
                    data.loc[i, 'close'] = data.loc[i-1, 'close'] * (1 + data.loc[i, 'return_1d'])
            
            # Adjust other price columns to maintain their relative relationships
            if all(col in data.columns for col in ['open', 'high', 'low']):
                for i in range(vol_start, vol_start + self.duration):
                    if i > 0:
                        # Calculate price range multiplier based on volatility multiplier
                        range_multiplier = np.sqrt(self.volatility_multiplier)
                        
                        # Base price is close from previous day
                        base_price = data.loc[i-1, 'close']
                        
                        # New close price
                        close_price = data.loc[i, 'close']
                        
                        # Original high and low relative to previous close
                        orig_high_rel = data.loc[i, 'high'] / close_price
                        orig_low_rel = data.loc[i, 'low'] / close_price
                        
                        # Amplify the range
                        new_high_rel = 1 + (orig_high_rel - 1) * range_multiplier
                        new_low_rel = 1 - (1 - orig_low_rel) * range_multiplier
                        
                        # Set new high and low
                        data.loc[i, 'high'] = close_price * new_high_rel
                        data.loc[i, 'low'] = close_price * new_low_rel
    
    def evaluate(self, model: torch.nn.Module, data: pd.DataFrame, device: str) -> Dict[str, Any]:
        """Evaluate model on high volatility scenario"""
        # Prepare data
        stress_data = self.prepare_data(data)
        
        # Extract features and target
        feature_cols = [col for col in stress_data.columns if col not in ['date', 'target', 'high_vol_period']]
        
        if 'target' in stress_data.columns:
            target_col = 'target'
        elif 'return_1d' in stress_data.columns:
            # Use next day's return as target
            stress_data['target'] = stress_data['return_1d'].shift(-1)
            target_col = 'target'
        else:
            raise ValueError("No target column found in data")
        
        # Create data loader
        sequence_length = 20  # Default sequence length
        
        # Filter out non-numeric columns
        numeric_cols = []
        for col in feature_cols:
            try:
                pd.to_numeric(stress_data[col])
                numeric_cols.append(col)
            except:
                print(f"Skipping non-numeric column: {col}")
                
        feature_cols = numeric_cols
        
        # Select most relevant features if there are too many
        if len(feature_cols) > 20:
            # Prioritize basic price and volatility features
            priority_features = ['price', 'volume', 'returns', 'volatility', 'momentum_5d', 'momentum_10d', 'rsi_14']
            selected_features = []
            
            # First include priority features
            for prefix in priority_features:
                for col in feature_cols:
                    if col.startswith(prefix) and col not in selected_features:
                        selected_features.append(col)
                        if len(selected_features) >= 7:  # Limit to 7 features
                            break
                            
            feature_cols = selected_features[:7]  # Take at most 7 features
            
        if len(feature_cols) == 0:
            raise ValueError("No numeric feature columns found in data")
        
        # Normalize features
        features = stress_data[feature_cols].values
        
        # Convert to numeric safely
        try:
            features = features.astype(np.float32)
        except ValueError:
            # Handle conversion errors by converting each column individually
            numeric_features = np.zeros((features.shape[0], len(feature_cols)), dtype=np.float32)
            for i, col in enumerate(feature_cols):
                try:
                    numeric_features[:, i] = stress_data[col].astype(np.float32)
                except:
                    # Fill with zeros if conversion fails
                    print(f"Failed to convert {col} to numeric, using zeros instead")
            features = numeric_features
        
        # Handle any remaining non-numeric values
        features = np.where(np.isnan(features), 0, features)
        
        feature_means = np.nanmean(features, axis=0)
        feature_stds = np.nanstd(features, axis=0)
        feature_stds[feature_stds == 0] = 1.0
        normalized_features = (features - feature_means) / feature_stds
        normalized_features = np.nan_to_num(normalized_features)
        
        # Get targets
        targets = stress_data[target_col].values
        
        # Prepare model
        model.to(device)
        model.eval()
        
        # Evaluate model
        predictions = []
        actual_values = []
        is_high_vol_period = []
        confidences = []  # For cognitive models with confidence estimation
        
        with torch.no_grad():
            for i in range(sequence_length, len(normalized_features)):
                # Get sequence and current features
                sequence = normalized_features[i-sequence_length:i]
                current_features = normalized_features[i]
                
                # Create tensors
                feature_tensor = torch.tensor(current_features, dtype=torch.float32).unsqueeze(0).to(device)
                sequence_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(device)
                
                # Forward pass - handle different model interfaces
                try:
                    # Try cognitive architecture interface first
                    outputs = model(financial_data=feature_tensor, financial_seq=sequence_tensor)
                    
                    # If outputs is a dict, extract the relevant prediction
                    if isinstance(outputs, dict) and 'market_state' in outputs:
                        # Get the prediction - could be multi-dimensional
                        pred = outputs['market_state'].cpu().numpy()
                        # If prediction has shape [batch, seq, features], take the first timestep
                        if len(pred.shape) > 2:
                            prediction = pred[0, 0, 0]  # Take first batch, first timestep, first feature
                        else:
                            prediction = pred[0, 0]  # Take first batch, first feature
                    else:
                        # Handle tensor output - could be multi-dimensional
                        pred = outputs.cpu().numpy()
                        if len(pred.shape) > 2:
                            prediction = pred[0, 0, 0]  # Take first batch, first timestep, first feature
                        else:
                            prediction = pred[0, 0]  # Take first batch, first feature
                except TypeError:
                    # Fall back to standard interface for baseline model
                    outputs = model(sequence_tensor)
                    prediction = outputs.cpu().numpy()[0, 0]
                
                # Store results
                predictions.append(prediction)
                actual_values.append(targets[i])
                is_high_vol_period.append(stress_data['high_vol_period'].iloc[i])
                confidences.append(1.0)  # Default confidence for baseline model
        
        # Convert to numpy arrays
        predictions = np.array(predictions)
        actual_values = np.array(actual_values)
        is_high_vol_period = np.array(is_high_vol_period)
        confidences = np.array(confidences)
        
        # Calculate metrics
        high_vol_mask = is_high_vol_period
        normal_vol_mask = ~is_high_vol_period
        
        # Overall metrics
        mse = np.mean((predictions - actual_values) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - actual_values))
        direction_accuracy = np.mean(np.sign(predictions) == np.sign(actual_values))
        
        # High volatility period metrics
        high_vol_mse = np.mean((predictions[high_vol_mask] - actual_values[high_vol_mask]) ** 2) if np.any(high_vol_mask) else np.nan
        high_vol_rmse = np.sqrt(high_vol_mse) if not np.isnan(high_vol_mse) else np.nan
        high_vol_direction_accuracy = np.mean(np.sign(predictions[high_vol_mask]) == np.sign(actual_values[high_vol_mask])) if np.any(high_vol_mask) else np.nan
        
        # Normal volatility period metrics
        normal_vol_mse = np.mean((predictions[normal_vol_mask] - actual_values[normal_vol_mask]) ** 2) if np.any(normal_vol_mask) else np.nan
        normal_vol_rmse = np.sqrt(normal_vol_mse) if not np.isnan(normal_vol_mse) else np.nan
        normal_vol_direction_accuracy = np.mean(np.sign(predictions[normal_vol_mask]) == np.sign(actual_values[normal_vol_mask])) if np.any(normal_vol_mask) else np.nan
        
        # Calculate volatility adaptation metric
        volatility_adaptation = np.nan
        if np.any(high_vol_mask) and np.any(normal_vol_mask):
            # Correlation between prediction error and volatility
            error = np.abs(predictions - actual_values)
            volatility_adaptation = 1.0 - (np.mean(error[high_vol_mask]) / np.mean(error[normal_vol_mask]))
            
            # Normalize to range [0, 1] where 1 means perfect adaptation (same error in both regimes)
            volatility_adaptation = max(0, min(1, volatility_adaptation + 1.0))
        
        # Calculate confidence calibration for cognitive models
        confidence_correlation = np.nan
        if not np.all(confidences == 1.0):  # If we have actual confidence values
            error = np.abs(predictions - actual_values)
            confidence_correlation = -np.corrcoef(error, confidences)[0, 1]  # Negative correlation is better
        
        # Create visualization
        plt.figure(figsize=(15, 10))
        
        # Plot returns and predictions
        plt.subplot(3, 1, 1)
        plt.plot(actual_values, label='Actual Returns')
        plt.plot(predictions, label='Predicted Returns')
        
        # Highlight high volatility period
        if np.any(high_vol_mask):
            high_vol_indices = np.where(high_vol_mask)[0]
            plt.axvspan(min(high_vol_indices), max(high_vol_indices), alpha=0.2, color='orange', label='High Volatility')
        
        plt.xlabel('Time')
        plt.ylabel('Returns')
        plt.legend()
        plt.title(f'High Volatility Scenario: Actual vs Predicted Returns\nDirection Accuracy: Overall={direction_accuracy:.4f}, High Vol={high_vol_direction_accuracy:.4f}')
        
        # Plot error
        plt.subplot(3, 1, 2)
        error = np.abs(predictions - actual_values)
        plt.plot(error, label='Absolute Error')
        
        # Highlight high volatility period
        if np.any(high_vol_mask):
            high_vol_indices = np.where(high_vol_mask)[0]
            plt.axvspan(min(high_vol_indices), max(high_vol_indices), alpha=0.2, color='orange', label='High Volatility')
        
        plt.xlabel('Time')
        plt.ylabel('Absolute Error')
        plt.legend()
        plt.title(f'Prediction Error: Overall RMSE={rmse:.4f}, High Vol={high_vol_rmse:.4f}, Normal Vol={normal_vol_rmse:.4f}')
        
        # Plot confidence if available
        plt.subplot(3, 1, 3)
        if not np.all(confidences == 1.0):
            plt.plot(confidences, label='Model Confidence')
            plt.plot(1.0 - error/np.max(error), label='Normalized Accuracy')
            
            # Highlight high volatility period
            if np.any(high_vol_mask):
                high_vol_indices = np.where(high_vol_mask)[0]
                plt.axvspan(min(high_vol_indices), max(high_vol_indices), alpha=0.2, color='orange', label='High Volatility')
            
            plt.xlabel('Time')
            plt.ylabel('Confidence / Accuracy')
            plt.legend()
            plt.title(f'Model Confidence vs Accuracy: Correlation={confidence_correlation:.4f}')
        else:
            plt.text(0.5, 0.5, 'No confidence data available', horizontalalignment='center', verticalalignment='center')
        
        plt.tight_layout()
        
        # Return metrics
        metrics = {
            'overall': {
                'mse': float(mse),
                'rmse': float(rmse),
                'mae': float(mae),
                'direction_accuracy': float(direction_accuracy)
            },
            'high_volatility_period': {
                'mse': float(high_vol_mse) if not np.isnan(high_vol_mse) else None,
                'rmse': float(high_vol_rmse) if not np.isnan(high_vol_rmse) else None,
                'direction_accuracy': float(high_vol_direction_accuracy) if not np.isnan(high_vol_direction_accuracy) else None,
                'samples': int(np.sum(high_vol_mask))
            },
            'normal_volatility_period': {
                'mse': float(normal_vol_mse) if not np.isnan(normal_vol_mse) else None,
                'rmse': float(normal_vol_rmse) if not np.isnan(normal_vol_rmse) else None,
                'direction_accuracy': float(normal_vol_direction_accuracy) if not np.isnan(normal_vol_direction_accuracy) else None,
                'samples': int(np.sum(normal_vol_mask))
            },
            'volatility_adaptation': float(volatility_adaptation) if not np.isnan(volatility_adaptation) else None,
            'confidence_correlation': float(confidence_correlation) if not np.isnan(confidence_correlation) else None
        }
        
        return metrics, plt.gcf()

class RegimeChangeScenario(StressTestScenario):
    """Simulate a market regime change scenario"""
    
    def __init__(self, from_regime: int = 1, to_regime: int = 2, transition_duration: int = 5):
        super().__init__(
            name="regime_change",
            description=f"Market regime change from {from_regime} to {to_regime} over {transition_duration} days"
        )
        self.from_regime = from_regime
        self.to_regime = to_regime
        self.transition_duration = transition_duration
    
    def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare regime change scenario data"""
        result = data.copy()
        
        # Check if market_regime exists or create it
        if 'market_regime' not in result.columns:
            # Create regime column if it doesn't exist
            # This is a simplified approach - in reality, regimes would be identified by clustering
            if 'volatility_20d' in result.columns and 'return_1d' in result.columns:
                # Create regimes based on volatility and returns
                vol = result['volatility_20d']
                returns = result['return_1d'].rolling(20).mean()
                
                # Normalize
                vol_norm = (vol - vol.mean()) / vol.std()
                returns_norm = (returns - returns.mean()) / returns.std()
                
                # Create regimes
                result['market_regime'] = 0  # Default regime
                result.loc[(vol_norm <= 0) & (returns_norm > 0), 'market_regime'] = 1  # Bull
                result.loc[(vol_norm > 0) & (returns_norm <= 0), 'market_regime'] = 2  # Bear
                result.loc[(vol_norm > 0) & (returns_norm > 0), 'market_regime'] = 3  # Volatile bull
            else:
                # If no suitable columns, create dummy regimes
                result['market_regime'] = 1  # Default to bull regime
        
        # Find segments with the from_regime
        from_regime_mask = result['market_regime'] == self.from_regime
        from_regime_segments = []
        in_segment = False
        segment_start = 0
        
        for i, is_from_regime in enumerate(from_regime_mask):
            if is_from_regime and not in_segment:
                # Start of segment
                in_segment = True
                segment_start = i
            elif not is_from_regime and in_segment:
                # End of segment
                from_regime_segments.append((segment_start, i-1))
                in_segment = False
        
        # Add last segment if still open
        if in_segment:
            from_regime_segments.append((segment_start, len(result)-1))
        
        # Find suitable segment (at least 20 days after start, 20 days before end, and long enough)
        suitable_segments = []
        min_segment_length = self.transition_duration + 40  # 20 days padding on each side
        
        for start, end in from_regime_segments:
            segment_length = end - start + 1
            if start >= 20 and end < len(result) - 20 and segment_length >= min_segment_length:
                suitable_segments.append((start, end))
        
        if not suitable_segments:
            print("Warning: No suitable segments found for regime change")
            return result
        
        # Select a random suitable segment
        segment_idx = np.random.randint(0, len(suitable_segments))
        segment_start, segment_end = suitable_segments[segment_idx]
        
        # Choose transition start point (at least 20 days after segment start)
        transition_start = segment_start + 20 + np.random.randint(0, segment_end - segment_start - 20 - self.transition_duration)
        transition_end = transition_start + self.transition_duration
        
        # Mark transition period
        result['regime_transition'] = False
        result.loc[transition_start:transition_end, 'regime_transition'] = True
        
        # Change regime during and after transition
        result.loc[transition_start:transition_end, 'market_regime'] = -1  # Transition
        result.loc[transition_end+1:, 'market_regime'] = self.to_regime
        
        # Adjust returns based on new regime
        if 'return_1d' in result.columns:
            # Calculate return characteristics for 'to' regime
            to_regime_mask = (result['market_regime'] == self.to_regime)
            if np.any(to_regime_mask):
                to_regime_returns = result.loc[to_regime_mask, 'return_1d']
                to_regime_mean = to_regime_returns.mean()
                to_regime_std = to_regime_returns.std()
                
                # Generate new returns for transition period
                # Start with current returns and gradually shift to new regime characteristics
                for i, idx in enumerate(range(transition_start, transition_end + 1)):
                    # Interpolation factor (0 at start, 1 at end)
                    alpha = i / self.transition_duration
                    
                    # Original return
                    orig_return = result.loc[idx, 'return_1d']
                    
                    # Target return pattern (sample from new regime distribution)
                    target_return = np.random.normal(to_regime_mean, to_regime_std)
                    
                    # Interpolate
                    result.loc[idx, 'return_1d'] = (1 - alpha) * orig_return + alpha * target_return
                
                # Generate new returns after transition
                for idx in range(transition_end + 1, len(result)):
                    result.loc[idx, 'return_1d'] = np.random.normal(to_regime_mean, to_regime_std)
                
                # Adjust price-based columns
                if 'close' in result.columns:
                    # Recalculate close prices based on new returns
                    for idx in range(transition_start, len(result)):
                        if idx > 0:
                            result.loc[idx, 'close'] = result.loc[idx-1, 'close'] * (1 + result.loc[idx, 'return_1d'])
                    
                    # Adjust other price columns to maintain their relative relationships
                    if all(col in result.columns for col in ['open', 'high', 'low']):
                        for idx in range(transition_start, len(result)):
                            if idx > 0:
                                # Calculate relative position of open, high, low to close
                                prev_close = result.loc[idx-1, 'close']
                                curr_close = result.loc[idx, 'close']
                                
                                # Adjust open, high, low to maintain relative position
                                result.loc[idx, 'open'] = curr_close * (result.loc[idx, 'open'] / prev_close)
                                result.loc[idx, 'high'] = curr_close * max(1, result.loc[idx, 'high'] / prev_close)
                                result.loc[idx, 'low'] = curr_close * min(1, result.loc[idx, 'low'] / prev_close)
        
        return result
    
    def evaluate(self, model: torch.nn.Module, data: pd.DataFrame, device: str) -> Dict[str, Any]:
        """Evaluate model on regime change scenario"""
        # Prepare data
        stress_data = self.prepare_data(data)
        
        # Extract features and target
        feature_cols = [col for col in stress_data.columns if col not in ['date', 'target', 'regime_transition']]
        
        if 'target' in stress_data.columns:
            target_col = 'target'
        elif 'return_1d' in stress_data.columns:
            # Use next day's return as target
            stress_data['target'] = stress_data['return_1d'].shift(-1)
            target_col = 'target'
        else:
            raise ValueError("No target column found in data")
        
        # Create data loader
        sequence_length = 20  # Default sequence length
        
        # Filter out non-numeric columns
        numeric_cols = []
        for col in feature_cols:
            try:
                pd.to_numeric(stress_data[col])
                numeric_cols.append(col)
            except:
                print(f"Skipping non-numeric column: {col}")
                
        feature_cols = numeric_cols
        
        # Select most relevant features if there are too many
        if len(feature_cols) > 20:
            # Prioritize basic price and volatility features
            priority_features = ['price', 'volume', 'returns', 'volatility', 'momentum_5d', 'momentum_10d', 'rsi_14']
            selected_features = []
            
            # First include priority features
            for prefix in priority_features:
                for col in feature_cols:
                    if col.startswith(prefix) and col not in selected_features:
                        selected_features.append(col)
                        if len(selected_features) >= 7:  # Limit to 7 features
                            break
                            
            feature_cols = selected_features[:7]  # Take at most 7 features
            
        if len(feature_cols) == 0:
            raise ValueError("No numeric feature columns found in data")
        
        # Normalize features
        features = stress_data[feature_cols].values
        feature_means = np.nanmean(features, axis=0)
        feature_stds = np.nanstd(features, axis=0)
        feature_stds[feature_stds == 0] = 1.0
        normalized_features = (features - feature_means) / feature_stds
        normalized_features = np.nan_to_num(normalized_features)
        
        # Get targets and regimes
        targets = stress_data[target_col].values
        regimes = stress_data['market_regime'].values
        is_transition = stress_data['regime_transition'].values
        
        # Prepare model
        model.to(device)
        model.eval()
        
        # Evaluate model
        predictions = []
        actual_values = []
        regime_values = []
        transition_mask = []
        
        with torch.no_grad():
            for i in range(sequence_length, len(normalized_features)):
                # Get sequence and current features
                sequence = normalized_features[i-sequence_length:i]
                current_features = normalized_features[i]
                
                # Create tensors
                feature_tensor = torch.tensor(current_features, dtype=torch.float32).unsqueeze(0).to(device)
                sequence_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(device)
                
                # Forward pass - handle different model interfaces
                try:
                    # Try cognitive architecture interface first
                    outputs = model(financial_data=feature_tensor, financial_seq=sequence_tensor)
                    
                    # If outputs is a dict, extract the relevant prediction
                    if isinstance(outputs, dict) and 'market_state' in outputs:
                        prediction = outputs['market_state'].cpu().numpy()[0, 0]
                    else:
                        prediction = outputs.cpu().numpy()[0, 0]
                except TypeError:
                    # Fall back to standard interface for baseline model
                    outputs = model(sequence_tensor)
                    prediction = outputs.cpu().numpy()[0, 0]
                
                # Store results
                predictions.append(prediction)
                actual_values.append(targets[i])
                regime_values.append(regimes[i])
                transition_mask.append(is_transition[i])
        
        # Convert to numpy arrays
        predictions = np.array(predictions)
        actual_values = np.array(actual_values)
        regime_values = np.array(regime_values)
        transition_mask = np.array(transition_mask)
        
        # Define masks for different periods
        pre_transition_mask = regime_values == self.from_regime
        during_transition_mask = transition_mask
        post_transition_mask = regime_values == self.to_regime
        
        # Calculate metrics for each period
        metrics = {
            'overall': self._calculate_period_metrics(predictions, actual_values),
            'pre_transition': self._calculate_period_metrics(
                predictions[pre_transition_mask], actual_values[pre_transition_mask]
            ) if np.any(pre_transition_mask) else None,
            'during_transition': self._calculate_period_metrics(
                predictions[during_transition_mask], actual_values[during_transition_mask]
            ) if np.any(during_transition_mask) else None,
            'post_transition': self._calculate_period_metrics(
                predictions[post_transition_mask], actual_values[post_transition_mask]
            ) if np.any(post_transition_mask) else None
        }
        
        # Calculate regime adaptation score
        regime_adaptation = np.nan
        if np.any(pre_transition_mask) and np.any(post_transition_mask):
            # Pre-transition error
            pre_error = np.mean(np.abs(
                predictions[pre_transition_mask] - actual_values[pre_transition_mask]
            ))
            
            # Post-transition error
            post_error = np.mean(np.abs(
                predictions[post_transition_mask] - actual_values[post_transition_mask]
            ))
            
            # Adaptation score (1 = perfect adaptation, 0 = no adaptation)
            max_error_ratio = 3.0  # Limit the ratio to avoid extreme values
            error_ratio = min(max_error_ratio, post_error / pre_error)
            regime_adaptation = 1.0 - (error_ratio - 1.0) / (max_error_ratio - 1.0)
            regime_adaptation = max(0.0, regime_adaptation)
        
        metrics['regime_adaptation'] = float(regime_adaptation) if not np.isnan(regime_adaptation) else None
        
        # Create visualization
        plt.figure(figsize=(15, 10))
        
        # Plot returns and predictions
        plt.subplot(2, 1, 1)
        plt.plot(actual_values, label='Actual Returns')
        plt.plot(predictions, label='Predicted Returns')
        
        # Highlight different periods
        if np.any(pre_transition_mask):
            pre_indices = np.where(pre_transition_mask)[0]
            if len(pre_indices) > 0:
                plt.axvspan(0, max(pre_indices), alpha=0.1, color='green', label=f'Regime {self.from_regime}')
        
        if np.any(during_transition_mask):
            trans_indices = np.where(during_transition_mask)[0]
            if len(trans_indices) > 0:
                plt.axvspan(min(trans_indices), max(trans_indices), alpha=0.2, color='red', label='Transition')
        
        if np.any(post_transition_mask):
            post_indices = np.where(post_transition_mask)[0]
            if len(post_indices) > 0:
                plt.axvspan(min(post_indices), len(actual_values), alpha=0.1, color='blue', label=f'Regime {self.to_regime}')
        
        plt.xlabel('Time')
        plt.ylabel('Returns')
        plt.legend()
        plt.title(f'Regime Change Scenario: Actual vs Predicted Returns\nRegime Adaptation Score: {metrics["regime_adaptation"]:.4f}')
        
        # Plot prediction errors across regimes
        plt.subplot(2, 1, 2)
        errors = np.abs(predictions - actual_values)
        plt.plot(errors, label='Absolute Error')
        
        # Add horizontal lines for average error in each regime
        if np.any(pre_transition_mask):
            pre_error = np.mean(errors[pre_transition_mask])
            plt.axhline(pre_error, color='green', linestyle='--', label=f'Avg Error Regime {self.from_regime}: {pre_error:.4f}')
        
        if np.any(during_transition_mask):
            during_error = np.mean(errors[during_transition_mask])
            plt.axhline(during_error, color='red', linestyle='--', label=f'Avg Error Transition: {during_error:.4f}')
        
        if np.any(post_transition_mask):
            post_error = np.mean(errors[post_transition_mask])
            plt.axhline(post_error, color='blue', linestyle='--', label=f'Avg Error Regime {self.to_regime}: {post_error:.4f}')
        
        # Highlight different periods
        if np.any(pre_transition_mask):
            pre_indices = np.where(pre_transition_mask)[0]
            if len(pre_indices) > 0:
                plt.axvspan(0, max(pre_indices), alpha=0.1, color='green')
        
        if np.any(during_transition_mask):
            trans_indices = np.where(during_transition_mask)[0]
            if len(trans_indices) > 0:
                plt.axvspan(min(trans_indices), max(trans_indices), alpha=0.2, color='red')
        
        if np.any(post_transition_mask):
            post_indices = np.where(post_transition_mask)[0]
            if len(post_indices) > 0:
                plt.axvspan(min(post_indices), len(actual_values), alpha=0.1, color='blue')
        
        plt.xlabel('Time')
        plt.ylabel('Absolute Error')
        plt.legend()
        plt.title('Prediction Error Across Regimes')
        
        plt.tight_layout()
        
        # Return metrics
        metrics = {
            'overall': {
                'mse': float(mse),
                'rmse': float(rmse),
                'mae': float(mae),
                'direction_accuracy': float(direction_accuracy)
            },
            'regime_adaptation': float(regime_adaptation) if not np.isnan(regime_adaptation) else None
        }
        
        return metrics, plt.gcf()

    def _calculate_period_metrics(self, predictions: np.ndarray, actual_values: np.ndarray) -> Dict[str, float]:
        """Calculate metrics for a specific period"""
        if len(predictions) == 0 or len(actual_values) == 0:
            return {
                'mse': None,
                'rmse': None,
                'mae': None,
                'direction_accuracy': None,
                'samples': 0
            }
        
        mse = np.mean((predictions - actual_values) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - actual_values))
        direction_accuracy = np.mean(np.sign(predictions) == np.sign(actual_values))
        
        return {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'direction_accuracy': float(direction_accuracy),
            'samples': len(predictions)
        }

class NovelPatternScenario(StressTestScenario):
    """Simulate a novel pattern scenario"""
    
    def __init__(self, pattern_duration: int = 15, pattern_type: str = "oscillation"):
        super().__init__(
            name="novel_pattern",
            description=f"Novel {pattern_type} pattern over {pattern_duration} days"
        )
        self.pattern_duration = pattern_duration
        self.pattern_type = pattern_type
    
    def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare novel pattern scenario data"""
        result = data.copy()
        
        # Find suitable location for pattern (at least 20% into the data)
        min_index = int(len(result) * 0.2)
        pattern_start = min_index + np.random.randint(0, len(result) - min_index - self.pattern_duration)
        
        # Generate the pattern
        pattern = self._generate_pattern()
        
        # Apply pattern to returns
        if 'return_1d' in result.columns:
            # Replace returns with pattern
            result.loc[pattern_start:pattern_start+self.pattern_duration-1, 'return_1d'] = pattern
            
            # Adjust price-based columns
            if 'close' in result.columns:
                # Recalculate close prices based on new returns
                for i in range(pattern_start, pattern_start + self.pattern_duration):
                    if i > 0:
                        result.loc[i, 'close'] = result.loc[i-1, 'close'] * (1 + result.loc[i, 'return_1d'])
                
                # Adjust other price columns
                if all(col in result.columns for col in ['open', 'high', 'low']):
                    for i in range(pattern_start, pattern_start + self.pattern_duration):
                        if i > 0:
                            # Calculate relative position of open, high, low to close
                            prev_close = result.loc[i-1, 'close']
                            curr_close = result.loc[i, 'close']
                            
                            # Use the pattern to determine the price range (higher volatility for larger changes)
                            price_range = abs(result.loc[i, 'return_1d']) * 2.0
                            
                            # Adjust open, high, low
                            result.loc[i, 'open'] = curr_close
                            result.loc[i, 'high'] = curr_close * (1 + price_range/2)
                            result.loc[i, 'low'] = curr_close * (1 - price_range/2)
            
            # Mark pattern period
            result['novel_pattern'] = False
            result.loc[pattern_start:pattern_start+self.pattern_duration-1, 'novel_pattern'] = True
        
        return result
    
    def _generate_pattern(self) -> np.ndarray:
        """Generate a novel price pattern"""
        if self.pattern_type == "oscillation":
            # Create oscillation pattern
            t = np.linspace(0, 2*np.pi, self.pattern_duration)
            pattern = 0.02 * np.sin(t) + 0.01 * np.sin(2*t + 0.5)
        
        elif self.pattern_type == "trend_reversal":
            # Create trend reversal pattern
            pattern = np.concatenate([
                0.005 * np.ones(self.pattern_duration // 3),  # Uptrend
                np.linspace(0.005, -0.005, self.pattern_duration // 3),  # Transition
                -0.005 * np.ones(self.pattern_duration - 2*(self.pattern_duration // 3))  # Downtrend
            ])
        
        elif self.pattern_type == "shock":
            # Create shock pattern (big move followed by reversion)
            pattern = np.zeros(self.pattern_duration)
            pattern[0] = -0.05  # Big drop
            pattern[1:4] = 0.01  # Small recovery
            # Rest remains near zero
        
        else:
            # Default to random walk with drift
            pattern = np.random.normal(0, 0.01, self.pattern_duration)
            pattern = pattern + np.linspace(0, 0.01, self.pattern_duration)
        
        return pattern
    
    def evaluate(self, model: torch.nn.Module, data: pd.DataFrame, device: str) -> Dict[str, Any]:
        """Evaluate model on novel pattern scenario"""
        # Prepare data
        stress_data = self.prepare_data(data)
        
        # Extract features and target
        feature_cols = [col for col in stress_data.columns if col not in ['date', 'target', 'novel_pattern']]
        
        if 'target' in stress_data.columns:
            target_col = 'target'
        elif 'return_1d' in stress_data.columns:
            # Use next day's return as target
            stress_data['target'] = stress_data['return_1d'].shift(-1)
            target_col = 'target'
        else:
            raise ValueError("No target column found in data")
        
        # Create data loader
        sequence_length = 20  # Default sequence length
        
        # Filter out non-numeric columns
        numeric_cols = []
        for col in feature_cols:
            try:
                pd.to_numeric(stress_data[col])
                numeric_cols.append(col)
            except:
                print(f"Skipping non-numeric column: {col}")
                
        feature_cols = numeric_cols
        
        # Select most relevant features if there are too many
        if len(feature_cols) > 20:
            # Prioritize basic price and volatility features
            priority_features = ['price', 'volume', 'returns', 'volatility', 'momentum_5d', 'momentum_10d', 'rsi_14']
            selected_features = []
            
            # First include priority features
            for prefix in priority_features:
                for col in feature_cols:
                    if col.startswith(prefix) and col not in selected_features:
                        selected_features.append(col)
                        if len(selected_features) >= 7:  # Limit to 7 features
                            break
                            
            feature_cols = selected_features[:7]  # Take at most 7 features
            
        if len(feature_cols) == 0:
            raise ValueError("No numeric feature columns found in data")
        
        # Normalize features
        features = stress_data[feature_cols].values
        
        # Convert to numeric safely
        try:
            features = features.astype(np.float32)
        except ValueError:
            # Handle conversion errors by converting each column individually
            numeric_features = np.zeros((features.shape[0], len(feature_cols)), dtype=np.float32)
            for i, col in enumerate(feature_cols):
                try:
                    numeric_features[:, i] = stress_data[col].astype(np.float32)
                except:
                    # Fill with zeros if conversion fails
                    print(f"Failed to convert {col} to numeric, using zeros instead")
            features = numeric_features
        
        # Handle any remaining non-numeric values
        features = np.where(np.isnan(features), 0, features)
        
        feature_means = np.nanmean(features, axis=0)
        feature_stds = np.nanstd(features, axis=0)
        feature_stds[feature_stds == 0] = 1.0
        normalized_features = (features - feature_means) / feature_stds
        normalized_features = np.nan_to_num(normalized_features)
        
        # Get targets
        targets = stress_data[target_col].values
        is_pattern = stress_data['novel_pattern'].values
        
        # Prepare model
        model.to(device)
        model.eval()
        
        # Evaluate model
        predictions = []
        actual_values = []
        pattern_mask = []
        
        with torch.no_grad():
            for i in range(sequence_length, len(normalized_features)):
                # Get sequence and current features
                sequence = normalized_features[i-sequence_length:i]
                current_features = normalized_features[i]
                
                # Create tensors
                feature_tensor = torch.tensor(current_features, dtype=torch.float32).unsqueeze(0).to(device)
                sequence_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(device)
                
                # Forward pass - handle different model interfaces
                try:
                    # Try cognitive architecture interface first
                    outputs = model(financial_data=feature_tensor, financial_seq=sequence_tensor)
                    
                    # If outputs is a dict, extract the relevant prediction
                    if isinstance(outputs, dict) and 'market_state' in outputs:
                        # Get the prediction - could be multi-dimensional
                        pred = outputs['market_state'].cpu().numpy()
                        # If prediction has shape [batch, seq, features], take the first timestep
                        if len(pred.shape) > 2:
                            prediction = pred[0, 0, 0]  # Take first batch, first timestep, first feature
                        else:
                            prediction = pred[0, 0]  # Take first batch, first feature
                    else:
                        # Handle tensor output - could be multi-dimensional
                        pred = outputs.cpu().numpy()
                        if len(pred.shape) > 2:
                            prediction = pred[0, 0, 0]  # Take first batch, first timestep, first feature
                        else:
                            prediction = pred[0, 0]  # Take first batch, first feature
                except TypeError:
                    # Fall back to standard interface for baseline model
                    outputs = model(sequence_tensor)
                    prediction = outputs.cpu().numpy()[0, 0]
                
                # Store results
                predictions.append(prediction)
                actual_values.append(targets[i])
                pattern_mask.append(is_pattern[i])
        
        # Convert to numpy arrays
        predictions = np.array(predictions)
        actual_values = np.array(actual_values)
        pattern_mask = np.array(pattern_mask)
        
        # Calculate metrics
        pattern_indices = np.where(pattern_mask)[0]
        normal_indices = np.where(~pattern_mask)[0]
        
        # Overall metrics
        overall_metrics = self._calculate_metrics(predictions, actual_values)
        
        # Pattern period metrics
        pattern_metrics = self._calculate_metrics(
            predictions[pattern_mask], actual_values[pattern_mask]
        ) if np.any(pattern_mask) else None
        
        # Normal period metrics
        normal_metrics = self._calculate_metrics(
            predictions[normal_indices], actual_values[normal_indices]
        ) if len(normal_indices) > 0 else None
        
        # Calculate novelty adaptation score
        novelty_adaptation = np.nan
        if pattern_metrics and normal_metrics:
            # Compare pattern error to normal error
            pattern_rmse = pattern_metrics['rmse']
            normal_rmse = normal_metrics['rmse']
            
            # Calculate adaptation score (1 = same performance on novel pattern, 0 = much worse)
            error_ratio = min(3.0, pattern_rmse / normal_rmse)  # Cap at 3x worse
            novelty_adaptation = 1.0 - (error_ratio - 1.0) / 2.0  # Scale to [0, 1]
            novelty_adaptation = max(0.0, novelty_adaptation)
        
        # Create visualization
        plt.figure(figsize=(15, 10))
        
        # Plot returns and predictions
        plt.subplot(2, 1, 1)
        plt.plot(actual_values, label='Actual Returns')
        plt.plot(predictions, label='Predicted Returns')
        
        # Highlight pattern period
        if np.any(pattern_mask):
            pattern_start = np.min(pattern_indices)
            pattern_end = np.max(pattern_indices)
            plt.axvspan(pattern_start, pattern_end, alpha=0.2, color='purple', label=f'Novel Pattern ({self.pattern_type})')
        
        plt.xlabel('Time')
        plt.ylabel('Returns')
        plt.legend()
        plt.title(f'Novel Pattern Scenario: Actual vs Predicted Returns\nNovelty Adaptation Score: {novelty_adaptation:.4f if not np.isnan(novelty_adaptation) else "N/A"}')
        
        # Plot prediction errors
        plt.subplot(2, 1, 2)
        errors = np.abs(predictions - actual_values)
        plt.plot(errors, label='Absolute Error')
        
        # Add horizontal lines for average error in each period
        if np.any(pattern_mask):
            pattern_error = np.mean(errors[pattern_mask])
            plt.axhline(pattern_error, color='purple', linestyle='--', 
                      label=f'Avg Error Novel Pattern: {pattern_error:.4f}')
        
        if len(normal_indices) > 0:
            normal_error = np.mean(errors[normal_indices])
            plt.axhline(normal_error, color='green', linestyle='--', 
                      label=f'Avg Error Normal: {normal_error:.4f}')
        
        # Highlight pattern period
        if np.any(pattern_mask):
            pattern_start = np.min(pattern_indices)
            pattern_end = np.max(pattern_indices)
            plt.axvspan(pattern_start, pattern_end, alpha=0.2, color='purple')
        
        plt.xlabel('Time')
        plt.ylabel('Absolute Error')
        plt.legend()
        plt.title('Prediction Error for Novel Pattern')
        
        plt.tight_layout()
        
        # Return metrics
        metrics = {
            'overall': overall_metrics,
            'pattern_period': pattern_metrics,
            'normal_period': normal_metrics,
            'novelty_adaptation': float(novelty_adaptation) if not np.isnan(novelty_adaptation) else None
        }
        
        return metrics, plt.gcf()
    
    def _calculate_metrics(self, predictions: np.ndarray, actual_values: np.ndarray) -> Dict[str, float]:
        """Calculate metrics"""
        if len(predictions) == 0:
            return None
        
        mse = np.mean((predictions - actual_values) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - actual_values))
        direction_accuracy = np.mean(np.sign(predictions) == np.sign(actual_values))
        
        return {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'direction_accuracy': float(direction_accuracy),
            'samples': len(predictions)
        }

def run_stress_tests(
    model_path: str,
    test_data_path: str,
    output_dir: str,
    scenarios: List[str] = None,
    model_type: str = "cognitive",
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> Dict[str, Any]:
    """
    Run stress tests on model
    
    Args:
        model_path: Path to model checkpoint
        test_data_path: Path to test data
        output_dir: Directory to save results
        scenarios: List of scenarios to run
        model_type: Model type ('cognitive' or 'baseline')
        device: Computation device
        
    Returns:
        Dictionary with stress test results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load test data
    test_data = pd.read_csv(test_data_path)
    print(f"Loaded test data with {len(test_data)} samples")
    
    # Load model
    print(f"Loading {model_type} model from {model_path}")
    
    try:
        # For cognitive models, use the new from_checkpoint method
        if model_type.lower() == "cognitive":
            try:
                model = CognitiveArchitecture.from_checkpoint(model_path, device)
                # Get configuration from model attributes
                config = {
                    'input_dim': model.input_dim,
                    'hidden_dim': model.hidden_dim,
                    'memory_size': getattr(model.financial_memory, 'num_slots', 50),
                    'output_dim': model.output_dim,
                    'seq_length': model.seq_length
                }
                print(f"Using model configuration: input_dim={config['input_dim']}, hidden_dim={config['hidden_dim']}, memory_size={config['memory_size']}")
            except Exception as e:
                print(f"Error loading cognitive model with from_checkpoint: {e}")
                raise
        else:
            # Load baseline model with legacy method
            checkpoint = torch.load(model_path, map_location=device)
            
            # Extract configuration based on actual dimensions in the checkpoint
            config = {}
            
            # Case 1: If 'config' exists in checkpoint, use it
            if isinstance(checkpoint, dict) and 'config' in checkpoint:
                config = checkpoint['config']
            
            # Case 2: If checkpoint is a state dict (most likely)
            else:
                # Try to determine dimensions directly from the weights
                if 'lstm.weight_ih_l0' in checkpoint:
                    # Extract input dimension from the LSTM input weights
                    input_dim = checkpoint['lstm.weight_ih_l0'].shape[1]
                    config['input_dim'] = input_dim
                    
                    # Extract hidden dimension from LSTM weights (LSTM has 4 gates)
                    hidden_dim = checkpoint['lstm.weight_ih_l0'].shape[0] // 4
                    config['hidden_dim'] = hidden_dim
                    
                    # Extract output dimension from final layer if available
                    if 'fc.weight' in checkpoint:
                        output_dim = checkpoint['fc.weight'].shape[0]
                        config['output_dim'] = output_dim
                
                # Default values if not found
                if 'input_dim' not in config:
                    config['input_dim'] = 7  # Default for baseline
                if 'hidden_dim' not in config:
                    config['hidden_dim'] = 64  # Default
                if 'output_dim' not in config:
                    config['output_dim'] = 7  # Default for baseline
            
            print(f"Using baseline model configuration: input_dim={config['input_dim']}, hidden_dim={config['hidden_dim']}")
            
            # Create model with extracted configuration
            model = FinancialLSTMBaseline(
                input_dim=config.get('input_dim', 7),
                hidden_dim=config.get('hidden_dim', 64),
                num_layers=config.get('num_layers', 2),
                output_dim=config.get('output_dim', 7)
            )
            
            # Load state dict
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
        
    model.to(device)
    model.eval()
    
    # Define available scenarios
    available_scenarios = {
        "market_crash": MarketCrashScenario(crash_magnitude=-0.1, crash_duration=5),
        "high_volatility": HighVolatilityScenario(volatility_multiplier=2.5, duration=10),
        "regime_change": RegimeChangeScenario(from_regime=1, to_regime=2, transition_duration=5),
        "novel_pattern": NovelPatternScenario(pattern_duration=15, pattern_type="oscillation")
    }
    
    # Select scenarios to run
    if scenarios is None:
        scenarios = list(available_scenarios.keys())
    
    selected_scenarios = [available_scenarios[name] for name in scenarios if name in available_scenarios]
    
    if not selected_scenarios:
        raise ValueError(f"No valid scenarios selected. Available scenarios: {list(available_scenarios.keys())}")
    
    # Run scenarios
    results = {}
    for scenario in selected_scenarios:
        print(f"Running stress test: {scenario.name} - {scenario.description}")
        
        # Run scenario
        metrics, figure = scenario.evaluate(model, test_data, device)
        
        # Save visualization
        figure_path = os.path.join(output_dir, f"{scenario.name}_visualization.png")
        figure.savefig(figure_path, dpi=300)
        plt.close(figure)
        
        # Save metrics
        metrics_path = os.path.join(output_dir, f"{scenario.name}_metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Store results
        results[scenario.name] = {
            'description': scenario.description,
            'metrics': metrics,
            'figure_path': figure_path
        }
        
        print(f"  Results saved to {figure_path} and {metrics_path}")
    
    # Create summary visualization
    create_summary_visualization(results, os.path.join(output_dir, "stress_test_summary.png"))
    
    # Create summary report
    summary = {
        'model_path': model_path,
        'model_type': model_type,
        'test_data_path': test_data_path,
        'timestamp': datetime.now().isoformat(),
        'scenarios': [{'name': name, **info} for name, info in results.items()]
    }
    
    summary_path = os.path.join(output_dir, "stress_test_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Stress testing completed. Summary saved to {summary_path}")
    
    return results

def create_summary_visualization(results: Dict[str, Any], output_path: str) -> None:
    """
    Create summary visualization of stress test results
    
    Args:
        results: Dictionary with stress test results
        output_path: Path to save visualization
    """
    # Extract metrics for visualization
    scenarios = list(results.keys())
    
    # Metrics to compare
    metric_sets = [
        {'name': 'RMSE', 'key': 'rmse', 'lower_is_better': True},
        {'name': 'Direction Accuracy', 'key': 'direction_accuracy', 'lower_is_better': False}
    ]
    
    # Create subplots
    fig, axes = plt.subplots(len(metric_sets), 1, figsize=(12, 4 * len(metric_sets)))
    
    if len(metric_sets) == 1:
        axes = [axes]
    
    # Plot each metric
    for i, metric_set in enumerate(metric_sets):
        ax = axes[i]
        metric_name = metric_set['name']
        metric_key = metric_set['key']
        lower_is_better = metric_set['lower_is_better']
        
        # Extract values
        overall_values = []
        stress_values = []
        
        for scenario in scenarios:
            scenario_results = results[scenario]
            metrics = scenario_results['metrics']
            
            # Get overall metric
            if 'overall' in metrics and metric_key in metrics['overall']:
                overall_value = metrics['overall'][metric_key]
                overall_values.append(overall_value)
            else:
                overall_values.append(0)
            
            # Get stress period metric
            stress_value = None
            for period_key in ['stress_period', 'crash_period', 'high_volatility_period', 'pattern_period', 'during_transition']:
                if period_key in metrics and metrics[period_key] is not None:
                    if metric_key in metrics[period_key]:
                        stress_value = metrics[period_key][metric_key]
                        break
            
            if stress_value is None:
                stress_value = overall_values[-1]
            
            stress_values.append(stress_value)
        
        # Create bar chart
        x = np.arange(len(scenarios))
        width = 0.35
        
        ax.bar(x - width/2, overall_values, width, label='Overall')
        ax.bar(x + width/2, stress_values, width, label='Stress Period')
        
        # Add labels and legend
        ax.set_xlabel('Scenario')
        ax.set_ylabel(metric_name)
        ax.set_title(f'{metric_name} by Scenario')
        ax.set_xticks(x)
        ax.set_xticklabels([s.replace('_', ' ').title() for s in scenarios])
        ax.legend()
        
        # Highlight better values
        for j, (overall, stress) in enumerate(zip(overall_values, stress_values)):
            color = 'green' if not lower_is_better else 'red'
            if stress > overall and not lower_is_better:
                color = 'green'
            elif stress < overall and lower_is_better:
                color = 'green'
            else:
                color = 'red'
            
            ax.text(j + width/2, stress, f'{stress:.4f}', ha='center', va='bottom', color=color)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Stress testing for cognitive models")
    parser.add_argument("--model_path", required=True, help="Path to model checkpoint")
    parser.add_argument("--test_data", required=True, help="Path to test data")
    parser.add_argument("--output_dir", default="evaluation/stress_test", help="Directory to save results")
    parser.add_argument("--model_type", default="cognitive", choices=["cognitive", "baseline"], help="Model type")
    parser.add_argument("--scenarios", default="market_crash,high_volatility", 
                       help="Comma-separated list of scenarios to run")
    parser.add_argument("--device", default=None, help="Computation device")
    
    args = parser.parse_args()
    
    # Set device
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    try:
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Parse scenarios
        scenario_names = args.scenarios.split(",")
        print(f"Running the following stress test scenarios: {scenario_names}")
        
        # Load data
        print(f"Loading test data from {args.test_data}")
        test_data = pd.read_csv(args.test_data)
        
        # Run stress tests
        results = run_stress_tests(
            model_path=args.model_path,
            test_data_path=args.test_data,
            output_dir=args.output_dir,
            scenarios=scenario_names,
            model_type=args.model_type,
            device=args.device
        )
        
        # Save overall results
        with open(os.path.join(args.output_dir, "stress_test_results.json"), "w") as f:
            json.dump(results, f, indent=2)
        
        # Create summary visualization
        create_summary_visualization(results, os.path.join(args.output_dir, "stress_test_summary.png"))
        
        print(f"Stress test complete. Results saved to {args.output_dir}")
        return 0
    
    except Exception as e:
        print(f"Error during stress testing: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
