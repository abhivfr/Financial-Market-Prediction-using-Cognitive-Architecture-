#!/usr/bin/env python
# model_server.py - Deployment server for cognitive models

import os
import sys
import torch
import json
import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple
from threading import Thread, Lock
import queue
from pathlib import Path

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.insert(0, root_dir)

from src.arch.cognitive import CognitiveArchitecture
from src.arch.baseline_lstm import FinancialLSTMBaseline
from src.utils.online_learning import OnlineLearner
from src.monitoring.introspect import ModelIntrospector

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('model_server.log')
    ]
)
logger = logging.getLogger('model_server')

class PredictionRequest:
    """Prediction request container"""
    
    def __init__(
        self, 
        features: Dict[str, float],
        sequence: Optional[List[Dict[str, float]]] = None,
        request_id: Optional[str] = None
    ):
        """
        Initialize prediction request
        
        Args:
            features: Feature dictionary
            sequence: Historical sequence (optional)
            request_id: Unique request ID
        """
        self.features = features
        self.sequence = sequence
        self.request_id = request_id or datetime.now().strftime("%Y%m%d%H%M%S%f")
        self.timestamp = datetime.now()
    
    def to_tensor(self, feature_names: List[str]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Convert features to tensor
        
        Args:
            feature_names: List of feature names
            
        Returns:
            Tensors for features and sequence
        """
        # Extract features in correct order
        feature_values = [self.features.get(name, 0.0) for name in feature_names]
        feature_tensor = torch.tensor([feature_values], dtype=torch.float32)
        
        # Process sequence if available
        sequence_tensor = None
        if self.sequence:
            # Extract sequence features in correct order
            seq_values = []
            for seq_item in self.sequence:
                item_values = [seq_item.get(name, 0.0) for name in feature_names]
                seq_values.append(item_values)
            
            sequence_tensor = torch.tensor([seq_values], dtype=torch.float32)
        
        return feature_tensor, sequence_tensor

class PredictionResponse:
    """Prediction response container"""
    
    def __init__(
        self,
        request_id: str,
        prediction: Union[float, Dict[str, float]],
        confidence: Optional[float] = None,
        processing_time: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize prediction response
        
        Args:
            request_id: Request ID
            prediction: Prediction value or dictionary
            confidence: Prediction confidence (optional)
            processing_time: Processing time in milliseconds
            metadata: Additional metadata
        """
        self.request_id = request_id
        self.prediction = prediction
        self.confidence = confidence
        self.processing_time = processing_time
        self.metadata = metadata or {}
        self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = {
            'request_id': self.request_id,
            'prediction': self.prediction,
            'timestamp': self.timestamp.isoformat(),
            'processing_time_ms': self.processing_time
        }
        
        if self.confidence is not None:
            result['confidence'] = self.confidence
        
        if self.metadata:
            result['metadata'] = self.metadata
        
        return result
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        result = self.to_dict()
        
        # Convert any numpy or torch values to native Python types
        def convert_numpy(obj):
            if isinstance(obj, (np.integer, np.floating, np.bool_)):
                return obj.item()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, datetime):
                return obj.isoformat()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(i) for i in obj]
            else:
                return obj
        
        result = convert_numpy(result)
        
        return json.dumps(result, indent=2)

class ModelServer:
    """Server for cognitive model inference and online learning"""
    
    def __init__(
        self,
        model_path: str,
        model_type: str = "cognitive",
        feature_names: Optional[List[str]] = None,
        sequence_length: int = 10,
        enable_online_learning: bool = True,
        prediction_history_size: int = 1000,
        enable_introspection: bool = True,
        telemetry_dir: str = "model_telemetry",
        device: str = "cpu"
    ):
        """
        Initialize model server
        
        Args:
            model_path: Path to model checkpoint
            model_type: Model type ('cognitive' or 'baseline')
            feature_names: List of feature names
            sequence_length: Sequence length for prediction
            enable_online_learning: Whether to enable online learning
            prediction_history_size: Size of prediction history
            enable_introspection: Whether to enable model introspection
            telemetry_dir: Directory for telemetry data
            device: Computation device
        """
        self.model_path = model_path
        self.model_type = model_type
        self.sequence_length = sequence_length
        self.enable_online_learning = enable_online_learning
        self.enable_introspection = enable_introspection
        self.telemetry_dir = telemetry_dir
        self.device = device
        
        # Initialize feature names if not provided
        self.feature_names = feature_names or [
            'open', 'high', 'low', 'close', 'volume',
            'return_1d', 'return_5d', 'volatility_10d',
            'rsi_14', 'macd', 'ma_20', 'ma_50'
        ]
        
        # Load model
        self.model = self._load_model()
        logger.info(f"Model loaded from {model_path}")
        
        # Initialize online learner if enabled
        self.online_learner = None
        if enable_online_learning:
            self.online_learner = OnlineLearner(
                model=self.model,
                buffer_size=1000,
                device=device
            )
            logger.info("Online learning enabled")
        
        # Initialize introspector if enabled
        self.introspector = None
        if enable_introspection and model_type.lower() == "cognitive":
            self.introspector = ModelIntrospector(self.model)
            logger.info("Model introspection enabled")
        
        # Initialize sequence buffer
        self.sequence_buffer = []
        
        # Initialize prediction history
        self.prediction_history = []
        self.prediction_history_size = prediction_history_size
        
        # Initialize telemetry
        os.makedirs(telemetry_dir, exist_ok=True)
        self.telemetry_file = os.path.join(telemetry_dir, f"telemetry_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        self.telemetry_lock = Lock()
        
        # Initialize worker queue for async predictions
        self.prediction_queue = queue.Queue()
        self.worker_thread = None
        self.running = False
        
        # Statistics
        self.stats = {
            'requests_processed': 0,
            'online_updates': 0,
            'avg_processing_time': 0.0,
            'start_time': datetime.now()
        }
    
    def _load_model(self) -> torch.nn.Module:
        """
        Load model from checkpoint
        
        Returns:
            PyTorch model
        """
        # Check if model file exists
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        # Create model based on type
        if self.model_type.lower() == "cognitive":
            model = CognitiveArchitecture()
        else:
            model = FinancialLSTMBaseline()
        
        # Load weights
        model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        
        # Set to evaluation mode
        model.eval()
        
        # Move to device
        model.to(self.device)
        
        return model
    
    def start(self) -> None:
        """Start model server"""
        if self.running:
            logger.warning("Server already running")
            return
        
        # Start worker thread for async predictions
        self.running = True
        self.worker_thread = Thread(target=self._prediction_worker)
        self.worker_thread.daemon = True
        self.worker_thread.start()
        
        logger.info("Model server started")
    
    def stop(self) -> None:
        """Stop model server"""
        self.running = False
        
        if self.worker_thread:
            self.worker_thread.join(timeout=2.0)
        
        # Save telemetry
        self._save_telemetry()
        
        # Save online learner if enabled
        if self.online_learner:
            online_dir = os.path.join(self.telemetry_dir, "online_learner")
            self.online_learner.save(online_dir)
        
        logger.info("Model server stopped")
    
    def predict(
        self, 
        features: Dict[str, float],
        sequence: Optional[List[Dict[str, float]]] = None,
        request_id: Optional[str] = None,
        async_mode: bool = False
    ) -> Union[PredictionResponse, str]:
        """
        Make prediction with model
        
        Args:
            features: Feature dictionary
            sequence: Historical sequence (optional)
            request_id: Request ID (optional)
            async_mode: Whether to process asynchronously
            
        Returns:
            Prediction response or request ID if async
        """
        # Create prediction request
        request = PredictionRequest(
            features=features,
            sequence=sequence,
            request_id=request_id
        )
        
        # Process asynchronously if requested
        if async_mode:
            self.prediction_queue.put(request)
            return request.request_id
        
        # Process synchronously
        return self._process_prediction(request)
    
    def _prediction_worker(self) -> None:
        """Worker thread for async predictions"""
        while self.running:
            try:
                # Get request from queue
                request = self.prediction_queue.get(timeout=1.0)
                
                # Process prediction
                self._process_prediction(request)
                
                # Mark task as done
                self.prediction_queue.task_done()
            
            except queue.Empty:
                # No requests
                continue
            
            except Exception as e:
                logger.error(f"Error in prediction worker: {e}")
    
    def _process_prediction(self, request: PredictionRequest) -> PredictionResponse:
        """
        Process prediction request
        
        Args:
            request: Prediction request
            
        Returns:
            Prediction response
        """
        start_time = time.time()
        
        # Convert request to tensors
        feature_tensor, sequence_tensor = request.to_tensor(self.feature_names)
        
        # Use sequence buffer if sequence not provided
        if sequence_tensor is None and self.sequence_buffer:
            # Create sequence tensor from buffer
            buffer_size = min(len(self.sequence_buffer), self.sequence_length)
            buffer_values = self.sequence_buffer[-buffer_size:]
            
            if buffer_size == self.sequence_length:
                seq_values = []
                for seq_item in buffer_values:
                    item_values = [seq_item.get(name, 0.0) for name in self.feature_names]
                    seq_values.append(item_values)
                
                sequence_tensor = torch.tensor([seq_values], dtype=torch.float32)
        
        # Move to device
        feature_tensor = feature_tensor.to(self.device)
        if sequence_tensor is not None:
            sequence_tensor = sequence_tensor.to(self.device)
        
        # Process with model
        with torch.no_grad():
            # Make prediction
            if sequence_tensor is not None:
                outputs = self.model(financial_data=feature_tensor, financial_seq=sequence_tensor)
            else:
                outputs = self.model(financial_data=feature_tensor)
        
        # Extract prediction and confidence
        prediction = None
        confidence = None
        
        if isinstance(outputs, dict) and 'market_state' in outputs:
            prediction = outputs['market_state'].cpu().numpy()[0]
            if 'uncertainty' in outputs:
                confidence = 1.0 - outputs['uncertainty'].cpu().numpy()[0]
        else:
            prediction = outputs.cpu().numpy()[0]
        
        # Process with online learner if enabled
        online_result = {}
        if self.online_learner and sequence_tensor is not None:
            # Calculate target if we have future data
            target = None
            
            # Process with online learner (no update yet until we have target)
            online_result = self.online_learner.process_sample(
                features=feature_tensor,
                sequence=sequence_tensor,
                target=None,
                update_model=False
            )
        
        # Get introspection data if enabled
        introspection_data = {}
        if self.introspector and self.enable_introspection:
            try:
                introspection_data = self.introspector.introspect_model(
                    financial_data=feature_tensor,
                    financial_seq=sequence_tensor if sequence_tensor is not None else None
                )
                
                # Process introspection data
                introspection_data = {
                    k: v.cpu().detach().numpy() if isinstance(v, torch.Tensor) else v
                    for k, v in introspection_data.items()
                }
            except Exception as e:
                logger.error(f"Error during introspection: {e}")
        
        # Update sequence buffer
        self.sequence_buffer.append(features)
        if len(self.sequence_buffer) > self.sequence_length * 2:
            self.sequence_buffer = self.sequence_buffer[-self.sequence_length * 2:]
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Update statistics
        self.stats['requests_processed'] += 1
        self.stats['avg_processing_time'] = (
            (self.stats['avg_processing_time'] * (self.stats['requests_processed'] - 1) + processing_time) / 
            self.stats['requests_processed']
        )
        
        # Create metadata
        metadata = {
            'model_type': self.model_type,
            'online_learning': self.enable_online_learning,
            'drift_detected': online_result.get('drift_detected', False)
        }
        
        # Add introspection data to metadata
        if introspection_data:
            metadata['introspection'] = introspection_data
        
        # Create response
        response = PredictionResponse(
            request_id=request.request_id,
            prediction=prediction.item() if isinstance(prediction, (np.ndarray, np.number)) else prediction,
            confidence=confidence.item() if isinstance(confidence, (np.ndarray, np.number)) else confidence,
            processing_time=processing_time,
            metadata=metadata
        )
        
        # Add to prediction history
        self._add_to_history(request, response)
        
        # Log telemetry
        self._log_telemetry(request, response)
        
        return response
    
    def _add_to_history(self, request: PredictionRequest, response: PredictionResponse) -> None:
        """
        Add prediction to history
        
        Args:
            request: Prediction request
            response: Prediction response
        """
        # Add to history
        self.prediction_history.append({
            'request_id': request.request_id,
            'timestamp': response.timestamp,
            'features': request.features,
            'prediction': response.prediction,
            'confidence': response.confidence,
            'processing_time': response.processing_time
        })
        
        # Limit history size
        if len(self.prediction_history) > self.prediction_history_size:
            self.prediction_history = self.prediction_history[-self.prediction_history_size:]
    
    def _log_telemetry(self, request: PredictionRequest, response: PredictionResponse) -> None:
        """
        Log telemetry data
        
        Args:
            request: Prediction request
            response: Prediction response
        """
        # Create telemetry record
        telemetry = {
            'timestamp': response.timestamp.isoformat(),
            'request_id': request.request_id,
            'processing_time_ms': response.processing_time,
            'prediction': response.prediction
        }
        
        # Add confidence if available
        if response.confidence is not None:
            telemetry['confidence'] = response.confidence
        
        # Add all features
        for name, value in request.features.items():
            telemetry[f'feature_{name}'] = value
        
        # Add online learning data if available
        if 'metadata' in response.to_dict() and 'drift_detected' in response.to_dict()['metadata']:
            telemetry['drift_detected'] = response.to_dict()['metadata']['drift_detected']
        
        # Write to telemetry file
        with self.telemetry_lock:
            # Create file if it doesn't exist
            file_exists = os.path.exists(self.telemetry_file)
            
            with open(self.telemetry_file, 'a') as f:
                # Write header if file doesn't exist
                if not file_exists:
                    f.write(','.join(telemetry.keys()) + '\n')
                
                # Write values
                f.write(','.join(str(v) for v in telemetry.values()) + '\n')
    
    def _save_telemetry(self) -> None:
        """Save telemetry data"""
        # Save prediction history
        history_file = os.path.join(self.telemetry_dir, 'prediction_history.json')
        
        with open(history_file, 'w') as f:
            # Convert history to serializable format
            history = []
            for item in self.prediction_history:
                # Convert timestamps
                item_copy = item.copy()
                if isinstance(item_copy['timestamp'], datetime):
                    item_copy['timestamp'] = item_copy['timestamp'].isoformat()
                
                history.append(item_copy)
            
            json.dump(history, f, indent=2)
        
        # Save server statistics
        stats_file = os.path.join(self.telemetry_dir, 'server_stats.json')
        
        with open(stats_file, 'w') as f:
            # Convert statistics to serializable format
            stats = self.stats.copy()
            if isinstance(stats['start_time'], datetime):
                stats['start_time'] = stats['start_time'].isoformat()
            
            stats['end_time'] = datetime.now().isoformat()
            stats['uptime_seconds'] = (datetime.now() - self.stats['start_time']).total_seconds()
            
            json.dump(stats, f, indent=2)
    
    def update_with_feedback(
        self, 
        request_id: str, 
        actual_value: float
    ) -> bool:
        """
        Update model with feedback
        
        Args:
            request_id: Request ID
            actual_value: Actual target value
            
        Returns:
            True if update successful
        """
        if not self.online_learner:
            logger.warning("Online learning not enabled")
            return False
        
        # Find request in history
        request_item = None
        for item in self.prediction_history:
            if item['request_id'] == request_id:
                request_item = item
                break
        
        if not request_item:
            logger.warning(f"Request {request_id} not found in history")
            return False
        
        # Get features from history
        features = request_item['features']
        
        # Convert to tensors
        feature_tensor = torch.tensor([[features.get(name, 0.0) for name in self.feature_names]], dtype=torch.float32)
        
        # Get sequence from buffer at the time of prediction
        # This is an approximation since we don't store the exact sequence
        target_tensor = torch.tensor([[actual_value]], dtype=torch.float32)
        
        # Process through online learner
        try:
            # Get sequence from buffer
            buffer_size = min(len(self.sequence_buffer), self.sequence_length)
            buffer_values = self.sequence_buffer[-buffer_size:]
            
            if buffer_size == self.sequence_length:
                seq_values = []
                for seq_item in buffer_values:
                    item_values = [seq_item.get(name, 0.0) for name in self.feature_names]
                    seq_values.append(item_values)
                
                sequence_tensor = torch.tensor([seq_values], dtype=torch.float32)
                
                # Update with feedback
                update_result = self.online_learner.process_sample(
                    features=feature_tensor,
                    sequence=sequence_tensor,
                    target=target_tensor,
                    update_model=True
                )
                
                # Update statistics if update performed
                if update_result.get('update_performed', False):
                    self.stats['online_updates'] += 1
                
                return update_result.get('update_performed', False)
            
        except Exception as e:
            logger.error(f"Error updating model with feedback: {e}")
        
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get server statistics
        
        Returns:
            Dictionary with statistics
        """
        stats = self.stats.copy()
        
        # Add uptime
        uptime_seconds = (datetime.now() - stats['start_time']).total_seconds()
        stats['uptime_seconds'] = uptime_seconds
        
        # Add queue size
        stats['queue_size'] = self.prediction_queue.qsize()
        
        # Add online learning stats if enabled
        if self.online_learner:
            stats['online_learning'] = {
                'buffer_size': len(self.online_learner.buffer.buffer) if hasattr(self.online_learner, 'buffer') else 0,
                'updates_performed': self.online_learner.updates_performed if hasattr(self.online_learner, 'updates_performed') else 0,
                'drift_detected_count': self.online_learner.drift_detected_count if hasattr(self.online_learner, 'drift_detected_count') else 0
            }
        
        return stats

def create_model_server(
    model_path: str,
    model_type: str = "cognitive",
    feature_names: Optional[List[str]] = None,
    enable_online_learning: bool = True,
    enable_introspection: bool = True,
    device: str = "cpu"
) -> ModelServer:
    """
    Create model server
    
    Args:
        model_path: Path to model checkpoint
        model_type: Model type ('cognitive' or 'baseline')
        feature_names: List of feature names (optional)
        enable_online_learning: Whether to enable online learning
        enable_introspection: Whether to enable introspection
        device: Computation device
        
    Returns:
        Model server instance
    """
    server = ModelServer(
        model_path=model_path,
        model_type=model_type,
        feature_names=feature_names,
        enable_online_learning=enable_online_learning,
        enable_introspection=enable_introspection,
        device=device
    )
    
    # Start server
    server.start()
    
    return server

if __name__ == "__main__":
    import argparse
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Model server for cognitive models")
    parser.add_argument("--model_path", required=True, help="Path to model checkpoint")
    parser.add_argument("--model_type", choices=["cognitive", "baseline"], default="cognitive", help="Model type")
    parser.add_argument("--no_online_learning", action="store_true", help="Disable online learning")
    parser.add_argument("--no_introspection", action="store_true", help="Disable introspection")
    parser.add_argument("--telemetry_dir", default="model_telemetry", help="Directory for telemetry data")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Computation device")
    
    args = parser.parse_args()
    
    # Create model server
    server = ModelServer(
        model_path=args.model_path,
        model_type=args.model_type,
        enable_online_learning=not args.no_online_learning,
        enable_introspection=not args.no_introspection,
        telemetry_dir=args.telemetry_dir,
        device=args.device
    )
    
    # Start server
    server.start()
    
    try:
        # Sleep until interrupted
        while True:
            time.sleep(1)
    
    except KeyboardInterrupt:
        # Stop server
        server.stop()
