#!/usr/bin/env python
# api_server.py - REST API server for cognitive models

import os
import sys
import json
import time
import logging
import argparse
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks, Body
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.insert(0, root_dir)

from src.deployment.model_server import ModelServer, create_model_server

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('api_server.log')
    ]
)
logger = logging.getLogger('api_server')

# Create FastAPI app
app = FastAPI(
    title="Cognitive Model API",
    description="API for cognitive model inference and online learning",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request models
class PredictionFeatures(BaseModel):
    """Prediction features"""
    
    class Config:
        extra = "allow"  # Allow extra fields

class SequenceItem(BaseModel):
    """Sequence item for historical data"""
    
    class Config:
        extra = "allow"  # Allow extra fields

class PredictionRequest(BaseModel):
    """Prediction request"""
    
    features: PredictionFeatures
    sequence: Optional[List[SequenceItem]] = None
    request_id: Optional[str] = None
    async_mode: bool = False

class FeedbackRequest(BaseModel):
    """Feedback request for online learning"""
    
    request_id: str
    actual_value: float

class ModelServerStatus(BaseModel):
    """Model server status"""
    
    status: str = "running"
    stats: Dict[str, Any] = Field(default_factory=dict)

# Global model server instance
model_server = None

@app.get("/health", response_model=Dict[str, str])
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

@app.get("/status", response_model=ModelServerStatus)
async def get_status():
    """Get model server status"""
    if not model_server:
        raise HTTPException(status_code=503, detail="Model server not initialized")
    
    return {
        "status": "running",
        "stats": model_server.get_stats()
    }

@app.post("/predict", response_model=Dict[str, Any])
async def predict(request: PredictionRequest, background_tasks: BackgroundTasks):
    """
    Make prediction with model
    
    Args:
        request: Prediction request
        background_tasks: Background tasks
    
    Returns:
        Prediction response
    """
    if not model_server:
        raise HTTPException(status_code=503, detail="Model server not initialized")
    
    try:
        # Convert request models to dictionaries
        features = {k: v for k, v in request.features.__dict__.items() if not k.startswith('_')}
        
        sequence = None
        if request.sequence:
            sequence = [
                {k: v for k, v in item.__dict__.items() if not k.startswith('_')}
                for item in request.sequence
            ]
        
        # Make prediction
        response = model_server.predict(
            features=features,
            sequence=sequence,
            request_id=request.request_id,
            async_mode=request.async_mode
        )
        
        # For async mode, return request ID
        if request.async_mode:
            return {"request_id": response, "async": True}
        
        # For sync mode, return full response
        return response.to_dict()
    
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/feedback", response_model=Dict[str, Any])
async def feedback(request: FeedbackRequest):
    """
    Provide feedback for online learning
    
    Args:
        request: Feedback request
    
    Returns:
        Update status
    """
    if not model_server:
        raise HTTPException(status_code=503, detail="Model server not initialized")
    
    if not model_server.enable_online_learning:
        raise HTTPException(status_code=400, detail="Online learning not enabled")
    
    try:
        # Update model with feedback
        updated = model_server.update_with_feedback(
            request_id=request.request_id,
            actual_value=request.actual_value
        )
        
        return {
            "request_id": request.request_id,
            "updated": updated
        }
    
    except Exception as e:
        logger.error(f"Error during feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("startup")
async def startup_event():
    """Startup event handler"""
    global model_server
    
    logger.info("Starting model server")
    
    # Get command line arguments
    args = get_command_line_args()
    
    try:
        # Create model server
        model_server = create_model_server(
            model_path=args.model_path,
            model_type=args.model_type,
            enable_online_learning=not args.no_online_learning,
            enable_introspection=not args.no_introspection,
            device=args.device
        )
        
        logger.info("Model server started")
    
    except Exception as e:
        logger.error(f"Error starting model server: {e}")
        model_server = None

@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown event handler"""
    global model_server
    
    if model_server:
        logger.info("Stopping model server")
        model_server.stop()
        model_server = None

def get_command_line_args() -> argparse.Namespace:
    """
    Get command line arguments
    
    Returns:
        Parsed arguments
    """
    # Check if running as script or through uvicorn
    if len(sys.argv) > 1 and not sys.argv[1].startswith('--'):
        # Running through uvicorn
        parser = argparse.ArgumentParser(description="REST API server for cognitive models")
        parser.add_argument("--model_path", required=True, help="Path to model checkpoint")
        parser.add_argument("--model_type", choices=["cognitive", "baseline"], default="cognitive", help="Model type")
        parser.add_argument("--no_online_learning", action="store_true", help="Disable online learning")
        parser.add_argument("--no_introspection", action="store_true", help="Disable introspection")
        parser.add_argument("--host", default="127.0.0.1", help="Host to listen on")
        parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
        parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Computation device")
        
        return parser.parse_args()
    else:
        # Running as script
        # Use environment variables instead
        import os
        
        class EnvArgs:
            model_path = os.environ.get("MODEL_PATH", "models/cognitive_model.pt")
            model_type = os.environ.get("MODEL_TYPE", "cognitive")
            no_online_learning = os.environ.get("NO_ONLINE_LEARNING", "").lower() in ("true", "1", "yes")
            no_introspection = os.environ.get("NO_INTROSPECTION", "").lower() in ("true", "1", "yes")
            host = os.environ.get("HOST", "127.0.0.1")
            port = int(os.environ.get("PORT", "8000"))
            device = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
        
        return EnvArgs()

if __name__ == "__main__":
    # Get command line arguments
    args = get_command_line_args()
    
    # Run server
    uvicorn.run(
        "api_server:app",
        host=args.host,
        port=args.port,
        reload=False
    )
