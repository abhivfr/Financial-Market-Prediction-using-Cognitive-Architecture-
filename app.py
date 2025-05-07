#!/usr/bin/env python
# app.py - Web interface for cognitive architecture

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import subprocess
import json
from datetime import datetime, timedelta
import torch
from PIL import Image
import gradio as gr
from pathlib import Path
import seaborn as sns

# Add project root to path
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Import necessary modules
from src.arch.cognitive import CognitiveArchitecture
from src.arch.baseline_lstm import FinancialLSTMBaseline
from src.data.financial_loader import EnhancedFinancialDataLoader
from src.visualization.plot_engine import CognitiveVisualizer
from evaluation_utils import comprehensive_model_evaluation, compare_models
from src.monitoring.introspect import Introspection

def main():
    """Main app function"""
    st.set_page_config(page_title="Cognitive Finance Platform", layout="wide")
    
    st.title("Cognitive Finance Platform")
    
    # Sidebar navigation
    page = st.sidebar.selectbox(
        "Select Page",
        ["Home", "Data Management", "Training", "Evaluation", "Visualization", "Live Demo"]
    )
    
    # Home page
    if page == "Home":
        show_home_page()
    
    # Data Management page
    elif page == "Data Management":
        show_data_management()
    
    # Training page
    elif page == "Training":
        show_training_page()
    
    # Evaluation page
    elif page == "Evaluation":
        show_evaluation_page()
    
    # Visualization page
    elif page == "Visualization":
        show_visualization_page()
    
    # Live Demo page
    elif page == "Live Demo":
        show_live_demo()

def show_home_page():
    """Display the home page"""
    st.header("Welcome to the Cognitive Finance Platform")
    
    st.markdown("""
    This platform provides tools for training, evaluating, and using cognitive models for financial forecasting.
    
    ### Key Features:
    - **4D Cognitive Architecture**: Advanced forecasting with memory, attention, and regime awareness
    - **Data Management**: Download and process financial data
    - **Training**: Train models with progressive component focus
    - **Evaluation**: Comprehensive evaluation of cognitive capabilities
    - **Visualization**: Explore model internals and predictions
    
    ### Getting Started:
    1. Go to the **Data Management** page to download and prepare data
    2. Use the **Training** page to train your models
    3. Evaluate performance on the **Evaluation** page
    4. Explore model behavior on the **Visualization** page
    5. Try real-time forecasting with the **Live Demo**
    """)
    
    # Display system status
    st.subheader("System Status")
    
    col1, col2, col3 = st.columns(3)
    
    # Data status
    with col1:
        st.metric("Data Files", count_files("data"))
    
    # Model status
    with col2:
        st.metric("Trained Models", count_files("models", ".pth"))
    
    # Evaluation status
    with col3:
        st.metric("Evaluation Reports", count_files("evaluation", ".json"))

def show_data_management():
    """Display the data management page"""
    st.header("Data Management")
    
    # Data download section
    st.subheader("Download Financial Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        tickers = st.text_input("Ticker Symbols (comma-separated)", "SPY")
        start_date = st.date_input("Start Date", datetime.now() - timedelta(days=365*5))
        end_date = st.date_input("End Date", datetime.now())
        
    with col2:
        output_dir = st.text_input("Output Directory", "data/raw")
        calculate_features = st.checkbox("Calculate Additional Features", True)
        
    if st.button("Download Data"):
        if not tickers or not start_date or not end_date:
            st.error("Please provide ticker symbols, start date, and end date.")
        else:
            with st.spinner("Downloading data..."):
                # Call the download function via CLI
                cmd = [
                    "python", "cognitive_cli.py", "download",
                    "--tickers", tickers,
                    "--start", start_date.strftime("%Y-%m-%d"),
                    "--end", end_date.strftime("%Y-%m-%d"),
                    "--output", output_dir
                ]
                
                if calculate_features:
                    cmd.append("--features")
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    st.success(f"Successfully downloaded data for {tickers}")
                    st.text(result.stdout)
                else:
                    st.error("Error downloading data")
                    st.text(result.stderr)
    
    # Data splitting section
    st.subheader("Split Data for Training")
    
    col1, col2 = st.columns(2)
    
    with col1:
        input_path = st.text_input("Input Data File", "data/raw/SPY.csv")
        output_dir = st.text_input("Output Directory", "data/processed")
        
    with col2:
        train_ratio = st.slider("Training Data Ratio", 0.5, 0.9, 0.7)
        val_ratio = st.slider("Validation Data Ratio", 0.05, 0.3, 0.15)
        test_ratio = st.slider("Test Data Ratio", 0.05, 0.3, 0.15)
        split_mode = st.selectbox("Split Mode", ["time", "random"])
        visualize = st.checkbox("Visualize Data Splits", True)
    
    if st.button("Split Data"):
        with st.spinner("Splitting data..."):
            # Call the split function via CLI
            cmd = [
                "python", "cognitive_cli.py", "split",
                "--input", input_path,
                "--output", output_dir,
                "--train", str(train_ratio),
                "--val", str(val_ratio),
                "--test", str(test_ratio),
                "--mode", split_mode
            ]
            
            if visualize:
                cmd.append("--visualize")
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                st.success("Successfully split data")
                st.text(result.stdout)
                
                # Display visualization if available
                if visualize and os.path.exists(os.path.join(output_dir, "data_splits.png")):
                    st.image(os.path.join(output_dir, "data_splits.png"))
            else:
                st.error("Error splitting data")
                st.text(result.stderr)
    
    # Data preview section
    st.subheader("Data Preview")
    
    preview_file = st.selectbox(
        "Select File to Preview",
        list_data_files(["data/raw", "data/processed"])
    )
    
    if preview_file:
        try:
            df = pd.read_csv(preview_file)
            df.columns = [col.lower() for col in df.columns]
            st.dataframe(df.head(10))
            
            st.write(f"Data Shape: {df.shape}")
            
            # Display basic statistics
            st.write("Basic Statistics:")
            st.dataframe(df.describe())
            
        except Exception as e:
            st.error(f"Error loading file: {e}")

def show_training_page():
    """Display the training page"""
    st.header("Model Training")
    
    # Training configuration
    st.subheader("Training Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        train_path = st.text_input("Training Data Path", "data/processed/train.csv")
        val_path = st.text_input("Validation Data Path", "data/processed/val.csv")
        output_dir = st.text_input("Output Directory", "models")
        model_type = st.selectbox("Model Type", ["Cognitive", "Baseline"])
        
    with col2:
        batch_size = st.number_input("Batch Size", 8, 128, 32)
        seq_length = st.number_input("Sequence Length", 10, 100, 20)
        epochs = st.number_input("Number of Epochs", 5, 200, 50)
        learning_rate = st.number_input("Learning Rate", 1e-6, 1e-2, 1e-4, format="%.6f")
        
        if model_type == "Cognitive":
            progressive = st.checkbox("Use Progressive Training", True)
            component_focus = st.multiselect(
                "Pre-train Components",
                ["memory", "attention", "temporal", "core"],
                ["memory", "attention"]
            )
    
    # Start training
    if st.button("Start Training"):
        with st.spinner(f"Training {model_type} model..."):
            if model_type == "Cognitive":
                # Pre-train components if selected
                if component_focus:
                    for component in component_focus:
                        st.write(f"Pre-training {component} component...")
                        cmd = [
                            "python", "cognitive_cli.py", "pretrain",
                            "--train", train_path,
                            "--val", val_path,
                            "--component", component,
                            "--output", f"{output_dir}/components",
                            "--batch_size", str(batch_size),
                            "--epochs", "10",
                            "--lr", str(learning_rate)
                        ]
                        result = subprocess.run(cmd, capture_output=True, text=True)
                        st.text(result.stdout)
                
                # Train full model
                cmd = [
                    "python", "cognitive_cli.py", "train",
                    "--train", train_path,
                    "--val", val_path,
                    "--output", output_dir,
                    "--batch_size", str(batch_size),
                    "--seq_length", str(seq_length),
                    "--epochs", str(epochs),
                    "--lr", str(learning_rate)
                ]
                
                if progressive:
                    cmd.append("--progressive")
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                
            else:  # Baseline model
                cmd = [
                    "python", "train_baseline.py",
                    "--train", train_path,
                    "--val", val_path,
                    "--output", f"{output_dir}/baseline",
                    "--batch_size", str(batch_size),
                    "--seq_length", str(seq_length),
                    "--epochs", str(epochs),
                    "--lr", str(learning_rate)
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                st.success(f"Successfully trained {model_type} model")
                st.text(result.stdout[-1000:])  # Show last 1000 chars
            else:
                st.error(f"Error training {model_type} model")
                st.text(result.stderr)
    
    # Training logs
    st.subheader("Training Logs")
    
    # List available log directories
    log_dirs = []
    if os.path.exists("models"):
        for d in os.listdir("models"):
            log_path = os.path.join("models", d, "logs")
            if os.path.exists(log_path):
                log_dirs.append(log_path)
    
    if os.path.exists("models/baseline"):
        for d in os.listdir("models/baseline"):
            log_path = os.path.join("models/baseline", d, "logs")
            if os.path.exists(log_path):
                log_dirs.append(log_path)
    
    selected_log = st.selectbox("Select Training Log", log_dirs)
    
    if selected_log and os.path.exists(os.path.join(selected_log, "training_logs.json")):
        with open(os.path.join(selected_log, "training_logs.json"), "r") as f:
            logs = json.load(f)
        
        # Plot training curves
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(logs["train_losses"], label="Train Loss")
        ax.plot(logs["val_losses"], label="Validation Loss")
        ax.axvline(x=logs["best_epoch"], color='r', linestyle='--', 
                  label=f"Best Epoch ({logs['best_epoch']+1})")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Training and Validation Loss")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

        # Show validation metrics
        st.write("Validation Metrics at Best Epoch:")
        best_metrics = logs["val_metrics"][logs["best_epoch"]]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Validation Loss", f"{best_metrics['val_loss']:.6f}")
        
        with col2:
            st.metric("Price Accuracy", f"{best_metrics.get('price_accuracy', 0):.4f}")
        
        with col3:
            st.metric("Direction Accuracy", f"{best_metrics.get('direction_accuracy', 0):.4f}")

def show_evaluation_page():
    """Display the evaluation page"""
    st.header("Model Evaluation")
    
    # List available models
    cognitive_models = list_model_files("models", ".pth")
    baseline_models = list_model_files("models/baseline", ".pth")
    
    st.subheader("Standard Evaluation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        model_path = st.selectbox("Select Model", cognitive_models + baseline_models)
        test_data = st.text_input("Test Data Path", "data/processed/test.csv")
        
    with col2:
        batch_size = st.number_input("Batch Size", 8, 128, 32, key="eval_batch")
        seq_length = st.number_input("Sequence Length", 10, 100, 20, key="eval_seq")
        output_dir = st.text_input("Output Directory", "evaluation")
    
    if st.button("Run Evaluation"):
        with st.spinner("Evaluating model..."):
            is_cognitive = any(cm in model_path for cm in cognitive_models)
            cmd = [
                "python", "cognitive_cli.py", "evaluate",
                "--model", model_path,
                "--data", test_data,
                "--output", f"{output_dir}/{'cognitive' if is_cognitive else 'baseline'}",
                "--batch_size", str(batch_size),
                "--seq_length", str(seq_length)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                st.success("Evaluation completed successfully")
                st.text(result.stdout)
            else:
                st.error("Error during evaluation")
                st.text(result.stderr)
    
    # Cognitive evaluation
    if cognitive_models:
        st.subheader("Cognitive Capability Evaluation")
        
        cognitive_model = st.selectbox("Select Cognitive Model", cognitive_models)
        
        if st.button("Evaluate Cognitive Capabilities"):
            with st.spinner("Evaluating cognitive capabilities..."):
                cmd = [
                    "python", "cognitive_cli.py", "cognitive-eval",
                    "--model", cognitive_model,
                    "--data", test_data,
                    "--output", f"{output_dir}/cognitive_capabilities",
                    "--batch_size", str(batch_size),
                    "--seq_length", str(seq_length)
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    st.success("Cognitive evaluation completed successfully")
                    
                    # Try to load the results
                    result_path = find_latest_file(f"{output_dir}/cognitive_capabilities", ".json")
                    if result_path:
                        with open(result_path, "r") as f:
                            results = json.load(f)
                        
                        # Display cognitive metrics
                        st.write("Cognitive Capability Scores:")
                        
                        if "capability_scores" in results:
                            scores = results["capability_scores"]
                            
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.metric("Memory", f"{scores.get('memory_capability', 0):.2f}")
                            
                            with col2:
                                st.metric("Regime Detection", f"{scores.get('regime_detection', 0):.2f}")
                            
                            with col3:
                                st.metric("Cross-Dimensional", f"{scores.get('cross_dimensional', 0):.2f}")
                            
                            with col4:
                                st.metric("Forecasting", f"{scores.get('forecasting', 0):.2f}")
                else:
                    st.error("Error during cognitive evaluation")
                    st.text(result.stderr)
    
    # Benchmark comparison
    if cognitive_models and baseline_models:
        st.subheader("Progressive Benchmark")
        
        col1, col2 = st.columns(2)
        
        with col1:
            cognitive_model = st.selectbox("Select Cognitive Model for Benchmark", cognitive_models)
        
        with col2:
            baseline_model = st.selectbox("Select Baseline Model", baseline_models)
        
        if st.button("Run Benchmark"):
            with st.spinner("Running progressive benchmark..."):
                cmd = [
                    "python", "cognitive_cli.py", "benchmark",
                    "--cognitive", cognitive_model,
                    "--baseline", baseline_model,
                    "--data", test_data,
                    "--output", f"{output_dir}/benchmark",
                    "--batch_size", str(batch_size),
                    "--seq_length", str(seq_length)
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    st.success("Benchmark completed successfully")
                    
                    # Display benchmark results
                    plot_path = os.path.join(output_dir, "benchmark/plots/advantage_progression.png")
                    if os.path.exists(plot_path):
                        st.image(plot_path)
                    
                    scores_path = os.path.join(output_dir, "benchmark/plots/scenario_scores.png")
                    if os.path.exists(scores_path):
                        st.image(scores_path)
                else:
                    st.error("Error during benchmark")
                    st.text(result.stderr)
    
    # Results comparison
    st.subheader("Results Comparison")
    
    # List evaluation results
    cognitive_results = list_files(f"{output_dir}/cognitive", ".json", recursive=True)
    baseline_results = list_files(f"{output_dir}/baseline", ".json", recursive=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        cognitive_result = st.selectbox("Select Cognitive Result", cognitive_results)
    
    with col2:
        baseline_result = st.selectbox("Select Baseline Result", baseline_results)
    
    if cognitive_result and baseline_result and st.button("Compare Results"):
        with st.spinner("Comparing results..."):
            cmd = [
                "python", "cognitive_cli.py", "visualize-results",
                "--cognitive", cognitive_result,
                "--baseline", baseline_result,
                "--output", f"{output_dir}/comparison"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                st.success("Comparison completed successfully")
                
                # Display comparison image
                comparison_path = os.path.join(output_dir, "comparison/financial_metrics_comparison.png")
                if os.path.exists(comparison_path):
                    st.image(comparison_path)
            else:
                st.error("Error during comparison")
                st.text(result.stderr)

def show_visualization_page():
    """Display the visualization page"""
    st.header("Model Visualization")
    
    # List available models
    models = list_model_files("models", ".pth")
    test_data = "data/processed/test.csv"
    
    st.subheader("Model Structure Visualization")
    
    selected_model = st.selectbox("Select Model", models)
    
    if selected_model and st.button("Visualize Model Architecture"):
        with st.spinner("Generating visualizations..."):
            output_dir = "evaluation/visualization"
            cmd = [
                "python", "cognitive_cli.py", "visualize",
                "--model", selected_model,
                "--data", test_data,
                "--output", output_dir,
                "--batch_size", "4",
                "--seq_length", "20"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                st.success("Visualization completed successfully")
                
                # Display key visualizations
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Attention Patterns")
                    attention_path = os.path.join(output_dir, "attention/attention_heatmap.png")
                    if os.path.exists(attention_path):
                        st.image(attention_path)
                
                with col2:
                    st.subheader("Memory Activity")
                    memory_path = os.path.join(output_dir, "memory/memory_activity_heatmap.png")
                    if os.path.exists(memory_path):
                        st.image(memory_path)
                
                st.subheader("Predictions")
                pred_paths = list_files(os.path.join(output_dir, "predictions"), ".png")
                for path in pred_paths:
                    st.image(path)
                
                st.subheader("Cognitive State Profile")
                profile_path = os.path.join(output_dir, "profile/cognitive_state_profile.png")
                if os.path.exists(profile_path):
                    st.image(profile_path)
            else:
                st.error("Error during visualization")
                st.text(result.stderr)
    
    # Custom visualization
    st.subheader("Custom Prediction Visualization")
    
    col1, col2 = st.columns(2)
    
    with col1:
        custom_model = st.selectbox("Select Model for Prediction", models, key="custom_model")
        prediction_data = st.text_input("Data for Prediction", test_data)
    
    with col2:
        window_size = st.slider("Visualization Window", 10, 100, 50)
        show_uncertainty = st.checkbox("Show Uncertainty", True)
    
    if custom_model and st.button("Generate Custom Visualization"):
        with st.spinner("Generating custom visualization..."):
            try:
                # Load model
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model = CognitiveArchitecture()
                model.load_state_dict(torch.load(custom_model, map_location=device))
                model.eval()
                
                # Load data
                data_loader = EnhancedFinancialDataLoader(
                    data_path=prediction_data,
                    sequence_length=window_size,
                    batch_size=1,
                    regime_aware=True,
                    augmentation=False
                )
                
                # Get a batch of data
                batch = next(iter(data_loader))
                features = batch['features'].to(device)
                sequence = batch['sequence'].to(device)
                targets = batch['target'].to(device)
                
                # Get predictions
                with torch.no_grad():
                    outputs = model(financial_data=features, financial_seq=sequence)
                
                # Create visualizer
                visualizer = CognitiveVisualizer(model)
                
                # Create prediction plot
                if isinstance(outputs, dict) and 'market_state' in outputs:
                    predictions = outputs['market_state']
                    uncertainty = outputs.get('uncertainty', None)
                else:
                    predictions = outputs
                    uncertainty = None
                
                fig = plt.figure(figsize=(12, 6))
                plt.plot(predictions.cpu().numpy()[0, :, 0], 'b-', label='Predicted', linewidth=2)
                plt.plot(targets.cpu().numpy()[0, :, 0], 'r-', label='Actual', linewidth=2)
                
                if show_uncertainty and uncertainty is not None:
                    # Add uncertainty bands
                    uncertainty_np = uncertainty.cpu().numpy()[0, :, 0]
                    pred_np = predictions.cpu().numpy()[0, :, 0]
                    plt.fill_between(
                        range(len(pred_np)),
                        pred_np - 2 * uncertainty_np,
                        pred_np + 2 * uncertainty_np,
                        color='blue',
                        alpha=0.2,
                        label='Uncertainty (±2σ)'
                    )
                
                plt.title("Price Prediction with Uncertainty")
                plt.xlabel("Time Step")
                plt.ylabel("Price")
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                st.pyplot(fig)
                
                # Display attention patterns
                if 'attention_weights' in outputs:
                    attention_weights = outputs['attention_weights'].cpu().numpy()
                    fig = plt.figure(figsize=(10, 8))
                    plt.imshow(attention_weights, cmap='viridis')
                    plt.colorbar(label='Attention Weight')
                    plt.title("Attention Weights")
                    plt.xlabel("Sequence Position")
                    plt.ylabel("Feature")
                    st.pyplot(fig)
                
                # Display regime probabilities if available
                if 'regime_probabilities' in outputs:
                    regime_probs = outputs['regime_probabilities'].cpu().numpy()[0]
                    fig = plt.figure(figsize=(8, 5))
                    plt.bar(['Bull', 'Bear', 'Sideways'], regime_probs)
                    plt.title("Regime Probabilities")
                    plt.ylabel("Probability")
                    st.pyplot(fig)
                
            except Exception as e:
                st.error(f"Error generating visualization: {e}")

    if st.button("Visualize Information Flow"):
        with st.spinner("Generating information flow visualization..."):
            # Get sample data
            sample_data = next(iter(data_loader))
            financial_data, financial_seq, _ = sample_data
            
            # Move to device
            financial_data = financial_data.to(device)
            financial_seq = financial_seq.to(device)
            
            # Create model introspector
            introspector = Introspection(model)
            
            # Generate visualizations
            viz_paths, _ = introspector.visualize_information_flow(
                financial_data, financial_seq,
                output_dir="evaluation/information_flow"
            )
            
            # Display visualizations
            st.subheader("Information Flow Visualization")
            st.image(viz_paths["flow_diagram"])
            
            st.subheader("Component Activation Analysis")
            st.image(viz_paths["component_activations"])

def show_live_demo():
    """Display the live demo page"""
    st.header("Live Forecasting Demo")
    
    # List available models
    models = list_model_files("models", ".pth")
    
    st.subheader("Real-time Forecasting")
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_model = st.selectbox("Select Model", models)
        ticker = st.text_input("Ticker Symbol", "SPY")
        days = st.number_input("Historical Days", 30, 365, 60)
    
    with col2:
        forecast_horizon = st.slider("Forecast Horizon (Days)", 1, 30, 5)
        show_uncertainty = st.checkbox("Show Uncertainty", True)
        show_regimes = st.checkbox("Show Regime Detection", True)
    
    if selected_model and st.button("Generate Forecast"):
        with st.spinner("Fetching data and generating forecast..."):
            try:
                # Download latest data
                import yfinance as yf
                end_date = datetime.now()
                start_date = end_date - timedelta(days=days+10)  # Extra days for features
                
                # Get data
                stock_data = yf.download(ticker, start=start_date, end=end_date)
                
                if len(stock_data) == 0:
                    st.error(f"No data found for ticker {ticker}")
                    return
    
                # Calculate features
                stock_data['returns'] = stock_data['Close'].pct_change()
                stock_data['volatility'] = stock_data['returns'].rolling(window=20).std()
                stock_data['volume_change'] = stock_data['Volume'].pct_change()
                stock_data.dropna(inplace=True)
                
                # Prepare data for model
                features = stock_data[['Open', 'High', 'Low', 'Close', 'Volume', 'returns', 'volatility', 'volume_change']].values
                seq_len = min(20, len(features))
                
                # Normalize data
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                features = scaler.fit_transform(features)
                
                # Create sequence
                sequence = np.array([features[-seq_len:]])
                
                # Load model
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model = CognitiveArchitecture()
                model.load_state_dict(torch.load(selected_model, map_location=device))
                model.eval()
                
                # Convert to torch tensors
                sequence_tensor = torch.tensor(sequence, dtype=torch.float32).to(device)
                
                # Get predictions
                with torch.no_grad():
                    outputs = model(financial_seq=sequence_tensor)
                
                # Extract predictions
                if isinstance(outputs, dict) and 'market_state' in outputs:
                    predictions = outputs['market_state'].cpu().numpy()[0]
                    uncertainty = outputs.get('uncertainty', None)
                    if uncertainty is not None:
                        uncertainty = uncertainty.cpu().numpy()[0]
                    
                    regime_probs = None
                    if 'regime_probabilities' in outputs:
                        regime_probs = outputs['regime_probabilities'].cpu().numpy()[0]
                else:
                    predictions = outputs.cpu().numpy()[0]
                    uncertainty = None
                    regime_probs = None
                
                # Scale predictions back
                last_close = stock_data['Close'].iloc[-1]
                predicted_returns = predictions[:, 0]  # Assume first column is returns
                predicted_prices = [last_close]
                
                for ret in predicted_returns:
                    predicted_prices.append(predicted_prices[-1] * (1 + ret))
                
                # Create dates for forecast
                last_date = stock_data.index[-1]
                forecast_dates = [last_date]
                for i in range(forecast_horizon):
                    # Skip weekends
                    next_date = forecast_dates[-1] + timedelta(days=1)
                    while next_date.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
                        next_date += timedelta(days=1)
                    forecast_dates.append(next_date)
                
                # Create plot
                fig, ax = plt.subplots(figsize=(12, 6))
                
                # Plot historical data
                hist_dates = stock_data.index[-30:]  # Last 30 days
                hist_prices = stock_data['Close'].iloc[-30:].values
                ax.plot(hist_dates, hist_prices, 'b-', label='Historical', linewidth=2)
                
                # Plot forecast
                ax.plot(forecast_dates, predicted_prices, 'r-', label='Forecast', linewidth=2)
                
                # Add uncertainty if available
                if show_uncertainty and uncertainty is not None:
                    lower_bound = []
                    upper_bound = []
                    for i, price in enumerate(predicted_prices):
                        if i < len(uncertainty):
                            lower_bound.append(price * (1 - 2 * uncertainty[i, 0]))
                            upper_bound.append(price * (1 + 2 * uncertainty[i, 0]))
                        else:
                            lower_bound.append(price)
                            upper_bound.append(price)
                    
                    ax.fill_between(
                        forecast_dates,
                        lower_bound,
                        upper_bound,
                        color='red',
                        alpha=0.2,
                        label='Uncertainty (±2σ)'
                    )
                
                ax.set_title(f"{ticker} Price Forecast")
                ax.set_xlabel("Date")
                ax.set_ylabel("Price")
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                # Format x-axis
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                st.pyplot(fig)
    
                # Display regime information if available
                if show_regimes and regime_probs is not None:
                    st.subheader("Market Regime Analysis")
                    
                    regime_names = ['Bull Market', 'Bear Market', 'Sideways/Consolidation']
                    
                    fig, ax = plt.subplots(figsize=(8, 5))
                    ax.bar(regime_names, regime_probs)
                    ax.set_title("Current Market Regime Probabilities")
                    ax.set_ylabel("Probability")
                    plt.tight_layout()
                    
                    st.pyplot(fig)

                    # Determine dominant regime
                    dominant_regime = regime_names[np.argmax(regime_probs)]
                    st.info(f"Dominant Regime: {dominant_regime} ({regime_probs.max():.2%} probability)")
                    
                    # Provide regime-specific insights
                    if dominant_regime == 'Bull Market':
                        st.success("Bull Market: Strong upward trend, positive momentum")
                    elif dominant_regime == 'Bear Market':
                        st.error("Bear Market: Downward trend, negative sentiment")
                    else:
                        st.warning("Sideways/Consolidation: Range-bound with no clear direction")
                
                # Display forecast values
                st.subheader("Forecast Values")
                
                forecast_df = pd.DataFrame({
                    'Date': forecast_dates,
                    'Predicted Price': predicted_prices
                })
                
                if uncertainty is not None:
                    forecast_df['Lower Bound'] = lower_bound
                    forecast_df['Upper Bound'] = upper_bound
                
                st.dataframe(forecast_df)
                
            except Exception as e:
                st.error(f"Error generating forecast: {e}")
                import traceback
                st.text(traceback.format_exc())

def count_files(directory, extension=None):
    """Count files in directory (recursively) with optional extension filter"""
    count = 0
    if not os.path.exists(directory):
        return count
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if extension is None or file.endswith(extension):
                count += 1
    
    return count

def list_data_files(directories):
    """List data files in the specified directories"""
    files = []
    for directory in directories:
        if os.path.exists(directory):
            for file in os.listdir(directory):
                if file.endswith('.csv'):
                    files.append(os.path.join(directory, file))
    return files

def list_model_files(directory, extension='.pth'):
    """List model files in the specified directory"""
    files = []
    if os.path.exists(directory):
        for root, dirs, files_list in os.walk(directory):
            for file in files_list:
                if file.endswith(extension):
                    files.append(os.path.join(root, file))
    return files

def list_files(directory, extension, recursive=False):
    """List files with the specified extension in the directory"""
    files = []
    if not os.path.exists(directory):
        return files
    
    if recursive:
        for root, dirs, files_list in os.walk(directory):
            for file in files_list:
                if file.endswith(extension):
                    files.append(os.path.join(root, file))
    else:  # Aligned with 'if recursive:'
        for file in os.listdir(directory):
            if file.endswith(extension):
                files.append(os.path.join(directory, file))
    
    return files

def find_latest_file(directory, extension):
    """Find the most recently modified file with the specified extension"""
    files = list_files(directory, extension, recursive=True)
    if not files:
        return None
    
    return max(files, key=os.path.getmtime)

def create_training_tab():
    with gr.Tab("Training"):
        gr.Markdown("## Model Training")
        
        with gr.Row():
            with gr.Column(scale=1):
                model_type = gr.Dropdown(
                    ["baseline", "cognitive", "regime_aware_cognitive"], 
                    label="Model Type", 
                    value="cognitive"
                )
                
                training_mode = gr.Radio(
                    ["Regular", "Progressive", "Component"], 
                    label="Training Mode",
                    value="Progressive"
                )
                
                with gr.Accordion("Model Configuration", open=False):
                    input_dim = gr.Slider(5, 100, value=20, step=1, label="Input Dimension")
                    hidden_dim = gr.Slider(16, 256, value=64, step=8, label="Hidden Dimension")
                    num_layers = gr.Slider(1, 4, value=2, step=1, label="Number of Layers")
                    dropout = gr.Slider(0.0, 0.5, value=0.2, step=0.05, label="Dropout Rate")
                    memory_size = gr.Slider(10, 500, value=100, step=10, label="Memory Size")
                    num_regimes = gr.Slider(2, 10, value=3, step=1, label="Number of Regimes")
                
                with gr.Accordion("Training Parameters", open=True):
                    learning_rate = gr.Slider(0.0001, 0.01, value=0.001, step=0.0001, label="Learning Rate")
                    batch_size = gr.Slider(8, 128, value=32, step=8, label="Batch Size")
                    epochs = gr.Slider(5, 200, value=50, step=5, label="Epochs")
                    sequence_length = gr.Slider(5, 60, value=20, step=1, label="Sequence Length")
                    
                    # Conditional UI for component training
                    component_name = gr.Dropdown(
                        ["memory", "attention", "regime_detector"], 
                        label="Component to Train",
                        value="memory",
                        visible=False
                    )
                    
                    # Show component selector only when Component training is selected
                    def update_component_visibility(mode):
                        return gr.update(visible=(mode == "Component"))
                    
                    training_mode.change(
                        update_component_visibility,
                        inputs=[training_mode],
                        outputs=[component_name]
                    )
                
                # Add training data selection
                train_data = gr.File(label="Training Data (CSV)")
                
                # Add validation data selection that appears only when Progressive training is selected
                val_data = gr.File(label="Validation Data (CSV)", visible=False)
                
                def update_val_data_visibility(mode):
                    return gr.update(visible=(mode == "Progressive"))
                
                training_mode.change(
                    update_val_data_visibility,
                    inputs=[training_mode],
                    outputs=[val_data]
                )
                
                output_dir = gr.Textbox(label="Output Directory", value="models/latest")
                
            with gr.Column(scale=2):
                output_log = gr.Textbox(label="Training Log", lines=20)
                progress = gr.Plot(label="Training Progress")
                
        with gr.Row():
            start_button = gr.Button("Start Training", variant="primary")
            stop_button = gr.Button("Stop Training", variant="stop")
        
        def start_training(model_type, training_mode, input_dim, hidden_dim, num_layers, 
                          dropout, memory_size, num_regimes, learning_rate, batch_size, 
                          epochs, sequence_length, component_name, train_data, val_data, output_dir):
            if not train_data or not os.path.exists(train_data.name):
                return "Error: Please upload a valid training data file", None
                
            os.makedirs(output_dir, exist_ok=True)
            
            # Create config file
            config = {
                "input_dim": input_dim,
                "hidden_dim": hidden_dim,
                "num_layers": num_layers,
                "dropout": dropout,
                "memory_size": memory_size,
                "num_regimes": num_regimes,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "epochs": epochs,
                "sequence_length": sequence_length
            }
            
            config_path = os.path.join(output_dir, "config.json")
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)
                
            # Select the appropriate script based on training mode
            if training_mode == "Regular":
                script_path = "scripts/train_baseline.py"
            elif training_mode == "Progressive":
                script_path = "scripts/train_progressive.py"
            else:  # Component
                script_path = "scripts/train_component.py"
                
            # Build command
            cmd = [sys.executable, script_path]
            
            if training_mode == "Component":
                cmd.extend([
                    "--component", component_name,
                    "--data_path", train_data.name,
                    "--output_dir", output_dir,
                    "--batch_size", str(batch_size),
                    "--epochs", str(epochs),
                    "--learning_rate", str(learning_rate)
                ])
            elif training_mode == "Progressive":
                # For progressive training, we need explicit train and val paths
                
                # Path to the uploaded training data file
                if hasattr(train_data, 'name') and os.path.exists(train_data.name):
                    train_data_path = os.path.abspath(train_data.name)
                    print(f"Training data path from upload: {train_data_path}")
                else:
                    # Default path if no file uploaded
                    train_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                               "data", "enhanced_features_train.csv")
                    if not os.path.exists(train_data_path):
                        train_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                                  "data", "train.csv")
                    print(f"Using default training data path: {train_data_path}")
                
                # Check if validation data was provided via UI
                if val_data and hasattr(val_data, 'name') and os.path.exists(val_data.name):
                    val_data_path = os.path.abspath(val_data.name)
                    print(f"Validation data path from upload: {val_data_path}")
                else:
                    # Try to find the validation file by replacing train with val in the filename
                    val_data_path = train_data_path.replace("_train", "_val")
                    if not os.path.exists(val_data_path):
                        val_data_path = train_data_path.replace("train", "val")
                    
                    # If still not found, look in the data directory for a val.csv file
                    if not os.path.exists(val_data_path):
                        val_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                                "data", "enhanced_features_val.csv")
                    if not os.path.exists(val_data_path):
                        val_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                                "data", "val.csv")
                    
                    # Print the validation path being used
                    if os.path.exists(val_data_path):
                        print(f"Using validation data at: {val_data_path}")
                    else:
                        return f"Error: Validation file not found. Please ensure you have a validation file available.", None
                
                # Use train_progressive.py directly instead of the wrapper
                script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts", "train_progressive.py")
                
                print(f"Using script at: {script_path}")
                print(f"Train data: {train_data_path}")
                print(f"Val data: {val_data_path}")
                
                # Create the output directory if it doesn't exist
                os.makedirs(output_dir, exist_ok=True)
                
                # Build the command with explicit arguments
                cmd = [
                    sys.executable,
                    script_path,
                    "--train_data", train_data_path,
                    "--val_data", val_data_path,
                    "--hidden_dim", str(hidden_dim),
                    "--memory_size", str(memory_size),
                    "--sequence_length", str(sequence_length),
                    "--batch_size", str(batch_size),
                    "--learning_rate", str(learning_rate),
                    "--epochs", str(epochs),
                    "--output_dir", output_dir
                ]
                
                # Add optional flags if needed
                # cmd.extend(["--start_easy"])
                # cmd.extend(["--increase_difficulty"])
                
                print(f"Command: {' '.join(cmd)}")
                
                # Create log output
                log_output = f"Starting progressive training:\n"
                log_output += f"- Python interpreter: {sys.executable}\n"
                log_output += f"- Training data: {train_data_path}\n"
                log_output += f"- Validation data: {val_data_path}\n"
                log_output += f"- Fixed data loading and tensor shape issues\n"
                
                # Run the command directly
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.STDOUT,
                    text=True, 
                    bufsize=1, 
                    universal_newlines=True
                )
                
                # Output initial log
                yield log_output, None
                
                # Process output
                for line in iter(process.stdout.readline, ""):
                    log_output += line + "\n"
                    yield log_output, None
                    
                process.wait()
                
                if process.returncode == 0:
                    log_output += f"\nTraining completed successfully. Model saved to {output_dir}"
                else:
                    log_output += f"\nTraining failed with return code {process.returncode}"
                    
                yield log_output, None
                return
            
            else:
                cmd.extend([
                    "--model", model_type,
                    "--config", config_path,
                    "--data_path", train_data.name,
                    "--output_dir", output_dir,
                    "--batch_size", str(batch_size),
                    "--epochs", str(epochs),
                    "--sequence_length", str(sequence_length)
                ])
            
            # Print the full command for debugging
            print(f"Running command: {' '.join(cmd)}")
            
            # Run training process
            process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, universal_newlines=True
            )
            
            # Initialize log_output with command info
            if training_mode == "Progressive":
                log_output = f"Starting progressive training with:\n"
                log_output += f"- Training data: {train_data_path}\n"
                log_output += f"- Validation data: {val_data_path}\n"
            else:
                log_output = f"Starting training with command:\n{' '.join(cmd)}\n\n"
            
            # Output initial log
            yield log_output, None
            
            losses = []
            val_losses = []
            
            for line in iter(process.stdout.readline, ""):
                log_output += line + "\n"
                
                # Update plot if it contains loss information
                if "Epoch" in line and "Train Loss" in line:
                    try:
                        # Extract epoch and loss values
                        parts = line.split()
                        epoch_idx = parts.index("Epoch")
                        epoch = int(parts[epoch_idx + 1].strip(':,'))
                        
                        train_idx = parts.index("Train")
                        train_loss = float(parts[train_idx + 2].strip(','))
                        losses.append(train_loss)
                        
                        val_idx = parts.index("Val")
                        val_loss = float(parts[val_idx + 2])
                        val_losses.append(val_loss)
                        
                        # Create plot
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.plot(range(1, len(losses) + 1), losses, 'b-', label='Training Loss')
                        ax.plot(range(1, len(val_losses) + 1), val_losses, 'r-', label='Validation Loss')
                        ax.set_xlabel('Epoch')
                        ax.set_ylabel('Loss')
                        ax.set_title('Training Progress')
                        ax.legend()
                        ax.grid(True)
                        
                        yield log_output, fig
                    except (ValueError, IndexError) as e:
                        print(f"DEBUG - Error parsing loss values: {e}")
                        pass
                        
                    yield log_output, None
                    
                yield log_output, None
                
            process.wait()
            
            if process.returncode == 0:
                log_output += f"\nTraining completed successfully. Model saved to {output_dir}"
            else:
                log_output += f"\nTraining failed with return code {process.returncode}"
                
            yield log_output, None
            
        def stop_training():
            # Find and kill training processes
            os.system("pkill -f 'scripts/train'")
            return "Training stopped by user", None
            
        start_button.click(
            start_training,
            inputs=[model_type, training_mode, input_dim, hidden_dim, num_layers, 
                   dropout, memory_size, num_regimes, learning_rate, batch_size, 
                   epochs, sequence_length, component_name, train_data, val_data, output_dir],
            outputs=[output_log, progress]
        )
        
        stop_button.click(
            stop_training,
            inputs=None,
            outputs=[output_log, progress]
        )
        
    return output_dir

def create_evaluation_tab():
    with gr.Tab("Evaluation"):
        gr.Markdown("## Model Evaluation")
        
        with gr.Row():
            with gr.Column(scale=1):
                model_path = gr.Dropdown(
                    choices=list_model_files("models"), 
                    label="Model Checkpoint",
                    value="models/latest/best_model.pt",
                    interactive=True,
                    allow_custom_value=True
                )
                refresh_models = gr.Button("Refresh Models")
                
                model_type = gr.Dropdown(
                    ["baseline", "cognitive"], 
                    label="Model Type", 
                    value="cognitive"
                )
                
                data_path = gr.File(label="Test Data (CSV)")
                
                with gr.Accordion("Evaluation Options", open=True):
                    eval_type = gr.Radio(
                        ["Standard Metrics", "Regime-Specific", "Stress Test"], 
                        label="Evaluation Type",
                        value="Standard Metrics"
                    )
                    
                    compare_model = gr.Dropdown(
                        choices=["None"] + list_model_files("models"), 
                        label="Comparison Model Checkpoint (Optional)",
                        value="None"
                    )
                    
                    compare_type = gr.Dropdown(
                        ["baseline", "cognitive"], 
                        label="Comparison Model Type", 
                        value="baseline"
                    )
                    
                    stress_scenarios = gr.CheckboxGroup(
                        ["Market Crash", "High Volatility", "Regime Change", "Novel Pattern"],
                        label="Stress Test Scenarios"
                    )
                
                output_dir = gr.Textbox(label="Output Directory", value="evaluation/latest")
                
            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.Tab("Logs"):
                        output_log = gr.Textbox(label="Evaluation Log", lines=15)
                    
                    with gr.Tab("Results"):
                        metrics_display = gr.JSON(label="Metrics")
                        
                    with gr.Tab("Visualization"):
                        result_image = gr.Image(label="Result Visualization", show_label=True)
                        
                        viz_dropdown = gr.Dropdown(
                            label="Select Visualization",
                            choices=[],
                            interactive=True
                        )
        
        with gr.Row():
            start_button = gr.Button("Start Evaluation", variant="primary")
            
        def refresh_model_list():
            return gr.Dropdown.update(choices=list_model_files("models"))
            
        refresh_models.click(
            refresh_model_list,
            inputs=None,
            outputs=[model_path, compare_model]
        )
        
        def start_evaluation(model_path, model_type, data_path, eval_type, 
                            compare_model, compare_type, stress_scenarios, output_dir):
            if not model_path:
                return "Error: Please select a model checkpoint", None, None, []
                
            if not data_path or not os.path.exists(data_path.name):
                return "Error: Please upload a valid test data file", None, None, []
                
            os.makedirs(output_dir, exist_ok=True)
            log_output = f"Starting {eval_type} evaluation...\n"
            
            # Build command based on evaluation type
            if eval_type == "Standard Metrics":
                # Use the selected comparison model if not "None", otherwise use the baseline model
                baseline_path = compare_model if compare_model != "None" else "models/baseline/baseline.pth"
                cmd = [
                    sys.executable, "scripts/analyze_performance.py",
                    "--cognitive", model_path,
                    "--baseline", baseline_path,
                    "--test_data", data_path.name,
                    "--seq_length", "20",
                    "--batch", "8",
                    "--output", output_dir
                ]
                    
            elif eval_type == "Regime-Specific":
                cmd = [
                    sys.executable, "scripts/evaluate_by_regime.py",
                    "--model_path", model_path,
                    "--model_type", model_type,
                    "--data_path", data_path.name,
                    "--output_dir", output_dir,
                    "--sequence_length", "20",
                    "--batch_size", "32"
                ]
                
                if compare_model != "None":
                    cmd.extend(["--compare_with", compare_model, "--compare_type", compare_type])
                    
            elif eval_type == "Stress Test":
                if not stress_scenarios:
                    return "Error: Please select at least one stress test scenario", None, None, []
                    
                cmd = [
                    sys.executable, "scripts/stress_test.py",
                    "--model_path", model_path,
                    "--model_type", model_type,
                    "--test_data", data_path.name,
                    "--output_dir", output_dir,
                    "--scenarios", ",".join(stress_scenarios)
                ]
            
            # Run evaluation process
            process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, universal_newlines=True
            )
            
            for line in iter(process.stdout.readline, ""):
                log_output += line
                yield log_output, None, None, []
            
            process.wait()
            
            if process.returncode == 0:
                log_output += f"\nEvaluation completed successfully. Results saved to {output_dir}"
                
                # Load metrics from output directory
                metrics_path = ""
                if eval_type == "Standard Metrics":
                    metrics_path = os.path.join(output_dir, "comparison.json")
                elif eval_type == "Regime-Specific":
                    metrics_path = os.path.join(output_dir, "regime_metrics.json")
                elif eval_type == "Stress Test":
                    metrics_path = os.path.join(output_dir, "stress_test_results.json")
                
                metrics = {}
                if os.path.exists(metrics_path):
                    with open(metrics_path, "r") as f:
                        metrics = json.load(f)
                
                # Find visualizations
                viz_files = []
                for ext in ['.png', '.jpg', '.jpeg']:
                    viz_files.extend(list(Path(output_dir).glob(f"*{ext}")))
                viz_files = [str(f) for f in viz_files]
                
                first_image = viz_files[0] if viz_files else None
                
                yield log_output, metrics, first_image, viz_files
            else:
                log_output += f"\nEvaluation failed with return code {process.returncode}"
                yield log_output, None, None, []
                
        def update_visualization(viz_path):
            return viz_path
                
        start_button.click(
            start_evaluation,
            inputs=[model_path, model_type, data_path, eval_type, 
                   compare_model, compare_type, stress_scenarios, output_dir],
            outputs=[output_log, metrics_display, result_image, viz_dropdown]
        )
        
        viz_dropdown.change(
            update_visualization,
            inputs=[viz_dropdown],
            outputs=[result_image]
        )
    
    return output_dir

def create_monitoring_tab():
    with gr.Tab("Monitoring"):
        gr.Markdown("## Model Monitoring & Analysis")
        
        def refresh_model_list():
            return gr.Dropdown.update(choices=list_model_files("models"))
        
        with gr.Tabs():
            with gr.Tab("Introspection"):
                with gr.Row():
                    with gr.Column(scale=1):
                        model_path = gr.Dropdown(
                            choices=list_model_files("models"), 
                            label="Model Checkpoint",
                            value="",
                            interactive=True,
                            allow_custom_value=True
                        )
                        refresh_models = gr.Button("Refresh Models")
                        
                        data_path = gr.File(label="Sample Data (CSV)")
                        num_samples = gr.Slider(1, 50, value=10, step=1, label="Number of Samples")
                        output_dir = gr.Textbox(label="Output Directory", value="monitoring/introspect")
                        
                        introspect_button = gr.Button("Analyze Model", variant="primary")
                        
                    with gr.Column(scale=2):
                        with gr.Tabs():
                            with gr.Tab("Flow Visualization"):
                                flow_image = gr.Image(label="Information Flow", height=500)
                                
                            with gr.Tab("Component Analysis"):
                                component_tabs = gr.Tabs()
                                with component_tabs:
                                    with gr.Tab("Attention"):
                                        attention_image = gr.Image(label="Attention Analysis")
                                    with gr.Tab("Memory"):
                                        memory_image = gr.Image(label="Memory Analysis")
                                    with gr.Tab("Regime"):
                                        regime_image = gr.Image(label="Regime Analysis")
                                        
                            with gr.Tab("Report"):
                                introspect_log = gr.Textbox(label="Analysis Log", lines=10)
                                report_json = gr.JSON(label="Full Report")
                                
            with gr.Tab("Early Warning"):
                with gr.Row():
                    with gr.Column(scale=1):
                        ew_model_path = gr.Dropdown(
                            choices=list_model_files("models"), 
                            label="Model Checkpoint",
                            value="",
                            interactive=True,
                            allow_custom_value=True
                        )
                        
                        ew_data_path = gr.File(label="Monitoring Data (CSV)")
                        lookback = gr.Slider(10, 100, value=30, step=5, label="Lookback Window")
                        threshold = gr.Slider(1.0, 5.0, value=2.5, step=0.1, label="Warning Threshold")
                        
                        simulation_speed = gr.Slider(0.1, 5.0, value=0.5, step=0.1, 
                                                   label="Simulation Speed (seconds/sample)")
                        
                        ew_output_dir = gr.Textbox(label="Output Directory", value="monitoring/early_warning")
                        
                        with gr.Row():
                            start_ew_button = gr.Button("Start Monitoring", variant="primary")
                            stop_ew_button = gr.Button("Stop Monitoring", variant="stop")
                        
                    with gr.Column(scale=2):
                        with gr.Tabs():
                            with gr.Tab("Live Status"):
                                status_log = gr.Textbox(label="Monitoring Log", lines=15)
                                warning_count = gr.Number(label="Warnings Detected", value=0)
                                
                            with gr.Tab("Warning Visualization"):
                                warning_image = gr.Image(label="Warning Visualization")
                                warning_dropdown = gr.Dropdown(
                                    label="Select Warning Event",
                                    choices=[],
                                    interactive=True
                                )
                                
                            with gr.Tab("Summary"):
                                summary_json = gr.JSON(label="Warning Summary")
            
            with gr.Tab("Information Flow"):
                with gr.Row():
                    with gr.Column(scale=1):
                        flow_model_path = gr.Dropdown(
                            choices=list_model_files("models"), 
                            label="Model Checkpoint",
                            value="",
                            interactive=True,
                            allow_custom_value=True
                        )
                        
                        flow_data_path = gr.File(label="Sample Data (CSV)")
                        flow_samples = gr.Slider(1, 20, value=5, step=1, label="Number of Samples")
                        flow_output_dir = gr.Textbox(label="Output Directory", value="visualizations/flow")
                        
                        generate_flow_button = gr.Button("Generate Flow Visualization", variant="primary")
                        
                    with gr.Column(scale=2):
                        with gr.Tabs():
                            with gr.Tab("Average Flow"):
                                avg_flow_image = gr.Image(label="Average Information Flow", height=500)
                                
                            with gr.Tab("Sample Flows"):
                                sample_flow_dropdown = gr.Dropdown(
                                    label="Select Sample",
                                    choices=[],
                                    interactive=True
                                )
                                sample_flow_image = gr.Image(label="Sample Flow", height=500)
                                
                            with gr.Tab("Flow Data"):
                                flow_data_json = gr.JSON(label="Flow Analysis Data")
                
        # Introspection functionality
        def run_introspection(model_path, data_path, num_samples, output_dir):
            if not model_path:
                return "Error: Please select a model checkpoint", None, None, None, None, None
                
            if not data_path or not os.path.exists(data_path.name):
                return "Error: Please upload a valid data file", None, None, None, None, None
                
            os.makedirs(output_dir, exist_ok=True)
            log_output = f"Starting model introspection with {num_samples} samples...\n"
                
            cmd = [
                sys.executable, "scripts/setup_monitoring.py",
                "--model_path", model_path,
                "--data_path", data_path.name,
                "--output_dir", output_dir,
                "--samples", str(num_samples)
            ]
                
            process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, universal_newlines=True
            )
                
            for line in iter(process.stdout.readline, ""):
                log_output += line
                yield log_output, None, None, None, None, None
                
            process.wait()
                
            if process.returncode == 0:
                log_output += f"\nIntrospection completed successfully. Results saved to {output_dir}"
                
                # Load report
                report_path = os.path.join(output_dir, "introspection_report.json")
                report = {}
                if os.path.exists(report_path):
                    with open(report_path, "r") as f:
                        report = json.load(f)
                
                # Load images
                flow_img_path = os.path.join(output_dir, "flow", "average_information_flow.png")
                attention_img_path = os.path.join(output_dir, "attention_analysis.png")
                memory_img_path = os.path.join(output_dir, "memory_analysis.png")
                regime_img_path = os.path.join(output_dir, "regime_analysis.png")
                
                # Some images might not exist if model doesn't have all components
                flow_img = flow_img_path if os.path.exists(flow_img_path) else None
                attention_img = attention_img_path if os.path.exists(attention_img_path) else None
                memory_img = memory_img_path if os.path.exists(memory_img_path) else None
                regime_img = regime_img_path if os.path.exists(regime_img_path) else None
                
                yield log_output, report, flow_img, attention_img, memory_img, regime_img
            else:
                log_output += f"\nIntrospection failed with return code {process.returncode}"
                yield log_output, None, None, None, None, None
        
        refresh_models.click(
            refresh_model_list,
            inputs=None,
            outputs=[model_path, ew_model_path, flow_model_path]
        )
        
        introspect_button.click(
            run_introspection,
            inputs=[model_path, data_path, num_samples, output_dir],
            outputs=[introspect_log, report_json, flow_image, attention_image, memory_image, regime_image]
        )
        
        # Early Warning functionality
        def start_early_warning(model_path, data_path, lookback, threshold, simulation_speed, output_dir):
            if not model_path:
                return "Error: Please select a model checkpoint", 0, None, [], None
                
            if not data_path or not os.path.exists(data_path.name):
                return "Error: Please upload a valid data file", 0, None, [], None
                
            os.makedirs(output_dir, exist_ok=True)
            log_output = f"Starting early warning monitoring...\n"
                
            cmd = [
                sys.executable, "scripts/monitor_live.py",
                "--model_path", model_path,
                "--data_path", data_path.name,
                "--output_dir", output_dir,
                "--lookback_window", str(lookback),
                "--threshold", str(threshold),
                "--delay", str(simulation_speed)
            ]
                
            # Start the monitoring process
            process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, universal_newlines=True
            )
            
            warning_count = 0
            warning_files = []
            
            for line in iter(process.stdout.readline, ""):
                log_output += line
                
                # Check for warning signals in output
                if "Warning" in line or "warning" in line:
                    warning_count += 1
                    
                    # Check if new visualization was created
                    warning_files = []
                    img_dir = os.path.join(output_dir)
                    if os.path.exists(img_dir):
                        warning_files = [f for f in os.listdir(img_dir) if f.startswith("warning_") and f.endswith(".png")]
                        warning_files = [os.path.join(img_dir, f) for f in warning_files]
                        warning_files.sort(reverse=True)  # Latest first
                
                # Get latest warning image if available
                latest_warning_img = warning_files[0] if warning_files else None
                
                yield log_output, warning_count, latest_warning_img, warning_files, None
                
            process.wait()
            
            if process.returncode == 0:
                log_output += f"\nMonitoring completed successfully."
                
                # Load summary report
                summary_path = os.path.join(output_dir, "summary_report.json")
                summary = {}
                if os.path.exists(summary_path):
                    with open(summary_path, "r") as f:
                        summary = json.load(f)
                
                yield log_output, warning_count, latest_warning_img, warning_files, summary
            else:
                log_output += f"\nMonitoring stopped with return code {process.returncode}"
                yield log_output, warning_count, latest_warning_img, warning_files, None
                
        def stop_early_warning():
            # Find and kill monitoring processes
            os.system("pkill -f 'scripts/monitor_live.py'")
            return "Monitoring stopped by user", 0, None, [], None
            
        def update_warning_image(warning_path):
            return warning_path
                
        start_ew_button.click(
            start_early_warning,
            inputs=[ew_model_path, ew_data_path, lookback, threshold, simulation_speed, ew_output_dir],
            outputs=[status_log, warning_count, warning_image, warning_dropdown, summary_json]
        )
        
        stop_ew_button.click(
            stop_early_warning,
            inputs=None,
            outputs=[status_log, warning_count, warning_image, warning_dropdown, summary_json]
        )
        
        warning_dropdown.change(
            update_warning_image,
            inputs=[warning_dropdown],
            outputs=[warning_image]
        )
        
        # Information Flow visualization
        def generate_flow_visualization(model_path, data_path, num_samples, output_dir):
            if not model_path:
                return "Error: Please select a model checkpoint", None, None, [], None
                
            if not data_path or not os.path.exists(data_path.name):
                return "Error: Please upload a valid data file", None, None, [], None
                
            os.makedirs(output_dir, exist_ok=True)
                
            cmd = [
                sys.executable, "scripts/enhanced_visualization.py",
                "--model_path", model_path,
                "--data_path", data_path.name,
                "--output_dir", output_dir,
                "--samples", str(num_samples),
                "--type", "flow"
            ]
                
            process = subprocess.run(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, universal_newlines=True
            )
                
            if process.returncode == 0:
                # Find generated files
                avg_flow_path = os.path.join(output_dir, "average_information_flow.png")
                
                # Find sample flows
                sample_paths = []
                for i in range(num_samples):
                    path = os.path.join(output_dir, f"flow_sample_{i}.png")
                    if os.path.exists(path):
                        sample_paths.append(path)
                
                # Load flow data
                flow_data_path = os.path.join(output_dir, "flow_data.json")
                flow_data = {}
                if os.path.exists(flow_data_path):
                    with open(flow_data_path, "r") as f:
                        flow_data = json.load(f)
                
                first_sample = sample_paths[0] if sample_paths else None
                
                return process.stdout, avg_flow_path, first_sample, sample_paths, flow_data
            else:
                return f"Error: {process.stdout}", None, None, [], None
                
        def update_sample_flow(sample_path):
            return sample_path
                
        generate_flow_button.click(
            generate_flow_visualization,
            inputs=[flow_model_path, flow_data_path, flow_samples, flow_output_dir],
            outputs=[introspect_log, avg_flow_image, sample_flow_image, sample_flow_dropdown, flow_data_json]
        )
        
        sample_flow_dropdown.change(
            update_sample_flow,
            inputs=[sample_flow_dropdown],
            outputs=[sample_flow_image]
        )

def create_data_tab():
    with gr.Tab("Data Management"):
        gr.Markdown("## Data Preparation and Management")
        
        with gr.Tabs():
            with gr.Tab("Download Data"):
                with gr.Row():
                    with gr.Column():
                        tickers = gr.Textbox(label="Tickers (comma-separated)", 
                                           value="SPY,QQQ,AAPL,MSFT,AMZN")
                        start_date = gr.Textbox(label="Start Date (YYYY-MM-DD)", 
                                             value="2015-01-01")
                        end_date = gr.Textbox(label="End Date (YYYY-MM-DD)", 
                                           value="2023-12-31")
                        output_dir = gr.Textbox(label="Output Directory", 
                                             value="data/raw")
                        
                        include_features = gr.Checkbox(label="Include Technical Indicators", value=True)
                        detect_regimes = gr.Checkbox(label="Detect Market Regimes", value=True)
                        add_market_context = gr.Checkbox(label="Add Market Context", value=True)
                        
                        download_button = gr.Button("Download Data", variant="primary")
                        
                    with gr.Column():
                        download_log = gr.Textbox(label="Download Log", lines=15)
            
            with gr.Tab("Build Dataset"):
                with gr.Row():
                    with gr.Column():
                        input_dir = gr.Textbox(label="Input Directory (with ticker files)", 
                                             value="data/raw")
                        dataset_path = gr.Textbox(label="Output Dataset Path", 
                                               value="data/combined_financial.csv")
                        min_date = gr.Textbox(label="Minimum Date (YYYY-MM-DD)", 
                                           value="2015-01-01")
                        
                        add_context = gr.Checkbox(label="Add Market Context", value=True)
                        context_file = gr.Textbox(label="Market Context File", 
                                               value="data/raw/SPY.csv")
                        
                        build_button = gr.Button("Build Dataset", variant="primary")
                        
                    with gr.Column():
                        build_log = gr.Textbox(label="Build Log", lines=15)
                        dataset_preview = gr.Dataframe(label="Dataset Preview")
            
            with gr.Tab("Data Splitting"):
                with gr.Row():
                    with gr.Column():
                        dataset_file = gr.File(label="Dataset CSV File")
                        train_size = gr.Slider(0.5, 0.9, value=0.7, step=0.05, 
                                             label="Training Set Size")
                        val_size = gr.Slider(0.05, 0.3, value=0.15, step=0.05, 
                                           label="Validation Set Size")
                        preserve_regimes = gr.Checkbox(label="Preserve Regimes in Splits", value=True)
                        splits_dir = gr.Textbox(label="Output Directory", 
                                              value="data/splits")
                        
                        split_button = gr.Button("Split Dataset", variant="primary")
                        
                    with gr.Column():
                        split_log = gr.Textbox(label="Split Log", lines=10)
                        split_viz = gr.Image(label="Split Visualization")
            
            with gr.Tab("Feature Enhancement"):
                with gr.Row():
                    with gr.Column():
                        enhance_input = gr.File(label="Input Data CSV")
                        enhance_output = gr.Textbox(label="Output Path", 
                                                 value="data/enhanced_features.csv")
                        
                        with gr.Accordion("Feature Options", open=True):
                            add_technical = gr.Checkbox(label="Technical Indicators", value=True)
                            add_statistical = gr.Checkbox(label="Statistical Features", value=True)
                            add_cyclical = gr.Checkbox(label="Cyclical Features", value=True)
                            add_lagged = gr.Checkbox(label="Lagged Features", value=True)
                            lag_periods = gr.Slider(1, 10, value=5, step=1, label="Number of Lag Periods")
                        
                        enhance_button = gr.Button("Enhance Features", variant="primary")
                        
                    with gr.Column():
                        enhance_log = gr.Textbox(label="Enhancement Log", lines=10)
                        feature_preview = gr.Dataframe(label="Enhanced Data Preview")
        
        # Data download functionality
        def download_financial_data(tickers, start_date, end_date, output_dir, 
                                  include_features, detect_regimes, add_market_context):
            if not tickers or not start_date or not end_date:
                return "Error: Please provide ticker symbols, start date, and end date"
                
            os.makedirs(output_dir, exist_ok=True)
            log_output = f"Downloading data for: {tickers}...\n"
                
            cmd = [
                sys.executable, "download_data.py",  # This file is likely in the main directory
                "--tickers", tickers,
                "--start_date", start_date,
                "--end_date", end_date,
                "--output_dir", output_dir
            ]
                
            if include_features:
                cmd.append("--include_features")
                
            if detect_regimes:
                cmd.append("--detect_regimes")
                
            if add_market_context:
                cmd.append("--add_market_context")
                
            process = subprocess.run(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, universal_newlines=True
            )
                
            log_output += process.stdout
                
            if process.returncode == 0:
                log_output += f"\nData download completed successfully."
            else:
                log_output += f"\nData download failed with return code {process.returncode}"
                
            return log_output
        
        download_button.click(
            download_financial_data,
            inputs=[tickers, start_date, end_date, output_dir, 
                   include_features, detect_regimes, add_market_context],
            outputs=[download_log]
        )
        
        # Build dataset functionality
        def build_financial_dataset(input_dir, dataset_path, min_date, add_context, context_file):
            if not os.path.exists(input_dir):
                return "Error: Input directory does not exist", None
                
            os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
            log_output = f"Building dataset from {input_dir}...\n"
                
            cmd = [
                sys.executable, "scripts/build_dataset.py",
                "--input_dir", input_dir,
                "--output_path", dataset_path,
                "--min_date", min_date
            ]
                
            if add_context:
                cmd.extend(["--add_market_context", "--spy_file", context_file])
                
            process = subprocess.run(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, universal_newlines=True
            )
                
            log_output += process.stdout
                
            if process.returncode == 0:
                log_output += f"\nDataset built successfully. Saved to {dataset_path}"
                
                # Load preview
                if os.path.exists(dataset_path):
                    df = pd.read_csv(dataset_path)
                    preview = df.head(10)
                    return log_output, preview
            else:
                log_output += f"\nDataset build failed with return code {process.returncode}"
                
            return log_output, None
            
        build_button.click(
            build_financial_dataset,
            inputs=[input_dir, dataset_path, min_date, add_context, context_file],
            outputs=[build_log, dataset_preview]
        )
        
        # Split dataset functionality
        def split_financial_dataset(dataset_file, train_size, val_size, preserve_regimes, splits_dir):
            if not dataset_file or not os.path.exists(dataset_file.name):
                return "Error: Please upload a valid dataset file", None
                
            os.makedirs(splits_dir, exist_ok=True)
            log_output = f"Splitting dataset {dataset_file.name}...\n"
                
            cmd = [
                "python", "scripts/split_data.py",
                "--input_path", dataset_file.name,
                "--output_dir", splits_dir,
                "--train_ratio", str(train_size),
                "--val_ratio", str(val_size)
            ]
            if preserve_regimes:
                cmd.append("--ensure_regime_coverage")
                
            process = subprocess.run(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, universal_newlines=True
            )
                
            log_output += process.stdout
                
            if process.returncode == 0:
                log_output += f"\nDataset split successfully. Files saved to {splits_dir}"
                
                # Find visualization if it exists
                viz_path = os.path.join(splits_dir, "split_visualization.png")
                if os.path.exists(viz_path):
                    return log_output, viz_path
                else:
                    return log_output, None
            else:
                log_output += f"\nDataset split failed with return code {process.returncode}"
                
            return log_output, None
            
        split_button.click(
            split_financial_dataset,
            inputs=[dataset_file, train_size, val_size, preserve_regimes, splits_dir],
            outputs=[split_log, split_viz]
        )
        
        # Feature enhancement functionality
        def enhance_features(enhance_input, enhance_output, add_technical, add_statistical, 
                            add_cyclical, add_lagged, lag_periods):
            if not enhance_input:
                return "Error: Please upload a valid data file", None

            try:
                # Get the input file path
                input_file = enhance_input.name
                
                # Set output path and directory
                output_path = enhance_output if enhance_output else "data/enhanced_features.csv"
                
                # Log information about input and output
                log_output = f"Starting feature enhancement process:\n"
                log_output += f"- Input file: {input_file}\n"
                log_output += f"- Output path: {output_path}\n"
                log_output += f"- Options: technical={add_technical}, statistical={add_statistical}, cyclical={add_cyclical}, lagged={add_lagged} (periods={lag_periods})\n\n"
                
                # Make sure output_dir is a directory, not a file path
                if os.path.isdir(output_path):
                    output_dir = output_path
                    log_output += f"Output is a directory: {output_dir}\n"
                else:
                    output_dir = os.path.dirname(output_path)
                    if not output_dir:  # If output_path is just a filename with no directory
                        output_dir = "."
                    log_output += f"Output is a file: {output_path} (directory: {output_dir})\n"
                
                # Create output directory if it doesn't exist
                if output_dir and not output_path.lower().endswith('.csv'):
                    os.makedirs(output_dir, exist_ok=True)
                    log_output += f"Created output directory: {output_dir}\n"
                elif output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                    log_output += f"Created parent directory: {output_dir}\n"
                
                # Build command
                cmd = [
                    "python", 
                    "scripts/enhance_features.py", 
                    "--input_file", input_file,  # Use --input_file parameter (not --input_path)
                    "--output_path", output_path  # Use full output path to specify exact file name
                ]
                
                # Add optional flags based on user selection
                if add_technical:
                    cmd.append("--calculate_technicals")  # Use --calculate_technicals (not --add_technical)
                if add_statistical:
                    cmd.append("--add_sentiment")  # Use --add_sentiment (not --add_statistical)
                if add_cyclical:
                    cmd.append("--detect_regimes")  # Use --detect_regimes (not --add_cyclical)
                if add_lagged and lag_periods > 0:
                    # Note: --lag_periods might not be supported by enhance_features.py
                    # This could be another mismatch between UI and script parameters
                    cmd.append("--add_lagged")
                    cmd.append("--lag_periods")
                    cmd.append(str(lag_periods))
                    
                log_output += f"\nExecuting command: {' '.join(cmd)}\n\n"
                
                # Run the command
                process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                stdout, stderr = process.communicate()
                
                if stdout:
                    log_output += f"Standard output:\n{stdout.decode('utf-8')}\n"
                
                if process.returncode != 0:
                    log_output += f"\nFeature enhancement failed with return code {process.returncode}\n"
                    if stderr:
                        log_output += f"Error output:\n{stderr.decode('utf-8')}\n"
                    return log_output, None
                
                # Process successful
                log_output += f"\nFeature enhancement completed successfully!\n"
                
                # Check if the output file exists
                if os.path.exists(output_path):
                    log_output += f"Output file confirmed at: {output_path}\n"
                    try:
                        df = pd.read_csv(output_path)
                        log_output += f"Read {len(df)} rows and {len(df.columns)} columns from output file.\n"
                        preview = df.head(10)
                        return log_output, preview
                    except Exception as e:
                        log_output += f"Warning: Could not read output file: {str(e)}\n"
                        return log_output, None
                else:
                    log_output += f"Warning: Expected output file not found at {output_path}\n"
                    
                    # Try to look for the file in different locations
                    possible_locations = [
                        output_path,
                        os.path.join(output_dir, os.path.basename(input_file)),
                        os.path.join(".", os.path.basename(output_path))
                    ]
                    
                    log_output += "Checking alternative locations:\n"
                    for loc in possible_locations:
                        log_output += f"- {loc}: {'Found' if os.path.exists(loc) else 'Not found'}\n"
                    
                    # List files in the output directory
                    log_output += f"\nFiles in {output_dir} directory:\n"
                    if os.path.exists(output_dir):
                        for f in os.listdir(output_dir):
                            loc = os.path.join(output_dir, f)
                            file_info = f"- {f} ({os.path.getsize(loc)} bytes)"
                            log_output += file_info + "\n"
                            
                            # If this looks like our output file, try to use it
                            if f.endswith('.csv'):
                                log_output += f"Attempting to use {loc} as output file...\n"
                                try:
                                    df = pd.read_csv(loc)
                                    log_output += f"Success! Read {len(df)} rows and {len(df.columns)} columns.\n"
                                    preview = df.head(10)
                                    return log_output, preview
                                except Exception as e:
                                    log_output += f"Failed to read: {str(e)}\n"
                    else:
                        log_output += f"Output directory {output_dir} doesn't exist.\n"
                    
                    return log_output, None
                    
            except Exception as e:
                import traceback
                log_output = f"Error in enhance_features: {str(e)}\n"
                log_output += traceback.format_exc()
                return log_output, None
        
        enhance_button.click(
            enhance_features,
            inputs=[enhance_input, enhance_output, add_technical, add_statistical, 
                   add_cyclical, add_lagged, lag_periods],
            outputs=[enhance_log, feature_preview]
        )
    
    return None

def create_visualization_tab():
    with gr.Tab("Visualization"):
        gr.Markdown("## Data and Model Visualization")
        
        with gr.Tabs():
            with gr.Tab("Data Explorer"):
                with gr.Row():
                    with gr.Column(scale=1):
                        data_file = gr.File(label="Data CSV File")
                        
                        plot_type = gr.Radio(
                            ["Time Series", "Correlation Matrix", "Feature Distribution", 
                             "Regime Analysis", "Return Distribution"],
                            label="Plot Type",
                            value="Time Series"
                        )
                        
                        with gr.Accordion("Plot Options", open=True):
                            features = gr.Textbox(label="Features to Plot (comma-separated, leave empty for all)", 
                                                value="close,volume")
                            start_date = gr.Textbox(label="Start Date (YYYY-MM-DD, optional)", value="")
                            end_date = gr.Textbox(label="End Date (YYYY-MM-DD, optional)", value="")
                            
                        visualize_button = gr.Button("Generate Visualization", variant="primary")
                        
                    with gr.Column(scale=2):
                        data_plot = gr.Plot(label="Data Visualization")
                        plot_desc = gr.Markdown("")
            
            with gr.Tab("Model Comparison"):
                with gr.Row():
                    with gr.Column(scale=1):
                        model1_path = gr.Dropdown(
                            choices=list_model_files("models"), 
                            label="Model 1 Checkpoint",
                            value="",
                            interactive=True,
                            allow_custom_value=True
                        )
                        model1_type = gr.Radio(
                            ["baseline", "cognitive"], 
                            label="Model 1 Type",
                            value="cognitive"
                        )
                        
                        model2_path = gr.Dropdown(
                            choices=list_model_files("models"), 
                            label="Model 2 Checkpoint",
                            value="",
                            interactive=True,
                            allow_custom_value=True
                        )
                        model2_type = gr.Radio(
                            ["baseline", "cognitive"], 
                            label="Model 2 Type",
                            value="baseline"
                        )
                        
                        refresh_model_btn = gr.Button("Refresh Models")
                        
                        comparison_data = gr.File(label="Test Data CSV File")
                        comparison_type = gr.Radio(
                            ["Performance Metrics", "Predictions", "Feature Importance", 
                             "Error Analysis", "Regime Performance"],
                            label="Comparison Type",
                            value="Performance Metrics"
                        )
                        
                        compare_button = gr.Button("Compare Models", variant="primary")
                        
                    with gr.Column(scale=2):
                        comparison_plot = gr.Plot(label="Comparison Visualization")
                        comparison_metrics = gr.JSON(label="Comparison Metrics")
        
        # Data visualization functionality
        def visualize_data(data_file, plot_type, features, start_date, end_date):
            if not data_file or not os.path.exists(data_file.name):
                return None, "Error: Please upload a valid data file"
    
    # Load data
            df = pd.read_csv(data_file.name)
            df.columns = [col.lower() for col in df.columns]
            
            # Convert date column if it exists
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                
                # Filter by date if provided
                if start_date:
                    df = df[df['date'] >= pd.to_datetime(start_date)]
                if end_date:
                    df = df[df['date'] <= pd.to_datetime(end_date)]
            
            # Parse features
            feature_list = []
            if features and len(features.strip()) > 0:
                feature_list = [f.strip() for f in features.split(',')]
                # Check if all features exist
                missing = [f for f in feature_list if f not in df.columns]
                if missing:
                    return None, f"Error: Features not found in data: {', '.join(missing)}"
            
            fig = plt.figure(figsize=(12, 8))
            
            if plot_type == "Time Series":
                if not feature_list:
                    # Default to close price if no features specified
                    feature_list = ['close'] if 'close' in df.columns else [df.columns[1]]
                
                if 'date' in df.columns:
                    x = df['date']
                else:
                    x = np.arange(len(df))
                    
                for feature in feature_list:
                    plt.plot(x, df[feature], label=feature)
                    
                plt.title(f"Time Series Plot of {', '.join(feature_list)}")
                plt.xlabel("Date" if 'date' in df.columns else "Index")
                plt.ylabel("Value")
                plt.legend()
                if len(x) > 20:  # Rotate labels if many points
                    plt.xticks(rotation=45)
                plt.tight_layout()
                
                desc = f"Time series plot showing {', '.join(feature_list)} over time."
                
            elif plot_type == "Correlation Matrix":
                # Select numeric columns only
                numeric_df = df.select_dtypes(include=[np.number])
                
                if feature_list:
                    # Filter to requested features that are numeric
                    valid_features = [f for f in feature_list if f in numeric_df.columns]
                    if not valid_features:
                        return None, "Error: No numeric features found in selection"
                    corr_df = numeric_df[valid_features].corr()
                else:
                    # Limit to reasonable number of features
                    if len(numeric_df.columns) > 15:
                        # Prioritize important columns
                        priority = ['close', 'open', 'high', 'low', 'volume', 'returns']
                        cols = [c for c in priority if c in numeric_df.columns]
                        # Add more columns up to 15
                        remaining = [c for c in numeric_df.columns if c not in cols]
                        cols.extend(remaining[:15-len(cols)])
                        corr_df = numeric_df[cols].corr()
                    else:
                        corr_df = numeric_df.corr()
                
                plt.figure(figsize=(10, 8))
                plt.imshow(corr_df, cmap='coolwarm', vmin=-1, vmax=1)
                plt.colorbar(label='Correlation')
                plt.title("Feature Correlation Matrix")
                
                # Add correlation values
                for i in range(len(corr_df)):
                    for j in range(len(corr_df)):
                        text = plt.text(j, i, f'{corr_df.iloc[i, j]:.2f}',
                                       ha="center", va="center", color="black")
                
                plt.xticks(range(len(corr_df)), corr_df.columns, rotation=45)
                plt.yticks(range(len(corr_df)), corr_df.columns)
                plt.tight_layout()
                
                desc = f"Correlation matrix showing relationships between numeric features."
                
            elif plot_type == "Feature Distribution":
                if not feature_list:
                    return None, "Error: No features selected for distribution analysis"
                
                # Check if features exist in dataframe
                missing = [f for f in feature_list if f not in df.columns]
                if missing:
                    return None, f"Error: Features not found: {', '.join(missing)}"
                
                # Create subplots - adjust rows and columns based on feature count
                n_features = len(feature_list)
                n_cols = min(3, n_features)
                n_rows = (n_features + n_cols - 1) // n_cols
                
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
                if n_features == 1:  # Handle single plot case
                    axes = np.array([axes])
                axes = axes.flatten()
                
                for i, feature in enumerate(feature_list):
                    ax = axes[i]
                    
                    # Handle numeric vs categorical features
                    if pd.api.types.is_numeric_dtype(df[feature]):
                        # Plot histogram with KDE
                        sns.histplot(df[feature].dropna(), kde=True, ax=ax)
                        
                        # Add lines for quartiles, median and mean
                        quartiles = df[feature].quantile([0.25, 0.5, 0.75])
                        mean = df[feature].mean()
                        
                        # Add vertical lines
                        for val, name, color in zip(
                            [quartiles[0.25], quartiles[0.5], quartiles[0.75], mean],
                            ['Q1', 'Median', 'Q3', 'Mean'],
                            ['green', 'red', 'green', 'blue']
                        ):
                            ax.axvline(val, color=color, linestyle='--', label=f'{name}: {val:.2f}')
                        
                        # Add skew and kurtosis
                        skew = df[feature].skew()
                        axes[i].axvline(mean, color='green', linestyle='-', alpha=0.7)
                        axes[i].text(mean, axes[i].get_ylim()[1]*0.8, f' Mean: {mean:.2f}', 
                                    color='green', verticalalignment='top')
                        
                        # Add text with skew and kurtosis
                        skew = df[feature].skew()
                        kurt = df[feature].kurtosis()
                        stats_text = f"Skew: {skew:.2f}, Kurtosis: {kurt:.2f}"
                        axes[i].text(0.95, 0.95, stats_text, transform=axes[i].transAxes, 
                                    ha='right', va='top', bbox=dict(facecolor='white', alpha=0.7))
                    else:
                        # For non-numeric features, show value counts
                        value_counts = df[feature].value_counts().sort_index()
                        value_counts.plot(kind='bar', ax=axes[i])
                        axes[i].set_ylabel('Count')
                        
                    axes[i].set_title(f'Distribution of {feature}')
                    axes[i].grid(True, alpha=0.3)
                
                plt.tight_layout()
                
                desc = f"Distribution analysis of selected features: {', '.join(feature_list)}"
            
            elif plot_type == "Regime Analysis":
                # Check if regime column exists
                if 'regime' not in df.columns:
                    return None, "Error: No 'regime' column found in data"
                
                # Count regimes
                regime_counts = df['regime'].value_counts()
                
                # Create a figure with two subplots
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                # Plot regime counts
                ax1.bar(regime_counts.index.astype(str), regime_counts.values)
                ax1.set_title("Count of Samples by Regime")
                ax1.set_xlabel("Regime")
                ax1.set_ylabel("Count")
                
                # Plot regime over time if date column exists
                if 'date' in df.columns:
                    unique_regimes = df['regime'].unique()
                    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_regimes)))
                    
                    for i, regime in enumerate(unique_regimes):
                        regime_data = df[df['regime'] == regime]
                        ax2.scatter(regime_data['date'], regime_data['close'] if 'close' in df.columns else i,
                                  c=[colors[i]], label=f"Regime {regime}", alpha=0.7)
                    
                    ax2.set_title("Regime Timeline")
                    ax2.set_xlabel("Date")
                    ax2.set_ylabel("Price" if 'close' in df.columns else "Regime")
                    ax2.legend()
                    plt.xticks(rotation=45)
                
                plt.tight_layout()
                
                desc = f"Regime analysis showing distribution and timeline of different market regimes."
                
            elif plot_type == "Return Distribution":
                # Check if returns column exists or can be calculated
                if 'returns' not in df.columns and 'close' in df.columns:
                    df['returns'] = df['close'].pct_change()
                    
                if 'returns' not in df.columns:
                    return None, "Error: No 'returns' column found and couldn't calculate from 'close'"
                
                # Create figure with multiple plots
                fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))
                
                # 1. Histogram of returns
                ax1.hist(df['returns'].dropna(), bins=50, alpha=0.7, density=True)
                
                # Add normal distribution for comparison
                from scipy.stats import norm
                x = np.linspace(df['returns'].min(), df['returns'].max(), 1000)
                ax1.plot(x, norm.pdf(x, df['returns'].mean(), df['returns'].std()), 'r-', lw=2)
                
                ax1.set_title("Return Distribution with Normal Curve")
                ax1.set_xlabel("Returns")
                ax1.set_ylabel("Density")
                
                # Add statistics
                stats_text = (f"Mean: {df['returns'].mean():.4f}\n"
                             f"Std Dev: {df['returns'].std():.4f}\n"
                             f"Skew: {df['returns'].skew():.4f}\n"
                             f"Kurtosis: {df['returns'].kurtosis():.4f}")
                ax1.text(0.02, 0.95, stats_text, transform=ax1.transAxes, 
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
                
                # 2. QQ plot
                from scipy.stats import probplot
                probplot(df['returns'].dropna(), plot=ax2)
                ax2.set_title("Q-Q Plot of Returns")
                
                # 3. Return time series
                if 'date' in df.columns:
                    ax3.plot(df['date'], df['returns'], linewidth=0.8)
                else:
                    ax3.plot(df['returns'], linewidth=0.8)
                    
                ax3.set_title("Return Time Series")
                ax3.set_xlabel("Date" if 'date' in df.columns else "Index")
                ax3.set_ylabel("Returns")
                plt.xticks(rotation=45)
                
                plt.tight_layout()
                
                desc = f"Return distribution analysis showing histogram, Q-Q plot, and time series."
            
            return fig, desc
            
        visualize_button.click(
            visualize_data,
            inputs=[data_file, plot_type, features, start_date, end_date],
            outputs=[data_plot, plot_desc]
        )
        
        # Model comparison functionality
        def compare_models(model1_path, model1_type, model2_path, model2_type, comparison_data, comparison_type):
            if not model1_path or not model2_path:
                return None, {"error": "Please select both model checkpoints"}
                
            if not comparison_data or not os.path.exists(comparison_data.name):
                return None, {"error": "Please upload test data for comparison"}
                
            # Create temporary output directory
            output_dir = f"evaluation/comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            os.makedirs(output_dir, exist_ok=True)
                
            # Select appropriate comparison script
            script_path = "scripts/compare_models.py"
            if comparison_type == "Performance Metrics":
                script_path = "scripts/model_compare.py"
            elif comparison_type == "Predictions":
                script_path = "scripts/compare_models_simple.py"
            elif comparison_type == "Feature Importance":
                script_path = "scripts/model_compare_fixed.py"
                
            # Build command
            cmd = [
                sys.executable, script_path,
                "--model1_path", model1_path,
                "--model1_type", model1_type,
                "--model2_path", model2_path,
                "--model2_type", model2_type,
                "--data_path", comparison_data.name,
                "--output_dir", output_dir,
                "--comparison_type", comparison_type.lower().replace(" ", "_")
            ]
                
            process = subprocess.run(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, universal_newlines=True
            )
                
            if process.returncode == 0:
                # Find visualization
                viz_files = list(Path(output_dir).glob("*.png"))
                viz_path = str(viz_files[0]) if viz_files else None
                
                # Load metrics
                metrics_path = os.path.join(output_dir, "comparison_metrics.json")
                metrics = {}
                if os.path.exists(metrics_path):
                    with open(metrics_path, "r") as f:
                        metrics = json.load(f)
                
                # Load visualization
                fig = None
                if viz_path:
                    img = plt.imread(viz_path)
                    fig, ax = plt.subplots(figsize=(12, 8))
                    ax.imshow(img)
                    ax.axis('off')
                
                return fig, metrics
            else:
                return None, {"error": f"Comparison failed: {process.stdout}"}
                
        refresh_model_btn.click(
            refresh_model_list,
            inputs=None,
            outputs=[model1_path, model2_path]
        )
                
        compare_button.click(
            compare_models,
            inputs=[model1_path, model1_type, model2_path, model2_type, comparison_data, comparison_type],
            outputs=[comparison_plot, comparison_metrics]
        )
    
    return None

def create_documentation_tab():
    with gr.Tab("Documentation"):
        gr.Markdown("""
        # Cognitive Architecture Documentation
        
        ## Overview
        
        This application provides a comprehensive interface to train, evaluate, and monitor cognitive architecture models for financial forecasting. The cognitive architecture integrates attention mechanisms, memory modules, and regime detection to create an adaptive forecasting system.
        
        ## Main Components
        
        ### 1. Training
        - **Model Types**: baseline, cognitive, regime-aware cognitive
        - **Progressive Training**: Adapts difficulty during training
        - **Configuration**: Customize architecture and training parameters
        
        ### 2. Evaluation
        - **Standard Metrics**: MSE, RMSE, MAE, Direction Accuracy
        - **Regime-Specific**: Analyze performance across market regimes
        - **Stress Tests**: Test robustness with market crash, high volatility, regime change scenarios
        
        ### 3. Monitoring
        - **Introspection**: Analyze internal model components
        - **Information Flow**: Visualize data flow through model
        - **Early Warning**: Detect anomalies and potential model issues
        
        ### 4. Data Management
        - **Download Data**: Get financial time series with technical indicators
        - **Build Datasets**: Combine multiple assets with market context
        - **Feature Enhancement**: Add technical, statistical, and derived features
        
        ## Command-Line Interface
        
        All functionality is available via command-line scripts:
        
        ```
        # Training
        python scripts/train_progressive.py --model cognitive --config configs/cognitive.json --data_path data/train.csv
        
        # Basic Training
        python scripts/train_baseline.py --model cognitive --config configs/cognitive.json --data_path data/train.csv
        
        # Component Training
        python scripts/train_component.py --component memory --data_path data/train.csv --output_dir models/components
        
        # Evaluation by Regime
        python scripts/evaluate_by_regime.py --model_path models/cognitive_latest.pth --data_path data/test.csv
        
        # Stress Testing
        python scripts/stress_test.py --model_path models/cognitive_latest.pth --data_path data/test.csv --scenarios "Market Crash,High Volatility"
        
        # Live Monitoring
        python scripts/monitor_live.py --model_path models/cognitive_latest.pth --data_path data/live.csv
        ```
        
        ## Tips for Optimal Results
        
        1. Start with baseline models to establish performance benchmarks
        2. Use progressive training for better generalization
        3. Monitor memory and attention patterns during training
        4. Evaluate across different market regimes
        5. Use stress tests to identify weaknesses
        6. Enhance features to improve model performance
        7. Monitor model with early warning system during deployment
        """)
    
    return None

def create_app():
    with gr.Blocks(title="Cognitive Architecture Framework") as app:
        gr.Markdown("# Financial Cognitive Architecture Framework")
        
        # Create tabs
        training_output = create_training_tab()
        evaluation_output = create_evaluation_tab()
        create_monitoring_tab()
        create_data_tab()
        create_visualization_tab()
        create_documentation_tab()
        
    return app

def refresh_model_list():
    return [list_model_files("models"), list_model_files("models")]

if __name__ == "__main__":
    app = create_app()
    app.launch(share=False)
