
A sophisticated cognitive architecture for financial prediction and market understanding, combining memory mechanisms, attention networks, and context-aware processing.

## Project Overview

This project implements a cognitive architecture for financial market prediction that aims to surpass traditional baseline models by incorporating advanced cognitive capabilities:

- **Memory Systems**: Multi-tiered memory for short and long-term pattern retention
- **Attention Mechanisms**: Selective focus on relevant financial features
- **Introspection**: Self-monitoring of model confidence and reasoning
- **Cross-dimensional Understanding**: Modeling relationships between different market aspects
- **Adaptive Learning**: Online adaptation to market regime changes
- **Contextual Awareness**: Processing of market patterns within broader context
- **Enhanced Feature Processing**: Robust handling of 7-dimensional financial features
- **Dimension-Aware Evaluation**: Sophisticated handling of prediction vs target dimension mismatches

## Architecture Components

The architecture consists of several interconnected components:

1. **Perception Module**: Processes raw financial data and extracts 7 key features (price, volume, returns, volatility_20d, volatility, momentum_5d, momentum_10d)
2. **Memory Bank**: Stores and retrieves relevant patterns
3. **Attention Network**: Focuses on critical features and patterns
4. **Introspection System**: Monitors model confidence and reasoning
5. **Regime Detection**: Identifies market regimes and transitions
6. **Market State Prediction**: Generates predictions with confidence estimates
7. **Cross-dimensional Processor**: Analyzes relationships between different aspects
8. **Dimension Handler**: Manages feature and prediction dimension consistency

## Project Structure
project/
├── src/
│ ├── arch/
│ │ ├── cognitive.py # Cognitive architecture implementation
│ │ └── baseline.py # Baseline model
│ ├── core/
│ │ └── attention.py # Attention mechanisms
│ ├── data/
│ │ ├── financial_loader.py # Enhanced data loading
│ │ ├── processor.py # Data processing
│ │ └── tiered_loader.py # Multi-tier data loading
│ ├── memory/
│ │ ├── bank.py # Memory bank implementation
│ │ ├── buffer.py # Memory buffer
│ │ ├── regime_aware_memory.py # Regime-aware memory access
│ │ ├── confidence.py # Confidence estimation
│ │ ├── encoding.py # Memory encoding
│ │ ├── replay.py # Memory replay
│ │ └── retrieval.py # Memory retrieval
│ ├── monitoring/
│ │ ├── introspect.py # Model introspection
│ │ └── early_warning.py # Early warning system
│ ├── perception/
│ │ └── [perception modules]
│ ├── evaluation/
│ │ └── regime_evaluator.py # Regime-specific evaluation
│ ├── utils/
│ │ ├── regularization.py # Regularization utilities
│ │ ├── interpretability.py # Model interpretability
│ │ └── online_learning.py # Online learning framework
│ ├── visualization/
│ │ ├── plot_engine.py # Visualization utilities
│ │ └── flow_visualizer.py # Information flow visualization
│ └── deployment/
│ ├── model_server.py # Model serving framework
│ └── api_server.py # REST API server
├── scripts/
│ ├── download_data.py # Data downloading script
│ ├── build_dataset.py # Dataset creation
│ ├── split_data.py # Data splitting script
│ ├── enhance_features.py # Feature engineering
│ ├── train_progressive.py # Progressive training script
│ ├── train_baseline.py # Baseline model training
│ ├── train_component.py # Component-specific training
│ ├── evaluate_by_regime.py # Regime-specific evaluation
│ ├── stress_test.py # Model stress testing
│ ├── monitor_live.py # Live monitoring tool
│ ├── setup_monitoring.py # Monitoring setup
│ └── enhanced_visualization.py # Advanced visualization
├── tests/
│ ├── test_e2e.py # End-to-end testing
│ ├── test_component.py # Component testing
│ └── [other test files]
├── app.py # Main application UI
└── README.md

## Installation

1. Clone the repository:
```bash
git clone https://github.com/username/cognitive-finance.git
cd cognitive-finance
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

You can use the system in two ways:

### 1. Using the App Interface

Run the app interface for a complete GUI experience:
```bash
python app.py
```

The app provides a comprehensive interface with:
- Training tab with different training modes
- Evaluation tab for performance metrics, regime-specific evaluation, and stress testing
- Monitoring tab for model introspection and early warning detection
- Data management tab for downloading, processing, and enhancing data
- Visualization tab for data exploration and model comparison

### 2. Using Command-Line Scripts

#### Data Preparation

1. Download financial data:
```bash
python download_data.py --tickers SPY,QQQ --start_date 2010-01-01 --end_date 2023-01-01 --output_dir data/raw --include_features --detect_regimes
```

2. Build a combined dataset:
```bash
python scripts/build_dataset.py --input_dir data/raw --output_path data/combined_financial.csv --add_market_context
```

3. Enhance features:
```bash
python scripts/enhance_features.py --input_path data/combined_financial.csv --output_path data/enhanced_features.csv --add_technical --add_statistical
```

4. Split the data:
```bash
python scripts/split_data.py --input_path data/enhanced_features.csv --output_dir data/splits --preserve_regimes
```

#### Training

1. Progressive Training (recommended):
```bash
python scripts/train_progressive.py --model cognitive --config configs/cognitive.json --data_path data/splits/train.csv --output_dir models/cognitive
```

2. Baseline Training:
```bash
python scripts/train_baseline.py --model baseline --config configs/baseline.json --data_path data/splits/train.csv --output_dir models/baseline
```

3. Component Training:
```bash
python scripts/train_component.py --component memory --data_path data/splits/train.csv --output_dir models/components
```

#### Evaluation

1. Regime-specific evaluation:
```bash
python scripts/evaluate_by_regime.py --model_path models/cognitive/model.pth --data_path data/splits/test.csv --output_dir evaluation/regime
```

2. Stress testing:
```bash
python scripts/stress_test.py --model_path models/cognitive/model.pth --data_path data/splits/test.csv --output_dir evaluation/stress --scenarios "Market Crash,High Volatility,Regime Change"
```

#### Monitoring and Analysis

1. Setup monitoring dashboard:
```bash
python scripts/setup_monitoring.py --model_path models/cognitive/model.pth --data_path data/splits/test.csv --output_dir monitoring/dashboard
```

2. Live monitoring:
```bash
python scripts/monitor_live.py --model_path models/cognitive/model.pth --data_path data/splits/test.csv --output_dir monitoring/live
```

3. Information flow visualization:
```bash
python scripts/enhanced_visualization.py --model_path models/cognitive/model.pth --data_path data/splits/test.csv --output_dir visualizations/flow --type flow
```

## Key Features

### Progressive Training
Train models on progressively more difficult market regimes to build robust understanding.

### Regime-Aware Memory
Memory access patterns that adapt based on detected market regimes.

### Early Warning System
Detect potential model failures and regime changes before they impact performance.

### Information Flow Visualization
Visualize how information flows through different components of the cognitive architecture.

### Stress Testing Framework
Test model performance under extreme market conditions, including crashes, high volatility, and regime changes.

### Enhanced Feature Processing
Robust handling of 7-dimensional financial features with proper dimension management and normalization.

### Dimension-Aware Evaluation
Sophisticated handling of prediction vs target dimension mismatches with proper truncation and alignment.

## Comparing Cognitive vs Baseline

The cognitive architecture offers several advantages over baseline models:

1. **Memory capabilities**: Retains and utilizes historical patterns
2. **Adaptive behavior**: Adjusts to changing market regimes
3. **Introspection**: Provides confidence estimates and reasoning traces
4. **Cross-dimensional understanding**: Models relationships between different market aspects
5. **Aims fro better generalization**: Aims to perform better on novel patterns and extreme market conditions
6. **Robust dimension handling**: Properly manages 7-dimensional feature space
7. **Enhanced evaluation**: Sophisticated handling of prediction vs target dimension mismatches

## Implementation Notes

- The architecture uses PyTorch for deep learning components
- Data loading utilizes memory mapping for efficient large dataset handling
- Online learning enables continuous model adaptation
- Regularization techniques prevent overfitting
- Interpretability tools provide insights into model reasoning
- Dimension handling ensures consistency between input features and predictions
- Enhanced feature processing includes proper normalization and error handling

This project is licensed under the MIT License, with the following additional restriction:

**Copying and distribution of this project are strictly prohibited.**

Users are permitted to view the project's code and functionality for educational or informational purposes only.
