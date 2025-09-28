# 🧠 Machine Learning Training Scripts

Scripts for training, optimizing, and managing ML models in Mini-XDR's adaptive detection system.

## Scripts Overview

### 🚀 Core Training Scripts

#### `massive-dataset-trainer.py`
**Train with ALL available datasets**
- **Purpose**: Use every available dataset for maximum ML performance
- **Usage**: `python3 massive-dataset-trainer.py`
- **Features**: Auto-discovery of datasets, comprehensive training, model validation

#### `train-models-with-datasets.py`
**Basic ML training with open source datasets**
- **Purpose**: Train ML models using synthetic cybersecurity datasets
- **Usage**: `python3 train-models-with-datasets.py`
- **Features**: Ensemble training, model verification, core ML pipeline

#### `train-with-real-datasets.py`
**Enhanced training with real-world data**
- **Purpose**: Train using both synthetic and real cybersecurity datasets
- **Usage**: `python3 train-with-real-datasets.py`
- **Features**: Real dataset integration, threat intelligence, balanced training

### 📈 Data Generation & Import

#### `generate-training-data.py`
**Synthetic training data generator**
- **Purpose**: Generate realistic baseline and attack data for ML training
- **Usage**: `python3 generate-training-data.py [--mode quick|comprehensive]`
- **Features**: Behavioral patterns, adaptive detection, continuous learning

#### `import-historical-data.py`
**Historical data importer**
- **Purpose**: Import existing log files and honeypot data for training
- **Usage**: `python3 import-historical-data.py --source [file|directory]`
- **Features**: Multi-format parsing, batch processing, automatic conversion

### ⚙️ Training Optimization

#### `optimize-training.py`
**ML training optimization**
- **Purpose**: Optimize and accelerate ML model training for adaptive detection
- **Usage**: `python3 optimize-training.py [--mode optimize|continuous|status]`
- **Features**: Training schedule optimization, continuous training, sensitivity adjustment

## Usage Workflows

### Quick Start Training
```bash
# 1. Generate basic training data
python3 ml-training/generate-training-data.py --mode quick

# 2. Train with synthetic datasets
python3 ml-training/train-models-with-datasets.py

# 3. Optimize training
python3 ml-training/optimize-training.py --mode optimize
```

### Comprehensive Training
```bash
# 1. Download real datasets first
python3 ../datasets/download-real-datasets.py --download-all

# 2. Train with all data
python3 ml-training/train-with-real-datasets.py

# 3. Massive training with everything
python3 ml-training/massive-dataset-trainer.py
```

### Continuous Learning
```bash
# Import historical data
python3 ml-training/import-historical-data.py --source /var/log/attacks/

# Set up continuous optimization
python3 ml-training/optimize-training.py --mode continuous --duration 60

# Monitor training status
python3 ml-training/optimize-training.py --mode status
```

## Training Pipeline

### Data Flow
```
Raw Datasets → Conversion → Feature Extraction → ML Training → Model Validation
```

### Model Types Trained
- **Isolation Forest**: Anomaly detection
- **LSTM Autoencoder**: Sequential pattern detection
- **Ensemble Models**: Combined detection approaches
- **Federated Models**: Privacy-preserving collaborative learning

### Training Features
- **Behavioral Analysis**: IP-based behavioral patterns
- **Statistical Baselines**: Automatic normal behavior learning
- **Attack Pattern Recognition**: Known attack signature detection
- **Zero-Day Detection**: Unknown attack method identification

## Performance Monitoring

### Training Metrics
- **Model Accuracy**: Detection precision and recall
- **False Positive Rate**: Benign traffic misclassification
- **Training Time**: Model training duration
- **Dataset Coverage**: Attack type representation

### Optimization Features
- **Continuous Learning**: Real-time model updates
- **Concept Drift Detection**: Model performance degradation detection
- **Hyperparameter Tuning**: Automated parameter optimization
- **Ensemble Coordination**: Multi-model decision fusion

---

**Status**: Production Ready  
**Last Updated**: September 27, 2025  
**Maintained by**: Mini-XDR ML Team
