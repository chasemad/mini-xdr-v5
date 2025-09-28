# ML Model Training Guide with Open Source Datasets

## 🎯 Overview

This guide shows you how to train the Mini-XDR ML models using various open source cybersecurity datasets and methods.

## 📊 Available Training Methods

### Method 1: Quick Training Script (Recommended)

Use the automated training script with existing datasets:

```bash
# Train with default datasets (combined, brute_force, web_attacks)
python scripts/train-models-with-datasets.py

# Train with specific datasets
python scripts/train-models-with-datasets.py --datasets brute_force_ssh ddos_attacks malware_behavior

# Train with all available datasets
python scripts/train-models-with-datasets.py --datasets all
```

### Method 2: API-Based Training

Train models through the REST API:

```bash
# Trigger training with recent data
curl -X POST http://localhost:8000/api/retrain_ml_models \
  -H "Content-Type: application/json" \
  -d '{"datasets": ["combined_cybersecurity"], "force": true}'

# Check ML model status
curl http://localhost:8000/api/ml_status
```

### Method 3: Generate Synthetic Training Data

Create realistic training data on-demand:

```bash
# Quick training boost with synthetic data
python scripts/generate-training-data.py --mode quick

# Comprehensive synthetic dataset
python scripts/generate-training-data.py --mode comprehensive \
  --web-count 200 --ssh-count 100 --attack-web 50 --attack-ssh 40
```

### Method 4: Load External Datasets

Import external cybersecurity datasets:

```bash
# Download and process external datasets
python scripts/download-open-datasets.py

# Import custom dataset format
python scripts/import-historical-data.py --format csv --file /path/to/dataset.csv
```

## 📋 Available Open Source Datasets

Your system includes these pre-loaded datasets:

| Dataset | Size | Description | Events |
|---------|------|-------------|---------|
| `combined_cybersecurity` | 410 KB | Mixed attack types | 983 |
| `brute_force_ssh` | 65 KB | SSH brute force attacks | 174 |
| `ddos_attacks` | 264 KB | DDoS attack patterns | 1000+ |
| `web_attacks` | 56 KB | Web application attacks | 125 |
| `network_scans` | 19 KB | Network reconnaissance | 50+ |
| `malware_behavior` | 5 KB | Malware signatures | 25+ |

## 🧠 ML Models Explained

### 1. Isolation Forest
- **Purpose**: Anomaly detection in high-dimensional data
- **Best for**: Detecting outlier IPs and unusual behavior patterns
- **Training data needed**: 50+ samples
- **Strength**: Fast, unsupervised learning

### 2. LSTM Autoencoder  
- **Purpose**: Sequential pattern analysis
- **Best for**: Detecting attack sequences and temporal anomalies
- **Training data needed**: 100+ sequential samples
- **Strength**: Deep learning, temporal patterns

### 3. Enhanced ML Ensemble
- **Purpose**: Multi-model consensus for higher accuracy
- **Models included**: Isolation Forest, One-Class SVM, Local Outlier Factor
- **Best for**: Robust anomaly detection with confidence scores
- **Strength**: Combines multiple algorithms

### 4. Federated Learning (Advanced)
- **Purpose**: Privacy-preserving distributed learning
- **Best for**: Learning from multiple honeypot deployments
- **Requirements**: Multiple nodes, secure aggregation
- **Strength**: Collaborative learning without data sharing

## 🔧 Training Configuration

### Feature Extraction

The ML engine automatically extracts these features from events:

```python
features = [
    'event_count_1h',         # Events in last hour
    'event_count_24h',        # Events in last 24 hours  
    'unique_ports',           # Number of unique ports accessed
    'failed_login_count',     # Failed authentication attempts
    'session_duration_avg',   # Average session length
    'password_diversity',     # Unique passwords tried
    'username_diversity',     # Unique usernames tried
    'event_rate_per_minute',  # Attack velocity
    'time_of_day',           # Temporal patterns
    'is_weekend',            # Time-based features
    'command_diversity',     # Unique commands executed
    'download_attempts',     # File download behavior
    'upload_attempts'        # File upload behavior
]
```

### Training Parameters

Default training configuration:

```python
TRAINING_CONFIG = {
    'min_samples_isolation_forest': 50,
    'min_samples_lstm': 100,
    'min_samples_ensemble': 25,
    'isolation_forest_contamination': 0.1,
    'lstm_sequence_length': 10,
    'lstm_hidden_size': 64,
    'training_epochs': 50,
    'batch_size': 32
}
```

## 📈 Training Best Practices

### 1. Data Quality
- **Clean data**: Remove corrupted or invalid events
- **Balanced dataset**: Include both normal and attack traffic
- **Recent data**: Use data from last 30 days for relevance
- **Diverse attacks**: Include multiple attack types

### 2. Feature Engineering
- **Temporal features**: Include time-based patterns
- **Behavioral metrics**: Focus on rate-based features
- **Statistical features**: Use aggregations (mean, std, percentiles)
- **Domain knowledge**: Include cybersecurity-specific features

### 3. Model Selection
- **Start simple**: Begin with Isolation Forest
- **Add complexity**: Progress to ensemble methods
- **Sequential data**: Use LSTM for time-series patterns
- **High accuracy needs**: Use ensemble with voting

### 4. Validation
- **Test with known attacks**: Validate against labeled data
- **False positive rate**: Monitor legitimate traffic classification
- **Performance metrics**: Track precision, recall, F1-score
- **Real-world testing**: Test with live honeypot data

## 🚀 Training Workflow

### Quick Start (5 minutes)
```bash
# 1. Generate synthetic training data
python scripts/generate-training-data.py --mode quick

# 2. Train models with existing datasets  
python scripts/train-models-with-datasets.py

# 3. Verify training
curl http://localhost:8000/api/ml_status

# 4. Test detection
python scripts/test-adaptive-detection.sh
```

### Production Setup (30 minutes)
```bash
# 1. Download additional datasets
python scripts/download-open-datasets.py

# 2. Import historical honeypot data
python scripts/import-historical-data.py --days 30

# 3. Train comprehensive models
python scripts/train-models-with-datasets.py --datasets all

# 4. Enable continuous learning
curl -X POST http://localhost:8000/api/adaptive/enable_learning

# 5. Monitor performance
curl http://localhost:8000/api/adaptive/status
```

## 📊 Monitoring Training

### Check Model Status
```bash
# API status
curl http://localhost:8000/api/ml_status

# Python status
python -c "from backend.app.ml_engine import ml_detector; print(ml_detector.get_model_status())"
```

### Training Metrics
```bash
# View training metadata
cat backend/models/training_metadata.json

# Check model files
ls -la backend/models/
```

## 🛠️ Troubleshooting

### Common Issues

1. **"Insufficient training data"**
   - Solution: Generate more synthetic data or use larger datasets
   - Command: `python scripts/generate-training-data.py --mode comprehensive`

2. **"LSTM training failed"**
   - Solution: Need sequential data with time patterns
   - Command: Import time-series attack data

3. **"Models not loading"**
   - Solution: Check model file permissions and paths
   - Command: `ls -la backend/models/`

4. **"High false positive rate"**
   - Solution: Retrain with more legitimate traffic examples
   - Command: Generate baseline traffic data

### Performance Optimization

```bash
# Retrain with more data
python scripts/train-models-with-datasets.py --datasets combined_cybersecurity brute_force_ssh ddos_attacks

# Tune thresholds
curl -X POST http://localhost:8000/api/adaptive/tune_thresholds

# Enable online learning
curl -X POST http://localhost:8000/api/adaptive/enable_online_learning
```

## 🔄 Continuous Learning

Enable automatic model updates:

```bash
# Start continuous learning pipeline
curl -X POST http://localhost:8000/api/adaptive/start_learning

# Schedule daily retraining
curl -X POST http://localhost:8000/api/adaptive/schedule_retraining \
  -d '{"interval": "daily", "min_events": 100}'
```

## 📚 External Dataset Sources

### Recommended Cybersecurity Datasets

1. **UNSW-NB15**: Network intrusion dataset
2. **KDD Cup 1999**: Classic network intrusion detection
3. **CSE-CIC-IDS2018**: Comprehensive intrusion dataset  
4. **Malware Training Sets**: Executable analysis datasets
5. **Honeypot Logs**: Real-world attack data

### Loading External Datasets

```python
# Example: Load custom CSV dataset
import pandas as pd
from backend.app.training_data_collector import TrainingDataCollector

collector = TrainingDataCollector()
df = pd.read_csv('external_dataset.csv')

# Convert to Mini-XDR format
events = collector.convert_external_dataset(df, format='csv')

# Train models
await ml_detector.train_models(events)
```

## 🎯 Success Metrics

Track these metrics to ensure effective training:

- **Detection Rate**: >95% for known attacks
- **False Positive Rate**: <5% for legitimate traffic
- **Model Confidence**: >0.7 average confidence
- **Training Time**: <5 minutes for standard datasets
- **Memory Usage**: <1GB for trained models

## 🔒 Security Considerations

- **Data Privacy**: Anonymize sensitive data before training
- **Model Security**: Protect trained models from tampering
- **Bias Prevention**: Ensure diverse training data
- **Version Control**: Track model versions and performance
- **Audit Trail**: Log all training activities

---

🎉 **Your models are now trained and ready for live threat detection!**
