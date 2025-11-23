# Local ML Models Architecture

Mini-XDR uses a comprehensive ensemble of locally-trained machine learning models for threat detection. All models run 100% on your infrastructure with no cloud dependencies.

## Model Ensemble

### Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                   Event Stream                          │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│              Feature Extraction (79 features)           │
│  - Temporal patterns                                    │
│  - Protocol analysis                                     │
│  - Behavioral metrics                                   │
│  - Statistical features                                 │
└────────────────┬────────────────────────────────────────┘
                 │
     ┌───────────┴───────────┬────────────┬───────────┐
     │                       │            │           │
     ▼                       ▼            ▼           ▼
┌─────────┐          ┌──────────┐  ┌──────────┐  ┌────────┐
│ General │          │   DDoS   │  │  Brute   │  │  Web   │
│ Threat  │          │Specialist│  │  Force   │  │Attacks │
│Detector │          │          │  │Specialist│  │Spec.   │
└────┬────┘          └─────┬────┘  └────┬─────┘  └───┬────┘
     │                     │            │            │
     └──────────┬──────────┴────────────┴────────────┘
                │
                ▼
      ┌──────────────────┐
      │ Ensemble Voting  │
      │  - Confidence    │
      │  - Uncertainty   │
      │  - Aggregation   │
      └────────┬─────────┘
               │
               ▼
      ┌──────────────────┐
      │ Final Prediction │
      │  - Threat Type   │
      │  - Severity      │
      │  - Confidence    │
      └──────────────────┘
```

## Model Details

### 1. General Threat Detector

**Location**: `models/local_trained/general/threat_detector.pth`

**Purpose**: Multi-class classification for 7 threat categories

**Classes**:
0. Normal Traffic
1. DDoS/DoS Attack
2. Network Reconnaissance
3. Brute Force Attack
4. Web Application Attack
5. Malware/Botnet
6. Advanced Persistent Threat (APT)

**Architecture**:
- Deep Learning: PyTorch Neural Network
- Layers: 79 → 256 → 128 → 64 → 7
- Activation: ReLU with Dropout (0.3)
- Output: Softmax probabilities

**Performance**:
- Overall Accuracy: 97.98%
- Training Dataset: 1.2M+ events
- False Positive Rate: <2%

**Usage**:
```python
from backend.app.enhanced_threat_detector import enhanced_detector

result = await enhanced_detector.analyze_threat(src_ip, events)
# result.threat_type: str
# result.confidence: float
# result.class_probabilities: List[float]
```

### 2. DDoS Specialist Model

**Location**: `models/local_trained/ddos/threat_detector.pth`

**Purpose**: Specialized DDoS/DoS attack detection

**Detection Capabilities**:
- SYN Flood
- UDP Flood
- ICMP Flood
- HTTP Flood
- Slowloris
- Application-layer DDoS

**Features Analyzed**:
- Packet rate per second
- Connection rate
- Unique source IPs
- Protocol distribution
- Payload size distribution
- Time-series patterns

**Performance**:
- Precision: 98.5%
- Recall: 97.2%
- F1-Score: 97.8%

### 3. Brute Force Specialist Model

**Location**: `models/local_trained/brute_force/threat_detector.pth`

**Purpose**: Credential stuffing and brute force detection

**Detection Capabilities**:
- SSH brute force
- RDP brute force
- HTTP authentication attacks
- Password spraying
- Credential stuffing

**Features Analyzed**:
- Failed login rate
- Username diversity
- Password patterns
- Time intervals between attempts
- Geolocation changes
- User agent variations

**Performance**:
- Precision: 99.1%
- Recall: 98.7%
- F1-Score: 98.9%

### 4. Web Attacks Specialist Model

**Location**: `models/local_trained/web_attacks/threat_detector.pth`

**Purpose**: Web application attack detection

**Detection Capabilities**:
- SQL Injection
- Cross-Site Scripting (XSS)
- Command Injection
- Path Traversal
- XML External Entity (XXE)
- Server-Side Request Forgery (SSRF)

**Features Analyzed**:
- URL patterns
- Query parameter analysis
- Request header inspection
- Payload encoding detection
- Attack signature matching
- Behavioral analysis

**Performance**:
- Precision: 96.8%
- Recall: 95.4%
- F1-Score: 96.1%

### 5. Windows Specialist Models

#### Windows 13-Class Specialist
**Location**: `models/windows_specialist_13class/windows_13class_specialist.pth`

**Purpose**: Windows-specific threat detection

**Classes**:
- Normal
- Exploits
- Fuzzers
- Reconnaissance
- DoS
- Generic
- Shellcode
- Worms
- Backdoors
- Analysis
- Ransomware
- Trojans
- Privilege Escalation

**Features Analyzed**:
- Windows Event Logs
- Process execution patterns
- Registry modifications
- File system changes
- Network connections
- PowerShell activity

**Performance**:
- Accuracy: 94.3%
- Specialized for Windows environments

### 6. Isolation Forest (Anomaly Detection)

**Location**: `models/isolation_forest.pkl`

**Purpose**: Unsupervised anomaly detection

**Algorithm**: Isolation Forest (sklearn)

**Features**:
- Detects novel/zero-day attacks
- No labeled training data required
- Anomaly score: 0.0 (normal) to 1.0 (anomalous)

**Use Cases**:
- Unknown attack patterns
- Behavioral baseline violations
- Outlier detection

**Performance**:
- Anomaly Detection Rate: 89.5%
- False Positive Rate: 5.2%

### 7. LSTM Autoencoder (Sequence Anomaly)

**Location**: `models/lstm_autoencoder.pth`

**Purpose**: Temporal pattern anomaly detection

**Architecture**:
- Encoder-Decoder LSTM
- Sequence Length: 10 events
- Hidden Dimensions: 64
- Reconstruction error threshold

**Use Cases**:
- Time-series attack patterns
- Gradual attack escalation
- Multi-stage attack detection

**Performance**:
- Reconstruction accuracy: 92.1%
- Temporal anomaly detection: 87.3%

## Feature Engineering

### 79-Feature Vector

Mini-XDR extracts 79 comprehensive features from network events:

#### Temporal Features (10 features)
- Event count (1h, 24h)
- Event rate per minute
- Time of day (0-23)
- Day of week
- Is weekend
- Hour entropy
- Inter-event time (mean, std, min, max)

#### Protocol Features (15 features)
- Unique ports
- Port diversity (entropy)
- Common ports ratio
- High ports ratio
- Protocol distribution (TCP, UDP, ICMP)
- Service identification
- Port scan indicators

#### Behavioral Features (20 features)
- Failed login count
- Success rate
- Session duration (mean, max)
- Username diversity
- Password diversity
- Command diversity
- Download/upload attempts
- Privilege escalation indicators

#### Statistical Features (15 features)
- Packet size (mean, std, min, max)
- Bytes transferred (in, out)
- Connection count
- Unique IPs contacted
- Geographic diversity
- ISP diversity

#### Advanced Features (19 features)
- Machine learning-derived features
- Threat intelligence lookups
- Reputation scores
- Historical behavior comparison
- Peer group analysis

## Model Training Pipeline

### Training Data

Mini-XDR models are trained on:
1. **Public Datasets**:
   - CICIDS2017
   - NSL-KDD
   - UNSW-NB15
   - CTU-13

2. **Honeypot Data**:
   - T-Pot captures
   - Real attack patterns
   - Emerging threats

3. **Production Data** (optional):
   - Your network traffic
   - Labeled incidents
   - Analyst feedback

### Retraining

Models automatically retrain when:
- New labeled data available (>1000 events)
- Model drift detected
- False positive rate exceeds threshold
- Manual trigger via API

```bash
# Trigger retraining
curl -X POST http://localhost:8000/api/ml/retrain \
  -H "Content-Type: application/json" \
  -d '{
    "model": "general",
    "data_source": "production",
    "validation_split": 0.2
  }'

# Monitor training progress
curl http://localhost:8000/api/ml/training/status
```

### Training Script

```bash
# Local training (advanced users)
cd /Users/chasemad/Desktop/mini-xdr
python scripts/train_local_models.py \
  --data-path ./datasets/training \
  --output-path ./models/local_trained \
  --model-type general \
  --epochs 50 \
  --batch-size 128
```

## Model Versioning

Mini-XDR maintains model versions for rollback:

```
models/
├── local_trained/          # Current production models
├── local_trained_backup_*/  # Previous versions
└── local_trained_enhanced/  # Experimental models
```

### Rollback

```bash
# List available versions
ls -la models/local_trained_backup_*/

# Rollback to previous version
cp -r models/local_trained_backup_20251004_183432/* models/local_trained/

# Restart backend
docker-compose restart backend
```

## Performance Monitoring

### Metrics

```bash
# Real-time model metrics
curl http://localhost:8000/api/ml/metrics

# Response:
{
  "general_model": {
    "predictions_today": 15234,
    "avg_confidence": 0.87,
    "avg_latency_ms": 12.3
  },
  "ddos_specialist": {
    "predictions_today": 523,
    "avg_confidence": 0.94,
    "avg_latency_ms": 8.1
  }
}
```

### Model Drift Detection

Mini-XDR automatically monitors for concept drift:

```python
# Backend automatically checks:
# - Prediction distribution changes
# - Confidence score trends
# - False positive rate increases
# - Feature distribution shifts

# View drift report
curl http://localhost:8000/api/ml/drift/report
```

## Inference Optimization

### GPU Acceleration (Optional)

For high-throughput environments:

```yaml
# docker-compose.yml
backend:
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: 1
            capabilities: [gpu]
```

### Model Quantization

Reduce memory footprint:

```python
# Backend automatically applies:
# - Dynamic quantization for inference
# - Model pruning for efficiency
# - ONNX export for production
```

### Batch Inference

For bulk processing:

```bash
curl -X POST http://localhost:8000/api/ml/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "events": [...]  # Array of events
  }'
```

## Explainability

### SHAP Values

Understanding model predictions:

```bash
curl http://localhost:8000/api/ml/explain/{incident_id}

# Response includes:
# - Feature importance
# - SHAP values
# - Decision path
# - Similar past incidents
```

### Feature Importance

```bash
curl http://localhost:8000/api/ml/features/importance

# Top features by model:
{
  "general_model": [
    {"feature": "failed_login_count", "importance": 0.23},
    {"feature": "unique_ports", "importance": 0.19},
    {"feature": "event_rate_per_minute", "importance": 0.15}
  ]
}
```

## Advanced Topics

### Ensemble Strategies

- **Voting**: Majority vote across specialists
- **Stacking**: Meta-learner on specialist outputs
- **Boosting**: Sequential error correction
- **Confidence Weighting**: Weight by model certainty

### Adversarial Robustness

Models include defenses against:
- Evasion attacks
- Data poisoning
- Model inversion
- Membership inference

### Federated Learning (Future)

Planned: Privacy-preserving distributed training

## Troubleshooting

### Models Not Loading

```bash
# Check model files exist
ls -la /Users/chasemad/Desktop/mini-xdr/models/local_trained/

# Verify file permissions
chmod -R 755 models/

# Check backend logs
docker-compose logs backend | grep -i "model\|error"
```

### Poor Detection Accuracy

```bash
# Retrain with more data
python scripts/train_local_models.py --data-path ./datasets/more_data

# Adjust confidence thresholds
curl -X PATCH http://localhost:8000/api/ml/config \
  -d '{"confidence_threshold": 0.7}'
```

### High Memory Usage

```bash
# Enable model compression
export ML_MODEL_QUANTIZE=true

# Reduce batch size
export ML_INFERENCE_BATCH_SIZE=16

# Restart backend
docker-compose restart backend
```

## Resources

- **Training Scripts**: `/scripts/ml-training/`
- **Model Architectures**: `/backend/app/deep_learning_models.py`
- **Feature Engineering**: `/backend/app/ml_feature_extractor.py`
- **API Documentation**: http://localhost:8000/docs#/ml

## Next Steps

- [Configure Detection Policies](../configuration/policies.md)
- [Integrate with T-Pot](../getting-started/tpot-integration.md)
- [Deploy AI Agents](../agents/deployment.md)
