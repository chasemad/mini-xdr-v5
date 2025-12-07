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
cd .
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

## False Positive Reduction (Phase 3)

Mini-XDR implements multiple layers of false positive reduction to ensure high-quality detections.

### Multi-Gate Detection System

Events pass through 5 verification gates before incident creation:

```
┌─────────────────────────────────────────────────────────────────┐
│                         EVENT STREAM                             │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│ GATE 1: Heuristic Pre-filter (microseconds)                     │
│ - Minimum event count check                                      │
│ - Clear attack indicator detection                               │
└────────────────────────────┬────────────────────────────────────┘
                             │ PASS
┌────────────────────────────▼────────────────────────────────────┐
│ GATE 2: ML Classification with Temperature Scaling               │
│ - General threat detector                                        │
│ - Temperature T=1.5 for confidence calibration                  │
│ - Uncertainty quantification                                     │
└────────────────────────────┬────────────────────────────────────┘
                             │ PASS
┌────────────────────────────▼────────────────────────────────────┐
│ GATE 3: Specialist Verification (Classes 1, 3, 4)               │
│ - Binary classifier confirms/rejects detection                  │
│ - 85% threshold for confirmation                                │
│ - Rejects reduce confidence by 70%                              │
└────────────────────────────┬────────────────────────────────────┘
                             │ PASS
┌────────────────────────────▼────────────────────────────────────┐
│ GATE 4: Vector Memory Check                                      │
│ - Compare to past false positives                               │
│ - 90% similarity threshold blocks detection                     │
│ - Learns from analyst feedback                                  │
└────────────────────────────┬────────────────────────────────────┘
                             │ PASS
┌────────────────────────────▼────────────────────────────────────┐
│ GATE 5: Council Verification (50-85% confidence)                │
│ - Gemini/OpenAI reasoning verification                          │
│ - Human-level threat assessment                                 │
│ - Can override or confirm ML                                    │
└────────────────────────────┬────────────────────────────────────┘
                             │ PASS
                             ▼
                    ┌─────────────────┐
                    │ CREATE INCIDENT │
                    └─────────────────┘
```

### Per-Class Confidence Thresholds

Different threat classes require different confidence levels based on training precision:

| Class | Threat Type | Training Precision | Required Confidence |
|-------|-------------|-------------------|---------------------|
| 0 | Normal | 99.9% | 95% |
| 1 | DDoS | 73.2% | 50% (specialist verified) |
| 2 | Reconnaissance | 73.3% | 50% |
| 3 | Brute Force | 17.8% | **80%** (high FP risk) |
| 4 | Web Attack | 32.4% | **75%** (high FP risk) |
| 5 | Malware | 84.2% | 45% |
| 6 | APT | 46.9% | 60% |

### Temperature Scaling

Confidence calibration using temperature scaling (T=1.5):

```python
# Before: raw model output
logits = model(features)
probs = softmax(logits)  # May be overconfident

# After: calibrated confidence
scaled_logits = logits / temperature  # T=1.5
calibrated_probs = softmax(scaled_logits)  # More realistic
```

### Specialist Model Routing

High false-positive classes (1, 3, 4) are verified by specialist binary classifiers:

- **DDoS Specialist**: 93.29% accuracy (vs 73% general)
- **Brute Force Specialist**: 90.52% accuracy (vs 17.8% precision)
- **Web Attack Specialist**: 95.29% accuracy (vs 32.4% precision)

```python
# Specialist verification flow
if predicted_class in [1, 3, 4]:
    confirmed, confidence, reason = specialist_manager.verify_prediction(
        predicted_class=predicted_class,
        features=features,
    )
    if not confirmed:
        # Reduce confidence by 70%
        final_confidence = original_confidence * 0.3
```

### Event Content Override

Pattern-based corrections when ML misclassifies due to statistical features:

- **Time-based analysis**: High-rate single-type events != DDoS
- **Username patterns**: Dictionary attacks (5+ usernames) = Brute Force
- **Command analysis**: Malware downloads (wget, curl) = Malware
- **Attack phase correlation**: 3+ phases = APT

### Vector Memory Learning

The system learns from past false positives:

```python
# Check similarity to past FPs before creating incident
is_similar, fp_details = await check_similar_false_positives(
    features=features,
    ml_prediction=threat_type,
    threshold=0.90,  # High similarity
)

if is_similar:
    # Block or significantly reduce confidence
    confidence *= 0.4
```

### Configuration

```python
# In backend/app/intelligent_detection.py
@dataclass
class DetectionConfig:
    confidence_thresholds: Dict[int, float]  # Per-class thresholds
    min_anomaly_score: float = 0.3           # Minimum anomaly score
    openai_verify_uncertain: bool = True     # Use OpenAI for uncertain
    temperature: float = 1.5                 # Temperature scaling
    specialist_threshold: float = 0.85       # Specialist confirmation
```

### Multi-Gate Detection Method

For explicit multi-gate detection with detailed gate results:

```python
from backend.app.intelligent_detection import intelligent_detector

# Use multi-gate detection (modular architecture)
result = await intelligent_detector.analyze_with_multi_gate(
    db=db_session,
    src_ip="192.168.1.100",
    events=event_list,
)

# Response includes gate results:
# {
#     "incident_created": True,
#     "detection_method": "multi_gate",
#     "gate_results": [
#         {"gate": "heuristic", "verdict": "pass", "reason": "..."},
#         {"gate": "ml_classification", "verdict": "pass", "reason": "..."},
#         {"gate": "specialist_verification", "verdict": "escalate", "reason": "..."},
#         {"gate": "vector_memory", "verdict": "pass", "reason": "..."},
#     ],
#     "escalation_reasons": [...],
#     "processing_time_ms": 125.3,
# }
```

## LangChain Agent Orchestration

Mini-XDR integrates LangChain for intelligent incident response orchestration with **36 specialized tools** across 6 capability domains.

### ReAct Agent

The LangChain orchestrator uses GPT-4o with ReAct (Reasoning + Acting) pattern:

```
Incident → Analysis → Tool Selection → Action → Verification → Report
```

### Available Tools (36 Total)

#### Network & Firewall (7 tools)
| Tool | Description |
|------|-------------|
| `block_ip` | Block malicious IP at firewall (T-Pot/UFW integration) |
| `dns_sinkhole` | Redirect malicious domains to sinkhole server |
| `traffic_redirection` | Redirect traffic to honeypot/analyzer for analysis |
| `network_segmentation` | Isolate network segments (VLAN/ACL-based) |
| `capture_traffic` | Capture network PCAP for forensic analysis |
| `deploy_waf_rules` | Deploy Web Application Firewall rules |

#### Endpoint & Host (7 tools)
| Tool | Description |
|------|-------------|
| `isolate_host` | Network isolation for compromised hosts |
| `memory_dump` | Capture RAM snapshot for malware analysis |
| `kill_process` | Terminate malicious processes by name/PID |
| `registry_hardening` | Apply Windows registry hardening profiles |
| `system_recovery` | Restore system to clean checkpoint |
| `malware_removal` | Scan and remove malware from endpoint |
| `endpoint_scan` | Full antivirus/EDR scan of endpoint |

#### Investigation & Forensics (6 tools)
| Tool | Description |
|------|-------------|
| `behavior_analysis` | Analyze attack patterns and TTPs |
| `threat_hunting` | Hunt for IOCs across environment |
| `threat_intel_lookup` | Query external threat intelligence feeds |
| `collect_evidence` | Gather and preserve forensic artifacts |
| `analyze_logs` | Correlate and analyze security logs |
| `attribution_analysis` | Identify threat actor using ML and OSINT |

#### Identity & Access (5 tools)
| Tool | Description |
|------|-------------|
| `reset_passwords` | Force password reset for compromised accounts |
| `revoke_sessions` | Terminate all active user sessions |
| `disable_user` | Disable compromised user accounts |
| `enforce_mfa` | Require multi-factor authentication |
| `privileged_access_review` | Audit and review privileged access |

#### Data Protection (4 tools)
| Tool | Description |
|------|-------------|
| `check_db_integrity` | Verify database for tampering |
| `emergency_backup` | Create immutable backup of critical data |
| `encrypt_data` | Apply encryption to sensitive data at rest |
| `enable_dlp` | Activate Data Loss Prevention policies |

#### Alerting & Notification (3 tools)
| Tool | Description |
|------|-------------|
| `alert_analysts` | Send urgent notification to SOC team |
| `create_case` | Generate incident case in ticketing system |
| `notify_stakeholders` | Alert executive leadership |

#### Legacy Compatibility (4 tools)
| Tool | Description |
|------|-------------|
| `check_ip_reputation` | Quick IP reputation check |
| `collect_forensics` | Legacy forensics collection |
| `query_threat_intel` | Legacy threat intel query |
| `send_alert` | Legacy alert sending |
| `get_attribution` | Legacy attribution analysis |

### Usage

LangChain orchestration is automatically enabled when:
1. OpenAI API key is configured
2. LangChain packages are installed

```python
# Automatic integration with agent orchestrator
result = await agent_orchestrator.orchestrate_incident_response(
    incident=incident,
    recent_events=events,
    use_langchain=True,  # Default: True
)

# Access all 36 tools programmatically
from backend.app.agents.tools import create_xdr_tools
tools = create_xdr_tools()
print(f"Available tools: {len(tools)}")  # 36 tools
```

### Tool Input Schemas

Each tool has a validated Pydantic input schema:

```python
from backend.app.agents.tools import (
    BlockIPInput,
    DNSSinkholeInput,
    MemoryDumpInput,
    # ... 30+ more input schemas
)

# Example: Block IP with custom parameters
block_input = BlockIPInput(
    ip_address="192.168.1.100",
    duration_seconds=7200,  # 2 hours
    reason="Detected brute force attack"
)
```

### Fallback Mode

When LangChain is unavailable (no API key, packages missing), the system
falls back to rule-based orchestration that still provides:
- Automatic IP blocking for critical threats
- Severity-based response escalation
- Event correlation and analysis
- All 32 UI-facing actions via REST API endpoints

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
ls -la ./models/local_trained/

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
