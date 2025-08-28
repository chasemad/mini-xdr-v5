# 🚀 Intelligent Adaptive Attack Detection Setup

## Quick Start (macOS)

### 1. Fix Dependencies and Start Backend

```bash
cd /Users/chasemad/Desktop/mini-xdr/backend

# Stop any existing processes
pkill -f "pip install"
pkill -f uvicorn

# Activate virtual environment
source .venv/bin/activate

# Install updated requirements (scipy-free)
pip install -r requirements.txt

# Start the enhanced backend
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 2. Test Adaptive Detection (New Terminal)

```bash
cd /Users/chasemad/Desktop/mini-xdr

# Run comprehensive test
python test_adaptive_detection.py
```

## What's New: Intelligent Adaptive Detection

### 🧠 Core Components

1. **BehaviorAnalyzer** - Detects attack patterns without hardcoded rules
   - Rapid enumeration detection
   - Error-seeking behavior analysis  
   - Progressive complexity detection
   - Timing anomaly detection
   - Parameter fuzzing detection

2. **BaselineEngine** - Learns normal behavior statistically
   - Per-IP behavioral baselines
   - Temporal pattern learning
   - Statistical deviation detection
   - Automatic sensitivity adjustment

3. **Enhanced ML Detector** - Multi-model ensemble
   - Isolation Forest for anomaly detection
   - One-Class SVM for outlier detection
   - Local Outlier Factor for density-based detection
   - Intelligent ensemble scoring

4. **AdaptiveDetectionEngine** - Orchestrates all detection layers
   - Rule-based detection (traditional)
   - Behavioral pattern analysis (new)
   - ML anomaly detection (enhanced)
   - Statistical baseline deviation (new)
   - Composite threat scoring

5. **ContinuousLearningPipeline** - Adapts over time
   - Hourly baseline updates
   - Daily ML model retraining
   - Real-time pattern learning
   - Automatic sensitivity tuning

### 🎯 Test Scenarios

The test script simulates:

1. **Web Application Attacks**
   - Admin panel scanning (`/admin.php`, `/wp-admin/`)
   - SQL injection attempts (`' OR 1=1--`)
   - Sensitive file access (`/.env`)
   - Multiple attack indicators per request

2. **Enhanced SSH Brute Force**
   - Multiple username/password combinations
   - Behavioral timing analysis
   - Credential diversity detection

3. **Adaptive Learning**
   - Real-time baseline learning
   - Pattern recognition improvement
   - False positive reduction

### 📊 Expected Results

**Traditional Detection:**
- SSH brute force: 6+ failed attempts → incident
- Web attacks: 3+ attack indicators → incident

**NEW Adaptive Detection:**
- Behavioral anomalies: Composite threat score > 0.6 → incident
- Statistical deviations: Baseline deviation > 0.3 → incident  
- ML anomalies: Ensemble score > 0.3 → incident
- **Multi-layer correlation**: Multiple detection methods → high-confidence incident

### 🔧 API Endpoints (New)

```bash
# Check adaptive detection status
curl http://localhost:8000/api/adaptive/status

# Force learning update (for testing)
curl -X POST http://localhost:8000/api/adaptive/force_learning

# Adjust sensitivity
curl -X POST http://localhost:8000/api/adaptive/sensitivity \
  -H "Content-Type: application/json" \
  -d '{"sensitivity": "high"}'
```

### 📈 Advanced Features

1. **Zero-Day Detection**: Identifies unknown attack patterns through behavioral analysis
2. **False Positive Reduction**: Statistical baselines reduce noise from legitimate traffic
3. **Attack Campaign Correlation**: Links related attack activities across time
4. **Adaptive Thresholds**: Self-tuning based on environment and performance
5. **Explainable AI**: Detailed reasoning for each detection decision

### 🎉 Success Indicators

When the test runs successfully, you should see:

✅ **Behavioral Detection**: Rapid enumeration and error-seeking patterns detected  
✅ **ML Ensemble**: Multiple models agreeing on anomalous behavior  
✅ **Baseline Learning**: Statistical patterns learned from clean data  
✅ **Adaptive Incidents**: Higher confidence, detailed threat reasoning  
✅ **Continuous Learning**: Background learning tasks running  

## Troubleshooting

### Missing Dependencies
```bash
# If you get import errors, install missing packages individually:
pip install scikit-learn numpy pandas torch torchvision
```

### Learning Pipeline Not Starting
```bash
# Check logs for learning pipeline status
curl http://localhost:8000/api/adaptive/status | jq .learning_pipeline
```

### No Adaptive Incidents Generated
```bash
# Lower the detection threshold temporarily:
curl -X POST http://localhost:8000/api/adaptive/sensitivity \
  -H "Content-Type: application/json" \
  -d '{"sensitivity": "high"}'
```

This completes the transformation of your Mini-XDR from rule-based to **intelligent adaptive detection**! 🚀
