# Local ML Training & Deployment Guide

This guide will help you train and deploy Mini-XDR ML models **locally** without AWS SageMaker.

## 🎯 Quick Start (5 minutes to start training)

```bash
# 1. Navigate to project directory
cd /Users/chasemad/Desktop/mini-xdr

# 2. Run the training script
./train_models_locally.sh
```

That's it! The script will:
- ✅ Check dependencies (install if needed)
- ✅ Detect your hardware (GPU/CPU)
- ✅ Train all 4 models
- ✅ Save models to `models/local_trained/`

## 📊 What You're Training

### Models

1. **General Model (7-class)**
   - Normal, DDoS, Reconnaissance, Brute Force, Web Attack, Malware, APT
   - Most important model - handles all threat types

2. **DDoS Specialist (binary)**
   - Specialized for DDoS detection
   - Higher accuracy for DDoS vs Normal

3. **Brute Force Specialist (binary)**
   - Specialized for brute force attacks
   - Detects SSH/RDP/credential attacks

4. **Web Attack Specialist (binary)**
   - Specialized for web attacks
   - SQL injection, XSS, path traversal, etc.

### Training Data

- **1.6 million samples** from real attack datasets
- **79 features** per sample
- **Pre-normalized** to [0, 1] range
- Datasets: UNSW-NB15, CIC-IDS2017, KDD Cup 99, threat intel feeds

## ⚙️ Hardware & Performance

### GPU Training (Recommended)

**Apple Silicon (M1/M2/M3/M4)**
- Device: MPS (Metal Performance Shaders)
- Speed: ~30-45 minutes per model
- Total time: ~2-3 hours for all 4 models

**NVIDIA GPU**
- Device: CUDA
- Speed: ~15-20 minutes per model
- Total time: ~1-1.5 hours for all 4 models

### CPU Training

- Device: CPU
- Speed: ~1-2 hours per model
- Total time: ~4-8 hours for all 4 models
- Still perfectly usable!

## 🔧 Manual Training (Advanced)

If you want more control:

```bash
# Train only specific models
python3 aws/train_local.py \
    --models general ddos \
    --epochs 50 \
    --batch-size 1024

# Adjust hyperparameters
python3 aws/train_local.py \
    --models general \
    --epochs 100 \
    --batch-size 256 \
    --learning-rate 0.0005 \
    --patience 15

# Force CPU training
python3 aws/train_local.py \
    --device cpu

# Train with different data
python3 aws/train_local.py \
    --data-dir /path/to/your/data
```

### Command-Line Options

```
--data-dir          Path to training data (.npy files)
--output-dir        Where to save trained models
--models            Which models to train (general, ddos, brute_force, web_attacks)
--epochs            Max training epochs (default: 30)
--batch-size        Batch size (default: 512)
--learning-rate     Learning rate (default: 0.001)
--patience          Early stopping patience (default: 10)
--device            Device: auto, cpu, cuda, mps
```

## 📁 Output Structure

After training, you'll have:

```
models/local_trained/
├── general/
│   ├── threat_detector.pth          # Model weights
│   ├── model_metadata.json          # Model info (accuracy, etc.)
│   └── training_history.json        # Training logs
├── ddos/
│   ├── threat_detector.pth
│   ├── model_metadata.json
│   └── training_history.json
├── brute_force/
│   └── ...
├── web_attacks/
│   └── ...
└── training_summary.json            # Overall summary
```

## 🧪 Testing Your Models

### Quick Test

```bash
python3 aws/local_inference.py
```

This will:
- Load all trained models
- Run test inference on sample events
- Display prediction results

### Integration Test

Test with the backend:

```python
from aws.local_inference import LocalMLClient
import asyncio

async def test():
    client = LocalMLClient(model_dir="models/local_trained")
    
    # Check if models loaded
    status = client.get_model_status()
    print(f"Models loaded: {status['models_loaded']}")
    
    # Test inference
    events = [
        {
            'src_ip': '192.168.1.100',
            'dst_port': 22,
            'eventid': 'cowrie.login.failed',
            'message': 'Failed SSH login'
        }
    ]
    
    results = await client.detect_threats(events)
    for result in results:
        print(f"Threat: {result['predicted_class']} ({result['confidence']:.2%})")

asyncio.run(test())
```

## 🔌 Backend Integration

### Option 1: Replace SageMaker Client (Recommended)

Update `backend/app/ml_engine.py` to use local models:

```python
# At the top of backend/app/ml_engine.py
from aws.local_inference import local_ml_client

# In EnhancedFederatedDetector.calculate_anomaly_score():
# Replace SageMaker client with:
if await local_ml_client.health_check():
    results = await local_ml_client.detect_threats(events)
    if results:
        return results[0]['anomaly_score']
```

### Option 2: Fallback Strategy

Keep SageMaker as primary, local as fallback:

```python
# Try SageMaker first
try:
    from backend.app.sagemaker_client import sagemaker_client
    if await sagemaker_client.health_check():
        return await sagemaker_client.detect_threats(events)
except Exception as e:
    logger.warning(f"SageMaker unavailable: {e}")

# Fallback to local models
from aws.local_inference import local_ml_client
return await local_ml_client.detect_threats(events)
```

### Option 3: Configuration Switch

Add to `backend/app/config.py`:

```python
class Settings:
    ML_BACKEND: str = "local"  # or "sagemaker"
    LOCAL_MODEL_DIR: str = "models/local_trained"
```

Then in your code:

```python
from backend.app.config import settings

if settings.ML_BACKEND == "local":
    from aws.local_inference import local_ml_client as ml_client
else:
    from backend.app.sagemaker_client import sagemaker_client as ml_client
```

## 📈 Monitoring Training

### Watch Training Progress

```bash
# Training outputs progress to stdout
# Look for:
Epoch [1/30] (25.3s)
  Train Loss: 0.3245 | Train Acc: 89.23%
  Val Loss:   0.2891 | Val Acc:   91.45%
  ✅ New best model!

# Early stopping:
⚠️  Early stopping triggered after 18 epochs
```

### Check Results

```bash
# View training summary
cat models/local_trained/training_summary.json

# View individual model accuracy
cat models/local_trained/general/model_metadata.json | grep accuracy

# View training curves
cat models/local_trained/general/training_history.json
```

## 🎯 Expected Performance

Based on 1.6M samples of real attack data:

### General Model (7-class)
- **Expected Accuracy**: 85-95%
- Best for: Overall threat detection
- Classes: Normal, DDoS, Recon, BruteForce, WebAttack, Malware, APT

### DDoS Specialist
- **Expected Accuracy**: 95-99%
- Best for: High-confidence DDoS detection
- Classes: DDoS vs Normal

### Brute Force Specialist  
- **Expected Accuracy**: 90-98%
- Best for: SSH/RDP credential attacks
- Classes: Brute Force vs Normal

### Web Attack Specialist
- **Expected Accuracy**: 88-96%
- Best for: HTTP/web-layer attacks
- Classes: Web Attack vs Normal

## 🐛 Troubleshooting

### "No training data found"

```bash
# Check if data exists
ls -lh aws/training_data/*.npy

# If missing, you need to download/generate training data first
```

### "CUDA out of memory"

```bash
# Reduce batch size
python3 aws/train_local.py --batch-size 256

# Or use CPU
python3 aws/train_local.py --device cpu
```

### "No models loaded"

```bash
# Make sure training completed successfully
ls -la models/local_trained/*/threat_detector.pth

# Check logs for errors
cat models/local_trained/training_summary.json
```

### Models have low accuracy

Possible causes:
1. Training stopped too early (increase `--epochs`)
2. Learning rate too high/low (try `--learning-rate 0.0005`)
3. Data mismatch (check feature extraction in `local_inference.py`)

## 📊 Model Architecture

All models use the same architecture:

```
Input: 79 features
  ↓
Feature Interaction Layer (79 → 79)
  ↓
Attention Layer (64-dim attention)
  ↓
Hidden Layers: [512 → 256 → 128 → 64]
  ├─ Batch Normalization
  ├─ ReLU Activation
  └─ Dropout (0.3)
  ↓
Skip Connections (residual)
  ↓
Classifier: 7 classes (general) or 2 classes (specialist)
  ↓
Output: Class probabilities + uncertainty
```

**Total Parameters**: ~700K per model

## 🔄 Retraining Models

To retrain models with new data:

```bash
# 1. Backup existing models
mv models/local_trained models/local_trained.backup

# 2. Place new training data in aws/training_data/

# 3. Run training
./train_models_locally.sh

# 4. Compare performance
python3 aws/compare_models.py \
    --old models/local_trained.backup \
    --new models/local_trained
```

## 💾 Model Deployment

### Copy to Production

```bash
# Package models
cd models
tar -czf mini-xdr-models-$(date +%Y%m%d).tar.gz local_trained/

# Copy to server
scp mini-xdr-models-*.tar.gz user@server:/opt/mini-xdr/models/

# On server, extract
cd /opt/mini-xdr/models
tar -xzf mini-xdr-models-*.tar.gz
```

### Update Backend Config

```python
# backend/app/config.py
LOCAL_MODEL_DIR = "/opt/mini-xdr/models/local_trained"
```

## 🆚 Local vs SageMaker

| Feature | Local Training | SageMaker |
|---------|---------------|-----------|
| Cost | $0 (free!) | $40-60 one-time + $120-200/month |
| Speed | 2-8 hours (all models) | 30-60 min (all models) |
| Control | Full control | Limited |
| Scalability | Single machine | Auto-scaling |
| Deployment | Manual | Managed endpoints |
| Best for | Development, testing, cost savings | Production, high-throughput |

## 📚 Next Steps

After training models locally:

1. ✅ **Test inference**: `python3 aws/local_inference.py`
2. ✅ **Integrate with backend**: Update `ml_engine.py`
3. ✅ **Test end-to-end**: Send real traffic through your XDR
4. ✅ **Monitor performance**: Check detection rates
5. ✅ **Retrain periodically**: Update with new attack patterns

## 💡 Tips

1. **Start with General model only** - Train just the general model first to validate everything works
   ```bash
   python3 aws/train_local.py --models general
   ```

2. **Use GPU if available** - Training is 4-10x faster with GPU

3. **Monitor validation accuracy** - If it stops improving, training will stop early

4. **Save training logs** - Redirect output to log file for later analysis
   ```bash
   ./train_models_locally.sh 2>&1 | tee training.log
   ```

5. **Test before deploying** - Always test models with real data before production use

## 🤝 Support

Questions? Check:
- Training logs: `models/local_trained/training_summary.json`
- Model metadata: `models/local_trained/*/model_metadata.json`
- Training history: `models/local_trained/*/training_history.json`

Happy training! 🚀


