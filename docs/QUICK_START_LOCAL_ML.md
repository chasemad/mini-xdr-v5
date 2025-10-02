# 🚀 Quick Start: Local ML Training

## Your System Status

✅ **Apple Silicon GPU (MPS)** - Training will be FAST (~30-45 min/model)
✅ **1.6M training samples** - High-quality real attack data
✅ **All dependencies installed** - PyTorch 2.7.1 ready

## Three Steps to Working ML Models

### Step 1: Train Models (~2-3 hours total)

```bash
cd /Users/chasemad/Desktop/mini-xdr
./train_models_locally.sh
```

This will train:
- General model (7 attack types)
- DDoS specialist
- Brute Force specialist  
- Web Attack specialist

**Expected Output:**
```
Epoch [1/30] (25.3s)
  Train Loss: 0.3245 | Train Acc: 89.23%
  Val Loss:   0.2891 | Val Acc:   91.45%
  ✅ New best model!
```

Models saved to: `models/local_trained/`

### Step 2: Test Models (1 minute)

```bash
python3 aws/local_inference.py
```

**Expected Output:**
```
✅ Loaded general model (accuracy: 92.45%)
✅ Loaded ddos model (accuracy: 97.23%)
✅ Loaded brute_force model (accuracy: 94.12%)
✅ Loaded web_attacks model (accuracy: 91.88%)

Running inference...
Event 1: BruteForce (confidence: 0.945, threat: high)
Event 2: Normal (confidence: 0.887, threat: none)
```

### Step 3: Integrate with Backend (5 minutes)

Edit `backend/app/ml_engine.py`:

```python
# Add at top of file
from aws.local_inference import local_ml_client

# In EnhancedFederatedDetector.calculate_anomaly_score() method
# Around line 876, replace SageMaker code with:

async def calculate_anomaly_score(self, src_ip: str, events: List[Event]) -> float:
    """Enhanced anomaly scoring with local ML models"""
    try:
        # Get traditional ML score
        traditional_score = await self.federated_detector.calculate_anomaly_score(src_ip, events)
        
        # Get local deep learning score
        local_score = 0.0
        if await local_ml_client.health_check():
            sagemaker_events = [
                {
                    'src_ip': event.src_ip,
                    'dst_port': event.dst_port or 0,
                    'eventid': event.eventid,
                    'message': event.message or '',
                    'timestamp': event.ts.isoformat() if event.ts else None,
                    'raw': event.raw or {}
                }
                for event in events
            ]
            
            results = await local_ml_client.detect_threats(sagemaker_events)
            if results:
                local_score = results[0].get('anomaly_score', 0.0)
                self.logger.info(f"Local ML inference score: {local_score:.3f}")
        
        # Combine scores (70% local ML, 30% traditional)
        if local_score > 0:
            combined_score = 0.7 * local_score + 0.3 * traditional_score
            self.logger.info(f"Combined scoring - Local: {local_score:.3f}, "
                           f"Traditional: {traditional_score:.3f}, "
                           f"Combined: {combined_score:.3f}")
        else:
            combined_score = traditional_score
            
        return min(combined_score, 1.0)
        
    except Exception as e:
        self.logger.error(f"ML scoring failed: {e}")
        return await self.federated_detector.calculate_anomaly_score(src_ip, events)
```

## Command Cheat Sheet

```bash
# Full training (all models)
./train_models_locally.sh

# Train specific models only
python3 aws/train_local.py --models general ddos

# Fast training (fewer epochs)
python3 aws/train_local.py --epochs 20

# More thorough training
python3 aws/train_local.py --epochs 50 --patience 15

# Test inference
python3 aws/local_inference.py

# Check training results
cat models/local_trained/training_summary.json

# View model accuracy
cat models/local_trained/general/model_metadata.json | grep accuracy
```

## Expected Results

### General Model
- **Accuracy**: 85-95%
- **Classes**: Normal, DDoS, Recon, BruteForce, WebAttack, Malware, APT
- **Use**: Primary threat classification

### Specialist Models
- **DDoS**: 95-99% accuracy
- **Brute Force**: 90-98% accuracy
- **Web Attacks**: 88-96% accuracy
- **Use**: Confirm specific attack types with high confidence

## Troubleshooting

### Training is slow
- Check: Are you using MPS? `python3 -c "import torch; print(torch.backends.mps.is_available())"`
- Fix: Script auto-detects, but you can force with `--device mps`

### Out of memory
- Reduce batch size: `--batch-size 256`
- Or use CPU: `--device cpu`

### Low accuracy
- Train longer: `--epochs 50`
- Check data: `ls -lh aws/training_data/*.npy`

## What You Get

After training, you'll have:

```
models/local_trained/
├── general/
│   ├── threat_detector.pth       ← Main model (92% accuracy)
│   ├── model_metadata.json       ← Model info
│   └── training_history.json     ← Training curves
├── ddos/                          ← DDoS specialist (97% accuracy)
├── brute_force/                   ← Brute force specialist (94% accuracy)
├── web_attacks/                   ← Web attack specialist (91% accuracy)
└── training_summary.json          ← Overall results
```

## Cost Comparison

| Approach | Cost | Time | Control |
|----------|------|------|---------|
| **Local (this)** | $0 | 2-3 hours | Full |
| SageMaker | $40-60 + $120-200/mo | 30-60 min | Limited |

## Next Steps After Training

1. ✅ **Test models**: Verify accuracy meets expectations
2. ✅ **Integrate**: Update `ml_engine.py` (see Step 3 above)
3. ✅ **Deploy**: Copy models to production server
4. ✅ **Monitor**: Track detection rates in your XDR dashboard
5. ✅ **Retrain**: Update models monthly with new attack patterns

## Documentation

- **Full guide**: `LOCAL_ML_SETUP.md`
- **Training script**: `aws/train_local.py`
- **Inference client**: `aws/local_inference.py`

## Support

Having issues? Check:
1. System status: Run the system check above
2. Training logs: `models/local_trained/training_summary.json`
3. Model metadata: `models/local_trained/*/model_metadata.json`

## Time Estimate

With your Apple Silicon GPU:

- ☕ **General model**: 30-45 minutes
- ☕ **DDoS specialist**: 20-30 minutes  
- ☕ **Brute Force specialist**: 20-30 minutes
- ☕ **Web Attack specialist**: 20-30 minutes

**Total**: ~2-3 hours (can run overnight)

---

**Ready?** Just run: `./train_models_locally.sh`

The script will guide you through everything! 🚀


