# 🚀 Enhanced Training Guide - Full Dataset

**Date**: October 2, 2025  
**Purpose**: Train improved models using ALL real data + synthetic supplement

---

## 🎯 What's New

### Previous Training
- ❌ Only 1.6M samples used (partial data)
- ❌ CICIDS2017: Only 600k of 2.8M samples used
- ❌ No synthetic data supplement
- ❌ Suboptimal hyperparameters (general model: 66% accuracy)
- ❌ No data augmentation
- ❌ Standard cross-entropy loss

### Enhanced Training
- ✅ **4M+ samples** (ALL available real data)
- ✅ **CICIDS2017**: All 2.8M samples (not just 600k)
- ✅ **Synthetic supplement**: 10% of dataset for augmentation
- ✅ **Better hyperparameters**:
  - Epochs: 30 → **50**
  - Batch size: 512 → **256** (better generalization)
  - Learning rate: 0.001 → **0.0005** (more stable)
  - Patience: 10 → **15** (more chances to improve)
- ✅ **Data augmentation**:
  - Gaussian noise (σ=0.01)
  - Feature dropout (10%)
  - Mixup (optional)
- ✅ **Focal loss** for class imbalance
- ✅ **Cosine annealing** learning rate schedule
- ✅ **Gradient clipping** for stability

---

## 📊 Dataset Breakdown

| Source | Samples | Status |
|--------|---------|--------|
| **UNSW-NB15** | 500k | ✅ Included |
| **CICIDS2017 (full)** | 2.8M | ✅ **NOW INCLUDED** (was 600k) |
| **KDD Cup 99** | 300k | ✅ Included |
| **Malware & Threat Intel** | 300k | ✅ Included |
| **Synthetic Supplement** | 400k | ✅ **NEW** (10% of real data) |
| **TOTAL** | **4.3M** | ✅ **vs 1.6M before** |

### Class Distribution (Expected)
```
Class 0 (Normal):        ~800k samples (18%)
Class 1 (DDoS):         ~900k samples (21%)
Class 2 (Reconnaissance): ~700k samples (16%)
Class 3 (Brute Force):   ~600k samples (14%)
Class 4 (Web Attack):    ~700k samples (16%)
Class 5 (Malware):       ~400k samples (9%)
Class 6 (APT):           ~200k samples (5%)
```

---

## 🚀 Quick Start (3 Commands)

```bash
# 1. Navigate to project directory
cd /Users/chasemad/Desktop/mini-xdr

# 2. Run the training script
./train_enhanced_models.sh

# 3. Wait ~60-90 minutes (4M samples take longer than 1.6M)
```

That's it! The script will:
- ✅ Load ALL 4M+ samples
- ✅ Train 4 models (general + 3 specialists)
- ✅ Apply all improvements automatically
- ✅ Save models to `models/local_trained_enhanced/`

---

## ⚙️ Advanced Options

### Train Specific Models Only
```bash
# General model only (7-class)
python3 aws/train_enhanced_full_dataset.py --models general

# Specialists only
python3 aws/train_enhanced_full_dataset.py --models ddos brute_force web_attacks
```

### Adjust Hyperparameters
```bash
# More epochs for better accuracy
python3 aws/train_enhanced_full_dataset.py --epochs 100 --patience 20

# Slower learning rate for stability
python3 aws/train_enhanced_full_dataset.py --learning-rate 0.0001

# Smaller batch size for better generalization (slower training)
python3 aws/train_enhanced_full_dataset.py --batch-size 128
```

### Disable Enhancements (for comparison)
```bash
# No data augmentation
python3 aws/train_enhanced_full_dataset.py --no-augmentation

# Standard cross-entropy instead of focal loss
python3 aws/train_enhanced_full_dataset.py --no-focal-loss

# No synthetic data
python3 aws/train_enhanced_full_dataset.py --no-synthetic
```

### Adjust Synthetic Data Ratio
```bash
# 20% synthetic (instead of default 10%)
python3 aws/train_enhanced_full_dataset.py --synthetic-ratio 0.2

# 5% synthetic (minimal supplement)
python3 aws/train_enhanced_full_dataset.py --synthetic-ratio 0.05
```

---

## ⏱️ Expected Training Times

**Hardware: Apple Silicon GPU (MPS)**

| Model | Old (1.6M samples) | New (4.3M samples) |
|-------|-------------------|-------------------|
| General | 13.6 min | ~35-40 min |
| DDoS Specialist | 4.6 min | ~12-15 min |
| BruteForce Specialist | 3.2 min | ~8-10 min |
| WebAttack Specialist | 2.8 min | ~7-9 min |
| **TOTAL** | **22.6 min** | **~60-75 min** |

**Note**: Training will be 2.5-3× longer due to 2.7× more data, but accuracy should improve significantly!

---

## 📈 Expected Improvements

### General Model (7-class)
```
Before: 66.02% accuracy
Target: 80-85% accuracy
Expected: 75-82% accuracy (with full dataset + improvements)
```

### Specialist Models
```
DDoS:
  Before: 99.37% (already excellent)
  Expected: 99.4-99.6% (marginal improvement)

Brute Force:
  Before: 94.70%
  Expected: 96-97% (more training data helps)

Web Attack:
  Before: 79.73%
  Expected: 85-88% (most improvement expected)
```

---

## 📁 Output Files

### Model Files
```
models/local_trained_enhanced/
├── general/
│   └── threat_detector.pth
├── ddos_specialist/
│   └── threat_detector.pth
├── brute_force_specialist/
│   └── threat_detector.pth
├── web_attacks_specialist/
│   └── threat_detector.pth
└── training_summary.json
```

### Training Logs
```
training_enhanced.log            # Detailed training log
best_general.pth                 # Best checkpoint (temp)
best_ddos_specialist.pth         # Best checkpoint (temp)
best_brute_force_specialist.pth  # Best checkpoint (temp)
best_web_attacks_specialist.pth  # Best checkpoint (temp)
```

---

## 🔍 Monitoring Training

### Watch Progress in Real-Time
```bash
# Tail the training log
tail -f training_enhanced.log

# Watch GPU usage (if on Apple Silicon)
sudo powermetrics --samplers gpu_power -i 5000

# Check training status
./check_training_status.sh
```

### Understanding Log Output
```
Epoch 1/50: Train Loss=0.5234, Train Acc=0.7823 | Val Loss=0.4891, Val Acc=0.8012 | LR=0.000500
             ↑                 ↑                   ↑                ↑                 ↑
          Epoch #          Training           Validation        Validation      Learning
                          accuracy            loss              accuracy        rate
```

**Good signs:**
- ✅ Val Acc increasing over epochs
- ✅ Val Loss decreasing
- ✅ Gap between Train Acc and Val Acc < 10% (not overfitting)

**Bad signs:**
- ❌ Val Acc not improving after many epochs → Stop and adjust hyperparameters
- ❌ Train Acc >> Val Acc (e.g., 95% vs 70%) → Overfitting, increase dropout
- ❌ Val Loss increasing while Train Loss decreasing → Overfitting

---

## 🔬 After Training: Validation

### 1. Check Training Summary
```bash
cat models/local_trained_enhanced/training_summary.json | python3 -m json.tool
```

### 2. Compare to Old Models
```python
import json

# Load old results
with open('models/local_trained/training_summary.json') as f:
    old = json.load(f)

# Load new results
with open('models/local_trained_enhanced/training_summary.json') as f:
    new = json.load(f)

# Compare
for model_type in ['general', 'ddos_specialist', 'brute_force_specialist', 'web_attacks_specialist']:
    old_acc = [r for r in old['results'] if r['model_name'] == model_type][0]['best_val_accuracy']
    new_acc = [r for r in new['results'] if r['model_name'] == model_type][0]['best_val_accuracy']
    improvement = (new_acc - old_acc) * 100
    print(f"{model_type:30s}: {old_acc:.4f} → {new_acc:.4f} ({improvement:+.2f}%)")
```

### 3. Test on Real Events
```bash
# Test with backend integration
python3 test_backend_integration_formats.py

# Test with real attack scenarios
python3 test_ml_integration.py
```

### 4. Deploy if Better
```bash
# Backup old models
cp -r models/local_trained models/local_trained_backup_$(date +%Y%m%d)

# Deploy new models
cp -r models/local_trained_enhanced/* models/local_trained/

# Restart backend
cd backend
pkill -f "uvicorn"
uvicorn app.main:app --reload
```

---

## 🐛 Troubleshooting

### Issue: Out of Memory Error
```
RuntimeError: CUDA out of memory
```

**Solution**: Reduce batch size
```bash
python3 aws/train_enhanced_full_dataset.py --batch-size 128
# Or even smaller: --batch-size 64
```

### Issue: Training Too Slow
```
Epoch 1/50 taking > 10 minutes
```

**Solutions**:
1. Check if GPU is being used:
   ```python
   import torch
   print(f"Using: {torch.backends.mps.is_available()}")
   ```

2. Reduce dataset size temporarily:
   ```bash
   # Use 50% of data for faster iteration
   python3 aws/train_enhanced_full_dataset.py --synthetic-ratio 0 --epochs 30
   ```

3. Train on fewer models first:
   ```bash
   # Train general model first to validate setup
   python3 aws/train_enhanced_full_dataset.py --models general --epochs 10
   ```

### Issue: Models Not Improving
```
Val Acc stuck at 60-65% after many epochs
```

**Solutions**:
1. **Lower learning rate**:
   ```bash
   python3 aws/train_enhanced_full_dataset.py --learning-rate 0.0001
   ```

2. **More epochs + patience**:
   ```bash
   python3 aws/train_enhanced_full_dataset.py --epochs 100 --patience 25
   ```

3. **Check data distribution**:
   ```python
   import numpy as np
   y = np.load('aws/training_data/training_labels_20250929_062520.npy')
   unique, counts = np.unique(y, return_counts=True)
   for u, c in zip(unique, counts):
       print(f"Class {u}: {c:,} ({c/len(y)*100:.1f}%)")
   ```

### Issue: Validation Accuracy Lower Than Expected
```
Expected 80%, got 70%
```

**Possible causes:**
1. **Data distribution mismatch** - Training data doesn't match real events
2. **Overfitting** - Model memorizing training data
3. **Insufficient epochs** - Need more training

**Solutions**:
1. Add more data augmentation
2. Increase dropout rate
3. Use ensemble of multiple models

---

## 📚 Key Improvements Explained

### 1. Focal Loss
**Why**: Standard cross-entropy treats all classes equally, but some classes have 10× more samples than others.

**How it helps**: Focuses on hard-to-classify examples and underrepresented classes.

### 2. Data Augmentation
**Why**: Prevents overfitting by creating slight variations of training samples.

**Types**:
- **Gaussian noise**: Adds random noise (σ=0.01) to make model robust
- **Feature dropout**: Randomly drops 10% of features to prevent feature co-adaptation

### 3. Cosine Annealing LR Schedule
**Why**: Learning rate that oscillates helps escape local minima.

**Benefit**: Better convergence compared to static or exponential decay.

### 4. Smaller Batch Size (512 → 256)
**Why**: Smaller batches have more noise, which helps generalization.

**Trade-off**: Slower training, but better final accuracy.

### 5. Lower Initial LR (0.001 → 0.0005)
**Why**: Larger datasets benefit from slower, more careful learning.

**Benefit**: More stable training with less overfitting.

---

## 🎯 Success Criteria

After training completes, your models should achieve:

| Model | Target Accuracy | Baseline | Status |
|-------|----------------|----------|--------|
| General | > 80% | 66% | 🎯 Goal |
| DDoS | > 99% | 99.37% | ✅ Maintain |
| BruteForce | > 95% | 94.70% | 📈 Improve |
| WebAttack | > 85% | 79.73% | 📈 Improve |

**If targets not met:**
- Retrain with 100 epochs
- Try ensemble methods
- Collect more targeted training data for weak classes

---

## 📞 Need Help?

1. **Check logs**: `training_enhanced.log`
2. **Review summary**: `models/local_trained_enhanced/training_summary.json`
3. **Compare to baseline**: Old models in `models/local_trained/`

---

**Ready to train? Run: `./train_enhanced_models.sh`** 🚀

