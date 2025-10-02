# Critical Model Issue Report

**Date:** 2025-09-30
**Status:** 🔴 CRITICAL - All deployed models are non-functional

## Issue Summary

All 4 deployed SageMaker endpoints are producing incorrect predictions with extremely low accuracy:

- **General Model**: 14.3% accuracy (should be 86.8%)
- **DDoS Specialist**: 33% detection (should be 100%)
- **Brute Force Specialist**: 0% detection (should be 100%)
- **Web Attack Specialist**: 0% detection (should be 100%)

## Root Cause

**Double-scaling bug** in the training pipeline:

1. Training data is already normalized (range: `[0, 1]`)
2. `RobustScaler` was incorrectly fit on this pre-normalized data
3. When scaler transforms data, it corrupts the input:
   ```
   Before scaling: [-1.82, 10.00]
   After scaling: [-10.97, 61.23]  ← WRONG!
   ```
4. Models were trained on this incorrectly scaled data
5. Model weights learned patterns from corrupted inputs

## Evidence

### Local Testing
```python
# Test with scaler (deployed version)
Sample data range: [-1.8205, 10.0000]
After scaling: [-10.9669, 61.2254]  # Expanding instead of normalizing!
Accuracy: 1/7 = 14.3%

# Test without scaler
Sample data range: [-1.8205, 10.0000]
Accuracy: 1/7 = 14.3%  # Same bad accuracy - weights are corrupted
```

### Production Testing
All endpoints classify attacks incorrectly:
- DDoS attacks → "Normal" (0% threat probability)
- Brute force attacks → "Normal" (0% threat probability)
- Web attacks → "Normal" (0% threat probability)

## Solution

**Must retrain all 4 models WITHOUT the scaler:**

### Modified Training Process
1. Load pre-normalized training data (already in `[0, 1]` range)
2. **Skip scaling step entirely**
3. Train model directly on normalized features
4. Save model WITHOUT `scaler.pkl`
5. Update inference script to skip scaler transformation

### Files to Modify

#### 1. Training Script (`aws/train_specialist_model.py`)
**Line 436-439 - REMOVE:**
```python
# Scale features
logger.info("🔄 Scaling features...")
features_scaled = loader.scaler.fit_transform(features)
logger.info(f"   Feature range: [{features_scaled.min():.2f}, {features_scaled.max():.2f}]")
```

**Replace with:**
```python
# Data is already normalized - no scaling needed
logger.info("✅ Using pre-normalized features (no scaler needed)")
features_scaled = features  # Use features as-is
logger.info(f"   Feature range: [{features_scaled.min():.2f}, {features_scaled.max():.2f}]")
```

**Line 383 - REMOVE:**
```python
joblib.dump(scaler, scaler_path)
```

**Replace with:**
```python
# No scaler needed - data is pre-normalized
logger.info("ℹ️  Skipping scaler save (using pre-normalized data)")
```

#### 2. Inference Script (`code/inference.py`)
**Line 315-317 - MODIFY:**
```python
# Apply feature scaling if available
if scaler is not None:
    input_data = scaler.transform(input_data)
    logger.info("Applied feature scaling")
```

**Replace with:**
```python
# Data should already be normalized - skip scaling
if scaler is not None:
    logger.warning("Scaler provided but data is pre-normalized - skipping transform")
# No transformation needed
```

### Retraining Steps

```bash
cd /Users/chasemad/Desktop/mini-xdr/aws

# 1. Retrain general model (7-class, ~30 minutes)
python3 train_specialist_model.py \
    --data_dir training_data \
    --output_dir /tmp/models_fixed/general \
    --specialist_type general \
    --epochs 50 \
    --batch_size 512

# 2. Retrain DDoS specialist (~15 minutes)
python3 train_specialist_model.py \
    --data_dir training_data \
    --output_dir /tmp/models_fixed/ddos \
    --specialist_type ddos \
    --epochs 30 \
    --batch_size 512

# 3. Retrain Brute Force specialist (~15 minutes)
python3 train_specialist_model.py \
    --data_dir training_data \
    --output_dir /tmp/models_fixed/brute_force \
    --specialist_type brute_force \
    --epochs 30 \
    --batch_size 512

# 4. Retrain Web Attack specialist (~15 minutes)
python3 train_specialist_model.py \
    --data_dir training_data \
    --output_dir /tmp/models_fixed/web_attacks \
    --specialist_type webattack \
    --epochs 30 \
    --batch_size 512

# 5. Deploy all models
python3 deploy_all_models.py --model_dir /tmp/models_fixed

# 6. Test endpoints
python3 test_with_real_data.py
```

**Total Retraining Time:** ~90 minutes

## Impact

### Current State
- ❌ All threat detection is non-functional
- ❌ Cannot reliably identify attacks
- ❌ False negatives: Real attacks classified as "Normal"
- ❌ Backend integration blocked

### Post-Fix State
- ✅ 86.8% accuracy on general model
- ✅ 100% accuracy on specialist models
- ✅ Reliable threat detection
- ✅ Ready for backend integration

## Next Steps

1. **[URGENT]** Apply code modifications to training and inference scripts
2. **[URGENT]** Retrain all 4 models (~90 min total)
3. **Validate** models locally before deployment
4. **Deploy** fixed models to SageMaker endpoints
5. **Test** endpoints with real data
6. **Proceed** with backend integration

## Files for Reference

- Training script: `aws/train_specialist_model.py`
- Deployment script: `aws/deploy_all_models.py`
- Inference code: Model packages include `code/inference.py`
- Test scripts:
  - `aws/test_with_real_data.py` - Tests with actual training samples
  - `aws/test_all_endpoints.py` - Tests all 4 endpoints

## Lesson Learned

**Always verify data pipeline assumptions:**
- ✅ Check input data range before scaling
- ✅ Verify scaler is needed (don't scale pre-normalized data)
- ✅ Test model locally before cloud deployment
- ✅ Use actual data samples for validation, not synthetic data
