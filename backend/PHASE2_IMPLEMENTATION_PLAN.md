# Phase 2 Implementation Plan: Advanced ML System

**Goal**: Improve ML accuracy from 72.7% → 85%+ through automated retraining, feature engineering, and model improvements.

---

## Overview

Phase 2 builds on the Council of Models foundation to create a self-improving ML system that:
1. Automatically retrains models using Council corrections
2. Pre-computes and caches features for faster inference
3. Balances training data to eliminate class imbalance
4. Implements advanced feature engineering (Feature #80-#100)

---

## Task Breakdown

### **Task 1: Automated Retraining Pipeline** (60 minutes)

**Goal**: Use Council corrections to continuously improve local ML models

**Components**:

1. **Training Data Collector** (`app/learning/training_collector.py`)
   - Collect incidents with Council verdicts
   - Extract features + Council labels
   - Store in training buffer (PostgreSQL + S3/local file)
   - Schema: `(features[79], council_label, ml_label, incident_id, timestamp)`

2. **Retrain Scheduler** (`app/learning/retrain_scheduler.py`)
   - Background task that triggers retraining
   - Conditions:
     - 1000+ new labeled samples
     - 7 days since last retrain
     - Council override rate > 15% (indicates model drift)
   - Uses APScheduler or Celery

3. **Model Retrainer** (`app/learning/model_retrainer.py`)
   - Load current model + new training data
   - Apply class balancing (SMOTE/ADASYN)
   - Retrain with early stopping
   - Validate on hold-out set
   - Only deploy if accuracy improves by 2%+
   - Backup old model before replacing

**Metrics Tracked**:
- Training samples collected
- Retraining frequency
- Model version history
- Accuracy improvements per version

**Files to Create**:
- `app/learning/training_collector.py`
- `app/learning/retrain_scheduler.py`
- `app/learning/model_retrainer.py`
- `app/learning/__init__.py` (update)

**Files to Modify**:
- `app/intelligent_detection.py` - Call training collector after Council verdict
- `app/models.py` - Add `training_samples` table

---

### **Task 2: Feature Store with Pre-computation** (45 minutes)

**Goal**: Pre-compute expensive features and cache for fast inference

**Components**:

1. **Feature Store** (`app/features/feature_store.py`)
   - Pre-compute baseline features for all IPs
   - Store in Redis with longer TTL (1 hour)
   - Incremental updates as new events arrive
   - Schema: `baseline_features:{src_ip}` → first 20 features

2. **Feature Pipeline** (`app/features/feature_pipeline.py`)
   - Separate feature extraction into:
     - **Static features**: Computed once (IP geolocation, reputation)
     - **Temporal features**: Computed per window (event rates, inter-arrival time)
     - **Behavioral features**: Computed from sequence (LSTM input)
   - Pipeline stages run in parallel

3. **Feature Versioning** (`app/features/feature_versions.py`)
   - Track feature schema versions
   - Handle migrations when adding new features
   - Ensure backward compatibility

**Performance Targets**:
- Baseline features: <5ms (from cache)
- Temporal features: <30ms (from recent events)
- Total inference: <50ms (cached) or <100ms (cold start)

**Files to Create**:
- `app/features/feature_store.py`
- `app/features/feature_pipeline.py`
- `app/features/feature_versions.py`
- `app/features/__init__.py`

**Files to Modify**:
- `app/ml_feature_extractor.py` - Use feature store
- `app/config.py` - Add feature store settings

---

### **Task 3: Enhanced ML Models with Class Balancing** (60 minutes)

**Goal**: Eliminate 79.6% class imbalance to boost minority class accuracy

**Improvements**:

1. **Data Augmentation** (`app/learning/data_augmentation.py`)
   - SMOTE for synthetic minority samples
   - ADASYN for adaptive sampling
   - Time-series augmentation (jitter, scaling, rotation)
   - Target distribution: 30% normal, 70% attacks (balanced across attack types)

2. **Model Architecture Updates** (`app/deep_learning_models.py`)
   - **Weighted Loss Functions**:
     - Class weights inversely proportional to frequency
     - Focal loss for hard examples
   - **Ensemble Voting**:
     - Soft voting with calibrated probabilities
     - Weighted by per-class accuracy
   - **Threshold Optimization**:
     - Per-class decision thresholds (not just 0.5)
     - Optimize for F1 score instead of accuracy

3. **Specialist Model Training** (`scripts/train_specialist_models.py`)
   - Train 6 specialist models (one per attack type)
   - Each specialist: 95%+ accuracy on its class
   - Router model: Predict which specialist to use
   - Fallback to general model if uncertain

**Expected Improvements**:
- General model: 72.7% → 82% (balanced)
- Specialist models: 93%+ per class
- Ensemble: 85%+ overall accuracy

**Files to Create**:
- `app/learning/data_augmentation.py`
- `scripts/train_specialist_models.py`

**Files to Modify**:
- `app/deep_learning_models.py` - Update loss functions, ensemble logic
- `app/training/train_models.py` - Add class balancing

---

### **Task 4: Advanced Feature Engineering (Features #80-#100)** (45 minutes)

**Goal**: Add 20 new features to improve detection accuracy

**New Features**:

**Threat Intelligence Features (5)**:
- `feature_80`: Grok threat score (0-1) [Already implemented via Council]
- `feature_81`: IP reputation score (AbuseIPDB)
- `feature_82`: Domain age (recently registered = suspicious)
- `feature_83`: Autonomous System (AS) reputation
- `feature_84`: Tor exit node indicator

**Behavioral Sequence Features (10)**:
- `feature_85-89`: LSTM hidden states (top 5 dimensions)
- `feature_90`: Command sequence entropy
- `feature_91`: Timing regularity score (human vs bot)
- `feature_92`: Mouse/keyboard pattern (if available)
- `feature_93`: Session duration z-score
- `feature_94`: Failed auth sequence pattern

**Network Graph Features (5)**:
- `feature_95`: IP connectivity (how many hosts contacted)
- `feature_96`: Clustering coefficient (part of botnet?)
- `feature_97`: Betweenness centrality
- `feature_98`: Source IP ASN change frequency
- `feature_99`: Destination port diversity entropy

**Composite Feature (1)**:
- `feature_100`: ML ensemble agreement score (0-1)

**Files to Create**:
- `app/features/advanced_features.py`
- `app/features/threat_intel_features.py`
- `app/features/behavioral_features.py`
- `app/features/network_graph_features.py`

**Files to Modify**:
- `app/ml_feature_extractor.py` - Extend to 100 dimensions
- `app/deep_learning_models.py` - Update input size (79→100)

---

## Implementation Priority

**Critical Path** (Must complete for Phase 2):
1. Task 3: Class balancing (biggest accuracy improvement)
2. Task 1: Automated retraining (enables continuous learning)
3. Task 2: Feature store (performance optimization)
4. Task 4: Advanced features (incremental accuracy gains)

**Quick Wins** (Low effort, high impact):
- Add weighted loss functions (30 min, +5% accuracy)
- Implement class balancing (45 min, +8% accuracy)
- Pre-compute baseline features (30 min, 2x faster inference)

---

## Success Criteria

**Phase 2 Complete When**:
- [ ] ML accuracy ≥ 85% on test set (measured per-class)
- [ ] Automated retraining pipeline operational
- [ ] Feature store reduces inference time by 30%+
- [ ] Class imbalance reduced to <40% (from 79.6%)
- [ ] 100-dimensional feature vector implemented
- [ ] All new models deployed and validated

**Key Metrics**:
- General model accuracy: 72.7% → 82%+
- Specialist model accuracy: 93%+ per class
- Inference latency: <50ms (cached), <100ms (cold)
- Retraining frequency: Weekly (or after 1000 new samples)
- False positive rate: <5%

---

## Estimated Timeline

- **Task 1** (Automated Retraining): 60 min
- **Task 2** (Feature Store): 45 min
- **Task 3** (Class Balancing): 60 min
- **Task 4** (Advanced Features): 45 min

**Total**: 3.5 hours (without testing)
**With Testing**: 4-5 hours

---

## Dependencies

**External Services**:
- Redis (already running) - Feature caching
- PostgreSQL (already configured) - Training data storage
- Qdrant (already running) - Vector similarity search

**Python Libraries** (add to requirements.txt):
- `imbalanced-learn==0.12.0` - SMOTE/ADASYN for class balancing
- `scikit-optimize==0.10.2` - Hyperparameter tuning
- `category_encoders==2.6.4` - Advanced feature encoding

**Training Data**:
- Existing: CICIDS2017 (4.4M samples, 79 features)
- New: Council-corrected incidents (accumulating)

---

## Rollback Plan

If Phase 2 models perform worse:
1. Keep old models in `models/v1/` directory
2. Feature flag: `USE_PHASE2_MODELS=false`
3. Revert to Phase 1 (Council + 72.7% ML)
4. Analyze failures, retrain, redeploy

---

**Next Steps**: Start with Task 3 (Class Balancing) as it provides the biggest immediate accuracy boost.
