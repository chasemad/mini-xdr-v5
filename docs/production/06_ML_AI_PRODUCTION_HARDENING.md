# 06: ML/AI Production Hardening

**Current State:** 4-model ensemble, basic federated learning  
**Target State:** MLOps with governance, monitoring, bias detection  
**Priority:** P1 (Required for AI-heavy customers)  
**Solo Effort:** 3-4 weeks

---

## Critical Production ML Requirements

### Model Governance
- ❌ No model versioning system
- ❌ No A/B testing framework  
- ❌ No model approval workflow
- ❌ No performance degradation alerts
- ❌ No bias/fairness testing

### ML Monitoring  
- ✅ Basic metrics in `/backend/app/ml_engine.py`
- ❌ No drift detection alerts
- ❌ No feature distribution monitoring
- ❌ No prediction explainability in production

---

## Implementation Checklist

### Task 1: Model Registry & Versioning

**File:** `/backend/app/models.py` - Add MLModel enhancements

```python
class MLModelVersion(Base):
    """Track model versions for governance"""
    __tablename__ = "ml_model_versions"
    
    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    model_name = Column(String(128), nullable=False, index=True)
    version = Column(String(32), nullable=False)  # v1.0.0, v2.1.3
    
    # Model artifacts
    model_path = Column(String(512), nullable=False)
    model_hash = Column(String(64), nullable=False)  # SHA256 of model file
    
    # Training metadata
    training_dataset_hash = Column(String(64))
    training_samples = Column(Integer)
    training_duration_seconds = Column(Integer)
    hyperparameters = Column(JSON)
    
    # Performance metrics
    accuracy = Column(Float)
    precision = Column(Float)
    recall = Column(Float)
    f1_score = Column(Float)
    
    # Deployment status
    status = Column(String(32), default="training")  # training|testing|staging|production|archived
    deployed_at = Column(DateTime(timezone=True), nullable=True)
    deprecated_at = Column(DateTime(timezone=True), nullable=True)
    
    # Governance
    approved_by_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    approval_notes = Column(Text, nullable=True)
    
    # A/B testing
    traffic_percentage = Column(Float, default=0.0)  # 0-100
```

**Checklist:**
- [ ] Add MLModelVersion table
- [ ] Create migration
- [ ] Implement model registration on training completion
- [ ] Add model approval workflow
- [ ] Track deployed model versions

### Task 2: Prediction Monitoring

**File:** `/backend/app/ml_monitoring.py` (NEW)

```python
"""ML model monitoring and drift detection"""
import numpy as np
from datetime import datetime, timedelta
from collections import deque

class DriftDetector:
    """Detect concept drift in ML models"""
    
    def __init__(self, window_size=1000, threshold=0.1):
        self.predictions = deque(maxlen=window_size)
        self.threshold = threshold
        self.baseline_mean = None
        self.baseline_std = None
    
    def update(self, prediction_score: float, actual_label: bool = None):
        """Update drift detector with new prediction"""
        self.predictions.append({
            "score": prediction_score,
            "actual": actual_label,
            "timestamp": datetime.now()
        })
        
        if self.baseline_mean is None and len(self.predictions) >= 100:
            self._establish_baseline()
    
    def _establish_baseline(self):
        """Establish baseline statistics"""
        scores = [p["score"] for p in list(self.predictions)[:100]]
        self.baseline_mean = np.mean(scores)
        self.baseline_std = np.std(scores)
    
    def check_drift(self) -> dict:
        """Check if drift has occurred"""
        if len(self.predictions) < 100:
            return {"drift_detected": False, "reason": "insufficient_data"}
        
        recent_scores = [p["score"] for p in list(self.predictions)[-100:]]
        recent_mean = np.mean(recent_scores)
        recent_std = np.std(recent_scores)
        
        # Statistical drift detection
        mean_shift = abs(recent_mean - self.baseline_mean) / self.baseline_std
        
        if mean_shift > self.threshold:
            return {
                "drift_detected": True,
                "metric": "mean_shift",
                "value": mean_shift,
                "baseline_mean": self.baseline_mean,
                "recent_mean": recent_mean
            }
        
        return {"drift_detected": False}


# Global drift detector
drift_detector = DriftDetector()


async def log_prediction(
    model_name: str,
    model_version: str,
    features: dict,
    prediction: float,
    confidence: float,
    db: AsyncSession
):
    """Log ML prediction for monitoring"""
    log = PredictionLog(
        model_name=model_name,
        model_version=model_version,
        features_hash=hashlib.md5(json.dumps(features, sort_keys=True).encode()).hexdigest(),
        prediction_score=prediction,
        confidence_score=confidence
    )
    db.add(log)
    
    # Update drift detector
    drift_detector.update(prediction)
    drift_status = drift_detector.check_drift()
    
    if drift_status["drift_detected"]:
        # Alert!
        logger.warning(f"Drift detected in model {model_name}: {drift_status}")
        await send_alert(f"ML Drift: {model_name}", drift_status)
```

**Checklist:**
- [ ] Implement drift detection
- [ ] Log all predictions
- [ ] Set up drift alerts
- [ ] Create drift visualization dashboard

### Task 3: Model Explainability

**File:** `/backend/app/explainability.py` (NEW)

```python
"""SHAP-based model explanations"""
import shap

class ModelExplainer:
    """Generate explanations for predictions"""
    
    def __init__(self, model, feature_names):
        self.model = model
        self.feature_names = feature_names
        self.explainer = None
    
    def initialize(self, background_data):
        """Initialize SHAP explainer with background data"""
        self.explainer = shap.TreeExplainer(self.model, background_data)
    
    def explain_prediction(self, features: np.ndarray) -> dict:
        """Generate SHAP explanation for a prediction"""
        shap_values = self.explainer.shap_values(features)
        
        # Get top contributing features
        feature_importance = {}
        for i, (feature, value) in enumerate(zip(self.feature_names, shap_values[0])):
            feature_importance[feature] = float(value)
        
        # Sort by absolute contribution
        sorted_features = sorted(
            feature_importance.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
        return {
            "top_positive": [(f, v) for f, v in sorted_features if v > 0][:5],
            "top_negative": [(f, v) for f, v in sorted_features if v < 0][:5],
            "all_features": dict(sorted_features)
        }
```

**File:** `/backend/app/main.py` - Add explanation endpoint

```python
@app.get("/api/ml/explain/{incident_id}")
async def explain_ml_prediction(
    incident_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Get ML explanation for an incident"""
    # Get incident and its features
    incident = await db.get(Incident, incident_id)
    features = incident.ml_features
    
    # Generate explanation
    explainer = ModelExplainer(ml_detector.models["xgboost"], ml_detector.feature_columns)
    explanation = explainer.explain_prediction(features)
    
    return {
        "incident_id": incident_id,
        "prediction_score": incident.risk_score,
        "explanation": explanation
    }
```

**Checklist:**
- [ ] Implement SHAP explainer
- [ ] Add explain endpoint
- [ ] Create explanation UI
- [ ] Test with various predictions

### Task 4: Bias Detection

**File:** `/backend/app/bias_detection.py` (NEW)

```python
"""Fairness and bias testing for ML models"""

def calculate_demographic_parity(predictions, protected_attribute):
    """Check if prediction rates are similar across demographics"""
    groups = {}
    for pred, attr in zip(predictions, protected_attribute):
        if attr not in groups:
            groups[attr] = []
        groups[attr].append(pred)
    
    # Calculate positive prediction rate per group
    rates = {}
    for group, preds in groups.items():
        rates[group] = sum(preds) / len(preds) if preds else 0
    
    # Check disparity
    max_rate = max(rates.values())
    min_rate = min(rates.values())
    disparity = (max_rate - min_rate) / max_rate if max_rate > 0 else 0
    
    return {
        "rates_by_group": rates,
        "disparity": disparity,
        "passes": disparity < 0.1  # 10% threshold
    }


async def test_model_fairness(model, test_data):
    """Run comprehensive fairness tests"""
    results = {}
    
    # Test 1: Demographic parity by IP geography
    geo_parity = calculate_demographic_parity(
        predictions=model.predict(test_data.features),
        protected_attribute=test_data.geo_region
    )
    results["geographic_parity"] = geo_parity
    
    # Test 2: Equal opportunity (true positive rate)
    # ... additional tests ...
    
    return results
```

**Checklist:**
- [ ] Implement bias detection
- [ ] Run quarterly fairness audits
- [ ] Document fairness metrics
- [ ] Add to compliance reporting

---

## Quick Wins (Solo Developer)

**Week 1:**
- [ ] Add model versioning table
- [ ] Log all predictions
- [ ] Create drift detector

**Week 2:**
- [ ] Implement SHAP explainer
- [ ] Add explanation API endpoint
- [ ] Test with production data

**Week 3:**
- [ ] Set up drift alerts
- [ ] Create monitoring dashboard
- [ ] Run initial bias audit

**Week 4:**
- [ ] Document ML governance process
- [ ] Create model approval workflow
- [ ] Test end-to-end

---

**Next:** `07_SECURITY_HARDENING.md`


