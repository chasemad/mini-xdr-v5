# ML Training Guide

This guide explains how to retrain and manage Mini-XDR machine learning models using the artefacts and scripts in the repository.

## 1. Datasets

- **CICIDS2017 Dataset**: Official CSV archive stored in `datasets/cicids2017_official/` used for network intrusion training.
- **Real Datasets**: Curated cybersecurity datasets in `datasets/real_datasets/` including KDD Cup, honeypot logs, and threat feeds.
- **Synthetic Datasets**: Generated attack scenarios in `datasets/*.json` for model augmentation and testing.
- **Training Data Collector**: Automated data collection from live incidents via `backend/app/training_data_collector.py`.
- **Federated Datasets**: Privacy-preserving data aggregation across distributed tenants for collaborative learning.

## 2. Training Methods

### Local Training (Async Pipeline)

The backend exposes multiple async training pipelines that can be triggered programmatically:

```python
import asyncio
from backend.app.db import AsyncSessionLocal
from backend.app.enhanced_model_manager import train_enhanced_model

async def run():
    async with AsyncSessionLocal() as session:
        result = await train_enhanced_model(session, model_name="local_experiment")
        print(result)

asyncio.run(run())
```

This uses `EnhancedTrainingPipeline` (`backend/app/enhanced_training_pipeline.py`) with hard-example mining, class balancing, and comprehensive evaluation.

### Online Learning

Continuous model adaptation without full retraining:

```python
from backend.app.online_learning import trigger_online_adaptation
await trigger_online_adaptation()  # Adapts models to new threat patterns
```

### Federated Learning

Privacy-preserving collaborative training across tenants:

```python
from backend.app.federated_learning import initialize_federated_coordinator
await initialize_federated_coordinator({
    "participants": ["tenant1", "tenant2"],
    "model_type": "threat_detection"
})
```

### Ensemble Optimization

Optimize model weights and combinations:

```python
from backend.app.ensemble_optimizer import trigger_ensemble_optimization
await trigger_ensemble_optimization()  # Optimizes model ensemble performance
```

## 3. Batch Training Script

`scripts/ml-training/train-with-real-datasets.py` wraps training across synthetic and real datasets.
Run it from the repository root:

```bash
python3 scripts/ml-training/train-with-real-datasets.py --help
```

Use the script to list datasets, ingest them into the database, and trigger training runs.

## 4. Model Outputs

- **Model Weights**: `models/<model_name>/enhanced_threat_detector.pth` (PyTorch format)
- **Training Metadata**: `models/<model_name>/training_summary.json` with performance metrics, hyperparameters, and dataset information
- **Model Registry**: `models/model_registry.json` managed by `EnhancedModelManager` with versioning and rollback capabilities
- **Performance Metrics**: Real-time model performance tracking via `/api/ml/models/performance`
- **Explainability Data**: Model interpretation data for AI explainability features

## 5. Model Deployment & Management

### Activating Trained Models

```python
from backend.app.enhanced_model_manager import deploy_enhanced_model
asyncio.run(deploy_enhanced_model("local_experiment"))
```

### A/B Testing

```python
from backend.app.enhanced_model_manager import create_ab_test
await create_ab_test("new_model", "current_model", traffic_split=0.2)
# Monitor results via /api/ml/ab-test/{test_id}/results
```

### Rollback & Versioning

```python
from backend.app.enhanced_model_manager import rollback_model
await rollback_model("previous_version")  # Rollback to previous version
```

The backend loads active models on startup. Ensure the models directory is accessible in your deployment environment. For distributed deployments, models are automatically synchronized across nodes.

## 6. Automated Training & Monitoring

### Scheduled Retraining

The background scheduler triggers automated retraining based on multiple conditions:

- **Time-based**: Every 24 hours if `settings.auto_retrain_enabled` is true
- **Concept Drift**: Automatic retraining when model accuracy degrades (via `backend/app/concept_drift.py`)
- **Performance Thresholds**: Retraining triggered when detection rates fall below configured thresholds
- **Data Volume**: Retraining when sufficient new training data has been collected

### Continuous Learning

- **Online Learning**: Real-time model adaptation to new threat patterns without full retraining
- **Federated Updates**: Incremental model updates from distributed participants
- **Behavioral Learning**: Continuous adaptation to organizational-specific threat patterns

### Monitoring & Alerts

- **Model Performance**: Real-time monitoring via `/api/ml/models/performance` and `/api/ml/drift/status`
- **Training Status**: Monitor active training jobs via `/api/ml/status` and `/api/federated/status`
- **Alert Integration**: Automatic alerts when models require attention or retraining

### Resource Considerations

Confirm GPU/CPU capacity before enabling automated training. Distributed deployments can distribute training load across multiple nodes. Monitor resource usage via the telemetry API.

Document all training runs, model deployments, and performance changes in `change-control/audit-log.md`, including dataset versions, parameters, and resulting metrics.
