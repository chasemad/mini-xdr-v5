# Model Operations Runbook

## Checking Active Models

- **API Method**: Use `/api/ml/status` to get comprehensive model status, active deployments, and performance metrics.
- **Registry Inspection**: Examine `models/model_registry.json` for persisted metadata, versions, and deployment history.
- **Performance Monitoring**: Check `/api/ml/models/performance` for real-time model performance metrics.
- **Ensemble Status**: Monitor `/api/ml/ensemble/status` for ensemble model health and component performance.
- **Federated Status**: Check `/api/federated/status` for distributed model coordination status.

## Deploying New Models

1. **Training**: Train or copy model artefacts into `models/<name>/` using the training pipelines.
2. **Validation**: Run performance validation and A/B testing using `/api/ml/ab-test/create`.
3. **Deployment**: Execute `deploy_enhanced_model("<name>")` or use the API endpoints for safe deployment.
4. **Verification**: Monitor `/api/ml/status` to confirm successful deployment and performance.
5. **Gradual Rollout**: Use A/B testing to gradually roll out new models with traffic splitting.

## Rolling Back Models

- **API Method**: Use `deploy_enhanced_model(<previous_name>)` or model rollback endpoints.
- **Version Control**: The enhanced model manager maintains version history and rollback capabilities.
- **Safe Rollback**: Previous model versions remain intact; rollback is instantaneous without restart.
- **Monitoring**: Track rollback impact via performance metrics and alert systems.

## Validating Performance

- **Real-time Monitoring**: Track detection accuracy and performance via `/api/ml/models/performance`.
- **A/B Testing**: Compare model performance using `/api/ml/ab-test/{test_id}/results`.
- **Explainability**: Use `/api/ml/explain/{incident_id}` to validate model decision-making.
- **Regression Testing**: Run automated tests including `tests/test_model_detection.py` and `tests/test_enhanced_capabilities.py`.
- **Concept Drift**: Monitor `/api/ml/drift/status` for model degradation over time.
- **Ensemble Validation**: Check component model performance via `/api/ml/ensemble/status`.

## Synchronising Artefacts

- **Object Storage**: Store models in S3/Azure Blob Storage for production deployments with versioning.
- **Distributed Sync**: Models automatically synchronize across distributed nodes via the model manager.
- **Container Deployments**: Use init containers to download models before application startup.
- **Permissions**: Ensure backend process has read access to model files and directories.
- **Federated Models**: Secure synchronization of federated learning updates across participants.

## Troubleshooting

| Issue | Likely Cause | Fix |
| --- | --- | --- |
| Backend logs `Enhanced detector models not found` | Models directory missing or inaccessible. | Copy models to expected path or adjust `ML_MODELS_PATH` in config. |
| Training fails with CUDA errors | GPU not available or CUDA not configured. | Set `TORCH_DEVICE=cpu` or install correct CUDA drivers and PyTorch GPU version. |
| Model performance degradation | Concept drift or dataset shift. | Check `/api/ml/drift/status` and trigger retraining if needed. |
| Federated learning not working | Network connectivity or participant auth issues. | Verify `/api/federated/status` and participant credentials. |
| A/B test shows no traffic | Traffic splitting configuration error. | Check A/B test configuration and ensure proper routing. |
| Model explainability fails | Missing model metadata or incompatible format. | Regenerate model with explainability features enabled. |
| Distributed model sync fails | Network or permission issues between nodes. | Check distributed system logs and network connectivity. |
