# System Overview

Mini-XDR is composed of the following tiers:

1. **FastAPI service** (`backend/app/main.py`)
   - Lifespan initialises secure environment (`secure_startup.py`), multi-tenant database, ML detectors,
     enhanced learning pipeline, distributed components, and APScheduler jobs.
   - Routers: incidents & response (`backend/app/response_optimizer.py`, `backend/app/responder.py`),
     NLP workflows (`backend/app/nlp_workflow_routes.py`, `backend/app/nlp_suggestion_routes.py`),
     trigger management (`backend/app/trigger_routes.py`), onboarding (`backend/app/onboarding_routes.py`),
     agent orchestration, webhook management, and telemetry.
   - Multi-tenancy: Tenant middleware (`backend/app/tenant_middleware.py`) provides organization-based
     isolation and data segmentation.
   - Security: HMAC middleware (`backend/app/security.py`), JWT authentication (`backend/app/auth.py`),
     API key guards, rate limiting, and strict security headers applied globally.

2. **Agent & automation layer**
   - Orchestrator (`backend/app/agent_orchestrator.py`) coordinates 12 specialised agents in
     `backend/app/agents/` including containment, forensics, deception, predictive hunter, attribution,
     DLP, EDR, IAM, ingestion, NLP analyzer, coordination hub, and HMAC signer.
   - Advanced response engine (`backend/app/advanced_response_engine.py`) and playbook engine
     (`backend/app/playbook_engine.py`) expose comprehensive action libraries consumed by SOC UI.
   - Enhanced learning pipeline (`backend/app/enhanced_training_pipeline.py`) with federated learning
     (`backend/app/federated_learning.py`), online learning (`backend/app/online_learning.py`),
     and concept drift detection (`backend/app/concept_drift.py`).
   - Agent enrollment and verification services (`backend/app/agent_enrollment_service.py`,
     `backend/app/agent_verification_service.py`) manage agent lifecycle and credentials.

3. **Machine-learning stack**
   - Ensemble detector (`backend/app/ml_engine.py`) and ensemble optimizer (`backend/app/ensemble_optimizer.py`)
     aggregate transformer, XGBoost, autoencoder, isolation forest, and deep learning models stored under `models/`.
   - Enhanced detector wrapper (`backend/app/enhanced_threat_detector.py`) and enhanced model manager
     (`backend/app/enhanced_model_manager.py`) handle model versioning and deployment.
   - Adaptive detection (`backend/app/adaptive_detection.py`), baseline engine (`backend/app/baseline_engine.py`),
     behavioral analyzer, and concept drift detection (`backend/app/concept_drift.py`) provide dynamic threat modeling.
   - Feature extraction (`backend/app/ml_feature_extractor.py`) and explainable AI (`backend/app/explainable_ai.py`)
     enhance model interpretability and performance.

4. **Frontend** (`frontend/app/`)
   - Next.js 15 App Router application with SOC surfaces, automation designer, and analytics widgets.
   - API access centralised in `frontend/app/lib/api.ts` with JWT + API key headers and
     incident-specific helpers.
   - Core SOC components in `frontend/components/` (TacticalDecisionCenter, EnhancedAIAnalysis,
     ThreatStatusBar), workflow components in `frontend/app/components/` (WorkflowDesigner,
     PlaybookCanvas), shared UI primitives under `frontend/components/ui/`, and SOC dashboards in
     `frontend/app/incidents/`.
   - Real-time updates via WebSocket manager (`backend/app/websocket_manager.py`) and webhook integration
     (`backend/app/webhook_manager.py`, `backend/app/webhook_routes.py`).

5. **Discovery & Network Services**
   - Network discovery service (`backend/app/discovery_service.py`) with asset classifier, network scanner,
     and vulnerability mapper in `backend/app/discovery/`.
   - Distributed coordination via Kafka manager (`backend/app/distributed/kafka_manager.py`) and Redis cluster
     (`backend/app/distributed/redis_cluster.py`).
   - MCP coordinator (`backend/app/distributed/mcp_coordinator.py`) for distributed processing.

6. **Infrastructure tooling**
   - Local orchestration via `scripts/start-all.sh` (handles dependency checks, port cleanup, backend
     + frontend launch, optional signed API calls).
   - Cloud deployment manifests located in `infrastructure/`, `k8s/`, and AWS buildspec files at repo
     root.
   - Documentation enforcement system with pre-commit hooks and CI/CD validation.

## Data Flow Summary

1. Logs/events arrive via ingest routes (`/ingest/*`, see `backend/app/multi_ingestion.py`).
2. Events are sanitised, stored through SQLAlchemy models (`backend/app/models.py`), and evaluated by
   `run_detection` (`backend/app/detect.py`).
3. Detected incidents trigger triage (`backend/app/triager.py`), context analysis, and may auto-block
   via `backend/app/responder.py` depending on policy and `settings.auto_contain`.
4. SOC operators interact through the Next.js UI, which consumes `/api/incidents/*`,
   `/api/response/*`, `/api/workflows/*` endpoints.
5. Advanced response actions and workflows use APScheduler + async tasks to execute on remote hosts or
   services where configured.
6. Telemetry and status data is served via `/api/telemetry/status` for dashboard summaries.

## Persistence

- SQLAlchemy async engine configured in `backend/app/db.py` with SQLite default, Postgres support via
  AsyncPG DSNs.
- Secrets retrieved from environment variables and optionally AWS Secrets Manager through
  `backend/app/config.py` + `backend/app/secrets_manager.py`.
- ML models loaded from local filesystem; S3/SageMaker integrations exist in
  `backend/app/sagemaker_client.py` and `backend/app/sagemaker_endpoint_manager.py` but require live
  credentials before use.
