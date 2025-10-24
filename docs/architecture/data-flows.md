# Data Flows & Integrations

## Ingestion

- **Sources**: `/ingest/multi` (`backend/app/multi_ingestion.py`) accepts Cowrie, Suricata, OSQuery,
  and generic JSON payloads. Ingestion agent (`backend/app/agents/ingestion_agent.py`) handles
  advanced parsing and normalization.
- **Multi-tenancy**: Tenant middleware (`backend/app/tenant_middleware.py`) ensures data isolation
  by organization during ingestion.
- **Authentication**: HMAC headers (`X-Device-ID`, `X-TS`, `X-Nonce`, `X-Signature`) validated by
  `AuthMiddleware` for ingest paths. Agent verification service (`backend/app/agent_verification_service.py`)
  manages device authentication.
- **Storage**: Events persisted through SQLAlchemy models in `backend/app/models.py` using the async
  session from `backend/app/db.py`. Training data collector (`backend/app/training_data_collector.py`)
  captures data for ML model improvement.

## Detection & Triage

1. **Initial Detection**: `run_detection` (`backend/app/detect.py`) evaluates new events against the ML ensemble
   (`backend/app/ml_engine.py`), intelligent detection (`backend/app/intelligent_detection.py`), and rule engines.
2. **Context Enrichment**: Adaptive detection (`backend/app/adaptive_detection.py`) and behavioral analyzer
   enrich event context, referencing behavioral baselines (`backend/app/baseline_engine.py`) and threat intel
   (`backend/app/external_intel.py`).
3. **Multi-stage Analysis**: Intelligent detection and system detector (`backend/app/system_detector.py`)
   perform advanced threat correlation and pattern analysis.
4. **Triage Process**: Local triager (`backend/app/local_triager.py`) and main triager (`backend/app/triager.py`)
   generate triage notes, risk scoring, and IOC extraction.
5. **Incident Creation**: Incidents are inserted into the incidents table with multi-tenant isolation and
   exposed via `/api/incidents` routes.

## Response Automation

- **Policy Engine**: Policy engine (`backend/app/policy_engine.py`) evaluates response actions against
  organizational policies and approval workflows.
- **Auto Containment**: Enhanced containment (`backend/app/enhanced_containment.py`) and responder
  (`backend/app/responder.py`) handle immediate actions (e.g., `block_ip`, `isolate_host`) based on
  `settings.auto_contain` and policy evaluation.
- **Advanced Actions**: Advanced response engine (`backend/app/advanced_response_engine.py`) and playbook engine
  (`backend/app/playbook_engine.py`) enumerate supported actions with rollback metadata and effectiveness tracking.
- **Agent Coordination**: Agent orchestrator (`backend/app/agent_orchestrator.py`) coordinates specialized agents
  (containment, forensics, deception, etc.) for complex response operations.
- **Workflows**: `/api/response/workflows/*` endpoints enable queued, multi-step playbooks with execution tracking,
  impact metrics (`backend/app/response_analytics.py`), and learning response engine (`backend/app/learning_response_engine.py`).
- **NLP Integration**: NLP workflow parser (`backend/app/nlp_workflow_parser.py`) and NLP suggestion routes
  (`backend/app/nlp_suggestion_routes.py`) convert natural language into executable workflows.

## Telemetry & Monitoring

- **Real-time Updates**: WebSocket manager (`backend/app/websocket_manager.py`) provides broadcasts for
  realtime UI updates via hooks such as `frontend/app/hooks/useIncidentRealtime.ts`.
- **Webhook Integration**: Webhook manager (`backend/app/webhook_manager.py`) and routes
  (`backend/app/webhook_routes.py`) enable external system notifications.
- **Telemetry API**: `/api/telemetry/status` returns per-organization counts for assets, agents, incidents,
  and last event timestamps (`backend/app/main.py`).
- **Security Monitoring**: Rate limiting and nonce tracking rely on the `agent_credentials` and
  `request_nonces` tables managed in `backend/app/security.py` and `backend/app/models.py`.
- **Distributed Coordination**: Kafka manager (`backend/app/distributed/kafka_manager.py`) and Redis cluster
  (`backend/app/distributed/redis_cluster.py`) enable distributed processing and state management.

## Machine Learning & Learning Flows

- **Continuous Learning**: Learning pipeline (`backend/app/learning_pipeline.py`) and enhanced training pipeline
  (`backend/app/enhanced_training_pipeline.py`) perform continuous model retraining.
- **Federated Learning**: Federated learning coordinator (`backend/app/federated_learning.py`) enables
  privacy-preserving model training across distributed tenants.
- **Online Learning**: Online learning engine (`backend/app/online_learning.py`) adapts models in real-time
  to new threat patterns.
- **Concept Drift**: Concept drift detection (`backend/app/concept_drift.py`) monitors model performance
  and triggers retraining when accuracy degrades.
- **Model Management**: Enhanced model manager (`backend/app/enhanced_model_manager.py`) handles versioning,
  deployment, and rollback of ML models.

## External Integrations

- **Secrets Management**: AWS Secrets Manager integration (`backend/app/secrets_manager.py`) for secure
  credential storage and retrieval.
- **Threat Intelligence**: External intel service (`backend/app/external_intel.py`) integrates with
  AbuseIPDB, VirusTotal, and other threat feeds.
- **Honeypot Integration**: T-Pot verifier (`backend/app/tpot_verifier.py`) manages honeypot state
  and integration with deception networks.
- **Cloud ML**: SageMaker client (`backend/app/sagemaker_client.py`) and endpoint manager
  (`backend/app/sagemaker_endpoint_manager.py`) for cloud-based model deployment.
- **Discovery Services**: Network discovery service (`backend/app/discovery_service.py`) integrates
  with asset management and vulnerability scanning systems.
