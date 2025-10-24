# Roadmap & Status

_Last updated: 2024-10-24_

## Release Snapshot

- **Backend**: FastAPI app `backend/app/main.py` v1.1.0 running with async SQLAlchemy and
  APScheduler-driven background jobs. Supports multi-tenant architecture with tenant middleware.
- **Frontend**: Next.js 15 App Router targeting Node 20 (`frontend/package.json`, `scripts/start-all.sh`).
- **Models**: Local ensemble artefacts under `models/` loaded by `backend/app/ml_engine.py` and
  `backend/app/enhanced_threat_detector.py`. Enhanced training pipeline with federated learning support.
- **Infrastructure**: Local orchestration via `scripts/start-all.sh`; AWS and Azure manifests are
  present but require validation before production deployments. Kubernetes manifests in `k8s/`.

## Green

- **Authentication & Security**: HMAC authentication and rate limiting enforced for ingest routes
  (`backend/app/security.py`). JWT auth, API key fallback, and comprehensive onboarding flows
  exposed via `/api/auth` and `/api/onboarding/*` (`backend/app/auth.py`, `backend/app/onboarding_routes.py`).
- **Multi-tenant Architecture**: Tenant middleware and organization-based isolation implemented
  (`backend/app/tenant_middleware.py`).
- **Incident Management**: Full incident command center with realtime polling and WebSocket updates
  (`frontend/app/incidents/incident/[id]/page.tsx` and `frontend/app/hooks/useIncidentRealtime.ts`).
- **Response Automation**: Comprehensive response automation APIs (`/api/response/*`) with workflow
  execution, impact metrics, AI advisor integration, and playbook engine
  (`backend/app/advanced_response_engine.py`, `backend/app/playbook_engine.py`).
- **Agent Orchestration**: 12 specialized agents available (containment, forensics, deception,
  predictive hunter, attribution, DLP, EDR, IAM, ingestion, NLP analyzer, coordination hub, HMAC signer)
  orchestrated through `backend/app/agent_orchestrator.py`.
- **Secrets Management**: AWS Secrets Manager integration when `SECRETS_MANAGER_ENABLED=true`
  (`backend/app/config.py`, `backend/app/secrets_manager.py`).
- **ML Pipeline**: Enhanced training pipeline with federated learning, online learning, and
  concept drift detection (`backend/app/enhanced_training_pipeline.py`, `backend/app/federated_learning.py`).

## In Progress / Planned

- **Database migrations**: `backend/app/db.py:init_db` creates tables on startup; for production,
  implement Alembic migrations to manage schema changes safely.
- **Deployment validation**: AWS and Azure deployment documentation is being actively updated. Test
  Terraform/K8s configurations against the current v1.1.0 codebase before production use.
- **Test coverage**: `tests/` contains historical suites (pytest and shell). Expand test coverage for
  new auth endpoints, workflow execution, and trigger evaluation. Integrate into CI/CD pipeline.
- **Documentation maintenance**: Continue updating docs as features evolve. Remove legacy status files
  from root directory.
- **Observability**: Implement Prometheus metrics export and OpenTelemetry tracing. Configure log
  aggregation (CloudWatch/Azure Monitor) for production deployments.

## Next Decisions

1. **Database Strategy**: Finalize the database engine for production (managed Postgres vs SQLite) and
   implement proper Alembic migrations for schema management.
2. **CI/CD Pipeline**: Decide whether AWS CodeBuild deployment remains part of the standard workflow;
   if not, remove or archive `buildspec-*.yml` and related documentation.
3. **Agent Management**: Implement centralized agent credential management and enrollment workflow
   (currently 12 specialized agents available in `backend/app/agents/`).
4. **Distributed Architecture**: Evaluate whether to activate distributed components (Kafka, Redis cluster,
   MCP coordinator) for high-availability deployments.
5. **Federated Learning**: Determine rollout strategy for federated learning capabilities across
   multi-tenant environments.

Track these items in the issue tracker and link updates back to this file when status changes.
