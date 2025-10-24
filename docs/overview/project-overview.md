# Project Overview

Mini-XDR is a full-stack extended detection and response platform built around a FastAPI backend, a
Next.js 15 frontend, and a collection of local machine-learning pipelines. The current codebase
supports autonomous containment workflows, human-in-the-loop SOC tooling, and ML-driven detection
without relying on external SaaS services.

## Codebase Layout

- `backend/`: FastAPI application with async SQLAlchemy models, ingestion pipelines, agent
  orchestration, and AI-assisted response engines.
- `frontend/`: Next.js 15 app (App Router) with realtime incident views, automated response
  workflows, and SOC operator tooling.
- `scripts/`: Operational helpers including `start-all.sh` for local orchestration and
  authentication utilities (`scripts/auth/send_signed_request.py`).
- `models/` and `datasets/`: Local ML artefacts used by `backend/app/ml_engine.py` and
  `backend/app/enhanced_threat_detector.py`.
- `infrastructure/` and `k8s/`: Terraform/Kubernetes manifests for cloud deployment and cluster
  services.

## Runtime Services

| Service | Location | Description |
| --- | --- | --- |
| API | `backend/app/main.py` | FastAPI app with HMAC-protected ingest endpoints, incident lifecycle management, response automation, and telemetry routes. |
| Frontend | `frontend/app/` | Next.js UI consuming the API via `frontend/app/lib/api.ts`, including the incident command center and automation designer. |
| Worker Jobs | `backend/app/main.py` | APScheduler tasks trigger containment clean-up and periodic ML retraining. |
| ML Engines | `backend/app/ml_engine.py`, `backend/app/enhanced_threat_detector.py` | Ensemble detection running locally, plus model loading from `models/`. |
| Agents | `backend/app/agents/` | Specialized agents (containment, forensics, deception, predictive hunter, etc.) orchestrated through `backend/app/agent_orchestrator.py`. |

## Key Capabilities

- Multi-source log ingestion (`backend/app/multi_ingestion.py`) with per-source sanitisation and IOC
  extraction.
- Incident enrichment and automated triage via `backend/app/triager.py` and
  `backend/app/context_analyzer.py`.
- Human-executable and automated response actions exposed under `/api/response` and consumed by
  React components like `frontend/components/TacticalDecisionCenter.tsx`.
- NLP workflow support through `backend/app/nlp_workflow_routes.py` and `frontend/app/lib/api.ts`
  helpers such as `parseNlpWorkflow`.
- Auth stack featuring JWT bearer authentication (`backend/app/auth.py`), API keys, and HMAC device
  signatures enforced by `backend/app/security.py`.

Mini-XDR ships with SQLite defaults for development (`DATABASE_URL` in `.env`), but the SQLAlchemy
layer in `backend/app/db.py` accepts Postgres AsyncPG URLs for production clusters.
