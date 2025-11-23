# API Reference

The FastAPI application in `backend/app/main.py` exposes REST endpoints grouped under the following
categories. Refer to the source code for full schema definitions; this document highlights the core
operations in the running build.

## Authentication (`/api/auth/*`)

| Method | Path | Description |
| --- | --- | --- |
| POST | `/api/auth/register` | Create organisation + admin user. Requires password meeting policy (12+ chars, complexity). |
| POST | `/api/auth/login` | Returns access (8h) and refresh (30d) JWT tokens. |
| POST | `/api/auth/refresh` | Exchange refresh token for new access token. |
| GET | `/api/auth/me` | Current user profile (requires bearer token). |
| POST | `/api/auth/invite` | Invite additional users (admin-only). |
| POST | `/api/auth/logout` | Invalidates refresh tokens. |

## Onboarding (`/api/onboarding/*`)

| Method | Path | Description |
| --- | --- | --- |
| POST | `/api/onboarding/start` | Initialize onboarding process for organization. |
| GET | `/api/onboarding/status` | Check onboarding completion status for current org. |
| POST | `/api/onboarding/permissions` | Save permissions and credentials during onboarding. |
| POST | `/api/onboarding/profile` | Save organization profile information. |
| POST | `/api/onboarding/network-scan` | Start network discovery scan. |
| GET | `/api/onboarding/scan-results` | Get discovered assets from network scan. |
| POST | `/api/onboarding/generate-deployment-plan` | Generate deployment plan for agents. |
| POST | `/api/onboarding/generate-agent-token` | Generate enrollment token for agent deployment. |
| GET | `/api/onboarding/enrolled-agents` | List enrolled agents. |
| POST | `/api/onboarding/validation` | Run validation checks for onboarding completion. |
| POST | `/api/onboarding/verify-agent-access/{enrollment_id}` | Verify agent access and connectivity. |
| POST | `/api/onboarding/verify-all-agents` | Verify all enrolled agents. |
| POST | `/api/onboarding/complete` | Mark onboarding as complete. |
| POST | `/api/onboarding/skip` | Skip onboarding process. |

## Ingestion

| Method | Path | Description |
| --- | --- | --- |
| POST | `/ingest/cowrie` | Cowrie honeypot event intake (expects HMAC headers). |
| POST | `/ingest/multi` | Multi-source ingestion wrapper accepting JSON payloads. |
| POST | `/ingest/synthetic` | Optional helper for test data generation. |

## Incidents (`/incidents` prefix without `/api`)

| Method | Path | Description |
| --- | --- | --- |
| GET | `/incidents` | List incidents (supports filters, pagination). Response includes risk scores, ML confidence, and escalation details. |
| GET | `/incidents/{id}` | Retrieve incident details including AI analysis and actions. |
| POST | `/incidents/{id}/contain` | Block offending IP (duration optional). |
| POST | `/incidents/{id}/actions/block-ip` | Granular action endpoints for SOC workflow buttons. |
| POST | `/incidents/{id}/actions/isolate-host` | Host isolation. |
| POST | `/incidents/{id}/actions/reset-passwords` | Example remediation action. |
| POST | `/incidents/{id}/schedule_unblock` | Schedule unblock via APScheduler job. |
| GET | `/incidents/{id}/block-status` | Current containment status. |
| GET | `/incidents/{id}/isolation-status` | Host isolation status. |

These routes require API key headers and (where noted) JWT bearer tokens.

## Response Workflows (`/api/response/*`)

| Method | Path | Description |
| --- | --- | --- |
| GET | `/api/response/actions` | List available response actions by category. |
| POST | `/api/response/actions/execute` | Execute a single action (used by Tactical Decision Center). |
| POST | `/api/response/workflows/create` | Persist a workflow playbook. |
| POST | `/api/response/workflows/execute` | Execute workflow by database ID. |
| GET | `/api/response/workflows` | List workflows with filters. |
| GET | `/api/response/workflows/{id}/status` | Execution progress. |
| GET | `/api/response/workflows/{id}/actions` | Action history for a workflow. |
| POST | `/api/response/workflows/{id}/cancel` | Cancel in-flight workflow. |
| GET | `/api/response/metrics/impact` | Aggregated response metrics. |

## Natural Language Processing (`/api/workflows/nlp/*`, `/api/nlp/*`)

| Method | Path | Description |
| --- | --- | --- |
| POST | `/api/workflows/nlp/parse` | Parse natural language into workflow structure. |
| POST | `/api/workflows/nlp/create` | Create and persist NLP-derived workflow. |
| GET | `/api/workflows/nlp/examples` | Get example NLP workflow templates. |
| GET | `/api/workflows/nlp/capabilities` | Get NLP system capabilities and supported operations. |
| POST | `/api/nlp/query` | Execute natural language query against incident data. |
| POST | `/api/nlp/threat-analysis` | Generate threat analysis using natural language processing. |
| POST | `/api/nlp/semantic-search` | Perform semantic search across incidents. |
| GET | `/api/nlp/status` | Get NLP system status and model information. |

## Agents & Orchestration (`/api/agents/*`, `/api/orchestrator/*`)

| Method | Path | Description |
| --- | --- | --- |
| POST | `/api/agents/orchestrate` | Coordinate specialized agents with natural language prompt. |
| GET | `/api/orchestrator/status` | Get orchestrator status and active workflows. |
| GET | `/api/orchestrator/workflows/{workflow_id}` | Get specific workflow execution status. |
| POST | `/api/orchestrator/workflows/{workflow_id}/cancel` | Cancel running workflow. |
| POST | `/api/orchestrator/workflows` | Create new orchestrated workflow. |

## Triggers & Automation (`/api/triggers/*`)

| Method | Path | Description |
| --- | --- | --- |
| GET | `/api/triggers` | List all triggers for current organization. |
| GET | `/api/triggers/{trigger_id}` | Get specific trigger details. |
| POST | `/api/triggers` | Create new workflow trigger. |
| PUT | `/api/triggers/{trigger_id}` | Update trigger configuration. |
| DELETE | `/api/triggers/{trigger_id}` | Delete trigger. |
| POST | `/api/triggers/{trigger_id}/enable` | Enable trigger. |
| POST | `/api/triggers/{trigger_id}/disable` | Disable trigger. |
| GET | `/api/triggers/stats/summary` | Get trigger execution statistics. |
| POST | `/api/triggers/bulk/pause` | Bulk pause triggers. |
| POST | `/api/triggers/bulk/resume` | Bulk resume triggers. |
| POST | `/api/triggers/bulk/archive` | Bulk archive triggers. |
| POST | `/api/triggers/{trigger_id}/simulate` | Simulate trigger execution. |
| PATCH | `/api/triggers/{trigger_id}/settings` | Update trigger settings. |

Triggers are evaluated via `backend/app/trigger_evaluator.py` and can auto-execute workflows when
conditions match.

## Webhooks (`/api/webhooks/*`)

| Method | Path | Description |
| --- | --- | --- |
| POST | `/api/webhooks/subscriptions` | Create webhook subscription for event notifications. |
| GET | `/api/webhooks/subscriptions` | List webhook subscriptions. |
| GET | `/api/webhooks/subscriptions/{webhook_id}` | Get specific webhook subscription. |
| PATCH | `/api/webhooks/subscriptions/{webhook_id}` | Update webhook subscription. |
| DELETE | `/api/webhooks/subscriptions/{webhook_id}` | Delete webhook subscription. |
| POST | `/api/webhooks/test` | Test webhook delivery. |
| GET | `/api/webhooks/events` | List available webhook events. |

## Machine Learning (`/api/ml/*`)

| Method | Path | Description |
| --- | --- | --- |
| POST | `/api/ml/retrain` | Trigger ML model retraining. |
| GET | `/api/ml/status` | Get ML system status. |
| GET | `/api/ml/online-learning/status` | Get online learning status. |
| POST | `/api/ml/online-learning/adapt` | Trigger online model adaptation. |
| GET | `/api/ml/ensemble/status` | Get ensemble model status. |
| POST | `/api/ml/ensemble/optimize` | Trigger ensemble optimization. |
| POST | `/api/ml/ab-test/create` | Create A/B test for model comparison. |
| GET | `/api/ml/ab-test/{test_id}/results` | Get A/B test results. |
| POST | `/api/ml/explain/{incident_id}` | Generate explanation for incident prediction. |
| GET | `/api/ml/models/performance` | Get model performance metrics. |
| GET | `/api/ml/drift/status` | Get concept drift detection status. |

## Federated Learning (`/api/federated/*`)

| Method | Path | Description |
| --- | --- | --- |
| GET | `/api/federated/status` | Get federated learning status. |
| POST | `/api/federated/coordinator/initialize` | Initialize federated learning coordinator. |
| POST | `/api/federated/participant/initialize` | Initialize federated learning participant. |
| POST | `/api/federated/training/start` | Start federated training session. |
| GET | `/api/federated/models/status` | Get federated model status. |
| POST | `/api/federated/models/train` | Train federated models. |
| GET | `/api/federated/insights` | Get federated learning insights. |

## Distributed Systems (`/api/distributed/*`)

| Method | Path | Description |
| --- | --- | --- |
| GET | `/api/distributed/status` | Get distributed system status. |
| GET | `/api/distributed/health` | Get distributed system health. |
| POST | `/api/distributed/initialize` | Initialize distributed system. |
| POST | `/api/distributed/shutdown` | Shutdown distributed system. |
| POST | `/api/distributed/broadcast` | Broadcast message to distributed nodes. |
| POST | `/api/distributed/execute-tool` | Execute tool on distributed nodes. |
| GET | `/api/distributed/nodes` | Get active distributed nodes. |
| GET | `/api/distributed/kafka/metrics` | Get Kafka metrics. |
| GET | `/api/distributed/redis/metrics` | Get Redis metrics. |
| POST | `/api/distributed/cache/set` | Set distributed cache value. |
| GET | `/api/distributed/cache/{key}` | Get distributed cache value. |
| POST | `/api/distributed/coordination/election` | Coordinate leader election. |

## Threat Intelligence (`/api/intelligence/*`)

| Method | Path | Description |
| --- | --- | --- |
| GET | `/api/intelligence/threats` | Get threat intelligence data. |
| GET | `/api/intelligence/distributed-threats` | Get distributed threat intelligence. |

## Incident Analytics (`/api/incidents/*`)

| Method | Path | Description |
| --- | --- | --- |
| GET | `/api/incidents/{inc_id}/context` | Get incident context for NLP processing. |
| GET | `/api/incidents/timeline` | Get incident timeline data. |
| GET | `/api/incidents/attack-paths` | Get attack path analysis. |
| POST | `/api/incidents/{incident_id}/ai-analysis` | Generate AI-powered incident analysis. |
| POST | `/api/incidents/{incident_id}/execute-ai-recommendation` | Execute AI-generated recommendations. |
| POST | `/api/incidents/{incident_id}/execute-ai-plan` | Execute AI-generated response plan. |
| GET | `/api/incidents/{incident_id}/threat-status` | Get comprehensive threat status. |

## System Management

| Method | Path | Description |
| --- | --- | --- |
| GET | `/api/sources` | List log sources. |
| GET | `/api/adaptive/status` | Get adaptive detection status. |
| POST | `/api/adaptive/force_learning` | Force adaptive learning cycle. |
| POST | `/api/adaptive/sensitivity` | Adjust detection sensitivity. |
| GET | `/settings/auto_contain` | Get auto-containment setting. |
| POST | `/settings/auto_contain` | Set auto-containment setting. |
| DELETE | `/admin/clear-database` | Clear database (admin only). |
| GET | `/incidents/real` | Get real incidents (filtered). |
| GET | `/honeypot/attacker-stats` | Get honeypot attacker statistics. |

## Telemetry

| Method | Path | Description |
| --- | --- | --- |
| GET | `/api/telemetry/status` | Per-organisation summary (assets, agents, incidents, last event). |
| GET | `/health` | Basic health check endpoint. |

## Authentication Headers

- `Authorization: Bearer <JWT>` for authenticated user routes.
- `x-api-key: <API_KEY>` for API key protected routes.
- HMAC headers (`X-Device-ID`, `X-TS`, `X-Nonce`, `X-Signature`) for secured ingest.

See `frontend/app/lib/api.ts` for examples of how the UI constructs requests.
