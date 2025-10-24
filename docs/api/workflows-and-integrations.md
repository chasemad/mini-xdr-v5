# Workflow & Automation Integrations

## Workflow Designer

- Frontend components (`frontend/app/components/WorkflowDesigner.tsx`,
  `frontend/app/components/PlaybookCanvas.tsx`) build workflow payloads matching the schema consumed
  by `/api/response/workflows/create` and `/api/orchestrator/workflows`.
- Each step includes `action_type`, `parameters`, retry metadata, approval requirements, and execution
  dependencies. Backend validation occurs in `backend/app/advanced_response_engine.py` and
  `backend/app/playbook_engine.py`.
- Workflows support conditional branching, parallel execution, approval workflows, and comprehensive
  audit logging.

## NLP Workflows

- `/api/workflows/nlp/parse` and `/api/workflows/nlp/create` endpoints handle natural language to
  structured workflow conversion via `backend/app/nlp_workflow_parser.py`.
- `/api/nlp/*` endpoints provide advanced NLP capabilities including threat analysis, semantic search,
  and natural language querying via `backend/app/nlp_suggestion_routes.py`.
- Frontend entry point lives in `frontend/app/components/NaturalLanguageInput.tsx` for natural
  language workflow input with real-time validation and suggestions.

## Trigger Engine

- Triggers are evaluated via `backend/app/trigger_evaluator.py` and managed by comprehensive routes in
  `backend/app/trigger_routes.py` (aliased under `/api/triggers`).
- Support for event-based, time-based, and conditional triggers with bulk operations and simulation capabilities.
- Triggers can auto-execute workflows, send notifications, or invoke agent orchestration when
  specific incident conditions occur.

## Agent Integrations

- `backend/app/agent_orchestrator.py` coordinates 12 specialized agents in `backend/app/agents/`:
  - **Containment Agent**: Network isolation and blocking operations
  - **Forensics Agent**: Evidence collection and analysis
  - **Deception Agent**: Honeypot management and decoy deployment
  - **Predictive Hunter**: Threat hunting and pattern analysis
  - **Attribution Agent**: Attack attribution and origin analysis
  - **DLP Agent**: Data loss prevention and monitoring
  - **EDR Agent**: Endpoint detection and response
  - **IAM Agent**: Identity and access management
  - **Ingestion Agent**: Advanced log parsing and normalization
  - **NLP Analyzer**: Natural language processing for logs
  - **Coordination Hub**: Multi-agent workflow orchestration
  - **HMAC Signer**: Cryptographic signing and verification
- External systems can call `/api/agents/orchestrate` with context to request responses or analysis.
- Agent enrollment and verification managed through `backend/app/agent_enrollment_service.py` and
  `backend/app/agent_verification_service.py`.
- HMAC credentials for agents are required when invoking high-trust operations.

## Response Workflows

- Advanced response workflows managed through `/api/response/workflows/*` endpoints with support for:
  - Workflow creation, execution, and monitoring
  - Approval workflows and role-based access control
  - Impact metrics and effectiveness tracking
  - Parallel and conditional execution paths
- Integration with policy engine (`backend/app/policy_engine.py`) for automated compliance checking.
- Learning response engine (`backend/app/learning_response_engine.py`) improves workflow effectiveness
  over time.

## Webhook Integrations

- Webhook system (`backend/app/webhook_manager.py`, `backend/app/webhook_routes.py`) enables real-time
  notifications to external systems for incident events, workflow completions, and system alerts.
- Support for custom webhook subscriptions with filtering and authentication.

## External Services

- **Threat Intelligence**: `backend/app/external_intel.py` integrates AbuseIPDB, VirusTotal, and
  distributed threat feeds when API keys are available.
- **Honeypot (T-Pot)**: `backend/app/tpot_verifier.py` verifies honeypot state via SSH/HTTP with
  advanced deception capabilities.
- **SageMaker**: Cloud ML integration through `backend/app/sagemaker_client.py` and
  `backend/app/sagemaker_endpoint_manager.py` with auto-scaling and model versioning.
- **Distributed Systems**: Kafka and Redis integration for multi-node deployments with leader election
  and distributed caching.

## Federated Learning

- Privacy-preserving model training across distributed tenants via `backend/app/federated_learning.py`.
- Coordinator and participant roles with secure aggregation and model updates.

Update this document when new automation connectors or integrations are introduced.
