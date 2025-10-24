# SOC Analyst Guide

This runbook describes how analysts interact with Mini-XDR using the current frontend and API.

## Logging in

1. Visit the UI at `http://localhost:3000` (or your deployed domain).
2. Register or sign in via the `/api/auth` endpoints exposed by the backend.
3. The UI stores JWT access tokens in `localStorage`; if requests fail with 401, reauthenticate.

## Incident Workspace

- Navigate to **Incidents**. The list view pulls data from `/api/incidents` via
  `frontend/app/lib/api.ts::getIncidents`.
- Selecting an incident opens `frontend/app/incidents/incident/[id]/page.tsx`, which:
  - Shows high-level metrics (risk score, containment confidence, threat category).
  - Streams updates through `useIncidentRealtime` (WebSocket + polling fallback).
  - Surfaces IOC evidence and AI analysis panels (`frontend/components/EnhancedAIAnalysis.tsx` and
    `frontend/components/ThreatStatusBar.tsx`).

### Tactical Decision Center

- Quick actions (`Contain Now`, `Hunt Threats`, `Isolate Host`, etc.) map to comprehensive backend
  response endpoints with agent coordination. Actions include automated threat hunting, host isolation,
  WAF rule deployment, and deception techniques.
- "Ask AI" integrates with the full agent orchestration system (`/api/agents/orchestrate`) for
  natural language interaction with specialized security agents.
- AI Analysis provides comprehensive threat assessment using `/api/incidents/{incident_id}/ai-analysis`
  with actionable recommendations.

### Advanced Response Panel

- Build sophisticated multi-step workflows using the automation designer components
  (`frontend/app/components/PlaybookCanvas.tsx`, `frontend/app/components/WorkflowDesigner.tsx`).
- Leverage NLP workflow creation (`/api/workflows/nlp/parse`) for natural language to workflow conversion.
- Workflows support conditional branching, parallel execution, approval workflows, and comprehensive audit logging.
- Submitting a playbook calls `/api/response/workflows/create` with options for immediate execution or approval routing.

## Executing Responses

1. Use the **Advanced Response Panel** to create and execute comprehensive response workflows including
   agent coordination, approval routing, and impact monitoring.
2. Monitor execution status via `/api/response/workflows/{workflow_id}/status` and approval workflows
   via `/api/response/workflows/{workflow_id}/approve` - the UI renders progress in `ResponseImpactMonitor`.
3. Execute AI-generated recommendations (`/api/incidents/{incident_id}/execute-ai-recommendation`) and
   complete response plans (`/api/incidents/{incident_id}/execute-ai-plan`).
4. Manual actions from the Tactical Decision Center trigger agent orchestration; all actions are logged
   to the audit trail and `actions` table with comprehensive tracking.

## Post-Incident Steps

- Document comprehensive findings including AI analysis, agent actions, and threat intelligence in
  your ticketing system; Mini-XDR stores extensive triage notes, IOC data, and analysis results.
- Generate incident reports using `/api/incidents/timeline` and `/api/incidents/attack-paths` for
  compliance and auditing purposes.
- Schedule automated cleanup operations using `/api/incidents/{id}/schedule_unblock` with APScheduler
  for controlled rollback of containment measures.
- Update `change-control/audit-log.md` with all remediation actions, infrastructure changes, and
  lessons learned from the incident response.

## Troubleshooting UI Requests

- If API calls fail, check browser dev tools to confirm `x-api-key` header is present. Configure
  `NEXT_PUBLIC_API_KEY` in `frontend/.env.local`.
- For WebSocket issues, inspect backend logs for authentication errors originating from
  `AuthMiddleware`.
- Use `tests/test_hmac_auth.py` locally to validate HMAC signature generation when agents cannot
  connect.
