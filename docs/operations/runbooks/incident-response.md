# Incident Response Runbook

This runbook standardises the process when Mini-XDR raises a high-severity incident.

## 1. Triage

- Confirm incident details in the dashboard (`frontend/app/incidents/incident/[id]/page.tsx`).
- Review AI-generated triage notes, threat analysis, and IOC extraction from multiple sources.
- Utilize AI analysis endpoint (`/api/incidents/{incident_id}/ai-analysis`) for comprehensive threat assessment.
- Examine ensemble ML scores, behavioral analysis, and cross-correlation results.
- Validate source attribution using the attribution agent and forensic analysis.
- Query incident context via `/api/incidents/{id}/context` for NLP processing and semantic analysis.

## 2. Containment

- If auto-containment did not trigger, execute `Contain Now` via the Tactical Decision Center or API
  (`/api/incidents/{id}/actions/block-ip` with configurable duration).
- Utilize the containment agent for automated network isolation and blocking operations.
- For host-level containment, execute `/api/incidents/{id}/actions/isolate-host` which coordinates
  with the EDR agent and checks policies in `backend/app/advanced_response_engine.py`.
- Deploy deception techniques using the deception agent (`/api/incidents/{id}/actions/honeypot-deploy-decoy`).
- Schedule automated rollback via `/api/incidents/{id}/schedule_unblock` with APScheduler-managed timing.
- Execute comprehensive containment workflows through `/api/response/workflows/execute` with approval workflows.

## 3. Investigation

- Execute forensic analysis using the forensics agent for evidence collection and timeline reconstruction.
- Run threat intelligence lookups via `/api/incidents/{id}/actions/threat-intel-lookup` and distributed
  threat intelligence (`/api/intelligence/threats`).
- Utilize the attribution agent to determine attack source and methodology.
- Perform threat hunting with the predictive hunter agent (`/api/incidents/{id}/actions/hunt-similar-attacks`).
- Generate NLP-based investigation playbooks (`/api/workflows/nlp/parse`) for systematic analysis.
- Execute AI-powered threat analysis (`/api/incidents/{incident_id}/execute-ai-recommendation`) for insights.
- Review ensemble ML scores, behavioral analysis, and distributed threat correlation results.

## 4. Remediation

- Execute automated remediation workflows via `/api/response/workflows/execute` with approval workflows.
- Utilize specialized agents for remediation:
  - IAM agent for credential resets and access control updates.
  - DLP agent for data protection and leakage prevention.
  - EDR agent for endpoint remediation and hardening.
- Deploy WAF rules and firewall updates (`/api/incidents/{id}/actions/deploy-waf-rules`).
- Monitor remediation effectiveness through `/api/response/metrics/impact` and the ResponseImpactMonitor component.
- Execute AI-generated remediation plans (`/api/incidents/{incident_id}/execute-ai-plan`).
- Document all remediation actions and manual steps in ticketing systems and `change-control/audit-log.md`.

## 5. Closure

- Confirm threat elimination through comprehensive validation using all available agents.
- Verify no residual alerts or suspicious activity through continuous monitoring.
- Set incident status to resolved via the UI or API (`/api/incidents/{id}/status`).
- Generate comprehensive incident reports using `/api/incidents/timeline` and `/api/incidents/attack-paths`.
- Export incident data and analysis for compliance and auditing purposes.
- Ensure all scheduled tasks (unblocks, retraining, monitoring) completed successfully via scheduler logs.

## 6. Post-Mortem & Learning

- Conduct thorough post-mortem analysis using AI insights and attack path reconstruction.
- Update detection rules and ML models based on lessons learned.
- Enhance agent capabilities and workflows for similar future incidents.
- Update `operations/monitoring-and-alerts.md` if monitoring or alerting gaps were identified.
- Create GitHub issues for product improvements and agent capability enhancements.
- Document new TTPs and update threat intelligence feeds.
