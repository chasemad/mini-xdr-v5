# Mini-XDR Documentation Hub

Welcome to the canonical documentation set for the Mini-XDR platform. This directory tracks the **current, running implementation** of the system across the FastAPI backend, Next.js 15 frontend, machine-learning services, and supporting infrastructure. Every document is kept in lock-step with the source tree; update the relevant file whenever behaviour in code changes.

## Navigation

- **Overview**
  - [Project Overview](overview/project-overview.md)
  - [Roadmap & Status](overview/roadmap-and-status.md)
- **Architecture**
  - [System Overview](architecture/system-overview.md)
  - [Data Flows & Integrations](architecture/data-flows.md)
  - [Component Deep Dives](architecture/component-deep-dives/README.md)
- **Getting Started**
  - [Local Quickstart](getting-started/local-quickstart.md)
  - [Environment Configuration](getting-started/environment-config.md)
  - [Documentation Enforcement](getting-started/docs-enforcement.md)
- **Deployment**
  - [Deployment Overview](deployment/overview.md)
  - [Kubernetes & Infrastructure](deployment/kubernetes-and-infra.md)
  - Archived AWS Guides: `docs/archived/aws/`
- **Operations**
  - Runbooks: [SOC Analyst Guide](operations/runbooks/soc-analyst-guide.md) · [Incident Response](operations/runbooks/incident-response.md)
  - [Monitoring & Alerts](operations/monitoring-and-alerts.md)
- **Security & Compliance**
  - [Security Hardening](security-compliance/security-hardening.md)
  - [Authentication & Access Control](security-compliance/auth-and-access.md)
  - [Privacy & Compliance](security-compliance/privacy-and-compliance.md)
- **Machine Learning**
  - [Training Guide](ml/training-guide.md)
  - [Model Operations Runbook](ml/model-ops-runbook.md)
  - [Data Sources](ml/data-sources.md)
- **API & Integrations**
  - [API Reference](api/reference.md)
  - [Workflow & Automation Integrations](api/workflows-and-integrations.md)
- **User Interface**
  - [Dashboard Guide](ui/dashboard-guide.md)
  - [Automation Designer](ui/automation-designer.md)
- **Demo & Presentations**
  - [Demo Package Index](../DEMO-PACKAGE-INDEX.md) - Complete demo materials
  - [Demo Walkthrough](../scripts/demo/DEMO-WALKTHROUGH.md) - Step-by-step guide
  - [Quick Reference](../scripts/demo/QUICK-REFERENCE.txt) - 1-page cheat sheet
- **Change Control**
  - [Release Notes](change-control/release-notes.md)
  - [Audit Log](change-control/audit-log.md)

## How to Contribute

1. Keep statements factual—mirror the behaviour in the FastAPI routes, React components, scripts, or infrastructure manifests.
2. Document production defaults **and** local overrides (e.g. SQLite vs Postgres) so readers can reproduce environments.
3. When code or configuration changes, update the corresponding document in the same pull request.
4. Link directly to files or modules (for example `backend/app/main.py`) so engineers can trace context quickly.
5. Use ASCII text and wrap at ~100 characters for readability.

For any material removed from this directory, migrate reusable snippets into the new structure and delete the legacy file instead of leaving it stale.
