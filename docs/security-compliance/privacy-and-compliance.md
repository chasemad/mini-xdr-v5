# Privacy & Compliance

Mini-XDR stores security telemetry, incidents, and workflow metadata. Use this summary to ensure
deployments meet legal and organisational requirements.

## Data Collected

- **Security Telemetry**: Event logs ingested via `/ingest/multi` including IP addresses, user agents,
  command strings, network flows, and system events.
- **Incident Data**: Comprehensive incident records with IOCs, AI analysis results, response actions,
  and audit trails (`backend/app/models.py`).
- **User & Organization Data**: User accounts, organization metadata, authentication tokens, and
  role assignments with multi-tenant isolation (`backend/app/auth.py`).
- **Workflow & Automation**: Workflow definitions, execution logs, approval chains, and impact metrics
  (`backend/app/advanced_response_engine.py`, `backend/app/playbook_engine.py`).
- **Agent Operations**: Agent enrollment data, orchestration logs, and specialized agent outputs
  (forensics, attribution, deception, etc.).
- **ML Training Data**: Event patterns and behavioral data collected for model improvement
  (`backend/app/training_data_collector.py`).
- **Audit Logs**: Comprehensive audit trails for all security events, user actions, and system changes.

## Storage Locations

- **Database**: PostgreSQL via `DATABASE_URL` with encryption at rest and automated backups for
  production. SQLite is no longer the default.
- **ML Artefacts**: Stored in `models/` directory; production deployments can publish to encrypted
  object storage with versioning and access controls.
- **Distributed Storage**: Kafka for event streaming, Redis for caching and session management in
  distributed deployments.
- **Secrets**: Environment variables or your preferred secret store; AWS-specific integrations are archived.
- **Audit Logs**: Tamper-evident storage with retention policies and compliance exports.

## Retention & Deletion

- **Data Retention**: Configurable retention policies for incidents, events, and audit logs (default: 1 year).
- **Automated Cleanup**: Scheduled jobs for data purging and archival (`process_scheduled_unblocks` includes cleanup).
- **Data Deletion**: Cascading delete operations respect foreign key constraints and maintain referential integrity.
- **Right to Deletion**: User data deletion capabilities with audit trails maintained for compliance.
- **Archival**: Automated data archival to long-term storage for compliance requirements.

## Compliance Considerations

- **Multi-tenant Isolation**: Strict data separation between organizations with tenant middleware
  and database-level isolation.
- **Access Controls**: Role-based access control (RBAC) with policy engine for fine-grained permissions.
- **Audit Logging**: Comprehensive audit trails for all privileged actions, data access, and system changes.
- **Data Residency**: Regional deployments with data sovereignty controls and cross-border transfer restrictions.
- **PII Handling**: Input sanitization and optional data masking/anonymization for sensitive information.
- **Encryption**: End-to-end encryption (TLS in transit, encryption at rest for all storage layers).
- **Federated Learning Privacy**: Privacy-preserving ML training that doesn't expose raw data between tenants.
- **Incident Response**: Documented procedures with evidence collection and chain of custody maintenance.
- **Regulatory Compliance**: Support for GDPR, CCPA, SOX, and other compliance frameworks with automated reporting.

## Privacy by Design

- **Data Minimization**: Only collect necessary security telemetry with configurable data collection levels.
- **Purpose Limitation**: Data used solely for security monitoring and incident response.
- **Consent Management**: Clear consent mechanisms for data collection in onboarding flows.
- **Data Portability**: Export capabilities for user data and incident reports.
- **Breach Notification**: Automated alerting and notification systems for security incidents.

Document any compliance assessments, certifications, or audit results under `change-control/audit-log.md`.
