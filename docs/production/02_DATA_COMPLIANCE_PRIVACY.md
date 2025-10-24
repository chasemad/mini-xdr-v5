# 02: Data Compliance & Privacy - Production Implementation

**Status:** Critical Gap - Required for Enterprise Customers  
**Current State:** Basic encryption, no compliance framework  
**Target State:** SOC 2, GDPR, HIPAA-ready with full data governance  
**Priority:** P0 (Blocker for regulated industries)

---

## Current State Analysis

### What EXISTS Now

**File:** `/backend/app/db.py`
```python
✅ AsyncIO SQLAlchemy setup
✅ Basic database connection management
⚠️ SQLite default (NOT production-ready)
⚠️ No encryption at rest
⚠️ No connection pooling limits
```

**File:** `/backend/app/models.py`
```python
✅ Timestamp tracking (created_at, updated_at)
⚠️ No soft deletes (required for GDPR right to erasure)
⚠️ No data classification fields
⚠️ No retention policy metadata
```

**File:** `/backend/app/config.py`
```python
✅ Environment-based configuration
⚠️ No secrets encryption
⚠️ No data residency controls
```

### What's MISSING

- ❌ **SOC 2 Controls** - No documented security controls
- ❌ **GDPR Compliance** - No consent management, no data subject rights
- ❌ **HIPAA Compliance** - No BAA, no PHI handling
- ❌ **Data Retention Policies** - Indefinite storage
- ❌ **Data Classification** - No PII/PHI identification
- ❌ **Encryption at Rest** - Database not encrypted
- ❌ **Data Residency** - No geographic controls
- ❌ **Backup & Recovery** - No tested procedures
- ❌ **Data Anonymization** - No PII scrubbing capabilities
- ❌ **Consent Management** - No user consent tracking

---

## Implementation Checklist

### Task 1: SOC 2 Type I Readiness

#### 1.1: Create Security Policy Documents
**New Directory:** `/docs/compliance/soc2/`

**Files to Create:**
```
/docs/compliance/soc2/
├── information_security_policy.md
├── access_control_policy.md
├── change_management_policy.md
├── incident_response_policy.md
├── data_classification_policy.md
├── vendor_management_policy.md
├── acceptable_use_policy.md
└── business_continuity_plan.md
```

**Checklist:**
- [ ] Create compliance directory structure
- [ ] Draft Information Security Policy (review industry templates)
- [ ] Draft Access Control Policy (align with RBAC implementation)
- [ ] Draft Change Management Policy (require peer review, staging testing)
- [ ] Draft Incident Response Policy (contact procedures, escalation)
- [ ] Draft Data Classification Policy (define Public, Internal, Confidential, Restricted)
- [ ] Draft Vendor Management Policy (3rd party security assessments)
- [ ] Draft Acceptable Use Policy (employee/contractor usage)
- [ ] Draft Business Continuity Plan (RTO, RPO targets)
- [ ] Have legal review all policies
- [ ] Get executive sign-off on policies
- [ ] Distribute to all employees (track acknowledgment)

#### 1.2: Implement Technical Controls
**File:** `/backend/app/models.py` - Add compliance metadata to ALL models

```python
# Add these mixins to ALL database models

class ComplianceMixin:
    """Compliance and audit fields for all models"""
    
    # Soft delete (GDPR requirement)
    deleted_at = Column(DateTime(timezone=True), nullable=True, index=True)
    deleted_by_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    deletion_reason = Column(String(256), nullable=True)
    
    # Data classification
    data_classification = Column(
        String(32),
        default="internal",
        nullable=False
    )  # public|internal|confidential|restricted|pii|phi
    
    # Retention policy
    retention_policy_id = Column(Integer, ForeignKey("retention_policies.id"), nullable=True)
    retention_expires_at = Column(DateTime(timezone=True), nullable=True)
    legal_hold = Column(Boolean, default=False)  # Prevent deletion for litigation
    
    # Data residency
    data_region = Column(String(32), default="us-east-1")
    
    # Change tracking
    last_modified_by_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    version = Column(Integer, default=1)
```

**Apply to existing models:**
```python
# Update Event model (line 7)
class Event(Base, ComplianceMixin):
    __tablename__ = "events"
    # ... existing fields ...

# Update Incident model (line 29)
class Incident(Base, ComplianceMixin):
    __tablename__ = "incidents"
    # ... existing fields ...

# Update all other models similarly
```

**Checklist:**
- [ ] Create ComplianceMixin class
- [ ] Apply ComplianceMixin to all 20+ models
- [ ] Create migration: `alembic revision --autogenerate -m "add_compliance_fields"`
- [ ] Test migration in dev environment
- [ ] Update all queries to filter deleted_at IS NULL
- [ ] Implement soft delete helper function
- [ ] Create retention policy cleanup job

#### 1.3: Add Retention Policy Model
**File:** `/backend/app/models.py`

```python
class RetentionPolicy(Base):
    """Data retention policies for compliance"""
    __tablename__ = "retention_policies"
    
    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Policy identification
    name = Column(String(128), unique=True, nullable=False)
    description = Column(Text)
    
    # Retention rules
    retention_period_days = Column(Integer, nullable=False)  # 30, 90, 365, 2555 (7 years)
    applies_to_tables = Column(JSON, nullable=False)  # ["events", "incidents"]
    applies_to_classification = Column(JSON, nullable=True)  # ["pii", "phi"]
    
    # Action after retention period
    action = Column(String(32), nullable=False)  # delete|anonymize|archive
    anonymization_rules = Column(JSON, nullable=True)  # Which fields to redact
    archive_location = Column(String(256), nullable=True)  # S3 bucket for archive
    
    # Compliance justification
    compliance_requirement = Column(String(128))  # "GDPR Article 17", "HIPAA 164.316"
    approved_by = Column(String(256))
    approved_at = Column(DateTime(timezone=True))
    
    # Status
    is_active = Column(Boolean, default=True)


class RetentionExecution(Base):
    """Log of retention policy executions"""
    __tablename__ = "retention_executions"
    
    id = Column(Integer, primary_key=True)
    executed_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    
    retention_policy_id = Column(Integer, ForeignKey("retention_policies.id"), nullable=False)
    
    # Execution details
    records_processed = Column(Integer, default=0)
    records_deleted = Column(Integer, default=0)
    records_anonymized = Column(Integer, default=0)
    records_archived = Column(Integer, default=0)
    records_failed = Column(Integer, default=0)
    
    # Result
    status = Column(String(32), nullable=False)  # success|partial|failed
    error_message = Column(Text, nullable=True)
    execution_time_seconds = Column(Float)
    
    # Audit trail
    execution_log = Column(JSON, nullable=True)
```

**Checklist:**
- [ ] Add RetentionPolicy and RetentionExecution models
- [ ] Create migration
- [ ] Create default retention policies (Events: 90 days, Incidents: 7 years, Audit Logs: 10 years)
- [ ] Implement retention policy execution job
- [ ] Schedule retention job to run daily at 2 AM
- [ ] Test retention execution with test data
- [ ] Add retention policy management UI
- [ ] Document retention policies in compliance docs

#### 1.4: Implement Encryption at Rest
**File:** `/backend/app/crypto_fields.py` (NEW)

```python
"""Encrypted database fields for sensitive data"""
from sqlalchemy.types import TypeDecorator, String, Text
from cryptography.fernet import Fernet
from .config import settings
import base64


# Initialize Fernet encryption
# In production, get this from AWS KMS or HashiCorp Vault
ENCRYPTION_KEY = base64.urlsafe_b64decode(settings.database_encryption_key)
cipher_suite = Fernet(ENCRYPTION_KEY)


class EncryptedString(TypeDecorator):
    """Transparently encrypt/decrypt string fields"""
    impl = Text
    cache_ok = True
    
    def process_bind_param(self, value, dialect):
        """Encrypt before saving to database"""
        if value is not None:
            encrypted = cipher_suite.encrypt(value.encode())
            return base64.urlsafe_b64encode(encrypted).decode()
        return value
    
    def process_result_value(self, value, dialect):
        """Decrypt when reading from database"""
        if value is not None:
            decoded = base64.urlsafe_b64decode(value.encode())
            decrypted = cipher_suite.decrypt(decoded)
            return decrypted.decode()
        return value


# Usage in models:
# from .crypto_fields import EncryptedString
#
# class User(Base):
#     password_hash = Column(EncryptedString(512))  # Encrypted in DB
#     ssn = Column(EncryptedString(32), nullable=True)  # PII encrypted
```

**File:** `/backend/app/config.py` - Add encryption config

```python
# Add to Settings class
class Settings(BaseSettings):
    # ... existing fields ...
    
    # Encryption configuration
    database_encryption_key: str = None  # Base64-encoded 32-byte key
    field_level_encryption_enabled: bool = True
    
    # AWS KMS configuration (for production)
    kms_key_id: Optional[str] = None
    kms_region: str = "us-east-1"
```

**Checklist:**
- [ ] Create crypto_fields.py
- [ ] Generate encryption key: `python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"`
- [ ] Add DATABASE_ENCRYPTION_KEY to .env and secrets manager
- [ ] Apply EncryptedString to sensitive fields (password_hash, MFA secrets, SSO tokens)
- [ ] Create migration to re-encrypt existing data
- [ ] Test encryption/decryption
- [ ] Document key rotation procedure
- [ ] Implement AWS KMS integration for production

#### 1.5: PostgreSQL Full Disk Encryption
**File:** `/ops/k8s/postgres-encrypted.yaml` (NEW)

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: postgres-encrypted-pvc
  namespace: mini-xdr
spec:
  accessModes:
    - ReadWriteOnce
  storageClassName: encrypted-gp3  # AWS EBS encrypted storage class
  resources:
    requests:
      storage: 100Gi
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: postgres-config
  namespace: mini-xdr
data:
  postgresql.conf: |
    # Enable SSL
    ssl = on
    ssl_cert_file = '/var/lib/postgresql/certs/server.crt'
    ssl_key_file = '/var/lib/postgresql/certs/server.key'
    ssl_ca_file = '/var/lib/postgresql/certs/ca.crt'
    
    # Enforce encrypted connections only
    ssl_min_protocol_version = 'TLSv1.2'
    
    # Connection limits
    max_connections = 200
    
    # Logging for audit
    log_connections = on
    log_disconnections = on
    log_statement = 'all'  # Log all statements for compliance
    log_line_prefix = '%t [%p]: [%l-1] user=%u,db=%d,app=%a,client=%h '
```

**Checklist:**
- [ ] Create encrypted storage class in K8s/cloud provider
- [ ] Enable PostgreSQL SSL/TLS
- [ ] Generate SSL certificates for PostgreSQL
- [ ] Update database connection string to require SSL
- [ ] Enable PostgreSQL audit logging
- [ ] Configure log retention (10 years for audit logs)
- [ ] Test encrypted backups
- [ ] Document disaster recovery procedures

---

### Task 2: GDPR Compliance

#### 2.1: Add Consent Management
**File:** `/backend/app/models.py`

```python
class DataProcessingConsent(Base):
    """Track user consent for GDPR compliance"""
    __tablename__ = "data_processing_consents"
    
    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    organization_id = Column(Integer, ForeignKey("organizations.id"), nullable=False)
    
    # Consent details
    purpose = Column(String(128), nullable=False)  # analytics|marketing|essential
    consent_given = Column(Boolean, nullable=False)
    consent_method = Column(String(64))  # web_form|email|api
    
    # Legal basis (GDPR Article 6)
    legal_basis = Column(String(64), nullable=False)  # consent|contract|legal_obligation|legitimate_interest
    
    # Tracking
    ip_address = Column(String(64))
    user_agent = Column(String(512))
    consent_text_version = Column(String(32))  # Track policy version consented to
    
    # Revocation
    revoked_at = Column(DateTime(timezone=True), nullable=True)
    revoked_method = Column(String(64), nullable=True)


class DataSubjectRequest(Base):
    """GDPR Data Subject Access Requests (DSAR)"""
    __tablename__ = "data_subject_requests"
    
    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    
    # Requester
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    email = Column(String(256), nullable=False, index=True)
    organization_id = Column(Integer, ForeignKey("organizations.id"), nullable=False)
    
    # Request type (GDPR Article 15-22)
    request_type = Column(String(64), nullable=False)
    # access|rectification|erasure|restrict_processing|data_portability|object
    
    # Request details
    request_details = Column(JSON)
    verification_method = Column(String(64))  # email|phone|id_document
    verified_at = Column(DateTime(timezone=True), nullable=True)
    
    # Processing
    status = Column(String(32), default="pending")  # pending|verified|processing|completed|rejected
    assigned_to_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    
    # Response
    response_data = Column(JSON, nullable=True)  # For access requests
    response_file_path = Column(String(512), nullable=True)  # Exported data location
    completion_notes = Column(Text, nullable=True)
    
    # Timeline tracking (GDPR requires response within 30 days)
    due_date = Column(DateTime(timezone=True), nullable=False)
    reminded_at = Column(DateTime(timezone=True), nullable=True)
```

**Checklist:**
- [ ] Add consent and DSAR models
- [ ] Create migration
- [ ] Implement consent tracking on user registration
- [ ] Create API endpoints for consent management
- [ ] Implement DSAR submission endpoint
- [ ] Implement data export (all user data in JSON/CSV)
- [ ] Implement data deletion (cascade delete all user data)
- [ ] Create DSAR processing UI for admins
- [ ] Set up 30-day deadline alerts
- [ ] Test complete DSAR workflow

#### 2.2: Implement Data Portability
**New File:** `/backend/app/gdpr_export.py`

```python
"""GDPR Data Portability - Export all user data"""
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from .models import User, Event, Incident, Action, AuditLog


async def export_user_data(
    user_id: int,
    db: AsyncSession,
    output_dir: Path
) -> Path:
    """
    Export all data for a user in machine-readable format (GDPR Article 20)
    Returns path to ZIP file containing JSON exports
    """
    
    export_data = {
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "user_id": user_id,
        "format_version": "1.0"
    }
    
    # Export user profile
    user_result = await db.execute(select(User).where(User.id == user_id))
    user = user_result.scalars().first()
    export_data["user_profile"] = {
        "email": user.email,
        "first_name": user.first_name,
        "last_name": user.last_name,
        "created_at": user.created_at.isoformat(),
        "timezone": user.timezone
    }
    
    # Export incidents assigned to user
    incidents_result = await db.execute(
        select(Incident).where(Incident.assigned_to_id == user_id)
    )
    export_data["incidents"] = [
        {
            "id": inc.id,
            "created_at": inc.created_at.isoformat(),
            "src_ip": inc.src_ip,
            "reason": inc.reason,
            "status": inc.status
        }
        for inc in incidents_result.scalars().all()
    ]
    
    # Export audit logs for user actions
    audit_result = await db.execute(
        select(AuditLog).where(AuditLog.user_id == user_id).limit(10000)
    )
    export_data["audit_log"] = [
        {
            "timestamp": log.created_at.isoformat(),
            "action": log.action_type,
            "resource": f"{log.resource_type}:{log.resource_id}" if log.resource_type else None,
            "status": log.status
        }
        for log in audit_result.scalars().all()
    ]
    
    # Create ZIP file
    export_filename = f"user_data_export_{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
    zip_path = output_dir / export_filename
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Add JSON export
        zipf.writestr("user_data.json", json.dumps(export_data, indent=2))
        
        # Add README
        readme = """
GDPR Data Export
================
This file contains all your personal data stored in Mini-XDR.

Contents:
- user_data.json: Your profile, incidents, and activity log

Format: JSON (machine-readable as per GDPR Article 20)
Exported: {date}
        """.format(date=datetime.now().isoformat())
        zipf.writestr("README.txt", readme)
    
    return zip_path
```

**Checklist:**
- [ ] Create gdpr_export.py
- [ ] Implement export_user_data function
- [ ] Add all user-related tables to export
- [ ] Create /api/gdpr/export endpoint
- [ ] Test export with sample user data
- [ ] Verify export is machine-readable JSON
- [ ] Add export to DSAR workflow
- [ ] Store exports securely (encrypted S3)

#### 2.3: Implement Right to Erasure
**New File:** `/backend/app/gdpr_erasure.py`

```python
"""GDPR Right to Erasure - Delete or anonymize user data"""
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import update, delete
from datetime import datetime, timezone

from .models import User, UserSession, UserRole, AuditLog, Incident


async def erase_user_data(
    user_id: int,
    db: AsyncSession,
    anonymize: bool = True
) -> dict:
    """
    Delete or anonymize all user data (GDPR Article 17)
    
    If anonymize=True: Replace PII with anonymous identifiers (preserves analytics)
    If anonymize=False: Hard delete all user data (not recommended)
    """
    
    result = {
        "user_id": user_id,
        "method": "anonymize" if anonymize else "delete",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "records_affected": {}
    }
    
    if anonymize:
        # Anonymize user profile
        anonymous_email = f"deleted_user_{user_id}@anonymized.local"
        await db.execute(
            update(User)
            .where(User.id == user_id)
            .values(
                email=anonymous_email,
                first_name="Deleted",
                last_name="User",
                password_hash=None,
                mfa_secret=None,
                sso_user_id=None,
                avatar_url=None,
                deleted_at=datetime.now(timezone.utc),
                status="deactivated"
            )
        )
        result["records_affected"]["users"] = 1
        
        # Keep audit logs but anonymize actor
        audit_update = await db.execute(
            update(AuditLog)
            .where(AuditLog.user_id == user_id)
            .values(actor_identifier=anonymous_email)
        )
        result["records_affected"]["audit_logs"] = audit_update.rowcount
        
        # Keep incident assignments but mark as system
        incident_update = await db.execute(
            update(Incident)
            .where(Incident.assigned_to_id == user_id)
            .values(assigned_to_id=None, assigned_note="User deleted - GDPR erasure")
        )
        result["records_affected"]["incidents"] = incident_update.rowcount
        
    else:
        # Hard delete (cascade to related tables)
        # Delete sessions
        sessions_deleted = await db.execute(
            delete(UserSession).where(UserSession.user_id == user_id)
        )
        result["records_affected"]["sessions"] = sessions_deleted.rowcount
        
        # Delete roles
        roles_deleted = await db.execute(
            delete(UserRole).where(UserRole.user_id == user_id)
        )
        result["records_affected"]["roles"] = roles_deleted.rowcount
        
        # Delete user (audit logs kept for compliance)
        user_deleted = await db.execute(
            delete(User).where(User.id == user_id)
        )
        result["records_affected"]["users"] = user_deleted.rowcount
    
    await db.commit()
    return result
```

**Checklist:**
- [ ] Create gdpr_erasure.py
- [ ] Implement anonymization (default)
- [ ] Implement hard deletion (for specific requests)
- [ ] Add /api/gdpr/erase endpoint
- [ ] Require two-factor confirmation for erasure
- [ ] Test erasure with test users
- [ ] Verify audit logs are preserved
- [ ] Document what data is kept vs deleted

#### 2.4: Create Privacy Policy & Terms
**New Directory:** `/docs/legal/`

**Files to Create:**
```
/docs/legal/
├── privacy_policy.md
├── terms_of_service.md
├── data_processing_agreement.md
├── cookie_policy.md
└── acceptable_use_policy.md
```

**Privacy Policy Must Include:**
- What data is collected
- Why data is collected (legal basis)
- How long data is retained
- Who data is shared with
- User rights (access, rectification, erasure, portability)
- How to exercise rights
- Contact information for DPO (Data Protection Officer)
- Cookie usage

**Checklist:**
- [ ] Draft privacy policy (use GDPR template)
- [ ] Draft terms of service
- [ ] Draft Data Processing Agreement (DPA) for enterprise customers
- [ ] Draft cookie policy
- [ ] Have lawyer review all legal documents
- [ ] Publish on website with version tracking
- [ ] Implement "I agree" checkboxes during signup
- [ ] Track which policy version user agreed to

---

### Task 3: HIPAA Readiness

#### 3.1: Add PHI Identification
**File:** `/backend/app/phi_detector.py` (NEW)

```python
"""Protected Health Information (PHI) Detection"""
import re
from typing import Dict, List, Any


# Common PHI patterns
PHI_PATTERNS = {
    "ssn": re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
    "mrn": re.compile(r'\bMRN[:\s]?\d{6,}\b', re.IGNORECASE),
    "dob": re.compile(r'\b\d{1,2}/\d{1,2}/\d{4}\b'),
    "phone": re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'),
    "email": re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
    "ip_address": re.compile(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'),
}


def detect_phi(text: str) -> List[str]:
    """Detect PHI in free text"""
    found_phi = []
    
    for phi_type, pattern in PHI_PATTERNS.items():
        if pattern.search(text):
            found_phi.append(phi_type)
    
    return found_phi


def redact_phi(text: str) -> str:
    """Redact PHI from text"""
    redacted = text
    
    for phi_type, pattern in PHI_PATTERNS.items():
        redacted = pattern.sub(f"[REDACTED-{phi_type.upper()}]", redacted)
    
    return redacted


def classify_data_for_hipaa(data: Dict[str, Any]) -> str:
    """
    Classify data as PHI, PII, or general
    Returns: phi|pii|general
    """
    # Check for known PHI fields
    phi_fields = {"ssn", "medical_record_number", "diagnosis", "treatment", "dob"}
    
    for key in data.keys():
        if any(phi_field in key.lower() for phi_field in phi_fields):
            return "phi"
    
    # Check content
    for value in data.values():
        if isinstance(value, str) and detect_phi(value):
            return "phi"
    
    # Check for PII
    pii_fields = {"email", "phone", "address", "name"}
    for key in data.keys():
        if any(pii_field in key.lower() for pii_field in pii_fields):
            return "pii"
    
    return "general"
```

**Checklist:**
- [ ] Create phi_detector.py
- [ ] Implement PHI pattern detection
- [ ] Implement PHI redaction
- [ ] Add PHI classification to data ingestion pipeline
- [ ] Auto-classify data on insertion
- [ ] Test with sample healthcare data
- [ ] Add PHI detection to search/export

#### 3.2: Create BAA Template
**File:** `/docs/legal/business_associate_agreement.md`

**Content must include:**
- Permitted uses of PHI
- Safeguards to prevent unauthorized use
- Breach notification requirements (60 days)
- Subcontractor requirements
- Return or destruction of PHI on termination
- Audit rights

**Checklist:**
- [ ] Draft BAA using HIPAA template
- [ ] Have healthcare attorney review
- [ ] Create BAA signing workflow
- [ ] Track executed BAAs per customer
- [ ] Implement breach notification system

#### 3.3: HIPAA Technical Safeguards
**Checklist:**
- [ ] Implement access controls (done in 01_AUTHENTICATION)
- [ ] Implement audit logging (done in 01_AUTHENTICATION)
- [ ] Implement encryption at rest (Task 1.4)
- [ ] Implement encryption in transit (TLS 1.3 everywhere)
- [ ] Automatic logoff after 15 minutes of inactivity
- [ ] Unique user identification (done - no shared accounts)
- [ ] Emergency access procedure
- [ ] Data backup and disaster recovery (Task 4)

---

### Task 4: Backup & Disaster Recovery

#### 4.1: Automated Backup Strategy
**New File:** `/scripts/backup/postgres_backup.sh`

```bash
#!/bin/bash
# Automated PostgreSQL backup with encryption

set -euo pipefail

# Configuration
BACKUP_DIR="/var/backups/postgres"
S3_BUCKET="s3://mini-xdr-backups-${AWS_ACCOUNT_ID}"
RETENTION_DAYS=90
ENCRYPTION_KEY_ID="alias/mini-xdr-backup-key"  # AWS KMS key

# Create backup
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="mini-xdr_backup_${TIMESTAMP}.sql"

echo "Creating backup: ${BACKUP_FILE}"
pg_dump -h $DB_HOST -U $DB_USER -d $DB_NAME | gzip > "${BACKUP_DIR}/${BACKUP_FILE}.gz"

# Calculate checksum
sha256sum "${BACKUP_DIR}/${BACKUP_FILE}.gz" > "${BACKUP_DIR}/${BACKUP_FILE}.sha256"

# Upload to S3 with encryption
echo "Uploading to S3"
aws s3 cp "${BACKUP_DIR}/${BACKUP_FILE}.gz" \
    "${S3_BUCKET}/${BACKUP_FILE}.gz" \
    --sse aws:kms \
    --sse-kms-key-id "${ENCRYPTION_KEY_ID}"

aws s3 cp "${BACKUP_DIR}/${BACKUP_FILE}.sha256" \
    "${S3_BUCKET}/${BACKUP_FILE}.sha256"

# Delete old backups
echo "Cleaning up old backups"
aws s3 ls "${S3_BUCKET}/" | while read -r line; do
    BACKUP_DATE=$(echo $line | awk '{print $1}')
    BACKUP_AGE=$(( ( $(date +%s) - $(date -d "$BACKUP_DATE" +%s) ) / 86400 ))
    
    if [ $BACKUP_AGE -gt $RETENTION_DAYS ]; then
        BACKUP_FILE=$(echo $line | awk '{print $4}')
        echo "Deleting old backup: ${BACKUP_FILE}"
        aws s3 rm "${S3_BUCKET}/${BACKUP_FILE}"
    fi
done

# Cleanup local backup
rm -f "${BACKUP_DIR}/${BACKUP_FILE}.gz"
rm -f "${BACKUP_DIR}/${BACKUP_FILE}.sha256"

echo "Backup completed successfully"
```

**Checklist:**
- [ ] Create backup script
- [ ] Set up AWS S3 bucket with versioning enabled
- [ ] Create AWS KMS key for backup encryption
- [ ] Test backup creation
- [ ] Test backup restoration
- [ ] Schedule daily backups (cron or K8s CronJob)
- [ ] Set up backup monitoring/alerting
- [ ] Document restore procedures
- [ ] Test restore in staging environment monthly

#### 4.2: Point-in-Time Recovery
**Checklist:**
- [ ] Enable PostgreSQL Write-Ahead Logging (WAL)
- [ ] Configure continuous WAL archiving to S3
- [ ] Test point-in-time recovery (PITR)
- [ ] Document PITR procedures
- [ ] Define Recovery Point Objective (RPO: <1 hour)
- [ ] Define Recovery Time Objective (RTO: <4 hours)

#### 4.3: Multi-Region Replication
**For Phase 2/3 - Enterprise deployments**

**Checklist:**
- [ ] Set up PostgreSQL read replica in different region
- [ ] Configure automatic failover
- [ ] Test failover procedures
- [ ] Document runbook for failover
- [ ] Implement cross-region backup replication

---

## Compliance Certification Roadmap

### SOC 2 Type I (Months 1-4)
**Cost:** $15K - $30K for auditor  
**Timeline:** 3-4 months

**Checklist:**
- [ ] Select SOC 2 auditor (Big 4 or specialized firm)
- [ ] Complete readiness assessment
- [ ] Implement all required controls
- [ ] Document all policies and procedures
- [ ] Conduct internal audit
- [ ] Remediate findings
- [ ] Schedule formal audit
- [ ] Receive SOC 2 Type I report
- [ ] Share report with customers under NDA

### SOC 2 Type II (Months 7-12)
**Cost:** $25K - $50K  
**Timeline:** 6+ months (must operate controls for 6-12 months)

**Checklist:**
- [ ] Operate SOC 2 Type I controls for 6+ months
- [ ] Collect evidence of control operation
- [ ] Schedule Type II audit
- [ ] Auditor performs testing
- [ ] Remediate any findings
- [ ] Receive SOC 2 Type II report
- [ ] Market certification to enterprise customers

### ISO 27001 (Months 13-24)
**Cost:** $30K - $80K  
**Timeline:** 9-12 months

**Checklist:**
- [ ] Gap analysis against ISO 27001 controls
- [ ] Implement missing controls (114 total controls)
- [ ] Create Information Security Management System (ISMS)
- [ ] Internal audit
- [ ] Select certification body
- [ ] Stage 1 audit (documentation review)
- [ ] Stage 2 audit (implementation testing)
- [ ] Receive ISO 27001 certification
- [ ] Annual surveillance audits

### HIPAA Compliance
**Note:** No official certification, but attestation required

**Checklist:**
- [ ] Complete HIPAA Security Risk Assessment
- [ ] Implement required safeguards
- [ ] Execute BAAs with all covered entities
- [ ] Train all employees on HIPAA
- [ ] Conduct annual HIPAA audit
- [ ] Maintain compliance documentation
- [ ] Provide HIPAA compliance attestation to customers

---

## Testing Checklist

### Compliance Tests
- [ ] Test data retention policy execution
- [ ] Test soft delete functionality
- [ ] Test GDPR data export for completeness
- [ ] Test GDPR data erasure (anonymization)
- [ ] Test backup creation and encryption
- [ ] Test backup restoration (full and point-in-time)
- [ ] Test PHI detection and classification
- [ ] Test access controls on sensitive data
- [ ] Verify all connections use TLS 1.2+
- [ ] Verify encryption at rest is enabled

### Audit Tests
- [ ] Verify all data access is logged
- [ ] Verify audit logs are immutable
- [ ] Test audit log export for compliance reporting
- [ ] Verify retention of audit logs (10 years)

---

## Production Deployment Checklist

### Pre-Launch
- [ ] All compliance policies approved and published
- [ ] SOC 2 Type I audit scheduled (can be in progress)
- [ ] Privacy policy and terms published on website
- [ ] DPA template ready for enterprise customers
- [ ] Data encryption enabled (at rest and in transit)
- [ ] Automated backups configured and tested
- [ ] Disaster recovery plan documented and tested
- [ ] GDPR rights (export, erasure) fully functional
- [ ] Data retention policies configured
- [ ] PHI detection enabled (if applicable)
- [ ] All employees trained on data protection

### Post-Launch
- [ ] Schedule quarterly compliance reviews
- [ ] Annual penetration testing
- [ ] Annual disaster recovery test
- [ ] Continuous monitoring of compliance controls
- [ ] Customer communication for policy updates

---

## Estimated Effort

**Total Effort:** 10-12 weeks, 1-2 engineers + compliance specialist

| Task | Effort | Priority |
|------|--------|----------|
| SOC 2 readiness | 3 weeks | P0 |
| GDPR implementation | 2 weeks | P0 |
| Encryption (at rest & in transit) | 1 week | P0 |
| Backup & disaster recovery | 1.5 weeks | P0 |
| HIPAA readiness | 2 weeks | P1 (if needed) |
| Legal document drafting | 2 weeks | P0 |
| Testing & documentation | 1.5 weeks | P0 |
| SOC 2 Type I audit | 12 weeks (parallel) | P0 |

**External Costs:**
- SOC 2 auditor: $15K-$30K
- Legal review: $5K-$15K
- Penetration testing: $10K-$25K
- Compliance consultant: $10K-$30K

---

**Status:** Ready for implementation  
**Next Document:** `03_ENTERPRISE_INTEGRATIONS.md`


