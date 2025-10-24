from sqlalchemy import Column, Integer, String, DateTime, JSON, Text, Index, Boolean, Float, ForeignKey
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from .db import Base


# Multi-Tenant Models
class Organization(Base):
    """Organization/Tenant model for multi-tenancy"""
    __tablename__ = "organizations"
    
    id = Column(Integer, primary_key=True)
    name = Column(String(255), unique=True, nullable=False, index=True)
    slug = Column(String(100), unique=True, nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    status = Column(String(20), default="active", index=True)  # active|suspended|trial
    settings = Column(JSON, nullable=True)  # Custom org settings
    max_users = Column(Integer, default=10)  # User limit for organization
    max_log_sources = Column(Integer, default=50)  # Log source limit
    
    # Onboarding state tracking
    onboarding_status = Column(String(20), default="not_started", index=True)  # not_started|in_progress|completed
    onboarding_step = Column(String(50), nullable=True)  # profile|network_scan|agents|integrations|validation
    onboarding_data = Column(JSON, nullable=True)  # Wizard state (network ranges, scan results, tokens)
    onboarding_completed_at = Column(DateTime(timezone=True), nullable=True)
    first_login_completed = Column(Boolean, default=False)


class User(Base):
    """User model with organization association"""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True)
    organization_id = Column(Integer, ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255))
    role = Column(String(50), default="analyst", index=True)  # admin|analyst|viewer|soc_lead
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    last_login = Column(DateTime(timezone=True), nullable=True)
    is_active = Column(Boolean, default=True, index=True)
    failed_login_attempts = Column(Integer, default=0)
    locked_until = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    organization = relationship("Organization", backref="users")


class Event(Base):
    __tablename__ = "events"
    
    id = Column(Integer, primary_key=True)
    organization_id = Column(Integer, ForeignKey("organizations.id", ondelete="CASCADE"), nullable=True, index=True)  # Nullable for migration
    ts = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    src_ip = Column(String(64), index=True)
    dst_ip = Column(String(64), nullable=True)
    dst_port = Column(Integer, nullable=True)
    eventid = Column(String(128), index=True)
    message = Column(Text, nullable=True)
    raw = Column(JSON)
    
    # Enhanced fields for multi-source ingestion
    source_type = Column(String(32), default="cowrie", index=True)  # cowrie, suricata, osquery, etc.
    hostname = Column(String(128), nullable=True, index=True)  # Source hostname
    signature = Column(String(256), nullable=True)  # For integrity validation
    agent_timestamp = Column(Float, nullable=True)  # Agent collection timestamp
    anomaly_score = Column(Float, nullable=True)  # ML-calculated anomaly score
    
    # Relationships
    organization = relationship("Organization")
    
    __table_args__ = (Index("ix_events_src_ts", "src_ip", "ts"), Index("ix_events_org_ts", "organization_id", "ts"))


class Incident(Base):
    __tablename__ = "incidents"
    
    id = Column(Integer, primary_key=True)
    organization_id = Column(Integer, ForeignKey("organizations.id", ondelete="CASCADE"), nullable=True, index=True)  # Nullable for migration
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    src_ip = Column(String(64), index=True)
    reason = Column(String(256))
    status = Column(String(32), default="open")  # open|contained|dismissed
    auto_contained = Column(Boolean, default=False)
    triage_note = Column(JSON, nullable=True)  # {summary, severity, recommendation, rationale}
    
    # Enhanced fields for AI Agent integration
    escalation_level = Column(String(16), default="medium")  # low|medium|high|critical
    risk_score = Column(Float, default=0.0)  # Calculated risk score (0.0-1.0)
    threat_category = Column(String(64), nullable=True)  # brute_force|password_spray|credential_stuffing|etc.
    containment_confidence = Column(Float, default=0.0)  # Agent confidence in containment decision
    containment_method = Column(String(32), nullable=True)  # rule_based|ml_driven|ai_agent
    
    # Agent orchestration fields
    agent_id = Column(String(64), nullable=True)  # ID of orchestrating agent
    agent_actions = Column(JSON, nullable=True)  # Log of agent-executed actions
    policy_id = Column(String(64), nullable=True)  # Reference to applied containment policy
    rollback_status = Column(String(16), default="none")  # none|pending|executed|failed
    agent_confidence = Column(Float, default=0.0)  # Agent's self-assessed decision confidence
    soar_integrations = Column(JSON, nullable=True)  # External SOAR integrations
    
    # ML correlation fields
    correlation_id = Column(String(64), nullable=True)  # Links related incidents
    ml_features = Column(JSON, nullable=True)  # Extracted features for ML analysis
    ensemble_scores = Column(JSON, nullable=True)  # Individual model scores
    
    # AI Analysis Caching
    ai_analysis = Column(JSON, nullable=True)  # Cached AI analysis results
    ai_analysis_timestamp = Column(DateTime(timezone=True), nullable=True)  # When analysis was done
    last_event_count = Column(Integer, default=0)  # Track new events to trigger re-analysis
    
    # Relationships
    organization = relationship("Organization")
    action_logs = relationship("ActionLog", back_populates="incident", cascade="all, delete-orphan")


class Action(Base):
    __tablename__ = "actions"
    
    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    incident_id = Column(Integer, index=True)
    action = Column(String(32))   # block|unblock|scheduled_unblock|isolate|notify
    result = Column(String(32))   # pending|success|failed|done
    detail = Column(Text, nullable=True)
    params = Column(JSON, nullable=True)
    due_at = Column(DateTime(timezone=True), nullable=True)
    
    # Enhanced fields for agent actions
    agent_id = Column(String(64), nullable=True)  # Agent that executed this action
    confidence_score = Column(Float, nullable=True)  # Agent confidence in action
    rollback_action_id = Column(Integer, nullable=True)  # Link to rollback action if any
    
    # T-Pot Verification
    verified_on_tpot = Column(Boolean, default=False)  # Verification status
    tpot_verification_timestamp = Column(DateTime(timezone=True), nullable=True)
    tpot_verification_details = Column(JSON, nullable=True)  # Verification results


class ActionLog(Base):
    """Complete audit trail of agent actions"""
    __tablename__ = "action_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    organization_id = Column(Integer, ForeignKey("organizations.id", ondelete="CASCADE"), nullable=True, index=True)  # Nullable for migration
    action_id = Column(String, unique=True, index=True, nullable=False)
    agent_id = Column(String, index=True, nullable=False)
    agent_type = Column(String, index=True)  # iam, edr, dlp, containment
    action_name = Column(String, index=True, nullable=False)
    incident_id = Column(Integer, ForeignKey("incidents.id"), nullable=True, index=True)
    
    params = Column(JSON, nullable=False)
    result = Column(JSON, nullable=True)
    status = Column(String, nullable=False)  # success, failed, rolled_back
    error = Column(Text, nullable=True)
    
    rollback_id = Column(String, unique=True, index=True, nullable=True)
    rollback_data = Column(JSON, nullable=True)
    rollback_executed = Column(Boolean, default=False)
    rollback_timestamp = Column(DateTime(timezone=True), nullable=True)
    rollback_result = Column(JSON, nullable=True)
    
    executed_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    organization = relationship("Organization")
    incident = relationship("Incident", back_populates="action_logs")


class LogSource(Base):
    __tablename__ = "log_sources"
    
    id = Column(Integer, primary_key=True)
    organization_id = Column(Integer, ForeignKey("organizations.id", ondelete="CASCADE"), nullable=True, index=True)  # Nullable for migration
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Basic source configuration
    source_type = Column(String(32), index=True)  # cowrie, suricata, osquery, syslog, etc.
    hostname = Column(String(128), index=True)
    endpoint_url = Column(String(256), nullable=True)
    status = Column(String(16), default="active")  # active|inactive|error
    last_event_ts = Column(DateTime(timezone=True), nullable=True)
    
    # Agent configuration
    agent_endpoint = Column(String(256), nullable=True)  # Agent's push URL
    validation_key = Column(String(128), nullable=True)  # For signed log validation
    agent_version = Column(String(16), default="v1.0")  # Track agent compatibility
    ingestion_rate_limit = Column(Integer, default=1000)  # Events per minute
    
    # Statistics
    events_processed = Column(Integer, default=0)
    events_failed = Column(Integer, default=0)
    config = Column(JSON, nullable=True)  # Source-specific configuration
    
    # Relationships
    organization = relationship("Organization")


class ThreatIntelSource(Base):
    __tablename__ = "threat_intel_sources"
    
    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    name = Column(String(64), unique=True)  # virustotal, abuseipdb, etc.
    endpoint_url = Column(String(256))
    api_key = Column(String(256), nullable=True)
    status = Column(String(16), default="active")
    rate_limit_per_hour = Column(Integer, default=1000)
    config = Column(JSON, nullable=True)


class MLModel(Base):
    __tablename__ = "ml_models"
    
    id = Column(Integer, primary_key=True)
    organization_id = Column(Integer, ForeignKey("organizations.id", ondelete="CASCADE"), nullable=True, index=True)  # Nullable for shared/global models
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    name = Column(String(64), unique=True)  # isolation_forest, lstm_autoencoder, xgboost_classifier
    model_type = Column(String(32))  # anomaly_detection, classification, ensemble
    version = Column(String(16), default="v1.0")
    status = Column(String(16), default="training")  # training|active|deprecated
    
    # Training metadata
    training_data_size = Column(Integer, nullable=True)
    training_accuracy = Column(Float, nullable=True)
    last_trained_at = Column(DateTime(timezone=True), nullable=True)
    
    # Model storage
    model_path = Column(String(256), nullable=True)  # Path to saved model file
    hyperparameters = Column(JSON, nullable=True)
    metrics = Column(JSON, nullable=True)  # Accuracy, precision, recall, etc.
    
    # Federated learning
    is_federated = Column(Boolean, default=False)
    federated_round = Column(Integer, default=0)
    
    # Relationships
    organization = relationship("Organization")


class ContainmentPolicy(Base):
    __tablename__ = "containment_policies"
    
    id = Column(Integer, primary_key=True)
    organization_id = Column(Integer, ForeignKey("organizations.id", ondelete="CASCADE"), nullable=True, index=True)  # Nullable for migration
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    name = Column(String(64), unique=True)
    description = Column(Text, nullable=True)
    priority = Column(Integer, default=100)  # Lower number = higher priority
    status = Column(String(16), default="active")  # active|inactive|draft
    
    # Policy conditions (YAML stored as JSON)
    conditions = Column(JSON)  # Conditions that trigger this policy
    actions = Column(JSON)     # Actions to take when conditions are met
    
    # Agent settings
    agent_override = Column(Boolean, default=True)  # Allow agent to override this policy
    escalation_threshold = Column(Float, default=0.8)  # When to escalate to human
    
    # Usage statistics
    times_triggered = Column(Integer, default=0)
    success_rate = Column(Float, default=0.0)
    
    # Relationships
    organization = relationship("Organization")


class AgentCredential(Base):
    __tablename__ = "agent_credentials"

    id = Column(Integer, primary_key=True)
    device_id = Column(String(64), unique=True, index=True)
    public_id = Column(String(64), unique=True, index=True)
    secret_hash = Column(String(128))
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    expires_at = Column(DateTime(timezone=True), nullable=True)
    revoked_at = Column(DateTime(timezone=True), nullable=True)
    description = Column(String(256), nullable=True)


class RequestNonce(Base):
    __tablename__ = "request_nonces"

    id = Column(Integer, primary_key=True)
    device_id = Column(String(64), index=True)
    nonce = Column(String(128), index=True)
    endpoint = Column(String(256), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)

    __table_args__ = (
        Index("ix_request_nonce_device_nonce", "device_id", "nonce", unique=True),
    )


# =============================================================================
# ENHANCED RESPONSE & WORKFLOW MODELS (Phase 1)
# =============================================================================

class ResponseWorkflow(Base):
    """Enhanced workflow model for advanced response orchestration"""
    __tablename__ = "response_workflows"
    
    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    completed_at = Column(DateTime(timezone=True), nullable=True)
    
    # Workflow identification
    workflow_id = Column(String(64), unique=True, index=True)  # Unique workflow identifier
    incident_id = Column(Integer, ForeignKey("incidents.id"), index=True)
    playbook_name = Column(String(128), index=True)
    playbook_version = Column(String(16), default="v1.0")
    
    # Workflow state
    status = Column(String(32), default="pending", index=True)  # pending|running|completed|failed|cancelled
    current_step = Column(Integer, default=0)
    total_steps = Column(Integer, default=0)
    progress_percentage = Column(Float, default=0.0)
    
    # Workflow definition and execution
    steps = Column(JSON)  # Array of workflow steps with configuration
    execution_log = Column(JSON, nullable=True)  # Detailed execution log
    current_step_data = Column(JSON, nullable=True)  # Current step execution data
    
    # AI and automation
    ai_confidence = Column(Float, default=0.0)  # AI confidence in workflow selection
    auto_executed = Column(Boolean, default=False)  # Whether workflow was auto-executed
    approval_required = Column(Boolean, default=True)  # Whether human approval is required
    approved_by = Column(String(64), nullable=True)  # User who approved execution
    approved_at = Column(DateTime(timezone=True), nullable=True)
    
    # Safety and rollback
    auto_rollback_enabled = Column(Boolean, default=True)
    rollback_plan = Column(JSON, nullable=True)  # Rollback steps if needed
    rollback_executed = Column(Boolean, default=False)
    rollback_reason = Column(String(256), nullable=True)
    
    # Performance metrics
    execution_time_ms = Column(Integer, nullable=True)
    success_rate = Column(Float, default=0.0)
    impact_score = Column(Float, default=0.0)  # Measured impact of the workflow
    
    # Relationships
    incident = relationship("Incident", backref="response_workflows")
    impact_metrics = relationship("ResponseImpactMetrics", backref="workflow", cascade="all, delete-orphan")


class ResponseImpactMetrics(Base):
    """Real-time impact metrics for response actions"""
    __tablename__ = "response_impact_metrics"
    
    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    
    # Associated workflow
    workflow_id = Column(Integer, ForeignKey("response_workflows.id"), index=True)
    step_number = Column(Integer, default=0)  # Which step this metric is for
    
    # Impact measurements
    attacks_blocked = Column(Integer, default=0)
    false_positives = Column(Integer, default=0)
    systems_affected = Column(Integer, default=0)
    users_affected = Column(Integer, default=0)
    
    # Performance metrics
    response_time_ms = Column(Integer, default=0)
    success_rate = Column(Float, default=0.0)
    confidence_score = Column(Float, default=0.0)
    
    # Business impact
    downtime_minutes = Column(Integer, default=0)
    cost_impact_usd = Column(Float, default=0.0)
    compliance_impact = Column(String(32), default="none")  # none|low|medium|high|critical
    
    # Detailed metrics
    metrics_data = Column(JSON, nullable=True)  # Flexible metrics storage
    external_metrics = Column(JSON, nullable=True)  # Metrics from external systems


class AdvancedResponseAction(Base):
    """Enhanced action model for advanced response capabilities"""
    __tablename__ = "advanced_response_actions"
    
    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    completed_at = Column(DateTime(timezone=True), nullable=True)
    
    # Action identification
    action_id = Column(String(64), unique=True, index=True)
    workflow_id = Column(Integer, ForeignKey("response_workflows.id"), nullable=True, index=True)
    incident_id = Column(Integer, ForeignKey("incidents.id"), index=True)
    parent_action_id = Column(Integer, ForeignKey("advanced_response_actions.id"), nullable=True)
    
    # Action definition
    action_type = Column(String(64), index=True)  # block_ip, isolate_host, deploy_firewall, etc.
    action_category = Column(String(32), index=True)  # network, endpoint, email, cloud, etc.
    action_name = Column(String(128))
    action_description = Column(Text, nullable=True)
    
    # Execution details
    status = Column(String(32), default="pending", index=True)  # pending|running|completed|failed|cancelled|rolled_back
    priority = Column(Integer, default=100)  # Lower number = higher priority
    timeout_seconds = Column(Integer, default=300)  # Action timeout
    retry_count = Column(Integer, default=0)
    max_retries = Column(Integer, default=3)
    
    # Action parameters and results
    parameters = Column(JSON)  # Action-specific parameters
    result_data = Column(JSON, nullable=True)  # Action execution results
    error_details = Column(JSON, nullable=True)  # Error information if failed
    
    # Safety and validation
    safety_checks = Column(JSON, nullable=True)  # Pre-execution safety validations
    impact_assessment = Column(JSON, nullable=True)  # Predicted impact assessment
    approval_required = Column(Boolean, default=False)
    approved_by = Column(String(64), nullable=True)
    approved_at = Column(DateTime(timezone=True), nullable=True)
    
    # Rollback capabilities
    rollback_action_id = Column(Integer, ForeignKey("advanced_response_actions.id"), nullable=True)
    rollback_data = Column(JSON, nullable=True)  # Data needed for rollback
    rollback_executed = Column(Boolean, default=False)
    
    # Agent and automation
    executed_by = Column(String(64), nullable=True)  # Agent or user who executed
    execution_method = Column(String(32), default="manual")  # manual|automated|ai_driven
    confidence_score = Column(Float, default=0.0)
    
    # Relationships
    workflow = relationship("ResponseWorkflow", backref="actions")
    incident = relationship("Incident", backref="advanced_actions")
    child_actions = relationship("AdvancedResponseAction", 
                                foreign_keys=[parent_action_id],
                                backref="parent", 
                                remote_side=[id])


class ResponsePlaybook(Base):
    """Playbook templates for response workflows"""
    __tablename__ = "response_playbooks"
    
    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Playbook identification
    name = Column(String(128), unique=True, index=True)
    version = Column(String(16), default="v1.0")
    description = Column(Text, nullable=True)
    category = Column(String(64), index=True)  # malware, ddos, insider_threat, etc.
    
    # Playbook definition
    steps = Column(JSON)  # Playbook step definitions
    conditions = Column(JSON, nullable=True)  # When this playbook should be used
    estimated_duration_minutes = Column(Integer, default=30)
    
    # Usage and effectiveness
    status = Column(String(16), default="active", index=True)  # active|draft|deprecated
    times_used = Column(Integer, default=0)
    success_rate = Column(Float, default=0.0)
    average_execution_time = Column(Integer, default=0)  # In minutes
    
    # Metadata
    created_by = Column(String(64), nullable=True)
    tags = Column(JSON, nullable=True)  # Searchable tags
    compliance_frameworks = Column(JSON, nullable=True)  # SOC2, GDPR, etc.


class ResponseApproval(Base):
    """Approval workflow for high-impact response actions"""
    __tablename__ = "response_approvals"
    
    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Approval request
    workflow_id = Column(Integer, ForeignKey("response_workflows.id"), nullable=True, index=True)
    action_id = Column(Integer, ForeignKey("advanced_response_actions.id"), nullable=True, index=True)
    requested_by = Column(String(64), index=True)
    
    # Approval details
    approval_type = Column(String(32), index=True)  # workflow|action|emergency
    impact_level = Column(String(16), index=True)  # low|medium|high|critical
    justification = Column(Text)
    
    # Approval status
    status = Column(String(16), default="pending", index=True)  # pending|approved|denied|expired
    approved_by = Column(String(64), nullable=True)
    approved_at = Column(DateTime(timezone=True), nullable=True)
    denial_reason = Column(Text, nullable=True)
    expires_at = Column(DateTime(timezone=True), nullable=True)
    
    # Emergency overrides
    emergency_override = Column(Boolean, default=False)
    override_reason = Column(Text, nullable=True)
    override_by = Column(String(64), nullable=True)
    
    # Relationships
    workflow = relationship("ResponseWorkflow", backref="approvals")
    action = relationship("AdvancedResponseAction", backref="approvals")


# =============================================================================
# WEBHOOK SYSTEM MODELS (Phase 2)
# =============================================================================

class WebhookSubscription(Base):
    """Webhook subscription model for event notifications"""
    __tablename__ = "webhook_subscriptions"

    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Webhook configuration
    url = Column(String(512), nullable=False)
    event_types = Column(JSON, nullable=False)  # List of event types to subscribe to
    name = Column(String(128), nullable=True)
    description = Column(Text, nullable=True)

    # Security
    signing_secret = Column(String(256), nullable=True)  # Custom HMAC secret

    # Status and statistics
    is_active = Column(Boolean, default=True, index=True)
    last_triggered_at = Column(DateTime(timezone=True), nullable=True)
    delivery_success_count = Column(Integer, default=0)
    delivery_failure_count = Column(Integer, default=0)

    # Metadata
    config = Column(JSON, nullable=True)  # Additional configuration options


class WebhookDeliveryLog(Base):
    """Log of webhook delivery attempts"""
    __tablename__ = "webhook_delivery_logs"

    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)

    webhook_subscription_id = Column(Integer, ForeignKey('webhook_subscriptions.id'), index=True)
    event_type = Column(String(64), index=True)
    payload = Column(JSON)

    # Delivery result
    status_code = Column(Integer, nullable=True)
    response_time_ms = Column(Float, nullable=True)
    success = Column(Boolean, default=False, index=True)
    error = Column(Text, nullable=True)
    attempt_number = Column(Integer, default=1)

    # Relationship
    subscription = relationship("WebhookSubscription", backref="delivery_logs")


class WorkflowTrigger(Base):
    """Automatic workflow triggers for threat detection"""
    __tablename__ = "workflow_triggers"

    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Trigger identification
    name = Column(String(128), unique=True, index=True)
    description = Column(Text, nullable=True)
    category = Column(String(64), index=True)  # honeypot, network, endpoint, etc.

    # Trigger state
    enabled = Column(Boolean, default=True, index=True)
    auto_execute = Column(Boolean, default=False)  # Execute without approval
    priority = Column(String(16), default="medium")  # low|medium|high|critical
    status = Column(String(16), default="active", index=True)  # active|paused|archived|error

    # Trigger conditions (JSON format)
    # Example: {"event_type": "cowrie.login.failed", "threshold": 6, "window_seconds": 60}
    conditions = Column(JSON, nullable=False)

    # Response workflow definition
    playbook_name = Column(String(128), index=True)  # Name of playbook to execute
    workflow_steps = Column(JSON, nullable=False)  # Steps to execute

    # NLP metadata (for NLP-generated triggers)
    source = Column(String(16), default="manual", index=True)  # nlp|manual|template|api
    source_prompt = Column(Text, nullable=True)  # Original NLP prompt
    parser_confidence = Column(Float, nullable=True)  # NLP parser confidence score
    parser_version = Column(String(16), nullable=True)  # Parser version used
    request_type = Column(String(32), nullable=True, index=True)  # response|investigation|automation|reporting|qa
    fallback_used = Column(Boolean, default=False)  # Whether fallback template was used

    # Trigger metadata
    created_by = Column(String(64), nullable=True)
    last_editor = Column(String(64), nullable=True)
    owner = Column(String(64), nullable=True)
    last_triggered_at = Column(DateTime(timezone=True), nullable=True)
    trigger_count = Column(Integer, default=0)  # How many times triggered
    success_count = Column(Integer, default=0)  # Successful executions
    failure_count = Column(Integer, default=0)  # Failed executions
    last_run_status = Column(String(16), nullable=True)  # success|failed|skipped

    # Performance metrics
    avg_response_time_ms = Column(Float, default=0.0)
    success_rate = Column(Float, default=0.0)

    # Additional configuration
    cooldown_seconds = Column(Integer, default=60)  # Min time between triggers
    max_triggers_per_day = Column(Integer, default=100)  # Rate limiting
    tags = Column(JSON, nullable=True)  # For categorization and filtering

    # Agent dependencies
    agent_requirements = Column(JSON, nullable=True)  # Required agents/tools

    # Versioning
    version = Column(Integer, default=1)  # Current version number


class WorkflowTriggerVersion(Base):
    """Version history for workflow triggers"""
    __tablename__ = "workflow_trigger_versions"

    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)

    # Link to trigger
    trigger_id = Column(Integer, ForeignKey("workflow_triggers.id"), index=True)
    version_number = Column(Integer)

    # Snapshot of trigger state at this version
    name = Column(String(128))
    description = Column(Text, nullable=True)
    enabled = Column(Boolean)
    auto_execute = Column(Boolean)
    priority = Column(String(16))
    conditions = Column(JSON)
    workflow_steps = Column(JSON)

    # Change metadata
    changed_by = Column(String(64))
    change_reason = Column(Text, nullable=True)
    changes_summary = Column(JSON, nullable=True)  # Diff summary

    # Relationship
    trigger = relationship("WorkflowTrigger", backref="versions")


class NLPWorkflowSuggestion(Base):
    """Queue of NLP-parsed workflows awaiting review/approval"""
    __tablename__ = "nlp_workflow_suggestions"

    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Original NLP input
    prompt = Column(Text, nullable=False)
    incident_id = Column(Integer, ForeignKey("incidents.id"), nullable=True, index=True)

    # Parsed workflow data
    request_type = Column(String(32), index=True)  # response|investigation|automation|reporting|qa
    priority = Column(String(16))
    confidence = Column(Float)
    fallback_used = Column(Boolean, default=False)

    # Parsed actions and workflow
    workflow_steps = Column(JSON)
    detected_actions = Column(JSON, nullable=True)  # List of detected action types
    missing_actions = Column(JSON, nullable=True)  # Actions that couldn't be parsed

    # Suggestion status
    status = Column(String(16), default="pending", index=True)  # pending|approved|dismissed|converted
    reviewed_by = Column(String(64), nullable=True)
    reviewed_at = Column(DateTime(timezone=True), nullable=True)

    # If approved, link to created trigger
    trigger_id = Column(Integer, ForeignKey("workflow_triggers.id"), nullable=True)

    # Parser diagnostics
    parser_version = Column(String(16))
    parser_diagnostics = Column(JSON, nullable=True)  # Detailed parser output

    # Relationships
    incident = relationship("Incident", backref="nlp_suggestions")
    trigger = relationship("WorkflowTrigger", backref="nlp_suggestion")


# =============================================================================
# ONBOARDING & ASSET DISCOVERY MODELS
# =============================================================================

class DiscoveredAsset(Base):
    """Network assets discovered during onboarding scan"""
    __tablename__ = "discovered_assets"
    
    id = Column(Integer, primary_key=True)
    organization_id = Column(Integer, ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    
    # Asset identification
    ip = Column(String(64), nullable=False, index=True)
    hostname = Column(String(255), nullable=True)
    mac_address = Column(String(17), nullable=True)
    
    # Discovery metadata
    os_type = Column(String(64), nullable=True)  # Windows, Linux/Unix, unknown
    os_role = Column(String(128), nullable=True)  # Domain Controller, Web Server, etc.
    classification = Column(String(64), nullable=True)  # From AssetClassifier
    classification_confidence = Column(Float, default=0.0)
    
    # Network information
    open_ports = Column(JSON, nullable=True)  # List of open port numbers
    services = Column(JSON, nullable=True)  # Service details per port
    
    # Deployment information
    deployment_profile = Column(JSON, nullable=True)  # Agent deployment recommendations
    agent_compatible = Column(Boolean, default=True)
    deployment_priority = Column(String(16), default="medium")  # critical|high|medium|low
    
    # Discovery details
    discovered_at = Column(DateTime(timezone=True), server_default=func.now())
    last_seen = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    scan_id = Column(String(64), nullable=True, index=True)  # Link to scan session
    
    # Relationships
    organization = relationship("Organization")
    
    __table_args__ = (Index("ix_discovered_assets_org_ip", "organization_id", "ip"),)


class AgentEnrollment(Base):
    """Agent enrollment tokens and registration tracking"""
    __tablename__ = "agent_enrollments"
    
    id = Column(Integer, primary_key=True)
    organization_id = Column(Integer, ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Token and identification
    agent_token = Column(String(128), unique=True, nullable=False, index=True)  # Unique enrollment token
    agent_id = Column(String(64), unique=True, nullable=True, index=True)  # Agent's self-reported ID after registration
    
    # Agent information
    hostname = Column(String(255), nullable=True)
    platform = Column(String(64), nullable=True)  # windows|linux|macos|docker
    ip_address = Column(String(64), nullable=True)
    
    # Status tracking
    status = Column(String(20), default="pending", index=True)  # pending|active|inactive|revoked
    first_checkin = Column(DateTime(timezone=True), nullable=True)
    last_heartbeat = Column(DateTime(timezone=True), nullable=True)
    
    # Agent metadata
    agent_metadata = Column(JSON, nullable=True)  # OS version, agent version, etc.
    enrollment_source = Column(String(64), nullable=True)  # onboarding_wizard|manual|api
    revoked_at = Column(DateTime(timezone=True), nullable=True)
    revoked_reason = Column(String(255), nullable=True)
    
    # Link to discovered asset if applicable
    discovered_asset_id = Column(Integer, ForeignKey("discovered_assets.id"), nullable=True)
    
    # Relationships
    organization = relationship("Organization")
    discovered_asset = relationship("DiscoveredAsset")
    
    __table_args__ = (Index("ix_agent_enrollments_org_status", "organization_id", "status"),)
