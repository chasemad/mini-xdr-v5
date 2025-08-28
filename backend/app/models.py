from sqlalchemy import Column, Integer, String, DateTime, JSON, Text, Index, Boolean, Float
from sqlalchemy.sql import func
from .db import Base


class Event(Base):
    __tablename__ = "events"
    
    id = Column(Integer, primary_key=True)
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
    
    __table_args__ = (Index("ix_events_src_ts", "src_ip", "ts"),)


class Incident(Base):
    __tablename__ = "incidents"
    
    id = Column(Integer, primary_key=True)
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


class LogSource(Base):
    __tablename__ = "log_sources"
    
    id = Column(Integer, primary_key=True)
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


class ContainmentPolicy(Base):
    __tablename__ = "containment_policies"
    
    id = Column(Integer, primary_key=True)
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
