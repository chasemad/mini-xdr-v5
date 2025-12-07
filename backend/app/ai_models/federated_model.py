"""
Federated Learning Model Schemas and Data Structures
Extended models specifically for federated learning operations
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from ..db import Base


class FederatedRoleEnum(str, Enum):
    """Federated learning roles"""

    COORDINATOR = "coordinator"
    PARTICIPANT = "participant"
    OBSERVER = "observer"


class FederatedStatusEnum(str, Enum):
    """Federated learning process status"""

    IDLE = "idle"
    PREPARING = "preparing"
    TRAINING = "training"
    AGGREGATING = "aggregating"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"


class FederatedModelTypeEnum(str, Enum):
    """Types of federated models"""

    ISOLATION_FOREST = "isolation_forest"
    LSTM_AUTOENCODER = "lstm_autoencoder"
    NEURAL_NETWORK = "neural_network"
    ENSEMBLE = "ensemble"
    TRANSFORMER = "transformer"


class FederatedNode(Base):
    """Represents a node in the federated learning network"""

    __tablename__ = "federated_nodes"

    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # Node identification
    node_id = Column(String(64), unique=True, index=True)  # UUID of the node
    node_name = Column(String(128), nullable=True)  # Human-readable name
    role = Column(
        String(16), default=FederatedRoleEnum.PARTICIPANT
    )  # coordinator|participant|observer
    status = Column(String(16), default="active")  # active|inactive|disconnected|error

    # Network information
    endpoint_url = Column(String(256), nullable=True)  # Node's API endpoint
    public_key = Column(Text, nullable=True)  # RSA public key for encryption
    last_heartbeat = Column(DateTime(timezone=True), nullable=True)

    # Capabilities and configuration
    supported_models = Column(JSON, nullable=True)  # List of supported model types
    max_data_size = Column(Integer, default=10000)  # Maximum training data size
    compute_capacity = Column(
        Float, default=1.0
    )  # Relative compute capacity (0.1 - 10.0)
    network_bandwidth = Column(Float, nullable=True)  # Network bandwidth in Mbps

    # Statistics
    total_rounds_participated = Column(Integer, default=0)
    successful_rounds = Column(Integer, default=0)
    failed_rounds = Column(Integer, default=0)
    average_training_time = Column(
        Float, nullable=True
    )  # Average training time in seconds

    # Security and trust
    trust_score = Column(Float, default=0.5)  # Trust score (0.0 - 1.0)
    security_level = Column(String(16), default="standard")  # basic|standard|high
    last_security_audit = Column(DateTime(timezone=True), nullable=True)

    # Relationships
    training_rounds = relationship(
        "FederatedTrainingRound",
        back_populates="participants",
        secondary="federated_round_participants",
    )


class FederatedTrainingRound(Base):
    """Represents a federated training round"""

    __tablename__ = "federated_training_rounds"

    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # Round identification
    round_id = Column(String(128), unique=True, index=True)
    round_number = Column(Integer, index=True)
    coordinator_node_id = Column(String(64), index=True)

    # Model information
    model_type = Column(
        String(32), index=True
    )  # isolation_forest, lstm_autoencoder, etc.
    model_version = Column(String(16), default="1.0")
    model_config = Column(JSON, nullable=True)  # Model hyperparameters and config

    # Round status and timing
    status = Column(String(16), default=FederatedStatusEnum.PREPARING)
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    timeout_at = Column(DateTime(timezone=True), nullable=True)

    # Participation requirements
    min_participants = Column(Integer, default=2)
    max_participants = Column(Integer, default=10)
    required_data_size = Column(Integer, default=100)

    # Aggregation settings
    aggregation_method = Column(
        String(32), default="federated_averaging"
    )  # federated_averaging, median, etc.
    convergence_threshold = Column(Float, default=0.001)
    max_iterations = Column(Integer, default=10)

    # Results and metrics
    final_accuracy = Column(Float, nullable=True)
    final_loss = Column(Float, nullable=True)
    model_weights_hash = Column(
        String(64), nullable=True
    )  # Hash of final aggregated weights
    aggregation_metrics = Column(JSON, nullable=True)  # Detailed aggregation statistics

    # Security
    encryption_enabled = Column(Boolean, default=True)
    differential_privacy = Column(Boolean, default=False)
    privacy_budget = Column(Float, nullable=True)  # For differential privacy

    # Relationships
    participants = relationship(
        "FederatedNode",
        back_populates="training_rounds",
        secondary="federated_round_participants",
    )
    model_updates = relationship(
        "FederatedModelUpdate", back_populates="training_round"
    )


class FederatedRoundParticipants(Base):
    """Association table for training round participants"""

    __tablename__ = "federated_round_participants"

    round_id = Column(
        Integer, ForeignKey("federated_training_rounds.id"), primary_key=True
    )
    node_id = Column(Integer, ForeignKey("federated_nodes.id"), primary_key=True)

    # Participation metadata
    joined_at = Column(DateTime(timezone=True), server_default=func.now())
    status = Column(
        String(16), default="invited"
    )  # invited|training|completed|failed|dropped
    data_size = Column(Integer, nullable=True)  # Size of local training data
    expected_completion = Column(DateTime(timezone=True), nullable=True)


class FederatedModelUpdate(Base):
    """Stores encrypted model updates from participants"""

    __tablename__ = "federated_model_updates"

    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)

    # Reference information
    round_id = Column(Integer, ForeignKey("federated_training_rounds.id"), index=True)
    node_id = Column(String(64), index=True)

    # Encrypted model data
    encrypted_weights = Column(Text)  # Base64-encoded encrypted model weights
    encryption_metadata = Column(JSON)  # Encryption keys, nonces, etc.
    weights_hash = Column(String(64))  # Hash of decrypted weights for validation

    # Training metadata
    local_epochs = Column(Integer, default=1)
    local_batch_size = Column(Integer, nullable=True)
    training_loss = Column(Float, nullable=True)
    training_accuracy = Column(Float, nullable=True)
    validation_loss = Column(Float, nullable=True)
    validation_accuracy = Column(Float, nullable=True)

    # Data and compute metrics
    training_data_size = Column(Integer)
    training_time_seconds = Column(Float, nullable=True)
    memory_usage_mb = Column(Float, nullable=True)
    cpu_usage_percent = Column(Float, nullable=True)

    # Security and validation
    digital_signature = Column(Text, nullable=True)  # Digital signature of the update
    validation_passed = Column(Boolean, default=False)
    validation_errors = Column(JSON, nullable=True)

    # Relationships
    training_round = relationship(
        "FederatedTrainingRound", back_populates="model_updates"
    )


class FederatedModelRegistry(Base):
    """Registry of federated models and their versions"""

    __tablename__ = "federated_model_registry"

    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # Model identification
    model_name = Column(String(128), index=True)
    model_type = Column(String(32), index=True)
    version = Column(String(16), index=True)

    # Model metadata
    description = Column(Text, nullable=True)
    architecture = Column(JSON, nullable=True)  # Model architecture description
    hyperparameters = Column(JSON, nullable=True)

    # Federated learning specific
    rounds_trained = Column(Integer, default=0)
    total_participants = Column(Integer, default=0)

    # Performance metrics
    global_accuracy = Column(Float, nullable=True)
    global_loss = Column(Float, nullable=True)
    convergence_rounds = Column(Integer, nullable=True)

    # Model storage
    model_weights_path = Column(String(256), nullable=True)
    model_size_mb = Column(Float, nullable=True)
    checksum = Column(String(64), nullable=True)

    # Status and lifecycle
    status = Column(
        String(16), default="training"
    )  # training|active|deprecated|archived
    deployment_ready = Column(Boolean, default=False)
    last_used_at = Column(DateTime(timezone=True), nullable=True)


class FederatedAuditLog(Base):
    """Audit log for federated learning operations"""

    __tablename__ = "federated_audit_log"

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), index=True)

    # Event information
    event_type = Column(
        String(32), index=True
    )  # node_join, round_start, model_update, etc.
    node_id = Column(String(64), index=True)
    round_id = Column(String(128), nullable=True, index=True)

    # Event details
    action = Column(String(128))  # Human-readable action description
    parameters = Column(JSON, nullable=True)  # Event parameters and data
    result = Column(String(16))  # success|failure|warning

    # Security and traceability
    ip_address = Column(String(45), nullable=True)  # Source IP address
    user_agent = Column(String(256), nullable=True)
    request_id = Column(String(64), nullable=True)

    # Performance metrics
    duration_ms = Column(Integer, nullable=True)
    data_size_bytes = Column(Integer, nullable=True)

    # Error information (if applicable)
    error_message = Column(Text, nullable=True)
    error_code = Column(String(32), nullable=True)
    stack_trace = Column(Text, nullable=True)


class FederatedPrivacyMetrics(Base):
    """Privacy-preserving metrics for federated learning"""

    __tablename__ = "federated_privacy_metrics"

    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Reference
    round_id = Column(Integer, ForeignKey("federated_training_rounds.id"), index=True)
    node_id = Column(String(64), index=True)

    # Privacy metrics (aggregated/anonymized)
    epsilon_used = Column(Float, nullable=True)  # Differential privacy epsilon
    delta_used = Column(Float, nullable=True)  # Differential privacy delta
    noise_scale = Column(Float, nullable=True)  # Noise added for privacy

    # Data distribution metrics (no individual data)
    feature_means = Column(JSON, nullable=True)  # Statistical means of features
    feature_stddevs = Column(JSON, nullable=True)  # Standard deviations
    data_quality_score = Column(Float, nullable=True)  # Overall data quality (0-1)

    # Model contribution metrics
    gradient_norm = Column(Float, nullable=True)  # L2 norm of gradients
    weight_contribution = Column(Float, nullable=True)  # Relative contribution weight
    influence_score = Column(Float, nullable=True)  # Model influence score

    # Security metrics
    encryption_strength = Column(Integer, nullable=True)  # Key size used
    secure_aggregation_used = Column(Boolean, default=True)
    integrity_verified = Column(Boolean, default=False)


class FederatedPerformanceMetrics(Base):
    """Performance and system metrics for federated learning"""

    __tablename__ = "federated_performance_metrics"

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), index=True)

    # Reference
    round_id = Column(
        Integer, ForeignKey("federated_training_rounds.id"), nullable=True, index=True
    )
    node_id = Column(String(64), index=True)

    # Timing metrics
    total_round_time_seconds = Column(Float, nullable=True)
    training_time_seconds = Column(Float, nullable=True)
    communication_time_seconds = Column(Float, nullable=True)
    aggregation_time_seconds = Column(Float, nullable=True)

    # Resource utilization
    cpu_usage_percent = Column(Float, nullable=True)
    memory_usage_mb = Column(Float, nullable=True)
    disk_io_mb = Column(Float, nullable=True)
    network_io_mb = Column(Float, nullable=True)

    # Model metrics
    model_size_mb = Column(Float, nullable=True)
    compression_ratio = Column(Float, nullable=True)
    update_sparsity = Column(Float, nullable=True)  # Percentage of zero weights

    # Quality metrics
    convergence_rate = Column(Float, nullable=True)
    accuracy_improvement = Column(Float, nullable=True)
    loss_reduction = Column(Float, nullable=True)

    # Network metrics
    bandwidth_used_mbps = Column(Float, nullable=True)
    latency_ms = Column(Float, nullable=True)
    packet_loss_rate = Column(Float, nullable=True)

    # Efficiency scores
    communication_efficiency = Column(
        Float, nullable=True
    )  # Data transferred per accuracy gain
    energy_efficiency = Column(Float, nullable=True)  # Performance per watt
    cost_efficiency = Column(Float, nullable=True)  # Performance per dollar


class FederatedConfiguration(Base):
    """Configuration settings for federated learning"""

    __tablename__ = "federated_configuration"

    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # Configuration scope
    config_name = Column(String(64), unique=True, index=True)
    config_type = Column(String(32), index=True)  # global|node|model|round
    target_id = Column(
        String(64), nullable=True, index=True
    )  # Target node/model ID if applicable

    # Configuration data
    config_data = Column(JSON)  # Configuration parameters
    schema_version = Column(String(16), default="1.0")

    # Status and validation
    status = Column(String(16), default="active")  # active|inactive|deprecated
    validated = Column(Boolean, default=False)
    validation_errors = Column(JSON, nullable=True)

    # Lifecycle
    effective_from = Column(DateTime(timezone=True), nullable=True)
    effective_until = Column(DateTime(timezone=True), nullable=True)
    applied_at = Column(DateTime(timezone=True), nullable=True)


# Utility functions for federated model operations


def create_federated_training_round(
    coordinator_id: str,
    model_type: FederatedModelTypeEnum,
    model_config: Dict[str, Any],
    min_participants: int = 2,
    max_participants: int = 10,
) -> FederatedTrainingRound:
    """Create a new federated training round"""

    round_id = f"fl_round_{int(datetime.now().timestamp())}_{model_type.value}"

    return FederatedTrainingRound(
        round_id=round_id,
        round_number=1,  # Will be updated based on previous rounds
        coordinator_node_id=coordinator_id,
        model_type=model_type.value,
        model_config=model_config,
        min_participants=min_participants,
        max_participants=max_participants,
        status=FederatedStatusEnum.PREPARING,
    )


def create_federated_node(
    node_id: str,
    role: FederatedRoleEnum,
    endpoint_url: str = None,
    node_name: str = None,
) -> FederatedNode:
    """Create a new federated learning node"""

    return FederatedNode(
        node_id=node_id,
        node_name=node_name or f"Node-{node_id[:8]}",
        role=role.value,
        endpoint_url=endpoint_url,
        status="active",
    )


def create_model_update_record(
    round_id: int,
    node_id: str,
    encrypted_weights: str,
    encryption_metadata: Dict[str, Any],
    training_metrics: Dict[str, Any] = None,
) -> FederatedModelUpdate:
    """Create a federated model update record"""

    metrics = training_metrics or {}

    return FederatedModelUpdate(
        round_id=round_id,
        node_id=node_id,
        encrypted_weights=encrypted_weights,
        encryption_metadata=encryption_metadata,
        training_data_size=metrics.get("data_size", 0),
        training_time_seconds=metrics.get("training_time", 0.0),
        training_loss=metrics.get("training_loss"),
        training_accuracy=metrics.get("training_accuracy"),
        validation_loss=metrics.get("validation_loss"),
        validation_accuracy=metrics.get("validation_accuracy"),
    )


# Schema validation helpers

FEDERATED_MODEL_SCHEMAS = {
    "isolation_forest": {
        "required_params": ["n_estimators", "contamination", "max_samples"],
        "optional_params": ["max_features", "bootstrap", "random_state"],
        "param_types": {
            "n_estimators": int,
            "contamination": float,
            "max_samples": (int, str),
            "max_features": (int, float),
            "bootstrap": bool,
            "random_state": int,
        },
    },
    "lstm_autoencoder": {
        "required_params": ["input_size", "hidden_size", "sequence_length"],
        "optional_params": [
            "num_layers",
            "dropout",
            "learning_rate",
            "batch_size",
            "epochs",
        ],
        "param_types": {
            "input_size": int,
            "hidden_size": int,
            "sequence_length": int,
            "num_layers": int,
            "dropout": float,
            "learning_rate": float,
            "batch_size": int,
            "epochs": int,
        },
    },
    "neural_network": {
        "required_params": ["input_size", "hidden_layers", "output_size"],
        "optional_params": [
            "activation",
            "optimizer",
            "learning_rate",
            "batch_size",
            "epochs",
        ],
        "param_types": {
            "input_size": int,
            "hidden_layers": list,
            "output_size": int,
            "activation": str,
            "optimizer": str,
            "learning_rate": float,
            "batch_size": int,
            "epochs": int,
        },
    },
}


def validate_federated_model_config(
    model_type: str, config: Dict[str, Any]
) -> Dict[str, Any]:
    """Validate federated model configuration against schema"""

    if model_type not in FEDERATED_MODEL_SCHEMAS:
        return {"valid": False, "errors": [f"Unsupported model type: {model_type}"]}

    schema = FEDERATED_MODEL_SCHEMAS[model_type]
    errors = []

    # Check required parameters
    for param in schema["required_params"]:
        if param not in config:
            errors.append(f"Missing required parameter: {param}")

    # Check parameter types
    for param, value in config.items():
        if param in schema["param_types"]:
            expected_type = schema["param_types"][param]
            if not isinstance(value, expected_type):
                errors.append(
                    f"Parameter {param} should be of type {expected_type}, got {type(value)}"
                )

    return {"valid": len(errors) == 0, "errors": errors, "validated_config": config}
