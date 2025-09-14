"""
Federated Learning System for Mini-XDR
Enables distributed machine learning across multiple Mini-XDR instances
with privacy-preserving secure aggregation protocols.
"""

import asyncio
import logging
import json
import hashlib
import time
import uuid
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
from enum import Enum
import pickle
import base64
import numpy as np

# TensorFlow imports for custom federated learning
try:
    import tensorflow as tf
    # Suppress TF warnings for cleaner logs
    tf.get_logger().setLevel('ERROR')
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    tf = None

# Alias for backward compatibility with main.py imports
FEDERATED_AVAILABLE = TF_AVAILABLE

# Cryptography imports
from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_OAEP, AES
from Crypto.Random import get_random_bytes
from Crypto.Hash import SHA256
from Crypto.Signature import pkcs1_15

# Core ML imports
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator
import joblib

# Local imports
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, func
from .models import Event, MLModel, Incident
from .db import AsyncSessionLocal
from .config import settings

logger = logging.getLogger(__name__)


class FederatedModelType(Enum):
    """Types of federated models supported"""
    ISOLATION_FOREST = "isolation_forest"
    LSTM_AUTOENCODER = "lstm_autoencoder"
    NEURAL_NETWORK = "neural_network"
    ENSEMBLE = "ensemble"


class FederatedRole(Enum):
    """Roles in federated learning"""
    COORDINATOR = "coordinator"  # Central aggregator
    PARTICIPANT = "participant"  # Contributing node
    OBSERVER = "observer"       # Read-only node


class FederatedStatus(Enum):
    """Status of federated learning process"""
    IDLE = "idle"
    PREPARING = "preparing"
    TRAINING = "training"
    AGGREGATING = "aggregating" 
    COMPLETED = "completed"
    FAILED = "failed"


class SecureAggregation:
    """Privacy-preserving secure aggregation using cryptographic protocols"""
    
    def __init__(self):
        self.key_size = 2048
        self.aes_key_size = 32  # 256 bits
        
    def generate_keypair(self) -> Tuple[bytes, bytes]:
        """Generate RSA public/private key pair"""
        key = RSA.generate(self.key_size)
        private_key = key.export_key('PEM')
        public_key = key.publickey().export_key('PEM')
        return public_key, private_key
    
    def encrypt_model_update(self, model_update: np.ndarray, public_key: bytes) -> Dict[str, str]:
        """Encrypt model updates using hybrid encryption (RSA + AES)"""
        try:
            # Generate AES key for data encryption
            aes_key = get_random_bytes(self.aes_key_size)
            
            # Encrypt model update with AES
            cipher_aes = AES.new(aes_key, AES.MODE_GCM)
            serialized_update = pickle.dumps(model_update)
            ciphertext, auth_tag = cipher_aes.encrypt_and_digest(serialized_update)
            
            # Encrypt AES key with RSA public key
            rsa_key = RSA.import_key(public_key)
            cipher_rsa = PKCS1_OAEP.new(rsa_key)
            encrypted_aes_key = cipher_rsa.encrypt(aes_key)
            
            return {
                'encrypted_data': base64.b64encode(ciphertext).decode('utf-8'),
                'encrypted_key': base64.b64encode(encrypted_aes_key).decode('utf-8'),
                'nonce': base64.b64encode(cipher_aes.nonce).decode('utf-8'),
                'auth_tag': base64.b64encode(auth_tag).decode('utf-8')
            }
            
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            raise
    
    def decrypt_model_update(self, encrypted_update: Dict[str, str], private_key: bytes) -> np.ndarray:
        """Decrypt model updates"""
        try:
            # Decrypt AES key with RSA private key
            rsa_key = RSA.import_key(private_key)
            cipher_rsa = PKCS1_OAEP.new(rsa_key)
            encrypted_aes_key = base64.b64decode(encrypted_update['encrypted_key'])
            aes_key = cipher_rsa.decrypt(encrypted_aes_key)
            
            # Decrypt model update with AES key
            nonce = base64.b64decode(encrypted_update['nonce'])
            auth_tag = base64.b64decode(encrypted_update['auth_tag'])
            ciphertext = base64.b64decode(encrypted_update['encrypted_data'])
            
            cipher_aes = AES.new(aes_key, AES.MODE_GCM, nonce=nonce)
            serialized_update = cipher_aes.decrypt_and_verify(ciphertext, auth_tag)
            
            return pickle.loads(serialized_update)
            
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise
    
    def aggregate_updates(self, encrypted_updates: List[Dict], private_key: bytes, 
                         weights: Optional[List[float]] = None) -> np.ndarray:
        """Securely aggregate encrypted model updates"""
        try:
            decrypted_updates = []
            
            # Decrypt all updates
            for encrypted_update in encrypted_updates:
                update = self.decrypt_model_update(encrypted_update, private_key)
                decrypted_updates.append(update)
            
            if not decrypted_updates:
                raise ValueError("No updates to aggregate")
            
            # Weighted average aggregation
            if weights is None:
                weights = [1.0 / len(decrypted_updates)] * len(decrypted_updates)
            
            # Ensure weights sum to 1.0
            weight_sum = sum(weights)
            normalized_weights = [w / weight_sum for w in weights]
            
            # Aggregate updates
            aggregated = np.zeros_like(decrypted_updates[0])
            for update, weight in zip(decrypted_updates, normalized_weights):
                aggregated += weight * update
            
            return aggregated
            
        except Exception as e:
            logger.error(f"Secure aggregation failed: {e}")
            raise


class FederatedModelWrapper:
    """Wrapper for different model types to support federated learning"""
    
    def __init__(self, model_type: FederatedModelType, model_config: Dict[str, Any] = None):
        self.model_type = model_type
        self.model_config = model_config or {}
        self.model = None
        self.is_trained = False
        
    def create_model(self) -> Any:
        """Create model based on type"""
        if self.model_type == FederatedModelType.LSTM_AUTOENCODER:
            return self._create_lstm_autoencoder()
        elif self.model_type == FederatedModelType.NEURAL_NETWORK:
            return self._create_neural_network()
        elif self.model_type == FederatedModelType.ISOLATION_FOREST:
            return self._create_isolation_forest()
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def _create_lstm_autoencoder(self) -> tf.keras.Model:
        """Create TensorFlow LSTM autoencoder for federated learning"""
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required for LSTM autoencoder")
        
        input_size = self.model_config.get('input_size', 15)
        hidden_size = self.model_config.get('hidden_size', 64)
        sequence_length = self.model_config.get('sequence_length', 10)
        
        # Build model with explicit input layer for better compatibility
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(sequence_length, input_size)),
            tf.keras.layers.LSTM(hidden_size, return_sequences=True),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(hidden_size, return_sequences=True),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(input_size))
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.MeanAbsoluteError()]
        )
        
        return model
    
    def _create_neural_network(self) -> tf.keras.Model:
        """Create TensorFlow neural network for federated learning"""
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required for neural network")
        
        input_size = self.model_config.get('input_size', 15)
        hidden_layers = self.model_config.get('hidden_layers', [64, 32])
        output_size = self.model_config.get('output_size', 1)
        
        layers = [tf.keras.layers.Input(shape=(input_size,))]
        layers.append(tf.keras.layers.Dense(hidden_layers[0], activation='relu'))
        layers.append(tf.keras.layers.Dropout(0.3))
        
        for hidden_size in hidden_layers[1:]:
            layers.append(tf.keras.layers.Dense(hidden_size, activation='relu'))
            layers.append(tf.keras.layers.Dropout(0.2))
        
        layers.append(tf.keras.layers.Dense(output_size, activation='sigmoid'))
        
        model = tf.keras.Sequential(layers)
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=[tf.keras.metrics.BinaryAccuracy()]
        )
        
        return model
    
    def _create_isolation_forest(self):
        """Create sklearn isolation forest (non-federated fallback)"""
        from sklearn.ensemble import IsolationForest
        
        return IsolationForest(
            contamination=self.model_config.get('contamination', 0.1),
            random_state=42,
            n_estimators=self.model_config.get('n_estimators', 100)
        )
    
    def get_model_weights(self) -> np.ndarray:
        """Get model weights as numpy array"""
        if hasattr(self.model, 'get_weights'):  # TensorFlow model
            weights = self.model.get_weights()
            return np.concatenate([w.flatten() for w in weights])
        else:
            # For sklearn models, we'll need to serialize the entire model
            return pickle.dumps(self.model)
    
    def set_model_weights(self, weights: np.ndarray):
        """Set model weights from numpy array"""
        if hasattr(self.model, 'set_weights'):  # TensorFlow model
            # Reconstruct weight shapes and set
            weight_shapes = [w.shape for w in self.model.get_weights()]
            reconstructed_weights = []
            start_idx = 0
            
            for shape in weight_shapes:
                size = np.prod(shape)
                weight_data = weights[start_idx:start_idx + size]
                reconstructed_weights.append(weight_data.reshape(shape))
                start_idx += size
            
            self.model.set_weights(reconstructed_weights)
        else:
            # For sklearn models, deserialize
            self.model = pickle.loads(weights)


class FederatedLearningCoordinator:
    """Central coordinator for federated learning across Mini-XDR instances"""
    
    def __init__(self, node_id: str = None):
        self.node_id = node_id or str(uuid.uuid4())
        self.role = FederatedRole.COORDINATOR
        self.status = FederatedStatus.IDLE
        
        # Cryptographic components
        self.secure_aggregation = SecureAggregation()
        self.public_key = None
        self.private_key = None
        
        # Federated learning state
        self.current_round = 0
        self.participants = {}  # node_id -> node_info
        self.model_updates = {}  # node_id -> encrypted_update
        self.global_model = None
        
        # Configuration
        self.min_participants = 2
        self.max_participants = 10
        self.rounds_per_epoch = 5
        self.convergence_threshold = 0.001
        
        # Performance tracking
        self.round_metrics = []
        self.aggregation_history = []
        
        self._initialize_crypto()
    
    def _initialize_crypto(self):
        """Initialize cryptographic keys"""
        try:
            self.public_key, self.private_key = self.secure_aggregation.generate_keypair()
            logger.info(f"Federated coordinator {self.node_id} initialized with crypto keys")
        except Exception as e:
            logger.error(f"Crypto initialization failed: {e}")
            raise
    
    async def register_participant(self, node_id: str, node_info: Dict[str, Any]) -> bool:
        """Register a new participant node"""
        try:
            if len(self.participants) >= self.max_participants:
                logger.warning(f"Cannot register {node_id}: max participants reached")
                return False
            
            # Validate participant info
            required_fields = ['endpoint', 'model_type', 'data_size']
            if not all(field in node_info for field in required_fields):
                logger.error(f"Invalid participant info from {node_id}")
                return False
            
            self.participants[node_id] = {
                'registered_at': datetime.now(timezone.utc),
                'status': 'active',
                'rounds_participated': 0,
                'last_update': None,
                **node_info
            }
            
            logger.info(f"Registered participant {node_id}, total: {len(self.participants)}")
            return True
            
        except Exception as e:
            logger.error(f"Participant registration failed: {e}")
            return False
    
    async def start_federated_round(self, model_type: FederatedModelType, 
                                  model_config: Dict[str, Any] = None) -> str:
        """Start a new federated learning round"""
        try:
            if self.status != FederatedStatus.IDLE:
                raise ValueError(f"Cannot start round, current status: {self.status}")
            
            if len(self.participants) < self.min_participants:
                raise ValueError(f"Need at least {self.min_participants} participants")
            
            self.status = FederatedStatus.PREPARING
            self.current_round += 1
            round_id = f"round_{self.current_round}_{int(time.time())}"
            
            # Initialize global model if first round
            if self.global_model is None:
                self.global_model = FederatedModelWrapper(model_type, model_config)
                self.global_model.model = self.global_model.create_model()
            
            # Reset round state
            self.model_updates.clear()
            
            # Notify participants to start training
            active_participants = [
                node_id for node_id, info in self.participants.items()
                if info['status'] == 'active'
            ]
            
            logger.info(f"Starting federated round {round_id} with {len(active_participants)} participants")
            
            # In a real implementation, you would send HTTP requests to participants
            # For now, we'll simulate the process
            self.status = FederatedStatus.TRAINING
            
            return round_id
            
        except Exception as e:
            logger.error(f"Failed to start federated round: {e}")
            self.status = FederatedStatus.FAILED
            raise
    
    async def receive_model_update(self, node_id: str, encrypted_update: Dict[str, str],
                                 round_id: str, metadata: Dict[str, Any] = None) -> bool:
        """Receive and store encrypted model update from participant"""
        try:
            if node_id not in self.participants:
                logger.warning(f"Received update from unregistered participant: {node_id}")
                return False
            
            if self.status != FederatedStatus.TRAINING:
                logger.warning(f"Received update outside training phase: {self.status}")
                return False
            
            # Store encrypted update
            self.model_updates[node_id] = {
                'encrypted_update': encrypted_update,
                'round_id': round_id,
                'received_at': datetime.now(timezone.utc),
                'metadata': metadata or {}
            }
            
            # Update participant status
            self.participants[node_id]['last_update'] = datetime.now(timezone.utc)
            self.participants[node_id]['rounds_participated'] += 1
            
            logger.info(f"Received model update from {node_id} ({len(self.model_updates)}/{len(self.participants)})")
            
            # Check if we have all updates
            if len(self.model_updates) >= len(self.participants):
                await self._aggregate_models()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to receive model update from {node_id}: {e}")
            return False
    
    async def _aggregate_models(self):
        """Aggregate received model updates using secure aggregation"""
        try:
            self.status = FederatedStatus.AGGREGATING
            
            # Extract encrypted updates
            encrypted_updates = [
                info['encrypted_update'] 
                for info in self.model_updates.values()
            ]
            
            # Calculate weights based on data size (if available)
            weights = []
            for node_id, update_info in self.model_updates.items():
                participant_info = self.participants[node_id]
                data_size = participant_info.get('data_size', 1)
                weights.append(data_size)
            
            # Normalize weights
            total_weight = sum(weights)
            normalized_weights = [w / total_weight for w in weights]
            
            # Perform secure aggregation
            aggregated_weights = self.secure_aggregation.aggregate_updates(
                encrypted_updates, self.private_key, normalized_weights
            )
            
            # Update global model
            self.global_model.set_model_weights(aggregated_weights)
            
            # Record metrics
            round_metric = {
                'round': self.current_round,
                'participants': len(self.model_updates),
                'aggregated_at': datetime.now(timezone.utc),
                'weights_norm': float(np.linalg.norm(aggregated_weights))
            }
            self.round_metrics.append(round_metric)
            
            self.status = FederatedStatus.COMPLETED
            logger.info(f"Completed federated round {self.current_round} with {len(self.model_updates)} participants")
            
        except Exception as e:
            logger.error(f"Model aggregation failed: {e}")
            self.status = FederatedStatus.FAILED
            raise
    
    async def get_global_model(self) -> Dict[str, Any]:
        """Get the current global model state"""
        if self.global_model is None:
            return {'error': 'No global model available'}
        
        return {
            'model_type': self.global_model.model_type.value,
            'current_round': self.current_round,
            'status': self.status.value,
            'participants': len(self.participants),
            'weights_available': self.global_model.model is not None
        }
    
    def get_coordinator_status(self) -> Dict[str, Any]:
        """Get comprehensive coordinator status"""
        return {
            'node_id': self.node_id,
            'role': self.role.value,
            'status': self.status.value,
            'current_round': self.current_round,
            'participants': {
                'count': len(self.participants),
                'active': sum(1 for p in self.participants.values() if p['status'] == 'active'),
                'details': self.participants
            },
            'global_model': self.global_model.model_type.value if self.global_model else None,
            'round_metrics': self.round_metrics[-10:],  # Last 10 rounds
            'crypto_initialized': self.public_key is not None
        }


class FederatedLearningParticipant:
    """Participant node in federated learning"""
    
    def __init__(self, node_id: str = None, coordinator_endpoint: str = None):
        self.node_id = node_id or str(uuid.uuid4())
        self.coordinator_endpoint = coordinator_endpoint
        self.role = FederatedRole.PARTICIPANT
        self.status = FederatedStatus.IDLE
        
        # Local model
        self.local_model = None
        self.training_data = None
        
        # Crypto
        self.secure_aggregation = SecureAggregation()
        self.coordinator_public_key = None
        
        # Metrics
        self.training_metrics = []
    
    async def register_with_coordinator(self, model_type: FederatedModelType, 
                                      data_size: int) -> bool:
        """Register with the federated learning coordinator"""
        try:
            participant_info = {
                'endpoint': f"http://localhost:8000/api/federated/participant/{self.node_id}",
                'model_type': model_type.value,
                'data_size': data_size,
                'node_version': '1.0'
            }
            
            # In a real implementation, make HTTP request to coordinator
            # For simulation, assume registration is successful
            logger.info(f"Participant {self.node_id} registered with coordinator")
            return True
            
        except Exception as e:
            logger.error(f"Registration with coordinator failed: {e}")
            return False
    
    async def train_local_model(self, training_data: List[Dict], 
                               model_config: Dict[str, Any]) -> np.ndarray:
        """Train local model on private data"""
        try:
            self.status = FederatedStatus.TRAINING
            
            # Create local model if needed
            if self.local_model is None:
                model_type = FederatedModelType(model_config.get('model_type', 'neural_network'))
                self.local_model = FederatedModelWrapper(model_type, model_config)
                self.local_model.model = self.local_model.create_model()
            
            # Convert training data to appropriate format
            X, y = self._prepare_training_data(training_data)
            
            # Train model locally
            if hasattr(self.local_model.model, 'fit'):
                if hasattr(self.local_model.model, 'partial_fit'):
                    # Incremental learning for sklearn models
                    self.local_model.model.partial_fit(X, y)
                else:
                    # Regular training for tensorflow models
                    history = self.local_model.model.fit(X, y, epochs=5, verbose=0, validation_split=0.2)
                    
                    # Record training metrics
                    self.training_metrics.append({
                        'loss': history.history['loss'][-1] if 'loss' in history.history else 0,
                        'accuracy': history.history.get('accuracy', [0])[-1],
                        'trained_at': datetime.now(timezone.utc)
                    })
            
            # Get model weights for federated aggregation
            model_weights = self.local_model.get_model_weights()
            
            self.status = FederatedStatus.COMPLETED
            logger.info(f"Local training completed for participant {self.node_id}")
            
            return model_weights
            
        except Exception as e:
            logger.error(f"Local model training failed: {e}")
            self.status = FederatedStatus.FAILED
            raise
    
    def _prepare_training_data(self, training_data: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data for model"""
        # Feature extraction similar to existing ML engine
        features = []
        labels = []
        
        for data_point in training_data:
            # Use existing feature columns from ml_engine
            feature_vector = [
                data_point.get('event_count_1h', 0),
                data_point.get('event_count_24h', 0),
                data_point.get('unique_ports', 0),
                data_point.get('failed_login_count', 0),
                data_point.get('session_duration_avg', 0),
                data_point.get('password_diversity', 0),
                data_point.get('username_diversity', 0),
                data_point.get('event_rate_per_minute', 0),
                data_point.get('time_of_day', 0),
                data_point.get('is_weekend', 0),
                data_point.get('unique_usernames', 0),
                data_point.get('password_length_avg', 0),
                data_point.get('command_diversity', 0),
                data_point.get('download_attempts', 0),
                data_point.get('upload_attempts', 0)
            ]
            
            features.append(feature_vector)
            
            # Simple anomaly label (1 if anomalous, 0 if normal)
            # In practice, this would be derived from incident data
            labels.append(data_point.get('is_anomaly', 0))
        
        return np.array(features), np.array(labels)
    
    async def send_model_update(self, model_weights: np.ndarray, 
                               round_id: str) -> bool:
        """Send encrypted model update to coordinator"""
        try:
            if self.coordinator_public_key is None:
                raise ValueError("Coordinator public key not available")
            
            # Encrypt model update
            encrypted_update = self.secure_aggregation.encrypt_model_update(
                model_weights, self.coordinator_public_key
            )
            
            # In real implementation, send HTTP request to coordinator
            # For simulation, assume it's sent successfully
            logger.info(f"Model update sent from participant {self.node_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to send model update: {e}")
            return False
    
    def get_participant_status(self) -> Dict[str, Any]:
        """Get participant status"""
        return {
            'node_id': self.node_id,
            'role': self.role.value,
            'status': self.status.value,
            'coordinator_endpoint': self.coordinator_endpoint,
            'local_model': self.local_model.model_type.value if self.local_model else None,
            'training_metrics': self.training_metrics[-5:],  # Last 5 training sessions
            'crypto_ready': self.coordinator_public_key is not None
        }


class FederatedLearningManager:
    """High-level manager for federated learning operations"""
    
    def __init__(self):
        self.coordinator = None
        self.participant = None
        self.current_mode = None
        
        # Integration with existing ML pipeline
        self.model_dir = Path("models/federated")
        self.model_dir.mkdir(exist_ok=True)
        
    async def initialize_coordinator(self, config: Dict[str, Any] = None) -> FederatedLearningCoordinator:
        """Initialize as federated learning coordinator"""
        try:
            self.coordinator = FederatedLearningCoordinator()
            self.current_mode = FederatedRole.COORDINATOR
            
            # Save coordinator info to database
            await self._save_federated_config('coordinator', self.coordinator.node_id, config or {})
            
            logger.info(f"Initialized federated learning coordinator: {self.coordinator.node_id}")
            return self.coordinator
            
        except Exception as e:
            logger.error(f"Coordinator initialization failed: {e}")
            raise
    
    async def initialize_participant(self, coordinator_endpoint: str, 
                                   config: Dict[str, Any] = None) -> FederatedLearningParticipant:
        """Initialize as federated learning participant"""
        try:
            self.participant = FederatedLearningParticipant(
                coordinator_endpoint=coordinator_endpoint
            )
            self.current_mode = FederatedRole.PARTICIPANT
            
            # Save participant info to database
            config_with_endpoint = {**(config or {}), 'coordinator_endpoint': coordinator_endpoint}
            await self._save_federated_config('participant', self.participant.node_id, config_with_endpoint)
            
            logger.info(f"Initialized federated learning participant: {self.participant.node_id}")
            return self.participant
            
        except Exception as e:
            logger.error(f"Participant initialization failed: {e}")
            raise
    
    async def _save_federated_config(self, role: str, node_id: str, config: Dict[str, Any]):
        """Save federated learning configuration to database"""
        try:
            async with AsyncSessionLocal() as db:
                # Check if federated config already exists
                query = select(MLModel).where(
                    and_(
                        MLModel.name == f"federated_{role}_{node_id}",
                        MLModel.model_type == "federated_config"
                    )
                )
                result = await db.execute(query)
                existing_model = result.scalar_one_or_none()
                
                if existing_model:
                    # Update existing config
                    existing_model.hyperparameters = config
                    existing_model.updated_at = datetime.now(timezone.utc)
                else:
                    # Create new config
                    federated_model = MLModel(
                        name=f"federated_{role}_{node_id}",
                        model_type="federated_config",
                        status="active",
                        is_federated=True,
                        hyperparameters=config
                    )
                    db.add(federated_model)
                
                await db.commit()
                logger.info(f"Saved federated config for {role}: {node_id}")
                
        except Exception as e:
            logger.error(f"Failed to save federated config: {e}")
    
    async def start_federated_training(self, model_type: str, 
                                     training_data: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """Start federated training process"""
        try:
            if self.current_mode == FederatedRole.COORDINATOR:
                return await self._coordinator_start_training(model_type)
            elif self.current_mode == FederatedRole.PARTICIPANT:
                return await self._participant_start_training(model_type, training_data)
            else:
                raise ValueError("Federated learning not initialized")
                
        except Exception as e:
            logger.error(f"Failed to start federated training: {e}")
            raise
    
    async def _coordinator_start_training(self, model_type: str) -> Dict[str, Any]:
        """Start training as coordinator"""
        try:
            federated_model_type = FederatedModelType(model_type)
            
            # Default model configuration
            model_config = {
                'input_size': 15,  # Based on existing feature columns
                'hidden_size': 64,
                'sequence_length': 10
            }
            
            round_id = await self.coordinator.start_federated_round(
                federated_model_type, model_config
            )
            
            return {
                'status': 'started',
                'round_id': round_id,
                'coordinator_id': self.coordinator.node_id,
                'participants': len(self.coordinator.participants)
            }
            
        except Exception as e:
            logger.error(f"Coordinator training start failed: {e}")
            raise
    
    async def _participant_start_training(self, model_type: str, 
                                        training_data: List[Dict]) -> Dict[str, Any]:
        """Start training as participant"""
        try:
            if not training_data:
                # Get training data from local events
                training_data = await self._get_local_training_data()
            
            # Register with coordinator
            federated_model_type = FederatedModelType(model_type)
            await self.participant.register_with_coordinator(
                federated_model_type, len(training_data)
            )
            
            # Train local model
            model_config = {
                'model_type': model_type,
                'input_size': 15,
                'hidden_size': 64
            }
            
            model_weights = await self.participant.train_local_model(
                training_data, model_config
            )
            
            # Send update to coordinator (simulated)
            round_id = f"round_{int(time.time())}"
            await self.participant.send_model_update(model_weights, round_id)
            
            return {
                'status': 'completed',
                'participant_id': self.participant.node_id,
                'training_samples': len(training_data),
                'model_weights_size': len(model_weights) if isinstance(model_weights, np.ndarray) else 0
            }
            
        except Exception as e:
            logger.error(f"Participant training failed: {e}")
            raise
    
    async def _get_local_training_data(self) -> List[Dict[str, Any]]:
        """Get training data from local events"""
        try:
            from .ml_engine import prepare_training_data_from_events
            
            async with AsyncSessionLocal() as db:
                # Get recent clean events for training
                window_start = datetime.now(timezone.utc) - timedelta(days=7)
                
                query = select(Event).where(
                    Event.ts >= window_start
                ).order_by(Event.ts.desc()).limit(1000)
                
                result = await db.execute(query)
                events = result.scalars().all()
                
                # Convert to training data format
                training_data = await prepare_training_data_from_events(events)
                
                logger.info(f"Prepared {len(training_data)} samples for federated training")
                return training_data
                
        except Exception as e:
            logger.error(f"Failed to get local training data: {e}")
            return []
    
    def get_federated_status(self) -> Dict[str, Any]:
        """Get current federated learning status"""
        status = {
            'initialized': self.current_mode is not None,
            'mode': self.current_mode.value if self.current_mode else None,
            'tensorflow_available': TF_AVAILABLE
        }
        
        if self.coordinator:
            status['coordinator'] = self.coordinator.get_coordinator_status()
        
        if self.participant:
            status['participant'] = self.participant.get_participant_status()
        
        return status
    
    async def cleanup(self):
        """Cleanup federated learning resources"""
        if self.coordinator:
            # In a real implementation, notify participants
            logger.info("Cleaning up federated coordinator")
        
        if self.participant:
            # In a real implementation, unregister from coordinator
            logger.info("Cleaning up federated participant")
        
        self.coordinator = None
        self.participant = None
        self.current_mode = None


# Global federated learning manager instance
federated_manager = FederatedLearningManager()
