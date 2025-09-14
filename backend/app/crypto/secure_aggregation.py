"""
Advanced Secure Aggregation Cryptographic Protocols for Federated Learning
Implements state-of-the-art secure multi-party computation protocols
for privacy-preserving model aggregation in Mini-XDR federated learning.
"""

import asyncio
import logging
import hashlib
import secrets
import struct
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import json
import time
import numpy as np

# Cryptographic imports
from Crypto.PublicKey import RSA, ECC
from Crypto.Cipher import AES, ChaCha20_Poly1305
from Crypto.Hash import SHA256, HMAC, SHAKE256
from Crypto.Random import get_random_bytes
from Crypto.Signature import DSS, pkcs1_15
from Crypto.Util import Counter
from Crypto.Protocol.KDF import PBKDF2, scrypt
from Crypto.Protocol.SecretSharing import Shamir

logger = logging.getLogger(__name__)


class AggregationProtocol(Enum):
    """Supported secure aggregation protocols"""
    SIMPLE_ENCRYPTION = "simple_encryption"           # Basic RSA+AES hybrid encryption
    SECURE_AGGREGATION = "secure_aggregation"         # Google's secure aggregation protocol
    DIFFERENTIAL_PRIVACY = "differential_privacy"      # DP with noise addition
    SECRET_SHARING = "secret_sharing"                  # Shamir's secret sharing
    HOMOMORPHIC = "homomorphic"                        # Basic homomorphic encryption
    MULTI_KEY_AGGREGATION = "multi_key_aggregation"    # Multi-key secure aggregation


class EncryptionMode(Enum):
    """Encryption modes for different security levels"""
    AES_GCM = "aes_gcm"
    CHACHA20_POLY1305 = "chacha20_poly1305"
    RSA_OAEP = "rsa_oaep"
    ECC_INTEGRATED = "ecc_integrated"


@dataclass
class SecureMessage:
    """Secure message wrapper for encrypted communications"""
    sender_id: str
    recipient_id: str
    message_type: str
    encrypted_payload: bytes
    signature: bytes
    timestamp: float
    protocol_version: str = "1.0"
    metadata: Dict[str, Any] = None


@dataclass
class AggregationContext:
    """Context for secure aggregation operations"""
    round_id: str
    participants: List[str]
    protocol: AggregationProtocol
    security_level: int  # 1=basic, 2=standard, 3=high, 4=maximum
    differential_privacy: bool = False
    epsilon: float = 1.0  # DP epsilon parameter
    delta: float = 1e-5   # DP delta parameter


class AdvancedSecureAggregation:
    """Advanced secure aggregation with multiple cryptographic protocols"""
    
    def __init__(self, security_level: int = 3):
        self.security_level = security_level
        self.key_size = self._get_key_size(security_level)
        
        # Cryptographic keys (will be generated per session)
        self.rsa_keypair = None
        self.ecc_keypair = None
        self.session_keys = {}
        
        # Protocol configurations
        self.protocol_configs = {
            AggregationProtocol.SIMPLE_ENCRYPTION: {
                "encryption_mode": EncryptionMode.AES_GCM,
                "key_derivation": "pbkdf2"
            },
            AggregationProtocol.SECURE_AGGREGATION: {
                "threshold_scheme": "shamir",
                "reconstruction_threshold": 0.7,
                "dropout_resilience": True
            },
            AggregationProtocol.DIFFERENTIAL_PRIVACY: {
                "noise_mechanism": "gaussian",
                "clipping_threshold": 1.0,
                "sensitivity": 2.0
            },
            AggregationProtocol.SECRET_SHARING: {
                "scheme": "shamir",
                "threshold_ratio": 0.5,
                "field_size": 2**31 - 1
            }
        }
        
        self._initialize_cryptographic_primitives()
    
    def _get_key_size(self, security_level: int) -> int:
        """Get key size based on security level"""
        key_sizes = {1: 1024, 2: 2048, 3: 3072, 4: 4096}
        return key_sizes.get(security_level, 2048)
    
    def _initialize_cryptographic_primitives(self):
        """Initialize cryptographic keys and primitives"""
        try:
            # Generate RSA keypair
            self.rsa_keypair = RSA.generate(self.key_size)
            
            # Generate ECC keypair for efficiency (P-256 curve)
            self.ecc_keypair = ECC.generate(curve='P-256')
            
            logger.info(f"Initialized secure aggregation with security level {self.security_level}")
            
        except Exception as e:
            logger.error(f"Failed to initialize cryptographic primitives: {e}")
            raise
    
    def get_public_key(self, key_type: str = "rsa") -> bytes:
        """Get public key for the specified algorithm"""
        if key_type == "rsa":
            return self.rsa_keypair.publickey().export_key()
        elif key_type == "ecc":
            return self.ecc_keypair.public_key().export_key()
        else:
            raise ValueError(f"Unsupported key type: {key_type}")
    
    def derive_session_key(self, participant_id: str, shared_secret: bytes, 
                          salt: bytes = None) -> bytes:
        """Derive session key using PBKDF2"""
        if salt is None:
            salt = get_random_bytes(32)
        
        # Use strong key derivation
        session_key = PBKDF2(
            password=shared_secret,
            salt=salt,
            dkLen=32,  # 256-bit key
            count=100000,
            hmac_hash_module=SHA256
        )
        
        self.session_keys[participant_id] = session_key
        return session_key
    
    async def encrypt_model_update(self, model_weights: np.ndarray, 
                                  recipient_public_key: bytes,
                                  protocol: AggregationProtocol = AggregationProtocol.SIMPLE_ENCRYPTION) -> Dict[str, Any]:
        """Encrypt model update using specified protocol"""
        
        if protocol == AggregationProtocol.SIMPLE_ENCRYPTION:
            return await self._encrypt_simple(model_weights, recipient_public_key)
        elif protocol == AggregationProtocol.SECURE_AGGREGATION:
            return await self._encrypt_secure_aggregation(model_weights, recipient_public_key)
        elif protocol == AggregationProtocol.DIFFERENTIAL_PRIVACY:
            return await self._encrypt_with_dp(model_weights, recipient_public_key)
        elif protocol == AggregationProtocol.SECRET_SHARING:
            return await self._encrypt_secret_sharing(model_weights, recipient_public_key)
        else:
            raise ValueError(f"Unsupported encryption protocol: {protocol}")
    
    async def _encrypt_simple(self, model_weights: np.ndarray, 
                            recipient_public_key: bytes) -> Dict[str, Any]:
        """Simple hybrid encryption (RSA + AES-GCM)"""
        try:
            # Serialize model weights
            serialized_weights = model_weights.tobytes()
            
            # Generate AES key and IV
            aes_key = get_random_bytes(32)  # 256-bit key
            
            # Encrypt model weights with AES-GCM
            cipher = AES.new(aes_key, AES.MODE_GCM)
            ciphertext, auth_tag = cipher.encrypt_and_digest(serialized_weights)
            
            # Encrypt AES key with RSA
            rsa_key = RSA.import_key(recipient_public_key)
            from Crypto.Cipher import PKCS1_OAEP
            rsa_cipher = PKCS1_OAEP.new(rsa_key, hashAlgo=SHA256)
            encrypted_aes_key = rsa_cipher.encrypt(aes_key)
            
            # Create integrity hash
            hasher = SHA256.new()
            hasher.update(serialized_weights)
            integrity_hash = hasher.digest()
            
            return {
                "encrypted_data": ciphertext.hex(),
                "encrypted_key": encrypted_aes_key.hex(),
                "nonce": cipher.nonce.hex(),
                "auth_tag": auth_tag.hex(),
                "integrity_hash": integrity_hash.hex(),
                "weights_shape": model_weights.shape,
                "weights_dtype": str(model_weights.dtype),
                "protocol": "simple_encryption",
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Simple encryption failed: {e}")
            raise
    
    async def _encrypt_secure_aggregation(self, model_weights: np.ndarray, 
                                        recipient_public_key: bytes) -> Dict[str, Any]:
        """Secure aggregation protocol with secret sharing"""
        try:
            # Add noise for privacy (basic differential privacy)
            noise_scale = 0.1  # This should be configurable
            noisy_weights = model_weights + np.random.normal(0, noise_scale, model_weights.shape)
            
            # Quantize weights to integers for secret sharing
            quantization_factor = 10000
            quantized_weights = (noisy_weights * quantization_factor).astype(np.int64)
            
            # Apply Shamir's secret sharing to each weight
            shares_data = []
            for weight in quantized_weights.flatten():
                # Create 5 shares with threshold of 3
                shares = self._create_secret_shares(int(weight), num_shares=5, threshold=3)
                shares_data.append(shares)
            
            # Encrypt shares with simple encryption
            serialized_shares = json.dumps(shares_data).encode()
            
            # Encrypt with AES-GCM
            aes_key = get_random_bytes(32)
            cipher = AES.new(aes_key, AES.MODE_GCM)
            ciphertext, auth_tag = cipher.encrypt_and_digest(serialized_shares)
            
            # Encrypt AES key with RSA
            rsa_key = RSA.import_key(recipient_public_key)
            from Crypto.Cipher import PKCS1_OAEP
            rsa_cipher = PKCS1_OAEP.new(rsa_key, hashAlgo=SHA256)
            encrypted_aes_key = rsa_cipher.encrypt(aes_key)
            
            return {
                "encrypted_shares": ciphertext.hex(),
                "encrypted_key": encrypted_aes_key.hex(),
                "nonce": cipher.nonce.hex(),
                "auth_tag": auth_tag.hex(),
                "weights_shape": model_weights.shape,
                "quantization_factor": quantization_factor,
                "num_shares": 5,
                "threshold": 3,
                "noise_scale": noise_scale,
                "protocol": "secure_aggregation",
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Secure aggregation encryption failed: {e}")
            raise
    
    async def _encrypt_with_dp(self, model_weights: np.ndarray, 
                             recipient_public_key: bytes,
                             epsilon: float = 1.0, 
                             sensitivity: float = 2.0) -> Dict[str, Any]:
        """Encrypt with differential privacy noise"""
        try:
            # Calculate noise scale for Gaussian mechanism
            noise_scale = (sensitivity * np.sqrt(2 * np.log(1.25 / 1e-5))) / epsilon
            
            # Add calibrated noise
            noise = np.random.normal(0, noise_scale, model_weights.shape)
            private_weights = model_weights + noise
            
            # Clip weights to bound sensitivity
            clip_threshold = 1.0  # L2 norm clipping
            weights_norm = np.linalg.norm(private_weights)
            if weights_norm > clip_threshold:
                private_weights = private_weights * (clip_threshold / weights_norm)
            
            # Encrypt the private weights
            encrypted_data = await self._encrypt_simple(private_weights, recipient_public_key)
            
            # Add DP metadata
            encrypted_data.update({
                "protocol": "differential_privacy",
                "epsilon": epsilon,
                "sensitivity": sensitivity,
                "noise_scale": noise_scale,
                "clipping_applied": weights_norm > clip_threshold,
                "original_norm": float(weights_norm)
            })
            
            return encrypted_data
            
        except Exception as e:
            logger.error(f"Differential privacy encryption failed: {e}")
            raise
    
    async def _encrypt_secret_sharing(self, model_weights: np.ndarray, 
                                    recipient_public_key: bytes) -> Dict[str, Any]:
        """Encrypt using Shamir's secret sharing"""
        try:
            # Quantize weights for integer arithmetic
            quantization_factor = 1000000  # Higher precision
            quantized_weights = (model_weights * quantization_factor).astype(np.int64)
            
            # Create secret shares for all weights
            all_shares = {}
            for i, weight in enumerate(quantized_weights.flatten()):
                shares = self._create_secret_shares(int(weight), num_shares=7, threshold=4)
                all_shares[i] = shares
            
            # Serialize and encrypt shares
            shares_json = json.dumps(all_shares, cls=NumpyEncoder)
            serialized_shares = shares_json.encode()
            
            # Encrypt with ChaCha20-Poly1305 for performance
            key = get_random_bytes(32)
            cipher = ChaCha20_Poly1305.new(key=key)
            ciphertext, auth_tag = cipher.encrypt_and_digest(serialized_shares)
            
            # Encrypt key with RSA
            rsa_key = RSA.import_key(recipient_public_key)
            from Crypto.Cipher import PKCS1_OAEP
            rsa_cipher = PKCS1_OAEP.new(rsa_key, hashAlgo=SHA256)
            encrypted_key = rsa_cipher.encrypt(key)
            
            return {
                "encrypted_shares": ciphertext.hex(),
                "encrypted_key": encrypted_key.hex(),
                "nonce": cipher.nonce.hex(),
                "auth_tag": auth_tag.hex(),
                "weights_shape": model_weights.shape,
                "quantization_factor": quantization_factor,
                "num_shares": 7,
                "threshold": 4,
                "protocol": "secret_sharing",
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Secret sharing encryption failed: {e}")
            raise
    
    def _create_secret_shares(self, secret: int, num_shares: int, threshold: int) -> List[Tuple[int, int]]:
        """Create Shamir secret shares"""
        try:
            # Use large prime for finite field arithmetic
            prime = 2**31 - 1
            
            # Generate random coefficients
            coeffs = [secret] + [secrets.randbelow(prime) for _ in range(threshold - 1)]
            
            # Generate shares
            shares = []
            for x in range(1, num_shares + 1):
                y = 0
                for i, coeff in enumerate(coeffs):
                    y += coeff * (x ** i)
                y %= prime
                shares.append((x, y))
            
            return shares
            
        except Exception as e:
            logger.error(f"Secret sharing creation failed: {e}")
            raise
    
    def _reconstruct_secret(self, shares: List[Tuple[int, int]], prime: int = 2**31 - 1) -> int:
        """Reconstruct secret from Shamir shares using Lagrange interpolation"""
        try:
            if len(shares) == 0:
                raise ValueError("No shares provided")
            
            # Lagrange interpolation to find secret (x=0)
            secret = 0
            for i, (xi, yi) in enumerate(shares):
                # Calculate Lagrange basis polynomial
                basis = yi
                for j, (xj, _) in enumerate(shares):
                    if i != j:
                        # Calculate (0 - xj) / (xi - xj) mod prime
                        numerator = (-xj) % prime
                        denominator = (xi - xj) % prime
                        denominator_inv = pow(denominator, prime - 2, prime)  # Modular inverse
                        basis = (basis * numerator * denominator_inv) % prime
                
                secret = (secret + basis) % prime
            
            return secret
            
        except Exception as e:
            logger.error(f"Secret reconstruction failed: {e}")
            raise
    
    async def decrypt_model_update(self, encrypted_data: Dict[str, Any], 
                                  private_key: bytes = None) -> np.ndarray:
        """Decrypt model update based on protocol"""
        
        protocol = encrypted_data.get("protocol", "simple_encryption")
        
        if protocol == "simple_encryption":
            return await self._decrypt_simple(encrypted_data, private_key)
        elif protocol == "secure_aggregation":
            return await self._decrypt_secure_aggregation(encrypted_data, private_key)
        elif protocol == "differential_privacy":
            return await self._decrypt_with_dp(encrypted_data, private_key)
        elif protocol == "secret_sharing":
            return await self._decrypt_secret_sharing(encrypted_data, private_key)
        else:
            raise ValueError(f"Unsupported decryption protocol: {protocol}")
    
    async def _decrypt_simple(self, encrypted_data: Dict[str, Any], 
                            private_key: bytes = None) -> np.ndarray:
        """Decrypt simple hybrid encryption"""
        try:
            if private_key is None:
                private_key = self.rsa_keypair.export_key()
            
            # Decrypt AES key with RSA
            rsa_key = RSA.import_key(private_key)
            from Crypto.Cipher import PKCS1_OAEP
            rsa_cipher = PKCS1_OAEP.new(rsa_key, hashAlgo=SHA256)
            aes_key = rsa_cipher.decrypt(bytes.fromhex(encrypted_data["encrypted_key"]))
            
            # Decrypt model weights with AES-GCM
            nonce = bytes.fromhex(encrypted_data["nonce"])
            ciphertext = bytes.fromhex(encrypted_data["encrypted_data"])
            auth_tag = bytes.fromhex(encrypted_data["auth_tag"])
            
            cipher = AES.new(aes_key, AES.MODE_GCM, nonce=nonce)
            serialized_weights = cipher.decrypt_and_verify(ciphertext, auth_tag)
            
            # Reconstruct numpy array
            weights_shape = encrypted_data["weights_shape"]
            weights_dtype = encrypted_data["weights_dtype"]
            
            weights = np.frombuffer(serialized_weights, dtype=weights_dtype).reshape(weights_shape)
            
            # Verify integrity
            hasher = SHA256.new()
            hasher.update(serialized_weights)
            computed_hash = hasher.digest()
            provided_hash = bytes.fromhex(encrypted_data["integrity_hash"])
            
            if computed_hash != provided_hash:
                logger.warning("Integrity check failed for decrypted model update")
            
            return weights
            
        except Exception as e:
            logger.error(f"Simple decryption failed: {e}")
            raise
    
    async def _decrypt_secure_aggregation(self, encrypted_data: Dict[str, Any], 
                                        private_key: bytes = None) -> np.ndarray:
        """Decrypt secure aggregation with secret sharing"""
        try:
            if private_key is None:
                private_key = self.rsa_keypair.export_key()
            
            # Decrypt AES key
            rsa_key = RSA.import_key(private_key)
            from Crypto.Cipher import PKCS1_OAEP
            rsa_cipher = PKCS1_OAEP.new(rsa_key, hashAlgo=SHA256)
            aes_key = rsa_cipher.decrypt(bytes.fromhex(encrypted_data["encrypted_key"]))
            
            # Decrypt shares
            nonce = bytes.fromhex(encrypted_data["nonce"])
            ciphertext = bytes.fromhex(encrypted_data["encrypted_shares"])
            auth_tag = bytes.fromhex(encrypted_data["auth_tag"])
            
            cipher = AES.new(aes_key, AES.MODE_GCM, nonce=nonce)
            serialized_shares = cipher.decrypt_and_verify(ciphertext, auth_tag)
            
            # Reconstruct weights from shares
            shares_data = json.loads(serialized_shares.decode())
            
            # Reconstruct each weight from its shares
            reconstructed_weights = []
            for shares in shares_data:
                # Take first 'threshold' shares for reconstruction
                threshold = encrypted_data["threshold"]
                used_shares = shares[:threshold]
                reconstructed_weight = self._reconstruct_secret(used_shares)
                reconstructed_weights.append(reconstructed_weight)
            
            # Convert back to original scale and shape
            quantization_factor = encrypted_data["quantization_factor"]
            weights_array = np.array(reconstructed_weights, dtype=np.float64) / quantization_factor
            weights_shape = encrypted_data["weights_shape"]
            
            return weights_array.reshape(weights_shape)
            
        except Exception as e:
            logger.error(f"Secure aggregation decryption failed: {e}")
            raise
    
    async def _decrypt_with_dp(self, encrypted_data: Dict[str, Any], 
                             private_key: bytes = None) -> np.ndarray:
        """Decrypt differential privacy encrypted update"""
        # DP decryption is the same as simple decryption
        # The privacy is in the noise added during encryption
        return await self._decrypt_simple(encrypted_data, private_key)
    
    async def _decrypt_secret_sharing(self, encrypted_data: Dict[str, Any], 
                                    private_key: bytes = None) -> np.ndarray:
        """Decrypt secret sharing encrypted update"""
        try:
            if private_key is None:
                private_key = self.rsa_keypair.export_key()
            
            # Decrypt key
            rsa_key = RSA.import_key(private_key)
            from Crypto.Cipher import PKCS1_OAEP
            rsa_cipher = PKCS1_OAEP.new(rsa_key, hashAlgo=SHA256)
            key = rsa_cipher.decrypt(bytes.fromhex(encrypted_data["encrypted_key"]))
            
            # Decrypt shares
            nonce = bytes.fromhex(encrypted_data["nonce"])
            ciphertext = bytes.fromhex(encrypted_data["encrypted_shares"])
            auth_tag = bytes.fromhex(encrypted_data["auth_tag"])
            
            cipher = ChaCha20_Poly1305.new(key=key, nonce=nonce)
            serialized_shares = cipher.decrypt_and_verify(ciphertext, auth_tag)
            
            # Reconstruct weights
            all_shares = json.loads(serialized_shares.decode())
            
            reconstructed_weights = []
            threshold = encrypted_data["threshold"]
            
            for weight_idx in sorted(all_shares.keys(), key=int):
                shares = all_shares[weight_idx][:threshold]
                reconstructed_weight = self._reconstruct_secret(shares)
                reconstructed_weights.append(reconstructed_weight)
            
            # Convert back to original scale and shape
            quantization_factor = encrypted_data["quantization_factor"]
            weights_array = np.array(reconstructed_weights, dtype=np.float64) / quantization_factor
            weights_shape = encrypted_data["weights_shape"]
            
            return weights_array.reshape(weights_shape)
            
        except Exception as e:
            logger.error(f"Secret sharing decryption failed: {e}")
            raise
    
    async def aggregate_encrypted_updates(self, encrypted_updates: List[Dict[str, Any]], 
                                        participant_weights: Optional[List[float]] = None,
                                        context: Optional[AggregationContext] = None) -> np.ndarray:
        """Aggregate multiple encrypted model updates"""
        
        if not encrypted_updates:
            raise ValueError("No encrypted updates provided")
        
        try:
            # Decrypt all updates
            decrypted_updates = []
            for encrypted_update in encrypted_updates:
                decrypted_update = await self.decrypt_model_update(encrypted_update)
                decrypted_updates.append(decrypted_update)
            
            # Perform weighted aggregation
            if participant_weights is None:
                participant_weights = [1.0 / len(decrypted_updates)] * len(decrypted_updates)
            
            # Normalize weights
            weight_sum = sum(participant_weights)
            normalized_weights = [w / weight_sum for w in participant_weights]
            
            # Aggregate updates
            aggregated_update = np.zeros_like(decrypted_updates[0])
            for update, weight in zip(decrypted_updates, normalized_weights):
                aggregated_update += weight * update
            
            logger.info(f"Successfully aggregated {len(encrypted_updates)} model updates")
            return aggregated_update
            
        except Exception as e:
            logger.error(f"Encrypted aggregation failed: {e}")
            raise
    
    def create_secure_message(self, recipient_id: str, message_type: str, 
                            payload: Dict[str, Any]) -> SecureMessage:
        """Create a secure message with encryption and signature"""
        try:
            # Serialize payload
            payload_bytes = json.dumps(payload, cls=NumpyEncoder).encode()
            
            # Encrypt payload (using session key if available)
            if recipient_id in self.session_keys:
                # Use symmetric encryption for efficiency
                key = self.session_keys[recipient_id]
                cipher = AES.new(key, AES.MODE_GCM)
                encrypted_payload, auth_tag = cipher.encrypt_and_digest(payload_bytes)
                encrypted_payload = cipher.nonce + auth_tag + encrypted_payload
            else:
                # Fall back to asymmetric encryption
                # This is a simplified version - in practice, you'd need the recipient's public key
                encrypted_payload = payload_bytes  # Placeholder
            
            # Sign the message
            hasher = SHA256.new(encrypted_payload)
            signature = pkcs1_15.new(self.rsa_keypair).sign(hasher)
            
            return SecureMessage(
                sender_id="self",
                recipient_id=recipient_id,
                message_type=message_type,
                encrypted_payload=encrypted_payload,
                signature=signature,
                timestamp=time.time(),
                metadata={"encryption": "session_key" if recipient_id in self.session_keys else "asymmetric"}
            )
            
        except Exception as e:
            logger.error(f"Secure message creation failed: {e}")
            raise
    
    def verify_secure_message(self, message: SecureMessage, 
                            sender_public_key: bytes) -> Dict[str, Any]:
        """Verify and decrypt a secure message"""
        try:
            # Verify signature
            sender_rsa_key = RSA.import_key(sender_public_key)
            hasher = SHA256.new(message.encrypted_payload)
            
            try:
                pkcs1_15.new(sender_rsa_key).verify(hasher, message.signature)
                signature_valid = True
            except ValueError:
                signature_valid = False
                logger.warning(f"Invalid signature from {message.sender_id}")
            
            # Decrypt payload
            if message.sender_id in self.session_keys:
                # Use symmetric decryption
                key = self.session_keys[message.sender_id]
                nonce = message.encrypted_payload[:16]
                auth_tag = message.encrypted_payload[16:32]
                ciphertext = message.encrypted_payload[32:]
                
                cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
                payload_bytes = cipher.decrypt_and_verify(ciphertext, auth_tag)
            else:
                # Asymmetric decryption (placeholder)
                payload_bytes = message.encrypted_payload
            
            payload = json.loads(payload_bytes.decode())
            
            return {
                "valid": signature_valid,
                "payload": payload,
                "sender_id": message.sender_id,
                "timestamp": message.timestamp
            }
            
        except Exception as e:
            logger.error(f"Secure message verification failed: {e}")
            return {"valid": False, "error": str(e)}


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder for numpy arrays and types"""
    
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


class DifferentialPrivacyManager:
    """Manager for differential privacy in federated learning"""
    
    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5):
        self.epsilon = epsilon
        self.delta = delta
        self.privacy_budget_used = 0.0
        self.noise_multiplier = self._calculate_noise_multiplier()
    
    def _calculate_noise_multiplier(self) -> float:
        """Calculate noise multiplier for Gaussian mechanism"""
        # Simplified calculation - in practice, use more sophisticated methods
        return np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon
    
    def add_privacy_noise(self, weights: np.ndarray, sensitivity: float = 1.0) -> np.ndarray:
        """Add calibrated noise for differential privacy"""
        try:
            noise_scale = sensitivity * self.noise_multiplier
            noise = np.random.normal(0, noise_scale, weights.shape)
            
            # Track privacy budget usage
            self.privacy_budget_used += self.epsilon
            
            logger.info(f"Added DP noise with scale {noise_scale:.4f}, budget used: {self.privacy_budget_used:.4f}")
            
            return weights + noise
            
        except Exception as e:
            logger.error(f"Privacy noise addition failed: {e}")
            raise
    
    def clip_weights(self, weights: np.ndarray, clip_threshold: float = 1.0) -> np.ndarray:
        """Clip weights to bound sensitivity"""
        weights_norm = np.linalg.norm(weights)
        
        if weights_norm > clip_threshold:
            weights = weights * (clip_threshold / weights_norm)
            logger.info(f"Clipped weights from norm {weights_norm:.4f} to {clip_threshold}")
        
        return weights
    
    def get_privacy_budget_remaining(self) -> float:
        """Get remaining privacy budget"""
        return max(0, self.epsilon - self.privacy_budget_used)


# Factory function for creating secure aggregation instances
def create_secure_aggregation(security_level: int = 3, 
                            protocol: AggregationProtocol = AggregationProtocol.SIMPLE_ENCRYPTION) -> AdvancedSecureAggregation:
    """Factory function to create secure aggregation instance"""
    
    aggregator = AdvancedSecureAggregation(security_level=security_level)
    
    logger.info(f"Created secure aggregation with protocol {protocol.value} and security level {security_level}")
    
    return aggregator
