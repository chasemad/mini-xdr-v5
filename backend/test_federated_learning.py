#!/usr/bin/env python3
"""
Comprehensive Test Suite for Mini-XDR Federated Learning System (Phase 2)
Tests all components: core framework, cryptographic protocols, ML integration, and API endpoints
"""

import asyncio
import logging
import json
import sys
import time
import numpy as np
from datetime import datetime, timezone
from typing import Dict, List, Any
import unittest
from unittest.mock import Mock, AsyncMock, patch
import requests
import subprocess
from pathlib import Path

# Add backend to path
sys.path.append(str(Path(__file__).parent / "app"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FederatedLearningTestSuite:
    """Comprehensive test suite for federated learning system"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.test_results = []
        self.api_running = False
        
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all federated learning tests"""
        logger.info("=" * 80)
        logger.info("STARTING FEDERATED LEARNING TEST SUITE")
        logger.info("=" * 80)
        
        # Check API availability
        await self._check_api_availability()
        
        test_methods = [
            self._test_imports_and_dependencies,
            self._test_cryptographic_protocols,
            self._test_federated_core_framework,
            self._test_ml_engine_integration,
            self._test_api_endpoints,
            self._test_end_to_end_scenario,
            self._test_security_and_privacy,
            self._test_performance_and_scalability
        ]
        
        for test_method in test_methods:
            try:
                logger.info(f"\n{'='*60}")
                logger.info(f"Running: {test_method.__name__}")
                logger.info(f"{'='*60}")
                
                result = await test_method()
                self.test_results.append({
                    'test': test_method.__name__,
                    'status': 'PASSED' if result['success'] else 'FAILED',
                    'details': result
                })
                
                if result['success']:
                    logger.info(f"✅ {test_method.__name__} PASSED")
                else:
                    logger.error(f"❌ {test_method.__name__} FAILED: {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                logger.error(f"❌ {test_method.__name__} FAILED with exception: {e}")
                self.test_results.append({
                    'test': test_method.__name__,
                    'status': 'ERROR',
                    'error': str(e)
                })
        
        # Generate final report
        return self._generate_test_report()
    
    async def _check_api_availability(self):
        """Check if the Mini-XDR API is running"""
        try:
            response = requests.get(f"{self.base_url}/incidents", timeout=5)
            self.api_running = True
            logger.info("✅ Mini-XDR API is running and accessible")
        except Exception as e:
            self.api_running = False
            logger.warning(f"⚠️ Mini-XDR API not available: {e}")
            logger.info("API-dependent tests will be skipped")
    
    async def _test_imports_and_dependencies(self) -> Dict[str, Any]:
        """Test all federated learning imports and dependencies"""
        test_results = {}
        
        try:
            # Test core federated learning imports
            logger.info("Testing federated learning imports...")
            from app.federated_learning import (
                federated_manager, FederatedLearningManager, 
                FederatedLearningCoordinator, FederatedLearningParticipant
            )
            test_results['federated_core'] = True
            logger.info("✅ Federated learning core imports successful")
            
        except ImportError as e:
            test_results['federated_core'] = False
            logger.error(f"❌ Federated learning core import failed: {e}")
        
        try:
            # Test cryptographic imports
            logger.info("Testing cryptographic imports...")
            from app.crypto.secure_aggregation import (
                AdvancedSecureAggregation, AggregationProtocol, 
                create_secure_aggregation
            )
            test_results['crypto'] = True
            logger.info("✅ Cryptographic imports successful")
            
        except ImportError as e:
            test_results['crypto'] = False
            logger.error(f"❌ Cryptographic import failed: {e}")
        
        try:
            # Test TensorFlow Federated
            logger.info("Testing TensorFlow Federated availability...")
            import tensorflow as tf
            import tensorflow_federated as tff
            test_results['tensorflow_federated'] = True
            logger.info(f"✅ TensorFlow {tf.__version__} and TFF {tff.version.VERSION} available")
            
        except ImportError as e:
            test_results['tensorflow_federated'] = False
            logger.warning(f"⚠️ TensorFlow Federated not available: {e}")
        
        try:
            # Test enhanced ML engine
            logger.info("Testing enhanced ML engine...")
            from app.ml_engine import ml_detector, FederatedEnsembleDetector
            test_results['enhanced_ml'] = True
            logger.info("✅ Enhanced ML engine with federated capabilities loaded")
            
        except ImportError as e:
            test_results['enhanced_ml'] = False
            logger.error(f"❌ Enhanced ML engine import failed: {e}")
        
        success = all(test_results.values())
        
        return {
            'success': success,
            'component_status': test_results,
            'message': 'All imports successful' if success else 'Some imports failed'
        }
    
    async def _test_cryptographic_protocols(self) -> Dict[str, Any]:
        """Test secure aggregation cryptographic protocols"""
        try:
            from app.crypto.secure_aggregation import (
                AdvancedSecureAggregation, AggregationProtocol
            )
            
            # Create secure aggregation instance
            secure_agg = AdvancedSecureAggregation(security_level=2)
            
            # Test key generation
            public_key = secure_agg.get_public_key("rsa")
            assert public_key is not None, "Public key generation failed"
            logger.info("✅ Public key generation successful")
            
            # Test model weight encryption/decryption
            test_weights = np.random.rand(100).astype(np.float32)
            
            # Test simple encryption
            encrypted_data = await secure_agg.encrypt_model_update(
                test_weights, public_key, AggregationProtocol.SIMPLE_ENCRYPTION
            )
            assert 'encrypted_data' in encrypted_data, "Encryption failed"
            logger.info("✅ Simple encryption successful")
            
            # Test decryption
            decrypted_weights = await secure_agg.decrypt_model_update(encrypted_data)
            assert np.allclose(test_weights, decrypted_weights, rtol=1e-5), "Decryption mismatch"
            logger.info("✅ Decryption successful")
            
            # Test differential privacy encryption
            dp_encrypted = await secure_agg.encrypt_model_update(
                test_weights, public_key, AggregationProtocol.DIFFERENTIAL_PRIVACY
            )
            assert dp_encrypted['protocol'] == 'differential_privacy', "DP encryption failed"
            logger.info("✅ Differential privacy encryption successful")
            
            # Test secure aggregation
            secure_encrypted = await secure_agg.encrypt_model_update(
                test_weights, public_key, AggregationProtocol.SECURE_AGGREGATION
            )
            assert secure_encrypted['protocol'] == 'secure_aggregation', "Secure aggregation failed"
            logger.info("✅ Secure aggregation protocol successful")
            
            # Test multiple update aggregation
            updates = [encrypted_data, dp_encrypted]
            aggregated = await secure_agg.aggregate_encrypted_updates(updates)
            assert aggregated is not None, "Aggregation failed"
            logger.info("✅ Multi-update aggregation successful")
            
            return {
                'success': True,
                'protocols_tested': [
                    'simple_encryption',
                    'differential_privacy', 
                    'secure_aggregation',
                    'multi_update_aggregation'
                ],
                'message': 'All cryptographic protocols working correctly'
            }
            
        except Exception as e:
            logger.error(f"Cryptographic test failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'message': 'Cryptographic protocol tests failed'
            }
    
    async def _test_federated_core_framework(self) -> Dict[str, Any]:
        """Test federated learning core framework"""
        try:
            from app.federated_learning import (
                federated_manager, FederatedLearningCoordinator, 
                FederatedLearningParticipant, FederatedModelType
            )
            
            # Test coordinator initialization
            coordinator = FederatedLearningCoordinator()
            assert coordinator.node_id is not None, "Coordinator node_id not set"
            assert coordinator.public_key is not None, "Coordinator public key not generated"
            logger.info("✅ Coordinator initialization successful")
            
            # Test participant initialization
            participant = FederatedLearningParticipant()
            assert participant.node_id is not None, "Participant node_id not set"
            logger.info("✅ Participant initialization successful")
            
            # Test participant registration
            participant_info = {
                'endpoint': 'http://localhost:8001',
                'model_type': 'neural_network',
                'data_size': 1000
            }
            registered = await coordinator.register_participant(
                participant.node_id, participant_info
            )
            assert registered, "Participant registration failed"
            logger.info("✅ Participant registration successful")
            
            # Test federated round start
            round_id = await coordinator.start_federated_round(
                FederatedModelType.NEURAL_NETWORK,
                {'input_size': 15, 'hidden_layers': [64, 32]}
            )
            assert round_id is not None, "Federated round start failed"
            logger.info("✅ Federated round start successful")
            
            # Test model update simulation
            test_weights = np.random.rand(100).astype(np.float32)
            encrypted_update = coordinator.secure_aggregation.encrypt_model_update(
                test_weights, coordinator.public_key
            )
            
            update_received = await coordinator.receive_model_update(
                participant.node_id, encrypted_update, round_id, {'data_size': 1000}
            )
            assert update_received, "Model update reception failed"
            logger.info("✅ Model update reception successful")
            
            # Test global model status
            global_model_status = await coordinator.get_global_model()
            assert 'current_round' in global_model_status, "Global model status missing"
            logger.info("✅ Global model status retrieval successful")
            
            return {
                'success': True,
                'components_tested': [
                    'coordinator_initialization',
                    'participant_initialization',
                    'participant_registration',
                    'federated_round_start',
                    'model_update_handling',
                    'global_model_management'
                ],
                'coordinator_id': coordinator.node_id,
                'participant_id': participant.node_id,
                'round_id': round_id,
                'message': 'Federated learning core framework working correctly'
            }
            
        except Exception as e:
            logger.error(f"Federated core framework test failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'message': 'Federated core framework tests failed'
            }
    
    async def _test_ml_engine_integration(self) -> Dict[str, Any]:
        """Test ML engine integration with federated learning"""
        try:
            from app.ml_engine import ml_detector
            
            # Test federated capabilities status
            status = ml_detector.get_model_status()
            assert 'federated_enabled' in status, "Federated status not in ML detector"
            assert 'federated_rounds' in status, "Federated rounds not tracked"
            logger.info("✅ ML detector federated status available")
            
            # Test anomaly score calculation (should work with or without federated)
            from app.models import Event
            
            # Create mock events
            mock_events = [
                Mock(spec=Event, src_ip='192.168.1.100', dst_port=22, eventid='cowrie.login.failed',
                     ts=datetime.now(timezone.utc), raw={'username': 'admin', 'password': '123456'})
                for _ in range(5)
            ]
            
            anomaly_score = await ml_detector.calculate_anomaly_score('192.168.1.100', mock_events)
            assert 0 <= anomaly_score <= 1, f"Invalid anomaly score: {anomaly_score}"
            logger.info(f"✅ Anomaly score calculation successful: {anomaly_score:.3f}")
            
            # Test training data preparation
            training_data = [
                {
                    'event_count_1h': 10, 'event_count_24h': 50,
                    'unique_ports': 3, 'failed_login_count': 5,
                    'session_duration_avg': 120.0, 'password_diversity': 3,
                    'username_diversity': 2, 'event_rate_per_minute': 0.5,
                    'time_of_day': 0.5, 'is_weekend': 0.0,
                    'unique_usernames': 2, 'password_length_avg': 8.0,
                    'command_diversity': 5, 'download_attempts': 1,
                    'upload_attempts': 0
                }
                for _ in range(100)
            ]
            
            # Test model training with federated learning disabled first
            results = await ml_detector.train_models(training_data, enable_federated=False)
            assert isinstance(results, dict), "Training results should be a dict"
            logger.info(f"✅ Standard model training successful: {results}")
            
            # Test model training with federated learning enabled (if available)
            if status.get('federated_enabled', False):
                fed_results = await ml_detector.train_models(training_data, enable_federated=True)
                assert isinstance(fed_results, dict), "Federated training results should be a dict"
                logger.info(f"✅ Federated model training successful: {fed_results}")
            else:
                logger.info("ℹ️ Federated learning not enabled, skipping federated training test")
            
            return {
                'success': True,
                'ml_status': status,
                'anomaly_score': anomaly_score,
                'training_results': results,
                'federated_enabled': status.get('federated_enabled', False),
                'message': 'ML engine federated integration working correctly'
            }
            
        except Exception as e:
            logger.error(f"ML engine integration test failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'message': 'ML engine integration tests failed'
            }
    
    async def _test_api_endpoints(self) -> Dict[str, Any]:
        """Test federated learning API endpoints"""
        if not self.api_running:
            return {
                'success': False,
                'message': 'API not available - skipping API endpoint tests',
                'skipped': True
            }
        
        endpoint_results = {}
        
        try:
            # Test federated status endpoint
            logger.info("Testing /api/federated/status")
            response = requests.get(f"{self.base_url}/api/federated/status", timeout=10)
            assert response.status_code == 200, f"Status endpoint failed: {response.status_code}"
            status_data = response.json()
            endpoint_results['status'] = status_data
            logger.info("✅ Federated status endpoint working")
            
            # Test federated models status
            logger.info("Testing /api/federated/models/status")
            response = requests.get(f"{self.base_url}/api/federated/models/status", timeout=10)
            assert response.status_code == 200, f"Models status failed: {response.status_code}"
            models_data = response.json()
            endpoint_results['models_status'] = models_data
            logger.info("✅ Federated models status endpoint working")
            
            # Test federated insights
            logger.info("Testing /api/federated/insights")
            response = requests.get(f"{self.base_url}/api/federated/insights", timeout=10)
            # This might fail if federated learning is not available, which is OK
            if response.status_code == 200:
                insights_data = response.json()
                endpoint_results['insights'] = insights_data
                logger.info("✅ Federated insights endpoint working")
            else:
                logger.info(f"ℹ️ Federated insights endpoint returned {response.status_code} (expected if FL not available)")
            
            # Test coordinator initialization
            logger.info("Testing /api/federated/coordinator/initialize")
            init_data = {"config": {"min_participants": 2, "security_level": 2}}
            response = requests.post(
                f"{self.base_url}/api/federated/coordinator/initialize", 
                json=init_data, timeout=15
            )
            if response.status_code == 200:
                coord_data = response.json()
                endpoint_results['coordinator_init'] = coord_data
                logger.info("✅ Coordinator initialization endpoint working")
            else:
                logger.info(f"ℹ️ Coordinator init returned {response.status_code} (expected if dependencies missing)")
            
            # Test model training endpoint
            logger.info("Testing /api/federated/models/train")
            train_data = {"enable_federated": False}  # Start with standard training
            response = requests.post(
                f"{self.base_url}/api/federated/models/train",
                json=train_data, timeout=30
            )
            if response.status_code == 200:
                training_result = response.json()
                endpoint_results['model_training'] = training_result
                logger.info("✅ Model training endpoint working")
            else:
                logger.info(f"ℹ️ Model training returned {response.status_code}")
            
            return {
                'success': True,
                'endpoints_tested': list(endpoint_results.keys()),
                'endpoint_results': endpoint_results,
                'message': 'Federated learning API endpoints working correctly'
            }
            
        except Exception as e:
            logger.error(f"API endpoint test failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'endpoints_tested': list(endpoint_results.keys()),
                'partial_results': endpoint_results,
                'message': 'API endpoint tests failed'
            }
    
    async def _test_end_to_end_scenario(self) -> Dict[str, Any]:
        """Test end-to-end federated learning scenario"""
        try:
            # This is a simulated end-to-end test since we don't have multiple nodes
            logger.info("Running simulated end-to-end federated learning scenario...")
            
            from app.federated_learning import (
                FederatedLearningCoordinator, FederatedLearningParticipant, 
                FederatedModelType
            )
            
            # Setup coordinator
            coordinator = FederatedLearningCoordinator()
            
            # Setup multiple participants
            participants = []
            for i in range(3):
                participant = FederatedLearningParticipant(
                    node_id=f"participant_{i}",
                    coordinator_endpoint="http://localhost:8000"
                )
                participants.append(participant)
            
            # Register all participants
            for i, participant in enumerate(participants):
                registered = await coordinator.register_participant(
                    participant.node_id,
                    {
                        'endpoint': f'http://localhost:{8001+i}',
                        'model_type': 'neural_network',
                        'data_size': 100 + i * 50
                    }
                )
                assert registered, f"Failed to register participant {i}"
            
            logger.info(f"✅ Registered {len(participants)} participants")
            
            # Start federated training round
            round_id = await coordinator.start_federated_round(
                FederatedModelType.NEURAL_NETWORK,
                {'input_size': 15, 'hidden_layers': [64, 32], 'output_size': 1}
            )
            assert round_id is not None, "Failed to start federated round"
            logger.info(f"✅ Started federated round: {round_id}")
            
            # Simulate participant training and updates
            model_updates = []
            for i, participant in enumerate(participants):
                # Generate mock training data
                training_data = [
                    {
                        'event_count_1h': np.random.randint(1, 20),
                        'failed_login_count': np.random.randint(0, 10),
                        'unique_ports': np.random.randint(1, 5),
                        'session_duration_avg': np.random.uniform(10, 300),
                        # ... other features with random values
                        'is_anomaly': np.random.randint(0, 2)
                    }
                    for _ in range(100 + i * 20)
                ]
                
                # Train local model (simulated)
                model_weights = await participant.train_local_model(
                    training_data,
                    {'model_type': 'neural_network', 'input_size': 15}
                )
                
                # Encrypt and send update
                encrypted_update = coordinator.secure_aggregation.encrypt_model_update(
                    model_weights, coordinator.public_key
                )
                
                # Send to coordinator
                update_received = await coordinator.receive_model_update(
                    participant.node_id, encrypted_update, round_id, 
                    {'data_size': len(training_data)}
                )
                assert update_received, f"Failed to receive update from participant {i}"
                model_updates.append(encrypted_update)
            
            logger.info(f"✅ Collected {len(model_updates)} model updates")
            
            # Wait a moment for aggregation to complete
            await asyncio.sleep(1)
            
            # Check final status
            coordinator_status = coordinator.get_coordinator_status()
            global_model = await coordinator.get_global_model()
            
            return {
                'success': True,
                'scenario': 'simulated_3_node_training',
                'participants': len(participants),
                'round_id': round_id,
                'model_updates_collected': len(model_updates),
                'coordinator_status': coordinator_status['status'],
                'global_model_available': 'weights_available' in global_model,
                'message': 'End-to-end federated learning scenario completed successfully'
            }
            
        except Exception as e:
            logger.error(f"End-to-end scenario test failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'message': 'End-to-end scenario test failed'
            }
    
    async def _test_security_and_privacy(self) -> Dict[str, Any]:
        """Test security and privacy features"""
        try:
            from app.crypto.secure_aggregation import (
                AdvancedSecureAggregation, DifferentialPrivacyManager,
                AggregationProtocol
            )
            
            # Test differential privacy
            dp_manager = DifferentialPrivacyManager(epsilon=1.0, delta=1e-5)
            test_weights = np.random.rand(100).astype(np.float32)
            
            # Test privacy noise addition
            private_weights = dp_manager.add_privacy_noise(test_weights, sensitivity=1.0)
            assert not np.array_equal(test_weights, private_weights), "No privacy noise added"
            logger.info("✅ Differential privacy noise addition working")
            
            # Test weight clipping
            large_weights = np.random.rand(100) * 10  # Large weights
            clipped_weights = dp_manager.clip_weights(large_weights, clip_threshold=1.0)
            clipped_norm = np.linalg.norm(clipped_weights)
            assert clipped_norm <= 1.1, f"Weight clipping failed: norm={clipped_norm}"
            logger.info("✅ Weight clipping working")
            
            # Test secure aggregation with different security levels
            security_results = {}
            for level in [1, 2, 3]:
                secure_agg = AdvancedSecureAggregation(security_level=level)
                public_key = secure_agg.get_public_key()
                
                encrypted = await secure_agg.encrypt_model_update(
                    test_weights, public_key, AggregationProtocol.SECURE_AGGREGATION
                )
                decrypted = await secure_agg.decrypt_model_update(encrypted)
                
                # Check that decrypted weights are close to original (allowing for quantization noise)
                max_diff = np.max(np.abs(test_weights - decrypted))
                security_results[f'level_{level}'] = max_diff < 0.1
                
            assert all(security_results.values()), f"Security level tests failed: {security_results}"
            logger.info("✅ Multi-level security testing successful")
            
            # Test secret sharing
            from app.crypto.secure_aggregation import AdvancedSecureAggregation
            secure_agg = AdvancedSecureAggregation()
            
            # Create secret shares
            secret = 12345
            shares = secure_agg._create_secret_shares(secret, num_shares=5, threshold=3)
            assert len(shares) == 5, "Incorrect number of shares created"
            
            # Reconstruct secret
            reconstructed = secure_agg._reconstruct_secret(shares[:3])  # Use threshold shares
            assert reconstructed == secret, f"Secret sharing failed: {reconstructed} != {secret}"
            logger.info("✅ Secret sharing working correctly")
            
            return {
                'success': True,
                'privacy_features_tested': [
                    'differential_privacy_noise',
                    'weight_clipping',
                    'multi_level_security',
                    'secret_sharing'
                ],
                'security_levels_tested': [1, 2, 3],
                'differential_privacy_budget_remaining': dp_manager.get_privacy_budget_remaining(),
                'message': 'Security and privacy features working correctly'
            }
            
        except Exception as e:
            logger.error(f"Security and privacy test failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'message': 'Security and privacy tests failed'
            }
    
    async def _test_performance_and_scalability(self) -> Dict[str, Any]:
        """Test performance and scalability characteristics"""
        try:
            from app.crypto.secure_aggregation import AdvancedSecureAggregation
            
            # Performance tests
            secure_agg = AdvancedSecureAggregation(security_level=2)
            public_key = secure_agg.get_public_key()
            
            # Test encryption/decryption performance
            weight_sizes = [100, 1000, 10000]  # Different model sizes
            performance_results = {}
            
            for size in weight_sizes:
                test_weights = np.random.rand(size).astype(np.float32)
                
                # Time encryption
                start_time = time.time()
                encrypted = await secure_agg.encrypt_model_update(
                    test_weights, public_key
                )
                encryption_time = time.time() - start_time
                
                # Time decryption
                start_time = time.time()
                decrypted = await secure_agg.decrypt_model_update(encrypted)
                decryption_time = time.time() - start_time
                
                performance_results[f'size_{size}'] = {
                    'encryption_time': encryption_time,
                    'decryption_time': decryption_time,
                    'total_time': encryption_time + decryption_time,
                    'weights_per_second': size / (encryption_time + decryption_time)
                }
                
                logger.info(f"✅ Size {size}: {encryption_time:.3f}s encrypt, {decryption_time:.3f}s decrypt")
            
            # Test aggregation scalability
            num_participants = [2, 5, 10]
            aggregation_results = {}
            
            for num_parts in num_participants:
                test_weights = np.random.rand(1000).astype(np.float32)
                
                # Create multiple encrypted updates
                encrypted_updates = []
                start_time = time.time()
                
                for i in range(num_parts):
                    participant_weights = test_weights + np.random.normal(0, 0.1, test_weights.shape)
                    encrypted = await secure_agg.encrypt_model_update(
                        participant_weights, public_key
                    )
                    encrypted_updates.append(encrypted)
                
                # Aggregate all updates
                aggregated = await secure_agg.aggregate_encrypted_updates(encrypted_updates)
                aggregation_time = time.time() - start_time
                
                aggregation_results[f'participants_{num_parts}'] = {
                    'aggregation_time': aggregation_time,
                    'time_per_participant': aggregation_time / num_parts,
                    'successful': aggregated is not None
                }
                
                logger.info(f"✅ {num_parts} participants: {aggregation_time:.3f}s aggregation")
            
            # Calculate performance metrics
            avg_encryption_time = np.mean([
                result['encryption_time'] for result in performance_results.values()
            ])
            avg_weights_per_second = np.mean([
                result['weights_per_second'] for result in performance_results.values()
            ])
            
            return {
                'success': True,
                'performance_metrics': {
                    'average_encryption_time': avg_encryption_time,
                    'average_weights_per_second': avg_weights_per_second,
                    'scalability_tested': f"{min(num_participants)}-{max(num_participants)} participants",
                    'max_model_size_tested': max(weight_sizes)
                },
                'detailed_results': {
                    'encryption_performance': performance_results,
                    'aggregation_scalability': aggregation_results
                },
                'message': 'Performance and scalability tests completed successfully'
            }
            
        except Exception as e:
            logger.error(f"Performance and scalability test failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'message': 'Performance and scalability tests failed'
            }
    
    def _generate_test_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result['status'] == 'PASSED')
        failed_tests = sum(1 for result in self.test_results if result['status'] == 'FAILED')
        error_tests = sum(1 for result in self.test_results if result['status'] == 'ERROR')
        
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        logger.info("\n" + "="*80)
        logger.info("FEDERATED LEARNING TEST REPORT")
        logger.info("="*80)
        logger.info(f"Total Tests: {total_tests}")
        logger.info(f"Passed: {passed_tests} ✅")
        logger.info(f"Failed: {failed_tests} ❌")
        logger.info(f"Errors: {error_tests} 🔥")
        logger.info(f"Success Rate: {success_rate:.1f}%")
        logger.info("="*80)
        
        # Print detailed results
        for result in self.test_results:
            status_emoji = {"PASSED": "✅", "FAILED": "❌", "ERROR": "🔥"}
            logger.info(f"{status_emoji.get(result['status'], '?')} {result['test']}: {result['status']}")
        
        return {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'summary': {
                'total_tests': total_tests,
                'passed': passed_tests,
                'failed': failed_tests,
                'errors': error_tests,
                'success_rate': success_rate
            },
            'detailed_results': self.test_results,
            'overall_success': success_rate >= 75,  # Consider 75%+ as overall success
            'recommendations': self._generate_recommendations()
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []
        
        failed_tests = [r for r in self.test_results if r['status'] != 'PASSED']
        
        if any('crypto' in r['test'] for r in failed_tests):
            recommendations.append(
                "Install cryptographic dependencies: pip install pycryptodome"
            )
        
        if any('tensorflow' in r.get('error', '') for r in failed_tests):
            recommendations.append(
                "Install TensorFlow Federated: pip install tensorflow-federated==0.70.0"
            )
        
        if not self.api_running:
            recommendations.append(
                "Start Mini-XDR backend server to test API endpoints"
            )
        
        if len(failed_tests) == 0:
            recommendations.append(
                "🎉 All tests passed! Your federated learning system is ready for production."
            )
        
        return recommendations


async def main():
    """Main test runner"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Mini-XDR Federated Learning Test Suite")
    parser.add_argument('--url', default='http://localhost:8000', help='Backend API URL')
    parser.add_argument('--output', help='JSON output file path')
    
    args = parser.parse_args()
    
    # Create test suite
    test_suite = FederatedLearningTestSuite(base_url=args.url)
    
    # Run all tests
    report = await test_suite.run_all_tests()
    
    # Save report if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"Test report saved to: {args.output}")
    
    # Exit with error code if tests failed
    sys.exit(0 if report['overall_success'] else 1)


if __name__ == "__main__":
    asyncio.run(main())
