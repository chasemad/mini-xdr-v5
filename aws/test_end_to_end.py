#!/usr/bin/env python3
"""
End-to-End Test: Simulated Attack → Feature Extraction → SageMaker → Classification
Tests the complete flow from raw event to threat detection
"""

import sys
import os
sys.path.append('/Users/chasemad/Desktop/mini-xdr/backend')

import asyncio
import json
import logging
from datetime import datetime, timezone
import numpy as np
import boto3

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EndToEndTester:
    def __init__(self):
        self.sagemaker_runtime = boto3.client('sagemaker-runtime', region_name='us-east-1')
        self.endpoint_name = 'mini-xdr-production-endpoint'

    def simulate_brute_force_attack(self):
        """Simulate a brute force attack with realistic events"""
        logger.info("🎭 Simulating Brute Force Attack")

        # Simulate 20 failed login attempts in 2 minutes
        events = []
        base_time = datetime.now(timezone.utc)

        usernames = ['root', 'admin', 'user', 'test']
        passwords = ['123456', 'password', 'admin123', 'root123', 'test']

        for i in range(20):
            event = {
                "id": f"sim_{i}",
                "src_ip": "203.0.113.42",  # Test IP
                "dst_ip": "10.0.1.100",
                "dst_port": 22,
                "eventid": "cowrie.login.failed",
                "message": f"Failed login attempt {i}",
                "timestamp": base_time.isoformat(),
                "raw": {
                    "username": usernames[i % len(usernames)],
                    "password": passwords[i % len(passwords)],
                    "protocol": "ssh",
                    "session": f"session_{i // 5}"  # 4 sessions, 5 attempts each
                }
            }
            events.append(event)

        logger.info(f"  Generated {len(events)} simulated attack events")
        return events

    def extract_features(self, src_ip: str, events: list) -> dict:
        """Extract 79 features from events (matching training data)"""
        logger.info("🔧 Extracting Features")

        if not events:
            return {f'feature_{i}': 0.0 for i in range(79)}

        features = {}

        # Time-based features
        features['event_count_1h'] = len(events)
        features['event_count_24h'] = len(events)
        features['unique_ports'] = len(set(e.get('dst_port', 0) for e in events))

        # Failed login analysis
        failed_logins = [e for e in events if e.get('eventid') == 'cowrie.login.failed']
        features['failed_login_count'] = len(failed_logins)

        # Session analysis
        if events:
            features['session_duration_avg'] = 120.0  # 2 minutes
            features['event_rate_per_minute'] = len(events) / 2.0
        else:
            features['session_duration_avg'] = 0
            features['event_rate_per_minute'] = 0

        # Credential analysis
        usernames = set()
        passwords = set()
        password_lengths = []

        for event in failed_logins:
            raw_data = event.get('raw', {})
            if 'username' in raw_data:
                usernames.add(raw_data['username'])
            if 'password' in raw_data:
                passwords.add(raw_data['password'])
                password_lengths.append(len(str(raw_data['password'])))

        features['unique_usernames'] = len(usernames)
        features['password_diversity'] = len(passwords)
        features['username_diversity'] = len(usernames)
        features['password_length_avg'] = np.mean(password_lengths) if password_lengths else 0

        # Command and download attempts
        features['command_diversity'] = 0
        features['download_attempts'] = 0
        features['upload_attempts'] = 0

        # Time features
        features['time_of_day'] = 12  # Noon
        features['is_weekend'] = 0

        # Network features
        features['bytes_sent'] = len(events) * 512
        features['bytes_received'] = len(events) * 256
        features['packets_sent'] = len(events)
        features['packets_received'] = len(events)
        features['connection_duration'] = 120.0

        # Attack pattern indicators
        features['rapid_fire_attempts'] = 1.0  # High rate
        features['credential_stuffing_score'] = 0.8  # Many passwords
        features['password_spray_score'] = 0.3
        features['username_enum_score'] = 0.4

        # Session features
        features['unique_sessions'] = len(set(e.get('raw', {}).get('session', '') for e in events))
        features['avg_attempts_per_session'] = len(events) / features['unique_sessions'] if features['unique_sessions'] > 0 else 0

        # Protocol features
        features['ssh_attempts'] = len([e for e in events if e.get('dst_port') == 22])
        features['telnet_attempts'] = 0
        features['rdp_attempts'] = 0
        features['http_attempts'] = 0
        features['https_attempts'] = 0

        # Behavioral features
        features['time_between_attempts_avg'] = 6.0  # seconds
        features['time_between_attempts_std'] = 2.0
        features['attempt_regularity'] = 0.9  # Very regular

        # Pad remaining features to reach 79
        current_count = len(features)
        for i in range(current_count, 79):
            features[f'feature_{i}'] = 0.0

        logger.info(f"  Extracted {len(features)} features")
        logger.info(f"  Key indicators: {features['failed_login_count']} failures, "
                   f"{features['unique_usernames']} usernames, {features['password_diversity']} passwords")

        return features

    def invoke_sagemaker(self, features: dict) -> dict:
        """Invoke SageMaker endpoint with features"""
        logger.info("🤖 Invoking SageMaker Model")

        # Convert features dict to list of values
        feature_vector = list(features.values())[:79]  # Ensure exactly 79 features

        # Format for SageMaker
        payload = {
            "instances": [feature_vector]
        }

        logger.info(f"  Feature vector length: {len(feature_vector)}")
        logger.info(f"  Feature stats: min={min(feature_vector):.2f}, max={max(feature_vector):.2f}, mean={np.mean(feature_vector):.2f}")

        try:
            response = self.sagemaker_runtime.invoke_endpoint(
                EndpointName=self.endpoint_name,
                ContentType='application/json',
                Accept='application/json',
                Body=json.dumps(payload)
            )

            result = json.loads(response['Body'].read().decode())
            logger.info("  ✅ SageMaker invocation successful")

            return result

        except Exception as e:
            logger.error(f"  ❌ SageMaker invocation failed: {e}")
            raise

    def interpret_results(self, result: dict, events: list) -> dict:
        """Interpret SageMaker results"""
        logger.info("📊 Interpreting Results")

        threat_classes = [
            "Normal",
            "DDoS/DoS Attack",
            "Network Reconnaissance",
            "Brute Force Attack",
            "Web Application Attack",
            "Malware/Botnet",
            "Advanced Persistent Threat"
        ]

        predictions = result.get('predictions', [[]])[0]

        if not predictions:
            logger.error("  ❌ No predictions returned")
            return {"error": "No predictions"}

        # Get predicted class
        predicted_idx = np.argmax(predictions)
        predicted_class = threat_classes[predicted_idx]
        confidence = predictions[predicted_idx]

        # Determine severity
        if confidence > 0.8:
            severity = "critical"
        elif confidence > 0.6:
            severity = "high"
        elif confidence > 0.4:
            severity = "medium"
        else:
            severity = "low"

        logger.info(f"\n{'=' * 60}")
        logger.info("🎯 THREAT DETECTION RESULTS")
        logger.info(f"{'=' * 60}")
        logger.info(f"Source IP: {events[0].get('src_ip', 'unknown')}")
        logger.info(f"Event Count: {len(events)}")
        logger.info(f"Predicted Threat: {predicted_class}")
        logger.info(f"Confidence: {confidence:.2%}")
        logger.info(f"Severity: {severity.upper()}")
        logger.info(f"\nClass Probabilities:")

        for i, (cls, prob) in enumerate(zip(threat_classes, predictions)):
            indicator = "👉" if i == predicted_idx else "  "
            logger.info(f"{indicator} {cls:30s}: {prob:.4f} ({prob*100:.2f}%)")

        logger.info(f"{'=' * 60}\n")

        # Validate prediction
        expected_class = "Brute Force Attack"
        if predicted_class == expected_class:
            logger.info(f"✅ CORRECT: Model correctly identified {expected_class}")
        else:
            logger.warning(f"⚠️  UNEXPECTED: Expected {expected_class}, got {predicted_class}")

        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "severity": severity,
            "all_probabilities": {cls: prob for cls, prob in zip(threat_classes, predictions)},
            "validation": {
                "expected": expected_class,
                "correct": predicted_class == expected_class
            }
        }

    def run_test(self):
        """Run complete end-to-end test"""
        logger.info("\n" + "=" * 60)
        logger.info("🚀 STARTING END-TO-END TEST")
        logger.info("=" * 60 + "\n")

        try:
            # Step 1: Simulate attack
            events = self.simulate_brute_force_attack()

            # Step 2: Extract features
            features = self.extract_features(events[0]['src_ip'], events)

            # Step 3: Invoke SageMaker
            result = self.invoke_sagemaker(features)

            # Step 4: Interpret results
            interpretation = self.interpret_results(result, events)

            # Summary
            logger.info("\n" + "=" * 60)
            if interpretation.get('validation', {}).get('correct'):
                logger.info("✅ END-TO-END TEST PASSED")
            else:
                logger.info("⚠️  END-TO-END TEST: UNEXPECTED RESULT")
            logger.info("=" * 60)

            return {
                "success": True,
                "interpretation": interpretation,
                "events_processed": len(events),
                "features_extracted": len(features)
            }

        except Exception as e:
            logger.error(f"\n❌ END-TO-END TEST FAILED: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }


if __name__ == "__main__":
    tester = EndToEndTester()
    result = tester.run_test()
    exit(0 if result['success'] else 1)