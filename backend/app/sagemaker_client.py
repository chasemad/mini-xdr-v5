"""
SageMaker Client for Mini-XDR Backend
Handles communication with SageMaker endpoints for ML inference
"""

import boto3
import json
import logging
import time
from typing import Dict, List, Any, Optional
import asyncio
import aiohttp
from datetime import datetime

from .config import settings

logger = logging.getLogger(__name__)

class SageMakerMLClient:
    """Client for SageMaker-based ML inference"""

    def __init__(self):
        self.sagemaker_runtime = boto3.client('sagemaker-runtime', region_name='us-east-1')
        self.endpoint_name = None
        self.endpoint_status = "unknown"
        self.last_health_check = 0
        self.health_check_interval = 300  # 5 minutes

        # Load endpoint configuration
        self._load_endpoint_config()

    def _load_endpoint_config(self):
        """Load SageMaker endpoint configuration"""
        try:
            config_file = "/Users/chasemad/Desktop/mini-xdr/config/sagemaker_endpoints.json"
            with open(config_file, 'r') as f:
                config = json.load(f)
                self.endpoint_name = config.get('endpoint_name')
                logger.info(f"Loaded SageMaker endpoint: {self.endpoint_name}")
        except FileNotFoundError:
            logger.warning("SageMaker endpoint configuration not found")
        except Exception as e:
            logger.error(f"Error loading endpoint configuration: {e}")

    async def health_check(self) -> bool:
        """Check if SageMaker endpoint is healthy"""
        if not self.endpoint_name:
            return False

        current_time = time.time()
        if current_time - self.last_health_check < self.health_check_interval:
            return self.endpoint_status == "InService"

        try:
            sagemaker = boto3.client('sagemaker', region_name='us-east-1')
            response = sagemaker.describe_endpoint(EndpointName=self.endpoint_name)
            self.endpoint_status = response['EndpointStatus']
            self.last_health_check = current_time

            logger.info(f"SageMaker endpoint status: {self.endpoint_status}")
            return self.endpoint_status == "InService"

        except Exception as e:
            logger.error(f"SageMaker health check failed: {e}")
            self.endpoint_status = "Error"
            return False

    async def invoke_endpoint(self, input_data: List[Dict]) -> Dict[str, Any]:
        """Invoke SageMaker endpoint for prediction"""
        if not self.endpoint_name:
            raise ValueError("SageMaker endpoint not configured")

        try:
            # For SageMaker, input_data should be [{"instances": [[79 floats]]}]
            if input_data and "instances" in input_data[0]:
                payload = json.dumps(input_data[0])  # Extract the {"instances": [[...]]} format
            else:
                payload = json.dumps(input_data)

            logger.info(f"Invoking SageMaker endpoint {self.endpoint_name} with payload length: {len(payload)}")

            # Invoke endpoint with timeout
            response = self.sagemaker_runtime.invoke_endpoint(
                EndpointName=self.endpoint_name,
                ContentType='application/json',
                Accept='application/json',
                Body=payload
            )

            # Parse response
            result = json.loads(response['Body'].read().decode())

            logger.info(f"SageMaker inference completed successfully")
            return result

        except Exception as e:
            logger.error(f"SageMaker invocation failed: {e}")
            raise

    async def detect_threats(self, network_events: List[Dict]) -> List[Dict]:
        """High-level threat detection using SageMaker"""
        try:
            # Check endpoint health
            if not await self.health_check():
                raise RuntimeError("SageMaker endpoint is not healthy")

            # Extract 79-dimensional features for SageMaker model
            from .deep_learning_models import deep_learning_manager
            from .models import Event
            from datetime import datetime

            # Convert network events to Event objects for feature extraction
            event_objects = []
            src_ip = network_events[0].get("src_ip", "unknown") if network_events else "unknown"

            for event_data in network_events:
                try:
                    # Create Event object from event data
                    event = Event(
                        src_ip=event_data.get("src_ip", src_ip),
                        dst_ip=event_data.get("dst_ip", ""),
                        dst_port=event_data.get("dst_port", 0),
                        eventid=event_data.get("eventid", "unknown"),
                        message=event_data.get("message", ""),
                        ts=datetime.fromisoformat(event_data.get("timestamp")) if event_data.get("timestamp") else datetime.utcnow(),
                        raw=event_data.get("raw", {})
                    )
                    event_objects.append(event)
                except Exception as e:
                    logger.warning(f"Failed to convert event data: {e}")
                    continue

            if not event_objects:
                logger.warning("No valid events to process")
                return []

            # Extract 79 features using the same feature extraction as the training
            features = deep_learning_manager._extract_features(src_ip, event_objects)

            # Convert features to the format expected by SageMaker: {"instances": [[79 floats]]}
            feature_vector = [list(features.values())]
            sagemaker_input = {"instances": feature_vector}

            logger.info(f"Prepared SageMaker input: {len(feature_vector[0])} features for {len(event_objects)} events")

            # Invoke SageMaker endpoint with proper format
            result = await self.invoke_endpoint([sagemaker_input])

            # Process results - SageMaker returns 7-class probabilities
            threats = []
            predictions = result.get("predictions", [])

            # Map SageMaker prediction to threat classification (7 classes)
            threat_classes = {
                0: "Normal",
                1: "DDoS/DoS Attack",
                2: "Network Reconnaissance",
                3: "Brute Force Attack",
                4: "Web Application Attack",
                5: "Malware/Botnet",
                6: "Advanced Persistent Threat"
            }

            for i, prediction in enumerate(predictions):
                # SageMaker returns probability distribution over 7 classes
                if isinstance(prediction, list) and len(prediction) == 7:
                    # Get the class with highest probability
                    predicted_class = prediction.index(max(prediction))
                    confidence = max(prediction)  # Highest class probability

                    # Calculate overall threat score (1 - normal probability)
                    normal_prob = prediction[0]
                    anomaly_score = 1.0 - normal_prob

                else:
                    # Fallback for unexpected format
                    predicted_class = 0
                    confidence = 0.5
                    anomaly_score = 0.5
                    logger.warning(f"Unexpected SageMaker prediction format: {prediction}")

                # Determine threat type and severity
                threat_type = threat_classes.get(predicted_class, "unknown")

                # Map severity based on threat class and confidence
                if threat_type in ["Advanced Persistent Threat", "Malware/Botnet"] or confidence > 0.8:
                    severity = "critical"
                elif threat_type in ["DDoS/DoS Attack", "Brute Force Attack"] or confidence > 0.6:
                    severity = "high"
                elif threat_type in ["Network Reconnaissance", "Web Application Attack"] or confidence > 0.4:
                    severity = "medium"
                else:
                    severity = "low"

                # Create threat result for this prediction
                threat = {
                    "id": f"sagemaker_{src_ip}_{i}",
                    "anomaly_score": anomaly_score,
                    "threat_type": threat_type,
                    "predicted_class": predicted_class,
                    "confidence": confidence,
                    "class_probabilities": prediction if isinstance(prediction, list) else [0.0] * 7,
                    "severity": severity,
                    "timestamp": network_events[0].get("timestamp") if network_events else datetime.utcnow().isoformat(),
                    "src_ip": src_ip,
                    "dst_ip": network_events[0].get("dst_ip", "") if network_events else "",
                    "ml_model": "sagemaker_pytorch_97.98%",
                    "processing_time": datetime.utcnow().isoformat()
                }
                threats.append(threat)

            return threats

        except Exception as e:
            logger.error(f"Threat detection failed: {e}")
            # Return fallback results
            return self._fallback_threat_detection(network_events)

    def _fallback_threat_detection(self, network_events: List[Dict]) -> List[Dict]:
        """Fallback threat detection when SageMaker is unavailable"""
        logger.warning("Using fallback threat detection")

        threats = []
        for event in network_events:
            # Simple rule-based fallback
            anomaly_score = 0.1  # Default low score

            # Basic heuristics
            src_port = event.get("src_port", 0)
            dst_port = event.get("dst_port", 0)

            if dst_port in [22, 23, 3389]:  # SSH, Telnet, RDP
                anomaly_score = 0.6
            elif src_port in [80, 443, 53]:  # Web traffic, DNS
                anomaly_score = 0.1
            elif event.get("packet_length", 0) > 1500:
                anomaly_score = 0.4

            threat = {
                "id": event.get("id"),
                "anomaly_score": anomaly_score,
                "threat_type": "unknown",
                "confidence": 0.3,
                "severity": "low" if anomaly_score < 0.5 else "medium",
                "timestamp": event.get("timestamp"),
                "src_ip": event.get("src_ip"),
                "dst_ip": event.get("dst_ip"),
                "ml_model": "fallback_rules",
                "processing_time": datetime.utcnow().isoformat()
            }
            threats.append(threat)

        return threats

    async def batch_predict(self, events_batch: List[List[Dict]]) -> List[List[Dict]]:
        """Process multiple batches of events"""
        results = []

        for batch in events_batch:
            try:
                batch_result = await self.detect_threats(batch)
                results.append(batch_result)
            except Exception as e:
                logger.error(f"Batch prediction failed: {e}")
                results.append([])

        return results

    def get_endpoint_info(self) -> Dict[str, Any]:
        """Get endpoint information and status"""
        return {
            "endpoint_name": self.endpoint_name,
            "endpoint_status": self.endpoint_status,
            "last_health_check": self.last_health_check,
            "region": "us-east-1",
            "instance_type": "ml.p3.2xlarge"
        }

# Global instance
sagemaker_client = SageMakerMLClient()

async def get_ml_predictions(network_events: List[Dict]) -> List[Dict]:
    """Main function for getting ML predictions from SageMaker"""
    return await sagemaker_client.detect_threats(network_events)

async def check_ml_health() -> Dict[str, Any]:
    """Check ML service health"""
    is_healthy = await sagemaker_client.health_check()
    endpoint_info = sagemaker_client.get_endpoint_info()

    return {
        "status": "healthy" if is_healthy else "unhealthy",
        "endpoint_info": endpoint_info,
        "service": "sagemaker_gpu"
    }