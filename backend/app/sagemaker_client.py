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
            # Prepare input data
            payload = json.dumps(input_data)

            # Invoke endpoint
            response = self.sagemaker_runtime.invoke_endpoint(
                EndpointName=self.endpoint_name,
                ContentType='application/json',
                Accept='application/json',
                Body=payload
            )

            # Parse response
            result = json.loads(response['Body'].read().decode())

            logger.info(f"SageMaker inference completed for {len(input_data)} records")
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

            # Prepare input for threat detection model
            model_input = []
            for event in network_events:
                input_record = {
                    "id": event.get("id"),
                    "timestamp": event.get("timestamp"),
                    "src_ip": event.get("src_ip"),
                    "dst_ip": event.get("dst_ip"),
                    "src_port": event.get("src_port", 0),
                    "dst_port": event.get("dst_port", 0),
                    "protocol": event.get("protocol", 0),
                    "packet_length": event.get("packet_length", 0),
                    "duration": event.get("duration", 0),
                    "flow_bytes_per_sec": event.get("flow_bytes_per_sec", 0),
                    "flow_packets_per_sec": event.get("flow_packets_per_sec", 0),
                }
                model_input.append(input_record)

            # Invoke SageMaker endpoint
            result = await self.invoke_endpoint(model_input)

            # Process results
            threats = []
            predictions = result.get("predictions", [])

            for prediction in predictions:
                threat = {
                    "id": prediction.get("record_id"),
                    "anomaly_score": prediction.get("anomaly_score", 0.0),
                    "threat_type": prediction.get("threat_classification", {}).get("threat_type", "unknown"),
                    "confidence": prediction.get("threat_classification", {}).get("confidence", 0.0),
                    "severity": prediction.get("severity", "low"),
                    "timestamp": prediction.get("timestamp"),
                    "src_ip": prediction.get("src_ip"),
                    "dst_ip": prediction.get("dst_ip"),
                    "ml_model": "sagemaker_gpu",
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