#!/usr/bin/env python3
"""
Test the deployed SageMaker endpoint with sample data
"""

import boto3
import json
import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_endpoint():
    """Test the endpoint with sample 79-feature vectors"""

    # Initialize SageMaker runtime client
    sagemaker_runtime = boto3.client('sagemaker-runtime', region_name='us-east-1')
    endpoint_name = 'mini-xdr-production-endpoint'

    try:
        # Create sample input - 79 features representing network traffic
        # These would normally come from real network events
        sample_features = np.random.rand(79).tolist()

        # Format input for SageMaker
        payload = {
            "instances": [sample_features]
        }

        logger.info(f"Testing endpoint: {endpoint_name}")
        logger.info(f"Input shape: 1 x 79 features")

        # Invoke endpoint
        response = sagemaker_runtime.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType='application/json',
            Accept='application/json',
            Body=json.dumps(payload)
        )

        # Parse response
        result = json.loads(response['Body'].read().decode())

        logger.info("=" * 60)
        logger.info("✅ ENDPOINT TEST SUCCESSFUL")
        logger.info("=" * 60)
        logger.info(f"Response: {result}")

        # Expected format: {"predictions": [[prob_class_0, prob_class_1, ..., prob_class_6]]}
        if 'predictions' in result:
            predictions = result['predictions'][0]
            logger.info(f"\n7-Class Probabilities:")
            threat_classes = [
                "Normal",
                "DDoS/DoS Attack",
                "Network Reconnaissance",
                "Brute Force Attack",
                "Web Application Attack",
                "Malware/Botnet",
                "Advanced Persistent Threat"
            ]

            for i, (threat_class, prob) in enumerate(zip(threat_classes, predictions)):
                logger.info(f"  {threat_class}: {prob:.4f}")

            predicted_class = np.argmax(predictions)
            logger.info(f"\nPredicted Threat: {threat_classes[predicted_class]}")
            logger.info(f"Confidence: {predictions[predicted_class]:.2%}")

        logger.info("=" * 60)

        return {
            "success": True,
            "endpoint_status": "healthy",
            "response": result
        }

    except Exception as e:
        logger.error(f"❌ Endpoint test failed: {e}")
        return {
            "success": False,
            "error": str(e)
        }

if __name__ == "__main__":
    result = test_endpoint()
    exit(0 if result['success'] else 1)