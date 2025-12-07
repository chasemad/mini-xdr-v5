#!/usr/bin/env python3
"""
Deploy Trained SageMaker Model to Real-Time Endpoint
Uses the completed training job artifacts
"""

import boto3
import json
import time
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TrainedModelDeployer:
    def __init__(self, training_job_name="mini-xdr-gpu-regular-20250927-061258", region='us-east-1'):
        self.training_job_name = training_job_name
        self.region = region
        self.sagemaker = boto3.client('sagemaker', region_name=region)
        self.timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')

    def wait_for_training_completion(self):
        """Wait for training job to complete"""
        logger.info(f"Checking training job: {self.training_job_name}")

        try:
            response = self.sagemaker.describe_training_job(TrainingJobName=self.training_job_name)
            status = response['TrainingJobStatus']

            if status == 'Completed':
                logger.info("✅ Training job completed successfully!")
                return response['ModelArtifacts']['S3ModelArtifacts']
            elif status == 'InProgress':
                logger.info("⏳ Training job still in progress...")
                return None
            elif status in ['Failed', 'Stopped']:
                logger.error(f"❌ Training job {status}: {response.get('FailureReason', 'Unknown')}")
                return None
            else:
                logger.info(f"Training job status: {status}")
                return None

        except Exception as e:
            logger.error(f"Error checking training job: {e}")
            return None

    def create_model_from_training(self, model_artifacts_url):
        """Create SageMaker model from training job artifacts"""
        model_name = f"mini-xdr-trained-model-{self.timestamp}"

        # Get the same container image used in training
        container_image = "763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:2.0.0-gpu-py310-cu118-ubuntu20.04-sagemaker"

        # Get execution role (should exist from previous deployment)
        role_arn = f"arn:aws:iam::675076709589:role/SageMakerExecutionRole-MiniXDR"

        try:
            response = self.sagemaker.create_model(
                ModelName=model_name,
                PrimaryContainer={
                    'Image': container_image,
                    'ModelDataUrl': model_artifacts_url,
                    'Environment': {
                        'SAGEMAKER_PROGRAM': 'inference.py',
                        'SAGEMAKER_SUBMIT_DIRECTORY': '/opt/ml/code',
                        'SAGEMAKER_REQUIREMENTS': 'requirements.txt'
                    }
                },
                ExecutionRoleArn=role_arn,
                Tags=[
                    {'Key': 'Project', 'Value': 'Mini-XDR'},
                    {'Key': 'Purpose', 'Value': 'TrainedThreatDetection'},
                    {'Key': 'TrainingJob', 'Value': self.training_job_name}
                ]
            )

            logger.info(f"✅ Created model: {model_name}")
            return model_name

        except Exception as e:
            logger.error(f"Failed to create model: {e}")
            raise

    def create_endpoint_config_with_autoscaling(self, model_name):
        """Create endpoint configuration with auto-scaling"""
        config_name = f"mini-xdr-trained-config-{self.timestamp}"

        try:
            response = self.sagemaker.create_endpoint_config(
                EndpointConfigName=config_name,
                ProductionVariants=[
                    {
                        'VariantName': 'primary',
                        'ModelName': model_name,
                        'InitialInstanceCount': 1,
                        'InstanceType': 'ml.p3.2xlarge',  # GPU instance for inference
                        'InitialVariantWeight': 1.0
                    }
                ],
                Tags=[
                    {'Key': 'Project', 'Value': 'Mini-XDR'},
                    {'Key': 'Purpose', 'Value': 'TrainedThreatDetection'}
                ]
            )

            logger.info(f"✅ Created endpoint configuration: {config_name}")
            return config_name

        except Exception as e:
            logger.error(f"Failed to create endpoint config: {e}")
            raise

    def deploy_endpoint(self, config_name):
        """Deploy the endpoint"""
        endpoint_name = f"mini-xdr-trained-endpoint-{self.timestamp}"

        try:
            response = self.sagemaker.create_endpoint(
                EndpointName=endpoint_name,
                EndpointConfigName=config_name,
                Tags=[
                    {'Key': 'Project', 'Value': 'Mini-XDR'},
                    {'Key': 'Purpose', 'Value': 'TrainedThreatDetection'},
                    {'Key': 'TrainingJob', 'Value': self.training_job_name}
                ]
            )

            logger.info(f"🚀 Deploying endpoint: {endpoint_name}")
            logger.info("⏳ Waiting for endpoint to be in service (this may take 10-15 minutes)...")

            # Wait for endpoint to be ready
            waiter = self.sagemaker.get_waiter('endpoint_in_service')
            waiter.wait(EndpointName=endpoint_name, WaiterConfig={'Delay': 30, 'MaxAttempts': 30})

            logger.info(f"✅ Endpoint {endpoint_name} is now in service!")
            return endpoint_name

        except Exception as e:
            logger.error(f"Failed to deploy endpoint: {e}")
            raise

    def update_backend_config(self, endpoint_name, model_name):
        """Update backend configuration with new endpoint"""
        config_data = {
            'endpoint_name': endpoint_name,
            'model_name': model_name,
            'training_job_name': self.training_job_name,
            'region': self.region,
            'instance_type': 'ml.p3.2xlarge',
            'deployed_at': datetime.now().isoformat(),
            'status': 'active'
        }

        # Save to config file
        config_file = '$(cd "$(dirname "$0")/../.." ${PROJECT_ROOT:-$(dirname $(dirname $(dirname $(realpath "$0"))))}${PROJECT_ROOT:-$(dirname $(dirname $(dirname $(realpath "$0"))))} pwd)/config/sagemaker_endpoints.json'
        with open(config_file, 'w') as f:
            json.dump(config_data, f, indent=2)

        logger.info(f"📝 Updated backend configuration: {config_file}")
        return config_data

    def deploy_trained_model(self):
        """Complete deployment of trained model"""
        try:
            # Step 1: Check if training is complete
            model_artifacts = self.wait_for_training_completion()
            if not model_artifacts:
                logger.error("❌ Training job not yet complete. Please wait and try again.")
                return None

            logger.info(f"📦 Model artifacts: {model_artifacts}")

            # Step 2: Create model from training artifacts
            model_name = self.create_model_from_training(model_artifacts)

            # Step 3: Create endpoint configuration
            config_name = self.create_endpoint_config_with_autoscaling(model_name)

            # Step 4: Deploy endpoint
            endpoint_name = self.deploy_endpoint(config_name)

            # Step 5: Update backend configuration
            config_data = self.update_backend_config(endpoint_name, model_name)

            print("\n" + "="*70)
            print("🎉 TRAINED MODEL DEPLOYMENT COMPLETE!")
            print("="*70)
            print(f"🎯 Endpoint Name: {endpoint_name}")
            print(f"🔬 Model Name: {model_name}")
            print(f"🏋️ Training Job: {self.training_job_name}")
            print(f"💻 Instance Type: ml.p3.2xlarge (GPU)")
            print(f"🌍 Region: {self.region}")
            print("="*70)
            print("🚀 Ready for real-time threat detection inference!")

            return config_data

        except Exception as e:
            logger.error(f"❌ Deployment failed: {e}")
            return None

def main():
    """Main deployment function"""
    logger.info("🚀 Starting trained model deployment for Mini-XDR")

    try:
        deployer = TrainedModelDeployer()
        result = deployer.deploy_trained_model()

        if result:
            return 0
        else:
            return 1

    except Exception as e:
        logger.error(f"❌ Main deployment failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())