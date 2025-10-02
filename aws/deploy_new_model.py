#!/usr/bin/env python3
"""
Deploy newly trained SageMaker model to existing endpoint
Updates mini-xdr-production-endpoint with the latest trained model
"""

import boto3
import json
import time
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelDeployer:
    def __init__(self, region='us-east-1'):
        self.region = region
        self.sagemaker = boto3.client('sagemaker', region_name=region)
        self.training_job_name = "enhanced-xdr-real-data-20250929-070000"
        self.endpoint_name = "mini-xdr-production-endpoint"

    def create_model_from_training_job(self):
        """Create SageMaker model from training job artifacts"""
        model_name = f"mini-xdr-model-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

        try:
            # Get training job details
            training_job = self.sagemaker.describe_training_job(
                TrainingJobName=self.training_job_name
            )

            model_data_url = training_job['ModelArtifacts']['S3ModelArtifacts']
            training_image = training_job['AlgorithmSpecification']['TrainingImage']
            role_arn = training_job['RoleArn']

            logger.info(f"Creating model from training job: {self.training_job_name}")
            logger.info(f"Model artifacts: {model_data_url}")

            # Use CPU inference container for cost efficiency
            inference_image = '763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:2.0.1-cpu-py310'

            # Create model
            response = self.sagemaker.create_model(
                ModelName=model_name,
                PrimaryContainer={
                    'Image': inference_image,
                    'ModelDataUrl': model_data_url,
                    'Environment': {
                        'SAGEMAKER_PROGRAM': 'enhanced_sagemaker_train_real.py',
                        'SAGEMAKER_SUBMIT_DIRECTORY': model_data_url,
                        'SAGEMAKER_REGION': self.region
                    }
                },
                ExecutionRoleArn=role_arn,
                Tags=[
                    {'Key': 'Project', 'Value': 'Mini-XDR'},
                    {'Key': 'TrainingJob', 'Value': self.training_job_name},
                    {'Key': 'DeployedAt', 'Value': datetime.now().isoformat()}
                ]
            )

            logger.info(f"✅ Created model: {model_name}")
            return model_name

        except Exception as e:
            logger.error(f"Failed to create model: {e}")
            raise

    def update_endpoint(self, model_name):
        """Update existing endpoint with new model"""
        config_name = f"mini-xdr-config-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

        try:
            # Create new endpoint configuration
            logger.info(f"Creating endpoint configuration: {config_name}")
            self.sagemaker.create_endpoint_config(
                EndpointConfigName=config_name,
                ProductionVariants=[
                    {
                        'VariantName': 'primary',
                        'ModelName': model_name,
                        'InitialInstanceCount': 1,
                        'InstanceType': 'ml.m5.large',  # Using CPU instance for cost efficiency
                        'InitialVariantWeight': 1.0
                    }
                ],
                Tags=[
                    {'Key': 'Project', 'Value': 'Mini-XDR'},
                    {'Key': 'Model', 'Value': model_name}
                ]
            )

            logger.info(f"✅ Created endpoint configuration: {config_name}")

            # Update endpoint
            logger.info(f"Updating endpoint: {self.endpoint_name}")
            self.sagemaker.update_endpoint(
                EndpointName=self.endpoint_name,
                EndpointConfigName=config_name
            )

            logger.info(f"⏳ Waiting for endpoint update to complete...")
            waiter = self.sagemaker.get_waiter('endpoint_in_service')
            waiter.wait(
                EndpointName=self.endpoint_name,
                WaiterConfig={'Delay': 30, 'MaxAttempts': 30}
            )

            logger.info(f"✅ Endpoint {self.endpoint_name} updated successfully!")

            return config_name

        except Exception as e:
            logger.error(f"Failed to update endpoint: {e}")
            raise

    def update_config_file(self, model_name):
        """Update config file with new model information"""
        config_path = "/Users/chasemad/Desktop/mini-xdr/config/sagemaker_endpoints.json"

        try:
            config = {
                "endpoint_name": self.endpoint_name,
                "model_name": model_name,
                "training_job": self.training_job_name,
                "model_accuracy": 0.9798,  # From training logs
                "features": 79,
                "region": self.region,
                "updated_at": datetime.now().isoformat(),
                "cost_optimization": {
                    "development_mode": True,
                    "auto_scale_enabled": False,
                    "scale_to_zero_when_idle": True,
                    "max_hourly_cost": 5.0
                }
            }

            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)

            logger.info(f"✅ Updated configuration file: {config_path}")

        except Exception as e:
            logger.error(f"Failed to update config file: {e}")
            raise

    def deploy(self):
        """Execute full deployment"""
        try:
            logger.info("=" * 60)
            logger.info("🚀 DEPLOYING NEW SAGEMAKER MODEL")
            logger.info("=" * 60)

            # Step 1: Create model from training job
            model_name = self.create_model_from_training_job()

            # Step 2: Update endpoint
            config_name = self.update_endpoint(model_name)

            # Step 3: Update config file
            self.update_config_file(model_name)

            logger.info("=" * 60)
            logger.info("✅ DEPLOYMENT COMPLETE")
            logger.info("=" * 60)
            logger.info(f"Endpoint: {self.endpoint_name}")
            logger.info(f"Model: {model_name}")
            logger.info(f"Training Job: {self.training_job_name}")
            logger.info(f"Status: InService")
            logger.info("=" * 60)

            return {
                "success": True,
                "endpoint_name": self.endpoint_name,
                "model_name": model_name,
                "training_job": self.training_job_name
            }

        except Exception as e:
            logger.error(f"❌ Deployment failed: {e}")
            return {"success": False, "error": str(e)}

if __name__ == "__main__":
    deployer = ModelDeployer()
    result = deployer.deploy()
    exit(0 if result.get("success") else 1)