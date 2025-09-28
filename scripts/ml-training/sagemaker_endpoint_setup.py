#!/usr/bin/env python3
"""
SageMaker Endpoint Setup for Mini-XDR Threat Detection
Creates real-time inference endpoints using approved ml.p3.8xlarge quota
"""

import boto3
import json
import time
import logging
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SageMakerEndpointManager:
    def __init__(self, region='us-east-1'):
        self.region = region
        self.sagemaker = boto3.client('sagemaker', region_name=region)
        self.s3 = boto3.client('s3', region_name=region)
        self.iam = boto3.client('iam', region_name=region)

        # Configuration
        self.bucket_name = 'mini-xdr-ml-models-675076709589'
        self.role_name = 'SageMakerExecutionRole-MiniXDR'
        self.timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')

    def create_s3_bucket(self):
        """Create S3 bucket for model artifacts if it doesn't exist"""
        try:
            self.s3.head_bucket(Bucket=self.bucket_name)
            logger.info(f"S3 bucket {self.bucket_name} already exists")
        except:
            try:
                if self.region == 'us-east-1':
                    self.s3.create_bucket(Bucket=self.bucket_name)
                else:
                    self.s3.create_bucket(
                        Bucket=self.bucket_name,
                        CreateBucketConfiguration={'LocationConstraint': self.region}
                    )
                logger.info(f"Created S3 bucket: {self.bucket_name}")
            except Exception as e:
                logger.error(f"Failed to create S3 bucket: {e}")
                raise

    def create_execution_role(self):
        """Create SageMaker execution role if it doesn't exist"""
        try:
            role = self.iam.get_role(RoleName=self.role_name)
            logger.info(f"IAM role {self.role_name} already exists")
            return role['Role']['Arn']
        except:
            # Create the role
            assume_role_policy = {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Principal": {"Service": "sagemaker.amazonaws.com"},
                        "Action": "sts:AssumeRole"
                    }
                ]
            }

            try:
                role = self.iam.create_role(
                    RoleName=self.role_name,
                    AssumeRolePolicyDocument=json.dumps(assume_role_policy),
                    Description='Execution role for Mini-XDR SageMaker endpoints'
                )

                # Attach necessary policies
                policies = [
                    'arn:aws:iam::aws:policy/AmazonSageMakerFullAccess',
                    'arn:aws:iam::aws:policy/AmazonS3FullAccess'
                ]

                for policy in policies:
                    self.iam.attach_role_policy(RoleName=self.role_name, PolicyArn=policy)

                # Wait for role to be ready
                time.sleep(10)

                logger.info(f"Created IAM role: {self.role_name}")
                return role['Role']['Arn']
            except Exception as e:
                logger.error(f"Failed to create IAM role: {e}")
                raise

    def create_threat_detection_model(self):
        """Create SageMaker model for threat detection"""
        model_name = f"mini-xdr-threat-detection-{self.timestamp}"

        # For now, we'll use a PyTorch container with our custom inference code
        container_image = "763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:2.0.0-gpu-py310-cu118-ubuntu20.04-sagemaker"

        # Model artifact location (we'll create a placeholder)
        model_artifact_path = f"s3://{self.bucket_name}/models/threat-detection/model.tar.gz"

        # Get execution role
        role_arn = self.create_execution_role()

        try:
            response = self.sagemaker.create_model(
                ModelName=model_name,
                PrimaryContainer={
                    'Image': container_image,
                    'ModelDataUrl': model_artifact_path,
                    'Environment': {
                        'SAGEMAKER_PROGRAM': 'inference.py',
                        'SAGEMAKER_SUBMIT_DIRECTORY': '/opt/ml/code'
                    }
                },
                ExecutionRoleArn=role_arn,
                Tags=[
                    {'Key': 'Project', 'Value': 'Mini-XDR'},
                    {'Key': 'Purpose', 'Value': 'ThreatDetection'},
                    {'Key': 'Environment', 'Value': 'Production'}
                ]
            )

            logger.info(f"Created SageMaker model: {model_name}")
            return model_name

        except Exception as e:
            logger.error(f"Failed to create SageMaker model: {e}")
            raise

    def create_endpoint_configuration(self, model_name):
        """Create endpoint configuration for threat detection model"""
        config_name = f"mini-xdr-endpoint-config-{self.timestamp}"

        try:
            response = self.sagemaker.create_endpoint_config(
                EndpointConfigName=config_name,
                ProductionVariants=[
                    {
                        'VariantName': 'primary',
                        'ModelName': model_name,
                        'InitialInstanceCount': 1,
                        'InstanceType': 'ml.p3.2xlarge',  # Using your approved quota
                        'InitialVariantWeight': 1.0
                    }
                ],
                Tags=[
                    {'Key': 'Project', 'Value': 'Mini-XDR'},
                    {'Key': 'Purpose', 'Value': 'ThreatDetection'}
                ]
            )

            logger.info(f"Created endpoint configuration: {config_name}")
            return config_name

        except Exception as e:
            logger.error(f"Failed to create endpoint configuration: {e}")
            raise

    def create_endpoint(self, config_name):
        """Create the actual SageMaker endpoint"""
        endpoint_name = f"mini-xdr-threat-detection-{self.timestamp}"

        try:
            response = self.sagemaker.create_endpoint(
                EndpointName=endpoint_name,
                EndpointConfigName=config_name,
                Tags=[
                    {'Key': 'Project', 'Value': 'Mini-XDR'},
                    {'Key': 'Purpose', 'Value': 'ThreatDetection'},
                    {'Key': 'Environment', 'Value': 'Production'}
                ]
            )

            logger.info(f"Creating SageMaker endpoint: {endpoint_name}")

            # Wait for endpoint to be ready
            logger.info("Waiting for endpoint to be in service...")
            waiter = self.sagemaker.get_waiter('endpoint_in_service')
            waiter.wait(EndpointName=endpoint_name, WaiterConfig={'Delay': 30, 'MaxAttempts': 20})

            logger.info(f"✅ Endpoint {endpoint_name} is now in service!")
            return endpoint_name

        except Exception as e:
            logger.error(f"Failed to create endpoint: {e}")
            raise

    def deploy_threat_detection_endpoint(self):
        """Complete deployment of threat detection endpoint"""
        try:
            # Step 1: Create S3 bucket
            self.create_s3_bucket()

            # Step 2: Create model
            model_name = self.create_threat_detection_model()

            # Step 3: Create endpoint configuration
            config_name = self.create_endpoint_configuration(model_name)

            # Step 4: Create endpoint
            endpoint_name = self.create_endpoint(config_name)

            # Save endpoint info
            endpoint_info = {
                'endpoint_name': endpoint_name,
                'model_name': model_name,
                'config_name': config_name,
                'region': self.region,
                'created_at': datetime.now().isoformat()
            }

            # Write endpoint info to file
            info_file = Path('/Users/chasemad/Desktop/mini-xdr/config/sagemaker_endpoints.json')
            info_file.parent.mkdir(exist_ok=True)

            with open(info_file, 'w') as f:
                json.dump(endpoint_info, f, indent=2)

            logger.info(f"🚀 Successfully deployed threat detection endpoint!")
            logger.info(f"📊 Endpoint name: {endpoint_name}")
            logger.info(f"🔗 Region: {self.region}")
            logger.info(f"💾 Configuration saved to: {info_file}")

            return endpoint_info

        except Exception as e:
            logger.error(f"❌ Failed to deploy endpoint: {e}")
            raise

def main():
    """Main function to deploy SageMaker endpoint"""
    logger.info("🚀 Starting SageMaker endpoint deployment for Mini-XDR")

    try:
        manager = SageMakerEndpointManager()
        endpoint_info = manager.deploy_threat_detection_endpoint()

        print("\n" + "="*60)
        print("✅ SAGEMAKER ENDPOINT DEPLOYMENT COMPLETE")
        print("="*60)
        print(f"Endpoint Name: {endpoint_info['endpoint_name']}")
        print(f"Model Name: {endpoint_info['model_name']}")
        print(f"Region: {endpoint_info['region']}")
        print(f"Instance Type: ml.p3.2xlarge (GPU)")
        print("="*60)

    except Exception as e:
        logger.error(f"❌ Deployment failed: {e}")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())