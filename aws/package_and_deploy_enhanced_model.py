#!/usr/bin/env python3
"""
Package and deploy the enhanced threat detection model to SageMaker
Downloads the trained model, adds inference script, repackages, and deploys
"""

import boto3
import json
import tarfile
import tempfile
import shutil
import logging
from pathlib import Path
from datetime import datetime
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EnhancedModelDeployer:
    def __init__(self, region='us-east-1'):
        self.region = region
        self.s3 = boto3.client('s3', region_name=region)
        self.sagemaker = boto3.client('sagemaker', region_name=region)

        self.bucket = 'mini-xdr-ml-data-bucket-675076709589'
        self.training_job_name = 'enhanced-xdr-real-data-20250929-070000'
        self.endpoint_name = 'mini-xdr-production-endpoint'

        # Paths
        self.model_s3_path = f's3://{self.bucket}/enhanced_models/{self.training_job_name}/output/model.tar.gz'
        self.inference_script = '/Users/chasemad/Desktop/mini-xdr/aws/inference_enhanced.py'

    def download_model_from_s3(self, temp_dir: Path) -> Path:
        """Download the trained model from S3"""
        logger.info(f"Downloading model from {self.model_s3_path}")

        model_archive = temp_dir / "model.tar.gz"

        # Parse S3 path
        s3_key = f'enhanced_models/{self.training_job_name}/output/model.tar.gz'

        try:
            self.s3.download_file(self.bucket, s3_key, str(model_archive))
            logger.info(f"Downloaded model to {model_archive}")
            return model_archive
        except Exception as e:
            logger.error(f"Failed to download model: {e}")
            raise

    def extract_model(self, model_archive: Path, extract_dir: Path):
        """Extract the model archive"""
        logger.info(f"Extracting model archive to {extract_dir}")

        try:
            with tarfile.open(model_archive, 'r:gz') as tar:
                tar.extractall(extract_dir)

            # List extracted files
            files = list(extract_dir.rglob('*'))
            logger.info(f"Extracted {len(files)} files")
            for f in files[:10]:  # Show first 10 files
                logger.info(f"  - {f.name}")

            return True
        except Exception as e:
            logger.error(f"Failed to extract model: {e}")
            raise

    def add_inference_script(self, model_dir: Path):
        """Add the inference script to the model directory"""
        logger.info("Adding inference script to model")

        try:
            # Copy inference script
            inference_dest = model_dir / "code" / "inference.py"
            inference_dest.parent.mkdir(parents=True, exist_ok=True)

            shutil.copy(self.inference_script, inference_dest)
            logger.info(f"Copied inference script to {inference_dest}")

            # Create requirements.txt for dependencies
            requirements = model_dir / "code" / "requirements.txt"
            with open(requirements, 'w') as f:
                f.write("torch>=2.0.0\n")
                f.write("numpy>=1.21.0\n")
                f.write("scikit-learn>=1.0.0\n")
                f.write("joblib>=1.0.0\n")

            logger.info("Created requirements.txt")

            return True
        except Exception as e:
            logger.error(f"Failed to add inference script: {e}")
            raise

    def repackage_model(self, model_dir: Path, output_archive: Path):
        """Repackage the model with inference script"""
        logger.info(f"Repackaging model to {output_archive}")

        try:
            with tarfile.open(output_archive, 'w:gz') as tar:
                for item in model_dir.rglob('*'):
                    if item.is_file():
                        arcname = item.relative_to(model_dir)
                        tar.add(item, arcname=arcname)
                        logger.debug(f"Added {arcname} to archive")

            logger.info(f"Created model package: {output_archive.stat().st_size / 1024 / 1024:.2f} MB")
            return True
        except Exception as e:
            logger.error(f"Failed to repackage model: {e}")
            raise

    def upload_to_s3(self, model_archive: Path) -> str:
        """Upload repackaged model to S3"""
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        s3_key = f'enhanced_models/packaged/enhanced-model-{timestamp}.tar.gz'

        logger.info(f"Uploading model to s3://{self.bucket}/{s3_key}")

        try:
            self.s3.upload_file(str(model_archive), self.bucket, s3_key)

            s3_uri = f's3://{self.bucket}/{s3_key}'
            logger.info(f"Uploaded model to {s3_uri}")
            return s3_uri
        except Exception as e:
            logger.error(f"Failed to upload model: {e}")
            raise

    def create_sagemaker_model(self, model_data_url: str) -> str:
        """Create SageMaker model"""
        model_name = f'mini-xdr-enhanced-{datetime.now().strftime("%Y%m%d-%H%M%S")}'

        logger.info(f"Creating SageMaker model: {model_name}")

        try:
            # Get training job for role ARN and image
            training_job = self.sagemaker.describe_training_job(
                TrainingJobName=self.training_job_name
            )

            role_arn = training_job['RoleArn']

            # Use CPU inference container
            inference_image = '763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:2.0.1-cpu-py310'

            response = self.sagemaker.create_model(
                ModelName=model_name,
                PrimaryContainer={
                    'Image': inference_image,
                    'ModelDataUrl': model_data_url,
                    'Environment': {
                        'SAGEMAKER_PROGRAM': 'inference.py',
                        'SAGEMAKER_SUBMIT_DIRECTORY': model_data_url,
                        'SAGEMAKER_REGION': self.region
                    }
                },
                ExecutionRoleArn=role_arn,
                Tags=[
                    {'Key': 'Project', 'Value': 'Mini-XDR'},
                    {'Key': 'ModelType', 'Value': 'EnhancedThreatDetector'},
                    {'Key': 'Accuracy', 'Value': '97.98'}
                ]
            )

            logger.info(f"✅ Created SageMaker model: {model_name}")
            return model_name

        except Exception as e:
            logger.error(f"Failed to create SageMaker model: {e}")
            raise

    def update_endpoint(self, model_name: str):
        """Update endpoint with new model"""
        config_name = f'mini-xdr-config-{datetime.now().strftime("%Y%m%d-%H%M%S")}'

        logger.info(f"Creating endpoint configuration: {config_name}")

        try:
            # Create new endpoint configuration
            self.sagemaker.create_endpoint_config(
                EndpointConfigName=config_name,
                ProductionVariants=[
                    {
                        'VariantName': 'primary',
                        'ModelName': model_name,
                        'InitialInstanceCount': 1,
                        'InstanceType': 'ml.m5.large',
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

            logger.info("⏳ Waiting for endpoint update to complete...")
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

    def update_config_file(self, model_name: str):
        """Update local config file"""
        config_path = "/Users/chasemad/Desktop/mini-xdr/config/sagemaker_endpoints.json"

        config = {
            "endpoint_name": self.endpoint_name,
            "model_name": model_name,
            "training_job": self.training_job_name,
            "model_accuracy": 0.9798,
            "features": 79,
            "region": self.region,
            "updated_at": datetime.now().isoformat(),
            "model_type": "EnhancedXDRThreatDetector",
            "capabilities": ["attention", "uncertainty_quantification", "skip_connections"],
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

    def deploy(self):
        """Execute full deployment pipeline"""
        logger.info("=" * 60)
        logger.info("🚀 DEPLOYING ENHANCED SAGEMAKER MODEL")
        logger.info("=" * 60)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            try:
                # Step 1: Download model from S3
                logger.info("\n📥 Step 1: Downloading model from S3...")
                model_archive = self.download_model_from_s3(temp_path)

                # Step 2: Extract model
                logger.info("\n📦 Step 2: Extracting model...")
                extract_dir = temp_path / "model_extracted"
                extract_dir.mkdir()
                self.extract_model(model_archive, extract_dir)

                # Step 3: Add inference script
                logger.info("\n📝 Step 3: Adding inference script...")
                self.add_inference_script(extract_dir)

                # Step 4: Repackage model
                logger.info("\n📦 Step 4: Repackaging model...")
                new_archive = temp_path / "model_packaged.tar.gz"
                self.repackage_model(extract_dir, new_archive)

                # Step 5: Upload to S3
                logger.info("\n☁️  Step 5: Uploading to S3...")
                model_data_url = self.upload_to_s3(new_archive)

                # Step 6: Create SageMaker model
                logger.info("\n🤖 Step 6: Creating SageMaker model...")
                model_name = self.create_sagemaker_model(model_data_url)

                # Step 7: Update endpoint
                logger.info("\n🔄 Step 7: Updating endpoint...")
                config_name = self.update_endpoint(model_name)

                # Step 8: Update config file
                logger.info("\n💾 Step 8: Updating config file...")
                self.update_config_file(model_name)

                logger.info("\n" + "=" * 60)
                logger.info("✅ DEPLOYMENT COMPLETE!")
                logger.info("=" * 60)
                logger.info(f"Endpoint: {self.endpoint_name}")
                logger.info(f"Model: {model_name}")
                logger.info(f"Training Job: {self.training_job_name}")
                logger.info(f"Model Data: {model_data_url}")
                logger.info(f"Accuracy: 97.98%")
                logger.info("=" * 60)

                return {
                    "success": True,
                    "endpoint_name": self.endpoint_name,
                    "model_name": model_name,
                    "model_data_url": model_data_url
                }

            except Exception as e:
                logger.error(f"\n❌ Deployment failed: {e}", exc_info=True)
                return {"success": False, "error": str(e)}


if __name__ == "__main__":
    deployer = EnhancedModelDeployer()
    result = deployer.deploy()
    exit(0 if result.get("success") else 1)