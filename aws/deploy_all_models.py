#!/usr/bin/env python3
"""
Deploy All Trained Models to SageMaker
Packages and deploys general + specialist models to production endpoints
"""

import boto3
import tarfile
import json
import time
from pathlib import Path
from datetime import datetime

# AWS Configuration
REGION = 'us-east-1'
ACCOUNT_ID = '675076709589'
BUCKET = f'mini-xdr-ml-data-bucket-{ACCOUNT_ID}'
ROLE_ARN = f'arn:aws:iam::{ACCOUNT_ID}:role/SageMakerExecutionRole-MiniXDR'

# Model configurations
MODELS = {
    'general': {
        'local_path': '/tmp/models/general',
        'endpoint_name': 'mini-xdr-general-endpoint',
        'instance_type': 'ml.m5.large',
        'description': 'General purpose 7-class threat detector'
    },
    'ddos': {
        'local_path': '/tmp/models/ddos',
        'endpoint_name': 'mini-xdr-ddos-specialist',
        'instance_type': 'ml.t2.medium',
        'description': 'DDoS attack specialist binary classifier'
    },
    'bruteforce': {
        'local_path': '/tmp/models/brute_force',
        'endpoint_name': 'mini-xdr-bruteforce-specialist',
        'instance_type': 'ml.t2.medium',
        'description': 'Brute force attack specialist binary classifier'
    },
    'webattack': {
        'local_path': '/tmp/models/web_attacks',
        'endpoint_name': 'mini-xdr-webattack-specialist',
        'instance_type': 'ml.t2.medium',
        'description': 'Web application attack specialist binary classifier'
    }
}

sagemaker = boto3.client('sagemaker', region_name=REGION)
s3 = boto3.client('s3', region_name=REGION)


def package_model(model_name, local_path):
    """Package model into tar.gz for SageMaker"""
    print(f"\n📦 Packaging {model_name} model...")

    model_dir = Path(local_path)
    tar_path = f'/tmp/{model_name}_model.tar.gz'

    with tarfile.open(tar_path, 'w:gz') as tar:
        # Add model files
        tar.add(model_dir / 'threat_detector.pth', arcname='threat_detector.pth')
        tar.add(model_dir / 'scaler.pkl', arcname='scaler.pkl')
        tar.add(model_dir / 'model_metadata.json', arcname='model_metadata.json')
        tar.add(model_dir / 'code' / 'inference.py', arcname='code/inference.py')

    print(f"   ✅ Created package: {tar_path}")
    return tar_path


def upload_to_s3(tar_path, model_name):
    """Upload model package to S3"""
    print(f"\n☁️  Uploading {model_name} to S3...")

    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    s3_key = f'models/{model_name}/{model_name}-{timestamp}.tar.gz'
    s3_uri = f's3://{BUCKET}/{s3_key}'

    s3.upload_file(tar_path, BUCKET, s3_key)
    print(f"   ✅ Uploaded to: {s3_uri}")

    return s3_uri


def create_sagemaker_model(model_name, s3_uri, description):
    """Create SageMaker model"""
    print(f"\n🤖 Creating SageMaker model: {model_name}...")

    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    model_name_full = f'mini-xdr-{model_name}-{timestamp}'

    # PyTorch inference container
    container_image = f'763104351884.dkr.ecr.{REGION}.amazonaws.com/pytorch-inference:2.0.1-cpu-py310'

    try:
        response = sagemaker.create_model(
            ModelName=model_name_full,
            PrimaryContainer={
                'Image': container_image,
                'ModelDataUrl': s3_uri,
                'Environment': {
                    'SAGEMAKER_PROGRAM': 'inference.py',
                    'SAGEMAKER_SUBMIT_DIRECTORY': s3_uri,
                    'SAGEMAKER_REGION': REGION
                }
            },
            ExecutionRoleArn=ROLE_ARN,
            Tags=[
                {'Key': 'Project', 'Value': 'mini-xdr'},
                {'Key': 'ModelType', 'Value': model_name},
                {'Key': 'Description', 'Value': description}
            ]
        )
        print(f"   ✅ Model created: {model_name_full}")
        return model_name_full
    except Exception as e:
        print(f"   ❌ Error creating model: {e}")
        raise


def deploy_endpoint(model_name_full, endpoint_name, instance_type):
    """Deploy model to endpoint"""
    print(f"\n🚀 Deploying endpoint: {endpoint_name}...")

    endpoint_config_name = f'{endpoint_name}-config-{int(time.time())}'

    # Create endpoint configuration
    print(f"   Creating endpoint configuration...")
    try:
        sagemaker.create_endpoint_configuration(
        EndpointConfigName=endpoint_config_name,
        ProductionVariants=[
            {
                'VariantName': 'primary',
                'ModelName': model_name_full,
                'InstanceType': instance_type,
                'InitialInstanceCount': 1,
                'InitialVariantWeight': 1.0
            }
        ],
        Tags=[
            {'Key': 'Project', 'Value': 'mini-xdr'},
            {'Key': 'Environment', 'Value': 'production'}
        ]
    )
    except sagemaker.exceptions.ClientError as e:
        if 'Cannot create already existing' in str(e):
            print(f"   Endpoint config already exists, continuing...")
        else:
            raise

    # Check if endpoint exists
    try:
        sagemaker.describe_endpoint(EndpointName=endpoint_name)
        print(f"   Updating existing endpoint...")
        sagemaker.update_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=endpoint_config_name
        )
    except sagemaker.exceptions.ClientError:
        print(f"   Creating new endpoint...")
        sagemaker.create_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=endpoint_config_name,
            Tags=[
                {'Key': 'Project', 'Value': 'mini-xdr'},
                {'Key': 'Environment', 'Value': 'production'}
            ]
        )

    # Wait for endpoint
    print(f"   Waiting for endpoint to be InService...")
    waiter = sagemaker.get_waiter('endpoint_in_service')
    waiter.wait(EndpointName=endpoint_name)

    print(f"   ✅ Endpoint deployed: {endpoint_name}")
    return endpoint_name


def save_deployment_config(deployed_endpoints):
    """Save deployment configuration"""
    config_path = '/Users/chasemad/Desktop/mini-xdr/config/deployed_models.json'
    Path(config_path).parent.mkdir(parents=True, exist_ok=True)

    config = {
        'deployment_date': datetime.now().isoformat(),
        'region': REGION,
        'models': deployed_endpoints
    }

    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\n💾 Saved deployment config: {config_path}")


def main():
    print("=" * 60)
    print("🚀 MINI-XDR MODEL DEPLOYMENT")
    print("=" * 60)

    deployed_endpoints = {}

    for model_name, config in MODELS.items():
        print(f"\n{'=' * 60}")
        print(f"Deploying: {model_name.upper()}")
        print(f"{'=' * 60}")

        try:
            # Package model
            tar_path = package_model(model_name, config['local_path'])

            # Upload to S3
            s3_uri = upload_to_s3(tar_path, model_name)

            # Create SageMaker model
            model_name_full = create_sagemaker_model(
                model_name,
                s3_uri,
                config['description']
            )

            # Deploy endpoint
            endpoint_name = deploy_endpoint(
                model_name_full,
                config['endpoint_name'],
                config['instance_type']
            )

            # Save deployment info
            deployed_endpoints[model_name] = {
                'endpoint_name': endpoint_name,
                'model_name': model_name_full,
                'instance_type': config['instance_type'],
                's3_uri': s3_uri,
                'description': config['description']
            }

            print(f"\n✅ {model_name.upper()} DEPLOYED SUCCESSFULLY!")

        except Exception as e:
            print(f"\n❌ Failed to deploy {model_name}: {e}")
            continue

    # Save deployment configuration
    save_deployment_config(deployed_endpoints)

    # Summary
    print("\n" + "=" * 60)
    print("📊 DEPLOYMENT SUMMARY")
    print("=" * 60)

    for model_name, info in deployed_endpoints.items():
        print(f"\n{model_name.upper()}:")
        print(f"  Endpoint: {info['endpoint_name']}")
        print(f"  Instance: {info['instance_type']}")
        print(f"  Description: {info['description']}")

    print("\n" + "=" * 60)
    print("✅ ALL MODELS DEPLOYED!")
    print("=" * 60)


if __name__ == '__main__':
    main()
