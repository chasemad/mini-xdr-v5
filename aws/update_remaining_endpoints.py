#!/usr/bin/env python3
"""
Update remaining 2 endpoints by deleting and recreating to avoid quota issues
"""

import boto3
import time
from datetime import datetime

REGION = 'us-east-1'
ACCOUNT_ID = '675076709589'
BUCKET = f'mini-xdr-ml-data-bucket-{ACCOUNT_ID}'
ROLE_ARN = f'arn:aws:iam::{ACCOUNT_ID}:role/SageMakerExecutionRole-MiniXDR'

sagemaker = boto3.client('sagemaker', region_name=REGION)

MODELS = {
    'bruteforce': {
        'model_data': f's3://{BUCKET}/models/bruteforce/mini-xdr-bruteforce-20250930-221433/output/model.tar.gz',
        'endpoint': 'mini-xdr-bruteforce-specialist',
        'instance': 'ml.t2.medium'  # Use cheaper instance
    },
    'webattack': {
        'model_data': f's3://{BUCKET}/models/webattack/mini-xdr-webattack-20250930-215040/output/model.tar.gz',
        'endpoint': 'mini-xdr-webattack-specialist',
        'instance': 'ml.t2.medium'  # Use cheaper instance
    }
}


def delete_endpoint(endpoint_name):
    """Delete endpoint"""
    print(f"🗑️  Deleting old endpoint: {endpoint_name}")
    try:
        sagemaker.delete_endpoint(EndpointName=endpoint_name)
        print(f"   ✅ Deleted: {endpoint_name}")
        time.sleep(5)
        return True
    except Exception as e:
        print(f"   ❌ Error deleting: {e}")
        return False


def create_model(model_name, model_data):
    """Create a new SageMaker model"""
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    model_full_name = f'mini-xdr-{model_name}-{timestamp}'

    print(f"\n📦 Creating model: {model_full_name}")

    try:
        sagemaker.create_model(
            ModelName=model_full_name,
            PrimaryContainer={
                'Image': f'763104351884.dkr.ecr.{REGION}.amazonaws.com/pytorch-inference:2.1.0-cpu-py310',
                'ModelDataUrl': model_data,
                'Environment': {
                    'SAGEMAKER_PROGRAM': 'inference.py',
                    'SAGEMAKER_SUBMIT_DIRECTORY': model_data,
                    'SAGEMAKER_REGION': REGION
                }
            },
            ExecutionRoleArn=ROLE_ARN
        )
        print(f"   ✅ Model created: {model_full_name}")
        return model_full_name
    except Exception as e:
        print(f"   ❌ Error creating model: {e}")
        return None


def create_endpoint(endpoint_name, model_name, instance_type):
    """Create new endpoint"""
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    config_name = f'{endpoint_name}-config-{timestamp}'

    print(f"\n⚙️  Creating endpoint config: {config_name}")

    try:
        # Create config
        sagemaker.create_endpoint_config(
            EndpointConfigName=config_name,
            ProductionVariants=[
                {
                    'VariantName': 'AllTraffic',
                    'ModelName': model_name,
                    'InitialInstanceCount': 1,
                    'InstanceType': instance_type,
                    'InitialVariantWeight': 1.0
                }
            ]
        )
        print(f"   ✅ Config created: {config_name}")

        # Create endpoint
        print(f"\n🚀 Creating endpoint: {endpoint_name}")
        sagemaker.create_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=config_name
        )
        print(f"   ✅ Endpoint creation initiated")

        return True
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def wait_for_endpoint(endpoint_name):
    """Wait for endpoint to be InService"""
    print(f"\n⏳ Waiting for {endpoint_name} to be InService...")

    while True:
        try:
            response = sagemaker.describe_endpoint(EndpointName=endpoint_name)
            status = response['EndpointStatus']

            if status == 'InService':
                print(f"   ✅ {endpoint_name} is InService")
                return True
            elif status in ['Failed', 'RollingBack']:
                print(f"   ❌ {endpoint_name} failed: {status}")
                return False
            else:
                print(f"   🔄 {status}...", end='\r')
                time.sleep(20)
        except Exception as e:
            print(f"   🔄 Creating...", end='\r')
            time.sleep(20)


print("=" * 60)
print("🚀 UPDATING REMAINING ENDPOINTS")
print("=" * 60)
print("Strategy: Delete old endpoints, then recreate with new models")
print("Instance type: ml.t2.medium (cheaper, simpler quota)")
print("=" * 60)

for model_name, config in MODELS.items():
    print(f"\n{'='*60}")
    print(f"📦 Processing: {model_name.upper()}")
    print(f"{'='*60}")

    # Step 1: Delete old endpoint
    delete_endpoint(config['endpoint'])

    # Step 2: Create new model
    sagemaker_model_name = create_model(model_name, config['model_data'])
    if not sagemaker_model_name:
        continue

    # Step 3: Create new endpoint
    success = create_endpoint(config['endpoint'], sagemaker_model_name, config['instance'])
    if not success:
        continue

    # Step 4: Wait for endpoint
    wait_for_endpoint(config['endpoint'])

print("\n" + "="*60)
print("✅ ALL ENDPOINTS UPDATED")
print("="*60)
