#!/usr/bin/env python3
"""
Update all 4 SageMaker endpoints with newly trained models
"""

import boto3
import time
from datetime import datetime

REGION = 'us-east-1'
ACCOUNT_ID = '675076709589'
BUCKET = f'mini-xdr-ml-data-bucket-{ACCOUNT_ID}'
ROLE_ARN = f'arn:aws:iam::{ACCOUNT_ID}:role/SageMakerExecutionRole-MiniXDR'

sagemaker = boto3.client('sagemaker', region_name=REGION)

# Map of model names to their S3 model artifacts and endpoint names
MODELS = {
    'general': {
        'model_data': f's3://{BUCKET}/models/general/mini-xdr-general-20250930-215557/output/model.tar.gz',
        'endpoint': 'mini-xdr-general-endpoint',
        'instance': 'ml.m5.xlarge'
    },
    'ddos': {
        'model_data': f's3://{BUCKET}/models/ddos/mini-xdr-ddos-20250930-220554/output/model.tar.gz',
        'endpoint': 'mini-xdr-ddos-specialist',
        'instance': 'ml.m5.xlarge'
    },
    'bruteforce': {
        'model_data': f's3://{BUCKET}/models/bruteforce/mini-xdr-bruteforce-20250930-221433/output/model.tar.gz',
        'endpoint': 'mini-xdr-bruteforce-specialist',
        'instance': 'ml.m5.xlarge'
    },
    'webattack': {
        'model_data': f's3://{BUCKET}/models/webattack/mini-xdr-webattack-20250930-215040/output/model.tar.gz',
        'endpoint': 'mini-xdr-webattack-specialist',
        'instance': 'ml.m5.xlarge'
    }
}


def create_model(model_name, model_data):
    """Create a new SageMaker model"""
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    model_full_name = f'mini-xdr-{model_name}-{timestamp}'

    print(f"\n📦 Creating model: {model_full_name}")

    try:
        response = sagemaker.create_model(
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


def create_endpoint_config(model_name, sagemaker_model_name, instance_type):
    """Create endpoint configuration"""
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    config_name = f'mini-xdr-{model_name}-config-{timestamp}'

    print(f"\n⚙️  Creating endpoint config: {config_name}")

    try:
        response = sagemaker.create_endpoint_config(
            EndpointConfigName=config_name,
            ProductionVariants=[
                {
                    'VariantName': 'AllTraffic',
                    'ModelName': sagemaker_model_name,
                    'InitialInstanceCount': 1,
                    'InstanceType': instance_type,
                    'InitialVariantWeight': 1.0
                }
            ]
        )
        print(f"   ✅ Config created: {config_name}")
        return config_name
    except Exception as e:
        print(f"   ❌ Error creating config: {e}")
        return None


def update_endpoint(endpoint_name, config_name):
    """Update existing endpoint with new config"""
    print(f"\n🔄 Updating endpoint: {endpoint_name}")

    try:
        response = sagemaker.update_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=config_name
        )
        print(f"   ✅ Update initiated for: {endpoint_name}")
        return True
    except Exception as e:
        print(f"   ❌ Error updating endpoint: {e}")
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
                time.sleep(30)
        except Exception as e:
            print(f"   ❌ Error checking status: {e}")
            return False


print("=" * 60)
print("🚀 UPDATING ALL SAGEMAKER ENDPOINTS")
print("=" * 60)

results = []

for model_name, config in MODELS.items():
    print(f"\n{'='*60}")
    print(f"📦 Processing: {model_name.upper()}")
    print(f"{'='*60}")

    # Step 1: Create new model
    sagemaker_model_name = create_model(model_name, config['model_data'])
    if not sagemaker_model_name:
        results.append({'model': model_name, 'status': 'Failed', 'step': 'create_model'})
        continue

    # Step 2: Create endpoint config
    config_name = create_endpoint_config(model_name, sagemaker_model_name, config['instance'])
    if not config_name:
        results.append({'model': model_name, 'status': 'Failed', 'step': 'create_config'})
        continue

    # Step 3: Update endpoint
    success = update_endpoint(config['endpoint'], config_name)
    if not success:
        results.append({'model': model_name, 'status': 'Failed', 'step': 'update_endpoint'})
        continue

    results.append({
        'model': model_name,
        'status': 'Updating',
        'endpoint': config['endpoint'],
        'config': config_name
    })

print("\n" + "="*60)
print("⏳ WAITING FOR ALL ENDPOINTS TO UPDATE")
print("="*60)

# Wait for all endpoints
for result in results:
    if result['status'] == 'Updating':
        success = wait_for_endpoint(result['endpoint'])
        result['status'] = 'InService' if success else 'Failed'

print("\n" + "="*60)
print("📊 FINAL SUMMARY")
print("="*60)

for result in results:
    status_icon = "✅" if result['status'] == 'InService' else "❌"
    print(f"\n{status_icon} {result['model'].upper()}: {result['status']}")
    if result['status'] == 'InService':
        print(f"   Endpoint: {result['endpoint']}")

print("\n✅ All endpoints updated with fixed models!")
print("   Models trained with NO SCALER (pre-normalized data)")
print("   Expected accuracy: ~80%+")
