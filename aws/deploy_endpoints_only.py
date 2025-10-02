#!/usr/bin/env python3
"""
Deploy endpoints for existing SageMaker models
"""

import boto3
import time

REGION = 'us-east-1'

sagemaker = boto3.client('sagemaker', region_name=REGION)

# Existing models from previous deployment
ENDPOINTS = [
    {
        'model_name': 'mini-xdr-general-20250930-210140',
        'endpoint_name': 'mini-xdr-general-endpoint',
        'instance_type': 'ml.m5.large'
    },
    {
        'model_name': 'mini-xdr-ddos-20250930-210142',
        'endpoint_name': 'mini-xdr-ddos-specialist',
        'instance_type': 'ml.t2.medium'
    },
    {
        'model_name': 'mini-xdr-bruteforce-20250930-210144',
        'endpoint_name': 'mini-xdr-bruteforce-specialist',
        'instance_type': 'ml.t2.medium'
    },
    {
        'model_name': 'mini-xdr-webattack-20250930-210146',
        'endpoint_name': 'mini-xdr-webattack-specialist',
        'instance_type': 'ml.t2.medium'
    }
]

def deploy_endpoint(model_name, endpoint_name, instance_type):
    """Deploy endpoint for existing model"""
    print(f"\n🚀 Deploying: {endpoint_name}")
    print(f"   Model: {model_name}")
    print(f"   Instance: {instance_type}")

    timestamp = int(time.time())
    config_name = f'{endpoint_name}-config-{timestamp}'

    # Create endpoint configuration
    print(f"   Creating configuration...")
    try:
        sagemaker.create_endpoint_configuration(
            EndpointConfigName=config_name,
            ProductionVariants=[{
                'VariantName': 'primary',
                'ModelName': model_name,
                'InstanceType': instance_type,
                'InitialInstanceCount': 1
            }]
        )
        print(f"   ✅ Configuration created: {config_name}")
    except Exception as e:
        print(f"   ❌ Config failed: {e}")
        return False

    # Deploy or update endpoint
    print(f"   Deploying endpoint...")
    try:
        # Check if endpoint exists
        sagemaker.describe_endpoint(EndpointName=endpoint_name)
        print(f"   Updating existing endpoint...")
        sagemaker.update_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=config_name
        )
    except sagemaker.exceptions.ClientError:
        print(f"   Creating new endpoint...")
        sagemaker.create_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=config_name
        )

    # Wait for endpoint
    print(f"   Waiting for endpoint (this takes 5-10 min)...")
    waiter = sagemaker.get_waiter('endpoint_in_service')
    try:
        waiter.wait(EndpointName=endpoint_name, WaiterConfig={'MaxAttempts': 60})
        print(f"   ✅ Endpoint deployed: {endpoint_name}")
        return True
    except Exception as e:
        print(f"   ❌ Deployment failed: {e}")
        return False

def main():
    print("=" * 60)
    print("🚀 DEPLOYING SAGEMAKER ENDPOINTS")
    print("=" * 60)

    deployed = []
    failed = []

    for config in ENDPOINTS:
        success = deploy_endpoint(
            config['model_name'],
            config['endpoint_name'],
            config['instance_type']
        )
        if success:
            deployed.append(config['endpoint_name'])
        else:
            failed.append(config['endpoint_name'])

    # Summary
    print("\n" + "=" * 60)
    print("📊 DEPLOYMENT SUMMARY")
    print("=" * 60)
    print(f"✅ Deployed: {len(deployed)}")
    for name in deployed:
        print(f"   - {name}")
    if failed:
        print(f"\n❌ Failed: {len(failed)}")
        for name in failed:
            print(f"   - {name}")

if __name__ == '__main__':
    main()
