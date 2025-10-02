#!/usr/bin/env python3
"""
Launch training jobs sequentially (quota allows only 1 at a time)
"""

import boto3
import time
from datetime import datetime

REGION = 'us-east-1'
ACCOUNT_ID = '675076709589'
BUCKET = f'mini-xdr-ml-data-bucket-{ACCOUNT_ID}'
ROLE_ARN = f'arn:aws:iam::{ACCOUNT_ID}:role/SageMakerExecutionRole-MiniXDR'
TRAINING_INSTANCE = 'ml.p3.8xlarge'

sagemaker = boto3.client('sagemaker', region_name=REGION)

MODELS = [
    ('ddos', 'ddos', 30),
    ('bruteforce', 'brute_force', 30),
    ('webattack', 'web_attacks', 30)
]


def launch_job(model_name, specialist_type, epochs):
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    job_name = f'mini-xdr-{model_name}-{timestamp}'

    print(f"\n🚀 Launching {model_name}...")

    params = {
        'TrainingJobName': job_name,
        'RoleArn': ROLE_ARN,
        'AlgorithmSpecification': {
            'TrainingImage': f'763104351884.dkr.ecr.{REGION}.amazonaws.com/pytorch-training:2.1.0-gpu-py310',
            'TrainingInputMode': 'File'
        },
        'InputDataConfig': [{
            'ChannelName': 'training',
            'DataSource': {
                'S3DataSource': {
                    'S3DataType': 'S3Prefix',
                    'S3Uri': f's3://{BUCKET}/data/comprehensive-train/',
                    'S3DataDistributionType': 'FullyReplicated'
                }
            },
            'ContentType': 'text/csv'
        }],
        'OutputDataConfig': {
            'S3OutputPath': f's3://{BUCKET}/models/{model_name}/'
        },
        'ResourceConfig': {
            'InstanceType': TRAINING_INSTANCE,
            'InstanceCount': 1,
            'VolumeSizeInGB': 50
        },
        'StoppingCondition': {
            'MaxRuntimeInSeconds': 3600
        },
        'HyperParameters': {
            'specialist-type': specialist_type,
            'epochs': str(epochs),
            'batch-size': '512',
            'learning-rate': '0.001',
            'sagemaker_program': 'train.py',
            'sagemaker_submit_directory': f's3://{BUCKET}/training/sourcedir.tar.gz'
        }
    }

    sagemaker.create_training_job(**params)
    print(f"   ✅ Created: {job_name}")

    # Wait for completion
    while True:
        response = sagemaker.describe_training_job(TrainingJobName=job_name)
        status = response['TrainingJobStatus']

        if status == 'InProgress':
            substatus = response.get('SecondaryStatus', 'Training')
            print(f"   🔄 {substatus}...", end='\r')
            time.sleep(30)
        elif status == 'Completed':
            training_time = response.get('TrainingTimeInSeconds', 0)
            print(f"\n   ✅ Completed in {training_time//60}min {training_time%60}s")
            return True
        elif status in ['Failed', 'Stopped']:
            failure = response.get('FailureReason', 'Unknown')
            print(f"\n   ❌ Failed: {failure}")
            return False
        else:
            time.sleep(30)


print("=" * 60)
print("🚀 SEQUENTIAL GPU TRAINING")
print("=" * 60)

for model_name, specialist_type, epochs in MODELS:
    success = launch_job(model_name, specialist_type, epochs)
    if not success:
        print(f"\n❌ Stopping - {model_name} failed")
        break
    time.sleep(5)  # Small delay between jobs

print("\n" + "=" * 60)
print("✅ ALL TRAINING COMPLETE")
print("=" * 60)
