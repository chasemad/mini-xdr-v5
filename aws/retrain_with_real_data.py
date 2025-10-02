#!/usr/bin/env python3
"""
Retrain all 4 models using the REAL training data
Previously trained on synthetic 'comprehensive-train' data (too easy, 100% accuracy)
Now training on 'full-training' data (real attack patterns)
"""

import boto3
import time
from datetime import datetime

REGION = 'us-east-1'
ACCOUNT_ID = '675076709589'
BUCKET = f'mini-xdr-ml-data-bucket-{ACCOUNT_ID}'
ROLE_ARN = f'arn:aws:iam::{ACCOUNT_ID}:role/SageMakerExecutionRole-MiniXDR'

# Use the REAL training data (not synthetic)
TRAINING_DATA = f's3://{BUCKET}/data/full-training/'

# ECR image for PyTorch training
ECR_IMAGE = f'763104351884.dkr.ecr.{REGION}.amazonaws.com/pytorch-training:2.1.0-gpu-py310'

sagemaker = boto3.client('sagemaker', region_name=REGION)

MODELS = [
    {
        'name': 'general',
        'specialist_type': 'general',
        'output_path': f's3://{BUCKET}/models/general-v2/',
        'epochs': 25
    },
    {
        'name': 'ddos',
        'specialist_type': 'ddos',
        'output_path': f's3://{BUCKET}/models/ddos-v2/',
        'epochs': 30
    },
    {
        'name': 'bruteforce',
        'specialist_type': 'brute_force',
        'output_path': f's3://{BUCKET}/models/bruteforce-v2/',
        'epochs': 30
    },
    {
        'name': 'webattack',
        'specialist_type': 'web_attacks',
        'output_path': f's3://{BUCKET}/models/webattack-v2/',
        'epochs': 30
    }
]


def launch_training_job(config):
    """Launch a SageMaker training job"""

    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    job_name = f"mini-xdr-{config['name']}-v2-{timestamp}"

    print(f"\n{'='*70}")
    print(f"🚀 Launching: {config['name'].upper()} ({config['specialist_type']})")
    print(f"{'='*70}")
    print(f"   Job name: {job_name}")
    print(f"   Data: {TRAINING_DATA}")
    print(f"   Epochs: {config['epochs']}")

    try:
        response = sagemaker.create_training_job(
            TrainingJobName=job_name,
            RoleArn=ROLE_ARN,
            AlgorithmSpecification={
                'TrainingImage': ECR_IMAGE,
                'TrainingInputMode': 'File'
            },
            InputDataConfig=[
                {
                    'ChannelName': 'training',
                    'DataSource': {
                        'S3DataSource': {
                            'S3DataType': 'S3Prefix',
                            'S3Uri': TRAINING_DATA,
                            'S3DataDistributionType': 'FullyReplicated'
                        }
                    }
                }
            ],
            OutputDataConfig={
                'S3OutputPath': config['output_path']
            },
            ResourceConfig={
                'InstanceType': 'ml.p3.8xlarge',  # GPU for faster training
                'InstanceCount': 1,
                'VolumeSizeInGB': 50
            },
            StoppingCondition={
                'MaxRuntimeInSeconds': 7200  # 2 hours max
            },
            HyperParameters={
                'specialist-type': config['specialist_type'],
                'epochs': str(config['epochs']),
                'batch-size': '256',
                'learning-rate': '0.001',
                'sagemaker_program': 'sagemaker_train.py',
                'sagemaker_submit_directory': f's3://{BUCKET}/code/sagemaker_train.tar.gz'
            }
        )

        print(f"   ✅ Job launched: {job_name}")
        return job_name

    except Exception as e:
        print(f"   ❌ Failed to launch: {e}")
        return None


def wait_for_job(job_name):
    """Wait for training job to complete"""

    print(f"\n⏳ Monitoring {job_name}...")

    while True:
        try:
            response = sagemaker.describe_training_job(TrainingJobName=job_name)
            status = response['TrainingJobStatus']

            if status == 'Completed':
                training_time = response.get('TrainingTimeInSeconds', 0)
                metrics = response.get('FinalMetricDataList', [])
                print(f"   ✅ COMPLETED in {training_time}s")
                if metrics:
                    for metric in metrics:
                        print(f"      {metric.get('MetricName')}: {metric.get('Value')}")
                return True

            elif status in ['Failed', 'Stopped']:
                failure_reason = response.get('FailureReason', 'Unknown')
                print(f"   ❌ {status}: {failure_reason}")
                return False

            elif status == 'InProgress':
                # Check for metrics
                secondary_status = response.get('SecondaryStatus', '')
                print(f"   🔄 {secondary_status}...", end='\r')
                time.sleep(30)
            else:
                print(f"   🔄 {status}...", end='\r')
                time.sleep(30)

        except Exception as e:
            print(f"   ❌ Error checking status: {e}")
            return False


print("="*70)
print("🔄 RETRAINING ALL MODELS WITH REAL DATA")
print("="*70)
print(f"Training data: {TRAINING_DATA}")
print(f"Models to train: {len(MODELS)}")
print("="*70)

# Upload training script to S3
print("\n📦 Uploading training script to S3...")
import subprocess
subprocess.run([
    'tar', '-czf', '/tmp/sagemaker_train.tar.gz',
    '-C', 'aws', 'sagemaker_train.py'
], check=True)

subprocess.run([
    'aws', 's3', 'cp', '/tmp/sagemaker_train.tar.gz',
    f's3://{BUCKET}/code/sagemaker_train.tar.gz'
], check=True)
print("   ✅ Training script uploaded")

# Train models sequentially (to avoid quota issues)
completed_jobs = []
failed_jobs = []

for config in MODELS:
    job_name = launch_training_job(config)

    if job_name:
        success = wait_for_job(job_name)
        if success:
            completed_jobs.append(job_name)
        else:
            failed_jobs.append(job_name)
    else:
        failed_jobs.append(config['name'])

    print()  # Blank line between jobs

# Summary
print("\n" + "="*70)
print("📊 TRAINING SUMMARY")
print("="*70)
print(f"✅ Completed: {len(completed_jobs)}")
for job in completed_jobs:
    print(f"   - {job}")

if failed_jobs:
    print(f"\n❌ Failed: {len(failed_jobs)}")
    for job in failed_jobs:
        print(f"   - {job}")

print("\n💡 Next steps:")
print("   1. Download trained models from S3")
print("   2. Update endpoints with new models")
print("   3. Test with real attack data")
print("="*70)
