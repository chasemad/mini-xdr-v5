#!/usr/bin/env python3
"""
Launch all 4 model training jobs on SageMaker with GPU
"""

import boto3
import json
import time
from datetime import datetime

# AWS Configuration
REGION = 'us-east-1'
ACCOUNT_ID = '675076709589'
BUCKET = f'mini-xdr-ml-data-bucket-{ACCOUNT_ID}'
ROLE_ARN = f'arn:aws:iam::{ACCOUNT_ID}:role/SageMakerExecutionRole-MiniXDR'

# Training instance - V100 GPU approved!
TRAINING_INSTANCE = 'ml.p3.8xlarge'  # 4x V100 GPUs, 32 vCPUs, 244GB RAM, $12.24/hr

sagemaker = boto3.client('sagemaker', region_name=REGION)

MODELS = {
    'general': {
        'epochs': 50,
        'specialist_type': 'general',
        'description': 'General 7-class threat detector'
    },
    'ddos': {
        'epochs': 30,
        'specialist_type': 'ddos',
        'description': 'DDoS specialist binary classifier'
    },
    'bruteforce': {
        'epochs': 30,
        'specialist_type': 'brute_force',
        'description': 'Brute force specialist binary classifier'
    },
    'webattack': {
        'epochs': 30,
        'specialist_type': 'web_attacks',
        'description': 'Web attack specialist binary classifier'
    }
}


def upload_training_script():
    """Check if training script exists on S3"""
    print("\n📦 Using existing training script on S3...")
    s3_uri = f's3://{BUCKET}/training/sourcedir.tar.gz'
    print(f"   ✅ Using: {s3_uri}")
    return s3_uri


def launch_training_job(model_name, config):
    """Launch a SageMaker training job"""

    print(f"\n🚀 Launching {model_name} training job...")
    print(f"   {config['description']}")

    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    job_name = f'mini-xdr-{model_name}-{timestamp}'

    # Training data location
    training_data_s3 = f's3://{BUCKET}/data/comprehensive-train/'
    output_s3 = f's3://{BUCKET}/models/{model_name}/'

    # Training job configuration
    training_params = {
        'TrainingJobName': job_name,
        'RoleArn': ROLE_ARN,
        'AlgorithmSpecification': {
            'TrainingImage': f'763104351884.dkr.ecr.{REGION}.amazonaws.com/pytorch-training:2.1.0-gpu-py310',
            'TrainingInputMode': 'File',
            'EnableSageMakerMetricsTimeSeries': True
        },
        'InputDataConfig': [
            {
                'ChannelName': 'training',
                'DataSource': {
                    'S3DataSource': {
                        'S3DataType': 'S3Prefix',
                        'S3Uri': training_data_s3,
                        'S3DataDistributionType': 'FullyReplicated'
                    }
                },
                'ContentType': 'text/csv',
                'CompressionType': 'None'
            }
        ],
        'OutputDataConfig': {
            'S3OutputPath': output_s3
        },
        'ResourceConfig': {
            'InstanceType': TRAINING_INSTANCE,
            'InstanceCount': 1,
            'VolumeSizeInGB': 50
        },
        'StoppingCondition': {
            'MaxRuntimeInSeconds': 7200  # 2 hours max
        },
        'HyperParameters': {
            'specialist-type': config['specialist_type'],
            'epochs': str(config['epochs']),
            'batch-size': '512',
            'learning-rate': '0.001',
            'sagemaker_program': 'train.py',
            'sagemaker_submit_directory': f's3://{BUCKET}/training/sourcedir.tar.gz'
        },
        'Tags': [
            {'Key': 'Project', 'Value': 'MiniXDR'},
            {'Key': 'ModelType', 'Value': model_name},
            {'Key': 'FixedScaler', 'Value': 'true'}
        ]
    }

    try:
        response = sagemaker.create_training_job(**training_params)
        print(f"   ✅ Training job created: {job_name}")
        print(f"   📊 Instance: {TRAINING_INSTANCE}")
        print(f"   ⏱️  Max runtime: 2 hours")
        print(f"   💰 Estimated cost: ~$1.50")

        return {
            'job_name': job_name,
            'model_name': model_name,
            'status': 'InProgress'
        }

    except Exception as e:
        print(f"   ❌ Failed to create training job: {e}")
        return None


def monitor_training_jobs(jobs):
    """Monitor all training jobs"""

    print("\n" + "=" * 60)
    print("📊 MONITORING TRAINING JOBS")
    print("=" * 60)

    while True:
        all_complete = True

        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Status:")

        for job_info in jobs:
            if not job_info:
                continue

            try:
                response = sagemaker.describe_training_job(
                    TrainingJobName=job_info['job_name']
                )

                status = response['TrainingJobStatus']
                model = job_info['model_name']

                if status == 'InProgress':
                    all_complete = False
                    # Get training metrics if available
                    if 'SecondaryStatusTransitions' in response:
                        last_update = response['SecondaryStatusTransitions'][-1]
                        substatus = last_update.get('Status', 'Training')
                        print(f"   {model:15} | 🔄 {status:15} | {substatus}")
                    else:
                        print(f"   {model:15} | 🔄 {status:15}")

                elif status == 'Completed':
                    training_time = response.get('TrainingTimeInSeconds', 0)
                    print(f"   {model:15} | ✅ {status:15} | {training_time//60}min {training_time%60}s")

                elif status in ['Failed', 'Stopped']:
                    failure_reason = response.get('FailureReason', 'Unknown')
                    print(f"   {model:15} | ❌ {status:15} | {failure_reason}")

            except Exception as e:
                print(f"   {job_info['model_name']:15} | ❌ Error checking status: {e}")

        if all_complete:
            break

        time.sleep(60)  # Check every minute

    print("\n" + "=" * 60)
    print("✅ ALL TRAINING JOBS COMPLETE")
    print("=" * 60)

    # Return summary
    summary = []
    for job_info in jobs:
        if not job_info:
            continue
        try:
            response = sagemaker.describe_training_job(
                TrainingJobName=job_info['job_name']
            )
            summary.append({
                'model': job_info['model_name'],
                'status': response['TrainingJobStatus'],
                'model_artifacts': response.get('ModelArtifacts', {}).get('S3ModelArtifacts'),
                'training_time': response.get('TrainingTimeInSeconds')
            })
        except:
            pass

    return summary


def main():
    print("=" * 60)
    print("🚀 SAGEMAKER GPU TRAINING LAUNCHER")
    print("=" * 60)
    print(f"\n📍 Region: {REGION}")
    print(f"💻 Instance: {TRAINING_INSTANCE} (Tesla T4 GPU)")
    print(f"📦 Training data: s3://{BUCKET}/data/comprehensive-train/")
    print(f"💰 Estimated total cost: ~$6 for all 4 models")

    # Step 1: Upload training script
    upload_training_script()

    # Step 2: Launch all training jobs
    print("\n" + "=" * 60)
    print("🚀 LAUNCHING TRAINING JOBS")
    print("=" * 60)

    jobs = []
    for model_name, config in MODELS.items():
        job = launch_training_job(model_name, config)
        jobs.append(job)
        time.sleep(2)  # Small delay between launches

    # Step 3: Monitor progress
    summary = monitor_training_jobs(jobs)

    # Step 4: Print summary
    print("\n📋 TRAINING SUMMARY:")
    for item in summary:
        print(f"\n{item['model'].upper()}:")
        print(f"  Status: {item['status']}")
        print(f"  Time: {item.get('training_time', 0) // 60}min")
        print(f"  Artifacts: {item.get('model_artifacts', 'N/A')}")

    print("\n✅ Next steps:")
    print("  1. Download model artifacts from S3")
    print("  2. Deploy models using deploy_all_models.py")
    print("  3. Test endpoints with test_with_real_data.py")


if __name__ == "__main__":
    main()
