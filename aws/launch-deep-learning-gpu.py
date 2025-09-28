#!/usr/bin/env python3
"""
Launch Deep Learning PyTorch GPU Training on SageMaker
Uses proper neural networks with full dataset
"""

import os
import boto3
import json
from datetime import datetime
from sagemaker import get_execution_role
from sagemaker.pytorch import PyTorch

def launch_deep_learning_training():
    """Launch deep learning training with proper neural networks"""

    # SageMaker setup
    sagemaker_session = boto3.Session().client('sagemaker')
    role = get_execution_role() if 'SM_' in os.environ else 'arn:aws:iam::675076709589:role/SageMakerExecutionRole'

    # S3 locations
    bucket = 'mini-xdr-ml-data-bucket-675076709589'
    s3_input_path = f's3://{bucket}/data/train'
    s3_output_path = f's3://{bucket}/models/'

    # Training job name with timestamp
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    training_job_name = f'mini-xdr-deep-learning-{timestamp}'

    print("🚀 LAUNCHING DEEP LEARNING TRAINING")
    print("=" * 50)
    print(f"📝 Job name: {training_job_name}")
    print(f"📊 Input: {s3_input_path}")
    print(f"💾 Output: {s3_output_path}")
    print(f"🧠 Architecture: Multi-Model Deep Learning")
    print(f"💎 Instance: ml.p3.2xlarge (1x V100 GPU) - NCCL workaround")

    # PyTorch estimator with proper deep learning
    estimator = PyTorch(
        entry_point='pytorch_deep_learning_train.py',
        source_dir='.',
        role=role,
        instance_type='ml.p3.2xlarge',  # 1x V100 GPU - avoids NCCL issues
        instance_count=1,
        framework_version='1.12.0',
        py_version='py38',
        job_name=training_job_name,
        output_path=s3_output_path,
        volume_size=50,  # Increased for full dataset
        max_run=7200,    # 2 hours max
        use_spot_instances=True,
        max_wait=14400,  # 4 hours max wait
        checkpoint_s3_uri=f's3://{bucket}/checkpoints/{training_job_name}/',
        hyperparameters={
            'epochs': 25,         # Good balance for 4 GPUs
            'batch-size': 128,    # Larger batch for single GPU with gradient accumulation
            'max-samples': 200000 # Reasonable dataset size
        },
        environment={
            'SAGEMAKER_ENABLE_CLOUDWATCH_METRICS': 'false',
            'CUDA_LAUNCH_BLOCKING': '1',  # Synchronous CUDA for better error reporting
            'PYTORCH_CUDA_ALLOC_CONF': 'max_split_size_mb:512'  # Conservative memory allocation
        },
        tags=[
            {'Key': 'Project', 'Value': 'Mini-XDR'},
            {'Key': 'Type', 'Value': 'DeepLearning'},
            {'Key': 'Architecture', 'Value': 'MultiModel'},
            {'Key': 'Dataset', 'Value': 'CICIDS2017-Full'}
        ]
    )

    # Launch training
    print("🔥 Starting deep learning training...")
    print("🧠 Models: Deep Threat Detector + Anomaly Autoencoder")
    print("📈 Full dataset training with efficient batching")
    print("⚡ GPU-accelerated with 1x V100 (NCCL workaround + gradient accumulation)")

    try:
        estimator.fit({
            'training': s3_input_path
        })

        print("✅ DEEP LEARNING TRAINING LAUNCHED SUCCESSFULLY!")
        print(f"🔗 Job ARN: {estimator.latest_training_job.job_arn}")
        print(f"📊 Monitor at: https://console.aws.amazon.com/sagemaker/home#/jobs/{training_job_name}")

        # Save job info
        job_info = {
            'job_name': training_job_name,
            'job_arn': estimator.latest_training_job.job_arn,
            'model_output': f'{s3_output_path}{training_job_name}/output/model.tar.gz',
            'training_type': 'deep_learning',
            'architecture': 'multi_model_neural_networks',
            'instance_type': 'ml.p3.2xlarge',
            'gpu_count': 1,
            'dataset_type': 'full_cicids2017',
            'timestamp': timestamp
        }

        with open(f'/tmp/deep_learning_job_info_{timestamp}.json', 'w') as f:
            json.dump(job_info, f, indent=2)

        print(f"💾 Job info saved to: /tmp/deep_learning_job_info_{timestamp}.json")

    except Exception as e:
        print(f"❌ TRAINING LAUNCH FAILED: {e}")
        raise

    return training_job_name

if __name__ == '__main__':
    launch_deep_learning_training()