#!/usr/bin/env python3
"""
🚀 SIMPLE ENHANCED SAGEMAKER TRAINING LAUNCHER
Direct launch of enhanced training job without interactive prompts
"""

import boto3
from datetime import datetime
from sagemaker.pytorch import PyTorch
from sagemaker.inputs import TrainingInput
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def launch_training():
    """Launch SageMaker training job directly"""

    # Configuration
    role = 'arn:aws:iam::675076709589:role/SageMakerExecutionRole'
    bucket = 'mini-xdr-ml-data-bucket-675076709589'
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    training_job_name = f'enhanced-xdr-{timestamp}'

    logger.info(f"🚀 Launching training job: {training_job_name}")

    # Training paths
    s3_input_path = f's3://{bucket}/data/train'
    s3_output_path = f's3://{bucket}/enhanced_models/'

    # Enhanced PyTorch estimator
    estimator = PyTorch(
        entry_point='enhanced_sagemaker_train.py',
        source_dir='.',
        role=role,
        instance_type='ml.p3.2xlarge',
        instance_count=1,
        framework_version='2.0.1',
        py_version='py310',
        job_name=training_job_name,
        output_path=s3_output_path,
        volume_size=100,
        max_run=14400,  # 4 hours max
        hyperparameters={
            'hidden-dims': '512,256,128,64',
            'num-classes': 7,
            'dropout-rate': 0.3,
            'use-attention': True,
            'batch-size': 256,
            'epochs': 50,
            'learning-rate': 0.001,
            'patience': 15,
            'use-cuda': True
        },
        environment={
            'SM_MODEL_DIR': '/opt/ml/model',
            'SM_OUTPUT_DIR': '/opt/ml/output',
            'SM_CHANNEL_TRAINING': '/opt/ml/input/data/training'
        },
        checkpoint_s3_uri=f's3://{bucket}/checkpoints/{training_job_name}/',
        use_spot_instances=False,
        enable_sagemaker_metrics=True,
        metric_definitions=[
            {'Name': 'train:accuracy', 'Regex': 'Train Acc: ([0-9\\.]+)'},
            {'Name': 'test:accuracy', 'Regex': 'Test Acc: ([0-9\\.]+)'},
            {'Name': 'test:f1_score', 'Regex': 'F1: ([0-9\\.]+)'},
            {'Name': 'train:loss', 'Regex': 'Train Loss: ([0-9\\.]+)'}
        ]
    )

    # Prepare training input
    training_input = TrainingInput(
        s3_data=s3_input_path,
        content_type='text/csv',
        s3_data_type='S3Prefix'
    )

    logger.info(f"📊 Input: {s3_input_path}")
    logger.info(f"💾 Output: {s3_output_path}")
    logger.info(f"💎 Instance: ml.p3.2xlarge (V100 GPU)")

    try:
        # Launch training (don't wait)
        estimator.fit({'training': training_input}, wait=False)

        logger.info("✅ Training job launched successfully!")
        logger.info(f"📝 Job name: {training_job_name}")
        logger.info(f"🔗 Console: https://console.aws.amazon.com/sagemaker/home#/jobs/{training_job_name}")

        return training_job_name

    except Exception as e:
        logger.error(f"❌ Training launch failed: {e}")
        return None

if __name__ == '__main__':
    job_name = launch_training()
    if job_name:
        print(f"\n🎯 TRAINING JOB LAUNCHED: {job_name}")
        print(f"Monitor at: https://console.aws.amazon.com/sagemaker/home#/jobs/{job_name}")
    else:
        print("❌ Training job launch failed")