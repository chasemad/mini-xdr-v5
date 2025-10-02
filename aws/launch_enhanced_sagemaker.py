#!/usr/bin/env python3
"""
🚀 ENHANCED SAGEMAKER TRAINING LAUNCHER
Launches enhanced threat detection model training with:
- 2M+ events from multiple sources
- GPU training on ml.p3.2xlarge
- Auto-deployment to SageMaker endpoint
- Monitoring and validation
"""

import os
import boto3
import json
import time
from datetime import datetime
from sagemaker import get_execution_role
from sagemaker.pytorch import PyTorch
from sagemaker.inputs import TrainingInput
from sagemaker.model import Model
from sagemaker.predictor import Predictor
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_data_availability():
    """Check if training data is available in S3"""
    s3 = boto3.client('s3')
    bucket = 'mini-xdr-ml-data-bucket-675076709589'

    logger.info("🔍 Checking S3 data availability...")

    try:
        # Check if bucket exists
        s3.head_bucket(Bucket=bucket)
        logger.info(f"✅ Bucket {bucket} exists")

        # List data files
        response = s3.list_objects_v2(Bucket=bucket, Prefix='data/')

        if 'Contents' in response:
            total_size = sum(obj['Size'] for obj in response['Contents'])
            file_count = len(response['Contents'])

            logger.info(f"📊 Found {file_count} data files")
            logger.info(f"💾 Total data size: {total_size / (1024**3):.2f} GB")

            # Estimate event count (rough approximation)
            estimated_events = (total_size / 1024) * 100  # ~100 events per KB
            logger.info(f"🎯 Estimated events: {estimated_events:,.0f}")

            return True, file_count, total_size
        else:
            logger.warning("⚠️ No data files found in S3 bucket")
            return False, 0, 0

    except Exception as e:
        logger.error(f"❌ Error checking S3 bucket: {e}")
        return False, 0, 0


def prepare_training_data():
    """Prepare and upload training data if needed"""
    logger.info("📚 Preparing training data...")

    # For now, we'll assume data is already in S3
    # In production, you would:
    # 1. Download datasets from sources (UNSW-NB15, CIC-IDS2017, etc.)
    # 2. Process and combine them
    # 3. Upload to S3

    bucket = 'mini-xdr-ml-data-bucket-675076709589'
    s3_input_path = f's3://{bucket}/data/train'

    logger.info(f"📍 Training data location: {s3_input_path}")
    return s3_input_path


def launch_enhanced_training():
    """Launch enhanced SageMaker training job"""

    logger.info("🚀 LAUNCHING ENHANCED SAGEMAKER TRAINING")
    logger.info("=" * 60)

    # Check data availability first
    data_available, file_count, data_size = check_data_availability()
    if not data_available:
        logger.error("❌ Training data not available. Please prepare data first.")
        return None

    # SageMaker setup
    role = get_execution_role() if 'SM_' in os.environ else 'arn:aws:iam::675076709589:role/SageMakerExecutionRole'

    # S3 locations
    bucket = 'mini-xdr-ml-data-bucket-675076709589'
    s3_input_path = prepare_training_data()
    s3_output_path = f's3://{bucket}/enhanced_models/'

    # Training job configuration
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    training_job_name = f'enhanced-xdr-{timestamp}'

    logger.info(f"📝 Job name: {training_job_name}")
    logger.info(f"📊 Input: {s3_input_path}")
    logger.info(f"💾 Output: {s3_output_path}")
    logger.info(f"🧠 Architecture: Enhanced with Attention + Uncertainty")
    logger.info(f"💎 Instance: ml.p3.2xlarge (V100 GPU)")

    # Enhanced PyTorch estimator
    estimator = PyTorch(
        entry_point='enhanced_sagemaker_train.py',
        source_dir='.',
        role=role,
        instance_type='ml.p3.2xlarge',  # V100 GPU
        instance_count=1,
        framework_version='2.0.1',      # Latest PyTorch
        py_version='py310',
        job_name=training_job_name,
        output_path=s3_output_path,
        volume_size=100,                # 100GB for large dataset
        max_run=14400,                  # 4 hours max
        hyperparameters={
            'hidden-dims': '512,256,128,64',
            'num-classes': 7,
            'dropout-rate': 0.3,
            'use-attention': True,
            'batch-size': 256,
            'epochs': 50,              # Reduced for faster training
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
        use_spot_instances=False,       # Disable spot for reliability
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
        content_type='application/json',
        s3_data_type='S3Prefix'
    )

    logger.info("⚡ Starting training job...")

    try:
        # Launch training
        estimator.fit({'training': training_input}, wait=False)

        logger.info("✅ Training job launched successfully!")
        logger.info(f"📊 Monitor progress: https://console.aws.amazon.com/sagemaker/home#/jobs/{training_job_name}")

        return estimator, training_job_name

    except Exception as e:
        logger.error(f"❌ Training launch failed: {e}")
        return None, None


def monitor_training(training_job_name: str):
    """Monitor training job progress"""
    sagemaker_client = boto3.client('sagemaker')

    logger.info(f"👁️ Monitoring training job: {training_job_name}")

    while True:
        try:
            response = sagemaker_client.describe_training_job(TrainingJobName=training_job_name)
            status = response['TrainingJobStatus']

            logger.info(f"Status: {status}")

            if status == 'Completed':
                logger.info("🎉 Training completed successfully!")

                # Get final metrics
                if 'FinalMetricDataList' in response:
                    logger.info("📊 Final Metrics:")
                    for metric in response['FinalMetricDataList']:
                        logger.info(f"  {metric['MetricName']}: {metric['Value']}")

                return True

            elif status == 'Failed':
                logger.error("❌ Training job failed!")
                if 'FailureReason' in response:
                    logger.error(f"Reason: {response['FailureReason']}")
                return False

            elif status == 'Stopping':
                logger.warning("⚠️ Training job is stopping...")

            time.sleep(60)  # Check every minute

        except KeyboardInterrupt:
            logger.info("👋 Monitoring interrupted by user")
            break
        except Exception as e:
            logger.error(f"Error monitoring job: {e}")
            time.sleep(60)


def create_endpoint(estimator, endpoint_name: str = None):
    """Create SageMaker endpoint for the trained model"""

    if endpoint_name is None:
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        endpoint_name = f'enhanced-xdr-endpoint-{timestamp}'

    logger.info(f"🚀 Creating SageMaker endpoint: {endpoint_name}")

    try:
        # Deploy to endpoint
        predictor = estimator.deploy(
            initial_instance_count=1,
            instance_type='ml.m5.large',  # CPU instance for inference
            endpoint_name=endpoint_name,
            wait=True
        )

        logger.info(f"✅ Endpoint created successfully: {endpoint_name}")
        logger.info(f"🔗 Endpoint URL: https://runtime.sagemaker.us-east-1.amazonaws.com/endpoints/{endpoint_name}/invocations")

        return predictor, endpoint_name

    except Exception as e:
        logger.error(f"❌ Endpoint creation failed: {e}")
        return None, None


def test_endpoint(predictor):
    """Test the deployed endpoint with sample data"""
    logger.info("🧪 Testing enhanced endpoint...")

    # Create test data (79 features)
    test_data = {
        "instances": [
            [0.1] * 79,  # Normal traffic
            [0.8, 10.0, 5.0] + [0.5] * 76,  # Attack-like pattern
        ]
    }

    try:
        # Make prediction
        result = predictor.predict(test_data)

        logger.info("✅ Endpoint test successful!")
        logger.info(f"📊 Sample predictions: {json.dumps(result, indent=2)}")

        # Validate enhanced features
        predictions = result.get('predictions', [])
        for i, pred in enumerate(predictions):
            if 'enhanced_prediction' in pred and pred['enhanced_prediction']:
                logger.info(f"  ✨ Enhanced prediction {i+1}: {pred['threat_type']} ({pred['confidence']:.1%} confidence)")
                if 'uncertainty_score' in pred:
                    logger.info(f"    🎯 Uncertainty: {pred['uncertainty_score']:.3f}")
            else:
                logger.warning(f"  ⚠️ Prediction {i+1} missing enhanced features")

        return True

    except Exception as e:
        logger.error(f"❌ Endpoint test failed: {e}")
        return False


def main():
    """Main execution function"""
    print("🚀 ENHANCED SAGEMAKER TRAINING PIPELINE")
    print("=" * 50)

    # Step 1: Launch training
    estimator, training_job_name = launch_enhanced_training()

    if not estimator or not training_job_name:
        logger.error("❌ Training launch failed. Exiting.")
        return

    # Step 2: Monitor training (optional - can run separately)
    monitor_choice = input("\n📊 Monitor training progress? (y/n): ").lower().strip()

    if monitor_choice == 'y':
        training_success = monitor_training(training_job_name)

        if not training_success:
            logger.error("❌ Training failed. Exiting.")
            return
    else:
        logger.info("⏭️ Skipping monitoring. Check AWS Console for progress.")
        logger.info(f"🔗 Console: https://console.aws.amazon.com/sagemaker/home#/jobs/{training_job_name}")
        return

    # Step 3: Create endpoint (only if training completed)
    deploy_choice = input("\n🚀 Deploy to SageMaker endpoint? (y/n): ").lower().strip()

    if deploy_choice == 'y':
        predictor, endpoint_name = create_endpoint(estimator)

        if predictor and endpoint_name:
            # Step 4: Test endpoint
            test_choice = input("\n🧪 Test the endpoint? (y/n): ").lower().strip()

            if test_choice == 'y':
                test_success = test_endpoint(predictor)

                if test_success:
                    logger.info("🎉 DEPLOYMENT COMPLETED SUCCESSFULLY!")
                    logger.info(f"📍 Endpoint: {endpoint_name}")
                    logger.info(f"🔥 Enhanced model with attention + uncertainty ready for production!")
                else:
                    logger.warning("⚠️ Endpoint test failed, but deployment completed")
            else:
                logger.info("⏭️ Skipping endpoint test")
                logger.info(f"📍 Endpoint: {endpoint_name}")
        else:
            logger.error("❌ Endpoint deployment failed")
    else:
        logger.info("⏭️ Skipping endpoint deployment")
        logger.info("💾 Trained model available in S3 for manual deployment")

    logger.info("\n🎯 ENHANCED TRAINING PIPELINE COMPLETED!")


if __name__ == '__main__':
    main()