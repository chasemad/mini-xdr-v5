#!/usr/bin/env python3
"""
Monitor SageMaker Training Job Progress
Real-time monitoring for deep learning training
"""

import boto3
import time
import json
from datetime import datetime

def monitor_training_job(job_name: str = "pytorch-training-2025-09-27-13-19-58-208"):
    """Monitor the training job and download models when complete"""

    sagemaker = boto3.client('sagemaker')
    s3 = boto3.client('s3')

    print(f"🔍 Monitoring training job: {job_name}")
    print("=" * 60)

    while True:
        try:
            # Get job status
            response = sagemaker.describe_training_job(TrainingJobName=job_name)

            status = response['TrainingJobStatus']
            secondary_status = response.get('SecondaryStatus', 'Unknown')

            # Get timing info
            creation_time = response['CreationTime']
            last_modified = response['LastModifiedTime']

            if 'TrainingStartTime' in response:
                start_time = response['TrainingStartTime']
                elapsed = (datetime.now(start_time.tzinfo) - start_time).total_seconds() / 60
                print(f"⏱️  Status: {status} | {secondary_status} | Runtime: {elapsed:.1f} minutes")
            else:
                print(f"⏱️  Status: {status} | {secondary_status}")

            # Check for completion
            if status == 'Completed':
                print("🎉 TRAINING COMPLETED SUCCESSFULLY!")

                # Get model artifacts location
                model_artifacts = response['ModelArtifacts']['S3ModelArtifacts']
                print(f"📦 Model artifacts: {model_artifacts}")

                # Download and extract models
                download_and_extract_models(model_artifacts)

                break

            elif status == 'Failed':
                print("❌ TRAINING FAILED!")

                if 'FailureReason' in response:
                    print(f"💥 Failure reason: {response['FailureReason']}")

                break

            elif status == 'Stopped':
                print("⏹️  TRAINING STOPPED")
                break

            # Wait before next check
            time.sleep(30)

        except Exception as e:
            print(f"❌ Monitoring error: {e}")
            time.sleep(30)

def download_and_extract_models(s3_model_path: str):
    """Download and extract trained models to local directory"""
    try:
        print(f"⬇️  Downloading models from: {s3_model_path}")

        s3 = boto3.client('s3')

        # Parse S3 path
        s3_parts = s3_model_path.replace('s3://', '').split('/', 1)
        bucket = s3_parts[0]
        key = s3_parts[1]

        # Download model.tar.gz
        local_path = "/tmp/deep_learning_model.tar.gz"
        s3.download_file(bucket, key, local_path)

        print(f"📥 Downloaded to: {local_path}")

        # Extract to models directory
        import tarfile
        import os

        models_dir = "/Users/chasemad/Desktop/mini-xdr/models"

        with tarfile.open(local_path, 'r:gz') as tar:
            tar.extractall(models_dir)

        print(f"📂 Extracted models to: {models_dir}")

        # List extracted files
        extracted_files = []
        for root, dirs, files in os.walk(models_dir):
            for file in files:
                if file.endswith(('.pth', '.pkl', '.json')):
                    extracted_files.append(os.path.join(root, file))

        print("📋 Extracted files:")
        for file_path in extracted_files:
            file_size = os.path.getsize(file_path)
            print(f"   📄 {file_path} ({file_size:,} bytes)")

        # Check if deep learning models are now available
        check_model_integration()

    except Exception as e:
        print(f"❌ Model download failed: {e}")

def check_model_integration():
    """Check if the backend can now load the deep learning models"""
    try:
        print("🔍 Checking model integration...")

        # Try to import and test the deep learning manager
        import sys
        sys.path.append('/Users/chasemad/Desktop/mini-xdr/backend/app')

        from deep_learning_models import DeepLearningModelManager

        manager = DeepLearningModelManager()
        results = manager.load_models()

        print("🧠 Deep Learning Model Status:")
        for model_name, loaded in results.items():
            status = "✅ Loaded" if loaded else "❌ Not loaded"
            print(f"   {model_name}: {status}")

        # Get overall status
        status = manager.get_model_status()
        print(f"📊 GPU Available: {status.get('gpu_available', False)}")
        print(f"🔥 Models Ready: {status.get('deep_learning_loaded', False)}")

        if status.get('deep_learning_loaded', False):
            print("🚀 SUCCESS: Deep learning models are now integrated with the backend!")
        else:
            print("⚠️  WARNING: Models downloaded but integration may need troubleshooting")

    except Exception as e:
        print(f"⚠️  Model integration check failed: {e}")
        print("💡 Models downloaded successfully, but backend integration needs manual verification")

def show_training_logs(job_name: str = "pytorch-training-2025-09-27-13-19-58-208"):
    """Show recent training logs"""
    try:
        logs = boto3.client('logs')

        # Get log stream
        log_group = "/aws/sagemaker/TrainingJobs"

        # Find the log stream for this job
        streams_response = logs.describe_log_streams(
            logGroupName=log_group,
            orderBy='LastEventTime',
            descending=True
        )

        job_stream = None
        for stream in streams_response['logStreams']:
            if job_name in stream['logStreamName']:
                job_stream = stream['logStreamName']
                break

        if job_stream:
            print(f"📋 Recent training logs from {job_stream}:")
            print("-" * 60)

            events_response = logs.get_log_events(
                logGroupName=log_group,
                logStreamName=job_stream,
                limit=20,
                startFromHead=False
            )

            for event in events_response['events']:
                timestamp = datetime.fromtimestamp(event['timestamp'] / 1000)
                message = event['message'].strip()
                print(f"[{timestamp.strftime('%H:%M:%S')}] {message}")
        else:
            print("📋 No logs available yet (job may still be starting)")

    except Exception as e:
        print(f"❌ Failed to fetch logs: {e}")

if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == 'logs':
            show_training_logs()
        else:
            monitor_training_job(sys.argv[1])
    else:
        monitor_training_job()