#!/usr/bin/env python3
"""
Optimized CICIDS2017 Data Upload for AWS S3
Breaks large dataset into manageable chunks for reliable upload
"""

import boto3
import pandas as pd
import numpy as np
import os
import json
import math
from datetime import datetime
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OptimizedDataUploader:
    """Optimized data uploader for large CICIDS2017 dataset"""

    def __init__(self, region: str = 'us-east-1'):
        self.region = region
        self.s3_client = boto3.client('s3', region_name=region)
        self.sts_client = boto3.client('sts', region_name=region)

        # Configuration
        self.dataset_path = '/Users/chasemad/Desktop/mini-xdr/datasets/cicids2017_official/MachineLearningCVE'
        self.s3_bucket = 'mini-xdr-ml-data-bucket-' + self.sts_client.get_caller_identity()['Account']

        # Optimized chunk settings
        self.chunk_size = 100000  # 100k records per chunk (manageable size)
        self.features_to_drop = ['Flow ID', 'Source IP', 'Destination IP', 'Timestamp']
        self.target_column = 'Label'

    def load_and_process_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load and preprocess data with memory optimization"""
        logger.info("Loading CICIDS2017 dataset with memory optimization...")

        csv_files = list(Path(self.dataset_path).glob('*.csv'))
        logger.info(f"Found {len(csv_files)} CSV files")

        # Process files one at a time to save memory
        all_chunks = []
        total_records = 0

        for csv_file in csv_files:
            logger.info(f"Processing {csv_file.name}...")

            # Read in chunks to manage memory
            chunk_iter = pd.read_csv(csv_file, chunksize=50000, low_memory=False)

            for chunk in chunk_iter:
                # Basic preprocessing
                chunk = chunk.replace([np.inf, -np.inf], np.nan)
                chunk = chunk.fillna(0)

                # Remove unnecessary columns
                columns_to_drop = [col for col in self.features_to_drop if col in chunk.columns]
                if columns_to_drop:
                    chunk = chunk.drop(columns=columns_to_drop)

                # Handle non-numeric columns
                for col in chunk.select_dtypes(include=['object']).columns:
                    if col != self.target_column:
                        chunk[col] = LabelEncoder().fit_transform(chunk[col].astype(str))

                all_chunks.append(chunk)
                total_records += len(chunk)

                # Log progress
                if total_records % 200000 == 0:
                    logger.info(f"Processed {total_records:,} records so far...")

        logger.info(f"Combining {len(all_chunks)} chunks with {total_records:,} total records")

        # Combine all chunks
        combined_df = pd.concat(all_chunks, ignore_index=True)

        # Handle labels
        if self.target_column in combined_df.columns:
            # Create binary labels: BENIGN=0, any attack=1
            combined_df[self.target_column] = (combined_df[self.target_column] != 'BENIGN').astype(int)

            X = combined_df.drop(columns=[self.target_column])
            y = combined_df[self.target_column].values
        else:
            X = combined_df
            y = np.zeros(len(combined_df))

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Save preprocessing artifacts
        import joblib
        joblib.dump(scaler, 'feature_scaler.pkl')

        logger.info(f"Final processed data: {X_scaled.shape[0]:,} samples, {X_scaled.shape[1]} features")
        logger.info(f"Attack ratio: {np.sum(y)}/{len(y)} = {np.sum(y)/len(y)*100:.1f}% attacks")

        return X_scaled, y

    def upload_data_in_chunks(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Upload training data in optimized chunks"""
        logger.info("Splitting data and uploading in chunks...")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Calculate number of chunks needed
        train_chunks_needed = math.ceil(len(X_train) / self.chunk_size)
        test_chunks_needed = math.ceil(len(X_test) / self.chunk_size)

        logger.info(f"Training data: {len(X_train):,} samples in {train_chunks_needed} chunks")
        logger.info(f"Test data: {len(X_test):,} samples in {test_chunks_needed} chunks")

        # Upload training chunks
        train_files = []
        for i in range(train_chunks_needed):
            start_idx = i * self.chunk_size
            end_idx = min((i + 1) * self.chunk_size, len(X_train))

            # Create chunk with target as first column (SageMaker format)
            chunk_data = np.column_stack((y_train[start_idx:end_idx], X_train[start_idx:end_idx]))
            chunk_df = pd.DataFrame(chunk_data)

            # Save chunk locally
            chunk_file = f'train_chunk_{i:03d}.csv'
            chunk_df.to_csv(chunk_file, index=False, header=False)

            # Check if chunk already exists in S3
            s3_key = f'data/train/train_chunk_{i:03d}.csv'
            try:
                self.s3_client.head_object(Bucket=self.s3_bucket, Key=s3_key)
                logger.info(f"⏭️  Skipping training chunk {i+1}/{train_chunks_needed} (already exists)")
                train_files.append(f"s3://{self.s3_bucket}/{s3_key}")
                os.remove(chunk_file)
                continue
            except:
                pass  # File doesn't exist, proceed with upload

            # Upload to S3
            try:
                self.s3_client.upload_file(chunk_file, self.s3_bucket, s3_key)
                train_files.append(f"s3://{self.s3_bucket}/{s3_key}")
                logger.info(f"✅ Uploaded training chunk {i+1}/{train_chunks_needed} ({end_idx-start_idx:,} samples)")

                # Clean up local file
                os.remove(chunk_file)

            except Exception as e:
                logger.error(f"❌ Failed to upload training chunk {i}: {e}")
                if os.path.exists(chunk_file):
                    os.remove(chunk_file)
                raise

        # Upload test chunks
        test_files = []
        for i in range(test_chunks_needed):
            start_idx = i * self.chunk_size
            end_idx = min((i + 1) * self.chunk_size, len(X_test))

            # Create chunk with target as first column (SageMaker format)
            chunk_data = np.column_stack((y_test[start_idx:end_idx], X_test[start_idx:end_idx]))
            chunk_df = pd.DataFrame(chunk_data)

            # Save chunk locally
            chunk_file = f'test_chunk_{i:03d}.csv'
            chunk_df.to_csv(chunk_file, index=False, header=False)

            # Check if chunk already exists in S3
            s3_key = f'data/test/test_chunk_{i:03d}.csv'
            try:
                self.s3_client.head_object(Bucket=self.s3_bucket, Key=s3_key)
                logger.info(f"⏭️  Skipping test chunk {i+1}/{test_chunks_needed} (already exists)")
                test_files.append(f"s3://{self.s3_bucket}/{s3_key}")
                os.remove(chunk_file)
                continue
            except:
                pass  # File doesn't exist, proceed with upload

            # Upload to S3
            try:
                self.s3_client.upload_file(chunk_file, self.s3_bucket, s3_key)
                test_files.append(f"s3://{self.s3_bucket}/{s3_key}")
                logger.info(f"✅ Uploaded test chunk {i+1}/{test_chunks_needed} ({end_idx-start_idx:,} samples)")

                # Clean up local file
                os.remove(chunk_file)

            except Exception as e:
                logger.error(f"❌ Failed to upload test chunk {i}: {e}")
                if os.path.exists(chunk_file):
                    os.remove(chunk_file)
                raise

        # Create manifest files for SageMaker
        train_manifest = {
            'total_samples': len(X_train),
            'chunks': len(train_files),
            'chunk_size': self.chunk_size,
            'files': train_files,
            'features': X_train.shape[1],
            'attack_ratio': float(np.sum(y_train) / len(y_train))
        }

        test_manifest = {
            'total_samples': len(X_test),
            'chunks': len(test_files),
            'chunk_size': self.chunk_size,
            'files': test_files,
            'features': X_test.shape[1],
            'attack_ratio': float(np.sum(y_test) / len(y_test))
        }

        # Upload manifests
        train_manifest_file = 'train_manifest.json'
        test_manifest_file = 'test_manifest.json'

        with open(train_manifest_file, 'w') as f:
            json.dump(train_manifest, f, indent=2)
        with open(test_manifest_file, 'w') as f:
            json.dump(test_manifest, f, indent=2)

        self.s3_client.upload_file(train_manifest_file, self.s3_bucket, 'data/train_manifest.json')
        self.s3_client.upload_file(test_manifest_file, self.s3_bucket, 'data/test_manifest.json')

        # Clean up manifest files
        os.remove(train_manifest_file)
        os.remove(test_manifest_file)

        logger.info(f"✅ Upload complete! Training manifest: s3://{self.s3_bucket}/data/train_manifest.json")

        return {
            'train_manifest': train_manifest,
            'test_manifest': test_manifest,
            'bucket': self.s3_bucket
        }

    def create_sagemaker_training_script(self, data_config: Dict) -> str:
        """Create optimized SageMaker training script for chunked data"""
        logger.info("Creating SageMaker training script for chunked data...")

        training_script = '''
import os
import json
import boto3
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import logging

logger = logging.getLogger(__name__)

def load_chunked_data(manifest_path, data_dir):
    """Load data from multiple chunks"""
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    chunks = []
    for i in range(manifest['chunks']):
        chunk_file = os.path.join(data_dir, f'train_chunk_{i:03d}.csv')
        if os.path.exists(chunk_file):
            chunk_df = pd.read_csv(chunk_file, header=None)
            chunks.append(chunk_df)

    if not chunks:
        raise ValueError("No training chunks found")

    combined_df = pd.concat(chunks, ignore_index=True)

    # First column is target, rest are features
    y = combined_df.iloc[:, 0].values
    X = combined_df.iloc[:, 1:].values

    return X, y, manifest

def train_model(X, y):
    """Train Isolation Forest model"""
    logger.info(f"Training on {len(X):,} samples with {X.shape[1]} features")

    model = IsolationForest(
        contamination=0.1,
        random_state=42,
        n_estimators=100,
        n_jobs=-1
    )

    model.fit(X)
    return model

if __name__ == "__main__":
    # Load training data
    train_dir = os.environ.get('SM_CHANNEL_TRAINING', '/opt/ml/input/data/training')
    model_dir = os.environ.get('SM_MODEL_DIR', '/opt/ml/model')

    manifest_path = os.path.join(train_dir, 'train_manifest.json')
    X, y, manifest = load_chunked_data(manifest_path, train_dir)

    # Train model
    model = train_model(X, y)

    # Save model
    joblib.dump(model, os.path.join(model_dir, 'model.pkl'))

    # Save training info
    training_info = {
        'samples_trained': len(X),
        'features': X.shape[1],
        'attack_ratio': float(np.sum(y) / len(y)),
        'model_type': 'IsolationForest'
    }

    with open(os.path.join(model_dir, 'training_info.json'), 'w') as f:
        json.dump(training_info, f, indent=2)

    print("Training completed successfully!")
'''

        # Save training script
        script_path = 'sagemaker_training_script.py'
        with open(script_path, 'w') as f:
            f.write(training_script)

        # Upload to S3
        script_s3_key = 'code/sagemaker_training_script.py'
        self.s3_client.upload_file(script_path, self.s3_bucket, script_s3_key)
        os.remove(script_path)

        script_uri = f"s3://{self.s3_bucket}/{script_s3_key}"
        logger.info(f"✅ Training script uploaded: {script_uri}")

        return script_uri

    def run_optimized_pipeline(self):
        """Execute the complete optimized pipeline"""
        try:
            logger.info("🚀 Starting Optimized CICIDS2017 Data Upload Pipeline")
            start_time = datetime.now()

            # Step 1: Load and process data
            X, y = self.load_and_process_data()

            # Step 2: Upload data in chunks
            data_config = self.upload_data_in_chunks(X, y)

            # Step 3: Create training script
            script_uri = self.create_sagemaker_training_script(data_config)

            # Step 4: Create deployment summary
            summary = {
                'total_samples': X.shape[0],
                'features': X.shape[1],
                'training_samples': data_config['train_manifest']['total_samples'],
                'test_samples': data_config['test_manifest']['total_samples'],
                'training_chunks': data_config['train_manifest']['chunks'],
                'test_chunks': data_config['test_manifest']['chunks'],
                'attack_ratio': data_config['train_manifest']['attack_ratio'],
                's3_bucket': self.s3_bucket,
                'training_script': script_uri,
                'upload_time': (datetime.now() - start_time).total_seconds(),
                'ready_for_sagemaker': True,
                'timestamp': datetime.now().isoformat()
            }

            # Save summary
            summary_file = 'upload_summary.json'
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)

            self.s3_client.upload_file(summary_file, self.s3_bucket, 'results/upload_summary.json')
            os.remove(summary_file)

            logger.info("🎉 Optimized Data Upload Pipeline Completed!")
            return summary

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise

def main():
    """Main execution function"""
    uploader = OptimizedDataUploader()
    summary = uploader.run_optimized_pipeline()

    print("\n" + "="*70)
    print("🛡️ MINI-XDR OPTIMIZED DATA UPLOAD COMPLETE")
    print("="*70)
    print(f"Total Samples: {summary['total_samples']:,}")
    print(f"Features: {summary['features']}")
    print(f"Training Set: {summary['training_samples']:,} samples in {summary['training_chunks']} chunks")
    print(f"Test Set: {summary['test_samples']:,} samples in {summary['test_chunks']} chunks")
    print(f"Attack Ratio: {summary['attack_ratio']*100:.1f}% attacks")
    print(f"S3 Bucket: {summary['s3_bucket']}")
    print(f"Upload Time: {summary['upload_time']:.1f} seconds")
    print(f"Ready for SageMaker: ✅ YES")
    print("="*70)

if __name__ == "__main__":
    main()