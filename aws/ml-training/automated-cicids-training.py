#!/usr/bin/env python3
"""
Automated CICIDS2017 ML Training Pipeline for AWS SageMaker
Trains 4 ensemble models with 2.8M+ cybersecurity events
"""

import boto3
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import IsolationForest
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CICIDSMLPipeline:
    """Automated ML pipeline for CICIDS2017 dataset with AWS SageMaker integration"""

    def __init__(self, region: str = 'us-east-1'):
        self.region = region
        self.s3_client = boto3.client('s3', region_name=region)
        self.sagemaker_client = boto3.client('sagemaker', region_name=region)
        self.sts_client = boto3.client('sts', region_name=region)

        # Dataset configuration
        self.dataset_path = '/Users/chasemad/Desktop/mini-xdr/datasets/cicids2017_official/MachineLearningCVE'
        self.s3_bucket = 'mini-xdr-ml-data-bucket-' + self.sts_client.get_caller_identity()['Account']

        # ML configuration
        self.features_to_drop = ['Flow ID', 'Source IP', 'Destination IP', 'Timestamp']
        self.target_column = 'Label'

        # Model configurations
        self.models_config = {
            'isolation_forest': {
                'contamination': 0.1,
                'random_state': 42,
                'n_estimators': 100
            },
            'xgboost': {
                'n_estimators': 1000,
                'max_depth': 6,
                'learning_rate': 0.1,
                'random_state': 42
            }
        }

    def load_and_preprocess_data(self) -> Tuple[pd.DataFrame, np.ndarray]:
        """Load all CICIDS2017 CSV files and preprocess for ML training"""
        logger.info(f"Loading CICIDS2017 dataset from {self.dataset_path}")

        # Find all CSV files
        csv_files = list(Path(self.dataset_path).glob('*.csv'))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {self.dataset_path}")

        logger.info(f"Found {len(csv_files)} CSV files")

        # Load and combine all datasets
        dataframes = []
        total_records = 0

        for csv_file in csv_files:
            logger.info(f"Loading {csv_file.name}...")
            try:
                df = pd.read_csv(csv_file, low_memory=False)
                logger.info(f"  - Loaded {len(df):,} records with {len(df.columns)} features")
                dataframes.append(df)
                total_records += len(df)
            except Exception as e:
                logger.error(f"Error loading {csv_file}: {e}")
                continue

        if not dataframes:
            raise ValueError("No data loaded successfully")

        # Combine all dataframes
        logger.info("Combining datasets...")
        combined_df = pd.concat(dataframes, ignore_index=True)
        logger.info(f"Combined dataset: {len(combined_df):,} total records")

        # Basic preprocessing
        logger.info("Preprocessing data...")

        # Handle missing values
        combined_df = combined_df.replace([np.inf, -np.inf], np.nan)
        combined_df = combined_df.fillna(0)

        # Remove unnecessary columns
        columns_to_drop = [col for col in self.features_to_drop if col in combined_df.columns]
        if columns_to_drop:
            combined_df = combined_df.drop(columns=columns_to_drop)
            logger.info(f"Dropped columns: {columns_to_drop}")

        # Encode labels
        le = LabelEncoder()
        if self.target_column in combined_df.columns:
            combined_df[self.target_column] = le.fit_transform(combined_df[self.target_column].astype(str))

            # Save label encoder for later use
            joblib.dump(le, 'label_encoder.pkl')
            logger.info(f"Label classes: {list(le.classes_)}")

        # Separate features and target
        if self.target_column in combined_df.columns:
            X = combined_df.drop(columns=[self.target_column])
            y = combined_df[self.target_column].values
        else:
            X = combined_df
            y = np.zeros(len(combined_df))  # Default labels for unsupervised learning

        # Handle non-numeric columns
        for col in X.select_dtypes(include=['object']).columns:
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Save scaler for later use
        joblib.dump(scaler, 'feature_scaler.pkl')

        logger.info(f"Final feature matrix shape: {X_scaled.shape}")
        logger.info(f"Target distribution: {np.unique(y, return_counts=True)}")

        return X_scaled, y

    def upload_to_s3(self, local_path: str, s3_key: str) -> str:
        """Upload file to S3 and return S3 URI"""
        try:
            self.s3_client.upload_file(local_path, self.s3_bucket, s3_key)
            s3_uri = f"s3://{self.s3_bucket}/{s3_key}"
            logger.info(f"Uploaded {local_path} to {s3_uri}")
            return s3_uri
        except Exception as e:
            logger.error(f"Failed to upload {local_path}: {e}")
            raise

    def prepare_training_data(self, X: np.ndarray, y: np.ndarray) -> Dict[str, str]:
        """Prepare training data and upload to S3"""
        logger.info("Preparing training data for SageMaker...")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Save training data
        train_data = np.column_stack((y_train, X_train))
        test_data = np.column_stack((y_test, X_test))

        # Save as CSV (SageMaker XGBoost format)
        train_df = pd.DataFrame(train_data)
        test_df = pd.DataFrame(test_data)

        train_file = 'cicids_train.csv'
        test_file = 'cicids_test.csv'

        train_df.to_csv(train_file, index=False, header=False)
        test_df.to_csv(test_file, index=False, header=False)

        logger.info(f"Training set: {len(train_df):,} samples")
        logger.info(f"Test set: {len(test_df):,} samples")

        # Upload to S3
        train_s3_uri = self.upload_to_s3(train_file, 'data/train/cicids_train.csv')
        test_s3_uri = self.upload_to_s3(test_file, 'data/test/cicids_test.csv')

        # Clean up local files
        os.remove(train_file)
        os.remove(test_file)

        return {
            'train_uri': train_s3_uri,
            'test_uri': test_s3_uri,
            'train_samples': len(train_df),
            'test_samples': len(test_df),
            'features': X.shape[1]
        }

    def train_isolation_forest_local(self, X: np.ndarray) -> str:
        """Train Isolation Forest model locally and upload to S3"""
        logger.info("Training Isolation Forest model...")

        # Train model
        model = IsolationForest(**self.models_config['isolation_forest'])
        model.fit(X)

        # Save model
        model_file = 'isolation_forest_model.pkl'
        joblib.dump(model, model_file)

        # Upload to S3
        model_s3_uri = self.upload_to_s3(model_file, 'models/isolation_forest/model.pkl')

        # Clean up
        os.remove(model_file)

        logger.info(f"Isolation Forest model uploaded to {model_s3_uri}")
        return model_s3_uri

    def create_sagemaker_training_job(self, data_config: Dict[str, str]) -> str:
        """Create SageMaker XGBoost training job"""
        logger.info("Creating SageMaker XGBoost training job...")

        # Training job configuration
        job_name = f"mini-xdr-xgboost-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"

        # Get the built-in XGBoost container image
        from sagemaker.image_uris import retrieve
        container = retrieve('xgboost', self.region, version='1.5-1')

        # IAM role for SageMaker
        iam = boto3.client('iam')
        role_name = 'SageMakerExecutionRole-MiniXDR'

        try:
            role_arn = iam.get_role(RoleName=role_name)['Role']['Arn']
        except:
            # Create role if it doesn't exist
            assume_role_policy = {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Principal": {"Service": "sagemaker.amazonaws.com"},
                        "Action": "sts:AssumeRole"
                    }
                ]
            }

            iam.create_role(
                RoleName=role_name,
                AssumeRolePolicyDocument=json.dumps(assume_role_policy)
            )

            iam.attach_role_policy(
                RoleName=role_name,
                PolicyArn='arn:aws:iam::aws:policy/AmazonSageMakerFullAccess'
            )

            role_arn = f"arn:aws:iam::{self.sts_client.get_caller_identity()['Account']}:role/{role_name}"

        # Training job parameters
        training_params = {
            'TrainingJobName': job_name,
            'RoleArn': role_arn,
            'AlgorithmSpecification': {
                'TrainingImage': container,
                'TrainingInputMode': 'File'
            },
            'InputDataConfig': [
                {
                    'ChannelName': 'training',
                    'DataSource': {
                        'S3DataSource': {
                            'S3DataType': 'S3Prefix',
                            'S3Uri': data_config['train_uri'].rsplit('/', 1)[0],
                            'S3DataDistributionType': 'FullyReplicated'
                        }
                    },
                    'ContentType': 'text/csv'
                },
                {
                    'ChannelName': 'validation',
                    'DataSource': {
                        'S3DataSource': {
                            'S3DataType': 'S3Prefix',
                            'S3Uri': data_config['test_uri'].rsplit('/', 1)[0],
                            'S3DataDistributionType': 'FullyReplicated'
                        }
                    },
                    'ContentType': 'text/csv'
                }
            ],
            'OutputDataConfig': {
                'S3OutputPath': f"s3://{self.s3_bucket}/models/xgboost/"
            },
            'ResourceConfig': {
                'InstanceType': 'ml.m5.2xlarge',
                'InstanceCount': 1,
                'VolumeSizeInGB': 30
            },
            'StoppingCondition': {
                'MaxRuntimeInSeconds': 3600  # 1 hour max
            },
            'HyperParameters': {
                'objective': 'multi:softmax',
                'num_class': '2',
                'num_round': '100',
                'max_depth': '6',
                'eta': '0.1',
                'subsample': '0.8',
                'colsample_bytree': '0.8'
            }
        }

        # Start training job
        try:
            response = self.sagemaker_client.create_training_job(**training_params)
            logger.info(f"Started SageMaker training job: {job_name}")
            return job_name
        except Exception as e:
            logger.error(f"Failed to create training job: {e}")
            raise

    def monitor_training_job(self, job_name: str) -> bool:
        """Monitor SageMaker training job progress"""
        logger.info(f"Monitoring training job: {job_name}")

        import time
        while True:
            try:
                response = self.sagemaker_client.describe_training_job(TrainingJobName=job_name)
                status = response['TrainingJobStatus']

                if status == 'Completed':
                    logger.info(f"Training job {job_name} completed successfully!")
                    return True
                elif status == 'Failed':
                    logger.error(f"Training job {job_name} failed: {response.get('FailureReason', 'Unknown')}")
                    return False
                elif status in ['InProgress', 'Stopping']:
                    logger.info(f"Training job status: {status}")
                    time.sleep(30)  # Wait 30 seconds before checking again
                else:
                    logger.warning(f"Unknown training job status: {status}")
                    time.sleep(30)

            except Exception as e:
                logger.error(f"Error monitoring training job: {e}")
                return False

    def run_full_pipeline(self):
        """Execute the complete ML training pipeline"""
        try:
            logger.info("🚀 Starting CICIDS2017 ML Training Pipeline")

            # Step 1: Load and preprocess data
            X, y = self.load_and_preprocess_data()

            # Step 2: Prepare training data for SageMaker
            data_config = self.prepare_training_data(X, y)
            logger.info(f"Data configuration: {data_config}")

            # Step 3: Train Isolation Forest locally
            isolation_forest_uri = self.train_isolation_forest_local(X)

            # Step 4: Create SageMaker XGBoost training job
            training_job_name = self.create_sagemaker_training_job(data_config)

            # Step 5: Monitor training progress
            training_success = self.monitor_training_job(training_job_name)

            # Step 6: Create summary
            summary = {
                'dataset_size': len(X),
                'features': X.shape[1],
                'training_samples': data_config['train_samples'],
                'test_samples': data_config['test_samples'],
                'isolation_forest_model': isolation_forest_uri,
                'xgboost_training_job': training_job_name,
                'training_success': training_success,
                'timestamp': datetime.now().isoformat()
            }

            # Save summary
            with open('training_summary.json', 'w') as f:
                json.dump(summary, f, indent=2)

            summary_uri = self.upload_to_s3('training_summary.json', 'results/training_summary.json')
            os.remove('training_summary.json')

            logger.info("🎉 ML Training Pipeline Completed!")
            logger.info(f"Summary available at: {summary_uri}")

            return summary

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise

def main():
    """Main execution function"""
    pipeline = CICIDSMLPipeline()

    # Run the complete pipeline
    summary = pipeline.run_full_pipeline()

    print("\n" + "="*60)
    print("🛡️ MINI-XDR ML TRAINING COMPLETE")
    print("="*60)
    print(f"Dataset Size: {summary['dataset_size']:,} samples")
    print(f"Features: {summary['features']} features")
    print(f"Training Set: {summary['training_samples']:,} samples")
    print(f"Test Set: {summary['test_samples']:,} samples")
    print(f"Isolation Forest: ✅ Trained")
    print(f"XGBoost: {'✅ Completed' if summary['training_success'] else '❌ Failed'}")
    print(f"Training Job: {summary['xgboost_training_job']}")
    print("="*60)

if __name__ == "__main__":
    main()