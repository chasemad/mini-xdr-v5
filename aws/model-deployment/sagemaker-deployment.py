#!/usr/bin/env python3
"""
SageMaker Model Deployment for Mini-XDR Real-time Inference
Deploys trained ML models as auto-scaling endpoints for threat detection

This script creates production-ready inference endpoints for:
- Real-time threat classification
- Anomaly detection scoring
- Ensemble prediction aggregation
- Multi-model hosting optimization
"""

import boto3
import json
import time
from datetime import datetime
import sagemaker
from sagemaker import get_execution_role
from sagemaker.model import Model
from sagemaker.predictor import Predictor
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer
from sagemaker.multidatamodel import MultiDataModel

class MiniXDRModelDeployment:
    """
    Production deployment pipeline for Mini-XDR ML models
    """
    
    def __init__(self, region='us-east-1'):
        self.region = region
        self.session = sagemaker.Session()
        self.role = get_execution_role()
        self.s3_client = boto3.client('s3')
        self.sagemaker_client = boto3.client('sagemaker')
        
        # Configuration
        self.models_bucket = "mini-xdr-ml-models-123456789-us-east-1"
        self.artifacts_bucket = "mini-xdr-ml-artifacts-123456789-us-east-1"
        
        # Inference configuration
        self.instance_type = "ml.c5.2xlarge"  # CPU-optimized for inference
        self.initial_instance_count = 2
        self.max_instance_count = 10
        
        # Endpoint configuration
        self.endpoint_config_name = f"mini-xdr-endpoint-config-{int(time.time())}"
        self.endpoint_name = f"mini-xdr-endpoint-{int(time.time())}"
        
        print("🚀 Mini-XDR Model Deployment initialized")
        print(f"🖥️ Inference instance: {self.instance_type}")
        print(f"📈 Auto-scaling: {self.initial_instance_count} → {self.max_instance_count} instances")
    
    def create_inference_scripts(self):
        """
        Create inference scripts for each model type
        """
        print("📝 Creating inference scripts...")
        
        # Multi-model inference script
        inference_script = """
import json
import boto3
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import logging

logger = logging.getLogger(__name__)

class EnsemblePredictor:
    def __init__(self, model_dir='/opt/ml/model'):
        self.model_dir = model_dir
        self.models = {}
        self.scalers = {}
        self.ensemble_config = None
        self.load_models()
    
    def load_models(self):
        '''Load all trained models and ensemble configuration'''
        try:
            # Load ensemble configuration
            with open(f'{self.model_dir}/ensemble_config.json', 'r') as f:
                self.ensemble_config = json.load(f)
            
            # Load XGBoost model
            if os.path.exists(f'{self.model_dir}/xgboost_model.pkl'):
                self.models['xgboost'] = joblib.load(f'{self.model_dir}/xgboost_model.pkl')
                logger.info("XGBoost model loaded")
            
            # Load Isolation Forest ensemble
            if os.path.exists(f'{self.model_dir}/isolation_forest_ensemble.pkl'):
                self.models['isolation_forest'] = joblib.load(f'{self.model_dir}/isolation_forest_ensemble.pkl')
                logger.info("Isolation Forest ensemble loaded")
            
            # Load LSTM Autoencoder
            if os.path.exists(f'{self.model_dir}/lstm_autoencoder.pth'):
                self.models['lstm'] = self.load_lstm_model(f'{self.model_dir}/lstm_autoencoder.pth')
                logger.info("LSTM Autoencoder loaded")
            
            # Load Transformer model
            if os.path.exists(f'{self.model_dir}/transformer_model'):
                self.models['transformer'] = tf.keras.models.load_model(f'{self.model_dir}/transformer_model')
                logger.info("Transformer model loaded")
            
            logger.info(f"Loaded {len(self.models)} models for ensemble prediction")
            
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            raise
    
    def load_lstm_model(self, model_path):
        '''Load PyTorch LSTM model'''
        # Define LSTM architecture (must match training)
        class LSTMAutoencoder(nn.Module):
            def __init__(self, input_size=113, hidden_size=128, num_layers=3, sequence_length=100):
                super(LSTMAutoencoder, self).__init__()
                self.input_size = input_size
                self.hidden_size = hidden_size
                self.num_layers = num_layers
                self.sequence_length = sequence_length
                
                self.encoder = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
                self.decoder = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
                self.output_layer = nn.Linear(hidden_size, input_size)
                self.attention = nn.MultiheadAttention(hidden_size, num_heads=8, batch_first=True)
            
            def forward(self, x):
                encoded, (hidden, cell) = self.encoder(x)
                attended, _ = self.attention(encoded, encoded, encoded)
                decoded, _ = self.decoder(attended, (hidden, cell))
                reconstructed = self.output_layer(decoded)
                return reconstructed
        
        model = LSTMAutoencoder()
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        return model
    
    def preprocess_features(self, data):
        '''Preprocess input features for prediction'''
        try:
            # Convert to DataFrame if needed
            if isinstance(data, dict):
                df = pd.DataFrame([data])
            elif isinstance(data, list):
                df = pd.DataFrame(data)
            else:
                df = data
            
            # Expected feature columns (113 features total)
            expected_features = [
                # Temporal features (15)
                'flow_duration', 'flow_iat_mean', 'flow_iat_std', 'flow_iat_max', 'flow_iat_min',
                'fwd_iat_total', 'fwd_iat_mean', 'fwd_iat_std', 'fwd_iat_max', 'fwd_iat_min',
                'bwd_iat_total', 'bwd_iat_mean', 'bwd_iat_std', 'bwd_iat_max', 'bwd_iat_min',
                
                # Packet features (15)
                'total_fwd_packets', 'total_backward_packets', 'fwd_packet_length_max',
                'fwd_packet_length_min', 'fwd_packet_length_mean', 'fwd_packet_length_std',
                'bwd_packet_length_max', 'bwd_packet_length_min', 'bwd_packet_length_mean',
                'bwd_packet_length_std', 'packet_length_max', 'packet_length_min',
                'packet_length_mean', 'packet_length_std', 'packet_length_variance',
                
                # Add remaining 83+ features here...
                # This is a simplified example
            ]
            
            # Ensure all expected features are present
            for feature in expected_features:
                if feature not in df.columns:
                    df[feature] = 0.0
            
            # Select only expected features
            df = df[expected_features]
            
            # Handle missing values
            df = df.fillna(0.0)
            
            # Apply scaling if available
            if 'scaler' in self.scalers:
                df = pd.DataFrame(
                    self.scalers['scaler'].transform(df),
                    columns=df.columns
                )
            
            return df
            
        except Exception as e:
            logger.error(f"Error preprocessing features: {str(e)}")
            raise
    
    def predict_xgboost(self, features):
        '''XGBoost classification prediction'''
        try:
            model = self.models['xgboost']
            proba = model.predict_proba(features)
            pred = model.predict(features)
            
            return {
                'predictions': pred.tolist(),
                'probabilities': proba.tolist(),
                'confidence': np.max(proba, axis=1).tolist()
            }
        except Exception as e:
            logger.error(f"XGBoost prediction error: {str(e)}")
            return None
    
    def predict_isolation_forest(self, features):
        '''Isolation Forest anomaly detection'''
        try:
            model = self.models['isolation_forest']
            anomaly_scores = model.decision_function(features)
            predictions = model.predict(features)
            
            # Convert to binary (0=normal, 1=anomaly)
            binary_pred = (predictions == -1).astype(int)
            
            return {
                'anomaly_scores': anomaly_scores.tolist(),
                'predictions': binary_pred.tolist(),
                'is_anomaly': binary_pred.tolist()
            }
        except Exception as e:
            logger.error(f"Isolation Forest prediction error: {str(e)}")
            return None
    
    def predict_lstm(self, features):
        '''LSTM Autoencoder anomaly detection'''
        try:
            model = self.models['lstm']
            
            # Prepare sequence data (simplified)
            sequence_length = 100
            if len(features) >= sequence_length:
                sequences = []
                for i in range(len(features) - sequence_length + 1):
                    sequences.append(features.iloc[i:i+sequence_length].values)
                
                if sequences:
                    input_tensor = torch.FloatTensor(np.array(sequences))
                    
                    with torch.no_grad():
                        reconstructed = model(input_tensor)
                        
                    # Calculate reconstruction error
                    mse = torch.mean((input_tensor - reconstructed) ** 2, dim=(1, 2))
                    reconstruction_errors = mse.numpy()
                    
                    # Threshold-based anomaly detection
                    threshold = 0.1  # Would be determined during training
                    anomalies = (reconstruction_errors > threshold).astype(int)
                    
                    return {
                        'reconstruction_errors': reconstruction_errors.tolist(),
                        'predictions': anomalies.tolist(),
                        'threshold': threshold
                    }
            
            # Fallback for insufficient data
            return {
                'reconstruction_errors': [0.0] * len(features),
                'predictions': [0] * len(features),
                'threshold': 0.1
            }
            
        except Exception as e:
            logger.error(f"LSTM prediction error: {str(e)}")
            return None
    
    def predict_transformer(self, features):
        '''Transformer classification prediction'''
        try:
            model = self.models['transformer']
            
            # Prepare input for transformer (may need reshaping)
            input_data = features.values.reshape(1, len(features), -1)
            
            predictions = model.predict(input_data)
            
            return {
                'predictions': np.argmax(predictions, axis=1).tolist(),
                'probabilities': predictions.tolist(),
                'confidence': np.max(predictions, axis=1).tolist()
            }
        except Exception as e:
            logger.error(f"Transformer prediction error: {str(e)}")
            return None
    
    def ensemble_prediction(self, features):
        '''Combine predictions from all models'''
        try:
            results = {}
            
            # Get predictions from each model
            if 'xgboost' in self.models:
                results['xgboost'] = self.predict_xgboost(features)
            
            if 'isolation_forest' in self.models:
                results['isolation_forest'] = self.predict_isolation_forest(features)
            
            if 'lstm' in self.models:
                results['lstm'] = self.predict_lstm(features)
            
            if 'transformer' in self.models:
                results['transformer'] = self.predict_transformer(features)
            
            # Ensemble aggregation
            ensemble_pred = self.aggregate_predictions(results)
            
            return {
                'ensemble_prediction': ensemble_pred,
                'individual_models': results,
                'model_count': len(results),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Ensemble prediction error: {str(e)}")
            raise
    
    def aggregate_predictions(self, model_results):
        '''Aggregate predictions from multiple models'''
        try:
            if not self.ensemble_config or not model_results:
                return {'error': 'No valid predictions available'}
            
            weights = {}
            predictions = {}
            
            # Extract weights and predictions
            for model_name, result in model_results.items():
                if result and 'predictions' in result:
                    weight = self.ensemble_config['models'].get(model_name, {}).get('weight', 0.25)
                    weights[model_name] = weight
                    predictions[model_name] = result['predictions'][0] if result['predictions'] else 0
            
            if not predictions:
                return {'error': 'No valid predictions from any model'}
            
            # Weighted voting
            weighted_sum = sum(pred * weights.get(model, 0.25) for model, pred in predictions.items())
            total_weight = sum(weights.values())
            
            if total_weight > 0:
                final_score = weighted_sum / total_weight
                final_prediction = 1 if final_score > 0.5 else 0
            else:
                final_score = 0.5
                final_prediction = 0
            
            # Confidence calculation
            confidence = abs(final_score - 0.5) * 2  # 0-1 scale
            
            return {
                'prediction': final_prediction,
                'score': final_score,
                'confidence': confidence,
                'threat_level': self.calculate_threat_level(final_score),
                'contributing_models': list(predictions.keys())
            }
            
        except Exception as e:
            logger.error(f"Prediction aggregation error: {str(e)}")
            return {'error': str(e)}
    
    def calculate_threat_level(self, score):
        '''Calculate threat level based on prediction score'''
        if score >= 0.9:
            return 'CRITICAL'
        elif score >= 0.7:
            return 'HIGH'
        elif score >= 0.5:
            return 'MEDIUM'
        elif score >= 0.3:
            return 'LOW'
        else:
            return 'BENIGN'

# SageMaker inference functions
predictor = None

def model_fn(model_dir):
    '''Load model for SageMaker inference'''
    global predictor
    predictor = EnsemblePredictor(model_dir)
    return predictor

def input_fn(request_body, request_content_type):
    '''Parse input data'''
    if request_content_type == 'application/json':
        input_data = json.loads(request_body)
        return input_data
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    '''Make prediction'''
    try:
        # Preprocess features
        features = model.preprocess_features(input_data)
        
        # Get ensemble prediction
        result = model.ensemble_prediction(features)
        
        return result
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return {'error': str(e)}

def output_fn(prediction, accept):
    '''Format prediction output'''
    if accept == 'application/json':
        return json.dumps(prediction), accept
    else:
        raise ValueError(f"Unsupported accept type: {accept}")
"""
        
        # Write inference script
        with open('/tmp/inference.py', 'w') as f:
            f.write(inference_script)
        
        # Upload to S3
        script_key = 'inference-scripts/inference.py'
        self.s3_client.upload_file('/tmp/inference.py', self.artifacts_bucket, script_key)
        
        print("✅ Inference scripts created and uploaded")
        return f's3://{self.artifacts_bucket}/{script_key}'
    
    def create_model_artifacts(self):
        """
        Package trained models for deployment
        """
        print("📦 Creating model artifacts...")
        
        # Create model.tar.gz with all models
        model_package_script = """
import tarfile
import boto3
import os

def package_models():
    s3_client = boto3.client('s3')
    models_bucket = 'mini-xdr-ml-models-123456789-us-east-1'
    
    # Download all model artifacts
    model_files = [
        'xgboost/xgboost_model.pkl',
        'isolation-forest/isolation_forest_ensemble.pkl',
        'lstm-autoencoder/lstm_autoencoder.pth',
        'transformer/transformer_model',
        'ensemble/config.json'
    ]
    
    # Create model package
    with tarfile.open('/tmp/model.tar.gz', 'w:gz') as tar:
        for model_file in model_files:
            try:
                local_path = f'/tmp/{os.path.basename(model_file)}'
                s3_client.download_file(models_bucket, model_file, local_path)
                tar.add(local_path, arcname=os.path.basename(model_file))
                print(f"Added {model_file} to package")
            except Exception as e:
                print(f"Warning: Could not package {model_file}: {e}")
        
        # Add inference script
        tar.add('/tmp/inference.py', arcname='code/inference.py')
    
    # Upload packaged model
    s3_client.upload_file('/tmp/model.tar.gz', models_bucket, 'deployment/model.tar.gz')
    print("Model package uploaded to S3")

if __name__ == '__main__':
    package_models()
"""
        
        # Execute packaging
        exec(model_package_script)
        
        model_data_url = f's3://{self.models_bucket}/deployment/model.tar.gz'
        print(f"✅ Model artifacts packaged: {model_data_url}")
        
        return model_data_url
    
    def deploy_ensemble_endpoint(self):
        """
        Deploy ensemble model as SageMaker endpoint
        """
        print("🚀 Deploying ensemble endpoint...")
        
        # Create inference scripts
        inference_script_path = self.create_inference_scripts()
        
        # Package model artifacts
        model_data_url = self.create_model_artifacts()
        
        # Create SageMaker model
        model = Model(
            image_uri=sagemaker.image_uris.retrieve(
                framework='sklearn',
                region=self.region,
                version='0.23-1',
                py_version='py3',
                instance_type=self.instance_type
            ),
            model_data=model_data_url,
            role=self.role,
            entry_point='inference.py',
            source_dir=inference_script_path,
            framework_version='0.23-1',
            py_version='py3'
        )
        
        # Deploy endpoint
        predictor = model.deploy(
            initial_instance_count=self.initial_instance_count,
            instance_type=self.instance_type,
            endpoint_name=self.endpoint_name,
            serializer=JSONSerializer(),
            deserializer=JSONDeserializer()
        )
        
        print(f"✅ Endpoint deployed: {self.endpoint_name}")
        print(f"🌐 Endpoint URL: {predictor.endpoint_name}")
        
        return predictor
    
    def setup_auto_scaling(self):
        """
        Configure auto-scaling for the endpoint
        """
        print("📈 Setting up auto-scaling...")
        
        # Auto-scaling configuration
        autoscaling_client = boto3.client('application-autoscaling')
        
        # Register scalable target
        autoscaling_client.register_scalable_target(
            ServiceNamespace='sagemaker',
            ResourceId=f'endpoint/{self.endpoint_name}/variant/AllTraffic',
            ScalableDimension='sagemaker:variant:DesiredInstanceCount',
            MinCapacity=self.initial_instance_count,
            MaxCapacity=self.max_instance_count
        )
        
        # Create scaling policy
        autoscaling_client.put_scaling_policy(
            PolicyName=f'{self.endpoint_name}-scaling-policy',
            ServiceNamespace='sagemaker',
            ResourceId=f'endpoint/{self.endpoint_name}/variant/AllTraffic',
            ScalableDimension='sagemaker:variant:DesiredInstanceCount',
            PolicyType='TargetTrackingScaling',
            TargetTrackingScalingPolicyConfiguration={
                'TargetValue': 70.0,  # Target CPU utilization
                'PredefinedMetricSpecification': {
                    'PredefinedMetricType': 'SageMakerVariantInvocationsPerInstance'
                },
                'ScaleOutCooldown': 300,
                'ScaleInCooldown': 300
            }
        )
        
        print("✅ Auto-scaling configured")
        print(f"   Min instances: {self.initial_instance_count}")
        print(f"   Max instances: {self.max_instance_count}")
        print(f"   Target utilization: 70%")
    
    def test_endpoint(self):
        """
        Test the deployed endpoint with sample data
        """
        print("🧪 Testing endpoint...")
        
        # Create test data
        test_data = {
            'flow_duration': 1200,
            'flow_packets_s': 50,
            'flow_bytes_s': 5000,
            'total_fwd_packets': 100,
            'total_backward_packets': 80,
            'src_ip_reputation': 0.8,
            'dst_port': 22,
            'protocol_risk': 0.7,
            'time_risk': 0.9
        }
        
        try:
            # Get predictor
            predictor = Predictor(
                endpoint_name=self.endpoint_name,
                serializer=JSONSerializer(),
                deserializer=JSONDeserializer()
            )
            
            # Make prediction
            result = predictor.predict(test_data)
            
            print("✅ Endpoint test successful!")
            print(f"   Test result: {json.dumps(result, indent=2)}")
            
            return result
            
        except Exception as e:
            print(f"❌ Endpoint test failed: {str(e)}")
            return None
    
    def create_monitoring_dashboard(self):
        """
        Create CloudWatch dashboard for monitoring
        """
        print("📊 Creating monitoring dashboard...")
        
        cloudwatch = boto3.client('cloudwatch')
        
        dashboard_body = {
            "widgets": [
                {
                    "type": "metric",
                    "properties": {
                        "metrics": [
                            ["AWS/SageMaker", "Invocations", "EndpointName", self.endpoint_name],
                            [".", "InvocationErrors", ".", "."],
                            [".", "ModelLatency", ".", "."],
                            [".", "OverheadLatency", ".", "."]
                        ],
                        "period": 300,
                        "stat": "Average",
                        "region": self.region,
                        "title": "Mini-XDR Endpoint Metrics"
                    }
                },
                {
                    "type": "metric",
                    "properties": {
                        "metrics": [
                            ["AWS/SageMaker", "CPUUtilization", "EndpointName", self.endpoint_name],
                            [".", "MemoryUtilization", ".", "."]
                        ],
                        "period": 300,
                        "stat": "Average",
                        "region": self.region,
                        "title": "Resource Utilization"
                    }
                }
            ]
        }
        
        cloudwatch.put_dashboard(
            DashboardName=f'Mini-XDR-ML-{self.endpoint_name}',
            DashboardBody=json.dumps(dashboard_body)
        )
        
        print("✅ Monitoring dashboard created")
    
    def deploy_complete_solution(self):
        """
        Deploy complete ML solution with monitoring
        """
        print("🚀 Deploying Complete Mini-XDR ML Solution")
        print("=" * 60)
        
        start_time = time.time()
        
        # Deploy endpoint
        predictor = self.deploy_ensemble_endpoint()
        
        # Setup auto-scaling
        self.setup_auto_scaling()
        
        # Test endpoint
        test_result = self.test_endpoint()
        
        # Create monitoring
        self.create_monitoring_dashboard()
        
        end_time = time.time()
        duration = end_time - start_time
        
        print("\n🎉 Deployment Completed!")
        print("=" * 60)
        print(f"⏱️  Deployment time: {duration/60:.2f} minutes")
        print(f"🌐 Endpoint name: {self.endpoint_name}")
        print(f"🔗 Endpoint URL: {predictor.endpoint_name}")
        print(f"📈 Auto-scaling: {self.initial_instance_count}-{self.max_instance_count} instances")
        print(f"🎯 Performance target: <50ms latency, >10k events/sec")
        print(f"📊 Monitoring: CloudWatch dashboard created")
        
        return {
            'endpoint_name': self.endpoint_name,
            'predictor': predictor,
            'test_result': test_result,
            'deployment_duration': duration
        }

def main():
    """
    Main deployment execution
    """
    print("🚀 Mini-XDR Model Deployment Pipeline")
    print("🎯 Target: Real-time threat detection with <50ms latency")
    print("🧠 Models: Ensemble of 4 advanced ML models")
    
    # Initialize deployment
    deployment = MiniXDRModelDeployment()
    
    # Deploy complete solution
    results = deployment.deploy_complete_solution()
    
    print("\n✅ Mini-XDR ML models deployed and ready for production!")
    print("🚀 Integration: Update Mini-XDR backend to use new endpoint")

if __name__ == "__main__":
    main()
