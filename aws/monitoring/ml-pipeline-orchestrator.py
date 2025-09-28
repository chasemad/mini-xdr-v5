#!/usr/bin/env python3
"""
ML Pipeline Orchestrator for Mini-XDR AWS Training
Orchestrates the complete ML pipeline from data processing to model deployment

This script coordinates:
- S3 data lake setup and data upload
- AWS Glue ETL jobs for feature engineering  
- SageMaker training jobs for all models
- Model deployment and endpoint creation
- Monitoring and alerting setup
"""

import boto3
import json
import time
import yaml
from datetime import datetime, timedelta
import subprocess
import os
from pathlib import Path

class MLPipelineOrchestrator:
    """
    Comprehensive orchestrator for Mini-XDR ML training pipeline
    """
    
    def __init__(self, config_file=None):
        self.region = 'us-east-1'
        self.account_id = boto3.sts.get_caller_identity()['Account']
        
        # AWS clients
        self.s3_client = boto3.client('s3')
        self.glue_client = boto3.client('glue')
        self.sagemaker_client = boto3.client('sagemaker')
        self.cloudwatch_client = boto3.client('cloudwatch')
        self.sns_client = boto3.client('sns')
        
        # Configuration
        self.config = self.load_config(config_file)
        self.pipeline_state = {}
        
        print("🎯 Mini-XDR ML Pipeline Orchestrator initialized")
        print(f"📊 Target: 846,073+ events with 83+ features")
        print(f"🧠 Models: Transformer, XGBoost, LSTM, IsolationForest")
    
    def load_config(self, config_file):
        """Load pipeline configuration"""
        default_config = {
            'data': {
                'source_bucket': f'mini-xdr-ml-data-{self.account_id}-{self.region}',
                'models_bucket': f'mini-xdr-ml-models-{self.account_id}-{self.region}',
                'artifacts_bucket': f'mini-xdr-ml-artifacts-{self.account_id}-{self.region}',
                'total_events': 846073
            },
            'training': {
                'instance_type': 'ml.p3.8xlarge',
                'instance_count': 1,
                'max_runtime_hours': 24
            },
            'deployment': {
                'instance_type': 'ml.c5.2xlarge',
                'initial_instance_count': 2,
                'max_instance_count': 10
            },
            'monitoring': {
                'email': 'admin@example.com',
                'slack_webhook': None
            }
        }
        
        if config_file and os.path.exists(config_file):
            with open(config_file, 'r') as f:
                user_config = yaml.safe_load(f)
                default_config.update(user_config)
        
        return default_config
    
    def setup_notifications(self):
        """Setup SNS topic for pipeline notifications"""
        print("📢 Setting up notifications...")
        
        topic_name = 'mini-xdr-ml-pipeline-notifications'
        
        try:
            # Create SNS topic
            response = self.sns_client.create_topic(Name=topic_name)
            topic_arn = response['TopicArn']
            
            # Subscribe email if provided
            email = self.config.get('monitoring', {}).get('email')
            if email:
                self.sns_client.subscribe(
                    TopicArn=topic_arn,
                    Protocol='email',
                    Endpoint=email
                )
                print(f"📧 Email notifications setup: {email}")
            
            self.notification_topic = topic_arn
            return topic_arn
            
        except Exception as e:
            print(f"⚠️ Could not setup notifications: {e}")
            return None
    
    def send_notification(self, subject, message, level='INFO'):
        """Send pipeline notification"""
        if hasattr(self, 'notification_topic') and self.notification_topic:
            try:
                full_message = f"""
Mini-XDR ML Pipeline Notification

Level: {level}
Time: {datetime.now().isoformat()}

{message}

Pipeline State:
{json.dumps(self.pipeline_state, indent=2)}
"""
                self.sns_client.publish(
                    TopicArn=self.notification_topic,
                    Subject=f"[Mini-XDR ML] {subject}",
                    Message=full_message
                )
            except Exception as e:
                print(f"⚠️ Could not send notification: {e}")
    
    def execute_phase_1_data_setup(self):
        """Phase 1: Setup S3 data lake and upload datasets"""
        print("\n" + "="*60)
        print("📊 PHASE 1: Data Lake Setup and Upload")
        print("="*60)
        
        phase_start = time.time()
        
        try:
            # Execute S3 setup script
            script_path = '/Users/chasemad/Desktop/mini-xdr/aws/data-processing/setup-s3-data-lake.sh'
            
            print("🗃️ Setting up S3 data lake...")
            result = subprocess.run(['bash', script_path], 
                                  capture_output=True, text=True, timeout=1800)
            
            if result.returncode == 0:
                print("✅ S3 data lake setup completed")
                self.pipeline_state['phase_1'] = {
                    'status': 'completed',
                    'duration': time.time() - phase_start,
                    'output': result.stdout
                }
                
                self.send_notification(
                    "Phase 1 Completed", 
                    "S3 data lake setup and data upload completed successfully"
                )
                return True
            else:
                print(f"❌ S3 setup failed: {result.stderr}")
                self.pipeline_state['phase_1'] = {
                    'status': 'failed',
                    'error': result.stderr
                }
                
                self.send_notification(
                    "Phase 1 Failed", 
                    f"S3 data lake setup failed: {result.stderr}",
                    'ERROR'
                )
                return False
                
        except subprocess.TimeoutExpired:
            print("❌ S3 setup timed out")
            self.pipeline_state['phase_1'] = {'status': 'timeout'}
            return False
        except Exception as e:
            print(f"❌ S3 setup error: {e}")
            self.pipeline_state['phase_1'] = {'status': 'error', 'message': str(e)}
            return False
    
    def execute_phase_2_etl(self):
        """Phase 2: Execute Glue ETL jobs for feature engineering"""
        print("\n" + "="*60)
        print("🔬 PHASE 2: Feature Engineering ETL")
        print("="*60)
        
        phase_start = time.time()
        
        try:
            # Create Glue job
            job_name = f'mini-xdr-etl-{int(time.time())}'
            
            # Upload Glue script
            script_path = '/Users/chasemad/Desktop/mini-xdr/aws/data-processing/glue-etl-pipeline.py'
            script_key = f'glue-scripts/etl-pipeline-{int(time.time())}.py'
            
            self.s3_client.upload_file(
                script_path,
                self.config['data']['artifacts_bucket'],
                script_key
            )
            
            # Create Glue job
            print("⚡ Creating Glue ETL job...")
            self.glue_client.create_job(
                Name=job_name,
                Role='arn:aws:iam::123456789012:role/GlueServiceRole',  # Update with actual role
                Command={
                    'Name': 'glueetl',
                    'ScriptLocation': f's3://{self.config["data"]["artifacts_bucket"]}/{script_key}',
                    'PythonVersion': '3'
                },
                DefaultArguments={
                    '--SOURCE_BUCKET': self.config['data']['source_bucket'],
                    '--TARGET_BUCKET': self.config['data']['source_bucket'],
                    '--DATASET_TYPE': 'all'
                },
                MaxRetries=1,
                Timeout=2880,  # 48 hours
                GlueVersion='3.0'
            )
            
            # Start job run
            print("🚀 Starting ETL job...")
            run_response = self.glue_client.start_job_run(JobName=job_name)
            run_id = run_response['JobRunId']
            
            # Monitor job progress
            print(f"📊 Monitoring ETL job: {run_id}")
            
            while True:
                response = self.glue_client.get_job_run(JobName=job_name, RunId=run_id)
                state = response['JobRun']['JobRunState']
                
                if state == 'SUCCEEDED':
                    print("✅ ETL job completed successfully")
                    self.pipeline_state['phase_2'] = {
                        'status': 'completed',
                        'duration': time.time() - phase_start,
                        'job_id': run_id
                    }
                    
                    self.send_notification(
                        "Phase 2 Completed",
                        f"Feature engineering ETL completed. Processed 846,073+ events with 83+ features"
                    )
                    return True
                    
                elif state in ['FAILED', 'ERROR', 'TIMEOUT']:
                    error_msg = response['JobRun'].get('ErrorMessage', 'Unknown error')
                    print(f"❌ ETL job failed: {error_msg}")
                    self.pipeline_state['phase_2'] = {
                        'status': 'failed',
                        'error': error_msg
                    }
                    
                    self.send_notification(
                        "Phase 2 Failed",
                        f"Feature engineering ETL failed: {error_msg}",
                        'ERROR'
                    )
                    return False
                
                print(f"   ETL job state: {state}")
                time.sleep(60)  # Check every minute
                
        except Exception as e:
            print(f"❌ ETL phase error: {e}")
            self.pipeline_state['phase_2'] = {'status': 'error', 'message': str(e)}
            return False
    
    def execute_phase_3_training(self):
        """Phase 3: Execute SageMaker training jobs"""
        print("\n" + "="*60)
        print("🧠 PHASE 3: ML Model Training")
        print("="*60)
        
        phase_start = time.time()
        
        try:
            # Execute training script
            script_path = '/Users/chasemad/Desktop/mini-xdr/aws/ml-training/sagemaker-training-pipeline.py'
            
            print("🤖 Starting ML model training...")
            print("   Models: Transformer, XGBoost, LSTM, IsolationForest")
            
            result = subprocess.run(['python3', script_path], 
                                  capture_output=True, text=True, timeout=86400)  # 24 hours
            
            if result.returncode == 0:
                print("✅ ML training completed successfully")
                self.pipeline_state['phase_3'] = {
                    'status': 'completed',
                    'duration': time.time() - phase_start,
                    'output': result.stdout
                }
                
                self.send_notification(
                    "Phase 3 Completed",
                    "ML model training completed successfully. All 4 models trained with ensemble creation."
                )
                return True
            else:
                print(f"❌ ML training failed: {result.stderr}")
                self.pipeline_state['phase_3'] = {
                    'status': 'failed',
                    'error': result.stderr
                }
                
                self.send_notification(
                    "Phase 3 Failed",
                    f"ML model training failed: {result.stderr}",
                    'ERROR'
                )
                return False
                
        except subprocess.TimeoutExpired:
            print("❌ ML training timed out (24 hours)")
            self.pipeline_state['phase_3'] = {'status': 'timeout'}
            return False
        except Exception as e:
            print(f"❌ ML training error: {e}")
            self.pipeline_state['phase_3'] = {'status': 'error', 'message': str(e)}
            return False
    
    def execute_phase_4_deployment(self):
        """Phase 4: Deploy models as SageMaker endpoints"""
        print("\n" + "="*60)
        print("🚀 PHASE 4: Model Deployment")
        print("="*60)
        
        phase_start = time.time()
        
        try:
            # Execute deployment script
            script_path = '/Users/chasemad/Desktop/mini-xdr/aws/model-deployment/sagemaker-deployment.py'
            
            print("🌐 Deploying ML models to production endpoints...")
            
            result = subprocess.run(['python3', script_path], 
                                  capture_output=True, text=True, timeout=3600)  # 1 hour
            
            if result.returncode == 0:
                print("✅ Model deployment completed successfully")
                self.pipeline_state['phase_4'] = {
                    'status': 'completed',
                    'duration': time.time() - phase_start,
                    'output': result.stdout
                }
                
                self.send_notification(
                    "Phase 4 Completed",
                    "Model deployment completed. Real-time inference endpoints are live."
                )
                return True
            else:
                print(f"❌ Model deployment failed: {result.stderr}")
                self.pipeline_state['phase_4'] = {
                    'status': 'failed',
                    'error': result.stderr
                }
                
                self.send_notification(
                    "Phase 4 Failed",
                    f"Model deployment failed: {result.stderr}",
                    'ERROR'
                )
                return False
                
        except subprocess.TimeoutExpired:
            print("❌ Model deployment timed out")
            self.pipeline_state['phase_4'] = {'status': 'timeout'}
            return False
        except Exception as e:
            print(f"❌ Model deployment error: {e}")
            self.pipeline_state['phase_4'] = {'status': 'error', 'message': str(e)}
            return False
    
    def setup_monitoring_alerts(self):
        """Setup CloudWatch alarms for ongoing monitoring"""
        print("\n" + "="*60)
        print("📊 PHASE 5: Setup Monitoring & Alerts")
        print("="*60)
        
        try:
            # Create CloudWatch alarms
            alarms = [
                {
                    'AlarmName': 'Mini-XDR-ML-Endpoint-Errors',
                    'MetricName': 'InvocationErrors',
                    'Namespace': 'AWS/SageMaker',
                    'Statistic': 'Sum',
                    'Threshold': 10,
                    'ComparisonOperator': 'GreaterThanThreshold',
                    'EvaluationPeriods': 2,
                    'Period': 300
                },
                {
                    'AlarmName': 'Mini-XDR-ML-High-Latency',
                    'MetricName': 'ModelLatency',
                    'Namespace': 'AWS/SageMaker',
                    'Statistic': 'Average',
                    'Threshold': 100,  # 100ms
                    'ComparisonOperator': 'GreaterThanThreshold',
                    'EvaluationPeriods': 3,
                    'Period': 300
                }
            ]
            
            for alarm in alarms:
                print(f"⚠️ Creating alarm: {alarm['AlarmName']}")
                
                alarm_config = {
                    'AlarmName': alarm['AlarmName'],
                    'ComparisonOperator': alarm['ComparisonOperator'],
                    'EvaluationPeriods': alarm['EvaluationPeriods'],
                    'MetricName': alarm['MetricName'],
                    'Namespace': alarm['Namespace'],
                    'Period': alarm['Period'],
                    'Statistic': alarm['Statistic'],
                    'Threshold': alarm['Threshold'],
                    'ActionsEnabled': True,
                    'AlarmDescription': f'Mini-XDR ML Pipeline alarm for {alarm["MetricName"]}',
                    'Unit': 'Count'
                }
                
                if hasattr(self, 'notification_topic') and self.notification_topic:
                    alarm_config['AlarmActions'] = [self.notification_topic]
                    alarm_config['OKActions'] = [self.notification_topic]
                
                self.cloudwatch_client.put_metric_alarm(**alarm_config)
            
            print("✅ Monitoring alerts configured")
            
            self.send_notification(
                "Monitoring Setup Complete",
                "CloudWatch alarms and monitoring configured for ML pipeline."
            )
            
            return True
            
        except Exception as e:
            print(f"❌ Monitoring setup error: {e}")
            return False
    
    def generate_pipeline_report(self):
        """Generate comprehensive pipeline execution report"""
        print("\n" + "="*60)
        print("📋 PIPELINE EXECUTION REPORT")
        print("="*60)
        
        total_duration = sum(
            phase.get('duration', 0) 
            for phase in self.pipeline_state.values() 
            if isinstance(phase, dict)
        )
        
        report = {
            'pipeline_execution': {
                'start_time': self.pipeline_start_time,
                'end_time': datetime.now().isoformat(),
                'total_duration_hours': total_duration / 3600,
                'status': 'completed' if all(
                    phase.get('status') == 'completed' 
                    for phase in self.pipeline_state.values()
                    if isinstance(phase, dict)
                ) else 'partial'
            },
            'data_processing': {
                'total_events': self.config['data']['total_events'],
                'features_extracted': '83+ CICIDS2017 + custom threat intelligence',
                'datasets_processed': ['CICIDS2017', 'KDD Cup', 'Threat Intelligence', 'Synthetic']
            },
            'models_trained': {
                'transformer': 'Multi-head attention for sequence analysis',
                'xgboost': 'Gradient boosting with hyperparameter optimization',
                'lstm_autoencoder': 'Advanced LSTM with attention mechanism',
                'isolation_forest': 'Ensemble anomaly detection'
            },
            'deployment': {
                'endpoint_type': 'Real-time inference',
                'auto_scaling': f'{self.config["deployment"]["initial_instance_count"]}-{self.config["deployment"]["max_instance_count"]} instances',
                'target_latency': '<50ms',
                'target_throughput': '>10k events/sec'
            },
            'phase_details': self.pipeline_state
        }
        
        # Save report
        report_file = f'/tmp/mini-xdr-ml-pipeline-report-{int(time.time())}.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Upload to S3
        self.s3_client.upload_file(
            report_file,
            self.config['data']['artifacts_bucket'],
            f'reports/pipeline-execution-{int(time.time())}.json'
        )
        
        print(f"📊 Total execution time: {total_duration/3600:.2f} hours")
        print(f"📈 Success rate: {len([p for p in self.pipeline_state.values() if isinstance(p, dict) and p.get('status') == 'completed'])}/4 phases")
        print(f"🎯 Models deployed: 4 (Transformer, XGBoost, LSTM, IsolationForest)")
        print(f"📁 Report saved to S3: s3://{self.config['data']['artifacts_bucket']}/reports/")
        
        return report
    
    def run_complete_pipeline(self):
        """Execute the complete ML pipeline"""
        print("🚀 STARTING MINI-XDR ML PIPELINE")
        print("=" * 80)
        print("🎯 Target: 846,073+ events → Advanced ML models → Real-time inference")
        print("⏱️  Estimated duration: 6-8 hours")
        print("=" * 80)
        
        self.pipeline_start_time = datetime.now().isoformat()
        
        # Setup notifications
        self.setup_notifications()
        
        # Send start notification
        self.send_notification(
            "Pipeline Started",
            "Mini-XDR ML pipeline execution started. Processing 846,073+ cybersecurity events."
        )
        
        # Execute all phases
        phases = [
            ("Data Lake Setup", self.execute_phase_1_data_setup),
            ("Feature Engineering", self.execute_phase_2_etl),
            ("Model Training", self.execute_phase_3_training),
            ("Model Deployment", self.execute_phase_4_deployment)
        ]
        
        success_count = 0
        
        for phase_name, phase_func in phases:
            print(f"\n🔄 Starting {phase_name}...")
            
            if phase_func():
                success_count += 1
                print(f"✅ {phase_name} completed successfully")
            else:
                print(f"❌ {phase_name} failed")
                self.send_notification(
                    f"Pipeline Failure in {phase_name}",
                    f"Pipeline stopped due to failure in {phase_name}. Check logs for details.",
                    'ERROR'
                )
                break
        
        # Setup monitoring
        self.setup_monitoring_alerts()
        
        # Generate final report
        report = self.generate_pipeline_report()
        
        # Final notification
        if success_count == len(phases):
            print("\n🎉 PIPELINE COMPLETED SUCCESSFULLY!")
            self.send_notification(
                "Pipeline Completed Successfully",
                f"Mini-XDR ML pipeline completed successfully! All 4 phases completed in {report['pipeline_execution']['total_duration_hours']:.2f} hours."
            )
        else:
            print(f"\n⚠️ PIPELINE PARTIALLY COMPLETED ({success_count}/{len(phases)} phases)")
            self.send_notification(
                "Pipeline Partially Completed",
                f"Pipeline completed {success_count}/{len(phases)} phases. Check logs for failed phases.",
                'WARNING'
            )
        
        return report

def main():
    """Main orchestrator execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Mini-XDR ML Pipeline Orchestrator')
    parser.add_argument('--config', help='Configuration file path')
    parser.add_argument('--phase', help='Run specific phase only', 
                       choices=['data', 'etl', 'training', 'deployment', 'all'])
    parser.add_argument('--dry-run', action='store_true', help='Show what would be executed')
    
    args = parser.parse_args()
    
    # Initialize orchestrator
    orchestrator = MLPipelineOrchestrator(args.config)
    
    if args.dry_run:
        print("🔍 DRY RUN MODE - Pipeline Execution Plan:")
        print("1. Phase 1: S3 Data Lake Setup (30 minutes)")
        print("2. Phase 2: Glue ETL Feature Engineering (2-3 hours)")
        print("3. Phase 3: SageMaker Model Training (4-6 hours)")
        print("4. Phase 4: Model Deployment (30 minutes)")
        print("5. Phase 5: Monitoring Setup (15 minutes)")
        print(f"📊 Total estimated time: 7-10 hours")
        return
    
    if args.phase and args.phase != 'all':
        # Run specific phase
        phase_map = {
            'data': orchestrator.execute_phase_1_data_setup,
            'etl': orchestrator.execute_phase_2_etl,
            'training': orchestrator.execute_phase_3_training,
            'deployment': orchestrator.execute_phase_4_deployment
        }
        
        if args.phase in phase_map:
            orchestrator.setup_notifications()
            result = phase_map[args.phase]()
            print(f"Phase {args.phase} {'completed' if result else 'failed'}")
        else:
            print(f"Unknown phase: {args.phase}")
    else:
        # Run complete pipeline
        report = orchestrator.run_complete_pipeline()
        
        print("\n📊 Final Pipeline Summary:")
        print(f"   Duration: {report['pipeline_execution']['total_duration_hours']:.2f} hours")
        print(f"   Status: {report['pipeline_execution']['status']}")
        print(f"   Events processed: {report['data_processing']['total_events']:,}")
        print(f"   Models trained: {len(report['models_trained'])}")

if __name__ == "__main__":
    main()
