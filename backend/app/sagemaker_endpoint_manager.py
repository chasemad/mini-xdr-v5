"""
SageMaker Endpoint Management with Auto-scaling and Cost Optimization
Manages endpoint lifecycle, auto-scaling, and cost controls for Mini-XDR
"""

import boto3
import json
import logging
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


class EndpointStatus(Enum):
    CREATING = "Creating"
    IN_SERVICE = "InService"
    UPDATING = "Updating"
    DELETING = "Deleting"
    FAILED = "Failed"
    OUT_OF_SERVICE = "OutOfService"


class ScalingPolicy(Enum):
    DEVELOPMENT = "development"  # Manual scaling, minimal instances
    TESTING = "testing"         # Auto-scale 0-2 instances
    PRODUCTION = "production"   # Auto-scale 2-10 instances
    COST_OPTIMIZED = "cost_optimized"  # Scale to zero when idle


class SageMakerEndpointManager:
    """
    Comprehensive SageMaker endpoint management with cost optimization
    """
    
    def __init__(self, region: str = "us-east-1"):
        self.region = region
        self.sagemaker = boto3.client('sagemaker', region_name=region)
        self.sagemaker_runtime = boto3.client('sagemaker-runtime', region_name=region)
        self.autoscaling = boto3.client('application-autoscaling', region_name=region)
        self.cloudwatch = boto3.client('cloudwatch', region_name=region)
        
        # Configuration
        self.endpoint_name = "mini-xdr-production-endpoint"
        self.model_name = "mini-xdr-production-model-fixed" 
        self.scaling_policy = ScalingPolicy.DEVELOPMENT
        
        # Cost tracking
        self.max_hourly_cost = 5.0  # $5/hour limit for development
        self.idle_timeout_minutes = 30  # Scale down after 30 minutes of inactivity
        
    async def create_endpoint_from_training_job(self, training_job_name: str) -> Dict[str, Any]:
        """Create SageMaker endpoint from completed training job"""
        try:
            logger.info(f"Creating endpoint from training job: {training_job_name}")
            
            # 1. Create model from training job
            model_response = self.sagemaker.create_model(
                ModelName=self.model_name,
                PrimaryContainer={
                    'Image': '763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:1.12.0-gpu-py38',
                    'ModelDataUrl': f's3://mini-xdr-ml-data-bucket-675076709589/models/{training_job_name}/output/model.tar.gz',
                    'Environment': {
                        'SAGEMAKER_PROGRAM': 'inference.py',
                        'SAGEMAKER_SUBMIT_DIRECTORY': f's3://mini-xdr-ml-data-bucket-675076709589/models/{training_job_name}/output/model.tar.gz'
                    }
                },
                ExecutionRoleArn='arn:aws:iam::675076709589:role/SageMakerExecutionRole'
            )
            
            # 2. Create endpoint configuration
            config_response = self.sagemaker.create_endpoint_config(
                EndpointConfigName=f"{self.endpoint_name}-config",
                ProductionVariants=[{
                    'VariantName': 'primary',
                    'ModelName': self.model_name,
                    'InitialInstanceCount': 1 if self.scaling_policy == ScalingPolicy.DEVELOPMENT else 0,
                    'InstanceType': 'ml.m5.large' if self.scaling_policy == ScalingPolicy.DEVELOPMENT else 'ml.c5.xlarge',
                    'InitialVariantWeight': 1
                }]
            )
            
            # 3. Create endpoint
            endpoint_response = self.sagemaker.create_endpoint(
                EndpointName=self.endpoint_name,
                EndpointConfigName=f"{self.endpoint_name}-config"
            )
            
            # 4. Setup auto-scaling if not development
            if self.scaling_policy != ScalingPolicy.DEVELOPMENT:
                await self._setup_auto_scaling()
            
            # 5. Setup cost monitoring
            await self._setup_cost_monitoring()
            
            logger.info(f"Endpoint creation initiated: {self.endpoint_name}")
            
            return {
                "success": True,
                "endpoint_name": self.endpoint_name,
                "model_name": self.model_name,
                "status": "Creating",
                "scaling_policy": self.scaling_policy.value,
                "estimated_cost_per_hour": self._estimate_hourly_cost()
            }
            
        except Exception as e:
            logger.error(f"Endpoint creation failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _setup_auto_scaling(self):
        """Setup auto-scaling for cost optimization"""
        try:
            # Register scalable target
            self.autoscaling.register_scalable_target(
                ServiceNamespace='sagemaker',
                ResourceId=f'endpoint/{self.endpoint_name}/variant/primary',
                ScalableDimension='sagemaker:variant:DesiredInstanceCount',
                MinCapacity=0 if self.scaling_policy == ScalingPolicy.COST_OPTIMIZED else 1,
                MaxCapacity=2 if self.scaling_policy == ScalingPolicy.TESTING else 10
            )
            
            # Create scaling policy based on invocations
            self.autoscaling.put_scaling_policy(
                PolicyName=f"{self.endpoint_name}-scaling-policy",
                ServiceNamespace='sagemaker',
                ResourceId=f'endpoint/{self.endpoint_name}/variant/primary',
                ScalableDimension='sagemaker:variant:DesiredInstanceCount',
                PolicyType='TargetTrackingScaling',
                TargetTrackingScalingPolicyConfiguration={
                    'TargetValue': 100.0,  # Target 100 invocations per minute
                    'PredefinedMetricSpecification': {
                        'PredefinedMetricType': 'SageMakerVariantInvocationsPerInstance'
                    },
                    'ScaleOutCooldown': 300,  # 5 minutes
                    'ScaleInCooldown': 600    # 10 minutes
                }
            )
            
            logger.info("Auto-scaling configured successfully")
            
        except Exception as e:
            logger.error(f"Auto-scaling setup failed: {e}")
    
    async def _setup_cost_monitoring(self):
        """Setup CloudWatch alarms for cost monitoring"""
        try:
            # Create alarm for high invocation costs
            self.cloudwatch.put_metric_alarm(
                AlarmName=f"{self.endpoint_name}-cost-alarm",
                ComparisonOperator='GreaterThanThreshold',
                EvaluationPeriods=1,
                MetricName='InvocationCount',
                Namespace='AWS/SageMaker',
                Period=3600,  # 1 hour
                Statistic='Sum',
                Threshold=1000.0,  # More than 1000 invocations per hour
                ActionsEnabled=True,
                AlarmActions=[
                    # Could add SNS topic for notifications
                ],
                AlarmDescription=f'Cost monitoring for {self.endpoint_name}',
                Dimensions=[
                    {
                        'Name': 'EndpointName',
                        'Value': self.endpoint_name
                    },
                    {
                        'Name': 'VariantName', 
                        'Value': 'primary'
                    }
                ]
            )
            
            logger.info("Cost monitoring configured")
            
        except Exception as e:
            logger.error(f"Cost monitoring setup failed: {e}")
    
    def _estimate_hourly_cost(self) -> float:
        """Estimate hourly cost based on instance type and scaling policy"""
        cost_per_hour = {
            'ml.m5.large': 0.115,    # Development
            'ml.c5.xlarge': 0.192,   # Testing  
            'ml.c5.2xlarge': 0.384,  # Production
            'ml.p3.2xlarge': 3.06    # GPU inference
        }
        
        instance_type = 'ml.m5.large' if self.scaling_policy == ScalingPolicy.DEVELOPMENT else 'ml.c5.xlarge'
        base_cost = cost_per_hour.get(instance_type, 0.115)
        
        # Estimate based on scaling policy
        if self.scaling_policy == ScalingPolicy.DEVELOPMENT:
            return base_cost  # 1 instance
        elif self.scaling_policy == ScalingPolicy.TESTING:
            return base_cost * 1.5  # Average 1.5 instances
        elif self.scaling_policy == ScalingPolicy.COST_OPTIMIZED:
            return base_cost * 0.3  # Mostly scaled to zero
        else:
            return base_cost * 3  # Production average
    
    async def scale_to_zero(self) -> Dict[str, Any]:
        """Scale endpoint to zero instances for cost savings"""
        try:
            logger.info(f"Scaling {self.endpoint_name} to zero instances")
            
            # Update auto-scaling to allow zero instances
            self.autoscaling.update_scalable_target(
                ServiceNamespace='sagemaker',
                ResourceId=f'endpoint/{self.endpoint_name}/variant/primary',
                ScalableDimension='sagemaker:variant:DesiredInstanceCount',
                MinCapacity=0,
                MaxCapacity=0
            )
            
            return {"success": True, "status": "scaling_to_zero", "cost_savings": "~$2.76/day"}
            
        except Exception as e:
            logger.error(f"Scale to zero failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def scale_up_on_demand(self) -> Dict[str, Any]:
        """Scale up endpoint when ML inference is needed"""
        try:
            logger.info(f"Scaling up {self.endpoint_name} for inference")
            
            # Update auto-scaling to allow instances
            self.autoscaling.update_scalable_target(
                ServiceNamespace='sagemaker',
                ResourceId=f'endpoint/{self.endpoint_name}/variant/primary', 
                ScalableDimension='sagemaker:variant:DesiredInstanceCount',
                MinCapacity=1,
                MaxCapacity=2
            )
            
            # Wait for endpoint to be ready
            waiter = self.sagemaker.get_waiter('endpoint_in_service')
            await asyncio.get_event_loop().run_in_executor(
                None, 
                lambda: waiter.wait(
                    EndpointName=self.endpoint_name,
                    WaiterConfig={'Delay': 30, 'MaxAttempts': 20}
                )
            )
            
            return {"success": True, "status": "ready_for_inference", "estimated_ready_time": "2-5 minutes"}
            
        except Exception as e:
            logger.error(f"Scale up failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def get_endpoint_metrics(self) -> Dict[str, Any]:
        """Get comprehensive endpoint metrics and cost information"""
        try:
            # Get endpoint status
            endpoint_response = self.sagemaker.describe_endpoint(EndpointName=self.endpoint_name)
            
            # Get instance count
            instance_count = 0
            for variant in endpoint_response.get('ProductionVariants', []):
                instance_count += variant.get('CurrentInstanceCount', 0)
            
            # Get recent invocation metrics
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=24)
            
            metrics_response = self.cloudwatch.get_metric_statistics(
                Namespace='AWS/SageMaker',
                MetricName='InvocationCount',
                Dimensions=[
                    {'Name': 'EndpointName', 'Value': self.endpoint_name},
                    {'Name': 'VariantName', 'Value': 'primary'}
                ],
                StartTime=start_time,
                EndTime=end_time,
                Period=3600,  # 1 hour periods
                Statistics=['Sum']
            )
            
            total_invocations = sum(point['Sum'] for point in metrics_response.get('Datapoints', []))
            estimated_cost_24h = self._estimate_hourly_cost() * 24
            
            return {
                "endpoint_name": self.endpoint_name,
                "status": endpoint_response['EndpointStatus'],
                "instance_count": instance_count,
                "instance_type": endpoint_response.get('ProductionVariants', [{}])[0].get('InstanceType'),
                "invocations_24h": int(total_invocations),
                "estimated_cost_24h": round(estimated_cost_24h, 2),
                "scaling_policy": self.scaling_policy.value,
                "last_updated": endpoint_response.get('LastModifiedTime', '').isoformat() if endpoint_response.get('LastModifiedTime') else None,
                "cost_optimization": {
                    "auto_scale_enabled": self.scaling_policy != ScalingPolicy.DEVELOPMENT,
                    "scale_to_zero_enabled": self.scaling_policy == ScalingPolicy.COST_OPTIMIZED,
                    "max_hourly_cost": self.max_hourly_cost,
                    "idle_timeout_minutes": self.idle_timeout_minutes
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get endpoint metrics: {e}")
            return {"error": str(e)}


# Global endpoint manager
_endpoint_manager = None

async def get_endpoint_manager() -> SageMakerEndpointManager:
    """Get or create endpoint manager instance"""
    global _endpoint_manager
    if _endpoint_manager is None:
        _endpoint_manager = SageMakerEndpointManager()
    return _endpoint_manager
