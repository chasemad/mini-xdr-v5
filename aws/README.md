# Mini-XDR AWS Infrastructure & ML Training

This directory contains the complete AWS infrastructure and advanced ML training pipeline for Mini-XDR cybersecurity system.

## 🎯 Overview

The AWS implementation provides enterprise-scale threat detection with advanced machine learning capabilities, processing 846,073+ cybersecurity events with 83+ sophisticated features.

### Key Components

- **🗃️ S3 Data Lake**: Organized storage for 846k+ cybersecurity events
- **⚡ AWS Glue ETL**: Distributed feature engineering with 83+ CICIDS2017 features
- **🧠 SageMaker Training**: 4 advanced ML models with ensemble learning
- **🚀 Real-time Inference**: Auto-scaling endpoints for live threat detection
- **📊 CloudWatch Monitoring**: Comprehensive observability and alerting

## 📁 Directory Structure

```
aws/
├── deployment/           # Core infrastructure deployment
│   ├── deploy-mini-xdr-aws.sh
│   ├── deploy-mini-xdr-code.sh
│   ├── deploy-frontend-aws.sh
│   └── deploy-complete-aws-system.sh
│
├── data-processing/      # Data lake and ETL pipelines
│   ├── setup-s3-data-lake.sh
│   └── glue-etl-pipeline.py
│
├── feature-engineering/  # Advanced feature extraction
│   └── advanced-feature-engineering.py
│
├── ml-training/          # Machine learning model training
│   └── sagemaker-training-pipeline.py
│
├── model-deployment/     # Production model deployment
│   └── sagemaker-deployment.py
│
├── monitoring/           # Pipeline orchestration and monitoring
│   └── ml-pipeline-orchestrator.py
│
├── utils/               # Utility and management scripts
│   ├── aws-services-control.sh
│   ├── tpot-security-control.sh
│   ├── update-pipeline.sh
│   └── configure-tpot-aws-connection.sh
│
└── deploy-complete-aws-ml-system.sh  # Master deployment script
```

## 🚀 Quick Start

### Complete System Deployment

Deploy the entire Mini-XDR system with advanced ML capabilities:

```bash
cd /Users/chasemad/Desktop/mini-xdr/aws
./deploy-complete-aws-ml-system.sh
```

**Duration**: 6-8 hours  
**Cost**: $200-500/month during training, $50-100/month for inference

### Step-by-Step Deployment

For more control, deploy components individually:

```bash
# 1. Deploy basic infrastructure
./deployment/deploy-mini-xdr-aws.sh
./deployment/deploy-mini-xdr-code.sh
./deployment/deploy-frontend-aws.sh

# 2. Setup ML data lake
./data-processing/setup-s3-data-lake.sh

# 3. Run feature engineering
python3 ./feature-engineering/advanced-feature-engineering.py

# 4. Train ML models
python3 ./ml-training/sagemaker-training-pipeline.py

# 5. Deploy models
python3 ./model-deployment/sagemaker-deployment.py
```

## 📊 Data Processing Pipeline

### Dataset Overview (846,073+ Events)

| Dataset | Events | Features | Description |
|---------|--------|----------|-------------|
| **CICIDS2017** | 799,989 | 83+ | Premium network flow features |
| **KDD Cup** | 41,000 | 41 | Classic intrusion detection |
| **Threat Intel** | 2,273 | Custom | Live threat feeds |
| **Synthetic** | 1,966 | Custom | Simulated attacks |

### Feature Engineering

The pipeline extracts **113 total features**:

- **83 CICIDS2017 Features**: Complete network flow analysis
- **30 Custom Features**: Threat intelligence integration

#### Feature Categories

```python
features = {
    'temporal': 15,           # Flow timing patterns
    'packet': 15,             # Packet size analysis  
    'traffic_rate': 6,        # Bandwidth utilization
    'protocol': 13,           # Protocol flag analysis
    'behavioral': 17,         # Network behavior patterns
    'threat_intelligence': 6, # Real-time threat feeds
    'behavioral_analysis': 5, # Attack pattern detection
    'attack_campaign': 6,     # Multi-stage attack indicators
    'time_based': 8,          # Temporal risk assessment
    'ensemble': 4             # Meta-features
}
```

## 🧠 Machine Learning Models

### Model Architecture

The system trains an ensemble of 4 advanced models:

#### 1. **Transformer Model**
- **Architecture**: Multi-head attention (6 layers, 8 heads)
- **Purpose**: Complex temporal pattern recognition
- **Features**: Sequence analysis with positional encoding
- **Training**: TensorFlow on ml.p3.8xlarge

#### 2. **XGBoost Ensemble** 
- **Architecture**: Gradient boosting with hyperparameter optimization
- **Purpose**: Feature importance analysis and classification
- **Features**: SHAP explainability, 1000+ estimators
- **Training**: Hyperparameter tuning with 20 jobs

#### 3. **LSTM Autoencoder**
- **Architecture**: Multi-layer LSTM with attention mechanism
- **Purpose**: Sequence reconstruction and anomaly scoring
- **Features**: 3 layers, 128 hidden units, attention heads
- **Training**: PyTorch on GPU instances

#### 4. **Isolation Forest Ensemble**
- **Architecture**: Multiple isolation forests with different parameters
- **Purpose**: Unsupervised anomaly detection
- **Features**: 5 model ensemble with weighted voting
- **Training**: Distributed scikit-learn

### Performance Targets

| Metric | Target | Description |
|--------|--------|-------------|
| **Detection Rate** | >99% | True positive rate |
| **False Positive Rate** | <0.5% | Minimize false alarms |
| **Precision** | >98% | Positive prediction accuracy |
| **F1 Score** | >98.5% | Balanced performance |
| **Inference Latency** | <50ms | Real-time response |
| **Throughput** | >10k events/sec | High-volume processing |

## ⚡ AWS Infrastructure

### SageMaker Configuration

```yaml
Training:
  Instance: ml.p3.8xlarge (4x V100 GPUs)
  Duration: 4-6 hours per model
  Hyperparameter Tuning: 20 parallel jobs
  
Inference:
  Instance: ml.c5.2xlarge
  Auto-scaling: 2-10 instances  
  Target Latency: <50ms
  
Storage:
  Data Lake: S3 with Intelligent Tiering
  Models: S3 with versioning
  Feature Store: SageMaker Feature Store
```

### Cost Optimization

| Phase | Monthly Cost | Optimization |
|-------|-------------|--------------|
| **Training** | $200-500 | Run only when retraining |
| **Inference** | $50-100 | Auto-scaling, spot instances |
| **Storage** | $10-20 | Intelligent tiering, lifecycle policies |

## 📋 Management Commands

### AWS ML Control

```bash
# Check ML system status
~/aws-ml-control.sh status

# Start/stop inference endpoints
~/aws-ml-control.sh start
~/aws-ml-control.sh stop

# Trigger model retraining
~/aws-ml-control.sh retrain

# View training logs
~/aws-ml-control.sh logs

# Cost analysis
~/aws-ml-control.sh costs
```

### System Management

```bash
# Overall system control
~/aws-services-control.sh status
~/aws-services-control.sh start
~/aws-services-control.sh stop

# TPOT security modes
~/tpot-security-control.sh testing   # Safe mode
~/tpot-security-control.sh live      # Production mode
~/tpot-security-control.sh lockdown  # Emergency stop

# Update deployments
~/update-pipeline.sh frontend
~/update-pipeline.sh backend
~/update-pipeline.sh both
```

## 🔄 ML Pipeline Orchestration

### Automated Pipeline

The orchestrator manages the complete ML workflow:

```bash
# Run complete pipeline
python3 monitoring/ml-pipeline-orchestrator.py --phase all

# Run specific phases
python3 monitoring/ml-pipeline-orchestrator.py --phase data
python3 monitoring/ml-pipeline-orchestrator.py --phase etl  
python3 monitoring/ml-pipeline-orchestrator.py --phase training
python3 monitoring/ml-pipeline-orchestrator.py --phase deployment

# Dry run (show execution plan)
python3 monitoring/ml-pipeline-orchestrator.py --dry-run
```

### Pipeline Phases

1. **Data Lake Setup** (30 minutes)
   - S3 bucket creation and configuration
   - Dataset upload and organization
   - Security policy configuration

2. **Feature Engineering** (2-3 hours)
   - AWS Glue ETL job execution
   - 83+ feature extraction from CICIDS2017
   - Custom threat intelligence integration

3. **Model Training** (4-6 hours)
   - Parallel SageMaker training jobs
   - Hyperparameter optimization
   - Model evaluation and selection

4. **Model Deployment** (30 minutes)
   - Endpoint creation and configuration
   - Auto-scaling setup
   - Health monitoring activation

## 📊 Monitoring & Alerting

### CloudWatch Dashboards

- **Model Performance**: Accuracy, latency, throughput
- **Resource Utilization**: CPU, memory, GPU usage
- **Cost Tracking**: Training and inference costs
- **Data Flow**: ETL job status and data quality

### Alerts

- **High Error Rate**: >10 errors in 10 minutes
- **High Latency**: >100ms average response time
- **Cost Threshold**: >$100/day spending
- **Training Failures**: Job completion status

### SNS Notifications

Configure email/SMS alerts:

```bash
# Update configuration
vim /tmp/ml-pipeline-config.yaml

monitoring:
  email: "your-email@domain.com"
  slack_webhook: "https://hooks.slack.com/..."
```

## 🔐 Security

### Access Control

- **IAM Roles**: Least privilege principle
- **VPC**: Private subnets for sensitive data
- **Encryption**: AES-256 at rest, TLS 1.3 in transit
- **API Keys**: Secure credential management

### Data Protection

- **PII Detection**: Automated sensitive data masking
- **Audit Trail**: Complete lineage tracking
- **Backup**: Automated S3 cross-region replication
- **Compliance**: SOC 2, ISO 27001 ready

## 🧪 Testing

### Unit Tests

```bash
# Test feature engineering
python3 -m pytest feature-engineering/test_features.py

# Test model training
python3 -m pytest ml-training/test_training.py

# Test deployment
python3 -m pytest model-deployment/test_deployment.py
```

### Integration Tests

```bash
# End-to-end pipeline test
python3 monitoring/ml-pipeline-orchestrator.py --dry-run

# Endpoint testing
curl -X POST https://sagemaker-endpoint.amazonaws.com/invocations \
  -H "Content-Type: application/json" \
  -d '{"features": {...}}'
```

## 📈 Performance Optimization

### Training Optimization

- **Distributed Training**: Multi-GPU, multi-instance
- **Mixed Precision**: FP16 for faster training
- **Data Loading**: Optimized S3 prefetch
- **Hyperparameter Tuning**: Bayesian optimization

### Inference Optimization

- **Model Compilation**: TensorRT/TorchScript optimization
- **Batch Prediction**: Efficient batch processing
- **Caching**: Feature and prediction caching
- **Auto-scaling**: Demand-based scaling

## 🔧 Troubleshooting

### Common Issues

#### Training Job Failures
```bash
# Check training logs
aws logs get-log-events --log-group-name /aws/sagemaker/TrainingJobs

# Resource limits
aws service-quotas get-service-quota --service-code sagemaker
```

#### Endpoint Issues
```bash
# Check endpoint status
aws sagemaker describe-endpoint --endpoint-name mini-xdr-endpoint

# Test endpoint health
curl -f https://endpoint-url/ping
```

#### Data Pipeline Issues
```bash
# Check Glue job logs
aws logs get-log-events --log-group-name /aws-glue/jobs/error

# Validate S3 permissions
aws s3 ls s3://mini-xdr-ml-data-bucket/
```

### Support Resources

- **AWS Documentation**: [SageMaker Developer Guide](https://docs.aws.amazon.com/sagemaker/)
- **Community**: [SageMaker Examples](https://github.com/aws/amazon-sagemaker-examples)
- **Debugging**: Enable detailed logging in CloudWatch

## 📚 Additional Resources

### Documentation

- [AWS ML Training Migration Prompt](../docs/AWS_ML_TRAINING_MIGRATION_PROMPT.md)
- [SageMaker Best Practices](https://docs.aws.amazon.com/sagemaker/latest/dg/best-practices.html)
- [Cost Optimization Guide](https://aws.amazon.com/sagemaker/pricing/)

### Examples

- [Feature Engineering Examples](feature-engineering/examples/)
- [Model Training Notebooks](ml-training/notebooks/)
- [Deployment Templates](model-deployment/templates/)

---

## 🎯 Success Criteria

After successful deployment, you should achieve:

✅ **Real-time Threat Detection**: <50ms inference latency  
✅ **High Accuracy**: >99% detection rate, <0.5% false positives  
✅ **Scalable Processing**: >10k events/sec throughput  
✅ **Advanced Features**: 83+ CICIDS2017 + custom intelligence  
✅ **Enterprise Security**: SOC-ready monitoring and alerting  
✅ **Cost Optimization**: Automated scaling and cost controls  

## 🎉 Getting Started

1. **Prerequisites**: AWS CLI, Python 3.8+, sufficient IAM permissions
2. **Deploy**: Run `./deploy-complete-aws-ml-system.sh`
3. **Monitor**: Use CloudWatch dashboards and management scripts  
4. **Optimize**: Adjust auto-scaling and cost controls
5. **Integrate**: Connect with existing Mini-XDR workflows

Your Mini-XDR system now has world-class ML capabilities for advanced threat detection! 🛡️
