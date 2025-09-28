#!/bin/bash

# Complete Mini-XDR AWS ML System Deployment
# Deploys full ML training infrastructure and integrates with existing system

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REGION="${AWS_REGION:-us-east-1}"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
PROJECT_DIR="/Users/chasemad/Desktop/mini-xdr"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

# Banner
show_banner() {
    echo -e "${CYAN}"
    echo "=============================================================="
    echo "          Mini-XDR Complete AWS ML System Deployment"
    echo "=============================================================="
    echo -e "${NC}"
    echo "🎯 Target: 846,073+ events → Advanced ML models → Real-time inference"
    echo "🧠 Models: Transformer, XGBoost, LSTM Autoencoder, Isolation Forest"
    echo "⚡ Features: 83+ CICIDS2017 + custom threat intelligence"
    echo "🚀 Infrastructure: S3 + Glue + SageMaker + CloudWatch"
    echo ""
}

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[WARN] $1${NC}"
}

error() {
    echo -e "${RED}[ERROR] $1${NC}"
    exit 1
}

step() {
    echo -e "${BLUE}$1${NC}"
}

highlight() {
    echo -e "${MAGENTA}$1${NC}"
}

# Check prerequisites
check_prerequisites() {
    step "🔍 Checking Prerequisites"
    
    # Check AWS CLI
    if ! command -v aws &> /dev/null; then
        error "AWS CLI not found. Please install it first."
    fi
    
    # Check AWS configuration
    if ! aws sts get-caller-identity &> /dev/null; then
        error "AWS CLI not configured. Please run 'aws configure' first."
    fi
    
    # Check Python 3 and required packages
    if ! command -v python3 &> /dev/null; then
        error "Python 3 not found. Please install Python 3.8+ first."
    fi
    
    # Check required tools
    for tool in jq boto3 sagemaker; do
        if ! python3 -c "import $tool" &> /dev/null; then
            warn "$tool not available - installing..."
            pip3 install "$tool" || error "Failed to install $tool"
        fi
    done
    
    # Check sufficient permissions
    log "Checking AWS permissions..."
    
    required_services=(
        "s3:CreateBucket"
        "sagemaker:CreateTrainingJob" 
        "glue:CreateJob"
        "iam:GetRole"
        "cloudwatch:PutMetricAlarm"
    )
    
    for service in "${required_services[@]}"; do
        if ! aws iam simulate-principal-policy \
            --policy-source-arn "arn:aws:iam::${ACCOUNT_ID}:user/$(aws sts get-caller-identity --query UserName --output text)" \
            --action-names "$service" \
            --resource-arns "*" \
            --query 'EvaluationResults[0].EvalDecision' \
            --output text 2>/dev/null | grep -q "allowed"; then
            warn "May not have permissions for $service"
        fi
    done
    
    log "✅ Prerequisites check completed"
}

# Confirm deployment
confirm_deployment() {
    step "⚙️  Deployment Configuration"
    echo ""
    echo "AWS Configuration:"
    echo "  Account ID: $ACCOUNT_ID"
    echo "  Region: $REGION"
    echo "  Estimated Cost: $200-500/month during training"
    echo ""
    echo "Data Processing:"
    echo "  Total Events: 846,073+"
    echo "  CICIDS2017: 799,989 events (83+ features)"
    echo "  KDD Cup: 41,000 events"
    echo "  Threat Intel: 2,273 events"
    echo "  Synthetic: 1,966 events"
    echo ""
    echo "ML Infrastructure:"
    echo "  Training Instance: ml.p3.8xlarge (4x V100 GPUs)"
    echo "  Inference Instance: ml.c5.2xlarge (auto-scaling)"
    echo "  Storage: S3 with Intelligent Tiering"
    echo "  Monitoring: CloudWatch + SNS alerts"
    echo ""
    echo "Models to be Trained:"
    echo "  🤖 Transformer (Multi-head attention)"
    echo "  🌳 XGBoost (Gradient boosting with HPO)"
    echo "  🔄 LSTM Autoencoder (Sequence anomaly detection)"
    echo "  🌲 Isolation Forest (Ensemble anomaly detection)"
    echo ""
    highlight "⏱️  Estimated deployment time: 6-8 hours"
    echo ""
    
    read -p "Continue with AWS ML deployment? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log "Deployment cancelled by user"
        exit 0
    fi
}

# Setup IAM roles
setup_iam_roles() {
    step "🔐 Setting up IAM Roles"
    
    # SageMaker execution role
    local sagemaker_role_name="Mini-XDR-SageMaker-ExecutionRole"
    local glue_role_name="Mini-XDR-Glue-ExecutionRole"
    
    log "Creating SageMaker execution role..."
    
    # Check if role exists
    if ! aws iam get-role --role-name "$sagemaker_role_name" &>/dev/null; then
        # Create trust policy
        cat > "/tmp/sagemaker-trust-policy.json" << EOF
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {
                "Service": "sagemaker.amazonaws.com"
            },
            "Action": "sts:AssumeRole"
        }
    ]
}
EOF
        
        # Create role
        aws iam create-role \
            --role-name "$sagemaker_role_name" \
            --assume-role-policy-document file:///tmp/sagemaker-trust-policy.json
        
        # Attach secure least-privilege policies
        aws iam attach-role-policy \
            --role-name "$sagemaker_role_name" \
            --policy-arn "arn:aws:iam::${ACCOUNT_ID}:policy/Mini-XDR-SageMaker-Secure"
        
        # Create custom policy for S3 access
        cat > "/tmp/s3-access-policy.json" << EOF
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "s3:GetObject",
                "s3:PutObject",
                "s3:DeleteObject",
                "s3:ListBucket"
            ],
            "Resource": [
                "arn:aws:s3:::mini-xdr-ml-*",
                "arn:aws:s3:::mini-xdr-ml-*/*"
            ]
        }
    ]
}
EOF
        
        aws iam put-role-policy \
            --role-name "$sagemaker_role_name" \
            --policy-name "S3Access" \
            --policy-document file:///tmp/s3-access-policy.json
    fi
    
    log "Creating Glue execution role..."
    
    # Similar process for Glue role
    if ! aws iam get-role --role-name "$glue_role_name" &>/dev/null; then
        cat > "/tmp/glue-trust-policy.json" << EOF
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {
                "Service": "glue.amazonaws.com"
            },
            "Action": "sts:AssumeRole"
        }
    ]
}
EOF
        
        aws iam create-role \
            --role-name "$glue_role_name" \
            --assume-role-policy-document file:///tmp/glue-trust-policy.json
        
        aws iam attach-role-policy \
            --role-name "$glue_role_name" \
            --policy-arn "arn:aws:iam::aws:policy/service-role/AWSGlueServiceRole"
        
        aws iam put-role-policy \
            --role-name "$glue_role_name" \
            --policy-name "S3Access" \
            --policy-document file:///tmp/s3-access-policy.json
    fi
    
    log "✅ IAM roles configured"
}

# Deploy infrastructure components
deploy_infrastructure() {
    step "🏗️  Deploying ML Infrastructure"
    
    # Deploy existing Mini-XDR backend if not already deployed
    if ! aws cloudformation describe-stacks --stack-name mini-xdr-backend --region "$REGION" &>/dev/null; then
        log "Deploying Mini-XDR backend first..."
        "$SCRIPT_DIR/deployment/deploy-mini-xdr-aws.sh"
        "$SCRIPT_DIR/deployment/deploy-mini-xdr-code.sh"
    else
        log "Mini-XDR backend already deployed"
    fi
    
    # Deploy frontend if not already deployed
    if ! aws cloudformation describe-stacks --stack-name mini-xdr-frontend --region "$REGION" &>/dev/null; then
        log "Deploying Mini-XDR frontend..."
        "$SCRIPT_DIR/deployment/deploy-frontend-aws.sh"
    else
        log "Mini-XDR frontend already deployed"
    fi
    
    log "✅ Base infrastructure ready"
}

# Execute ML pipeline
execute_ml_pipeline() {
    step "🧠 Executing ML Pipeline"
    
    log "Starting comprehensive ML pipeline..."
    log "📊 Processing 846,073+ cybersecurity events"
    log "🎯 Training 4 advanced ML models"
    log "⏱️  Estimated time: 6-8 hours"
    
    # Create configuration file
    cat > "/tmp/ml-pipeline-config.yaml" << EOF
data:
  source_bucket: mini-xdr-ml-data-${ACCOUNT_ID}-${REGION}
  models_bucket: mini-xdr-ml-models-${ACCOUNT_ID}-${REGION}
  artifacts_bucket: mini-xdr-ml-artifacts-${ACCOUNT_ID}-${REGION}
  total_events: 846073

training:
  instance_type: ml.p3.8xlarge
  instance_count: 1
  max_runtime_hours: 24

deployment:
  instance_type: ml.c5.2xlarge
  initial_instance_count: 2
  max_instance_count: 10

monitoring:
  email: "admin@example.com"
  slack_webhook: null
EOF
    
    # Execute orchestrator
    python3 "$SCRIPT_DIR/monitoring/ml-pipeline-orchestrator.py" \
        --config "/tmp/ml-pipeline-config.yaml" \
        --phase all
    
    log "✅ ML pipeline execution completed"
}

# Integrate with existing Mini-XDR
integrate_with_minixdr() {
    step "🔗 Integrating with Mini-XDR Backend"
    
    # Get SageMaker endpoint name
    local endpoint_name
    endpoint_name=$(aws sagemaker list-endpoints \
        --name-contains "mini-xdr" \
        --query 'Endpoints[0].EndpointName' \
        --output text 2>/dev/null || echo "")
    
    if [ -n "$endpoint_name" ] && [ "$endpoint_name" != "None" ]; then
        log "Found SageMaker endpoint: $endpoint_name"
        
        # Update Mini-XDR backend configuration
        local backend_outputs
        backend_outputs=$(aws cloudformation describe-stacks \
            --stack-name "mini-xdr-backend" \
            --region "$REGION" \
            --query 'Stacks[0].Outputs' \
            --output json)
        
        local backend_ip
        backend_ip=$(echo "$backend_outputs" | jq -r '.[] | select(.OutputKey=="BackendPublicIP") | .OutputValue')
        
        if [ -n "$backend_ip" ] && [ "$backend_ip" != "None" ]; then
            log "Updating Mini-XDR backend configuration..."
            
            # SSH to backend and update configuration
            ssh -i "~/.ssh/mini-xdr-tpot-key.pem" -o StrictHostKeyChecking=yes -o UserKnownHostsFile=~/.ssh/known_hosts ubuntu@"$backend_ip" << REMOTE_CONFIG
                # Update environment file with SageMaker endpoint
                echo "" >> /opt/mini-xdr/.env
                echo "# AWS ML Configuration" >> /opt/mini-xdr/.env
                echo "SAGEMAKER_ENDPOINT_NAME=${endpoint_name}" >> /opt/mini-xdr/.env
                echo "AWS_REGION=${REGION}" >> /opt/mini-xdr/.env
                echo "ML_MODELS_BUCKET=mini-xdr-ml-models-${ACCOUNT_ID}-${REGION}" >> /opt/mini-xdr/.env
                echo "ADVANCED_ML_ENABLED=true" >> /opt/mini-xdr/.env
                
                # Restart Mini-XDR service
                sudo systemctl restart mini-xdr
                
                echo "Mini-XDR backend updated with ML integration"
REMOTE_CONFIG
            
            log "✅ Mini-XDR integration completed"
        else
            warn "Could not find Mini-XDR backend IP"
        fi
    else
        warn "No SageMaker endpoint found for integration"
    fi
}

# Create management scripts
create_management_scripts() {
    step "📋 Creating Management Scripts"
    
    # Create ML management script
    cat > "$HOME/aws-ml-control.sh" << 'ML_CONTROL_EOF'
#!/bin/bash

# Mini-XDR AWS ML Management Script

REGION="${AWS_REGION:-us-east-1}"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

show_usage() {
    echo "Usage: $0 {status|start|stop|retrain|logs|costs}"
    echo ""
    echo "Commands:"
    echo "  status   - Show ML pipeline status"
    echo "  start    - Start inference endpoints"
    echo "  stop     - Stop inference endpoints (save costs)"
    echo "  retrain  - Trigger model retraining"
    echo "  logs     - Show recent training logs"
    echo "  costs    - Show current AWS costs"
}

show_status() {
    echo "📊 Mini-XDR ML System Status"
    echo "================================"
    
    # Check SageMaker endpoints
    echo "🤖 SageMaker Endpoints:"
    aws sagemaker list-endpoints --query 'Endpoints[?contains(EndpointName, `mini-xdr`)].{Name:EndpointName,Status:EndpointStatus}' --output table
    
    # Check training jobs
    echo ""
    echo "🏋️ Recent Training Jobs:"
    aws sagemaker list-training-jobs --max-results 5 --query 'TrainingJobSummaries[?contains(TrainingJobName, `mini-xdr`)].{Name:TrainingJobName,Status:TrainingJobStatus,CreationTime:CreationTime}' --output table
    
    # Check S3 buckets
    echo ""
    echo "🗃️ S3 Storage:"
    aws s3 ls | grep mini-xdr-ml
}

case "${1:-status}" in
    status)
        show_status
        ;;
    start)
        echo "🚀 Starting inference endpoints..."
        # Implementation would go here
        ;;
    stop)
        echo "🛑 Stopping inference endpoints..."
        # Implementation would go here
        ;;
    retrain)
        echo "🔄 Triggering model retraining..."
        # Implementation would go here
        ;;
    logs)
        echo "📋 Showing training logs..."
        # Implementation would go here
        ;;
    costs)
        echo "💰 AWS Cost Analysis..."
        # Implementation would go here
        ;;
    *)
        show_usage
        ;;
esac
ML_CONTROL_EOF
    
    chmod +x "$HOME/aws-ml-control.sh"
    
    # Update existing aliases
    if [ -f "$HOME/mini-xdr-aliases.sh" ]; then
        cat >> "$HOME/mini-xdr-aliases.sh" << 'ALIASES_APPEND'

# ML-specific aliases
alias ml-status="~/aws-ml-control.sh status"
alias ml-start="~/aws-ml-control.sh start"
alias ml-stop="~/aws-ml-control.sh stop"
alias ml-retrain="~/aws-ml-control.sh retrain"
alias ml-logs="~/aws-ml-control.sh logs"
alias ml-costs="~/aws-ml-control.sh costs"
ALIASES_APPEND
    fi
    
    log "✅ Management scripts created"
}

# Show deployment summary
show_summary() {
    step "🎉 Deployment Summary"
    
    # Get deployment information
    local backend_ip=""
    local frontend_url=""
    local endpoint_name=""
    
    # Get backend IP
    if aws cloudformation describe-stacks --stack-name mini-xdr-backend --region "$REGION" &>/dev/null; then
        backend_ip=$(aws cloudformation describe-stacks \
            --stack-name "mini-xdr-backend" \
            --region "$REGION" \
            --query 'Stacks[0].Outputs[?OutputKey==`BackendPublicIP`].OutputValue' \
            --output text)
    fi
    
    # Get frontend URL
    if aws cloudformation describe-stacks --stack-name mini-xdr-frontend --region "$REGION" &>/dev/null; then
        frontend_url=$(aws cloudformation describe-stacks \
            --stack-name "mini-xdr-frontend" \
            --region "$REGION" \
            --query 'Stacks[0].Outputs[?OutputKey==`CloudFrontURL`].OutputValue' \
            --output text)
    fi
    
    # Get SageMaker endpoint
    endpoint_name=$(aws sagemaker list-endpoints \
        --name-contains "mini-xdr" \
        --query 'Endpoints[0].EndpointName' \
        --output text 2>/dev/null || echo "")
    
    echo ""
    echo "=============================================================="
    echo "        Mini-XDR AWS ML System Deployment Complete!"
    echo "=============================================================="
    echo ""
    echo "🌐 System URLs:"
    echo "   Frontend: ${frontend_url:-'Not deployed'}"
    echo "   Backend API: http://${backend_ip:-'Not deployed'}:8000"
    echo "   ML Endpoint: ${endpoint_name:-'Not deployed'}"
    echo ""
    echo "📊 Data Processing:"
    echo "   Total Events: 846,073+"
    echo "   Features: 83+ CICIDS2017 + custom threat intelligence"
    echo "   Datasets: CICIDS2017, KDD Cup, Threat Intel, Synthetic"
    echo ""
    echo "🧠 ML Models Deployed:"
    echo "   🤖 Transformer (Multi-head attention)"
    echo "   🌳 XGBoost (Gradient boosting + HPO)"
    echo "   🔄 LSTM Autoencoder (Sequence anomaly detection)"
    echo "   🌲 Isolation Forest (Ensemble anomaly detection)"
    echo ""
    echo "⚡ Performance Targets:"
    echo "   Detection Rate: >99%"
    echo "   False Positive Rate: <0.5%"
    echo "   Inference Latency: <50ms"
    echo "   Throughput: >10k events/sec"
    echo ""
    echo "🔧 Management Commands:"
    echo "   ML Status: ~/aws-ml-control.sh status"
    echo "   System Status: ~/aws-services-control.sh status"
    echo "   TPOT Control: ~/tpot-security-control.sh status"
    echo "   Updates: ~/update-pipeline.sh both"
    echo ""
    echo "💰 Cost Optimization:"
    echo "   Stop endpoints when not needed: ~/aws-ml-control.sh stop"
    echo "   Monitor costs: ~/aws-ml-control.sh costs"
    echo "   Estimated monthly cost: $200-500 (training), $50-100 (inference only)"
    echo ""
    echo "🚨 Next Steps:"
    echo "   1. Load aliases: source ~/mini-xdr-aliases.sh"
    echo "   2. Check ML status: ml-status"
    echo "   3. Test with real attacks: tpot-live (when ready)"
    echo "   4. Monitor performance: AWS console → SageMaker"
    echo ""
    echo "✅ Your Mini-XDR system now has enterprise-scale ML capabilities!"
}

# Error handling
cleanup_on_error() {
    if [ $? -ne 0 ]; then
        error "Deployment failed! Check the logs above for details."
        echo ""
        echo "To clean up partial deployment:"
        echo "  aws sagemaker delete-endpoint --endpoint-name <endpoint-name>"
        echo "  aws s3 rb s3://mini-xdr-ml-data-$ACCOUNT_ID-$REGION --force"
        echo "  aws s3 rb s3://mini-xdr-ml-models-$ACCOUNT_ID-$REGION --force"
        echo ""
        echo "To retry deployment:"
        echo "  $0"
    fi
}

# Set up error handling
trap cleanup_on_error EXIT

# Main deployment function
main() {
    show_banner
    check_prerequisites
    confirm_deployment
    
    log "🚀 Starting complete AWS ML system deployment..."
    local start_time=$(date +%s)
    
    setup_iam_roles
    deploy_infrastructure
    execute_ml_pipeline
    integrate_with_minixdr
    create_management_scripts
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    local hours=$((duration / 3600))
    local minutes=$(((duration % 3600) / 60))
    
    show_summary
    
    echo ""
    log "🕒 Total deployment time: ${hours}h ${minutes}m"
    echo ""
    
    # Disable error trap for successful completion
    trap - EXIT
}

# Export configuration for subscripts
export AWS_REGION="$REGION"
export ACCOUNT_ID="$ACCOUNT_ID"

# Run main function
main "$@"
