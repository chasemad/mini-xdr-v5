#!/bin/bash

# S3 Data Lake Setup for Mini-XDR ML Training
# Creates S3 infrastructure for 846,073+ events with proper organization

set -euo pipefail

# Configuration
REGION="${AWS_REGION:-us-east-1}"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
DATA_BUCKET_NAME="${DATA_BUCKET_NAME:-mini-xdr-ml-data-${ACCOUNT_ID}-${REGION}}"
MODELS_BUCKET_NAME="${MODELS_BUCKET_NAME:-mini-xdr-ml-models-${ACCOUNT_ID}-${REGION}}"
ARTIFACTS_BUCKET_NAME="${ARTIFACTS_BUCKET_NAME:-mini-xdr-ml-artifacts-${ACCOUNT_ID}-${REGION}}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

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

# Banner
show_banner() {
    echo -e "${CYAN}"
    echo "=============================================="
    echo "    Mini-XDR S3 Data Lake Setup"
    echo "      Processing 846,073+ Events"
    echo "=============================================="
    echo -e "${NC}"
}

# Create S3 buckets with proper configuration
create_s3_buckets() {
    step "🗃️ Creating S3 Data Lake Infrastructure"
    
    local buckets=(
        "$DATA_BUCKET_NAME"
        "$MODELS_BUCKET_NAME"
        "$ARTIFACTS_BUCKET_NAME"
    )
    
    for bucket in "${buckets[@]}"; do
        log "Creating bucket: $bucket"
        
        # Create bucket with region-specific configuration
        if [ "$REGION" = "us-east-1" ]; then
            aws s3api create-bucket \
                --bucket "$bucket" \
                --region "$REGION" || warn "Bucket $bucket may already exist"
        else
            aws s3api create-bucket \
                --bucket "$bucket" \
                --region "$REGION" \
                --create-bucket-configuration LocationConstraint="$REGION" || warn "Bucket $bucket may already exist"
        fi
        
        # Enable versioning
        aws s3api put-bucket-versioning \
            --bucket "$bucket" \
            --versioning-configuration Status=Enabled
        
        # Enable server-side encryption
        aws s3api put-bucket-encryption \
            --bucket "$bucket" \
            --server-side-encryption-configuration '{
                "Rules": [{
                    "ApplyServerSideEncryptionByDefault": {
                        "SSEAlgorithm": "AES256"
                    }
                }]
            }'
        
        # Set intelligent tiering for cost optimization
        aws s3api put-bucket-intelligent-tiering-configuration \
            --bucket "$bucket" \
            --id "EntireBucket" \
            --intelligent-tiering-configuration '{
                "Id": "EntireBucket",
                "Status": "Enabled",
                "Filter": {"Prefix": ""},
                "Tierings": [{
                    "Days": 1,
                    "AccessTier": "ARCHIVE_ACCESS"
                }, {
                    "Days": 90,
                    "AccessTier": "DEEP_ARCHIVE_ACCESS"
                }]
            }' || warn "Intelligent tiering may not be supported in this region"
        
        log "✅ Bucket $bucket configured successfully"
    done
}

# Create organized folder structure
create_folder_structure() {
    step "📁 Creating Organized Data Structure"
    
    log "Setting up data lake folder structure..."
    
    # Raw datasets (846,073+ events)
    aws s3api put-object --bucket "$DATA_BUCKET_NAME" --key "raw-datasets/"
    aws s3api put-object --bucket "$DATA_BUCKET_NAME" --key "raw-datasets/cicids2017/"
    aws s3api put-object --bucket "$DATA_BUCKET_NAME" --key "raw-datasets/kdd-cup/"
    aws s3api put-object --bucket "$DATA_BUCKET_NAME" --key "raw-datasets/unsw-nb15/"
    aws s3api put-object --bucket "$DATA_BUCKET_NAME" --key "raw-datasets/real-honeypot/"
    aws s3api put-object --bucket "$DATA_BUCKET_NAME" --key "raw-datasets/threat-intelligence/"
    aws s3api put-object --bucket "$DATA_BUCKET_NAME" --key "raw-datasets/synthetic/"
    
    # Processed data with 83+ features
    aws s3api put-object --bucket "$DATA_BUCKET_NAME" --key "processed-data/"
    aws s3api put-object --bucket "$DATA_BUCKET_NAME" --key "processed-data/features-83plus/"
    aws s3api put-object --bucket "$DATA_BUCKET_NAME" --key "processed-data/engineered-features/"
    aws s3api put-object --bucket "$DATA_BUCKET_NAME" --key "processed-data/training-sets/"
    aws s3api put-object --bucket "$DATA_BUCKET_NAME" --key "processed-data/validation-sets/"
    aws s3api put-object --bucket "$DATA_BUCKET_NAME" --key "processed-data/test-sets/"
    
    # Feature engineering outputs
    aws s3api put-object --bucket "$DATA_BUCKET_NAME" --key "feature-store/"
    aws s3api put-object --bucket "$DATA_BUCKET_NAME" --key "feature-store/temporal-features/"
    aws s3api put-object --bucket "$DATA_BUCKET_NAME" --key "feature-store/packet-features/"
    aws s3api put-object --bucket "$DATA_BUCKET_NAME" --key "feature-store/behavioral-features/"
    aws s3api put-object --bucket "$DATA_BUCKET_NAME" --key "feature-store/threat-intel-features/"
    
    # Model artifacts
    aws s3api put-object --bucket "$MODELS_BUCKET_NAME" --key "trained-models/"
    aws s3api put-object --bucket "$MODELS_BUCKET_NAME" --key "trained-models/transformers/"
    aws s3api put-object --bucket "$MODELS_BUCKET_NAME" --key "trained-models/xgboost/"
    aws s3api put-object --bucket "$MODELS_BUCKET_NAME" --key "trained-models/isolation-forest/"
    aws s3api put-object --bucket "$MODELS_BUCKET_NAME" --key "trained-models/lstm-autoencoder/"
    aws s3api put-object --bucket "$MODELS_BUCKET_NAME" --key "trained-models/ensemble/"
    
    # Training artifacts
    aws s3api put-object --bucket "$ARTIFACTS_BUCKET_NAME" --key "experiments/"
    aws s3api put-object --bucket "$ARTIFACTS_BUCKET_NAME" --key "hyperparameter-tuning/"
    aws s3api put-object --bucket "$ARTIFACTS_BUCKET_NAME" --key "model-evaluation/"
    aws s3api put-object --bucket "$ARTIFACTS_BUCKET_NAME" --key "feature-importance/"
    
    log "✅ Folder structure created successfully"
}

# Create bucket policies for secure access
create_bucket_policies() {
    step "🔐 Setting Up Security Policies"
    
    # Data bucket policy - restrict to ML services
    cat > "/tmp/data-bucket-policy.json" << EOF
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "SageMakerAccess",
            "Effect": "Allow",
            "Principal": {
                "Service": "sagemaker.amazonaws.com"
            },
            "Action": [
                "s3:GetObject",
                "s3:PutObject",
                "s3:DeleteObject",
                "s3:ListBucket"
            ],
            "Resource": [
                "arn:aws:s3:::${DATA_BUCKET_NAME}",
                "arn:aws:s3:::${DATA_BUCKET_NAME}/*"
            ]
        },
        {
            "Sid": "GlueAccess",
            "Effect": "Allow",
            "Principal": {
                "Service": "glue.amazonaws.com"
            },
            "Action": [
                "s3:GetObject",
                "s3:PutObject",
                "s3:DeleteObject",
                "s3:ListBucket"
            ],
            "Resource": [
                "arn:aws:s3:::${DATA_BUCKET_NAME}",
                "arn:aws:s3:::${DATA_BUCKET_NAME}/*"
            ]
        }
    ]
}
EOF
    
    aws s3api put-bucket-policy \
        --bucket "$DATA_BUCKET_NAME" \
        --policy file:///tmp/data-bucket-policy.json
    
    log "✅ Security policies configured"
}

# Upload existing datasets
upload_existing_datasets() {
    step "⬆️ Uploading Existing Datasets (846,073+ Events)"
    
    local project_dir="/Users/chasemad/Desktop/mini-xdr"
    
    # Upload CICIDS2017 dataset (799,989 events)
    if [ -f "$project_dir/datasets/real_datasets/cicids2017_enhanced_minixdr.json" ]; then
        log "Uploading CICIDS2017 Enhanced Dataset (799,989 events)..."
        aws s3 cp "$project_dir/datasets/real_datasets/cicids2017_enhanced_minixdr.json" \
            "s3://$DATA_BUCKET_NAME/raw-datasets/cicids2017/"
    fi
    
    # Upload KDD datasets (41,000 events)
    for kdd_file in kdd_full_minixdr.json kdd_10_percent_minixdr.json; do
        if [ -f "$project_dir/datasets/real_datasets/$kdd_file" ]; then
            log "Uploading $kdd_file..."
            aws s3 cp "$project_dir/datasets/real_datasets/$kdd_file" \
                "s3://$DATA_BUCKET_NAME/raw-datasets/kdd-cup/"
        fi
    done
    
    # Upload threat intelligence feeds (2,273 events)
    if [ -d "$project_dir/datasets/threat_feeds" ]; then
        log "Uploading threat intelligence feeds..."
        aws s3 sync "$project_dir/datasets/threat_feeds/" \
            "s3://$DATA_BUCKET_NAME/raw-datasets/threat-intelligence/"
    fi
    
    # Upload synthetic datasets (1,966 events)
    for dataset in combined_cybersecurity_dataset.json ddos_attacks_dataset.json \
                  brute_force_ssh_dataset.json web_attacks_dataset.json \
                  network_scans_dataset.json malware_behavior_dataset.json; do
        if [ -f "$project_dir/datasets/$dataset" ]; then
            log "Uploading $dataset..."
            aws s3 cp "$project_dir/datasets/$dataset" \
                "s3://$DATA_BUCKET_NAME/raw-datasets/synthetic/"
        fi
    done
    
    # Upload existing models
    if [ -d "$project_dir/models" ]; then
        log "Uploading existing trained models..."
        aws s3 sync "$project_dir/models/" \
            "s3://$MODELS_BUCKET_NAME/existing-models/"
    fi
    
    log "✅ Dataset upload completed"
}

# Create CloudFormation template for additional resources
create_cloudformation_template() {
    step "📋 Creating CloudFormation Template for ML Infrastructure"
    
    cat > "/tmp/ml-infrastructure.yaml" << EOF
AWSTemplateFormatVersion: '2010-09-09'
Description: 'Mini-XDR ML Training Infrastructure'

Parameters:
  DataBucketName:
    Type: String
    Default: ${DATA_BUCKET_NAME}
    Description: S3 bucket for ML training data
    
  ModelsBucketName:
    Type: String
    Default: ${MODELS_BUCKET_NAME}
    Description: S3 bucket for trained models
    
  ArtifactsBucketName:
    Type: String
    Default: ${ARTIFACTS_BUCKET_NAME}
    Description: S3 bucket for training artifacts

Resources:
  # IAM Role for SageMaker
  SageMakerExecutionRole:
    Type: AWS::IAM::Role
    Properties:
      AssumeRolePolicyDocument:
        Version: '2012-10-17'
        Statement:
          - Effect: Allow
            Principal:
              Service: sagemaker.amazonaws.com
            Action: sts:AssumeRole
      ManagedPolicyArns:
        - arn:aws:iam::aws:policy/CloudWatchAgentServerPolicy
      Policies:
        - PolicyName: S3Access
          PolicyDocument:
            Version: '2012-10-17'
            Statement:
              - Effect: Allow
                Action:
                  - s3:GetObject
                  - s3:PutObject
                  - s3:DeleteObject
                  - s3:ListBucket
                Resource:
                  - !Sub "arn:aws:s3:::\${DataBucketName}"
                  - !Sub "arn:aws:s3:::\${DataBucketName}/*"
                  - !Sub "arn:aws:s3:::\${ModelsBucketName}"
                  - !Sub "arn:aws:s3:::\${ModelsBucketName}/*"
                  - !Sub "arn:aws:s3:::\${ArtifactsBucketName}"
                  - !Sub "arn:aws:s3:::\${ArtifactsBucketName}/*"

  # IAM Role for Glue
  GlueExecutionRole:
    Type: AWS::IAM::Role
    Properties:
      AssumeRolePolicyDocument:
        Version: '2012-10-17'
        Statement:
          - Effect: Allow
            Principal:
              Service: glue.amazonaws.com
            Action: sts:AssumeRole
      ManagedPolicyArns:
        - arn:aws:iam::aws:policy/service-role/AWSGlueServiceRole
      Policies:
        - PolicyName: S3Access
          PolicyDocument:
            Version: '2012-10-17'
            Statement:
              - Effect: Allow
                Action:
                  - s3:GetObject
                  - s3:PutObject
                  - s3:DeleteObject
                  - s3:ListBucket
                Resource:
                  - !Sub "arn:aws:s3:::\${DataBucketName}"
                  - !Sub "arn:aws:s3:::\${DataBucketName}/*"

  # Glue Database for Data Catalog
  MLDataCatalog:
    Type: AWS::Glue::Database
    Properties:
      CatalogId: !Ref AWS::AccountId
      DatabaseInput:
        Name: mini_xdr_ml_catalog
        Description: Data catalog for Mini-XDR ML training datasets

Outputs:
  SageMakerRoleArn:
    Description: IAM Role ARN for SageMaker
    Value: !GetAtt SageMakerExecutionRole.Arn
    Export:
      Name: !Sub "\${AWS::StackName}-SageMakerRole"
      
  GlueRoleArn:
    Description: IAM Role ARN for Glue
    Value: !GetAtt GlueExecutionRole.Arn
    Export:
      Name: !Sub "\${AWS::StackName}-GlueRole"
      
  DataCatalogName:
    Description: Glue Data Catalog Database Name
    Value: !Ref MLDataCatalog
    Export:
      Name: !Sub "\${AWS::StackName}-DataCatalog"
EOF
    
    log "✅ CloudFormation template created at /tmp/ml-infrastructure.yaml"
}

# Show summary information
show_summary() {
    step "📊 S3 Data Lake Setup Summary"
    
    echo ""
    echo "=============================================="
    echo "     Mini-XDR ML Data Lake Ready!"
    echo "=============================================="
    echo ""
    echo "📦 S3 Buckets Created:"
    echo "   Data Lake: s3://$DATA_BUCKET_NAME"
    echo "   Models: s3://$MODELS_BUCKET_NAME"
    echo "   Artifacts: s3://$ARTIFACTS_BUCKET_NAME"
    echo ""
    echo "📊 Data Capacity:"
    echo "   Total Events: 846,073+"
    echo "   CICIDS2017: 799,989 events (83+ features)"
    echo "   KDD Cup: 41,000 events"
    echo "   Threat Intel: 2,273 events"
    echo "   Synthetic: 1,966 events"
    echo ""
    echo "🔗 Folder Structure:"
    echo "   /raw-datasets/ - Original 846k+ events"
    echo "   /processed-data/ - 83+ engineered features"
    echo "   /feature-store/ - ML-ready feature sets"
    echo "   /trained-models/ - Model artifacts"
    echo ""
    echo "🔐 Security Features:"
    echo "   ✅ Encryption at rest (AES-256)"
    echo "   ✅ Versioning enabled"
    echo "   ✅ Intelligent tiering"
    echo "   ✅ Service-specific access policies"
    echo ""
    echo "🚀 Next Steps:"
    echo "   1. Deploy CloudFormation template: aws cloudformation deploy --template-file /tmp/ml-infrastructure.yaml --stack-name mini-xdr-ml-infra --capabilities CAPABILITY_IAM"
    echo "   2. Run feature engineering pipeline"
    echo "   3. Start SageMaker training jobs"
    echo ""
}

# Main function
main() {
    show_banner
    
    log "Setting up S3 Data Lake for 846,073+ cybersecurity events..."
    
    create_s3_buckets
    create_folder_structure
    create_bucket_policies
    upload_existing_datasets
    create_cloudformation_template
    
    show_summary
    
    log "✅ S3 Data Lake setup completed successfully!"
}

# Export configuration
export DATA_BUCKET_NAME="$DATA_BUCKET_NAME"
export MODELS_BUCKET_NAME="$MODELS_BUCKET_NAME"
export ARTIFACTS_BUCKET_NAME="$ARTIFACTS_BUCKET_NAME"
export AWS_REGION="$REGION"

# Run main function
main "$@"
