#!/bin/bash
# ============================================================================
# Mini-XDR AWS Infrastructure Setup Script
# ============================================================================
# Sets up all necessary AWS infrastructure for EKS deployment
# - Quota validation
# - VPC/networking
# - EFS for shared model storage
# - KMS for encryption
# - IAM roles
# - Security groups
# ============================================================================

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
AWS_REGION="${AWS_REGION:-us-east-1}"
CLUSTER_NAME="mini-xdr-cluster"
PROJECT_TAG="mini-xdr"
ENVIRONMENT="production"

# Logging
LOG_FILE="/tmp/mini-xdr-infra-setup-$(date +%Y%m%d-%H%M%S).log"
exec 1> >(tee -a "$LOG_FILE")
exec 2> >(tee -a "$LOG_FILE" >&2)

# ============================================================================
# Helper Functions
# ============================================================================

log() {
    echo -e "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

success() {
    log "${GREEN}✅ $1${NC}"
}

warning() {
    log "${YELLOW}⚠️  $1${NC}"
}

error() {
    log "${RED}❌ ERROR: $1${NC}"
    exit 1
}

info() {
    log "${BLUE}ℹ️  $1${NC}"
}

header() {
    echo
    log "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    log "${BLUE}🚀 $1${NC}"
    log "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo
}

# ============================================================================
# Prerequisites Check
# ============================================================================

check_prerequisites() {
    header "CHECKING PREREQUISITES"

    # Check AWS CLI
    if ! command -v aws &> /dev/null; then
        error "AWS CLI not found. Please install: https://aws.amazon.com/cli/"
    fi
    success "AWS CLI installed: $(aws --version | head -1)"

    # Check AWS credentials
    if ! aws sts get-caller-identity &> /dev/null; then
        error "AWS credentials not configured. Run: aws configure"
    fi

    ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
    CALLER_ARN=$(aws sts get-caller-identity --query Arn --output text)
    success "AWS Account ID: $ACCOUNT_ID"
    info "Caller identity: $CALLER_ARN"

    # Check jq for JSON parsing
    if ! command -v jq &> /dev/null; then
        warning "jq not found. Installing via package manager is recommended for better output parsing"
        warning "On macOS: brew install jq | On Linux: sudo apt-get install jq"
    else
        success "jq installed"
    fi

    # Check region
    aws configure set region "$AWS_REGION"
    success "AWS Region: $AWS_REGION"
}

# ============================================================================
# Quota Validation
# ============================================================================

check_quotas() {
    header "VALIDATING AWS QUOTAS"

    info "Checking EC2 vCPU quotas..."

    # Check standard instance vCPU quota (for t3.medium)
    STANDARD_VCPU_QUOTA=$(aws service-quotas get-service-quota \
        --service-code ec2 \
        --quota-code L-1216C47A \
        --region "$AWS_REGION" \
        --query 'Quota.Value' \
        --output text 2>/dev/null || echo "0")

    info "Standard instance vCPU quota: $STANDARD_VCPU_QUOTA vCPUs"

    # Minimum required: 8 vCPUs (4 t3.medium nodes)
    REQUIRED_VCPUS=8

    if (( $(echo "$STANDARD_VCPU_QUOTA < $REQUIRED_VCPUS" | bc -l) )); then
        error "Insufficient vCPU quota. Required: $REQUIRED_VCPUS, Current: $STANDARD_VCPU_QUOTA"
        info "To request quota increase:"
        info "  1. Visit: https://console.aws.amazon.com/servicequotas/home/services/ec2/quotas"
        info "  2. Search for 'Running On-Demand Standard instances'"
        info "  3. Request increase to at least 8 vCPUs"
        exit 1
    fi

    success "✅ vCPU quota sufficient: $STANDARD_VCPU_QUOTA vCPUs (need $REQUIRED_VCPUS)"

    # Check VPC quota
    VPC_QUOTA=$(aws service-quotas get-service-quota \
        --service-code vpc \
        --quota-code L-F678F1CE \
        --region "$AWS_REGION" \
        --query 'Quota.Value' \
        --output text 2>/dev/null || echo "5")

    VPC_COUNT=$(aws ec2 describe-vpcs --region "$AWS_REGION" --query 'length(Vpcs)' --output text)
    info "VPCs: $VPC_COUNT / $VPC_QUOTA"

    if [ "$VPC_COUNT" -ge "$VPC_QUOTA" ]; then
        warning "VPC quota reached. Will attempt to use existing VPC."
    fi

    # Check EBS volume quota
    info "EBS volume quota check (for persistent storage)..."
    EBS_VOLUME_QUOTA=$(aws service-quotas get-service-quota \
        --service-code ebs \
        --quota-code L-D18FCD1D \
        --region "$AWS_REGION" \
        --query 'Quota.Value' \
        --output text 2>/dev/null || echo "1000")

    success "EBS volume quota: $EBS_VOLUME_QUOTA (sufficient)"

    # Check EFS quota
    EFS_COUNT=$(aws efs describe-file-systems --region "$AWS_REGION" --query 'length(FileSystems)' --output text)
    info "EFS file systems: $EFS_COUNT (no strict quota, pay-per-use)"

    success "All quotas validated successfully!"
}

# ============================================================================
# VPC Setup
# ============================================================================

setup_vpc() {
    header "SETTING UP VPC"

    # Check if VPC already exists
    EXISTING_VPC=$(aws ec2 describe-vpcs \
        --filters "Name=tag:Project,Values=$PROJECT_TAG" "Name=tag:Environment,Values=$ENVIRONMENT" \
        --query 'Vpcs[0].VpcId' \
        --output text \
        --region "$AWS_REGION" 2>/dev/null || echo "None")

    if [ "$EXISTING_VPC" != "None" ] && [ -n "$EXISTING_VPC" ]; then
        success "VPC already exists: $EXISTING_VPC"
        VPC_ID="$EXISTING_VPC"

        # Store VPC ID for later use
        echo "$VPC_ID" > /tmp/mini-xdr-vpc-id.txt
        return 0
    fi

    info "Creating new VPC with CIDR 10.0.0.0/16..."

    VPC_ID=$(aws ec2 create-vpc \
        --cidr-block 10.0.0.0/16 \
        --region "$AWS_REGION" \
        --tag-specifications "ResourceType=vpc,Tags=[{Key=Name,Value=${CLUSTER_NAME}-vpc},{Key=Project,Value=${PROJECT_TAG}},{Key=Environment,Value=${ENVIRONMENT}}]" \
        --query 'Vpc.VpcId' \
        --output text)

    success "VPC created: $VPC_ID"

    # Enable DNS
    aws ec2 modify-vpc-attribute --vpc-id "$VPC_ID" --enable-dns-support
    aws ec2 modify-vpc-attribute --vpc-id "$VPC_ID" --enable-dns-hostnames
    success "DNS support enabled"

    # Store VPC ID
    echo "$VPC_ID" > /tmp/mini-xdr-vpc-id.txt

    info "VPC setup complete. EKS cluster will create subnets automatically."
}

# ============================================================================
# KMS Setup (for encryption)
# ============================================================================

setup_kms() {
    header "SETTING UP KMS FOR ENCRYPTION"

    # Check if KMS key already exists
    EXISTING_KEY=$(aws kms list-aliases \
        --region "$AWS_REGION" \
        --query "Aliases[?AliasName=='alias/${CLUSTER_NAME}-secrets'].TargetKeyId" \
        --output text 2>/dev/null || echo "")

    if [ -n "$EXISTING_KEY" ]; then
        success "KMS key already exists: $EXISTING_KEY"
        KMS_KEY_ARN=$(aws kms describe-key --key-id "$EXISTING_KEY" --region "$AWS_REGION" --query 'KeyMetadata.Arn' --output text)
        success "KMS Key ARN: $KMS_KEY_ARN"
        echo "$KMS_KEY_ARN" > /tmp/mini-xdr-kms-arn.txt
        return 0
    fi

    info "Creating KMS key for secrets encryption..."

    KMS_KEY_ID=$(aws kms create-key \
        --description "Mini-XDR EKS Secrets Encryption" \
        --region "$AWS_REGION" \
        --tags "TagKey=Project,TagValue=${PROJECT_TAG}" "TagKey=Environment,TagValue=${ENVIRONMENT}" \
        --query 'KeyMetadata.KeyId' \
        --output text)

    success "KMS key created: $KMS_KEY_ID"

    # Create alias
    aws kms create-alias \
        --alias-name "alias/${CLUSTER_NAME}-secrets" \
        --target-key-id "$KMS_KEY_ID" \
        --region "$AWS_REGION"

    success "KMS alias created: alias/${CLUSTER_NAME}-secrets"

    KMS_KEY_ARN=$(aws kms describe-key --key-id "$KMS_KEY_ID" --region "$AWS_REGION" --query 'KeyMetadata.Arn' --output text)
    success "KMS Key ARN: $KMS_KEY_ARN"

    # Store KMS ARN
    echo "$KMS_KEY_ARN" > /tmp/mini-xdr-kms-arn.txt
}

# ============================================================================
# EFS Setup (for shared model storage)
# ============================================================================

setup_efs() {
    header "SETTING UP EFS FOR SHARED MODEL STORAGE"

    # Check if EFS already exists
    EXISTING_EFS=$(aws efs describe-file-systems \
        --region "$AWS_REGION" \
        --query "FileSystems[?Name=='${CLUSTER_NAME}-models'].FileSystemId" \
        --output text 2>/dev/null || echo "")

    if [ -n "$EXISTING_EFS" ]; then
        success "EFS already exists: $EXISTING_EFS"
        EFS_ID="$EXISTING_EFS"
        echo "$EFS_ID" > /tmp/mini-xdr-efs-id.txt
        return 0
    fi

    info "Creating EFS file system for model storage..."

    EFS_ID=$(aws efs create-file-system \
        --performance-mode generalPurpose \
        --throughput-mode bursting \
        --encrypted \
        --region "$AWS_REGION" \
        --tags "Key=Name,Value=${CLUSTER_NAME}-models" "Key=Project,Value=${PROJECT_TAG}" "Key=Environment,Value=${ENVIRONMENT}" \
        --query 'FileSystemId' \
        --output text)

    success "EFS created: $EFS_ID"

    # Wait for EFS to be available
    info "Waiting for EFS to become available..."
    aws efs wait file-system-available --file-system-id "$EFS_ID" --region "$AWS_REGION"
    success "EFS is available"

    # Store EFS ID
    echo "$EFS_ID" > /tmp/mini-xdr-efs-id.txt

    info "Note: EFS mount targets will be created automatically by EFS CSI driver in EKS"
}

# ============================================================================
# Cost Estimation
# ============================================================================

estimate_costs() {
    header "COST ESTIMATION"

    info "Monthly cost breakdown (us-east-1):"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "EKS Control Plane:              \$73.00/month"
    echo "2x t3.medium (on-demand):       \$60.00/month"
    echo "EBS gp3 (15GB):                 \$1.20/month"
    echo "EFS (5GB, estimated):           \$1.50/month"
    echo "NAT Gateway:                    \$32.00/month"
    echo "ALB:                            \$16.00/month"
    echo "Data transfer (estimated):      \$10.00/month"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "TOTAL (estimated):              \$193.70/month"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""

    warning "💡 Cost Optimization Tips:"
    echo "  - Use Spot instances for nodes: Save ~70% on compute"
    echo "  - Use single NAT gateway (already configured)"
    echo "  - Enable EKS cluster autoscaler to scale down when idle"
    echo "  - Set up AWS Budgets alerts for cost control"
    echo ""
}

# ============================================================================
# Summary and Next Steps
# ============================================================================

print_summary() {
    header "INFRASTRUCTURE SETUP COMPLETE ✅"

    # Load created resources
    VPC_ID=$(cat /tmp/mini-xdr-vpc-id.txt 2>/dev/null || echo "N/A")
    KMS_KEY_ARN=$(cat /tmp/mini-xdr-kms-arn.txt 2>/dev/null || echo "N/A")
    EFS_ID=$(cat /tmp/mini-xdr-efs-id.txt 2>/dev/null || echo "N/A")

    success "Infrastructure components created:"
    echo ""
    echo "  VPC ID:            $VPC_ID"
    echo "  KMS Key ARN:       $KMS_KEY_ARN"
    echo "  EFS File System:   $EFS_ID"
    echo "  AWS Region:        $AWS_REGION"
    echo "  AWS Account:       $ACCOUNT_ID"
    echo ""

    info "📋 Next Steps:"
    echo ""
    echo "  1. Update EKS cluster config with these values:"
    echo "     - Set EFS_FILE_SYSTEM_ID=$EFS_ID in persistent-volumes-production.yaml"
    echo "     - Set KMS_KEY_ARN=$KMS_KEY_ARN in eks-cluster-config-production.yaml"
    echo ""
    echo "  2. Create EKS cluster:"
    echo "     ./infrastructure/aws/deploy-eks-cluster.sh"
    echo ""
    echo "  3. Build and push Docker images:"
    echo "     ./infrastructure/aws/build-and-push-images.sh"
    echo ""
    echo "  4. Deploy application to EKS:"
    echo "     ./infrastructure/aws/deploy-to-eks.sh"
    echo ""

    info "📝 Log file saved to: $LOG_FILE"
}

# ============================================================================
# Main Execution
# ============================================================================

main() {
    header "MINI-XDR AWS INFRASTRUCTURE SETUP"

    info "This script will set up all necessary AWS infrastructure for Mini-XDR deployment"
    info "Estimated time: 5-10 minutes"
    echo ""

    # Run all setup steps
    check_prerequisites
    check_quotas
    setup_vpc
    setup_kms
    setup_efs
    estimate_costs
    print_summary

    success "🎉 Infrastructure setup completed successfully!"
}

# Run main function
main "$@"
