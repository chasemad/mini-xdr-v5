#!/bin/bash
# ============================================================================
# Mini-XDR EKS Cluster Deployment Script
# ============================================================================
# Creates EKS cluster using eksctl with production configuration
# - CPU-only t3.medium nodes (within 8 vCPU quota)
# - 2-4 nodes with autoscaling
# - Required add-ons: EBS CSI, EFS CSI, AWS Load Balancer Controller
# ============================================================================

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
AWS_REGION="${AWS_REGION:-us-east-1}"
CLUSTER_NAME="mini-xdr-cluster"
CLUSTER_CONFIG="$SCRIPT_DIR/eks-cluster-config-production.yaml"

# Load infrastructure resources
KMS_KEY_ARN=$(cat /tmp/mini-xdr-kms-arn.txt 2>/dev/null || echo "")
EFS_ID=$(cat /tmp/mini-xdr-efs-id.txt 2>/dev/null || echo "")

# Logging
LOG_FILE="/tmp/mini-xdr-eks-deploy-$(date +%Y%m%d-%H%M%S).log"
exec 1> >(tee -a "$LOG_FILE")
exec 2> >(tee -a "$LOG_FILE" >&2)

# ============================================================================
# Helper Functions
# ============================================================================

log() {
    echo -e "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
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
    log "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    log "${BLUE}🚀 $1${NC}"
    log "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo
}

# ============================================================================
# Prerequisites
# ============================================================================

check_prerequisites() {
    header "CHECKING PREREQUISITES"

    # Check eksctl
    if ! command -v eksctl &> /dev/null; then
        error "eksctl not found. Install: https://eksctl.io/installation/"
    fi
    success "eksctl installed: $(eksctl version)"

    # Check kubectl
    if ! command -v kubectl &> /dev/null; then
        error "kubectl not found. Install: https://kubernetes.io/docs/tasks/tools/"
    fi
    success "kubectl installed: $(kubectl version --client --short 2>/dev/null || kubectl version --client)"

    # Check AWS CLI
    if ! aws sts get-caller-identity &> /dev/null; then
        error "AWS credentials not configured"
    fi
    success "AWS credentials configured"

    # Check if infrastructure was set up
    if [ -z "$KMS_KEY_ARN" ] || [ -z "$EFS_ID" ]; then
        warning "Infrastructure resources not found in /tmp/"
        warning "Run: ./infrastructure/aws/setup-infrastructure.sh first"
        info "Continuing anyway (will prompt for values)..."
    else
        success "Infrastructure resources loaded:"
        info "  KMS Key: $KMS_KEY_ARN"
        info "  EFS ID: $EFS_ID"
    fi

    # Check cluster config
    if [ ! -f "$CLUSTER_CONFIG" ]; then
        error "Cluster config not found: $CLUSTER_CONFIG"
    fi
    success "Cluster config found: $CLUSTER_CONFIG"
}

# ============================================================================
# Update Cluster Config with Infrastructure IDs
# ============================================================================

update_cluster_config() {
    header "UPDATING CLUSTER CONFIGURATION"

    # Create temporary config with substituted values
    TMP_CONFIG="/tmp/eks-cluster-config-$(date +%Y%m%d-%H%M%S).yaml"
    cp "$CLUSTER_CONFIG" "$TMP_CONFIG"

    # Substitute KMS key ARN if available
    if [ -n "$KMS_KEY_ARN" ]; then
        info "Substituting KMS Key ARN..."
        sed -i.bak "s|\${KMS_KEY_ARN}|$KMS_KEY_ARN|g" "$TMP_CONFIG"
        success "KMS Key ARN configured"
    else
        warning "No KMS Key ARN found - secrets encryption will not be enabled"
        # Comment out secretsEncryption section
        sed -i.bak '/secretsEncryption:/,+1 d' "$TMP_CONFIG"
    fi

    success "Cluster config updated: $TMP_CONFIG"
    echo "$TMP_CONFIG" > /tmp/mini-xdr-cluster-config.txt
}

# ============================================================================
# Create EKS Cluster
# ============================================================================

create_eks_cluster() {
    header "CREATING EKS CLUSTER"

    TMP_CONFIG=$(cat /tmp/mini-xdr-cluster-config.txt)

    # Check if cluster already exists
    if eksctl get cluster --name "$CLUSTER_NAME" --region "$AWS_REGION" &> /dev/null; then
        success "Cluster '$CLUSTER_NAME' already exists"
        return 0
    fi

    info "Creating EKS cluster '$CLUSTER_NAME'..."
    info "This will take 15-20 minutes..."
    echo ""

    eksctl create cluster --config-file="$TMP_CONFIG" --verbose 4

    success "EKS cluster created successfully!"
}

# ============================================================================
# Configure kubectl
# ============================================================================

configure_kubectl() {
    header "CONFIGURING KUBECTL"

    info "Updating kubeconfig..."

    eksctl utils write-kubeconfig \
        --cluster="$CLUSTER_NAME" \
        --region="$AWS_REGION"

    success "kubeconfig updated"

    # Test cluster access
    info "Testing cluster access..."
    kubectl get nodes

    success "kubectl configured and cluster accessible"
}

# ============================================================================
# Install AWS Load Balancer Controller
# ============================================================================

install_alb_controller() {
    header "INSTALLING AWS LOAD BALANCER CONTROLLER"

    # Check if Helm is installed
    if ! command -v helm &> /dev/null; then
        warning "Helm not found. Installing Helm..."
        curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash
        success "Helm installed"
    fi

    # Add EKS Helm repo
    info "Adding EKS Helm repository..."
    helm repo add eks https://aws.github.io/eks-charts
    helm repo update

    # Install AWS Load Balancer Controller
    info "Installing AWS Load Balancer Controller..."

    kubectl create namespace kube-system --dry-run=client -o yaml | kubectl apply -f -

    helm install aws-load-balancer-controller eks/aws-load-balancer-controller \
        -n kube-system \
        --set clusterName="$CLUSTER_NAME" \
        --set serviceAccount.create=false \
        --set serviceAccount.name=aws-load-balancer-controller \
        --set region="$AWS_REGION" \
        --set vpcId=$(aws eks describe-cluster --name "$CLUSTER_NAME" --region "$AWS_REGION" --query "cluster.resourcesVpcConfig.vpcId" --output text) \
        || warning "ALB Controller may already be installed"

    success "AWS Load Balancer Controller installed"
}

# ============================================================================
# Configure EFS CSI Driver
# ============================================================================

configure_efs_csi() {
    header "CONFIGURING EFS CSI DRIVER"

    if [ -z "$EFS_ID" ]; then
        warning "EFS ID not found. Skipping EFS CSI configuration."
        warning "You will need to configure this manually later."
        return 0
    fi

    info "EFS File System ID: $EFS_ID"

    # Create EFS mount targets in cluster VPC (if not already created)
    info "Creating EFS mount targets in cluster subnets..."

    VPC_ID=$(aws eks describe-cluster --name "$CLUSTER_NAME" --region "$AWS_REGION" --query "cluster.resourcesVpcConfig.vpcId" --output text)
    SUBNET_IDS=$(aws eks describe-cluster --name "$CLUSTER_NAME" --region "$AWS_REGION" --query "cluster.resourcesVpcConfig.subnetIds[]" --output text)

    for subnet in $SUBNET_IDS; do
        # Check if mount target already exists
        EXISTING_MT=$(aws efs describe-mount-targets \
            --file-system-id "$EFS_ID" \
            --region "$AWS_REGION" \
            --query "MountTargets[?SubnetId=='$subnet'].MountTargetId" \
            --output text 2>/dev/null || echo "")

        if [ -z "$EXISTING_MT" ]; then
            info "Creating EFS mount target in subnet $subnet..."

            # Get security group for cluster
            SG_ID=$(aws eks describe-cluster --name "$CLUSTER_NAME" --region "$AWS_REGION" --query "cluster.resourcesVpcConfig.clusterSecurityGroupId" --output text)

            aws efs create-mount-target \
                --file-system-id "$EFS_ID" \
                --subnet-id "$subnet" \
                --security-groups "$SG_ID" \
                --region "$AWS_REGION" \
                > /dev/null 2>&1 || warning "Could not create mount target in $subnet (may already exist)"
        fi
    done

    success "EFS mount targets configured"

    # Update persistent volumes config with EFS ID
    info "Updating persistent volumes config with EFS ID..."
    PV_CONFIG="$PROJECT_ROOT/ops/k8s/persistent-volumes-production.yaml"
    if [ -f "$PV_CONFIG" ]; then
        sed -i.bak "s/\${EFS_FILE_SYSTEM_ID}/$EFS_ID/g" "$PV_CONFIG"
        success "PV config updated with EFS ID"
    fi
}

# ============================================================================
# Summary
# ============================================================================

print_summary() {
    header "EKS CLUSTER DEPLOYMENT COMPLETE ✅"

    success "EKS cluster is ready!"
    echo ""
    echo "  Cluster Name:  $CLUSTER_NAME"
    echo "  Region:        $AWS_REGION"
    echo "  Nodes:         $(kubectl get nodes --no-headers | wc -l) nodes"
    echo ""

    info "Cluster nodes:"
    kubectl get nodes -o wide

    echo ""
    info "📋 Next Steps:"
    echo ""
    echo "  1. Build and push Docker images:"
    echo "     ./infrastructure/aws/build-and-push-images.sh"
    echo ""
    echo "  2. Deploy application:"
    echo "     ./infrastructure/aws/deploy-to-eks.sh"
    echo ""
    echo "  3. Access cluster:"
    echo "     kubectl get pods -n mini-xdr"
    echo ""

    info "📝 Log file saved to: $LOG_FILE"
}

# ============================================================================
# Main Execution
# ============================================================================

main() {
    header "MINI-XDR EKS CLUSTER DEPLOYMENT"

    info "This script will create an EKS cluster for Mini-XDR"
    info "Estimated time: 15-20 minutes"
    echo ""

    check_prerequisites
    update_cluster_config
    create_eks_cluster
    configure_kubectl
    install_alb_controller
    configure_efs_csi
    print_summary

    success "🎉 EKS cluster deployment completed successfully!"
}

# Run main function
main "$@"
