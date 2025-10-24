#!/bin/bash
# ============================================================================
# Mini-XDR Complete AWS Production Deployment Script
# ============================================================================
# Master script that orchestrates the entire deployment process:
# 1. Setup infrastructure (VPC, EFS, KMS)
# 2. Build and push Docker images
# 3. Deploy EKS cluster
# 4. Deploy application to EKS
#
# Requirements:
# - AWS CLI configured
# - Docker installed
# - kubectl, eksctl, helm installed
# - 8 vCPU quota for standard EC2 instances
#
# Usage:
#   ./deploy-aws-production.sh            # Full deployment
#   ./deploy-aws-production.sh --skip-infra   # Skip infrastructure setup
#   ./deploy-aws-production.sh --skip-build   # Skip Docker build
# ============================================================================

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AWS_REGION="${AWS_REGION:-us-east-1}"

# Flags
SKIP_INFRA=false
SKIP_BUILD=false
SKIP_CLUSTER=false
SKIP_DEPLOY=false

# Logging
MASTER_LOG="/tmp/mini-xdr-deployment-$(date +%Y%m%d-%H%M%S).log"
exec 1> >(tee -a "$MASTER_LOG")
exec 2> >(tee -a "$MASTER_LOG" >&2)

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
    log "${BOLD}${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    log "${BOLD}${CYAN}🚀 $1${NC}"
    log "${BOLD}${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo
}

# ============================================================================
# Parse Arguments
# ============================================================================

parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --skip-infra)
                SKIP_INFRA=true
                shift
                ;;
            --skip-build)
                SKIP_BUILD=true
                shift
                ;;
            --skip-cluster)
                SKIP_CLUSTER=true
                shift
                ;;
            --skip-deploy)
                SKIP_DEPLOY=true
                shift
                ;;
            --help|-h)
                show_help
                exit 0
                ;;
            *)
                error "Unknown option: $1. Use --help for usage."
                ;;
        esac
    done
}

show_help() {
    cat << EOF
${BOLD}${CYAN}Mini-XDR AWS Production Deployment${NC}

${BOLD}USAGE:${NC}
    $0 [OPTIONS]

${BOLD}OPTIONS:${NC}
    --skip-infra      Skip infrastructure setup (VPC, EFS, KMS)
    --skip-build      Skip Docker image build and push
    --skip-cluster    Skip EKS cluster creation
    --skip-deploy     Skip application deployment
    -h, --help        Show this help message

${BOLD}EXAMPLES:${NC}
    # Full deployment (recommended for first time)
    $0

    # Re-deploy application only (after code changes)
    $0 --skip-infra --skip-cluster --skip-build

    # Create cluster only (infrastructure already exists)
    $0 --skip-infra --skip-deploy

${BOLD}REQUIREMENTS:${NC}
    - AWS CLI configured (aws configure)
    - Docker installed and running
    - kubectl, eksctl, helm installed
    - 8 vCPU quota for standard EC2 instances

${BOLD}ESTIMATED TIME:${NC}
    - Full deployment: 30-40 minutes
    - Skip infra: 25-30 minutes
    - Deploy only: 5-10 minutes

${BOLD}COST ESTIMATE:${NC}
    ~\$180-200/month for production deployment

For more information, see: docs/AWS_DEPLOYMENT_GUIDE.md
EOF
}

# ============================================================================
# Pre-flight Checks
# ============================================================================

preflight_checks() {
    header "PRE-FLIGHT CHECKS"

    info "Checking required tools..."

    # Check AWS CLI
    if ! command -v aws &> /dev/null; then
        error "AWS CLI not installed. Install from: https://aws.amazon.com/cli/"
    fi
    success "✓ AWS CLI installed"

    # Check Docker
    if ! command -v docker &> /dev/null; then
        error "Docker not installed. Install from: https://docs.docker.com/get-docker/"
    fi
    if ! docker info &> /dev/null 2>&1; then
        error "Docker daemon not running. Please start Docker."
    fi
    success "✓ Docker installed and running"

    # Check kubectl
    if ! command -v kubectl &> /dev/null; then
        warning "kubectl not installed. Install from: https://kubernetes.io/docs/tasks/tools/"
        warning "kubectl is required for deployment"
    else
        success "✓ kubectl installed"
    fi

    # Check eksctl
    if ! command -v eksctl &> /dev/null; then
        warning "eksctl not installed. Install from: https://eksctl.io/installation/"
        warning "eksctl is required for EKS cluster creation"
    else
        success "✓ eksctl installed"
    fi

    # Check AWS credentials
    if ! aws sts get-caller-identity &> /dev/null; then
        error "AWS credentials not configured. Run: aws configure"
    fi

    ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
    success "✓ AWS credentials configured (Account: $ACCOUNT_ID)"

    # Check AWS region
    success "✓ AWS Region: $AWS_REGION"

    echo ""
    info "All pre-flight checks passed!"
}

# ============================================================================
# Deployment Phases
# ============================================================================

phase_1_infrastructure() {
    if [ "$SKIP_INFRA" = true ]; then
        warning "Skipping infrastructure setup (--skip-infra)"
        return 0
    fi

    header "PHASE 1: INFRASTRUCTURE SETUP"

    info "Setting up AWS infrastructure..."
    info "This will create: VPC, KMS key, EFS file system"
    info "Estimated time: 5-10 minutes"
    echo ""

    "$SCRIPT_DIR/infrastructure/aws/setup-infrastructure.sh"

    success "Phase 1 complete: Infrastructure ready"
}

phase_2_docker_images() {
    if [ "$SKIP_BUILD" = true ]; then
        warning "Skipping Docker image build (--skip-build)"
        return 0
    fi

    header "PHASE 2: BUILD AND PUSH DOCKER IMAGES"

    info "Building Docker images..."
    info "This will build CPU-only images without training data"
    info "Estimated time: 10-15 minutes"
    echo ""

    "$SCRIPT_DIR/infrastructure/aws/build-and-push-images.sh"

    success "Phase 2 complete: Docker images pushed to ECR"
}

phase_3_eks_cluster() {
    if [ "$SKIP_CLUSTER" = true ]; then
        warning "Skipping EKS cluster creation (--skip-cluster)"
        return 0
    fi

    header "PHASE 3: CREATE EKS CLUSTER"

    info "Creating EKS cluster..."
    info "This will create a production-ready Kubernetes cluster"
    info "Estimated time: 15-20 minutes"
    echo ""

    "$SCRIPT_DIR/infrastructure/aws/deploy-eks-cluster.sh"

    success "Phase 3 complete: EKS cluster ready"
}

phase_4_deploy_application() {
    if [ "$SKIP_DEPLOY" = true ]; then
        warning "Skipping application deployment (--skip-deploy)"
        return 0
    fi

    header "PHASE 4: DEPLOY APPLICATION"

    info "Deploying Mini-XDR to EKS..."
    info "This will deploy backend, frontend, and configure ALB"
    info "Estimated time: 5-10 minutes"
    echo ""

    "$SCRIPT_DIR/infrastructure/aws/deploy-to-eks.sh"

    success "Phase 4 complete: Application deployed"
}

# ============================================================================
# Final Summary
# ============================================================================

print_final_summary() {
    header "🎉 DEPLOYMENT COMPLETE! 🎉"

    # Get deployment info
    ALB_DNS=$(kubectl get ingress mini-xdr-ingress -n mini-xdr -o jsonpath='{.status.loadBalancer.ingress[0].hostname}' 2>/dev/null || echo "Pending...")

    success "Mini-XDR is now deployed to AWS!"
    echo ""
    echo "${BOLD}Deployment Details:${NC}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  AWS Account:     $ACCOUNT_ID"
    echo "  Region:          $AWS_REGION"
    echo "  Cluster:         mini-xdr-cluster"
    echo "  Namespace:       mini-xdr"

    if [ "$ALB_DNS" != "Pending..." ]; then
        echo "  ALB Endpoint:    ${GREEN}$ALB_DNS${NC}"
        echo ""
        echo "${BOLD}🌐 Access URLs:${NC}"
        echo "  Frontend:        ${CYAN}http://$ALB_DNS${NC}"
        echo "  Backend API:     ${CYAN}http://$ALB_DNS/api${NC}"
        echo "  Health Check:    ${CYAN}http://$ALB_DNS/api/health${NC}"
    else
        echo "  ALB Endpoint:    ${YELLOW}Provisioning... (check in 2-3 minutes)${NC}"
    fi

    echo ""
    echo "${BOLD}📊 Resource Usage:${NC}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    NODE_COUNT=$(kubectl get nodes --no-headers 2>/dev/null | wc -l || echo "0")
    BACKEND_PODS=$(kubectl get pods -n mini-xdr -l app=mini-xdr-backend --no-headers 2>/dev/null | wc -l || echo "0")
    FRONTEND_PODS=$(kubectl get pods -n mini-xdr -l app=mini-xdr-frontend --no-headers 2>/dev/null | wc -l || echo "0")

    echo "  Nodes:           $NODE_COUNT"
    echo "  Backend Pods:    $BACKEND_PODS"
    echo "  Frontend Pods:   $FRONTEND_PODS"

    echo ""
    echo "${BOLD}💰 Estimated Monthly Cost:${NC}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  EKS Control Plane:      \$73.00"
    echo "  2x t3.medium nodes:     \$60.00"
    echo "  Storage (EBS + EFS):    \$3.00"
    echo "  ALB:                    \$16.00"
    echo "  NAT Gateway:            \$32.00"
    echo "  Data Transfer:          ~\$10.00"
    echo "  ${BOLD}Total (estimated):      ~\$194.00/month${NC}"

    echo ""
    echo "${BOLD}📋 Next Steps:${NC}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "  1. Test your deployment:"
    echo "     ${CYAN}curl http://$ALB_DNS/api/health${NC}"
    echo ""
    echo "  2. View application logs:"
    echo "     ${CYAN}kubectl logs -f deployment/mini-xdr-backend -n mini-xdr${NC}"
    echo ""
    echo "  3. Monitor your deployment:"
    echo "     ${CYAN}kubectl get pods -n mini-xdr -w${NC}"
    echo ""
    echo "  4. Set up monitoring (optional):"
    echo "     ${CYAN}./infrastructure/aws/setup-monitoring.sh${NC}"
    echo ""
    echo "  5. Configure custom domain (optional):"
    echo "     - Point your domain to: $ALB_DNS"
    echo "     - Update ingress with SSL certificate"
    echo ""

    echo "${BOLD}📝 Documentation:${NC}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  - Full deployment guide:  docs/AWS_DEPLOYMENT_GUIDE.md"
    echo "  - Operations guide:       docs/AWS_OPERATIONS_GUIDE.md"
    echo "  - Troubleshooting:        docs/TROUBLESHOOTING.md"
    echo ""
    echo "  - Deployment log:         $MASTER_LOG"
    echo ""

    success "🚀 Your Mini-XDR deployment is complete and ready to use!"
}

# ============================================================================
# Main Execution
# ============================================================================

main() {
    # ASCII Art Banner
    cat << 'EOF'
╔══════════════════════════════════════════════════════════════════════════╗
║                                                                          ║
║   ███╗   ███╗██╗███╗   ██╗██╗    ██╗  ██╗██████╗ ██████╗              ║
║   ████╗ ████║██║████╗  ██║██║    ╚██╗██╔╝██╔══██╗██╔══██╗             ║
║   ██╔████╔██║██║██╔██╗ ██║██║     ╚███╔╝ ██║  ██║██████╔╝             ║
║   ██║╚██╔╝██║██║██║╚██╗██║██║     ██╔██╗ ██║  ██║██╔══██╗             ║
║   ██║ ╚═╝ ██║██║██║ ╚████║██║    ██╔╝ ██╗██████╔╝██║  ██║             ║
║   ╚═╝     ╚═╝╚═╝╚═╝  ╚═══╝╚═╝    ╚═╝  ╚═╝╚═════╝ ╚═╝  ╚═╝             ║
║                                                                          ║
║                AWS Production Deployment Script                          ║
║                        v1.0.0 - CPU-Only                                 ║
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝
EOF

    echo ""
    info "Starting Mini-XDR AWS Production Deployment"
    info "This will deploy a complete XDR system to AWS EKS"
    echo ""

    # Parse arguments
    parse_args "$@"

    # Run deployment phases
    preflight_checks
    phase_1_infrastructure
    phase_2_docker_images
    phase_3_eks_cluster
    phase_4_deploy_application
    print_final_summary
}

# Run main function
main "$@"
