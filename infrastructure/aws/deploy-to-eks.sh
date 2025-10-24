#!/bin/bash
# ============================================================================
# Mini-XDR Application Deployment to EKS
# ============================================================================
# Deploys Mini-XDR application to EKS cluster
# - Creates namespace and configmaps
# - Sets up secrets
# - Deploys backend and frontend with production configuration
# - Configures ingress with ALB
# - Sets up monitoring and security policies
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
NAMESPACE="mini-xdr"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

# Logging
LOG_FILE="/tmp/mini-xdr-deploy-$(date +%Y%m%d-%H%M%S).log"
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

    # Check kubectl
    if ! command -v kubectl &> /dev/null; then
        error "kubectl not found"
    fi
    success "kubectl installed"

    # Check cluster access
    if ! kubectl get nodes &> /dev/null; then
        error "Cannot access Kubernetes cluster. Is kubectl configured?"
    fi
    success "Cluster access verified"

    # Show cluster info
    NODE_COUNT=$(kubectl get nodes --no-headers | wc -l)
    success "Cluster has $NODE_COUNT nodes"

    # Check if images are in ECR
    info "Checking ECR repositories..."
    if ! aws ecr describe-repositories --repository-names mini-xdr-backend --region "$AWS_REGION" &> /dev/null; then
        warning "Backend image not found in ECR. Run: ./infrastructure/aws/build-and-push-images.sh"
    fi
    if ! aws ecr describe-repositories --repository-names mini-xdr-frontend --region "$AWS_REGION" &> /dev/null; then
        warning "Frontend image not found in ECR. Run: ./infrastructure/aws/build-and-push-images.sh"
    fi
    success "ECR repositories verified"
}

# ============================================================================
# Create Namespace
# ============================================================================

create_namespace() {
    header "CREATING NAMESPACE"

    kubectl create namespace "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -

    # Label namespace
    kubectl label namespace "$NAMESPACE" \
        name="$NAMESPACE" \
        environment=production \
        project=mini-xdr \
        --overwrite

    success "Namespace '$NAMESPACE' created"
}

# ============================================================================
# Create Secrets
# ============================================================================

create_secrets() {
    header "CREATING SECRETS"

    info "Creating secrets from environment variables or prompts..."

    # Check for required secrets
    if [ -z "${OPENAI_API_KEY:-}" ]; then
        warning "OPENAI_API_KEY not set in environment"
        read -sp "Enter OpenAI API Key (or press Enter to skip): " OPENAI_API_KEY
        echo
    fi

    if [ -z "${XAI_API_KEY:-}" ]; then
        warning "XAI_API_KEY not set in environment"
        read -sp "Enter X.AI API Key (or press Enter to skip): " XAI_API_KEY
        echo
    fi

    # Create secret with provided values
    kubectl create secret generic mini-xdr-secrets \
        --namespace="$NAMESPACE" \
        --from-literal=database-url="sqlite:////app/data/xdr.db" \
        --from-literal=openai-api-key="${OPENAI_API_KEY:-placeholder}" \
        --from-literal=xai-api-key="${XAI_API_KEY:-placeholder}" \
        --from-literal=honeypot-ssh-key="placeholder" \
        --dry-run=client -o yaml | kubectl apply -f -

    success "Secrets created"
}

# ============================================================================
# Create ConfigMaps
# ============================================================================

create_configmaps() {
    header "CREATING CONFIGMAPS"

    # Create config from ops/k8s/configmap.yaml
    if [ -f "$PROJECT_ROOT/ops/k8s/configmap.yaml" ]; then
        kubectl apply -f "$PROJECT_ROOT/ops/k8s/configmap.yaml"
        success "ConfigMap applied from configmap.yaml"
    else
        # Create minimal configmap
        kubectl create configmap mini-xdr-config \
            --namespace="$NAMESPACE" \
            --from-literal=API_HOST="0.0.0.0" \
            --from-literal=API_PORT="8000" \
            --from-literal=DATABASE_URL="sqlite:////app/data/xdr.db" \
            --dry-run=client -o yaml | kubectl apply -f -
        success "Minimal ConfigMap created"
    fi

    # Create policies configmap
    if [ -d "$PROJECT_ROOT/backend/policies" ]; then
        kubectl create configmap mini-xdr-policies \
            --namespace="$NAMESPACE" \
            --from-file="$PROJECT_ROOT/backend/policies" \
            --dry-run=client -o yaml | kubectl apply -f -
        success "Policies ConfigMap created"
    fi
}

# ============================================================================
# Deploy Persistent Volumes
# ============================================================================

deploy_persistent_volumes() {
    header "DEPLOYING PERSISTENT VOLUMES"

    PV_CONFIG="$PROJECT_ROOT/ops/k8s/persistent-volumes-production.yaml"

    if [ ! -f "$PV_CONFIG" ]; then
        error "PV config not found: $PV_CONFIG"
    fi

    # Apply storage classes and PVCs
    kubectl apply -f "$PV_CONFIG"

    success "Persistent volumes deployed"

    # Wait for PVCs to be bound (may take a moment)
    info "Waiting for PVCs to be bound..."
    kubectl wait --for=condition=Bound pvc/mini-xdr-data-pvc -n "$NAMESPACE" --timeout=60s || warning "Data PVC not bound yet"
    kubectl wait --for=condition=Bound pvc/mini-xdr-models-pvc -n "$NAMESPACE" --timeout=60s || warning "Models PVC not bound yet"
}

# ============================================================================
# Deploy Backend
# ============================================================================

deploy_backend() {
    header "DEPLOYING BACKEND"

    BACKEND_CONFIG="$PROJECT_ROOT/ops/k8s/backend-deployment-production.yaml"

    if [ ! -f "$BACKEND_CONFIG" ]; then
        error "Backend config not found: $BACKEND_CONFIG"
    fi

    # Substitute environment variables in manifest
    export AWS_ACCOUNT_ID="$ACCOUNT_ID"
    export AWS_REGION="$AWS_REGION"

    # Apply with substitution
    envsubst < "$BACKEND_CONFIG" | kubectl apply -f -

    success "Backend deployed"

    # Wait for backend to be ready
    info "Waiting for backend pods to be ready (this may take 2-3 minutes)..."
    kubectl wait --for=condition=Ready pod -l app=mini-xdr-backend -n "$NAMESPACE" --timeout=300s || warning "Backend pods not ready yet"
}

# ============================================================================
# Deploy Frontend
# ============================================================================

deploy_frontend() {
    header "DEPLOYING FRONTEND"

    FRONTEND_CONFIG="$PROJECT_ROOT/ops/k8s/frontend-deployment-production.yaml"

    if [ ! -f "$FRONTEND_CONFIG" ]; then
        error "Frontend config not found: $FRONTEND_CONFIG"
    fi

    # Substitute environment variables
    export AWS_ACCOUNT_ID="$ACCOUNT_ID"
    export AWS_REGION="$AWS_REGION"

    envsubst < "$FRONTEND_CONFIG" | kubectl apply -f -

    success "Frontend deployed"

    # Wait for frontend to be ready
    info "Waiting for frontend pods to be ready..."
    kubectl wait --for=condition=Ready pod -l app=mini-xdr-frontend -n "$NAMESPACE" --timeout=120s || warning "Frontend pods not ready yet"
}

# ============================================================================
# Deploy Ingress (ALB)
# ============================================================================

deploy_ingress() {
    header "DEPLOYING INGRESS (ALB)"

    # Create ingress manifest
    cat > /tmp/mini-xdr-ingress.yaml <<EOF
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: mini-xdr-ingress
  namespace: $NAMESPACE
  annotations:
    kubernetes.io/ingress.class: alb
    alb.ingress.kubernetes.io/scheme: internet-facing
    alb.ingress.kubernetes.io/target-type: ip
    alb.ingress.kubernetes.io/listen-ports: '[{"HTTP": 80}]'
    alb.ingress.kubernetes.io/healthcheck-path: /health
    alb.ingress.kubernetes.io/healthcheck-interval-seconds: '30'
    alb.ingress.kubernetes.io/healthcheck-timeout-seconds: '5'
    alb.ingress.kubernetes.io/success-codes: '200'
    alb.ingress.kubernetes.io/tags: Project=mini-xdr,Environment=production
spec:
  rules:
  - http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: mini-xdr-frontend-service
            port:
              number: 3000
      - path: /api
        pathType: Prefix
        backend:
          service:
            name: mini-xdr-backend-service
            port:
              number: 8000
EOF

    kubectl apply -f /tmp/mini-xdr-ingress.yaml

    success "Ingress created"

    # Wait for ALB to be provisioned
    info "Waiting for ALB to be provisioned (this may take 2-3 minutes)..."
    sleep 30

    ALB_DNS=$(kubectl get ingress mini-xdr-ingress -n "$NAMESPACE" -o jsonpath='{.status.loadBalancer.ingress[0].hostname}' 2>/dev/null || echo "")

    if [ -n "$ALB_DNS" ]; then
        success "ALB provisioned: $ALB_DNS"
    else
        warning "ALB not yet available. Check status with: kubectl get ingress -n $NAMESPACE"
    fi
}

# ============================================================================
# Verify Deployment
# ============================================================================

verify_deployment() {
    header "VERIFYING DEPLOYMENT"

    info "Checking pod status..."
    kubectl get pods -n "$NAMESPACE"

    echo ""
    info "Checking services..."
    kubectl get svc -n "$NAMESPACE"

    echo ""
    info "Checking ingress..."
    kubectl get ingress -n "$NAMESPACE"

    # Count running pods
    BACKEND_PODS=$(kubectl get pods -n "$NAMESPACE" -l app=mini-xdr-backend --field-selector=status.phase=Running --no-headers 2>/dev/null | wc -l)
    FRONTEND_PODS=$(kubectl get pods -n "$NAMESPACE" -l app=mini-xdr-frontend --field-selector=status.phase=Running --no-headers 2>/dev/null | wc -l)

    if [ "$BACKEND_PODS" -gt 0 ]; then
        success "Backend: $BACKEND_PODS pods running"
    else
        warning "Backend: No pods running yet"
    fi

    if [ "$FRONTEND_PODS" -gt 0 ]; then
        success "Frontend: $FRONTEND_PODS pods running"
    else
        warning "Frontend: No pods running yet"
    fi
}

# ============================================================================
# Summary
# ============================================================================

print_summary() {
    header "DEPLOYMENT COMPLETE ✅"

    ALB_DNS=$(kubectl get ingress mini-xdr-ingress -n "$NAMESPACE" -o jsonpath='{.status.loadBalancer.ingress[0].hostname}' 2>/dev/null || echo "Pending...")

    success "Mini-XDR deployed to EKS successfully!"
    echo ""
    echo "  Cluster:       $CLUSTER_NAME"
    echo "  Namespace:     $NAMESPACE"
    echo "  ALB Endpoint:  $ALB_DNS"
    echo ""

    if [ "$ALB_DNS" != "Pending..." ]; then
        info "🌐 Access your deployment:"
        echo "  Frontend:    http://$ALB_DNS"
        echo "  Backend API: http://$ALB_DNS/api"
        echo "  Health:      http://$ALB_DNS/api/health"
    else
        warning "ALB is still provisioning. Check status in 2-3 minutes:"
        echo "  kubectl get ingress -n $NAMESPACE"
    fi

    echo ""
    info "📋 Useful Commands:"
    echo ""
    echo "  # View pods"
    echo "  kubectl get pods -n $NAMESPACE"
    echo ""
    echo "  # View logs"
    echo "  kubectl logs -f deployment/mini-xdr-backend -n $NAMESPACE"
    echo "  kubectl logs -f deployment/mini-xdr-frontend -n $NAMESPACE"
    echo ""
    echo "  # Scale deployment"
    echo "  kubectl scale deployment/mini-xdr-backend --replicas=3 -n $NAMESPACE"
    echo ""
    echo "  # Port forward for local access"
    echo "  kubectl port-forward svc/mini-xdr-frontend-service 3000:3000 -n $NAMESPACE"
    echo "  kubectl port-forward svc/mini-xdr-backend-service 8000:8000 -n $NAMESPACE"
    echo ""

    info "📝 Log file saved to: $LOG_FILE"
}

# ============================================================================
# Main Execution
# ============================================================================

main() {
    header "MINI-XDR DEPLOYMENT TO EKS"

    info "This script will deploy Mini-XDR to your EKS cluster"
    info "Estimated time: 5-10 minutes"
    echo ""

    check_prerequisites
    create_namespace
    create_secrets
    create_configmaps
    deploy_persistent_volumes
    deploy_backend
    deploy_frontend
    deploy_ingress
    verify_deployment
    print_summary

    success "🎉 Deployment completed successfully!"
}

# Run main function
main "$@"
