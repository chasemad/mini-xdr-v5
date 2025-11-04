#!/bin/bash
# ============================================================================
# AWS Security Verification Script for Mini-XDR
# ============================================================================
# Comprehensive security verification for AWS EKS deployment
# Checks: Kubernetes security, Network policies, Secrets, IAM, SSL/TLS
# ============================================================================

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
AWS_REGION="${AWS_REGION:-us-east-1}"
CLUSTER_NAME="${CLUSTER_NAME:-mini-xdr-cluster}"
NAMESPACE="${NAMESPACE:-mini-xdr}"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text 2>/dev/null || echo "UNKNOWN")

# Counters
PASSED=0
FAILED=0
WARNINGS=0

# ============================================================================
# Helper Functions
# ============================================================================

log() {
    echo -e "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

success() {
    echo -e "${GREEN}✅ $1${NC}"
    ((PASSED++))
}

warning() {
    echo -e "${YELLOW}⚠️  WARNING: $1${NC}"
    ((WARNINGS++))
}

error() {
    echo -e "${RED}❌ FAILED: $1${NC}"
    ((FAILED++))
}

info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

header() {
    echo
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}🔒 $1${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo
}

# ============================================================================
# Verification Functions
# ============================================================================

verify_kubectl_access() {
    header "Verifying Kubernetes Cluster Access"

    if ! command -v kubectl &> /dev/null; then
        error "kubectl not found"
        return 1
    fi

    if ! kubectl get nodes &> /dev/null; then
        error "Cannot access Kubernetes cluster"
        return 1
    fi

    success "kubectl access verified"

    local node_count=$(kubectl get nodes --no-headers | wc -l | tr -d ' ')
    info "Cluster has $node_count nodes"

    return 0
}

verify_namespace() {
    header "Verifying Namespace Security"

    if kubectl get namespace "$NAMESPACE" &> /dev/null; then
        success "Namespace '$NAMESPACE' exists"

        # Check Pod Security Standards
        local pss_enforce=$(kubectl get namespace "$NAMESPACE" -o jsonpath='{.metadata.labels.pod-security\.kubernetes\.io/enforce}' 2>/dev/null || echo "")
        if [ -n "$pss_enforce" ]; then
            success "Pod Security Standard enforced: $pss_enforce"
        else
            warning "Pod Security Standard not configured"
        fi
    else
        error "Namespace '$NAMESPACE' does not exist"
        return 1
    fi

    return 0
}

verify_network_policies() {
    header "Verifying Network Policies"

    local np_count=$(kubectl get networkpolicies -n "$NAMESPACE" --no-headers 2>/dev/null | wc -l | tr -d ' ')

    if [ "$np_count" -gt 0 ]; then
        success "$np_count network policies found"
        kubectl get networkpolicies -n "$NAMESPACE"
    else
        warning "No network policies found"
    fi

    return 0
}

verify_pod_security() {
    header "Verifying Pod Security Contexts"

    local pods=$(kubectl get pods -n "$NAMESPACE" -o json 2>/dev/null || echo '{"items":[]}')
    local pod_count=$(echo "$pods" | jq '.items | length')

    if [ "$pod_count" -eq 0 ]; then
        warning "No pods found in namespace"
        return 0
    fi

    success "$pod_count pods found"

    # Check each pod
    for i in $(seq 0 $((pod_count - 1))); do
        local pod_name=$(echo "$pods" | jq -r ".items[$i].metadata.name")
        local run_as_user=$(echo "$pods" | jq -r ".items[$i].spec.securityContext.runAsUser // .items[$i].spec.containers[0].securityContext.runAsUser // 'not-set'")
        local run_as_non_root=$(echo "$pods" | jq -r ".items[$i].spec.securityContext.runAsNonRoot // .items[$i].spec.containers[0].securityContext.runAsNonRoot // false")

        if [ "$run_as_user" != "not-set" ] && [ "$run_as_user" != "0" ]; then
            success "Pod $pod_name: Running as user $run_as_user"
        elif [ "$run_as_non_root" == "true" ]; then
            success "Pod $pod_name: Run as non-root enforced"
        else
            error "Pod $pod_name: Running as root or no security context"
        fi

        # Check capabilities
        local capabilities=$(echo "$pods" | jq -r ".items[$i].spec.containers[0].securityContext.capabilities.drop[]?" 2>/dev/null | grep -c "ALL" || echo "0")
        if [ "$capabilities" -gt 0 ]; then
            success "Pod $pod_name: ALL capabilities dropped"
        else
            warning "Pod $pod_name: Capabilities not fully restricted"
        fi
    done

    return 0
}

verify_secrets_management() {
    header "Verifying Secrets Management"

    # Check Kubernetes secrets
    if kubectl get secrets -n "$NAMESPACE" mini-xdr-secrets &> /dev/null; then
        success "Kubernetes secret 'mini-xdr-secrets' exists"
    else
        warning "Kubernetes secret 'mini-xdr-secrets' not found"
    fi

    # Check AWS Secrets Manager integration
    if aws secretsmanager list-secrets --region "$AWS_REGION" &> /dev/null; then
        local secrets_count=$(aws secretsmanager list-secrets --region "$AWS_REGION" --query "SecretList[?contains(Name, 'mini-xdr')].Name" --output text | wc -w)

        if [ "$secrets_count" -gt 0 ]; then
            success "$secrets_count Mini-XDR secrets found in AWS Secrets Manager"

            # List secrets
            info "Secrets in AWS Secrets Manager:"
            aws secretsmanager list-secrets --region "$AWS_REGION" \
                --query "SecretList[?contains(Name, 'mini-xdr')].Name" \
                --output table
        else
            warning "No Mini-XDR secrets found in AWS Secrets Manager"
        fi
    else
        warning "Cannot access AWS Secrets Manager"
    fi

    # Check service account IAM permissions
    local sa_policy=$(kubectl get serviceaccount -n "$NAMESPACE" mini-xdr-backend -o jsonpath='{.metadata.annotations.eks\.amazonaws\.com/role-arn}' 2>/dev/null || echo "")
    if [ -n "$sa_policy" ]; then
        success "Service account has IAM role: $sa_policy"
    else
        warning "Service account IAM role not configured"
    fi

    return 0
}

verify_resource_limits() {
    header "Verifying Resource Limits"

    # Check ResourceQuota
    if kubectl get resourcequota -n "$NAMESPACE" mini-xdr-quota &> /dev/null; then
        success "Resource quota configured"
        kubectl get resourcequota -n "$NAMESPACE" mini-xdr-quota
    else
        warning "Resource quota not found"
    fi

    # Check LimitRange
    if kubectl get limitrange -n "$NAMESPACE" mini-xdr-limits &> /dev/null; then
        success "Limit range configured"
        kubectl get limitrange -n "$NAMESPACE" mini-xdr-limits
    else
        warning "Limit range not found"
    fi

    return 0
}

verify_ingress_security() {
    header "Verifying Ingress Security"

    local ingress=$(kubectl get ingress -n "$NAMESPACE" mini-xdr-ingress -o json 2>/dev/null || echo "{}")

    if [ "$ingress" != "{}" ]; then
        success "Ingress 'mini-xdr-ingress' found"

        # Check SSL/TLS configuration
        local tls_hosts=$(echo "$ingress" | jq -r '.spec.tls[0].hosts[]?' 2>/dev/null || echo "")
        if [ -n "$tls_hosts" ]; then
            success "TLS configured for: $tls_hosts"
        else
            warning "TLS not configured - HTTPS not enabled"
        fi

        # Check ALB annotations
        local alb_scheme=$(echo "$ingress" | jq -r '.metadata.annotations."alb\.ingress\.kubernetes\.io/scheme"?' 2>/dev/null || echo "")
        if [ -n "$alb_scheme" ]; then
            success "ALB scheme: $alb_scheme"
        fi

        # Check IP whitelist
        local ip_whitelist=$(echo "$ingress" | jq -r '.metadata.annotations."alb\.ingress\.kubernetes\.io/inbound-cidrs"?' 2>/dev/null || echo "")
        if [ -n "$ip_whitelist" ]; then
            info "IP whitelist configured: $ip_whitelist"
        else
            warning "No IP whitelist configured - ingress is publicly accessible"
        fi
    else
        warning "Ingress not found"
    fi

    return 0
}

verify_iam_permissions() {
    header "Verifying IAM Permissions"

    # Check EKS cluster access
    if aws eks describe-cluster --name "$CLUSTER_NAME" --region "$AWS_REGION" &> /dev/null; then
        success "EKS cluster '$CLUSTER_NAME' accessible"
    else
        error "Cannot access EKS cluster '$CLUSTER_NAME'"
        return 1
    fi

    # Check current IAM identity
    info "Current AWS Identity:"
    aws sts get-caller-identity

    return 0
}

verify_monitoring() {
    header "Verifying Monitoring & Logging"

    # Check CloudWatch log groups
    local log_groups=$(aws logs describe-log-groups --region "$AWS_REGION" \
        --log-group-name-prefix "/aws/eks/$CLUSTER_NAME" \
        --query 'logGroups[*].logGroupName' \
        --output text 2>/dev/null || echo "")

    if [ -n "$log_groups" ] && [ "$log_groups" != "None" ]; then
        local count=$(echo "$log_groups" | tr ' ' '\n' | wc -l | tr -d ' ')
        success "$count CloudWatch log groups found for cluster"
    else
        warning "CloudWatch log groups not found"
    fi

    return 0
}

verify_deployments() {
    header "Verifying Deployments"

    local deployments=$(kubectl get deployments -n "$NAMESPACE" --no-headers 2>/dev/null | wc -l | tr -d ' ')

    if [ "$deployments" -gt 0 ]; then
        success "$deployments deployments found"

        # Check deployment status
        kubectl get deployments -n "$NAMESPACE"

        # Check if all pods are ready
        local ready_pods=$(kubectl get pods -n "$NAMESPACE" -o json | jq '[.items[] | select(.status.phase=="Running" and ([.status.conditions[] | select(.type=="Ready" and .status=="True")] | length > 0))] | length')
        local total_pods=$(kubectl get pods -n "$NAMESPACE" --no-headers 2>/dev/null | wc -l | tr -d ' ')

        if [ "$ready_pods" -eq "$total_pods" ] && [ "$total_pods" -gt 0 ]; then
            success "All $total_pods pods are running and ready"
        else
            warning "$ready_pods/$total_pods pods are ready"
        fi
    else
        warning "No deployments found"
    fi

    return 0
}

# ============================================================================
# Main
# ============================================================================

main() {
    header "AWS SECURITY VERIFICATION FOR MINI-XDR"

    info "Configuration:"
    info "  AWS Region: $AWS_REGION"
    info "  EKS Cluster: $CLUSTER_NAME"
    info "  Namespace: $NAMESPACE"
    info "  Account ID: $ACCOUNT_ID"
    echo

    # Run all verifications
    verify_kubectl_access
    verify_iam_permissions
    verify_namespace
    verify_network_policies
    verify_pod_security
    verify_secrets_management
    verify_resource_limits
    verify_ingress_security
    verify_monitoring
    verify_deployments

    # Summary
    header "VERIFICATION SUMMARY"

    echo -e "✅ Passed: ${GREEN}$PASSED${NC}"
    echo -e "⚠️  Warnings: ${YELLOW}$WARNINGS${NC}"
    echo -e "❌ Failed: ${RED}$FAILED${NC}"
    echo

    if [ "$FAILED" -eq 0 ]; then
        success "All critical security checks passed!"
        if [ "$WARNINGS" -gt 0 ]; then
            warning "Review warnings above for optional security enhancements"
        fi
        exit 0
    else
        error "$FAILED critical security check(s) failed"
        exit 1
    fi
}

# Run main function
main "$@"
