#!/bin/bash
# ============================================================================
# SSL/TLS Certificate Setup Script for Mini-XDR
# ============================================================================
# Creates ACM certificate and updates ingress configuration
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
DOMAIN_NAME="${1:-}"
EMAIL="${2:-admin@example.com}"
NAMESPACE="${NAMESPACE:-mini-xdr}"

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
    log "${BLUE}🔒 $1${NC}"
    log "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo
}

# ============================================================================
# Main
# ============================================================================

main() {
    header "SSL/TLS CERTIFICATE SETUP"

    if [ -z "$DOMAIN_NAME" ]; then
        error "Usage: $0 <domain-name> [email]"
        echo "Example: $0 mini-xdr.example.com admin@example.com"
        exit 1
    fi

    info "Domain: $DOMAIN_NAME"
    info "Email: $EMAIL"
    info "Region: $AWS_REGION"
    echo

    # Request certificate
    header "Requesting ACM Certificate"

    CERT_ARN=$(aws acm request-certificate \
        --domain-name "$DOMAIN_NAME" \
        --validation-method DNS \
        --region "$AWS_REGION" \
        --query 'CertificateArn' \
        --output text 2>/dev/null || echo "")

    if [ -z "$CERT_ARN" ]; then
        error "Failed to request certificate"
    fi

    success "Certificate requested: $CERT_ARN"
    echo

    # Get validation records
    header "Certificate Validation Records"

    info "Waiting for certificate validation records..."
    sleep 5

    aws acm describe-certificate \
        --certificate-arn "$CERT_ARN" \
        --region "$AWS_REGION" \
        --query 'Certificate.DomainValidationOptions[*].[DomainName,ResourceRecord.Name,ResourceRecord.Value]' \
        --output table

    info "Add these DNS records to your domain's DNS provider:"
    echo

    # Update ingress configuration
    header "Updating Ingress Configuration"

    INGRESS_FILE="k8s/ingress-alb.yaml"

    if [ -f "$INGRESS_FILE" ]; then
        # Backup original
        cp "$INGRESS_FILE" "${INGRESS_FILE}.backup"

        # Update certificate ARN annotation
        sed -i.bak "s|# alb.ingress.kubernetes.io/certificate-arn:.*|alb.ingress.kubernetes.io/certificate-arn: $CERT_ARN|g" "$INGRESS_FILE"

        # Enable HTTPS
        sed -i.bak 's|alb.ingress.kubernetes.io/listen-ports:.*|alb.ingress.kubernetes.io/listen-ports: '"'"'[{"HTTP": 80}, {"HTTPS": 443}]'"'"'|g' "$INGRESS_FILE"

        # Enable SSL redirect
        sed -i.bak 's|# alb.ingress.kubernetes.io/ssl-redirect:.*|alb.ingress.kubernetes.io/ssl-redirect: '"'"'443'"'"'|g' "$INGRESS_FILE"

        # Uncomment TLS section
        sed -i.bak "s/# tls:/tls:/g" "$INGRESS_FILE"
        sed -i.bak "s/# - hosts:/  - hosts:/g" "$INGRESS_FILE"
        sed -i.bak "s/#   - mini-xdr.example.com/    - $DOMAIN_NAME/g" "$INGRESS_FILE"
        sed -i.bak "s/#   secretName: mini-xdr-tls-cert/    secretName: mini-xdr-tls-cert/g" "$INGRESS_FILE"

        success "Ingress configuration updated"
        info "Backup saved to: ${INGRESS_FILE}.backup"
        info "Review the changes before applying: git diff $INGRESS_FILE"
    else
        warning "Ingress file not found: $INGRESS_FILE"
    fi

    # Summary
    header "SETUP COMPLETE"

    success "Certificate ARN: $CERT_ARN"
    echo
    info "Next steps:"
    echo "  1. Add DNS validation records to your domain"
    echo "  2. Wait for certificate validation (usually 5-30 minutes)"
    echo "  3. Review updated ingress configuration: $INGRESS_FILE"
    echo "  4. Apply ingress: kubectl apply -f $INGRESS_FILE"
    echo "  5. Update DNS to point $DOMAIN_NAME to your ALB"
    echo
}

main "$@"
