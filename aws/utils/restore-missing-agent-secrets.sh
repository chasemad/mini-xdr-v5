#!/bin/bash

# RESTORE MISSING AI AGENT SECRETS
# This script restores the AI agent secrets that were in your original backup

set -euo pipefail

REGION="${AWS_REGION:-us-east-1}"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

log() { echo -e "${GREEN}[$(date +'%H:%M:%S')] $1${NC}"; }
warn() { echo -e "${YELLOW}[WARN] $1${NC}"; }
error() { echo -e "${RED}[ERROR] $1${NC}"; }

echo "🤖 RESTORING MISSING AI AGENT SECRETS"
echo "====================================="

# The agent secrets from your original backup file
# (I extracted these from the grep output earlier)
AGENT_SECRETS=(
    "attribution:VSYKH1g2YN-BlQ4uPgh3kgDxu0QXedbf3adrfEWe-Jg"
    "forensics:Bv9ZPk21eCNM_z10gRyWEBmLK22NOnhbEr5opo1y9g8"
    "deception:N_l6Ifpt8OWMnfNJsN7fLtbzh3_cKw3A94kMHCVhw1A"
    "hunter:LYl21D1ooRyTAJ_LYZ802zSOq9t0CQyZLAN0R8k31oA"
    "rollback:1RRRp1GGiJfIQsVoNzPtj_cs7LOLcoKuuF5YQoY7PnQ"
)

log "Found containment agent secret already stored"
log "Restoring ${#AGENT_SECRETS[@]} missing agent secrets..."

# Store each missing agent secret
for agent_data in "${AGENT_SECRETS[@]}"; do
    IFS=':' read -r agent_name secret_value <<< "$agent_data"
    secret_name="mini-xdr/agents/${agent_name}-secret"
    
    log "Storing secret: $secret_name"
    
    if aws secretsmanager create-secret \
        --name "$secret_name" \
        --secret-string "$secret_value" \
        --description "Mini-XDR $agent_name agent secret" \
        --region "$REGION" >/dev/null 2>&1; then
        log "✅ Created: $secret_name"
    else
        aws secretsmanager update-secret \
            --secret-id "$secret_name" \
            --secret-string "$secret_value" \
            --region "$REGION" >/dev/null
        log "✅ Updated: $secret_name"
    fi
done

log "Updating backend configuration with all agent secrets..."

# Add the missing agent secret references to .env
cat >> /Users/chasemad/Desktop/mini-xdr/backend/.env << 'EOF'

# AI Agent Secrets - Retrieved from AWS Secrets Manager
ATTRIBUTION_AGENT_SECRET_NAME=mini-xdr/agents/attribution-secret
FORENSICS_AGENT_SECRET_NAME=mini-xdr/agents/forensics-secret
DECEPTION_AGENT_SECRET_NAME=mini-xdr/agents/deception-secret
HUNTER_AGENT_SECRET_NAME=mini-xdr/agents/hunter-secret
ROLLBACK_AGENT_SECRET_NAME=mini-xdr/agents/rollback-secret
CONTAINMENT_AGENT_SECRET_NAME=mini-xdr/agents/containment-secret
EOF

log "✅ Backend configuration updated with all agent secrets"

echo ""
log "📋 Listing all agent secrets:"
aws secretsmanager list-secrets --query 'SecretList[?contains(Name, `agents`)].Name' --output table

echo ""
log "✅ All AI agent secrets have been restored!"
echo "🤖 Your agents should now have access to their authentication secrets"