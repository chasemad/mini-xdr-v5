#!/bin/bash
# ========================================================================
# Generate and Store Agent Credentials in Azure Key Vault
# ========================================================================
# Creates HMAC credentials for all Mini-XDR agents and stores in Azure
# ========================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND_DIR="$PROJECT_ROOT/backend"
AZURE_KEY_VAULT="${1:-minixdrchasemad}"

# Agent types
AGENT_TYPES=("containment" "attribution" "forensics" "deception" "hunter" "rollback")

log() {
    echo -e "${BLUE}[$(date +'%H:%M:%S')]${NC} $1"
}

success() {
    echo -e "${GREEN}✅ $1${NC}"
}

error() {
    echo -e "${RED}❌ $1${NC}"
}

echo -e "${BLUE}╔════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  Mini-XDR Agent Credentials Generator (Azure)         ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check prerequisites
if ! command -v az &> /dev/null; then
    error "Azure CLI not installed"
    exit 1
fi

if ! az account show &> /dev/null; then
    error "Not logged into Azure. Run: az login"
    exit 1
fi

if ! az keyvault show --name "$AZURE_KEY_VAULT" &> /dev/null; then
    error "Key Vault '$AZURE_KEY_VAULT' not found"
    exit 1
fi

log "Using Key Vault: $AZURE_KEY_VAULT"
echo ""

# Activate virtual environment
cd "$BACKEND_DIR"
if [ ! -d "venv" ]; then
    error "Virtual environment not found. Run: python3 -m venv venv"
    exit 1
fi

source venv/bin/activate

# Generate credentials for each agent
for agent in "${AGENT_TYPES[@]}"; do
    echo -e "${YELLOW}━━━ Generating credentials for ${agent} agent ━━━${NC}"
    
    # Generate credential using mint script
    log "Running mint_agent_cred.py..."
    output=$(python "$PROJECT_ROOT/scripts/auth/mint_agent_cred.py" 90 2>&1)
    
    # Extract values from output
    device_id=$(echo "$output" | grep "Device ID" | awk '{print $4}')
    public_id=$(echo "$output" | grep "Public ID" | awk '{print $4}')
    secret=$(echo "$output" | grep "Secret   " | awk '{print $3}')
    hmac_key=$(echo "$output" | grep "HMAC Key" | awk '{print $4}')
    
    if [ -z "$device_id" ] || [ -z "$public_id" ] || [ -z "$secret" ] || [ -z "$hmac_key" ]; then
        error "Failed to generate credentials for $agent"
        continue
    fi
    
    log "Generated credentials for $agent agent"
    echo "  Device ID: $device_id"
    echo "  Public ID: $public_id"
    echo "  Secret: ${secret:0:10}..."
    echo "  HMAC Key: ${hmac_key:0:16}..."
    
    # Store in Azure Key Vault
    log "Storing in Azure Key Vault..."
    
    az keyvault secret set \
        --vault-name "$AZURE_KEY_VAULT" \
        --name "${agent}-agent-device-id" \
        --value "$device_id" \
        --output none 2>/dev/null
    
    az keyvault secret set \
        --vault-name "$AZURE_KEY_VAULT" \
        --name "${agent}-agent-public-id" \
        --value "$public_id" \
        --output none 2>/dev/null
    
    az keyvault secret set \
        --vault-name "$AZURE_KEY_VAULT" \
        --name "${agent}-agent-secret" \
        --value "$secret" \
        --output none 2>/dev/null
    
    az keyvault secret set \
        --vault-name "$AZURE_KEY_VAULT" \
        --name "${agent}-agent-hmac-key" \
        --value "$hmac_key" \
        --output none 2>/dev/null
    
    success "Stored $agent agent credentials in Key Vault"
    echo ""
done

# Generate updated .env snippet
echo -e "${YELLOW}━━━ .env Configuration ━━━${NC}"
echo ""
echo "Add these to your backend/.env file (or re-run sync-secrets-from-azure.sh):"
echo ""

for agent in "${AGENT_TYPES[@]}"; do
    device_id=$(az keyvault secret show --vault-name "$AZURE_KEY_VAULT" --name "${agent}-agent-device-id" --query value -o tsv 2>/dev/null)
    public_id=$(az keyvault secret show --vault-name "$AZURE_KEY_VAULT" --name "${agent}-agent-public-id" --query value -o tsv 2>/dev/null)
    hmac_key=$(az keyvault secret show --vault-name "$AZURE_KEY_VAULT" --name "${agent}-agent-hmac-key" --query value -o tsv 2>/dev/null)
    secret=$(az keyvault secret show --vault-name "$AZURE_KEY_VAULT" --name "${agent}-agent-secret" --query value -o tsv 2>/dev/null)
    
    echo "# $agent Agent"
    echo "${agent^^}_AGENT_DEVICE_ID=$device_id"
    echo "${agent^^}_AGENT_PUBLIC_ID=$public_id"
    echo "${agent^^}_AGENT_HMAC_KEY=$hmac_key"
    echo "${agent^^}_AGENT_SECRET=$secret"
    echo ""
done

echo -e "${GREEN}╔════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║  ✨ Agent credentials generated and stored!            ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════╝${NC}"
echo ""
echo "Next steps:"
echo "  1. Re-sync .env: ./scripts/sync-secrets-from-azure.sh $AZURE_KEY_VAULT"
echo "  2. Restart backend to load new credentials"
echo "  3. Test agents with: ./scripts/test-azure-deployment.sh"


