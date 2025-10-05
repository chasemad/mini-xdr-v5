#!/bin/bash
# ========================================================================
# Sync Secrets from Azure Key Vault to .env
# ========================================================================
# Use this script to update your local .env with secrets from Azure
# ========================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
KEY_VAULT_NAME="${1:-minixdr$(whoami | tr '[:upper:]' '[:lower:]' | tr -d '-')}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND_DIR="$PROJECT_ROOT/backend"
ENV_FILE="$BACKEND_DIR/.env"

echo -e "${BLUE}🔐 Syncing secrets from Azure Key Vault...${NC}"
echo -e "Key Vault: ${GREEN}$KEY_VAULT_NAME${NC}"
echo ""

# Verify Azure CLI
if ! command -v az &> /dev/null; then
    echo -e "${RED}❌ Azure CLI not found${NC}"
    exit 1
fi

# Verify logged in
if ! az account show &> /dev/null; then
    echo -e "${RED}❌ Not logged into Azure. Run: az login${NC}"
    exit 1
fi

# Verify Key Vault exists
if ! az keyvault show --name "$KEY_VAULT_NAME" &> /dev/null; then
    echo -e "${RED}❌ Key Vault '$KEY_VAULT_NAME' not found${NC}"
    echo -e "${YELLOW}Available Key Vaults:${NC}"
    az keyvault list --query "[].name" -o tsv
    exit 1
fi

# Backup existing .env
if [ -f "$ENV_FILE" ]; then
    cp "$ENV_FILE" "$ENV_FILE.backup-$(date +%Y%m%d-%H%M%S)"
    echo -e "${GREEN}✅ Backed up existing .env${NC}"
fi

# Retrieve secrets
echo -e "${BLUE}Retrieving secrets...${NC}"

API_KEY=$(az keyvault secret show --vault-name "$KEY_VAULT_NAME" --name "mini-xdr-api-key" --query value -o tsv 2>/dev/null || echo "")
TPOT_KEY=$(az keyvault secret show --vault-name "$KEY_VAULT_NAME" --name "tpot-api-key" --query value -o tsv 2>/dev/null || echo "")
TPOT_HOST=$(az keyvault secret show --vault-name "$KEY_VAULT_NAME" --name "tpot-host" --query value -o tsv 2>/dev/null || echo "")
OPENAI_KEY=$(az keyvault secret show --vault-name "$KEY_VAULT_NAME" --name "openai-api-key" --query value -o tsv 2>/dev/null || echo "")
XAI_KEY=$(az keyvault secret show --vault-name "$KEY_VAULT_NAME" --name "xai-api-key" --query value -o tsv 2>/dev/null || echo "")
ABUSEIPDB_KEY=$(az keyvault secret show --vault-name "$KEY_VAULT_NAME" --name "abuseipdb-api-key" --query value -o tsv 2>/dev/null || echo "")
VIRUSTOTAL_KEY=$(az keyvault secret show --vault-name "$KEY_VAULT_NAME" --name "virustotal-api-key" --query value -o tsv 2>/dev/null || echo "")

# Create .env file
cat > "$ENV_FILE" << ENVEOF
# Mini-XDR Configuration
# Synced from Azure Key Vault: $(date)

# API Configuration
API_KEY=${API_KEY:-GENERATE_NEW_KEY}
API_HOST=127.0.0.1
API_PORT=8000
UI_ORIGIN=http://localhost:3000

# Database
DATABASE_URL=sqlite+aiosqlite:///./xdr.db

# Detection Configuration
FAIL_WINDOW_SECONDS=60
FAIL_THRESHOLD=6
AUTO_CONTAIN=false
ALLOW_PRIVATE_IP_BLOCKING=true

# T-Pot Honeypot (Azure)
TPOT_API_KEY=${TPOT_KEY:-GENERATE_NEW_KEY}
TPOT_HOST=${TPOT_HOST:-CONFIGURE_TPOT_IP}
TPOT_SSH_PORT=64295
TPOT_WEB_PORT=64297
HONEYPOT_HOST=${TPOT_HOST:-CONFIGURE_TPOT_IP}
HONEYPOT_USER=azureuser
HONEYPOT_SSH_KEY=$HOME/.ssh/mini-xdr-tpot-azure
HONEYPOT_SSH_PORT=64295

# LLM Configuration
LLM_PROVIDER=openai
OPENAI_API_KEY=${OPENAI_KEY:-CONFIGURE_IN_AZURE_KEY_VAULT}
OPENAI_MODEL=gpt-4o-mini
XAI_API_KEY=${XAI_KEY:-CONFIGURE_IN_AZURE_KEY_VAULT}
XAI_MODEL=grok-beta

# Threat Intelligence
ABUSEIPDB_API_KEY=${ABUSEIPDB_KEY:-CONFIGURE_IN_AZURE_KEY_VAULT}
VIRUSTOTAL_API_KEY=${VIRUSTOTAL_KEY:-CONFIGURE_IN_AZURE_KEY_VAULT}

# Azure Key Vault
AZURE_KEY_VAULT_NAME=$KEY_VAULT_NAME
AZURE_KEY_VAULT_URL=https://$KEY_VAULT_NAME.vault.azure.net

# Agent Credentials (retrieved from Key Vault if available)
MINIXDR_AGENT_PROFILE=HUNTER
MINIXDR_AGENT_DEVICE_ID=$(az keyvault secret show --vault-name "$KEY_VAULT_NAME" --name "hunter-agent-device-id" --query value -o tsv 2>/dev/null || echo "hunter-device-001")
MINIXDR_AGENT_HMAC_KEY=$(az keyvault secret show --vault-name "$KEY_VAULT_NAME" --name "hunter-agent-hmac-key" --query value -o tsv 2>/dev/null || openssl rand -hex 32)

# Individual Agent Credentials
CONTAINMENT_AGENT_DEVICE_ID=$(az keyvault secret show --vault-name "$KEY_VAULT_NAME" --name "containment-agent-device-id" --query value -o tsv 2>/dev/null || echo "")
CONTAINMENT_AGENT_PUBLIC_ID=$(az keyvault secret show --vault-name "$KEY_VAULT_NAME" --name "containment-agent-public-id" --query value -o tsv 2>/dev/null || echo "")
CONTAINMENT_AGENT_HMAC_KEY=$(az keyvault secret show --vault-name "$KEY_VAULT_NAME" --name "containment-agent-hmac-key" --query value -o tsv 2>/dev/null || echo "")
CONTAINMENT_AGENT_SECRET=$(az keyvault secret show --vault-name "$KEY_VAULT_NAME" --name "containment-agent-secret" --query value -o tsv 2>/dev/null || echo "")

ATTRIBUTION_AGENT_DEVICE_ID=$(az keyvault secret show --vault-name "$KEY_VAULT_NAME" --name "attribution-agent-device-id" --query value -o tsv 2>/dev/null || echo "")
ATTRIBUTION_AGENT_PUBLIC_ID=$(az keyvault secret show --vault-name "$KEY_VAULT_NAME" --name "attribution-agent-public-id" --query value -o tsv 2>/dev/null || echo "")
ATTRIBUTION_AGENT_HMAC_KEY=$(az keyvault secret show --vault-name "$KEY_VAULT_NAME" --name "attribution-agent-hmac-key" --query value -o tsv 2>/dev/null || echo "")
ATTRIBUTION_AGENT_SECRET=$(az keyvault secret show --vault-name "$KEY_VAULT_NAME" --name "attribution-agent-secret" --query value -o tsv 2>/dev/null || echo "")

FORENSICS_AGENT_DEVICE_ID=$(az keyvault secret show --vault-name "$KEY_VAULT_NAME" --name "forensics-agent-device-id" --query value -o tsv 2>/dev/null || echo "")
FORENSICS_AGENT_PUBLIC_ID=$(az keyvault secret show --vault-name "$KEY_VAULT_NAME" --name "forensics-agent-public-id" --query value -o tsv 2>/dev/null || echo "")
FORENSICS_AGENT_HMAC_KEY=$(az keyvault secret show --vault-name "$KEY_VAULT_NAME" --name "forensics-agent-hmac-key" --query value -o tsv 2>/dev/null || echo "")
FORENSICS_AGENT_SECRET=$(az keyvault secret show --vault-name "$KEY_VAULT_NAME" --name "forensics-agent-secret" --query value -o tsv 2>/dev/null || echo "")

DECEPTION_AGENT_DEVICE_ID=$(az keyvault secret show --vault-name "$KEY_VAULT_NAME" --name "deception-agent-device-id" --query value -o tsv 2>/dev/null || echo "")
DECEPTION_AGENT_PUBLIC_ID=$(az keyvault secret show --vault-name "$KEY_VAULT_NAME" --name "deception-agent-public-id" --query value -o tsv 2>/dev/null || echo "")
DECEPTION_AGENT_HMAC_KEY=$(az keyvault secret show --vault-name "$KEY_VAULT_NAME" --name "deception-agent-hmac-key" --query value -o tsv 2>/dev/null || echo "")
DECEPTION_AGENT_SECRET=$(az keyvault secret show --vault-name "$KEY_VAULT_NAME" --name "deception-agent-secret" --query value -o tsv 2>/dev/null || echo "")

HUNTER_AGENT_DEVICE_ID=$(az keyvault secret show --vault-name "$KEY_VAULT_NAME" --name "hunter-agent-device-id" --query value -o tsv 2>/dev/null || echo "")
HUNTER_AGENT_PUBLIC_ID=$(az keyvault secret show --vault-name "$KEY_VAULT_NAME" --name "hunter-agent-public-id" --query value -o tsv 2>/dev/null || echo "")
HUNTER_AGENT_HMAC_KEY=$(az keyvault secret show --vault-name "$KEY_VAULT_NAME" --name "hunter-agent-hmac-key" --query value -o tsv 2>/dev/null || echo "")
HUNTER_AGENT_SECRET=$(az keyvault secret show --vault-name "$KEY_VAULT_NAME" --name "hunter-agent-secret" --query value -o tsv 2>/dev/null || echo "")

ROLLBACK_AGENT_DEVICE_ID=$(az keyvault secret show --vault-name "$KEY_VAULT_NAME" --name "rollback-agent-device-id" --query value -o tsv 2>/dev/null || echo "")
ROLLBACK_AGENT_PUBLIC_ID=$(az keyvault secret show --vault-name "$KEY_VAULT_NAME" --name "rollback-agent-public-id" --query value -o tsv 2>/dev/null || echo "")
ROLLBACK_AGENT_HMAC_KEY=$(az keyvault secret show --vault-name "$KEY_VAULT_NAME" --name "rollback-agent-hmac-key" --query value -o tsv 2>/dev/null || echo "")
ROLLBACK_AGENT_SECRET=$(az keyvault secret show --vault-name "$KEY_VAULT_NAME" --name "rollback-agent-secret" --query value -o tsv 2>/dev/null || echo "")
ENVEOF

echo -e "${GREEN}✅ Secrets synced to: $ENV_FILE${NC}"
echo ""
echo -e "${BLUE}Retrieved secrets:${NC}"
[ -n "$API_KEY" ] && echo -e "  ✅ Mini-XDR API Key" || echo -e "  ⚠️  Mini-XDR API Key (missing)"
[ -n "$TPOT_KEY" ] && echo -e "  ✅ T-Pot API Key" || echo -e "  ⚠️  T-Pot API Key (missing)"
[ -n "$TPOT_HOST" ] && echo -e "  ✅ T-Pot Host: $TPOT_HOST" || echo -e "  ⚠️  T-Pot Host (missing)"
[ -n "$OPENAI_KEY" ] && echo -e "  ✅ OpenAI API Key" || echo -e "  ⚠️  OpenAI API Key (missing)"
[ -n "$XAI_KEY" ] && echo -e "  ✅ XAI API Key" || echo -e "  ⚠️  XAI API Key (missing)"
[ -n "$ABUSEIPDB_KEY" ] && echo -e "  ✅ AbuseIPDB API Key" || echo -e "  ⚠️  AbuseIPDB API Key (missing)"
[ -n "$VIRUSTOTAL_KEY" ] && echo -e "  ✅ VirusTotal API Key" || echo -e "  ⚠️  VirusTotal API Key (missing)"
echo ""
echo -e "${GREEN}✨ Backend secrets synced!${NC}"
echo ""

# Also sync to frontend
FRONTEND_ENV="$PROJECT_ROOT/frontend/.env.local"
echo -e "${BLUE}Syncing API key to frontend...${NC}"

if [ -f "$FRONTEND_ENV" ]; then
    cp "$FRONTEND_ENV" "${FRONTEND_ENV}.backup-$(date +%Y%m%d-%H%M%S)" 2>/dev/null
fi

cat > "$FRONTEND_ENV" << FRONTENDEOF
# MINI-XDR FRONTEND LOCAL DEVELOPMENT CONFIGURATION
# Auto-synced from Azure Key Vault on $(date)

NEXT_PUBLIC_API_BASE=http://localhost:8000
NEXT_PUBLIC_API_KEY=${API_KEY:-GENERATE_NEW_KEY}
NEXT_PUBLIC_ENVIRONMENT=development
NEXT_PUBLIC_CSP_ENABLED=false
NEXT_PUBLIC_DEBUG=true

# Security Configuration (disabled for local development)
NEXT_PUBLIC_SECRETS_MANAGER_ENABLED=false
FRONTENDEOF

echo -e "${GREEN}✅ Frontend .env.local synced${NC}"
echo ""
echo -e "${GREEN}✨ Sync complete!${NC}"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo "  1. Restart backend:  pkill -f uvicorn && cd $PROJECT_ROOT/backend && uvicorn app.entrypoint:app --reload"
echo "  2. Restart frontend: pkill -f 'next dev' && cd $PROJECT_ROOT/frontend && npm run dev"
echo ""

