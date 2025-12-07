#!/bin/bash
# ========================================================================
# Add API Keys to Azure Key Vault
# ========================================================================

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

KEY_VAULT_NAME="minixdr-keyvault"

echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║           Add API Keys to Azure Key Vault                     ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${YELLOW}Please enter your API keys (press Enter to skip any):${NC}"
echo ""

# OpenAI API Key
read -p "OpenAI API Key (sk-...): " OPENAI_KEY
if [ -n "$OPENAI_KEY" ]; then
    az keyvault secret set --vault-name "$KEY_VAULT_NAME" --name "openai-api-key" --value "$OPENAI_KEY" --output none
    echo -e "${GREEN}✅ Stored OpenAI API key${NC}"
fi

# XAI (Grok) API Key
read -p "XAI (Grok) API Key: " XAI_KEY
if [ -n "$XAI_KEY" ]; then
    az keyvault secret set --vault-name "$KEY_VAULT_NAME" --name "xai-api-key" --value "$XAI_KEY" --output none
    echo -e "${GREEN}✅ Stored XAI API key${NC}"
fi

# AbuseIPDB API Key
read -p "AbuseIPDB API Key: " ABUSEIPDB_KEY
if [ -n "$ABUSEIPDB_KEY" ]; then
    az keyvault secret set --vault-name "$KEY_VAULT_NAME" --name "abuseipdb-api-key" --value "$ABUSEIPDB_KEY" --output none
    echo -e "${GREEN}✅ Stored AbuseIPDB API key${NC}"
fi

# VirusTotal API Key
read -p "VirusTotal API Key: " VIRUSTOTAL_KEY
if [ -n "$VIRUSTOTAL_KEY" ]; then
    az keyvault secret set --vault-name "$KEY_VAULT_NAME" --name "virustotal-api-key" --value "$VIRUSTOTAL_KEY" --output none
    echo -e "${GREEN}✅ Stored VirusTotal API key${NC}"
fi

echo ""
echo -e "${GREEN}✅ API keys added to Azure Key Vault!${NC}"
echo ""
echo -e "${BLUE}Next steps:${NC}"
echo -e "  1. Sync secrets to .env: ${YELLOW}./sync-secrets-from-azure.sh${NC}"
echo -e "  2. Continue setup: ${YELLOW}./setup-azure-mini-xdr-auto.sh${NC}"
echo ""

