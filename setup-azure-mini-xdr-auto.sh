#!/bin/bash
# ========================================================================
# Mini-XDR Azure Setup Script (Non-Interactive)
# ========================================================================
# This version runs without prompts - API keys can be added later
# ========================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
RESOURCE_GROUP="mini-xdr-rg"
LOCATION="eastus"
KEY_VAULT_NAME="minixdr$(whoami | tr '[:upper:]' '[:lower:]' | tr -d '-')"
TPOT_VM_NAME="mini-xdr-tpot"
TPOT_VM_SIZE="Standard_B2s"  # 2 vCPU, 4GB RAM - good for testing
YOUR_IP_V4=""
YOUR_IP_V6=""

# Get current directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BACKEND_DIR="$SCRIPT_DIR/backend"
ENV_FILE="$BACKEND_DIR/.env"

echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║         Mini-XDR Azure Setup - Automated (Non-Interactive)     ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# ========================================================================
# STEP 1: Verify Prerequisites
# ========================================================================
echo -e "${YELLOW}[STEP 1/7]${NC} Verifying prerequisites..."

if ! command -v az &> /dev/null; then
    echo -e "${RED}❌ Azure CLI not found${NC}"
    exit 1
fi

if ! az account show &> /dev/null; then
    echo -e "${RED}❌ Not logged into Azure${NC}"
    exit 1
fi

YOUR_IP_V4=$(curl -4 -s ifconfig.me 2>/dev/null || echo "")
YOUR_IP_V6=$(curl -6 -s ifconfig.me 2>/dev/null || echo "")

echo -e "${GREEN}✅ Azure CLI: Installed${NC}"
echo -e "${GREEN}✅ Azure Login: Authenticated${NC}"
echo -e "${GREEN}✅ Subscription: $(az account show --query name -o tsv)${NC}"
if [ -n "$YOUR_IP_V4" ]; then
    echo -e "${GREEN}✅ Your IPv4: $YOUR_IP_V4${NC}"
fi
if [ -n "$YOUR_IP_V6" ]; then
    echo -e "${GREEN}✅ Your IPv6: $YOUR_IP_V6${NC}"
fi
echo ""

# ========================================================================
# STEP 2: Create Resource Group
# ========================================================================
echo -e "${YELLOW}[STEP 2/7]${NC} Creating Azure Resource Group..."

if az group show --name "$RESOURCE_GROUP" &> /dev/null; then
    echo -e "${BLUE}ℹ️  Resource group '$RESOURCE_GROUP' already exists${NC}"
else
    az group create \
        --name "$RESOURCE_GROUP" \
        --location "$LOCATION" \
        --output none
    echo -e "${GREEN}✅ Created resource group: $RESOURCE_GROUP${NC}"
fi
echo ""

# ========================================================================
# STEP 3: Create Azure Key Vault
# ========================================================================
echo -e "${YELLOW}[STEP 3/7]${NC} Creating Azure Key Vault..."

if az keyvault show --name "$KEY_VAULT_NAME" &> /dev/null; then
    echo -e "${BLUE}ℹ️  Key Vault '$KEY_VAULT_NAME' already exists${NC}"
else
    az keyvault create \
        --name "$KEY_VAULT_NAME" \
        --resource-group "$RESOURCE_GROUP" \
        --location "$LOCATION" \
        --enable-rbac-authorization false \
        --output none
    
    # Grant yourself access
    az keyvault set-policy \
        --name "$KEY_VAULT_NAME" \
        --upn "$(az ad signed-in-user show --query userPrincipalName -o tsv)" \
        --secret-permissions get list set delete backup restore recover purge \
        --output none
    
    echo -e "${GREEN}✅ Created Key Vault: $KEY_VAULT_NAME${NC}"
fi
echo ""

# ========================================================================
# STEP 4: Generate and Store Secrets
# ========================================================================
echo -e "${YELLOW}[STEP 4/7]${NC} Generating and storing secrets..."

# Generate API keys
MINI_XDR_API_KEY=$(openssl rand -hex 32)
TPOT_API_KEY=$(openssl rand -hex 32)

# Store secrets (no external API keys for now)
az keyvault secret set --vault-name "$KEY_VAULT_NAME" --name "mini-xdr-api-key" --value "$MINI_XDR_API_KEY" --output none
az keyvault secret set --vault-name "$KEY_VAULT_NAME" --name "tpot-api-key" --value "$TPOT_API_KEY" --output none

echo -e "${GREEN}✅ Stored Mini-XDR API key${NC}"
echo -e "${GREEN}✅ Stored T-Pot API key${NC}"
echo -e "${YELLOW}ℹ️  External API keys (OpenAI, XAI, etc.) can be added later${NC}"
echo ""

# ========================================================================
# STEP 5: Deploy T-Pot Honeypot VM
# ========================================================================
echo -e "${YELLOW}[STEP 5/7]${NC} Deploying T-Pot honeypot on Azure..."

# Check if VM already exists
if az vm show --resource-group "$RESOURCE_GROUP" --name "$TPOT_VM_NAME" &> /dev/null; then
    echo -e "${BLUE}ℹ️  VM '$TPOT_VM_NAME' already exists${NC}"
    TPOT_PUBLIC_IP=$(az vm show -d --resource-group "$RESOURCE_GROUP" --name "$TPOT_VM_NAME" --query publicIps -o tsv)
    echo -e "${GREEN}✅ T-Pot VM IP: $TPOT_PUBLIC_IP${NC}"
else
    echo -e "${BLUE}Creating T-Pot VM (this may take 5-10 minutes)...${NC}"
    
    # Generate SSH key if not exists
    SSH_KEY_PATH="$HOME/.ssh/mini-xdr-tpot-azure"
    if [ ! -f "$SSH_KEY_PATH" ]; then
        ssh-keygen -t rsa -b 4096 -f "$SSH_KEY_PATH" -N "" -C "mini-xdr-tpot"
        echo -e "${GREEN}✅ Generated SSH key: $SSH_KEY_PATH${NC}"
    fi
    
    # Create VM with Ubuntu 22.04 (T-Pot supported)
    az vm create \
        --resource-group "$RESOURCE_GROUP" \
        --name "$TPOT_VM_NAME" \
        --image "Canonical:0001-com-ubuntu-server-jammy:22_04-lts:latest" \
        --size "$TPOT_VM_SIZE" \
        --admin-username "azureuser" \
        --ssh-key-values "$SSH_KEY_PATH.pub" \
        --public-ip-sku Standard \
        --output none
    
    echo -e "${GREEN}✅ VM created${NC}"
    
    # Get public IP
    TPOT_PUBLIC_IP=$(az vm show -d --resource-group "$RESOURCE_GROUP" --name "$TPOT_VM_NAME" --query publicIps -o tsv)
    echo -e "${GREEN}✅ T-Pot VM IP: $TPOT_PUBLIC_IP${NC}"
    
    # Configure Network Security Group rules
    echo -e "${BLUE}Configuring firewall rules...${NC}"
    
    NSG_NAME=$(az network nsg list --resource-group "$RESOURCE_GROUP" --query "[?contains(name, '$TPOT_VM_NAME')].name" -o tsv)
    
    # Delete default SSH rule (open to all)
    az network nsg rule delete \
        --resource-group "$RESOURCE_GROUP" \
        --nsg-name "$NSG_NAME" \
        --name "default-allow-ssh" \
        --output none 2>/dev/null || true
    
    # Add restricted SSH rule (port 64295 for T-Pot)
    if [ -n "$YOUR_IP_V4" ]; then
        az network nsg rule create \
            --resource-group "$RESOURCE_GROUP" \
            --nsg-name "$NSG_NAME" \
            --name "allow-ssh-your-ip-v4" \
            --priority 100 \
            --source-address-prefixes "$YOUR_IP_V4/32" \
            --destination-port-ranges 22 64295 \
            --access Allow \
            --protocol Tcp \
            --output none
        echo -e "${GREEN}✅ Added SSH rule for IPv4: $YOUR_IP_V4${NC}"
    fi
    
    # Add T-Pot web interface (restricted)
    if [ -n "$YOUR_IP_V4" ]; then
        az network nsg rule create \
            --resource-group "$RESOURCE_GROUP" \
            --nsg-name "$NSG_NAME" \
            --name "allow-tpot-web-v4" \
            --priority 200 \
            --source-address-prefixes "$YOUR_IP_V4/32" \
            --destination-port-ranges 64297 \
            --access Allow \
            --protocol Tcp \
            --output none
        echo -e "${GREEN}✅ Added T-Pot web access rule${NC}"
    fi
    
    # Add honeypot ports (OPEN to internet)
    az network nsg rule create \
        --resource-group "$RESOURCE_GROUP" \
        --nsg-name "$NSG_NAME" \
        --name "allow-honeypot-ports" \
        --priority 300 \
        --source-address-prefixes "*" \
        --destination-port-ranges 21 23 25 80 110 143 443 445 1433 3306 3389 5432 8080 \
        --access Allow \
        --protocol "*" \
        --output none
    echo -e "${GREEN}✅ Opened honeypot ports to internet${NC}"
    
    echo -e "${GREEN}✅ Firewall configured${NC}"
    
    # Store T-Pot host in Key Vault
    az keyvault secret set --vault-name "$KEY_VAULT_NAME" --name "tpot-host" --value "$TPOT_PUBLIC_IP" --output none
    echo -e "${GREEN}✅ Stored T-Pot IP in Key Vault${NC}"
fi

echo ""

# ========================================================================
# STEP 6: Install T-Pot on VM
# ========================================================================
echo -e "${YELLOW}[STEP 6/7]${NC} Installing T-Pot on VM..."

SSH_KEY_PATH="$HOME/.ssh/mini-xdr-tpot-azure"

if [ -f "$SSH_KEY_PATH" ]; then
    echo -e "${BLUE}Waiting 30 seconds for VM to fully boot...${NC}"
    sleep 30
    
    echo -e "${BLUE}Connecting to VM and installing T-Pot...${NC}"
    echo -e "${YELLOW}Note: T-Pot installation takes 15-30 minutes${NC}"
    
    # Create install script
    cat > /tmp/install-tpot.sh << 'TPOT_SCRIPT'
#!/bin/bash
set -e

echo "Installing T-Pot prerequisites..."
sudo apt-get update -qq
sudo apt-get install -y git curl

echo "Cloning T-Pot repository..."
cd /opt
sudo git clone https://github.com/telekom-security/tpotce

echo "Running T-Pot installer (auto mode)..."
cd tpotce
# Auto install with STANDARD edition
sudo ./install.sh --type=auto --conf=standard

echo "T-Pot installation complete! VM will reboot..."
TPOT_SCRIPT
    
    # Copy and run install script
    echo -e "${BLUE}Uploading T-Pot install script...${NC}"
    scp -o StrictHostKeyChecking=no -o ConnectTimeout=30 -i "$SSH_KEY_PATH" /tmp/install-tpot.sh "azureuser@$TPOT_PUBLIC_IP:/tmp/" || {
        echo -e "${YELLOW}⚠️  Could not upload script. VM may still be booting.${NC}"
        echo -e "${YELLOW}   You can manually install T-Pot later with:${NC}"
        echo -e "${YELLOW}   ssh -i $SSH_KEY_PATH azureuser@$TPOT_PUBLIC_IP${NC}"
    }
    
    echo -e "${BLUE}Running T-Pot installation...${NC}"
    ssh -o StrictHostKeyChecking=no -o ConnectTimeout=30 -i "$SSH_KEY_PATH" "azureuser@$TPOT_PUBLIC_IP" "bash /tmp/install-tpot.sh" || {
        echo -e "${BLUE}ℹ️  Installation started (VM will reboot)${NC}"
    }
    
    echo -e "${GREEN}✅ T-Pot installation initiated${NC}"
else
    echo -e "${YELLOW}⚠️  SSH key not found${NC}"
fi

echo ""

# ========================================================================
# STEP 7: Create .env File
# ========================================================================
echo -e "${YELLOW}[STEP 7/7]${NC} Creating backend .env file..."

# Retrieve secrets
STORED_API_KEY=$(az keyvault secret show --vault-name "$KEY_VAULT_NAME" --name "mini-xdr-api-key" --query value -o tsv)
STORED_TPOT_KEY=$(az keyvault secret show --vault-name "$KEY_VAULT_NAME" --name "tpot-api-key" --query value -o tsv)
STORED_TPOT_HOST=$(az keyvault secret show --vault-name "$KEY_VAULT_NAME" --name "tpot-host" --query value -o tsv 2>/dev/null || echo "$TPOT_PUBLIC_IP")

# Backup existing .env
if [ -f "$ENV_FILE" ]; then
    cp "$ENV_FILE" "$ENV_FILE.backup-$(date +%Y%m%d-%H%M%S)"
    echo -e "${BLUE}ℹ️  Backed up existing .env file${NC}"
fi

# Create new .env
cat > "$ENV_FILE" << ENVEOF
# Mini-XDR Configuration
# Generated by Azure setup script: $(date)
# Secrets managed in Azure Key Vault: $KEY_VAULT_NAME

# API Configuration
API_KEY=$STORED_API_KEY
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

# T-Pot Honeypot Configuration (Azure)
TPOT_API_KEY=$STORED_TPOT_KEY
TPOT_HOST=$STORED_TPOT_HOST
TPOT_SSH_PORT=64295
TPOT_WEB_PORT=64297
HONEYPOT_HOST=$STORED_TPOT_HOST
HONEYPOT_USER=azureuser
HONEYPOT_SSH_KEY=$SSH_KEY_PATH
HONEYPOT_SSH_PORT=22

# LLM Configuration (add keys later with sync-secrets-from-azure.sh)
LLM_PROVIDER=openai
OPENAI_API_KEY=ADD_TO_AZURE_KEY_VAULT
OPENAI_MODEL=gpt-4o-mini
XAI_API_KEY=ADD_TO_AZURE_KEY_VAULT
XAI_MODEL=grok-beta

# Threat Intelligence (add keys later)
ABUSEIPDB_API_KEY=ADD_TO_AZURE_KEY_VAULT
VIRUSTOTAL_API_KEY=ADD_TO_AZURE_KEY_VAULT

# Azure Key Vault
AZURE_KEY_VAULT_NAME=$KEY_VAULT_NAME
AZURE_KEY_VAULT_URL=https://$KEY_VAULT_NAME.vault.azure.net

# Agent Credentials
MINIXDR_AGENT_PROFILE=HUNTER
MINIXDR_AGENT_DEVICE_ID=hunter-device-001
MINIXDR_AGENT_HMAC_KEY=$(openssl rand -hex 32)
ENVEOF

echo -e "${GREEN}✅ Created .env file: $ENV_FILE${NC}"
echo ""

# ========================================================================
# Summary
# ========================================================================
echo -e "${GREEN}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                    Setup Complete! ✅                          ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${BLUE}📋 Summary:${NC}"
echo -e "  • Resource Group: ${GREEN}$RESOURCE_GROUP${NC}"
echo -e "  • Key Vault: ${GREEN}$KEY_VAULT_NAME${NC}"
echo -e "  • T-Pot VM: ${GREEN}$TPOT_VM_NAME${NC}"
echo -e "  • T-Pot IP: ${GREEN}$TPOT_PUBLIC_IP${NC}"
echo -e "  • SSH Key: ${GREEN}$SSH_KEY_PATH${NC}"
echo ""
echo -e "${BLUE}🔐 Access T-Pot:${NC}"
echo -e "  • Web UI: ${GREEN}https://$TPOT_PUBLIC_IP:64297${NC}"
echo -e "  • SSH: ${GREEN}ssh -i $SSH_KEY_PATH azureuser@$TPOT_PUBLIC_IP${NC}"
echo ""
echo -e "${BLUE}🚀 Next Steps:${NC}"
echo -e "  1. Wait for T-Pot to finish installing (check with: ./check-tpot-status.sh)"
echo -e "  2. Add external API keys:"
echo -e "     ${YELLOW}az keyvault secret set --vault-name $KEY_VAULT_NAME --name openai-api-key --value YOUR_KEY${NC}"
echo -e "     ${YELLOW}./sync-secrets-from-azure.sh${NC}"
echo -e "  3. Start Mini-XDR backend:"
echo -e "     ${YELLOW}cd $BACKEND_DIR && uvicorn app.main:app --reload${NC}"
echo -e "  4. Start Mini-XDR frontend:"
echo -e "     ${YELLOW}cd $SCRIPT_DIR/frontend && npm run dev${NC}"
echo -e "  5. Run test attack:"
echo -e "     ${YELLOW}./test-honeypot-attack.sh${NC}"
echo ""
echo -e "${GREEN}✨ Azure setup complete!${NC}"


