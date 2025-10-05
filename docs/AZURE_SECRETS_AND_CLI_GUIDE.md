# Azure Key Vault & CLI Setup Guide

## 🎯 Overview

Moving from AWS Secrets Manager to Azure Key Vault for managing your Mini-XDR secrets and API keys.

**Azure has excellent CLI tools** - just like AWS! The Azure CLI (`az`) is powerful and easy to use.

---

## 📦 Prerequisites

### 1. Install Azure CLI

**macOS (Homebrew):**
```bash
brew update && brew install azure-cli
```

**macOS (Direct install):**
```bash
curl -L https://aka.ms/InstallAzureCli | bash
```

**Verify installation:**
```bash
az --version
```

### 2. Login to Azure

```bash
# Interactive login
az login

# You'll be redirected to browser to authenticate
# After login, you'll see your subscriptions listed
```

**Set your subscription:**
```bash
# List subscriptions
az account list --output table

# Set active subscription
az account set --subscription "YOUR_SUBSCRIPTION_NAME_OR_ID"

# Verify
az account show
```

---

## 🔐 Azure Key Vault Setup

### Step 1: Create Resource Group (if needed)

```bash
# Create resource group in your preferred region
az group create \
  --name mini-xdr-rg \
  --location eastus

# Or use existing resource group
az group list --output table
```

### Step 2: Create Key Vault

```bash
# Create Key Vault
az keyvault create \
  --name mini-xdr-secrets \
  --resource-group mini-xdr-rg \
  --location eastus \
  --enable-rbac-authorization false

# Note: Key Vault names must be globally unique
# If 'mini-xdr-secrets' is taken, try: mini-xdr-secrets-YOUR_NAME
```

**Verify creation:**
```bash
az keyvault show \
  --name mini-xdr-secrets \
  --resource-group mini-xdr-rg
```

### Step 3: Set Access Policy

```bash
# Get your Azure AD user ID
USER_ID=$(az ad signed-in-user show --query id -o tsv)

# Grant yourself full secret permissions
az keyvault set-policy \
  --name mini-xdr-secrets \
  --upn $(az ad signed-in-user show --query userPrincipalName -o tsv) \
  --secret-permissions get list set delete backup restore recover purge
```

---

## 🔑 Storing Secrets in Azure Key Vault

### Store All Your API Keys

```bash
# 1. Mini-XDR API Key
az keyvault secret set \
  --vault-name mini-xdr-secrets \
  --name mini-xdr-api-key \
  --value "$(openssl rand -hex 32)"

# 2. OpenAI API Key
az keyvault secret set \
  --vault-name mini-xdr-secrets \
  --name openai-api-key \
  --value "YOUR_OPENAI_API_KEY"

# 3. XAI (Grok) API Key
az keyvault secret set \
  --vault-name mini-xdr-secrets \
  --name xai-api-key \
  --value "YOUR_XAI_API_KEY"

# 4. AbuseIPDB API Key
az keyvault secret set \
  --vault-name mini-xdr-secrets \
  --name abuseipdb-api-key \
  --value "YOUR_ABUSEIPDB_KEY"

# 5. VirusTotal API Key
az keyvault secret set \
  --vault-name mini-xdr-secrets \
  --name virustotal-api-key \
  --value "YOUR_VIRUSTOTAL_KEY"

# 6. T-Pot API Key (generate or use existing)
az keyvault secret set \
  --vault-name mini-xdr-secrets \
  --name tpot-api-key \
  --value "$(openssl rand -hex 32)"

# 7. T-Pot Host (your Azure T-Pot IP)
az keyvault secret set \
  --vault-name mini-xdr-secrets \
  --name tpot-host \
  --value "YOUR_AZURE_TPOT_PUBLIC_IP"
```

### Verify Secrets

```bash
# List all secrets
az keyvault secret list \
  --vault-name mini-xdr-secrets \
  --output table

# Get a specific secret (for testing)
az keyvault secret show \
  --vault-name mini-xdr-secrets \
  --name mini-xdr-api-key \
  --query value -o tsv
```

---

## 🚀 Retrieving Secrets in Your Application

### Option 1: Manual Retrieval (Development)

```bash
# Create a script to populate .env from Key Vault
cat > sync-secrets-from-azure.sh << 'EOF'
#!/bin/bash

echo "🔐 Syncing secrets from Azure Key Vault..."

VAULT_NAME="mini-xdr-secrets"
ENV_FILE="/Users/chasemad/Desktop/mini-xdr/backend/.env"

# Backup existing .env
if [ -f "$ENV_FILE" ]; then
    cp "$ENV_FILE" "$ENV_FILE.backup-$(date +%Y%m%d-%H%M%S)"
fi

# Retrieve secrets
API_KEY=$(az keyvault secret show --vault-name $VAULT_NAME --name mini-xdr-api-key --query value -o tsv)
OPENAI_KEY=$(az keyvault secret show --vault-name $VAULT_NAME --name openai-api-key --query value -o tsv)
XAI_KEY=$(az keyvault secret show --vault-name $VAULT_NAME --name xai-api-key --query value -o tsv)
ABUSEIPDB_KEY=$(az keyvault secret show --vault-name $VAULT_NAME --name abuseipdb-api-key --query value -o tsv)
VIRUSTOTAL_KEY=$(az keyvault secret show --vault-name $VAULT_NAME --name virustotal-api-key --query value -o tsv)
TPOT_KEY=$(az keyvault secret show --vault-name $VAULT_NAME --name tpot-api-key --query value -o tsv)
TPOT_HOST=$(az keyvault secret show --vault-name $VAULT_NAME --name tpot-host --query value -o tsv)

# Write to .env
cat > "$ENV_FILE" << ENVEOF
# Mini-XDR Configuration
# Synced from Azure Key Vault: $(date)

# API Configuration
API_KEY=$API_KEY

# LLM Configuration
OPENAI_API_KEY=$OPENAI_KEY
OPENAI_MODEL=gpt-4
XAI_API_KEY=$XAI_KEY
XAI_MODEL=grok-beta
LLM_PROVIDER=openai

# Threat Intelligence
ABUSEIPDB_API_KEY=$ABUSEIPDB_KEY
VIRUSTOTAL_API_KEY=$VIRUSTOTAL_KEY

# T-Pot Honeypot
TPOT_API_KEY=$TPOT_KEY
TPOT_HOST=$TPOT_HOST
TPOT_SSH_PORT=64295
TPOT_WEB_PORT=64297

# Detection Configuration
FAIL_THRESHOLD=6
FAIL_WINDOW_SECONDS=60
AUTO_CONTAIN=false
ALLOW_PRIVATE_IP_BLOCKING=true

# Database
DATABASE_URL=sqlite+aiosqlite:///./xdr.db
ENVEOF

echo "✅ Secrets synced to $ENV_FILE"
echo "   Restart backend to apply changes"
EOF

chmod +x sync-secrets-from-azure.sh
```

**Run the sync script:**
```bash
./sync-secrets-from-azure.sh
```

### Option 2: Azure VM with Managed Identity (Production)

When deploying to Azure VM, use managed identity for automatic secret retrieval:

```bash
# Enable managed identity on your VM
az vm identity assign \
  --resource-group mini-xdr-rg \
  --name mini-xdr-vm

# Get the VM's managed identity
VM_IDENTITY=$(az vm identity show \
  --resource-group mini-xdr-rg \
  --name mini-xdr-vm \
  --query principalId -o tsv)

# Grant VM access to Key Vault
az keyvault set-policy \
  --name mini-xdr-secrets \
  --object-id $VM_IDENTITY \
  --secret-permissions get list
```

Then use Azure SDK in your Python code:

```python
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient

# This works automatically on Azure VM with managed identity
credential = DefaultAzureCredential()
vault_url = "https://mini-xdr-secrets.vault.azure.net"
client = SecretClient(vault_url=vault_url, credential=credential)

# Retrieve secrets
api_key = client.get_secret("mini-xdr-api-key").value
openai_key = client.get_secret("openai-api-key").value
```

---

## 🔄 Migrating Your Current .env to Azure

### Quick Migration Script

```bash
#!/bin/bash
# migrate-env-to-azure.sh - Upload your current .env to Azure Key Vault

VAULT_NAME="mini-xdr-secrets"
ENV_FILE="/Users/chasemad/Desktop/mini-xdr/backend/.env"

echo "🔐 Migrating .env secrets to Azure Key Vault..."

# Read current .env and upload to Key Vault
while IFS='=' read -r key value; do
    # Skip comments and empty lines
    [[ $key =~ ^#.*$ ]] || [[ -z $key ]] && continue
    
    # Remove any quotes from value
    value=$(echo "$value" | sed 's/^"\(.*\)"$/\1/' | sed "s/^'\(.*\)'$/\1/")
    
    # Skip if value is empty or placeholder
    [[ -z $value ]] || [[ $value == "CONFIGURE_IN_AWS_SECRETS_MANAGER" ]] && continue
    
    # Convert key to Azure Key Vault naming (lowercase, hyphens)
    vault_key=$(echo "$key" | tr '[:upper:]' '[:lower:]' | tr '_' '-')
    
    echo "  Uploading: $key -> $vault_key"
    az keyvault secret set \
      --vault-name "$VAULT_NAME" \
      --name "$vault_key" \
      --value "$value" \
      --output none
done < "$ENV_FILE"

echo "✅ Migration complete!"
echo ""
echo "Secrets uploaded to: https://$VAULT_NAME.vault.azure.net"
echo ""
echo "To retrieve secrets, run:"
echo "  az keyvault secret list --vault-name $VAULT_NAME"
```

**Run the migration:**
```bash
chmod +x migrate-env-to-azure.sh
./migrate-env-to-azure.sh
```

---

## 📊 Managing Secrets

### List All Secrets

```bash
az keyvault secret list \
  --vault-name mini-xdr-secrets \
  --output table
```

### Get Secret Value

```bash
az keyvault secret show \
  --vault-name mini-xdr-secrets \
  --name mini-xdr-api-key \
  --query value -o tsv
```

### Update Secret

```bash
az keyvault secret set \
  --vault-name mini-xdr-secrets \
  --name mini-xdr-api-key \
  --value "NEW_VALUE"
```

### Delete Secret

```bash
# Soft delete (can be recovered)
az keyvault secret delete \
  --vault-name mini-xdr-secrets \
  --name secret-name

# Purge permanently (cannot be recovered)
az keyvault secret purge \
  --vault-name mini-xdr-secrets \
  --name secret-name
```

### Secret Versions

```bash
# List all versions of a secret
az keyvault secret list-versions \
  --vault-name mini-xdr-secrets \
  --name mini-xdr-api-key

# Get specific version
az keyvault secret show \
  --vault-name mini-xdr-secrets \
  --name mini-xdr-api-key \
  --version VERSION_ID
```

---

## 🔒 Security Best Practices

### 1. Enable Soft Delete

```bash
az keyvault update \
  --name mini-xdr-secrets \
  --enable-soft-delete true \
  --enable-purge-protection true
```

### 2. Enable Logging

```bash
# Create Log Analytics workspace
az monitor log-analytics workspace create \
  --resource-group mini-xdr-rg \
  --workspace-name mini-xdr-logs

# Get workspace ID
WORKSPACE_ID=$(az monitor log-analytics workspace show \
  --resource-group mini-xdr-rg \
  --workspace-name mini-xdr-logs \
  --query id -o tsv)

# Enable diagnostic logs
az monitor diagnostic-settings create \
  --name key-vault-logs \
  --resource $(az keyvault show --name mini-xdr-secrets --query id -o tsv) \
  --workspace $WORKSPACE_ID \
  --logs '[{"category": "AuditEvent", "enabled": true}]'
```

### 3. Set Expiration Dates

```bash
# Set secret with expiration
az keyvault secret set \
  --vault-name mini-xdr-secrets \
  --name temp-api-key \
  --value "TEMP_VALUE" \
  --expires $(date -u -d "+90 days" +%Y-%m-%dT%H:%M:%SZ)
```

### 4. Restrict Network Access

```bash
# Allow access only from specific IPs
az keyvault network-rule add \
  --vault-name mini-xdr-secrets \
  --ip-address YOUR_OFFICE_IP/32

# Or allow from Azure services only
az keyvault update \
  --name mini-xdr-secrets \
  --bypass AzureServices \
  --default-action Deny
```

---

## 🚀 Azure CLI Quick Reference

### Common Commands

```bash
# Login
az login

# List subscriptions
az account list

# Set subscription
az account set --subscription "SUBSCRIPTION_NAME"

# List resource groups
az group list

# List Key Vaults
az keyvault list

# Get Key Vault URL
az keyvault show --name mini-xdr-secrets --query properties.vaultUri -o tsv

# List all secrets
az keyvault secret list --vault-name mini-xdr-secrets

# Backup secret
az keyvault secret backup --vault-name mini-xdr-secrets --name api-key --file api-key.backup

# Restore secret
az keyvault secret restore --vault-name mini-xdr-secrets --file api-key.backup
```

---

## 🔄 Development Workflow

### 1. Local Development

Keep using `.env` file locally:

```bash
# Sync from Azure when needed
./sync-secrets-from-azure.sh

# Work locally
cd backend
uvicorn app.main:app --reload
```

### 2. Staging/Production on Azure VM

Use managed identity:

```python
# backend/app/config.py
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
import os

def load_secrets_from_azure():
    """Load secrets from Azure Key Vault if running on Azure VM"""
    if os.getenv("AZURE_VM"):  # Set this env var on Azure VM
        credential = DefaultAzureCredential()
        vault_url = "https://mini-xdr-secrets.vault.azure.net"
        client = SecretClient(vault_url=vault_url, credential=credential)
        
        # Load secrets
        os.environ["API_KEY"] = client.get_secret("mini-xdr-api-key").value
        os.environ["OPENAI_API_KEY"] = client.get_secret("openai-api-key").value
        # ... load other secrets
```

---

## 💰 Cost Optimization

Azure Key Vault pricing:
- **Secret operations:** $0.03 per 10,000 operations
- **Secrets stored:** First 10,000 secrets free, then minimal cost

**Best practices:**
- Cache secrets in your application
- Don't retrieve on every request
- Use managed identity (free)

---

## 🆘 Troubleshooting

### Authentication Issues

```bash
# Re-login
az logout
az login

# Check current account
az account show

# Clear cache
rm -rf ~/.azure
az login
```

### Permission Denied

```bash
# Check your permissions
az keyvault show --name mini-xdr-secrets

# Re-grant access
az keyvault set-policy \
  --name mini-xdr-secrets \
  --upn $(az ad signed-in-user show --query userPrincipalName -o tsv) \
  --secret-permissions get list set delete
```

### Key Vault Not Found

```bash
# List all Key Vaults
az keyvault list --output table

# Check resource group
az group list --output table
```

---

## 📚 Additional Resources

- **Azure CLI Docs:** https://learn.microsoft.com/en-us/cli/azure/
- **Key Vault Docs:** https://learn.microsoft.com/en-us/azure/key-vault/
- **Python SDK:** https://learn.microsoft.com/en-us/python/api/overview/azure/keyvault-secrets-readme

---

## ✅ Quick Start Checklist

- [ ] Install Azure CLI
- [ ] Login to Azure (`az login`)
- [ ] Create Resource Group
- [ ] Create Key Vault
- [ ] Upload secrets from .env
- [ ] Test secret retrieval
- [ ] Create sync script
- [ ] Update backend to use secrets
- [ ] Test locally
- [ ] Deploy to Azure VM with managed identity

---

**Summary:** Azure CLI is just as powerful as AWS CLI! Use Azure Key Vault for secure secret management, and managed identity for seamless access on Azure VMs. 🔐


