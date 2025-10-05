# 🔐 Mini-XDR Azure Setup Guide

Complete guide for setting up Mini-XDR with Azure Key Vault and T-Pot honeypot.

---

## 📋 Quick Start (5 Minutes)

### Step 1: Run the Setup Script

```bash
cd /Users/chasemad/Desktop/mini-xdr
./setup-azure-mini-xdr.sh
```

This will:
- ✅ Create Azure Resource Group
- ✅ Create Azure Key Vault
- ✅ Generate and store all secrets
- ✅ Deploy T-Pot honeypot VM (with restricted access)
- ✅ Configure Mini-XDR backend `.env` file

**Your IPs detected:**
- IPv4: `24.11.0.176`
- IPv6: `2601:681:8b01:36b0:1435:f6bd:f64:47fe`

The script will automatically restrict T-Pot admin access to ONLY your IP addresses for security.

### Step 2: Wait for T-Pot Installation

T-Pot takes 15-30 minutes to install. Check status:

```bash
./check-tpot-status.sh
```

### Step 3: Start Mini-XDR

```bash
# Terminal 1: Start backend
cd backend
source venv/bin/activate  # or: . venv/bin/activate
uvicorn app.main:app --reload

# Terminal 2: Start frontend
cd frontend
npm run dev
```

### Step 4: Test the System

```bash
# Run attack simulation
./test-honeypot-attack.sh
```

This simulates:
- SSH brute force (10 attempts)
- Port scanning
- HTTP probing
- Telnet connections
- FTP attempts

---

## 🔑 Azure Key Vault Management

### View All Secrets

```bash
# List secrets
az keyvault secret list \
  --vault-name mini-xdr-secrets-$(whoami | tr '[:upper:]' '[:lower:]') \
  --output table

# Get specific secret
az keyvault secret show \
  --vault-name mini-xdr-secrets-$(whoami | tr '[:upper:]' '[:lower:]') \
  --name mini-xdr-api-key \
  --query value -o tsv
```

### Update Secrets

```bash
# Update a secret
az keyvault secret set \
  --vault-name mini-xdr-secrets-$(whoami | tr '[:upper:]' '[:lower:]') \
  --name openai-api-key \
  --value "YOUR_NEW_KEY"

# Sync to .env
./sync-secrets-from-azure.sh
```

### Add New Secrets

```bash
# Add a new secret
az keyvault secret set \
  --vault-name mini-xdr-secrets-$(whoami | tr '[:upper:]' '[:lower:]') \
  --name new-api-key \
  --value "YOUR_VALUE"
```

---

## 🍯 T-Pot Honeypot Management

### Check Status

```bash
./check-tpot-status.sh
```

### Access T-Pot

**Web Interface:**
```
https://YOUR_TPOT_IP:64297
```

Default credentials:
- Username: `tsec`
- Password: Generated on first boot (check via SSH)

**SSH Access:**
```bash
ssh -i ~/.ssh/mini-xdr-tpot-azure azureuser@YOUR_TPOT_IP
```

### View T-Pot Logs

```bash
# SSH to VM
ssh -i ~/.ssh/mini-xdr-tpot-azure azureuser@YOUR_TPOT_IP

# View logs
sudo docker logs nginx
sudo docker logs cowrie  # SSH honeypot
sudo docker logs dionaea  # Multi-protocol honeypot

# View all containers
sudo docker ps
```

### Restart T-Pot Services

```bash
ssh -i ~/.ssh/mini-xdr-tpot-azure azureuser@YOUR_TPOT_IP

cd /opt/tpotce
sudo docker-compose restart
```

---

## 🧪 Testing Mini-XDR Detection

### Run Attack Simulation

```bash
./test-honeypot-attack.sh
```

### Monitor Mini-XDR

**Backend Logs:**
```bash
tail -f backend/backend.log
```

**Dashboard:**
```
http://localhost:3000
```

Check for:
- 🚨 Detected alerts
- 📊 Threat intelligence enrichment
- 🤖 AI-generated recommendations
- 🛡️ Automated containment actions

---

## 🔄 Daily Workflow

### Update Local Secrets

```bash
# Pull latest secrets from Azure
./sync-secrets-from-azure.sh

# Restart backend
cd backend
uvicorn app.main:app --reload
```

### Check Honeypot Health

```bash
./check-tpot-status.sh
```

### Run Test Attacks (for development)

```bash
./test-honeypot-attack.sh
```

---

## 🛠️ Troubleshooting

### Azure CLI Issues

```bash
# Re-login
az logout
az login

# Check subscription
az account show

# List available Key Vaults
az keyvault list --output table
```

### T-Pot Not Responding

```bash
# Check VM status
az vm show -d \
  --resource-group mini-xdr-rg \
  --name mini-xdr-tpot \
  --query powerState

# Start VM if stopped
az vm start \
  --resource-group mini-xdr-rg \
  --name mini-xdr-tpot

# SSH and check services
ssh -i ~/.ssh/mini-xdr-tpot-azure azureuser@YOUR_IP
sudo docker ps
sudo systemctl status docker
```

### Mini-XDR Not Detecting Events

1. **Check backend logs:**
   ```bash
   tail -f backend/backend.log
   ```

2. **Verify .env configuration:**
   ```bash
   cat backend/.env | grep TPOT
   ```

3. **Test T-Pot connectivity:**
   ```bash
   ssh -i ~/.ssh/mini-xdr-tpot-azure azureuser@YOUR_TPOT_IP
   ```

4. **Run manual test:**
   ```bash
   ./test-honeypot-attack.sh
   ```

### Permission Denied (Azure Key Vault)

```bash
# Re-grant access policy
az keyvault set-policy \
  --name mini-xdr-secrets-$(whoami | tr '[:upper:]' '[:lower:]') \
  --upn $(az ad signed-in-user show --query userPrincipalName -o tsv) \
  --secret-permissions get list set delete
```

---

## 💰 Azure Cost Management

### Current Setup Costs (Estimated)

**Monthly costs:**
- Azure Key Vault: ~$0.03 (minimal usage)
- T-Pot VM (Standard_B2s): ~$30-50/month
- Storage: ~$1-5/month
- **Total: ~$35-60/month**

### Cost Optimization

**Stop VM when not testing:**
```bash
# Stop VM (no compute charges, keep disk)
az vm deallocate \
  --resource-group mini-xdr-rg \
  --name mini-xdr-tpot

# Start VM when needed
az vm start \
  --resource-group mini-xdr-rg \
  --name mini-xdr-tpot
```

**Delete resources when done:**
```bash
# Delete entire resource group (WARNING: IRREVERSIBLE)
az group delete \
  --name mini-xdr-rg \
  --yes --no-wait

# Secrets are soft-deleted and can be recovered for 90 days
```

---

## 🔒 Security Best Practices

### ✅ Implemented

- 🔐 Secrets stored in Azure Key Vault (not in .env)
- 🛡️ T-Pot admin access restricted to your IP only
- 🔑 SSH key-based authentication (no passwords)
- 🚫 Firewall rules blocking all unauthorized access

### 📋 Recommended

1. **Rotate secrets monthly:**
   ```bash
   # Generate new API key
   NEW_KEY=$(openssl rand -hex 32)
   
   # Update in Key Vault
   az keyvault secret set \
     --vault-name mini-xdr-secrets-$(whoami | tr '[:upper:]' '[:lower:]') \
     --name mini-xdr-api-key \
     --value "$NEW_KEY"
   
   # Sync to .env
   ./sync-secrets-from-azure.sh
   ```

2. **Enable Key Vault logging:**
   ```bash
   # Create Log Analytics workspace
   az monitor log-analytics workspace create \
     --resource-group mini-xdr-rg \
     --workspace-name mini-xdr-logs
   
   # Enable diagnostic logs (see AZURE_SECRETS_AND_CLI_GUIDE.md)
   ```

3. **Review T-Pot logs regularly:**
   ```bash
   ssh -i ~/.ssh/mini-xdr-tpot-azure azureuser@YOUR_IP
   cd /opt/tpotce
   sudo docker-compose logs --tail=100
   ```

---

## 📚 Additional Resources

- **Azure Key Vault Docs:** https://learn.microsoft.com/en-us/azure/key-vault/
- **T-Pot Documentation:** https://github.com/telekom-security/tpotce
- **Mini-XDR Docs:** `docs/` directory

---

## 🆘 Quick Commands Reference

```bash
# Azure Login
az login

# List secrets
az keyvault secret list --vault-name mini-xdr-secrets-$(whoami | tr '[:upper:]' '[:lower:]')

# Sync secrets to .env
./sync-secrets-from-azure.sh

# Check T-Pot status
./check-tpot-status.sh

# Run attack test
./test-honeypot-attack.sh

# Start Mini-XDR backend
cd backend && uvicorn app.main:app --reload

# Start Mini-XDR frontend
cd frontend && npm run dev

# View backend logs
tail -f backend/backend.log

# SSH to T-Pot
ssh -i ~/.ssh/mini-xdr-tpot-azure azureuser@YOUR_TPOT_IP

# Stop T-Pot VM (save costs)
az vm deallocate --resource-group mini-xdr-rg --name mini-xdr-tpot

# Start T-Pot VM
az vm start --resource-group mini-xdr-rg --name mini-xdr-tpot
```

---

## ✅ Setup Checklist

- [ ] Azure CLI installed and logged in
- [ ] Run `./setup-azure-mini-xdr.sh`
- [ ] Wait for T-Pot installation (15-30 min)
- [ ] Verify T-Pot status: `./check-tpot-status.sh`
- [ ] Add API keys to Azure Key Vault
- [ ] Start Mini-XDR backend and frontend
- [ ] Run test attack: `./test-honeypot-attack.sh`
- [ ] Verify detection in Mini-XDR dashboard
- [ ] Access T-Pot web UI: `https://YOUR_IP:64297`

---

**🎉 You're all set! Your Mini-XDR is now running with Azure-managed secrets and a secure T-Pot honeypot.**

For detailed CLI commands, see: `AZURE_SECRETS_AND_CLI_GUIDE.md`

