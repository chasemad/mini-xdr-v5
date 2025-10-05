# 🎯 Azure Deployment Summary

**Status:** Ready to Deploy ✅  
**Date:** October 5, 2025  
**Your IP:** 24.11.0.176 (IPv4), 2601:681:8b01:36b0:1435:f6bd:f64:47fe (IPv6)

---

## 🎬 What We've Prepared

### 1. Azure Secret Management
- **Migration from AWS Secrets Manager → Azure Key Vault**
- All API keys will be stored securely in Azure
- Local `.env` file will be auto-generated from Key Vault
- Script to sync secrets on-demand

### 2. T-Pot Honeypot Deployment
- **Fully automated deployment on Azure VM**
- Ubuntu 22.04 LTS with T-Pot CE (Community Edition)
- Restricted admin access (only your IP can manage it)
- Honeypot ports open to internet (for capturing real attacks)
- Automatic installation of all honeypot services

### 3. Testing Framework
- Attack simulation script (SSH brute force, port scans, HTTP probing)
- Health check script for T-Pot
- End-to-end testing workflow

---

## 📁 Files Created

```
/Users/chasemad/Desktop/mini-xdr/
├── setup-azure-mini-xdr.sh         # Main setup (creates everything)
├── sync-secrets-from-azure.sh      # Sync secrets to .env
├── test-honeypot-attack.sh         # Simulate attacks
├── check-tpot-status.sh            # Check T-Pot health
├── AZURE_SETUP_GUIDE.md            # Complete guide
├── AZURE_QUICK_START.md            # Quick reference
├── AZURE_SECRETS_AND_CLI_GUIDE.md  # CLI commands
└── AZURE_DEPLOYMENT_SUMMARY.md     # This file
```

---

## 🚀 Deployment Options

### OPTION A: One-Command Setup (Recommended)

**Perfect if you want:**
- Everything set up automatically
- T-Pot honeypot ready to test
- Production-ready configuration

**Run this:**
```bash
cd /Users/chasemad/Desktop/mini-xdr
./setup-azure-mini-xdr.sh
```

**Timeline:**
- 5-10 min: Azure resources created
- 15-30 min: T-Pot installation (automatic)
- 2 min: Test attack simulation
- **Total: ~30-40 minutes**

**What happens:**
1. ✅ Creates Azure Resource Group (`mini-xdr-rg`)
2. ✅ Creates Azure Key Vault with your secrets
3. ✅ Generates SSH key (`~/.ssh/mini-xdr-tpot-azure`)
4. ✅ Deploys Ubuntu 22.04 VM (Standard_B2s - 2 vCPU, 4GB RAM)
5. ✅ Configures firewall (restricts admin access to YOUR IP)
6. ✅ Installs T-Pot honeypot on VM
7. ✅ Creates `backend/.env` with Azure secrets
8. ✅ Ready to test!

---

### OPTION B: Key Vault Only (No VM)

**Perfect if you want:**
- Just Azure Key Vault for secret management
- Deploy T-Pot manually later
- Test locally first

**Run these commands:**
```bash
# Create resource group
az group create --name mini-xdr-rg --location eastus

# Create Key Vault
az keyvault create \
  --name mini-xdr-secrets-$(whoami | tr '[:upper:]' '[:lower:]') \
  --resource-group mini-xdr-rg \
  --location eastus \
  --enable-rbac-authorization false

# Grant yourself access
az keyvault set-policy \
  --name mini-xdr-secrets-$(whoami | tr '[:upper:]' '[:lower:]') \
  --upn $(az ad signed-in-user show --query userPrincipalName -o tsv) \
  --secret-permissions get list set delete

# Add your secrets
az keyvault secret set \
  --vault-name mini-xdr-secrets-$(whoami | tr '[:upper:]' '[:lower:]') \
  --name mini-xdr-api-key \
  --value "$(openssl rand -hex 32)"

# Sync to .env
./sync-secrets-from-azure.sh
```

---

## 🔐 Security Configuration

### Your Protected Access
- **SSH (port 22):** Only `24.11.0.176` (your IP)
- **T-Pot SSH (port 64295):** Only `24.11.0.176` (your IP)
- **T-Pot Web (port 64297):** Only `24.11.0.176` (your IP)

### Public Honeypot Ports (Intentionally Open)
These ports are OPEN to the internet to attract attackers:
- FTP: 21
- SSH: 22 (honeypot, not real SSH)
- Telnet: 23
- SMTP: 25
- HTTP: 80
- POP3: 110
- IMAP: 143
- HTTPS: 443
- SMB: 445
- RDP: 3389
- MySQL: 3306
- PostgreSQL: 5432

**Why?** These are decoy services that log attacker behavior without compromising security.

---

## 💰 Cost Breakdown

### Monthly Costs
| Resource | Monthly Cost | Notes |
|----------|--------------|-------|
| Azure Key Vault | ~$0.03 | Per 10,000 operations |
| VM (Standard_B2s) | ~$30-50 | 2 vCPU, 4GB RAM |
| Managed Disk | ~$5 | 30GB Premium SSD |
| Public IP | ~$3 | Standard Static IP |
| Outbound Data | ~$0-5 | First 100GB free |
| **TOTAL** | **~$40-65** | Per month |

### Cost Optimization Tips

**Stop VM when not testing:**
```bash
# Stop VM (no compute charges, keep disk)
az vm deallocate --resource-group mini-xdr-rg --name mini-xdr-tpot

# Saves: ~$30-50/month
# Storage charges still apply: ~$8/month
```

**Start VM when needed:**
```bash
az vm start --resource-group mini-xdr-rg --name mini-xdr-tpot
```

**Delete everything (CAUTION):**
```bash
# WARNING: This deletes ALL resources (irreversible)
az group delete --name mini-xdr-rg --yes --no-wait
```

---

## 🧪 Testing Workflow

### Step 1: Deploy (Option A)
```bash
./setup-azure-mini-xdr.sh
```

### Step 2: Wait for T-Pot
```bash
# Check every 5 minutes
./check-tpot-status.sh
```

**Expected output when ready:**
```
✅ VM is reachable
✅ SSH access working
✅ T-Pot web interface responding
✅ T-Pot containers running: 15+
```

### Step 3: Start Mini-XDR

**Terminal 1 (Backend):**
```bash
cd backend
source venv/bin/activate
uvicorn app.main:app --reload
```

**Terminal 2 (Frontend):**
```bash
cd frontend
npm run dev
```

**Terminal 3 (Logs):**
```bash
tail -f backend/backend.log
```

### Step 4: Run Test Attack
```bash
./test-honeypot-attack.sh
```

**This simulates:**
- 10x SSH brute force attempts
- Port scan of 10 common ports
- 5x HTTP probing (including path traversal)
- 3x Telnet connection attempts
- 1x FTP connection attempt

### Step 5: Verify Detection

**Check Mini-XDR Dashboard:**
```
http://localhost:3000
```

**Look for:**
- 🚨 Threat alerts
- 📊 Attack patterns detected
- 🤖 AI recommendations
- 🛡️ Containment actions

**Check T-Pot Dashboard:**
```
https://YOUR_TPOT_IP:64297
```

**Default credentials:**
- Username: `tsec`
- Password: Generated on first boot (check via SSH)

---

## 📊 What Mini-XDR Should Detect

### Expected Alerts

1. **SSH Brute Force Detection**
   - Multiple failed SSH attempts
   - Source IP flagged as suspicious
   - Recommended action: Block IP

2. **Port Scan Detection**
   - Sequential port probing detected
   - Pattern matches reconnaissance activity
   - Recommended action: Monitor or block

3. **HTTP Attack Patterns**
   - Path traversal attempt (`../../../etc/passwd`)
   - Web shell upload attempt (`shell.php`)
   - Recommended action: WAF rules or block

4. **Honeypot Interactions**
   - Telnet login attempts
   - FTP anonymous login
   - Logged as malicious behavior

### Threat Intelligence Enrichment

If you've configured API keys:
- **AbuseIPDB:** Reputation score and abuse reports
- **VirusTotal:** IP/domain reputation
- **OpenAI/XAI:** AI-generated threat analysis

---

## 🛠️ Management Commands

### Azure Key Vault

```bash
# List all secrets
az keyvault secret list \
  --vault-name mini-xdr-secrets-$(whoami | tr '[:upper:]' '[:lower:]') \
  --output table

# Get a secret
az keyvault secret show \
  --vault-name mini-xdr-secrets-$(whoami | tr '[:upper:]' '[:lower:]') \
  --name mini-xdr-api-key \
  --query value -o tsv

# Update a secret
az keyvault secret set \
  --vault-name mini-xdr-secrets-$(whoami | tr '[:upper:]' '[:lower:]') \
  --name openai-api-key \
  --value "sk-YOUR_NEW_KEY"

# Sync to .env
./sync-secrets-from-azure.sh
```

### T-Pot VM

```bash
# SSH to VM
ssh -i ~/.ssh/mini-xdr-tpot-azure azureuser@YOUR_TPOT_IP

# Check T-Pot services
ssh -i ~/.ssh/mini-xdr-tpot-azure azureuser@YOUR_TPOT_IP 'docker ps'

# View honeypot logs
ssh -i ~/.ssh/mini-xdr-tpot-azure azureuser@YOUR_TPOT_IP 'docker logs cowrie'

# Restart T-Pot
ssh -i ~/.ssh/mini-xdr-tpot-azure azureuser@YOUR_TPOT_IP \
  'cd /opt/tpotce && sudo docker-compose restart'

# Stop VM (save costs)
az vm deallocate --resource-group mini-xdr-rg --name mini-xdr-tpot

# Start VM
az vm start --resource-group mini-xdr-rg --name mini-xdr-tpot
```

---

## 🆘 Troubleshooting

### "Setup script fails at VM creation"

**Check quota:**
```bash
az vm list-usage --location eastus --output table | grep StandardBSFamily
```

**Try different region:**
```bash
./setup-azure-mini-xdr.sh --location westus2
```

### "T-Pot web interface not responding"

**Wait longer (installation takes 15-30 min):**
```bash
./check-tpot-status.sh
```

**Check VM logs:**
```bash
ssh -i ~/.ssh/mini-xdr-tpot-azure azureuser@YOUR_TPOT_IP
tail -f /opt/tpotce/install.log
```

### "Mini-XDR not detecting attacks"

**1. Check backend logs:**
```bash
tail -f backend/backend.log
```

**2. Verify .env configuration:**
```bash
cat backend/.env | grep -E "TPOT|HONEYPOT"
```

**3. Test T-Pot connectivity:**
```bash
ssh -i ~/.ssh/mini-xdr-tpot-azure azureuser@YOUR_TPOT_IP
```

**4. Re-run attack:**
```bash
./test-honeypot-attack.sh
```

### "Permission denied on Key Vault"

**Re-grant access:**
```bash
az keyvault set-policy \
  --name mini-xdr-secrets-$(whoami | tr '[:upper:]' '[:lower:]') \
  --upn $(az ad signed-in-user show --query userPrincipalName -o tsv) \
  --secret-permissions get list set delete
```

---

## ✅ Pre-Flight Checklist

Before running setup, verify:

- [ ] Azure CLI installed: `az --version`
- [ ] Logged into Azure: `az account show`
- [ ] Active subscription visible
- [ ] No conflicting resource group named `mini-xdr-rg`
- [ ] Sufficient VM quota in region (Standard_B2s)
- [ ] Have API keys ready (optional, can add later)

---

## 🎯 Ready to Deploy?

**Recommended: Option A (Full Setup)**
```bash
cd /Users/chasemad/Desktop/mini-xdr
./setup-azure-mini-xdr.sh
```

**Then follow the on-screen prompts!**

---

## 📚 Additional Resources

- **Quick Start:** `AZURE_QUICK_START.md`
- **Full Guide:** `AZURE_SETUP_GUIDE.md`
- **CLI Reference:** `AZURE_SECRETS_AND_CLI_GUIDE.md`
- **T-Pot Docs:** https://github.com/telekom-security/tpotce

---

**Questions? Check the troubleshooting sections in `AZURE_SETUP_GUIDE.md`**

🚀 **Let's get started!**

