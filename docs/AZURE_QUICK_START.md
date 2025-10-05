# ⚡ Mini-XDR Azure Quick Start

## 🚀 Get Started in 3 Commands

```bash
cd /Users/chasemad/Desktop/mini-xdr

# 1. Setup Azure (creates Key Vault, deploys T-Pot, configures secrets)
./setup-azure-mini-xdr.sh

# 2. Check T-Pot status (wait until ready)
./check-tpot-status.sh

# 3. Test the system
./test-honeypot-attack.sh
```

---

## 📊 What Gets Created

### Azure Resources
- **Resource Group:** `mini-xdr-rg` (East US)
- **Key Vault:** `mini-xdr-secrets-<your-username>`
- **VM:** `mini-xdr-tpot` (Ubuntu 22.04, Standard_B2s)
- **Network:** Public IP with firewall (restricted to YOUR IP only)

### Local Files
- **Backend .env:** Configured with Azure secrets
- **SSH Key:** `~/.ssh/mini-xdr-tpot-azure`

### Secrets Stored in Azure Key Vault
- `mini-xdr-api-key` (auto-generated)
- `tpot-api-key` (auto-generated)
- `tpot-host` (VM public IP)
- `openai-api-key` (you provide)
- `xai-api-key` (you provide)
- `abuseipdb-api-key` (you provide)
- `virustotal-api-key` (you provide)

---

## 🔐 Your Security Setup

**Your detected IPs:**
- IPv4: `24.11.0.176`
- IPv6: `2601:681:8b01:36b0:1435:f6bd:f64:47fe`

**Access Control:**
- ✅ SSH (port 22) - ONLY your IP
- ✅ T-Pot SSH (port 64295) - ONLY your IP
- ✅ T-Pot Web (port 64297) - ONLY your IP
- 🌐 Honeypot ports (21, 22, 23, 25, 80, etc.) - OPEN to internet (for capturing attacks)

---

## ⏱️ Timeline

| Step | Time | Action |
|------|------|--------|
| 1. Run setup script | 5-10 min | Creates Azure resources |
| 2. T-Pot installation | 15-30 min | Automatic VM setup |
| 3. First test | 2 min | Run attack simulation |
| **Total** | **~30-40 min** | Fully operational |

---

## 🎯 Next Steps After Setup

### 1. Add Your API Keys (Optional but Recommended)

```bash
# Update secrets in Azure Key Vault
az keyvault secret set \
  --vault-name mini-xdr-secrets-$(whoami | tr '[:upper:]' '[:lower:]') \
  --name openai-api-key \
  --value "sk-YOUR_OPENAI_KEY"

# Sync to local .env
./sync-secrets-from-azure.sh
```

### 2. Start Mini-XDR

**Terminal 1 - Backend:**
```bash
cd backend
source venv/bin/activate
uvicorn app.main:app --reload
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm run dev
```

**Terminal 3 - Test:**
```bash
./test-honeypot-attack.sh
```

### 3. Monitor Results

- **Frontend Dashboard:** http://localhost:3000
- **Backend Logs:** `tail -f backend/backend.log`
- **T-Pot Web UI:** https://YOUR_TPOT_IP:64297

---

## 🛠️ Common Tasks

### Check T-Pot Status
```bash
./check-tpot-status.sh
```

### Update Secrets
```bash
# Edit in Azure Key Vault
az keyvault secret set --vault-name VAULT_NAME --name KEY_NAME --value VALUE

# Sync locally
./sync-secrets-from-azure.sh
```

### Run Test Attacks
```bash
./test-honeypot-attack.sh
```

### SSH to T-Pot
```bash
ssh -i ~/.ssh/mini-xdr-tpot-azure azureuser@YOUR_TPOT_IP
```

### Stop/Start VM (Cost Savings)
```bash
# Stop (no compute charges)
az vm deallocate --resource-group mini-xdr-rg --name mini-xdr-tpot

# Start
az vm start --resource-group mini-xdr-rg --name mini-xdr-tpot
```

---

## 📚 Full Documentation

- **Complete Guide:** `AZURE_SETUP_GUIDE.md`
- **CLI Reference:** `AZURE_SECRETS_AND_CLI_GUIDE.md`
- **Troubleshooting:** See "🛠️ Troubleshooting" section in `AZURE_SETUP_GUIDE.md`

---

## 💰 Monthly Costs

Estimated: **$35-60/month**
- Key Vault: ~$0.03
- VM (Standard_B2s): ~$30-50
- Storage: ~$1-5

**Stop VM when not testing to save ~$30-50/month!**

---

## ✅ Verification Checklist

After setup, verify:
- [ ] Azure resources created: `az group show --name mini-xdr-rg`
- [ ] Key Vault accessible: `az keyvault secret list --vault-name VAULT_NAME`
- [ ] T-Pot VM running: `./check-tpot-status.sh`
- [ ] Backend .env configured: `cat backend/.env`
- [ ] Mini-XDR backend starts: `cd backend && uvicorn app.main:app`
- [ ] Mini-XDR frontend starts: `cd frontend && npm run dev`
- [ ] Test attacks detected: `./test-honeypot-attack.sh`

---

## 🆘 Need Help?

**Common Issues:**

1. **"Key Vault not found"**
   - Check name: `az keyvault list --output table`
   - Re-run setup: `./setup-azure-mini-xdr.sh`

2. **"T-Pot not responding"**
   - Still installing? Wait 20-30 min
   - Check status: `./check-tpot-status.sh`
   - SSH and check: `ssh -i ~/.ssh/mini-xdr-tpot-azure azureuser@IP`

3. **"No events detected"**
   - Check backend logs: `tail -f backend/backend.log`
   - Verify TPOT_HOST in .env: `cat backend/.env | grep TPOT`
   - Re-run test: `./test-honeypot-attack.sh`

**Contact:** Check `AZURE_SETUP_GUIDE.md` for detailed troubleshooting

---

🎉 **That's it! You're ready to detect and respond to threats with Mini-XDR on Azure!**

