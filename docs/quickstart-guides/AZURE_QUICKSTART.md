# 🚀 Mini-XDR Azure Deployment Quickstart

**Complete deployment in 4 simple steps (~90 minutes)**

---

## Prerequisites (5 minutes)

### Install Required Tools

```bash
# Azure CLI
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash

# Terraform
brew install terraform  # macOS
# OR
wget https://releases.hashicorp.com/terraform/1.6.0/terraform_1.6.0_linux_amd64.zip
unzip terraform_1.6.0_linux_amd64.zip
sudo mv terraform /usr/local/bin/

# Docker
# Visit: https://docs.docker.com/get-docker/

# kubectl
brew install kubectl  # macOS
# OR
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl
```

### Login to Azure

```bash
az login
az account list
az account set --subscription "YOUR_SUBSCRIPTION_NAME"
```

---

## Step 1: Pre-Deployment Check ✅

Run the validation script to ensure everything is ready:

```bash
cd /Users/chasemad/Desktop/mini-xdr
./ops/azure/scripts/pre-deployment-check.sh
```

**Expected Output:**
```
✨ All critical checks passed! Ready to deploy.
```

---

## Step 2: Configure Deployment (Optional) ⚙️

### Option A: Use Defaults (Recommended for Testing)
Skip this step - defaults work out of the box!

### Option B: Customize Configuration

```bash
cd ops/azure/terraform
cp terraform.tfvars.example terraform.tfvars
nano terraform.tfvars
```

**Key Settings to Customize:**
- `location` - Azure region (default: eastus)
- `windows_endpoint_count` - Number of Windows workstations (default: 3)
- `linux_server_count` - Number of Linux servers (default: 2)
- `enable_mini_corp_network` - Enable test network (default: true)
- `enable_bastion` - Enable Azure Bastion (default: true)

---

## Step 3: Deploy Everything 🚀

### One-Command Deployment

```bash
cd /Users/chasemad/Desktop/mini-xdr
./ops/azure/scripts/deploy-all.sh
```

**This automated script will:**
1. ✅ Deploy Azure infrastructure (45 min)
   - AKS cluster (3 nodes)
   - PostgreSQL database
   - Redis cache
   - Application Gateway with WAF
   - Key Vault for secrets
   - Mini corporate network (6 VMs)

2. ✅ Build and push Docker images (10 min)
   - Backend API image
   - Frontend Next.js image
   - Agent image

3. ✅ Deploy to Kubernetes (5 min)
   - Backend pods (3 replicas)
   - Frontend pods (2 replicas)
   - ConfigMaps and Secrets

4. ✅ Configure mini corporate network (15 min)
   - Promote Domain Controller
   - Create Active Directory domain
   - Configure DNS
   - Create test users

5. ✅ Display access information (5 min)

**Total Time:** ~90 minutes (fully automated)

---

## Step 4: Access & Verify 🔍

### Get Your Access Information

```bash
cd /Users/chasemad/Desktop/mini-xdr/ops/azure/terraform

# Get Application Gateway IP
APPGW_IP=$(terraform output -raw appgw_public_ip)
echo "Access Mini-XDR at: https://$APPGW_IP"

# Get Key Vault name
KEY_VAULT=$(terraform output -raw key_vault_name)

# Get VM admin password
az keyvault secret show --vault-name $KEY_VAULT --name vm-admin-password --query value -o tsv
```

### Access the Application

1. **Web Interface:**
   ```bash
   open https://$APPGW_IP
   ```

2. **View Kubernetes Pods:**
   ```bash
   kubectl get pods -n mini-xdr
   ```

3. **View Backend Logs:**
   ```bash
   kubectl logs -n mini-xdr -l app=mini-xdr-backend -f
   ```

4. **Access VMs via Azure Bastion:**
   - Go to Azure Portal
   - Navigate to your VM
   - Click "Connect" → "Bastion"
   - Use credentials from Key Vault

---

## What You Get 📦

### Application Infrastructure
- ✅ **AKS Cluster**: 3-node Kubernetes cluster (auto-scales 2-5)
- ✅ **Backend API**: FastAPI with 9 AI agents, ML detection (98.73% accuracy)
- ✅ **Frontend**: Next.js 15 + React 19 dashboard
- ✅ **PostgreSQL**: Zone-redundant database (128GB)
- ✅ **Redis**: Standard C1 cache
- ✅ **Application Gateway**: WAF enabled (OWASP 3.2)

### Mini Corporate Network
- ✅ **Domain Controller**: Windows Server 2022 (minicorp.local)
- ✅ **3x Windows 11 Workstations**: Domain-joined
- ✅ **2x Ubuntu 22.04 Servers**: File/app servers
- ✅ **8 Test Users**: Realistic organizational structure
- ✅ **Azure Bastion**: Secure RDP/SSH access

### Security Features
- ✅ **IP Whitelisting**: Your IP only (auto-detected)
- ✅ **Private Networking**: No public IPs on VMs
- ✅ **Key Vault**: All secrets secured
- ✅ **Managed Identities**: No stored credentials
- ✅ **TLS 1.2+**: Enforced everywhere
- ✅ **WAF Protection**: OWASP ruleset

---

## Next Steps 🎯

### 1. Check Deployment Status

```bash
./ops/azure/scripts/deployment-status.sh
```

### 2. Install Agents on VMs

```bash
# Automated (all VMs)
./ops/azure/scripts/deploy-agents-to-corp.sh

# OR Manual (single VM via Bastion):
# Windows: Run install-agent-windows.ps1
# Linux: Run install-agent-linux.sh
```

### 3. Run Attack Simulations

```bash
# All attacks
./ops/azure/attacks/run-all-tests.sh

# Specific attacks
./ops/azure/attacks/kerberos-attacks.sh
./ops/azure/attacks/lateral-movement.sh
./ops/azure/attacks/data-exfiltration.sh
```

### 4. Verify Detections

```bash
# End-to-end validation
./ops/azure/tests/e2e-azure-test.sh

# View incidents in dashboard
open https://$APPGW_IP/incidents
```

---

## Cost Management 💰

### Monthly Cost Estimate
**Total:** $800-1,400/month

**Breakdown:**
- AKS (3 nodes): $250-400
- PostgreSQL: $80-150
- Redis: $15-50
- Application Gateway: $150-200
- 6 Windows/Linux VMs: $260-520
- Other (storage, network): $50-100

### Reduce Costs

**Save 60%** by stopping VMs when not testing:

```bash
# Stop all VMs
az vm deallocate --ids $(az vm list -g mini-xdr-prod-rg --query "[].id" -o tsv)

# Cost after stopping VMs: $500-700/month
```

**Auto-shutdown** is enabled at 10 PM daily (included by default).

### Set Budget Alerts

```bash
az consumption budget create \
  --name mini-xdr-budget \
  --category Cost \
  --amount 1000 \
  --time-grain Monthly \
  --resource-group mini-xdr-prod-rg
```

---

## Monitoring & Troubleshooting 🔧

### View Logs

```bash
# Backend logs
kubectl logs -n mini-xdr -l app=mini-xdr-backend -f

# Frontend logs
kubectl logs -n mini-xdr -l app=mini-xdr-frontend -f

# All events
kubectl get events -n mini-xdr --sort-by='.lastTimestamp'
```

### Check Resource Status

```bash
# Kubernetes
kubectl get all -n mini-xdr

# Azure resources
az resource list -g mini-xdr-prod-rg -o table

# VMs
az vm list -g mini-xdr-prod-rg -d -o table
```

### Common Issues

**1. Terraform errors:**
```bash
cd ops/azure/terraform
terraform refresh
```

**2. kubectl not configured:**
```bash
az aks get-credentials -g mini-xdr-prod-rg -n mini-xdr-aks --overwrite-existing
```

**3. Pods not starting:**
```bash
kubectl describe pod -n mini-xdr <pod-name>
kubectl logs -n mini-xdr <pod-name>
```

**4. VM connection issues:**
- Verify your IP is whitelisted: `curl ifconfig.me`
- Check NSG rules in Azure Portal
- Use Azure Bastion for secure access

---

## Cleanup 🗑️

### Stop Resources (Preserve Data)

```bash
# Stop VMs only
az vm deallocate --ids $(az vm list -g mini-xdr-prod-rg --query "[].id" -o tsv)
```

### Complete Removal

```bash
cd ops/azure/terraform
terraform destroy
```

**⚠️ Warning:** This deletes ALL resources including data. Make backups first!

---

## Detection Capabilities 🎯

**13 Attack Classes (98.73% Accuracy):**
1. Normal (100%)
2. DDoS (99.7%)
3. Reconnaissance (95.5%)
4. Brute Force (99.9%)
5. Web Attack (97.7%)
6. Malware (98.9%)
7. APT (99.7%)
8. **Kerberos Attack (99.98%)**
9. **Lateral Movement (98.9%)**
10. **Credential Theft (99.8%)**
11. **Privilege Escalation (97.7%)**
12. **Data Exfiltration (97.7%)**
13. **Insider Threat (98.0%)**

**9 AI Response Agents:**
- IAM Agent (AD management)
- EDR Agent (endpoint protection)
- DLP Agent (data protection)
- Containment Agent (network isolation)
- Attribution Agent (threat intelligence)
- Forensics Agent (evidence collection)
- Deception Agent (honeypots)
- Hunter Agent (proactive hunting)
- NLP Agent (natural language queries)

---

## Additional Resources 📚

### Documentation
- [Complete Deployment Guide](ops/azure/README.md)
- [Architecture Diagram](ops/azure/ARCHITECTURE_DIAGRAM.md)
- [Attack Simulations](ops/azure/attacks/README.md)
- [Technical Implementation](AZURE_DEPLOYMENT_IMPLEMENTATION.md)

### Quick Commands Reference

```bash
# Pre-deployment check
./ops/azure/scripts/pre-deployment-check.sh

# Full deployment
./ops/azure/scripts/deploy-all.sh

# Check status
./ops/azure/scripts/deployment-status.sh

# Run tests
./ops/azure/tests/e2e-azure-test.sh

# Attack simulations
./ops/azure/attacks/run-all-tests.sh

# View logs
kubectl logs -n mini-xdr -l app=mini-xdr-backend -f
```

---

## Support & Troubleshooting 💬

### Get Help

1. **Check logs:** `kubectl logs -n mini-xdr -l app=mini-xdr-backend`
2. **View status:** `./ops/azure/scripts/deployment-status.sh`
3. **Validate:** `./ops/azure/tests/e2e-azure-test.sh`
4. **Azure Activity:** Check Azure Portal → Activity Log

### Useful Azure CLI Commands

```bash
# View all resources
az resource list -g mini-xdr-prod-rg -o table

# Check costs
az consumption usage list --start-date $(date -d '30 days ago' +%Y-%m-%d)

# View activity log
az monitor activity-log list -g mini-xdr-prod-rg

# Export Terraform outputs
terraform output -json > outputs.json
```

---

## Security Best Practices 🔒

✅ **Implemented by Default:**
- IP whitelisting (your IP only)
- Private networking (no public IPs on VMs)
- Key Vault for all secrets
- Managed identities (no credentials)
- TLS 1.2+ enforced
- WAF with OWASP 3.2 rules
- Auto-shutdown at 10 PM
- Azure AD integration
- Network segmentation

🔐 **Additional Recommendations:**
- Enable MFA on Azure account
- Review Key Vault access policies regularly
- Monitor NSG flow logs
- Enable Azure Security Center
- Set up Log Analytics alerts
- Regular patch management for VMs

---

## Success! 🎉

Your Mini-XDR is now running on Azure with:
- ✅ Production-grade infrastructure
- ✅ Enterprise security
- ✅ Full ML detection capabilities
- ✅ 9 AI response agents
- ✅ Mini corporate network for testing
- ✅ Automated attack simulations

**Access your deployment:**
```bash
open https://$(terraform -chdir=ops/azure/terraform output -raw appgw_public_ip)
```

**Questions?** Review the documentation in `ops/azure/README.md`

**Ready to test?** Run `./ops/azure/attacks/run-all-tests.sh`

---

**🚀 Happy threat hunting!**

