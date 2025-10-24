# Azure Deployment Quick Reference

**One-page reference for Mini-XDR Azure deployment**

---

## ⚡ Quick Deploy

```bash
cd /Users/chasemad/Desktop/mini-xdr
./ops/azure/scripts/deploy-all.sh
```

**Time:** 90 minutes | **Cost:** $800-1,400/month

---

## 📁 Key Files

### Deployment
- `ops/azure/scripts/deploy-all.sh` - **START HERE** (one-command deploy)
- `ops/azure/terraform/*.tf` - Infrastructure code (8 modules)
- `ops/azure/README.md` - Complete deployment guide

### Testing
- `ops/azure/tests/e2e-azure-test.sh` - End-to-end validation (25+ tests)
- `ops/azure/attacks/run-all-tests.sh` - All attack simulations
- `ops/azure/attacks/kerberos-attacks.sh` - Kerberos tests
- `ops/azure/attacks/lateral-movement.sh` - Lateral movement tests
- `ops/azure/attacks/data-exfiltration.sh` - Data theft tests

### Agents
- `ops/azure/scripts/install-agent-windows.ps1` - Windows agent
- `ops/azure/scripts/install-agent-linux.sh` - Linux agent
- `ops/azure/scripts/deploy-agents-to-corp.sh` - Automated deployment

### Documentation
- `AZURE_DEPLOYMENT_READY.md` - **READ THIS FIRST**
- `AZURE_DEPLOYMENT_IMPLEMENTATION.md` - Technical details
- `ops/azure/ARCHITECTURE_DIAGRAM.md` - Visual architecture

---

## 🔧 Essential Commands

### Deploy
```bash
# Full deployment
./ops/azure/scripts/deploy-all.sh

# Infrastructure only
cd ops/azure/terraform && terraform apply

# Application only
./ops/azure/scripts/deploy-mini-xdr-to-aks.sh
```

### Access
```bash
# Get Application Gateway IP
APPGW_IP=$(terraform -chdir=ops/azure/terraform output -raw appgw_public_ip)

# Open dashboard
open https://$APPGW_IP

# Get credentials
KEY_VAULT=$(terraform -chdir=ops/azure/terraform output -raw key_vault_name)
az keyvault secret show --vault-name $KEY_VAULT --name vm-admin-password --query value -o tsv
```

### Test
```bash
# End-to-end validation
./ops/azure/tests/e2e-azure-test.sh

# Attack simulations
./ops/azure/attacks/run-all-tests.sh

# Specific attacks
./ops/azure/attacks/kerberos-attacks.sh
```

### Monitor
```bash
# Kubernetes pods
kubectl get pods -n mini-xdr

# Backend logs
kubectl logs -n mini-xdr -l app=mini-xdr-backend -f

# View VMs
az vm list -g mini-xdr-prod-rg -o table

# View all resources
az resource list -g mini-xdr-prod-rg -o table
```

### Cleanup
```bash
# Destroy everything
cd ops/azure/terraform
terraform destroy

# Stop VMs only (save money)
az vm deallocate --ids $(az vm list -g mini-xdr-prod-rg --query "[].id" -o tsv)
```

---

## 🏗️ What Gets Deployed

### Application Stack
- ✅ AKS cluster (3 nodes, auto-scale 2-5)
- ✅ Backend API (FastAPI, 9 agents, ML models)
- ✅ Frontend (Next.js, React 19, dashboards)
- ✅ PostgreSQL (zone-redundant, 128GB)
- ✅ Redis (Standard C1)
- ✅ App Gateway (WAF enabled)

### Mini Corporate Network
- ✅ Domain Controller (Windows Server 2022)
- ✅ 3 Windows 11 workstations
- ✅ 2 Ubuntu 22.04 servers
- ✅ Active Directory (minicorp.local)
- ✅ 8 test users, 7 groups
- ✅ Azure Bastion access

### Security
- ✅ NSG: Your IP only
- ✅ Private subnets
- ✅ Key Vault secrets
- ✅ Managed identities
- ✅ TLS 1.2+
- ✅ WAF protection

---

## 🎯 Detection Capabilities

**13 Attack Classes (98.73% accuracy):**
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

**9 Response Agents:**
- IAM Agent (AD management)
- EDR Agent (endpoint protection)
- DLP Agent (data protection)
- Containment Agent (network blocking)
- Attribution Agent (threat intel)
- Forensics Agent (evidence collection)
- Deception Agent (honeypots)
- Hunter Agent (proactive hunting)
- NLP Agent (natural language)

---

## 💰 Cost Management

**Monthly:** $800-1,400

**Reduce to $500-700:**
```bash
# Stop VMs when not testing
az vm deallocate --ids $(az vm list -g mini-xdr-prod-rg --query "[].id" -o tsv)
```

**Auto-shutdown:** All VMs stop at 10 PM daily (included)

**Budget alerts:**
```bash
az consumption budget create --name mini-xdr-budget --category Cost --amount 1000
```

---

## 🔍 Troubleshooting

### Terraform Issues
```bash
terraform refresh
terraform force-unlock <LOCK_ID>
rm -rf .terraform && terraform init
```

### Kubernetes Issues
```bash
az aks get-credentials -g mini-xdr-prod-rg -n mini-xdr-aks --overwrite-existing
kubectl get nodes
kubectl describe pod -n mini-xdr <pod-name>
```

### VM Issues
```bash
az vm list -g mini-xdr-prod-rg -o table
az vm start -g mini-xdr-prod-rg -n mini-corp-dc01
```

---

## ✅ Checklist

**Before Deploy:**
- [ ] Azure CLI installed and logged in
- [ ] Terraform installed
- [ ] Docker installed
- [ ] kubectl installed

**After Deploy:**
- [ ] Verify pods running: `kubectl get pods -n mini-xdr`
- [ ] Access dashboard: `https://APPGW_IP`
- [ ] Install agents on VMs
- [ ] Run attack simulations
- [ ] Verify detections in dashboard

---

## 📞 Files Reference

| Task | File |
|------|------|
| Deploy all | `ops/azure/scripts/deploy-all.sh` |
| Infrastructure | `ops/azure/terraform/*.tf` |
| Install Windows agent | `ops/azure/scripts/install-agent-windows.ps1` |
| Install Linux agent | `ops/azure/scripts/install-agent-linux.sh` |
| Test attacks | `ops/azure/attacks/run-all-tests.sh` |
| Validate deployment | `ops/azure/tests/e2e-azure-test.sh` |
| Complete guide | `ops/azure/README.md` |
| Status report | `AZURE_DEPLOYMENT_READY.md` |

---

**Ready to deploy!** 🚀

```bash
./ops/azure/scripts/deploy-all.sh
```

