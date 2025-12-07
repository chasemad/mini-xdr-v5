# Mini-XDR Azure Production Deployment

Complete infrastructure-as-code for deploying Mini-XDR to Azure with a mini corporate network for testing and validation.

## 🏗️ Architecture Overview

### Infrastructure Components

- **Azure Kubernetes Service (AKS)**: 3-node cluster for Mini-XDR application
- **Azure Container Registry (ACR)**: Private Docker image registry
- **Azure PostgreSQL Flexible Server**: Production database
- **Azure Cache for Redis**: Session management and caching
- **Azure Application Gateway**: WAF-enabled ingress controller
- **Azure Bastion**: Secure RDP/SSH access to VMs
- **Azure Key Vault**: Secrets management
- **Virtual Network**: Isolated network with 5 subnets

### Mini Corporate Network

- **1x Domain Controller**: Windows Server 2022 (Active Directory)
- **3x Windows Endpoints**: Windows 11 Pro workstations
- **2x Linux Servers**: Ubuntu 22.04 LTS file/app servers
- **Network Segmentation**: Isolated corp-network subnet
- **Active Directory**: minicorp.local domain with test users

## 📁 Directory Structure

```
ops/azure/
├── terraform/                 # Infrastructure as Code
│   ├── provider.tf           # Azure provider configuration
│   ├── variables.tf          # Input variables
│   ├── networking.tf         # VNet, subnets, NSGs
│   ├── security.tf           # ACR, Key Vault, managed identities
│   ├── aks.tf                # AKS cluster and App Gateway
│   ├── databases.tf          # PostgreSQL and Redis
│   ├── vms.tf                # Mini corporate network VMs
│   └── outputs.tf            # Output values
├── scripts/                  # Deployment automation
│   ├── deploy-all.sh         # One-command deployment
│   ├── build-and-push-images.sh   # Docker image build/push
│   ├── deploy-mini-xdr-to-aks.sh  # K8s deployment
│   ├── setup-mini-corp-network.sh # Corporate network setup
│   ├── configure-active-directory.ps1  # AD configuration
│   ├── create-ad-structure.ps1    # OUs, users, groups
│   ├── install-agent-windows.ps1  # Windows agent installer
│   └── install-agent-linux.sh     # Linux agent installer
├── attacks/                  # Attack simulations (to be created)
│   ├── kerberos-attacks.sh
│   ├── lateral-movement.sh
│   └── data-exfiltration.sh
└── tests/                    # End-to-end tests (to be created)
    └── e2e-azure-test.sh
```

## 🚀 Quick Start

### Prerequisites

1. **Azure CLI**: [Install](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli)
2. **Terraform**: [Install](https://www.terraform.io/downloads)
3. **Docker**: [Install](https://docs.docker.com/get-docker/)
4. **kubectl**: [Install](https://kubernetes.io/docs/tasks/tools/)

### One-Command Deployment

```bash
# Login to Azure
az login

# Navigate to project root
cd .

# Run complete deployment
./ops/azure/scripts/deploy-all.sh
```

This will:
1. ✅ Deploy all Azure infrastructure with Terraform
2. ✅ Build and push Docker images to ACR
3. ✅ Deploy Mini-XDR application to AKS
4. ✅ Setup mini corporate network with Active Directory
5. ✅ Configure Domain Controller and create test users
6. ✅ Display access information and next steps

**Estimated time:** 45-60 minutes

## 📝 Manual Step-by-Step Deployment

### Step 1: Deploy Infrastructure

```bash
cd ops/azure/terraform

# Initialize Terraform
terraform init

# Review planned changes
terraform plan

# Apply configuration
terraform apply
```

### Step 2: Build and Push Images

```bash
# Get ACR name from Terraform
ACR_NAME=$(terraform output -raw acr_login_server | cut -d'.' -f1)

# Build and push
./ops/azure/scripts/build-and-push-images.sh $ACR_NAME
```

### Step 3: Deploy to AKS

```bash
# Configure kubectl
az aks get-credentials --resource-group mini-xdr-prod-rg --name mini-xdr-aks

# Deploy application
./ops/azure/scripts/deploy-mini-xdr-to-aks.sh
```

### Step 4: Setup Mini Corporate Network

```bash
# Configure Active Directory and endpoints
./ops/azure/scripts/setup-mini-corp-network.sh
```

## 🔐 Security Features

### Network Security

- **NSG Rules**: Restrict access to your IP only
- **Private Subnets**: VMs have no public IPs
- **Azure Bastion**: Secure RDP/SSH without exposing ports
- **Application Gateway WAF**: OWASP 3.2 rule set enabled

### Identity & Access

- **Azure AD Integration**: AKS uses Azure AD for RBAC
- **Managed Identities**: No stored credentials for services
- **Key Vault**: All secrets stored in Azure Key Vault
- **Least Privilege**: RBAC configured for minimal permissions

### Data Protection

- **TLS 1.2+**: All communications encrypted
- **Encryption at Rest**: PostgreSQL and storage encrypted
- **Private Endpoints**: Databases not exposed to internet
- **Auto-Shutdown**: VMs automatically shut down at 10 PM

## 💰 Cost Management

### Monthly Costs (Approximate)

| Resource | Cost |
|----------|------|
| AKS cluster (3 nodes) | $250-400 |
| Azure PostgreSQL | $80-150 |
| Azure Cache for Redis | $15-50 |
| Application Gateway | $150-200 |
| 6 Windows VMs | $200-400 |
| 2 Linux VMs | $60-120 |
| Storage & Networking | $50-100 |
| **Total** | **$800-1,400/month** |

### Cost Reduction Tips

1. **Use B-series VMs**: Burstable instances for workloads
2. **Stop VMs when not testing**: Auto-shutdown enabled by default
3. **Azure Dev/Test Pricing**: Save 40-55% on Windows VMs
4. **Reserved Instances**: Save up to 72% with 1-3 year commitments
5. **Spot VMs**: Use for non-critical workloads (up to 90% savings)

### Cost Monitoring

```bash
# Set up budget alerts
az consumption budget create \
  --name mini-xdr-monthly-budget \
  --category Cost \
  --amount 1000 \
  --time-grain Monthly \
  --resource-group mini-xdr-prod-rg
```

## 🔍 Accessing Resources

### Application

```bash
# Get Application Gateway IP
APPGW_IP=$(terraform -chdir=ops/azure/terraform output -raw appgw_public_ip)

# Access frontend
open https://$APPGW_IP
```

### Kubernetes

```bash
# View pods
kubectl get pods -n mini-xdr

# View logs
kubectl logs -n mini-xdr -l app=mini-xdr-backend -f

# Port forward for debugging
kubectl port-forward -n mini-xdr svc/mini-xdr-backend-service 8000:8000
```

### Virtual Machines

Access VMs through Azure Bastion in the Azure Portal:

1. Navigate to the VM in Azure Portal
2. Click "Connect" → "Bastion"
3. Enter credentials (stored in Key Vault)

Or use Azure CLI:

```bash
# Get VM admin password from Key Vault
KEY_VAULT=$(terraform -chdir=ops/azure/terraform output -raw key_vault_name)
az keyvault secret show --vault-name $KEY_VAULT --name vm-admin-password --query value -o tsv
```

## 🧪 Testing

### Deploy Agents to Mini Corp Network

```bash
# Windows VMs (via PowerShell)
BACKEND_URL="https://$APPGW_IP"
API_KEY=$(az keyvault secret show --vault-name $KEY_VAULT --name mini-xdr-api-key --query value -o tsv)

# For each Windows VM via Bastion:
./ops/azure/scripts/install-agent-windows.ps1 -BackendUrl $BACKEND_URL -ApiKey $API_KEY

# Linux VMs (via SSH through Bastion)
./ops/azure/scripts/install-agent-linux.sh $BACKEND_URL $API_KEY
```

### Run Attack Simulations

```bash
# Kerberos attacks
./ops/azure/attacks/kerberos-attacks.sh

# Lateral movement
./ops/azure/attacks/lateral-movement.sh

# Data exfiltration
./ops/azure/attacks/data-exfiltration.sh
```

### Verify Detection

```bash
# Check backend logs
kubectl logs -n mini-xdr -l app=mini-xdr-backend | grep "Detected"

# Access dashboard
open https://$APPGW_IP

# Check incidents
curl -H "X-API-Key: $API_KEY" https://$APPGW_IP/api/incidents
```

## 🛠️ Maintenance

### Update Images

```bash
# Build new images
VERSION=v1.1 ./ops/azure/scripts/build-and-push-images.sh

# Update deployments
kubectl set image deployment/mini-xdr-backend \
  backend=$ACR_LOGIN_SERVER/mini-xdr-backend:v1.1 \
  -n mini-xdr

kubectl set image deployment/mini-xdr-frontend \
  frontend=$ACR_LOGIN_SERVER/mini-xdr-frontend:v1.1 \
  -n mini-xdr
```

### Database Backup

```bash
# Manual backup
az postgres flexible-server backup list \
  --resource-group mini-xdr-prod-rg \
  --server-name mini-xdr-postgres

# Restore to point in time
az postgres flexible-server restore \
  --resource-group mini-xdr-prod-rg \
  --name mini-xdr-postgres-restored \
  --source-server mini-xdr-postgres \
  --restore-time "2024-01-01T00:00:00Z"
```

### Scale Resources

```bash
# Scale AKS nodes
az aks scale \
  --resource-group mini-xdr-prod-rg \
  --name mini-xdr-aks \
  --node-count 5

# Scale Kubernetes deployments
kubectl scale deployment/mini-xdr-backend --replicas=5 -n mini-xdr
```

## 🗑️ Cleanup

### Destroy All Resources

```bash
cd ops/azure/terraform
terraform destroy
```

**Warning:** This will delete ALL resources including data. Make backups first!

### Stop VMs Only

```bash
# Stop all VMs in resource group
az vm list --resource-group mini-xdr-prod-rg --query "[].name" -o tsv | \
  xargs -I {} az vm deallocate --resource-group mini-xdr-prod-rg --name {}
```

## 📚 Additional Documentation

- [Terraform Configuration](./terraform/README.md)
- [Network Discovery Implementation](../../backend/app/discovery/README.md)
- [Agent Management Guide](../../docs/AGENT_MANAGEMENT.md)
- [Attack Simulation Guide](./attacks/README.md)

## 🐛 Troubleshooting

### Common Issues

**Terraform errors:**
```bash
# Reset state if corrupted
terraform force-unlock <LOCK_ID>

# Refresh state
terraform refresh
```

**AKS authentication issues:**
```bash
# Re-get credentials
az aks get-credentials --resource-group mini-xdr-prod-rg --name mini-xdr-aks --overwrite-existing

# Check connection
kubectl cluster-info
```

**VM connection issues:**
```bash
# Check NSG rules
az network nsg rule list --resource-group mini-xdr-prod-rg --nsg-name mini-xdr-corp-nsg -o table

# Verify your IP is whitelisted
curl ifconfig.me
```

### Support

For issues or questions:
1. Check logs: `kubectl logs -n mini-xdr -l app=mini-xdr-backend`
2. Review Azure Activity Log in the Azure Portal
3. Check Terraform state: `terraform show`

## 📄 License

MIT License - See LICENSE file for details.

---

**Ready to deploy!** Start with `./ops/azure/scripts/deploy-all.sh`

