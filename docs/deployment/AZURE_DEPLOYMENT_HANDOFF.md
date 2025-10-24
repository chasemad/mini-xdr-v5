# 🎯 MINI-XDR AZURE DEPLOYMENT - COMPLETE HANDOFF DOCUMENT

**Date**: October 8, 2025
**Status**: Infrastructure 100% Deployed | Application Deployment Pending
**Location**: East US
**Your IP**: 149.40.58.153 (Whitelisted)

---

## 📋 TABLE OF CONTENTS

1. [Project Overview](#project-overview)
2. [What We're Building](#what-were-building)
3. [Current Status](#current-status)
4. [Infrastructure Deployed](#infrastructure-deployed)
5. [Files & Configuration](#files--configuration)
6. [What's Running Now](#whats-running-now)
7. [What Still Needs Setup](#what-still-needs-setup)
8. [Complete Deployment Steps](#complete-deployment-steps)
9. [Testing & Validation](#testing--validation)
10. [Access & Credentials](#access--credentials)
11. [Troubleshooting](#troubleshooting)
12. [Cost Management](#cost-management)

---

## 🎯 PROJECT OVERVIEW

### What We're Building

**Mini-XDR** - A complete Extended Detection and Response (XDR) platform deployed on Azure that:
- Monitors a simulated corporate network (mini-corp) for security threats
- Detects 13 different attack classes using ML models (98.73% accuracy)
- Uses 9 AI-powered agents for automated incident response
- Provides a web dashboard for security monitoring and threat hunting
- Integrates with T-Pot honeypots for real-world threat intelligence

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     AZURE CLOUD (East US)                   │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │            AKS CLUSTER (mini-xdr-aks)                │  │
│  │  - Frontend (Next.js Dashboard)                      │  │
│  │  - Backend (FastAPI + ML Models)                     │  │
│  │  - PostgreSQL (Database)                             │  │
│  │  - Redis (Cache)                                     │  │
│  └──────────────────────────────────────────────────────┘  │
│                          ▲                                   │
│                          │ monitors                          │
│  ┌──────────────────────┴───────────────────────────────┐  │
│  │         MINI CORPORATE NETWORK                        │  │
│  │  ┌────────────────┐  ┌────────────────┐             │  │
│  │  │ Domain         │  │ Windows 11     │             │  │
│  │  │ Controller     │  │ Endpoint       │             │  │
│  │  │ (DC01)         │  │ (WS01)         │             │  │
│  │  └────────────────┘  └────────────────┘             │  │
│  │  ┌────────────────┐                                  │  │
│  │  │ Linux Server   │                                  │  │
│  │  │ (SRV01)        │                                  │  │
│  │  └────────────────┘                                  │  │
│  │  * All VMs have Mini-XDR agents installed           │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Application Gateway (WAF) - 20.168.241.208          │  │
│  │  ↓ Your Access Point (from 149.40.58.153)           │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## ✅ CURRENT STATUS

### Phase 1: Infrastructure Deployment ✅ **COMPLETE**
- All 33 Azure resources deployed
- AKS cluster running (Kubernetes 1.31)
- Mini corporate network VMs running
- Databases and caching configured
- Security and networking in place

### Phase 2: Application Deployment ⏳ **PENDING**
- Docker images need to be built and pushed to ACR
- Kubernetes manifests need to be deployed to AKS
- Application configuration needs to be completed
- Health checks need to be verified

### Phase 3: Testing & Validation ⏳ **PENDING**
- Attack simulations need to be run
- Detection capabilities need to be verified
- Dashboard access needs to be confirmed
- End-to-end workflow testing

---

## 🏗️ INFRASTRUCTURE DEPLOYED

### Resource Summary (33 Total Resources)

#### 1. **AKS Cluster** ✅
```yaml
Name: mini-xdr-aks
Status: Running
Kubernetes Version: 1.31
Node Count: 1
Node Size: Standard_D2s_v3 (2 vCPU, 8 GB RAM)
Subnet: 10.0.4.0/22 (1,024 IPs)
Features:
  - Azure CNI networking
  - Azure RBAC enabled
  - Key Vault secrets provider
  - Log Analytics integration
  - Container Registry pull access
```

**Location**: `/subscriptions/e5636423-8514-4bdd-bfef-f7ecdb934260/resourceGroups/mini-xdr-prod-rg/providers/Microsoft.ContainerService/managedClusters/mini-xdr-aks`

#### 2. **Container Registry** ✅
```yaml
Name: minixdracr
Login Server: minixdracr.azurecr.io
SKU: Standard
Admin Enabled: Yes
Status: Ready
```

**Purpose**: Stores Docker images for Mini-XDR frontend and backend

#### 3. **Databases** ✅

**PostgreSQL Flexible Server**:
```yaml
Name: mini-xdr-postgres
FQDN: mini-xdr-postgres.postgres.database.azure.com
Version: 14
SKU: GP_Standard_D2s_v3
Storage: 128 GB
Database: minixdr
Status: Ready
Zone: 1
Private Access: Yes (via VNet integration)
```

**Redis Cache**:
```yaml
Name: mini-xdr-redis
Hostname: mini-xdr-redis.redis.cache.windows.net
Version: 6.0
SKU: Standard C1
TLS: 1.2+ enforced
Status: Running
```

#### 4. **Networking** ✅

**Virtual Network**:
```yaml
Name: mini-xdr-vnet
Address Space: 10.0.0.0/16
Location: East US

Subnets:
  - aks-subnet: 10.0.4.0/22 (AKS pods & nodes)
  - services-subnet: 10.0.2.0/24 (PostgreSQL)
  - appgw-subnet: 10.0.3.0/24 (Application Gateway)
  - corp-network-subnet: 10.0.10.0/24 (Mini-corp VMs)
  - agents-subnet: 10.0.20.0/24 (Future agent VMs)
  - AzureBastionSubnet: 10.0.255.0/24 (Bastion)
```

**Application Gateway**:
```yaml
Name: mini-xdr-appgw
Public IP: 20.168.241.208
SKU: WAF_v2 (2 instances)
Features:
  - WAF enabled (Prevention mode)
  - OWASP 3.2 rule set
  - TLS 1.2+ enforced
  - Backend pool: AKS cluster
Status: Running
```

**Azure Bastion**:
```yaml
Name: mini-xdr-bastion
Public IP: 20.51.154.66
Purpose: Secure RDP/SSH to VMs (no public IPs on VMs)
Status: Running
```

**Network Security Groups**:
- `mini-xdr-aks-nsg` - Protects AKS subnet
- `mini-xdr-appgw-nsg` - Protects Application Gateway
- `mini-xdr-corp-nsg` - Protects mini-corp network

**IP Whitelisting**: Only your IP (149.40.58.153) can access the Application Gateway

#### 5. **Mini Corporate Network VMs** ✅

**Domain Controller**:
```yaml
Name: mini-corp-dc01
Size: Standard_D2s_v3 (2 vCPU, 8 GB RAM)
OS: Windows Server 2022 Datacenter
Private IP: 10.0.10.10
Role: Active Directory Domain Services
Extension: configure-ad-ds (Succeeded)
Status: VM running
Auto-shutdown: 10:00 PM EST daily
```

**Linux Server**:
```yaml
Name: mini-corp-srv01
Size: Standard_B2s (2 vCPU, 4 GB RAM)
OS: Ubuntu Server 22.04 LTS
Private IP: 10.0.10.5
Status: VM running
Auto-shutdown: 10:00 PM EST daily
```

**Windows Endpoint**:
```yaml
Name: mini-corp-ws01
Size: Standard_B2s (2 vCPU, 4 GB RAM)
OS: Windows 11 Enterprise (22H2)
Private IP: 10.0.10.6
Status: VM running
Auto-shutdown: 10:00 PM EST daily
```

**Note**: We originally had 2 Linux servers but reduced to 1 to stay within the 10 vCPU quota limit.

#### 6. **Key Vault** ✅
```yaml
Name: mini-xdr-kv-f6hbfb
URI: https://mini-xdr-kv-f6hbfb.vault.azure.net/
SKU: Standard
Soft Delete: 7 days
Network Access: Your IP (149.40.58.153) whitelisted

Secrets Stored (5):
  1. vm-admin-password - Admin password for all VMs
  2. dc-restore-mode-password - AD restore mode password
  3. postgres-admin-password - PostgreSQL admin password
  4. postgres-connection-string - Full PostgreSQL connection string
  5. redis-connection-string - Full Redis connection string
```

#### 7. **Monitoring & Logging** ✅
```yaml
Log Analytics Workspace:
  Name: mini-xdr-law
  ID: d9160899-1ff1-4341-adb1-a1ac874622d9
  Retention: 30 days
  Connected Services:
    - AKS cluster (Container Insights)
    - All VMs (diagnostic logs)
```

---

## 📁 FILES & CONFIGURATION

### Terraform Infrastructure Code

**Location**: `/Users/chasemad/Desktop/mini-xdr/ops/azure/terraform/`

#### Main Configuration Files:

1. **`provider.tf`** - Azure provider configuration
   ```hcl
   - Provider version: hashicorp/azurerm ~> 4.0
   - Subscription ID: e5636423-8514-4bdd-bfef-f7ecdb934260
   ```

2. **`variables.tf`** - Variable definitions ⚠️ **MODIFIED**
   ```hcl
   Key Changes Made:
   - aks_subnet_prefix: "10.0.4.0/22" (was 10.0.1.0/22)
   - linux_server_count: 1 (was 2)
   ```

3. **`networking.tf`** - VNet, subnets, NSGs, Application Gateway
   - Defines 6 subnets with proper CIDR allocation
   - NSG rules for AKS, AppGW, and Corp network
   - Application Gateway with WAF configuration

4. **`aks.tf`** - AKS cluster configuration ⚠️ **MODIFIED**
   ```hcl
   Key Changes Made:
   - upgrade_settings.max_surge: "1" (was "33%", then "0")
   - Reason: Azure requires non-zero surge OR unavailable
   ```

5. **`databases.tf`** - PostgreSQL and Redis ⚠️ **MODIFIED**
   ```hcl
   Key Changes Made:
   - Added explicit zone: "1" to PostgreSQL config
   - Reason: Prevent zone modification errors
   ```

6. **`vms.tf`** - Mini corporate network VMs
   - Domain controller with AD-DS extension
   - Windows 11 endpoint
   - Linux server (count controlled by variable)

7. **`security.tf`** - Key Vault, secrets, managed identities
   - Stores all sensitive credentials
   - Managed identities for AKS and AppGW

8. **`outputs.tf`** - Output values
   - All resource IDs, endpoints, and configuration details

#### Terraform State:
```bash
State File: terraform.tfstate (115 KB)
Backup: terraform.tfstate.backup
Resources in State: 49
```

### Deployment Scripts

**Location**: `/Users/chasemad/Desktop/mini-xdr/ops/azure/scripts/`

All scripts are **executable** (`chmod +x`):

1. **`build-and-push-images.sh`** ⏳ **NOT RUN YET**
   - Builds Docker images for frontend and backend
   - Tags with ACR registry name
   - Pushes to Azure Container Registry
   - **Usage**: `./build-and-push-images.sh minixdracr.azurecr.io`

2. **`deploy-mini-xdr-to-aks.sh`** ⏳ **NOT RUN YET**
   - Deploys Kubernetes manifests to AKS
   - Creates namespace, secrets, deployments, services
   - Configures ingress and load balancer
   - **Usage**: `./deploy-mini-xdr-to-aks.sh`

3. **`setup-mini-corp-network.sh`** ⏳ **NOT RUN YET**
   - Configures Active Directory domain
   - Joins Windows endpoint to domain
   - Sets up DNS and DHCP
   - Creates test users and OUs
   - **Usage**: `./setup-mini-corp-network.sh`

4. **`deploy-agents-to-corp.sh`** ⏳ **NOT RUN YET**
   - Installs Mini-XDR agents on all corp VMs
   - Configures agents to report to backend
   - Enables log collection
   - **Usage**: `./deploy-agents-to-corp.sh`

5. **`deployment-status.sh`** ✅ **CAN USE NOW**
   - Shows status of all Azure resources
   - Checks health of deployments
   - **Usage**: `./deployment-status.sh`

6. **`pre-deployment-check.sh`** ✅ **ALREADY RAN**
   - Validates Azure CLI, Terraform, kubectl
   - Checks authentication
   - Verifies quotas

7. **`install-agent-linux.sh`** ⏳ **NOT RUN YET**
   - Linux agent installation script
   - To be run on each Linux VM

### Kubernetes Manifests

**Location**: `/Users/chasemad/Desktop/mini-xdr/ops/k8s/`

Expected structure (these will be used during deployment):
```
ops/k8s/
├── namespace.yaml              - mini-xdr namespace
├── secrets.yaml                - Database & Redis credentials
├── backend-deployment.yaml     - FastAPI backend pods
├── backend-service.yaml        - Backend service
├── frontend-deployment.yaml    - Next.js frontend pods
├── frontend-service.yaml       - Frontend service
├── ingress.yaml                - Application Gateway ingress
└── postgres-init.yaml          - Database initialization job
```

### Application Code

**Backend Location**: `/Users/chasemad/Desktop/mini-xdr/backend/`

Key files:
- `app/main.py` - FastAPI application entry point
- `app/models.py` - Database models
- `app/detect.py` - ML-based threat detection
- `app/agents/*.py` - 9 AI-powered response agents
- `Dockerfile` - Backend container definition
- `requirements.txt` - Python dependencies

**Frontend Location**: `/Users/chasemad/Desktop/mini-xdr/frontend/`

Key files:
- `app/page.tsx` - Main dashboard page
- `app/incidents/page.tsx` - Incidents view
- `app/hunt/page.tsx` - Threat hunting interface
- `Dockerfile` - Frontend container definition
- `package.json` - Node.js dependencies

---

## 🏃 WHAT'S RUNNING NOW

### Azure Resources (All Running)

```bash
# Check status of all resources
az resource list -g mini-xdr-prod-rg -o table

# Expected output: 33 resources, all in "Succeeded" state
```

#### Active Services:

1. **AKS Cluster** - `mini-xdr-aks`
   - 1 node running (Standard_D2s_v3)
   - No pods deployed yet (cluster is empty)
   - Ready to accept deployments

2. **Application Gateway** - `20.168.241.208`
   - Listening on ports 80 and 443
   - Backend pool configured (empty - no AKS pods yet)
   - WAF rules active

3. **PostgreSQL** - `mini-xdr-postgres.postgres.database.azure.com`
   - Database `minixdr` created
   - No tables/schema yet (will be created during app deployment)
   - Listening on port 5432
   - Only accessible from AKS subnet (10.0.4.0/22)

4. **Redis** - `mini-xdr-redis.redis.cache.windows.net`
   - Listening on port 6380 (TLS)
   - Empty cache
   - Ready for session storage and caching

5. **Domain Controller** - `mini-corp-dc01` (10.0.10.10)
   - Windows Server 2022 running
   - Active Directory Domain Services installed
   - Domain NOT configured yet (needs setup script)

6. **Linux Server** - `mini-corp-srv01` (10.0.10.5)
   - Ubuntu 22.04 running
   - No agent installed yet

7. **Windows Endpoint** - `mini-corp-ws01` (10.0.10.6)
   - Windows 11 running
   - Not joined to domain yet
   - No agent installed yet

8. **Bastion** - `20.51.154.66`
   - Ready for secure RDP/SSH connections

9. **Key Vault** - `mini-xdr-kv-f6hbfb`
   - 5 secrets stored and accessible

### Current vCPU Usage

```
Total Quota:  10 vCPUs (East US region)
Currently Used: 8 vCPUs (80%)

Breakdown:
- AKS node (1):         2 vCPUs
- Domain Controller:    2 vCPUs
- Linux Server:         2 vCPUs
- Windows Endpoint:     2 vCPUs

Available:  2 vCPUs (20%)
```

**⚠️ IMPORTANT**: You're at 80% quota utilization. Don't create additional VMs without deallocating existing ones first.

---

## ⏳ WHAT STILL NEEDS SETUP

### Phase 2: Application Deployment

#### Step 1: Configure kubectl Access
```bash
cd /Users/chasemad/Desktop/mini-xdr/ops/azure/terraform

# Get AKS credentials
az aks get-credentials \
  --resource-group mini-xdr-prod-rg \
  --name mini-xdr-aks \
  --overwrite-existing

# Verify connection
kubectl get nodes
# Should show 1 node in "Ready" state
```

#### Step 2: Build and Push Docker Images
```bash
cd /Users/chasemad/Desktop/mini-xdr

# Login to ACR
az acr login --name minixdracr

# Build and push images (this will take 5-10 minutes)
./ops/azure/scripts/build-and-push-images.sh minixdracr.azurecr.io

# Expected output:
# - Backend image: minixdracr.azurecr.io/mini-xdr-backend:latest
# - Frontend image: minixdracr.azurecr.io/mini-xdr-frontend:latest
```

**What this does:**
- Builds backend Docker image from `backend/Dockerfile`
- Builds frontend Docker image from `frontend/Dockerfile`
- Tags images with ACR registry URL
- Pushes both images to Azure Container Registry
- Verifies images were uploaded successfully

#### Step 3: Create Kubernetes Secrets
```bash
# Get secrets from Key Vault
POSTGRES_PASSWORD=$(az keyvault secret show \
  --vault-name mini-xdr-kv-f6hbfb \
  --name postgres-admin-password \
  --query value -o tsv)

REDIS_CONNECTION=$(az keyvault secret show \
  --vault-name mini-xdr-kv-f6hbfb \
  --name redis-connection-string \
  --query value -o tsv)

# Create Kubernetes secret
kubectl create namespace mini-xdr

kubectl create secret generic mini-xdr-secrets \
  --namespace mini-xdr \
  --from-literal=postgres-password=$POSTGRES_PASSWORD \
  --from-literal=redis-connection=$REDIS_CONNECTION \
  --from-literal=postgres-host=mini-xdr-postgres.postgres.database.azure.com \
  --from-literal=postgres-database=minixdr
```

#### Step 4: Deploy Mini-XDR Application to AKS
```bash
cd /Users/chasemad/Desktop/mini-xdr

# Deploy all Kubernetes manifests
./ops/azure/scripts/deploy-mini-xdr-to-aks.sh

# This script will:
# 1. Create mini-xdr namespace (if not exists)
# 2. Deploy backend (FastAPI + ML models)
# 3. Deploy frontend (Next.js dashboard)
# 4. Create services (LoadBalancer type)
# 5. Configure ingress to Application Gateway
# 6. Run database migrations
# 7. Wait for all pods to be ready

# Monitor deployment
kubectl get pods -n mini-xdr -w

# Expected pods (this will take 5-10 minutes):
# - mini-xdr-backend-xxxxx (1/1 Running)
# - mini-xdr-frontend-xxxxx (1/1 Running)
```

#### Step 5: Verify Application is Running
```bash
# Check backend health
kubectl port-forward -n mini-xdr service/mini-xdr-backend 8000:8000 &
curl http://localhost:8000/health
# Expected: {"status":"healthy","database":"connected","redis":"connected"}

# Check frontend
kubectl port-forward -n mini-xdr service/mini-xdr-frontend 3000:3000 &
curl http://localhost:3000
# Expected: HTML response

# Stop port-forwards
pkill -f "kubectl port-forward"
```

### Phase 3: Mini Corporate Network Setup

#### Step 6: Configure Active Directory Domain
```bash
cd /Users/chasemad/Desktop/mini-xdr

# Run domain setup script
./ops/azure/scripts/setup-mini-corp-network.sh

# This script will (via Bastion):
# 1. Promote DC01 to domain controller (minicorp.local)
# 2. Create OUs (Users, Computers, Servers)
# 3. Create test users (john.doe, jane.smith, admin)
# 4. Configure DNS and DHCP
# 5. Join WS01 (Windows endpoint) to domain
# 6. Join SRV01 (Linux) to domain (realm)

# This will take 15-20 minutes
```

**Expected Domain Structure:**
```
Domain: minicorp.local
Domain Controller: DC01 (10.0.10.10)

OUs:
├── MiniCorp Users
│   ├── john.doe (Standard User)
│   ├── jane.smith (Standard User)
│   └── security.admin (IT Admin)
├── MiniCorp Computers
│   └── WS01 (Windows 11 Endpoint)
└── MiniCorp Servers
    └── SRV01 (Linux Server)
```

#### Step 7: Install Mini-XDR Agents on All VMs
```bash
cd /Users/chasemad/Desktop/mini-xdr

# Install agents on all mini-corp VMs
./ops/azure/scripts/deploy-agents-to-corp.sh

# This script will (via Bastion):
# 1. Copy agent binaries to each VM
# 2. Install agent service on Windows VMs (DC01, WS01)
# 3. Install agent daemon on Linux VM (SRV01)
# 4. Configure agents to report to backend (via Application Gateway)
# 5. Start agent services
# 6. Verify agents are connected and sending data

# Agent installation takes ~5 minutes per VM (15 min total)
```

**Agent Configuration:**
```yaml
Backend URL: http://20.168.241.208/api
Collection Interval: 30 seconds
Log Types: Process, Network, File, Registry, Authentication
```

#### Step 8: Verify Agent Connectivity
```bash
# Check that agents are sending data to backend
kubectl logs -n mini-xdr -l app=mini-xdr-backend --tail=100 | grep "agent"

# Expected output:
# "Received heartbeat from agent: mini-corp-dc01"
# "Received heartbeat from agent: mini-corp-srv01"
# "Received heartbeat from agent: mini-corp-ws01"

# Check database for agent registrations
kubectl exec -n mini-xdr -it deployment/mini-xdr-backend -- \
  python -c "
from app.models import Agent
from app.database import SessionLocal
db = SessionLocal()
agents = db.query(Agent).all()
for agent in agents:
    print(f'{agent.hostname}: {agent.last_seen}')
"

# Expected output:
# mini-corp-dc01: 2025-10-08 12:34:56
# mini-corp-srv01: 2025-10-08 12:35:01
# mini-corp-ws01: 2025-10-08 12:35:12
```

### Phase 4: Access & Testing

#### Step 9: Access the Dashboard
```bash
# The Application Gateway should now route traffic to the frontend
# Open your browser and navigate to:
open https://20.168.241.208

# Or with curl:
curl -k https://20.168.241.208

# Expected: Mini-XDR login page
```

**⚠️ SSL Certificate Note**: The Application Gateway is currently using a self-signed certificate, so you'll see a browser warning. Click "Advanced" → "Proceed to site" (this is expected for a test deployment).

#### Step 10: Create Admin User
```bash
# Connect to backend pod
kubectl exec -n mini-xdr -it deployment/mini-xdr-backend -- /bin/bash

# Inside pod, create admin user
python -c "
from app.auth import create_user
create_user(username='admin', password='ChangeMe123!', role='admin')
print('Admin user created successfully')
"

exit
```

**Login Credentials:**
```
URL: https://20.168.241.208
Username: admin
Password: ChangeMe123!
```

#### Step 11: Verify Dashboard Shows Mini-Corp Assets
Once logged in, you should see:

1. **Dashboard (Home Page)**:
   - 3 monitored endpoints (DC01, SRV01, WS01)
   - 0 active incidents (no attacks yet)
   - Agent status: 3/3 connected
   - Real-time metrics graphs

2. **Assets Page**:
   ```
   Hostname          IP           OS                    Status     Last Seen
   mini-corp-dc01    10.0.10.10   Windows Server 2022   Online     Just now
   mini-corp-srv01   10.0.10.5    Ubuntu 22.04          Online     Just now
   mini-corp-ws01    10.0.10.6    Windows 11 Ent        Online     Just now
   ```

3. **Incidents Page**:
   - Should be empty (no attacks detected yet)

---

## 🧪 TESTING & VALIDATION

### Test 1: Run Attack Simulations

**Location**: `/Users/chasemad/Desktop/mini-xdr/ops/azure/attacks/`

#### Available Attack Scripts:

1. **`run-all-tests.sh`** - Runs all attack simulations
2. **`kerberos-attacks.sh`** - Kerberoasting, AS-REP roasting
3. **`lateral-movement.sh`** - Pass-the-hash, pass-the-ticket
4. **`data-exfiltration.sh`** - Large file transfers, suspicious DNS
5. **`privilege-escalation.sh`** - UAC bypass, service exploitation
6. **`ransomware-simulation.sh`** - File encryption behavior

#### Running Attack Simulations:

```bash
cd /Users/chasemad/Desktop/mini-xdr/ops/azure/attacks

# Connect to Windows endpoint via Bastion (run attacks from here)
# In Azure Portal:
# 1. Go to mini-corp-ws01 VM
# 2. Click "Connect" → "Bastion"
# 3. Username: minicorp\admin (or azureadmin)
# 4. Password: (get from Key Vault)

# Once connected to WS01, open PowerShell and run:

# Test 1: Suspicious process creation
Start-Process "cmd.exe" -ArgumentList "/c powershell -encodedCommand <base64>"

# Test 2: Kerberoasting attempt
Get-ADUser -Filter * -Properties ServicePrincipalName |
  Where {$_.ServicePrincipalName -ne $null}

# Test 3: Lateral movement attempt
Invoke-Command -ComputerName DC01 -ScriptBlock {Get-Process}

# Test 4: Port scanning
1..1024 | % { Test-NetConnection -ComputerName DC01 -Port $_ -InformationLevel Quiet }
```

#### Expected Detection Results:

After running attacks, check the dashboard:

1. **Incidents Page** should show new incidents:
   ```
   ID  Type                  Severity  Source        Status      Time
   1   Suspicious Process    High      WS01          New         12:45 PM
   2   Kerberos Attack       Critical  WS01          New         12:46 PM
   3   Lateral Movement      High      WS01→DC01     New         12:47 PM
   4   Port Scan             Medium    WS01          New         12:48 PM
   ```

2. **Each incident should have**:
   - AI-generated analysis
   - Recommended response actions
   - MITRE ATT&CK technique mapping
   - Timeline of events
   - Affected assets

3. **Agents should show activity**:
   ```bash
   # Check backend logs for detections
   kubectl logs -n mini-xdr -l app=mini-xdr-backend --tail=200 | grep "DETECTION"

   # Expected output:
   # [DETECTION] Suspicious process detected on mini-corp-ws01: encoded PowerShell
   # [DETECTION] Kerberos attack detected on mini-corp-ws01: GetUserSPNs query
   # [DETECTION] Lateral movement detected: WS01 → DC01 via WinRM
   # [DETECTION] Port scan detected from mini-corp-ws01
   ```

### Test 2: Verify ML Detection Models

```bash
# Check that ML models are loaded
kubectl exec -n mini-xdr -it deployment/mini-xdr-backend -- \
  python -c "
from app.detect import load_models
models = load_models()
print(f'Loaded {len(models)} ML models')
for name, model in models.items():
    print(f'  - {name}: {type(model).__name__}')
"

# Expected output:
# Loaded 3 ML models
#   - windows_specialist_13class: RandomForestClassifier
#   - comprehensive_classifier: GradientBoostingClassifier
#   - ensemble: EnsembleDetector
```

### Test 3: Verify AI Agent Responses

```bash
# Trigger a test incident and check agent responses
kubectl exec -n mini-xdr -it deployment/mini-xdr-backend -- \
  python -c "
from app.agents.containment_agent import ContainmentAgent
from app.agents.iam_agent import IAMAgent

# Simulate a high-severity incident
incident = {
    'id': 'test-001',
    'type': 'lateral_movement',
    'severity': 'critical',
    'source': 'mini-corp-ws01',
    'destination': 'mini-corp-dc01'
}

# Test containment agent
containment = ContainmentAgent()
actions = containment.analyze(incident)
print('Containment Agent Actions:')
for action in actions:
    print(f'  - {action}')

# Test IAM agent
iam = IAMAgent()
recommendations = iam.analyze(incident)
print('IAM Agent Recommendations:')
for rec in recommendations:
    print(f'  - {rec}')
"

# Expected output:
# Containment Agent Actions:
#   - Isolate host mini-corp-ws01 from network
#   - Block outbound connections from WS01
#   - Suspend user session on WS01
# IAM Agent Recommendations:
#   - Disable compromised user account
#   - Force password reset for affected users
#   - Enable MFA for admin accounts
```

### Test 4: End-to-End Workflow Test

```bash
cd /Users/chasemad/Desktop/mini-xdr/ops/azure/tests

# Run comprehensive end-to-end test
./e2e-azure-test.sh

# This script will:
# 1. Verify all infrastructure is running
# 2. Check agent connectivity (3/3 agents)
# 3. Inject test events into backend
# 4. Verify ML detection (should detect 13/13 attack types)
# 5. Verify AI agent responses (9/9 agents responding)
# 6. Check dashboard accessibility
# 7. Verify database queries work
# 8. Generate test report

# Expected output:
# ✅ Infrastructure: All 33 resources running
# ✅ Agents: 3/3 connected and reporting
# ✅ ML Detection: 13/13 attack types detected
# ✅ AI Agents: 9/9 agents responding correctly
# ✅ Dashboard: Accessible at https://20.168.241.208
# ✅ Database: All queries successful
# ✅ Overall: PASSED (100% success rate)
```

---

## 🔑 ACCESS & CREDENTIALS

### Your IP Address
```
IP: 149.40.58.153
Whitelisted on:
  - Application Gateway NSG
  - Key Vault firewall
Status: Verified and active
```

### Application URLs

```bash
# Main Dashboard
URL: https://20.168.241.208
Method: HTTPS (self-signed cert - browser warning is normal)
Access: Only from your IP (149.40.58.153)

# API Endpoint
URL: https://20.168.241.208/api
Docs: https://20.168.241.208/api/docs (Swagger UI)
```

### Login Credentials

#### Dashboard Login (After User Creation)
```
Username: admin
Password: ChangeMe123!
Role: Administrator (full access)
```

#### VM Access (via Azure Bastion)

**Method 1: Azure Portal**
1. Go to https://portal.azure.com
2. Navigate to the VM (mini-corp-dc01, mini-corp-srv01, or mini-corp-ws01)
3. Click "Connect" → "Bastion"
4. Enter credentials below

**Method 2: Azure CLI**
```bash
# Windows VMs (RDP via Bastion)
az network bastion rdp \
  --name mini-xdr-bastion \
  --resource-group mini-xdr-prod-rg \
  --target-resource-id /subscriptions/e5636423-8514-4bdd-bfef-f7ecdb934260/resourceGroups/mini-xdr-prod-rg/providers/Microsoft.Compute/virtualMachines/mini-corp-dc01

# Linux VMs (SSH via Bastion)
az network bastion ssh \
  --name mini-xdr-bastion \
  --resource-group mini-xdr-prod-rg \
  --target-resource-id /subscriptions/e5636423-8514-4bdd-bfef-f7ecdb934260/resourceGroups/mini-xdr-prod-rg/providers/Microsoft.Compute/virtualMachines/mini-corp-srv01 \
  --auth-type password \
  --username azureadmin
```

**VM Credentials** (all VMs):
```bash
Username: azureadmin
Password: (get from Key Vault)

# Retrieve password:
az keyvault secret show \
  --vault-name mini-xdr-kv-f6hbfb \
  --name vm-admin-password \
  --query value -o tsv
```

#### Domain Admin (After AD Setup)
```
Domain: minicorp.local
Username: minicorp\admin
Password: (same as VM admin password from Key Vault)
```

#### Database Access

**PostgreSQL**:
```bash
Host: mini-xdr-postgres.postgres.database.azure.com
Port: 5432
Database: minixdr
Username: minixdradmin
Password: (get from Key Vault)

# Retrieve password:
az keyvault secret show \
  --vault-name mini-xdr-kv-f6hbfb \
  --name postgres-admin-password \
  --query value -o tsv

# Connection string (full):
az keyvault secret show \
  --vault-name mini-xdr-kv-f6hbfb \
  --name postgres-connection-string \
  --query value -o tsv

# Connect via psql (from AKS pod):
kubectl exec -n mini-xdr -it deployment/mini-xdr-backend -- \
  psql "$(az keyvault secret show --vault-name mini-xdr-kv-f6hbfb --name postgres-connection-string --query value -o tsv)"
```

**Redis**:
```bash
Host: mini-xdr-redis.redis.cache.windows.net
Port: 6380 (TLS)
Password: (from connection string)

# Retrieve connection string:
az keyvault secret show \
  --vault-name mini-xdr-kv-f6hbfb \
  --name redis-connection-string \
  --query value -o tsv
```

#### Container Registry

```bash
Registry: minixdracr.azurecr.io
Admin: Enabled

# Login:
az acr login --name minixdracr

# Or with Docker:
docker login minixdracr.azurecr.io
# Username: minixdracr
# Password: (get from Azure Portal → ACR → Access keys)
```

#### Kubernetes Access

```bash
# Get kubeconfig:
az aks get-credentials \
  --resource-group mini-xdr-prod-rg \
  --name mini-xdr-aks \
  --overwrite-existing

# Verify:
kubectl cluster-info
kubectl get nodes
kubectl get namespaces

# Access dashboard (if deployed):
kubectl port-forward -n mini-xdr service/mini-xdr-frontend 3000:3000
# Then open: http://localhost:3000
```

---

## 🔧 TROUBLESHOOTING

### Issue 1: Dashboard Not Accessible

**Symptom**: Browser shows "Connection refused" or "This site can't be reached"

**Diagnosis**:
```bash
# Check if Application Gateway is running
az network application-gateway show \
  -g mini-xdr-prod-rg \
  -n mini-xdr-appgw \
  --query "operationalState" -o tsv
# Should output: Running

# Check if backend pods are running
kubectl get pods -n mini-xdr
# Should show: backend and frontend pods in "Running" state

# Check Application Gateway backend health
az network application-gateway show-backend-health \
  -g mini-xdr-prod-rg \
  -n mini-xdr-appgw
# All backends should show "Healthy"
```

**Solutions**:
1. If Application Gateway is stopped:
   ```bash
   az network application-gateway start \
     -g mini-xdr-prod-rg \
     -n mini-xdr-appgw
   ```

2. If pods are not running:
   ```bash
   kubectl get pods -n mini-xdr
   kubectl describe pod <pod-name> -n mini-xdr
   kubectl logs <pod-name> -n mini-xdr
   ```

3. If backend health is unhealthy:
   ```bash
   # Check backend service
   kubectl get svc -n mini-xdr
   kubectl get endpoints -n mini-xdr
   ```

4. If your IP changed:
   ```bash
   # Update NSG rule with new IP
   NEW_IP=$(curl -s ifconfig.me)
   az network nsg rule update \
     -g mini-xdr-prod-rg \
     --nsg-name mini-xdr-appgw-nsg \
     --name AllowClientIP \
     --source-address-prefixes $NEW_IP/32
   ```

### Issue 2: Agents Not Connecting

**Symptom**: Dashboard shows 0/3 agents connected

**Diagnosis**:
```bash
# Check if VMs are running
az vm list -g mini-xdr-prod-rg -d --query "[].{Name:name, PowerState:powerState}" -o table

# Check backend logs for agent heartbeats
kubectl logs -n mini-xdr -l app=mini-xdr-backend --tail=100 | grep "agent"

# SSH to a VM and check agent status
az network bastion ssh \
  --name mini-xdr-bastion \
  --resource-group mini-xdr-prod-rg \
  --target-resource-id <vm-resource-id> \
  --auth-type password \
  --username azureadmin

# On Linux VM:
sudo systemctl status mini-xdr-agent

# On Windows VM (PowerShell):
Get-Service -Name "MiniXDRAgent"
```

**Solutions**:
1. If VMs are deallocated (stopped):
   ```bash
   az vm start -g mini-xdr-prod-rg -n mini-corp-srv01
   az vm start -g mini-xdr-prod-rg -n mini-corp-dc01
   az vm start -g mini-xdr-prod-rg -n mini-corp-ws01
   ```

2. If agent service is not running:
   ```bash
   # Linux:
   sudo systemctl start mini-xdr-agent
   sudo systemctl enable mini-xdr-agent

   # Windows:
   Start-Service -Name "MiniXDRAgent"
   Set-Service -Name "MiniXDRAgent" -StartupType Automatic
   ```

3. If agents are not installed:
   ```bash
   cd /Users/chasemad/Desktop/mini-xdr
   ./ops/azure/scripts/deploy-agents-to-corp.sh
   ```

4. If agents can't reach backend:
   ```bash
   # From VM, test connectivity to Application Gateway
   curl http://20.168.241.208/health
   # Should return: {"status":"healthy"}
   ```

### Issue 3: Database Connection Errors

**Symptom**: Backend logs show "Connection to database failed"

**Diagnosis**:
```bash
# Check PostgreSQL status
az postgres flexible-server show \
  -g mini-xdr-prod-rg \
  -n mini-xdr-postgres \
  --query "state" -o tsv
# Should output: Ready

# Check firewall rules
az postgres flexible-server firewall-rule list \
  -g mini-xdr-prod-rg \
  -n mini-xdr-postgres -o table

# Test connection from AKS pod
kubectl exec -n mini-xdr -it deployment/mini-xdr-backend -- \
  pg_isready -h mini-xdr-postgres.postgres.database.azure.com -p 5432
```

**Solutions**:
1. If PostgreSQL is stopped:
   ```bash
   az postgres flexible-server start \
     -g mini-xdr-prod-rg \
     -n mini-xdr-postgres
   ```

2. If firewall rule is missing:
   ```bash
   # Add rule to allow AKS subnet
   az postgres flexible-server firewall-rule create \
     -g mini-xdr-prod-rg \
     -n mini-xdr-postgres \
     --rule-name AllowAKS \
     --start-ip-address 10.0.4.0 \
     --end-ip-address 10.0.7.255
   ```

3. If credentials are wrong:
   ```bash
   # Update Kubernetes secret with correct credentials
   kubectl delete secret mini-xdr-secrets -n mini-xdr

   POSTGRES_PASSWORD=$(az keyvault secret show \
     --vault-name mini-xdr-kv-f6hbfb \
     --name postgres-admin-password \
     --query value -o tsv)

   kubectl create secret generic mini-xdr-secrets \
     --namespace mini-xdr \
     --from-literal=postgres-password=$POSTGRES_PASSWORD \
     --from-literal=postgres-host=mini-xdr-postgres.postgres.database.azure.com

   # Restart backend pods
   kubectl rollout restart deployment/mini-xdr-backend -n mini-xdr
   ```

### Issue 4: Out of vCPU Quota

**Symptom**: "InsufficientVCPUQuota" error when creating resources

**Current Quota**: 10 vCPUs in East US region

**Check Usage**:
```bash
# List all VMs and their sizes
az vm list -g mini-xdr-prod-rg --query "[].{Name:name, Size:hardwareProfile.vmSize}" -o table

# Calculate vCPU usage:
# Standard_D2s_v3 = 2 vCPUs
# Standard_B2s = 2 vCPUs
```

**Solutions**:
1. **Deallocate unused VMs** (stops billing too):
   ```bash
   # Stop a VM (frees vCPUs but keeps disk)
   az vm deallocate -g mini-xdr-prod-rg -n mini-corp-ws01

   # Start it again when needed
   az vm start -g mini-xdr-prod-rg -n mini-corp-ws01
   ```

2. **Request quota increase** (takes 1-2 business days):
   ```bash
   # Go to Azure Portal → Subscriptions → Your Subscription → Usage + quotas
   # Search for "Standard DSv3 Family vCPUs"
   # Click "Request increase"
   # Request at least 14-16 vCPUs for comfort
   ```

3. **Use smaller VM sizes** (if you need more VMs):
   - Change from Standard_D2s_v3 → Standard_B2s (same vCPUs but burstable)
   - Change from Standard_B2s → Standard_B1s (1 vCPU instead of 2)

### Issue 5: Terraform State Locked

**Symptom**: "Error locking state: state blob is already locked"

**Solution**:
```bash
cd /Users/chasemad/Desktop/mini-xdr/ops/azure/terraform

# Force unlock (only if you're sure no other terraform process is running)
terraform force-unlock <LOCK_ID>

# Or remove lock file
rm -rf .terraform/terraform.tfstate
terraform init
```

### Issue 6: Pods in CrashLoopBackOff

**Symptom**: `kubectl get pods` shows pods repeatedly restarting

**Diagnosis**:
```bash
kubectl get pods -n mini-xdr
kubectl describe pod <pod-name> -n mini-xdr
kubectl logs <pod-name> -n mini-xdr --previous
```

**Common Causes & Solutions**:

1. **Missing secrets**:
   ```bash
   kubectl get secrets -n mini-xdr
   # Should show: mini-xdr-secrets

   # If missing, create it:
   kubectl create secret generic mini-xdr-secrets -n mini-xdr \
     --from-literal=postgres-password=$POSTGRES_PASSWORD \
     --from-literal=postgres-host=mini-xdr-postgres.postgres.database.azure.com
   ```

2. **Database not accessible**:
   ```bash
   # Check from pod
   kubectl exec -n mini-xdr -it deployment/mini-xdr-backend -- \
     pg_isready -h mini-xdr-postgres.postgres.database.azure.com
   ```

3. **Insufficient resources**:
   ```bash
   kubectl describe node
   # Check "Allocated resources" section

   # If needed, reduce resource requests in deployment:
   kubectl edit deployment mini-xdr-backend -n mini-xdr
   ```

---

## 💰 COST MANAGEMENT

### Current Monthly Cost Estimate

```
Service                       Tier                    Est. Monthly Cost
─────────────────────────────────────────────────────────────────────
AKS Cluster (1 node)          Standard_D2s_v3         ~$70
Application Gateway           WAF_v2 (2 instances)    ~$350
PostgreSQL                    GP_Standard_D2s_v3      ~$250
Redis Cache                   Standard C1             ~$60
Azure Bastion                 Standard                ~$140
Storage (diagnostics)         Standard LRS            ~$5
Log Analytics                 Pay-as-you-go           ~$20
VMs (3 total):
  - Domain Controller         Standard_D2s_v3         ~$70
  - Linux Server              Standard_B2s            ~$30
  - Windows Endpoint          Standard_B2s            ~$40
Container Registry            Standard                ~$20
Key Vault                     Standard                ~$3
Virtual Network               Standard                ~$5
NSGs, NICs, Disks             Various                 ~$30
─────────────────────────────────────────────────────────────────────
TOTAL ESTIMATED                                       ~$1,093/month
```

**⚠️ IMPORTANT**: This is a TEST deployment. You should **deallocate or delete resources** when not actively testing to avoid charges.

### Cost Optimization Tips

#### 1. **Enable Auto-Shutdown** (Already Configured ✅)
All VMs are configured to auto-shutdown at 10:00 PM EST daily.

```bash
# Verify auto-shutdown schedules
az resource list -g mini-xdr-prod-rg --resource-type "Microsoft.DevTestLab/schedules" -o table
```

#### 2. **Deallocate Resources When Not Testing**

```bash
# Stop all VMs (saves ~$140/month)
az vm deallocate --ids $(az vm list -g mini-xdr-prod-rg --query "[].id" -o tsv)

# Stop AKS cluster (saves ~$70/month)
az aks stop -g mini-xdr-prod-rg -n mini-xdr-aks

# Stop PostgreSQL (saves ~$250/month)
az postgres flexible-server stop -g mini-xdr-prod-rg -n mini-xdr-postgres

# Start everything again when needed:
az vm start --ids $(az vm list -g mini-xdr-prod-rg --query "[].id" -o tsv)
az aks start -g mini-xdr-prod-rg -n mini-xdr-aks
az postgres flexible-server start -g mini-xdr-prod-rg -n mini-xdr-postgres
```

#### 3. **Delete Entire Deployment When Done Testing**

```bash
cd /Users/chasemad/Desktop/mini-xdr/ops/azure/terraform

# THIS WILL DELETE EVERYTHING - MAKE SURE YOU WANT TO DO THIS
terraform destroy -auto-approve

# Or via Azure CLI:
az group delete -g mini-xdr-prod-rg --yes --no-wait
```

#### 4. **Set Up Budget Alerts**

```bash
# Create a budget alert at $500/month
az consumption budget create \
  --budget-name mini-xdr-budget \
  --category Cost \
  --amount 500 \
  --time-grain Monthly \
  --time-period start=2025-10-01 end=2026-10-01 \
  --resource-group mini-xdr-prod-rg \
  --notifications \
    Actual-GreaterThan-80-Percent="{thresholdType:Actual,threshold:80,contactEmails:[your-email@example.com]}" \
    Forecasted-GreaterThan-100-Percent="{thresholdType:Forecasted,threshold:100,contactEmails:[your-email@example.com]}"
```

#### 5. **Monitor Costs Daily**

```bash
# Check current month costs
az consumption usage list \
  --start-date $(date -d '30 days ago' +%Y-%m-%d) \
  --end-date $(date +%Y-%m-%d) \
  --query "[].{Service:instanceName, Cost:pretaxCost, Currency:currency}" \
  -o table

# Or use Azure Cost Management in portal:
# https://portal.azure.com → Cost Management + Billing → Cost analysis
```

---

## 📝 QUICK COMMAND REFERENCE

### Essential Commands

```bash
# === TERRAFORM ===
cd /Users/chasemad/Desktop/mini-xdr/ops/azure/terraform
terraform plan                    # Preview changes
terraform apply                   # Apply changes
terraform destroy                 # Delete everything
terraform output                  # Show output values
terraform state list              # List all resources

# === KUBERNETES ===
az aks get-credentials --resource-group mini-xdr-prod-rg --name mini-xdr-aks
kubectl get nodes                 # List AKS nodes
kubectl get pods -n mini-xdr      # List Mini-XDR pods
kubectl get svc -n mini-xdr       # List services
kubectl logs -n mini-xdr <pod>    # View pod logs
kubectl describe pod <pod> -n mini-xdr  # Pod details
kubectl exec -it <pod> -n mini-xdr -- /bin/bash  # Shell into pod

# === AZURE CLI ===
az account show                   # Show current subscription
az resource list -g mini-xdr-prod-rg -o table  # List all resources
az vm list -g mini-xdr-prod-rg -d -o table     # List VMs with status
az aks show -g mini-xdr-prod-rg -n mini-xdr-aks  # AKS details
az postgres flexible-server show -g mini-xdr-prod-rg -n mini-xdr-postgres  # PostgreSQL details

# === KEY VAULT ===
az keyvault secret list --vault-name mini-xdr-kv-f6hbfb  # List secrets
az keyvault secret show --vault-name mini-xdr-kv-f6hbfb --name vm-admin-password --query value -o tsv  # Get secret

# === BASTION / VM ACCESS ===
az network bastion rdp \
  --name mini-xdr-bastion \
  --resource-group mini-xdr-prod-rg \
  --target-resource-id <vm-id>

az network bastion ssh \
  --name mini-xdr-bastion \
  --resource-group mini-xdr-prod-rg \
  --target-resource-id <vm-id> \
  --auth-type password \
  --username azureadmin

# === MONITORING ===
kubectl top nodes                 # Node resource usage
kubectl top pods -n mini-xdr      # Pod resource usage
az monitor metrics list --resource <resource-id>  # Azure Monitor metrics

# === LOGS ===
kubectl logs -f -n mini-xdr -l app=mini-xdr-backend  # Follow backend logs
kubectl logs -f -n mini-xdr -l app=mini-xdr-frontend  # Follow frontend logs
az monitor log-analytics query --workspace mini-xdr-law --analytics-query "ContainerLog | limit 100"

# === TROUBLESHOOTING ===
kubectl get events -n mini-xdr --sort-by='.lastTimestamp'  # Recent events
kubectl describe pod <pod> -n mini-xdr                      # Pod issues
az network application-gateway show-backend-health -g mini-xdr-prod-rg -n mini-xdr-appgw  # AppGW health
```

---

## 🎯 SUCCESS CRITERIA

When everything is working correctly, you should be able to:

1. ✅ **Access Dashboard**:
   - Navigate to https://20.168.241.208
   - See login page
   - Login with admin credentials
   - View dashboard showing 3 connected agents

2. ✅ **See Mini-Corp Assets**:
   - Assets page shows: DC01, SRV01, WS01
   - All 3 assets show "Online" status
   - Last seen timestamps update every 30 seconds

3. ✅ **Detect Attacks**:
   - Run attack simulation from WS01
   - See new incident appear within 60 seconds
   - Incident shows ML classification (attack type)
   - AI agents provide recommended actions

4. ✅ **View Threat Intelligence**:
   - Dashboard shows real-time metrics
   - Attack timeline visualized
   - MITRE ATT&CK techniques mapped
   - Network topology diagram shows mini-corp

5. ✅ **Execute Responses**:
   - Click "Execute Response" on an incident
   - See agent actions taken (isolate, block, etc.)
   - Verify action was performed on target VM
   - Status changes to "Contained"

---

## 📞 SUPPORT & RESOURCES

### Documentation
- Mini-XDR Docs: `/Users/chasemad/Desktop/mini-xdr/docs/`
- Azure AKS Docs: https://learn.microsoft.com/en-us/azure/aks/
- Terraform Azure Provider: https://registry.terraform.io/providers/hashicorp/azurerm/latest/docs

### Log Files
```
Terraform Logs: /tmp/terraform-*.log
  - terraform-quota-fixed.log (initial deployment)
  - terraform-fix-apply.log (subnet fix)
  - terraform-final-apply.log (Linux server reduction)
  - terraform-aks-surge1.log (FINAL successful deployment)

Kubernetes Logs:
  kubectl logs -n mini-xdr <pod-name>

Azure Logs:
  Azure Portal → Monitor → Logs
  Query workspace: mini-xdr-law
```

### Git Repository Status
```bash
cd /Users/chasemad/Desktop/mini-xdr
git status

# Current branch: main
# Modified files:
#   - ops/azure/terraform/variables.tf (Linux server count, AKS subnet)
#   - ops/azure/terraform/aks.tf (max_surge setting)
#   - ops/azure/terraform/databases.tf (PostgreSQL zone)

# To commit changes:
git add ops/azure/terraform/
git commit -m "Azure deployment configuration - 10 vCPU quota optimizations"
git push origin main
```

---

## ✅ FINAL CHECKLIST

Before proceeding to application deployment, verify:

- [ ] All 33 Azure resources show "Succeeded" state
- [ ] AKS cluster is running with 1 node
- [ ] All 3 VMs are running and accessible via Bastion
- [ ] Key Vault has all 5 secrets
- [ ] PostgreSQL database is Ready
- [ ] Redis cache is Running
- [ ] Application Gateway is Running
- [ ] Your IP (149.40.58.153) is whitelisted
- [ ] You can retrieve VM password from Key Vault
- [ ] kubectl is configured and can connect to AKS
- [ ] You have ACR credentials and can login

**If all items are checked ✅, proceed with Phase 2: Application Deployment**

---

## 🚀 NEXT SESSION START COMMAND

When you start your next session and want to continue deployment:

```bash
cd /Users/chasemad/Desktop/mini-xdr/ops/azure/terraform

# Verify infrastructure status
terraform output
az resource list -g mini-xdr-prod-rg --query "length(@)"  # Should be 33

# Configure kubectl
az aks get-credentials --resource-group mini-xdr-prod-rg --name mini-xdr-aks --overwrite-existing

# Verify AKS connectivity
kubectl get nodes  # Should show 1 node Ready

# Then proceed to build and deploy application
cd /Users/chasemad/Desktop/mini-xdr
./ops/azure/scripts/build-and-push-images.sh minixdracr.azurecr.io
```

---

**End of Handoff Document**
**Status**: Infrastructure Complete | Ready for Application Deployment
**Last Updated**: October 8, 2025
**Total Resources Deployed**: 33
**Total Cost**: ~$1,093/month (optimize by deallocating when not testing)
