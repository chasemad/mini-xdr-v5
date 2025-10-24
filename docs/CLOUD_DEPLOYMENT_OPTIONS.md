# Mini-XDR Cloud Deployment Options

**Complete guide to deploying Mini-XDR across different cloud providers**

---

## Table of Contents

1. [Cloud Provider Comparison](#cloud-provider-comparison)
2. [Google Cloud Platform (GCP)](#google-cloud-platform-gcp)
3. [DigitalOcean](#digitalocean)
4. [Linode (Akamai Cloud)](#linode-akamai-cloud)
5. [Oracle Cloud (Free Tier)](#oracle-cloud-free-tier)
6. [Hetzner Cloud](#hetzner-cloud)
7. [Vultr](#vultr)
8. [On-Premises / Self-Hosted](#on-premises--self-hosted)
9. [Multi-Cloud Deployment](#multi-cloud-deployment)
10. [Recommendations by Use Case](#recommendations-by-use-case)

---

## Cloud Provider Comparison

### Feature Matrix

| Provider | Kubernetes | Cost/Month* | Free Tier | Ease of Use | Global Reach | Best For |
|----------|-----------|-------------|-----------|-------------|--------------|----------|
| **AWS** | EKS | $156-250 | ❌ Limited | ⭐⭐⭐ | 🌍🌍🌍 | Enterprise, Scale |
| **Azure** | AKS | $150-240 | ✅ $200 credit | ⭐⭐⭐ | 🌍🌍🌍 | Enterprise, Microsoft stack |
| **GCP** | GKE | $140-220 | ✅ $300 credit | ⭐⭐⭐⭐ | 🌍🌍🌍 | ML/AI, Data analytics |
| **DigitalOcean** | DOKS | $60-120 | ❌ | ⭐⭐⭐⭐⭐ | 🌍🌍 | Startups, Simple deployments |
| **Linode** | LKE | $50-100 | ❌ | ⭐⭐⭐⭐⭐ | 🌍🌍 | Cost-conscious, Developers |
| **Oracle Cloud** | OKE | $0-80 | ✅ Always Free | ⭐⭐⭐ | 🌍🌍 | Free tier, Testing |
| **Hetzner** | N/A** | $30-60 | ❌ | ⭐⭐⭐⭐ | 🌍 (EU) | Budget, EU data residency |
| **Vultr** | VKE | $50-100 | ❌ | ⭐⭐⭐⭐ | 🌍🌍 | Performance, Gaming |
| **On-Prem** | k3s/k8s | Hardware | N/A | ⭐⭐ | Local | Full control, Compliance |

*Estimated for 2-3 nodes, basic setup. **Hetzner requires manual k3s setup

### Cost Breakdown (Monthly USD)

**Budget Option (~$30-60/month):**
- Hetzner: 2x CPX11 ($6 each) = $12 + LB $5 = **$17/month**
- Linode: 2x 4GB ($24 each) = $48 + LB $10 = **$58/month**
- Oracle: Free tier = **$0/month** (with limits)

**Mid-Range (~$60-120/month):**
- DigitalOcean: DOKS ($12) + 2x 2vCPU ($24 each) = **$60/month**
- Vultr: VKE ($10) + 2x 2vCPU ($24 each) = **$58/month**

**Enterprise (~$150-250/month):**
- AWS: EKS + EC2 + RDS = **$156-250/month**
- Azure: AKS + VMs + Database = **$150-240/month**
- GCP: GKE + Compute + Cloud SQL = **$140-220/month**

---

## Google Cloud Platform (GCP)

### Why Choose GCP?

✅ **Pros:**
- Best-in-class ML/AI tools (great for Mini-XDR's ML features)
- $300 free credit for new users (3 months)
- Excellent network performance
- Simple pricing, fewer hidden costs
- GKE Autopilot (serverless Kubernetes)
- BigQuery for log analysis

❌ **Cons:**
- Smaller ecosystem than AWS
- Fewer regions than AWS/Azure
- Learning curve if coming from AWS

### Quick Deployment Guide

#### Prerequisites

```bash
# Install gcloud CLI
curl https://sdk.cloud.google.com | bash
exec -l $SHELL

# Initialize gcloud
gcloud init

# Install kubectl (if not already installed)
gcloud components install kubectl

# Set project
export PROJECT_ID=mini-xdr-prod
gcloud config set project $PROJECT_ID
```

#### 1. Enable Required APIs

```bash
gcloud services enable \
  container.googleapis.com \
  compute.googleapis.com \
  sqladmin.googleapis.com \
  redis.googleapis.com \
  secretmanager.googleapis.com
```

#### 2. Create GKE Cluster

```bash
# Standard GKE cluster
gcloud container clusters create mini-xdr-cluster \
  --region us-central1 \
  --num-nodes 2 \
  --machine-type e2-medium \
  --disk-size 30 \
  --enable-autoscaling \
  --min-nodes 2 \
  --max-nodes 5 \
  --enable-autorepair \
  --enable-autoupgrade \
  --addons HorizontalPodAutoscaling,HttpLoadBalancing,GcePersistentDiskCsiDriver

# OR GKE Autopilot (serverless, recommended)
gcloud container clusters create-auto mini-xdr-autopilot \
  --region us-central1

# Get credentials
gcloud container clusters get-credentials mini-xdr-cluster --region us-central1
```

#### 3. Create Cloud SQL (PostgreSQL)

```bash
gcloud sql instances create mini-xdr-db \
  --database-version=POSTGRES_15 \
  --tier=db-f1-micro \
  --region=us-central1 \
  --root-password=YourSecurePassword123! \
  --backup \
  --backup-start-time=03:00 \
  --enable-bin-log \
  --storage-auto-increase

# Create database
gcloud sql databases create xdrdb --instance=mini-xdr-db

# Get connection name
gcloud sql instances describe mini-xdr-db --format='value(connectionName)'
```

#### 4. Create Memorystore (Redis)

```bash
gcloud redis instances create mini-xdr-redis \
  --size=1 \
  --region=us-central1 \
  --redis-version=redis_7_0 \
  --tier=basic

# Get IP
gcloud redis instances describe mini-xdr-redis --region=us-central1 --format='value(host)'
```

#### 5. Deploy Application

```bash
# Create namespace
kubectl create namespace mini-xdr

# Deploy using same k8s manifests as AWS
# Update image registry to GCR
export GCP_PROJECT_ID=$(gcloud config get-value project)

# Push images to GCR
docker tag mini-xdr-frontend gcr.io/${GCP_PROJECT_ID}/mini-xdr-frontend:latest
docker tag mini-xdr-backend gcr.io/${GCP_PROJECT_ID}/mini-xdr-backend:latest
docker push gcr.io/${GCP_PROJECT_ID}/mini-xdr-frontend:latest
docker push gcr.io/${GCP_PROJECT_ID}/mini-xdr-backend:latest

# Deploy
kubectl apply -f k8s/
```

#### 6. Setup Ingress with Google Cloud Load Balancer

```bash
# Install NGINX Ingress (or use GCP Load Balancer)
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.8.2/deploy/static/provider/cloud/deploy.yaml

# Create Ingress
cat <<EOF | kubectl apply -f -
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: mini-xdr-ingress
  namespace: mini-xdr
  annotations:
    kubernetes.io/ingress.class: "nginx"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
  tls:
  - hosts:
    - mini-xdr.yourdomain.com
    secretName: mini-xdr-tls
  rules:
  - host: mini-xdr.yourdomain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: mini-xdr-frontend-service
            port:
              number: 3000
EOF
```

#### Estimated GCP Costs

- GKE cluster: Free (only pay for nodes)
- 2x e2-medium: $48/month
- Cloud SQL (db-f1-micro): $15/month
- Memorystore (1GB): $30/month
- Load Balancer: $18/month
- Egress: ~$10/month
- **Total: ~$121/month**

**With Autopilot:** ~$80-140/month (pay per pod resource)

---

## DigitalOcean

### Why Choose DigitalOcean?

✅ **Pros:**
- **Simplest interface** of all providers
- Predictable, transparent pricing
- Excellent documentation and tutorials
- Great community support
- Good performance/price ratio
- Managed databases included
- $200 free credit (60 days) via referral

❌ **Cons:**
- Fewer features than AWS/GCP
- Limited regions (12 vs 30+)
- No advanced ML services
- Smaller ecosystem

### Quick Deployment Guide

#### Prerequisites

```bash
# Install doctl (DigitalOcean CLI)
brew install doctl

# Authenticate
doctl auth init

# Install kubectl
brew install kubectl
```

#### 1. Create Kubernetes Cluster (DOKS)

```bash
# Create cluster (takes 4-5 minutes)
doctl kubernetes cluster create mini-xdr-cluster \
  --region nyc1 \
  --version 1.28.2-do.0 \
  --node-pool "name=worker-pool;size=s-2vcpu-4gb;count=2;auto-scale=true;min-nodes=2;max-nodes=5"

# Get kubeconfig
doctl kubernetes cluster kubeconfig save mini-xdr-cluster

# Verify
kubectl get nodes
```

#### 2. Create Managed Database

```bash
# Create PostgreSQL database
doctl databases create mini-xdr-db \
  --engine pg \
  --region nyc1 \
  --size db-s-1vcpu-1gb \
  --version 15

# Create Redis cluster
doctl databases create mini-xdr-redis \
  --engine redis \
  --region nyc1 \
  --size db-s-1vcpu-1gb \
  --version 7

# Get connection details
doctl databases connection mini-xdr-db
doctl databases connection mini-xdr-redis
```

#### 3. Setup Container Registry

```bash
# Create registry
doctl registry create mini-xdr-registry

# Login to registry
doctl registry login

# Tag and push images
docker tag mini-xdr-frontend registry.digitalocean.com/mini-xdr-registry/frontend:latest
docker tag mini-xdr-backend registry.digitalocean.com/mini-xdr-registry/backend:latest
docker push registry.digitalocean.com/mini-xdr-registry/frontend:latest
docker push registry.digitalocean.com/mini-xdr-registry/backend:latest
```

#### 4. Deploy Application

```bash
# Create namespace
kubectl create namespace mini-xdr

# Create secrets
kubectl create secret generic mini-xdr-secrets \
  --from-literal=database-url="postgresql://..." \
  --from-literal=redis-url="redis://..." \
  -n mini-xdr

# Deploy (use k8s manifests, update image paths)
kubectl apply -f k8s/
```

#### 5. Setup Load Balancer

```bash
# Install NGINX Ingress (DigitalOcean auto-provisions LB)
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.8.2/deploy/static/provider/cloud/deploy.yaml

# Get Load Balancer IP
kubectl get svc -n ingress-nginx ingress-nginx-controller

# Create DNS A record pointing to LB IP
```

#### Estimated DigitalOcean Costs

- DOKS: $12/month
- 2x 2vCPU 4GB nodes: $48/month ($24 each)
- PostgreSQL (1GB): $15/month
- Redis (1GB): $15/month
- Load Balancer: $12/month
- Container Registry: $5/month
- **Total: ~$107/month**

**Budget option:**
- 2x 1vCPU 2GB nodes: $24/month ($12 each)
- Total: ~$68/month

---

## Linode (Akamai Cloud)

### Why Choose Linode?

✅ **Pros:**
- **Best price/performance** ratio
- Simple, clean interface
- Excellent documentation
- Strong community
- $100 free credit (60 days)
- Good network performance
- Recently acquired by Akamai (more resources)

❌ **Cons:**
- Fewer managed services than AWS/GCP
- Smaller feature set
- Limited to 11 regions

### Quick Deployment Guide

#### Prerequisites

```bash
# Install Linode CLI
brew install linode-cli

# Configure
linode-cli configure
```

#### 1. Create LKE Cluster

```bash
# Create cluster
linode-cli lke cluster-create \
  --label mini-xdr-cluster \
  --region us-east \
  --k8s_version 1.28 \
  --node_pools.type g6-standard-2 \
  --node_pools.count 2 \
  --node_pools.autoscaler.enabled true \
  --node_pools.autoscaler.min 2 \
  --node_pools.autoscaler.max 5

# Get kubeconfig
linode-cli lke kubeconfig-view CLUSTER_ID --json | jq -r '.[0].kubeconfig' | base64 -d > ~/.kube/linode-config
export KUBECONFIG=~/.kube/linode-config

kubectl get nodes
```

#### 2. Create Database (Manual setup)

```bash
# Create Linode for PostgreSQL
linode-cli linodes create \
  --label mini-xdr-db \
  --region us-east \
  --type g6-nanode-1 \
  --image linode/debian11 \
  --root_pass YourSecurePassword!

# SSH and install PostgreSQL
# (Or use managed database when available)
```

#### 3. Deploy Application

```bash
# Similar to DigitalOcean
kubectl create namespace mini-xdr
kubectl apply -f k8s/

# Install NGINX Ingress
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.8.2/deploy/static/provider/cloud/deploy.yaml

# Linode auto-provisions NodeBalancer
```

#### Estimated Linode Costs

- LKE: Free (only pay for nodes)
- 2x Linode 4GB: $48/month ($24 each)
- NodeBalancer: $10/month
- Block Storage (optional): $10/month
- **Total: ~$68/month**

**Budget option with 2GB nodes: ~$40/month**

---

## Oracle Cloud (Free Tier)

### Why Choose Oracle Cloud?

✅ **Pros:**
- **Always Free tier** (not just trial!)
  - 2x AMD E2.1 Micro instances (1/8 OCPU, 1GB RAM) - FOREVER
  - 4x ARM Ampere A1 cores + 24GB RAM - FOREVER
  - 200GB block storage - FOREVER
  - 10TB egress/month - FOREVER
- Can run entire Mini-XDR on free tier
- Enterprise-grade infrastructure
- Good for learning, testing, personal projects

❌ **Cons:**
- Complex interface
- Less popular (smaller community)
- Free tier can be hard to claim (high demand)
- Limited support on free tier

### Free Tier Deployment

#### Architecture on Free Tier

```
┌─────────────────────────────────────────┐
│  Free Tier Resources (Always Free)      │
│                                         │
│  ┌──────────────────────────────────┐  │
│  │ 4x ARM A1 cores (24GB RAM)       │  │
│  │ - k3s Kubernetes cluster         │  │
│  │ - Mini-XDR Frontend (6GB)        │  │
│  │ - Mini-XDR Backend (6GB)         │  │
│  │ - PostgreSQL (4GB)               │  │
│  │ - Redis (2GB)                    │  │
│  │ - Nginx (1GB)                    │  │
│  └──────────────────────────────────┘  │
│                                         │
│  200GB Block Storage (Free)            │
│  10TB Egress/month (Free)              │
│  Flexible Load Balancer (Free)         │
└─────────────────────────────────────────┘
```

#### Setup Guide

```bash
# 1. Create OCI account and claim free tier

# 2. Create ARM-based compute instance (4 cores, 24GB)
# Via OCI Console or CLI
oci compute instance launch \
  --compartment-id <compartment-ocid> \
  --availability-domain <ad> \
  --shape VM.Standard.A1.Flex \
  --shape-config '{"ocpus":4.0,"memoryInGBs":24.0}' \
  --image-id <ubuntu-22.04-arm-image-id> \
  --subnet-id <subnet-ocid> \
  --assign-public-ip true

# 3. SSH and install k3s
ssh ubuntu@<instance-ip>

curl -sfL https://get.k3s.io | sh -

# 4. Install Mini-XDR
kubectl apply -f k8s/

# 5. Setup port forwarding or Cloudflare Tunnel
# (Free tier Load Balancer has limitations)
```

#### Cost

- **$0/month** if you stay within free tier limits
- Upgrades available if needed

---

## Hetzner Cloud

### Why Choose Hetzner?

✅ **Pros:**
- **Cheapest option** ($3/month per small server)
- Excellent performance
- EU data centers (good for GDPR)
- Simple pricing
- Great for bootstrapped startups

❌ **Cons:**
- No managed Kubernetes (must use k3s)
- EU-only (Germany, Finland)
- Smaller ecosystem
- Limited support

### Deployment on Hetzner

#### Costs (Incredibly Low)

- 2x CPX11 (2vCPU, 2GB): $6/month each = $12/month
- Load Balancer: $5/month
- Volume (100GB): $5/month
- **Total: $22/month**

#### Setup with k3s

```bash
# Install hcloud CLI
brew install hcloud

# Authenticate
hcloud context create mini-xdr

# Create servers
hcloud server create \
  --name k3s-master \
  --type cpx11 \
  --image ubuntu-22.04 \
  --ssh-key my-key

hcloud server create \
  --name k3s-worker-1 \
  --type cpx11 \
  --image ubuntu-22.04 \
  --ssh-key my-key

# Install k3s on master
ssh root@<master-ip>
curl -sfL https://get.k3s.io | sh -s - server \
  --token=my-secret-token

# Join workers
ssh root@<worker-ip>
curl -sfL https://get.k3s.io | K3S_URL=https://<master-ip>:6443 \
  K3S_TOKEN=my-secret-token sh -

# Deploy Mini-XDR
kubectl apply -f k8s/
```

---

## Vultr

Similar to DigitalOcean with competitive pricing:

- VKE (Kubernetes): $10/month
- 2x 2vCPU 4GB: $48/month
- Managed Database: $15/month
- **Total: ~$73/month**

**Pros:** Global presence, good performance, simple
**Cons:** Smaller than DO/Linode

---

## On-Premises / Self-Hosted

### Why Self-Host?

✅ **Pros:**
- Full control
- No monthly cloud costs
- Better for compliance/security requirements
- Great for learning
- Use existing hardware

❌ **Cons:**
- Upfront hardware cost
- Maintenance burden
- No auto-scaling
- You manage security, backups, HA

### Options

#### 1. **k3s** (Lightweight Kubernetes)

Perfect for:
- Home lab
- Edge computing
- Resource-constrained environments
- Raspberry Pi clusters

```bash
# Install k3s (single node)
curl -sfL https://get.k3s.io | sh -

# Multi-node setup
# Master:
curl -sfL https://get.k3s.io | sh -s - server --token=SECRET

# Workers:
curl -sfL https://get.k3s.io | K3S_URL=https://master:6443 K3S_TOKEN=SECRET sh -
```

#### 2. **microk8s** (Canonical's Kubernetes)

```bash
# Install (Ubuntu)
sudo snap install microk8s --classic

# Enable addons
microk8s enable dns dashboard ingress storage

# Deploy
microk8s kubectl apply -f k8s/
```

#### 3. **Kubernetes (full)**

For production on-prem:
- Use kubeadm
- Requires 3+ master nodes for HA
- More complex but production-ready

#### 4. **Docker Compose** (No Kubernetes)

Simplest option for small deployments:

```yaml
# docker-compose.yml
version: '3.8'
services:
  frontend:
    build: ./frontend
    ports:
      - "3000:3000"
  backend:
    build: ./backend
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://postgres:5432/xdrdb
  postgres:
    image: postgres:15
    environment:
      POSTGRES_PASSWORD: password
  redis:
    image: redis:7
```

```bash
docker-compose up -d
```

---

## Multi-Cloud Deployment

### Why Multi-Cloud?

- Avoid vendor lock-in
- Geographic redundancy
- Cost optimization
- Compliance requirements

### Tools for Multi-Cloud

#### 1. **Terraform** (Infrastructure as Code)

```hcl
# main.tf - Deploy to AWS, GCP, Azure simultaneously

# AWS
module "aws_cluster" {
  source = "./modules/aws-eks"
  region = "us-east-1"
}

# GCP
module "gcp_cluster" {
  source = "./modules/gcp-gke"
  region = "us-central1"
}

# Azure
module "azure_cluster" {
  source = "./modules/azure-aks"
  location = "eastus"
}
```

#### 2. **Crossplane** (Kubernetes-native)

Deploy infrastructure using Kubernetes CRDs:

```yaml
apiVersion: eks.aws.upbound.io/v1beta1
kind: Cluster
metadata:
  name: mini-xdr-eks
spec:
  forProvider:
    region: us-east-1
    version: "1.28"
---
apiVersion: container.gcp.upbound.io/v1beta1
kind: Cluster
metadata:
  name: mini-xdr-gke
spec:
  forProvider:
    location: us-central1
```

#### 3. **Rancher** (Multi-cluster Management)

Single pane of glass for managing multiple clusters across clouds.

---

## Recommendations by Use Case

### 🎓 **Learning / Portfolio Projects**
**Best: Oracle Cloud Free Tier**
- Cost: $0/month
- Fully functional setup
- Learn Kubernetes without spending

**Alternative: DigitalOcean** ($60-100/month)
- Simple interface
- Great tutorials
- Easy to understand pricing

---

### 💼 **Job Hunting / Demo for Recruiters**
**Best: DigitalOcean or Linode**
- Cost: $60-80/month
- Reliable uptime
- Professional appearance
- Easy to set up quickly

**Why not free tier?** 
- May have downtimes
- Performance issues
- Looks less professional

---

### 🚀 **Startup / MVP**
**Best: DigitalOcean or GCP (with credits)**
- DigitalOcean: Simplicity, predictable costs
- GCP: $300 credit buys 2-3 months, great ML tools

**Scale path:**
- Start: DigitalOcean ($60/month)
- Growth: Migrate to AWS/GCP ($200+/month)

---

### 🏢 **Enterprise / Production**
**Best: AWS or GCP**
- AWS: Most mature, largest ecosystem
- GCP: Better for ML/AI workloads
- Azure: If already using Microsoft stack

**Multi-cloud:** Use Terraform + AWS (primary) + GCP (DR/analytics)

---

### 💰 **Budget-Conscious**
**Best: Hetzner or Oracle**
- Hetzner: $22/month for good performance
- Oracle: $0/month (free tier)
- Linode: $40/month (budget nodes)

**Trade-off:** Less managed services, more manual work

---

### 🏠 **Home Lab / Learning**
**Best: On-premises with k3s**
- Use old hardware or Raspberry Pi
- Learn full stack (networking, storage, security)
- No monthly costs after hardware

**Hardware recommendation:**
- 3x Raspberry Pi 4 (4GB): ~$180
- Or old laptop/desktop
- Total: One-time $200-500

---

### 🌍 **EU/GDPR Compliance**
**Best: Hetzner or GCP (EU regions)**
- Hetzner: German company, EU data centers
- GCP: Strong EU presence (Frankfurt, Belgium)
- AWS: EU regions available

---

### 🎮 **High Performance / Low Latency**
**Best: Vultr or AWS**
- Vultr: Good for gaming/real-time
- AWS: Edge locations, CloudFront CDN

---

## Quick Decision Tree

```
Do you need it for free?
├─ Yes → Oracle Cloud Free Tier
└─ No
   └─ What's your budget?
      ├─ $0-50/month → Hetzner or Linode
      ├─ $50-150/month → DigitalOcean or Linode
      └─ $150+/month
         └─ What's your priority?
            ├─ Simplicity → DigitalOcean
            ├─ ML/AI features → GCP
            ├─ Enterprise ecosystem → AWS
            └─ Microsoft stack → Azure
```

---

## Migration Guide

### Moving Between Clouds

All use Kubernetes, so migration is relatively easy:

```bash
# 1. Export current configs
kubectl get all -n mini-xdr -o yaml > mini-xdr-backup.yaml

# 2. Export secrets
kubectl get secrets -n mini-xdr -o yaml > secrets-backup.yaml

# 3. Setup new cloud cluster (use guides above)

# 4. Apply to new cluster
kubectl apply -f mini-xdr-backup.yaml
kubectl apply -f secrets-backup.yaml

# 5. Update DNS to point to new cluster

# 6. Verify and decommission old cluster
```

**Database migration:**
- Use `pg_dump` for PostgreSQL
- Use `redis-cli --rdb` for Redis backups

---

## Cost Comparison Summary

| Provider | Monthly Cost | Setup Time | Best For |
|----------|-------------|------------|----------|
| Oracle Free | $0 | 2-3 hours | Learning, testing |
| Hetzner | $22 | 2 hours | Budget, EU |
| Linode | $40-70 | 30 min | Developers, cost-conscious |
| DigitalOcean | $60-110 | 20 min | Startups, simplicity |
| Vultr | $70-120 | 30 min | Performance |
| GCP | $120-220 | 45 min | ML/AI, data analytics |
| AWS | $156-250 | 60 min | Enterprise, scale |
| Azure | $150-240 | 60 min | Microsoft ecosystem |

---

## Next Steps

1. **Choose your provider** based on recommendations above
2. **Follow the specific guide** for that provider
3. **Adapt the k8s manifests** (change image registries, endpoints)
4. **Set up monitoring** (each provider has their own)
5. **Configure backups** (database, configs)
6. **Test thoroughly** before showing to recruiters

---

## Additional Resources

- **Terraform modules**: `terraform/` folder (create multi-cloud IaC)
- **Kubernetes manifests**: `k8s/` folder (cloud-agnostic)
- **Cloud-specific guides**: Individual detailed docs for each provider
- **Cost calculators**:
  - AWS: https://calculator.aws
  - GCP: https://cloud.google.com/products/calculator
  - Azure: https://azure.microsoft.com/en-us/pricing/calculator/

---

**Questions?** Each cloud option has trade-offs. Pick based on:
- Budget
- Use case (demo vs. production)
- Technical requirements (ML, compliance, etc.)
- Learning goals

For **quick demo setup**, I recommend **DigitalOcean** - simple, fast, professional.
For **free learning**, use **Oracle Cloud Free Tier** - fully functional at $0/month.


