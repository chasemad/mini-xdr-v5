# Mini-XDR AWS Deployment Guide

**Comprehensive Guide to Mini-XDR Production Deployment on Amazon Web Services**

**Last Updated:** October 11, 2025
**AWS Account ID:** 116912495274
**Region:** us-east-1 (N. Virginia)
**Deployment Date:** October 2025

---

## Table of Contents

1. [Deployment Overview](#deployment-overview)
2. [Architecture Diagram](#architecture-diagram)
3. [AWS Infrastructure Components](#aws-infrastructure-components)
4. [Kubernetes Resources](#kubernetes-resources)
5. [Container Images](#container-images)
6. [Network Flow & Security](#network-flow--security)
7. [Files & Configurations](#files--configurations)
8. [Access Instructions](#access-instructions)
9. [Common Operations](#common-operations)
10. [Troubleshooting](#troubleshooting)
11. [Cost Analysis](#cost-analysis)
12. [Backup & Disaster Recovery](#backup--disaster-recovery)

---

## Deployment Overview

### Current Status
**PRODUCTION READY** - All services operational as of October 10, 2025

### Quick Access
- **Application URL:** http://k8s-minixdr-minixdri-dc5fc1df8b-1132128475.us-east-1.elb.amazonaws.com
- **Whitelisted IP:** 24.11.0.176/32 (your current IP)
- **Region:** us-east-1
- **Kubernetes Version:** 1.31
- **Backend Version:** v1.0.1 (AMD64)
- **Frontend Version:** v1.0.1 (AMD64 with CSP fix)

### Services Running
```
Backend:  1/1 pods  ✓ HEALTHY
Frontend: 2/2 pods  ✓ HEALTHY
Database: RDS PostgreSQL 17.4 ✓ RUNNING
Storage:  EFS + EBS ✓ MOUNTED
```

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           INTERNET                                       │
│                        (Your IP: 24.11.0.176)                           │
└───────────────────────────────────┬─────────────────────────────────────┘
                                    │
                                    │ HTTPS/HTTP
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                   Application Load Balancer (ALB)                        │
│  DNS: k8s-minixdr-minixdri-dc5fc1df8b-1132128475.us-east-1...          │
│  Security Group: sg-0e958a76b787f7689 (mini-xdr-alb-sg)                │
│  IP Whitelist: 24.11.0.176/32                                          │
└───────────────┬────────────────────────┬────────────────────────────────┘
                │                        │
                │ Path: /                │ Path: /api, /health
                │ Target: 3000           │ Target: 8000
                ▼                        ▼
┌───────────────────────────┐  ┌──────────────────────────────┐
│  Frontend Service         │  │  Backend Service             │
│  ClusterIP: 172.20.71.88  │  │  ClusterIP: 172.20.158.62    │
│  Port: 3000               │  │  Port: 8000, 9090            │
└─────────┬─────────────────┘  └───────────┬──────────────────┘
          │                                 │
          │ Kubernetes Service              │ Kubernetes Service
          │ mini-xdr-frontend-service       │ mini-xdr-backend-service
          ▼                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       Kubernetes Cluster (EKS)                           │
│                    mini-xdr-cluster (Kubernetes 1.31)                   │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────┐   │
│  │  Node: ip-10-0-11-108.ec2.internal (us-east-1a)                │   │
│  │  Type: t3.medium (2 vCPU, 4GB RAM)                             │   │
│  │  ┌──────────────────────┐  ┌──────────────────────────────┐  │   │
│  │  │ Frontend Pod         │  │ Backend Pod                  │  │   │
│  │  │ IP: 10.0.11.250      │  │ IP: 10.0.11.145              │  │   │
│  │  │ Image: frontend:     │  │ Image: backend:v1.0.1        │  │   │
│  │  │   latest (AMD64)     │  │   (AMD64)                    │  │   │
│  │  └──────────────────────┘  └──────────────────────────────┘  │   │
│  └────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────┐   │
│  │  Node: ip-10-0-13-168.ec2.internal (us-east-1c)                │   │
│  │  Type: t3.medium (2 vCPU, 4GB RAM)                             │   │
│  │  ┌──────────────────────┐                                      │   │
│  │  │ Frontend Pod         │                                      │   │
│  │  │ IP: 10.0.13.232      │                                      │   │
│  │  │ Image: frontend:     │                                      │   │
│  │  │   latest (AMD64)     │                                      │   │
│  │  └──────────────────────┘                                      │   │
│  └────────────────────────────────────────────────────────────────┘   │
└──────────┬───────────────────────────┬───────────────┬────────────────┘
           │                           │               │
           │ NFS (EFS)                 │ RDS Conn      │ ECR Pull
           ▼                           ▼               ▼
┌──────────────────────┐  ┌──────────────────────┐  ┌─────────────────┐
│  EFS (Shared)        │  │  RDS PostgreSQL      │  │  ECR Repos      │
│  fs-0109cfbea9b...   │  │  mini-xdr-postgres   │  │  mini-xdr-*     │
│  Encrypted w/ KMS    │  │  db.t3.micro         │  │  2 repos        │
│  5 GiB (Models)      │  │  PostgreSQL 17.4     │  │                 │
│  Mount Points: 3 AZs │  │  Encrypted           │  │                 │
└──────────────────────┘  └──────────────────────┘  └─────────────────┘
```

### Data Flow
1. **User Request** → Your browser (24.11.0.176)
2. **ALB Ingress** → Security check, IP whitelist validation
3. **Path Routing:**
   - `/` → Frontend Service:3000 → Frontend Pods
   - `/api` → Backend Service:8000 → Backend Pod
4. **Backend Processing:**
   - Database queries → RDS PostgreSQL
   - ML model loading → EFS (shared across pods)
   - Logs → CloudWatch (via EKS)
5. **Response** → ALB → User

---

## AWS Infrastructure Components

### 1. Virtual Private Cloud (VPC)

**VPC ID:** `vpc-0d474acd38d418e98`
**CIDR Block:** 10.0.0.0/16
**Region:** us-east-1

#### Subnets

**Public Subnets** (for ALB):
```
subnet-08e1fac778ab053a7  us-east-1a  10.0.1.0/24
subnet-0f70f92833d4a5b54  us-east-1b  10.0.2.0/24
subnet-05dde5c466b5264c5  us-east-1c  10.0.3.0/24
```

**Private Subnets** (for EKS nodes):
```
subnet-0a0622bf540f3849c  us-east-1a  10.0.11.0/24
subnet-0d116b9d4a8ac6b49  us-east-1b  10.0.12.0/24
subnet-0e69d3bc882f061db  us-east-1c  10.0.13.0/24
```

**Purpose:**
- Public subnets host the Application Load Balancer
- Private subnets host EKS worker nodes
- Multi-AZ deployment for high availability

#### Internet Gateway & NAT
- Internet Gateway attached for public subnet internet access
- NAT Gateways in each public subnet for private subnet egress

---

### 2. Elastic Kubernetes Service (EKS)

**Cluster Name:** mini-xdr-cluster
**Kubernetes Version:** 1.31
**Control Plane Endpoint:** https://2782A66117D2F687ED9E7F0A8F89E490.gr7.us-east-1.eks.amazonaws.com

#### Node Group Configuration

**Node Group:** mini-xdr-ng-1
**Instance Type:** t3.medium
**Compute:**
- 2 vCPU per node
- 4 GB RAM per node
- AMD64 (x86_64) architecture

**Nodes:**
```
ip-10-0-11-108.ec2.internal  10.0.11.108  us-east-1a  t3.medium
ip-10-0-13-168.ec2.internal  10.0.13.168  us-east-1c  t3.medium
```

**Scaling:**
- Min nodes: 2
- Max nodes: 4 (via Horizontal Pod Autoscaler)
- Desired: 2

**IAM Roles:**
- Cluster role: AmazonEKSClusterRole
- Node role: AmazonEKSNodeRole
- Additional policies: AmazonEKS_CNI_Policy, AmazonEKSWorkerNodePolicy, AmazonEC2ContainerRegistryReadOnly

#### EKS Add-ons Installed

1. **AWS Load Balancer Controller**
   - Manages ALB creation from Kubernetes Ingress
   - Version: Latest
   - Service Account: aws-load-balancer-controller

2. **EFS CSI Driver**
   - Enables EFS volume mounting in pods
   - Version: Latest
   - Service Account: efs-csi-controller-sa
   - IAM Role: AmazonEKS_EFS_CSI_DriverRole

3. **EBS CSI Driver**
   - Enables EBS volume provisioning
   - Default for single-node persistent volumes

4. **CoreDNS**
   - Kubernetes DNS service
   - Provides service discovery

5. **kube-proxy**
   - Network proxy on each node
   - Implements Kubernetes Service networking

6. **VPC CNI**
   - AWS VPC networking for pods
   - Each pod gets VPC IP address

---

### 3. Elastic Container Registry (ECR)

**Repositories:**

```
Repository: mini-xdr-backend
URI: 116912495274.dkr.ecr.us-east-1.amazonaws.com/mini-xdr-backend
Images:
  - v1.0.1 (AMD64) - Current production
  - amd64 (AMD64) - Previous version
  - latest (ARM64) - Development build (not used)
Size: ~2.1 GB per image

Repository: mini-xdr-frontend
URI: 116912495274.dkr.ecr.us-east-1.amazonaws.com/mini-xdr-frontend
Images:
  - v1.0.1 (AMD64) - Current production (CSP fix)
  - latest (AMD64) - Same as v1.0.1
  - amd64 (AMD64) - Previous tagged AMD64 build
Size: ~1.8 GB per image
```

**Image Scanning:** Enabled
**Encryption:** AES-256
**Lifecycle Policy:** Not configured (manual cleanup)

---

### 4. Elastic File System (EFS)

**File System ID:** fs-0109cfbea9b55373c
**Encryption:** Enabled
**KMS Key ARN:** arn:aws:kms:us-east-1:116912495274:key/431cb645-f4d9-41f6-8d6e-6c26c79c5c04
**Performance Mode:** General Purpose
**Throughput Mode:** Bursting
**Storage Class:** Standard
**Size:** 24 KB (will grow with ML models)

#### Mount Targets

```
us-east-1a: subnet-0a0622bf540f3849c  IP: 10.0.11.102  SG: sg-0bd21dae4146cde52
us-east-1b: subnet-0d116b9d4a8ac6b49  IP: 10.0.12.111  SG: sg-0bd21dae4146cde52
us-east-1c: subnet-0e69d3bc882f061db  IP: 10.0.13.54   SG: sg-0bd21dae4146cde52
```

**Purpose:**
- Shared storage for ML models across pods
- ReadWriteMany (RWX) access mode
- Automatically syncs model updates to all backend pods

**Mounted At:**
- Backend pods: `/app/models`
- Size: 5 GiB (PVC: mini-xdr-models-pvc)

---

### 5. Relational Database Service (RDS)

**Instance Identifier:** mini-xdr-postgres
**Endpoint:** mini-xdr-postgres.ccnkck4wij3r.us-east-1.rds.amazonaws.com
**Port:** 5432
**Engine:** PostgreSQL 17.4
**Instance Class:** db.t3.micro
**Storage:** 20 GB gp3 (SSD)
**Multi-AZ:** No (single instance for cost)
**Encryption:** Enabled (KMS)
**Backup Retention:** 7 days
**Public Access:** No (VPC internal only)

**Connection String Format:**
```
postgresql://username:password@mini-xdr-postgres.ccnkck4wij3r.us-east-1.rds.amazonaws.com:5432/mini_xdr_db
```

**Database Schema:**
- Incidents table
- Response playbooks
- ML training logs
- Agent logs
- User sessions
- Threat intelligence data

---

### 6. Key Management Service (KMS)

**Key ID:** 431cb645-f4d9-41f6-8d6e-6c26c79c5c04
**Key ARN:** arn:aws:kms:us-east-1:116912495274:key/431cb645-f4d9-41f6-8d6e-6c26c79c5c04
**Key Type:** Symmetric
**Usage:** Encrypt/Decrypt
**Origin:** AWS_KMS
**Rotation:** Enabled (annual)

**Used For:**
- EFS file system encryption
- RDS database encryption
- EKS secrets encryption
- EBS volume encryption

---

### 7. Application Load Balancer (ALB)

**DNS Name:** k8s-minixdr-minixdri-dc5fc1df8b-1132128475.us-east-1.elb.amazonaws.com
**ARN:** arn:aws:elasticloadbalancing:us-east-1:116912495274:loadbalancer/app/k8s-minixdr-minixdri-dc5fc1df8b/fb05f47242292703
**Scheme:** internet-facing
**IP Address Type:** ipv4
**Security Group:** sg-0e958a76b787f7689 (mini-xdr-alb-sg)

#### Listeners
```
Port 80 (HTTP)
  → Default action: Forward to target group
```

#### Target Groups

**Backend Target Group:**
```
Name: k8s-minixdr-minixdrb-407ce4b45d
Protocol: HTTP
Port: 8000
Health Check:
  Path: /health
  Interval: 30s
  Timeout: 5s
  Healthy Threshold: 2
  Unhealthy Threshold: 2
Targets:
  - 10.0.11.145:8000 (backend pod) - HEALTHY
```

**Frontend Target Group:**
```
Name: k8s-minixdr-minixdrf-c279e7e8e8
Protocol: HTTP
Port: 3000
Health Check:
  Path: /
  Interval: 30s
  Timeout: 5s
  Healthy Threshold: 2
  Unhealthy Threshold: 2
Targets:
  - 10.0.11.250:3000 (frontend pod) - HEALTHY
  - 10.0.13.232:3000 (frontend pod) - HEALTHY
```

#### Path-Based Routing

```
Path: /api/*         → Backend Target Group (8000)
Path: /ingest/*      → Backend Target Group (8000)
Path: /incidents/*   → Backend Target Group (8000)
Path: /health        → Backend Target Group (8000)
Path: /              → Frontend Target Group (3000)
```

---

### 8. Security Groups

#### ALB Security Group (sg-0e958a76b787f7689)
**Name:** mini-xdr-alb-sg
**Purpose:** Controls inbound traffic to Application Load Balancer

**Inbound Rules:**
```
Type        Protocol  Port   Source              Description
HTTP        TCP       80     24.11.0.176/32      Your IP only
HTTPS       TCP       443    24.11.0.176/32      Your IP only (if configured)
```

**Outbound Rules:**
```
All traffic to 10.0.0.0/16 (VPC CIDR)
```

#### Node Security Group 1 (sg-0beefcaa22b6dc37e)
**Name:** eksctl-mini-xdr-cluster-nodegroup-mini-xdr-ng-1-remoteAccess
**Purpose:** SSH access and pod communication

**Inbound Rules:**
```
Type        Protocol  Port   Source                Description
SSH         TCP       22     0.0.0.0/0             SSH access (restrict in production)
Custom TCP  TCP       3000   sg-0e958a76b787f7689  ALB → Frontend
Custom TCP  TCP       8000   sg-0e958a76b787f7689  ALB → Backend
All traffic ALL       ALL    sg-059f716b6776b2f6c  Inter-node communication
```

#### Node Security Group 2 (sg-059f716b6776b2f6c)
**Name:** eks-cluster-sg-mini-xdr-cluster-699676227
**Purpose:** EKS control plane and node communication

**Inbound Rules:**
```
Type        Protocol  Port   Source                Description
Custom TCP  TCP       3000   sg-0e958a76b787f7689  ALB → Frontend
Custom TCP  TCP       8000   sg-0e958a76b787f7689  ALB → Backend
All traffic ALL       ALL    sg-0beefcaa22b6dc37e  Inter-node communication
```

#### EFS Security Group (sg-0bd21dae4146cde52)
**Name:** mini-xdr-efs-sg
**Purpose:** Controls access to EFS mount targets

**Inbound Rules:**
```
Type        Protocol  Port   Source                Description
NFS         TCP       2049   sg-0beefcaa22b6dc37e  Nodes → EFS
NFS         TCP       2049   sg-059f716b6776b2f6c  Nodes → EFS
```

---

## Kubernetes Resources

### Namespace
**Name:** mini-xdr
All Mini-XDR resources are deployed in this namespace.

### Deployments

#### Backend Deployment

**Name:** mini-xdr-backend
**File:** ops/k8s/backend-deployment.yaml

```yaml
Replicas: 1 (scaled down due to resource constraints)
Strategy: RollingUpdate
  maxSurge: 1
  maxUnavailable: 0

Container:
  Name: backend
  Image: 116912495274.dkr.ecr.us-east-1.amazonaws.com/mini-xdr-backend:v1.0.1
  Platform: linux/amd64
  Port: 8000

  Command Override:
    sh -c "uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 2"
    (Skips database migrations to avoid conflicts)

  Environment Variables:
    DATABASE_URL:        <from secret>
    ENVIRONMENT:         production
    ML_MODEL_PATH:       /app/models
    OPENAI_API_KEY:      <from secret>
    XAI_API_KEY:         <from secret>
    REDIS_URL:           <from secret>
    ABUSEIPDB_API_KEY:   <from secret>
    VIRUSTOTAL_API_KEY:  <from secret>

  Resources:
    Requests:
      CPU: 500m
      Memory: 768Mi
    Limits:
      CPU: 1000m
      Memory: 1536Mi

  Volume Mounts:
    /app/backend/policies → ConfigMap (mini-xdr-policies)
    /app/data             → PVC (mini-xdr-data-pvc, EBS)
    /app/models           → PVC (mini-xdr-models-pvc, EFS)

  Health Checks:
    Liveness:  HTTP GET :8000/health (delay: 60s, period: 30s)
    Readiness: HTTP GET :8000/health (delay: 30s, period: 10s)
```

**Current Pod:**
```
Pod: mini-xdr-backend-769ffd6d9b-q289l
Node: ip-10-0-11-108.ec2.internal (us-east-1a)
IP: 10.0.11.145
Status: Running (1/1 Ready)
Restart Count: 0
Age: 16m
```

#### Frontend Deployment

**Name:** mini-xdr-frontend
**File:** ops/k8s/frontend-deployment.yaml

```yaml
Replicas: 2
Strategy: RollingUpdate
  maxSurge: 1
  maxUnavailable: 0

Container:
  Name: frontend
  Image: 116912495274.dkr.ecr.us-east-1.amazonaws.com/mini-xdr-frontend:v1.0.1
  Platform: linux/amd64
  Port: 3000

  Environment Variables:
    NEXT_PUBLIC_API_URL: http://mini-xdr-backend-service:8000

  Resources:
    Requests:
      CPU: 200m
      Memory: 512Mi
    Limits:
      CPU: 500m
      Memory: 1024Mi

  Health Checks:
    Liveness:  HTTP GET :3000/ (delay: 30s, period: 30s)
    Readiness: HTTP GET :3000/ (delay: 10s, period: 10s)
```

**Current Pods:**
```
Pod: mini-xdr-frontend-7d7945d4f7-2tcrk
Node: ip-10-0-11-108.ec2.internal (us-east-1a)
IP: 10.0.11.250
Status: Running (1/1 Ready)

Pod: mini-xdr-frontend-7d7945d4f7-8slhh
Node: ip-10-0-13-168.ec2.internal (us-east-1c)
IP: 10.0.13.232
Status: Running (1/1 Ready)
```

---

### Services

#### Backend ClusterIP Service

**Name:** mini-xdr-backend-service
**File:** ops/k8s/backend-deployment.yaml

```yaml
Type: ClusterIP
Cluster IP: 172.20.158.62
Ports:
  - name: http
    port: 8000
    targetPort: 8000
  - name: metrics
    port: 9090
    targetPort: 9090
Selector:
  app: mini-xdr-backend
```

#### Backend NodePort Service

**Name:** mini-xdr-backend-nodeport
**File:** ops/k8s/backend-deployment.yaml

```yaml
Type: NodePort
Cluster IP: 172.20.106.25
Port: 8000
NodePort: 30800
Selector:
  app: mini-xdr-backend
```

#### Frontend ClusterIP Service

**Name:** mini-xdr-frontend-service
**File:** ops/k8s/frontend-deployment.yaml

```yaml
Type: ClusterIP
Cluster IP: 172.20.71.88
Port: 3000
Selector:
  app: mini-xdr-frontend
```

#### Frontend NodePort Service

**Name:** mini-xdr-frontend-nodeport
**File:** ops/k8s/frontend-deployment.yaml

```yaml
Type: NodePort
Cluster IP: 172.20.69.14
Port: 3000
NodePort: 30300
Selector:
  app: mini-xdr-frontend
```

---

### Ingress

**Name:** mini-xdr-ingress
**File:** ops/k8s/ingress.yaml

```yaml
Annotations:
  alb.ingress.kubernetes.io/scheme: internet-facing
  alb.ingress.kubernetes.io/target-type: ip
  alb.ingress.kubernetes.io/inbound-cidrs: 24.11.0.176/32
  alb.ingress.kubernetes.io/healthcheck-path: /health
  alb.ingress.kubernetes.io/healthcheck-interval-seconds: "30"
  alb.ingress.kubernetes.io/security-groups: mini-xdr-alb-sg

Rules:
  - http:
      paths:
        - path: /api
          pathType: Prefix
          backend:
            service:
              name: mini-xdr-backend-service
              port: 8000

        - path: /ingest
          pathType: Prefix
          backend:
            service:
              name: mini-xdr-backend-service
              port: 8000

        - path: /incidents
          pathType: Prefix
          backend:
            service:
              name: mini-xdr-backend-service
              port: 8000

        - path: /health
          pathType: Prefix
          backend:
            service:
              name: mini-xdr-backend-service
              port: 8000

        - path: /
          pathType: Prefix
          backend:
            service:
              name: mini-xdr-frontend-service
              port: 3000

Status:
  LoadBalancer:
    Ingress:
      - hostname: k8s-minixdr-minixdri-dc5fc1df8b-1132128475.us-east-1.elb.amazonaws.com
```

---

### ConfigMaps

#### mini-xdr-policies

**File:** ops/k8s/configmap.yaml
**Purpose:** Stores policy configurations for response playbooks

Contains YAML files from `backend/policies/` directory:
- Default playbook policies
- Response action definitions
- Agent configurations

Mounted at: `/app/backend/policies` in backend pods

---

### Secrets

#### mini-xdr-secrets

**File:** ops/k8s/secrets.yaml (not in version control)
**Type:** Opaque

**Keys:**
```
DATABASE_URL            PostgreSQL connection string
OPENAI_API_KEY          OpenAI API key for AI features
XAI_API_KEY             Alternative AI API key
REDIS_URL               Redis connection string (if used)
ABUSEIPDB_API_KEY       AbuseIPDB threat intelligence API
VIRUSTOTAL_API_KEY      VirusTotal malware scanning API
```

**Creation Command:**
```bash
kubectl create secret generic mini-xdr-secrets \
  --from-literal=DATABASE_URL="postgresql://..." \
  --from-literal=OPENAI_API_KEY="sk-..." \
  --from-literal=XAI_API_KEY="..." \
  --from-literal=REDIS_URL="redis://..." \
  --from-literal=ABUSEIPDB_API_KEY="..." \
  --from-literal=VIRUSTOTAL_API_KEY="..." \
  -n mini-xdr
```

---

### Persistent Volumes

#### Data Volume (EBS)

**PVC Name:** mini-xdr-data-pvc
**StorageClass:** mini-xdr-gp3
**Access Mode:** ReadWriteOnce (RWO)
**Size:** 10 GiB
**Volume ID:** vol-xxxxx (AWS EBS gp3)
**Bound To:** Backend pod only

**Purpose:**
- Application data
- Temporary files
- Logs

#### Models Volume (EFS)

**PVC Name:** mini-xdr-models-pvc
**StorageClass:** efs-sc
**Access Mode:** ReadWriteMany (RWX)
**Size:** 5 GiB
**File System:** fs-0109cfbea9b55373c
**Bound To:** All backend pods (shared)

**Purpose:**
- ML model files
- Shared training data
- Model weights and checkpoints

**EFS Access Points:**
- Base Path: /mini-xdr
- POSIX User: 1000:1000
- Permissions: 755

---

### Storage Classes

#### efs-sc (EFS Storage Class)

**File:** ops/k8s/persistent-volumes.yaml

```yaml
Provisioner: efs.csi.aws.com
Parameters:
  provisioningMode: efs-ap
  fileSystemId: fs-0109cfbea9b55373c
  directoryPerms: "755"
  basePath: /mini-xdr
Reclaim Policy: Retain
Volume Binding Mode: Immediate
```

#### mini-xdr-gp3 (EBS Storage Class)

**File:** ops/k8s/persistent-volumes.yaml

```yaml
Provisioner: ebs.csi.aws.com
Parameters:
  type: gp3
  iops: "3000"
  throughput: "125"
  encrypted: "true"
  kmsKeyId: arn:aws:kms:us-east-1:116912495274:key/431cb645...
Reclaim Policy: Retain
Volume Binding Mode: WaitForFirstConsumer
```

---

### Horizontal Pod Autoscalers (HPA)

#### Backend HPA

**Name:** mini-xdr-backend-hpa

```yaml
Min Replicas: 2 (currently overridden to 1 due to resource constraints)
Max Replicas: 4
Metrics:
  - CPU: 70%
  - Memory: 80%
Current: CPU: 1%, Memory: 171% (overprovisioned)
```

**Note:** Backend memory usage is high (171% of request) but within limits. Consider increasing memory request to 1024Mi.

#### Frontend HPA

**Name:** mini-xdr-frontend-hpa

```yaml
Min Replicas: 2
Max Replicas: 4
Metrics:
  - CPU: 70%
  - Memory: 80%
Current: CPU: 1%, Memory: 25%
Status: Healthy
```

---

## Container Images

### Backend Image

**Repository:** 116912495274.dkr.ecr.us-east-1.amazonaws.com/mini-xdr-backend
**Current Tag:** v1.0.1
**Architecture:** linux/amd64
**Base Image:** python:3.11-slim
**Size:** ~2.1 GB

**Dockerfile:** ops/Dockerfile.backend.production

**Key Dependencies:**
```
fastapi==0.116.1
uvicorn[standard]==0.32.1
sqlalchemy==2.0.36
torch==2.8.0
scikit-learn==1.6.0
langchain==0.3.11
email-validator==2.1.0  # Critical fix in v1.0.1
```

**Layers:**
1. Python 3.11 base
2. System dependencies (gcc, postgresql-dev)
3. Python packages (pip install)
4. Application code
5. Policy files
6. Entrypoint configuration

**Build Command:**
```bash
docker buildx build \
  --platform linux/amd64 \
  --file ops/Dockerfile.backend.production \
  --build-arg BUILD_DATE="2025-10-10T05:00:00Z" \
  --build-arg VCS_REF="04aa6fe" \
  --build-arg VERSION="1.0.1" \
  --tag 116912495274.dkr.ecr.us-east-1.amazonaws.com/mini-xdr-backend:v1.0.1 \
  --push \
  .
```

**v1.0.1 Changes:**
- Added email-validator==2.1.0 (fixes Pydantic email validation)
- Fixed database migration conflicts
- Optimized for AMD64 architecture

### Frontend Image

**Repository:** 116912495274.dkr.ecr.us-east-1.amazonaws.com/mini-xdr-frontend
**Current Tag:** v1.0.1
**Architecture:** linux/amd64
**Base Image:** node:20-alpine
**Size:** ~1.8 GB

**Dockerfile:** ops/Dockerfile.frontend.production

**Key Dependencies:**
```
next@15.5.0
react@19.0.0
tailwindcss@3.4.x
```

**Build Command:**
```bash
# Swap .dockerignore to frontend-specific version
if [ -f .dockerignore ]; then mv .dockerignore .dockerignore.backend.tmp; fi
if [ -f .dockerignore.frontend ]; then cp .dockerignore.frontend .dockerignore; fi

# Build and push
docker buildx build \
  --platform linux/amd64 \
  --file ops/Dockerfile.frontend.production \
  --build-arg BUILD_DATE="2025-10-11T05:00:00Z" \
  --build-arg VCS_REF="04aa6fe" \
  --build-arg VERSION="1.0.1" \
  --build-arg NEXT_PUBLIC_API_URL="http://mini-xdr-backend-service:8000" \
  --tag 116912495274.dkr.ecr.us-east-1.amazonaws.com/mini-xdr-frontend:v1.0.1 \
  --tag 116912495274.dkr.ecr.us-east-1.amazonaws.com/mini-xdr-frontend:latest \
  --push \
  .

# Restore original .dockerignore
rm -f .dockerignore
if [ -f .dockerignore.backend.tmp ]; then mv .dockerignore.backend.tmp .dockerignore; fi
```

**Multi-Stage Build:**
1. Dependencies stage (npm install)
2. Build stage (next build)
3. Production stage (minimal runtime)

**v1.0.1 Changes (CSP Fix):**
- Added 'unsafe-inline' to script-src directive in Content Security Policy
- Fixed connect-src to use internal Kubernetes service URL
- Removed upgrade-insecure-requests directive (not needed without HTTPS)
- **Critical Fix:** Allows Next.js inline scripts to execute properly

---

## Network Flow & Security

### Inbound Traffic Flow

```
Internet (Your IP: 24.11.0.176)
  ↓
  ↓ HTTP/HTTPS (Port 80/443)
  ↓
ALB Security Group (sg-0e958a76b787f7689)
  ├─ Inbound: 24.11.0.176/32:80 ✓ ALLOWED
  ├─ Inbound: Other IPs ✗ DENIED
  ↓
Application Load Balancer
  ├─ Health check: /health (backend), / (frontend)
  ├─ Path routing based on URL
  ↓
Target Groups
  ├─ Backend TG: 10.0.11.145:8000
  └─ Frontend TG: 10.0.11.250:3000, 10.0.13.232:3000
  ↓
Node Security Groups (sg-0beefcaa22b6dc37e, sg-059f716b6776b2f6c)
  ├─ Inbound: sg-0e958a76b787f7689:3000 ✓ ALLOWED
  ├─ Inbound: sg-0e958a76b787f7689:8000 ✓ ALLOWED
  ↓
Kubernetes Service (ClusterIP)
  ├─ mini-xdr-backend-service: 172.20.158.62:8000
  └─ mini-xdr-frontend-service: 172.20.71.88:3000
  ↓
Pods (via iptables DNAT)
  ├─ Backend: 10.0.11.145:8000
  └─ Frontend: 10.0.11.250:3000, 10.0.13.232:3000
```

### Pod-to-RDS Flow

```
Backend Pod (10.0.11.145)
  ↓
  ↓ PostgreSQL (Port 5432)
  ↓
Node Security Group (sg-0beefcaa22b6dc37e)
  ├─ Outbound: All traffic ✓ ALLOWED
  ↓
VPC Routing (10.0.0.0/16)
  ↓
RDS Security Group
  ├─ Inbound: VPC CIDR:5432 ✓ ALLOWED
  ↓
RDS PostgreSQL Instance
  └─ mini-xdr-postgres.ccnkck4wij3r.us-east-1.rds.amazonaws.com:5432
```

### Pod-to-EFS Flow

```
Backend Pod (10.0.11.145)
  ↓
  ↓ NFS (Port 2049)
  ↓
Node Security Group (sg-0beefcaa22b6dc37e)
  ├─ Outbound: All traffic ✓ ALLOWED
  ↓
EFS Mount Target (10.0.11.102, 10.0.12.111, 10.0.13.54)
  ↓
EFS Security Group (sg-0bd21dae4146cde52)
  ├─ Inbound: sg-0beefcaa22b6dc37e:2049 ✓ ALLOWED
  ├─ Inbound: sg-059f716b6776b2f6c:2049 ✓ ALLOWED
  ↓
EFS File System (fs-0109cfbea9b55373c)
  └─ Encrypted with KMS (431cb645-f4d9-41f6-8d6e-6c26c79c5c04)
```

### Pod-to-ECR Flow

```
Kubelet on Node (10.0.11.108, 10.0.13.168)
  ↓
  ↓ HTTPS (Port 443)
  ↓
NAT Gateway (in public subnet)
  ↓
Internet Gateway
  ↓
ECR API Endpoint
  ↓ Authentication via IAM
  ↓
ECR Repository (116912495274.dkr.ecr.us-east-1.amazonaws.com)
  └─ Image pull: mini-xdr-backend:v1.0.1, mini-xdr-frontend:latest
```

### Security Summary

**Network Segmentation:**
- ✓ Public subnets (ALB) isolated from private subnets (EKS)
- ✓ No direct public IP on EKS nodes
- ✓ All egress through NAT gateways

**Access Control:**
- ✓ IP whitelist on ALB (24.11.0.176/32 only)
- ✓ Security groups control all inter-service communication
- ✓ IAM roles restrict AWS API access

**Data Protection:**
- ✓ EFS encrypted at rest (KMS)
- ✓ RDS encrypted at rest (KMS)
- ✓ EBS volumes encrypted (KMS)
- ✓ ECR images encrypted (AES-256)
- ✓ Kubernetes secrets base64 encoded (consider encrypting with KMS)

**Secrets Management:**
- ⚠️ Kubernetes secrets (base64 only, not encrypted)
- ✓ RDS password managed by AWS Secrets Manager (recommended)
- ✓ IAM roles for service accounts (IRSA) for EFS, ALB

---

## Files & Configurations

### Project Structure

```
mini-xdr/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI application entry
│   │   ├── models.py            # SQLAlchemy database models
│   │   ├── config.py            # Configuration management
│   │   ├── detect.py            # Threat detection engine
│   │   ├── ml_engine.py         # Machine learning inference
│   │   ├── agents/              # AI agent implementations
│   │   ├── policies/            # Response playbook policies
│   │   └── ...
│   ├── requirements.txt         # Python dependencies
│   ├── Dockerfile               # Local development Dockerfile
│   └── migrations/              # Alembic database migrations
│
├── frontend/
│   ├── app/
│   │   ├── page.tsx             # Main dashboard
│   │   ├── layout.tsx           # Application layout
│   │   ├── incidents/           # Incident management pages
│   │   ├── components/          # Reusable React components
│   │   └── ...
│   ├── package.json             # Node.js dependencies
│   ├── next.config.ts           # Next.js configuration
│   ├── tailwind.config.ts       # Tailwind CSS configuration
│   └── Dockerfile               # Local development Dockerfile
│
├── ops/
│   ├── Dockerfile.backend.production    # Production backend image
│   ├── Dockerfile.frontend.production   # Production frontend image
│   ├── k8s/
│   │   ├── namespace.yaml               # Kubernetes namespace
│   │   ├── backend-deployment.yaml      # Backend deployment + services
│   │   ├── frontend-deployment.yaml     # Frontend deployment + services
│   │   ├── ingress.yaml                 # ALB Ingress configuration
│   │   ├── configmap.yaml               # Policy ConfigMap
│   │   ├── secrets.yaml.example         # Secret template (not checked in)
│   │   ├── persistent-volumes.yaml      # Storage classes + PVCs
│   │   └── hpa.yaml                     # Horizontal Pod Autoscalers
│   └── docker-compose.yml               # Local development environment
│
├── aws/
│   ├── start-mini-xdr-aws-v4.sh         # AWS infrastructure setup script
│   ├── train_enhanced_full_dataset.py   # ML model training (AWS)
│   └── ...
│
├── infrastructure/
│   ├── aws/
│   │   ├── deploy-eks-cluster.sh        # EKS cluster creation
│   │   ├── deploy-to-eks.sh             # Application deployment
│   │   └── eks-cluster-config.yaml      # eksctl configuration
│   └── terraform/                        # Terraform IaC (if used)
│
├── scripts/
│   ├── testing/                         # Test scripts
│   ├── security/                        # Security hardening scripts
│   └── ...
│
├── docs/
│   └── AWS_DEPLOYMENT_GUIDE.md          # This file
│
├── .dockerignore                        # Docker build exclusions
├── .dockerignore.frontend               # Frontend-specific exclusions
├── buildspec-backend.yml                # AWS CodeBuild spec (if used)
└── README.md                            # Project overview
```

### Key Configuration Files

#### 1. Backend Deployment (ops/k8s/backend-deployment.yaml)

Defines:
- Backend deployment (replicas, image, resources)
- ClusterIP service (internal)
- NodePort service (external, for debugging)
- ConfigMap mount for policies
- PVC mounts for data and models
- Environment variables from secrets
- Health check endpoints

**Used by:** kubectl apply

#### 2. Frontend Deployment (ops/k8s/frontend-deployment.yaml)

Defines:
- Frontend deployment (replicas, image, resources)
- ClusterIP service (internal)
- NodePort service (external, for debugging)
- Environment variable for API URL
- Health check endpoints

**Used by:** kubectl apply

#### 3. Ingress (ops/k8s/ingress.yaml)

Defines:
- ALB creation and configuration
- IP whitelist annotation
- Path-based routing rules
- Health check configuration
- Security group assignment

**Used by:** AWS Load Balancer Controller

#### 4. Persistent Volumes (ops/k8s/persistent-volumes.yaml)

Defines:
- EFS StorageClass (efs-sc)
- EBS StorageClass (mini-xdr-gp3)
- Data PVC (10 GiB EBS)
- Models PVC (5 GiB EFS)

**Used by:** kubectl apply

#### 5. ConfigMap (ops/k8s/configmap.yaml)

Defines:
- Response playbook policies
- Agent configurations
- Detection rules

**Generated from:** backend/policies/ directory

#### 6. EKS Cluster Config (infrastructure/aws/eks-cluster-config.yaml)

Defines:
- EKS cluster version
- Node group configuration
- VPC and subnet mapping
- IAM roles and policies
- Add-on configurations

**Used by:** eksctl create cluster

#### 7. Backend Production Dockerfile (ops/Dockerfile.backend.production)

Multi-stage build:
1. Base stage: Python 3.11 + system deps
2. Dependencies stage: pip install requirements.txt
3. Production stage: Copy app + install packages
4. Entrypoint: uvicorn

**Used by:** docker buildx build

#### 8. Frontend Production Dockerfile (ops/Dockerfile.frontend.production)

Multi-stage build:
1. Dependencies stage: npm install
2. Build stage: next build
3. Production stage: node server
4. Entrypoint: npm start

**Used by:** docker buildx build

#### 9. Backend Requirements (backend/requirements.txt)

Defines all Python dependencies:
- Web framework: fastapi, uvicorn
- Database: sqlalchemy, psycopg2-binary, asyncpg
- ML: torch, scikit-learn, xgboost, pandas
- AI: langchain, langchain-openai
- Security: cryptography, pycryptodome
- **Critical:** email-validator==2.1.0 (added in v1.0.1)

**Used by:** pip install -r requirements.txt (in Dockerfile)

#### 10. Frontend Package.json (frontend/package.json)

Defines all Node.js dependencies:
- Framework: next@15.5.0, react@19.0.0
- UI: tailwindcss, @headlessui/react
- State: zustand (if used)
- HTTP: axios (if used)

**Used by:** npm install (in Dockerfile)

---

## Access Instructions

### Prerequisites

1. **AWS CLI configured:**
   ```bash
   aws configure
   AWS Access Key ID: [your-access-key]
   AWS Secret Access Key: [your-secret-key]
   Default region: us-east-1
   Default output format: json
   ```

2. **kubectl configured:**
   ```bash
   aws eks update-kubeconfig --name mini-xdr-cluster --region us-east-1
   ```

3. **Your IP whitelisted:**
   - Current whitelist: 24.11.0.176/32
   - If your IP changes, update ingress:
     ```bash
     kubectl annotate ingress mini-xdr-ingress \
       -n mini-xdr \
       alb.ingress.kubernetes.io/inbound-cidrs="NEW_IP/32" \
       --overwrite
     ```

### Accessing the Application

#### 1. Web Browser Access

**URL:** http://k8s-minixdr-minixdri-dc5fc1df8b-1132128475.us-east-1.elb.amazonaws.com

**Available Pages:**
- `/` - Main dashboard
- `/incidents` - Incident management
- `/hunt` - Threat hunting interface
- `/analytics/response` - Response analytics

**Note:** HTTPS not yet configured. For production, add ACM certificate to ALB.

#### 2. API Access

**Base URL:** http://k8s-minixdr-minixdri-dc5fc1df8b-1132128475.us-east-1.elb.amazonaws.com

**Endpoints:**
```bash
# Health check
curl http://k8s-minixdr-minixdri-dc5fc1df8b-1132128475.us-east-1.elb.amazonaws.com/health

# Get incidents
curl http://k8s-minixdr-minixdri-dc5fc1df8b-1132128475.us-east-1.elb.amazonaws.com/api/incidents

# API documentation
curl http://k8s-minixdr-minixdri-dc5fc1df8b-1132128475.us-east-1.elb.amazonaws.com/api/docs
```

#### 3. Direct Node Access (for debugging)

**NodePort Services:**

Backend:
```bash
# Get node public IP
NODE_IP=$(kubectl get nodes -o jsonpath='{.items[0].status.addresses[?(@.type=="ExternalIP")].address}')

# Access backend (if NodePort 30800 is exposed in SG)
curl http://$NODE_IP:30800/health
```

Frontend:
```bash
# Access frontend (if NodePort 30300 is exposed in SG)
curl http://$NODE_IP:30300/
```

**Note:** NodePorts are not publicly accessible by default. Requires security group modification.

#### 4. Kubernetes Dashboard (optional)

Deploy Kubernetes Dashboard:
```bash
kubectl apply -f https://raw.githubusercontent.com/kubernetes/dashboard/v2.7.0/aio/deploy/recommended.yaml

# Create service account and get token
kubectl create serviceaccount dashboard-admin -n kubernetes-dashboard
kubectl create clusterrolebinding dashboard-admin \
  --clusterrole=cluster-admin \
  --serviceaccount=kubernetes-dashboard:dashboard-admin

# Get token
kubectl -n kubernetes-dashboard create token dashboard-admin

# Proxy to dashboard
kubectl proxy

# Access at: http://localhost:8001/api/v1/namespaces/kubernetes-dashboard/services/https:kubernetes-dashboard:/proxy/
```

---

## Common Operations

### Viewing Logs

#### Backend Logs
```bash
# All backend pods
kubectl logs -n mini-xdr -l app=mini-xdr-backend --tail=100 -f

# Specific pod
kubectl logs -n mini-xdr mini-xdr-backend-769ffd6d9b-q289l --tail=100 -f

# Previous crashed pod
kubectl logs -n mini-xdr mini-xdr-backend-769ffd6d9b-q289l --previous
```

#### Frontend Logs
```bash
# All frontend pods
kubectl logs -n mini-xdr -l app=mini-xdr-frontend --tail=100 -f

# Specific pod
kubectl logs -n mini-xdr mini-xdr-frontend-7d7945d4f7-2tcrk --tail=100 -f
```

#### ALB Logs (CloudWatch)
```bash
# View ALB access logs (if enabled)
aws logs tail /aws/elasticloadbalancing/app/k8s-minixdr-minixdri-dc5fc1df8b --follow
```

### Scaling Resources

#### Manual Scaling

**Backend:**
```bash
# Scale to 2 replicas (if resources allow)
kubectl scale deployment mini-xdr-backend -n mini-xdr --replicas=2

# Verify scaling
kubectl get pods -n mini-xdr -l app=mini-xdr-backend
```

**Frontend:**
```bash
# Scale to 3 replicas
kubectl scale deployment mini-xdr-frontend -n mini-xdr --replicas=3
```

**Nodes (EKS):**
```bash
# Update node group desired capacity
aws eks update-nodegroup-config \
  --cluster-name mini-xdr-cluster \
  --nodegroup-name mini-xdr-ng-1 \
  --scaling-config minSize=2,maxSize=4,desiredSize=3 \
  --region us-east-1
```

#### Autoscaling Configuration

**View HPA status:**
```bash
kubectl get hpa -n mini-xdr

# Detailed HPA info
kubectl describe hpa mini-xdr-backend-hpa -n mini-xdr
```

**Modify HPA thresholds:**
```bash
# Edit HPA
kubectl edit hpa mini-xdr-backend-hpa -n mini-xdr

# Change target CPU from 70% to 80%
# Change target memory from 80% to 85%
```

### Updating Application

#### Rolling Update (zero downtime)

**Backend:**
```bash
# Build and push new image
docker buildx build \
  --platform linux/amd64 \
  --file ops/Dockerfile.backend.production \
  --tag 116912495274.dkr.ecr.us-east-1.amazonaws.com/mini-xdr-backend:v1.0.2 \
  --push \
  .

# Update deployment image
kubectl set image deployment/mini-xdr-backend \
  backend=116912495274.dkr.ecr.us-east-1.amazonaws.com/mini-xdr-backend:v1.0.2 \
  -n mini-xdr

# Watch rollout
kubectl rollout status deployment/mini-xdr-backend -n mini-xdr

# Rollback if needed
kubectl rollout undo deployment/mini-xdr-backend -n mini-xdr
```

**Frontend:**
```bash
# Similar process for frontend
docker buildx build \
  --platform linux/amd64 \
  --file ops/Dockerfile.frontend.production \
  --tag 116912495274.dkr.ecr.us-east-1.amazonaws.com/mini-xdr-frontend:v1.0.1 \
  --push \
  .

kubectl set image deployment/mini-xdr-frontend \
  frontend=116912495274.dkr.ecr.us-east-1.amazonaws.com/mini-xdr-frontend:v1.0.1 \
  -n mini-xdr
```

#### Updating Configuration

**Environment Variables:**
```bash
# Update secret
kubectl edit secret mini-xdr-secrets -n mini-xdr

# Restart pods to pick up new secret
kubectl rollout restart deployment/mini-xdr-backend -n mini-xdr
```

**ConfigMap:**
```bash
# Update ConfigMap
kubectl edit configmap mini-xdr-policies -n mini-xdr

# Restart to pick up changes
kubectl rollout restart deployment/mini-xdr-backend -n mini-xdr
```

### Monitoring Resources

#### Cluster Status
```bash
# Node status and resource usage
kubectl top nodes

# All pods resource usage
kubectl top pods -n mini-xdr

# Specific pod resource usage
kubectl top pod mini-xdr-backend-769ffd6d9b-q289l -n mini-xdr
```

#### Deployment Status
```bash
# All resources in namespace
kubectl get all -n mini-xdr

# Deployment status
kubectl get deployments -n mini-xdr -o wide

# Pod status with node info
kubectl get pods -n mini-xdr -o wide

# Events (troubleshooting)
kubectl get events -n mini-xdr --sort-by='.lastTimestamp'
```

#### Health Checks
```bash
# Check ALB target health
aws elbv2 describe-target-health \
  --target-group-arn arn:aws:elasticloadbalancing:us-east-1:116912495274:targetgroup/k8s-minixdr-minixdrb-407ce4b45d/cdb5470305250908 \
  --region us-east-1

# Check pod readiness
kubectl get pods -n mini-xdr -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.status.conditions[?(@.type=="Ready")].status}{"\n"}{end}'
```

### Database Operations

#### Connecting to RDS

**From local machine (via bastion or VPN):**
```bash
psql -h mini-xdr-postgres.ccnkck4wij3r.us-east-1.rds.amazonaws.com \
     -U postgres \
     -d mini_xdr_db \
     -p 5432
```

**From within EKS (kubectl exec):**
```bash
# Exec into backend pod
kubectl exec -it mini-xdr-backend-769ffd6d9b-q289l -n mini-xdr -- /bin/bash

# Inside pod
psql $DATABASE_URL

# Run query
SELECT COUNT(*) FROM incidents;
```

#### Database Backup

**RDS Automated Backups:**
```bash
# View backups
aws rds describe-db-snapshots \
  --db-instance-identifier mini-xdr-postgres \
  --region us-east-1

# Create manual snapshot
aws rds create-db-snapshot \
  --db-snapshot-identifier mini-xdr-snapshot-$(date +%Y%m%d) \
  --db-instance-identifier mini-xdr-postgres \
  --region us-east-1
```

**Restore from snapshot:**
```bash
aws rds restore-db-instance-from-db-snapshot \
  --db-instance-identifier mini-xdr-postgres-restored \
  --db-snapshot-identifier mini-xdr-snapshot-20251010 \
  --region us-east-1
```

#### Running Migrations

**Manually run Alembic migrations:**
```bash
# Exec into backend pod
kubectl exec -it mini-xdr-backend-769ffd6d9b-q289l -n mini-xdr -- /bin/bash

# Inside pod
cd /app/backend
alembic upgrade head
```

**Note:** Current deployment skips migrations on startup to avoid conflicts. Run manually if needed.

### Storage Operations

#### EFS Operations

**Check EFS usage:**
```bash
# Describe file system
aws efs describe-file-systems \
  --file-system-id fs-0109cfbea9b55373c \
  --region us-east-1

# Check mount targets
aws efs describe-mount-targets \
  --file-system-id fs-0109cfbea9b55373c \
  --region us-east-1
```

**Access EFS data from pod:**
```bash
# Exec into backend pod
kubectl exec -it mini-xdr-backend-769ffd6d9b-q289l -n mini-xdr -- /bin/bash

# Inside pod
ls -lah /app/models/
du -sh /app/models/
```

**Backup EFS data:**
```bash
# Create AWS Backup plan (recommended)
aws backup create-backup-plan --backup-plan file://backup-plan.json

# Or manual backup via tar
kubectl exec mini-xdr-backend-769ffd6d9b-q289l -n mini-xdr -- \
  tar czf /tmp/models-backup.tar.gz /app/models/
```

#### EBS Operations

**Check EBS volumes:**
```bash
# List volumes
aws ec2 describe-volumes \
  --filters "Name=tag:kubernetes.io/cluster/mini-xdr-cluster,Values=owned" \
  --region us-east-1

# Get PVC volume
kubectl get pvc mini-xdr-data-pvc -n mini-xdr -o jsonpath='{.spec.volumeName}'
```

**Resize EBS volume:**
```bash
# Edit PVC to increase size
kubectl edit pvc mini-xdr-data-pvc -n mini-xdr
# Change spec.resources.requests.storage: 10Gi → 20Gi

# Restart pod to apply
kubectl rollout restart deployment/mini-xdr-backend -n mini-xdr
```

### Security Operations

#### Updating IP Whitelist

**Add new IP:**
```bash
# Current: 24.11.0.176/32
# New: 203.0.113.0/32

kubectl annotate ingress mini-xdr-ingress \
  -n mini-xdr \
  alb.ingress.kubernetes.io/inbound-cidrs="24.11.0.176/32,203.0.113.0/32" \
  --overwrite
```

**Allow all IPs (NOT recommended for production):**
```bash
kubectl annotate ingress mini-xdr-ingress \
  -n mini-xdr \
  alb.ingress.kubernetes.io/inbound-cidrs="0.0.0.0/0" \
  --overwrite
```

#### Rotating Secrets

**Update Kubernetes secret:**
```bash
# Delete old secret
kubectl delete secret mini-xdr-secrets -n mini-xdr

# Create new secret with updated values
kubectl create secret generic mini-xdr-secrets \
  --from-literal=DATABASE_URL="postgresql://new_password@..." \
  --from-literal=OPENAI_API_KEY="sk-new-key-..." \
  -n mini-xdr

# Restart pods to pick up new secret
kubectl rollout restart deployment/mini-xdr-backend -n mini-xdr
```

#### Viewing Security Groups

**List security groups:**
```bash
aws ec2 describe-security-groups \
  --group-ids sg-0e958a76b787f7689 sg-0beefcaa22b6dc37e \
  --region us-east-1
```

**Add security group rule:**
```bash
# Example: Allow SSH from your IP to nodes
aws ec2 authorize-security-group-ingress \
  --group-id sg-0beefcaa22b6dc37e \
  --protocol tcp \
  --port 22 \
  --cidr 24.11.0.176/32 \
  --region us-east-1
```

---

## Troubleshooting

### Common Issues & Solutions

#### 1. Pod Not Starting (Pending/ContainerCreating)

**Symptoms:**
```bash
kubectl get pods -n mini-xdr
NAME                               READY   STATUS              RESTARTS   AGE
mini-xdr-backend-769ffd6d9b-xxxxx  0/1     ContainerCreating   0          5m
```

**Diagnosis:**
```bash
# Describe pod for events
kubectl describe pod mini-xdr-backend-769ffd6d9b-xxxxx -n mini-xdr

# Common causes:
# - Insufficient CPU/memory on nodes
# - Image pull errors
# - Volume mount failures
```

**Solutions:**

**Insufficient resources:**
```bash
# Check node capacity
kubectl describe nodes

# Scale down other deployments
kubectl scale deployment mini-xdr-backend -n mini-xdr --replicas=1

# Or add more nodes
aws eks update-nodegroup-config \
  --cluster-name mini-xdr-cluster \
  --nodegroup-name mini-xdr-ng-1 \
  --scaling-config minSize=2,maxSize=4,desiredSize=3 \
  --region us-east-1
```

**Image pull errors:**
```bash
# Check image exists in ECR
aws ecr describe-images \
  --repository-name mini-xdr-backend \
  --region us-east-1

# Re-push image if missing
docker push 116912495274.dkr.ecr.us-east-1.amazonaws.com/mini-xdr-backend:v1.0.1
```

**EFS mount failures:**
```bash
# Check EFS mount targets
aws efs describe-mount-targets \
  --file-system-id fs-0109cfbea9b55373c \
  --region us-east-1

# Verify security group allows NFS (port 2049)
aws ec2 describe-security-groups --group-ids sg-0bd21dae4146cde52 --region us-east-1
```

#### 2. Pod CrashLoopBackOff

**Symptoms:**
```bash
kubectl get pods -n mini-xdr
NAME                               READY   STATUS             RESTARTS   AGE
mini-xdr-backend-769ffd6d9b-xxxxx  0/1     CrashLoopBackOff   5          10m
```

**Diagnosis:**
```bash
# Check logs
kubectl logs mini-xdr-backend-769ffd6d9b-xxxxx -n mini-xdr --previous

# Common causes:
# - Application startup errors
# - Database connection failures
# - Missing environment variables
```

**Solutions:**

**Database connection:**
```bash
# Verify DATABASE_URL secret
kubectl get secret mini-xdr-secrets -n mini-xdr -o jsonpath='{.data.DATABASE_URL}' | base64 -d

# Test connection from pod
kubectl run test-db --rm -it --image=postgres:17 -- \
  psql "postgresql://..."

# Check RDS status
aws rds describe-db-instances \
  --db-instance-identifier mini-xdr-postgres \
  --query 'DBInstances[0].DBInstanceStatus' \
  --region us-east-1
```

**Missing dependencies:**
```bash
# Check if email-validator is installed (v1.0.1 fix)
kubectl exec mini-xdr-backend-769ffd6d9b-xxxxx -n mini-xdr -- \
  pip list | grep email-validator

# If missing, rebuild image with requirements.txt updated
```

#### 3. ALB Health Check Failing

**Symptoms:**
```bash
aws elbv2 describe-target-health \
  --target-group-arn arn:aws:elasticloadbalancing:us-east-1:116912495274:targetgroup/k8s-minixdr-minixdrb-407ce4b45d/cdb5470305250908 \
  --region us-east-1

# Output: "State": "unhealthy", "Reason": "Target.Timeout"
```

**Diagnosis:**
```bash
# Check security group allows ALB → Pods
aws ec2 describe-security-groups \
  --group-ids sg-0beefcaa22b6dc37e \
  --region us-east-1

# Check if pod is actually running
kubectl get pods -n mini-xdr -o wide

# Test health endpoint from within cluster
kubectl run test-curl --rm -it --image=curlimages/curl -- \
  curl -v http://10.0.11.145:8000/health
```

**Solutions:**

**Add security group rules:**
```bash
# Allow ALB SG to access backend port 8000
aws ec2 authorize-security-group-ingress \
  --group-id sg-0beefcaa22b6dc37e \
  --protocol tcp \
  --port 8000 \
  --source-group sg-0e958a76b787f7689 \
  --region us-east-1

# Allow ALB SG to access frontend port 3000
aws ec2 authorize-security-group-ingress \
  --group-id sg-0beefcaa22b6dc37e \
  --protocol tcp \
  --port 3000 \
  --source-group sg-0e958a76b787f7689 \
  --region us-east-1
```

**Fix health check path:**
```bash
# Frontend doesn't have /health endpoint, use /
aws elbv2 modify-target-group \
  --target-group-arn arn:aws:elasticloadbalancing:us-east-1:116912495274:targetgroup/k8s-minixdr-minixdrf-c279e7e8e8/b852eced28b4e308 \
  --health-check-path / \
  --region us-east-1
```

#### 4. Cannot Access Application from Browser

**Symptoms:**
- Connection timeout
- "This site can't be reached"

**Diagnosis:**
```bash
# Check your current IP
curl -4 ifconfig.me

# Check ALB ingress whitelist
kubectl get ingress mini-xdr-ingress -n mini-xdr \
  -o jsonpath='{.metadata.annotations.alb\.ingress\.kubernetes\.io/inbound-cidrs}'
```

**Solutions:**

**Update IP whitelist:**
```bash
# Replace with your current IP
YOUR_IP=$(curl -4 -s ifconfig.me)

kubectl annotate ingress mini-xdr-ingress \
  -n mini-xdr \
  alb.ingress.kubernetes.io/inbound-cidrs="$YOUR_IP/32" \
  --overwrite
```

**Verify ALB is active:**
```bash
aws elbv2 describe-load-balancers \
  --names k8s-minixdr-minixdri-dc5fc1df8b \
  --query 'LoadBalancers[0].State.Code' \
  --region us-east-1
# Should output: "active"
```

#### 5. Content Security Policy (CSP) Error

**Symptoms:**
- Application loads HTML but JavaScript doesn't execute
- Browser console error: "Refused to execute inline script because it violates the following Content Security Policy directive: 'script-src...'"
- Page appears stuck or only partially loads

**Diagnosis:**
```bash
# Check browser console (F12) for errors
# Look for CSP violation messages

# Test if HTML is loading
curl -I http://k8s-minixdr-minixdri-dc5fc1df8b-1132128475.us-east-1.elb.amazonaws.com/

# Check current CSP header
curl -s -I http://k8s-minixdr-minixdri-dc5fc1df8b-1132128475.us-east-1.elb.amazonaws.com/ | \
  grep "Content-Security-Policy"
```

**Root Cause:**
- Next.js requires `'unsafe-inline'` in the `script-src` directive to execute inline scripts
- Production CSP in `frontend/next.config.ts` was too restrictive

**Solutions:**

**Update CSP in next.config.ts:**
```typescript
// File: frontend/next.config.ts (line 40)

// Change from:
const prodCSP = "...script-src 'self' 'wasm-unsafe-eval' blob:..."

// To:
const prodCSP = "...script-src 'self' 'unsafe-inline' 'wasm-unsafe-eval' blob:..."
```

**Full fix process:**
```bash
# 1. Update frontend/next.config.ts
# Edit line 40 to add 'unsafe-inline' to script-src

# 2. Swap .dockerignore to frontend-specific version
if [ -f .dockerignore ]; then mv .dockerignore .dockerignore.backend.tmp; fi
if [ -f .dockerignore.frontend ]; then cp .dockerignore.frontend .dockerignore; fi

# 3. Build new frontend image
docker buildx build \
  --platform linux/amd64 \
  --file ops/Dockerfile.frontend.production \
  --build-arg NEXT_PUBLIC_API_URL="http://mini-xdr-backend-service:8000" \
  --tag 116912495274.dkr.ecr.us-east-1.amazonaws.com/mini-xdr-frontend:v1.0.1 \
  --push \
  .

# 4. Restore .dockerignore
rm -f .dockerignore
if [ -f .dockerignore.backend.tmp ]; then mv .dockerignore.backend.tmp .dockerignore; fi

# 5. Update deployment to use new image
kubectl set image deployment/mini-xdr-frontend \
  frontend=116912495274.dkr.ecr.us-east-1.amazonaws.com/mini-xdr-frontend:v1.0.1 \
  -n mini-xdr

# 6. Watch rollout
kubectl rollout status deployment/mini-xdr-frontend -n mini-xdr

# 7. Verify pods are running
kubectl get pods -n mini-xdr -l app=mini-xdr-frontend

# 8. Test application
curl -s http://k8s-minixdr-minixdri-dc5fc1df8b-1132128475.us-east-1.elb.amazonaws.com/ | head -50
```

**Verification:**
```bash
# Check new CSP header includes 'unsafe-inline'
curl -s -I http://k8s-minixdr-minixdri-dc5fc1df8b-1132128475.us-east-1.elb.amazonaws.com/ | \
  grep -A 1 "Content-Security-Policy"

# Should show:
# Content-Security-Policy: ...script-src 'self' 'unsafe-inline' 'wasm-unsafe-eval' blob:...

# Open browser and check console - no more CSP errors
```

**Note:** This was fixed in frontend v1.0.1. Current production image includes this fix.

#### 6. High Memory Usage

**Symptoms:**
```bash
kubectl top pods -n mini-xdr
NAME                               CPU(cores)   MEMORY(bytes)
mini-xdr-backend-769ffd6d9b-q289l  10m          1300Mi
```

**Diagnosis:**
```bash
# Check pod memory limits
kubectl describe pod mini-xdr-backend-769ffd6d9b-q289l -n mini-xdr | grep -A 5 "Limits"

# Memory limit: 1536Mi
# Current usage: 1300Mi (85% of limit)
```

**Solutions:**

**Increase memory limits:**
```bash
# Edit deployment
kubectl edit deployment mini-xdr-backend -n mini-xdr

# Update resources:
#   requests:
#     memory: 1024Mi  # Increase from 768Mi
#   limits:
#     memory: 2048Mi  # Increase from 1536Mi

# Apply changes
kubectl rollout restart deployment/mini-xdr-backend -n mini-xdr
```

**Optimize application:**
```bash
# Reduce Uvicorn workers (current: 2)
kubectl set env deployment/mini-xdr-backend -n mini-xdr \
  UVICORN_WORKERS=1

# Or modify deployment command:
# sh -c "uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1"
```

#### 6. Database Connection Pool Exhausted

**Symptoms:**
```
sqlalchemy.exc.TimeoutError: QueuePool limit of size 5 overflow 10 reached
```

**Solutions:**

**Increase connection pool size:**
```python
# In backend/app/db.py
engine = create_engine(
    DATABASE_URL,
    pool_size=10,      # Increase from 5
    max_overflow=20,   # Increase from 10
    pool_pre_ping=True
)
```

**Scale backend pods:**
```bash
# Distribute load across multiple pods
kubectl scale deployment mini-xdr-backend -n mini-xdr --replicas=2
```

#### 7. ECR Image Pull Rate Limit

**Symptoms:**
```
Failed to pull image: toomanyrequests: Rate exceeded
```

**Solutions:**

**Use VPC endpoints for ECR:**
```bash
# Create VPC endpoint for ECR
aws ec2 create-vpc-endpoint \
  --vpc-id vpc-0d474acd38d418e98 \
  --service-name com.amazonaws.us-east-1.ecr.dkr \
  --route-table-ids rtb-xxxxx \
  --region us-east-1

# No additional cost, avoids rate limits
```

**Enable image caching:**
```yaml
# In deployment, set imagePullPolicy: IfNotPresent
imagePullPolicy: IfNotPresent
```

---

## Cost Analysis

### Monthly Cost Breakdown (Estimated)

**Compute (EKS + EC2):**
```
EKS Control Plane:              $73.00/month  (2 AZs)
EC2 t3.medium nodes (2):        $60.96/month  ($0.0416/hour × 2 × 730 hours)
EKS add-ons:                    $0.00         (included)
---
Total Compute:                  $133.96/month
```

**Storage:**
```
EFS (Standard, 5 GiB):          $1.50/month   ($0.30/GB × 5)
EBS gp3 (10 GiB):               $0.80/month   ($0.08/GB × 10)
RDS storage (20 GB gp3):        $2.76/month   ($0.138/GB × 20)
---
Total Storage:                  $5.06/month
```

**Database:**
```
RDS db.t3.micro (730 hours):    $13.14/month  ($0.018/hour × 730)
RDS backup storage (7 days):    ~$0.50/month  (estimated)
---
Total Database:                 $13.64/month
```

**Networking:**
```
ALB:                            $16.43/month  ($0.0225/hour × 730)
ALB LCU charges:                ~$5.00/month  (estimated, light traffic)
NAT Gateway (3 AZs):            $97.92/month  ($0.045/hour × 3 × 730)
Data transfer (out):            ~$10.00/month (estimated, 100 GB)
---
Total Networking:               $129.35/month
```

**Container Registry:**
```
ECR storage (4 GB):             $0.40/month   ($0.10/GB × 4)
ECR data transfer:              $0.00         (within region)
---
Total ECR:                      $0.40/month
```

**Security & Monitoring:**
```
KMS (1 key, 1000 requests):     $1.00/month
CloudWatch Logs (5 GB):         $2.50/month   ($0.50/GB × 5)
CloudWatch Metrics:             $0.00         (free tier: 10 metrics)
---
Total Security:                 $3.50/month
```

**TOTAL ESTIMATED COST:**        **$286.91/month**

### Cost Optimization Recommendations

#### 1. Use Fargate Instead of EC2 Nodes
**Savings:** ~$40-50/month
**Tradeoff:** Less control, cold start delays

```bash
# Fargate only charges for actual pod usage
eksctl create fargateprofile \
  --cluster mini-xdr-cluster \
  --name mini-xdr-fargate \
  --namespace mini-xdr
```

#### 2. Use Single NAT Gateway
**Savings:** ~$65/month (remove 2 of 3 NAT Gateways)
**Tradeoff:** Single point of failure

```bash
# Update VPC to use single NAT in one AZ
# Route all private subnet traffic through one NAT
```

#### 3. Use RDS Aurora Serverless v2
**Savings:** ~$5-10/month (scales to zero during idle)
**Tradeoff:** Cold start delays

```bash
aws rds create-db-cluster \
  --db-cluster-identifier mini-xdr-cluster \
  --engine aurora-postgresql \
  --engine-mode serverless \
  --scaling-configuration MinCapacity=0.5,MaxCapacity=1
```

#### 4. Use Spot Instances for Nodes
**Savings:** ~70% on EC2 costs (~$43/month)
**Tradeoff:** Nodes can be terminated with 2-minute notice

```yaml
# In eks-cluster-config.yaml
nodeGroups:
  - name: mini-xdr-ng-spot
    instancesDistribution:
      instanceTypes: ["t3.medium", "t3a.medium"]
      onDemandBaseCapacity: 0
      onDemandPercentageAboveBaseCapacity: 0
      spotAllocationStrategy: capacity-optimized
```

#### 5. Enable S3 Gateway Endpoint
**Savings:** Data transfer costs (varies)
**Tradeoff:** None

```bash
# Free VPC endpoint for S3 access
aws ec2 create-vpc-endpoint \
  --vpc-id vpc-0d474acd38d418e98 \
  --service-name com.amazonaws.us-east-1.s3 \
  --route-table-ids rtb-xxxxx
```

#### 6. Use Graviton2 (ARM) Nodes
**Savings:** ~20% on EC2 costs (~$12/month)
**Tradeoff:** Must rebuild images for ARM64

```yaml
# Use t4g.medium instead of t3.medium
instanceType: t4g.medium
```

**With All Optimizations Applied: ~$150-180/month**

---

## Backup & Disaster Recovery

### Backup Strategy

#### 1. RDS Automated Backups

**Configuration:**
```
Backup retention: 7 days
Backup window: 03:00-04:00 UTC (low traffic)
Snapshot frequency: Daily
Point-in-time recovery: Enabled (5-minute granularity)
```

**Manual snapshot:**
```bash
aws rds create-db-snapshot \
  --db-snapshot-identifier mini-xdr-manual-$(date +%Y%m%d) \
  --db-instance-identifier mini-xdr-postgres \
  --region us-east-1
```

**Restore from snapshot:**
```bash
# Restore to new instance
aws rds restore-db-instance-from-db-snapshot \
  --db-instance-identifier mini-xdr-postgres-restored \
  --db-snapshot-identifier mini-xdr-manual-20251010 \
  --db-instance-class db.t3.micro \
  --publicly-accessible false \
  --region us-east-1

# Update backend DATABASE_URL to point to new instance
kubectl set env deployment/mini-xdr-backend -n mini-xdr \
  DATABASE_URL="postgresql://...mini-xdr-postgres-restored..."
```

#### 2. EFS Backups (AWS Backup)

**Setup AWS Backup:**
```bash
# Create backup vault
aws backup create-backup-vault \
  --backup-vault-name mini-xdr-efs-vault \
  --region us-east-1

# Create backup plan
aws backup create-backup-plan --backup-plan '{
  "BackupPlanName": "mini-xdr-efs-daily",
  "Rules": [{
    "RuleName": "DailyBackup",
    "TargetBackupVaultName": "mini-xdr-efs-vault",
    "ScheduleExpression": "cron(0 5 ? * * *)",
    "StartWindowMinutes": 60,
    "CompletionWindowMinutes": 120,
    "Lifecycle": {
      "DeleteAfterDays": 30
    }
  }]
}' --region us-east-1

# Assign EFS to backup plan
aws backup create-backup-selection --backup-plan-id <plan-id> --backup-selection '{
  "SelectionName": "mini-xdr-efs",
  "IamRoleArn": "arn:aws:iam::116912495274:role/AWSBackupServiceRole",
  "Resources": ["arn:aws:elasticfilesystem:us-east-1:116912495274:file-system/fs-0109cfbea9b55373c"]
}' --region us-east-1
```

**Manual EFS backup (tar):**
```bash
# From backend pod
kubectl exec mini-xdr-backend-769ffd6d9b-q289l -n mini-xdr -- \
  tar czf /tmp/efs-backup-$(date +%Y%m%d).tar.gz /app/models/

# Copy to S3
kubectl exec mini-xdr-backend-769ffd6d9b-q289l -n mini-xdr -- \
  aws s3 cp /tmp/efs-backup-20251010.tar.gz s3://mini-xdr-backups/
```

#### 3. Kubernetes Configuration Backup

**Export all manifests:**
```bash
# Create backup directory
mkdir -p backups/k8s/$(date +%Y%m%d)

# Export all resources
kubectl get all,ingress,pvc,configmap,secret -n mini-xdr -o yaml > \
  backups/k8s/$(date +%Y%m%d)/mini-xdr-full-backup.yaml

# Export secrets separately (encrypted)
kubectl get secrets -n mini-xdr -o yaml > \
  backups/k8s/$(date +%Y%m%d)/secrets.yaml

# Commit to private Git repo
git add backups/
git commit -m "Kubernetes backup $(date +%Y%m%d)"
git push origin main
```

#### 4. Container Images Backup

**Export images to S3:**
```bash
# Pull and save backend image
docker pull 116912495274.dkr.ecr.us-east-1.amazonaws.com/mini-xdr-backend:v1.0.1
docker save 116912495274.dkr.ecr.us-east-1.amazonaws.com/mini-xdr-backend:v1.0.1 | \
  gzip > mini-xdr-backend-v1.0.1.tar.gz

# Upload to S3
aws s3 cp mini-xdr-backend-v1.0.1.tar.gz s3://mini-xdr-backups/images/

# Same for frontend
docker save 116912495274.dkr.ecr.us-east-1.amazonaws.com/mini-xdr-frontend:latest | \
  gzip > mini-xdr-frontend-latest.tar.gz
aws s3 cp mini-xdr-frontend-latest.tar.gz s3://mini-xdr-backups/images/
```

### Disaster Recovery Plan

#### Scenario 1: EKS Cluster Failure

**Recovery Steps:**

1. **Create new EKS cluster:**
   ```bash
   eksctl create cluster -f infrastructure/aws/eks-cluster-config.yaml
   ```

2. **Restore Kubernetes resources:**
   ```bash
   kubectl apply -f backups/k8s/20251010/mini-xdr-full-backup.yaml
   ```

3. **Update ingress DNS:**
   ```bash
   # ALB DNS will change, update your DNS records
   kubectl get ingress mini-xdr-ingress -n mini-xdr
   ```

4. **Verify services:**
   ```bash
   kubectl get pods -n mini-xdr
   kubectl get svc -n mini-xdr
   ```

**Recovery Time Objective (RTO):** 30-60 minutes
**Recovery Point Objective (RPO):** 24 hours (last backup)

#### Scenario 2: Database Corruption

**Recovery Steps:**

1. **Stop backend pods:**
   ```bash
   kubectl scale deployment mini-xdr-backend -n mini-xdr --replicas=0
   ```

2. **Restore from snapshot:**
   ```bash
   aws rds restore-db-instance-from-db-snapshot \
     --db-instance-identifier mini-xdr-postgres-restored \
     --db-snapshot-identifier mini-xdr-manual-20251010 \
     --region us-east-1
   ```

3. **Update backend connection:**
   ```bash
   kubectl set env deployment/mini-xdr-backend -n mini-xdr \
     DATABASE_URL="postgresql://...mini-xdr-postgres-restored..."
   ```

4. **Restart backend:**
   ```bash
   kubectl scale deployment mini-xdr-backend -n mini-xdr --replicas=1
   ```

**RTO:** 15-30 minutes
**RPO:** 5 minutes (point-in-time recovery)

#### Scenario 3: EFS Data Loss

**Recovery Steps:**

1. **Stop backend pods:**
   ```bash
   kubectl scale deployment mini-xdr-backend -n mini-xdr --replicas=0
   ```

2. **Restore from AWS Backup:**
   ```bash
   # List recovery points
   aws backup list-recovery-points-by-resource \
     --resource-arn arn:aws:elasticfilesystem:us-east-1:116912495274:file-system/fs-0109cfbea9b55373c \
     --region us-east-1

   # Restore
   aws backup start-restore-job \
     --recovery-point-arn <recovery-point-arn> \
     --metadata file-system-id=fs-0109cfbea9b55373c \
     --iam-role-arn arn:aws:iam::116912495274:role/AWSBackupServiceRole \
     --region us-east-1
   ```

3. **Or restore from tar backup:**
   ```bash
   # Download from S3
   aws s3 cp s3://mini-xdr-backups/efs-backup-20251010.tar.gz /tmp/

   # Extract to EFS
   kubectl exec mini-xdr-backend-769ffd6d9b-q289l -n mini-xdr -- \
     tar xzf /tmp/efs-backup-20251010.tar.gz -C /
   ```

4. **Restart backend:**
   ```bash
   kubectl scale deployment mini-xdr-backend -n mini-xdr --replicas=1
   ```

**RTO:** 30-60 minutes
**RPO:** 24 hours

#### Scenario 4: Complete Region Failure

**Multi-Region Setup (Future Enhancement):**

1. Deploy identical infrastructure in `us-west-2`
2. Configure RDS cross-region read replica
3. Use Route 53 failover routing
4. Replicate EFS data to S3, restore in new region

**Current State:** Single region deployment. For true DR, implement multi-region.

---

## Appendix

### Useful Commands Reference

```bash
# --- EKS Cluster ---
aws eks describe-cluster --name mini-xdr-cluster --region us-east-1
aws eks update-kubeconfig --name mini-xdr-cluster --region us-east-1
eksctl get cluster --name mini-xdr-cluster --region us-east-1

# --- Kubernetes Resources ---
kubectl get all -n mini-xdr
kubectl get pods -n mini-xdr -o wide
kubectl logs -n mini-xdr -l app=mini-xdr-backend --tail=100 -f
kubectl describe pod <pod-name> -n mini-xdr
kubectl exec -it <pod-name> -n mini-xdr -- /bin/bash

# --- Deployments ---
kubectl scale deployment mini-xdr-backend -n mini-xdr --replicas=2
kubectl rollout status deployment/mini-xdr-backend -n mini-xdr
kubectl rollout restart deployment/mini-xdr-backend -n mini-xdr
kubectl rollout undo deployment/mini-xdr-backend -n mini-xdr

# --- Services & Ingress ---
kubectl get svc -n mini-xdr
kubectl get ingress -n mini-xdr
kubectl describe ingress mini-xdr-ingress -n mini-xdr

# --- Storage ---
kubectl get pvc -n mini-xdr
kubectl get storageclass
kubectl describe pvc mini-xdr-models-pvc -n mini-xdr

# --- Secrets & ConfigMaps ---
kubectl get secrets -n mini-xdr
kubectl get configmap -n mini-xdr
kubectl describe secret mini-xdr-secrets -n mini-xdr

# --- Events & Monitoring ---
kubectl get events -n mini-xdr --sort-by='.lastTimestamp'
kubectl top nodes
kubectl top pods -n mini-xdr

# --- ECR ---
aws ecr describe-repositories --region us-east-1
aws ecr list-images --repository-name mini-xdr-backend --region us-east-1
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 116912495274.dkr.ecr.us-east-1.amazonaws.com

# --- ALB ---
aws elbv2 describe-load-balancers --region us-east-1
aws elbv2 describe-target-groups --region us-east-1
aws elbv2 describe-target-health --target-group-arn <tg-arn> --region us-east-1

# --- RDS ---
aws rds describe-db-instances --region us-east-1
aws rds describe-db-snapshots --db-instance-identifier mini-xdr-postgres --region us-east-1
aws rds create-db-snapshot --db-snapshot-identifier <name> --db-instance-identifier mini-xdr-postgres --region us-east-1

# --- EFS ---
aws efs describe-file-systems --region us-east-1
aws efs describe-mount-targets --file-system-id fs-0109cfbea9b55373c --region us-east-1

# --- Security Groups ---
aws ec2 describe-security-groups --group-ids <sg-id> --region us-east-1
aws ec2 authorize-security-group-ingress --group-id <sg-id> --protocol tcp --port 8000 --source-group <source-sg-id> --region us-east-1
```

### Environment Variables Reference

**Backend Pod:**
```bash
DATABASE_URL                 # PostgreSQL connection string
ENVIRONMENT                  # production
ML_MODEL_PATH               # /app/models
OPENAI_API_KEY              # OpenAI API key
XAI_API_KEY                 # Alternative AI API key
REDIS_URL                   # Redis connection (optional)
ABUSEIPDB_API_KEY           # Threat intelligence API
VIRUSTOTAL_API_KEY          # Malware scanning API
```

**Frontend Pod:**
```bash
NEXT_PUBLIC_API_URL         # http://mini-xdr-backend-service:8000
```

### Important ARNs & IDs

```
AWS Account ID:              116912495274
Region:                      us-east-1

VPC ID:                      vpc-0d474acd38d418e98
EKS Cluster:                 mini-xdr-cluster
RDS Instance:                mini-xdr-postgres
EFS File System:             fs-0109cfbea9b55373c
KMS Key:                     431cb645-f4d9-41f6-8d6e-6c26c79c5c04

ECR Backend:                 116912495274.dkr.ecr.us-east-1.amazonaws.com/mini-xdr-backend
ECR Frontend:                116912495274.dkr.ecr.us-east-1.amazonaws.com/mini-xdr-frontend

ALB:                         k8s-minixdr-minixdri-dc5fc1df8b
ALB ARN:                     arn:aws:elasticloadbalancing:us-east-1:116912495274:loadbalancer/app/k8s-minixdr-minixdri-dc5fc1df8b/fb05f47242292703

Security Groups:
  ALB:                       sg-0e958a76b787f7689 (mini-xdr-alb-sg)
  EKS Nodes:                 sg-0beefcaa22b6dc37e, sg-059f716b6776b2f6c
  EFS:                       sg-0bd21dae4146cde52 (mini-xdr-efs-sg)
```

### Contacts & Support

**AWS Support:**
- Console: https://console.aws.amazon.com/support/
- Phone: 1-888-759-3840 (Basic support included)

**Mini-XDR Documentation:**
- GitHub: https://github.com/your-org/mini-xdr
- Issues: https://github.com/your-org/mini-xdr/issues

**Monitoring:**
- CloudWatch: https://console.aws.amazon.com/cloudwatch/
- EKS Console: https://console.aws.amazon.com/eks/

---

## Summary

This Mini-XDR deployment on AWS provides a production-ready Extended Detection and Response platform with the following highlights:

**✓ High Availability:**
- Multi-AZ deployment (us-east-1a, us-east-1b, us-east-1c)
- ALB with health checks
- Auto-scaling (HPA) for frontend and backend

**✓ Security:**
- IP whitelist (24.11.0.176/32)
- Private subnets for compute
- Encrypted storage (EFS, RDS, EBS via KMS)
- Security groups controlling all traffic flows

**✓ Scalability:**
- Kubernetes-based orchestration
- Horizontal pod autoscaling
- Shared EFS for ML models
- RDS for centralized data

**✓ Monitoring:**
- CloudWatch Logs integration
- Kubernetes metrics (kubectl top)
- ALB target health monitoring
- Application health checks

**✓ Cost Optimized:**
- t3.medium nodes (right-sized)
- Single replica backend (resource constrained)
- 7-day backup retention
- Potential savings: ~$100/month with optimizations

**Current Status: PRODUCTION READY**
- All services healthy
- ALB responding
- Database connected
- Storage mounted

**Access Now:**
http://k8s-minixdr-minixdri-dc5fc1df8b-1132128475.us-east-1.elb.amazonaws.com

---

**Document Version:** 1.1
**Last Updated:** October 11, 2025 (Added CSP fix documentation and frontend v1.0.1 updates)
**Next Review:** November 11, 2025
