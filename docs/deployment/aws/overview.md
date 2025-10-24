# AWS Deployment – Overview

Mini-XDR is deployed to AWS using **EKS (Elastic Kubernetes Service)** with containerized backend and
frontend services. Docker images are built on a dedicated EC2 instance and stored in ECR (Elastic
Container Registry).

## Current Production Environment

| Component | Details |
| --- | --- |
| **EKS Cluster** | `mini-xdr-cluster` in `us-east-1` |
| **Namespace** | `mini-xdr` |
| **Load Balancer** | Application Load Balancer (ALB) managed by AWS Load Balancer Controller |
| **ALB URL** | `http://k8s-minixdr-minixdri-dc5fc1df8b-1132128475.us-east-1.elb.amazonaws.com` |
| **Current Version** | `v1.1.8` (CloudAsset model + seamless onboarding) |
| **ECR Repositories** | `mini-xdr-backend`, `mini-xdr-frontend` |
| **Build Instance** | EC2 t3.medium (54.82.186.21, Amazon Linux 2023) |
| **Database** | RDS PostgreSQL with Alembic migrations |
| **Secrets** | AWS Secrets Manager integration |
| **Security** | IP-restricted access (configured via ingress annotations) |

## Architecture

```
┌─────────────┐
│   Internet  │
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│  ALB (Port 80)  │
│  kubernetes.io/ │
│  ingress        │
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌─────────┐ ┌──────────┐
│Frontend │ │ Backend  │
│Service  │ │ Service  │
│ClusterIP│ │ClusterIP │
└────┬────┘ └────┬─────┘
     │           │
     ▼           ▼
┌─────────┐ ┌──────────┐
│Frontend │ │ Backend  │
│Pods     │ │ Pods     │
│(3000)   │ │ (8000)   │
└─────────┘ └──────────┘
```

## Prerequisites

- **AWS CLI v2** authenticated (`aws sts get-caller-identity` succeeds)
- **kubectl** configured with EKS cluster credentials
- **IAM permissions** for EKS, ECR, EC2, Secrets Manager, and RDS
- **SSH key pair** for EC2 build instance (e.g., `mini-xdr-eks-key.pem`)
- **Docker** installed on EC2 build instance

## Deployment Workflow

1. **Code Changes** → Commit to Git repository
2. **Build Images** → SSH to EC2 build instance, pull latest code, build Docker images
3. **Push to ECR** → Tag and push images with version tags (e.g., `1.1.0`, `59c483b`, `latest`)
4. **Update K8s** → Patch deployments to use new image digest or force pull with `imagePullPolicy: Always`
5. **Verify** → Check pod status, logs, and test application endpoints

## Key Configuration Files

| File | Purpose |
| --- | --- |
| `backend/Dockerfile` | Multi-stage backend image with Python dependencies and ML models |
| `frontend/Dockerfile` | Multi-stage frontend image with Next.js build |
| `k8s/backend-deployment.yaml` | Backend deployment, service, and configuration |
| `k8s/frontend-deployment.yaml` | Frontend deployment and service |
| `k8s/ingress.yaml` | ALB ingress configuration |
| `buildspec-backend.yml` | CodeBuild CI/CD for backend (optional automation) |
| `buildspec-frontend.yml` | CodeBuild CI/CD for frontend (optional automation) |

## Why EC2 for Image Builds?

Docker builds are performed on a dedicated EC2 instance (Amazon Linux 2023 on x86_64) rather than
locally because:

1. **Architecture Consistency**: Ensures linux/amd64 images compatible with EKS nodes
2. **Avoid ARM64/AMD64 Issues**: Local M1/M2 Macs produce ARM64 images that may have compatibility issues
3. **Build Speed**: EC2 instance has faster network connectivity to ECR
4. **ML Models**: Backend image includes large ML models (~10GB total), faster to build in AWS

## Deployment Considerations

### ⚠️ Critical Deployment Steps

**Always perform these steps after code deployment:**

1. **Rebuild Docker Images**: Ensure container images include latest code changes
2. **Run Database Migrations**: Execute `alembic upgrade head` for schema changes
3. **Verify Security Groups**: Ensure ALB security groups match ingress annotations
4. **Check Target Health**: Confirm ALB targets are healthy before considering deployment complete

### Recent Architecture Updates

- **Version 1.1.8**: Added CloudAsset model for seamless onboarding workflow
- **Database Schema**: Extended organizations table with onboarding flow tracking
- **Security**: IP-restricted ALB access with dynamic security group management
- **Monitoring**: Enhanced health checks and readiness probes for ML model loading

### Common Failure Points

- **Database Schema Drift**: Local SQLite ≠ AWS RDS PostgreSQL
- **Image Version Mismatch**: Deployed containers running outdated code
- **Security Group Sync**: Ingress annotations ≠ ALB security group rules
- **ALB Target Registration**: Controller may deregister targets during rollouts

Refer to [operations](operations.md) for detailed deployment commands and
[troubleshooting](troubleshooting.md) for common issues.
