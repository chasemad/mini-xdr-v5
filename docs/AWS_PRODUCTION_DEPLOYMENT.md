# Mini-XDR AWS Production Deployment Guide

## Overview

This guide provides complete instructions for deploying Mini-XDR to AWS EKS in a production-ready configuration that:
- ✅ **Excludes training data** (9.6GB not uploaded)
- ✅ **Works within quota limits** (8 vCPU standard instance quota)
- ✅ **CPU-only ML inference** (no GPU required)
- ✅ **All capabilities maintained** (pre-trained models included)
- ✅ **Production-ready** (secure, scalable, monitored)

## Quick Start

### Prerequisites

1. **AWS Account** with:
   - 8 vCPU quota for standard EC2 instances (t3.medium)
   - Billing enabled
   - IAM permissions for EKS, EC2, VPC, ECR

2. **Local Tools**:
   ```bash
   # Check installations
   aws --version        # AWS CLI 2.x
   docker --version     # Docker 20.x+
   kubectl version      # Kubernetes 1.28+
   eksctl version       # eksctl 0.150+
   helm version         # Helm 3.x
   ```

3. **AWS Configuration**:
   ```bash
   aws configure
   # Enter:
   #   AWS Access Key ID
   #   AWS Secret Access Key
   #   Default region: us-east-1
   #   Output format: json
   ```

### One-Command Deployment

```bash
# Full deployment (30-40 minutes)
./deploy-aws-production.sh
```

That's it! The script will:
1. ✅ Check quotas and prerequisites
2. ✅ Set up infrastructure (VPC, EFS, KMS)
3. ✅ Build CPU-only Docker images (no training data)
4. ✅ Create EKS cluster (2 t3.medium nodes)
5. ✅ Deploy application with ALB
6. ✅ Provide access URLs

## Step-by-Step Deployment

If you prefer manual control or need to troubleshoot:

### Step 1: Infrastructure Setup (5-10 minutes)

```bash
./infrastructure/aws/setup-infrastructure.sh
```

This creates:
- VPC with public/private subnets
- KMS key for secrets encryption
- EFS file system for shared model storage
- Security groups

**Quota Check**: Script automatically verifies you have sufficient quotas.

### Step 2: Build and Push Images (10-15 minutes)

```bash
./infrastructure/aws/build-and-push-images.sh
```

This:
- Creates ECR repositories
- Builds production Docker images:
  - **Backend**: CPU-only PyTorch, no training data (~200MB)
  - **Frontend**: Optimized Next.js standalone (~150MB)
- Verifies no training data in images
- Pushes to ECR with tags (latest + git SHA)

### Step 3: Create EKS Cluster (15-20 minutes)

```bash
./infrastructure/aws/deploy-eks-cluster.sh
```

This creates:
- EKS 1.31 cluster
- 2x t3.medium nodes (4 vCPUs total, within quota)
- AWS Load Balancer Controller
- EBS and EFS CSI drivers
- IAM roles for service accounts

### Step 4: Deploy Application (5-10 minutes)

```bash
./infrastructure/aws/deploy-to-eks.sh
```

This deploys:
- Backend (2 pods, CPU-only ML inference)
- Frontend (2 pods)
- Persistent volumes (EBS + EFS)
- Application Load Balancer
- Auto-scaling policies

## Architecture

### Resource Configuration

| Component | Instance Type | Count | vCPUs | Memory | Storage |
|-----------|--------------|-------|-------|--------|---------|
| EKS Nodes | t3.medium | 2-4 | 2/node | 4GB/node | 30GB gp3 |
| Backend Pods | - | 2-4 | 0.5-1/pod | 768Mi-1.5Gi | - |
| Frontend Pods | - | 2-4 | 0.1-0.25/pod | 256Mi-512Mi | - |
| Data Volume | EBS gp3 | 1 | - | - | 10GB |
| Models Volume | EFS | 1 | - | - | 5GB |

**Total vCPU Usage**: 4-8 vCPUs (within 8 vCPU quota)

### Optimizations for Quota Constraints

1. **CPU-Only ML**: PyTorch configured for CPU inference
   - No CUDA dependencies
   - Multi-threaded NumPy/MKL operations
   - ~10-100x slower than GPU, but functional

2. **Efficient Resource Allocation**:
   - Backend: 500m-1000m CPU requests/limits
   - Frontend: 100m-250m CPU requests/limits
   - Autoscaling: 2-4 pods max per deployment

3. **No Training Data**:
   - Excluded 9.6GB training data via .dockerignore
   - Only pre-trained models included (~92MB)
   - Models loaded on startup, not trained

## Cost Breakdown

| Service | Monthly Cost |
|---------|--------------|
| EKS Control Plane | $73.00 |
| 2x t3.medium (on-demand) | $60.00 |
| EBS gp3 (15GB) | $1.20 |
| EFS (5GB) | $1.50 |
| NAT Gateway | $32.00 |
| Application Load Balancer | $16.00 |
| Data Transfer (estimated) | $10.00 |
| **Total** | **~$194/month** |

### Cost Optimization Tips

1. **Use Spot Instances**: Save ~70% on compute
   ```yaml
   # In eks-cluster-config-production.yaml, uncomment:
   spot: true
   spotAllocationStrategy: capacity-optimized
   ```

2. **Auto-scale Down**: Cluster autoscaler scales to min (2 nodes) when idle

3. **Single NAT Gateway**: Already configured (shared across AZs)

4. **Set Budget Alerts**:
   ```bash
   aws budgets create-budget \
     --account-id YOUR_ACCOUNT_ID \
     --budget '{"BudgetName":"MiniXDR","BudgetLimit":{"Amount":"200","Unit":"USD"},"BudgetType":"COST","TimeUnit":"MONTHLY"}' \
     --notifications-with-subscribers '{"Notification":{"NotificationType":"ACTUAL","ComparisonOperator":"GREATER_THAN","Threshold":80},"Subscribers":[{"SubscriptionType":"EMAIL","Address":"your@email.com"}]}'
   ```

## Deployment Scenarios

### Scenario 1: First-Time Deployment

```bash
./deploy-aws-production.sh
```

### Scenario 2: Update Application Only

After code changes:

```bash
# Rebuild and redeploy (skip infra and cluster)
./deploy-aws-production.sh --skip-infra --skip-cluster
```

Or manually:

```bash
./infrastructure/aws/build-and-push-images.sh
./infrastructure/aws/deploy-to-eks.sh
```

### Scenario 3: Scale Up/Down

```bash
# Scale backend to 3 replicas
kubectl scale deployment/mini-xdr-backend --replicas=3 -n mini-xdr

# Scale frontend to 3 replicas
kubectl scale deployment/mini-xdr-frontend --replicas=3 -n mini-xdr

# Check autoscaler status
kubectl get hpa -n mini-xdr
```

### Scenario 4: Clean Up Everything

```bash
# Delete application
kubectl delete namespace mini-xdr

# Delete EKS cluster (takes 10-15 min)
eksctl delete cluster --name mini-xdr-cluster --region us-east-1

# Delete infrastructure
aws efs delete-file-system --file-system-id fs-xxxxx
aws kms schedule-key-deletion --key-id xxxxx --pending-window-in-days 7

# Delete ECR images
aws ecr delete-repository --repository-name mini-xdr-backend --force
aws ecr delete-repository --repository-name mini-xdr-frontend --force
```

## Verification

### Check Deployment Status

```bash
# Get all resources
kubectl get all -n mini-xdr

# Check pod logs
kubectl logs -f deployment/mini-xdr-backend -n mini-xdr
kubectl logs -f deployment/mini-xdr-frontend -n mini-xdr

# Check ingress
kubectl get ingress -n mini-xdr

# Get ALB DNS
ALB_DNS=$(kubectl get ingress mini-xdr-ingress -n mini-xdr -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')
echo "Access at: http://$ALB_DNS"
```

### Test Endpoints

```bash
# Health check
curl http://$ALB_DNS/api/health

# API docs
open http://$ALB_DNS/api/docs

# Frontend
open http://$ALB_DNS
```

### Verify ML Models

```bash
# Check backend logs for model loading
kubectl logs deployment/mini-xdr-backend -n mini-xdr | grep -i "model"

# Should see:
# ✅ Isolation Forest model loaded
# ✅ LSTM autoencoder model loaded
# ✅ Enhanced ML ensemble initialized
```

## Troubleshooting

### Issue: Pods Not Starting

```bash
# Check pod status
kubectl describe pod <pod-name> -n mini-xdr

# Common causes:
# 1. Image pull error: Check ECR permissions
# 2. Resource limits: Nodes at capacity
# 3. Mount errors: Check EFS/EBS status
```

### Issue: Out of vCPU Quota

```bash
# Check current usage
aws service-quotas get-service-quota \
  --service-code ec2 \
  --quota-code L-1216C47A \
  --region us-east-1

# Reduce replicas temporarily
kubectl scale deployment/mini-xdr-backend --replicas=1 -n mini-xdr
kubectl scale deployment/mini-xdr-frontend --replicas=1 -n mini-xdr
```

### Issue: Slow ML Inference

This is expected with CPU-only inference. To improve performance:

1. **Increase CPU allocation**:
   ```yaml
   # In backend-deployment-production.yaml
   resources:
     requests:
       cpu: "1000m"  # Increase from 500m
     limits:
       cpu: "2000m"  # Increase from 1000m
   ```

2. **Reduce model ensemble**:
   ```python
   # In backend/app/ml_engine.py, disable some models
   # Comment out LSTM or OneClassSVM to reduce computation
   ```

### Issue: ALB Not Provisioning

```bash
# Check AWS Load Balancer Controller logs
kubectl logs -n kube-system deployment/aws-load-balancer-controller

# Verify IAM roles
eksctl get iamserviceaccount --cluster mini-xdr-cluster --region us-east-1
```

## Security Best Practices

### 1. Enable Secrets Encryption

Already configured with KMS encryption at rest.

### 2. Update Secrets

```bash
# Update OpenAI API key
kubectl create secret generic mini-xdr-secrets \
  --from-literal=openai-api-key=sk-xxxxx \
  --namespace=mini-xdr \
  --dry-run=client -o yaml | kubectl apply -f -

# Restart pods to pick up new secret
kubectl rollout restart deployment/mini-xdr-backend -n mini-xdr
```

### 3. Restrict ALB Access

```bash
# Add IP whitelist to ingress
kubectl annotate ingress mini-xdr-ingress \
  -n mini-xdr \
  alb.ingress.kubernetes.io/inbound-cidrs='YOUR.IP.ADDRESS/32'
```

### 4. Enable HTTPS

```bash
# Request ACM certificate
aws acm request-certificate \
  --domain-name your-domain.com \
  --validation-method DNS

# Update ingress with HTTPS
kubectl annotate ingress mini-xdr-ingress \
  -n mini-xdr \
  alb.ingress.kubernetes.io/listen-ports='[{"HTTPS":443}]' \
  alb.ingress.kubernetes.io/certificate-arn='arn:aws:acm:...'
```

## Monitoring

### CloudWatch Logs

```bash
# View logs in CloudWatch
aws logs tail /aws/eks/mini-xdr-cluster/cluster --follow
```

### Metrics

```bash
# Install metrics server
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml

# View resource usage
kubectl top nodes
kubectl top pods -n mini-xdr
```

### Alarms

```bash
# Create CloudWatch alarm for high CPU
aws cloudwatch put-metric-alarm \
  --alarm-name mini-xdr-high-cpu \
  --alarm-description "Alert when CPU > 80%" \
  --metric-name CPUUtilization \
  --namespace AWS/EKS \
  --statistic Average \
  --period 300 \
  --threshold 80 \
  --comparison-operator GreaterThanThreshold \
  --evaluation-periods 2
```

## Maintenance

### Update Application

```bash
# Pull latest code
git pull

# Rebuild and deploy
./infrastructure/aws/build-and-push-images.sh
kubectl rollout restart deployment/mini-xdr-backend -n mini-xdr
kubectl rollout restart deployment/mini-xdr-frontend -n mini-xdr
```

### Update EKS

```bash
# Check available updates
eksctl get cluster --name mini-xdr-cluster --region us-east-1

# Upgrade cluster (e.g., 1.31 -> 1.32)
eksctl upgrade cluster --name mini-xdr-cluster --region us-east-1 --approve
```

### Backup Data

```bash
# Create EBS snapshot
VOLUME_ID=$(kubectl get pv -o jsonpath='{.items[?(@.spec.claimRef.name=="mini-xdr-data-pvc")].spec.awsElasticBlockStore.volumeID}' | cut -d'/' -f4)
aws ec2 create-snapshot --volume-id $VOLUME_ID --description "Mini-XDR data backup $(date +%Y%m%d)"
```

## Support

### Logs and Diagnostics

All deployment logs are saved to `/tmp/mini-xdr-*.log`

```bash
# Find latest log
ls -lt /tmp/mini-xdr-*.log | head -1

# View deployment log
cat /tmp/mini-xdr-deployment-YYYYMMDD-HHMMSS.log
```

### Common Commands

```bash
# Get pod shell
kubectl exec -it deployment/mini-xdr-backend -n mini-xdr -- /bin/bash

# Port forward for local access
kubectl port-forward svc/mini-xdr-frontend-service 3000:3000 -n mini-xdr
kubectl port-forward svc/mini-xdr-backend-service 8000:8000 -n mini-xdr

# View events
kubectl get events -n mini-xdr --sort-by='.lastTimestamp'
```

## FAQ

**Q: Can I use GPU instances?**
A: Yes, if you have GPU quota. Change `instanceType` to `g4dn.xlarge` in `eks-cluster-config-production.yaml` and rebuild backend image without CPU-only flags.

**Q: What if I need more than 8 vCPUs?**
A: Request quota increase via AWS Service Quotas console. Approval typically takes 24-48 hours.

**Q: Can I deploy to multiple regions?**
A: Yes, set `AWS_REGION` environment variable before running deployment scripts.

**Q: How do I add custom models?**
A: Copy `.pth` files to `models/` directory before building images. They'll be included automatically.

**Q: Is this production-ready?**
A: Yes, with proper security hardening (HTTPS, network policies, IAM) and monitoring.

## Summary

You now have a complete, production-ready Mini-XDR deployment running on AWS that:
- ✅ Works within your 8 vCPU quota
- ✅ Uses CPU-only ML inference
- ✅ Excludes training data (only pre-trained models)
- ✅ Costs ~$200/month
- ✅ Scales automatically based on load
- ✅ Includes all XDR capabilities

For questions or issues, check logs in `/tmp/` or review Kubernetes events.

**Happy Detecting! 🛡️**
