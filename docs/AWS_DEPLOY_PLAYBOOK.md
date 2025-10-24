# Mini-XDR AWS Deployment Playbook

**Quick Reference Guide for Deploying to AWS EKS**

Last Updated: October 23, 2025  
Status: ✅ Production Ready

---

## Quick Start

### One-Command Deployment

```bash
# Build, push, and deploy everything
./scripts/build-and-deploy-aws.sh --all --push --deploy

# Build and push backend only
./scripts/build-and-deploy-aws.sh --backend --push

# Just deploy (use existing images)
./scripts/build-and-deploy-aws.sh --deploy
```

---

## Prerequisites

### 1. AWS Configuration
```bash
# Configure AWS CLI
aws configure
# Enter: Access Key ID, Secret Access Key, Region (us-east-1)

# Update kubeconfig for EKS
aws eks update-kubeconfig --name mini-xdr-cluster --region us-east-1

# Verify connection
kubectl get nodes
```

### 2. Docker Setup
```bash
# Ensure Docker is running
docker info

# For Mac M1/M2, buildx will automatically handle AMD64 cross-compilation
```

---

## Deployment Scenarios

### Scenario 1: Deploy Code Changes (Backend)

**When:** You've updated Python code, added endpoints, fixed bugs

```bash
# 1. Test locally first
cd backend
source venv/bin/activate
pytest tests/

# 2. Build and deploy
cd ..
./scripts/build-and-deploy-aws.sh --backend --push --deploy

# 3. Monitor rollout
kubectl rollout status deployment/mini-xdr-backend -n mini-xdr
kubectl logs -f deployment/mini-xdr-backend -n mini-xdr
```

**Rollback if needed:**
```bash
kubectl rollout undo deployment/mini-xdr-backend -n mini-xdr
```

### Scenario 2: Deploy UI Changes (Frontend)

**When:** You've updated React components, styling, or pages

```bash
# 1. Test locally
cd frontend
npm run build
npm start

# 2. Build and deploy
cd ..
./scripts/build-and-deploy-aws.sh --frontend --push --deploy

# 3. Verify
kubectl get pods -n mini-xdr
```

### Scenario 3: Deploy Both Backend & Frontend

**When:** Major release with changes to both services

```bash
# Build everything
./scripts/build-and-deploy-aws.sh --all --push --deploy --version 1.1.0

# Monitor both
kubectl get pods -n mini-xdr -w
```

### Scenario 4: Configuration Changes Only

**When:** You've updated K8s manifests without code changes

```bash
# Apply updated manifests
kubectl apply -f k8s/backend-deployment.yaml
kubectl apply -f k8s/frontend-deployment.yaml

# Or just restart
kubectl rollout restart deployment/mini-xdr-backend -n mini-xdr
```

### Scenario 5: Emergency Rollback

**When:** New deployment is failing

```bash
# Rollback backend
kubectl rollout undo deployment/mini-xdr-backend -n mini-xdr

# Rollback frontend
kubectl rollout undo deployment/mini-xdr-frontend -n mini-xdr

# Verify health
kubectl get pods -n mini-xdr
curl http://ALB-URL/health
```

---

## Manual Build Process

If you need more control or the script fails:

### Backend

```bash
cd backend

# Build for AMD64
docker buildx build \
  --platform linux/amd64 \
  --file Dockerfile \
  --tag 116912495274.dkr.ecr.us-east-1.amazonaws.com/mini-xdr-backend:v1.0.2 \
  --load \
  .

# Login to ECR
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin \
  116912495274.dkr.ecr.us-east-1.amazonaws.com

# Push
docker push 116912495274.dkr.ecr.us-east-1.amazonaws.com/mini-xdr-backend:v1.0.2

# Deploy
kubectl set image deployment/mini-xdr-backend \
  backend=116912495274.dkr.ecr.us-east-1.amazonaws.com/mini-xdr-backend:v1.0.2 \
  -n mini-xdr

# Wait for rollout
kubectl rollout status deployment/mini-xdr-backend -n mini-xdr
```

### Frontend

```bash
cd frontend

# Build
docker buildx build \
  --platform linux/amd64 \
  --file Dockerfile \
  --tag 116912495274.dkr.ecr.us-east-1.amazonaws.com/mini-xdr-frontend:v1.0.2 \
  --load \
  .

# Push (after ECR login)
docker push 116912495274.dkr.ecr.us-east-1.amazonaws.com/mini-xdr-frontend:v1.0.2

# Deploy
kubectl set image deployment/mini-xdr-frontend \
  frontend=116912495274.dkr.ecr.us-east-1.amazonaws.com/mini-xdr-frontend:v1.0.2 \
  -n mini-xdr
```

---

## Health Checks

### Check Pod Status
```bash
# All pods
kubectl get pods -n mini-xdr

# Detailed pod info
kubectl describe pod <pod-name> -n mini-xdr

# Logs
kubectl logs -f deployment/mini-xdr-backend -n mini-xdr
```

### Check Resource Usage
```bash
kubectl top pods -n mini-xdr
kubectl top nodes
```

### Check Application Health
```bash
# Get ALB URL
ALB_URL=$(kubectl get ingress mini-xdr-ingress -n mini-xdr -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')

# Test health endpoint
curl http://$ALB_URL/health

# Test API
curl http://$ALB_URL/api/docs

# Visit frontend
open http://$ALB_URL
```

---

## Troubleshooting

### Pod Won't Start

**Check events:**
```bash
kubectl describe pod <pod-name> -n mini-xdr
kubectl get events -n mini-xdr --sort-by='.lastTimestamp'
```

**Common causes:**
- Insufficient resources → Scale down or add nodes
- Image pull failure → Check ECR login and image exists
- Health probe timeout → Increase initialDelaySeconds

### CrashLoopBackOff

**Check logs:**
```bash
kubectl logs <pod-name> -n mini-xdr --previous
```

**Common causes:**
- Database connection failure → Verify RDS endpoint in secrets
- Missing environment variables → Check secrets exist
- Application error → Review logs for stack trace

### ImagePullBackOff

**Verify image:**
```bash
aws ecr describe-images \
  --repository-name mini-xdr-backend \
  --region us-east-1
```

**Re-authenticate:**
```bash
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin \
  116912495274.dkr.ecr.us-east-1.amazonaws.com
```

### Out of Resources

**Check capacity:**
```bash
kubectl describe nodes
```

**Options:**
1. Scale down replicas
2. Add more nodes
3. Reduce resource requests

---

## Environment Variables

### Backend Environment

Set via K8s secrets (`mini-xdr-secrets`):
```bash
kubectl edit secret mini-xdr-secrets -n mini-xdr
```

Required variables:
- `DATABASE_URL` - PostgreSQL connection string
- `OPENAI_API_KEY` - OpenAI API key
- `REDIS_URL` - Redis connection (if used)
- `ABUSEIPDB_API_KEY` - Threat intel API
- `VIRUSTOTAL_API_KEY` - Malware scanning API

### Frontend Environment

Set via deployment environment:
- `NEXT_PUBLIC_API_URL` - Backend service URL (http://mini-xdr-backend-service:8000)

---

## Database Migrations

### Run Migrations on RDS

```bash
# Get DATABASE_URL from secret
DATABASE_URL=$(kubectl get secret mini-xdr-secrets -n mini-xdr -o jsonpath='{.data.DATABASE_URL}' | base64 -d)

# Export for alembic
export DATABASE_URL

# Run migrations
cd backend
source venv/bin/activate
alembic upgrade head

# Verify
alembic current
```

### From within pod:

```bash
kubectl exec -it deployment/mini-xdr-backend -n mini-xdr -- /bin/bash
cd /app/backend
alembic upgrade head
```

---

## Scaling

### Manual Scaling

```bash
# Scale backend to 2 replicas (if resources allow)
kubectl scale deployment mini-xdr-backend -n mini-xdr --replicas=2

# Scale frontend to 3 replicas
kubectl scale deployment mini-xdr-frontend -n mini-xdr --replicas=3
```

### Auto-Scaling (HPA)

Currently disabled due to resource constraints. To re-enable:

```bash
kubectl create -f k8s/hpa.yaml
```

Or create manually:
```bash
kubectl autoscale deployment mini-xdr-backend \
  --cpu-percent=70 \
  --min=1 \
  --max=3 \
  -n mini-xdr
```

---

## Cost Management

### Current Monthly Cost: ~$287

**Breakdown:**
- EKS Control Plane: $73
- EC2 (2× t3.medium): $61
- NAT Gateways (3): $98
- RDS (db.t3.micro): $13
- ALB: $21
- Storage (EFS + EBS): $5
- Other: $16

### Cost Optimization Options:

1. **Single NAT Gateway:** Save ~$65/month
2. **Spot Instances:** Save ~$43/month
3. **Fargate instead of EC2:** Save ~$40/month
4. **Aurora Serverless v2:** Save ~$10/month

---

## Monitoring

### CloudWatch Logs

```bash
# View logs in CloudWatch
aws logs tail /aws/eks/mini-xdr-cluster/cluster --follow

# ALB logs (if enabled)
aws logs tail /aws/elasticloadbalancing/app/mini-xdr-alb --follow
```

### Kubernetes Metrics

```bash
# Pod metrics
kubectl top pods -n mini-xdr

# Node metrics
kubectl top nodes

# Describe for detailed info
kubectl describe pod <pod-name> -n mini-xdr
```

---

## Next Steps

### After Successful Deployment:

1. ✅ Verify all pods are running and healthy
2. ✅ Test health endpoint via ALB
3. ✅ Run end-to-end onboarding flow
4. ✅ Check application logs for errors
5. ✅ Monitor resource usage
6. ⏳ Upload ML models to EFS (if needed)
7. ⏳ Enable HTTPS with ACM certificate
8. ⏳ Set up CloudWatch dashboards
9. ⏳ Configure backups

### For Production:

- Enable HTTPS/TLS on ALB
- Set up proper monitoring and alerting
- Configure automated backups
- Implement CI/CD pipeline
- Add more nodes for redundancy
- Enable HPA with proper minReplicas

---

## Quick Reference Commands

```bash
# Common operations
kubectl get pods -n mini-xdr                    # View pods
kubectl logs -f <pod-name> -n mini-xdr          # Stream logs
kubectl describe pod <pod-name> -n mini-xdr     # Pod details
kubectl top pods -n mini-xdr                    # Resource usage
kubectl rollout restart deployment/<name> -n mini-xdr  # Restart
kubectl rollout undo deployment/<name> -n mini-xdr     # Rollback
kubectl scale deployment/<name> --replicas=N -n mini-xdr  # Scale

# Deployment script
./scripts/build-and-deploy-aws.sh --all --push --deploy  # Full deploy
./scripts/build-and-deploy-aws.sh --backend --push       # Backend only
./scripts/build-and-deploy-aws.sh --deploy               # Deploy only
```

---

## Support

**Documentation:**
- AWS Deployment Guide: `/docs/AWS_DEPLOYMENT_GUIDE.md`
- Stabilization Report: `/docs/AWS_STABILIZATION_REPORT.md`
- Test & Deploy Guide: `/docs/TEST_AND_DEPLOY_GUIDE.md`

**Get Help:**
```bash
# Deployment script help
./scripts/build-and-deploy-aws.sh --help

# Check cluster status
kubectl cluster-info

# View all resources
kubectl get all -n mini-xdr
```

---

**Status:** ✅ **System is stable and operational!**

For issues or questions, review the troubleshooting section or check the AWS Stabilization Report.

