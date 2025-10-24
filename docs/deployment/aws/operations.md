# AWS Deployment – Operations

This guide covers day-to-day operations for the Mini-XDR EKS deployment, including building images,
deploying updates, and verifying the system.

## 1. Deploying New Code

### Step 1: Commit and Tag Code Changes

```bash
# Commit changes to Git
git add .
git commit -m "feat: your feature description"

# Tag with version (optional but recommended)
git tag v1.1.1
git push origin main --tags
```

### Step 2: SSH to EC2 Build Instance

```bash
# SSH to the build instance
ssh -i ~/.ssh/mini-xdr-eks-key.pem ec2-user@<EC2-IP>

# Navigate to repository
cd /home/ec2-user/mini-xdr-v2

# Pull latest code
git fetch --all --tags
git checkout main  # or specific tag like v1.1.0
git pull origin main
```

### Step 3: Build Docker Images

**Backend Image:**

```bash
cd /home/ec2-user/mini-xdr-v2/backend

# Login to ECR
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin \
  116912495274.dkr.ecr.us-east-1.amazonaws.com

# Build with version tags
docker build \
  --build-arg VERSION="1.1.1" \
  --build-arg VCS_REF="$(git rev-parse --short HEAD)" \
  --build-arg BUILD_DATE="$(date -u +"%Y-%m-%dT%H:%M:%SZ")" \
  -t 116912495274.dkr.ecr.us-east-1.amazonaws.com/mini-xdr-backend:1.1.1 \
  -t 116912495274.dkr.ecr.us-east-1.amazonaws.com/mini-xdr-backend:latest \
  .
```

**Frontend Image:**

```bash
cd /home/ec2-user/mini-xdr-v2/frontend

# Build with ALB URL (IMPORTANT: Use actual ALB endpoint)
docker build \
  --build-arg NEXT_PUBLIC_API_BASE="http://k8s-minixdr-minixdri-dc5fc1df8b-1132128475.us-east-1.elb.amazonaws.com" \
  --build-arg NEXT_PUBLIC_API_URL="http://k8s-minixdr-minixdri-dc5fc1df8b-1132128475.us-east-1.elb.amazonaws.com" \
  --build-arg VERSION="1.1.1" \
  --build-arg VCS_REF="$(git rev-parse --short HEAD)" \
  --build-arg BUILD_DATE="$(date -u +"%Y-%m-%dT%H:%M:%SZ")" \
  -t 116912495274.dkr.ecr.us-east-1.amazonaws.com/mini-xdr-frontend:1.1.1 \
  -t 116912495274.dkr.ecr.us-east-1.amazonaws.com/mini-xdr-frontend:latest \
  .
```

### Step 4: Push Images to ECR

```bash
# Push backend
docker push 116912495274.dkr.ecr.us-east-1.amazonaws.com/mini-xdr-backend:1.1.1
docker push 116912495274.dkr.ecr.us-east-1.amazonaws.com/mini-xdr-backend:latest

# Push frontend
docker push 116912495274.dkr.ecr.us-east-1.amazonaws.com/mini-xdr-frontend:1.1.1
docker push 116912495274.dkr.ecr.us-east-1.amazonaws.com/mini-xdr-frontend:latest
```

### Step 5: Update Kubernetes Deployments

**Option A: Force Pull Latest Images (for same tag):**

```bash
# Patch deployments to always pull
kubectl patch deployment mini-xdr-backend -n mini-xdr -p \
  '{"spec":{"template":{"spec":{"containers":[{"name":"backend","imagePullPolicy":"Always"}]}}}}'

kubectl patch deployment mini-xdr-frontend -n mini-xdr -p \
  '{"spec":{"template":{"spec":{"containers":[{"name":"frontend","imagePullPolicy":"Always"}]}}}}'

# Restart deployments
kubectl rollout restart deployment/mini-xdr-backend -n mini-xdr
kubectl rollout restart deployment/mini-xdr-frontend -n mini-xdr
```

**Option B: Update to Specific Version Tag:**

```bash
# Update image tags in deployment files
kubectl set image deployment/mini-xdr-backend \
  backend=116912495274.dkr.ecr.us-east-1.amazonaws.com/mini-xdr-backend:1.1.1 \
  -n mini-xdr

kubectl set image deployment/mini-xdr-frontend \
  frontend=116912495274.dkr.ecr.us-east-1.amazonaws.com/mini-xdr-frontend:1.1.1 \
  -n mini-xdr
```

### Step 6: Verify Deployment

```bash
# Check pod status
kubectl get pods -n mini-xdr

# Watch rollout status
kubectl rollout status deployment/mini-xdr-backend -n mini-xdr
kubectl rollout status deployment/mini-xdr-frontend -n mini-xdr

# Check pod logs
kubectl logs -f deployment/mini-xdr-backend -n mini-xdr --tail=50
kubectl logs -f deployment/mini-xdr-frontend -n mini-xdr --tail=50
```

## 2. Verifying System Health

### Check Pod Status

```bash
# Get all pods in mini-xdr namespace
kubectl get pods -n mini-xdr -o wide

# Expected output: All pods should be Running with 1/1 ready
# NAME                                  READY   STATUS    RESTARTS   AGE
# mini-xdr-backend-xxxxx-xxxxx          1/1     Running   0          5m
# mini-xdr-frontend-xxxxx-xxxxx         1/1     Running   0          5m
```

### Check Services and Ingress

```bash
# Get services
kubectl get svc -n mini-xdr

# Get ingress and ALB URL
kubectl get ingress -n mini-xdr

# Describe ingress for ALB details
kubectl describe ingress mini-xdr-ingress -n mini-xdr
```

### Test Application Endpoints

```bash
# Health check
curl http://k8s-minixdr-minixdri-dc5fc1df8b-1132128475.us-east-1.elb.amazonaws.com/health

# Frontend access (should return HTML)
curl http://k8s-minixdr-minixdri-dc5fc1df8b-1132128475.us-east-1.elb.amazonaws.com/

# Backend API docs
curl http://k8s-minixdr-minixdri-dc5fc1df8b-1132128475.us-east-1.elb.amazonaws.com/docs
```

## 3. Monitoring and Logs

### View Pod Logs

```bash
# Tail backend logs
kubectl logs -f deployment/mini-xdr-backend -n mini-xdr

# Tail frontend logs
kubectl logs -f deployment/mini-xdr-frontend -n mini-xdr

# Get logs from all pods with label
kubectl logs -l app=mini-xdr-backend -n mini-xdr --tail=100

# Get previous container logs (if pod crashed)
kubectl logs <pod-name> -n mini-xdr --previous
```

### Check Resource Usage

```bash
# Pod resource usage
kubectl top pods -n mini-xdr

# Node resource usage
kubectl top nodes
```

### Describe Pods for Events

```bash
# Get detailed pod information and events
kubectl describe pod <pod-name> -n mini-xdr

# Get events in namespace
kubectl get events -n mini-xdr --sort-by='.lastTimestamp'
```

## 4. Database Operations

### Access RDS PostgreSQL

```bash
# Get database connection details from ConfigMap
kubectl get configmap mini-xdr-config -n mini-xdr -o yaml

# Connect via psql (requires network access)
psql postgresql://user:password@rds-endpoint:5432/minixdr

# Run migrations from backend pod
kubectl exec -it deployment/mini-xdr-backend -n mini-xdr -- \
  alembic upgrade head
```

### Critical: Run Database Migrations After Code Deployment

**⚠️ IMPORTANT:** Always run database migrations after deploying code that includes database schema changes.

```bash
# After deploying new backend code with schema changes
kubectl exec -it deployment/mini-xdr-backend -n mini-xdr -- alembic upgrade head

# Check migration status
kubectl logs deployment/mini-xdr-backend -n mini-xdr --tail=10 | grep -i migration

# Restart pods if needed
kubectl rollout restart deployment/mini-xdr-backend -n mini-xdr
```

**Symptoms of missing migrations:**
- 500 Internal Server Error on API endpoints
- `column X does not exist` errors in logs
- Authentication endpoints failing

**Prevention:** Include `alembic upgrade head` in your deployment checklist.

## 5. Secrets Management

### Access AWS Secrets Manager

```bash
# List secrets
aws secretsmanager list-secrets --region us-east-1

# Get secret value
aws secretsmanager get-secret-value \
  --secret-id mini-xdr-backend-secrets \
  --region us-east-1 \
  --query SecretString \
  --output text
```

### Update Kubernetes Secrets

```bash
# Create/update secret from AWS Secrets Manager
kubectl create secret generic mini-xdr-secrets \
  --from-literal=API_KEY=<value> \
  --from-literal=JWT_SECRET_KEY=<value> \
  -n mini-xdr \
  --dry-run=client -o yaml | kubectl apply -f -
```

## 6. Scaling

### Manual Scaling

```bash
# Scale backend replicas
kubectl scale deployment mini-xdr-backend --replicas=5 -n mini-xdr

# Scale frontend replicas
kubectl scale deployment mini-xdr-frontend --replicas=3 -n mini-xdr
```

### Horizontal Pod Autoscaling (HPA)

```bash
# Create HPA for backend (requires metrics-server)
kubectl autoscale deployment mini-xdr-backend \
  --cpu-percent=70 \
  --min=2 \
  --max=10 \
  -n mini-xdr

# Check HPA status
kubectl get hpa -n mini-xdr
```

## 7. Emergency Operations

### Quick Rollback

```bash
# Rollback to previous version
kubectl rollout undo deployment/mini-xdr-backend -n mini-xdr
kubectl rollout undo deployment/mini-xdr-frontend -n mini-xdr

# Check rollout history
kubectl rollout history deployment/mini-xdr-backend -n mini-xdr
```

### Force Pod Restart

```bash
# Delete pod (will be recreated automatically)
kubectl delete pod <pod-name> -n mini-xdr

# Or restart entire deployment
kubectl rollout restart deployment/mini-xdr-backend -n mini-xdr
```

### Access Pod Shell

```bash
# Get shell in backend pod
kubectl exec -it deployment/mini-xdr-backend -n mini-xdr -- /bin/bash

# Get shell in frontend pod
kubectl exec -it deployment/mini-xdr-frontend -n mini-xdr -- /bin/sh
```

## 8. Maintenance

### Rotate Secrets

```bash
# Update secret in AWS Secrets Manager
aws secretsmanager update-secret \
  --secret-id mini-xdr-backend-secrets \
  --secret-string '{"API_KEY":"new-key"}' \
  --region us-east-1

# Restart pods to pick up new secrets
kubectl rollout restart deployment/mini-xdr-backend -n mini-xdr
```

### Clean Up Old Images

```bash
# On EC2 build instance, clean up dangling images
ssh -i ~/.ssh/mini-xdr-eks-key.pem ec2-user@<EC2-IP>
docker image prune -a --filter "until=168h"  # Remove images older than 7 days
```

### Update EKS Cluster

```bash
# Check cluster version
aws eks describe-cluster --name mini-xdr-cluster --region us-east-1

# Update cluster (follow AWS documentation for cluster upgrades)
aws eks update-cluster-version \
  --name mini-xdr-cluster \
  --kubernetes-version 1.29 \
  --region us-east-1
```
