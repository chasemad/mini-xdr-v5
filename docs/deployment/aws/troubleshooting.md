# AWS Deployment – Troubleshooting

This guide covers common issues encountered during EKS deployment and their resolutions.

## Pod Issues

### Backend Pod: "No module named uvicorn"

**Symptom:**
```
/usr/local/bin/python: No module named uvicorn
Pod Status: CrashLoopBackOff
```

**Cause:**
The Dockerfile uses a multi-stage build where Python packages are installed to `/root/.local` in the
builder stage and copied to `/home/xdr/.local` in the production stage. The issue occurs when:

1. The `CMD` uses `python -m uvicorn` which invokes the system Python at `/usr/local/bin/python`
2. The `PYTHONUSERBASE` environment variable is not set, so Python doesn't know to look in `/home/xdr/.local`

**Resolution:**

Update `backend/Dockerfile` to:

1. **Add PYTHONUSERBASE environment variable:**
```dockerfile
ENV PATH=/home/xdr/.local/bin:$PATH \
    PYTHONUSERBASE=/home/xdr/.local
```

2. **Use uvicorn binary directly instead of `python -m uvicorn`:**
```dockerfile
CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--log-level", "info"]
```

See commit `3466bc7` for the complete fix in `backend/Dockerfile:62-63,92-96`.

---

### Frontend Pod: npm cache permission errors

**Symptom:**
```
npm error path /home/xdr/.npm
npm error errno ENOENT
npm error enoent ENOENT: no such file or directory, mkdir '/home/xdr/.npm'
Pod Status: CrashLoopBackOff
```

**Cause:**
The frontend pod tries to install TypeScript at runtime but fails because:

1. The `HOME` environment variable is not set, causing npm to try creating `/.npm`
2. The `/home/xdr/.npm` directory doesn't exist for the non-root user
3. TypeScript is not installed as a production dependency

**Resolution:**

Update `frontend/Dockerfile` to:

1. **Create npm cache directories with proper ownership:**
```dockerfile
RUN addgroup -g 1000 -S xdr 2>/dev/null || true && \
    adduser -u 1000 -S xdr -h /home/xdr 2>/dev/null || true && \
    mkdir -p /home/xdr/.npm /home/xdr/.cache && \
    chown -R xdr:xdr /home/xdr 2>/dev/null || chown -R 1000:1000 /home/xdr
```

2. **Set HOME environment variable:**
```dockerfile
ENV NODE_ENV=production \
    PORT=3000 \
    HOSTNAME="0.0.0.0" \
    NEXT_TELEMETRY_DISABLED=1 \
    HOME=/home/xdr
```

3. **Install TypeScript as production dependency:**
```dockerfile
RUN npm ci --only=production --ignore-scripts && \
    npm install --save-exact typescript@5.9.3 && \
    npm cache clean --force
```

See commit `6616444` for the complete fix in `frontend/Dockerfile:49-79`.

---

### Frontend: Content Security Policy (CSP) Violation

**Symptom:**
```
TypeError: Failed to fetch
Refused to connect to http://localhost:8000 because it violates CSP
```

**Cause:**
Frontend was built with default `NEXT_PUBLIC_API_BASE=http://localhost:8000` instead of the actual
ALB URL. Next.js bakes these environment variables into the build at build-time.

**Resolution:**

Rebuild frontend image with correct ALB URL:

```bash
docker build \
  --build-arg NEXT_PUBLIC_API_BASE="http://k8s-minixdr-minixdri-dc5fc1df8b-1132128475.us-east-1.elb.amazonaws.com" \
  --build-arg NEXT_PUBLIC_API_URL="http://k8s-minixdr-minixdri-dc5fc1df8b-1132128475.us-east-1.elb.amazonaws.com" \
  -t 116912495274.dkr.ecr.us-east-1.amazonaws.com/mini-xdr-frontend:1.1.0 \
  .
```

**Verification:**
Check frontend logs for correct API configuration:
```bash
kubectl logs deployment/mini-xdr-frontend -n mini-xdr | grep "ThreatDataService configuration"
```

Should show:
```
🔧 ThreatDataService configuration: {
  API_BASE_URL: 'http://k8s-minixdr-minixdri-dc5fc1df8b-1132128475.us-east-1.elb.amazonaws.com',
  ...
}
```

---

### Pods Using Cached Images After Push

**Symptom:**
```
Container image already present on machine
Pods still using old version after pushing new image
```

**Cause:**
Kubernetes caches images by tag. When pushing a new image with the same tag (e.g., `latest`),
nodes don't pull the new image if they already have that tag cached.

**Resolution:**

**Option 1: Force image pull policy:**
```bash
kubectl patch deployment mini-xdr-backend -n mini-xdr -p \
  '{"spec":{"template":{"spec":{"containers":[{"name":"backend","imagePullPolicy":"Always"}]}}}}'

kubectl rollout restart deployment/mini-xdr-backend -n mini-xdr
```

**Option 2: Use image digest instead of tag:**
```bash
# Get image digest from ECR
aws ecr describe-images \
  --repository-name mini-xdr-backend \
  --region us-east-1 \
  --query 'imageDetails[0].imageDigest' \
  --output text

# Update deployment with digest
kubectl set image deployment/mini-xdr-backend \
  backend=116912495274.dkr.ecr.us-east-1.amazonaws.com/mini-xdr-backend@sha256:abc123... \
  -n mini-xdr
```

**Option 3: Use unique version tags:**
```bash
# Tag with git commit SHA
docker tag mini-xdr-backend:latest mini-xdr-backend:59c483b
docker push 116912495274.dkr.ecr.us-east-1.amazonaws.com/mini-xdr-backend:59c483b

kubectl set image deployment/mini-xdr-backend \
  backend=116912495274.dkr.ecr.us-east-1.amazonaws.com/mini-xdr-backend:59c483b \
  -n mini-xdr
```

---

## Build Issues

### Docker Build: "addgroup: gid '1000' in use"

**Symptom:**
```
addgroup: gid '1000' in use
Build fails on Alpine Linux base image
```

**Cause:**
Alpine base image already has a group with GID 1000.

**Resolution:**

Add error suppression with fallback:
```dockerfile
RUN addgroup -g 1000 -S xdr 2>/dev/null || true && \
    adduser -u 1000 -S xdr -h /home/xdr 2>/dev/null || true
```

---

### Architecture Mismatch: ARM64 vs AMD64

**Symptom:**
```
WARNING: The requested image's platform (linux/arm64) does not match the detected host platform
Pods fail to start or have performance issues
```

**Cause:**
Building images on M1/M2 Mac produces `linux/arm64` images, but EKS nodes are `linux/amd64`.

**Resolution:**

Build images on a dedicated EC2 instance (Amazon Linux 2023 x86_64):

```bash
# SSH to EC2 build instance
ssh -i ~/.ssh/mini-xdr-eks-key.pem ec2-user@<EC2-IP>

# Verify architecture
uname -m  # Should output: x86_64

# Build images
cd /home/ec2-user/mini-xdr-v2/backend
docker build -t mini-xdr-backend:latest .
```

Alternatively, use Docker buildx for multi-platform builds:
```bash
docker buildx build --platform linux/amd64 -t mini-xdr-backend:latest .
```

---

## Deployment Issues

### ALB Ingress Not Creating

**Symptom:**
```
kubectl get ingress -n mini-xdr
No resources found
```

**Cause:**
AWS Load Balancer Controller not installed or ingress annotations incorrect.

**Resolution:**

1. **Verify AWS Load Balancer Controller:**
```bash
kubectl get deployment -n kube-system aws-load-balancer-controller
```

2. **Check ingress annotations:**
```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  annotations:
    kubernetes.io/ingress.class: alb
    alb.ingress.kubernetes.io/scheme: internet-facing
    alb.ingress.kubernetes.io/target-type: ip
```

3. **Check controller logs:**
```bash
kubectl logs -n kube-system deployment/aws-load-balancer-controller
```

---

### Backend Cannot Connect to RDS

**Symptom:**
```
sqlalchemy.exc.OperationalError: could not connect to server
Connection timeout or refused
```

**Cause:**
Security group rules or network ACLs blocking traffic between EKS and RDS.

**Resolution:**

1. **Check RDS security group:**
```bash
aws rds describe-db-instances \
  --db-instance-identifier <instance-id> \
  --query 'DBInstances[0].VpcSecurityGroups'
```

2. **Add inbound rule to RDS security group:**
```bash
aws ec2 authorize-security-group-ingress \
  --group-id <rds-sg-id> \
  --protocol tcp \
  --port 5432 \
  --source-group <eks-node-sg-id>
```

3. **Verify DATABASE_URL in ConfigMap:**
```bash
kubectl get configmap mini-xdr-config -n mini-xdr -o yaml | grep DATABASE_URL
```

---

## General Debugging Commands

### Check Pod Events
```bash
kubectl describe pod <pod-name> -n mini-xdr | tail -50
```

### View All Pod Logs
```bash
kubectl logs -l app=mini-xdr-backend -n mini-xdr --tail=100 --timestamps
```

### Get Previous Container Logs (if crashed)
```bash
kubectl logs <pod-name> -n mini-xdr --previous
```

### Check Image Pull Errors
```bash
kubectl get events -n mini-xdr --field-selector reason=Failed
```

### Verify Image Exists in ECR
```bash
aws ecr describe-images \
  --repository-name mini-xdr-backend \
  --image-ids imageTag=1.1.0 \
  --region us-east-1
```

### Test Network Connectivity from Pod
```bash
kubectl exec -it deployment/mini-xdr-backend -n mini-xdr -- curl http://mini-xdr-frontend:3000
```

### Check Resource Limits
```bash
kubectl describe node | grep -A 5 "Allocated resources"
```
