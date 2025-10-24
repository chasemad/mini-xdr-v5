# Dockerfile Fixes for Production Deployment

This document explains the critical Dockerfile fixes required for successful EKS deployment. These
fixes resolve module resolution, permission issues, and build-time configuration problems.

## Background

Mini-XDR uses multi-stage Docker builds for both backend and frontend. The builder stage compiles
dependencies, and the production stage runs as a non-root user (`xdr:xdr` with UID/GID 1000) for
security. Several issues arose when deploying these images to EKS that required specific fixes.

---

## Backend Dockerfile Fixes

### Issue 1: "No module named uvicorn"

**Problem:**
The backend pod crashed immediately with `ModuleNotFoundError: No module named 'uvicorn'` despite
uvicorn being installed.

**Root Cause Analysis:**

The Dockerfile uses a multi-stage build:

1. **Builder Stage**: Installs packages to `/root/.local` using `pip install --user`
2. **Production Stage**: Copies packages to `/home/xdr/.local` for non-root user

Original problematic code:
```dockerfile
# Builder stage
RUN pip install --user --no-cache-dir -r requirements.txt
# Packages installed to: /root/.local/lib/python3.11/site-packages

# Production stage
COPY --from=builder --chown=xdr:xdr /root/.local /home/xdr/.local
# Packages copied to: /home/xdr/.local/lib/python3.11/site-packages

CMD ["python", "-m", "uvicorn", "app.main:app", ...]
```

The issue occurred because:

1. `CMD ["python", "-m", "uvicorn", ...]` invoked system Python at `/usr/local/bin/python`
2. System Python has default `sys.path` that doesn't include `/home/xdr/.local/lib/python3.11/site-packages`
3. Missing `PYTHONUSERBASE` environment variable meant Python didn't know about user-installed packages

**Verification:**
```bash
# Inside failing container
$ python -c "import sys; print(sys.path)"
['/usr/local/lib/python311.zip', '/usr/local/lib/python3.11', ...]
# Missing: /home/xdr/.local/lib/python3.11/site-packages

$ echo $PYTHONUSERBASE
# Empty - not set!
```

**Solution:**

Two changes fixed this issue:

**1. Set PYTHONUSERBASE environment variable:**
```dockerfile
ENV PATH=/home/xdr/.local/bin:$PATH \
    PYTHONUSERBASE=/home/xdr/.local
```

This tells Python to include `/home/xdr/.local/lib/python3.11/site-packages` in `sys.path`.

**2. Use uvicorn binary directly:**
```dockerfile
CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--log-level", "info"]
```

This uses the uvicorn binary at `/home/xdr/.local/bin/uvicorn` which is already in PATH and knows
where its dependencies are located.

**Commit:** `3466bc7` - backend/Dockerfile:62-63,92-96

**Verification:**
```bash
# After fix
$ docker run --rm --entrypoint python backend-image -c "import sys; print(sys.path)"
[..., '/home/xdr/.local/lib/python3.11/site-packages', ...]

$ docker run --rm --entrypoint sh backend-image -c "echo \$PYTHONUSERBASE"
/home/xdr/.local
```

### Why This Matters

- **Security**: Maintains non-root user execution
- **Python User Installs**: Correctly handles `pip install --user` in multi-stage builds
- **Portability**: Works consistently across different base images

---

## Frontend Dockerfile Fixes

### Issue 2: npm Cache Permission Errors

**Problem:**
Frontend pod crashed with:
```
npm error path /home/xdr/.npm
npm error errno ENOENT
npm error enoent ENOENT: no such file or directory, mkdir '/home/xdr/.npm'
```

**Root Cause Analysis:**

The frontend Dockerfile had three related problems:

1. **Missing HOME environment variable**: npm defaulted to trying to create `/.npm` instead of `/home/xdr/.npm`
2. **Missing npm cache directory**: `/home/xdr/.npm` didn't exist for the non-root user
3. **TypeScript not in production**: `next.config.ts` requires TypeScript at runtime, but it wasn't
   installed with `npm ci --only=production`

Original problematic code:
```dockerfile
# No HOME environment variable set
RUN adduser -u 1000 -S xdr ...
# But /home/xdr/.npm directory not created

RUN npm ci --only=production
# TypeScript not installed (it's a devDependency)

CMD ["npm", "start"]
# Tries to load next.config.ts but TypeScript not available
```

**Solution:**

Three coordinated fixes:

**1. Create npm cache directories with proper ownership:**
```dockerfile
RUN addgroup -g 1000 -S xdr 2>/dev/null || true && \
    adduser -u 1000 -S xdr -h /home/xdr 2>/dev/null || true && \
    mkdir -p /home/xdr/.npm /home/xdr/.cache && \
    chown -R xdr:xdr /home/xdr 2>/dev/null || chown -R 1000:1000 /home/xdr
```

Note: `2>/dev/null || true` handles cases where GID 1000 already exists in Alpine base image.

**2. Set HOME environment variable:**
```dockerfile
ENV NODE_ENV=production \
    PORT=3000 \
    HOSTNAME="0.0.0.0" \
    NEXT_TELEMETRY_DISABLED=1 \
    HOME=/home/xdr
```

**3. Install TypeScript as production dependency:**
```dockerfile
RUN npm ci --only=production --ignore-scripts && \
    npm install --save-exact typescript@5.9.3 && \
    npm cache clean --force
```

**Commit:** `6616444` - frontend/Dockerfile:49-79

**Verification:**
```bash
# After fix
$ docker run --rm --entrypoint sh frontend-image -c "echo \$HOME"
/home/xdr

$ docker run --rm --entrypoint sh frontend-image -c "ls -la /home/xdr/.npm"
drwxr-xr-x  2 xdr  xdr  4096 ... /home/xdr/.npm

$ docker run --rm --entrypoint sh frontend-image -c "npm list typescript"
typescript@5.9.3
```

### Why This Matters

- **Next.js TypeScript Support**: `next.config.ts` requires TypeScript at runtime
- **npm Security**: Non-root user can create cache files without privilege escalation
- **Alpine Compatibility**: Handles edge cases with existing GIDs

---

## Frontend Build-Time Configuration

### Issue 3: Content Security Policy (CSP) Violations

**Problem:**
Frontend loaded successfully but couldn't connect to backend:
```
TypeError: Failed to fetch
Refused to connect to http://localhost:8000 because it violates CSP
```

**Root Cause:**

Next.js environment variables starting with `NEXT_PUBLIC_*` are **baked into the JavaScript bundle
at build time**, not runtime. The frontend was built with defaults:

```dockerfile
ARG NEXT_PUBLIC_API_BASE=http://localhost:8000
ARG NEXT_PUBLIC_API_URL=http://localhost:8000
```

When deployed to EKS, the frontend tried to call `http://localhost:8000`, which:
1. Doesn't exist (backend is at ALB URL)
2. Violates Content Security Policy (CSP) from production domain

**Solution:**

Build frontend with **actual ALB URL** as build argument:

```bash
docker build \
  --build-arg NEXT_PUBLIC_API_BASE="http://k8s-minixdr-minixdri-dc5fc1df8b-1132128475.us-east-1.elb.amazonaws.com" \
  --build-arg NEXT_PUBLIC_API_URL="http://k8s-minixdr-minixdri-dc5fc1df8b-1132128475.us-east-1.elb.amazonaws.com" \
  --build-arg NEXT_PUBLIC_API_KEY="demo-minixdr-api-key" \
  -t 116912495274.dkr.ecr.us-east-1.amazonaws.com/mini-xdr-frontend:1.1.0 \
  .
```

**Critical:** This must be done on the EC2 build instance, not locally, because the ALB URL changes
per deployment and needs to match the EKS environment.

**Verification:**

Check frontend logs for correct API configuration:
```bash
kubectl logs deployment/mini-xdr-frontend -n mini-xdr | grep "ThreatDataService configuration"
```

Should show:
```
🔧 ThreatDataService configuration: {
  API_BASE_URL: 'http://k8s-minixdr-minixdri-dc5fc1df8b-1132128475.us-east-1.elb.amazonaws.com',
  MAX_RETRY_ATTEMPTS: 3,
  RETRY_DELAY: 1000
}
```

### Why This Matters

- **Next.js Build-Time Variables**: Understanding that `NEXT_PUBLIC_*` variables are compiled into the bundle
- **CSP Security**: Browser prevents connections to mismatched origins
- **Environment-Specific Builds**: Frontend must be rebuilt for each target environment

---

## Complete Build Workflow

### Backend Build (on EC2 instance)

```bash
cd /home/ec2-user/mini-xdr-v2/backend

# Login to ECR
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin \
  116912495274.dkr.ecr.us-east-1.amazonaws.com

# Build with fixed Dockerfile
docker build \
  --build-arg VERSION="1.1.0" \
  --build-arg VCS_REF="$(git rev-parse --short HEAD)" \
  --build-arg BUILD_DATE="$(date -u +"%Y-%m-%dT%H:%M:%SZ")" \
  -t 116912495274.dkr.ecr.us-east-1.amazonaws.com/mini-xdr-backend:1.1.0 \
  -t 116912495274.dkr.ecr.us-east-1.amazonaws.com/mini-xdr-backend:latest \
  .

# Push to ECR
docker push 116912495274.dkr.ecr.us-east-1.amazonaws.com/mini-xdr-backend:1.1.0
docker push 116912495274.dkr.ecr.us-east-1.amazonaws.com/mini-xdr-backend:latest
```

### Frontend Build (on EC2 instance)

```bash
cd /home/ec2-user/mini-xdr-v2/frontend

# Get ALB URL from Kubernetes
ALB_URL=$(kubectl get ingress mini-xdr-ingress -n mini-xdr -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')

# Build with correct API endpoints and fixed Dockerfile
docker build \
  --build-arg NEXT_PUBLIC_API_BASE="http://${ALB_URL}" \
  --build-arg NEXT_PUBLIC_API_URL="http://${ALB_URL}" \
  --build-arg VERSION="1.1.0" \
  --build-arg VCS_REF="$(git rev-parse --short HEAD)" \
  --build-arg BUILD_DATE="$(date -u +"%Y-%m-%dT%H:%M:%SZ")" \
  -t 116912495274.dkr.ecr.us-east-1.amazonaws.com/mini-xdr-frontend:1.1.0 \
  -t 116912495274.dkr.ecr.us-east-1.amazonaws.com/mini-xdr-frontend:latest \
  .

# Push to ECR
docker push 116912495274.dkr.ecr.us-east-1.amazonaws.com/mini-xdr-frontend:1.1.0
docker push 116912495274.dkr.ecr.us-east-1.amazonaws.com/mini-xdr-frontend:latest
```

### Deploy to Kubernetes

```bash
# Force pull new images
kubectl patch deployment mini-xdr-backend -n mini-xdr -p \
  '{"spec":{"template":{"spec":{"containers":[{"name":"backend","imagePullPolicy":"Always"}]}}}}'

kubectl patch deployment mini-xdr-frontend -n mini-xdr -p \
  '{"spec":{"template":{"spec":{"containers":[{"name":"frontend","imagePullPolicy":"Always"}]}}}}'

# Restart deployments
kubectl rollout restart deployment/mini-xdr-backend -n mini-xdr
kubectl rollout restart deployment/mini-xdr-frontend -n mini-xdr

# Verify rollout
kubectl rollout status deployment/mini-xdr-backend -n mini-xdr
kubectl rollout status deployment/mini-xdr-frontend -n mini-xdr
```

---

## Key Takeaways

1. **PYTHONUSERBASE is critical** for Python user-installed packages in multi-stage builds
2. **HOME environment variable** is required for npm to function as non-root user
3. **NEXT_PUBLIC_* variables** are baked into Next.js builds at build-time, not runtime
4. **Always build on linux/amd64** architecture for EKS compatibility
5. **Use `imagePullPolicy: Always`** when reusing image tags to force fresh pulls

## Testing New Dockerfile Changes

Before deploying, test locally:

```bash
# Test backend can start
docker run --rm -p 8000:8000 \
  -e DATABASE_URL=sqlite:///./data/test.db \
  mini-xdr-backend:latest

# In another terminal, test health endpoint
curl http://localhost:8000/health

# Test frontend can start
docker run --rm -p 3000:3000 mini-xdr-frontend:latest

# In another terminal, test frontend loads
curl http://localhost:3000
```

## Recent Production Fixes (v1.1.8)

### Code Import Errors in Production

**Issue:** Backend pods crashed with import errors despite successful local testing.

**Root Cause:** Missing imports in production code that were added during development.

**Examples Fixed:**
1. **CloudIntegration Import:**
   ```python
   # Added to backend/app/integrations/manager.py
   from .base import CloudIntegration
   ```

2. **Optional Type Import:**
   ```python
   # Added to backend/app/onboarding_v2/auto_discovery.py
   from typing import Dict, List, Any, Optional
   ```

**Prevention:** Always verify imports after adding new models or integrations. Test container builds locally before pushing to production.

### Database Schema Synchronization

**Issue:** Production database missing columns added in development.

**Fix:** Always run database migrations in production:
```bash
kubectl exec -it deployment/mini-xdr-backend -n mini-xdr -- alembic upgrade head
```

**Added to deployment checklist:** Database migration step is now mandatory after code deployments.

## Related Documentation

- [AWS Deployment Overview](overview.md) - Current EKS architecture
- [Operations Guide](operations.md) - Step-by-step deployment commands
- [Troubleshooting](troubleshooting.md) - Common deployment issues
