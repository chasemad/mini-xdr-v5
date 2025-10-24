# 🚀 Mini-XDR Onboarding - AWS Deployment Instructions

**Status:** ✅ READY TO DEPLOY  
**Date:** October 10, 2025  
**Prerequisites:** AWS CLI configured, kubectl connected to EKS cluster

---

## ⚡ Quick Deploy (30 Minutes)

### Step 1: Apply Database Migration to RDS (5 min)

```bash
# Get RDS credentials from Secrets Manager
aws secretsmanager get-secret-value \
  --secret-id mini-xdr-secrets \
  --query SecretString \
  --output text | jq -r '.DATABASE_URL'

# Set environment variable
export DATABASE_URL="postgresql+asyncpg://USER:PASS@mini-xdr-postgres.XXXXX.us-east-1.rds.amazonaws.com:5432/minixdr"

# Run migration
cd /Users/chasemad/Desktop/mini-xdr/backend
source venv/bin/activate
alembic upgrade head

# Verify
alembic current
# Expected output: 5093d5f3c7d4 (head)

# Confirm tables created
psql $DATABASE_URL -c "\dt" | grep -E "(discovered_assets|agent_enrollments)"
# Should show both tables
```

**✅ Checkpoint:** Migration 5093d5f3c7d4 applied to RDS

---

### Step 2: Build & Push Backend Image (10 min)

```bash
# Get ECR login
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin YOUR_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com

# Build backend image
cd /Users/chasemad/Desktop/mini-xdr/backend
docker build -t YOUR_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/mini-xdr-backend:onboarding-v1.0 .

# Push to ECR
docker push YOUR_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/mini-xdr-backend:onboarding-v1.0

# Update EKS deployment
kubectl set image deployment/backend-deployment \
  backend=YOUR_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/mini-xdr-backend:onboarding-v1.0 \
  -n mini-xdr

# Wait for rollout
kubectl rollout status deployment/backend-deployment -n mini-xdr
# Expected: deployment "backend-deployment" successfully rolled out

# Verify pods running
kubectl get pods -n mini-xdr | grep backend
# Should show 2/2 READY
```

**✅ Checkpoint:** Backend pods running with onboarding code

---

### Step 3: Build & Push Frontend Image (10 min)

```bash
# Get ALB endpoint
export ALB_DOMAIN=$(kubectl get ingress mini-xdr-ingress -n mini-xdr \
  -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')

echo "ALB Domain: $ALB_DOMAIN"

# Update frontend environment
cd /Users/chasemad/Desktop/mini-xdr/frontend
cat > .env.production << EOF
NEXT_PUBLIC_API_URL=http://$ALB_DOMAIN
EOF

# Build production bundle
npm run build

# Build Docker image
docker build -t YOUR_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/mini-xdr-frontend:onboarding-v1.0 .

# Push to ECR
docker push YOUR_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/mini-xdr-frontend:onboarding-v1.0

# Update EKS deployment
kubectl set image deployment/frontend-deployment \
  frontend=YOUR_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/mini-xdr-frontend:onboarding-v1.0 \
  -n mini-xdr

# Wait for rollout
kubectl rollout status deployment/frontend-deployment -n mini-xdr

# Verify pods running
kubectl get pods -n mini-xdr | grep frontend
# Should show 3/3 READY
```

**✅ Checkpoint:** Frontend pods running with onboarding wizard

---

### Step 4: Test Onboarding Flow (5 min)

```bash
# Get ALB endpoint
kubectl get ingress mini-xdr-ingress -n mini-xdr

# Navigate to registration
open http://YOUR_ALB_DOMAIN/register
```

**Test Checklist:**
- [ ] Register new organization (pick unique name)
- [ ] Redirects to dashboard after registration
- [ ] Navigate to `/onboarding`
- [ ] Complete Step 1 (Profile) - saves successfully
- [ ] Complete Step 2 (Network Scan) - enter test CIDR like `10.0.0.0/28`
- [ ] Scan completes and shows discovered assets
- [ ] Complete Step 3 (Agents) - generate token, see install script
- [ ] Complete Step 4 (Validation) - checks run
- [ ] Click "Complete Setup" - redirects to dashboard
- [ ] Dashboard shows completed status

**✅ Checkpoint:** Onboarding flow completes successfully

---

## 🔍 Verification Commands

### Check Backend Health
```bash
# Get backend pod
BACKEND_POD=$(kubectl get pods -n mini-xdr -l app=backend -o jsonpath='{.items[0].metadata.name}')

# Check logs for onboarding activity
kubectl logs $BACKEND_POD -n mini-xdr | grep -i onboarding

# Verify onboarding routes registered
kubectl exec $BACKEND_POD -n mini-xdr -- \
  curl -s http://localhost:8000/docs | grep onboarding

# Check database connection
kubectl exec $BACKEND_POD -n mini-xdr -- \
  python -c "from app.models import DiscoveredAsset; print('✅ Models accessible')"
```

### Check Frontend Health
```bash
# Get frontend pod
FRONTEND_POD=$(kubectl get pods -n mini-xdr -l app=frontend -o jsonpath='{.items[0].metadata.name}')

# Check environment variables
kubectl exec $FRONTEND_POD -n mini-xdr -- env | grep NEXT_PUBLIC_API_URL

# Verify wizard page exists
kubectl exec $FRONTEND_POD -n mini-xdr -- \
  ls -la /app/.next/server/app/onboarding/page.js
```

### Check Database
```bash
# Connect to RDS
export DATABASE_URL=$(aws secretsmanager get-secret-value \
  --secret-id mini-xdr-secrets \
  --query SecretString \
  --output text | jq -r '.DATABASE_URL')

psql $DATABASE_URL << SQL
-- Check migration applied
SELECT version_num FROM alembic_version;
-- Expected: 5093d5f3c7d4

-- Check tables exist
\dt discovered_assets
\dt agent_enrollments

-- Check organizations have onboarding columns
\d organizations

-- Check for test data
SELECT id, name, onboarding_status, onboarding_step FROM organizations;
SQL
```

---

## 🐛 Troubleshooting

### Issue: Migration Not Applied
```bash
# Check current version
alembic current

# If not at 5093d5f3c7d4, run upgrade
export DATABASE_URL="postgresql+asyncpg://..."
alembic upgrade head

# If error about existing tables, stamp the migration
alembic stamp 5093d5f3c7d4
```

### Issue: Backend Can't Scan Network
```bash
# K8s network policies may block ICMP
# Check if backend pod can ping
kubectl exec -it $BACKEND_POD -n mini-xdr -- ping -c 1 8.8.8.8

# If fails, update network policies to allow egress
kubectl get networkpolicies -n mini-xdr
```

### Issue: Agent Tokens Not Generating
```bash
# Check logs
kubectl logs $BACKEND_POD -n mini-xdr | grep -i "agent token"

# Verify database table exists
kubectl exec $BACKEND_POD -n mini-xdr -- \
  psql $DATABASE_URL -c "SELECT COUNT(*) FROM agent_enrollments;"

# Test API directly
curl -X POST http://$ALB_DOMAIN/api/onboarding/generate-agent-token \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"platform":"linux"}'
```

### Issue: Frontend Can't Reach Backend
```bash
# Check backend service
kubectl get svc backend-service -n mini-xdr

# Test from frontend pod
kubectl exec $FRONTEND_POD -n mini-xdr -- \
  curl http://backend-service:8000/api/onboarding/status

# Check CORS settings
kubectl logs $BACKEND_POD -n mini-xdr | grep CORS
```

---

## 🔒 Optional: Infrastructure Hardening

### Enable Redis Encryption (30 min)
```bash
# Create encrypted Redis replication group
aws elasticache create-replication-group \
  --replication-group-id mini-xdr-redis-encrypted \
  --replication-group-description "Mini-XDR Redis (Encrypted)" \
  --engine redis \
  --cache-node-type cache.t3.micro \
  --num-cache-clusters 2 \
  --automatic-failover-enabled \
  --at-rest-encryption-enabled \
  --transit-encryption-enabled \
  --auth-token $(openssl rand -base64 32 | tr -d '\n') \
  --cache-subnet-group-name mini-xdr-subnet-group \
  --security-group-ids sg-XXXXX

# Get auth token and store in Secrets Manager
aws secretsmanager create-secret \
  --name mini-xdr-secrets/redis-password \
  --description "Redis authentication token" \
  --secret-string "YOUR_REDIS_AUTH_TOKEN"

# Update backend K8s secret
kubectl create secret generic mini-xdr-redis-auth \
  --from-literal=REDIS_PASSWORD="YOUR_TOKEN" \
  -n mini-xdr

# Update backend deployment to use secret
kubectl set env deployment/backend-deployment \
  REDIS_PASSWORD_SECRET=mini-xdr-redis-auth \
  -n mini-xdr
```

### Enable TLS/HTTPS (30 min)
```bash
# Request ACM certificate
aws acm request-certificate \
  --domain-name xdr.yourdomain.com \
  --validation-method DNS \
  --region us-east-1

# Get certificate ARN
CERT_ARN=$(aws acm list-certificates \
  --query 'CertificateSummaryList[?DomainName==`xdr.yourdomain.com`].CertificateArn' \
  --output text)

echo "Certificate ARN: $CERT_ARN"

# Validate certificate (follow DNS validation steps in AWS Console)

# Update ALB ingress
kubectl patch ingress mini-xdr-ingress -n mini-xdr --type=json -p='[
  {
    "op": "add",
    "path": "/metadata/annotations/alb.ingress.kubernetes.io~1certificate-arn",
    "value": "'$CERT_ARN'"
  },
  {
    "op": "add",
    "path": "/metadata/annotations/alb.ingress.kubernetes.io~1listen-ports",
    "value": "[{\"HTTPS\":443}]"
  },
  {
    "op": "add",
    "path": "/metadata/annotations/alb.ingress.kubernetes.io~1ssl-redirect",
    "value": "443"
  }
]'

# Wait for ALB to update (2-3 minutes)
# Then test HTTPS
curl -I https://xdr.yourdomain.com
```

---

## ✅ Deployment Verification Checklist

### Backend
- [ ] Migration 5093d5f3c7d4 shows in `alembic current`
- [ ] Tables exist: `discovered_assets`, `agent_enrollments`
- [ ] Organizations table has onboarding columns
- [ ] Backend pods running (2/2 ready)
- [ ] `/api/onboarding/status` endpoint responds
- [ ] Logs show "onboarding_router" registered

### Frontend
- [ ] Frontend pods running (3/3 ready)
- [ ] `/onboarding` page loads
- [ ] `/register` page loads
- [ ] NEXT_PUBLIC_API_URL points to ALB
- [ ] No console errors

### Integration
- [ ] Can register new organization
- [ ] Can complete onboarding wizard
- [ ] Network scan discovers assets
- [ ] Agent tokens generate
- [ ] Validation checks execute
- [ ] Completion marks onboarding_status='completed'

### Database
- [ ] Organizations table updated
- [ ] Discovered assets persist per org
- [ ] Agent enrollments tracked
- [ ] Multi-tenant isolation verified

---

## 📊 Success Metrics to Monitor

After deployment, track these metrics:

1. **Onboarding Completion Rate**
   ```sql
   SELECT 
     COUNT(*) FILTER (WHERE onboarding_status = 'completed') * 100.0 / COUNT(*) as completion_rate,
     AVG(EXTRACT(EPOCH FROM (onboarding_completed_at - created_at))/60) as avg_time_minutes
   FROM organizations
   WHERE onboarding_status != 'not_started';
   ```

2. **Assets Discovered Per Org**
   ```sql
   SELECT 
     o.name,
     COUNT(da.id) as assets_discovered,
     COUNT(DISTINCT da.classification) as unique_classifications
   FROM organizations o
   LEFT JOIN discovered_assets da ON da.organization_id = o.id
   GROUP BY o.id, o.name;
   ```

3. **Agent Enrollment Success**
   ```sql
   SELECT 
     COUNT(*) FILTER (WHERE status = 'active') as active_agents,
     COUNT(*) FILTER (WHERE status = 'pending') as pending_agents,
     COUNT(*) FILTER (WHERE status = 'inactive') as inactive_agents
   FROM agent_enrollments;
   ```

---

## 🎯 Post-Deployment Tasks

### Immediate (Day 1)
1. ✅ Deploy to AWS (follow steps above)
2. ✅ Test onboarding with test organization
3. ✅ Verify database isolation
4. ✅ Check logs for errors
5. ✅ Monitor performance

### Week 1
1. 🔄 Onboard first real customer
2. 🔄 Collect feedback on wizard UX
3. 🔄 Monitor completion rates
4. 🔄 Optimize network scan performance
5. 🔄 Document common issues

### Week 2+
1. 📋 Enable Redis encryption
2. 📋 Add TLS certificate
3. 📋 Implement advanced validation checks
4. 📋 Add agent binary downloads
5. 📋 Build admin dashboard for org management

---

## 📞 Support

### If Deployment Fails

**Backend Pod Won't Start:**
```bash
# Check logs
kubectl logs deployment/backend-deployment -n mini-xdr

# Common issues:
# - Database connection failed → verify DATABASE_URL in secret
# - Migration not applied → run alembic upgrade head
# - Import errors → verify all new files are in Docker image
```

**Frontend Pod Won't Start:**
```bash
# Check logs
kubectl logs deployment/frontend-deployment -n mini-xdr

# Common issues:
# - Build failed → check npm run build output
# - API_URL wrong → verify .env.production before build
# - Missing dependencies → verify package.json includes new deps
```

**Onboarding Endpoints 404:**
```bash
# Verify routes registered
kubectl exec $BACKEND_POD -n mini-xdr -- \
  python -c "from app.main import app; print([r.path for r in app.routes if 'onboarding' in r.path])"

# Should show 10 /api/onboarding/* routes
```

### Contact & Escalation
- Check `TEST_AND_DEPLOY_GUIDE.md` for detailed troubleshooting
- Review logs in CloudWatch (if configured)
- Check `FINAL_DEPLOYMENT_STATUS.md` for implementation reference

---

## 📋 Rollback Procedure

If deployment fails and you need to rollback:

```bash
# Rollback backend
kubectl rollout undo deployment/backend-deployment -n mini-xdr

# Rollback frontend  
kubectl rollout undo deployment/frontend-deployment -n mini-xdr

# Rollback database migration (if needed)
export DATABASE_URL="..."
cd backend
alembic downgrade -1  # Goes back one migration

# Verify rollback
kubectl get pods -n mini-xdr
alembic current
```

---

## 🎉 YOU'RE READY!

**Everything needed for deployment:**
- ✅ Code complete and tested
- ✅ Database migration ready
- ✅ Docker images buildable
- ✅ Deployment commands documented
- ✅ Verification steps provided
- ✅ Troubleshooting guide included
- ✅ Rollback procedure documented

**Execute the Quick Deploy steps above and you'll be live in 30 minutes.**

**Questions? See:**
- `TEST_AND_DEPLOY_GUIDE.md` - Comprehensive testing
- `FINAL_DEPLOYMENT_STATUS.md` - Implementation summary
- `README_ONBOARDING.md` - Quick reference

---

**🚀 Deploy now and start onboarding enterprise customers!**


