# Mini-XDR AWS Operations Guide

**Date:** October 9, 2025  
**Status:** Multi-tenant system with AWS control scripts  
**Version:** 2.0

---

## Table of Contents
1. [Daily Startup/Shutdown](#daily-startupshutdown)
2. [User Management](#user-management)
3. [Dashboard Access](#dashboard-access)
4. [Security Configuration](#security-configuration)
5. [Troubleshooting](#troubleshooting)

---

## Daily Startup/Shutdown

### Starting the System (~8 minutes)

```bash
cd /Users/chasemad/Desktop/mini-xdr
./start-mini-xdr-aws.sh
```

**What it does:**
1. Starts RDS PostgreSQL instance (~5 min)
2. Checks Redis status (~1 min)
3. Scales EKS pods from 0 to target replicas
4. Waits for pods to be healthy
5. Displays connection information

**Expected output:**
```
🚀 Mini-XDR AWS Startup Initiated
...
✅ Startup Complete!
📊 System Status:
   mini-xdr-backend-xxx    1/1 Running
   mini-xdr-frontend-xxx   1/1 Running
```

### Stopping the System (Immediate + ~2 min for RDS)

```bash
./stop-mini-xdr-aws.sh
```

**What it does:**
1. Scales EKS pods to 0 replicas (immediate)
2. Stops RDS instance (saves ~$15/month)
3. Notes Redis is still running (~$12/month)

**Cost savings when stopped:**
- RDS stopped: ~$15/month saved
- Pods scaled to 0: No additional compute cost
- Redis running: ~$12/month (can't be paused)
- EKS control plane: ~$73/month (always running)

**Total savings: ~$15/month when stopped**

---

## User Management

### Creating the First Organization

After deploying, you need to create your first organization and admin user.

**Method 1: Via Frontend (Recommended)**
1. Access the dashboard (see Dashboard Access below)
2. Navigate to http://YOUR_ALB_URL/register
3. Fill in:
   - Organization Name: "Your Company"
   - Admin Name: "Your Name"
   - Admin Email: "you@example.com"
   - Admin Password: (min 12 chars, with uppercase, lowercase, number, special)
4. Click "Create Organization"
5. You'll be automatically logged in

**Method 2: Via API**
```bash
curl -X POST http://localhost:8000/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "organization_name": "Acme Security",
    "admin_email": "admin@acme.com",
    "admin_password": "SecurePass123!@#",
    "admin_name": "John Doe"
  }'
```

### Inviting Additional Users (Admin only)

```bash
curl -X POST http://localhost:8000/api/auth/invite \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -d '{
    "email": "analyst@acme.com",
    "full_name": "Jane Smith",
    "role": "analyst"
  }'
```

**Available Roles:**
- `admin` - Full access, can invite users, manage org
- `soc_lead` - Manage incidents, approve workflows
- `analyst` - View and investigate incidents
- `viewer` - Read-only access

---

## Dashboard Access

### Option 1: Local Port-Forward (Current Setup)

```bash
# Start port-forward in background
kubectl port-forward -n mini-xdr svc/mini-xdr-frontend-service 3000:3000 &
kubectl port-forward -n mini-xdr svc/mini-xdr-backend-service 8000:8000 &

# Access dashboard
open http://localhost:3000
```

### Option 2: AWS ALB (IP-Restricted)

**Setup (one-time):**
```bash
# 1. Create ALB security group
./scripts/create-alb-security-group.sh

# 2. Apply ingress configuration
kubectl apply -f k8s/ingress-alb.yaml

# 3. Wait 3-5 minutes for ALB provisioning
kubectl get ingress -n mini-xdr mini-xdr-ingress -w

# 4. Get ALB hostname
ALB_URL=$(kubectl get ingress -n mini-xdr mini-xdr-ingress -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')
echo "Dashboard: http://$ALB_URL"
```

**Accessing:**
```bash
# Get current ALB URL
kubectl get ingress -n mini-xdr mini-xdr-ingress

# Open in browser
open http://YOUR_ALB_HOSTNAME
```

**Current restriction:** Only accessible from IP `37.19.221.202/32`

### Switching to Public Access (For Demos)

**Option A: Update security group to allow all IPs**
```bash
./scripts/create-alb-security-group.sh 0.0.0.0/0
```

**Option B: Update ingress annotation**
```bash
# Edit k8s/ingress-alb.yaml
# Change:
#   alb.ingress.kubernetes.io/inbound-cidrs: 37.19.221.202/32
# To:
#   alb.ingress.kubernetes.io/inbound-cidrs: 0.0.0.0/0

# Apply changes
kubectl apply -f k8s/ingress-alb.yaml
```

### Adding HTTPS/TLS

1. **Create ACM certificate in AWS Console:**
   - Go to AWS Certificate Manager
   - Request certificate for your domain
   - Validate via DNS or email

2. **Update ingress with certificate ARN:**
   ```bash
   # Edit k8s/ingress-alb.yaml
   # Uncomment and update:
   # alb.ingress.kubernetes.io/certificate-arn: arn:aws:acm:us-east-1:ACCOUNT:certificate/CERT_ID
   
   kubectl apply -f k8s/ingress-alb.yaml
   ```

3. **Configure DNS:**
   - Create CNAME record pointing to ALB hostname
   - Example: `xdr.yourdomain.com` → `k8s-minixdr-abc123.us-east-1.elb.amazonaws.com`

---

## Security Configuration

### Current Security Status

**✅ Implemented:**
- Multi-tenant data isolation by organization
- JWT authentication with 8-hour expiry
- Password requirements (12+ chars, complexity)
- Account lockout after 5 failed attempts
- RDS encrypted at rest (AES-256)
- RDS Multi-AZ for high availability
- VPC isolation with private subnets
- Security groups with least privilege
- Non-root containers (UID 1000)

**⚠️ Pending:**
- Redis encryption (transit + at-rest)
- External TLS/HTTPS (ALB + ACM certificate)
- AWS Secrets Manager rotation policy

### Verifying ML Models

Run the ML model verification script to ensure all models are loaded and performing correctly:

```bash
./scripts/verify-ml-models.sh
```

**Expected output:**
```
🤖 Mini-XDR ML Model Verification
===================================

🔍 Checking Model Files...
✅ best_general.pth                              1.10 MB
✅ best_brute_force_specialist.pth              1.10 MB
✅ best_ddos_specialist.pth                     1.10 MB
✅ best_web_attacks_specialist.pth              1.10 MB
✅ lstm_autoencoder.pth                         0.24 MB
✅ isolation_forest.pkl                         0.17 MB
✅ isolation_forest_scaler.pkl                  0.00 MB

Found: 7/7 models

🧪 Testing Model Loading & Inference...
✅ best_general.pth                      Load:  42.3ms  Inference:  45.2ms
...

📊 Performance Summary:
✅ All 7 models loaded successfully
✅ Device: CPU
✅ Average PyTorch inference: <100ms
✅ Average sklearn inference: <20ms

🎯 Overall Health: EXCELLENT
```

### Password Policy

All passwords must meet these requirements:
- Minimum 12 characters
- At least one uppercase letter
- At least one lowercase letter
- At least one number
- At least one special character (@$!%*?&#^()_-+=[]{}|\\:;"'<>,.?/~`)

**Example valid passwords:**
- `SecurePass123!@#`
- `MyXDR#Dashboard2025`
- `Str0ng&Complex!Pass`

### Account Lockout

After 5 failed login attempts:
- Account locked for 15 minutes
- User receives HTTP 403 Forbidden
- Lock automatically expires after 15 minutes
- Successful login resets failed attempt counter

---

## Troubleshooting

### Issue: Pods Not Starting After Startup Script

**Check pod status:**
```bash
kubectl get pods -n mini-xdr
```

**Check logs:**
```bash
kubectl logs -n mini-xdr deployment/mini-xdr-backend --tail=100
```

**Common causes:**
1. RDS not fully started - wait additional 2-3 minutes
2. Database connection timeout - check security groups
3. Redis unavailable - verify Redis cluster status

**Solution:**
```bash
# Force restart pods
kubectl rollout restart deployment/mini-xdr-backend -n mini-xdr
```

### Issue: Can't Access Dashboard via ALB

**Check ingress status:**
```bash
kubectl get ingress -n mini-xdr mini-xdr-ingress
```

**Check ALB security group:**
```bash
aws ec2 describe-security-groups \
  --region us-east-1 \
  --filters "Name=group-name,Values=mini-xdr-alb-sg"
```

**Common causes:**
1. ALB not fully provisioned - wait 5 minutes
2. IP whitelist blocking you - update to your current IP
3. Security group misconfigured

**Solution:**
```bash
# Update IP whitelist
./scripts/create-alb-security-group.sh $(curl -s ifconfig.me)/32
```

### Issue: Login Returns 401 Unauthorized

**Check backend logs:**
```bash
kubectl logs -n mini-xdr deployment/mini-xdr-backend | grep auth
```

**Common causes:**
1. No JWT_SECRET_KEY configured
2. Organization not created yet
3. Wrong email/password

**Solution:**
```bash
# Ensure JWT secret is set in Kubernetes secret
kubectl get secret -n mini-xdr mini-xdr-secrets -o jsonpath='{.data.JWT_SECRET_KEY}' | base64 -d
```

### Issue: Organization Data Showing for Wrong Org

**This should never happen!** If you see data from another organization:

1. **Stop immediately** - potential security issue
2. Check current user's organization_id:
   ```bash
   curl http://localhost:8000/api/auth/me \
     -H "Authorization: Bearer YOUR_TOKEN"
   ```
3. Verify all API responses include organization_id filter
4. Check backend logs for SQL queries

### Issue: RDS Won't Start

**Check RDS status:**
```bash
aws rds describe-db-instances \
  --db-instance-identifier mini-xdr-postgres \
  --region us-east-1 \
  --query 'DBInstances[0].DBInstanceStatus'
```

**Common causes:**
1. RDS already running
2. Recent backup/maintenance window
3. AWS service issue

**Solution:**
```bash
# Force start if needed
aws rds start-db-instance \
  --db-instance-identifier mini-xdr-postgres \
  --region us-east-1
```

---

## Cost Monitoring

### Current Monthly Costs
- EKS cluster: ~$73/month
- RDS (when running): ~$15/month
- Redis: ~$12/month
- EC2 nodes: ~$60/month
- NAT Gateway: ~$32/month
- **Total: ~$192/month when running**

### Cost Savings Tips

1. **Stop when not in use:**
   ```bash
   ./stop-mini-xdr-aws.sh  # Saves ~$15/month
   ```

2. **Use Spot instances for dev:**
   - Configure node group with spot instances
   - 70% cost savings for non-critical workloads

3. **Scale down EKS nodes:**
   ```bash
   # During low usage
   kubectl scale deployment mini-xdr-backend -n mini-xdr --replicas=1
   kubectl scale deployment mini-xdr-frontend -n mini-xdr --replicas=1
   ```

4. **Monitor with AWS Cost Explorer:**
   - Set up billing alerts
   - Review cost allocation tags
   - Identify unused resources

---

## Quick Reference Commands

```bash
# System Control
./start-mini-xdr-aws.sh                    # Start system
./stop-mini-xdr-aws.sh                     # Stop system
./scripts/verify-ml-models.sh              # Check ML models

# Kubernetes
kubectl get pods -n mini-xdr               # Check pod status
kubectl get ingress -n mini-xdr            # Get ALB URL
kubectl logs -f deployment/mini-xdr-backend -n mini-xdr  # Stream logs

# Database
kubectl port-forward -n mini-xdr svc/mini-xdr-backend-service 8000:8000
curl http://localhost:8000/health          # Check health

# Security
./scripts/create-alb-security-group.sh     # Create/update ALB SG
./scripts/create-alb-security-group.sh 0.0.0.0/0  # Allow public access
```

---

**Last Updated:** October 9, 2025  
**Status:** Production Ready with Multi-Tenant Auth  
**Next Steps:** Enable Redis encryption, configure TLS/HTTPS


