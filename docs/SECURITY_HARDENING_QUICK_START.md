# 🔐 Mini-XDR Security Hardening - Quick Start Guide

**Time Required:** 15-30 minutes
**Prerequisites:** AWS CLI, kubectl, admin access

---

## 🚨 Critical Actions (Do These First)

### 1. Generate and Deploy Strong Keys (5 minutes)

```bash
# Generate cryptographically secure keys
cd /Users/chasemad/Desktop/mini-xdr
./scripts/security/generate-secure-keys.sh

# Review the generated keys
cat .secure-keys/mini-xdr-secrets-*.env

# Deploy to Kubernetes
kubectl create secret generic mini-xdr-secrets-new \
  --from-env-file=.secure-keys/mini-xdr-secrets-*.env \
  --namespace mini-xdr

# Update backend deployment to use new secret
kubectl set env deployment/mini-xdr-backend \
  --from=secret/mini-xdr-secrets-new \
  -n mini-xdr

# Securely delete the local file
shred -vfz -n 10 .secure-keys/mini-xdr-secrets-*.env
```

### 2. Deploy Non-Root Backend Pods (2 minutes)

```bash
# Apply updated backend deployment
kubectl apply -f ops/k8s/backend-deployment.yaml -n mini-xdr

# Verify pods are running as UID 1000
kubectl get pods -n mini-xdr -o jsonpath='{.items[].spec.securityContext}'

# Expected output should include:
# "runAsUser":1000,"runAsGroup":1000
```

### 3. Enable KMS Key Rotation (2 minutes)

```bash
# Enable rotation on existing RDS encryption key
./scripts/security/enable-kms-rotation.sh

# Verify rotation is enabled
aws kms get-key-rotation-status \
  --key-id 431cb645-f4d9-41f6-8d6e-6c26c79c5c04 \
  --region us-east-1
```

---

## 🛡️ High Priority Actions (Do Within 24 Hours)

### 4. Restrict EKS API Endpoint Access (5 minutes)

```bash
# Restrict to your IP only
./scripts/security/restrict-eks-api-access.sh

# This will:
# - Detect your current IP
# - Update EKS cluster to only allow your IP
# - Wait for update to complete (3-5 minutes)

# Verify kubectl still works after restriction
kubectl get nodes
```

### 5. Enable Kubernetes Secrets Encryption (10 minutes)

```bash
# Enable KMS encryption for K8s secrets
./scripts/security/enable-secrets-encryption.sh

# This will:
# - Create new KMS key for secrets encryption
# - Configure EKS cluster to use the key
# - Re-encrypt all existing secrets
# - Takes ~10 minutes total

# Verify encryption is enabled
aws eks describe-cluster \
  --name mini-xdr-cluster \
  --region us-east-1 \
  --query 'cluster.encryptionConfig'
```

---

## ✅ Verification Steps

### Test Agent Verification System

```bash
# Get enrolled agents
curl -X GET http://your-alb-url/api/onboarding/enrolled-agents \
  -H "Authorization: Bearer YOUR_JWT_TOKEN"

# Run verification on agent
curl -X POST http://your-alb-url/api/onboarding/verify-agent-access/1 \
  -H "Authorization: Bearer YOUR_JWT_TOKEN"

# Expected response:
{
  "status": "ready|warning|fail",
  "checks": [
    {"check_name": "Agent Connectivity", "status": "pass", ...},
    {"check_name": "Platform Access", "status": "pass", ...},
    {"check_name": "Containment Capability", "status": "pass", ...},
    {"check_name": "Rollback Capability", "status": "pass", ...}
  ],
  "ready_for_production": true
}
```

---

## 📊 Security Status Check

Run this to verify everything is properly secured:

```bash
#!/bin/bash
echo "=== Mini-XDR Security Status ==="
echo ""

echo "1. Backend Pod Security:"
kubectl get pods -n mini-xdr -l app=mini-xdr-backend \
  -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.spec.securityContext.runAsUser}{"\n"}{end}'
echo "   Expected: UID 1000"
echo ""

echo "2. EKS API Access:"
aws eks describe-cluster --name mini-xdr-cluster --region us-east-1 \
  --query 'cluster.resourcesVpcConfig.publicAccessCidrs'
echo "   Expected: Your IP only"
echo ""

echo "3. KMS Key Rotation:"
aws kms get-key-rotation-status \
  --key-id 431cb645-f4d9-41f6-8d6e-6c26c79c5c04 \
  --region us-east-1 \
  --query 'KeyRotationEnabled'
echo "   Expected: true"
echo ""

echo "4. Secrets Encryption:"
aws eks describe-cluster --name mini-xdr-cluster --region us-east-1 \
  --query 'cluster.encryptionConfig[0].resources'
echo "   Expected: [secrets]"
echo ""

echo "5. Network Policies:"
kubectl get networkpolicies -n mini-xdr --no-headers | wc -l
echo "   Expected: 3 policies"
echo ""

echo "=== End Status Check ==="
```

---

## 🎯 Next Steps (Within 1-2 Weeks)

### When You Get a Domain:

```bash
# 1. Purchase domain (e.g., minixdr.io)
# 2. Request ACM certificate via AWS Console
#    - Go to Certificate Manager
#    - Request certificate
#    - Validate via DNS
# 3. Update ingress with certificate ARN
kubectl annotate ingress mini-xdr-ingress -n mini-xdr \
  alb.ingress.kubernetes.io/certificate-arn=YOUR_CERT_ARN \
  alb.ingress.kubernetes.io/ssl-redirect="true" \
  --overwrite
```

### Additional Hardening:

1. **Enable CloudWatch Alarms**
   - Failed authentication attempts > 10/hour
   - RDS connection spikes
   - KMS key usage anomalies

2. **Enable AWS GuardDuty**
   ```bash
   aws guardduty create-detector \
     --enable \
     --region us-east-1
   ```

3. **Enable VPC Flow Logs**
   ```bash
   aws ec2 create-flow-logs \
     --resource-type VPC \
     --resource-id vpc-0d474acd38d418e98 \
     --traffic-type ALL \
     --log-destination-type cloud-watch-logs \
     --log-group-name /aws/vpc/mini-xdr
   ```

4. **Deploy AWS WAF**
   - OWASP Top 10 ruleset
   - Rate limiting rules
   - Geo-blocking (if needed)

---

## 🆘 Troubleshooting

### Issue: Backend pods won't start after security context update

```bash
# Check pod logs
kubectl logs -n mini-xdr -l app=mini-xdr-backend --tail=50

# Common issue: Permissions on mounted volumes
# Fix: Update PVC permissions
kubectl exec -n mini-xdr -it <backend-pod> -- chown -R 1000:1000 /app
```

### Issue: kubectl access denied after EKS restriction

```bash
# Your IP changed - update the restriction
./scripts/security/restrict-eks-api-access.sh

# Or manually update
aws eks update-cluster-config \
  --name mini-xdr-cluster \
  --region us-east-1 \
  --resources-vpc-config \
    endpointPublicAccess=true,publicAccessCidrs=["YOUR_NEW_IP/32"]
```

### Issue: Secrets not encrypted after running script

```bash
# Check cluster encryption config
aws eks describe-cluster --name mini-xdr-cluster --region us-east-1 \
  --query 'cluster.encryptionConfig'

# If empty, run the script again and wait for full completion (10 min)

# Check update status
aws eks list-updates --name mini-xdr-cluster --region us-east-1
```

---

## 📞 Support

**Documentation:**
- Full Audit Report: `docs/SECURITY_AUDIT_REPORT.md`
- Deployment Guide: `AWS_DEPLOYMENT_GUIDE.md`

**Quick Help:**
```bash
# View available security scripts
ls -lh scripts/security/

# Check script help
./scripts/security/<script-name>.sh --help  # (if supported)
```

---

## ✨ Summary

After completing these steps, you will have:

✅ Strong cryptographic keys for JWT/encryption
✅ Non-root container execution
✅ KMS key rotation enabled
✅ EKS API restricted to your IP
✅ Kubernetes secrets encrypted with KMS
✅ Agent verification system ready for customer demos

**Security Rating: A- (Excellent for Testing/Demo Phase)**

When you add HTTPS and monitoring, you'll be at **A+ (Production Ready)**.

---

**Last Updated:** January 2025
**Next Review:** After Phase 1 deployment
