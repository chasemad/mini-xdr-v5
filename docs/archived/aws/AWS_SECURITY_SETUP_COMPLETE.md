# AWS Security Setup - Complete Verification

## ✅ Security Configuration Summary

### 1. Kubernetes Security (EKS)
**Status**: ✅ CONFIGURED

#### Network Policies
- ✅ Backend network policy: Only accepts traffic from frontend and ALB
- ✅ Frontend network policy: Only accepts traffic from ALB
- ✅ Default deny-all policy: Prevents unauthorized pod-to-pod communication
- **Location**: `infrastructure/aws/security-hardening.yaml`

#### Pod Security Standards
- ✅ Namespace configured with `restricted` Pod Security Standard
- ✅ All pods run as non-root (UID 1000)
- ✅ All capabilities dropped (`ALL` capabilities removed)
- ✅ Read-only root filesystem (where applicable)
- ✅ No privilege escalation allowed
- **Location**: `k8s/backend-deployment.yaml`, `k8s/frontend-deployment.yaml`

#### RBAC (Role-Based Access Control)
- ✅ Service accounts with minimal permissions
- ✅ Role bindings restrict access to only necessary resources
- ✅ `automountServiceAccountToken: false` for security
- **Location**: `infrastructure/aws/security-hardening.yaml`

#### Resource Limits
- ✅ Resource quotas prevent resource exhaustion attacks
- ✅ Limit ranges prevent pod bombing
- ✅ CPU/Memory limits set appropriately
- **Location**: `infrastructure/aws/security-hardening.yaml`

### 2. Network Security

#### Security Groups
- ✅ EKS cluster in private subnets (recommended)
- ✅ ALB security group configured
- ✅ Network policies restrict pod communication
- **Location**: `infrastructure/aws/eks-cluster-config.yaml`

#### VPC Configuration
- ✅ Cluster uses existing VPC with private/public subnets
- ✅ Private networking enabled for node groups
- ✅ Subnets configured across multiple AZs
- **Location**: `infrastructure/aws/eks-cluster-config.yaml`

#### IP Whitelisting (Optional)
- ⚠️ Currently configured for specific IP: `24.11.0.176/32`
- 📝 For production, consider removing IP restriction or using VPN
- **Location**: `k8s/ingress-alb.yaml` line 23

### 3. Secrets Management

#### AWS Secrets Manager Integration
- ✅ Backend configured to use AWS Secrets Manager
- ✅ Service account has Secrets Manager read permissions
- ✅ Secrets loaded at startup via `secrets_manager.py`
- ✅ Fallback to environment variables for local development
- **Location**:
  - `backend/app/secrets_manager.py`
  - `backend/app/config.py`
  - `infrastructure/aws/eks-cluster-config.yaml` (IAM policy)

#### Kubernetes Secrets
- ✅ Secrets stored in Kubernetes `Secret` objects
- ✅ Secrets referenced from `mini-xdr-secrets` Secret
- ✅ Not hardcoded in deployment files
- **Location**: `k8s/backend-deployment.yaml`

### 4. Application Security

#### Container Security
- ✅ Non-root user (UID 1000)
- ✅ Minimal base images (python:3.11.9-slim, node:18-alpine)
- ✅ Multi-stage builds reduce attack surface
- ✅ Health checks configured
- **Location**: `backend/Dockerfile`, `frontend/Dockerfile`

#### API Security
- ✅ JWT authentication configured
- ✅ API keys managed via Secrets Manager
- ✅ CORS configured appropriately
- **Location**: `backend/app/main.py`, `backend/app/auth.py`

### 5. Ingress & Load Balancer Security

#### AWS Application Load Balancer (ALB)
- ✅ ALB Ingress Controller configured
- ✅ Health checks configured
- ✅ Access logs enabled (S3 bucket required)
- ✅ Deletion protection enabled
- ✅ HTTP/2 enabled
- **Location**: `k8s/ingress-alb.yaml`, `infrastructure/aws/security-hardening.yaml`

#### SSL/TLS Configuration
- ⚠️ **ACTION REQUIRED**: SSL certificate not yet configured
- 📝 To enable HTTPS:
  1. Create ACM certificate in us-east-1
  2. Update `k8s/ingress-alb.yaml` with certificate ARN
  3. Uncomment TLS section
  4. Enable SSL redirect
- **Location**: `k8s/ingress-alb.yaml` lines 12-18, 286-289

### 6. IAM & Access Control

#### EKS Cluster IAM
- ✅ OIDC provider configured
- ✅ Service accounts use IRSA (IAM Roles for Service Accounts)
- ✅ AWS Load Balancer Controller has required permissions
- ✅ Backend service account has Secrets Manager + S3 access
- **Location**: `infrastructure/aws/eks-cluster-config.yaml`

#### Node Group IAM
- ✅ AutoScaler permissions
- ✅ ALB Ingress permissions
- ✅ CloudWatch logging
- ✅ EBS volume permissions
- **Location**: `infrastructure/aws/eks-cluster-config.yaml`

### 7. Monitoring & Logging

#### CloudWatch Logging
- ✅ API server logging enabled
- ✅ Audit logging enabled
- ✅ Authenticator logging enabled
- ✅ Controller manager logging enabled
- ✅ Scheduler logging enabled
- ✅ 7-day log retention
- **Location**: `infrastructure/aws/eks-cluster-config.yaml`

#### ALB Access Logs
- ⚠️ **ACTION REQUIRED**: S3 bucket for ALB logs not yet created
- 📝 Create bucket: `mini-xdr-alb-logs`
- **Location**: `infrastructure/aws/security-hardening.yaml` line 250

### 8. Backup & Disaster Recovery

#### Database Backups
- ⚠️ **VERIFY**: RDS automated backups should be configured
- ⚠️ **VERIFY**: Backup retention period should be set
- 📝 Check RDS configuration separately

#### Persistent Volumes
- ✅ PVCs configured for models and data
- ✅ EBS CSI driver installed
- **Location**: `k8s/backend-deployment.yaml`

## 🔒 Security Checklist

### Pre-Deployment Verification

- [x] Kubernetes secrets created (not hardcoded)
- [x] Network policies applied
- [x] Pod security contexts configured
- [x] Resource limits set
- [x] Service accounts with minimal permissions
- [ ] SSL/TLS certificate configured (⚠️ ACTION REQUIRED)
- [ ] ALB access logs S3 bucket created (⚠️ ACTION REQUIRED)
- [ ] WAF configured (optional but recommended)
- [x] Secrets Manager integration enabled
- [x] CloudWatch logging enabled

### Runtime Security Verification

- [x] All pods running as non-root
- [x] No privileged containers
- [x] Network policies enforced
- [x] Secrets not exposed in environment variables
- [x] Health checks passing
- [x] Resource limits respected

## 🚀 Deployment Verification Steps

### 1. Verify Secrets Manager Setup

```bash
# Check if secrets exist in AWS Secrets Manager
aws secretsmanager list-secrets --region us-east-1 | grep mini-xdr

# Test secret retrieval (replace with your secret name)
aws secretsmanager get-secret-value \
  --secret-id mini-xdr/api-key \
  --region us-east-1
```

### 2. Verify EKS Cluster

```bash
# Verify cluster access
kubectl get nodes

# Verify namespaces
kubectl get namespaces

# Verify service accounts
kubectl get serviceaccounts -n mini-xdr

# Verify network policies
kubectl get networkpolicies -n mini-xdr

# Verify resource quotas
kubectl get resourcequota -n mini-xdr
```

### 3. Verify Deployments

```bash
# Check pod security contexts
kubectl get pods -n mini-xdr -o jsonpath='{.items[*].spec.securityContext}'

# Verify all pods are running as non-root
kubectl get pods -n mini-xdr -o jsonpath='{.items[*].spec.containers[*].securityContext.runAsUser}'

# Check resource limits
kubectl describe pods -n mini-xdr | grep -A 5 "Limits:"
```

### 4. Verify Network Security

```bash
# Check ingress configuration
kubectl get ingress -n mini-xdr

# Verify ALB security groups
aws ec2 describe-security-groups \
  --group-names mini-xdr-alb-sg \
  --region us-east-1

# Test network policies (should deny unauthorized access)
kubectl exec -it <backend-pod> -n mini-xdr -- curl http://<unauthorized-pod>:8000
```

### 5. Verify Secrets Management

```bash
# Verify secrets are loaded from Secrets Manager
kubectl logs -n mini-xdr deployment/mini-xdr-backend | grep "Secrets Manager"

# Check that secrets are not in environment variables
kubectl get pods -n mini-xdr -o json | jq '.items[].spec.containers[].env[] | select(.name | contains("KEY") or contains("SECRET"))'
```

## 📋 Required Actions for 100% Security

### Critical (Must Complete)

1. **Enable SSL/TLS**
   ```bash
   # Create ACM certificate
   aws acm request-certificate \
     --domain-name your-domain.com \
     --validation-method DNS \
     --region us-east-1

   # Update k8s/ingress-alb.yaml with certificate ARN
   ```

2. **Create ALB Access Logs S3 Bucket**
   ```bash
   aws s3 mb s3://mini-xdr-alb-logs --region us-east-1
   aws s3api put-bucket-policy --bucket mini-xdr-alb-logs --policy file://alb-logs-policy.json
   ```

### Recommended (Enhanced Security)

3. **Configure AWS WAF**
   - Create WAF web ACL
   - Attach to ALB (see annotation in security-hardening.yaml line 238)
   - Enable rate limiting rules

4. **Enable GuardDuty**
   - Enable AWS GuardDuty in us-east-1
   - Configure findings notifications

5. **Configure CloudTrail**
   - Enable CloudTrail for API audit logging
   - Store logs in S3 with encryption

6. **Review IP Whitelisting**
   - For production, consider VPN instead of IP whitelist
   - Or use AWS Client VPN for secure access

## 🔐 Secrets to Configure in AWS Secrets Manager

Create these secrets in AWS Secrets Manager:

```bash
# Template for creating secrets
aws secretsmanager create-secret \
  --name mini-xdr/api-key \
  --secret-string "your-api-key-here" \
  --region us-east-1

# Required secrets:
# - mini-xdr/api-key
# - mini-xdr/openai-api-key
# - mini-xdr/xai-api-key (optional)
# - mini-xdr/abuseipdb-api-key
# - mini-xdr/virustotal-api-key
# - mini-xdr/database-password (if using RDS)
# - mini-xdr/jwt-secret-key
```

## ✅ Verification Script

Run the comprehensive verification script:

```bash
./scripts/verify-aws-security.sh
```

This script checks:
- ✅ Kubernetes cluster security
- ✅ Network policies
- ✅ Pod security contexts
- ✅ Secrets management
- ✅ IAM permissions
- ✅ SSL/TLS configuration
- ✅ Logging configuration

## 📊 Security Compliance Status

| Category | Status | Notes |
|----------|--------|-------|
| Pod Security | ✅ Complete | Restricted policy enforced |
| Network Security | ✅ Complete | Network policies configured |
| Secrets Management | ✅ Complete | AWS Secrets Manager integrated |
| IAM & RBAC | ✅ Complete | Least privilege configured |
| SSL/TLS | ⚠️ Pending | Certificate setup required |
| Monitoring | ✅ Complete | CloudWatch logging enabled |
| Access Control | ✅ Complete | IP whitelist + service accounts |
| Container Security | ✅ Complete | Non-root, minimal images |

## 🎯 Next Steps

1. **Complete SSL/TLS setup** (Critical)
2. **Create ALB logs bucket** (Recommended)
3. **Configure WAF** (Recommended)
4. **Review and update IP whitelist** (Production consideration)
5. **Test disaster recovery procedures** (Recommended)

## 📚 Additional Resources

- [EKS Security Best Practices](https://aws.github.io/aws-eks-best-practices/security/)
- [Kubernetes Pod Security Standards](https://kubernetes.io/docs/concepts/security/pod-security-standards/)
- [AWS Secrets Manager Best Practices](https://docs.aws.amazon.com/secretsmanager/latest/userguide/best-practices.html)
