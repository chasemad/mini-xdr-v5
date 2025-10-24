# Seamless Onboarding Testing Guide

## Quick Start - Testing Without Mini Corp

This guide walks you through testing the new seamless onboarding system using simple test EC2 instances instead of waiting for the Mini Corp network deployment.

---

## Prerequisites Checklist

- [x] EKS cluster running (`mini-xdr-cluster`)
- [x] Backend code with seamless onboarding implemented
- [x] AWS CLI configured (`aws sts get-caller-identity` works)
- [x] kubectl configured for EKS cluster
- [x] SSH key for EC2 build instance (`~/.ssh/mini-xdr-eks-key.pem`)

---

## Part 1: Deploy Updated Backend Code

### Step 1: SSH to Build Instance

```bash
# Replace <EC2-BUILD-IP> with your actual build instance IP
ssh -i ~/.ssh/mini-xdr-eks-key.pem ec2-user@<EC2-BUILD-IP>
```

### Step 2: Pull Latest Code

```bash
cd /home/ec2-user/mini-xdr-v2
git fetch --all
git checkout main
git pull origin main
```

### Step 3: Install boto3 (if not already installed)

```bash
# In the backend directory
cd /home/ec2-user/mini-xdr-v2/backend
pip install boto3>=1.28.0
```

### Step 4: Build Backend Image

```bash
cd /home/ec2-user/mini-xdr-v2/backend

# Login to ECR
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin \
  116912495274.dkr.ecr.us-east-1.amazonaws.com

# Build image
docker build \
  --build-arg VERSION="1.2.0-seamless" \
  --build-arg VCS_REF="$(git rev-parse --short HEAD)" \
  --build-arg BUILD_DATE="$(date -u +"%Y-%m-%dT%H:%M:%SZ")" \
  -t 116912495274.dkr.ecr.us-east-1.amazonaws.com/mini-xdr-backend:1.2.0-seamless \
  -t 116912495274.dkr.ecr.us-east-1.amazonaws.com/mini-xdr-backend:latest \
  .
```

### Step 5: Push to ECR

```bash
docker push 116912495274.dkr.ecr.us-east-1.amazonaws.com/mini-xdr-backend:1.2.0-seamless
docker push 116912495274.dkr.ecr.us-east-1.amazonaws.com/mini-xdr-backend:latest
```

### Step 6: Deploy to EKS

```bash
# Exit SSH session
exit

# From your local machine:
kubectl patch deployment mini-xdr-backend -n mini-xdr -p \
  '{"spec":{"template":{"spec":{"containers":[{"name":"backend","imagePullPolicy":"Always"}]}}}}'

kubectl rollout restart deployment/mini-xdr-backend -n mini-xdr

# Watch deployment
kubectl rollout status deployment/mini-xdr-backend -n mini-xdr
```

---

## Part 2: Run Database Migration

```bash
# Run Alembic migration
kubectl exec -it deployment/mini-xdr-backend -n mini-xdr -- \
  alembic upgrade head

# Verify migration
kubectl exec -it deployment/mini-xdr-backend -n mini-xdr -- \
  alembic current

# Should show: 99d70952c5da (head)
```

---

## Part 3: Create Test Organization

```bash
# Connect to backend pod
kubectl exec -it deployment/mini-xdr-backend -n mini-xdr -- bash

# Create test organization and user
python << 'EOF'
import asyncio
from app.db import get_async_session_local
from app.models import Organization, User
from app.auth import get_password_hash

async def create_test_org():
    async for db in get_async_session_local():
        # Create test organization
        test_org = Organization(
            name="Test Organization",
            slug="test-org",
            onboarding_flow_version="seamless",
            auto_discovery_enabled=True,
            integration_settings={
                "agent_public_base_url": "http://k8s-minixdr-minixdri-dc5fc1df8b-1132128475.us-east-1.elb.amazonaws.com"
            },
            onboarding_status="not_started"
        )
        db.add(test_org)
        await db.commit()
        await db.refresh(test_org)

        # Create test user
        test_user = User(
            organization_id=test_org.id,
            email="test@minixdr.com",
            hashed_password=get_password_hash("TestPassword123!"),
            full_name="Test User",
            role="admin",
            is_active=True
        )
        db.add(test_user)
        await db.commit()

        print(f"✅ Created test org: ID={test_org.id}, slug={test_org.slug}")
        print(f"✅ Created test user: email=test@minixdr.com, password=TestPassword123!")
        print(f"✅ ALB URL configured: {test_org.integration_settings.get('agent_public_base_url')}")
        break

asyncio.run(create_test_org())
EOF

# Exit pod
exit
```

**Save these credentials:**
- Email: `test@minixdr.com`
- Password: `TestPassword123!`

---

## Part 4: Set Up AWS IAM Roles

### 4.1: Create Seamless Onboarding IAM Role

```bash
# Create trust policy
cat > /tmp/trust-policy.json <<'EOF'
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": {"AWS": "arn:aws:iam::116912495274:root"},
    "Action": "sts:AssumeRole",
    "Condition": {
      "StringEquals": {"sts:ExternalId": "mini-xdr-test-org"}
    }
  }]
}
EOF

# Create role
aws iam create-role \
  --role-name MiniXDR-SeamlessOnboarding-Test \
  --assume-role-policy-document file:///tmp/trust-policy.json \
  --description "Mini-XDR Seamless Onboarding Test Role"

# Create permissions policy
cat > /tmp/permissions-policy.json <<'EOF'
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "AssetDiscovery",
      "Effect": "Allow",
      "Action": [
        "ec2:DescribeRegions",
        "ec2:DescribeInstances",
        "ec2:DescribeVpcs",
        "ec2:DescribeSubnets",
        "ec2:DescribeSecurityGroups",
        "rds:DescribeDBInstances"
      ],
      "Resource": "*"
    },
    {
      "Sid": "AgentDeployment",
      "Effect": "Allow",
      "Action": [
        "ssm:DescribeInstanceInformation",
        "ssm:SendCommand",
        "ssm:GetCommandInvocation",
        "ssm:ListCommandInvocations"
      ],
      "Resource": "*"
    }
  ]
}
EOF

# Create policy
aws iam create-policy \
  --policy-name MiniXDR-SeamlessOnboarding-Test-Policy \
  --policy-document file:///tmp/permissions-policy.json

# Attach policy to role
aws iam attach-role-policy \
  --role-name MiniXDR-SeamlessOnboarding-Test \
  --policy-arn arn:aws:iam::116912495274:policy/MiniXDR-SeamlessOnboarding-Test-Policy

# Get role ARN
echo "Save this Role ARN:"
aws iam get-role --role-name MiniXDR-SeamlessOnboarding-Test --query 'Role.Arn' --output text
```

### 4.2: Create EC2 Instance Profile (for SSM)

```bash
# Create role
aws iam create-role \
  --role-name MiniXDR-Test-EC2-SSM \
  --assume-role-policy-document '{
    "Version": "2012-10-17",
    "Statement": [{
      "Effect": "Allow",
      "Principal": {"Service": "ec2.amazonaws.com"},
      "Action": "sts:AssumeRole"
    }]
  }'

# Attach SSM policy
aws iam attach-role-policy \
  --role-name MiniXDR-Test-EC2-SSM \
  --policy-arn arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore

# Create instance profile
aws iam create-instance-profile \
  --instance-profile-name MiniXDR-Test-EC2-Profile

# Add role to profile
aws iam add-role-to-instance-profile \
  --instance-profile-name MiniXDR-Test-EC2-Profile \
  --role-name MiniXDR-Test-EC2-SSM

# Wait for profile to be ready
sleep 10
```

### 4.3: Test AssumeRole

```bash
# Test that you can assume the role
ROLE_ARN="arn:aws:iam::116912495274:role/MiniXDR-SeamlessOnboarding-Test"

aws sts assume-role \
  --role-arn $ROLE_ARN \
  --role-session-name test-session \
  --external-id mini-xdr-test-org

# If successful, you'll see temporary credentials
```

---

## Part 5: Launch Test EC2 Instances

```bash
# Get default VPC and subnet
DEFAULT_VPC=$(aws ec2 describe-vpcs --filters "Name=is-default,Values=true" --query 'Vpcs[0].VpcId' --output text)
DEFAULT_SUBNET=$(aws ec2 describe-subnets --filters "Name=vpc-id,Values=$DEFAULT_VPC" --query 'Subnets[0].SubnetId' --output text)

echo "Using VPC: $DEFAULT_VPC"
echo "Using Subnet: $DEFAULT_SUBNET"

# Launch 3 test instances
aws ec2 run-instances \
  --image-id ami-0c02fb55cc1f0c4c4 \
  --instance-type t3.micro \
  --count 3 \
  --iam-instance-profile Name=MiniXDR-Test-EC2-Profile \
  --subnet-id $DEFAULT_SUBNET \
  --tag-specifications \
    'ResourceType=instance,Tags=[{Key=Name,Value=mini-xdr-test},{Key=Purpose,Value=seamless-onboarding-test},{Key=Environment,Value=test}]' \
  --user-data '#!/bin/bash
yum update -y
yum install -y amazon-ssm-agent
systemctl enable amazon-ssm-agent
systemctl start amazon-ssm-agent'

# Wait for instances to be running
echo "Waiting for instances to start..."
aws ec2 wait instance-running --instance-ids $(
  aws ec2 describe-instances \
    --filters "Name=tag:Purpose,Values=seamless-onboarding-test" "Name=instance-state-name,Values=pending,running" \
    --query 'Reservations[].Instances[].InstanceId' \
    --output text
)

# Verify instances
echo "Test instances launched:"
aws ec2 describe-instances \
  --filters "Name=tag:Purpose,Values=seamless-onboarding-test" \
  --query 'Reservations[].Instances[].[InstanceId,State.Name,InstanceType,Tags[?Key==`Name`].Value|[0]]' \
  --output table
```

---

## Part 6: Test Seamless Onboarding

### 6.1: Login and Get JWT Token

```bash
ALB_URL="http://k8s-minixdr-minixdri-dc5fc1df8b-1132128475.us-east-1.elb.amazonaws.com"

# Login
JWT_TOKEN=$(curl -s -X POST \
  $ALB_URL/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"test@minixdr.com","password":"TestPassword123!"}' \
  | jq -r '.access_token')

if [ "$JWT_TOKEN" = "null" ] || [ -z "$JWT_TOKEN" ]; then
  echo "❌ Login failed! Check credentials."
else
  echo "✅ Login successful!"
  echo "Token: ${JWT_TOKEN:0:50}..."
fi
```

### 6.2: Start Seamless Onboarding

```bash
ROLE_ARN="arn:aws:iam::116912495274:role/MiniXDR-SeamlessOnboarding-Test"

# Start quick-start onboarding
curl -X POST \
  $ALB_URL/api/onboarding/v2/quick-start \
  -H "Authorization: Bearer $JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d "{
    \"provider\": \"aws\",
    \"credentials\": {
      \"role_arn\": \"$ROLE_ARN\",
      \"external_id\": \"mini-xdr-test-org\"
    }
  }" | jq '.'
```

### 6.3: Monitor Progress

```bash
# Watch progress (refreshes every 5 seconds)
while true; do
  clear
  echo "=== Seamless Onboarding Progress ==="
  echo "Time: $(date '+%H:%M:%S')"
  echo ""

  curl -s -X GET \
    $ALB_URL/api/onboarding/v2/progress \
    -H "Authorization: Bearer $JWT_TOKEN" | jq '.'

  sleep 5
done

# Press Ctrl+C to stop monitoring
```

### 6.4: View Discovered Assets

```bash
# Get discovered assets
curl -s -X GET \
  $ALB_URL/api/onboarding/v2/assets \
  -H "Authorization: Bearer $JWT_TOKEN" | jq '.'
```

### 6.5: Check Deployment Summary

```bash
# Get deployment summary
curl -s -X GET \
  $ALB_URL/api/onboarding/v2/deployment/summary \
  -H "Authorization: Bearer $JWT_TOKEN" | jq '.'
```

### 6.6: View Validation Results

```bash
# Get validation summary
curl -s -X GET \
  $ALB_URL/api/onboarding/v2/validation/summary \
  -H "Authorization: Bearer $JWT_TOKEN" | jq '.'
```

---

## Part 7: Verify Backend Logs

```bash
# Watch backend logs
kubectl logs -f deployment/mini-xdr-backend -n mini-xdr | grep -E "(discovery|deployment|integration|AWS)"

# Check for errors
kubectl logs deployment/mini-xdr-backend -n mini-xdr --tail=500 | grep -i error
```

---

## Expected Results

### ✅ What You Should See:

1. **Quick-Start Response:**
   ```json
   {
     "status": "initiated",
     "message": "Auto-discovery started for aws...",
     "provider": "aws",
     "organization_id": <test-org-id>
   }
   ```

2. **Progress Updates:**
   - Discovery: 0% → 100% (finding 3 EC2 instances)
   - Deployment: 0% → 100% (sending SSM commands)
   - Validation: Running checks

3. **Discovered Assets:**
   ```json
   {
     "total": 3,
     "assets": [
       {
         "provider": "aws",
         "asset_type": "ec2",
         "asset_id": "i-xxxxx",
         "region": "us-east-1",
         "agent_status": "pending"
       }
     ]
   }
   ```

4. **Deployment Summary:**
   ```json
   {
     "total_assets": 3,
     "agent_deployed": 0,
     "deployment_pending": 3,
     "by_provider": {"aws": 3}
   }
   ```

---

## Troubleshooting

### Login Fails
```bash
# Check if backend is healthy
curl $ALB_URL/health

# Check backend logs
kubectl logs deployment/mini-xdr-backend -n mini-xdr --tail=100
```

### Discovery Fails
```bash
# Test AssumeRole manually
aws sts assume-role \
  --role-arn arn:aws:iam::116912495274:role/MiniXDR-SeamlessOnboarding-Test \
  --role-session-name manual-test \
  --external-id mini-xdr-test-org

# Check if instances are visible
aws ec2 describe-instances --filters "Name=tag:Purpose,Values=seamless-onboarding-test"
```

### No Assets Found
```bash
# Verify instances are running
aws ec2 describe-instances \
  --filters "Name=tag:Purpose,Values=seamless-onboarding-test" \
  --query 'Reservations[].Instances[].[InstanceId,State.Name]' \
  --output table
```

---

## Cleanup After Testing

```bash
# Terminate test instances
aws ec2 terminate-instances --instance-ids $(
  aws ec2 describe-instances \
    --filters "Name=tag:Purpose,Values=seamless-onboarding-test" \
    --query 'Reservations[].Instances[].InstanceId' \
    --output text
)

# Delete IAM resources
aws iam detach-role-policy \
  --role-name MiniXDR-SeamlessOnboarding-Test \
  --policy-arn arn:aws:iam::116912495274:policy/MiniXDR-SeamlessOnboarding-Test-Policy

aws iam delete-policy \
  --policy-arn arn:aws:iam::116912495274:policy/MiniXDR-SeamlessOnboarding-Test-Policy

aws iam delete-role --role-name MiniXDR-SeamlessOnboarding-Test

aws iam remove-role-from-instance-profile \
  --instance-profile-name MiniXDR-Test-EC2-Profile \
  --role-name MiniXDR-Test-EC2-SSM

aws iam delete-instance-profile --instance-profile-name MiniXDR-Test-EC2-Profile

aws iam detach-role-policy \
  --role-name MiniXDR-Test-EC2-SSM \
  --policy-arn arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore

aws iam delete-role --role-name MiniXDR-Test-EC2-SSM
```

---

## Next Steps After Successful Testing

1. **Deploy Mini Corp Network** (follow `MINI_CORP_AWS_NETWORK_README.md`)
2. **Update Mini Corp Organization** with seamless onboarding settings
3. **Test with Mini Corp** infrastructure
4. **Build Frontend Components** (QuickStartOnboarding.tsx, etc.)
5. **Production Hardening** (proper credential encryption, tests)

---

**You're now ready to test seamless onboarding end-to-end!** 🚀
