# 🚀 Quick Start - Testing Seamless Onboarding

## One-Command Testing

```bash
cd .
./scripts/test-seamless-onboarding.sh full
```

This single command will:
- ✅ Setup all AWS IAM roles
- ✅ Launch 3 test EC2 instances
- ✅ Run seamless onboarding
- ✅ Display all results

---

## Pre-Requisites (One-Time Setup)

### 1. Deploy Updated Backend

```bash
# From your Mac
cd .

# Make sure latest code is committed
git add .
git commit -m "feat: add seamless onboarding system"
git push origin main
```

### 2. Build & Deploy Backend Image

```bash
# SSH to EC2 build instance
ssh -i ~/.ssh/mini-xdr-eks-key.pem ec2-user@<EC2-IP>

cd /home/ec2-user/mini-xdr-v2
git pull origin main

cd backend
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin 116912495274.dkr.ecr.us-east-1.amazonaws.com

docker build -t 116912495274.dkr.ecr.us-east-1.amazonaws.com/mini-xdr-backend:latest .
docker push 116912495274.dkr.ecr.us-east-1.amazonaws.com/mini-xdr-backend:latest

exit

# Restart deployment
kubectl rollout restart deployment/mini-xdr-backend -n mini-xdr
kubectl rollout status deployment/mini-xdr-backend -n mini-xdr
```

### 3. Run Migration

```bash
kubectl exec -it deployment/mini-xdr-backend -n mini-xdr -- alembic upgrade head
```

### 4. Create Test Organization

```bash
kubectl exec -it deployment/mini-xdr-backend -n mini-xdr -- bash -c '
python << "EOF"
import asyncio
from app.db import get_async_session_local
from app.models import Organization, User
from app.auth import get_password_hash

async def create():
    async for db in get_async_session_local():
        org = Organization(
            name="Test Organization", slug="test-org",
            onboarding_flow_version="seamless", auto_discovery_enabled=True,
            integration_settings={"agent_public_base_url": "http://k8s-minixdr-minixdri-dc5fc1df8b-1132128475.us-east-1.elb.amazonaws.com"},
            onboarding_status="not_started"
        )
        db.add(org)
        await db.commit()
        await db.refresh(org)

        user = User(
            organization_id=org.id, email="test@minixdr.com",
            hashed_password=get_password_hash("TestPassword123!"),
            full_name="Test User", role="admin", is_active=True
        )
        db.add(user)
        await db.commit()
        print(f"✅ Created: test@minixdr.com / TestPassword123!")
        break

asyncio.run(create())
EOF
'
```

---

## Run Full Test

```bash
cd .
./scripts/test-seamless-onboarding.sh full
```

**Expected output:**
```
🚀 CHECKING PREREQUISITES
✅ AWS CLI installed
✅ kubectl installed
✅ jq installed
✅ AWS credentials configured
✅ kubectl connected to EKS cluster
✅ Backend is reachable

🚀 SETTING UP IAM ROLES
✅ Created IAM role: MiniXDR-SeamlessOnboarding-Test
✅ Created IAM policy
✅ IAM role ARN: arn:aws:iam::116912495274:role/MiniXDR-SeamlessOnboarding-Test
✅ AssumeRole test successful

🚀 LAUNCHING TEST EC2 INSTANCES
✅ Instances launched: 3

🚀 TESTING SEAMLESS ONBOARDING
✅ Login successful
✅ Onboarding initiated!
--- Progress monitoring ---
Discovery: 100% - 3 assets found
Deployment: 100% - 3 agents deployed
Validation: Running

=== FINAL RESULTS ===
Discovered Assets: 3 EC2 instances
Deployment Summary: 3 pending/deploying
Validation: 3/5 checks passed
✅ Testing complete!
```

---

## Manual Verification

### Check Backend Logs

```bash
kubectl logs -f deployment/mini-xdr-backend -n mini-xdr | grep -E "(discovery|deployment|AWS)"
```

### View Progress via API

```bash
ALB_URL="http://k8s-minixdr-minixdri-dc5fc1df8b-1132128475.us-east-1.elb.amazonaws.com"

# Login
JWT=$(curl -s -X POST $ALB_URL/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"test@minixdr.com","password":"TestPassword123!"}' \
  | jq -r '.access_token')

# Check progress
curl -s "$ALB_URL/api/onboarding/v2/progress" \
  -H "Authorization: Bearer $JWT" | jq '.'
```

---

## Cleanup

```bash
./scripts/test-seamless-onboarding.sh cleanup
```

Or manually:

```bash
# Terminate instances
aws ec2 terminate-instances --instance-ids $(
  aws ec2 describe-instances \
    --filters "Name=tag:Purpose,Values=seamless-onboarding-test" \
    --query 'Reservations[].Instances[].InstanceId' \
    --output text
)
```

---

## Troubleshooting Commands

### Backend Health

```bash
# Check pods
kubectl get pods -n mini-xdr

# Check logs
kubectl logs deployment/mini-xdr-backend -n mini-xdr --tail=100

# Test health
curl http://k8s-minixdr-minixdri-dc5fc1df8b-1132128475.us-east-1.elb.amazonaws.com/health
```

### Migration Issues

```bash
# Check migration status
kubectl exec -it deployment/mini-xdr-backend -n mini-xdr -- alembic current

# Rollback if needed
kubectl exec -it deployment/mini-xdr-backend -n mini-xdr -- alembic downgrade -1
```

### AWS Issues

```bash
# Test AssumeRole
aws sts assume-role \
  --role-arn arn:aws:iam::116912495274:role/MiniXDR-SeamlessOnboarding-Test \
  --role-session-name test \
  --external-id mini-xdr-test-org

# Check test instances
aws ec2 describe-instances \
  --filters "Name=tag:Purpose,Values=seamless-onboarding-test" \
  --query 'Reservations[].Instances[].[InstanceId,State.Name]' \
  --output table
```

---

## Full Documentation

| Document | Purpose |
|----------|---------|
| `TESTING_READY_SUMMARY.md` | **Complete testing overview** |
| `SEAMLESS_ONBOARDING_TESTING_GUIDE.md` | Detailed step-by-step guide |
| `SEAMLESS_ONBOARDING_IMPLEMENTATION_SUMMARY.md` | Implementation details |
| `AWS_IAM_SETUP_FOR_SEAMLESS_ONBOARDING.md` | IAM setup instructions |

---

## Test Credentials

**Test Organization:**
- Email: `test@minixdr.com`
- Password: `TestPassword123!`
- Org Slug: `test-org`
- ALB URL: `http://k8s-minixdr-minixdri-dc5fc1df8b-1132128475.us-east-1.elb.amazonaws.com`

---

## Success Criteria

✅ Backend deploys without errors
✅ Migration creates new tables
✅ Test organization created
✅ AWS IAM role can be assumed
✅ 3 EC2 instances discovered
✅ Agent deployment commands sent
✅ Progress updates work
✅ Validation checks pass

---

**Ready to test! Run: `./scripts/test-seamless-onboarding.sh full`** 🚀
