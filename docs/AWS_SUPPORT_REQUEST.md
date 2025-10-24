# AWS Support Request - Enable Load Balancer Creation

## How to Request Load Balancer Access

### Step 1: Go to AWS Support Center
https://console.aws.amazon.com/support/home

### Step 2: Create a Case

**Case Type:** Service Limit Increase

**Limit Type:** Elastic Load Balancing (ELB)

**Request Details:**
```
Subject: Enable Application Load Balancer Creation

Description:
I need to create an Application Load Balancer (ALB) for my Kubernetes cluster 
running in us-east-1. I'm getting the error:

"This AWS account currently does not support creating load balancers."

Use Case: 
I'm running a security monitoring platform (XDR) on EKS and need an ALB to 
expose my web application with proper SSL/TLS encryption and health checks.

Required Resources:
- 1 Application Load Balancer in us-east-1
- Internet-facing scheme
- Used with EKS cluster

Region: us-east-1
Service: Elastic Load Balancing v2 (Application Load Balancer)
```

**Expected Resolution Time:** 1-2 hours during business hours

### Option 2: If You Have AWS Credits/Education Account

Some account types (free tier, education) have restrictions. Options:
1. Upgrade to paid account (no immediate charge, pay-as-you-go)
2. Apply for AWS Educate/Activate credits if you're a student/startup

### Option 3: Check Account Status

```bash
# Check if your account has load balancer restrictions
aws elbv2 describe-load-balancers --region us-east-1

# Check your account limits
aws service-quotas get-service-quota \
  --service-code elasticloadbalancing \
  --quota-code L-53DA6B97 \
  --region us-east-1
```

### After Approval

Once AWS enables load balancers, run:
```bash
cd /Users/chasemad/Desktop/mini-xdr
./deploy-alb-with-org.sh
```

---

## Alternative: Use NodePort (Temporary Solution)

While waiting for AWS approval, you can expose via NodePort:

**Pros:**
- Works immediately, no approval needed
- Gets you a public IP

**Cons:**
- Less professional (uses port like :30000)
- No SSL/TLS without extra work
- Not recommended for production

See: NODEPORT_SETUP.md for instructions


