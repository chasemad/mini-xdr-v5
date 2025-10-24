# AWS IAM Setup for Seamless Onboarding

## Quick Setup Guide for Mini Corp Testing

### Step 1: Create IAM Role

```bash
# Create the trust policy file
cat > /tmp/trust-policy.json <<'EOF'
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "AWS": "arn:aws:iam::YOUR_ACCOUNT_ID:root"
      },
      "Action": "sts:AssumeRole",
      "Condition": {
        "StringEquals": {
          "sts:ExternalId": "mini-xdr-org-2"
        }
      }
    }
  ]
}
EOF

# Replace YOUR_ACCOUNT_ID with your AWS account ID
sed -i '' 's/YOUR_ACCOUNT_ID/675076709589/g' /tmp/trust-policy.json

# Create the role
aws iam create-role \
  --role-name MiniXDR-SeamlessOnboarding \
  --assume-role-policy-document file:///tmp/trust-policy.json \
  --description "Mini-XDR Seamless Onboarding Role for asset discovery and agent deployment"
```

### Step 2: Create IAM Policy

```bash
# Create the permissions policy file
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
        "ec2:DescribeNetworkInterfaces",
        "rds:DescribeDBInstances",
        "rds:DescribeDBClusters"
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

# Create the policy
aws iam create-policy \
  --policy-name MiniXDR-SeamlessOnboarding-Policy \
  --policy-document file:///tmp/permissions-policy.json \
  --description "Permissions for Mini-XDR seamless onboarding"

# Attach policy to role
aws iam attach-role-policy \
  --role-name MiniXDR-SeamlessOnboarding \
  --policy-arn arn:aws:iam::675076709589:policy/MiniXDR-SeamlessOnboarding-Policy
```

### Step 3: Get Role ARN

```bash
# Get the role ARN (you'll need this for Mini Corp configuration)
aws iam get-role --role-name MiniXDR-SeamlessOnboarding --query 'Role.Arn' --output text
```

**Expected output:**
```
arn:aws:iam::675076709589:role/MiniXDR-SeamlessOnboarding
```

### Step 4: Test AssumeRole

```bash
# Test that you can assume the role
aws sts assume-role \
  --role-arn arn:aws:iam::675076709589:role/MiniXDR-SeamlessOnboarding \
  --role-session-name test-session \
  --external-id mini-xdr-org-2

# If successful, you'll see temporary credentials
```

---

## Configuration for Mini-XDR

Once the role is created, use these credentials in the seamless onboarding flow:

```json
{
  "provider": "aws",
  "credentials": {
    "role_arn": "arn:aws:iam::675076709589:role/MiniXDR-SeamlessOnboarding",
    "external_id": "mini-xdr-org-2"
  }
}
```

---

## EC2 Instance Requirements (for Agent Deployment)

For SSM-based agent deployment to work, your EC2 instances need:

### 1. SSM Agent Installed
Most Amazon Linux and Ubuntu AMIs have it pre-installed. To verify:
```bash
# On the EC2 instance
sudo systemctl status amazon-ssm-agent
```

### 2. IAM Instance Profile
Attach a role with `AmazonSSMManagedInstanceCore` policy:

```bash
# Create instance role
aws iam create-role \
  --role-name MiniXDR-EC2-SSM-Role \
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
  --role-name MiniXDR-EC2-SSM-Role \
  --policy-arn arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore

# Create instance profile
aws iam create-instance-profile \
  --instance-profile-name MiniXDR-EC2-SSM-Profile

# Add role to profile
aws iam add-role-to-instance-profile \
  --instance-profile-name MiniXDR-EC2-SSM-Profile \
  --role-name MiniXDR-EC2-SSM-Role

# Attach to Mini Corp instances
aws ec2 associate-iam-instance-profile \
  --instance-id i-XXXXXXXXX \
  --iam-instance-profile Name=MiniXDR-EC2-SSM-Profile
```

### 3. Network Connectivity
Ensure security groups allow:
- **Outbound HTTPS (443)** to SSM endpoints
- **Outbound HTTPS (443)** to Mini-XDR backend (ALB)

---

## Verification Checklist

- [ ] IAM role created with correct trust policy
- [ ] External ID matches: `mini-xdr-org-2`
- [ ] Permissions policy attached
- [ ] Can successfully assume role with AWS CLI
- [ ] EC2 instances have SSM agent running
- [ ] EC2 instances have IAM instance profile attached
- [ ] Security groups allow required outbound traffic
- [ ] Role ARN saved for Mini-XDR configuration

---

## Security Best Practices

1. **Use External ID**: Always use external ID for cross-account access
2. **Least Privilege**: Only grant required permissions (discovery + SSM)
3. **Rotate Credentials**: Use AssumeRole for temporary credentials
4. **Monitor Usage**: Enable CloudTrail logging for AssumeRole calls
5. **Scope Resources**: Consider adding resource constraints to policies

---

## Troubleshooting

### "Access Denied" when assuming role
- Verify trust policy includes your account ARN
- Check external ID matches exactly
- Ensure IAM user/role has `sts:AssumeRole` permission

### "SSM agent not available"
- Install SSM agent: `sudo snap install amazon-ssm-agent --classic`
- Verify IAM instance profile is attached
- Check VPC endpoints or internet gateway for SSM connectivity

### Instances not discovered
- Verify EC2 describe permissions
- Check instances are in `running` state
- Ensure at least one region has instances

---

## Quick Test Command

```bash
# Test discovery (requires jq)
aws ec2 describe-instances \
  --profile mini-xdr-assumed \
  --region us-east-1 \
  --filters "Name=instance-state-name,Values=running" \
  --query 'Reservations[].Instances[].[InstanceId,InstanceType,State.Name,Tags[?Key==`Name`].Value|[0]]' \
  --output table
```

This shows you what Mini-XDR will discover!
