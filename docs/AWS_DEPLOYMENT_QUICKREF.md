# Mini-XDR AWS Deployment - Quick Reference

**Everything you need to know on one page**

---

## 📁 Files You Have

1. **`AWS_COMPLETE_DEPLOYMENT.md`** ⭐ **MAIN GUIDE**
   - Complete step-by-step deployment
   - 2,700+ lines of detailed instructions
   - Deploys Mini-XDR + 13-server Mini Corp network
   - Copy-paste ready commands

2. **`scripts/aws/deploy-complete-mini-xdr.sh`**
   - Automated deployment script
   - Deploys Mini-XDR application
   - Helps with Mini Corp network setup

3. **`docs/AWS_DEPLOYMENT_COMPLETE_GUIDE.md`**
   - Technical reference
   - Advanced configurations
   - Production best practices

4. **`docs/CLOUD_DEPLOYMENT_OPTIONS.md`**
   - Alternative cloud providers
   - Cost comparisons
   - Migration guides

---

## 🚀 Quick Deploy (Choose One)

### Option 1: Fully Manual (Recommended for Learning)
```bash
# Follow step-by-step in AWS_COMPLETE_DEPLOYMENT.md
# Understand every component as you deploy
# Total time: 60 minutes
```

### Option 2: Semi-Automated
```bash
# Run the automation script
cd /Users/chasemad/Desktop/mini-xdr
./scripts/aws/deploy-complete-mini-xdr.sh

# Then manually deploy Mini Corp network (Part 4)
# Total time: 45 minutes
```

### Option 3: Minimal (Just Mini-XDR, No Test Network)
```bash
# Follow Parts 1-3 only
# Skip Mini Corp network to save costs
# Total time: 35 minutes
# Cost: ~$215/month
```

---

## 📊 What Gets Deployed

### Full Deployment (All Components)

```
┌─────────────────────────────────────────────────────────────┐
│                     Your IP Address                         │
│                     (ONLY ACCESS)                            │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
    ┌────────────────────────────────────────────────────┐
    │  Application Load Balancer (Secured to your IP)    │
    └────────────────┬───────────────────────────────────┘
                     │
         ┌───────────┴───────────┐
         │                       │
         ▼                       ▼
┌─────────────────┐    ┌─────────────────┐
│  Mini-XDR       │    │  Mini-XDR       │
│  Frontend       │    │  Backend        │
│  (Next.js)      │    │  (FastAPI)      │
│  2 pods         │    │  2 pods         │
└─────────────────┘    └────────┬────────┘
                                │
                    ┌───────────┴──────────┐
                    │                      │
                    ▼                      ▼
           ┌─────────────────┐    ┌─────────────────┐
           │  RDS PostgreSQL │    │  ElastiCache    │
           │  (encrypted)    │    │  Redis          │
           └─────────────────┘    └─────────────────┘
                                
═══════════════════════════════════════════════════════════════

            Mini Corp Network (13 Servers)
            All monitored by Mini-XDR above

Infrastructure:      File & Collab:       Applications:
• Domain Controller  • File Server        • Website (DVWA)
• DNS Server        • Email Server       • Database (MySQL)
                                         • CRM App

Workstations:       Security:            Honeypots:
• Finance PC        • VPN Gateway        • SSH Honeypot
• Engineering PC                         • FTP Server
• HR PC
```

---

## 💵 Cost Summary

| Configuration | Monthly Cost | Use Case |
|--------------|--------------|----------|
| **Minimal** (No test network) | ~$215 | Demo only, lowest cost |
| **Standard** (2 nodes, test network) | ~$459 | Full demo, all features |
| **Optimized** (Spot + auto-stop) | ~$200 | Cost-conscious full demo |
| **Production** (Multi-AZ, HA) | ~$800+ | Real production use |

**To minimize costs:**
- Deploy only when demoing
- Stop test network instances when not testing
- Use spot instances
- Delete after demo, redeploy when needed

---

## ⚡ Quick Commands

### Check Status
```bash
# Cluster
kubectl get nodes

# Mini-XDR
kubectl get pods -n mini-xdr
kubectl get svc -n mini-xdr
kubectl get ingress -n mini-xdr

# Test Network
aws ec2 describe-instances \
  --filters "Name=tag:MonitoredBy,Values=mini-xdr" \
  --query 'Reservations[].Instances[].[Tags[?Key==`Name`].Value | [0], State.Name, PublicIpAddress]' \
  --output table
```

### Access Application
```bash
# Get URL
ALB_URL=$(kubectl get ingress mini-xdr-ingress -n mini-xdr -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')
echo "Dashboard: http://$ALB_URL"

# Test access
curl -I http://$ALB_URL

# Open in browser
open http://$ALB_URL
```

### View Logs
```bash
# Backend logs
kubectl logs -n mini-xdr -l app=mini-xdr-backend -f

# Frontend logs
kubectl logs -n mini-xdr -l app=mini-xdr-frontend -f

# All Mini-XDR logs
kubectl logs -n mini-xdr --all-containers -f
```

### Update Application
```bash
# Rebuild and push new image
cd backend
docker build -t ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/mini-xdr-backend:latest .
docker push ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/mini-xdr-backend:latest

# Restart deployment
kubectl rollout restart deployment/mini-xdr-backend -n mini-xdr
kubectl rollout status deployment/mini-xdr-backend -n mini-xdr
```

### Control Test Network
```bash
# Start all test instances
aws ec2 start-instances --instance-ids $(aws ec2 describe-instances \
  --filters "Name=tag:MonitoredBy,Values=mini-xdr" "Name=instance-state-name,Values=stopped" \
  --query 'Reservations[].Instances[].InstanceId' --output text)

# Stop all test instances (save money)
aws ec2 stop-instances --instance-ids $(aws ec2 describe-instances \
  --filters "Name=tag:MonitoredBy,Values=mini-xdr" "Name=instance-state-name,Values=running" \
  --query 'Reservations[].Instances[].InstanceId' --output text)
```

### Update IP Whitelist
```bash
# If your IP changes
NEW_IP=$(curl -s https://ifconfig.me)
ALB_SG=$(aws cloudformation describe-stacks \
  --stack-name mini-xdr-infrastructure \
  --query 'Stacks[0].Outputs[?OutputKey==`ALBSecurityGroup`].OutputValue' \
  --output text)

# Remove old rule
aws ec2 revoke-security-group-ingress \
  --group-id $ALB_SG \
  --ip-permissions IpProtocol=tcp,FromPort=80,ToPort=80,IpRanges="[{CidrIp=$MY_IP/32}]"

# Add new rule
aws ec2 authorize-security-group-ingress \
  --group-id $ALB_SG \
  --ip-permissions IpProtocol=tcp,FromPort=80,ToPort=80,IpRanges="[{CidrIp=$NEW_IP/32}]"

echo "Updated whitelist to: $NEW_IP"
```

---

## 🧪 Test Attack Scenarios

```bash
# Get test instance IPs
WEB_IP=$(aws ec2 describe-instances \
  --filters "Name=tag:Name,Values=corp-web01" "Name=instance-state-name,Values=running" \
  --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)

# SQL Injection
curl "http://${WEB_IP}/dvwa/vulnerabilities/sqli/?id=1'+OR+'1'='1"

# Port Scan
nmap -sT $WEB_IP

# Brute Force (get SSH honeypot IP first)
SSH_IP=$(aws ec2 describe-instances \
  --filters "Name=tag:Name,Values=corp-honeypot-ssh" "Name=instance-state-name,Values=running" \
  --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)

for i in {1..10}; do
  timeout 1 ssh -o StrictHostKeyChecking=no test@${SSH_IP} 2>/dev/null &
done
```

---

## 🗑️ Clean Up Everything

```bash
# WARNING: This deletes EVERYTHING and is irreversible

# 1. Delete EKS cluster
eksctl delete cluster --name mini-xdr-prod --region us-east-1

# 2. Terminate test instances
aws ec2 terminate-instances --instance-ids $(aws ec2 describe-instances \
  --filters "Name=tag:MonitoredBy,Values=mini-xdr" \
  --query 'Reservations[].Instances[].InstanceId' --output text)

# 3. Delete CloudFormation stack
aws cloudformation delete-stack --stack-name mini-xdr-infrastructure

# 4. Delete ECR repos
aws ecr delete-repository --repository-name mini-xdr-backend --force
aws ecr delete-repository --repository-name mini-xdr-frontend --force

# 5. Empty and delete S3 buckets
aws s3 rb s3://mini-xdr-prod-logs-${AWS_ACCOUNT_ID} --force
aws s3 rb s3://mini-xdr-prod-ml-models-${AWS_ACCOUNT_ID} --force

# Done! All resources deleted.
```

---

## 🎯 For Recruiter Demos

### 5-Minute Demo Flow

1. **Show Dashboard** (1 min)
   - "This is Mini-XDR monitoring a 13-server corporate network"
   - Open: http://[ALB-URL]

2. **Show Architecture** (1 min)
   - Pull up network diagram
   - Show AWS Console (EC2, EKS)

3. **Live Attack** (2 min)
   - Run SQL injection script
   - Show real-time detection in dashboard

4. **Explain Tech** (1 min)
   - "Built with Python, TypeScript, Kubernetes"
   - "ML models detect threats, AI agents respond"

### Key Talking Points
- ✅ "Production-grade Kubernetes deployment on AWS EKS"
- ✅ "13 servers across 6 tiers: Infrastructure, File, App, Workstation, Security, Honeypot"
- ✅ "Real-time ML threat detection with specialist models"
- ✅ "Zero-trust architecture, everything encrypted"
- ✅ "Can handle thousands of events per second"

---

## 📞 Get Help

**If something breaks:**

1. Check logs:
   ```bash
   kubectl logs -n mini-xdr -l app=mini-xdr-backend --tail=100
   ```

2. Verify pods are running:
   ```bash
   kubectl get pods -n mini-xdr
   ```

3. Check AWS CloudWatch:
   - EKS logs: CloudWatch → Log groups → `/aws/eks/mini-xdr-prod/cluster`
   - VPC Flow Logs: `/aws/vpc/mini-xdr-flow-logs`

4. Check deployment guide: `AWS_COMPLETE_DEPLOYMENT.md`

5. Troubleshooting section in main guide

---

## ✅ Pre-Demo Checklist

Before showing to recruiters:

- [ ] `curl http://[ALB-URL]` returns 200 OK
- [ ] Dashboard loads in browser
- [ ] Can login to dashboard
- [ ] Test instances are running (if using them)
- [ ] Can successfully run attack simulation
- [ ] Alerts appear in dashboard
- [ ] Have screenshots as backup
- [ ] GitHub repo is updated
- [ ] Resume mentions the project

---

## 🔥 Pro Tips

1. **Test 24 hours before demo**
   - Ensure everything works
   - Take screenshots/videos as backup

2. **Have ALB URL ready**
   - Save it somewhere accessible
   - Test from different networks

3. **Prepare to go technical**
   - Have code open in IDE
   - Know your architecture cold
   - Explain design decisions confidently

4. **Highlight business value**
   - "Reduces time to detect from hours to seconds"
   - "Automates tier-1 SOC analyst work"
   - "Scales from 10 to 10,000 hosts"

5. **Cost management**
   - Stop instances after demo
   - Use spot instances
   - Delete and redeploy as needed

---

## 📈 What Recruiters Will Love

- **Scale**: Kubernetes cluster, auto-scaling, multi-tier architecture
- **Security**: Zero-trust, encryption, least privilege, compliance-ready
- **ML/AI**: Custom models, real-time classification, automated response  
- **Modern Stack**: Python, TypeScript, Docker, Kubernetes, AWS
- **Production-Ready**: HA, monitoring, logging, security hardening
- **Complete Solution**: Not just code - full infrastructure, testing, deployment

---

**Ready to deploy?** Open `AWS_COMPLETE_DEPLOYMENT.md` and start at Part 1!

**Questions?** Every step is documented with explanations and expected outputs.

**Good luck with your demos!** 🚀


