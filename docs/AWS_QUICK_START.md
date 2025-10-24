# Mini-XDR AWS Quick Start Guide

**Get Mini-XDR running on AWS in under 30 minutes**

---

## Prerequisites Checklist

- [ ] AWS Account with billing enabled
- [ ] AWS CLI v2 installed and configured
- [ ] kubectl installed
- [ ] eksctl installed
- [ ] Helm installed
- [ ] Docker installed
- [ ] Domain name (optional)
- [ ] $50-100 budget for testing

---

## Step-by-Step Quick Deploy

### 1. Initial Setup (5 minutes)

```bash
# Clone the repo
cd ~/Desktop/mini-xdr

# Set environment variables
export AWS_REGION=us-east-1
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export CLUSTER_NAME=mini-xdr-cluster
export PROJECT_NAME=mini-xdr-prod

# Verify AWS access
aws sts get-caller-identity
```

### 2. Create EKS Cluster with eksctl (15 minutes)

```bash
# Create cluster config
cat > /tmp/eks-quick-config.yaml <<EOF
apiVersion: eksctl.io/v1alpha5
kind: ClusterConfig

metadata:
  name: ${CLUSTER_NAME}
  region: ${AWS_REGION}
  version: "1.28"

managedNodeGroups:
  - name: ng-1
    instanceType: t3.medium
    desiredCapacity: 2
    minSize: 2
    maxSize: 4
    volumeSize: 30
    privateNetworking: true
    
iam:
  withOIDC: true
  
addons:
  - name: vpc-cni
  - name: coredns
  - name: kube-proxy
  - name: aws-ebs-csi-driver
EOF

# Create cluster (takes ~15 minutes)
eksctl create cluster -f /tmp/eks-quick-config.yaml

# Verify
kubectl get nodes
```

### 3. Install AWS Load Balancer Controller (5 minutes)

```bash
# Download IAM policy
curl -o /tmp/iam_policy.json https://raw.githubusercontent.com/kubernetes-sigs/aws-load-balancer-controller/v2.6.2/docs/install/iam_policy.json

# Create policy
aws iam create-policy \
  --policy-name AWSLoadBalancerControllerIAMPolicy \
  --policy-document file:///tmp/iam_policy.json

# Create service account
eksctl create iamserviceaccount \
  --cluster=${CLUSTER_NAME} \
  --namespace=kube-system \
  --name=aws-load-balancer-controller \
  --role-name AmazonEKSLoadBalancerControllerRole \
  --attach-policy-arn=arn:aws:iam::${AWS_ACCOUNT_ID}:policy/AWSLoadBalancerControllerIAMPolicy \
  --approve

# Install controller
helm repo add eks https://aws.github.io/eks-charts
helm repo update
helm install aws-load-balancer-controller eks/aws-load-balancer-controller \
  -n kube-system \
  --set clusterName=${CLUSTER_NAME} \
  --set serviceAccount.create=false \
  --set serviceAccount.name=aws-load-balancer-controller

# Verify
kubectl get deployment -n kube-system aws-load-balancer-controller
```

### 4. Deploy Mini-XDR Application (5 minutes)

```bash
# Create namespace
kubectl create namespace mini-xdr

# Create quick deployment (using public images for demo)
cat <<EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mini-xdr-frontend
  namespace: mini-xdr
spec:
  replicas: 2
  selector:
    matchLabels:
      app: mini-xdr-frontend
  template:
    metadata:
      labels:
        app: mini-xdr-frontend
    spec:
      containers:
      - name: frontend
        image: node:18-alpine
        command: ["/bin/sh", "-c"]
        args:
          - |
            echo '<!DOCTYPE html>
            <html>
            <head><title>Mini-XDR Demo</title></head>
            <body>
              <h1>Mini-XDR Security Platform</h1>
              <p>Welcome to Mini-XDR on AWS EKS!</p>
              <p>Cluster: ${CLUSTER_NAME}</p>
              <p>Region: ${AWS_REGION}</p>
            </body>
            </html>' > /tmp/index.html
            npx -y http-server /tmp -p 3000
        ports:
        - containerPort: 3000
---
apiVersion: v1
kind: Service
metadata:
  name: mini-xdr-frontend
  namespace: mini-xdr
spec:
  selector:
    app: mini-xdr-frontend
  ports:
  - port: 80
    targetPort: 3000
  type: ClusterIP
---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: mini-xdr-ingress
  namespace: mini-xdr
  annotations:
    alb.ingress.kubernetes.io/scheme: internet-facing
    alb.ingress.kubernetes.io/target-type: ip
    alb.ingress.kubernetes.io/healthcheck-path: /
spec:
  ingressClassName: alb
  rules:
  - http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: mini-xdr-frontend
            port:
              number: 80
EOF

# Wait for ALB to be created (3-5 minutes)
kubectl get ingress -n mini-xdr -w
```

### 5. Access Your Application

```bash
# Get the ALB URL
ALB_URL=$(kubectl get ingress -n mini-xdr mini-xdr-ingress \
  -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')

echo "Your Mini-XDR is accessible at: http://$ALB_URL"

# Test it
curl http://$ALB_URL

# Open in browser
open http://$ALB_URL
```

---

## What You've Deployed

- ✅ EKS Kubernetes cluster (v1.28)
- ✅ 2 EC2 worker nodes (t3.medium)
- ✅ AWS Load Balancer Controller
- ✅ Application Load Balancer (internet-facing)
- ✅ Mini-XDR frontend (demo version)

---

## Next Steps

### Add SSL/TLS

```bash
# Request certificate
aws acm request-certificate \
  --domain-name yourdomain.com \
  --validation-method DNS

# Get certificate ARN
CERT_ARN=$(aws acm list-certificates --query 'CertificateSummaryList[0].CertificateArn' --output text)

# Update ingress with HTTPS
kubectl annotate ingress mini-xdr-ingress -n mini-xdr \
  alb.ingress.kubernetes.io/certificate-arn=$CERT_ARN \
  alb.ingress.kubernetes.io/listen-ports='[{"HTTP": 80}, {"HTTPS": 443}]' \
  alb.ingress.kubernetes.io/ssl-redirect='443'
```

### Deploy Real Application

```bash
# Build and push your images to ECR
aws ecr create-repository --repository-name mini-xdr-frontend
aws ecr create-repository --repository-name mini-xdr-backend

# Get ECR login
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin \
  ${AWS_ACCOUNT_ID}.dkr.ecr.us-east-1.amazonaws.com

# Build and push
cd frontend
docker build -t ${AWS_ACCOUNT_ID}.dkr.ecr.us-east-1.amazonaws.com/mini-xdr-frontend:latest .
docker push ${AWS_ACCOUNT_ID}.dkr.ecr.us-east-1.amazonaws.com/mini-xdr-frontend:latest

cd ../backend
docker build -t ${AWS_ACCOUNT_ID}.dkr.ecr.us-east-1.amazonaws.com/mini-xdr-backend:latest .
docker push ${AWS_ACCOUNT_ID}.dkr.ecr.us-east-1.amazonaws.com/mini-xdr-backend:latest

# Deploy using manifests from the full guide
kubectl apply -f k8s/
```

### Add Database

```bash
# Create RDS PostgreSQL
aws rds create-db-instance \
  --db-instance-identifier mini-xdr-db \
  --db-instance-class db.t3.micro \
  --engine postgres \
  --master-username admin \
  --master-user-password YourSecurePassword123! \
  --allocated-storage 20

# Update backend deployment with DB connection
```

---

## Clean Up (When Done Testing)

```bash
# Delete the cluster (removes everything)
eksctl delete cluster --name ${CLUSTER_NAME} --region ${AWS_REGION}

# This will delete:
# - EKS cluster
# - EC2 instances
# - Load Balancer
# - VPC (if created by eksctl)
# - IAM roles

# Manually delete (if created):
# - RDS instances
# - ECR repositories
# - CloudWatch logs
```

---

## Costs

**Quick setup costs (running 24/7):**
- EKS cluster: $73/month
- 2x t3.medium: $60/month
- ALB: $23/month
- **Total: ~$156/month**

**To reduce costs:**
- Stop cluster when not using: Free
- Use spot instances: ~$90/month
- Use Fargate instead: ~$120/month

---

## Troubleshooting

**"No nodes available"**
```bash
kubectl get nodes
eksctl get nodegroup --cluster ${CLUSTER_NAME}
```

**"Ingress not getting external IP"**
```bash
kubectl logs -n kube-system deployment/aws-load-balancer-controller
kubectl describe ingress -n mini-xdr mini-xdr-ingress
```

**"Can't access ALB URL"**
```bash
# Check ALB target health
aws elbv2 describe-target-health \
  --target-group-arn $(aws elbv2 describe-target-groups \
    --query 'TargetGroups[0].TargetGroupArn' --output text)
```

---

## Get Help

- AWS EKS Docs: https://docs.aws.amazon.com/eks/
- kubectl Cheat Sheet: https://kubernetes.io/docs/reference/kubectl/cheatsheet/
- Refer to full guide: `docs/AWS_DEPLOYMENT_COMPLETE_GUIDE.md`

---

**Success!** 🎉

Your Mini-XDR demo is now running on AWS. For production deployment, follow the complete guide.


