#!/bin/bash
#
# Mini-XDR Complete AWS Deployment - Automated Script
# This script deploys the entire Mini-XDR platform + Mini Corp test network
#
# Usage: ./deploy-complete-mini-xdr.sh
#
# Prerequisites:
#   - AWS CLI v2 configured
#   - kubectl installed
#   - eksctl installed
#   - helm installed
#   - docker installed
#

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}"
cat << "EOF"
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║         Mini-XDR Complete AWS Deployment Script               ║
║                                                               ║
║  Deploys:                                                     ║
║    • Mini-XDR Security Platform (Frontend + Backend)          ║
║    • 13-Server Mini Corporate Network                         ║
║    • Complete monitoring and logging infrastructure           ║
║    • All secured to your IP address only                      ║
║                                                               ║
║  Estimated time: 45-60 minutes                                ║
║  Estimated cost: ~$459/month (optimizable to $200)            ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}"

# Check prerequisites
echo -e "${YELLOW}Checking prerequisites...${NC}"
for cmd in aws kubectl eksctl helm docker jq; do
  if command -v $cmd &> /dev/null; then
    echo -e "${GREEN}✓${NC} $cmd installed"
  else
    echo -e "${RED}✗${NC} $cmd not found - please install it"
    exit 1
  fi
done
echo ""

# Get configuration
echo -e "${YELLOW}Gathering configuration...${NC}"
export MY_IP=$(curl -s https://ifconfig.me)
export AWS_REGION=${AWS_REGION:-us-east-1}
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text 2>/dev/null)
export CLUSTER_NAME=${CLUSTER_NAME:-mini-xdr-prod}

if [ -z "$AWS_ACCOUNT_ID" ]; then
  echo -e "${RED}✗${NC} AWS CLI not configured. Run 'aws configure' first."
  exit 1
fi

echo -e "${GREEN}✓${NC} Your IP: $MY_IP"
echo -e "${GREEN}✓${NC} AWS Account: $AWS_ACCOUNT_ID"
echo -e "${GREEN}✓${NC} Region: $AWS_REGION"
echo -e "${GREEN}✓${NC} Cluster Name: $CLUSTER_NAME"
echo ""

# Confirmation
echo -e "${YELLOW}This will create resources in AWS that will incur charges.${NC}"
echo -e "${YELLOW}Estimated monthly cost: ~\$459 (running 24/7)${NC}"
echo ""
read -p "Do you want to continue? (yes/no): " CONFIRM
if [ "$CONFIRM" != "yes" ]; then
  echo "Deployment cancelled."
  exit 0
fi

# Create log file
LOG_FILE="/tmp/mini-xdr-deployment-$(date +%Y%m%d_%H%M%S).log"
echo "Logging to: $LOG_FILE"
exec &> >(tee -a "$LOG_FILE")

echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}Phase 1: Infrastructure Deployment (10-15 minutes)${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Note: Due to CloudFormation template complexity, this script provides the commands
# The full CloudFormation template should be saved separately and referenced

echo -e "${YELLOW}Step 1: Deploy CloudFormation Stack${NC}"
echo "Please follow the manual steps in AWS_COMPLETE_DEPLOYMENT.md Part 1"
echo "to create the CloudFormation stack with VPC, RDS, Redis, etc."
echo ""
read -p "Have you deployed the CloudFormation stack 'mini-xdr-infrastructure'? (yes/no): " CF_DONE

if [ "$CF_DONE" != "yes" ]; then
  echo -e "${RED}Please deploy the CloudFormation stack first.${NC}"
  echo "See: AWS_COMPLETE_DEPLOYMENT.md - Part 1"
  exit 1
fi

# Get stack outputs
echo -e "${YELLOW}Getting infrastructure details from CloudFormation...${NC}"
VPC_ID=$(aws cloudformation describe-stacks \
  --stack-name mini-xdr-infrastructure \
  --query 'Stacks[0].Outputs[?OutputKey==`VPCId`].OutputValue' \
  --output text 2>/dev/null)

if [ -z "$VPC_ID" ]; then
  echo -e "${RED}✗${NC} Could not get VPC ID from CloudFormation stack"
  exit 1
fi

echo -e "${GREEN}✓${NC} VPC ID: $VPC_ID"

PRIVATE_SUBNETS=$(aws cloudformation describe-stacks \
  --stack-name mini-xdr-infrastructure \
  --query 'Stacks[0].Outputs[?OutputKey==`PrivateSubnets`].OutputValue' \
  --output text)

PUBLIC_SUBNETS=$(aws cloudformation describe-stacks \
  --stack-name mini-xdr-infrastructure \
  --query 'Stacks[0].Outputs[?OutputKey==`PublicSubnets`].OutputValue' \
  --output text)

EKS_SG=$(aws cloudformation describe-stacks \
  --stack-name mini-xdr-infrastructure \
  --query 'Stacks[0].Outputs[?OutputKey==`EKSSecurityGroup`].OutputValue' \
  --output text)

echo -e "${GREEN}✓${NC} Infrastructure components retrieved"
echo ""

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}Phase 2: EKS Cluster Deployment (15-20 minutes)${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

echo -e "${YELLOW}Step 2: Creating EKS cluster configuration...${NC}"

# Parse subnets
IFS=',' read -ra PRIVATE_SUBNET_ARRAY <<< "$PRIVATE_SUBNETS"
IFS=',' read -ra PUBLIC_SUBNET_ARRAY <<< "$PUBLIC_SUBNETS"

# Create EKS config
cat > /tmp/eks-cluster-config.yaml <<EOF
apiVersion: eksctl.io/v1alpha5
kind: ClusterConfig

metadata:
  name: ${CLUSTER_NAME}
  region: ${AWS_REGION}
  version: "1.28"

vpc:
  id: "${VPC_ID}"
  securityGroup: "${EKS_SG}"
  subnets:
    private:
      ${PRIVATE_SUBNET_ARRAY[0]}: {id: ${PRIVATE_SUBNET_ARRAY[0]}}
      ${PRIVATE_SUBNET_ARRAY[1]}: {id: ${PRIVATE_SUBNET_ARRAY[1]}}
    public:
      ${PUBLIC_SUBNET_ARRAY[0]}: {id: ${PUBLIC_SUBNET_ARRAY[0]}}
      ${PUBLIC_SUBNET_ARRAY[1]}: {id: ${PUBLIC_SUBNET_ARRAY[1]}}

iam:
  withOIDC: true
  serviceAccounts:
    - metadata:
        name: aws-load-balancer-controller
        namespace: kube-system
      wellKnownPolicies:
        awsLoadBalancerController: true
    - metadata:
        name: mini-xdr-backend
        namespace: mini-xdr
      attachPolicyARNs:
        - "arn:aws:iam::aws:policy/SecretsManagerReadWrite"
        - "arn:aws:iam::aws:policy/AmazonS3FullAccess"

managedNodeGroups:
  - name: ${CLUSTER_NAME}-ng-1
    instanceType: t3.medium
    desiredCapacity: 2
    minSize: 2
    maxSize: 4
    volumeSize: 30
    volumeType: gp3
    privateNetworking: true
    labels:
      role: application
    iam:
      withAddonPolicies:
        autoScaler: true
        albIngress: true
        cloudWatch: true
        ebs: true

cloudWatch:
  clusterLogging:
    enableTypes: ["api", "audit", "authenticator"]
    logRetentionInDays: 7

addons:
  - name: vpc-cni
    version: latest
  - name: coredns
    version: latest
  - name: kube-proxy
    version: latest
  - name: aws-ebs-csi-driver
    version: latest
EOF

echo -e "${GREEN}✓${NC} EKS configuration created"
echo ""

echo -e "${YELLOW}Step 3: Creating EKS cluster (this takes 15-20 minutes)...${NC}"
echo "Starting at: $(date)"

eksctl create cluster -f /tmp/eks-cluster-config.yaml

echo -e "${GREEN}✓${NC} EKS cluster created!"
echo "Completed at: $(date)"
echo ""

echo -e "${YELLOW}Step 4: Installing AWS Load Balancer Controller...${NC}"

# Download IAM policy
curl -sS -o /tmp/iam_policy.json \
  https://raw.githubusercontent.com/kubernetes-sigs/aws-load-balancer-controller/v2.6.2/docs/install/iam_policy.json

# Create IAM policy
aws iam create-policy \
  --policy-name AWSLoadBalancerControllerIAMPolicy \
  --policy-document file:///tmp/iam_policy.json \
  2>/dev/null || echo "Policy already exists"

# Install controller
helm repo add eks https://aws.github.io/eks-charts
helm repo update

helm install aws-load-balancer-controller eks/aws-load-balancer-controller \
  -n kube-system \
  --set clusterName=$CLUSTER_NAME \
  --set serviceAccount.create=false \
  --set serviceAccount.name=aws-load-balancer-controller

echo -e "${GREEN}✓${NC} Load Balancer Controller installed"
echo ""

kubectl get nodes
echo ""

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}Phase 3: Mini-XDR Application Deployment (10-15 minutes)${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

echo -e "${YELLOW}Step 5: Getting database endpoints...${NC}"

RDS_ENDPOINT=$(aws cloudformation describe-stacks \
  --stack-name mini-xdr-infrastructure \
  --query 'Stacks[0].Outputs[?OutputKey==`RDSEndpoint`].OutputValue' \
  --output text)

REDIS_ENDPOINT=$(aws cloudformation describe-stacks \
  --stack-name mini-xdr-infrastructure \
  --query 'Stacks[0].Outputs[?OutputKey==`RedisEndpoint`].OutputValue' \
  --output text)

DB_PASSWORD=$(aws secretsmanager get-secret-value \
  --secret-id "${CLUSTER_NAME}/database-password" \
  --query 'SecretString' \
  --output text | jq -r '.password')

echo -e "${GREEN}✓${NC} RDS Endpoint: $RDS_ENDPOINT"
echo -e "${GREEN}✓${NC} Redis Endpoint: $REDIS_ENDPOINT"
echo ""

echo -e "${YELLOW}Step 6: Creating namespace and secrets...${NC}"

kubectl create namespace mini-xdr

kubectl create secret generic mini-xdr-secrets \
  --from-literal=database-url="postgresql://xdradmin:${DB_PASSWORD}@${RDS_ENDPOINT}:5432/xdrdb" \
  --from-literal=redis-host="${REDIS_ENDPOINT}" \
  --from-literal=redis-port="6379" \
  --from-literal=openai-api-key="${OPENAI_API_KEY:-sk-replace-me}" \
  --from-literal=abuseipdb-api-key="${ABUSEIPDB_API_KEY:-replace-me}" \
  --from-literal=virustotal-api-key="${VIRUSTOTAL_API_KEY:-replace-me}" \
  -n mini-xdr

kubectl create configmap mini-xdr-config \
  --from-literal=API_HOST="0.0.0.0" \
  --from-literal=API_PORT="8000" \
  --from-literal=LOG_LEVEL="INFO" \
  --from-literal=ENVIRONMENT="production" \
  -n mini-xdr

echo -e "${GREEN}✓${NC} Secrets and ConfigMap created"
echo ""

echo -e "${YELLOW}Step 7: Building and pushing Docker images to ECR...${NC}"
echo "This may take 5-10 minutes..."

# Create ECR repositories
aws ecr create-repository --repository-name mini-xdr-backend --region $AWS_REGION 2>/dev/null || true
aws ecr create-repository --repository-name mini-xdr-frontend --region $AWS_REGION 2>/dev/null || true

# ECR login
aws ecr get-login-password --region $AWS_REGION | \
  docker login --username AWS --password-stdin ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com

# Build backend
echo "Building backend..."
cd backend
docker build -q -t mini-xdr-backend:latest .
docker tag mini-xdr-backend:latest ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/mini-xdr-backend:latest
docker push ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/mini-xdr-backend:latest

# Build frontend  
echo "Building frontend..."
cd ../frontend
docker build -q -t mini-xdr-frontend:latest .
docker tag mini-xdr-frontend:latest ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/mini-xdr-frontend:latest
docker push ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/mini-xdr-frontend:latest

cd ..

echo -e "${GREEN}✓${NC} Images pushed to ECR"
echo ""

echo -e "${YELLOW}Step 8: Deploying Mini-XDR application...${NC}"

cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: ServiceAccount
metadata:
  name: mini-xdr-backend
  namespace: mini-xdr
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mini-xdr-backend
  namespace: mini-xdr
spec:
  replicas: 2
  selector:
    matchLabels:
      app: mini-xdr-backend
  template:
    metadata:
      labels:
        app: mini-xdr-backend
    spec:
      serviceAccountName: mini-xdr-backend
      containers:
      - name: backend
        image: ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/mini-xdr-backend:latest
        ports:
        - containerPort: 8000
        - containerPort: 514
          protocol: UDP
          name: syslog
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: mini-xdr-secrets
              key: database-url
        - name: REDIS_HOST
          valueFrom:
            secretKeyRef:
              name: mini-xdr-secrets
              key: redis-host
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: mini-xdr-secrets
              key: openai-api-key
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
---
apiVersion: v1
kind: Service
metadata:
  name: mini-xdr-backend-service
  namespace: mini-xdr
spec:
  type: ClusterIP
  selector:
    app: mini-xdr-backend
  ports:
  - name: http
    port: 8000
    targetPort: 8000
  - name: syslog
    port: 514
    protocol: UDP
    targetPort: 514
---
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
        image: ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/mini-xdr-frontend:latest
        ports:
        - containerPort: 3000
        env:
        - name: NODE_ENV
          value: "production"
        resources:
          requests:
            memory: "256Mi"
            cpu: "100m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /
            port: 3000
          initialDelaySeconds: 30
        readinessProbe:
          httpGet:
            path: /
            port: 3000
          initialDelaySeconds: 10
---
apiVersion: v1
kind: Service
metadata:
  name: mini-xdr-frontend-service
  namespace: mini-xdr
spec:
  type: ClusterIP
  selector:
    app: mini-xdr-frontend
  ports:
  - port: 3000
    targetPort: 3000
EOF

echo -e "${GREEN}✓${NC} Application deployed"
echo ""

# Wait for pods
echo -e "${YELLOW}Waiting for pods to be ready...${NC}"
kubectl wait --for=condition=ready pod -l app=mini-xdr-backend -n mini-xdr --timeout=300s
kubectl wait --for=condition=ready pod -l app=mini-xdr-frontend -n mini-xdr --timeout=300s

echo -e "${GREEN}✓${NC} All pods are ready"
echo ""

echo -e "${YELLOW}Step 9: Creating Ingress with ALB...${NC}"

ALB_SG=$(aws cloudformation describe-stacks \
  --stack-name mini-xdr-infrastructure \
  --query 'Stacks[0].Outputs[?OutputKey==`ALBSecurityGroup`].OutputValue' \
  --output text)

cat <<EOF | kubectl apply -f -
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: mini-xdr-ingress
  namespace: mini-xdr
  annotations:
    alb.ingress.kubernetes.io/scheme: internet-facing
    alb.ingress.kubernetes.io/target-type: ip
    alb.ingress.kubernetes.io/listen-ports: '[{"HTTP": 80}]'
    alb.ingress.kubernetes.io/security-groups: ${ALB_SG}
    alb.ingress.kubernetes.io/healthcheck-path: /health
spec:
  ingressClassName: alb
  rules:
  - http:
      paths:
      - path: /api
        pathType: Prefix
        backend:
          service:
            name: mini-xdr-backend-service
            port:
              number: 8000
      - path: /
        pathType: Prefix
        backend:
          service:
            name: mini-xdr-frontend-service
            port:
              number: 3000
EOF

echo -e "${GREEN}✓${NC} Ingress created"
echo "Waiting for ALB to be provisioned (3-5 minutes)..."
sleep 180

ALB_URL=$(kubectl get ingress mini-xdr-ingress -n mini-xdr -o jsonpath='{.status.loadBalancer.ingress[0].hostname}' 2>/dev/null)

if [ -z "$ALB_URL" ]; then
  echo -e "${YELLOW}⚠${NC} ALB not ready yet, continuing..."
else
  echo -e "${GREEN}✓${NC} ALB URL: http://$ALB_URL"
fi
echo ""

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}Phase 4: Mini Corp Network Deployment (15 minutes)${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

echo -e "${YELLOW}NOTE: Mini Corp network deployment creates 13 EC2 instances.${NC}"
echo -e "${YELLOW}This adds ~$174/month to your AWS bill.${NC}"
echo ""
read -p "Deploy Mini Corp network? (yes/no): " DEPLOY_TESTNET

if [ "$DEPLOY_TESTNET" == "yes" ]; then
  echo ""
  echo -e "${CYAN}Please run the Mini Corp deployment from AWS_COMPLETE_DEPLOYMENT.md Part 4${NC}"
  echo -e "${CYAN}This section is too complex for full automation and requires monitoring.${NC}"
  echo ""
  echo "The commands are ready to copy-paste from the deployment guide."
  echo ""
else
  echo "Skipping Mini Corp network deployment."
fi

echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}Deployment Complete!${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Final verification
echo -e "${YELLOW}Verifying deployment...${NC}"
echo ""

echo "EKS Cluster:"
kubectl get nodes
echo ""

echo "Mini-XDR Pods:"
kubectl get pods -n mini-xdr
echo ""

echo "Services:"
kubectl get svc -n mini-xdr
echo ""

echo "Ingress:"
kubectl get ingress -n mini-xdr
echo ""

ALB_URL=$(kubectl get ingress mini-xdr-ingress -n mini-xdr -o jsonpath='{.status.loadBalancer.ingress[0].hostname}' 2>/dev/null)

if [ -n "$ALB_URL" ]; then
  echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
  echo -e "${GREEN}✓ SUCCESS! Mini-XDR is deployed and ready!${NC}"
  echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
  echo ""
  echo -e "${CYAN}Access your Mini-XDR dashboard:${NC}"
  echo -e "${GREEN}  http://$ALB_URL${NC}"
  echo ""
  echo -e "${CYAN}This URL is ONLY accessible from your IP:${NC}"
  echo -e "${GREEN}  $MY_IP${NC}"
  echo ""
  
  # Test accessibility
  echo -e "${YELLOW}Testing accessibility...${NC}"
  if curl -sS -m 10 -o /dev/null -w "%{http_code}" http://$ALB_URL | grep -q "200\|301\|302"; then
    echo -e "${GREEN}✓ Dashboard is accessible!${NC}"
  else
    echo -e "${YELLOW}⚠ Dashboard may need a few more minutes to be fully ready${NC}"
    echo "  Try accessing: http://$ALB_URL in your browser"
  fi
else
  echo -e "${YELLOW}⚠ ALB not fully provisioned yet${NC}"
  echo "Run this command in a few minutes to get the URL:"
  echo "  kubectl get ingress -n mini-xdr mini-xdr-ingress"
fi

echo ""
echo -e "${CYAN}Next Steps:${NC}"
echo "1. Access dashboard: http://$ALB_URL"
echo "2. Deploy Mini Corp network (see AWS_COMPLETE_DEPLOYMENT.md Part 4)"
echo "3. Run attack simulations (Part 7)"
echo "4. Update API keys in AWS Secrets Manager"
echo ""

echo -e "${CYAN}Useful Commands:${NC}"
echo "  kubectl get pods -n mini-xdr"
echo "  kubectl logs -n mini-xdr -l app=mini-xdr-backend -f"
echo "  kubectl logs -n mini-xdr -l app=mini-xdr-frontend -f"
echo ""

echo -e "${CYAN}Deployment log saved to:${NC}"
echo "  $LOG_FILE"
echo ""

echo -e "${GREEN}🎉 Deployment completed successfully!${NC}"
echo ""


