# Mini-XDR Complete AWS Deployment Guide

**A Production-Ready Deployment of Mini-XDR on AWS**

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Prerequisites](#prerequisites)
3. [Phase 1: AWS Foundation Setup](#phase-1-aws-foundation-setup)
4. [Phase 2: EKS Cluster Deployment](#phase-2-eks-cluster-deployment)
5. [Phase 3: Application Deployment](#phase-3-application-deployment)
6. [Phase 4: Networking & Ingress](#phase-4-networking--ingress)
7. [Phase 5: Security Hardening](#phase-5-security-hardening)
8. [Phase 6: Monitoring & Observability](#phase-6-monitoring--observability)
9. [Phase 7: CI/CD Pipeline](#phase-7-cicd-pipeline)
10. [Cost Optimization](#cost-optimization)
11. [Troubleshooting](#troubleshooting)

---

## Architecture Overview

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Internet Users                           │
│                    (Recruiters, You, etc.)                       │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            │ HTTPS (443)
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                  AWS Route 53 (DNS)                              │
│              mini-xdr.yourdomain.com                             │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│              AWS Application Load Balancer (ALB)                 │
│                  - SSL/TLS Termination                           │
│                  - Health Checks                                 │
│                  - WAF Integration                               │
└───────────────────────────┬─────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│   EKS Node   │   │   EKS Node   │   │   EKS Node   │
│ us-east-1a   │   │ us-east-1b   │   │ us-east-1c   │
└──────┬───────┘   └──────┬───────┘   └──────┬───────┘
       │                  │                  │
       │   Kubernetes Cluster (EKS)         │
       │                  │                  │
       ▼                  ▼                  ▼
┌────────────────────────────────────────────────────┐
│              Kubernetes Services                    │
│  ┌──────────────┐  ┌──────────────┐               │
│  │   Frontend   │  │   Backend    │               │
│  │   (Next.js)  │  │  (FastAPI)   │               │
│  │   Pods x3    │  │   Pods x2    │               │
│  └──────────────┘  └──────────────┘               │
└────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│     RDS      │   │  ElastiCache │   │     S3       │
│  PostgreSQL  │   │    Redis     │   │   Storage    │
│  (Multi-AZ)  │   │              │   │   Logs/Data  │
└──────────────┘   └──────────────┘   └──────────────┘
        │                   │                   │
        └───────────────────┴───────────────────┘
                            │
                            ▼
              ┌─────────────────────────┐
              │  AWS Secrets Manager    │
              │  (API Keys, Creds)      │
              └─────────────────────────┘
                            │
                            ▼
              ┌─────────────────────────┐
              │   CloudWatch Logs       │
              │   + Metrics             │
              └─────────────────────────┘
```

### Component Breakdown

**Networking:**
- VPC with public and private subnets across 3 AZs
- NAT Gateways for private subnet internet access
- Internet Gateway for public subnets
- Security Groups for fine-grained access control
- Network ACLs for subnet-level security

**Compute:**
- EKS managed Kubernetes cluster (v1.28+)
- EC2 instances (t3.medium or t3a.medium for cost)
- Auto-scaling node groups (2-6 nodes)
- Spot instances for non-critical workloads

**Storage:**
- RDS PostgreSQL (production-grade SQL database)
- ElastiCache Redis (session management, caching)
- EBS volumes for persistent storage
- S3 for logs, backups, static assets

**Networking/Ingress:**
- AWS Load Balancer Controller
- Application Load Balancer (ALB)
- AWS Certificate Manager (ACM) for SSL/TLS
- Route 53 for DNS management

**Security:**
- AWS WAF for web application firewall
- Security Groups (network firewall)
- IAM roles with least privilege
- AWS Secrets Manager for secrets
- KMS for encryption

**Monitoring:**
- CloudWatch for logs and metrics
- CloudWatch Container Insights
- X-Ray for distributed tracing
- CloudWatch Alarms for alerts

---

## Prerequisites

### Required Tools

```bash
# Install AWS CLI v2
curl "https://awscli.amazonaws.com/AWSCLIV2.pkg" -o "AWSCLIV2.pkg"
sudo installer -pkg AWSCLIV2.pkg -target /

# Install kubectl
brew install kubectl

# Install eksctl
brew tap weaveworks/tap
brew install weaveworks/tap/eksctl

# Install Helm
brew install helm

# Install terraform (optional, for infrastructure as code)
brew install terraform

# Verify installations
aws --version       # Should be 2.x
kubectl version --client
eksctl version
helm version
```

### AWS Account Setup

1. **AWS Account:** Active AWS account with billing enabled
2. **IAM User:** Create an IAM user with appropriate permissions:
   - AmazonEC2FullAccess
   - AmazonEKSClusterPolicy
   - AmazonEKSServicePolicy
   - AmazonVPCFullAccess
   - IAMFullAccess (for service accounts)
   - AmazonRDSFullAccess
   - AmazonS3FullAccess
   - SecretsManagerReadWrite

3. **Configure AWS CLI:**
```bash
aws configure
# AWS Access Key ID: YOUR_ACCESS_KEY
# AWS Secret Access Key: YOUR_SECRET_KEY
# Default region: us-east-1
# Default output format: json
```

4. **Budget Alert:** Set up billing alerts (recommended $50-$100/month for testing)

### Domain Name (Optional but Recommended)

- Purchase a domain from Route 53 or use existing domain
- Example: `mini-xdr.yourdomain.com`
- Alternative: Use AWS-provided LoadBalancer DNS name

---

## Phase 1: AWS Foundation Setup

### Step 1.1: Create VPC with CloudFormation

Create `infrastructure/aws/vpc-stack.yaml`:

```yaml
AWSTemplateFormatVersion: '2010-09-09'
Description: 'VPC for Mini-XDR with public and private subnets across 3 AZs'

Parameters:
  EnvironmentName:
    Type: String
    Default: mini-xdr-prod
    Description: Environment name prefix

  VpcCIDR:
    Type: String
    Default: 10.0.0.0/16
    Description: CIDR block for VPC

Resources:
  VPC:
    Type: AWS::EC2::VPC
    Properties:
      CidrBlock: !Ref VpcCIDR
      EnableDnsHostnames: true
      EnableDnsSupport: true
      Tags:
        - Key: Name
          Value: !Sub '${EnvironmentName}-vpc'
        - Key: Environment
          Value: !Ref EnvironmentName

  InternetGateway:
    Type: AWS::EC2::InternetGateway
    Properties:
      Tags:
        - Key: Name
          Value: !Sub '${EnvironmentName}-igw'

  InternetGatewayAttachment:
    Type: AWS::EC2::VPCGatewayAttachment
    Properties:
      InternetGatewayId: !Ref InternetGateway
      VpcId: !Ref VPC

  # Public Subnets
  PublicSubnet1:
    Type: AWS::EC2::Subnet
    Properties:
      VpcId: !Ref VPC
      AvailabilityZone: !Select [0, !GetAZs '']
      CidrBlock: 10.0.1.0/24
      MapPublicIpOnLaunch: true
      Tags:
        - Key: Name
          Value: !Sub '${EnvironmentName}-public-subnet-1a'
        - Key: kubernetes.io/role/elb
          Value: 1

  PublicSubnet2:
    Type: AWS::EC2::Subnet
    Properties:
      VpcId: !Ref VPC
      AvailabilityZone: !Select [1, !GetAZs '']
      CidrBlock: 10.0.2.0/24
      MapPublicIpOnLaunch: true
      Tags:
        - Key: Name
          Value: !Sub '${EnvironmentName}-public-subnet-1b'
        - Key: kubernetes.io/role/elb
          Value: 1

  PublicSubnet3:
    Type: AWS::EC2::Subnet
    Properties:
      VpcId: !Ref VPC
      AvailabilityZone: !Select [2, !GetAZs '']
      CidrBlock: 10.0.3.0/24
      MapPublicIpOnLaunch: true
      Tags:
        - Key: Name
          Value: !Sub '${EnvironmentName}-public-subnet-1c'
        - Key: kubernetes.io/role/elb
          Value: 1

  # Private Subnets
  PrivateSubnet1:
    Type: AWS::EC2::Subnet
    Properties:
      VpcId: !Ref VPC
      AvailabilityZone: !Select [0, !GetAZs '']
      CidrBlock: 10.0.11.0/24
      Tags:
        - Key: Name
          Value: !Sub '${EnvironmentName}-private-subnet-1a'
        - Key: kubernetes.io/role/internal-elb
          Value: 1

  PrivateSubnet2:
    Type: AWS::EC2::Subnet
    Properties:
      VpcId: !Ref VPC
      AvailabilityZone: !Select [1, !GetAZs '']
      CidrBlock: 10.0.12.0/24
      Tags:
        - Key: Name
          Value: !Sub '${EnvironmentName}-private-subnet-1b'
        - Key: kubernetes.io/role/internal-elb
          Value: 1

  PrivateSubnet3:
    Type: AWS::EC2::Subnet
    Properties:
      VpcId: !Ref VPC
      AvailabilityZone: !Select [2, !GetAZs '']
      CidrBlock: 10.0.13.0/24
      Tags:
        - Key: Name
          Value: !Sub '${EnvironmentName}-private-subnet-1c'
        - Key: kubernetes.io/role/internal-elb
          Value: 1

  # NAT Gateways
  NatGateway1EIP:
    Type: AWS::EC2::EIP
    DependsOn: InternetGatewayAttachment
    Properties:
      Domain: vpc
      Tags:
        - Key: Name
          Value: !Sub '${EnvironmentName}-nat-eip-1a'

  NatGateway1:
    Type: AWS::EC2::NatGateway
    Properties:
      AllocationId: !GetAtt NatGateway1EIP.AllocationId
      SubnetId: !Ref PublicSubnet1
      Tags:
        - Key: Name
          Value: !Sub '${EnvironmentName}-nat-1a'

  # Public Route Table
  PublicRouteTable:
    Type: AWS::EC2::RouteTable
    Properties:
      VpcId: !Ref VPC
      Tags:
        - Key: Name
          Value: !Sub '${EnvironmentName}-public-routes'

  DefaultPublicRoute:
    Type: AWS::EC2::Route
    DependsOn: InternetGatewayAttachment
    Properties:
      RouteTableId: !Ref PublicRouteTable
      DestinationCidrBlock: 0.0.0.0/0
      GatewayId: !Ref InternetGateway

  PublicSubnet1RouteTableAssociation:
    Type: AWS::EC2::SubnetRouteTableAssociation
    Properties:
      RouteTableId: !Ref PublicRouteTable
      SubnetId: !Ref PublicSubnet1

  PublicSubnet2RouteTableAssociation:
    Type: AWS::EC2::SubnetRouteTableAssociation
    Properties:
      RouteTableId: !Ref PublicRouteTable
      SubnetId: !Ref PublicSubnet2

  PublicSubnet3RouteTableAssociation:
    Type: AWS::EC2::SubnetRouteTableAssociation
    Properties:
      RouteTableId: !Ref PublicRouteTable
      SubnetId: !Ref PublicSubnet3

  # Private Route Table
  PrivateRouteTable1:
    Type: AWS::EC2::RouteTable
    Properties:
      VpcId: !Ref VPC
      Tags:
        - Key: Name
          Value: !Sub '${EnvironmentName}-private-routes-1a'

  DefaultPrivateRoute1:
    Type: AWS::EC2::Route
    Properties:
      RouteTableId: !Ref PrivateRouteTable1
      DestinationCidrBlock: 0.0.0.0/0
      NatGatewayId: !Ref NatGateway1

  PrivateSubnet1RouteTableAssociation:
    Type: AWS::EC2::SubnetRouteTableAssociation
    Properties:
      RouteTableId: !Ref PrivateRouteTable1
      SubnetId: !Ref PrivateSubnet1

  PrivateSubnet2RouteTableAssociation:
    Type: AWS::EC2::SubnetRouteTableAssociation
    Properties:
      RouteTableId: !Ref PrivateRouteTable1
      SubnetId: !Ref PrivateSubnet2

  PrivateSubnet3RouteTableAssociation:
    Type: AWS::EC2::SubnetRouteTableAssociation
    Properties:
      RouteTableId: !Ref PrivateRouteTable1
      SubnetId: !Ref PrivateSubnet3

Outputs:
  VPC:
    Description: VPC ID
    Value: !Ref VPC
    Export:
      Name: !Sub '${EnvironmentName}-VPC'

  PublicSubnets:
    Description: List of public subnets
    Value: !Join [',', [!Ref PublicSubnet1, !Ref PublicSubnet2, !Ref PublicSubnet3]]
    Export:
      Name: !Sub '${EnvironmentName}-PublicSubnets'

  PrivateSubnets:
    Description: List of private subnets
    Value: !Join [',', [!Ref PrivateSubnet1, !Ref PrivateSubnet2, !Ref PrivateSubnet3]]
    Export:
      Name: !Sub '${EnvironmentName}-PrivateSubnets'
```

**Deploy the VPC:**

```bash
aws cloudformation create-stack \
  --stack-name mini-xdr-vpc \
  --template-body file://infrastructure/aws/vpc-stack.yaml \
  --region us-east-1

# Wait for completion
aws cloudformation wait stack-create-complete \
  --stack-name mini-xdr-vpc \
  --region us-east-1

# Get outputs
aws cloudformation describe-stacks \
  --stack-name mini-xdr-vpc \
  --query 'Stacks[0].Outputs' \
  --region us-east-1
```

### Step 1.2: Create Security Groups

```bash
# Get VPC ID
VPC_ID=$(aws cloudformation describe-stacks \
  --stack-name mini-xdr-vpc \
  --query 'Stacks[0].Outputs[?OutputKey==`VPC`].OutputValue' \
  --output text)

# EKS Cluster Security Group
aws ec2 create-security-group \
  --group-name mini-xdr-eks-cluster-sg \
  --description "Security group for Mini-XDR EKS cluster" \
  --vpc-id $VPC_ID

EKS_SG_ID=$(aws ec2 describe-security-groups \
  --filters "Name=group-name,Values=mini-xdr-eks-cluster-sg" \
  --query 'SecurityGroups[0].GroupId' \
  --output text)

# Allow all traffic within the security group
aws ec2 authorize-security-group-ingress \
  --group-id $EKS_SG_ID \
  --protocol all \
  --source-group $EKS_SG_ID

# RDS Security Group
aws ec2 create-security-group \
  --group-name mini-xdr-rds-sg \
  --description "Security group for Mini-XDR RDS" \
  --vpc-id $VPC_ID

RDS_SG_ID=$(aws ec2 describe-security-groups \
  --filters "Name=group-name,Values=mini-xdr-rds-sg" \
  --query 'SecurityGroups[0].GroupId' \
  --output text)

# Allow PostgreSQL from EKS nodes
aws ec2 authorize-security-group-ingress \
  --group-id $RDS_SG_ID \
  --protocol tcp \
  --port 5432 \
  --source-group $EKS_SG_ID

# Redis Security Group
aws ec2 create-security-group \
  --group-name mini-xdr-redis-sg \
  --description "Security group for Mini-XDR ElastiCache" \
  --vpc-id $VPC_ID

REDIS_SG_ID=$(aws ec2 describe-security-groups \
  --filters "Name=group-name,Values=mini-xdr-redis-sg" \
  --query 'SecurityGroups[0].GroupId' \
  --output text)

# Allow Redis from EKS nodes
aws ec2 authorize-security-group-ingress \
  --group-id $REDIS_SG_ID \
  --protocol tcp \
  --port 6379 \
  --source-group $EKS_SG_ID

echo "Security Groups Created:"
echo "EKS SG: $EKS_SG_ID"
echo "RDS SG: $RDS_SG_ID"
echo "Redis SG: $REDIS_SG_ID"
```

### Step 1.3: Create RDS PostgreSQL Database

```bash
# Get private subnet IDs
PRIVATE_SUBNETS=$(aws cloudformation describe-stacks \
  --stack-name mini-xdr-vpc \
  --query 'Stacks[0].Outputs[?OutputKey==`PrivateSubnets`].OutputValue' \
  --output text)

# Create DB subnet group
aws rds create-db-subnet-group \
  --db-subnet-group-name mini-xdr-db-subnet-group \
  --db-subnet-group-description "Subnet group for Mini-XDR RDS" \
  --subnet-ids $(echo $PRIVATE_SUBNETS | tr ',' ' ')

# Create RDS PostgreSQL instance
aws rds create-db-instance \
  --db-instance-identifier mini-xdr-postgres \
  --db-instance-class db.t3.micro \
  --engine postgres \
  --engine-version 15.4 \
  --master-username xdradmin \
  --master-user-password 'ChangeThisP@ssw0rd!' \
  --allocated-storage 20 \
  --storage-type gp3 \
  --vpc-security-group-ids $RDS_SG_ID \
  --db-subnet-group-name mini-xdr-db-subnet-group \
  --backup-retention-period 7 \
  --preferred-backup-window "03:00-04:00" \
  --preferred-maintenance-window "mon:04:00-mon:05:00" \
  --multi-az \
  --storage-encrypted \
  --publicly-accessible false \
  --db-name xdrdb \
  --tags Key=Name,Value=mini-xdr-postgres Key=Environment,Value=production

# Wait for RDS to be available (takes 5-10 minutes)
aws rds wait db-instance-available \
  --db-instance-identifier mini-xdr-postgres

# Get RDS endpoint
RDS_ENDPOINT=$(aws rds describe-db-instances \
  --db-instance-identifier mini-xdr-postgres \
  --query 'DBInstances[0].Endpoint.Address' \
  --output text)

echo "RDS Endpoint: $RDS_ENDPOINT"
```

### Step 1.4: Create ElastiCache Redis

```bash
# Create ElastiCache subnet group
aws elasticache create-cache-subnet-group \
  --cache-subnet-group-name mini-xdr-redis-subnet-group \
  --cache-subnet-group-description "Subnet group for Mini-XDR Redis" \
  --subnet-ids $(echo $PRIVATE_SUBNETS | tr ',' ' ')

# Create Redis cluster
aws elasticache create-cache-cluster \
  --cache-cluster-id mini-xdr-redis \
  --cache-node-type cache.t3.micro \
  --engine redis \
  --engine-version 7.0 \
  --num-cache-nodes 1 \
  --cache-parameter-group default.redis7 \
  --cache-subnet-group-name mini-xdr-redis-subnet-group \
  --security-group-ids $REDIS_SG_ID \
  --tags Key=Name,Value=mini-xdr-redis Key=Environment,Value=production

# Wait for Redis to be available (takes 3-5 minutes)
aws elasticache wait cache-cluster-available \
  --cache-cluster-id mini-xdr-redis

# Get Redis endpoint
REDIS_ENDPOINT=$(aws elasticache describe-cache-clusters \
  --cache-cluster-id mini-xdr-redis \
  --show-cache-node-info \
  --query 'CacheClusters[0].CacheNodes[0].Endpoint.Address' \
  --output text)

echo "Redis Endpoint: $REDIS_ENDPOINT"
```

### Step 1.5: Store Secrets in AWS Secrets Manager

```bash
# Create database credentials secret
aws secretsmanager create-secret \
  --name mini-xdr/database \
  --description "Mini-XDR Database Credentials" \
  --secret-string "{
    \"username\": \"xdradmin\",
    \"password\": \"ChangeThisP@ssw0rd!\",
    \"engine\": \"postgres\",
    \"host\": \"$RDS_ENDPOINT\",
    \"port\": 5432,
    \"dbname\": \"xdrdb\"
  }"

# Create API keys secret (update with your actual keys)
aws secretsmanager create-secret \
  --name mini-xdr/api-keys \
  --description "Mini-XDR API Keys" \
  --secret-string "{
    \"OPENAI_API_KEY\": \"sk-your-openai-key\",
    \"ABUSEIPDB_API_KEY\": \"your-abuseipdb-key\",
    \"VIRUSTOTAL_API_KEY\": \"your-virustotal-key\"
  }"

# Create Redis connection secret
aws secretsmanager create-secret \
  --name mini-xdr/redis \
  --description "Mini-XDR Redis Connection" \
  --secret-string "{
    \"host\": \"$REDIS_ENDPOINT\",
    \"port\": 6379
  }"

echo "Secrets created in AWS Secrets Manager"
```

---

## Phase 2: EKS Cluster Deployment

### Step 2.1: Create EKS Cluster Configuration

Create `infrastructure/aws/eks-cluster-config.yaml`:

```yaml
apiVersion: eksctl.io/v1alpha5
kind: ClusterConfig

metadata:
  name: mini-xdr-cluster
  region: us-east-1
  version: "1.28"
  tags:
    Environment: production
    Project: mini-xdr

# Import existing VPC
vpc:
  id: "vpc-xxxxx"  # Replace with your VPC ID from Step 1.1
  subnets:
    private:
      us-east-1a:
        id: "subnet-xxxxx"  # Replace with private subnet IDs
      us-east-1b:
        id: "subnet-xxxxx"
      us-east-1c:
        id: "subnet-xxxxx"
    public:
      us-east-1a:
        id: "subnet-xxxxx"  # Replace with public subnet IDs
      us-east-1b:
        id: "subnet-xxxxx"
      us-east-1c:
        id: "subnet-xxxxx"

# IAM OIDC provider (required for service accounts)
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

# Managed Node Groups
managedNodeGroups:
  - name: mini-xdr-ng-1
    instanceType: t3.medium
    desiredCapacity: 2
    minSize: 2
    maxSize: 6
    volumeSize: 30
    volumeType: gp3
    privateNetworking: true
    ssh:
      allow: true
      publicKeyName: mini-xdr-eks-key  # Create this key pair first
    labels:
      role: application
      environment: production
    tags:
      k8s.io/cluster-autoscaler/mini-xdr-cluster: owned
      k8s.io/cluster-autoscaler/enabled: "true"
    iam:
      withAddonPolicies:
        autoScaler: true
        albIngress: true
        cloudWatch: true
        ebs: true

# CloudWatch Logging
cloudWatch:
  clusterLogging:
    enableTypes:
      - "api"
      - "audit"
      - "authenticator"
      - "controllerManager"
      - "scheduler"
    logRetentionInDays: 7

# Add-ons
addons:
  - name: vpc-cni
    version: latest
  - name: coredns
    version: latest
  - name: kube-proxy
    version: latest
  - name: aws-ebs-csi-driver
    version: latest
```

### Step 2.2: Deploy EKS Cluster

```bash
# Create EC2 key pair for SSH access to nodes
aws ec2 create-key-pair \
  --key-name mini-xdr-eks-key \
  --query 'KeyMaterial' \
  --output text > ~/.ssh/mini-xdr-eks-key.pem
chmod 400 ~/.ssh/mini-xdr-eks-key.pem

# Get VPC and subnet IDs
VPC_ID=$(aws cloudformation describe-stacks \
  --stack-name mini-xdr-vpc \
  --query 'Stacks[0].Outputs[?OutputKey==`VPC`].OutputValue' \
  --output text)

PRIVATE_SUBNETS=$(aws cloudformation describe-stacks \
  --stack-name mini-xdr-vpc \
  --query 'Stacks[0].Outputs[?OutputKey==`PrivateSubnets`].OutputValue' \
  --output text)

PUBLIC_SUBNETS=$(aws cloudformation describe-stacks \
  --stack-name mini-xdr-vpc \
  --query 'Stacks[0].Outputs[?OutputKey==`PublicSubnets`].OutputValue' \
  --output text)

# Update the config file with actual IDs
# (Do this manually or with sed)

# Create EKS cluster (takes 15-20 minutes)
eksctl create cluster -f infrastructure/aws/eks-cluster-config.yaml

# Verify cluster
kubectl get nodes
kubectl get pods -A

# Enable kubectl context
aws eks update-kubeconfig \
  --region us-east-1 \
  --name mini-xdr-cluster
```

### Step 2.3: Install AWS Load Balancer Controller

```bash
# Download IAM policy
curl -o iam_policy.json https://raw.githubusercontent.com/kubernetes-sigs/aws-load-balancer-controller/v2.6.2/docs/install/iam_policy.json

# Create IAM policy
aws iam create-policy \
  --policy-name AWSLoadBalancerControllerIAMPolicy \
  --policy-document file://iam_policy.json

# Get your AWS account ID
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

# Associate IAM OIDC provider
eksctl utils associate-iam-oidc-provider \
  --region us-east-1 \
  --cluster mini-xdr-cluster \
  --approve

# Create service account
eksctl create iamserviceaccount \
  --cluster=mini-xdr-cluster \
  --namespace=kube-system \
  --name=aws-load-balancer-controller \
  --role-name AmazonEKSLoadBalancerControllerRole \
  --attach-policy-arn=arn:aws:iam::${AWS_ACCOUNT_ID}:policy/AWSLoadBalancerControllerIAMPolicy \
  --approve

# Add Helm repo
helm repo add eks https://aws.github.io/eks-charts
helm repo update

# Install AWS Load Balancer Controller
helm install aws-load-balancer-controller eks/aws-load-balancer-controller \
  -n kube-system \
  --set clusterName=mini-xdr-cluster \
  --set serviceAccount.create=false \
  --set serviceAccount.name=aws-load-balancer-controller

# Verify installation
kubectl get deployment -n kube-system aws-load-balancer-controller
```

### Step 2.4: Install Cluster Autoscaler

```bash
# Create IAM policy for autoscaler
cat > cluster-autoscaler-policy.json <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "autoscaling:DescribeAutoScalingGroups",
        "autoscaling:DescribeAutoScalingInstances",
        "autoscaling:DescribeLaunchConfigurations",
        "autoscaling:DescribeScalingActivities",
        "autoscaling:DescribeTags",
        "ec2:DescribeInstanceTypes",
        "ec2:DescribeLaunchTemplateVersions"
      ],
      "Resource": ["*"]
    },
    {
      "Effect": "Allow",
      "Action": [
        "autoscaling:SetDesiredCapacity",
        "autoscaling:TerminateInstanceInAutoScalingGroup",
        "ec2:DescribeImages",
        "ec2:GetInstanceTypesFromInstanceRequirements",
        "eks:DescribeNodegroup"
      ],
      "Resource": ["*"]
    }
  ]
}
EOF

aws iam create-policy \
  --policy-name AmazonEKSClusterAutoscalerPolicy \
  --policy-document file://cluster-autoscaler-policy.json

# Create service account
eksctl create iamserviceaccount \
  --cluster=mini-xdr-cluster \
  --namespace=kube-system \
  --name=cluster-autoscaler \
  --attach-policy-arn=arn:aws:iam::${AWS_ACCOUNT_ID}:policy/AmazonEKSClusterAutoscalerPolicy \
  --override-existing-serviceaccounts \
  --approve

# Deploy autoscaler
kubectl apply -f https://raw.githubusercontent.com/kubernetes/autoscaler/master/cluster-autoscaler/cloudprovider/aws/examples/cluster-autoscaler-autodiscover.yaml

# Patch deployment
kubectl patch deployment cluster-autoscaler \
  -n kube-system \
  -p '{"spec":{"template":{"metadata":{"annotations":{"cluster-autoscaler.kubernetes.io/safe-to-evict": "false"}}}}}'

kubectl set image deployment cluster-autoscaler \
  -n kube-system \
  cluster-autoscaler=registry.k8s.io/autoscaling/cluster-autoscaler:v1.28.2
```

---

## Phase 3: Application Deployment

### Step 3.1: Create Kubernetes Namespace

```bash
# Create namespace
kubectl create namespace mini-xdr

# Label namespace
kubectl label namespace mini-xdr environment=production
```

### Step 3.2: Create Secrets from AWS Secrets Manager

Create `k8s/secrets-from-aws.yaml`:

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: mini-xdr-secrets
  namespace: mini-xdr
type: Opaque
stringData:
  database-url: "postgresql://xdradmin:ChangeThisP@ssw0rd!@REPLACE_RDS_ENDPOINT:5432/xdrdb"
  redis-url: "redis://REPLACE_REDIS_ENDPOINT:6379"
  openai-api-key: "REPLACE_WITH_YOUR_KEY"
  abuseipdb-api-key: "REPLACE_WITH_YOUR_KEY"
  virustotal-api-key: "REPLACE_WITH_YOUR_KEY"
```

Or use External Secrets Operator (recommended):

```bash
# Install External Secrets Operator
helm repo add external-secrets https://charts.external-secrets.io
helm install external-secrets \
  external-secrets/external-secrets \
  -n external-secrets-system \
  --create-namespace

# Create SecretStore
cat <<EOF | kubectl apply -f -
apiVersion: external-secrets.io/v1beta1
kind: SecretStore
metadata:
  name: aws-secrets-manager
  namespace: mini-xdr
spec:
  provider:
    aws:
      service: SecretsManager
      region: us-east-1
      auth:
        jwt:
          serviceAccountRef:
            name: mini-xdr-backend
EOF

# Create ExternalSecret
cat <<EOF | kubectl apply -f -
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: mini-xdr-secrets
  namespace: mini-xdr
spec:
  refreshInterval: 1h
  secretStoreRef:
    name: aws-secrets-manager
    kind: SecretStore
  target:
    name: mini-xdr-secrets
    creationPolicy: Owner
  data:
    - secretKey: database-url
      remoteRef:
        key: mini-xdr/database
        property: username
    - secretKey: openai-api-key
      remoteRef:
        key: mini-xdr/api-keys
        property: OPENAI_API_KEY
EOF
```

### Step 3.3: Create ConfigMap

```bash
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: ConfigMap
metadata:
  name: mini-xdr-config
  namespace: mini-xdr
data:
  API_HOST: "0.0.0.0"
  API_PORT: "8000"
  LOG_LEVEL: "INFO"
  ENVIRONMENT: "production"
  REDIS_HOST: "$REDIS_ENDPOINT"
  REDIS_PORT: "6379"
  DATABASE_HOST: "$RDS_ENDPOINT"
  DATABASE_PORT: "5432"
  DATABASE_NAME: "xdrdb"
EOF
```

### Step 3.4: Deploy Backend Application

Create `k8s/backend-deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mini-xdr-backend
  namespace: mini-xdr
  labels:
    app: mini-xdr-backend
    version: v1
spec:
  replicas: 2
  selector:
    matchLabels:
      app: mini-xdr-backend
  template:
    metadata:
      labels:
        app: mini-xdr-backend
        version: v1
    spec:
      serviceAccountName: mini-xdr-backend
      containers:
      - name: backend
        image: YOUR_ECR_REPO/mini-xdr-backend:latest  # Build and push first
        ports:
        - containerPort: 8000
          name: http
        - containerPort: 9090
          name: metrics
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: mini-xdr-secrets
              key: database-url
        - name: REDIS_HOST
          valueFrom:
            configMapKeyRef:
              name: mini-xdr-config
              key: REDIS_HOST
        - name: REDIS_PORT
          valueFrom:
            configMapKeyRef:
              name: mini-xdr-config
              key: REDIS_PORT
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: mini-xdr-secrets
              key: openai-api-key
        - name: API_HOST
          valueFrom:
            configMapKeyRef:
              name: mini-xdr-config
              key: API_HOST
        - name: API_PORT
          valueFrom:
            configMapKeyRef:
              name: mini-xdr-config
              key: API_PORT
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: mini-xdr-backend-service
  namespace: mini-xdr
  labels:
    app: mini-xdr-backend
spec:
  type: ClusterIP
  selector:
    app: mini-xdr-backend
  ports:
  - name: http
    port: 8000
    targetPort: 8000
  - name: metrics
    port: 9090
    targetPort: 9090
```

### Step 3.5: Deploy Frontend Application

Create `k8s/frontend-deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mini-xdr-frontend
  namespace: mini-xdr
  labels:
    app: mini-xdr-frontend
    version: v1
spec:
  replicas: 3
  selector:
    matchLabels:
      app: mini-xdr-frontend
  template:
    metadata:
      labels:
        app: mini-xdr-frontend
        version: v1
    spec:
      containers:
      - name: frontend
        image: YOUR_ECR_REPO/mini-xdr-frontend:latest  # Build and push first
        ports:
        - containerPort: 3000
          name: http
        env:
        - name: NEXT_PUBLIC_API_URL
          value: "https://api.mini-xdr.yourdomain.com"
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
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /
            port: 3000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: mini-xdr-frontend-service
  namespace: mini-xdr
  labels:
    app: mini-xdr-frontend
spec:
  type: ClusterIP
  selector:
    app: mini-xdr-frontend
  ports:
  - name: http
    port: 3000
    targetPort: 3000
```

### Step 3.6: Build and Push Docker Images to ECR

```bash
# Create ECR repositories
aws ecr create-repository --repository-name mini-xdr-backend --region us-east-1
aws ecr create-repository --repository-name mini-xdr-frontend --region us-east-1

# Get ECR login
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin ${AWS_ACCOUNT_ID}.dkr.ecr.us-east-1.amazonaws.com

# Build and push backend
cd backend
docker build -t mini-xdr-backend:latest .
docker tag mini-xdr-backend:latest ${AWS_ACCOUNT_ID}.dkr.ecr.us-east-1.amazonaws.com/mini-xdr-backend:latest
docker push ${AWS_ACCOUNT_ID}.dkr.ecr.us-east-1.amazonaws.com/mini-xdr-backend:latest

# Build and push frontend
cd ../frontend
docker build -t mini-xdr-frontend:latest .
docker tag mini-xdr-frontend:latest ${AWS_ACCOUNT_ID}.dkr.ecr.us-east-1.amazonaws.com/mini-xdr-frontend:latest
docker push ${AWS_ACCOUNT_ID}.dkr.ecr.us-east-1.amazonaws.com/mini-xdr-frontend:latest

cd ..
```

### Step 3.7: Deploy Applications

```bash
# Deploy backend
kubectl apply -f k8s/backend-deployment.yaml

# Deploy frontend
kubectl apply -f k8s/frontend-deployment.yaml

# Verify deployments
kubectl get pods -n mini-xdr
kubectl get svc -n mini-xdr

# Check logs
kubectl logs -n mini-xdr -l app=mini-xdr-backend --tail=50
kubectl logs -n mini-xdr -l app=mini-xdr-frontend --tail=50
```

---

## Phase 4: Networking & Ingress

### Step 4.1: Request SSL Certificate from ACM

```bash
# Request certificate (requires domain verification)
aws acm request-certificate \
  --domain-name mini-xdr.yourdomain.com \
  --domain-name www.mini-xdr.yourdomain.com \
  --validation-method DNS \
  --region us-east-1

# Get certificate ARN
CERT_ARN=$(aws acm list-certificates \
  --region us-east-1 \
  --query 'CertificateSummaryList[?DomainName==`mini-xdr.yourdomain.com`].CertificateArn' \
  --output text)

echo "Certificate ARN: $CERT_ARN"

# Follow DNS validation instructions in AWS Console
# Add CNAME records to your DNS provider
```

### Step 4.2: Create ALB Ingress

Create `k8s/ingress.yaml`:

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: mini-xdr-ingress
  namespace: mini-xdr
  annotations:
    # AWS Load Balancer Controller annotations
    alb.ingress.kubernetes.io/scheme: internet-facing
    alb.ingress.kubernetes.io/target-type: ip
    alb.ingress.kubernetes.io/listen-ports: '[{"HTTP": 80}, {"HTTPS": 443}]'
    alb.ingress.kubernetes.io/ssl-redirect: '443'
    alb.ingress.kubernetes.io/certificate-arn: "arn:aws:acm:us-east-1:ACCOUNT:certificate/CERT_ID"
    
    # Health check settings
    alb.ingress.kubernetes.io/healthcheck-path: /health
    alb.ingress.kubernetes.io/healthcheck-interval-seconds: '15'
    alb.ingress.kubernetes.io/healthcheck-timeout-seconds: '5'
    alb.ingress.kubernetes.io/healthy-threshold-count: '2'
    alb.ingress.kubernetes.io/unhealthy-threshold-count: '2'
    
    # Security
    alb.ingress.kubernetes.io/security-groups: sg-xxxxx  # ALB security group
    alb.ingress.kubernetes.io/wafv2-acl-arn: "arn:aws:wafv2:us-east-1:ACCOUNT:regional/webacl/NAME/ID"
    
    # Performance
    alb.ingress.kubernetes.io/load-balancer-attributes: |
      idle_timeout.timeout_seconds=60,
      deletion_protection.enabled=true
    
    # Tags
    alb.ingress.kubernetes.io/tags: |
      Environment=production,
      Project=mini-xdr
spec:
  ingressClassName: alb
  rules:
  - host: mini-xdr.yourdomain.com
    http:
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
```

Deploy:

```bash
# Update certificate ARN in ingress.yaml
kubectl apply -f k8s/ingress.yaml

# Wait for ALB to be provisioned (3-5 minutes)
kubectl get ingress -n mini-xdr -w

# Get ALB DNS name
ALB_DNS=$(kubectl get ingress -n mini-xdr mini-xdr-ingress \
  -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')

echo "ALB DNS Name: $ALB_DNS"
```

### Step 4.3: Configure Route 53 DNS

```bash
# Get hosted zone ID for your domain
HOSTED_ZONE_ID=$(aws route53 list-hosted-zones-by-name \
  --query "HostedZones[?Name=='yourdomain.com.'].Id" \
  --output text | cut -d'/' -f3)

# Get ALB hosted zone ID (for us-east-1)
ALB_ZONE_ID="Z35SXDOTRQ7X7K"

# Create Route 53 record
cat > route53-change.json <<EOF
{
  "Changes": [
    {
      "Action": "UPSERT",
      "ResourceRecordSet": {
        "Name": "mini-xdr.yourdomain.com",
        "Type": "A",
        "AliasTarget": {
          "HostedZoneId": "$ALB_ZONE_ID",
          "DNSName": "$ALB_DNS",
          "EvaluateTargetHealth": true
        }
      }
    }
  ]
}
EOF

aws route53 change-resource-record-sets \
  --hosted-zone-id $HOSTED_ZONE_ID \
  --change-batch file://route53-change.json

echo "DNS configured: https://mini-xdr.yourdomain.com"
```

### Step 4.4: Configure AWS WAF (Web Application Firewall)

```bash
# Create WAF WebACL
cat > waf-rules.json <<EOF
{
  "Name": "mini-xdr-waf",
  "Scope": "REGIONAL",
  "DefaultAction": {
    "Allow": {}
  },
  "Description": "WAF rules for Mini-XDR",
  "Rules": [
    {
      "Name": "RateLimitRule",
      "Priority": 1,
      "Statement": {
        "RateBasedStatement": {
          "Limit": 2000,
          "AggregateKeyType": "IP"
        }
      },
      "Action": {
        "Block": {}
      },
      "VisibilityConfig": {
        "SampledRequestsEnabled": true,
        "CloudWatchMetricsEnabled": true,
        "MetricName": "RateLimit"
      }
    },
    {
      "Name": "AWSManagedRulesCommonRuleSet",
      "Priority": 2,
      "Statement": {
        "ManagedRuleGroupStatement": {
          "VendorName": "AWS",
          "Name": "AWSManagedRulesCommonRuleSet"
        }
      },
      "OverrideAction": {
        "None": {}
      },
      "VisibilityConfig": {
        "SampledRequestsEnabled": true,
        "CloudWatchMetricsEnabled": true,
        "MetricName": "CommonRuleSet"
      }
    }
  ],
  "VisibilityConfig": {
    "SampledRequestsEnabled": true,
    "CloudWatchMetricsEnabled": true,
    "MetricName": "mini-xdr-waf"
  }
}
EOF

aws wafv2 create-web-acl \
  --region us-east-1 \
  --cli-input-json file://waf-rules.json

# Associate WAF with ALB (update ingress annotation with WAF ARN)
```

---

## Phase 5: Security Hardening

### Step 5.1: Enable Pod Security Standards

```bash
# Label namespace for restricted pod security
kubectl label namespace mini-xdr \
  pod-security.kubernetes.io/enforce=restricted \
  pod-security.kubernetes.io/audit=restricted \
  pod-security.kubernetes.io/warn=restricted
```

### Step 5.2: Create Network Policies

Create `k8s/network-policies.yaml`:

```yaml
# Allow frontend to backend
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: frontend-to-backend
  namespace: mini-xdr
spec:
  podSelector:
    matchLabels:
      app: mini-xdr-backend
  policyTypes:
  - Ingress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: mini-xdr-frontend
    - podSelector:
        matchLabels:
          app.kubernetes.io/name: ingress-nginx
    ports:
    - protocol: TCP
      port: 8000
---
# Allow ingress to frontend
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: ingress-to-frontend
  namespace: mini-xdr
spec:
  podSelector:
    matchLabels:
      app: mini-xdr-frontend
  policyTypes:
  - Ingress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: kube-system
    ports:
    - protocol: TCP
      port: 3000
---
# Deny all other traffic
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: default-deny
  namespace: mini-xdr
spec:
  podSelector: {}
  policyTypes:
  - Ingress
  - Egress
  egress:
  - to:
    - namespaceSelector: {}
  - to:
    - podSelector:
        matchLabels:
          k8s-app: kube-dns
    ports:
    - protocol: UDP
      port: 53
```

```bash
kubectl apply -f k8s/network-policies.yaml
```

### Step 5.3: Enable Encryption at Rest

```bash
# EBS volumes are encrypted by default in EKS
# RDS encryption enabled in Step 1.3
# Enable S3 bucket encryption

aws s3api put-bucket-encryption \
  --bucket mini-xdr-logs \
  --server-side-encryption-configuration '{
    "Rules": [{
      "ApplyServerSideEncryptionByDefault": {
        "SSEAlgorithm": "AES256"
      }
    }]
  }'
```

### Step 5.4: Setup RBAC

Create `k8s/rbac.yaml`:

```yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: mini-xdr-backend
  namespace: mini-xdr
---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: mini-xdr-backend-role
  namespace: mini-xdr
rules:
- apiGroups: [""]
  resources: ["secrets", "configmaps"]
  verbs: ["get", "list"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: mini-xdr-backend-rolebinding
  namespace: mini-xdr
subjects:
- kind: ServiceAccount
  name: mini-xdr-backend
  namespace: mini-xdr
roleRef:
  kind: Role
  name: mini-xdr-backend-role
  apiGroup: rbac.authorization.k8s.io
```

```bash
kubectl apply -f k8s/rbac.yaml
```

---

## Phase 6: Monitoring & Observability

### Step 6.1: Enable Container Insights

```bash
# Enable Container Insights
aws eks update-cluster-config \
  --region us-east-1 \
  --name mini-xdr-cluster \
  --logging '{"clusterLogging":[{"types":["api","audit","authenticator","controllerManager","scheduler"],"enabled":true}]}'

# Install CloudWatch agent
kubectl apply -f https://raw.githubusercontent.com/aws-samples/amazon-cloudwatch-container-insights/latest/k8s-deployment-manifest-templates/deployment-mode/daemonset/container-insights-monitoring/quickstart/cwagent-fluentd-quickstart.yaml
```

### Step 6.2: Install Prometheus & Grafana

```bash
# Add Helm repos
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

# Install Prometheus Stack
helm install prometheus prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --create-namespace \
  --set prometheus.prometheusSpec.serviceMonitorSelectorNilUsesHelmValues=false \
  --set grafana.adminPassword='YourSecurePassword'

# Expose Grafana (for demo - use Ingress for production)
kubectl port-forward -n monitoring svc/prometheus-grafana 3000:80
```

### Step 6.3: Configure CloudWatch Alarms

```bash
# Create CloudWatch alarm for high CPU
aws cloudwatch put-metric-alarm \
  --alarm-name mini-xdr-high-cpu \
  --alarm-description "Alert when EKS cluster CPU > 80%" \
  --metric-name node_cpu_utilization \
  --namespace ContainerInsights \
  --statistic Average \
  --period 300 \
  --evaluation-periods 2 \
  --threshold 80 \
  --comparison-operator GreaterThanThreshold \
  --dimensions Name=ClusterName,Value=mini-xdr-cluster

# Create alarm for high memory
aws cloudwatch put-metric-alarm \
  --alarm-name mini-xdr-high-memory \
  --alarm-description "Alert when EKS cluster memory > 80%" \
  --metric-name node_memory_utilization \
  --namespace ContainerInsights \
  --statistic Average \
  --period 300 \
  --evaluation-periods 2 \
  --threshold 80 \
  --comparison-operator GreaterThanThreshold \
  --dimensions Name=ClusterName,Value=mini-xdr-cluster
```

---

## Phase 7: CI/CD Pipeline

### Step 7.1: Setup AWS CodePipeline

Create `infrastructure/aws/codepipeline.yaml`:

```yaml
AWSTemplateFormatVersion: '2010-09-09'
Description: 'CI/CD Pipeline for Mini-XDR'

Resources:
  CodeBuildProject:
    Type: AWS::CodeBuild::Project
    Properties:
      Name: mini-xdr-build
      ServiceRole: !GetAtt CodeBuildRole.Arn
      Artifacts:
        Type: CODEPIPELINE
      Environment:
        Type: LINUX_CONTAINER
        ComputeType: BUILD_GENERAL1_SMALL
        Image: aws/codebuild/standard:7.0
        PrivilegedMode: true
        EnvironmentVariables:
          - Name: AWS_ACCOUNT_ID
            Value: !Ref AWS::AccountId
          - Name: ECR_REPO_BACKEND
            Value: mini-xdr-backend
          - Name: ECR_REPO_FRONTEND
            Value: mini-xdr-frontend
          - Name: EKS_CLUSTER_NAME
            Value: mini-xdr-cluster
      Source:
        Type: CODEPIPELINE
        BuildSpec: |
          version: 0.2
          phases:
            pre_build:
              commands:
                - echo Logging into ECR...
                - aws ecr get-login-password --region $AWS_DEFAULT_REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_DEFAULT_REGION.amazonaws.com
                - COMMIT_HASH=$(echo $CODEBUILD_RESOLVED_SOURCE_VERSION | cut -c 1-7)
                - IMAGE_TAG=${COMMIT_HASH:=latest}
            build:
              commands:
                - echo Building backend...
                - cd backend
                - docker build -t $ECR_REPO_BACKEND:$IMAGE_TAG .
                - docker tag $ECR_REPO_BACKEND:$IMAGE_TAG $AWS_ACCOUNT_ID.dkr.ecr.$AWS_DEFAULT_REGION.amazonaws.com/$ECR_REPO_BACKEND:$IMAGE_TAG
                - cd ..
                - echo Building frontend...
                - cd frontend
                - docker build -t $ECR_REPO_FRONTEND:$IMAGE_TAG .
                - docker tag $ECR_REPO_FRONTEND:$IMAGE_TAG $AWS_ACCOUNT_ID.dkr.ecr.$AWS_DEFAULT_REGION.amazonaws.com/$ECR_REPO_FRONTEND:$IMAGE_TAG
                - cd ..
            post_build:
              commands:
                - echo Pushing images to ECR...
                - docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_DEFAULT_REGION.amazonaws.com/$ECR_REPO_BACKEND:$IMAGE_TAG
                - docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_DEFAULT_REGION.amazonaws.com/$ECR_REPO_FRONTEND:$IMAGE_TAG
                - echo Updating EKS deployment...
                - aws eks update-kubeconfig --name $EKS_CLUSTER_NAME --region $AWS_DEFAULT_REGION
                - kubectl set image deployment/mini-xdr-backend mini-xdr-backend=$AWS_ACCOUNT_ID.dkr.ecr.$AWS_DEFAULT_REGION.amazonaws.com/$ECR_REPO_BACKEND:$IMAGE_TAG -n mini-xdr
                - kubectl set image deployment/mini-xdr-frontend mini-xdr-frontend=$AWS_ACCOUNT_ID.dkr.ecr.$AWS_DEFAULT_REGION.amazonaws.com/$ECR_REPO_FRONTEND:$IMAGE_TAG -n mini-xdr
                - kubectl rollout status deployment/mini-xdr-backend -n mini-xdr
                - kubectl rollout status deployment/mini-xdr-frontend -n mini-xdr

  CodeBuildRole:
    Type: AWS::IAM::Role
    Properties:
      AssumeRolePolicyDocument:
        Version: '2012-10-17'
        Statement:
          - Effect: Allow
            Principal:
              Service: codebuild.amazonaws.com
            Action: sts:AssumeRole
      ManagedPolicyArns:
        - arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryPowerUser
        - arn:aws:iam::aws:policy/AmazonEKSClusterPolicy
      Policies:
        - PolicyName: CodeBuildPolicy
          PolicyDocument:
            Version: '2012-10-17'
            Statement:
              - Effect: Allow
                Action:
                  - logs:CreateLogGroup
                  - logs:CreateLogStream
                  - logs:PutLogEvents
                Resource: '*'
              - Effect: Allow
                Action:
                  - eks:DescribeCluster
                Resource: '*'
```

---

## Cost Optimization

### Estimated Monthly Costs (us-east-1)

| Service | Configuration | Estimated Cost |
|---------|--------------|----------------|
| EKS Cluster | Control plane | $73/month |
| EC2 Instances | 2x t3.medium (on-demand) | $60/month |
| RDS PostgreSQL | db.t3.micro (Multi-AZ) | $34/month |
| ElastiCache Redis | cache.t3.micro | $12/month |
| Application Load Balancer | 1 ALB | $23/month |
| Data Transfer | ~100GB/month | $9/month |
| NAT Gateway | 1 NAT Gateway | $32/month |
| CloudWatch Logs | 10GB/month | $5/month |
| **Total** | | **~$248/month** |

### Cost Reduction Strategies

1. **Use Spot Instances** (save ~70%)
```bash
# Add spot instance node group
eksctl create nodegroup \
  --cluster mini-xdr-cluster \
  --name spot-nodes \
  --instance-types t3.medium,t3a.medium \
  --spot \
  --nodes 2 \
  --nodes-min 1 \
  --nodes-max 4
```

2. **Schedule scaling** (turn off at night)
```bash
# Add to cron: Scale down at night, up in morning
0 22 * * * kubectl scale deployment --all --replicas=0 -n mini-xdr
0 8 * * * kubectl scale deployment mini-xdr-backend --replicas=2 -n mini-xdr
0 8 * * * kubectl scale deployment mini-xdr-frontend --replicas=3 -n mini-xdr
```

3. **Use S3 instead of EBS where possible**
4. **Enable AWS Cost Explorer** and set up budget alerts
5. **Consider Fargate** for serverless (no EC2 costs)

---

## Troubleshooting

### Common Issues

**Issue 1: Pods can't pull images from ECR**
```bash
# Check IAM permissions
kubectl describe pod -n mini-xdr POD_NAME | grep -A 5 Events

# Verify ECR login
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin ${AWS_ACCOUNT_ID}.dkr.ecr.us-east-1.amazonaws.com
```

**Issue 2: ALB not created**
```bash
# Check AWS Load Balancer Controller logs
kubectl logs -n kube-system deployment/aws-load-balancer-controller

# Verify service account
kubectl get sa -n kube-system aws-load-balancer-controller -o yaml
```

**Issue 3: Can't connect to RDS**
```bash
# Test from pod
kubectl run -it --rm debug --image=postgres:15 --restart=Never -- psql -h $RDS_ENDPOINT -U xdradmin -d xdrdb

# Check security groups
aws ec2 describe-security-groups --group-ids $RDS_SG_ID
```

**Issue 4: High costs**
```bash
# Check current costs
aws ce get-cost-and-usage \
  --time-period Start=2025-01-01,End=2025-01-31 \
  --granularity MONTHLY \
  --metrics BlendedCost \
  --group-by Type=SERVICE

# Identify large resources
aws ec2 describe-volumes --filters "Name=status,Values=available"
aws rds describe-db-snapshots --query 'DBSnapshots[*].[DBSnapshotIdentifier,AllocatedStorage]'
```

---

## Next Steps

1. **SSL/TLS**: Ensure ACM certificate is validated
2. **Backup**: Set up automated RDS and EBS backups
3. **DR Plan**: Document disaster recovery procedures
4. **Load Testing**: Use k6 or Locust to test at scale
5. **Security Audit**: Run AWS Trusted Advisor
6. **Documentation**: Keep this guide updated with your specific settings

---

## Quick Reference Commands

```bash
# Get cluster info
kubectl cluster-info
aws eks describe-cluster --name mini-xdr-cluster

# Update kubeconfig
aws eks update-kubeconfig --region us-east-1 --name mini-xdr-cluster

# Check all resources
kubectl get all -n mini-xdr

# View logs
kubectl logs -n mini-xdr -l app=mini-xdr-backend --tail=100 -f

# Scale deployment
kubectl scale deployment mini-xdr-backend --replicas=3 -n mini-xdr

# Exec into pod
kubectl exec -it -n mini-xdr POD_NAME -- /bin/bash

# Port forward for local testing
kubectl port-forward -n mini-xdr svc/mini-xdr-frontend-service 3000:3000

# Delete everything (CAREFUL!)
eksctl delete cluster --name mini-xdr-cluster
```

---

## Support Resources

- **AWS EKS Documentation**: https://docs.aws.amazon.com/eks/
- **Kubernetes Documentation**: https://kubernetes.io/docs/
- **AWS Load Balancer Controller**: https://kubernetes-sigs.github.io/aws-load-balancer-controller/
- **eksctl**: https://eksctl.io/

---

**Document Version**: 1.0  
**Last Updated**: October 9, 2025  
**Maintained By**: Mini-XDR Team


