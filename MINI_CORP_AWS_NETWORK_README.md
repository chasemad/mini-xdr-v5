# Mini Corporate Network for Mini-XDR Testing

## Overview

This document provides a comprehensive plan for creating a "mini corporate network" on AWS that will be used to test Mini-XDR's onboarding, network discovery, agent deployment, and security monitoring capabilities. The network is designed to mimic a small corporate environment with diverse systems that provide realistic security events for ML model training and agent testing.

## Current AWS Environment Prerequisites

- **EKS foundation already running**: `docs/deployment/aws/overview.md` confirms the production cluster `mini-xdr-cluster` in `us-east-1` fronted by the ALB `http://k8s-minixdr-minixdri-dc5fc1df8b-1132128475.us-east-1.elb.amazonaws.com`. Treat this as the source VPC that must reach the new corporate lab.
- **Capture live identifiers up front**:
  ```bash
  AWS_REGION=us-east-1
  MINIXDR_CLUSTER=mini-xdr-cluster

  MINIXDR_VPC_ID=$(aws eks describe-cluster \
    --name "$MINIXDR_CLUSTER" \
    --region "$AWS_REGION" \
    --query "cluster.resourcesVpcConfig.vpcId" \
    --output text)

  MINIXDR_SUBNETS=$(aws eks describe-cluster \
    --name "$MINIXDR_CLUSTER" \
    --region "$AWS_REGION" \
    --query "cluster.resourcesVpcConfig.subnetIds" \
    --output text)

  echo "Mini-XDR VPC:        $MINIXDR_VPC_ID"
  echo "Mini-XDR Subnets:    $MINIXDR_SUBNETS"
  ```
- **Note the VPC CIDR**: The CloudFormation VPC stack under `infrastructure/aws/vpc-stack.yaml` defaults to `10.0.0.0/16`; verify the actual allocation and reuse it when authorising traffic from Mini-XDR into the corporate lab.
- **IAM prerequisites**: Create an EC2 instance profile that carries `AmazonSSMManagedInstanceCore`. All corporate systems rely on SSM Session Manager instead of exposing SSH/RDP directly to the internet.
- **Secrets & configuration**: Mini-XDR backend still serves agent installers from the internal service name `backend-service:8000`. We will adjust the deployment to expose a `AGENT_PUBLIC_BASE_URL` env var so the existing agent enrollment service can hand out ALB-aware installers.

## Current Mini-XDR Deployment Status

### ✅ Successfully Deployed Components
- **EKS Cluster**: `mini-xdr-cluster` in `us-east-1`
- **Namespace**: `mini-xdr` with backend and frontend pods running
- **ALB Endpoint**: `http://k8s-minixdr-minixdri-dc5fc1df8b-1132128475.us-east-1.elb.amazonaws.com`
- **Authentication**: Working JWT authentication with user `chasemadrian@protonmail.com`
- **Onboarding System**: Complete 4-step wizard (Profile → Network Scan → Agents → Validation)

### 🔧 Onboarding Architecture
- **Network Discovery**: ICMP sweep + TCP port scanning (ports: 22, 80, 443, 445, 3389, 3306, 5432, etc.)
- **Agent Deployment**: Supports Windows, Linux, macOS, Docker platforms
- **Agent Types**: 12 specialized agents (containment, forensics, EDR, IAM, attribution, etc.)
- **HMAC Authentication**: Secure agent-to-backend communication

## Mini Corporate Network Architecture

### Network Design
```
VPC: 10.100.0.0/16 (Separate from Mini-XDR's 10.0.0.0/16)
├── Public Subnet: 10.100.1.0/24 (Bastion host)
├── Private Subnet: 10.100.10.0/24 (Corporate systems)
│   ├── Domain Controller (Windows Server)
│   ├── File Server (Windows Server)
│   ├── Web Server (Linux)
│   ├── Database Server (Linux)
│   ├── Workstation 1 (Windows 11)
│   ├── Workstation 2 (Ubuntu Desktop)
│   └── Honeypot (T-Pot)
└── Security Groups: Restricted to your IP only
```

### Target Systems for Detection & Monitoring

| System | OS | Services | Detection Purpose | Agent Deployment |
|--------|----|----------|------------------|------------------|
| **DC-01** | Windows Server 2022 | AD DS, DNS, DHCP | Domain authentication, user management | EDR + IAM agents |
| **FS-01** | Windows Server 2019 | SMB, File Shares | File access monitoring, lateral movement | DLP + Forensics agents |
| **WEB-01** | Ubuntu 22.04 | Apache/Nginx, SSH | Web attacks, service enumeration | Containment + Attribution agents |
| **DB-01** | Ubuntu 22.04 | PostgreSQL, MySQL | Database attacks, data exfiltration | DLP + Predictive Hunter agents |
| **WK-01** | Windows 11 Pro | RDP, SMB | Endpoint behavior, credential theft | EDR + Forensics agents |
| **WK-02** | Ubuntu 22.04 Desktop | SSH, VNC | Linux endpoint monitoring | Ingestion + NLP agents |
| **HP-01** | T-Pot (Ubuntu) | 20+ honeypot services | Attack simulation, threat intelligence | Deception + Coordination agents |

## Security & Access Control

### IP-Based Access Restrictions
```bash
# Your IP address (replace with actual)
YOUR_IP="YOUR.ACTUAL.IP.ADDRESS/32"

# Security group rules - INBOUND ONLY from your IP
aws ec2 authorize-security-group-ingress \
  --group-id $CORP_SG_ID \
  --protocol tcp \
  --port 22 \
  --cidr $YOUR_IP \
  --description "SSH access for testing"

aws ec2 authorize-security-group-ingress \
  --group-id $CORP_SG_ID \
  --protocol tcp \
  --port 3389 \
  --cidr $YOUR_IP \
  --description "RDP access for testing"
```

### Zero Trust Network Design
- **No internet access** for private subnet systems
- **Bastion host** for secure access to private systems
- **Security groups** restrict all traffic to your IP only
- **Network ACLs** provide additional layer of network-level filtering

## Deployment Steps

### 0. Prepare IAM Role and Session Manager Access

1. **Create the EC2 instance profile for all lab instances** (one-time).
   ```bash
   cat <<'EOF' > /tmp/minicorp-ec2-trust.json
   {
     "Version": "2012-10-17",
     "Statement": [
       {
         "Effect": "Allow",
         "Principal": { "Service": "ec2.amazonaws.com" },
         "Action": "sts:AssumeRole"
       }
     ]
   }
   EOF

   aws iam create-role \
     --role-name MiniCorpSSMRole \
     --assume-role-policy-document file:///tmp/minicorp-ec2-trust.json

   aws iam attach-role-policy \
     --role-name MiniCorpSSMRole \
     --policy-arn arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore

   aws iam create-instance-profile \
     --instance-profile-name MiniCorpSSMInstanceProfile

   aws iam add-role-to-instance-profile \
     --instance-profile-name MiniCorpSSMInstanceProfile \
     --role-name MiniCorpSSMRole
   ```
2. **Plan for VPC endpoints**: The lab uses Session Manager and needs access to S3, SSM, and SSM Messages without opening the private subnet to the internet. The CloudFormation template below provisions these interface endpoints; no extra action required post stack creation.

### 1. Create Corporate VPC & Networking

1. **Check the template into source control**: place the CloudFormation file at `infrastructure/aws/mini-corp-vpc.yaml` (full template shown later in this document). The template provisions the VPC, subnets, NAT gateway, interface endpoints, and base security group.
2. **Deploy the stack with parameter overrides**:
   ```bash
   ADMIN_CIDR="$(curl -s https://checkip.amazonaws.com)/32"
   MINIXDR_VPC_CIDR=$(aws ec2 describe-vpcs \
     --vpc-ids "$MINIXDR_VPC_ID" \
     --region "$AWS_REGION" \
     --query "Vpcs[0].CidrBlock" \
     --output text)

   aws cloudformation deploy \
     --stack-name mini-corp-vpc \
     --template-file infrastructure/aws/mini-corp-vpc.yaml \
     --parameter-overrides \
       AllowedAdminCidr="$ADMIN_CIDR" \
       MiniXdrVpcCidr="$MINIXDR_VPC_CIDR" \
     --capabilities CAPABILITY_NAMED_IAM
   ```
3. **Capture stack outputs for later steps**:
   ```bash
   CORP_VPC_ID=$(aws cloudformation describe-stacks \
     --stack-name mini-corp-vpc \
     --query 'Stacks[0].Outputs[?OutputKey==`VpcId`].OutputValue' \
     --output text)

   CORP_PUBLIC_SUBNET=$(aws cloudformation describe-stacks \
     --stack-name mini-corp-vpc \
     --query 'Stacks[0].Outputs[?OutputKey==`PublicSubnetId`].OutputValue' \
     --output text)

   CORP_PRIVATE_SUBNET=$(aws cloudformation describe-stacks \
     --stack-name mini-corp-vpc \
     --query 'Stacks[0].Outputs[?OutputKey==`PrivateSubnetId`].OutputValue' \
     --output text)

   CORP_PRIVATE_RT=$(aws cloudformation describe-stacks \
     --stack-name mini-corp-vpc \
     --query 'Stacks[0].Outputs[?OutputKey==`PrivateRouteTableId`].OutputValue' \
     --output text)

   CORP_SG_ID=$(aws cloudformation describe-stacks \
     --stack-name mini-corp-vpc \
     --query 'Stacks[0].Outputs[?OutputKey==`CorporateSecurityGroupId`].OutputValue' \
     --output text)
   ```

### 2. Deploy Bastion Host

```bash
# Launch bastion host in public subnet
aws ec2 run-instances \
  --image-id ami-0abcdef1234567890 \  # Ubuntu 22.04 AMI
  --instance-type t3.medium \
  --key-name mini-corp-key \
  --security-group-ids $CORP_SG_ID \
  --subnet-id $CORP_PUBLIC_SUBNET \
  --associate-public-ip-address \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=Mini-Corp-Bastion}]'

# Configure bastion for SSH proxy to private instances
# Install tools: nmap, ansible, terraform, awscli
```

### 3. Deploy Corporate Systems

#### Windows Domain Controller (DC-01)
```bash
# Windows Server 2022 with AD DS
aws ec2 run-instances \
  --image-id ami-0abcdef1234567890 \  # Windows Server 2022 AMI
  --instance-type t3.medium \
  --key-name mini-corp-key \
  --security-group-ids $CORP_SG_ID \
  --subnet-id $CORP_PRIVATE_SUBNET \
  --private-ip-address 10.100.10.10 \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=DC-01}]'

# Post-deployment:
# - Promote to domain controller
# - Create domain: corp.local
# - Configure DNS and DHCP
# - Create test user accounts
```

#### Windows File Server (FS-01)
```bash
aws ec2 run-instances \
  --image-id ami-0abcdef1234567890 \  # Windows Server 2019 AMI
  --instance-type t3.small \
  --key-name mini-corp-key \
  --security-group-ids $CORP_SG_ID \
  --subnet-id $CORP_PRIVATE_SUBNET \
  --private-ip-address 10.100.10.11 \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=FS-01}]'

# Post-deployment:
# - Join to corp.local domain
# - Configure SMB shares
# - Enable file auditing
# - Create test files and permissions
```

#### Linux Web Server (WEB-01)
```bash
aws ec2 run-instances \
  --image-id ami-0abcdef1234567890 \  # Ubuntu 22.04 AMI
  --instance-type t3.small \
  --key-name mini-corp-key \
  --security-group-ids $CORP_SG_ID \
  --subnet-id $CORP_PRIVATE_SUBNET \
  --private-ip-address 10.100.10.12 \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=WEB-01}]'

# Post-deployment:
# - Install Apache/Nginx
# - Deploy test web applications
# - Configure SSL certificates
# - Enable access logging
```

#### Linux Database Server (DB-01)
```bash
aws ec2 run-instances \
  --image-id ami-0abcdef1234567890 \  # Ubuntu 22.04 AMI
  --instance-type t3.small \
  --key-name mini-corp-key \
  --security-group-ids $CORP_SG_ID \
  --subnet-id $CORP_PRIVATE_SUBNET \
  --private-ip-address 10.100.10.13 \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=DB-01}]'

# Post-deployment:
# - Install PostgreSQL and MySQL
# - Create test databases
# - Configure database users and permissions
# - Enable query logging
```

#### Windows Workstation (WK-01)
```bash
aws ec2 run-instances \
  --image-id ami-0abcdef1234567890 \  # Windows 11 AMI
  --instance-type t3.small \
  --key-name mini-corp-key \
  --security-group-ids $CORP_SG_ID \
  --subnet-id $CORP_PRIVATE_SUBNET \
  --private-ip-address 10.100.10.14 \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=WK-01}]'

# Post-deployment:
# - Join to corp.local domain
# - Install common business applications
# - Configure RDP access
# - Create local user accounts
```

#### Linux Workstation (WK-02)
```bash
aws ec2 run-instances \
  --image-id ami-0abcdef1234567890 \  # Ubuntu 22.04 Desktop AMI
  --instance-type t3.small \
  --key-name mini-corp-key \
  --security-group-ids $CORP_SG_ID \
  --subnet-id $CORP_PRIVATE_SUBNET \
  --private-ip-address 10.100.10.15 \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=WK-02}]'

# Post-deployment:
# - Install desktop environment
# - Configure SSH and VNC
# - Install development tools
# - Create user accounts
```

#### T-Pot Honeypot (HP-01)
```bash
aws ec2 run-instances \
  --image-id ami-0abcdef1234567890 \  # Ubuntu 22.04 AMI
  --instance-type t3.medium \
  --key-name mini-corp-key \
  --security-group-ids $CORP_SG_ID \
  --subnet-id $CORP_PRIVATE_SUBNET \
  --private-ip-address 10.100.10.16 \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=HP-01}]'

# Post-deployment:
# - Install T-Pot honeypot framework
# - Configure 20+ honeypot services
# - Enable log shipping to Mini-XDR
# - Set up deception scenarios
```

## Mini-XDR Onboarding Integration

### Network Scan Configuration
When onboarding through the Mini-XDR wizard:

1. **Network Range**: `10.100.10.0/24`
2. **Discovery Mode**: Quick scan (recommended for initial testing)
3. **Expected Detection**:
   - 7 live hosts (DC-01 through HP-01)
   - Mixed OS detection (Windows Server, Ubuntu, Windows 11)
   - Service enumeration (SSH, RDP, SMB, HTTP, databases)

### Agent Deployment Strategy
```
DC-01 (Windows Server): EDR + IAM agents
FS-01 (Windows Server): DLP + Forensics agents
WEB-01 (Ubuntu): Containment + Attribution agents
DB-01 (Ubuntu): DLP + Predictive Hunter agents
WK-01 (Windows 11): EDR + Forensics agents
WK-02 (Ubuntu Desktop): Ingestion + NLP agents
HP-01 (T-Pot): Deception + Coordination agents
```

### Expected Security Events for ML Training

#### Authentication Events
- Windows domain logins/logouts (DC-01)
- SSH authentication attempts (Linux systems)
- RDP connection logs (WK-01)

#### Network Events
- SMB file access (FS-01)
- HTTP/HTTPS requests (WEB-01)
- Database connections (DB-01)
- Honeypot interaction attempts (HP-01)

#### System Events
- Process creation/termination
- File system changes
- Registry modifications (Windows)
- Package installations (Linux)

#### Security Events
- Failed authentication attempts
- Privilege escalation attempts
- Malware simulation (honeypot)
- Data exfiltration attempts

## CloudFormation Templates

### Mini Corp VPC Template (`mini-corp-vpc.yaml`)

```yaml
AWSTemplateFormatVersion: '2010-09-09'
Description: 'Mini Corporate Network VPC for Mini-XDR Testing'

Parameters:
  YourIPAddress:
    Type: String
    Description: Your IP address with /32 CIDR for security group restrictions

Resources:
  VPC:
    Type: AWS::EC2::VPC
    Properties:
      CidrBlock: 10.100.0.0/16
      EnableDnsHostnames: true
      EnableDnsSupport: true
      Tags:
        - Key: Name
          Value: mini-corp-vpc

  InternetGateway:
    Type: AWS::EC2::InternetGateway
    Properties:
      Tags:
        - Key: Name
          Value: mini-corp-igw

  InternetGatewayAttachment:
    Type: AWS::EC2::InternetGateway
    Properties:
      InternetGatewayId: !Ref InternetGateway
      VpcId: !Ref VPC

  # Public Subnet for Bastion
  PublicSubnet:
    Type: AWS::EC2::Subnet
    Properties:
      VpcId: !Ref VPC
      AvailabilityZone: us-east-1a
      CidrBlock: 10.100.1.0/24
      MapPublicIpOnLaunch: true
      Tags:
        - Key: Name
          Value: mini-corp-public-subnet

  # Private Subnet for Corporate Systems
  PrivateSubnet:
    Type: AWS::EC2::Subnet
    Properties:
      VpcId: !Ref VPC
      AvailabilityZone: us-east-1a
      CidrBlock: 10.100.10.0/24
      Tags:
        - Key: Name
          Value: mini-corp-private-subnet

  # Security Group - Restricted to your IP only
  CorpSecurityGroup:
    Type: AWS::EC2::SecurityGroup
    Properties:
      GroupDescription: Security group for mini corp network
      VpcId: !Ref VPC
      SecurityGroupIngress:
        - IpProtocol: tcp
          FromPort: 22
          ToPort: 22
          CidrIp: !Ref YourIPAddress
          Description: SSH access
        - IpProtocol: tcp
          FromPort: 3389
          ToPort: 3389
          CidrIp: !Ref YourIPAddress
          Description: RDP access
        - IpProtocol: tcp
          FromPort: 80
          ToPort: 80
          CidrIp: !Ref YourIPAddress
          Description: HTTP access
        - IpProtocol: tcp
          FromPort: 443
          ToPort: 443
          CidrIp: !Ref YourIPAddress
          Description: HTTPS access
        - IpProtocol: tcp
          FromPort: 3389
          ToPort: 3389
          CidrIp: 10.100.0.0/16
          Description: Internal RDP
        - IpProtocol: tcp
          FromPort: 445
          ToPort: 445
          CidrIp: 10.100.0.0/16
          Description: SMB access
        - IpProtocol: tcp
          FromPort: 22
          ToPort: 22
          CidrIp: 10.100.0.0/16
          Description: Internal SSH
        - IpProtocol: tcp
          FromPort: 3306
          ToPort: 3306
          CidrIp: 10.100.0.0/16
          Description: MySQL access
        - IpProtocol: tcp
          FromPort: 5432
          ToPort: 5432
          CidrIp: 10.100.0.0/16
          Description: PostgreSQL access

  # Route Tables
  PublicRouteTable:
    Type: AWS::EC2::RouteTable
    Properties:
      VpcId: !Ref VPC
      Tags:
        - Key: Name
          Value: mini-corp-public-routes

  DefaultPublicRoute:
    Type: AWS::EC2::Route
    Properties:
      RouteTableId: !Ref PublicRouteTable
      DestinationCidrBlock: 0.0.0.0/0
      GatewayId: !Ref InternetGateway

  PublicSubnetRouteTableAssociation:
    Type: AWS::EC2::SubnetRouteTableAssociation
    Properties:
      RouteTableId: !Ref PublicRouteTable
      SubnetId: !Ref PublicSubnet

Outputs:
  VPC:
    Description: VPC ID
    Value: !Ref VPC
    Export:
      Name: mini-corp-vpc

  PublicSubnet:
    Description: Public subnet ID
    Value: !Ref PublicSubnet
    Export:
      Name: mini-corp-public-subnet

  PrivateSubnet:
    Description: Private subnet ID
    Value: !Ref PrivateSubnet
    Export:
      Name: mini-corp-private-subnet

  CorpSecurityGroup:
    Description: Security group ID
    Value: !Ref CorpSecurityGroup
    Export:
      Name: mini-corp-security-group
```

## Testing & Validation Plan

### 1. Network Discovery Testing
```bash
# From bastion host, test network scanning
nmap -sn 10.100.10.0/24  # ICMP sweep
nmap -p 22,80,443,445,3389 10.100.10.0/24  # Port scanning
```

### 2. Mini-XDR Onboarding
1. Access Mini-XDR frontend: `http://k8s-minixdr-minixdri-dc5fc1df8b-1132128475.us-east-1.elb.amazonaws.com`
2. Login with `chasemadrian@protonmail.com`
3. Start onboarding wizard
4. Configure network scan for `10.100.10.0/24`
5. Verify asset discovery and classification
6. Deploy agents to detected systems
7. Validate agent connectivity and data ingestion

### 3. Agent Functionality Testing
- **EDR Agent**: Monitor process creation, network connections
- **DLP Agent**: File access monitoring, data exfiltration detection
- **Forensics Agent**: Event log collection, evidence gathering
- **Containment Agent**: IP blocking, host isolation testing
- **Deception Agent**: Honeypot management and attack simulation

### 4. ML Model Validation
- Verify event ingestion from all system types
- Test anomaly detection across Windows/Linux environments
- Validate behavioral analysis and pattern recognition
- Assess model performance with real corporate network data

## Cost Optimization

### Instance Types (Testing Environment)
- **Bastion**: t3.medium ($0.05/hour)
- **Domain Controller**: t3.medium ($0.05/hour)
- **Other servers**: t3.small ($0.02/hour each)
- **Total**: ~$0.20/hour for 7 instances

### Usage Recommendations
- **Testing Hours**: 9 AM - 6 PM EST, Monday-Friday only
- **Automated Shutdown**: Use AWS Instance Scheduler for cost control
- **Spot Instances**: Consider spot instances for non-critical systems

## Potential Issues & Solutions

### Onboarding System Issues

#### Issue: Network Scan Not Detecting All Hosts
**Symptoms**: Some systems not appearing in discovery results
**Solutions**:
- Verify security groups allow ICMP and required ports
- Check that systems are actually running and network-accessible
- Review discovery service logs in Mini-XDR backend

#### Issue: Agent Deployment Failures
**Symptoms**: Agents fail to install or connect
**Solutions**:
- Verify target system credentials and permissions
- Check HMAC authentication configuration
- Validate agent platform compatibility (Windows vs Linux)
- Review agent enrollment service logs

#### Issue: No Security Events Generated
**Symptoms**: Low or no event ingestion for ML training
**Solutions**:
- Manually generate test events (logins, file access, network connections)
- Configure additional logging on target systems
- Verify agent configurations and data shipping
- Use honeypot to simulate attack traffic

### AWS-Specific Issues

#### Issue: IP Address Changes
**Symptoms**: Access blocked due to IP restrictions
**Solutions**:
- Update security groups with new IP address
- Consider using VPN or bastion host for consistent access
- Use AWS Client VPN for secure remote access

#### Issue: Cross-VPC Communication
**Symptoms**: Mini-XDR cannot reach corporate network
**Solutions**:
- Establish VPC peering between Mini-XDR VPC (10.0.0.0/16) and corporate VPC (10.100.0.0/16)
- Configure route tables for cross-VPC routing
- Update security groups to allow Mini-XDR access

## Potential Issues & Solutions

### Onboarding System Issues Identified

Based on analysis of the Mini-XDR onboarding system, here are critical issues that could prevent successful detection of the mini corporate network:

#### Issue 1: Cross-VPC Communication Required
**Problem**: Mini-XDR is deployed in VPC `10.0.0.0/16`, but the corporate network uses `10.100.0.0/16`. Network scanning will fail without VPC peering.

**Solution**:
```bash
# Create VPC peering connection
aws ec2 create-vpc-peering-connection \
  --vpc-id vpc-12345678 \  # Mini-XDR VPC
  --peer-vpc-id vpc-87654321 \  # Corporate VPC

# Accept the peering connection
aws ec2 accept-vpc-peering-connection \
  --vpc-peering-connection-id pcx-xxxxxxxx

# Update route tables for cross-VPC routing
aws ec2 create-route \
  --route-table-id rtb-mini-xdr-private \
  --destination-cidr-block 10.100.0.0/16 \
  --vpc-peering-connection-id pcx-xxxxxxxx

aws ec2 create-route \
  --route-table-id rtb-corp-private \
  --destination-cidr-block 10.0.0.0/16 \
  --vpc-peering-connection-id pcx-xxxxxxxx
```

#### Issue 2: Agent Installation Scripts Hardcode Internal URLs
**Problem**: Agent installation scripts use `backend-service:8000` which won't resolve from the corporate VPC.

**Solution**: Update the installation scripts to use the public ALB URL:
```bash
# In backend/app/agent_enrollment_service.py, change:
backend_url = "http://backend-service:8000"  # Internal only

# To use public ALB URL:
backend_url = "http://k8s-minixdr-minixdri-dc5fc1df8b-1132128475.us-east-1.elb.amazonaws.com"
```

#### Issue 3: Security Groups Block Network Discovery
**Problem**: Corporate security groups restrict all traffic to your IP only, but Mini-XDR needs ICMP and port scanning access.

**Solution**: Add temporary rules for Mini-XDR scanning:
```bash
# Allow ICMP from Mini-XDR VPC for host discovery
aws ec2 authorize-security-group-ingress \
  --group-id $CORP_SG_ID \
  --protocol icmp \
  --source-group sg-minixdr-eks-nodes

# Allow port scanning from Mini-XDR VPC
aws ec2 authorize-security-group-ingress \
  --group-id $CORP_SG_ID \
  --protocol tcp \
  --port 22,80,443,445,3389,3306,5432 \
  --source-group sg-minixdr-eks-nodes
```

#### Issue 4: Network Scan Range Configuration
**Problem**: Onboarding wizard needs correct CIDR range input.

**Solution**: During onboarding, use network range `10.100.10.0/24` for the private subnet scan.

#### Issue 5: OS Fingerprinting Accuracy
**Problem**: Network scanner may not correctly identify Windows vs Linux systems based on ports alone.

**Solution**: Ensure systems have expected services running:
- Windows: SMB (445), RDP (3389)
- Linux: SSH (22), HTTP/HTTPS (80/443)
- Domain Controller: LDAP (389), DNS (53)

#### Issue 6: Agent Download URLs Not Accessible
**Problem**: Agents need to download installation packages, but corporate VPC may not have internet access.

**Solution**: Configure NAT Gateway for corporate VPC to allow outbound internet access for agent downloads.

## Expected Onboarding Flow & Validation

### Step 1: Access Mini-XDR Frontend
1. **URL**: `http://k8s-minixdr-minixdri-dc5fc1df8b-1132128475.us-east-1.elb.amazonaws.com`
2. **Login**: Use `chasemadrian@protonmail.com`
3. **Navigation**: Access Organization Settings → Onboarding

### Step 2: Profile Setup
1. **Input**: Organization details (company name, industry, size)
2. **Expected**: Profile saves successfully, progresses to network scan step
3. **Validation**: Check backend logs for profile creation

### Step 3: Network Discovery Scan
1. **Input**: Network range `10.100.10.0/24`
2. **Scan Type**: Quick scan (recommended for initial testing)
3. **Expected Results**:
   - 7 live hosts detected (DC-01 through HP-01)
   - Systems classified as:
     - DC-01: Domain Controller (LDAP port 389)
     - FS-01: Windows Server (SMB port 445)
     - WEB-01: Web Server (HTTP/HTTPS ports 80/443)
     - DB-01: Database Server (MySQL/PostgreSQL ports)
     - WK-01: Windows Workstation (SMB + RDP)
     - WK-02: Linux Server (SSH port 22)
     - HP-01: Network Device or Unknown (multiple ports)

4. **Troubleshooting**:
   - **No hosts found**: Check VPC peering and security groups
   - **ICMP blocked**: Verify ICMP rules between VPCs
   - **Port scan fails**: Check TCP port rules (22, 80, 443, 445, 3389, 3306, 5432)

### Step 4: Agent Deployment Planning
1. **Expected**: Deployment matrix shows recommended agents for each system
2. **Priority Order**:
   - DC-01: EDR + IAM agents (critical priority)
   - FS-01: DLP + Forensics agents (high priority)
   - WEB-01: Containment + Attribution agents (high priority)
   - DB-01: DLP + Predictive Hunter agents (high priority)
   - WK-01: EDR + Forensics agents (high priority)
   - WK-02: Ingestion + NLP agents (high priority)
   - HP-01: Deception + Coordination agents (medium priority)

### Step 5: Agent Token Generation & Installation
1. **For each system**: Generate platform-specific installation scripts
2. **Expected**: Scripts include correct ALB URL for backend communication
3. **Installation Methods**:
   - **Linux systems**: SSH to bastion, then SSH to target systems
   - **Windows systems**: RDP through bastion host
   - **Domain Controller**: Use domain admin credentials

4. **Troubleshooting**:
   - **Script download fails**: Verify internet access via NAT Gateway
   - **Agent won't start**: Check systemd/Windows services
   - **HMAC authentication fails**: Verify agent credentials in database

### Step 6: Agent Verification & Validation
1. **Expected**: All agents report connectivity and basic capabilities
2. **Validation Tests**:
   - Agent heartbeat monitoring
   - Dry-run containment testing
   - Rollback capability verification
   - Platform-specific access validation

2. **Troubleshooting**:
   - **Connectivity fails**: Check VPC peering and security groups
   - **Permission denied**: Verify agent credentials and system access
   - **Dry-run fails**: Check agent service accounts and permissions

### Step 7: Onboarding Completion
1. **Expected**: Onboarding status changes to "completed"
2. **Post-Completion**: Systems should begin generating telemetry data
3. **Dashboard Validation**: Check incident dashboard for baseline activity

## Data Collection & ML Training Phase

### Baseline Data Generation (1-2 weeks)
1. **Natural Activity Monitoring**: Let systems run normal operations
2. **Expected Events**:
   - Windows authentication events (DC-01)
   - File access logs (FS-01)
   - Web server access logs (WEB-01)
   - Database query logs (DB-01)
   - User login sessions (WK-01, WK-02)

### ML Model Enhancement
1. **Trigger Retraining**: Use `/api/ml/retrain` endpoint
2. **Expected Improvements**:
   - Better anomaly detection for Windows domain environment
   - Enhanced recognition of legitimate database activity
   - Improved web application attack detection
   - More accurate behavioral profiling

### Testing Scenarios
1. **Controlled Attacks**: Simulate attacks using honeypot and known-bad traffic
2. **Performance Testing**: Generate high-volume legitimate traffic
3. **Failover Testing**: Test agent redundancy and recovery
4. **Multi-System Correlation**: Verify cross-system threat detection

## Performance Metrics & Success Criteria

### Detection Coverage
- **Target**: 95% of systems with active agent coverage
- **Expected**: 7/7 systems with comprehensive agent deployment
- **Validation**: Check `/api/telemetry/status` for agent counts

### Event Ingestion Rate
- **Target**: 100+ events/hour baseline, 1000+ events/hour under load
- **Expected**: Steady stream of authentication, file, and network events
- **Validation**: Monitor `/api/telemetry/status` metrics

### Incident Detection Accuracy
- **Target**: <5% false positive rate, >90% true positive rate
- **Expected**: Clean baseline with minimal false incidents
- **Validation**: Review incident dashboard and AI analysis quality

### Response Capability
- **Target**: <30 second response time for containment actions
- **Expected**: Automated responses working within SLA
- **Validation**: Execute test containment actions

## Next Steps After Deployment

1. **Complete Onboarding**: Run through full Mini-XDR onboarding wizard
2. **Agent Deployment**: Install and configure agents on all detected systems
3. **Data Collection**: Allow systems to run for 1-2 weeks to generate baseline data
4. **ML Training**: Trigger ML model retraining with real network data
5. **Testing Scenarios**: Execute controlled security tests (penetration testing, attack simulation)
6. **Performance Validation**: Assess Mini-XDR detection and response capabilities
7. **Documentation**: Update findings in `docs/change-control/audit-log.md`

## Implementation Scripts & Automation

### Corporate Network Deployment Script

```bash
#!/bin/bash
# deploy-mini-corp-network.sh
# Comprehensive deployment script for mini corporate network

set -e

# Configuration
CORP_VPC_CIDR="10.100.0.0/16"
CORP_PUBLIC_CIDR="10.100.1.0/24"
CORP_PRIVATE_CIDR="10.100.10.0/24"
YOUR_IP="${YOUR_IP:-$(curl -s ifconfig.me)/32}"
MINIXDR_VPC_ID="vpc-xxxxxxxx"  # Replace with actual Mini-XDR VPC ID

echo "🚀 Deploying Mini Corporate Network for Mini-XDR Testing"
echo "======================================================"

# 1. Create Corporate VPC
echo "📡 Creating corporate VPC..."
CORP_VPC_ID=$(aws ec2 create-vpc \
  --cidr-block $CORP_VPC_CIDR \
  --query 'Vpc.VpcId' \
  --output text)

aws ec2 create-tags \
  --resources $CORP_VPC_ID \
  --tags Key=Name,Value=mini-corp-vpc

# 2. Create subnets
echo "🏗️  Creating subnets..."
CORP_PUBLIC_SUBNET=$(aws ec2 create-subnet \
  --vpc-id $CORP_VPC_ID \
  --cidr-block $CORP_PUBLIC_CIDR \
  --availability-zone us-east-1a \
  --query 'Subnet.SubnetId' \
  --output text)

CORP_PRIVATE_SUBNET=$(aws ec2 create-subnet \
  --vpc-id $CORP_VPC_ID \
  --cidr-block $CORP_PRIVATE_CIDR \
  --availability-zone us-east-1a \
  --query 'Subnet.SubnetId' \
  --output text)

# 3. Create security group
echo "🔒 Creating security group..."
CORP_SG_ID=$(aws ec2 create-security-group \
  --group-name mini-corp-sg \
  --description "Mini corporate network security group" \
  --vpc-id $CORP_VPC_ID \
  --query 'GroupId' \
  --output text)

# Add your IP restrictions
aws ec2 authorize-security-group-ingress \
  --group-id $CORP_SG_ID \
  --protocol tcp \
  --port 22 \
  --cidr $YOUR_IP

aws ec2 authorize-security-group-ingress \
  --group-id $CORP_SG_ID \
  --protocol tcp \
  --port 3389 \
  --cidr $YOUR_IP

# Allow Mini-XDR VPC access for network scanning
aws ec2 authorize-security-group-ingress \
  --group-id $CORP_SG_ID \
  --protocol icmp \
  --source-group sg-xxxxxxxx  # Mini-XDR EKS node security group

# 4. Create VPC peering (requires Mini-XDR VPC ID)
echo "🔗 Creating VPC peering..."
PEERING_ID=$(aws ec2 create-vpc-peering-connection \
  --vpc-id $MINIXDR_VPC_ID \
  --peer-vpc-id $CORP_VPC_ID \
  --query 'VpcPeeringConnection.VpcPeeringConnectionId' \
  --output text)

echo "✅ Mini corporate network infrastructure deployed!"
echo "VPC ID: $CORP_VPC_ID"
echo "Public Subnet: $CORP_PUBLIC_SUBNET"
echo "Private Subnet: $CORP_PRIVATE_SUBNET"
echo "Security Group: $CORP_SG_ID"
echo "VPC Peering: $PEERING_ID"
echo ""
echo "📋 Next steps:"
echo "1. Accept VPC peering connection in AWS console"
echo "2. Update route tables for cross-VPC routing"
echo "3. Deploy bastion host and corporate systems"
echo "4. Run Mini-XDR onboarding wizard"
```

### System Configuration Scripts

#### Windows Domain Controller Setup
```powershell
# setup-dc.ps1 - Run on DC-01 after deployment
param(
    [string]$DomainName = "corp.local",
    [string]$AdminPassword = "ChangeMe123!"
)

# Install AD DS
Install-WindowsFeature -Name AD-Domain-Services -IncludeManagementTools

# Promote to domain controller
$SecurePassword = ConvertTo-SecureString $AdminPassword -AsPlainText -Force
Install-ADDSForest `
    -DomainName $DomainName `
    -DomainNetbiosName "CORP" `
    -SafeModeAdministratorPassword $SecurePassword `
    -InstallDns:$true `
    -Force:$true

# Configure DNS
Set-DnsServerForwarder -IPAddress "8.8.8.8", "1.1.1.1"

# Create test users
New-ADUser -Name "TestUser1" -SamAccountName "testuser1" -UserPrincipalName "testuser1@corp.local" -Enabled $true -AccountPassword $SecurePassword
New-ADUser -Name "TestUser2" -SamAccountName "testuser2" -UserPrincipalName "testuser2@corp.local" -Enabled $true -AccountPassword $SecurePassword

Write-Host "Domain controller setup complete. Reboot required."
```

#### Linux Web Server Setup
```bash
#!/bin/bash
# setup-web-server.sh - Run on WEB-01

# Install Apache
sudo apt update
sudo apt install -y apache2

# Configure basic site
sudo mkdir -p /var/www/html/test
cat << 'EOF' | sudo tee /var/www/html/index.html
<!DOCTYPE html>
<html>
<head><title>Mini Corp Web Server</title></head>
<body>
<h1>Mini Corporate Web Server</h1>
<p>This is a test web application for Mini-XDR monitoring.</p>
</body>
</html>
EOF

# Enable SSL
sudo a2enmod ssl
sudo systemctl restart apache2

# Configure logging
sudo a2enmod log_forensic
sudo systemctl restart apache2

echo "Web server setup complete"
```

#### Linux Database Server Setup
```bash
#!/bin/bash
# setup-db-server.sh - Run on DB-01

# Install PostgreSQL and MySQL
sudo apt update
sudo apt install -y postgresql postgresql-contrib mysql-server

# Configure PostgreSQL
sudo -u postgres createdb testdb
sudo -u postgres psql -c "CREATE USER testuser WITH PASSWORD 'testpass';"
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE testdb TO testuser;"

# Configure MySQL
sudo mysql -e "CREATE DATABASE testdb;"
sudo mysql -e "CREATE USER 'testuser'@'localhost' IDENTIFIED BY 'testpass';"
sudo mysql -e "GRANT ALL PRIVILEGES ON testdb.* TO 'testuser'@'localhost';"

# Enable query logging
# PostgreSQL
echo "log_statement = 'all'" | sudo tee -a /etc/postgresql/12/main/postgresql.conf
sudo systemctl restart postgresql

# MySQL
echo "[mysqld]" | sudo tee -a /etc/mysql/mysql.conf.d/mysqld.cnf
echo "general_log = 1" | sudo tee -a /etc/mysql/mysql.conf.d/mysqld.cnf
echo "general_log_file = /var/log/mysql/mysql.log" | sudo tee -a /etc/mysql/mysql.conf.d/mysqld.cnf
sudo systemctl restart mysql

echo "Database server setup complete"
```

### Monitoring & Validation Scripts

#### Network Discovery Test
```bash
#!/bin/bash
# test-network-discovery.sh
# Test if Mini-XDR can discover corporate network systems

CORP_RANGE="10.100.10.0/24"
MINIXDR_BACKEND="http://k8s-minixdr-minixdri-dc5fc1df8b-1132128475.us-east-1.elb.amazonaws.com"

echo "🕵️  Testing Mini-XDR network discovery..."

# Test basic connectivity
echo "Testing ICMP connectivity..."
for ip in {1..7}; do
    if ping -c 1 -W 2 10.100.10.$ip &>/dev/null; then
        echo "✅ 10.100.10.$ip is reachable"
    else
        echo "❌ 10.100.10.$ip is not reachable"
    fi
done

# Test port scanning
echo "Testing port scanning..."
nmap -p 22,80,443,445,3389 -T4 $CORP_RANGE

echo "Network discovery test complete"
```

#### Agent Deployment Verification
```bash
#!/bin/bash
# verify-agent-deployment.sh
# Verify all agents are properly deployed and communicating

MINIXDR_BACKEND="http://k8s-minixdr-minixdri-dc5fc1df8b-1132128475.us-east-1.elb.amazonaws.com"
API_KEY="your-api-key-here"

echo "🔍 Verifying agent deployment..."

# Check enrolled agents
curl -H "x-api-key: $API_KEY" \
     "$MINIXDR_BACKEND/api/onboarding/enrolled-agents" | jq .

# Run validation checks
curl -X POST \
     -H "x-api-key: $API_KEY" \
     "$MINIXDR_BACKEND/api/onboarding/validation" | jq .

# Check telemetry
curl -H "x-api-key: $API_KEY" \
     "$MINIXDR_BACKEND/api/telemetry/status" | jq .

echo "Agent verification complete"
```

## Cost Optimization Strategy

### Scheduled Operations
```bash
# Stop instances outside business hours (Monday-Friday, 9 AM - 6 PM EST)
aws ec2 stop-instances --instance-ids $INSTANCE_IDS

# Start instances for testing
aws ec2 start-instances --instance-ids $INSTANCE_IDS
```

### Instance Scheduler
Use AWS Instance Scheduler to automatically start/stop instances:
```bash
# Create schedule for business hours only
aws scheduler create-schedule \
  --name mini-corp-business-hours \
  --schedule-expression "cron(0 9 ? * MON-FRI *)" \
  --target '{"Arn": "arn:aws:lambda:us-east-1:123456789012:function:StartInstances", "Input": "{\"InstanceIds\": [\"i-1234567890abcdef0\"]}"}'
```

## Support Resources

- **Mini-XDR Documentation**: `docs/` directory
- **Onboarding Guide**: `docs/getting-started/local-quickstart.md`
- **API Reference**: `docs/api/reference.md`
- **Deployment Guide**: `docs/deployment/aws/`
- **Troubleshooting**: `docs/deployment/aws/troubleshooting.md`

---

**Created**: October 24, 2025
**Mini-XDR Version**: v1.1.0
**Target Environment**: AWS us-east-1
**Total Systems**: 7 (1 bastion + 6 corporate systems)
**Expected Cost**: ~$0.20/hour during testing
**Network Range**: 10.100.10.0/24 (private subnet)
