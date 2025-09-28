#!/bin/bash

# Mini-XDR AWS Deployment Script
# Deploys the complete Mini-XDR backend to AWS

set -euo pipefail

# Configuration
REGION="${AWS_REGION:-us-east-1}"
INSTANCE_TYPE="${INSTANCE_TYPE:-t3.medium}"
KEY_NAME="${KEY_NAME:-mini-xdr-tpot-key}"
YOUR_IP="${YOUR_IP:-24.11.0.176}"
PROJECT_NAME="mini-xdr"
STACK_NAME="mini-xdr-backend"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[WARN] $1${NC}"
}

error() {
    echo -e "${RED}[ERROR] $1${NC}"
    exit 1
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    if ! command -v aws &> /dev/null; then
        error "AWS CLI not found. Please install it first."
    fi
    
    if ! aws sts get-caller-identity &> /dev/null; then
        error "AWS CLI not configured. Please run 'aws configure' first."
    fi
    
    if ! aws ec2 describe-key-pairs --key-names "$KEY_NAME" &> /dev/null; then
        error "Key pair '$KEY_NAME' not found. Please create it first."
    fi
    
    log "Prerequisites check passed!"
}

# Get latest Ubuntu AMI
get_ubuntu_ami() {
    log "Getting latest Ubuntu 22.04 LTS AMI..."
    aws ec2 describe-images \
        --owners 099720109477 \
        --filters "Name=name,Values=ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*" \
        --query 'Images | sort_by(@, &CreationDate) | [-1].ImageId' \
        --output text
}

# Create CloudFormation template
create_cfn_template() {
    local ubuntu_ami="$1"
    
    log "Creating CloudFormation template..."
    cat > "/tmp/mini-xdr-backend.yaml" << EOF
AWSTemplateFormatVersion: '2010-09-09'
Description: 'Mini-XDR Backend Infrastructure'

Parameters:
  KeyPairName:
    Type: String
    Default: ${KEY_NAME}
    Description: EC2 Key Pair for SSH access
    
  YourPublicIP:
    Type: String
    Default: ${YOUR_IP}
    Description: Your public IP address for access control
    
  InstanceType:
    Type: String
    Default: ${INSTANCE_TYPE}
    Description: EC2 instance type for Mini-XDR backend

Resources:
  # VPC and Networking
  VPC:
    Type: AWS::EC2::VPC
    Properties:
      CidrBlock: 10.0.0.0/16
      EnableDnsHostnames: true
      EnableDnsSupport: true
      Tags:
        - Key: Name
          Value: mini-xdr-vpc
        - Key: Project
          Value: ${PROJECT_NAME}

  PublicSubnet:
    Type: AWS::EC2::Subnet
    Properties:
      VpcId: !Ref VPC
      CidrBlock: 10.0.1.0/24
      AvailabilityZone: !Select [0, !GetAZs '']
      MapPublicIpOnLaunch: true
      Tags:
        - Key: Name
          Value: mini-xdr-public-subnet
        - Key: Project
          Value: ${PROJECT_NAME}

  PrivateSubnet:
    Type: AWS::EC2::Subnet
    Properties:
      VpcId: !Ref VPC
      CidrBlock: 10.0.2.0/24
      AvailabilityZone: !Select [1, !GetAZs '']
      Tags:
        - Key: Name
          Value: mini-xdr-private-subnet
        - Key: Project
          Value: ${PROJECT_NAME}

  InternetGateway:
    Type: AWS::EC2::InternetGateway
    Properties:
      Tags:
        - Key: Name
          Value: mini-xdr-igw
        - Key: Project
          Value: ${PROJECT_NAME}

  AttachGateway:
    Type: AWS::EC2::VPCGatewayAttachment
    Properties:
      VpcId: !Ref VPC
      InternetGatewayId: !Ref InternetGateway

  PublicRouteTable:
    Type: AWS::EC2::RouteTable
    Properties:
      VpcId: !Ref VPC
      Tags:
        - Key: Name
          Value: mini-xdr-public-rt
        - Key: Project
          Value: ${PROJECT_NAME}

  PublicRoute:
    Type: AWS::EC2::Route
    DependsOn: AttachGateway
    Properties:
      RouteTableId: !Ref PublicRouteTable
      DestinationCidrBlock: 0.0.0.0/0
      GatewayId: !Ref InternetGateway

  PublicSubnetRouteTableAssociation:
    Type: AWS::EC2::SubnetRouteTableAssociation
    Properties:
      SubnetId: !Ref PublicSubnet
      RouteTableId: !Ref PublicRouteTable

  # Security Groups
  BackendSecurityGroup:
    Type: AWS::EC2::SecurityGroup
    Properties:
      GroupDescription: Security group for Mini-XDR backend
      VpcId: !Ref VPC
      SecurityGroupIngress:
        # SSH access from your IP
        - IpProtocol: tcp
          FromPort: 22
          ToPort: 22
          CidrIp: !Sub "\${YourPublicIP}/32"
          Description: SSH access
        # API access from your IP
        - IpProtocol: tcp
          FromPort: 8000
          ToPort: 8000
          CidrIp: !Sub "\${YourPublicIP}/32"
          Description: Mini-XDR API
        # TPOT honeypot access
        - IpProtocol: tcp
          FromPort: 8000
          ToPort: 8000
          CidrIp: 172.31.0.0/16
          Description: TPOT data ingestion
        # Internal VPC access
        - IpProtocol: tcp
          FromPort: 8000
          ToPort: 8000
          CidrIp: 10.0.0.0/16
          Description: Internal VPC access
      Tags:
        - Key: Name
          Value: mini-xdr-backend-sg
        - Key: Project
          Value: ${PROJECT_NAME}

  DatabaseSecurityGroup:
    Type: AWS::EC2::SecurityGroup
    Properties:
      GroupDescription: Security group for Mini-XDR database
      VpcId: !Ref VPC
      SecurityGroupIngress:
        - IpProtocol: tcp
          FromPort: 5432
          ToPort: 5432
          SourceSecurityGroupId: !Ref BackendSecurityGroup
          Description: PostgreSQL access from backend
      Tags:
        - Key: Name
          Value: mini-xdr-db-sg
        - Key: Project
          Value: ${PROJECT_NAME}

  # Database Subnet Group
  DBSubnetGroup:
    Type: AWS::RDS::DBSubnetGroup
    Properties:
      DBSubnetGroupDescription: Subnet group for Mini-XDR database
      SubnetIds:
        - !Ref PublicSubnet
        - !Ref PrivateSubnet
      Tags:
        - Key: Name
          Value: mini-xdr-db-subnet-group
        - Key: Project
          Value: ${PROJECT_NAME}

  # RDS PostgreSQL Database
  Database:
    Type: AWS::RDS::DBInstance
    DeletionPolicy: Snapshot
    Properties:
      DBInstanceIdentifier: mini-xdr-db
      DBInstanceClass: db.t3.micro
      Engine: postgres
      EngineVersion: "15.4"
      MasterUsername: postgres
      MasterUserPassword: !Sub "minixdr\${AWS::StackId}"
      AllocatedStorage: 20
      StorageType: gp2
      StorageEncrypted: true
      VPCSecurityGroups:
        - !Ref DatabaseSecurityGroup
      DBSubnetGroupName: !Ref DBSubnetGroup
      BackupRetentionPeriod: 7
      MultiAZ: false
      PubliclyAccessible: false
      Tags:
        - Key: Name
          Value: mini-xdr-database
        - Key: Project
          Value: ${PROJECT_NAME}

  # IAM Role for EC2 Instance
  EC2Role:
    Type: AWS::IAM::Role
    Properties:
      AssumeRolePolicyDocument:
        Version: '2012-10-17'
        Statement:
          - Effect: Allow
            Principal:
              Service: ec2.amazonaws.com
            Action: sts:AssumeRole
      ManagedPolicyArns:
        - arn:aws:iam::aws:policy/CloudWatchAgentServerPolicy
      Policies:
        - PolicyName: S3ModelAccess
          PolicyDocument:
            Version: '2012-10-17'
            Statement:
              - Effect: Allow
                Action:
                  - s3:GetObject
                  - s3:PutObject
                  - s3:DeleteObject
                  - s3:ListBucket
                Resource:
                  - !Sub "\${ModelsBucket}/*"
                  - !Ref ModelsBucket
      Tags:
        - Key: Project
          Value: ${PROJECT_NAME}

  EC2InstanceProfile:
    Type: AWS::IAM::InstanceProfile
    Properties:
      Roles:
        - !Ref EC2Role

  # S3 Bucket for ML Models
  ModelsBucket:
    Type: AWS::S3::Bucket
    Properties:
      BucketName: !Sub "mini-xdr-models-\${AWS::AccountId}-\${AWS::Region}"
      VersioningConfiguration:
        Status: Enabled
      PublicAccessBlockConfiguration:
        BlockPublicAcls: true
        BlockPublicPolicy: true
        IgnorePublicAcls: true
        RestrictPublicBuckets: true
      Tags:
        - Key: Name
          Value: mini-xdr-models
        - Key: Project
          Value: ${PROJECT_NAME}

  # EC2 Instance for Mini-XDR Backend
  BackendInstance:
    Type: AWS::EC2::Instance
    Properties:
      ImageId: ${ubuntu_ami}
      InstanceType: !Ref InstanceType
      KeyName: !Ref KeyPairName
      SecurityGroupIds:
        - !Ref BackendSecurityGroup
      SubnetId: !Ref PublicSubnet
      IamInstanceProfile: !Ref EC2InstanceProfile
      UserData:
        Fn::Base64: !Sub |
          #!/bin/bash
          set -euo pipefail
          
          # Update system
          apt-get update -y
          apt-get upgrade -y
          
          # Install dependencies
          apt-get install -y python3 python3-pip python3-venv git curl wget htop vim \
                             docker.io docker-compose nginx postgresql-client-15 \
                             build-essential libssl-dev libffi-dev python3-dev \
                             pkg-config libhdf5-dev
          
          # Configure Docker
          systemctl enable docker
          systemctl start docker
          usermod -aG docker ubuntu
          
          # Install Python 3.12 (required for ML dependencies)
          add-apt-repository ppa:deadsnakes/ppa -y
          apt-get update -y
          apt-get install -y python3.12 python3.12-venv python3.12-dev
          
          # Clone Mini-XDR repository (placeholder - will copy from local)
          mkdir -p /opt/mini-xdr
          chown ubuntu:ubuntu /opt/mini-xdr
          
          # Create virtual environment
          su - ubuntu -c "cd /opt/mini-xdr && python3.12 -m venv venv"
          
          # Create environment file template
          cat > /opt/mini-xdr/.env << 'ENVEOF'
          # API Configuration
          API_HOST=0.0.0.0
          API_PORT=8000
          UI_ORIGIN=http://localhost:3000,http://${YourPublicIP}:3000
          API_KEY=WILL_BE_GENERATED_DURING_DEPLOYMENT
          
          # Database
          DATABASE_URL=postgresql://postgres:minixdr\${AWS::StackId}@\${Database.Endpoint.Address}:5432/postgres
          
          # Detection Configuration
          FAIL_WINDOW_SECONDS=60
          FAIL_THRESHOLD=6
          AUTO_CONTAIN=false
          
          # Honeypot Configuration - TPOT
          HONEYPOT_HOST=34.193.101.171
          HONEYPOT_USER=admin
          HONEYPOT_SSH_KEY=/home/ubuntu/.ssh/mini-xdr-tpot-key.pem
          HONEYPOT_SSH_PORT=64295
          
          # LLM Configuration
          LLM_PROVIDER=openai
          OPENAI_API_KEY=YOUR_OPENAI_API_KEY_HERE
          OPENAI_MODEL=gpt-4o-mini
          
          # ML Models Path
          ML_MODELS_PATH=/opt/mini-xdr/models
          POLICIES_PATH=/opt/mini-xdr/policies
          
          # S3 Configuration
          MODELS_BUCKET=\${ModelsBucket}
          AWS_REGION=\${AWS::Region}
          ENVEOF
          
          chown ubuntu:ubuntu /opt/mini-xdr/.env
          
          # Configure nginx reverse proxy
          cat > /etc/nginx/sites-available/mini-xdr << 'NGINXEOF'
          server {
              listen 80;
              server_name _;
              
              location / {
                  proxy_pass http://127.0.0.1:8000;
                  proxy_set_header Host \$host;
                  proxy_set_header X-Real-IP \$remote_addr;
                  proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
                  proxy_set_header X-Forwarded-Proto \$scheme;
              }
              
              # Health check endpoint
              location /health {
                  proxy_pass http://127.0.0.1:8000/health;
              }
          }
          NGINXEOF
          
          ln -s /etc/nginx/sites-available/mini-xdr /etc/nginx/sites-enabled/
          rm -f /etc/nginx/sites-enabled/default
          systemctl enable nginx
          systemctl restart nginx
          
          # Create systemd service for Mini-XDR
          cat > /etc/systemd/system/mini-xdr.service << 'SERVICEEOF'
          [Unit]
          Description=Mini-XDR Backend
          After=network.target postgresql.service
          
          [Service]
          Type=exec
          User=ubuntu
          Group=ubuntu
          WorkingDirectory=/opt/mini-xdr/backend
          Environment=PATH=/opt/mini-xdr/venv/bin
          EnvironmentFile=/opt/mini-xdr/.env
          ExecStart=/opt/mini-xdr/venv/bin/uvicorn app.entrypoint:app --host 127.0.0.1 --port 8000 --workers 1
          Restart=always
          RestartSec=3
          
          [Install]
          WantedBy=multi-user.target
          SERVICEEOF
          
          systemctl daemon-reload
          systemctl enable mini-xdr
          
          # Install CloudWatch agent
          wget https://s3.amazonaws.com/amazoncloudwatch-agent/ubuntu/amd64/latest/amazon-cloudwatch-agent.deb
          dpkg -i amazon-cloudwatch-agent.deb
          
          echo "Mini-XDR backend instance setup complete!" > /var/log/mini-xdr-setup.log
          
      Tags:
        - Key: Name
          Value: mini-xdr-backend
        - Key: Project
          Value: ${PROJECT_NAME}

  # Elastic IP for consistent addressing
  BackendEIP:
    Type: AWS::EC2::EIP
    Properties:
      Domain: vpc
      InstanceId: !Ref BackendInstance
      Tags:
        - Key: Name
          Value: mini-xdr-backend-eip
        - Key: Project
          Value: ${PROJECT_NAME}

Outputs:
  BackendPublicIP:
    Description: Public IP of the Mini-XDR backend
    Value: !Ref BackendEIP
    Export:
      Name: !Sub "\${AWS::StackName}-PublicIP"
      
  BackendInstanceId:
    Description: Instance ID of the Mini-XDR backend
    Value: !Ref BackendInstance
    Export:
      Name: !Sub "\${AWS::StackName}-InstanceId"
      
  DatabaseEndpoint:
    Description: RDS database endpoint
    Value: !GetAtt Database.Endpoint.Address
    Export:
      Name: !Sub "\${AWS::StackName}-DatabaseEndpoint"
      
  ModelsBucket:
    Description: S3 bucket for ML models
    Value: !Ref ModelsBucket
    Export:
      Name: !Sub "\${AWS::StackName}-ModelsBucket"
      
  SSHCommand:
    Description: SSH command to connect to backend
    Value: !Sub "ssh -i ~/.ssh/\${KeyPairName}.pem ubuntu@\${BackendEIP}"
    
  APIEndpoint:
    Description: Mini-XDR API endpoint
    Value: !Sub "http://\${BackendEIP}:8000"

EOF
}

# Deploy CloudFormation stack
deploy_stack() {
    log "Deploying CloudFormation stack..."
    
    aws cloudformation deploy \
        --template-file "/tmp/mini-xdr-backend.yaml" \
        --stack-name "$STACK_NAME" \
        --parameter-overrides \
            KeyPairName="$KEY_NAME" \
            YourPublicIP="$YOUR_IP" \
            InstanceType="$INSTANCE_TYPE" \
        --capabilities CAPABILITY_IAM \
        --region "$REGION" || error "Stack deployment failed"
    
    log "Stack deployment completed successfully!"
}

# Get stack outputs
get_outputs() {
    log "Retrieving stack outputs..."
    
    local outputs
    outputs=$(aws cloudformation describe-stacks \
        --stack-name "$STACK_NAME" \
        --region "$REGION" \
        --query 'Stacks[0].Outputs' \
        --output json)
    
    echo "$outputs" | jq -r '.[] | "\(.OutputKey): \(.OutputValue)"'
    
    # Store outputs for later use
    echo "$outputs" > "/tmp/mini-xdr-outputs.json"
}

# Main deployment function
main() {
    log "Starting Mini-XDR AWS deployment..."
    
    check_prerequisites
    
    local ubuntu_ami
    ubuntu_ami=$(get_ubuntu_ami)
    log "Using Ubuntu AMI: $ubuntu_ami"
    
    create_cfn_template "$ubuntu_ami"
    deploy_stack
    get_outputs
    
    log "✅ Mini-XDR AWS infrastructure deployment completed!"
    log ""
    log "Next steps:"
    log "1. Upload application code to EC2 instance"
    log "2. Configure environment variables"
    log "3. Start Mini-XDR services"
    log "4. Configure TPOT data forwarding"
    log ""
    log "Run './deploy-mini-xdr-code.sh' to complete the deployment."
}

# Run main function
main "$@"
EOF
