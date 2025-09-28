#!/bin/bash
# T-Pot Honeypot AWS Deployment Script for Mini-XDR
# Deploys T-Pot honeypot with comprehensive security configuration and auto-management

set -e

# Configuration
REGION="us-east-1"  # Default region
INSTANCE_TYPE="t3.xlarge"  # T-Pot requires minimum 8GB RAM, 128GB SSD
KEY_NAME="mini-xdr-tpot-key"
VPC_NAME="mini-xdr-tpot-vpc"
SUBNET_NAME="mini-xdr-tpot-subnet"
SECURITY_GROUP_NAME="mini-xdr-tpot-sg"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log() { echo -e "${BLUE}[$(date +'%H:%M:%S')]${NC} $1"; }
success() { echo -e "${GREEN}✅ $1${NC}"; }
warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }
error() { echo -e "${RED}❌ $1${NC}"; }

# Function to get your public IP automatically
get_your_ip() {
    log "Detecting your public IP address..."
    YOUR_IP=$(curl -s -4 icanhazip.com 2>/dev/null || curl -s -4 ifconfig.me 2>/dev/null || curl -s -4 ipecho.net/plain 2>/dev/null)
    
    if [ -z "$YOUR_IP" ]; then
        error "Could not detect your public IP address"
        echo "Please manually enter your public IP address:"
        read -p "Your public IP: " YOUR_IP
    fi
    
    success "Your public IP: $YOUR_IP"
}

# Function to check AWS CLI configuration
check_aws_config() {
    log "Checking AWS CLI configuration..."
    
    if ! command -v aws &> /dev/null; then
        error "AWS CLI not installed. Please install it first."
        exit 1
    fi
    
    if ! aws sts get-caller-identity &> /dev/null; then
        error "AWS CLI not configured or credentials invalid"
        echo "Please run: aws configure"
        exit 1
    fi
    
    local account_id=$(aws sts get-caller-identity --query 'Account' --output text)
    local user_arn=$(aws sts get-caller-identity --query 'Arn' --output text)
    
    success "AWS CLI configured"
    echo "   Account: $account_id"
    echo "   User: $user_arn"
    echo "   Region: $REGION"
}

# Function to create or use existing key pair
setup_key_pair() {
    log "Setting up EC2 key pair..."
    
    # Check if key already exists
    if aws ec2 describe-key-pairs --key-names $KEY_NAME --region $REGION &> /dev/null; then
        success "Key pair '$KEY_NAME' already exists"
        
        # Check if local private key file exists
        if [ ! -f ~/.ssh/${KEY_NAME}.pem ]; then
            warning "Private key file not found locally"
            echo "If you don't have the private key file, I'll create a new key pair."
            read -p "Do you want to create a new key pair? (y/n): " create_new
            
            if [[ $create_new =~ ^[Yy]$ ]]; then
                aws ec2 delete-key-pair --key-name $KEY_NAME --region $REGION
                create_new_key_pair
            else
                echo "Please ensure ~/.ssh/${KEY_NAME}.pem exists with correct permissions"
                exit 1
            fi
        else
            success "Private key file found: ~/.ssh/${KEY_NAME}.pem"
        fi
    else
        create_new_key_pair
    fi
}

# Function to create new key pair
create_new_key_pair() {
    log "Creating new EC2 key pair..."
    
    # Create key pair and save private key
    aws ec2 create-key-pair \
        --key-name $KEY_NAME \
        --region $REGION \
        --query 'KeyMaterial' \
        --output text > ~/.ssh/${KEY_NAME}.pem
    
    # Set correct permissions
    chmod 600 ~/.ssh/${KEY_NAME}.pem
    
    success "Key pair created: $KEY_NAME"
    echo "   Private key saved: ~/.ssh/${KEY_NAME}.pem"
}

# Function to get Debian AMI ID for the region (T-Pot requires Debian)
get_debian_ami() {
    log "Finding latest Debian 12 AMI..."
    
    AMI_ID=$(aws ec2 describe-images \
        --region $REGION \
        --owners 136693071363 \
        --filters "Name=name,Values=debian-12-amd64-*" \
        --query 'Images[*].[ImageId,CreationDate]' \
        --output text | sort -k2 -r | head -n1 | cut -f1)
    
    if [ -z "$AMI_ID" ]; then
        error "Could not find Debian AMI"
        exit 1
    fi
    
    success "Debian AMI: $AMI_ID"
}

# Function to create VPC infrastructure
create_vpc_infrastructure() {
    log "Creating VPC infrastructure for T-Pot..."
    
    # Create VPC
    VPC_ID=$(aws ec2 create-vpc \
        --region $REGION \
        --cidr-block 10.0.0.0/16 \
        --tag-specifications "ResourceType=vpc,Tags=[{Key=Name,Value=$VPC_NAME}]" \
        --query 'Vpc.VpcId' \
        --output text)
    
    # Enable DNS hostnames
    aws ec2 modify-vpc-attribute \
        --region $REGION \
        --vpc-id $VPC_ID \
        --enable-dns-hostnames
    
    success "VPC created: $VPC_ID"
    
    # Create Internet Gateway
    IGW_ID=$(aws ec2 create-internet-gateway \
        --region $REGION \
        --tag-specifications "ResourceType=internet-gateway,Tags=[{Key=Name,Value=$VPC_NAME-igw}]" \
        --query 'InternetGateway.InternetGatewayId' \
        --output text)
    
    aws ec2 attach-internet-gateway \
        --region $REGION \
        --internet-gateway-id $IGW_ID \
        --vpc-id $VPC_ID
    
    success "Internet Gateway created: $IGW_ID"
    
    # Create Subnet
    SUBNET_ID=$(aws ec2 create-subnet \
        --region $REGION \
        --vpc-id $VPC_ID \
        --cidr-block 10.0.1.0/24 \
        --availability-zone ${REGION}a \
        --tag-specifications "ResourceType=subnet,Tags=[{Key=Name,Value=$SUBNET_NAME}]" \
        --query 'Subnet.SubnetId' \
        --output text)
    
    # Enable auto-assign public IP
    aws ec2 modify-subnet-attribute \
        --region $REGION \
        --subnet-id $SUBNET_ID \
        --map-public-ip-on-launch
    
    success "Subnet created: $SUBNET_ID"
    
    # Create Route Table
    ROUTE_TABLE_ID=$(aws ec2 create-route-table \
        --region $REGION \
        --vpc-id $VPC_ID \
        --tag-specifications "ResourceType=route-table,Tags=[{Key=Name,Value=$VPC_NAME-rt}]" \
        --query 'RouteTable.RouteTableId' \
        --output text)
    
    # Add route to internet gateway
    aws ec2 create-route \
        --region $REGION \
        --route-table-id $ROUTE_TABLE_ID \
        --destination-cidr-block 0.0.0.0/0 \
        --gateway-id $IGW_ID
    
    # Associate with subnet
    aws ec2 associate-route-table \
        --region $REGION \
        --subnet-id $SUBNET_ID \
        --route-table-id $ROUTE_TABLE_ID
    
    success "Route table created and configured: $ROUTE_TABLE_ID"
}

# Function to create security group for T-Pot
create_security_group() {
    log "Creating security group for T-Pot..."
    
    SG_ID=$(aws ec2 create-security-group \
        --region $REGION \
        --group-name $SECURITY_GROUP_NAME \
        --description "Security group for T-Pot honeypot with management access" \
        --vpc-id $VPC_ID \
        --tag-specifications "ResourceType=security-group,Tags=[{Key=Name,Value=$SECURITY_GROUP_NAME}]" \
        --query 'GroupId' \
        --output text)
    
    success "Security group created: $SG_ID"
    
    # Add rules for T-Pot management (restricted to your IP)
    log "Adding T-Pot management security rules..."
    
    # SSH Management (port 64295) - restricted to your IP
    aws ec2 authorize-security-group-ingress \
        --region $REGION \
        --group-id $SG_ID \
        --protocol tcp \
        --port 64295 \
        --cidr "$YOUR_IP/32" \
        --output text 2>/dev/null || true
    
    # T-Pot Web Interface (port 64297) - restricted to your IP
    aws ec2 authorize-security-group-ingress \
        --region $REGION \
        --group-id $SG_ID \
        --protocol tcp \
        --port 64297 \
        --cidr "$YOUR_IP/32" \
        --output text 2>/dev/null || true
    
    # Add rules for honeypot services (open to internet for threat collection)
    log "Adding honeypot service security rules..."
    
    # Common honeypot ports - open to internet
    honeypot_ports=(21 22 23 25 53 80 135 139 443 445 993 995 1433 1521 3306 3389 5432 5900 6379 8080 8443 9200 27017)
    
    for port in "${honeypot_ports[@]}"; do
        aws ec2 authorize-security-group-ingress \
            --region $REGION \
            --group-id $SG_ID \
            --protocol tcp \
            --port $port \
            --cidr 0.0.0.0/0 \
            --output text 2>/dev/null || true
    done
    
    # UDP ports for DNS and other services
    udp_ports=(53 123 161 500 1434 1900)
    for port in "${udp_ports[@]}"; do
        aws ec2 authorize-security-group-ingress \
            --region $REGION \
            --group-id $SG_ID \
            --protocol udp \
            --port $port \
            --cidr 0.0.0.0/0 \
            --output text 2>/dev/null || true
    done
    
    # Wide port range for additional honeypot services
    aws ec2 authorize-security-group-ingress \
        --region $REGION \
        --group-id $SG_ID \
        --protocol tcp \
        --port 8000-9999 \
        --cidr 0.0.0.0/0 \
        --output text 2>/dev/null || true
    
    success "Security group configured with T-Pot and honeypot rules"
}

# Function to create T-Pot user data script
create_tpot_user_data() {
    cat << EOF
#!/bin/bash
set -e

# T-Pot Installation Script for AWS EC2
echo "=== Starting T-Pot Honeypot Setup ==="

# Update system
apt-get update -y
apt-get upgrade -y

# Install essential dependencies
apt-get install -y git curl wget htop vim net-tools jq build-essential

# Configure timezone
timedatectl set-timezone UTC

# Set hostname
echo "tpot-honeypot" > /etc/hostname
hostnamectl set-hostname tpot-honeypot
echo "127.0.0.1 tpot-honeypot" >> /etc/hosts

# Create T-Pot installation directory
mkdir -p /opt/tpot
cd /opt/tpot

# Clone T-Pot repository
git clone https://github.com/telekom-security/tpotce.git
cd tpotce

# Create automated installation configuration
cat > /tmp/tpot-install.conf << 'TPOTCONF'
# T-Pot Automated Installation Configuration
TPOT_USER=tpot
TPOT_PASSWORD=\$(openssl rand -base64 32 | tr -d "=+/" | cut -c1-16)
TPOT_EDITION=STANDARD
TPOT_WEB_USER=admin
TPOT_WEB_PASSWORD=\$(openssl rand -base64 32 | tr -d "=+/" | cut -c1-16)
TPOT_SSH_PORT=64295
TPOT_WEB_PORT=64297
TPOTCONF

# Save credentials for later reference
echo "T-Pot Credentials:" > /root/tpot-credentials.txt
echo "SSH User: tpot" >> /root/tpot-credentials.txt
echo "SSH Password: \$(grep TPOT_PASSWORD /tmp/tpot-install.conf | cut -d'=' -f2)" >> /root/tpot-credentials.txt
echo "Web User: admin" >> /root/tpot-credentials.txt
echo "Web Password: \$(grep TPOT_WEB_PASSWORD /tmp/tpot-install.conf | cut -d'=' -f2)" >> /root/tpot-credentials.txt
echo "SSH Port: 64295" >> /root/tpot-credentials.txt
echo "Web Port: 64297" >> /root/tpot-credentials.txt
chmod 600 /root/tpot-credentials.txt

# Create automated installation script
cat > /tmp/tpot-auto-install.sh << 'AUTOINSTALL'
#!/bin/bash
export DEBIAN_FRONTEND=noninteractive

# Source configuration
source /tmp/tpot-install.conf

# Run T-Pot installation with automated responses
echo "Running T-Pot installation..."
cd /opt/tpot/tpotce

# Create expect script for automated installation
cat > /tmp/install-expect.exp << 'EXPECTEOF'
#!/usr/bin/expect -f
set timeout 300

spawn sudo ./install.sh

# Select STANDARD edition
expect "Please select your T-Pot type"
send "1\r"

# Confirm STANDARD selection
expect "You selected"
send "y\r"

# Enter username
expect "Enter your username"
send "tpot\r"

# Enter password
expect "Enter your password"
send "\$env(TPOT_PASSWORD)\r"

# Confirm password
expect "Repeat your password"
send "\$env(TPOT_PASSWORD)\r"

# Wait for installation to complete
expect "Please reboot your system"
send "\r"

expect eof
EXPECTEOF

chmod +x /tmp/install-expect.exp
/tmp/install-expect.exp

AUTOINSTALL

chmod +x /tmp/tpot-auto-install.sh

# Install expect for automated installation
apt-get install -y expect

# Run T-Pot installation
/tmp/tpot-auto-install.sh

# Configure log forwarding to Mini-XDR
mkdir -p /opt/mini-xdr-integration

# Install Fluent Bit for log forwarding
curl https://raw.githubusercontent.com/fluent/fluent-bit/master/install.sh | sh

# Configure Fluent Bit to forward T-Pot logs
cat > /etc/fluent-bit/fluent-bit.conf << 'FLUENTEOF'
[SERVICE]
    Flush         5
    Log_Level     info
    Daemon        off
    Parsers_File  parsers.conf

[INPUT]
    Name              tail
    Path              /data/elk/logstash/logs/logstash-plain.log
    Tag               tpot.logstash
    Refresh_Interval  5
    Read_from_Head    false

[INPUT]
    Name              tail
    Path              /data/suricata/log/eve.json
    Parser            json
    Tag               tpot.suricata
    Refresh_Interval  5
    Read_from_Head    false

[INPUT]
    Name              tail
    Path              /data/cowrie/log/cowrie.json
    Parser            json
    Tag               tpot.cowrie
    Refresh_Interval  5
    Read_from_Head    false

[INPUT]
    Name              tail
    Path              /data/dionaea/log/dionaea.json
    Parser            json
    Tag               tpot.dionaea
    Refresh_Interval  5
    Read_from_Head    false

[OUTPUT]
    Name  http
    Match tpot.*
    Host  $YOUR_IP
    Port  8000
    URI   /ingest/multi
    Format json
    Header Authorization Bearer tpot-honeypot-key
    Header Content-Type application/json
    Retry_Limit 3
FLUENTEOF

# Create T-Pot management scripts
mkdir -p /opt/mini-xdr-tpot

# Create status script
cat > /opt/mini-xdr-tpot/tpot-status.sh << 'STATUSEOF'
#!/bin/bash
echo "=== T-Pot Honeypot Status ==="
echo "Timestamp: \$(date)"
echo ""
echo "T-Pot Services:"
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep -E "(cowrie|dionaea|suricata|elk|nginx)"
echo ""
echo "System Resources:"
echo "CPU Usage: \$(top -bn1 | grep "Cpu(s)" | awk '{print \$2}' | cut -d'%' -f1)%"
echo "Memory Usage: \$(free | grep Mem | awk '{printf("%.1f%%", \$3/\$2 * 100.0)}')"
echo "Disk Usage: \$(df -h / | awk 'NR==2{printf "%s", \$5}')"
echo ""
echo "Network Connections:"
netstat -tulnp | grep -E ":22|:80|:443|:64295|:64297" | head -10
echo ""
echo "Recent Attack Activity:"
if [ -f /data/suricata/log/eve.json ]; then
    echo "Suricata Alerts (last 5):"
    tail -5 /data/suricata/log/eve.json | jq -r '.timestamp + " " + .alert.signature' 2>/dev/null || echo "No recent alerts"
fi
STATUSEOF
chmod +x /opt/mini-xdr-tpot/tpot-status.sh

# Create restart script
cat > /opt/mini-xdr-tpot/tpot-restart.sh << 'RESTARTEOF'
#!/bin/bash
echo "=== Restarting T-Pot Services ==="
cd /opt/tpot/tpotce
sudo docker-compose down
sudo docker-compose up -d
echo "T-Pot services restarted"
RESTARTEOF
chmod +x /opt/mini-xdr-tpot/tpot-restart.sh

# Create stop script
cat > /opt/mini-xdr-tpot/tpot-stop.sh << 'STOPEOF'
#!/bin/bash
echo "=== Stopping T-Pot Services ==="
cd /opt/tpot/tpotce
sudo docker-compose down
echo "T-Pot services stopped"
STOPEOF
chmod +x /opt/mini-xdr-tpot/tpot-stop.sh

# Create start script
cat > /opt/mini-xdr-tpot/tpot-start.sh << 'STARTEOF'
#!/bin/bash
echo "=== Starting T-Pot Services ==="
cd /opt/tpot/tpotce
sudo docker-compose up -d
echo "T-Pot services started"
STARTEOF
chmod +x /opt/mini-xdr-tpot/tpot-start.sh

# Create systemd service for auto-start
cat > /etc/systemd/system/tpot-startup.service << 'SYSTEMDEOF'
[Unit]
Description=T-Pot Honeypot Startup Service
After=network-online.target docker.service
Wants=network-online.target
Requires=docker.service

[Service]
Type=oneshot
ExecStart=/opt/mini-xdr-tpot/tpot-start.sh
RemainAfterExit=true
StandardOutput=journal
StandardError=journal
TimeoutStartSec=600

[Install]
WantedBy=multi-user.target
SYSTEMDEOF

# Enable auto-start service
systemctl daemon-reload
systemctl enable tpot-startup

# Enable Fluent Bit for log forwarding
systemctl enable fluent-bit

# Create initial status file
/opt/mini-xdr-tpot/tpot-status.sh > /var/lib/tpot-status.txt

echo "=== T-Pot Installation Complete ==="
echo "The system will reboot automatically to complete setup"
echo "After reboot, T-Pot will be accessible at:"
echo "  SSH: ssh -p 64295 tpot@\$(curl -s icanhazip.com)"
echo "  Web: https://\$(curl -s icanhazip.com):64297/"
echo ""
echo "Credentials saved in /root/tpot-credentials.txt"

# Schedule reboot
shutdown -r +2 "T-Pot installation complete, rebooting in 2 minutes"
EOF
}

# Function to launch T-Pot instance
launch_tpot_instance() {
    log "Launching T-Pot honeypot instance..."
    
    # Create user data
    local user_data=$(create_tpot_user_data | sed "s/\$YOUR_IP/$YOUR_IP/g")
    
    INSTANCE_ID=$(aws ec2 run-instances \
        --region $REGION \
        --image-id $AMI_ID \
        --count 1 \
        --instance-type $INSTANCE_TYPE \
        --key-name $KEY_NAME \
        --security-group-ids $SG_ID \
        --subnet-id $SUBNET_ID \
        --block-device-mappings '[{"DeviceName":"/dev/xvda","Ebs":{"VolumeSize":128,"VolumeType":"gp3"}}]' \
        --user-data "$user_data" \
        --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=mini-xdr-tpot-honeypot},{Key=Project,Value=mini-xdr},{Key=Type,Value=tpot-honeypot}]" \
        --query 'Instances[0].InstanceId' \
        --output text)
    
    success "T-Pot instance launched: $INSTANCE_ID"
    
    # Wait for instance to be running
    log "Waiting for instance to be running..."
    aws ec2 wait instance-running --region $REGION --instance-ids $INSTANCE_ID
    
    # Allocate and associate Elastic IP
    log "Allocating Elastic IP..."
    local allocation_id=$(aws ec2 allocate-address \
        --region $REGION \
        --domain vpc \
        --tag-specifications "ResourceType=elastic-ip,Tags=[{Key=Name,Value=mini-xdr-tpot-ip},{Key=Project,Value=mini-xdr}]" \
        --query 'AllocationId' \
        --output text)
    
    aws ec2 associate-address \
        --region $REGION \
        --instance-id $INSTANCE_ID \
        --allocation-id $allocation_id \
        --output text > /dev/null
    
    PUBLIC_IP=$(aws ec2 describe-addresses \
        --region $REGION \
        --allocation-ids $allocation_id \
        --query 'Addresses[0].PublicIp' \
        --output text)
    
    success "T-Pot honeypot deployed successfully!"
    
    # Get instance details
    PRIVATE_IP=$(aws ec2 describe-instances \
        --region $REGION \
        --instance-ids $INSTANCE_ID \
        --query 'Reservations[0].Instances[0].PrivateIpAddress' \
        --output text)
}

# Function to create local management scripts
create_management_scripts() {
    log "Creating local T-Pot management scripts..."
    
    # Create local management directory
    mkdir -p ~/.mini-xdr/tpot-management
    
    # Create connection script
    cat > ~/.mini-xdr/tpot-management/connect.sh << EOF
#!/bin/bash
# Connect to T-Pot honeypot for management
echo "Connecting to T-Pot honeypot..."
echo "Note: Use 'tpot' user after T-Pot installation is complete"
ssh -i ~/.ssh/$KEY_NAME.pem -p 64295 admin@$PUBLIC_IP "\$@"
EOF
    chmod +x ~/.mini-xdr/tpot-management/connect.sh
    
    # Create status check script
    cat > ~/.mini-xdr/tpot-management/status.sh << EOF
#!/bin/bash
# Check T-Pot honeypot status remotely
echo "=== Remote T-Pot Status Check ==="
ssh -i ~/.ssh/$KEY_NAME.pem -p 64295 admin@$PUBLIC_IP 'sudo /opt/mini-xdr-tpot/tpot-status.sh' 2>/dev/null || \
ssh -i ~/.ssh/$KEY_NAME.pem -p 22 admin@$PUBLIC_IP 'sudo /opt/mini-xdr-tpot/tpot-status.sh'
EOF
    chmod +x ~/.mini-xdr/tpot-management/status.sh
    
    # Create restart script
    cat > ~/.mini-xdr/tpot-management/restart.sh << EOF
#!/bin/bash
# Restart T-Pot honeypot services remotely
echo "=== Restarting T-Pot Services ==="
ssh -i ~/.ssh/$KEY_NAME.pem -p 64295 admin@$PUBLIC_IP 'sudo /opt/mini-xdr-tpot/tpot-restart.sh' 2>/dev/null || \
ssh -i ~/.ssh/$KEY_NAME.pem -p 22 admin@$PUBLIC_IP 'sudo /opt/mini-xdr-tpot/tpot-restart.sh'
echo "Restart initiated. Check status in 60 seconds."
EOF
    chmod +x ~/.mini-xdr/tpot-management/restart.sh
    
    # Create credentials retrieval script
    cat > ~/.mini-xdr/tpot-management/get-credentials.sh << EOF
#!/bin/bash
# Get T-Pot credentials
echo "=== T-Pot Credentials ==="
ssh -i ~/.ssh/$KEY_NAME.pem -p 64295 admin@$PUBLIC_IP 'sudo cat /root/tpot-credentials.txt' 2>/dev/null || \
ssh -i ~/.ssh/$KEY_NAME.pem -p 22 admin@$PUBLIC_IP 'sudo cat /root/tpot-credentials.txt'
EOF
    chmod +x ~/.mini-xdr/tpot-management/get-credentials.sh
    
    # Create web interface access script
    cat > ~/.mini-xdr/tpot-management/web-access.sh << EOF
#!/bin/bash
# Open T-Pot web interface
echo "=== T-Pot Web Interface ==="
echo "URL: https://$PUBLIC_IP:64297/"
echo "Opening in default browser..."
if command -v open >/dev/null 2>&1; then
    open "https://$PUBLIC_IP:64297/"
elif command -v xdg-open >/dev/null 2>&1; then
    xdg-open "https://$PUBLIC_IP:64297/"
else
    echo "Please manually open: https://$PUBLIC_IP:64297/"
fi
EOF
    chmod +x ~/.mini-xdr/tpot-management/web-access.sh
    
    success "T-Pot management scripts created in ~/.mini-xdr/tpot-management/"
}

# Function to display deployment summary
show_deployment_summary() {
    echo ""
    echo "=== 🍯 T-Pot Honeypot AWS Deployment Complete ==="
    echo ""
    echo "📊 Infrastructure:"
    echo "   • Instance ID:     $INSTANCE_ID"
    echo "   • Public IP:       $PUBLIC_IP"
    echo "   • Private IP:      $PRIVATE_IP"
    echo "   • Region:          $REGION"
    echo "   • Instance Type:   $INSTANCE_TYPE"
    echo "   • VPC:             $VPC_ID"
    echo "   • Subnet:          $SUBNET_ID"
    echo "   • Security Group:  $SG_ID"
    echo ""
    echo "🔗 Access Points (Available after installation completes):"
    echo "   • SSH Management:  ssh -i ~/.ssh/$KEY_NAME.pem -p 64295 tpot@$PUBLIC_IP"
    echo "   • Web Interface:   https://$PUBLIC_IP:64297/"
    echo "   • Honeypot Services: Multiple ports (22, 80, 443, 3306, etc.)"
    echo ""
    echo "🛠️  Local Management Scripts:"
    echo "   • Connect:         ~/.mini-xdr/tpot-management/connect.sh"
    echo "   • Status Check:    ~/.mini-xdr/tpot-management/status.sh"
    echo "   • Restart:         ~/.mini-xdr/tpot-management/restart.sh"
    echo "   • Get Credentials: ~/.mini-xdr/tpot-management/get-credentials.sh"
    echo "   • Web Access:      ~/.mini-xdr/tpot-management/web-access.sh"
    echo ""
    echo "📁 Remote T-Pot Files:"
    echo "   • Installation:    /opt/tpot/tpotce/"
    echo "   • Management:      /opt/mini-xdr-tpot/"
    echo "   • Credentials:     /root/tpot-credentials.txt"
    echo "   • Logs:            /data/*/log/"
    echo ""
    echo "⚙️  Installation Status:"
    echo "   • T-Pot Installation: In Progress (15-30 minutes)"
    echo "   • Auto-reboot: Scheduled after installation"
    echo "   • Auto-start: Enabled on boot"
    echo "   • Log forwarding: Configured to $YOUR_IP:8000"
    echo ""
    echo "🧪 Testing Commands (Use after installation completes):"
    echo "   • Test SSH:        ssh admin@$PUBLIC_IP"
    echo "   • Test Web:        curl http://$PUBLIC_IP/"
    echo "   • Check Status:    ~/.mini-xdr/tpot-management/status.sh"
    echo "   • Get Credentials: ~/.mini-xdr/tpot-management/get-credentials.sh"
    echo "   • Web Interface:   ~/.mini-xdr/tpot-management/web-access.sh"
    echo ""
    echo "💡 Next Steps:"
    echo "   1. Wait 15-30 minutes for T-Pot installation to complete"
    echo "   2. Get credentials: ~/.mini-xdr/tpot-management/get-credentials.sh"
    echo "   3. Access web interface: https://$PUBLIC_IP:64297/"
    echo "   4. Update your Mini-XDR backend .env file:"
    echo "      TPOT_HOST=$PUBLIC_IP"
    echo "      TPOT_SSH_PORT=64295"
    echo "      TPOT_WEB_PORT=64297"
    echo "      TPOT_SSH_KEY=~/.ssh/$KEY_NAME.pem"
    echo ""
    echo "   5. Start your local Mini-XDR:"
    echo "      ./scripts/start-all.sh"
    echo ""
    echo "   6. Test the complete system:"
    echo "      python3 attack_simulation.py --target $PUBLIC_IP"
    echo ""
    echo "🔒 Security:"
    echo "   • Management access restricted to your IP ($YOUR_IP)"
    echo "   • Honeypot services open to internet for threat collection"
    echo "   • All logs forwarded to your Mini-XDR instance"
    echo ""
    echo "💰 Cost Estimate: ~\$50-80/month (t3.xlarge + EIP + data transfer)"
    echo ""
    echo "✅ T-Pot honeypot deployment initiated! Check status in 30 minutes."
}

# Main execution function
main() {
    clear
    echo "=== 🍯 T-Pot Honeypot AWS Deployment for Mini-XDR ==="
    echo "Comprehensive T-Pot honeypot with multiple honeypot services"
    echo ""
    
    # Step 1: Get your IP
    get_your_ip
    echo ""
    
    # Step 2: Check AWS configuration
    check_aws_config
    echo ""
    
    # Step 3: Setup key pair
    setup_key_pair
    echo ""
    
    # Step 4: Get Debian AMI
    get_debian_ami
    echo ""
    
    # Step 5: Create VPC infrastructure
    create_vpc_infrastructure
    echo ""
    
    # Step 6: Create security group
    create_security_group
    echo ""
    
    # Step 7: Launch T-Pot instance
    launch_tpot_instance
    echo ""
    
    # Step 8: Create local management scripts
    create_management_scripts
    echo ""
    
    # Step 9: Show summary
    show_deployment_summary
    
    echo ""
    echo "🎉 T-Pot AWS honeypot deployment completed successfully!"
    echo "Installation will continue automatically on the remote instance."
}

# Handle command line arguments
case "${1:-}" in
    --help|-h)
        echo "Usage: $0 [--help]"
        echo ""
        echo "This script creates a comprehensive T-Pot honeypot on AWS with:"
        echo "  • Complete VPC setup with security groups"
        echo "  • T-Pot installation with multiple honeypot services"
        echo "  • Management scripts for easy administration"
        echo "  • Log forwarding to your Mini-XDR instance"
        echo "  • Web interface for monitoring and analysis"
        echo ""
        echo "Prerequisites:"
        echo "  • AWS CLI installed and configured"
        echo "  • Sufficient AWS permissions for EC2, VPC operations"
        echo "  • Mini-XDR system ready to receive logs"
        echo "  • At least t3.xlarge instance type (8GB RAM, 128GB SSD)"
        echo ""
        exit 0
        ;;
esac

# Run main function
main "$@"


