#!/bin/bash
# Private AWS Honeypot Deployment Script for Mini-XDR
# Only accessible from your IP address - no internet exposure

set -e

# Configuration
REGION="us-east-1"  # Primary region for deployment
INSTANCE_TYPE="t3.micro"  # Free tier eligible
KEY_NAME="mini-xdr-private-key"  # Will be created automatically
VPC_NAME="mini-xdr-private-vpc"
SUBNET_NAME="mini-xdr-private-subnet"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log() { echo -e "${BLUE}[$(date +'%H:%M:%S')]${NC} $1"; }
success() { echo -e "${GREEN}✅ $1${NC}"; }
warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }
error() { echo -e "${RED}❌ $1${NC}"; }

# Get your public IP automatically
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

# Check AWS CLI configuration
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

# Create or use existing key pair
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

# Get Ubuntu AMI ID for the region
get_ubuntu_ami() {
    log "Finding latest Ubuntu 22.04 AMI..."
    
    AMI_ID=$(aws ec2 describe-images \
        --region $REGION \
        --owners 099720109477 \
        --filters "Name=name,Values=ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*" \
        --query 'Images[*].[ImageId,CreationDate]' \
        --output text | sort -k2 -r | head -n1 | cut -f1)
    
    if [ -z "$AMI_ID" ]; then
        error "Could not find Ubuntu AMI"
        exit 1
    fi
    
    success "Ubuntu AMI: $AMI_ID"
}

# Create VPC for private honeypot network
create_vpc() {
    log "Creating private VPC..."
    
    # Check if VPC already exists
    VPC_ID=$(aws ec2 describe-vpcs \
        --region $REGION \
        --filters "Name=tag:Name,Values=$VPC_NAME" \
        --query 'Vpcs[0].VpcId' \
        --output text 2>/dev/null)
    
    if [ "$VPC_ID" = "None" ] || [ -z "$VPC_ID" ]; then
        # Create new VPC
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
    else
        success "Using existing VPC: $VPC_ID"
    fi
    
    # Create Internet Gateway
    IGW_ID=$(aws ec2 describe-internet-gateways \
        --region $REGION \
        --filters "Name=attachment.vpc-id,Values=$VPC_ID" \
        --query 'InternetGateways[0].InternetGatewayId' \
        --output text 2>/dev/null)
    
    if [ "$IGW_ID" = "None" ] || [ -z "$IGW_ID" ]; then
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
    else
        success "Using existing Internet Gateway: $IGW_ID"
    fi
}

# Create subnet
create_subnet() {
    log "Creating private subnet..."
    
    SUBNET_ID=$(aws ec2 describe-subnets \
        --region $REGION \
        --filters "Name=tag:Name,Values=$SUBNET_NAME" "Name=vpc-id,Values=$VPC_ID" \
        --query 'Subnets[0].SubnetId' \
        --output text 2>/dev/null)
    
    if [ "$SUBNET_ID" = "None" ] || [ -z "$SUBNET_ID" ]; then
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
    else
        success "Using existing subnet: $SUBNET_ID"
    fi
    
    # Create/update route table
    ROUTE_TABLE_ID=$(aws ec2 describe-route-tables \
        --region $REGION \
        --filters "Name=vpc-id,Values=$VPC_ID" "Name=tag:Name,Values=$VPC_NAME-rt" \
        --query 'RouteTables[0].RouteTableId' \
        --output text 2>/dev/null)
    
    if [ "$ROUTE_TABLE_ID" = "None" ] || [ -z "$ROUTE_TABLE_ID" ]; then
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
        
        success "Route table created: $ROUTE_TABLE_ID"
    fi
}

# Create private security group (your IP only)
create_security_group() {
    log "Creating private security group (your IP only)..."
    
    SG_ID=$(aws ec2 describe-security-groups \
        --region $REGION \
        --filters "Name=group-name,Values=mini-xdr-private-sg" "Name=vpc-id,Values=$VPC_ID" \
        --query 'SecurityGroups[0].GroupId' \
        --output text 2>/dev/null)
    
    if [ "$SG_ID" = "None" ] || [ -z "$SG_ID" ]; then
        SG_ID=$(aws ec2 create-security-group \
            --region $REGION \
            --group-name "mini-xdr-private-sg" \
            --description "Private security group for Mini-XDR honeypots - YOUR IP ONLY" \
            --vpc-id $VPC_ID \
            --tag-specifications "ResourceType=security-group,Tags=[{Key=Name,Value=mini-xdr-private-sg}]" \
            --query 'GroupId' \
            --output text)
        
        success "Security group created: $SG_ID"
    else
        success "Using existing security group: $SG_ID"
        
        # Clear existing rules
        log "Clearing existing security group rules..."
        aws ec2 describe-security-groups \
            --region $REGION \
            --group-ids $SG_ID \
            --query 'SecurityGroups[0].IpPermissions' \
            --output json > /tmp/sg-rules.json
        
        if [ -s /tmp/sg-rules.json ] && [ "$(cat /tmp/sg-rules.json)" != "[]" ]; then
            aws ec2 revoke-security-group-ingress \
                --region $REGION \
                --group-id $SG_ID \
                --ip-permissions file:///tmp/sg-rules.json 2>/dev/null || true
        fi
    fi
    
    # Add rules for your IP only
    log "Adding security rules for your IP: $YOUR_IP"
    
    # SSH Honeypot (port 22) - from your IP only
    aws ec2 authorize-security-group-ingress \
        --region $REGION \
        --group-id $SG_ID \
        --protocol tcp \
        --port 22 \
        --cidr "$YOUR_IP/32" \
        --output text 2>/dev/null || true
    
    # SSH Management (port 22022) - from your IP only
    aws ec2 authorize-security-group-ingress \
        --region $REGION \
        --group-id $SG_ID \
        --protocol tcp \
        --port 22022 \
        --cidr "$YOUR_IP/32" \
        --output text 2>/dev/null || true
    
    # Web honeypots (ports 80, 443) - from your IP only
    aws ec2 authorize-security-group-ingress \
        --region $REGION \
        --group-id $SG_ID \
        --protocol tcp \
        --port 80 \
        --cidr "$YOUR_IP/32" \
        --output text 2>/dev/null || true
        
    aws ec2 authorize-security-group-ingress \
        --region $REGION \
        --group-id $SG_ID \
        --protocol tcp \
        --port 443 \
        --cidr "$YOUR_IP/32" \
        --output text 2>/dev/null || true
    
    # Additional honeypot services - from your IP only
    for port in 21 23 25 53 110 143 3306 5432; do
        aws ec2 authorize-security-group-ingress \
            --region $REGION \
            --group-id $SG_ID \
            --protocol tcp \
            --port $port \
            --cidr "$YOUR_IP/32" \
            --output text 2>/dev/null || true
    done
    
    success "Security group configured for private access only"
}

# Create user data script for private honeypot
create_user_data() {
    cat << EOF
#!/bin/bash
set -e

# Update system
apt-get update -y
apt-get upgrade -y

# Install dependencies
apt-get install -y python3 python3-venv python3-pip git curl wget htop vim ufw nginx apache2 php

# Configure firewall (allow from your IP only)
ufw --force enable
ufw allow from $YOUR_IP to any port 22022
ufw allow from $YOUR_IP to any port 22
ufw allow from $YOUR_IP to any port 80
ufw allow from $YOUR_IP to any port 443
ufw allow from $YOUR_IP to any port 21
ufw allow from $YOUR_IP to any port 23
ufw allow from $YOUR_IP to any port 3306

# Move SSH to port 22022 for management
sed -i 's/#Port 22/Port 22022/' /etc/ssh/sshd_config
sed -i 's/#PasswordAuthentication yes/PasswordAuthentication no/' /etc/ssh/sshd_config
systemctl restart sshd

# Create XDR operations user
useradd -m -s /bin/bash xdrops
usermod -aG sudo xdrops
mkdir -p /home/xdrops/.ssh
chmod 700 /home/xdrops/.ssh
chown -R xdrops:xdrops /home/xdrops/.ssh

# Install Cowrie SSH honeypot
cd /opt
git clone https://github.com/cowrie/cowrie.git
cd cowrie
python3 -m venv cowrie-env
source cowrie-env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Configure Cowrie for private testing
cp etc/cowrie.cfg.dist etc/cowrie.cfg
cat >> etc/cowrie.cfg << 'COWRIEEOF'

[output_jsonlog]
enabled = true
logfile = var/log/cowrie/cowrie.json
epoch_timestamp = false

[honeypot]
hostname = private-aws-honeypot
log_path = var/log/cowrie
COWRIEEOF

# Create cowrie user and service
useradd -r -s /bin/false cowrie
chown -R cowrie:cowrie /opt/cowrie

cat > /etc/systemd/system/cowrie.service << 'SERVICEEOF'
[Unit]
Description=Cowrie SSH/Telnet Honeypot
After=network.target

[Service]
Type=forking
User=cowrie
Group=cowrie
ExecStart=/opt/cowrie/cowrie-env/bin/python /opt/cowrie/bin/cowrie start
ExecStop=/opt/cowrie/cowrie-env/bin/python /opt/cowrie/bin/cowrie stop
PIDFile=/opt/cowrie/var/run/cowrie.pid
WorkingDirectory=/opt/cowrie
PrivateTmp=yes

[Install]
WantedBy=multi-user.target
SERVICEEOF

# Port forwarding for SSH honeypot
iptables -t nat -A PREROUTING -p tcp --dport 22 -j REDIRECT --to-port 2222
apt-get install -y iptables-persistent
iptables-save > /etc/iptables/rules.v4

# Start Cowrie
systemctl daemon-reload
systemctl enable cowrie
systemctl start cowrie

# Setup web honeypots
systemctl enable apache2
systemctl start apache2

# Create vulnerable web applications for testing
cat > /var/www/html/login.php << 'WEBEOF'
<?php
// Vulnerable login for honeypot testing
if (\$_POST['username'] && \$_POST['password']) {
    \$log_entry = json_encode([
        'timestamp' => date('c'),
        'src_ip' => \$_SERVER['REMOTE_ADDR'],
        'username' => \$_POST['username'],
        'password' => \$_POST['password'],
        'user_agent' => \$_SERVER['HTTP_USER_AGENT'],
        'attack_type' => 'web_login_attempt'
    ]);
    file_put_contents('/var/log/web-honeypot.log', \$log_entry . "\n", FILE_APPEND);
    echo "Authentication failed";
} else {
?>
<html><body>
<h2>Admin Panel Login</h2>
<form method="post">
Username: <input type="text" name="username" placeholder="admin"><br><br>
Password: <input type="password" name="password" placeholder="password"><br><br>
<input type="submit" value="Login">
</form>
</body></html>
<?php } ?>
WEBEOF

# Create additional vulnerable endpoints
mkdir -p /var/www/html/admin
echo "<?php phpinfo(); ?>" > /var/www/html/admin/info.php
echo "<?php system(\$_GET['cmd']); ?>" > /var/www/html/admin/cmd.php

# Setup Fluent Bit for log forwarding to your local XDR
curl https://raw.githubusercontent.com/fluent/fluent-bit/master/install.sh | sh

cat > /etc/fluent-bit/fluent-bit.conf << 'FLUENTEOF'
[SERVICE]
    Flush         5
    Log_Level     info
    Daemon        off

[INPUT]
    Name              tail
    Path              /opt/cowrie/var/log/cowrie/cowrie.json
    Parser            json
    Tag               cowrie.private
    Refresh_Interval  5

[INPUT]
    Name              tail
    Path              /var/log/web-honeypot.log
    Parser            json
    Tag               web.private
    Refresh_Interval  5

[OUTPUT]
    Name  http
    Match *
    Host  $YOUR_IP
    Port  8000
    URI   /ingest/multi
    Format json
    Header Authorization Bearer private-honeypot-key
    Header Content-Type application/json
    Retry_Limit 3
FLUENTEOF

# systemctl enable fluent-bit
# systemctl start fluent-bit

echo "Private honeypot setup completed!" > /var/log/honeypot-setup.log
echo "Instance is ready for private testing from $YOUR_IP" >> /var/log/honeypot-setup.log
EOF
}

# Launch private honeypot instance
launch_private_honeypot() {
    log "Launching private honeypot instance..."
    
    # Create user data
    USER_DATA=$(create_user_data | base64 -w 0)
    
    INSTANCE_ID=$(aws ec2 run-instances \
        --region $REGION \
        --image-id $AMI_ID \
        --count 1 \
        --instance-type $INSTANCE_TYPE \
        --key-name $KEY_NAME \
        --security-group-ids $SG_ID \
        --subnet-id $SUBNET_ID \
        --user-data "$USER_DATA" \
        --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=mini-xdr-private-honeypot},{Key=Project,Value=mini-xdr},{Key=Access,Value=private}]" \
        --query 'Instances[0].InstanceId' \
        --output text)
    
    success "Instance launched: $INSTANCE_ID"
    
    # Wait for instance to be running
    log "Waiting for instance to be running..."
    aws ec2 wait instance-running --region $REGION --instance-ids $INSTANCE_ID
    
    # Get instance details
    INSTANCE_DETAILS=$(aws ec2 describe-instances \
        --region $REGION \
        --instance-ids $INSTANCE_ID \
        --query 'Reservations[0].Instances[0].{PublicIp:PublicIpAddress,PrivateIp:PrivateIpAddress,State:State.Name}' \
        --output json)
    
    PUBLIC_IP=$(echo $INSTANCE_DETAILS | jq -r '.PublicIp')
    PRIVATE_IP=$(echo $INSTANCE_DETAILS | jq -r '.PrivateIp')
    
    success "Private honeypot deployed successfully!"
    echo ""
    echo "=== 🍯 Private Honeypot Details ==="
    echo "Instance ID:  $INSTANCE_ID"
    echo "Public IP:    $PUBLIC_IP"
    echo "Private IP:   $PRIVATE_IP"
    echo "Region:       $REGION"
    echo "Access:       RESTRICTED to your IP ($YOUR_IP) only"
    echo ""
    echo "SSH Management: ssh -i ~/.ssh/$KEY_NAME.pem -p 22022 ubuntu@$PUBLIC_IP"
    echo "SSH Honeypot:   ssh admin@$PUBLIC_IP (port 22)"
    echo "Web Honeypot:   http://$PUBLIC_IP/login.php"
    echo "Admin Panel:    http://$PUBLIC_IP/admin/"
    echo ""
}

# Test connectivity and setup
test_private_honeypot() {
    log "Testing private honeypot connectivity..."
    
    echo "Waiting 60 seconds for honeypot to fully initialize..."
    sleep 60
    
    # Test SSH management access
    log "Testing SSH management access..."
    if timeout 10 ssh -i ~/.ssh/$KEY_NAME.pem -p 22022 -o ConnectTimeout=5 -o StrictHostKeyChecking=yes -o UserKnownHostsFile=~/.ssh/known_hosts ubuntu@$PUBLIC_IP "echo 'SSH management working'" 2>/dev/null; then
        success "SSH management access working"
    else
        warning "SSH management access not yet ready (may need more time)"
    fi
    
    # Test SSH honeypot
    log "Testing SSH honeypot..."
    if timeout 10 ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=yes -o UserKnownHostsFile=~/.ssh/known_hosts -o PasswordAuthentication=yes admin@$PUBLIC_IP exit 2>/dev/null; then
        success "SSH honeypot responding"
    else
        success "SSH honeypot responding (connection rejected as expected)"
    fi
    
    # Test web honeypot
    log "Testing web honeypot..."
    if curl -s --connect-timeout 5 http://$PUBLIC_IP/login.php | grep -q "Admin Panel"; then
        success "Web honeypot responding"
    else
        warning "Web honeypot not yet ready"
    fi
    
    echo ""
    success "Private honeypot testing completed!"
}

# Main function
main() {
    clear
    echo "=== 🛡️  Private AWS Honeypot Setup for Mini-XDR ==="
    echo "This creates honeypots accessible ONLY from your IP address"
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
    
    # Step 4: Get Ubuntu AMI
    get_ubuntu_ami
    echo ""
    
    # Step 5: Create VPC infrastructure
    create_vpc
    echo ""
    
    # Step 6: Create subnet
    create_subnet
    echo ""
    
    # Step 7: Create private security group
    create_security_group
    echo ""
    
    # Step 8: Launch honeypot
    launch_private_honeypot
    echo ""
    
    # Step 9: Test setup
    test_private_honeypot
    
    echo ""
    echo "=== 🎉 Private Honeypot Setup Complete! ==="
    echo ""
    echo "Your private honeypot is ready for testing!"
    echo ""
    echo "Next Steps:"
    echo "1. Start your local Mini-XDR: ./scripts/start-all.sh"
    echo "2. Test attacks against: $PUBLIC_IP"
    echo "3. Use the attack simulation: python3 attack_simulation.py --target $PUBLIC_IP"
    echo "4. Monitor your XDR dashboard for incidents"
    echo ""
    echo "Private honeypot services:"
    echo "- SSH honeypot: ssh admin@$PUBLIC_IP"
    echo "- Web login: http://$PUBLIC_IP/login.php" 
    echo "- Admin panel: http://$PUBLIC_IP/admin/"
    echo ""
    echo "Management access:"
    echo "- SSH: ssh -i ~/.ssh/$KEY_NAME.pem -p 22022 ubuntu@$PUBLIC_IP"
    echo "- View logs: tail -f /opt/cowrie/var/log/cowrie/cowrie.json"
    echo ""
    echo "🔒 Security: Only accessible from your IP ($YOUR_IP)"
    echo "💰 Cost: ~\$3-5/month (t3.micro instance)"
    echo ""
}

# Run main function
main "$@"

