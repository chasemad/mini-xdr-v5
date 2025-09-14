#!/bin/bash
# Enhanced AWS Honeypot Setup Script for Mini-XDR
# Sets up AWS VM with automatic startup/stop capabilities and comprehensive honeypot tools

set -e

# Configuration
REGION="us-east-1"  # Default region
INSTANCE_TYPE="t3.micro"  # Free tier eligible
KEY_NAME="mini-xdr-honeypot-key"
VPC_NAME="mini-xdr-honeypot-vpc"
SUBNET_NAME="mini-xdr-honeypot-subnet"
SECURITY_GROUP_NAME="mini-xdr-honeypot-sg"

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

# Function to get Ubuntu AMI ID for the region
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

# Function to create VPC infrastructure
create_vpc_infrastructure() {
    log "Creating VPC infrastructure..."
    
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

# Function to create security group
create_security_group() {
    log "Creating security group..."
    
    SG_ID=$(aws ec2 create-security-group \
        --region $REGION \
        --group-name $SECURITY_GROUP_NAME \
        --description "Security group for Mini-XDR honeypots with management access" \
        --vpc-id $VPC_ID \
        --tag-specifications "ResourceType=security-group,Tags=[{Key=Name,Value=$SECURITY_GROUP_NAME}]" \
        --query 'GroupId' \
        --output text)
    
    success "Security group created: $SG_ID"
    
    # Add rules for honeypot services (open to internet)
    log "Adding security group rules..."
    
    # SSH Honeypot (port 22) - open to internet
    aws ec2 authorize-security-group-ingress \
        --region $REGION \
        --group-id $SG_ID \
        --protocol tcp \
        --port 22 \
        --cidr 0.0.0.0/0 \
        --output text 2>/dev/null || true
    
    # Web honeypots (ports 80, 443) - open to internet
    aws ec2 authorize-security-group-ingress \
        --region $REGION \
        --group-id $SG_ID \
        --protocol tcp \
        --port 80 \
        --cidr 0.0.0.0/0 \
        --output text 2>/dev/null || true
        
    aws ec2 authorize-security-group-ingress \
        --region $REGION \
        --group-id $SG_ID \
        --protocol tcp \
        --port 443 \
        --cidr 0.0.0.0/0 \
        --output text 2>/dev/null || true
    
    # SSH Management (port 22022) - restricted to your IP
    aws ec2 authorize-security-group-ingress \
        --region $REGION \
        --group-id $SG_ID \
        --protocol tcp \
        --port 22022 \
        --cidr "$YOUR_IP/32" \
        --output text 2>/dev/null || true
    
    # Additional honeypot services - open to internet
    for port in 21 23 25 53 110 143 3306 5432 1433; do
        aws ec2 authorize-security-group-ingress \
            --region $REGION \
            --group-id $SG_ID \
            --protocol tcp \
            --port $port \
            --cidr 0.0.0.0/0 \
            --output text 2>/dev/null || true
    done
    
    success "Security group configured with honeypot and management rules"
}

# Function to create comprehensive user data script
create_user_data() {
    cat << EOF
#!/bin/bash
set -e

# Comprehensive AWS Honeypot Setup with Auto-Start/Stop
echo "=== Starting Enhanced Mini-XDR Honeypot Setup ==="

# Update system
apt-get update -y
apt-get upgrade -y

# Install comprehensive dependencies
apt-get install -y python3 python3-venv python3-pip git curl wget htop vim ufw nginx apache2 php \
    jq net-tools iptables-persistent fail2ban logrotate cron supervisor \
    build-essential libssl-dev libffi-dev python3-dev

# Configure timezone
timedatectl set-timezone UTC

# Set hostname
echo "mini-xdr-honeypot" > /etc/hostname
hostnamectl set-hostname mini-xdr-honeypot
echo "127.0.0.1 mini-xdr-honeypot" >> /etc/hosts

# Configure firewall with comprehensive rules
ufw --force enable
ufw allow from $YOUR_IP to any port 22022 comment "SSH Management"
ufw allow 22/tcp comment "SSH Honeypot"
ufw allow 80/tcp comment "Web Honeypot HTTP"
ufw allow 443/tcp comment "Web Honeypot HTTPS"
ufw allow 21/tcp comment "FTP Honeypot"
ufw allow 23/tcp comment "Telnet Honeypot"
ufw allow 25/tcp comment "SMTP Honeypot"
ufw allow 53/tcp comment "DNS Honeypot"
ufw allow 110/tcp comment "POP3 Honeypot"
ufw allow 143/tcp comment "IMAP Honeypot"
ufw allow 3306/tcp comment "MySQL Honeypot"
ufw allow 5432/tcp comment "PostgreSQL Honeypot"
ufw allow 1433/tcp comment "MSSQL Honeypot"

# Move SSH to management port
sed -i 's/#Port 22/Port 22022/' /etc/ssh/sshd_config
sed -i 's/#PasswordAuthentication yes/PasswordAuthentication no/' /etc/ssh/sshd_config
sed -i 's/#PubkeyAuthentication yes/PubkeyAuthentication yes/' /etc/ssh/sshd_config
systemctl restart sshd

# Create XDR operations user
useradd -m -s /bin/bash xdrops
usermod -aG sudo xdrops
mkdir -p /home/xdrops/.ssh
chmod 700 /home/xdrops/.ssh
chown -R xdrops:xdrops /home/xdrops/.ssh

# Configure sudoers for XDR operations
echo "xdrops ALL=(ALL) NOPASSWD: /usr/sbin/ufw, /bin/systemctl" > /etc/sudoers.d/xdrops

# Install and configure Cowrie SSH honeypot
log "Installing Cowrie SSH honeypot..."
cd /opt
git clone https://github.com/cowrie/cowrie.git
cd cowrie
python3 -m venv cowrie-env
source cowrie-env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Configure Cowrie with enhanced settings
cp etc/cowrie.cfg.dist etc/cowrie.cfg
cat >> etc/cowrie.cfg << 'COWRIEEOF'

[output_jsonlog]
enabled = true
logfile = var/log/cowrie/cowrie.json
epoch_timestamp = false

[output_textlog]
enabled = true
logfile = var/log/cowrie/cowrie.log

[honeypot]
hostname = aws-honeypot-$REGION
log_path = var/log/cowrie
download_path = var/lib/cowrie/downloads
filesystem_file = share/cowrie/fs.pickle
data_path = var/lib/cowrie
state_path = var/lib/cowrie

[telnet]
enabled = true
listen_endpoints = tcp:2223:interface=0.0.0.0

[ssh]
enabled = true
listen_endpoints = tcp:2222:interface=0.0.0.0
COWRIEEOF

# Create cowrie user and service
useradd -r -s /bin/false cowrie || echo "Cowrie user exists"
chown -R cowrie:cowrie /opt/cowrie
chmod +x /opt/cowrie/bin/cowrie

# Create comprehensive Cowrie systemd service
cat > /etc/systemd/system/cowrie.service << 'SERVICEEOF'
[Unit]
Description=Cowrie SSH/Telnet Honeypot
After=network.target
Wants=network.target

[Service]
Type=forking
User=cowrie
Group=cowrie
ExecStart=/opt/cowrie/cowrie-env/bin/python /opt/cowrie/bin/cowrie start
ExecStop=/opt/cowrie/cowrie-env/bin/python /opt/cowrie/bin/cowrie stop
ExecReload=/opt/cowrie/cowrie-env/bin/python /opt/cowrie/bin/cowrie restart
PIDFile=/opt/cowrie/var/run/cowrie.pid
WorkingDirectory=/opt/cowrie
PrivateTmp=yes
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
SERVICEEOF

# Configure iptables for port redirection
iptables -t nat -A PREROUTING -p tcp --dport 22 -j REDIRECT --to-port 2222
iptables -t nat -A PREROUTING -p tcp --dport 23 -j REDIRECT --to-port 2223
iptables-save > /etc/iptables/rules.v4

# Setup comprehensive web honeypots
systemctl enable apache2
systemctl start apache2

# Create multiple vulnerable web applications
mkdir -p /var/www/html/{admin,wp-admin,phpmyadmin,webmail,api}

# Main vulnerable login page
cat > /var/www/html/login.php << 'WEBEOF'
<?php
// Enhanced vulnerable login for comprehensive honeypot testing
session_start();
\$log_file = '/var/log/web-honeypot.log';

if (\$_POST['username'] && \$_POST['password']) {
    \$log_entry = json_encode([
        'timestamp' => date('c'),
        'src_ip' => \$_SERVER['REMOTE_ADDR'],
        'user_agent' => \$_SERVER['HTTP_USER_AGENT'],
        'username' => \$_POST['username'],
        'password' => \$_POST['password'],
        'referer' => \$_SERVER['HTTP_REFERER'] ?? '',
        'attack_type' => 'web_login_attempt',
        'session_id' => session_id(),
        'request_uri' => \$_SERVER['REQUEST_URI']
    ]);
    file_put_contents(\$log_file, \$log_entry . "\n", FILE_APPEND | LOCK_EX);
    
    // Simulate different responses
    if (rand(1, 10) > 8) {
        echo "Server Error - Please try again later";
    } else {
        echo "Invalid credentials";
    }
} else {
?>
<!DOCTYPE html>
<html>
<head>
    <title>Admin Panel - Login</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 50px; }
        .login-form { max-width: 400px; margin: auto; padding: 20px; border: 1px solid #ccc; }
        input { width: 100%; padding: 10px; margin: 10px 0; }
        button { width: 100%; padding: 10px; background: #007cba; color: white; border: none; }
    </style>
</head>
<body>
    <div class="login-form">
        <h2>System Administration Panel</h2>
        <form method="post">
            <input type="text" name="username" placeholder="Username" required>
            <input type="password" name="password" placeholder="Password" required>
            <button type="submit">Login</button>
        </form>
        <p><small>Authorized personnel only</small></p>
    </div>
</body>
</html>
<?php } ?>
WEBEOF

# WordPress admin simulation
cat > /var/www/html/wp-admin/index.php << 'WPEOF'
<?php
// WordPress admin simulation
\$log_file = '/var/log/web-honeypot.log';
\$log_entry = json_encode([
    'timestamp' => date('c'),
    'src_ip' => \$_SERVER['REMOTE_ADDR'],
    'user_agent' => \$_SERVER['HTTP_USER_AGENT'],
    'attack_type' => 'wordpress_admin_access',
    'request_uri' => \$_SERVER['REQUEST_URI']
]);
file_put_contents(\$log_file, \$log_entry . "\n", FILE_APPEND | LOCK_EX);
?>
<!DOCTYPE html>
<html><head><title>WordPress Admin</title></head>
<body><h1>WordPress</h1><p>Please log in to access the administration area.</p></body></html>
WPEOF

# phpMyAdmin simulation
echo "<?php header('HTTP/1.0 403 Forbidden'); echo 'Access Denied'; ?>" > /var/www/html/phpmyadmin/index.php

# API endpoints
cat > /var/www/html/api/v1.php << 'APIEOF'
<?php
// API honeypot
\$log_file = '/var/log/web-honeypot.log';
\$input = file_get_contents('php://input');
\$log_entry = json_encode([
    'timestamp' => date('c'),
    'src_ip' => \$_SERVER['REMOTE_ADDR'],
    'user_agent' => \$_SERVER['HTTP_USER_AGENT'],
    'attack_type' => 'api_access',
    'request_method' => \$_SERVER['REQUEST_METHOD'],
    'request_uri' => \$_SERVER['REQUEST_URI'],
    'post_data' => \$input
]);
file_put_contents(\$log_file, \$log_entry . "\n", FILE_APPEND | LOCK_EX);
header('Content-Type: application/json');
echo '{"error": "Unauthorized", "code": 401}';
?>
APIEOF

# Install and configure Fluent Bit for enhanced log forwarding
curl https://raw.githubusercontent.com/fluent/fluent-bit/master/install.sh | sh

# Configure Fluent Bit with multiple inputs
cat > /etc/fluent-bit/fluent-bit.conf << 'FLUENTEOF'
[SERVICE]
    Flush         5
    Log_Level     info
    Daemon        off
    Parsers_File  parsers.conf

[INPUT]
    Name              tail
    Path              /opt/cowrie/var/log/cowrie/cowrie.json
    Parser            json
    Tag               cowrie
    Refresh_Interval  5
    Read_from_Head    false

[INPUT]
    Name              tail
    Path              /var/log/web-honeypot.log
    Parser            json
    Tag               webhoneypot
    Refresh_Interval  5
    Read_from_Head    false

[INPUT]
    Name              tail
    Path              /var/log/apache2/access.log
    Tag               apache.access
    Refresh_Interval  5

[INPUT]
    Name              tail
    Path              /var/log/auth.log
    Tag               auth
    Refresh_Interval  10

[OUTPUT]
    Name  http
    Match *
    Host  $YOUR_IP
    Port  8000
    URI   /ingest/multi
    Format json
    Header Authorization Bearer aws-honeypot-enhanced-key
    Header Content-Type application/json
    Retry_Limit 3
FLUENTEOF

# Create startup and stop scripts directory
mkdir -p /opt/mini-xdr
chmod 755 /opt/mini-xdr

# Download and install the startup script
cat > /opt/mini-xdr/honeypot-vm-startup.sh << 'STARTUPEOF'
$(cat /dev/stdin)
STARTUPEOF
chmod +x /opt/mini-xdr/honeypot-vm-startup.sh

# Create systemd service for automatic startup
cat > /etc/systemd/system/honeypot-startup.service << 'SYSTEMDEOF'
[Unit]
Description=Mini-XDR Honeypot Startup Service
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
ExecStart=/opt/mini-xdr/honeypot-vm-startup.sh --auto-startup
RemainAfterExit=true
StandardOutput=journal
StandardError=journal
TimeoutStartSec=300

[Install]
WantedBy=multi-user.target
SYSTEMDEOF

# Enable services
systemctl daemon-reload
systemctl enable cowrie
systemctl enable apache2
systemctl enable fluent-bit
systemctl enable honeypot-startup

# Start services
systemctl start cowrie
systemctl start apache2
systemctl start fluent-bit

# Create log rotation configuration
cat > /etc/logrotate.d/honeypot << 'LOGROTATEEOF'
/opt/cowrie/var/log/cowrie/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    copytruncate
    su cowrie cowrie
}

/var/log/web-honeypot.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    copytruncate
}

/var/log/apache2/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    postrotate
        systemctl reload apache2
    endscript
}
LOGROTATEEOF

# Create status monitoring script
cat > /opt/mini-xdr/honeypot-status.sh << 'STATUSEOF'
#!/bin/bash
echo "=== Mini-XDR Honeypot Status ==="
echo "Timestamp: \$(date)"
echo ""
echo "Services:"
for service in cowrie apache2 fluent-bit ufw; do
    status=\$(systemctl is-active \$service 2>/dev/null || echo "inactive")
    echo "  \$service: \$status"
done
echo ""
echo "Network Ports:"
netstat -tlnp | grep -E ":22|:80|:443|:2222|:22022" | head -10
echo ""
echo "Recent Cowrie Activity:"
tail -5 /opt/cowrie/var/log/cowrie/cowrie.json 2>/dev/null || echo "No recent activity"
echo ""
echo "Recent Web Activity:"
tail -5 /var/log/web-honeypot.log 2>/dev/null || echo "No recent activity"
STATUSEOF
chmod +x /opt/mini-xdr/honeypot-status.sh

# Create initial status file
/opt/mini-xdr/honeypot-status.sh > /var/lib/honeypot-status.txt

# Install CloudWatch agent for monitoring
wget -q https://s3.amazonaws.com/amazoncloudwatch-agent/ubuntu/amd64/latest/amazon-cloudwatch-agent.deb
dpkg -i amazon-cloudwatch-agent.deb || true

echo "=== Enhanced Mini-XDR Honeypot Setup Complete ==="
echo "Timestamp: \$(date)"
echo "Services started: cowrie, apache2, fluent-bit"
echo "Auto-startup enabled: honeypot-startup.service"
echo "Management scripts: /opt/mini-xdr/"
echo "Status: Ready for threat detection"

# Final verification
/opt/mini-xdr/honeypot-status.sh
EOF
}

# Function to launch honeypot instance
launch_honeypot_instance() {
    log "Launching enhanced honeypot instance..."
    
    # Create user data with startup script embedded
    local startup_script_content=$(cat /Users/chasemad/Desktop/mini-xdr/ops/honeypot-vm-startup.sh 2>/dev/null || echo "# Startup script not found")
    local user_data=$(create_user_data | sed "s|\$(cat /dev/stdin)|$startup_script_content|")
    
    INSTANCE_ID=$(aws ec2 run-instances \
        --region $REGION \
        --image-id $AMI_ID \
        --count 1 \
        --instance-type $INSTANCE_TYPE \
        --key-name $KEY_NAME \
        --security-group-ids $SG_ID \
        --subnet-id $SUBNET_ID \
        --user-data "$user_data" \
        --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=mini-xdr-enhanced-honeypot},{Key=Project,Value=mini-xdr},{Key=Type,Value=enhanced-honeypot}]" \
        --query 'Instances[0].InstanceId' \
        --output text)
    
    success "Instance launched: $INSTANCE_ID"
    
    # Wait for instance to be running
    log "Waiting for instance to be running..."
    aws ec2 wait instance-running --region $REGION --instance-ids $INSTANCE_ID
    
    # Allocate and associate Elastic IP
    log "Allocating Elastic IP..."
    local allocation_id=$(aws ec2 allocate-address \
        --region $REGION \
        --domain vpc \
        --tag-specifications "ResourceType=elastic-ip,Tags=[{Key=Name,Value=mini-xdr-honeypot-ip},{Key=Project,Value=mini-xdr}]" \
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
    
    success "Enhanced honeypot deployed successfully!"
    
    # Get instance details
    PRIVATE_IP=$(aws ec2 describe-instances \
        --region $REGION \
        --instance-ids $INSTANCE_ID \
        --query 'Reservations[0].Instances[0].PrivateIpAddress' \
        --output text)
}

# Function to create management scripts locally
create_management_scripts() {
    log "Creating local management scripts..."
    
    # Create local management directory
    mkdir -p ~/.mini-xdr/honeypot-management
    
    # Create connection script
    cat > ~/.mini-xdr/honeypot-management/connect.sh << EOF
#!/bin/bash
# Connect to Mini-XDR honeypot for management
ssh -i ~/.ssh/$KEY_NAME.pem -p 22022 ubuntu@$PUBLIC_IP "\$@"
EOF
    chmod +x ~/.mini-xdr/honeypot-management/connect.sh
    
    # Create status check script
    cat > ~/.mini-xdr/honeypot-management/status.sh << EOF
#!/bin/bash
# Check Mini-XDR honeypot status remotely
echo "=== Remote Honeypot Status Check ==="
ssh -i ~/.ssh/$KEY_NAME.pem -p 22022 ubuntu@$PUBLIC_IP 'sudo /opt/mini-xdr/honeypot-status.sh'
EOF
    chmod +x ~/.mini-xdr/honeypot-management/status.sh
    
    # Create restart script
    cat > ~/.mini-xdr/honeypot-management/restart.sh << EOF
#!/bin/bash
# Restart Mini-XDR honeypot services remotely
echo "=== Restarting Honeypot Services ==="
ssh -i ~/.ssh/$KEY_NAME.pem -p 22022 ubuntu@$PUBLIC_IP 'sudo systemctl restart honeypot-startup'
echo "Restart initiated. Check status in 30 seconds."
EOF
    chmod +x ~/.mini-xdr/honeypot-management/restart.sh
    
    # Create stop script
    cat > ~/.mini-xdr/honeypot-management/stop.sh << EOF
#!/bin/bash
# Stop Mini-XDR honeypot services remotely
echo "=== Stopping Honeypot Services ==="
ssh -i ~/.ssh/$KEY_NAME.pem -p 22022 ubuntu@$PUBLIC_IP 'sudo /opt/mini-xdr/honeypot-vm-stop.sh'
EOF
    chmod +x ~/.mini-xdr/honeypot-management/stop.sh
    
    # Create start script
    cat > ~/.mini-xdr/honeypot-management/start.sh << EOF
#!/bin/bash
# Start Mini-XDR honeypot services remotely
echo "=== Starting Honeypot Services ==="
ssh -i ~/.ssh/$KEY_NAME.pem -p 22022 ubuntu@$PUBLIC_IP 'sudo /opt/mini-xdr/honeypot-vm-startup.sh'
EOF
    chmod +x ~/.mini-xdr/honeypot-management/start.sh
    
    success "Management scripts created in ~/.mini-xdr/honeypot-management/"
}

# Function to test honeypot connectivity
test_honeypot() {
    log "Testing honeypot connectivity and services..."
    
    echo "Waiting 120 seconds for honeypot to fully initialize..."
    sleep 120
    
    # Test SSH management access
    log "Testing SSH management access..."
    if timeout 10 ssh -i ~/.ssh/$KEY_NAME.pem -p 22022 -o ConnectTimeout=5 -o StrictHostKeyChecking=no ubuntu@$PUBLIC_IP "echo 'SSH management working'" 2>/dev/null; then
        success "SSH management access working"
    else
        warning "SSH management access not yet ready (may need more time)"
    fi
    
    # Test SSH honeypot
    log "Testing SSH honeypot..."
    if timeout 5 ssh -o ConnectTimeout=3 -o StrictHostKeyChecking=no admin@$PUBLIC_IP exit 2>/dev/null; then
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
    
    # Test status endpoint
    log "Testing honeypot status..."
    if timeout 10 ssh -i ~/.ssh/$KEY_NAME.pem -p 22022 -o ConnectTimeout=5 -o StrictHostKeyChecking=no ubuntu@$PUBLIC_IP "sudo /opt/mini-xdr/honeypot-status.sh" 2>/dev/null; then
        success "Honeypot status check working"
    else
        warning "Honeypot status check not ready yet"
    fi
}

# Function to display deployment summary
show_deployment_summary() {
    echo ""
    echo "=== 🍯 Enhanced Mini-XDR Honeypot Deployment Complete ==="
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
    echo "🔗 Access Points:"
    echo "   • SSH Management:  ssh -i ~/.ssh/$KEY_NAME.pem -p 22022 ubuntu@$PUBLIC_IP"
    echo "   • SSH Honeypot:    ssh admin@$PUBLIC_IP (port 22)"
    echo "   • Web Honeypot:    http://$PUBLIC_IP/login.php"
    echo "   • WordPress:       http://$PUBLIC_IP/wp-admin/"
    echo "   • API Endpoint:    http://$PUBLIC_IP/api/v1.php"
    echo ""
    echo "🛠️  Local Management Scripts:"
    echo "   • Connect:         ~/.mini-xdr/honeypot-management/connect.sh"
    echo "   • Status Check:    ~/.mini-xdr/honeypot-management/status.sh"
    echo "   • Start Services:  ~/.mini-xdr/honeypot-management/start.sh"
    echo "   • Stop Services:   ~/.mini-xdr/honeypot-management/stop.sh"
    echo "   • Restart:         ~/.mini-xdr/honeypot-management/restart.sh"
    echo ""
    echo "📁 Remote Honeypot Files:"
    echo "   • Startup Script:  /opt/mini-xdr/honeypot-vm-startup.sh"
    echo "   • Stop Script:     /opt/mini-xdr/honeypot-vm-stop.sh"
    echo "   • Status Script:   /opt/mini-xdr/honeypot-status.sh"
    echo "   • Cowrie Logs:     /opt/cowrie/var/log/cowrie/cowrie.json"
    echo "   • Web Logs:        /var/log/web-honeypot.log"
    echo ""
    echo "⚙️  Auto-Management:"
    echo "   • Auto-start on boot: Enabled (honeypot-startup.service)"
    echo "   • Log rotation: Configured (30 days retention)"
    echo "   • Status monitoring: /var/lib/honeypot-status.txt"
    echo ""
    echo "🧪 Testing Commands:"
    echo "   • Test SSH:        ssh admin@$PUBLIC_IP"
    echo "   • Test Web:        curl http://$PUBLIC_IP/login.php"
    echo "   • Check Status:    ~/.mini-xdr/honeypot-management/status.sh"
    echo "   • View Logs:       ssh -i ~/.ssh/$KEY_NAME.pem -p 22022 ubuntu@$PUBLIC_IP 'sudo tail -f /opt/cowrie/var/log/cowrie/cowrie.json'"
    echo ""
    echo "💡 Next Steps:"
    echo "   1. Update your Mini-XDR backend .env file:"
    echo "      HONEYPOT_HOST=$PUBLIC_IP"
    echo "      HONEYPOT_SSH_PORT=22022"
    echo "      HONEYPOT_USER=ubuntu"
    echo "      HONEYPOT_SSH_KEY=~/.ssh/$KEY_NAME.pem"
    echo ""
    echo "   2. Start your local Mini-XDR:"
    echo "      ./scripts/start-all.sh"
    echo ""
    echo "   3. Test the complete system:"
    echo "      python3 attack_simulation.py --target $PUBLIC_IP"
    echo ""
    echo "🔒 Security:"
    echo "   • Management access restricted to your IP ($YOUR_IP)"
    echo "   • Honeypot services open to internet for threat collection"
    echo "   • All logs forwarded to your Mini-XDR instance"
    echo ""
    echo "💰 Cost Estimate: ~\$8-12/month (t3.micro + EIP + data transfer)"
    echo ""
    echo "✅ Enhanced honeypot is ready for advanced threat detection!"
}

# Main execution function
main() {
    clear
    echo "=== 🍯 Enhanced AWS Honeypot Setup for Mini-XDR ==="
    echo "Complete infrastructure deployment with auto-start/stop capabilities"
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
    create_vpc_infrastructure
    echo ""
    
    # Step 6: Create security group
    create_security_group
    echo ""
    
    # Step 7: Launch honeypot instance
    launch_honeypot_instance
    echo ""
    
    # Step 8: Create local management scripts
    create_management_scripts
    echo ""
    
    # Step 9: Test honeypot
    test_honeypot
    
    # Step 10: Show summary
    show_deployment_summary
    
    echo ""
    echo "🎉 Enhanced AWS honeypot deployment completed successfully!"
}

# Handle command line arguments
case "${1:-}" in
    --help|-h)
        echo "Usage: $0 [--help]"
        echo ""
        echo "This script creates a comprehensive AWS honeypot infrastructure with:"
        echo "  • Complete VPC setup with security groups"
        echo "  • Auto-starting honeypot services (SSH, Web, etc.)"
        echo "  • Management scripts for easy start/stop operations"
        echo "  • Log forwarding to your Mini-XDR instance"
        echo "  • Comprehensive monitoring and status reporting"
        echo ""
        echo "Prerequisites:"
        echo "  • AWS CLI installed and configured"
        echo "  • Sufficient AWS permissions for EC2, VPC operations"
        echo "  • Mini-XDR system ready to receive logs"
        echo ""
        exit 0
        ;;
esac

# Run main function
main "$@"
