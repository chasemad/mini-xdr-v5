#!/bin/bash
# AWS Honeypot Deployment Script for Mini-XDR
# Deploys multiple honeypots across AWS regions with optimal security configuration

set -e

# Configuration
REGIONS=("us-east-1" "eu-west-1" "ap-southeast-1")  # Diverse geographic coverage
INSTANCE_TYPE="t3.micro"  # Free tier eligible
AMI_ID="ami-0abcdef1234567890"  # Ubuntu 22.04 LTS (update for your region)
KEY_NAME="mini-xdr-keypair"  # Your AWS key pair name
YOUR_LOCAL_IP="YOUR_LOCAL_IP_HERE"  # Replace with your public IP

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

# Check prerequisites
check_aws_cli() {
    if ! command -v aws &> /dev/null; then
        error "AWS CLI not installed. Install from: https://aws.amazon.com/cli/"
        exit 1
    fi
    
    if ! aws sts get-caller-identity &> /dev/null; then
        error "AWS CLI not configured. Run: aws configure"
        exit 1
    fi
    
    success "AWS CLI configured and ready"
}

# Get Ubuntu 22.04 AMI for region
get_ubuntu_ami() {
    local region=$1
    aws ec2 describe-images \
        --region $region \
        --owners 099720109477 \
        --filters "Name=name,Values=ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*" \
        --query 'Images[*].[ImageId,CreationDate]' \
        --output text | sort -k2 -r | head -n1 | cut -f1
}

# Create security group for honeypots
create_security_group() {
    local region=$1
    local vpc_id=$(aws ec2 describe-vpcs --region $region --filters "Name=is-default,Values=true" --query 'Vpcs[0].VpcId' --output text)
    
    log "Creating security group in $region..."
    
    local sg_id=$(aws ec2 create-security-group \
        --region $region \
        --group-name "mini-xdr-honeypot-sg" \
        --description "Security group for Mini-XDR honeypots" \
        --vpc-id $vpc_id \
        --query 'GroupId' --output text 2>/dev/null || \
        aws ec2 describe-security-groups \
        --region $region \
        --filters "Name=group-name,Values=mini-xdr-honeypot-sg" \
        --query 'SecurityGroups[0].GroupId' --output text)
    
    if [ "$sg_id" != "None" ]; then
        # Configure security group rules
        log "Configuring security group rules..."
        
        # Allow SSH honeypot (port 22)
        aws ec2 authorize-security-group-ingress \
            --region $region \
            --group-id $sg_id \
            --protocol tcp \
            --port 22 \
            --cidr 0.0.0.0/0 \
            --output text 2>/dev/null || true
        
        # Allow web honeypots (ports 80, 443)
        aws ec2 authorize-security-group-ingress \
            --region $region \
            --group-id $sg_id \
            --protocol tcp \
            --port 80 \
            --cidr 0.0.0.0/0 \
            --output text 2>/dev/null || true
            
        aws ec2 authorize-security-group-ingress \
            --region $region \
            --group-id $sg_id \
            --protocol tcp \
            --port 443 \
            --cidr 0.0.0.0/0 \
            --output text 2>/dev/null || true
        
        # Allow SSH management from your IP only (port 22022)
        aws ec2 authorize-security-group-ingress \
            --region $region \
            --group-id $sg_id \
            --protocol tcp \
            --port 22022 \
            --cidr "$YOUR_LOCAL_IP/32" \
            --output text 2>/dev/null || true
        
        # Allow additional honeypot services
        for port in 21 23 25 53 110 143 993 995 1433 3306 5432; do
            aws ec2 authorize-security-group-ingress \
                --region $region \
                --group-id $sg_id \
                --protocol tcp \
                --port $port \
                --cidr 0.0.0.0/0 \
                --output text 2>/dev/null || true
        done
        
        success "Security group created: $sg_id"
        echo $sg_id
    else
        error "Failed to create security group in $region"
        exit 1
    fi
}

# Create user data script for honeypot setup
create_user_data() {
    cat << 'EOF'
#!/bin/bash
# AWS EC2 User Data Script for Mini-XDR Honeypot
set -e

# Update system
apt-get update -y
apt-get upgrade -y

# Install dependencies
apt-get install -y python3 python3-venv python3-pip git curl wget htop vim ufw fail2ban

# Configure firewall
ufw --force enable
ufw allow 22022/tcp comment "SSH management"
ufw allow 22/tcp comment "SSH honeypot" 
ufw allow 80/tcp comment "Web honeypot"
ufw allow 443/tcp comment "Web honeypot SSL"
ufw allow 21/tcp comment "FTP honeypot"
ufw allow 23/tcp comment "Telnet honeypot"

# Setup SSH on alternate port for management
sed -i 's/#Port 22/Port 22022/' /etc/ssh/sshd_config
sed -i 's/#PasswordAuthentication yes/PasswordAuthentication no/' /etc/ssh/sshd_config
systemctl restart sshd

# Create XDR operations user
useradd -m -s /bin/bash xdrops || echo "User xdrops already exists"
usermod -aG sudo xdrops

# Setup SSH directory for xdrops user
mkdir -p /home/xdrops/.ssh
chmod 700 /home/xdrops/.ssh
chown -R xdrops:xdrops /home/xdrops/.ssh

# Install Cowrie honeypot
cd /opt
git clone https://github.com/cowrie/cowrie.git
cd cowrie
python3 -m venv cowrie-env
source cowrie-env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Configure Cowrie
cp etc/cowrie.cfg.dist etc/cowrie.cfg

# Enable JSON logging
cat >> etc/cowrie.cfg << COWRIEEOF

[output_jsonlog]
enabled = true
logfile = var/log/cowrie/cowrie.json
epoch_timestamp = false

[output_elasticsearch]
enabled = false

[honeypot]
hostname = aws-honeypot
log_path = var/log/cowrie
COWRIEEOF

# Create Cowrie service
cat > /etc/systemd/system/cowrie.service << SERVICEEOF
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

# Create cowrie user
useradd -r -s /bin/false cowrie || echo "Cowrie user exists"
chown -R cowrie:cowrie /opt/cowrie
chmod +x /opt/cowrie/bin/cowrie

# Redirect port 22 to Cowrie port 2222
iptables -t nat -A PREROUTING -p tcp --dport 22 -j REDIRECT --to-port 2222
iptables-save > /etc/iptables/rules.v4

# Install iptables-persistent to save rules
DEBIAN_FRONTEND=noninteractive apt-get install -y iptables-persistent

# Start and enable Cowrie
systemctl daemon-reload
systemctl enable cowrie
systemctl start cowrie

# Install Fluent Bit for log forwarding
curl https://raw.githubusercontent.com/fluent/fluent-bit/master/install.sh | sh

# Configure log forwarding (will be updated with actual XDR IP)
cat > /etc/fluent-bit/fluent-bit.conf << FLUENTEOF
[SERVICE]
    Flush         1
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

[OUTPUT]
    Name  http
    Match cowrie
    Host  YOUR_LOCAL_IP_HERE
    Port  8000
    URI   /ingest/multi
    Format json
    Header Authorization Bearer honeypot-api-key
    Retry_Limit 5
FLUENTEOF

# Setup web honeypot (simple vulnerable PHP app)
apt-get install -y apache2 php php-mysql
systemctl enable apache2
systemctl start apache2

# Create vulnerable web app
cat > /var/www/html/login.php << WEBEOF
<?php
// Intentionally vulnerable login for honeypot
if (\$_POST['username'] && \$_POST['password']) {
    // Log all login attempts
    \$log = date('Y-m-d H:i:s') . " - Login attempt: " . \$_POST['username'] . ":" . \$_POST['password'] . " from " . \$_SERVER['REMOTE_ADDR'] . "\n";
    file_put_contents('/var/log/web-honeypot.log', \$log, FILE_APPEND);
    
    // Always fail but make it look real
    echo "Login failed";
} else {
?>
<html><body>
<h2>Admin Login</h2>
<form method="post">
Username: <input type="text" name="username"><br>
Password: <input type="password" name="password"><br>
<input type="submit" value="Login">
</form>
</body></html>
<?php } ?>
WEBEOF

# Create admin directory
mkdir -p /var/www/html/admin
echo "<?php phpinfo(); ?>" > /var/www/html/admin/info.php

# Setup log rotation
cat > /etc/logrotate.d/honeypot << LOGROTATEEOF
/opt/cowrie/var/log/cowrie/*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    copytruncate
}

/var/log/web-honeypot.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    copytruncate
}
LOGROTATEEOF

# Install CloudWatch agent for monitoring
wget https://s3.amazonaws.com/amazoncloudwatch-agent/ubuntu/amd64/latest/amazon-cloudwatch-agent.deb
dpkg -i amazon-cloudwatch-agent.deb

echo "AWS Honeypot setup completed!"
EOF
}

# Launch EC2 instance
launch_honeypot() {
    local region=$1
    local sg_id=$2
    local ami_id=$3
    local instance_name="mini-xdr-honeypot-$region"
    
    log "Launching honeypot instance in $region..."
    
    # Create user data with your local IP
    local user_data=$(create_user_data | sed "s/YOUR_LOCAL_IP_HERE/$YOUR_LOCAL_IP/g")
    
    # Launch instance
    local instance_id=$(aws ec2 run-instances \
        --region $region \
        --image-id $ami_id \
        --count 1 \
        --instance-type $INSTANCE_TYPE \
        --key-name $KEY_NAME \
        --security-group-ids $sg_id \
        --user-data "$user_data" \
        --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$instance_name},{Key=Project,Value=mini-xdr},{Key=Type,Value=honeypot}]" \
        --query 'Instances[0].InstanceId' \
        --output text)
    
    if [ "$instance_id" != "None" ]; then
        success "Instance launched: $instance_id in $region"
        
        # Wait for instance to be running
        log "Waiting for instance to be running..."
        aws ec2 wait instance-running --region $region --instance-ids $instance_id
        
        # Allocate and associate Elastic IP
        log "Allocating Elastic IP..."
        local allocation_id=$(aws ec2 allocate-address \
            --region $region \
            --domain vpc \
            --tag-specifications "ResourceType=elastic-ip,Tags=[{Key=Name,Value=$instance_name-ip},{Key=Project,Value=mini-xdr}]" \
            --query 'AllocationId' \
            --output text)
        
        aws ec2 associate-address \
            --region $region \
            --instance-id $instance_id \
            --allocation-id $allocation_id \
            --output text > /dev/null
        
        local public_ip=$(aws ec2 describe-addresses \
            --region $region \
            --allocation-ids $allocation_id \
            --query 'Addresses[0].PublicIp' \
            --output text)
        
        success "Honeypot deployed: $public_ip ($region)"
        echo "Instance: $instance_id"
        echo "Public IP: $public_ip"
        echo "Region: $region"
        echo "SSH: ssh -i ~/.ssh/$KEY_NAME.pem -p 22022 ubuntu@$public_ip"
        echo ""
        
        return 0
    else
        error "Failed to launch instance in $region"
        return 1
    fi
}

# Main deployment function
main() {
    echo "=== 🍯 AWS Honeypot Deployment for Mini-XDR ==="
    echo ""
    
    # Check prerequisites
    check_aws_cli
    
    if [ "$YOUR_LOCAL_IP" = "YOUR_LOCAL_IP_HERE" ]; then
        error "Please update YOUR_LOCAL_IP in the script with your public IP"
        echo "Get your IP: curl -4 icanhazip.com"
        exit 1
    fi
    
    echo "Deploying honeypots in regions: ${REGIONS[*]}"
    echo "Instance type: $INSTANCE_TYPE"
    echo "Your IP: $YOUR_LOCAL_IP"
    echo ""
    
    # Deploy to each region
    for region in "${REGIONS[@]}"; do
        log "=== Deploying to $region ==="
        
        # Get AMI ID for region
        ami_id=$(get_ubuntu_ami $region)
        if [ -z "$ami_id" ]; then
            error "Could not find Ubuntu AMI in $region"
            continue
        fi
        log "Using AMI: $ami_id"
        
        # Create security group
        sg_id=$(create_security_group $region)
        if [ -z "$sg_id" ]; then
            error "Could not create security group in $region"
            continue
        fi
        
        # Launch honeypot
        if launch_honeypot $region $sg_id $ami_id; then
            success "Deployment successful in $region"
        else
            error "Deployment failed in $region"
        fi
        
        echo ""
    done
    
    echo "=== 🎉 AWS Honeypot Deployment Complete ==="
    echo ""
    echo "Next steps:"
    echo "1. Wait 5-10 minutes for instances to fully initialize"
    echo "2. Test SSH access to each honeypot"
    echo "3. Start your local Mini-XDR: ./scripts/start-all.sh"
    echo "4. Verify log forwarding is working"
    echo "5. Run attack simulations or wait for real attacks"
    echo ""
    echo "Monitoring:"
    echo "- Check instance status: aws ec2 describe-instances --region REGION"
    echo "- SSH to honeypot: ssh -i ~/.ssh/$KEY_NAME.pem -p 22022 ubuntu@PUBLIC_IP"
    echo "- View logs: tail -f /opt/cowrie/var/log/cowrie/cowrie.json"
    echo ""
}

# Run main function
main "$@"

