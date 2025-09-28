#!/bin/bash
# Create AWS-based Mini-XDR relay to enable TPOT connectivity
# This creates a lightweight AWS instance that forwards TPOT logs to your local system

set -e

REGION="us-east-1"
INSTANCE_TYPE="t3.micro"  # Free tier eligible
KEY_NAME="mini-xdr-tpot-key"
SECURITY_GROUP="mini-xdr-relay-sg"

echo "🚀 Setting up AWS Mini-XDR Relay for TPOT connectivity"
echo "This enables TPOT → AWS Relay → Your Local Mini-XDR"
echo ""

# Check if AWS CLI is configured
if ! aws sts get-caller-identity >/dev/null 2>&1; then
    echo "❌ AWS CLI not configured. Please run: aws configure"
    exit 1
fi

# Get your current public IP
YOUR_IP=$(curl -s ifconfig.me)
echo "📍 Your public IP: $YOUR_IP"

# Create security group for relay
echo "🔒 Creating security group for relay..."
RELAY_SG_ID=$(aws ec2 create-security-group \
    --region $REGION \
    --group-name $SECURITY_GROUP \
    --description "Mini-XDR Relay Security Group" \
    --query 'GroupId' \
    --output text 2>/dev/null || \
    aws ec2 describe-security-groups \
    --region $REGION \
    --filters "Name=group-name,Values=$SECURITY_GROUP" \
    --query 'SecurityGroups[0].GroupId' \
    --output text)

echo "✅ Security group: $RELAY_SG_ID"

# Add security rules
echo "🔐 Configuring security rules..."

# Allow SSH from your IP
aws ec2 authorize-security-group-ingress \
    --region $REGION \
    --group-id $RELAY_SG_ID \
    --protocol tcp \
    --port 22 \
    --cidr "$YOUR_IP/32" 2>/dev/null || true

# Allow HTTP from TPOT
aws ec2 authorize-security-group-ingress \
    --region $REGION \
    --group-id $RELAY_SG_ID \
    --protocol tcp \
    --port 8000 \
    --cidr "34.193.101.171/32" 2>/dev/null || true

# Get latest Ubuntu AMI
echo "🔍 Getting latest Ubuntu AMI..."
AMI_ID=$(aws ec2 describe-images \
    --region $REGION \
    --owners 099720109477 \
    --filters "Name=name,Values=ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*" \
    --query 'Images | sort_by(@, &CreationDate) | [-1].ImageId' \
    --output text)

echo "✅ Ubuntu AMI: $AMI_ID"

# Create user data script
echo "📝 Creating relay configuration..."
cat > /tmp/relay-user-data.sh << 'EOF'
#!/bin/bash
set -e

# Update system
apt-get update -y
apt-get install -y nginx python3 python3-pip curl

# Install simple relay service
cat > /home/ubuntu/relay.py << 'RELAY_EOF'
#!/usr/bin/env python3
import http.server
import socketserver
import requests
import json
from urllib.parse import urlparse, parse_qs

LOCAL_MINIXDR_IP = "YOUR_LOCAL_IP_PLACEHOLDER"
LOCAL_PORT = 8000

class RelayHandler(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            # Read the request data
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            # Forward to local Mini-XDR
            local_url = f"http://{LOCAL_MINIXDR_IP}:{LOCAL_PORT}{self.path}"
            
            # Forward headers
            headers = {}
            for key, value in self.headers.items():
                if key.lower() not in ['host', 'content-length']:
                    headers[key] = value
            
            # Make request to local Mini-XDR
            response = requests.post(
                local_url,
                data=post_data,
                headers=headers,
                timeout=30
            )
            
            # Send response back
            self.send_response(response.status_code)
            for key, value in response.headers.items():
                if key.lower() not in ['content-length', 'transfer-encoding']:
                    self.send_header(key, value)
            self.end_headers()
            self.wfile.write(response.content)
            
        except Exception as e:
            self.send_response(500)
            self.end_headers()
            self.wfile.write(f"Relay error: {str(e)}".encode())

    def log_message(self, format, *args):
        # Log to file
        with open('/home/ubuntu/relay.log', 'a') as f:
            f.write(f"{self.log_date_time_string()} - {format % args}\n")

if __name__ == "__main__":
    PORT = 8000
    with socketserver.TCPServer(("", PORT), RelayHandler) as httpd:
        print(f"Relay server serving at port {PORT}")
        httpd.serve_forever()
RELAY_EOF

# Create systemd service
cat > /etc/systemd/system/minixdr-relay.service << 'SERVICE_EOF'
[Unit]
Description=Mini-XDR Relay Service
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu
ExecStart=/usr/bin/python3 /home/ubuntu/relay.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
SERVICE_EOF

# Don't start yet - need to configure IP first
systemctl daemon-reload

echo "✅ Mini-XDR Relay installed and configured"
EOF

# Launch instance
echo "🚀 Launching relay instance..."
INSTANCE_ID=$(aws ec2 run-instances \
    --region $REGION \
    --image-id $AMI_ID \
    --count 1 \
    --instance-type $INSTANCE_TYPE \
    --key-name $KEY_NAME \
    --security-group-ids $RELAY_SG_ID \
    --user-data file:///tmp/relay-user-data.sh \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=mini-xdr-relay}]" \
    --query 'Instances[0].InstanceId' \
    --output text)

echo "✅ Instance launched: $INSTANCE_ID"
echo "⏳ Waiting for instance to be ready..."

# Wait for instance to be running
aws ec2 wait instance-running --region $REGION --instance-ids $INSTANCE_ID

# Get instance IP
RELAY_IP=$(aws ec2 describe-instances \
    --region $REGION \
    --instance-ids $INSTANCE_ID \
    --query 'Reservations[0].Instances[0].PublicIpAddress' \
    --output text)

echo "✅ Relay instance ready at: $RELAY_IP"
echo ""
echo "🔧 Next steps:"
echo "1. Configure your local IP in the relay"
echo "2. Update TPOT to send logs to relay"
echo "3. Start the relay service"
echo ""
echo "Run: ./configure-relay.sh $RELAY_IP $YOUR_IP"
