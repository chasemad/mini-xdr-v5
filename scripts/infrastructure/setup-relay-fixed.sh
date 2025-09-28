#!/bin/bash
# Fixed AWS relay setup with proper VPC and IPv4 handling

set -e

REGION="us-east-1"
INSTANCE_TYPE="t3.micro"
KEY_NAME="mini-xdr-tpot-key"
SECURITY_GROUP="mini-xdr-relay-sg"
VPC_ID="vpc-0d7f8e7006dea45c5"
YOUR_IPV4="24.11.0.176"

echo "🚀 Setting up AWS Mini-XDR Relay (Fixed Version)"
echo "📍 Your IPv4: $YOUR_IPV4"
echo "🌐 VPC ID: $VPC_ID"
echo ""

# Create security group
echo "🔒 Creating security group for relay..."
RELAY_SG_ID=$(aws ec2 create-security-group \
    --region $REGION \
    --group-name $SECURITY_GROUP-$(date +%s) \
    --description "Mini-XDR Relay Security Group" \
    --vpc-id $VPC_ID \
    --query 'GroupId' \
    --output text)

echo "✅ Security group created: $RELAY_SG_ID"

# Add security rules
echo "🔐 Adding security rules..."

# SSH from your IP
aws ec2 authorize-security-group-ingress \
    --region $REGION \
    --group-id $RELAY_SG_ID \
    --protocol tcp \
    --port 22 \
    --cidr "$YOUR_IPV4/32" || true

# HTTP from TPOT
aws ec2 authorize-security-group-ingress \
    --region $REGION \
    --group-id $RELAY_SG_ID \
    --protocol tcp \
    --port 8000 \
    --cidr "34.193.101.171/32" || true

echo "✅ Security rules configured"

# Get Ubuntu AMI
AMI_ID=$(aws ec2 describe-images \
    --region $REGION \
    --owners 099720109477 \
    --filters "Name=name,Values=ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*" \
    --query 'Images | sort_by(@, &CreationDate) | [-1].ImageId' \
    --output text)

echo "✅ Ubuntu AMI: $AMI_ID"

# Create user data for relay service
cat > /tmp/relay-user-data.sh << 'EOF'
#!/bin/bash
set -e

apt-get update -y
apt-get install -y python3 python3-pip

pip3 install requests

# Create relay service
cat > /home/ubuntu/relay.py << 'RELAY_EOF'
#!/usr/bin/env python3
import http.server
import socketserver
import requests
import json
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/home/ubuntu/relay.log'),
        logging.StreamHandler()
    ]
)

LOCAL_MINIXDR_IP = "24.11.0.176"
LOCAL_PORT = 8000

class RelayHandler(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length)
            
            # Log incoming request
            logging.info(f"Received {len(post_data)} bytes from {self.client_address[0]}")
            
            # Forward to local Mini-XDR
            local_url = f"http://{LOCAL_MINIXDR_IP}:{LOCAL_PORT}{self.path}"
            
            headers = {}
            for key, value in self.headers.items():
                if key.lower() not in ['host', 'content-length']:
                    headers[key] = value
            
            logging.info(f"Forwarding to {local_url}")
            
            response = requests.post(
                local_url,
                data=post_data,
                headers=headers,
                timeout=30
            )
            
            logging.info(f"Local response: {response.status_code}")
            
            # Send response back
            self.send_response(response.status_code)
            for key, value in response.headers.items():
                if key.lower() not in ['content-length', 'transfer-encoding']:
                    self.send_header(key, value)
            self.end_headers()
            self.wfile.write(response.content)
            
        except Exception as e:
            logging.error(f"Relay error: {e}")
            self.send_response(500)
            self.end_headers()
            self.wfile.write(f"Relay error: {str(e)}".encode())

    def log_message(self, format, *args):
        pass  # Use our custom logging instead

if __name__ == "__main__":
    PORT = 8000
    logging.info(f"Starting relay server on port {PORT}")
    logging.info(f"Forwarding to {LOCAL_MINIXDR_IP}:{LOCAL_PORT}")
    
    with socketserver.TCPServer(("", PORT), RelayHandler) as httpd:
        httpd.serve_forever()
RELAY_EOF

chmod +x /home/ubuntu/relay.py

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

systemctl daemon-reload
systemctl enable minixdr-relay
systemctl start minixdr-relay

echo "✅ Mini-XDR Relay service started"
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
    --subnet-id $(aws ec2 describe-subnets --region $REGION --filters "Name=vpc-id,Values=$VPC_ID" --query 'Subnets[0].SubnetId' --output text) \
    --user-data file:///tmp/relay-user-data.sh \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=mini-xdr-relay}]" \
    --query 'Instances[0].InstanceId' \
    --output text)

echo "✅ Instance launching: $INSTANCE_ID"
echo "⏳ Waiting for instance to be ready..."

aws ec2 wait instance-running --region $REGION --instance-ids $INSTANCE_ID

RELAY_IP=$(aws ec2 describe-instances \
    --region $REGION \
    --instance-ids $INSTANCE_ID \
    --query 'Reservations[0].Instances[0].PublicIpAddress' \
    --output text)

echo ""
echo "🎉 RELAY SETUP COMPLETE!"
echo "📡 Relay IP: $RELAY_IP"
echo "🏠 Your IP: $YOUR_IPV4"
echo ""
echo "🔧 Now configuring TPOT connection..."

# Update TPOT Fluent Bit configuration
sed "s/10\.0\.0\.222/$RELAY_IP/" config/tpot/fluent-bit-tpot.conf > /tmp/fluent-bit-relay.conf
sed -i "s/\$TPOT_API_KEY/6c49b95dd921e0003ce159e6b3c0b6eb4e126fc2b19a1530a0f72a4a9c0c1eee/" /tmp/fluent-bit-relay.conf

echo "📡 Deploying new config to TPOT..."
scp -i ~/.ssh/mini-xdr-tpot-key.pem -P 64295 -o StrictHostKeyChecking=yes -o UserKnownHostsFile=~/.ssh/known_hosts /tmp/fluent-bit-relay.conf admin@34.193.101.171:/tmp/

ssh -i ~/.ssh/mini-xdr-tpot-key.pem -p 64295 -o StrictHostKeyChecking=yes -o UserKnownHostsFile=~/.ssh/known_hosts admin@34.193.101.171 << 'TPOT_CONFIG'
sudo cp /tmp/fluent-bit-relay.conf /etc/fluent-bit/fluent-bit.conf
sudo systemctl restart tpot-fluent-bit
echo "✅ TPOT Fluent Bit restarted with relay configuration"
sudo systemctl status tpot-fluent-bit --no-pager
TPOT_CONFIG

echo ""
echo "🎉 COMPLETE! TPOT → RELAY → MINI-XDR CONNECTION ESTABLISHED!"
echo ""
echo "📊 Data Flow:"
echo "   TPOT (34.193.101.171) → AWS Relay ($RELAY_IP) → Your Mini-XDR ($YOUR_IPV4:8000)"
echo ""
echo "✅ Your system now receives real attack data for:"
echo "   🤖 ML Models & AI Agents"  
echo "   🌍 Globe Visualization"
echo "   📊 Training Data Collection"
echo ""
echo "🔍 Monitor activity:"
echo "   Relay: ssh -i ~/.ssh/mini-xdr-tpot-key.pem ubuntu@$RELAY_IP 'tail -f /home/ubuntu/relay.log'"
echo "   Backend: tail -f backend/backend.log"
