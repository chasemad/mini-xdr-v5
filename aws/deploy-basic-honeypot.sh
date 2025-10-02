#!/bin/bash
# 🍯 Basic AWS Honeypot Deployment for Mini-XDR Testing
# Creates a simple, secure honeypot that sends data to local Mini-XDR instance

set -e

# Configuration
REGION="us-east-1"
INSTANCE_TYPE="t3.micro"
AMI_ID="ami-0866a3c8686eaeeba"  # Ubuntu 24.04 LTS us-east-1
KEY_NAME="mini-xdr-test-key"
SECURITY_GROUP_NAME="mini-xdr-honeypot-test"

# Network Configuration  
LOCAL_MINI_XDR_IP=$(curl -s ipv4.icanhazip.com)  # Get IPv4 public IP where Mini-XDR is running
MINI_XDR_API_PORT="8000"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

log() { echo -e "${BLUE}[$(date +'%H:%M:%S')]${NC} $1"; }
success() { echo -e "${GREEN}✅ $1${NC}"; }
warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }
error() { echo -e "${RED}❌ $1${NC}"; }

echo -e "${BOLD}${PURPLE}"
echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║                   🍯 Basic AWS Honeypot for Mini-XDR Testing                ║"
echo "║              Simple, Secure Honeypot with Local Data Flow                   ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Check prerequisites
log "Checking prerequisites..."
if ! command -v aws &> /dev/null; then
    error "AWS CLI not found. Please install it first."
    exit 1
fi

if ! aws sts get-caller-identity &> /dev/null; then
    error "AWS credentials not configured. Run 'aws configure' first."
    exit 1
fi

success "AWS CLI configured and ready"
log "Your public IP: ${LOCAL_MINI_XDR_IP}"
log "Mini-XDR will receive data from honeypot at: ${LOCAL_MINI_XDR_IP}:${MINI_XDR_API_PORT}"

# Create key pair if it doesn't exist
log "Setting up SSH key pair..."
if ! aws ec2 describe-key-pairs --key-names "$KEY_NAME" &> /dev/null; then
    log "Creating new key pair: $KEY_NAME"
    aws ec2 create-key-pair --key-name "$KEY_NAME" --query 'KeyMaterial' --output text > ~/.ssh/${KEY_NAME}.pem
    chmod 600 ~/.ssh/${KEY_NAME}.pem
    success "Created SSH key: ~/.ssh/${KEY_NAME}.pem"
else
    success "SSH key pair already exists: $KEY_NAME"
fi

# Use existing VPC (avoid VPC limit issues)
log "Setting up VPC and security group..."
VPC_ID=$(aws ec2 describe-vpcs --filters "Name=state,Values=available" --query 'Vpcs[0].VpcId' --output text)
success "Using existing VPC: $VPC_ID"

# Get a public subnet from the VPC
SUBNET_ID=$(aws ec2 describe-subnets --filters "Name=vpc-id,Values=$VPC_ID" "Name=map-public-ip-on-launch,Values=true" --query 'Subnets[0].SubnetId' --output text 2>/dev/null || echo "None")

if [ "$SUBNET_ID" = "None" ]; then
    # Get any subnet and make it public-facing
    SUBNET_ID=$(aws ec2 describe-subnets --filters "Name=vpc-id,Values=$VPC_ID" --query 'Subnets[0].SubnetId' --output text)
    log "Using subnet: $SUBNET_ID (will associate public IP)"
else
    success "Using public subnet: $SUBNET_ID"
fi

# Create security group
SECURITY_GROUP_ID=$(aws ec2 describe-security-groups --filters "Name=group-name,Values=$SECURITY_GROUP_NAME" "Name=vpc-id,Values=$VPC_ID" --query 'SecurityGroups[0].GroupId' --output text 2>/dev/null || echo "None")

if [ "$SECURITY_GROUP_ID" = "None" ]; then
    log "Creating security group: $SECURITY_GROUP_NAME"
    SECURITY_GROUP_ID=$(aws ec2 create-security-group \
        --group-name "$SECURITY_GROUP_NAME" \
        --description "Basic honeypot security group for Mini-XDR testing" \
        --vpc-id "$VPC_ID" \
        --query 'GroupId' --output text)
    
    # Add security group rules
    log "Configuring security group rules..."
    
    # SSH access from your IP only
    aws ec2 authorize-security-group-ingress \
        --group-id "$SECURITY_GROUP_ID" \
        --protocol tcp \
        --port 22 \
        --cidr "${LOCAL_MINI_XDR_IP}/32"
    
    # Common honeypot ports (restricted to known attack sources)
    for port in 23 80 443 2222 3389; do
        aws ec2 authorize-security-group-ingress \
            --group-id "$SECURITY_GROUP_ID" \
            --protocol tcp \
            --port $port \
            --cidr "0.0.0.0/0"
    done
    
    # ICMP for ping
    aws ec2 authorize-security-group-ingress \
        --group-id "$SECURITY_GROUP_ID" \
        --protocol icmp \
        --port -1 \
        --cidr "0.0.0.0/0"
    
    success "Security group created: $SECURITY_GROUP_ID"
else
    success "Security group already exists: $SECURITY_GROUP_ID"
fi

# Create user data script for honeypot setup
log "Preparing honeypot installation script..."
cat > /tmp/honeypot-userdata.sh << 'EOF'
#!/bin/bash
exec > >(tee /var/log/user-data.log) 2>&1

echo "Starting Mini-XDR Honeypot Setup..."
date

# Update system
apt-get update
apt-get upgrade -y

# Install prerequisites
apt-get install -y curl wget git python3 python3-pip docker.io jq

# Start Docker
systemctl start docker
systemctl enable docker
usermod -aG docker ubuntu

# Install basic honeypot services
log() { echo "[$(date +'%H:%M:%S')] $1"; }

log "Installing SSH honeypot (Cowrie)..."
mkdir -p /opt/cowrie
cd /opt/cowrie

# Production-grade Python SSH honeypot with HMAC authentication
cat > honeypot.py << 'PYTHON_EOF'
#!/usr/bin/env python3
import socket
import threading
import json
import time
import requests
import hashlib
import hmac
import uuid
from datetime import datetime, timezone
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Mini-XDR API Configuration with HMAC Authentication
MINI_XDR_API = "MINI_XDR_ENDPOINT_PLACEHOLDER"
DEVICE_ID = "ffb56f4f-b0c8-4258-8922-0f976e536a7b"
HMAC_KEY = "678aae7bdf9e61cbb5fd059f0c774baf6d3143495cd091b2759265fe15c0beb3"
API_KEY = "demo-minixdr-api-key"

class SimpleHoneypot:
    def __init__(self, port=2222):
        self.port = port
        self.socket = None
        
    def build_hmac_headers(self, method, path, body):
        """Build HMAC authentication headers for Mini-XDR"""
        timestamp = str(int(datetime.now(timezone.utc).timestamp()))
        nonce = str(uuid.uuid4())
        
        # Build canonical message
        canonical_message = "|".join([method.upper(), path, body, timestamp, nonce])
        
        # Generate HMAC signature
        signature = hmac.new(
            HMAC_KEY.encode('utf-8'),
            canonical_message.encode('utf-8'), 
            hashlib.sha256
        ).hexdigest()
        
        return {
            'X-Device-ID': DEVICE_ID,
            'X-TS': timestamp,
            'X-Nonce': nonce,
            'X-Signature': signature,
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {API_KEY}'
        }
        
    def send_to_mini_xdr(self, event_data):
        """Send event to Mini-XDR API with proper HMAC authentication"""
        try:
            # Format as multi-source payload for /ingest/multi endpoint
            payload = {
                'source_type': 'cowrie',
                'hostname': 'aws-honeypot-001',
                'events': [{
                    'eventid': event_data['event_type'],
                    'src_ip': event_data['src_ip'],
                    'dst_ip': event_data.get('dst_ip', ''),
                    'dst_port': event_data['dst_port'],
                    'message': event_data['message'],
                    'raw': event_data['raw_data'],
                    'timestamp': event_data['timestamp']
                }]
            }
            
            body = json.dumps(payload)
            headers = self.build_hmac_headers('POST', '/ingest/multi', body)
            
            response = requests.post(
                f"{MINI_XDR_API}/ingest/multi",
                data=body,
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                logger.info(f"✅ Sent event to Mini-XDR: {event_data['event_type']}")
                result = response.json()
                logger.info(f"   Response: {result.get('processed', 0)} events processed")
            else:
                logger.warning(f"⚠️ Mini-XDR API error: {response.status_code}")
                logger.warning(f"   Response: {response.text}")
                
        except Exception as e:
            logger.error(f"❌ Failed to send to Mini-XDR: {e}")
    
    def handle_connection(self, client_socket, addr):
        """Handle incoming connection"""
        try:
            src_ip = addr[0]
            logger.info(f"🎯 New connection from {src_ip}")
            
            # Send SSH banner
            client_socket.send(b"SSH-2.0-OpenSSH_8.9p1 Ubuntu-3ubuntu0.6\r\n")
            
            # Read client data
            data = client_socket.recv(1024)
            
            # Log the attempt
            event_data = {
                'src_ip': src_ip,
                'dst_ip': '10.0.1.100',  # Honeypot internal IP
                'dst_port': self.port,
                'event_type': 'cowrie.session.connect',
                'message': f'SSH connection attempt from {src_ip}',
                'timestamp': datetime.utcnow().isoformat() + 'Z',
                'raw_data': {
                    'session': f'session_{int(time.time())}',
                    'src_ip': src_ip,
                    'src_port': addr[1],
                    'data_received': data.decode('utf-8', errors='ignore')[:100]
                }
            }
            
            # Send to Mini-XDR
            self.send_to_mini_xdr(event_data)
            
            # Simulate authentication attempts
            time.sleep(0.5)
            client_socket.send(b"Password: ")
            
            auth_data = client_socket.recv(1024)
            
            # Log failed authentication
            auth_event = {
                'src_ip': src_ip,
                'dst_ip': '10.0.1.100',
                'dst_port': self.port,
                'event_type': 'cowrie.login.failed',
                'message': f'Failed SSH login from {src_ip}',
                'timestamp': datetime.utcnow().isoformat() + 'Z',
                'raw_data': {
                    'username': 'admin',
                    'password': auth_data.decode('utf-8', errors='ignore').strip()[:20],
                    'src_ip': src_ip
                }
            }
            
            self.send_to_mini_xdr(auth_event)
            
        except Exception as e:
            logger.error(f"Connection handling error: {e}")
        finally:
            client_socket.close()
    
    def start(self):
        """Start the honeypot"""
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind(('0.0.0.0', self.port))
        self.socket.listen(5)
        
        logger.info(f"🍯 Honeypot listening on port {self.port}")
        logger.info(f"🎯 Sending data to Mini-XDR: {MINI_XDR_API}")
        
        while True:
            try:
                client_socket, addr = self.socket.accept()
                thread = threading.Thread(target=self.handle_connection, args=(client_socket, addr))
                thread.daemon = True
                thread.start()
            except Exception as e:
                logger.error(f"Server error: {e}")

if __name__ == "__main__":
    honeypot = SimpleHoneypot()
    honeypot.start()
PYTHON_EOF

# Replace placeholder with actual Mini-XDR endpoint
sed -i "s|MINI_XDR_ENDPOINT_PLACEHOLDER|http://${LOCAL_MINI_XDR_IP}:${MINI_XDR_API_PORT}|g" honeypot.py

chmod +x honeypot.py

log "Installing Python dependencies..."
pip3 install requests

log "Starting honeypot service..."
# Run honeypot in background
nohup python3 honeypot.py > /var/log/honeypot.log 2>&1 &

log "Setting up HTTP honeypot on port 80..."
# Simple HTTP honeypot
cat > http_honeypot.py << 'HTTP_EOF'
#!/usr/bin/env python3
import socket
import threading
import json
import requests
import hashlib
import hmac
import uuid
from datetime import datetime, timezone
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MINI_XDR_API = "MINI_XDR_ENDPOINT_PLACEHOLDER"
DEVICE_ID = "ffb56f4f-b0c8-4258-8922-0f976e536a7b"
HMAC_KEY = "678aae7bdf9e61cbb5fd059f0c774baf6d3143495cd091b2759265fe15c0beb3"
API_KEY = "demo-minixdr-api-key"

def build_hmac_headers(method, path, body):
    """Build HMAC authentication headers for Mini-XDR"""
    timestamp = str(int(datetime.now(timezone.utc).timestamp()))
    nonce = str(uuid.uuid4())
    
    # Build canonical message
    canonical_message = "|".join([method.upper(), path, body, timestamp, nonce])
    
    # Generate HMAC signature
    signature = hmac.new(
        HMAC_KEY.encode('utf-8'),
        canonical_message.encode('utf-8'), 
        hashlib.sha256
    ).hexdigest()
    
    return {
        'X-Device-ID': DEVICE_ID,
        'X-TS': timestamp,
        'X-Nonce': nonce,
        'X-Signature': signature,
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {API_KEY}'
    }

def send_to_mini_xdr(event_data):
    try:
        # Format as multi-source payload
        payload = {
            'source_type': 'webhoneypot',
            'hostname': 'aws-honeypot-001',
            'events': [event_data]
        }
        
        body = json.dumps(payload)
        headers = build_hmac_headers('POST', '/ingest/multi', body)
        
        response = requests.post(f"{MINI_XDR_API}/ingest/multi", data=body, headers=headers, timeout=10)
        
        if response.status_code == 200:
            logger.info(f"✅ Sent HTTP event: {event_data['eventid']}")
        else:
            logger.warning(f"⚠️ HTTP event failed: {response.status_code} - {response.text}")
    except Exception as e:
        logger.error(f"Failed to send HTTP event: {e}")

def handle_http_request(client_socket, addr):
    try:
        data = client_socket.recv(4096).decode('utf-8', errors='ignore')
        
        # Parse HTTP request
        lines = data.split('\n')
        if lines:
            request_line = lines[0]
            method, path, _ = request_line.split(' ', 2)
            
            # Detect attack patterns
            suspicious_patterns = ['union', 'select', 'script', 'alert', '../', 'etc/passwd']
            attack_detected = any(pattern in path.lower() for pattern in suspicious_patterns)
            
            event_data = {
                'src_ip': addr[0],
                'dst_ip': '10.0.1.100',
                'dst_port': 80,
                'eventid': 'webhoneypot.attack' if attack_detected else 'webhoneypot.request',
                'message': f'{method} {path}',
                'raw': {
                    'method': method,
                    'path': path,
                    'user_agent': next((line.split(': ', 1)[1] for line in lines if line.startswith('User-Agent:')), ''),
                    'attack_detected': attack_detected
                },
                'timestamp': datetime.utcnow().isoformat() + 'Z'
            }
            
            send_to_mini_xdr(event_data)
            
            # Send basic HTTP response
            response = "HTTP/1.1 200 OK\r\n\r\n<html><body><h1>Welcome</h1></body></html>"
            client_socket.send(response.encode())
            
    except Exception as e:
        logger.error(f"HTTP handler error: {e}")
    finally:
        client_socket.close()

def start_http_honeypot():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(('0.0.0.0', 80))
    server.listen(5)
    
    logger.info("🌐 HTTP Honeypot listening on port 80")
    
    while True:
        client_socket, addr = server.accept()
        thread = threading.Thread(target=handle_http_request, args=(client_socket, addr))
        thread.daemon = True
        thread.start()

if __name__ == "__main__":
    start_http_honeypot()
HTTP_EOF

sed -i "s|MINI_XDR_ENDPOINT_PLACEHOLDER|http://${LOCAL_MINI_XDR_IP}:${MINI_XDR_API_PORT}|g" http_honeypot.py
chmod +x http_honeypot.py

# Start HTTP honeypot
nohup python3 http_honeypot.py > /var/log/http-honeypot.log 2>&1 &

log "Setting up log forwarding to Mini-XDR..."
# Create status endpoint
cat > status.py << 'STATUS_EOF'
#!/usr/bin/env python3
from http.server import HTTPServer, BaseHTTPRequestHandler
import json

class StatusHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        
        status = {
            "honeypot_status": "active",
            "services": ["ssh:2222", "http:80"],
            "mini_xdr_endpoint": "MINI_XDR_ENDPOINT_PLACEHOLDER",
            "uptime": "running"
        }
        
        self.wfile.write(json.dumps(status).encode())

if __name__ == "__main__":
    server = HTTPServer(('0.0.0.0', 8080), StatusHandler)
    server.serve_forever()
STATUS_EOF

sed -i "s|MINI_XDR_ENDPOINT_PLACEHOLDER|http://${LOCAL_MINI_XDR_IP}:${MINI_XDR_API_PORT}|g" status.py
chmod +x status.py

# Start status service
nohup python3 status.py > /var/log/status.log 2>&1 &

echo "✅ Honeypot services started successfully"
echo "🍯 SSH Honeypot: port 2222"
echo "🌐 HTTP Honeypot: port 80"  
echo "📊 Status API: port 8080"
echo "🎯 Sending data to: http://${LOCAL_MINI_XDR_IP}:${MINI_XDR_API_PORT}"
echo "📝 Logs: /var/log/honeypot.log, /var/log/http-honeypot.log"

# Signal completion
touch /tmp/honeypot-setup-complete
EOF

# Get IPv4 address for configuration
log "Configuring honeypot to send data to: $LOCAL_MINI_XDR_IP"

# Launch EC2 instance
log "Launching EC2 instance for honeypot..."

INSTANCE_ID=$(aws ec2 run-instances \
    --image-id "$AMI_ID" \
    --count 1 \
    --instance-type "$INSTANCE_TYPE" \
    --key-name "$KEY_NAME" \
    --security-group-ids "$SECURITY_GROUP_ID" \
    --subnet-id "$SUBNET_ID" \
    --associate-public-ip-address \
    --user-data file:///tmp/honeypot-userdata.sh \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=mini-xdr-basic-honeypot},{Key=Project,Value=mini-xdr},{Key=Environment,Value=test}]" \
    --query 'Instances[0].InstanceId' \
    --output text)

if [ -z "$INSTANCE_ID" ]; then
    error "Failed to launch EC2 instance"
    exit 1
fi

success "Launched honeypot instance: $INSTANCE_ID"

# Wait for instance to start
log "Waiting for instance to start..."
aws ec2 wait instance-running --instance-ids "$INSTANCE_ID"

# Get public IP
PUBLIC_IP=$(aws ec2 describe-instances \
    --instance-ids "$INSTANCE_ID" \
    --query 'Reservations[0].Instances[0].PublicIpAddress' \
    --output text)

success "Honeypot is ready!"
echo ""
echo -e "${BOLD}${GREEN}🍯 HONEYPOT DEPLOYMENT SUCCESSFUL${NC}"
echo "=================================================="
echo "Instance ID: $INSTANCE_ID"
echo "Public IP: $PUBLIC_IP"
echo "SSH Access: ssh -i ~/.ssh/${KEY_NAME}.pem ubuntu@${PUBLIC_IP}"
echo "Status: http://${PUBLIC_IP}:8080"
echo ""
echo -e "${CYAN}📊 Testing Instructions:${NC}"
echo "1. Wait 2-3 minutes for honeypot services to start"
echo "2. Test SSH: ssh admin@${PUBLIC_IP} -p 2222"
echo "3. Test HTTP: curl http://${PUBLIC_IP}/admin.php"
echo "4. Check Mini-XDR incidents: http://localhost:3000/incidents"
echo ""
echo -e "${YELLOW}⚠️  Remember to terminate when done:${NC}"
echo "aws ec2 terminate-instances --instance-ids $INSTANCE_ID"
echo ""
echo -e "${BLUE}📝 Logs:${NC}"
echo "Honeypot logs: ssh -i ~/.ssh/${KEY_NAME}.pem ubuntu@${PUBLIC_IP} 'tail -f /var/log/honeypot.log'"
echo "Setup logs: ssh -i ~/.ssh/${KEY_NAME}.pem ubuntu@${PUBLIC_IP} 'tail -f /var/log/user-data.log'"

# Save instance info
echo "$INSTANCE_ID" > /tmp/mini-xdr-honeypot-instance.txt
echo "$PUBLIC_IP" > /tmp/mini-xdr-honeypot-ip.txt

log "Deployment complete. Instance details saved to /tmp/mini-xdr-honeypot-*.txt"
