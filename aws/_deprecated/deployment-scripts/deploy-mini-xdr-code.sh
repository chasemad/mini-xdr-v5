#!/bin/bash
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Configuration
SSH_KEY="~/.ssh/mini-xdr-tpot-key.pem"
BACKEND_IP="98.81.155.222"
PROJECT_DIR="/Users/chasemad/Desktop/mini-xdr"

echo -e "${BLUE}🚀 Mini-XDR Code Deployment to AWS${NC}"
echo "=============================================="

# Expand SSH key path
expanded_ssh_key="${SSH_KEY/#\~/$HOME}"
if [[ ! -f "$expanded_ssh_key" ]]; then
    echo -e "${RED}❌ SSH key not found: $SSH_KEY${NC}"
    exit 1
fi

echo -e "${GREEN}✅ SSH key found: $expanded_ssh_key${NC}"

# Test SSH connection
echo -e "${BLUE}🔍 Testing SSH connection...${NC}"
if ! ssh -i "$expanded_ssh_key" -o ConnectTimeout=10 -o StrictHostKeyChecking=no ubuntu@$BACKEND_IP "echo 'SSH connection successful'" >/dev/null 2>&1; then
    echo -e "${RED}❌ SSH connection failed to $BACKEND_IP${NC}"
    exit 1
fi
echo -e "${GREEN}✅ SSH connection successful${NC}"

# Create project directory on remote server
echo -e "${BLUE}📁 Creating project directory on server...${NC}"
ssh -i "$expanded_ssh_key" -o StrictHostKeyChecking=no ubuntu@$BACKEND_IP "
    sudo mkdir -p /opt/mini-xdr
    sudo chown ubuntu:ubuntu /opt/mini-xdr
"

# Copy backend code
echo -e "${BLUE}📦 Copying backend code...${NC}"
rsync -avz -e "ssh -i $expanded_ssh_key -o StrictHostKeyChecking=no" \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.env*' \
    --exclude='*.log' \
    --exclude='.venv' \
    --exclude='venv' \
    --exclude='*.db' \
    --exclude='xdr.db' \
    "$PROJECT_DIR/backend/" ubuntu@$BACKEND_IP:/opt/mini-xdr/backend/

# Copy frontend code
echo -e "${BLUE}📦 Copying frontend code...${NC}"
rsync -avz -e "ssh -i $expanded_ssh_key -o StrictHostKeyChecking=no" \
    --exclude='node_modules' \
    --exclude='.next' \
    --exclude='*.log' \
    --exclude='.env*' \
    "$PROJECT_DIR/frontend/" ubuntu@$BACKEND_IP:/opt/mini-xdr/frontend/

# Copy configuration files
echo -e "${BLUE}⚙️ Setting up configuration...${NC}"
ssh -i "$expanded_ssh_key" -o StrictHostKeyChecking=no ubuntu@$BACKEND_IP "
    cd /opt/mini-xdr

    # Create backend .env file
    cat > backend/.env << 'EOF'
OPENAI_API_KEY=your-openai-key-here
XAI_API_KEY=your-xai-key-here
DATABASE_URL=sqlite:///./mini_xdr.db
SECURITY_SECRET_KEY=mini-xdr-2024-secure-production-key
CORS_ORIGINS=http://98.81.155.222:3000,http://localhost:3000
LOG_LEVEL=INFO
ENVIRONMENT=production
EOF

    # Create frontend .env.local file
    cat > frontend/.env.local << 'EOF'
NEXT_PUBLIC_API_URL=http://98.81.155.222:8000
NEXT_PUBLIC_API_KEY=demo-minixdr-api-key
EOF

    # Set proper permissions
    chmod 600 backend/.env frontend/.env.local
"

# Install system dependencies
echo -e "${BLUE}📋 Installing system dependencies...${NC}"
ssh -i "$expanded_ssh_key" -o StrictHostKeyChecking=no ubuntu@$BACKEND_IP "
    sudo apt update -qq

    # Install Python dependencies
    if ! command -v pip3 &> /dev/null; then
        echo 'Installing pip3...'
        sudo apt install -y python3-pip
    fi

    # Install Node.js if not present
    if ! command -v node &> /dev/null; then
        echo 'Installing Node.js...'
        curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
        sudo apt-get install -y nodejs
    fi
"

# Install Python dependencies
echo -e "${BLUE}🐍 Installing Python dependencies...${NC}"
ssh -i "$expanded_ssh_key" -o StrictHostKeyChecking=no ubuntu@$BACKEND_IP "
    cd /opt/mini-xdr/backend
    pip3 install --user -r requirements.txt
"

# Install Node.js dependencies and build frontend
echo -e "${BLUE}📦 Installing Node.js dependencies and building frontend...${NC}"
ssh -i "$expanded_ssh_key" -o StrictHostKeyChecking=no ubuntu@$BACKEND_IP "
    cd /opt/mini-xdr/frontend
    npm install
    npm run build
"

# Create and start services
echo -e "${BLUE}🚀 Starting Mini-XDR services...${NC}"
ssh -i "$expanded_ssh_key" -o StrictHostKeyChecking=no ubuntu@$BACKEND_IP "
    cd /opt/mini-xdr

    # Stop any existing services
    pkill -f 'uvicorn.*main:app' || true
    pkill -f 'next.*start' || true

    # Start backend
    cd backend
    nohup python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000 > /tmp/mini-xdr-backend.log 2>&1 &

    # Wait for backend to start
    sleep 5

    # Start frontend
    cd ../frontend
    nohup npm start > /tmp/mini-xdr-frontend.log 2>&1 &

    # Wait for services to initialize
    sleep 10
"

# Verify services
echo -e "${BLUE}✅ Verifying services...${NC}"
if curl -s http://$BACKEND_IP:8000/health >/dev/null; then
    echo -e "${GREEN}✅ Backend service is running and healthy${NC}"
else
    echo -e "${YELLOW}⚠️  Backend service check failed, checking logs...${NC}"
    ssh -i "$expanded_ssh_key" -o StrictHostKeyChecking=no ubuntu@$BACKEND_IP "tail -20 /tmp/mini-xdr-backend.log"
fi

if curl -s http://$BACKEND_IP:3000 >/dev/null; then
    echo -e "${GREEN}✅ Frontend service is running${NC}"
else
    echo -e "${YELLOW}⚠️  Frontend service check failed, checking logs...${NC}"
    ssh -i "$expanded_ssh_key" -o StrictHostKeyChecking=no ubuntu@$BACKEND_IP "tail -20 /tmp/mini-xdr-frontend.log"
fi

echo -e "${GREEN}🎉 Mini-XDR deployment completed!${NC}"
echo "=============================================="
echo -e "Backend API: ${BLUE}http://$BACKEND_IP:8000${NC}"
echo -e "Frontend UI: ${BLUE}http://$BACKEND_IP:3000${NC}"
echo -e "Health Check: ${BLUE}http://$BACKEND_IP:8000/health${NC}"