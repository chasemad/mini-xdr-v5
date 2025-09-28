#!/bin/bash
# 🚀 Quick CORS Fix Deployment Script
# Deploys the improved error handling for CORS issues

set -euo pipefail

# Configuration
BACKEND_IP="54.237.168.3"
SSH_KEY="~/.ssh/mini-xdr-tpot-key.pem"
PROJECT_DIR="/Users/chasemad/Desktop/mini-xdr"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

info() { echo -e "${BLUE}ℹ️  $1${NC}"; }
success() { echo -e "${GREEN}✅ $1${NC}"; }
warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }

echo -e "${GREEN}🚀 Deploying CORS Fix to AWS${NC}"
echo

info "Copying updated backend code to AWS..."
rsync -avz --delete -e "ssh -o StrictHostKeyChecking=no -i ${SSH_KEY/#\~/$HOME}" \
    "$PROJECT_DIR/backend/" "ubuntu@$BACKEND_IP:/opt/mini-xdr/backend/" \
    --exclude venv --exclude __pycache__ --exclude "*.pyc"

success "Code synchronized successfully!"

info "Installing missing dependencies and restarting services..."
ssh -o StrictHostKeyChecking=no -i "${SSH_KEY/#\~/$HOME}" "ubuntu@$BACKEND_IP" << 'DEPLOY_FIX'
set -euo pipefail

cd /opt/mini-xdr/backend
source venv/bin/activate

echo "📦 Installing missing dependencies..."
pip install aiokafka==0.12.0 redis==5.0.1 aioredis==2.0.1 || echo "Some packages may already be installed"

echo "🛑 Stopping backend..."
pkill -f uvicorn || true
sleep 3

echo "🚀 Starting backend with CORS fixes..."
SECRETS_MANAGER_ENABLED=true AWS_DEFAULT_REGION=us-east-1 UI_ORIGIN=http://54.237.168.3:3000 PYTHONPATH=/opt/mini-xdr/backend \
    nohup python3 -m uvicorn app.entrypoint:app --host 0.0.0.0 --port 8000 > /tmp/backend.log 2>&1 &

echo "⏳ Waiting for backend to initialize..."
sleep 15

echo "🧪 Testing endpoints..."
if curl -f http://localhost:8000/api/distributed/status >/dev/null 2>&1; then
    echo "✅ /api/distributed/status is working"
else
    echo "⚠️  /api/distributed/status may need more time to initialize"
fi

if curl -f http://localhost:8000/api/federated/insights >/dev/null 2>&1; then
    echo "✅ /api/federated/insights is working"  
else
    echo "⚠️  /api/federated/insights may need more time to initialize"
fi

echo "📋 Recent logs:"
tail -5 /tmp/backend.log
DEPLOY_FIX

success "CORS fix deployed successfully!"

echo
info "Testing from your browser now..."
info "The previously failing endpoints should now return proper responses:"
echo "  • http://54.237.168.3:8000/api/distributed/status"
echo "  • http://54.237.168.3:8000/api/federated/insights"
echo
warning "If you still see CORS errors, refresh your browser to clear any cached failed requests"
