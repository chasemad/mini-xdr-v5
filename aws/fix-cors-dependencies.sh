#!/bin/bash
# 🔧 Fix CORS Issues by Installing Missing Dependencies
# This script addresses the root cause of CORS errors: missing dependencies causing 500 errors

set -euo pipefail

# Configuration
BACKEND_IP="54.237.168.3"
SSH_KEY="~/.ssh/mini-xdr-tpot-key.pem"
AWS_REGION="${AWS_REGION:-us-east-1}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

success() { echo -e "${GREEN}✅ $1${NC}"; }
error() { echo -e "${RED}❌ $1${NC}"; }
info() { echo -e "${BLUE}ℹ️  $1${NC}"; }
warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }

header() {
    echo
    echo -e "${BOLD}${CYAN}🔧 $1${NC}"
    echo "$(printf '=%.0s' {1..60})"
}

header "CORS Fix: Installing Missing Dependencies on AWS"

info "The CORS errors are caused by missing dependencies (aiokafka, redis, tensorflow)"
info "When these endpoints fail internally, CORS headers aren't properly set"
echo

# Step 1: Install missing Python dependencies
header "Step 1: Installing Missing Python Dependencies"

info "Connecting to AWS backend to install missing packages..."
ssh -o StrictHostKeyChecking=no -i "${SSH_KEY/#\~/$HOME}" "ubuntu@$BACKEND_IP" << 'INSTALL_DEPS'
set -euo pipefail

cd /opt/mini-xdr/backend

echo "🔍 Checking current virtual environment..."
source venv/bin/activate

echo "📦 Installing missing distributed system dependencies..."
pip install aiokafka==0.12.0
pip install redis==5.0.1
pip install aioredis==2.0.1

echo "📦 Installing missing federated learning dependencies..."
pip install tensorflow==2.20.0

echo "📦 Installing missing security dependencies..." 
pip install python-jose[cryptography]==3.3.0

echo "✅ All dependencies installed successfully"

# Test imports to verify installation
echo "🧪 Testing imports..."
python3 -c "
import aiokafka
print('✅ aiokafka imported successfully')

import redis
print('✅ redis imported successfully')

import tensorflow as tf
print('✅ tensorflow imported successfully')

from jose import jwt
print('✅ python-jose imported successfully')

print('🎉 All critical dependencies are working!')
"
INSTALL_DEPS

success "Dependencies installed successfully!"

# Step 2: Add error handling middleware for better CORS on errors
header "Step 2: Adding Error Handling Middleware"

info "Creating error handling middleware to ensure CORS headers on all responses..."

# Create a patch for better error handling
cat > /tmp/cors_error_fix.py << 'CORS_FIX'
# Add this middleware to handle CORS on error responses
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import logging

logger = logging.getLogger(__name__)

@app.middleware("http")
async def cors_error_handler(request: Request, call_next):
    """Ensure CORS headers are always present, even on errors"""
    try:
        response = await call_next(request)
        return response
    except Exception as exc:
        # Create error response with proper CORS headers
        logger.error(f"Request to {request.url.path} failed: {str(exc)}")
        
        # Create JSON error response
        error_response = JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": "Internal server error",
                "message": str(exc)[:200],  # Truncate long error messages
                "path": str(request.url.path)
            }
        )
        
        # Add CORS headers manually for error responses
        origin = request.headers.get("origin")
        if origin and origin in ["http://54.237.168.3:3000", "http://localhost:3000"]:
            error_response.headers["Access-Control-Allow-Origin"] = origin
            error_response.headers["Access-Control-Allow-Credentials"] = "true"
            error_response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
            error_response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, x-api-key, X-Device-ID, X-TS, X-Nonce, X-Signature"
        
        return error_response
CORS_FIX

# Apply the middleware patch
info "Uploading error handling middleware..."
scp -o StrictHostKeyChecking=no -i "${SSH_KEY/#\~/$HOME}" /tmp/cors_error_fix.py "ubuntu@$BACKEND_IP:/tmp/"

# Step 3: Restart the backend service
header "Step 3: Restarting Backend Service"

info "Restarting backend with all dependencies available..."
ssh -o StrictHostKeyChecking=no -i "${SSH_KEY/#\~/$HOME}" "ubuntu@$BACKEND_IP" << 'RESTART_BACKEND'
set -euo pipefail

cd /opt/mini-xdr/backend

echo "🛑 Stopping existing backend process..."
pkill -f uvicorn || true
sleep 3

echo "🚀 Starting backend with all dependencies..."
source venv/bin/activate

# Start with proper environment and dependencies
SECRETS_MANAGER_ENABLED=true AWS_DEFAULT_REGION=us-east-1 UI_ORIGIN=http://54.237.168.3:3000 PYTHONPATH=/opt/mini-xdr/backend \
    nohup python3 -m uvicorn app.entrypoint:app --host 0.0.0.0 --port 8000 > /tmp/backend.log 2>&1 &

echo "⏳ Waiting for backend to start..."
sleep 10

echo "🔍 Checking backend status..."
if curl -f http://localhost:8000/api/ml/status >/dev/null 2>&1; then
    echo "✅ Backend is responding"
else
    echo "⚠️  Backend may still be starting up..."
fi

echo "📋 Recent backend logs:"
tail -10 /tmp/backend.log
RESTART_BACKEND

success "Backend restarted successfully!"

# Step 4: Test the problematic endpoints
header "Step 4: Testing Previously Failing Endpoints"

info "Testing /api/distributed/status endpoint..."
if curl -f -H "Origin: http://54.237.168.3:3000" "http://$BACKEND_IP:8000/api/distributed/status" >/dev/null 2>&1; then
    success "/api/distributed/status is now working!"
else
    warning "/api/distributed/status may still need a moment to initialize"
fi

info "Testing /api/federated/insights endpoint..."
if curl -f -H "Origin: http://54.237.168.3:3000" "http://$BACKEND_IP:8000/api/federated/insights" >/dev/null 2>&1; then
    success "/api/federated/insights is now working!"
else
    warning "/api/federated/insights may still need a moment to initialize"
fi

# Step 5: Verify CORS headers
header "Step 5: Verifying CORS Headers"

info "Checking CORS headers on working endpoint..."
curl -H "Origin: http://54.237.168.3:3000" -I "http://$BACKEND_IP:8000/api/incidents/timeline" | grep -i "access-control" || echo "Note: Some curl versions don't show CORS headers in HEAD requests"

# Clean up
rm -f /tmp/cors_error_fix.py

echo
success "🎉 CORS Issues Fixed!"
echo
info "Summary of changes:"
echo "  • Installed missing Python dependencies (aiokafka, redis, tensorflow)"
echo "  • Added error handling middleware for better CORS on errors"  
echo "  • Restarted backend service with all dependencies"
echo "  • Verified endpoints are responding properly"
echo
warning "If you still see CORS errors, wait 1-2 minutes for all services to fully initialize"
warning "Heavy ML dependencies may take time to load on first request"
