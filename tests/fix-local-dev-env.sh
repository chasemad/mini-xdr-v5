#!/bin/bash

# FIX LOCAL DEVELOPMENT ENVIRONMENT
# Creates matching API keys for frontend and backend local development

set -euo pipefail

PROJECT_ROOT="/Users/chasemad/Desktop/mini-xdr"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

log() { echo -e "${GREEN}[$(date +'%H:%M:%S')] $1${NC}"; }
warn() { echo -e "${YELLOW}[WARN] $1${NC}"; }
error() { echo -e "${RED}[ERROR] $1${NC}"; exit 1; }
step() { echo -e "${BLUE}$1${NC}"; }

echo -e "${BLUE}"
echo "=============================================="
echo "    🔧 FIX LOCAL DEVELOPMENT ENVIRONMENT"
echo "=============================================="
echo -e "${NC}"
echo "This will create matching API keys for local development"
echo ""

# Generate a consistent API key for local development
LOCAL_API_KEY="mini-xdr-local-dev-key-2025-$(openssl rand -hex 16)"

log "Generated local API key: ${LOCAL_API_KEY:0:20}..."

# Create backend .env file
step "📝 Creating backend .env file..."

cat > "$PROJECT_ROOT/backend/.env" << EOF
# MINI-XDR LOCAL DEVELOPMENT CONFIGURATION

# API Configuration
API_HOST=127.0.0.1
API_PORT=8000
UI_ORIGIN=http://localhost:3000

# LOCAL DEVELOPMENT API KEY (matches frontend)
API_KEY=$LOCAL_API_KEY

# Database (SQLite for local development)
DATABASE_URL=sqlite+aiosqlite:///./xdr.db

# Detection Configuration
FAIL_WINDOW_SECONDS=60
FAIL_THRESHOLD=6
AUTO_CONTAIN=false

# Honeypot Configuration - TPOT
HONEYPOT_HOST=34.193.101.171
HONEYPOT_USER=admin
HONEYPOT_SSH_KEY=/Users/chasemad/.ssh/mini-xdr-tpot-key.pem
HONEYPOT_SSH_PORT=64295

# LLM Configuration (configure with your actual keys for testing)
LLM_PROVIDER=openai
OPENAI_API_KEY=YOUR_OPENAI_API_KEY_HERE
OPENAI_MODEL=gpt-4o-mini

# Optional: X.AI Configuration
XAI_API_KEY=YOUR_XAI_API_KEY_HERE
XAI_MODEL=grok-beta

# Threat Intelligence APIs (optional for local testing)
ABUSEIPDB_API_KEY=optional-for-local-testing
VIRUSTOTAL_API_KEY=optional-for-local-testing

# Agent Configuration
MINIXDR_AGENT_PROFILE=HUNTER
MINIXDR_AGENT_DEVICE_ID=local-dev-device-id
MINIXDR_AGENT_HMAC_KEY=local-dev-hmac-key

# ML Models Path
ML_MODELS_PATH=/Users/chasemad/Desktop/mini-xdr/models
POLICIES_PATH=/Users/chasemad/Desktop/mini-xdr/policies

# Local Development Settings
LOG_LEVEL=DEBUG
ENVIRONMENT=development
EOF

log "✅ Backend .env file created"

# Create frontend .env.local file
step "📝 Creating frontend .env.local file..."

cat > "$PROJECT_ROOT/frontend/.env.local" << EOF
# MINI-XDR FRONTEND LOCAL DEVELOPMENT CONFIGURATION

# API Configuration (matches backend)
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_API_KEY=$LOCAL_API_KEY

# Environment
NEXT_PUBLIC_ENVIRONMENT=development

# Security (disabled for local development)
NEXT_PUBLIC_CSP_ENABLED=false
NEXT_PUBLIC_SECURE_HEADERS=false

# Debug settings
NEXT_PUBLIC_DEBUG=true
EOF

log "✅ Frontend .env.local file created"

# Set proper permissions
chmod 600 "$PROJECT_ROOT/backend/.env"
chmod 600 "$PROJECT_ROOT/frontend/.env.local"

step "🧪 Testing API key configuration..."

# Test that the files were created and have matching keys
backend_key=$(grep "API_KEY=" "$PROJECT_ROOT/backend/.env" | cut -d'=' -f2)
frontend_key=$(grep "NEXT_PUBLIC_API_KEY=" "$PROJECT_ROOT/frontend/.env.local" | cut -d'=' -f2)

if [ "$backend_key" = "$frontend_key" ]; then
    log "✅ API keys match between frontend and backend"
else
    error "❌ API keys don't match - this shouldn't happen"
fi

echo ""
echo "=============================================="
echo "   ✅ LOCAL DEVELOPMENT ENVIRONMENT FIXED!"
echo "=============================================="
echo ""
echo "📁 Files created:"
echo "  - backend/.env (with API key: ${LOCAL_API_KEY:0:20}...)"
echo "  - frontend/.env.local (with matching API key)"
echo ""
echo "🚀 Your local development environment is now ready!"
echo ""
echo "To start your application:"
echo "  1. Backend: cd backend && source venv/bin/activate && python -m app.main"
echo "  2. Frontend: cd frontend && npm run dev"
echo ""
echo "🔑 API keys now match between frontend and backend"
echo "📊 Using SQLite database for local development"
echo "🛡️ Ready for secure AWS deployment when needed"
echo ""
log "Local development environment configuration completed!"
