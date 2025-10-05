#!/bin/bash
# ========================================================================
# Sync Frontend API Key from Backend (Azure Key Vault)
# ========================================================================
# Run this after syncing secrets from Azure to update frontend config
# ========================================================================

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND_ENV="$PROJECT_ROOT/backend/.env"
FRONTEND_ENV="$PROJECT_ROOT/frontend/.env.local"

echo -e "${BLUE}🔐 Syncing Frontend API Key from Backend${NC}"
echo ""

# Check if backend .env exists
if [ ! -f "$BACKEND_ENV" ]; then
    echo -e "${RED}❌ Backend .env not found${NC}"
    echo "Run: ./scripts/sync-secrets-from-azure.sh first"
    exit 1
fi

# Get API key from backend
API_KEY=$(grep "^API_KEY=" "$BACKEND_ENV" | cut -d'=' -f2)

if [ -z "$API_KEY" ]; then
    echo -e "${RED}❌ API_KEY not found in backend .env${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Found API key in backend .env${NC}"
echo -e "   Key: ${API_KEY:0:20}...${NC}"

# Backup existing frontend .env.local
if [ -f "$FRONTEND_ENV" ]; then
    cp "$FRONTEND_ENV" "${FRONTEND_ENV}.backup-$(date +%Y%m%d-%H%M%S)"
    echo -e "${GREEN}✅ Backed up existing frontend .env.local${NC}"
fi

# Update frontend .env.local
cat > "$FRONTEND_ENV" << EOF
# MINI-XDR FRONTEND LOCAL DEVELOPMENT CONFIGURATION
# Auto-synced from backend on $(date)

NEXT_PUBLIC_API_BASE=http://localhost:8000
NEXT_PUBLIC_API_KEY=$API_KEY
NEXT_PUBLIC_ENVIRONMENT=development
NEXT_PUBLIC_CSP_ENABLED=false
NEXT_PUBLIC_DEBUG=true

# Security Configuration (disabled for local development)
NEXT_PUBLIC_SECRETS_MANAGER_ENABLED=false
EOF

echo -e "${GREEN}✅ Updated frontend .env.local${NC}"
echo ""

# Check if frontend is running
if ps aux | grep -q "[n]ext dev"; then
    echo -e "${YELLOW}⚠️  Frontend is currently running${NC}"
    echo -e "${YELLOW}   Restart required to apply changes${NC}"
    echo ""
    echo -e "${BLUE}To restart frontend:${NC}"
    echo "  pkill -f 'next dev'"
    echo "  cd $PROJECT_ROOT/frontend && npm run dev"
else
    echo -e "${GREEN}✅ Frontend not running (changes will apply on next start)${NC}"
fi

echo ""
echo -e "${GREEN}✨ Frontend API key synced successfully!${NC}"


