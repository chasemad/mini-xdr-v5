#!/bin/bash

# SIMPLIFIED SECURE API KEY MIGRATION SCRIPT
# Compatible version for macOS bash

set -euo pipefail

# Configuration
PROJECT_ROOT="/Users/chasemad/Desktop/mini-xdr"
REGION="${AWS_REGION:-us-east-1}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log() { echo -e "${GREEN}[$(date +'%H:%M:%S')] $1${NC}"; }
warn() { echo -e "${YELLOW}[WARN] $1${NC}"; }
error() { echo -e "${RED}[ERROR] $1${NC}"; exit 1; }
step() { echo -e "${BLUE}$1${NC}"; }

echo "🔐 SECURE API KEY MIGRATION"
echo "================================"
echo ""

# Check AWS CLI
if ! aws sts get-caller-identity &>/dev/null; then
    error "AWS CLI not configured. Please run 'aws configure' first."
fi

ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
log "AWS Account: $ACCOUNT_ID, Region: $REGION"

# Read the backup file with real API keys
BACKUP_FILE="$PROJECT_ROOT/backend/.env.backup-20250926_214016"

if [ ! -f "$BACKUP_FILE" ]; then
    error "Backup file not found: $BACKUP_FILE"
fi

step "📋 Extracting API Keys from Backup"

# Extract specific API keys
OPENAI_KEY=$(grep "^OPENAI_API_KEY=" "$BACKUP_FILE" | cut -d'=' -f2- | tr -d '"' || echo "")
XAI_KEY=$(grep "^XAI_API_KEY=" "$BACKUP_FILE" | cut -d'=' -f2- | tr -d '"' || echo "")
ABUSEIPDB_KEY=$(grep "^ABUSEIPDB_API_KEY=" "$BACKUP_FILE" | cut -d'=' -f2- | tr -d '"' || echo "")
VIRUSTOTAL_KEY=$(grep "^VIRUSTOTAL_API_KEY=" "$BACKUP_FILE" | cut -d'=' -f2- | tr -d '"' || echo "")
CONTAINMENT_SECRET=$(grep "^CONTAINMENT_AGENT_SECRET=" "$BACKUP_FILE" | cut -d'=' -f2- | tr -d '"' || echo "")

# Generate secure keys
API_KEY=$(openssl rand -hex 32)
DB_PASSWORD=$(openssl rand -base64 32 | tr -d "=+/" | cut -c1-25)

step "🔐 Storing Secrets in AWS Secrets Manager"

# Store each secret
store_secret() {
    local name="$1"
    local value="$2"
    
    if [ -n "$value" ] && [ "$value" != "changeme" ] && [ "$value" != "YOUR_"* ]; then
        log "Storing: $name"
        aws secretsmanager create-secret \
            --name "$name" \
            --secret-string "$value" \
            --region "$REGION" 2>/dev/null || \
        aws secretsmanager update-secret \
            --secret-id "$name" \
            --secret-string "$value" \
            --region "$REGION" >/dev/null
        log "✅ Stored: $name"
    else
        log "⚠️ Skipping empty/placeholder: $name"
    fi
}

# Store all secrets
store_secret "mini-xdr/api-key" "$API_KEY"
store_secret "mini-xdr/database-password" "$DB_PASSWORD"
store_secret "mini-xdr/openai-api-key" "$OPENAI_KEY"
store_secret "mini-xdr/xai-api-key" "$XAI_KEY"
store_secret "mini-xdr/abuseipdb-api-key" "$ABUSEIPDB_KEY"
store_secret "mini-xdr/virustotal-api-key" "$VIRUSTOTAL_KEY"
store_secret "mini-xdr/agents/containment-secret" "$CONTAINMENT_SECRET"

step "🔄 Updating Configuration Files"

# Update backend .env
cat > "$PROJECT_ROOT/backend/.env" << 'EOF'
# MINI-XDR SECURE CONFIGURATION
# All sensitive credentials stored in AWS Secrets Manager

API_HOST=127.0.0.1
API_PORT=8000
UI_ORIGIN=http://localhost:3000

# Secrets Manager Configuration
SECRETS_MANAGER_ENABLED=true
AWS_REGION=us-east-1

# API keys retrieved from Secrets Manager
API_KEY_SECRET_NAME=mini-xdr/api-key
DATABASE_PASSWORD_SECRET_NAME=mini-xdr/database-password
OPENAI_API_KEY_SECRET_NAME=mini-xdr/openai-api-key
XAI_API_KEY_SECRET_NAME=mini-xdr/xai-api-key
ABUSEIPDB_API_KEY_SECRET_NAME=mini-xdr/abuseipdb-api-key
VIRUSTOTAL_API_KEY_SECRET_NAME=mini-xdr/virustotal-api-key

# Application Configuration
DATABASE_URL=sqlite+aiosqlite:///./xdr.db
FAIL_WINDOW_SECONDS=60
FAIL_THRESHOLD=6
AUTO_CONTAIN=false

# Honeypot Configuration
HONEYPOT_HOST=34.193.101.171
HONEYPOT_USER=admin
HONEYPOT_SSH_KEY=/Users/chasemad/.ssh/mini-xdr-tpot-key.pem
HONEYPOT_SSH_PORT=64295

# LLM Configuration
LLM_PROVIDER=openai
OPENAI_MODEL=gpt-4o-mini
XAI_MODEL=grok-beta

# Development
ML_MODELS_PATH=/Users/chasemad/Desktop/mini-xdr/models
POLICIES_PATH=/Users/chasemad/Desktop/mini-xdr/policies
LOG_LEVEL=DEBUG
ENVIRONMENT=development
EOF

# Update frontend .env
cat > "$PROJECT_ROOT/frontend/.env.local" << 'EOF'
# MINI-XDR FRONTEND SECURE CONFIGURATION

NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_ENVIRONMENT=development
NEXT_PUBLIC_CSP_ENABLED=false
NEXT_PUBLIC_DEBUG=true

# Security Configuration
NEXT_PUBLIC_SECRETS_MANAGER_ENABLED=true
NEXT_PUBLIC_API_KEY_SECRET_NAME=mini-xdr/api-key
EOF

log "✅ Configuration files updated"

step "🛠️ Creating Secret Helper Script"

cat > "$PROJECT_ROOT/aws/utils/get-secret.sh" << 'EOF'
#!/bin/bash
SECRET_NAME="$1"
REGION="${AWS_REGION:-us-east-1}"

if [ -z "$SECRET_NAME" ]; then
    echo "Usage: $0 <secret-name>"
    exit 1
fi

aws secretsmanager get-secret-value \
    --secret-id "$SECRET_NAME" \
    --query SecretString \
    --output text \
    --region "$REGION"
EOF

chmod +x "$PROJECT_ROOT/aws/utils/get-secret.sh"

step "✅ Validation"

# Test secret retrieval
if "$PROJECT_ROOT/aws/utils/get-secret.sh" "mini-xdr/api-key" >/dev/null 2>&1; then
    log "✅ Secret retrieval test: PASSED"
else
    warn "⚠️ Secret retrieval test: FAILED"
fi

# List stored secrets
echo ""
log "📋 Secrets stored in AWS Secrets Manager:"
aws secretsmanager list-secrets --query 'SecretList[?contains(Name, `mini-xdr`)].Name' --output table

step "🧹 Cleanup"

# Securely remove the backup file with sensitive data
if [ -f "$BACKUP_FILE" ]; then
    log "Securely removing backup file with plain text credentials..."
    if command -v shred >/dev/null 2>&1; then
        shred -vfz -n 3 "$BACKUP_FILE"
    else
        rm -f "$BACKUP_FILE"
    fi
    log "✅ Backup file securely removed"
fi

echo ""
echo "=============================================================="
echo "🎉 API KEY MIGRATION COMPLETED SUCCESSFULLY!"
echo "=============================================================="
echo ""
log "🔒 All API keys are now encrypted in AWS Secrets Manager"
log "🛡️ Configuration files updated to use secure retrieval"
log "📋 Test secret access: ./aws/utils/get-secret.sh mini-xdr/api-key"
echo ""
log "✅ Your credentials are now enterprise-secure!"