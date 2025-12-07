#!/bin/bash
# ========================================================================
# Mini-XDR Azure Deployment Full System Test
# ========================================================================
# Comprehensive testing of T-Pot connectivity, secrets, agents, and APIs
# ========================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND_DIR="$PROJECT_ROOT/backend"
BACKEND_PORT=8000
TPOT_HOST="74.235.242.205"
TPOT_SSH_PORT=64295
TPOT_WEB_PORT=64297
SSH_KEY="$HOME/.ssh/mini-xdr-tpot-azure"
AZURE_KEY_VAULT="minixdr-keyvault"

# Test counters
TESTS_PASSED=0
TESTS_FAILED=0
TESTS_WARNING=0

# Logging functions
log() {
    echo -e "${BLUE}[$(date +'%H:%M:%S')]${NC} $1"
}

success() {
    echo -e "${GREEN}✅ $1${NC}"
    ((TESTS_PASSED++))
}

error() {
    echo -e "${RED}❌ $1${NC}"
    ((TESTS_FAILED++))
}

warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
    ((TESTS_WARNING++))
}

section() {
    echo ""
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${CYAN}$1${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
}

# Test 1: Azure CLI and Authentication
test_azure_cli() {
    section "1. Testing Azure CLI Configuration"
    
    if ! command -v az &> /dev/null; then
        error "Azure CLI not installed"
        return 1
    fi
    success "Azure CLI installed"
    
    if ! az account show &> /dev/null; then
        error "Not logged into Azure. Run: az login"
        return 1
    fi
    success "Azure authentication valid"
    
    local subscription=$(az account show --query name -o tsv 2>/dev/null)
    log "Subscription: $subscription"
    
    return 0
}

# Test 2: Azure Key Vault Access
test_azure_key_vault() {
    section "2. Testing Azure Key Vault Access"
    
    if ! az keyvault show --name "$AZURE_KEY_VAULT" &> /dev/null; then
        error "Cannot access Key Vault: $AZURE_KEY_VAULT"
        return 1
    fi
    success "Key Vault accessible: $AZURE_KEY_VAULT"
    
    log "Checking secrets in Key Vault..."
    local secrets=(
        "mini-xdr-api-key"
        "tpot-api-key"
        "tpot-host"
        "openai-api-key"
        "xai-api-key"
        "abuseipdb-api-key"
        "virustotal-api-key"
    )
    
    local missing_secrets=()
    for secret in "${secrets[@]}"; do
        if az keyvault secret show --vault-name "$AZURE_KEY_VAULT" --name "$secret" &> /dev/null; then
            success "Secret exists: $secret"
        else
            error "Secret missing: $secret"
            missing_secrets+=("$secret")
        fi
    done
    
    # Check for agent secrets
    log "Checking agent credentials in Key Vault..."
    local agent_types=("containment" "attribution" "forensics" "deception" "hunter" "rollback")
    local missing_agents=()
    
    for agent in "${agent_types[@]}"; do
        if az keyvault secret show --vault-name "$AZURE_KEY_VAULT" --name "${agent}-agent-device-id" &> /dev/null; then
            success "Agent credentials exist: ${agent}"
        else
            warning "Agent credentials missing: ${agent}"
            missing_agents+=("$agent")
        fi
    done
    
    if [ ${#missing_agents[@]} -gt 0 ]; then
        echo ""
        warning "Missing agent credentials for: ${missing_agents[*]}"
        echo -e "${YELLOW}Run this to generate agent credentials:${NC}"
        echo "  cd $BACKEND_DIR"
        echo "  source venv/bin/activate"
        for agent in "${missing_agents[@]}"; do
            echo "  python scripts/auth/mint_agent_cred.py  # Then store in Azure Key Vault as ${agent}-agent-*"
        done
    fi
    
    return 0
}

# Test 3: Backend .env Configuration
test_env_configuration() {
    section "3. Testing Backend .env Configuration"
    
    if [ ! -f "$BACKEND_DIR/.env" ]; then
        warning ".env file not found"
        log "Running sync from Azure Key Vault..."
        
        cd "$PROJECT_ROOT"
        if ./scripts/sync-secrets-from-azure.sh "$AZURE_KEY_VAULT"; then
            success ".env created from Azure secrets"
        else
            error "Failed to create .env from Azure"
            return 1
        fi
    else
        success ".env file exists"
    fi
    
    # Check critical env variables
    cd "$BACKEND_DIR"
    source venv/bin/activate 2>/dev/null || {
        error "Virtual environment not found. Run: python3 -m venv venv"
        return 1
    }
    
    log "Checking configuration values..."
    python3 <<EOF
import os
from pathlib import Path

# Load .env
env_file = Path("$BACKEND_DIR/.env")
if env_file.exists():
    with open(env_file) as f:
        for line in f:
            if '=' in line and not line.startswith('#'):
                key, value = line.strip().split('=', 1)
                os.environ[key] = value

# Import config
import sys
sys.path.insert(0, "$BACKEND_DIR")
from app.config import settings

# Check values
checks = {
    'HONEYPOT_HOST': settings.honeypot_host,
    'HONEYPOT_USER': settings.honeypot_user,
    'HONEYPOT_SSH_KEY': settings.honeypot_ssh_key,
    'API_KEY': settings.api_key or "NOT SET",
}

print("\nConfiguration values:")
for key, value in checks.items():
    if value and value != "NOT SET" and not value.startswith("CONFIGURE"):
        print(f"  ✅ {key}: {value[:20]}..." if len(str(value)) > 20 else f"  ✅ {key}: {value}")
    else:
        print(f"  ⚠️  {key}: Not configured")
        
# Check LLM keys
llm_configured = False
if settings.openai_api_key and not settings.openai_api_key.startswith("CONFIGURE"):
    print(f"  ✅ OPENAI_API_KEY: Configured")
    llm_configured = True
elif settings.xai_api_key and not settings.xai_api_key.startswith("CONFIGURE"):
    print(f"  ✅ XAI_API_KEY: Configured")
    llm_configured = True
else:
    print(f"  ⚠️  LLM API Keys: Not configured (optional)")

# Check threat intel keys
if settings.abuseipdb_api_key and not settings.abuseipdb_api_key.startswith("CONFIGURE"):
    print(f"  ✅ ABUSEIPDB_API_KEY: Configured")
else:
    print(f"  ⚠️  ABUSEIPDB_API_KEY: Not configured (optional)")

if settings.virustotal_api_key and not settings.virustotal_api_key.startswith("CONFIGURE"):
    print(f"  ✅ VIRUSTOTAL_API_KEY: Configured")
else:
    print(f"  ⚠️  VIRUSTOTAL_API_KEY: Not configured (optional)")
EOF
    
    success "Configuration check complete"
    return 0
}

# Test 4: T-Pot SSH Connectivity
test_tpot_ssh() {
    section "4. Testing T-Pot SSH Connectivity"
    
    if [ ! -f "$SSH_KEY" ]; then
        error "SSH key not found: $SSH_KEY"
        log "Generate with: ssh-keygen -t ed25519 -f $SSH_KEY"
        return 1
    fi
    success "SSH key exists: $SSH_KEY"
    
    # Check key permissions
    local perms=$(stat -f "%OLp" "$SSH_KEY" 2>/dev/null || stat -c "%a" "$SSH_KEY" 2>/dev/null)
    if [ "$perms" != "600" ]; then
        log "Fixing SSH key permissions..."
        chmod 600 "$SSH_KEY"
    fi
    success "SSH key permissions correct (600)"
    
    # Test SSH connection
    log "Testing SSH connection to T-Pot ($TPOT_HOST:$TPOT_SSH_PORT)..."
    if ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no -p "$TPOT_SSH_PORT" -i "$SSH_KEY" "azureuser@$TPOT_HOST" "echo 'SSH connection successful'" &>/dev/null; then
        success "SSH connection to T-Pot successful"
    else
        error "SSH connection to T-Pot failed"
        log "Debug command: ssh -v -p $TPOT_SSH_PORT -i $SSH_KEY azureuser@$TPOT_HOST"
        return 1
    fi
    
    # Test Docker access
    log "Testing Docker access on T-Pot..."
    if ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no -p "$TPOT_SSH_PORT" -i "$SSH_KEY" "azureuser@$TPOT_HOST" "sudo docker ps" &>/dev/null; then
        success "Docker access verified"
        
        # List running honeypots
        log "T-Pot containers running:"
        ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no -p "$TPOT_SSH_PORT" -i "$SSH_KEY" "azureuser@$TPOT_HOST" "sudo docker ps --format 'table {{.Names}}\t{{.Status}}'" 2>/dev/null | head -10
    else
        warning "Docker access not available (may need sudo configuration)"
    fi
    
    return 0
}

# Test 5: T-Pot Web Interface
test_tpot_web() {
    section "5. Testing T-Pot Web Interface"
    
    log "Testing T-Pot web interface (https://$TPOT_HOST:$TPOT_WEB_PORT)..."
    if curl -k -s -o /dev/null -w "%{http_code}" "https://$TPOT_HOST:$TPOT_WEB_PORT" | grep -q "200\|301\|302\|401"; then
        success "T-Pot web interface accessible"
        log "Access at: https://$TPOT_HOST:$TPOT_WEB_PORT"
        log "Username: tsec"
        log "Password: minixdrtpot2025"
    else
        error "T-Pot web interface not accessible"
        log "Check firewall rules and VM status"
    fi
    
    return 0
}

# Test 6: Backend API Endpoints
test_backend_api() {
    section "6. Testing Backend API Endpoints"
    
    # Check if backend is running
    if ! curl -s "http://localhost:$BACKEND_PORT/health" > /dev/null 2>&1; then
        warning "Backend not running on port $BACKEND_PORT"
        log "Start with: cd $BACKEND_DIR && source venv/bin/activate && uvicorn app.entrypoint:app --reload"
        return 1
    fi
    success "Backend API running"
    
    # Test health endpoint
    local health_response=$(curl -s "http://localhost:$BACKEND_PORT/health")
    if echo "$health_response" | grep -q "status"; then
        success "Health endpoint responding"
    else
        error "Health endpoint not responding correctly"
    fi
    
    # Test incidents endpoint
    log "Testing incidents endpoint..."
    if curl -s "http://localhost:$BACKEND_PORT/incidents" > /dev/null 2>&1; then
        success "Incidents endpoint accessible"
    else
        warning "Incidents endpoint not responding"
    fi
    
    # Test authenticated endpoints using HMAC
    log "Testing authenticated endpoints..."
    if [ -f "$PROJECT_ROOT/scripts/auth/send_signed_request.py" ]; then
        cd "$BACKEND_DIR"
        source venv/bin/activate
        
        # Test ML status
        local ml_response=$(python3 "$PROJECT_ROOT/scripts/auth/send_signed_request.py" \
            --base-url "http://localhost:$BACKEND_PORT" \
            --path "/api/ml/status" \
            --method GET 2>/dev/null)
        
        if echo "$ml_response" | grep -q "models"; then
            success "ML Status API responding"
        else
            warning "ML Status API not responding correctly"
        fi
    fi
    
    return 0
}

# Test 7: Agent Credentials
test_agent_credentials() {
    section "7. Testing Agent Credentials"
    
    cd "$BACKEND_DIR"
    source venv/bin/activate 2>/dev/null
    
    log "Checking agent credentials in database..."
    python3 <<EOF
import asyncio
import sys
sys.path.insert(0, "$BACKEND_DIR")

async def check_agents():
    from app.db import init_db, AsyncSessionLocal
    from app.models import AgentCredential
    from sqlalchemy import select
    
    await init_db()
    
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(AgentCredential))
        credentials = result.scalars().all()
        
        if credentials:
            print(f"\n✅ Found {len(credentials)} agent credential(s) in database")
            for cred in credentials:
                print(f"   - Device ID: {cred.device_id}")
                print(f"     Public ID: {cred.public_id}")
                print(f"     Expires: {cred.expires_at or 'Never'}")
        else:
            print("\n⚠️  No agent credentials found in database")
            print("\nGenerate with:")
            print("  cd $BACKEND_DIR")
            print("  source venv/bin/activate")
            print("  python scripts/auth/mint_agent_cred.py")
            return False
    return True

result = asyncio.run(check_agents())
sys.exit(0 if result else 1)
EOF
    
    if [ $? -eq 0 ]; then
        success "Agent credentials configured"
    else
        warning "Agent credentials need to be generated"
    fi
    
    return 0
}

# Test 8: Test Event Ingestion
test_event_ingestion() {
    section "8. Testing Event Ingestion from T-Pot"
    
    if ! curl -s "http://localhost:$BACKEND_PORT/health" > /dev/null 2>&1; then
        warning "Backend not running, skipping ingestion test"
        return 1
    fi
    
    log "Sending test event to backend..."
    cd "$BACKEND_DIR"
    source venv/bin/activate 2>/dev/null
    
    local test_payload=$(cat <<JSON
{
  "source_type": "cowrie",
  "hostname": "azure-tpot-test",
  "events": [{
    "eventid": "cowrie.login.failed",
    "src_ip": "192.168.1.100",
    "dst_port": 2222,
    "message": "Test event from Azure deployment test",
    "raw": {
      "username": "admin",
      "password": "test123",
      "test_event": true,
      "test_type": "azure_deployment_validation",
      "test_timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    },
    "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  }]
}
JSON
)
    
    local response=$(python3 "$PROJECT_ROOT/scripts/auth/send_signed_request.py" \
        --base-url "http://localhost:$BACKEND_PORT" \
        --path "/ingest/multi" \
        --method POST \
        --body "$test_payload" 2>/dev/null)
    
    if echo "$response" | grep -q "processed"; then
        success "Event ingestion test successful"
        log "Response: $response"
    else
        error "Event ingestion test failed"
        log "Response: $response"
    fi
    
    return 0
}

# Test 9: Generate Missing Secrets
generate_missing_secrets() {
    section "9. Generating Missing Secrets"
    
    log "This would generate and store missing secrets in Azure Key Vault"
    log "Run manually if needed:"
    echo ""
    echo "  # Generate Mini-XDR API key"
    echo "  az keyvault secret set --vault-name $AZURE_KEY_VAULT \\"
    echo "    --name mini-xdr-api-key \\"
    echo "    --value \"\$(openssl rand -hex 32)\""
    echo ""
    echo "  # Generate T-Pot API key"
    echo "  az keyvault secret set --vault-name $AZURE_KEY_VAULT \\"
    echo "    --name tpot-api-key \\"
    echo "    --value \"\$(openssl rand -hex 32)\""
    echo ""
    echo "  # Store agent credentials (after generating with mint_agent_cred.py)"
    echo "  az keyvault secret set --vault-name $AZURE_KEY_VAULT \\"
    echo "    --name containment-agent-device-id \\"
    echo "    --value \"<device-id>\""
    echo ""
    
    return 0
}

# Main test execution
main() {
    clear
    echo -e "${CYAN}╔════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║     Mini-XDR Azure Deployment System Test             ║${NC}"
    echo -e "${CYAN}║     Comprehensive Testing Suite                        ║${NC}"
    echo -e "${CYAN}╚════════════════════════════════════════════════════════╝${NC}"
    echo ""
    
    test_azure_cli
    test_azure_key_vault
    test_env_configuration
    test_tpot_ssh
    test_tpot_web
    test_backend_api
    test_agent_credentials
    test_event_ingestion
    generate_missing_secrets
    
    # Final summary
    section "Test Summary"
    echo -e "${GREEN}Passed:  $TESTS_PASSED${NC}"
    echo -e "${YELLOW}Warnings: $TESTS_WARNING${NC}"
    echo -e "${RED}Failed:  $TESTS_FAILED${NC}"
    echo ""
    
    if [ $TESTS_FAILED -eq 0 ]; then
        echo -e "${GREEN}╔════════════════════════════════════════════════════════╗${NC}"
        echo -e "${GREEN}║     ✨ All Critical Tests Passed!                      ║${NC}"
        echo -e "${GREEN}║     System Ready for Production Use                   ║${NC}"
        echo -e "${GREEN}╚════════════════════════════════════════════════════════╝${NC}"
        exit 0
    else
        echo -e "${RED}╔════════════════════════════════════════════════════════╗${NC}"
        echo -e "${RED}║     ⚠️  Some Tests Failed                              ║${NC}"
        echo -e "${RED}║     Review errors above and fix issues                ║${NC}"
        echo -e "${RED}╚════════════════════════════════════════════════════════╝${NC}"
        exit 1
    fi
}

# Run tests
main


