#!/bin/bash
# Verify T-Pot Workflow Configuration
# Checks that all workflows are properly configured and ready

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log() { echo -e "${BLUE}[INFO]${NC} $1"; }
success() { echo -e "${GREEN}✅ $1${NC}"; }
warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }
error() { echo -e "${RED}❌ $1${NC}"; }

echo ""
echo -e "${BLUE}================================================================${NC}"
echo -e "${BLUE}T-Pot Workflow Verification${NC}"
echo -e "${BLUE}================================================================${NC}"
echo ""

# Check if backend is running
log "Checking Mini-XDR backend status..."
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    success "Backend is running"
else
    error "Backend is not running. Start it with: cd backend && source venv/bin/activate && uvicorn app.main:app"
    exit 1
fi

# Check API key - read from backend/.env
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

if [ -f "$PROJECT_ROOT/backend/.env" ]; then
    API_KEY=$(grep '^API_KEY=' "$PROJECT_ROOT/backend/.env" | cut -d '=' -f2)
    if [ -z "$API_KEY" ]; then
        warning "API_KEY not found in backend/.env"
        API_KEY="dev-api-key-12345"
    else
        success "Using API key from backend/.env"
    fi
else
    warning "backend/.env not found. Using default key..."
    API_KEY="dev-api-key-12345"
fi

# Count triggers
log "Fetching workflow triggers..."
RESPONSE=$(curl -s -L http://localhost:8000/api/triggers/ \
    -H "X-API-Key: $API_KEY" 2>/dev/null)

if [ $? -ne 0 ]; then
    error "Failed to fetch triggers. Check API key and backend status."
    exit 1
fi

TOTAL_COUNT=$(echo "$RESPONSE" | python3 -c "import sys, json; print(len(json.load(sys.stdin)))" 2>/dev/null || echo "0")

if [ "$TOTAL_COUNT" -eq 0 ]; then
    warning "No triggers found in database"
    echo ""
    echo "Run the setup script first:"
    echo "  python3 scripts/tpot-management/setup-tpot-workflows.py"
    exit 1
fi

success "Found $TOTAL_COUNT workflow triggers"

# Check for T-Pot specific triggers
log "Checking T-Pot trigger coverage..."
TPOT_COUNT=$(echo "$RESPONSE" | python3 -c "import sys, json; data = json.load(sys.stdin); print(sum(1 for t in data if 'tpot' in t.get('tags', [])))" 2>/dev/null || echo "0")

if [ "$TPOT_COUNT" -gt 0 ]; then
    success "$TPOT_COUNT T-Pot workflows configured"
else
    warning "No T-Pot specific workflows found"
fi

# Check enabled triggers
log "Checking enabled triggers..."
ENABLED_COUNT=$(echo "$RESPONSE" | python3 -c "import sys, json; data = json.load(sys.stdin); print(sum(1 for t in data if t.get('enabled', False)))" 2>/dev/null || echo "0")

if [ "$ENABLED_COUNT" -gt 0 ]; then
    success "$ENABLED_COUNT triggers are enabled"
else
    warning "No enabled triggers found"
fi

# Check auto-execute triggers
log "Checking auto-execute configuration..."
AUTO_COUNT=$(echo "$RESPONSE" | python3 -c "import sys, json; data = json.load(sys.stdin); print(sum(1 for t in data if t.get('auto_execute', False)))" 2>/dev/null || echo "0")

if [ "$AUTO_COUNT" -gt 0 ]; then
    success "$AUTO_COUNT triggers will auto-execute"
    MANUAL_COUNT=$((ENABLED_COUNT - AUTO_COUNT))
    if [ "$MANUAL_COUNT" -gt 0 ]; then
        log "$MANUAL_COUNT triggers require manual approval"
    fi
else
    warning "No auto-execute triggers found"
fi

# List trigger categories
echo ""
log "Trigger categories:"
echo "$RESPONSE" | python3 -c "
import sys, json
from collections import Counter
data = json.load(sys.stdin)
categories = Counter(t.get('category', 'unknown') for t in data)
for cat, count in categories.items():
    print(f'  • {cat}: {count}')
" 2>/dev/null || echo "  Unable to parse categories"

# List trigger priorities
echo ""
log "Priority distribution:"
echo "$RESPONSE" | python3 -c "
import sys, json
from collections import Counter
data = json.load(sys.stdin)
priorities = Counter(t.get('priority', 'unknown') for t in data)
for pri, count in sorted(priorities.items(), reverse=True):
    print(f'  • {pri}: {count}')
" 2>/dev/null || echo "  Unable to parse priorities"

# Check for common T-Pot triggers
echo ""
log "Verifying critical T-Pot workflows..."

EXPECTED_TRIGGERS=(
    "T-Pot: SSH Brute Force Attack"
    "T-Pot: Successful SSH Compromise"
    "T-Pot: Malicious Command Execution"
    "T-Pot: Malware Upload Detection (Dionaea)"
    "T-Pot: Ransomware Indicators"
    "T-Pot: Data Exfiltration Attempt"
    "T-Pot: Cryptomining Detection"
    "T-Pot: DDoS Attack Detection"
)

FOUND=0
MISSING=0

for trigger in "${EXPECTED_TRIGGERS[@]}"; do
    if echo "$RESPONSE" | grep -q "\"$trigger\""; then
        success "Found: $trigger"
        ((FOUND++))
    else
        warning "Missing: $trigger"
        ((MISSING++))
    fi
done

echo ""
if [ "$MISSING" -eq 0 ]; then
    success "All critical T-Pot workflows are configured! ✅"
else
    warning "$MISSING critical workflows are missing. Run setup script:"
    echo "  python3 scripts/tpot-management/setup-tpot-workflows.py"
fi

# Check T-Pot connectivity
echo ""
log "Checking T-Pot connectivity..."

# Read T-Pot config if available
TPOT_CONFIG="/Users/chasemad/Desktop/mini-xdr/config/tpot/tpot-config.json"
if [ -f "$TPOT_CONFIG" ]; then
    TPOT_IP=$(grep -o '"allowed_ips":\s*\["[^"]*"' "$TPOT_CONFIG" | grep -o '[0-9.]*' | head -1)
    if [ -n "$TPOT_IP" ]; then
        log "T-Pot IP configured: $TPOT_IP"
        
        # Check if T-Pot is reachable (might be on Azure)
        if ping -c 1 -W 2 "$TPOT_IP" > /dev/null 2>&1; then
            success "T-Pot is reachable"
        else
            warning "T-Pot is not reachable (might be normal if on Azure)"
        fi
    else
        warning "Could not determine T-Pot IP from config"
    fi
else
    warning "T-Pot config not found: $TPOT_CONFIG"
fi

# Summary
echo ""
echo -e "${BLUE}================================================================${NC}"
echo -e "${GREEN}Verification Summary${NC}"
echo -e "${BLUE}================================================================${NC}"
echo ""
echo -e "${GREEN}✅ Backend:${NC} Running"
echo -e "${GREEN}✅ Total Triggers:${NC} $TOTAL_COUNT"
echo -e "${GREEN}✅ Enabled:${NC} $ENABLED_COUNT"
echo -e "${GREEN}✅ Auto-Execute:${NC} $AUTO_COUNT"
echo -e "${GREEN}✅ Manual Approval:${NC} $((ENABLED_COUNT - AUTO_COUNT))"
echo -e "${GREEN}✅ T-Pot Workflows:${NC} $TPOT_COUNT"
echo -e "${GREEN}✅ Critical Workflows Found:${NC} $FOUND / ${#EXPECTED_TRIGGERS[@]}"
echo ""

if [ "$FOUND" -eq "${#EXPECTED_TRIGGERS[@]}" ] && [ "$ENABLED_COUNT" -gt 0 ]; then
    echo -e "${GREEN}🎉 Your T-Pot workflows are fully configured and ready!${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. View workflows in UI: http://localhost:3000/workflows"
    echo "  2. Deploy T-Pot on Azure if not already done"
    echo "  3. Configure T-Pot log forwarding to Mini-XDR"
    echo "  4. Test with simulated attacks"
else
    echo -e "${YELLOW}⚠️  Setup incomplete. Run the setup script:${NC}"
    echo "  python3 scripts/tpot-management/setup-tpot-workflows.py"
fi

echo ""
echo -e "${BLUE}================================================================${NC}"
echo ""

