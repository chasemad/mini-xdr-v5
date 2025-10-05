#!/bin/bash
# Quick test script to verify .env configuration
# Run this to check your API keys and T-Pot workflows

cd /Users/chasemad/Desktop/mini-xdr/backend

echo ""
echo "🔍 Mini-XDR Configuration Test"
echo "=============================="
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Check .env exists
if [ ! -f .env ]; then
    echo -e "${RED}❌ .env file not found!${NC}"
    echo "Create it from env.example:"
    echo "  cp env.example .env"
    exit 1
fi

echo -e "${GREEN}✅ .env file found${NC}"
echo ""

# Check backend health
echo "1️⃣  Backend Health:"
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo -e "${GREEN}   ✅ Backend is running${NC}"
    curl -s http://localhost:8000/health | python3 -c "import sys, json; d=json.load(sys.stdin); print(f'   Status: {d[\"status\"]}, Orchestrator: {d[\"orchestrator\"]}')" 2>/dev/null || echo "   (could not parse health response)"
else
    echo -e "${RED}   ❌ Backend not running${NC}"
    echo "   Start it with: cd backend && uvicorn app.main:app"
fi
echo ""

# Check API key
echo "2️⃣  API Key Configuration:"
if grep -q '^API_KEY=.\+' .env 2>/dev/null; then
    API_KEY=$(grep '^API_KEY=' .env | cut -d '=' -f2)
    if [ -n "$API_KEY" ] && [ "$API_KEY" != "WILL_BE_GENERATED_DURING_DEPLOYMENT" ]; then
        echo -e "${GREEN}   ✅ API key configured${NC}"
        echo "   Key length: ${#API_KEY} characters"
        
        # Test API access
        if curl -s -H "X-API-Key: $API_KEY" http://localhost:8000/api/triggers > /dev/null 2>&1; then
            TRIGGER_COUNT=$(curl -s -H "X-API-Key: $API_KEY" http://localhost:8000/api/triggers | python3 -c "import sys, json; print(len(json.load(sys.stdin)))" 2>/dev/null || echo "0")
            echo -e "${GREEN}   ✅ API access working${NC}"
            echo "   Found $TRIGGER_COUNT workflow triggers"
        else
            echo -e "${YELLOW}   ⚠️  API access test failed${NC}"
            echo "   This might be normal if backend just restarted"
        fi
    else
        echo -e "${YELLOW}   ⚠️  API key not set or using placeholder${NC}"
        echo "   Set a real API key in .env:"
        echo "   API_KEY=\$(openssl rand -hex 32)"
    fi
else
    echo -e "${RED}   ❌ API_KEY not found in .env${NC}"
fi
echo ""

# Check T-Pot configuration
echo "3️⃣  T-Pot Honeypot Configuration:"
TPOT_CONFIGURED=0
if grep -q '^TPOT_API_KEY=.\+' .env 2>/dev/null; then
    TPOT_KEY=$(grep '^TPOT_API_KEY=' .env | cut -d '=' -f2)
    if [ -n "$TPOT_KEY" ] && [ "$TPOT_KEY" != "your-tpot-key" ]; then
        echo -e "${GREEN}   ✅ T-Pot API key configured${NC}"
        TPOT_CONFIGURED=1
    else
        echo -e "${YELLOW}   ⚠️  T-Pot API key not set${NC}"
    fi
else
    echo -e "${YELLOW}   ⚠️  TPOT_API_KEY not in .env${NC}"
fi

if grep -q '^TPOT_HOST=.\+' .env 2>/dev/null; then
    TPOT_HOST=$(grep '^TPOT_HOST=' .env | cut -d '=' -f2)
    if [ -n "$TPOT_HOST" ] && [ "$TPOT_HOST" != "your-tpot-ip-or-hostname" ]; then
        echo -e "${GREEN}   ✅ T-Pot host configured: $TPOT_HOST${NC}"
        
        # Test connectivity (with timeout)
        if timeout 2 ping -c 1 $TPOT_HOST > /dev/null 2>&1; then
            echo -e "${GREEN}   ✅ T-Pot host is reachable${NC}"
        else
            echo -e "${YELLOW}   ⚠️  T-Pot host not reachable (might be on Azure)${NC}"
        fi
    else
        echo -e "${YELLOW}   ⚠️  T-Pot host not set${NC}"
    fi
else
    echo -e "${YELLOW}   ⚠️  TPOT_HOST not in .env${NC}"
fi

if [ $TPOT_CONFIGURED -eq 0 ]; then
    echo ""
    echo "   To configure T-Pot, add to .env:"
    echo "   TPOT_API_KEY=your-tpot-api-key"
    echo "   TPOT_HOST=your-azure-tpot-ip"
fi
echo ""

# Check LLM configuration
echo "4️⃣  LLM Configuration:"
LLM_COUNT=0
if grep -q '^OPENAI_API_KEY=.\+' .env 2>/dev/null; then
    OPENAI_KEY=$(grep '^OPENAI_API_KEY=' .env | cut -d '=' -f2)
    if [ -n "$OPENAI_KEY" ] && [ "$OPENAI_KEY" != "CONFIGURE_IN_AWS_SECRETS_MANAGER" ]; then
        echo -e "${GREEN}   ✅ OpenAI API key configured${NC}"
        LLM_COUNT=$((LLM_COUNT + 1))
    else
        echo -e "${YELLOW}   ⚠️  OpenAI key not configured${NC}"
    fi
else
    echo -e "${YELLOW}   ⚠️  OpenAI key not in .env${NC}"
fi

if grep -q '^XAI_API_KEY=.\+' .env 2>/dev/null; then
    XAI_KEY=$(grep '^XAI_API_KEY=' .env | cut -d '=' -f2)
    if [ -n "$XAI_KEY" ] && [ "$XAI_KEY" != "CONFIGURE_IN_AWS_SECRETS_MANAGER" ]; then
        echo -e "${GREEN}   ✅ XAI (Grok) API key configured${NC}"
        LLM_COUNT=$((LLM_COUNT + 1))
    else
        echo -e "${YELLOW}   ⚠️  XAI key not configured${NC}"
    fi
else
    echo -e "${YELLOW}   ⚠️  XAI key not in .env${NC}"
fi

if [ $LLM_COUNT -eq 0 ]; then
    echo -e "${YELLOW}   ⚠️  No LLM provider configured${NC}"
    echo "   AI features will be limited"
else
    echo -e "${GREEN}   ✅ $LLM_COUNT LLM provider(s) configured${NC}"
fi
echo ""

# Check threat intelligence
echo "5️⃣  Threat Intelligence APIs:"
INTEL_COUNT=0
if grep -q '^ABUSEIPDB_API_KEY=.\+' .env 2>/dev/null; then
    ABUSE_KEY=$(grep '^ABUSEIPDB_API_KEY=' .env | cut -d '=' -f2)
    if [ -n "$ABUSE_KEY" ] && [ "$ABUSE_KEY" != "CONFIGURE_IN_AWS_SECRETS_MANAGER" ]; then
        echo -e "${GREEN}   ✅ AbuseIPDB configured${NC}"
        INTEL_COUNT=$((INTEL_COUNT + 1))
    else
        echo -e "${YELLOW}   ⚠️  AbuseIPDB not configured${NC}"
    fi
else
    echo -e "${YELLOW}   ⚠️  AbuseIPDB not in .env${NC}"
fi

if grep -q '^VIRUSTOTAL_API_KEY=.\+' .env 2>/dev/null; then
    VT_KEY=$(grep '^VIRUSTOTAL_API_KEY=' .env | cut -d '=' -f2)
    if [ -n "$VT_KEY" ] && [ "$VT_KEY" != "CONFIGURE_IN_AWS_SECRETS_MANAGER" ]; then
        echo -e "${GREEN}   ✅ VirusTotal configured${NC}"
        INTEL_COUNT=$((INTEL_COUNT + 1))
    else
        echo -e "${YELLOW}   ⚠️  VirusTotal not configured${NC}"
    fi
else
    echo -e "${YELLOW}   ⚠️  VirusTotal not in .env${NC}"
fi

if [ $INTEL_COUNT -eq 0 ]; then
    echo -e "${YELLOW}   ⚠️  No threat intel APIs configured${NC}"
    echo "   IP reputation checks will be limited"
else
    echo -e "${GREEN}   ✅ $INTEL_COUNT threat intel API(s) configured${NC}"
fi
echo ""

# Check workflow triggers
echo "6️⃣  Workflow Triggers:"
if [ -f xdr.db ]; then
    WORKFLOW_COUNT=$(sqlite3 xdr.db "SELECT COUNT(*) FROM workflow_triggers;" 2>/dev/null || echo "0")
    TPOT_WORKFLOW_COUNT=$(sqlite3 xdr.db "SELECT COUNT(*) FROM workflow_triggers WHERE name LIKE 'T-Pot:%';" 2>/dev/null || echo "0")
    
    if [ "$WORKFLOW_COUNT" -gt 0 ]; then
        echo -e "${GREEN}   ✅ $WORKFLOW_COUNT total workflow triggers${NC}"
        echo -e "${GREEN}   ✅ $TPOT_WORKFLOW_COUNT T-Pot specific workflows${NC}"
        
        # Show auto-execute count
        AUTO_COUNT=$(sqlite3 xdr.db "SELECT COUNT(*) FROM workflow_triggers WHERE auto_execute=1;" 2>/dev/null || echo "0")
        echo "   Auto-execute: $AUTO_COUNT workflows"
        
        # Show enabled count
        ENABLED_COUNT=$(sqlite3 xdr.db "SELECT COUNT(*) FROM workflow_triggers WHERE enabled=1;" 2>/dev/null || echo "0")
        echo "   Enabled: $ENABLED_COUNT workflows"
    else
        echo -e "${YELLOW}   ⚠️  No workflow triggers found${NC}"
        echo "   Run: python3 scripts/tpot-management/setup-tpot-workflows.py"
    fi
else
    echo -e "${RED}   ❌ Database not found (xdr.db)${NC}"
fi
echo ""

# Summary
echo "=============================="
echo "📊 Configuration Summary"
echo "=============================="
echo ""

TOTAL_SCORE=0
MAX_SCORE=6

# Score each component
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    TOTAL_SCORE=$((TOTAL_SCORE + 1))
fi

if grep -q '^API_KEY=.\+' .env 2>/dev/null; then
    TOTAL_SCORE=$((TOTAL_SCORE + 1))
fi

if [ $TPOT_CONFIGURED -eq 1 ]; then
    TOTAL_SCORE=$((TOTAL_SCORE + 1))
fi

if [ $LLM_COUNT -gt 0 ]; then
    TOTAL_SCORE=$((TOTAL_SCORE + 1))
fi

if [ $INTEL_COUNT -gt 0 ]; then
    TOTAL_SCORE=$((TOTAL_SCORE + 1))
fi

if [ "${WORKFLOW_COUNT:-0}" -gt 0 ]; then
    TOTAL_SCORE=$((TOTAL_SCORE + 1))
fi

# Display score
echo "Score: $TOTAL_SCORE / $MAX_SCORE"
echo ""

if [ $TOTAL_SCORE -eq $MAX_SCORE ]; then
    echo -e "${GREEN}🎉 Perfect! All components configured!${NC}"
elif [ $TOTAL_SCORE -ge 4 ]; then
    echo -e "${GREEN}✅ Good! Core components working${NC}"
    echo "Consider configuring remaining components for full functionality"
elif [ $TOTAL_SCORE -ge 2 ]; then
    echo -e "${YELLOW}⚠️  Partial configuration${NC}"
    echo "Several components need configuration"
else
    echo -e "${RED}❌ Minimal configuration${NC}"
    echo "Please configure your .env file"
fi

echo ""
echo "📖 For detailed configuration guide, see:"
echo "   ENV_CONFIGURATION_GUIDE.md"
echo ""


