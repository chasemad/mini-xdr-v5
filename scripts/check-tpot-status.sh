#!/bin/bash
# ========================================================================
# Check T-Pot Honeypot Status
# ========================================================================
# Quick script to check if T-Pot is up and running
# ========================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Load .env
ENV_FILE="$(dirname "${BASH_SOURCE[0]}")/backend/.env"
if [ -f "$ENV_FILE" ]; then
    export $(grep -v '^#' "$ENV_FILE" | xargs)
fi

TPOT_IP="${TPOT_HOST:-}"
SSH_KEY="${HONEYPOT_SSH_KEY:-$HOME/.ssh/mini-xdr-tpot-azure}"

if [ -z "$TPOT_IP" ]; then
    echo -e "${RED}❌ T-Pot IP not configured${NC}"
    exit 1
fi

echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║              T-Pot Honeypot Status Check                       ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check if VM is reachable
echo -e "${YELLOW}[1/4]${NC} Checking VM connectivity..."
if ping -c 2 "$TPOT_IP" &> /dev/null; then
    echo -e "${GREEN}✅ VM is reachable: $TPOT_IP${NC}"
else
    echo -e "${RED}❌ VM is not reachable${NC}"
    exit 1
fi
echo ""

# Check SSH access
echo -e "${YELLOW}[2/4]${NC} Checking SSH access..."
if [ -f "$SSH_KEY" ]; then
    if timeout 5 ssh -o StrictHostKeyChecking=no -o ConnectTimeout=3 -i "$SSH_KEY" "azureuser@$TPOT_IP" "echo ok" &> /dev/null; then
        echo -e "${GREEN}✅ SSH access working${NC}"
        SSH_WORKING=true
    else
        echo -e "${RED}❌ Cannot SSH to VM${NC}"
        SSH_WORKING=false
    fi
else
    echo -e "${RED}❌ SSH key not found: $SSH_KEY${NC}"
    SSH_WORKING=false
fi
echo ""

# Check T-Pot web interface
echo -e "${YELLOW}[3/4]${NC} Checking T-Pot web interface..."
if timeout 5 curl -k -s -o /dev/null -w "%{http_code}" "https://$TPOT_IP:64297" | grep -q "200\|301\|302\|401"; then
    echo -e "${GREEN}✅ T-Pot web interface is responding${NC}"
    echo -e "${BLUE}   Access at: https://$TPOT_IP:64297${NC}"
else
    echo -e "${YELLOW}⚠️  T-Pot web interface not responding (may still be installing)${NC}"
fi
echo ""

# Check T-Pot services (if SSH is working)
echo -e "${YELLOW}[4/4]${NC} Checking T-Pot services..."
if [ "$SSH_WORKING" = true ]; then
    echo -e "${BLUE}Running diagnostics on VM...${NC}"
    ssh -o StrictHostKeyChecking=no -i "$SSH_KEY" "azureuser@$TPOT_IP" << 'EOF'
#!/bin/bash

# Check if T-Pot is installed
if [ -d "/opt/tpotce" ]; then
    echo "  ✅ T-Pot installed"
else
    echo "  ⚠️  T-Pot not yet installed"
    exit 0
fi

# Check Docker
if command -v docker &> /dev/null; then
    echo "  ✅ Docker installed"
    
    # Check Docker containers
    CONTAINERS=$(docker ps --format "{{.Names}}" 2>/dev/null | wc -l)
    if [ "$CONTAINERS" -gt 0 ]; then
        echo "  ✅ T-Pot containers running: $CONTAINERS"
        echo ""
        echo "  Active honeypots:"
        docker ps --format "    • {{.Names}}" | grep -v "compose\|logstash\|elasticsearch\|kibana\|nginx" || echo "    (still starting up...)"
    else
        echo "  ⚠️  No T-Pot containers running (may be starting)"
    fi
else
    echo "  ⚠️  Docker not installed"
fi

# Check system resources
echo ""
echo "  System Resources:"
echo "    • Uptime: $(uptime -p 2>/dev/null || uptime)"
echo "    • Memory: $(free -h | awk '/^Mem:/ {print $3 "/" $2}')"
echo "    • Disk: $(df -h / | awk 'NR==2 {print $3 "/" $2 " (" $5 " used)"}')"
EOF
else
    echo -e "${YELLOW}⚠️  Cannot check T-Pot services (SSH not working)${NC}"
fi

echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                    Status Check Complete                       ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${BLUE}📋 Quick Commands:${NC}"
echo -e "  • SSH to VM: ${YELLOW}ssh -i $SSH_KEY azureuser@$TPOT_IP${NC}"
echo -e "  • View T-Pot logs: ${YELLOW}ssh -i $SSH_KEY azureuser@$TPOT_IP 'docker logs nginx'${NC}"
echo -e "  • Restart T-Pot: ${YELLOW}ssh -i $SSH_KEY azureuser@$TPOT_IP 'cd /opt/tpotce && sudo docker-compose restart'${NC}"
echo ""

