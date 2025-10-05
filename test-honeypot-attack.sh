#!/bin/bash
# ========================================================================
# Test Honeypot Attack Simulation
# ========================================================================
# This script simulates various attacks against your T-Pot honeypot
# to test if Mini-XDR can detect and respond
# ========================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Load .env file
ENV_FILE="$(dirname "${BASH_SOURCE[0]}")/backend/.env"
if [ -f "$ENV_FILE" ]; then
    export $(grep -v '^#' "$ENV_FILE" | xargs)
fi

TPOT_IP="${TPOT_HOST:-}"

if [ -z "$TPOT_IP" ]; then
    echo -e "${RED}❌ T-Pot IP not configured in .env${NC}"
    echo -e "${YELLOW}Please run: ./setup-azure-mini-xdr.sh${NC}"
    exit 1
fi

echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║           T-Pot Honeypot Attack Simulation                     ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${YELLOW}⚠️  This script will simulate attacks against: $TPOT_IP${NC}"
echo -e "${YELLOW}⚠️  Ensure Mini-XDR backend is running to capture events${NC}"
echo ""
read -p "Press Enter to continue or Ctrl+C to cancel..."
echo ""

# ========================================================================
# Test 1: SSH Brute Force Simulation
# ========================================================================
echo -e "${YELLOW}[TEST 1/5]${NC} SSH Brute Force Simulation..."
echo -e "${BLUE}Attempting multiple SSH login failures...${NC}"

for i in {1..10}; do
    echo -e "${BLUE}  Attempt $i/10...${NC}"
    timeout 2 ssh -o StrictHostKeyChecking=no -o ConnectTimeout=2 \
        "fakeuser$i@$TPOT_IP" -p 22 2>/dev/null || true
    sleep 1
done

echo -e "${GREEN}✅ SSH brute force simulation complete${NC}"
echo ""
sleep 2

# ========================================================================
# Test 2: Port Scan Simulation
# ========================================================================
echo -e "${YELLOW}[TEST 2/5]${NC} Port Scan Simulation..."
echo -e "${BLUE}Scanning common ports...${NC}"

if command -v nmap &> /dev/null; then
    nmap -Pn -p 21,22,23,25,80,110,143,443,445,3389 "$TPOT_IP" --max-retries 1 --host-timeout 10s 2>/dev/null || true
    echo -e "${GREEN}✅ Port scan complete${NC}"
else
    echo -e "${YELLOW}⚠️  nmap not found, using nc instead${NC}"
    for port in 21 22 23 25 80 110 143 443 445 3389; do
        echo -e "${BLUE}  Scanning port $port...${NC}"
        timeout 2 nc -zv "$TPOT_IP" "$port" 2>/dev/null || true
    done
    echo -e "${GREEN}✅ Port scan complete${NC}"
fi

echo ""
sleep 2

# ========================================================================
# Test 3: HTTP Probing
# ========================================================================
echo -e "${YELLOW}[TEST 3/5]${NC} HTTP Probing..."
echo -e "${BLUE}Probing web server...${NC}"

# Common web attack patterns
curl -s -o /dev/null --max-time 5 "http://$TPOT_IP/" 2>/dev/null || true
curl -s -o /dev/null --max-time 5 "http://$TPOT_IP/admin" 2>/dev/null || true
curl -s -o /dev/null --max-time 5 "http://$TPOT_IP/login" 2>/dev/null || true
curl -s -o /dev/null --max-time 5 "http://$TPOT_IP/../../../etc/passwd" 2>/dev/null || true
curl -s -o /dev/null --max-time 5 "http://$TPOT_IP/shell.php" 2>/dev/null || true

echo -e "${GREEN}✅ HTTP probing complete${NC}"
echo ""
sleep 2

# ========================================================================
# Test 4: Telnet Connection Attempts
# ========================================================================
echo -e "${YELLOW}[TEST 4/5]${NC} Telnet Connection Attempts..."
echo -e "${BLUE}Connecting to Telnet honeypot...${NC}"

for i in {1..3}; do
    echo -e "${BLUE}  Attempt $i/3...${NC}"
    timeout 3 telnet "$TPOT_IP" 23 2>/dev/null << EOF || true
admin
admin
exit
EOF
    sleep 1
done

echo -e "${GREEN}✅ Telnet probing complete${NC}"
echo ""
sleep 2

# ========================================================================
# Test 5: FTP Connection Attempts
# ========================================================================
echo -e "${YELLOW}[TEST 5/5]${NC} FTP Connection Attempts..."
echo -e "${BLUE}Connecting to FTP honeypot...${NC}"

if command -v ftp &> /dev/null; then
    timeout 5 ftp -n "$TPOT_IP" 21 << EOF || true
user anonymous
pass anonymous@
quit
EOF
    echo -e "${GREEN}✅ FTP probing complete${NC}"
else
    echo -e "${YELLOW}⚠️  ftp command not found, using nc instead${NC}"
    timeout 3 nc "$TPOT_IP" 21 << EOF || true
USER anonymous
PASS anonymous@
QUIT
EOF
    echo -e "${GREEN}✅ FTP probing complete${NC}"
fi

echo ""

# ========================================================================
# Summary
# ========================================================================
echo -e "${GREEN}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║              Attack Simulation Complete! ✅                    ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${BLUE}📊 Summary:${NC}"
echo -e "  • SSH brute force: ${GREEN}10 attempts${NC}"
echo -e "  • Port scan: ${GREEN}10 ports${NC}"
echo -e "  • HTTP probing: ${GREEN}5 requests${NC}"
echo -e "  • Telnet attempts: ${GREEN}3 connections${NC}"
echo -e "  • FTP attempts: ${GREEN}1 connection${NC}"
echo ""
echo -e "${BLUE}🔍 Next Steps:${NC}"
echo -e "  1. Check Mini-XDR dashboard for detected events"
echo -e "  2. Review T-Pot logs at: ${GREEN}https://$TPOT_IP:64297${NC}"
echo -e "  3. Verify alerts and containment actions in Mini-XDR"
echo ""
echo -e "${YELLOW}💡 Tip: Check Mini-XDR logs with:${NC}"
echo -e "  ${YELLOW}tail -f backend/backend.log${NC}"
echo ""


