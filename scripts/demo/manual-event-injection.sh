#!/bin/bash
# ========================================================================
# Manual Event Injection - Backup for T-Pot Connection Issues
# ========================================================================
# Use this if T-Pot is not accessible during demo
# Directly injects events into Mini-XDR via API
# ========================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

API_URL="${API_URL:-http://localhost:8000}"
ATTACKER_IP="${ATTACKER_IP:-203.0.113.100}"

echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║           Manual Event Injection (T-Pot Backup)                ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${CYAN}API Endpoint: ${GREEN}$API_URL${NC}"
echo -e "${CYAN}Simulated Attacker IP: ${GREEN}$ATTACKER_IP${NC}"
echo ""

# ========================================================================
# Function: Inject Event
# ========================================================================
inject_event() {
    local event_type="$1"
    local description="$2"
    local payload="$3"

    echo -e "${BLUE}Injecting: ${CYAN}$description${NC}"

    response=$(curl -s -X POST "$API_URL/ingest/multi" \
        -H "Content-Type: application/json" \
        -d "$payload")

    if echo "$response" | grep -q "success\|processed\|created" 2>/dev/null; then
        echo -e "  ${GREEN}✓${NC} Event injected successfully"
    else
        echo -e "  ${YELLOW}⚠${NC} Response: $(echo $response | head -c 100)"
    fi

    sleep 1
}

# ========================================================================
# Event 1: SSH Brute Force Attack
# ========================================================================
echo -e "${YELLOW}[1/5] SSH Brute Force Attack${NC}"

TIMESTAMP=$(date -u +%Y-%m-%dT%H:%M:%S)Z

SSH_PAYLOAD=$(cat <<EOF
{
  "source_type": "cowrie",
  "hostname": "demo-honeypot",
  "events": [
    {
      "eventid": "cowrie.login.failed",
      "src_ip": "$ATTACKER_IP",
      "dst_port": 22,
      "username": "root",
      "password": "admin123",
      "timestamp": "$TIMESTAMP",
      "session": "demo-session-001"
    },
    {
      "eventid": "cowrie.login.failed",
      "src_ip": "$ATTACKER_IP",
      "dst_port": 22,
      "username": "admin",
      "password": "password",
      "timestamp": "$TIMESTAMP",
      "session": "demo-session-002"
    },
    {
      "eventid": "cowrie.login.failed",
      "src_ip": "$ATTACKER_IP",
      "dst_port": 22,
      "username": "test",
      "password": "123456",
      "timestamp": "$TIMESTAMP",
      "session": "demo-session-003"
    },
    {
      "eventid": "cowrie.login.failed",
      "src_ip": "$ATTACKER_IP",
      "dst_port": 22,
      "username": "administrator",
      "password": "admin",
      "timestamp": "$TIMESTAMP",
      "session": "demo-session-004"
    },
    {
      "eventid": "cowrie.login.failed",
      "src_ip": "$ATTACKER_IP",
      "dst_port": 22,
      "username": "user",
      "password": "root",
      "timestamp": "$TIMESTAMP",
      "session": "demo-session-005"
    }
  ]
}
EOF
)

inject_event "ssh_bruteforce" "5 SSH brute force attempts" "$SSH_PAYLOAD"
echo ""

# ========================================================================
# Event 2: SQL Injection Attempt
# ========================================================================
echo -e "${YELLOW}[2/5] SQL Injection Attempt${NC}"

SQL_PAYLOAD=$(cat <<EOF
{
  "source_type": "suricata",
  "hostname": "demo-honeypot",
  "events": [
    {
      "eventid": "suricata.alert",
      "src_ip": "$ATTACKER_IP",
      "dst_port": 80,
      "alert_signature": "SQL Injection Attempt",
      "alert_category": "Web Application Attack",
      "alert_severity": 1,
      "http_url": "/login?user=admin' OR 1=1--",
      "http_method": "GET",
      "http_user_agent": "sqlmap/1.5.2",
      "timestamp": "$TIMESTAMP"
    }
  ]
}
EOF
)

inject_event "sql_injection" "SQL injection attempt via web" "$SQL_PAYLOAD"
echo ""

# ========================================================================
# Event 3: Port Scanning
# ========================================================================
echo -e "${YELLOW}[3/5] Port Scanning Activity${NC}"

PORTSCAN_PAYLOAD=$(cat <<EOF
{
  "source_type": "suricata",
  "hostname": "demo-honeypot",
  "events": [
    {
      "eventid": "suricata.alert",
      "src_ip": "$ATTACKER_IP",
      "dst_port": 22,
      "alert_signature": "Port Scan Detected",
      "alert_category": "Attempted Information Leak",
      "alert_severity": 2,
      "proto": "TCP",
      "timestamp": "$TIMESTAMP"
    },
    {
      "eventid": "suricata.alert",
      "src_ip": "$ATTACKER_IP",
      "dst_port": 80,
      "alert_signature": "Port Scan Detected",
      "alert_category": "Attempted Information Leak",
      "alert_severity": 2,
      "proto": "TCP",
      "timestamp": "$TIMESTAMP"
    },
    {
      "eventid": "suricata.alert",
      "src_ip": "$ATTACKER_IP",
      "dst_port": 443,
      "alert_signature": "Port Scan Detected",
      "alert_category": "Attempted Information Leak",
      "alert_severity": 2,
      "proto": "TCP",
      "timestamp": "$TIMESTAMP"
    }
  ]
}
EOF
)

inject_event "port_scan" "Port scanning on 3 ports" "$PORTSCAN_PAYLOAD"
echo ""

# ========================================================================
# Event 4: Malware Download Attempt
# ========================================================================
echo -e "${YELLOW}[4/5] Malware Download Attempt${NC}"

MALWARE_PAYLOAD=$(cat <<EOF
{
  "source_type": "dionaea",
  "hostname": "demo-honeypot",
  "events": [
    {
      "eventid": "dionaea.download.offer",
      "src_ip": "$ATTACKER_IP",
      "dst_port": 445,
      "url": "http://malicious-site.com/payload.exe",
      "md5": "d41d8cd98f00b204e9800998ecf8427e",
      "timestamp": "$TIMESTAMP"
    }
  ]
}
EOF
)

inject_event "malware" "Malware download via SMB" "$MALWARE_PAYLOAD"
echo ""

# ========================================================================
# Event 5: Web Path Traversal
# ========================================================================
echo -e "${YELLOW}[5/5] Web Path Traversal Attack${NC}"

TRAVERSAL_PAYLOAD=$(cat <<EOF
{
  "source_type": "suricata",
  "hostname": "demo-honeypot",
  "events": [
    {
      "eventid": "suricata.alert",
      "src_ip": "$ATTACKER_IP",
      "dst_port": 80,
      "alert_signature": "Directory Traversal Attempt",
      "alert_category": "Web Application Attack",
      "alert_severity": 1,
      "http_url": "/admin/../../etc/passwd",
      "http_method": "GET",
      "http_user_agent": "Mozilla/5.0 (compatible; SecurityScanner/1.0)",
      "timestamp": "$TIMESTAMP"
    }
  ]
}
EOF
)

inject_event "path_traversal" "Directory traversal attempt" "$TRAVERSAL_PAYLOAD"
echo ""

# ========================================================================
# Summary
# ========================================================================
echo -e "${GREEN}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║              Event Injection Complete!                         ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${BLUE}📊 Events Injected:${NC}"
echo -e "  • SSH Brute Force: ${GREEN}5 attempts${NC}"
echo -e "  • SQL Injection: ${GREEN}1 attempt${NC}"
echo -e "  • Port Scanning: ${GREEN}3 ports${NC}"
echo -e "  • Malware Download: ${GREEN}1 attempt${NC}"
echo -e "  • Path Traversal: ${GREEN}1 attempt${NC}"
echo ""
echo -e "${BLUE}🔍 Check Mini-XDR Dashboard:${NC}"
echo -e "  • Incidents: ${CYAN}http://localhost:3000/incidents${NC}"
echo -e "  • Dashboard: ${CYAN}http://localhost:3000${NC}"
echo ""
echo -e "${BLUE}📝 Verify Events:${NC}"
echo -e "  ${CYAN}curl http://localhost:8000/incidents | jq 'length'${NC}"
echo ""
echo -e "${GREEN}✨ Events ready for demo analysis! ✨${NC}"
echo ""
