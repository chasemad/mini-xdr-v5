#!/bin/bash
# Advanced Attack Chain Simulation for SOC Training
# Simulates a complete web application attack with post-exploitation activities

set -e

# Load .env file to get Azure T-Pot IP
ENV_FILE="$(cd "$(dirname "$0")/../.." && pwd)/backend/.env"
if [ -f "$ENV_FILE" ]; then
    export $(grep -v '^#' "$ENV_FILE" | grep 'TPOT_HOST\|HONEYPOT_HOST' | xargs)
fi

# Use Azure T-Pot IP if available, otherwise prompt user
HONEYPOT_IP="${TPOT_HOST:-${HONEYPOT_HOST:-}}"

if [ -z "$HONEYPOT_IP" ]; then
    echo "❌ T-Pot IP not configured!"
    echo "Please enter your Azure T-Pot public IP address:"
    read -p "T-Pot IP: " HONEYPOT_IP
    if [ -z "$HONEYPOT_IP" ]; then
        echo "Error: No IP provided. Exiting."
        exit 1
    fi
fi

XDR_API="http://localhost:8000"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SEND_SCRIPT="$SCRIPT_DIR/../auth/send_signed_request.py"

send_ingest() {
  local payload="$1"
  python3 "$SEND_SCRIPT" --base-url "$XDR_API" --path /ingest/multi --body "$payload" >/dev/null
}

ATTACKER_IP=$(curl -s ifconfig.me || echo "203.0.113.10")


echo "🎯 Starting Advanced Attack Chain Simulation"
echo "Target: $HONEYPOT_IP"
echo "Monitoring: $XDR_API"
echo ""

# Phase 1: Initial Reconnaissance
echo "🔍 Phase 1: Reconnaissance"
sleep 2
curl -s "http://$HONEYPOT_IP/robots.txt" > /dev/null || true
curl -s "http://$HONEYPOT_IP/.git/config" > /dev/null || true
curl -s "http://$HONEYPOT_IP/admin/" > /dev/null || true
curl -s "http://$HONEYPOT_IP/wp-admin/" > /dev/null || true
curl -s "http://$HONEYPOT_IP/phpmyadmin/" > /dev/null || true
echo "   ✓ Directory enumeration complete"
sleep 1

# Phase 2: SQL Injection Testing
echo "🚨 Phase 2: SQL Injection Testing"
sleep 2
curl -G "http://$HONEYPOT_IP/login.php" --data-urlencode "user=admin' OR 1=1--" --data-urlencode "pass=test" || true
sleep 1
curl -G "http://$HONEYPOT_IP/search.php" --data-urlencode "q=' UNION SELECT version()--" || true
sleep 1
curl -G "http://$HONEYPOT_IP/product.php" --data-urlencode "id=1' AND (SELECT COUNT(*) FROM information_schema.tables)>0--" || true
echo "   ✓ SQL injection vectors tested"
sleep 1

# Phase 3: Successful Exploitation (Simulated)
echo "💥 Phase 3: Successful Database Access"
sleep 2
timestamp=$(date -u +%Y-%m-%dT%H:%M:%S.%6NZ)
payload=$(cat <<JSON
{
  "source_type": "webhoneypot",
  "hostname": "honeypot-vm",
  "events": [{
    "eventid": "webhoneypot.attack",
    "timestamp": "$timestamp",
    "src_ip": "$ATTACKER_IP",
    "dst_port": 80,
    "message": "POST /admin/users.php",
    "raw": {
      "method": "POST",
      "path": "/admin/users.php",
      "status_code": 200,
      "response_size": 2048,
      "attack_indicators": ["sql_injection", "admin_access"],
      "success_indicators": ["SELECT * FROM users", "admin password hash retrieved"]
    }
  }]
}
JSON
)
send_ingest "$payload"
echo "   ✓ Database access achieved"
sleep 1

# Phase 4: Privilege Escalation Simulation
echo "⬆️ Phase 4: Privilege Escalation"
sleep 2
timestamp=$(date -u +%Y-%m-%dT%H:%M:%S.%6NZ)
payload=$(cat <<JSON
{
  "source_type": "webhoneypot",
  "hostname": "honeypot-vm",
  "events": [{
    "eventid": "webhoneypot.privilege",
    "timestamp": "$timestamp",
    "src_ip": "$ATTACKER_IP",
    "dst_port": 80,
    "message": "POST /admin/exec.php",
    "raw": {
      "method": "POST",
      "path": "/admin/exec.php",
      "status_code": 200,
      "command": "sudo -l; whoami; id",
      "attack_indicators": ["privilege_escalation", "command_injection"],
      "success_indicators": ["root privileges obtained"]
    }
  }]
}
JSON
)
send_ingest "$payload"
echo "   ✓ Privilege escalation attempted"
sleep 1

# Phase 5: Data Exfiltration
echo "📤 Phase 5: Data Exfiltration"
sleep 2
timestamp=$(date -u +%Y-%m-%dT%H:%M:%S.%6NZ)
payload=$(cat <<JSON
{
  "source_type": "webhoneypot",
  "hostname": "honeypot-vm",
  "events": [{
    "eventid": "webhoneypot.exfil",
    "timestamp": "$timestamp",
    "src_ip": "$ATTACKER_IP",
    "dst_port": 80,
    "message": "GET /admin/export.php?table=users&format=sql",
    "raw": {
      "method": "GET",
      "path": "/admin/export.php",
      "query": {
        "table": "users",
        "format": "sql"
      },
      "status_code": 200,
      "attack_indicators": ["data_exfiltration"],
      "success_indicators": ["mysqldump executed", "user database exported"],
      "data_size": "15MB"
    }
  }]
}
JSON
)
send_ingest "$payload"
echo "   ✓ Database export completed"
sleep 1

# Phase 6: Persistence Establishment
echo "🔒 Phase 6: Persistence Mechanisms"
sleep 2
timestamp=$(date -u +%Y-%m-%dT%H:%M:%S.%6NZ)
payload=$(cat <<JSON
{
  "source_type": "webhoneypot",
  "hostname": "honeypot-vm",
  "events": [{
    "eventid": "webhoneypot.persistence",
    "timestamp": "$timestamp",
    "src_ip": "$ATTACKER_IP",
    "dst_port": 80,
    "message": "POST /admin/backdoor.php",
    "raw": {
      "method": "POST",
      "path": "/admin/backdoor.php",
      "status_code": 200,
      "attack_indicators": ["persistence", "backdoor_creation"],
      "success_indicators": ["crontab entry created", "web shell uploaded", "CREATE USER backdoor_user"],
      "files_created": ["/var/www/html/system.php", "/tmp/.hidden_shell"]
    }
  }]
}
JSON
)
send_ingest "$payload"
echo "   ✓ Backdoor mechanisms installed"
sleep 1

# Phase 7: Lateral Movement Preparation
echo "↔️ Phase 7: Network Reconnaissance"
sleep 2
timestamp=$(date -u +%Y-%m-%dT%H:%M:%S.%6NZ)
payload=$(cat <<JSON
{
  "source_type": "webhoneypot",
  "hostname": "honeypot-vm",
  "events": [{
    "eventid": "webhoneypot.recon",
    "timestamp": "$timestamp",
    "src_ip": "$ATTACKER_IP",
    "dst_port": 80,
    "message": "POST /admin/scan.php",
    "raw": {
      "method": "POST",
      "path": "/admin/scan.php",
      "status_code": 200,
      "attack_indicators": ["lateral_movement", "network_scanning"],
      "success_indicators": ["nmap -sn 192.168.168.0/24", "netstat -rn", "arp -a"],
      "targets_discovered": ["192.168.168.1", "192.168.168.100", "192.168.168.134"]
    }
  }]
}
JSON
)
send_ingest "$payload"
echo "   ✓ Network mapping completed"
sleep 2

echo ""
echo "🎯 Attack Chain Complete!"
echo ""
echo "📊 SOC Analysis Available:"
echo "   • Navigate to: http://localhost:3000/incidents"
echo "   • Look for the new incident with multi-phase attack"
echo "   • Review compromise assessment indicators"
echo "   • Check post-exploitation activities"
echo ""
echo "🚨 Expected Detection Categories:"
echo "   ✓ SQL Injection"
echo "   ✓ Database Access Patterns"
echo "   ✓ Privilege Escalation Indicators"
echo "   ✓ Data Exfiltration Indicators"
echo "   ✓ Persistence Mechanisms"
echo "   ✓ Lateral Movement Indicators"
echo "   ✓ Reconnaissance Patterns"
echo ""
echo "This simulates a complete APT-style attack for SOC training!"
