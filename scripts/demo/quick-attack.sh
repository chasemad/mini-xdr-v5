#!/bin/bash
#
# 🚀 Quick Attack Injection for Mini-XDR
#
# A minimal script that generates a simple brute force attack
# for quick testing. Use run-demo.sh for full demonstrations.
#

API_URL="http://localhost:8000"

# Generate random external IP
ATTACKER_IP="45.$((RANDOM % 255)).$((RANDOM % 255)).$((RANDOM % 255))"
SESSION="quick_demo_$(date +%s)"

echo ""
echo "🎯 Quick Attack Injection"
echo "========================="
echo ""
echo "Attacker IP: $ATTACKER_IP"
echo "Target: SSH (port 22)"
echo ""

# Common credentials for brute force
CREDENTIALS=(
    "root:root"
    "root:admin"
    "root:123456"
    "root:password"
    "root:toor"
    "admin:admin"
    "admin:123456"
    "admin:password"
    "ubuntu:ubuntu"
    "test:test"
    "root:qwerty"
    "root:letmein"
    "root:welcome"
    "admin:admin123"
    "user:user123"
)

echo "📤 Injecting ${#CREDENTIALS[@]} brute force events..."
echo ""

EVENTS="["
FIRST=true

for cred in "${CREDENTIALS[@]}"; do
    username="${cred%%:*}"
    password="${cred##*:}"
    timestamp=$(date -u +"%Y-%m-%dT%H:%M:%S.%3NZ")

    if [ "$FIRST" = true ]; then
        FIRST=false
    else
        EVENTS+=","
    fi

    EVENTS+='{
        "eventid": "cowrie.login.failed",
        "src_ip": "'$ATTACKER_IP'",
        "src_port": '$((40000 + RANDOM % 10000))',
        "dst_ip": "203.0.113.42",
        "dst_port": 22,
        "username": "'$username'",
        "password": "'$password'",
        "timestamp": "'$timestamp'",
        "session": "'$SESSION'_'$RANDOM'",
        "message": "login attempt ['$username'/'$password'] failed",
        "sensor": "cowrie",
        "protocol": "ssh"
    }'

    echo "  • $username / $password"
done

# Add a successful login at the end
timestamp=$(date -u +"%Y-%m-%dT%H:%M:%S.%3NZ")
EVENTS+=',{
    "eventid": "cowrie.login.success",
    "src_ip": "'$ATTACKER_IP'",
    "src_port": 46000,
    "dst_ip": "203.0.113.42",
    "dst_port": 22,
    "username": "root",
    "password": "toor",
    "timestamp": "'$timestamp'",
    "session": "'$SESSION'_success",
    "message": "login success [root/toor]",
    "sensor": "cowrie",
    "protocol": "ssh"
}'

EVENTS+="]"

echo ""
echo "  🚨 root / toor (SUCCESS!)"
echo ""

# Send batch
echo "📡 Sending to API..."
response=$(curl -s -X POST "$API_URL/api/ingest/cowrie" \
    -H "Content-Type: application/json" \
    -d "$EVENTS" 2>&1)

if echo "$response" | grep -qi "error"; then
    echo "❌ Error: $response"
    exit 1
fi

echo "✅ Events sent successfully"
echo ""
echo "⏳ Waiting for detection (5 seconds)..."
sleep 5

# Check for incident
echo ""
echo "🔍 Checking for incident..."

incident=$(curl -s "$API_URL/api/incidents" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    for inc in data:
        if inc.get('src_ip') == '$ATTACKER_IP':
            print(f'''
✅ INCIDENT DETECTED!

  ID:         {inc.get('id')}
  IP:         {inc.get('src_ip')}
  Severity:   {inc.get('escalation_level', 'N/A')}
  Status:     {inc.get('status')}
  Confidence: {inc.get('ml_confidence', inc.get('containment_confidence', 0)) * 100:.1f}%

🎯 View in UI: http://localhost:3000/incidents/incident/{inc.get('id')}
''')
            sys.exit(0)
    print('⏳ No incident found yet - check UI manually')
except Exception as e:
    print(f'Could not parse response: {e}')
" 2>&1)

echo "$incident"
echo ""
