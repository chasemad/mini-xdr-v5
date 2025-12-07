#!/bin/bash

# Inject Fake SSH Brute Force Attack (WITH AUTHENTICATION)
# Uses proper API authentication

API_URL="http://localhost:8000"

# Get API key from .env
API_KEY=$(grep "^API_KEY=" /Users/chasemad/Desktop/mini-xdr/backend/.env | cut -d'=' -f2)

if [ -z "$API_KEY" ] || [ "$API_KEY" = "WILL_BE_GENERATED_DURING_DEPLOYMENT" ]; then
    echo "⚠️  No API key found in .env"
    echo "Using no authentication (ingestion endpoints should allow this)"
    AUTH_HEADER=""
else
    echo "✅ Using API key authentication"
    AUTH_HEADER="-H \"X-API-Key: $API_KEY\""
fi

# Generate random external IP (not private)
FAKE_IP="45.$((RANDOM % 255)).$((RANDOM % 255)).$((RANDOM % 255))"

echo "═══════════════════════════════════════════════════════════════"
echo "🎯 Injecting Fake SSH Brute Force Attack"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "Attacker IP: $FAKE_IP (fake external IP)"
echo "Target: T-Pot Cowrie honeypot"
echo "Attack: 10 failed SSH login attempts"
echo ""

# Inject 10 failed login attempts
for i in {1..10}; do
    timestamp=$(date -u +"%Y-%m-%dT%H:%M:%S.%3NZ")

    event=$(cat <<EOF
{
    "eventid": "cowrie.login.failed",
    "src_ip": "$FAKE_IP",
    "src_port": $((40000 + i)),
    "dst_ip": "203.0.113.42",
    "dst_port": 22,
    "username": "root",
    "password": "password$i",
    "timestamp": "$timestamp",
    "session": "fake_session_$RANDOM",
    "message": "login attempt [root/password$i] failed",
    "sensor": "cowrie",
    "protocol": "ssh"
}
EOF
)

    # Try with auth, fall back to no auth
    if [ -n "$AUTH_HEADER" ]; then
        response=$(curl -s -X POST "$API_URL/ingest/cowrie" \
            -H "Content-Type: application/json" \
            -H "X-API-Key: $API_KEY" \
            -d "$event" 2>&1)
    else
        response=$(curl -s -X POST "$API_URL/ingest/cowrie" \
            -H "Content-Type: application/json" \
            -d "$event" 2>&1)
    fi

    if echo "$response" | grep -qi "error\|unauthorized\|401"; then
        echo "  [ERROR] Authentication failed on event $i"
        echo "  Response: $response"
        echo ""
        echo "Try disabling ingestion auth temporarily..."
        exit 1
    fi

    echo "  [$i/10] Failed login: root/password$i"
    sleep 0.3
done

echo ""
echo "✅ All 10 attack events injected!"
echo ""
echo "Waiting 3 seconds for detection engine..."
sleep 3

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "📊 Checking for Incident"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# Check for incident
incident_response=$(curl -s "$API_URL/api/incidents")

if echo "$incident_response" | python3 -c "
import sys, json
data = json.load(sys.stdin)
if isinstance(data, list) and len(data) > 0:
    incident = data[0]
    print(f'''
✅ SUCCESS! INCIDENT CREATED!

  Incident ID:     {incident.get('id')}
  Severity:        {incident.get('severity', 'N/A').upper()}
  Source IP:       {incident.get('src_ip', 'N/A')}
  Attack Type:     SSH Brute Force
  Status:          {incident.get('status', 'N/A')}

  Detection Reason:
    {incident.get('reason', 'N/A')}
''')

    # Check for actions
    actions = incident.get('actions', [])
    if actions:
        print(f'  🤖 Automated Actions: {len(actions)}')
        for action in actions[:5]:
            atype = action.get('action_type', 'unknown')
            status = action.get('status', 'unknown')
            print(f'     • {atype}: {status}')
    else:
        print(f'  ⏳ Automated response pending (auto_contain=False)')

    print('')
    print('═══════════════════════════════════════════════════════════════')
    print('')
    print('🎉 Check your dashboard: http://localhost:3000')
    exit(0)
else:
    print('⚠️  No incident found via /api/incidents')
    print('    Trying alternate endpoint...')
    exit(1)
" 2>&1; then
    :
else
    # Check database directly
    echo ""
    python3 << 'PYEOF'
import sqlite3
db = sqlite3.connect("/Users/chasemad/Desktop/mini-xdr/backend/xdr.db")
cursor = db.cursor()
cursor.execute("SELECT COUNT(*) FROM incidents WHERE src_ip LIKE '45.%'")
count = cursor.fetchone()[0]
if count > 0:
    print(f"✅ Found {count} incident(s) in database!")
    cursor.execute("""
        SELECT id, src_ip, severity, reason
        FROM incidents
        WHERE src_ip LIKE '45.%'
        ORDER BY created_at DESC
        LIMIT 1
    """)
    row = cursor.fetchone()
    print(f"\n  ID: {row[0]}")
    print(f"  IP: {row[1]}")
    print(f"  Severity: {row[2]}")
    print(f"  Reason: {row[3][:100]}")
else:
    print("❌ Still no incident in database")
    print("\nPossible issue: Detection logic not running")
db.close()
PYEOF
fi

echo ""
echo "═══════════════════════════════════════════════════════════════"
