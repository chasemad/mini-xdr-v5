#!/bin/bash

# Inject Fake SSH Brute Force Attack
# Uses a fake external IP to trigger incident detection

API_URL="http://localhost:8000"

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

    curl -s -X POST "$API_URL/ingest/cowrie" \
        -H "Content-Type: application/json" \
        -d "$event" > /dev/null

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
        print(f'  ⏳ Automated response pending...')

    exit(0)
else:
    print('❌ No incident created yet')
    print('')
    print('Checking events...')
    exit(1)
" 2>&1; then
    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    echo ""
    echo "🎉 Demo successful! Check your dashboard:"
    echo "   http://localhost:3000"
    echo ""
    echo "You should see:"
    echo "  • New incident for $FAKE_IP"
    echo "  • AI agents analyzing the threat"
    echo "  • Automated containment actions"
    echo ""
else
    # Still check if incident exists with different query
    echo ""
    echo "Trying alternate query..."
    curl -s "$API_URL/api/incidents?limit=1" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    incidents = data.get('incidents', []) if isinstance(data, dict) else data
    if incidents and len(incidents) > 0:
        incident = incidents[0]
        print(f'''
✅ INCIDENT FOUND!

  ID: {incident.get('id')}
  IP: {incident.get('src_ip')}
  Severity: {incident.get('severity')}
''')
    else:
        print('Still no incident. Checking backend logs...')
        print('')
        print('In your backend terminal, look for:')
        print('  • \"SSH brute-force check\"')
        print('  • \"NEW INCIDENT\"')
        print('  • Any ERROR messages')
except Exception as e:
    print(f'Error: {e}')
" 2>&1
fi

echo ""
echo "═══════════════════════════════════════════════════════════════"
