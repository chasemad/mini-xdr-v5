#!/bin/bash

# Test SSH Brute Force Detection
# Injects fake Cowrie events to verify the detection engine works

API_URL="http://localhost:8000"
TEST_IP="198.51.100.1"  # TEST-NET-2 (safe, reserved IP)

echo "═══════════════════════════════════════════════════════════════"
echo "🧪 Testing SSH Brute Force Detection"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "This will inject 6 fake SSH login failures to trigger detection"
echo "Test IP: $TEST_IP (safe test IP, won't block real traffic)"
echo ""

# Inject 6 failed login events
echo "Injecting attack events..."
for i in {1..6}; do
    timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

    event=$(cat <<EOF
{
    "eventid": "cowrie.login.failed",
    "src_ip": "$TEST_IP",
    "src_port": $((40000 + i)),
    "dst_port": 22,
    "username": "admin",
    "password": "password$i",
    "timestamp": "$timestamp",
    "session": "test_session_$i",
    "message": "login attempt [admin/password$i] failed"
}
EOF
)

    response=$(curl -s -X POST "$API_URL/api/ingest/cowrie" \
        -H "Content-Type: application/json" \
        -d "$event")

    echo "  ✓ Event $i/6 ingested"
    sleep 0.5
done

echo ""
echo "Waiting 3 seconds for detection..."
sleep 3

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "📊 Checking Results"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# Check incidents
echo "Checking for incident..."
incidents=$(curl -s "$API_URL/api/incidents" | python3 -c "
import sys, json
data = json.load(sys.stdin)
if data and len(data) > 0:
    incident = data[0]
    print(f'''
✅ INCIDENT CREATED!

  ID: {incident.get('id')}
  Severity: {incident.get('severity', 'N/A')}
  Source IP: {incident.get('src_ip', 'N/A')}
  Status: {incident.get('status', 'N/A')}

  Reason: {incident.get('reason', 'N/A')}
''')

    # Check actions
    actions = incident.get('actions', [])
    if actions:
        print(f'  Actions taken: {len(actions)}')
        for action in actions[:3]:
            print(f'    • {action.get('action_type')}: {action.get('status')}')

    exit(0)
else:
    print('❌ No incident created')
    print('')
    print('Possible issues:')
    print('  • Detection threshold not met')
    print('  • Detector not enabled')
    print('  • Events not processed')
    exit(1)
" 2>&1)

echo "$incidents"

if [ $? -eq 0 ]; then
    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    echo "✅ SUCCESS! Detection engine is working!"
    echo "═══════════════════════════════════════════════════════════════"
    echo ""
    echo "The issue is that Cowrie on T-Pot is not logging your attacks"
    echo "from 172.16.110.1 (local subnet)."
    echo ""
    echo "To test with real T-Pot attacks:"
    echo "  1. Wait for internet attackers (T-Pot gets scanned constantly)"
    echo "  2. Or attack from external IP/network"
    echo "  3. Or use public IP via VPN"
else
    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    echo "⚠️  Detection engine may need configuration"
    echo "═══════════════════════════════════════════════════════════════"
    echo ""
    echo "Check backend logs for errors"
fi
