#!/bin/bash

# Watch for Real Internet Attacks on T-Pot
# T-Pot gets scanned constantly - just wait and watch!

echo "═══════════════════════════════════════════════════════════════"
echo "👀 Watching for Real Internet Attacks on T-Pot"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "T-Pot is connected and monitoring Cowrie honeypot logs."
echo "The internet constantly scans T-Pot - attacks will appear soon!"
echo ""
echo "Press Ctrl+C to stop"
echo ""
echo "══════════════════════════════════════════════════════════════="
echo ""

last_incident_count=0
last_event_count=0

while true; do
    # Check incidents
    incident_data=$(curl -s http://localhost:8000/api/incidents 2>/dev/null)
    incident_count=$(echo "$incident_data" | python3 -c "import sys, json; print(len(json.load(sys.stdin)))" 2>/dev/null || echo "0")

    # Check for new incidents
    if [ "$incident_count" -gt "$last_incident_count" ]; then
        echo ""
        echo "🚨 NEW INCIDENT DETECTED!"
        echo "$incident_data" | python3 -c "
import sys, json
data = json.load(sys.stdin)
if data and len(data) > 0:
    inc = data[0]
    print(f'''
  ID: {inc.get('id')}
  Attacker IP: {inc.get('src_ip')}
  Severity: {inc.get('severity', 'N/A').upper()}
  Status: {inc.get('status')}
  Reason: {inc.get('reason', 'N/A')[:80]}
''')
"
        echo "View in dashboard: http://localhost:3000"
        echo "══════════════════════════════════════════════════════════════="
    fi

    last_incident_count=$incident_count

    # Check database for events
    event_count=$(sqlite3 $(cd "$(dirname "$0")/../.." .. pwd)/backend/xdr.db "SELECT COUNT(*) FROM events WHERE eventid LIKE '%cowrie%'" 2>/dev/null || echo "0")

    if [ "$event_count" -gt "$last_event_count" ]; then
        new_events=$((event_count - last_event_count))
        echo "[$(date +%H:%M:%S)] 📥 +$new_events new events (Total: $event_count events, $incident_count incidents)"

        # Show latest event
        sqlite3 $(cd "$(dirname "$0")/../.." .. pwd)/backend/xdr.db "
            SELECT src_ip, eventid, ts
            FROM events
            WHERE eventid LIKE '%cowrie%'
            ORDER BY ts DESC
            LIMIT 1
        " 2>/dev/null | while read line; do
            echo "    Latest: $line"
        done
    fi

    last_event_count=$event_count

    sleep 5
done
