#!/bin/bash
# Complete End-to-End Mini-XDR Test

echo "🛡️  Mini-XDR End-to-End Test"
echo "============================="

API_KEY="xdr-secure-api-key-2024"
BACKEND_URL="http://localhost:8000"

# Step 1: Check system health
echo "1. System Health Check..."
health=$(curl -s $BACKEND_URL/health)
echo "$health" | jq .

# Step 2: Inject test attack events
echo -e "\n2. Injecting Test Attack Events..."
attack_events='[
  {
    "src_ip": "203.0.113.45",
    "dst_ip": "10.0.0.23",
    "dst_port": 2222,
    "eventid": "cowrie.login.failed",
    "message": "login attempt [admin/password123] failed",
    "timestamp": "'$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)'"
  },
  {
    "src_ip": "203.0.113.45",
    "dst_ip": "10.0.0.23", 
    "dst_port": 2222,
    "eventid": "cowrie.login.failed",
    "message": "login attempt [root/admin] failed",
    "timestamp": "'$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)'"
  },
  {
    "src_ip": "203.0.113.45",
    "dst_ip": "10.0.0.23",
    "dst_port": 2222,
    "eventid": "cowrie.login.failed", 
    "message": "login attempt [test/test] failed",
    "timestamp": "'$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)'"
  },
  {
    "src_ip": "203.0.113.45",
    "dst_ip": "10.0.0.23",
    "dst_port": 2222,
    "eventid": "cowrie.login.failed",
    "message": "login attempt [guest/guest] failed", 
    "timestamp": "'$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)'"
  },
  {
    "src_ip": "203.0.113.45",
    "dst_ip": "10.0.0.23",
    "dst_port": 2222,
    "eventid": "cowrie.login.failed",
    "message": "login attempt [oracle/oracle] failed",
    "timestamp": "'$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)'"
  },
  {
    "src_ip": "203.0.113.45", 
    "dst_ip": "10.0.0.23",
    "dst_port": 2222,
    "eventid": "cowrie.login.failed",
    "message": "login attempt [postgres/postgres] failed",
    "timestamp": "'$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)'"
  },
  {
    "src_ip": "203.0.113.45",
    "dst_ip": "10.0.0.23", 
    "dst_port": 2222,
    "eventid": "cowrie.login.failed",
    "message": "login attempt [ubuntu/password] failed",
    "timestamp": "'$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)'"
  }
]'

# Inject the events
ingest_response=$(curl -s -X POST $BACKEND_URL/ingest/cowrie \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $API_KEY" \
  -d "$attack_events")

echo "Ingestion Response:"
echo "$ingest_response" | jq .

# Get the incident ID from response
incident_id=$(echo "$ingest_response" | jq -r '.incident_id')
echo "Detected Incident ID: $incident_id"

# Step 3: Wait for processing
echo -e "\n3. Waiting for incident processing..."
sleep 3

# Step 4: Check incident details
echo -e "\n4. Checking Incident Details..."
if [[ "$incident_id" != "null" ]] && [[ -n "$incident_id" ]]; then
    incident_details=$(curl -s $BACKEND_URL/incidents/$incident_id)
    echo "$incident_details" | jq .
else
    echo "No specific incident ID, checking recent incidents..."
    recent_incidents=$(curl -s $BACKEND_URL/incidents | head -c 500)
    echo "$recent_incidents" | jq .
fi

# Step 5: Test AI Agent Analysis
echo -e "\n5. Testing AI Agent Analysis..."
agent_response=$(curl -s -X POST $BACKEND_URL/api/agents/orchestrate \
  -H "Content-Type: application/json" \
  -H "x-api-key: $API_KEY" \
  -d '{"query": "analyze IP 203.0.113.45 and recommend containment actions"}')

echo "AI Agent Response:"
echo "$agent_response" | jq .

# Step 6: Test ML Model Status
echo -e "\n6. Checking ML Model Performance..."
ml_status=$(curl -s $BACKEND_URL/api/ml/status)
echo "$ml_status" | jq .

# Step 7: Test Log Sources
echo -e "\n7. Checking Log Source Status..."
sources=$(curl -s $BACKEND_URL/api/sources)
echo "$sources" | jq .

# Step 8: Final System Status
echo -e "\n8. Final System Status..."
final_status=$(curl -s $BACKEND_URL/health)
echo "$final_status" | jq .

echo -e "\n🎉 End-to-End Test Complete!"
echo "================================"

# Summary
echo "Summary:"
echo "- Test events injected: 7 failed login attempts from 203.0.113.45"
echo "- Incident detection: $(echo "$ingest_response" | jq -r '.detected') incidents detected"
echo "- Events stored: $(echo "$ingest_response" | jq -r '.stored') events"
echo "- AI Agent confidence: $(echo "$agent_response" | jq -r '.confidence // "N/A"')"
echo "- ML models active: $(echo "$ml_status" | jq -r '.metrics.models_trained // "N/A"')/$(echo "$ml_status" | jq -r '.metrics.total_models // "N/A"')"

echo -e "\nNext steps:"
echo "1. Open http://localhost:3000 to view the incidents in the UI"
echo "2. Test AI agent chat at http://localhost:3000/agents"
echo "3. Enable auto-containment if desired"
echo "4. Configure your real honeypot IP in backend/.env"
