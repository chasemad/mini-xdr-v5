#!/bin/bash

# Test script for Unified Agent Actions UI
# This script tests that agent actions appear in the unified interface

echo "🧪 Testing Unified Agent Actions UI Integration"
echo "=============================================="
echo ""

API_BASE="http://localhost:8000"

# Check if backend is running
echo "1️⃣ Checking backend health..."
if ! curl -s "${API_BASE}/health" > /dev/null; then
    echo "❌ Backend is not running. Start it with:"
    echo "   cd backend && source venv/bin/activate && uvicorn app.main:app --reload"
    exit 1
fi
echo "✅ Backend is running"
echo ""

# Get first incident ID
echo "2️⃣ Finding an incident to test with..."
INCIDENT_ID=$(curl -s "${API_BASE}/api/incidents" | jq -r '.[0].id // empty')

if [ -z "$INCIDENT_ID" ]; then
    echo "⚠️  No incidents found. Creating a test incident..."
    # You would create a test incident here
    echo "   Please create an incident manually first"
    exit 1
fi

echo "✅ Found incident ID: ${INCIDENT_ID}"
echo ""

# Execute a test IAM action
echo "3️⃣ Executing test IAM agent action..."
IAM_RESPONSE=$(curl -s -X POST "${API_BASE}/api/agents/iam/execute" \
  -H "Content-Type: application/json" \
  -d "{
    \"action_name\": \"disable_user_account\",
    \"params\": {
      \"username\": \"test.user@corp.local\",
      \"reason\": \"Testing unified UI - Suspicious activity detected\"
    },
    \"incident_id\": ${INCIDENT_ID}
  }")

echo "IAM Action Response:"
echo "$IAM_RESPONSE" | jq '.'
echo ""

# Execute a test EDR action
echo "4️⃣ Executing test EDR agent action..."
EDR_RESPONSE=$(curl -s -X POST "${API_BASE}/api/agents/edr/execute" \
  -H "Content-Type: application/json" \
  -d "{
    \"action_name\": \"kill_process\",
    \"params\": {
      \"process_name\": \"malicious.exe\",
      \"endpoint\": \"WORKSTATION-001\",
      \"reason\": \"Testing unified UI - Malware detected\"
    },
    \"incident_id\": ${INCIDENT_ID}
  }")

echo "EDR Action Response:"
echo "$EDR_RESPONSE" | jq '.'
echo ""

# Execute a test DLP action
echo "5️⃣ Executing test DLP agent action..."
DLP_RESPONSE=$(curl -s -X POST "${API_BASE}/api/agents/dlp/execute" \
  -H "Content-Type: application/json" \
  -d "{
    \"action_name\": \"scan_file\",
    \"params\": {
      \"file_path\": \"/home/user/sensitive_data.xlsx\",
      \"reason\": \"Testing unified UI - Data leak suspected\"
    },
    \"incident_id\": ${INCIDENT_ID}
  }")

echo "DLP Action Response:"
echo "$DLP_RESPONSE" | jq '.'
echo ""

# Fetch all actions for the incident
echo "6️⃣ Fetching all agent actions for incident ${INCIDENT_ID}..."
ACTIONS=$(curl -s "${API_BASE}/api/agents/actions/${INCIDENT_ID}")
ACTION_COUNT=$(echo "$ACTIONS" | jq 'length')

echo "✅ Found ${ACTION_COUNT} agent actions"
echo "$ACTIONS" | jq '.[] | {action: .action_name, agent: .agent_type, status: .status, rollback_id: .rollback_id}'
echo ""

echo "=============================================="
echo "✅ Test Complete!"
echo ""
echo "📋 Next Steps:"
echo "   1. Open http://localhost:3000 in your browser"
echo "   2. Navigate to incident #${INCIDENT_ID}"
echo "   3. Scroll down to 'Unified Response Actions' section"
echo "   4. You should see:"
echo "      - 👤 IAM: Disable User Account"
echo "      - 🖥️  EDR: Kill Process"
echo "      - 🔒 DLP: Scan File"
echo "   5. Click on any action to see details in modal"
echo "   6. Try rollback buttons if available"
echo ""
echo "🎯 SUCCESS CRITERIA:"
echo "   ✓ All 3 action types visible (manual, workflow, agent)"
echo "   ✓ Agent-specific color coding (IAM=Blue, EDR=Purple, DLP=Green)"
echo "   ✓ Rollback buttons appear for agent actions"
echo "   ✓ Click opens detailed modal"
echo "   ✓ Actions auto-refresh every 5 seconds"
echo ""


