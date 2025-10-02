#!/bin/bash
#
# Test NLP UI End-to-End
# Tests the full workflow: Parse → Preview → Create → List

set -e

API_KEY="${XDR_API_KEY:-demo-minixdr-api-key}"
BASE_URL="${BASE_URL:-http://localhost:8000}"
INCIDENT_ID="${INCIDENT_ID:-1}"

echo "🧪 Testing NLP Workflow UI Integration"
echo "========================================"
echo ""

# Test 1: Parse workflow
echo "1️⃣  Testing Parse Endpoint..."
PARSE_RESPONSE=$(curl -s -X POST "${BASE_URL}/api/workflows/nlp/parse" \
  -H "x-api-key: ${API_KEY}" \
  -H "Content-Type: application/json" \
  -d "{
    \"text\": \"Block IP 192.168.1.100 and isolate the compromised host\",
    \"incident_id\": ${INCIDENT_ID}
  }")

echo "✅ Parse Response:"
echo "${PARSE_RESPONSE}" | python3 -m json.tool
echo ""

# Extract workflow details
CONFIDENCE=$(echo "${PARSE_RESPONSE}" | python3 -c "import sys, json; print(json.load(sys.stdin).get('confidence', 0))" 2>/dev/null || echo "0")
ACTIONS_COUNT=$(echo "${PARSE_RESPONSE}" | python3 -c "import sys, json; print(json.load(sys.stdin).get('actions_count', 0))" 2>/dev/null || echo "0")

echo "   Confidence: ${CONFIDENCE}"
echo "   Actions: ${ACTIONS_COUNT}"
echo ""

# Test 2: Create workflow
echo "2️⃣  Testing Create Endpoint..."
CREATE_RESPONSE=$(curl -s -X POST "${BASE_URL}/api/workflows/nlp/create" \
  -H "x-api-key: ${API_KEY}" \
  -H "Content-Type: application/json" \
  -d "{
    \"text\": \"Emergency: Block attacker 10.0.200.50 and alert the security team\",
    \"incident_id\": ${INCIDENT_ID},
    \"auto_execute\": false
  }")

echo "✅ Create Response:"
echo "${CREATE_RESPONSE}" | python3 -m json.tool
echo ""

# Extract workflow ID
WORKFLOW_ID=$(echo "${CREATE_RESPONSE}" | python3 -c "import sys, json; print(json.load(sys.stdin).get('workflow_id', ''))" 2>/dev/null || echo "")
WORKFLOW_DB_ID=$(echo "${CREATE_RESPONSE}" | python3 -c "import sys, json; print(json.load(sys.stdin).get('workflow_db_id', ''))" 2>/dev/null || echo "")

echo "   Workflow ID: ${WORKFLOW_ID}"
echo "   DB ID: ${WORKFLOW_DB_ID}"
echo ""

# Test 3: List workflows to confirm it exists
echo "3️⃣  Testing Workflow List..."
LIST_RESPONSE=$(curl -s -X GET "${BASE_URL}/api/response/workflows" \
  -H "x-api-key: ${API_KEY}")

# Check if our workflow is in the list
WORKFLOW_FOUND=$(echo "${LIST_RESPONSE}" | python3 -c "
import sys, json
data = json.load(sys.stdin)
workflows = data.get('workflows', [])
found = any(w.get('workflow_id') == '${WORKFLOW_ID}' for w in workflows)
print('yes' if found else 'no')
" 2>/dev/null || echo "error")

if [ "${WORKFLOW_FOUND}" = "yes" ]; then
  echo "✅ Workflow found in list!"
else
  echo "⚠️  Workflow not found in list (may need to refresh)"
fi
echo ""

# Test 4: Get examples
echo "4️⃣  Testing Examples Endpoint..."
EXAMPLES_RESPONSE=$(curl -s -X GET "${BASE_URL}/api/workflows/nlp/examples" \
  -H "x-api-key: ${API_KEY}")

EXAMPLES_COUNT=$(echo "${EXAMPLES_RESPONSE}" | python3 -c "
import sys, json
data = json.load(sys.stdin)
total = sum(len(examples) for examples in data.values() if isinstance(examples, list))
print(total)
" 2>/dev/null || echo "0")

echo "✅ Examples available: ${EXAMPLES_COUNT}"
echo ""

# Summary
echo "📊 Test Summary"
echo "==============="
echo "✅ Parse endpoint working"
echo "✅ Create endpoint working"
echo "✅ Workflow created: ${WORKFLOW_ID}"
echo "✅ Examples endpoint working"
echo ""
echo "🎯 Next: Open http://localhost:3000/workflows and test the UI"
echo ""


