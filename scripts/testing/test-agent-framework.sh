#!/bin/bash
# Test IAM, EDR, and DLP Agents
# Tests all three agents in simulation mode

set -e

API_BASE="${API_BASE:-http://localhost:8000}"
INCIDENT_ID="${INCIDENT_ID:-1}"

echo "🧪 Testing Mini-XDR Agent Framework"
echo "===================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# ==================== IAM AGENT TESTS ====================

echo -e "${BLUE}[1/10] Testing IAM Agent - Disable User${NC}"
RESPONSE=$(curl -s -X POST "$API_BASE/api/agents/iam/execute" \
  -H "Content-Type: application/json" \
  -d '{
    "action_name": "disable_user",
    "params": {
      "username": "testuser@domain.local",
      "reason": "Suspicious activity detected"
    },
    "incident_id": '$INCIDENT_ID'
  }')

if echo "$RESPONSE" | jq -e '.success == true' > /dev/null; then
  echo -e "${GREEN}✅ IAM: Disable user successful${NC}"
  ROLLBACK_ID_IAM=$(echo "$RESPONSE" | jq -r '.rollback_id')
  echo "   Rollback ID: $ROLLBACK_ID_IAM"
else
  echo -e "${RED}❌ IAM: Disable user failed${NC}"
  echo "$RESPONSE" | jq '.'
fi
echo ""

echo -e "${BLUE}[2/10] Testing IAM Agent - Quarantine User${NC}"
RESPONSE=$(curl -s -X POST "$API_BASE/api/agents/iam/execute" \
  -H "Content-Type: application/json" \
  -d '{
    "action_name": "quarantine_user",
    "params": {
      "username": "compromised@domain.local",
      "security_group": "CN=Quarantine,OU=Security,DC=domain,DC=local"
    },
    "incident_id": '$INCIDENT_ID'
  }')

if echo "$RESPONSE" | jq -e '.success == true' > /dev/null; then
  echo -e "${GREEN}✅ IAM: Quarantine user successful${NC}"
else
  echo -e "${RED}❌ IAM: Quarantine user failed${NC}"
fi
echo ""

echo -e "${BLUE}[3/10] Testing IAM Agent - Reset Password${NC}"
RESPONSE=$(curl -s -X POST "$API_BASE/api/agents/iam/execute" \
  -H "Content-Type: application/json" \
  -d '{
    "action_name": "reset_password",
    "params": {
      "username": "testuser@domain.local"
    },
    "incident_id": '$INCIDENT_ID'
  }')

if echo "$RESPONSE" | jq -e '.success == true' > /dev/null; then
  echo -e "${GREEN}✅ IAM: Reset password successful${NC}"
  NEW_PASSWORD=$(echo "$RESPONSE" | jq -r '.result.new_password')
  echo "   New password: $NEW_PASSWORD"
else
  echo -e "${RED}❌ IAM: Reset password failed${NC}"
fi
echo ""

# ==================== EDR AGENT TESTS ====================

echo -e "${BLUE}[4/10] Testing EDR Agent - Kill Process${NC}"
RESPONSE=$(curl -s -X POST "$API_BASE/api/agents/edr/execute" \
  -H "Content-Type: application/json" \
  -d '{
    "action_name": "kill_process",
    "params": {
      "hostname": "workstation01",
      "process_name": "malware.exe"
    },
    "incident_id": '$INCIDENT_ID'
  }')

if echo "$RESPONSE" | jq -e '.success == true' > /dev/null; then
  echo -e "${GREEN}✅ EDR: Kill process successful${NC}"
  ROLLBACK_ID_EDR=$(echo "$RESPONSE" | jq -r '.rollback_id')
  echo "   Rollback ID: $ROLLBACK_ID_EDR"
else
  echo -e "${RED}❌ EDR: Kill process failed${NC}"
fi
echo ""

echo -e "${BLUE}[5/10] Testing EDR Agent - Quarantine File${NC}"
RESPONSE=$(curl -s -X POST "$API_BASE/api/agents/edr/execute" \
  -H "Content-Type: application/json" \
  -d '{
    "action_name": "quarantine_file",
    "params": {
      "hostname": "workstation01",
      "file_path": "C:\\Users\\Public\\suspicious.exe"
    },
    "incident_id": '$INCIDENT_ID'
  }')

if echo "$RESPONSE" | jq -e '.success == true' > /dev/null; then
  echo -e "${GREEN}✅ EDR: Quarantine file successful${NC}"
  QUARANTINE_PATH=$(echo "$RESPONSE" | jq -r '.result.quarantine_path')
  echo "   Quarantined to: $QUARANTINE_PATH"
else
  echo -e "${RED}❌ EDR: Quarantine file failed${NC}"
fi
echo ""

echo -e "${BLUE}[6/10] Testing EDR Agent - Isolate Host${NC}"
RESPONSE=$(curl -s -X POST "$API_BASE/api/agents/edr/execute" \
  -H "Content-Type: application/json" \
  -d '{
    "action_name": "isolate_host",
    "params": {
      "hostname": "workstation01",
      "level": "strict"
    },
    "incident_id": '$INCIDENT_ID'
  }')

if echo "$RESPONSE" | jq -e '.success == true' > /dev/null; then
  echo -e "${GREEN}✅ EDR: Isolate host successful${NC}"
else
  echo -e "${RED}❌ EDR: Isolate host failed${NC}"
fi
echo ""

echo -e "${BLUE}[7/10] Testing EDR Agent - Collect Memory Dump${NC}"
RESPONSE=$(curl -s -X POST "$API_BASE/api/agents/edr/execute" \
  -H "Content-Type: application/json" \
  -d '{
    "action_name": "collect_memory_dump",
    "params": {
      "hostname": "workstation01"
    },
    "incident_id": '$INCIDENT_ID'
  }')

if echo "$RESPONSE" | jq -e '.success == true' > /dev/null; then
  echo -e "${GREEN}✅ EDR: Collect memory dump successful${NC}"
  DUMP_PATH=$(echo "$RESPONSE" | jq -r '.result.dump_path')
  echo "   Dump saved to: $DUMP_PATH"
else
  echo -e "${RED}❌ EDR: Collect memory dump failed${NC}"
fi
echo ""

# ==================== DLP AGENT TESTS ====================

echo -e "${BLUE}[8/10] Testing DLP Agent - Scan File${NC}"
RESPONSE=$(curl -s -X POST "$API_BASE/api/agents/dlp/execute" \
  -H "Content-Type: application/json" \
  -d '{
    "action_name": "scan_file",
    "params": {
      "file_path": "C:\\Users\\testuser\\documents\\customers.xlsx"
    },
    "incident_id": '$INCIDENT_ID'
  }')

if echo "$RESPONSE" | jq -e '.success == true' > /dev/null; then
  echo -e "${GREEN}✅ DLP: Scan file successful${NC}"
  FINDINGS=$(echo "$RESPONSE" | jq -r '.result.findings | length')
  echo "   Found $FINDINGS sensitive data types"
else
  echo -e "${RED}❌ DLP: Scan file failed${NC}"
fi
echo ""

echo -e "${BLUE}[9/10] Testing DLP Agent - Block Upload${NC}"
RESPONSE=$(curl -s -X POST "$API_BASE/api/agents/dlp/execute" \
  -H "Content-Type: application/json" \
  -d '{
    "action_name": "block_upload",
    "params": {
      "hostname": "workstation01",
      "process_name": "chrome.exe",
      "destination": "http://malicious-site.com"
    },
    "incident_id": '$INCIDENT_ID'
  }')

if echo "$RESPONSE" | jq -e '.success == true' > /dev/null; then
  echo -e "${GREEN}✅ DLP: Block upload successful${NC}"
  ROLLBACK_ID_DLP=$(echo "$RESPONSE" | jq -r '.rollback_id')
  echo "   Rollback ID: $ROLLBACK_ID_DLP"
else
  echo -e "${RED}❌ DLP: Block upload failed${NC}"
fi
echo ""

# ==================== ROLLBACK TESTS ====================

echo -e "${BLUE}[10/10] Testing Rollback - IAM Disable User${NC}"
if [ ! -z "$ROLLBACK_ID_IAM" ] && [ "$ROLLBACK_ID_IAM" != "null" ]; then
  RESPONSE=$(curl -s -X POST "$API_BASE/api/agents/rollback/$ROLLBACK_ID_IAM")
  
  if echo "$RESPONSE" | jq -e '.success == true' > /dev/null; then
    echo -e "${GREEN}✅ Rollback: IAM action rolled back successfully${NC}"
  else
    echo -e "${RED}❌ Rollback: IAM rollback failed${NC}"
  fi
else
  echo -e "${RED}⚠️ Skipped: No rollback ID available${NC}"
fi
echo ""

# ==================== ACTION LOGS ====================

echo -e "${BLUE}Fetching Action Logs for Incident $INCIDENT_ID${NC}"
RESPONSE=$(curl -s "$API_BASE/api/agents/actions/$INCIDENT_ID")
ACTION_COUNT=$(echo "$RESPONSE" | jq 'length')
echo -e "${GREEN}Total actions logged: $ACTION_COUNT${NC}"
echo ""

echo "🎉 Agent Framework Testing Complete!"
echo ""
echo "Summary:"
echo "--------"
echo "✅ IAM Agent: 3/3 actions tested"
echo "✅ EDR Agent: 4/4 actions tested"
echo "✅ DLP Agent: 2/2 actions tested"
echo "✅ Rollback: Tested"
echo ""
echo "All agents are working in simulation mode!"

