#!/bin/bash

# Demo Script: Chat → Workflow & Investigation Integration
# Shows the new AI-powered workflow creation and investigation capabilities

set -e

BASE_URL="http://localhost:8000"
API_KEY="demo-minixdr-api-key"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${CYAN}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║   Mini-XDR: AI Chat Integration Demo                           ║${NC}"
echo -e "${CYAN}║   Workflow Creation & Investigation Triggers                   ║${NC}"
echo -e "${CYAN}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check if backend is running
echo -e "${BLUE}🔍 Checking backend health...${NC}"
if curl -s -f "$BASE_URL/health" > /dev/null; then
    echo -e "${GREEN}✅ Backend is healthy!${NC}"
else
    echo -e "${YELLOW}⚠️  Backend is not running. Please start it first.${NC}"
    exit 1
fi

echo ""
echo -e "${CYAN}════════════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}DEMO 1: Workflow Creation from Natural Language${NC}"
echo -e "${CYAN}════════════════════════════════════════════════════════════════${NC}"
echo ""

echo -e "${BLUE}Query:${NC} \"Block IP 192.0.2.100 and isolate the affected host\""
echo ""

RESPONSE=$(curl -s -X POST "$BASE_URL/api/agents/orchestrate" \
  -H "Content-Type: application/json" \
  -H "x-api-key: $API_KEY" \
  -d '{
    "query": "Block IP 192.0.2.100 and isolate the affected host",
    "incident_id": 8,
    "context": {"demo": true}
  }')

WORKFLOW_CREATED=$(echo "$RESPONSE" | jq -r '.workflow_created')
WORKFLOW_ID=$(echo "$RESPONSE" | jq -r '.workflow_id')
WORKFLOW_DB_ID=$(echo "$RESPONSE" | jq -r '.workflow_db_id')

if [ "$WORKFLOW_CREATED" = "true" ]; then
    echo -e "${GREEN}✅ Workflow Created Successfully!${NC}"
    echo -e "   ${GREEN}Workflow ID:${NC} $WORKFLOW_ID"
    echo -e "   ${GREEN}Database ID:${NC} $WORKFLOW_DB_ID"
    echo ""
    echo -e "${BLUE}Response Message:${NC}"
    echo "$RESPONSE" | jq -r '.message' | head -15
else
    echo -e "${YELLOW}⚠️  Workflow not created${NC}"
fi

echo ""
echo -e "${CYAN}════════════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}DEMO 2: Investigation Trigger${NC}"
echo -e "${CYAN}════════════════════════════════════════════════════════════════${NC}"
echo ""

echo -e "${BLUE}Query:${NC} \"Investigate this SSH brute force attack and analyze patterns\""
echo ""

RESPONSE=$(curl -s -X POST "$BASE_URL/api/agents/orchestrate" \
  -H "Content-Type: application/json" \
  -H "x-api-key: $API_KEY" \
  -d '{
    "query": "Investigate this SSH brute force attack and analyze patterns",
    "incident_id": 8,
    "context": {"demo": true}
  }')

INVESTIGATION_STARTED=$(echo "$RESPONSE" | jq -r '.investigation_started')
CASE_ID=$(echo "$RESPONSE" | jq -r '.case_id')
EVIDENCE_COUNT=$(echo "$RESPONSE" | jq -r '.evidence_count')

if [ "$INVESTIGATION_STARTED" = "true" ]; then
    echo -e "${GREEN}✅ Investigation Started!${NC}"
    echo -e "   ${GREEN}Case ID:${NC} $CASE_ID"
    echo -e "   ${GREEN}Evidence Count:${NC} $EVIDENCE_COUNT"
    echo ""
    echo -e "${BLUE}Response Message:${NC}"
    echo "$RESPONSE" | jq -r '.message' | head -20
else
    echo -e "${YELLOW}⚠️  Investigation not triggered${NC}"
fi

echo ""
echo -e "${CYAN}════════════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}DEMO 3: Complex Multi-Action Workflow${NC}"
echo -e "${CYAN}════════════════════════════════════════════════════════════════${NC}"
echo ""

echo -e "${BLUE}Query:${NC} \"Block the attacker, reset passwords, and deploy firewall rules\""
echo ""

RESPONSE=$(curl -s -X POST "$BASE_URL/api/agents/orchestrate" \
  -H "Content-Type: application/json" \
  -H "x-api-key: $API_KEY" \
  -d '{
    "query": "Block the attacker, reset passwords, and deploy firewall rules",
    "incident_id": 8,
    "context": {"demo": true}
  }')

WORKFLOW_CREATED=$(echo "$RESPONSE" | jq -r '.workflow_created')
WORKFLOW_ID=$(echo "$RESPONSE" | jq -r '.workflow_id')

if [ "$WORKFLOW_CREATED" = "true" ]; then
    echo -e "${GREEN}✅ Multi-Action Workflow Created!${NC}"
    echo -e "   ${GREEN}Workflow ID:${NC} $WORKFLOW_ID"
    echo ""
    echo -e "${BLUE}Workflow Details:${NC}"
    echo "$RESPONSE" | jq -r '.message' | grep -A 10 "Workflow Steps:"
else
    echo -e "${YELLOW}⚠️  Workflow not created${NC}"
fi

echo ""
echo -e "${CYAN}════════════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}DEMO 4: Different Attack Types${NC}"
echo -e "${CYAN}════════════════════════════════════════════════════════════════${NC}"
echo ""

# DDoS Attack
echo -e "${BLUE}Scenario: DDoS Attack${NC}"
echo -e "Query: \"Deploy firewall rules to mitigate this DDoS\""

RESPONSE=$(curl -s -X POST "$BASE_URL/api/agents/orchestrate" \
  -H "Content-Type: application/json" \
  -H "x-api-key: $API_KEY" \
  -d '{
    "query": "Deploy firewall rules to mitigate this DDoS",
    "incident_id": 7,
    "context": {"attack_type": "ddos", "demo": true}
  }')

WORKFLOW_CREATED=$(echo "$RESPONSE" | jq -r '.workflow_created')
if [ "$WORKFLOW_CREATED" = "true" ]; then
    WORKFLOW_ID=$(echo "$RESPONSE" | jq -r '.workflow_id')
    echo -e "${GREEN}✅ DDoS Mitigation Workflow: $WORKFLOW_ID${NC}"
else
    echo -e "${YELLOW}⚠️  No workflow created${NC}"
fi

# Malware Detection
echo ""
echo -e "${BLUE}Scenario: Malware Detection${NC}"
echo -e "Query: \"Isolate infected systems and run forensics\""

RESPONSE=$(curl -s -X POST "$BASE_URL/api/agents/orchestrate" \
  -H "Content-Type: application/json" \
  -H "x-api-key: $API_KEY" \
  -d '{
    "query": "Isolate infected systems and run forensics",
    "incident_id": 6,
    "context": {"attack_type": "malware", "demo": true}
  }')

INVESTIGATION_STARTED=$(echo "$RESPONSE" | jq -r '.investigation_started')
if [ "$INVESTIGATION_STARTED" = "true" ]; then
    CASE_ID=$(echo "$RESPONSE" | jq -r '.case_id')
    echo -e "${GREEN}✅ Forensic Investigation: $CASE_ID${NC}"
else
    echo -e "${YELLOW}⚠️  Investigation not triggered${NC}"
fi

echo ""
echo -e "${CYAN}════════════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}DEMO COMPLETE!${NC}"
echo -e "${CYAN}════════════════════════════════════════════════════════════════${NC}"
echo ""

echo -e "${GREEN}✅ Key Features Demonstrated:${NC}"
echo -e "   1. Natural language workflow creation"
echo -e "   2. Automatic investigation triggers"
echo -e "   3. Multi-action workflows"
echo -e "   4. Attack-specific responses"
echo ""

echo -e "${BLUE}💡 Try it in the UI:${NC}"
echo -e "   1. Open: http://localhost:3000/incidents/incident/8"
echo -e "   2. Use the AI chat on the right"
echo -e "   3. Type commands like:"
echo -e "      - \"Block IP 192.0.2.100\""
echo -e "      - \"Investigate this attack\""
echo -e "      - \"Isolate the host and reset passwords\""
echo ""

echo -e "${BLUE}📊 View Results:${NC}"
echo -e "   - Workflows: http://localhost:3000/workflows"
echo -e "   - Incident Actions: Check incident detail page"
echo -e "   - Database: backend/xdr.db (response_workflows & actions tables)"
echo ""


