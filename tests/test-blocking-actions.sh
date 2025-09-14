#!/bin/bash
# Test script to validate all blocking actions work correctly

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

BACKEND_URL="http://localhost:8000"

echo "=== 🧪 Testing Mini-XDR Blocking Actions ==="
echo ""

# Test 1: Get incidents
echo -e "${BLUE}🔍 Test 1: Fetching incidents...${NC}"
incidents_response=$(curl -s "$BACKEND_URL/incidents")
if [ $? -eq 0 ]; then
    incident_count=$(echo "$incidents_response" | jq length 2>/dev/null || echo "unknown")
    echo -e "${GREEN}✅ Success: Found $incident_count incidents${NC}"
    
    # Get first incident ID
    first_incident_id=$(echo "$incidents_response" | jq -r '.[0].id' 2>/dev/null)
    if [ "$first_incident_id" != "null" ] && [ "$first_incident_id" != "" ]; then
        echo "   Using incident ID: $first_incident_id for tests"
    else
        echo -e "${RED}❌ No incidents found to test with${NC}"
        exit 1
    fi
else
    echo -e "${RED}❌ Failed to fetch incidents${NC}"
    exit 1
fi

echo ""

# Test 2: Test permanent blocking
echo -e "${BLUE}🔍 Test 2: Testing permanent blocking...${NC}"
block_response=$(curl -s -X POST "$BACKEND_URL/incidents/$first_incident_id/contain")
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Success: Permanent block request completed${NC}"
    echo "   Response: $block_response"
else
    echo -e "${RED}❌ Failed permanent block request${NC}"
fi

echo ""

# Test 3: Test temporary blocking (30 seconds)
echo -e "${BLUE}🔍 Test 3: Testing temporary blocking (30 seconds)...${NC}"
temp_block_response=$(curl -s -X POST "$BACKEND_URL/incidents/$first_incident_id/contain?duration_seconds=30")
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Success: Temporary block request completed${NC}"
    echo "   Response: $temp_block_response"
else
    echo -e "${RED}❌ Failed temporary block request${NC}"
fi

echo ""

# Test 4: Test unblocking
echo -e "${BLUE}🔍 Test 4: Testing unblocking...${NC}"
unblock_response=$(curl -s -X POST "$BACKEND_URL/incidents/$first_incident_id/unblock")
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Success: Unblock request completed${NC}"
    echo "   Response: $unblock_response"
else
    echo -e "${RED}❌ Failed unblock request${NC}"
fi

echo ""

# Test 5: Test scheduled unblocking
echo -e "${BLUE}🔍 Test 5: Testing scheduled unblock (5 minutes)...${NC}"
schedule_response=$(curl -s -X POST "$BACKEND_URL/incidents/$first_incident_id/schedule_unblock?minutes=5")
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Success: Scheduled unblock request completed${NC}"
    echo "   Response: $schedule_response"
else
    echo -e "${RED}❌ Failed scheduled unblock request${NC}"
fi

echo ""

# Test 6: Check incident details after actions
echo -e "${BLUE}🔍 Test 6: Checking incident details after actions...${NC}"
detail_response=$(curl -s "$BACKEND_URL/incidents/$first_incident_id")
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Success: Retrieved incident details${NC}"
    action_count=$(echo "$detail_response" | jq '.actions | length' 2>/dev/null || echo "unknown")
    echo "   Actions recorded: $action_count"
    
    # Show recent actions
    echo "   Recent actions:"
    echo "$detail_response" | jq -r '.actions[] | "     - \(.action): \(.result) (\(.created_at))"' 2>/dev/null || echo "     Could not parse actions"
else
    echo -e "${RED}❌ Failed to get incident details${NC}"
fi

echo ""

# Test 7: Test auto-contain settings
echo -e "${BLUE}🔍 Test 7: Testing auto-contain settings...${NC}"
auto_contain_get=$(curl -s "$BACKEND_URL/settings/auto_contain")
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Success: Retrieved auto-contain setting${NC}"
    echo "   Current setting: $auto_contain_get"
    
    # Toggle it
    current_setting=$(echo "$auto_contain_get" | jq -r '.enabled' 2>/dev/null)
    new_setting=$([[ "$current_setting" == "true" ]] && echo "false" || echo "true")
    
    auto_contain_set=$(curl -s -X POST -H "Content-Type: application/json" -d "$new_setting" "$BACKEND_URL/settings/auto_contain")
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Success: Updated auto-contain setting${NC}"
        echo "   New setting: $auto_contain_set"
    else
        echo -e "${RED}❌ Failed to update auto-contain setting${NC}"
    fi
else
    echo -e "${RED}❌ Failed to get auto-contain setting${NC}"
fi

echo ""
echo "=== 🎉 All blocking action tests completed! ==="
echo ""
echo "💡 Check the frontend dashboard to see the action results"
echo "🌐 Dashboard: http://localhost:3000"
echo "📋 Incident Details: http://localhost:3000/incidents/$first_incident_id"
echo ""
