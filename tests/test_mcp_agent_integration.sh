#!/bin/bash

# ===============================================
# MCP Agent Integration Test Script
# Tests all new agent tools in MCP server
# ===============================================

set -e  # Exit on error

API_BASE="${API_BASE:-http://localhost:8000}"
INCIDENT_ID=1

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Track test results
TESTS_PASSED=0
TESTS_FAILED=0

echo "🚀 MCP Agent Integration Test Suite"
echo "===================================="
echo "API Base: $API_BASE"
echo "Test Incident ID: $INCIDENT_ID"
echo ""

# Function to run test and check result
run_test() {
    local test_name="$1"
    local endpoint="$2"
    local method="$3"
    local data="$4"
    
    echo -e "${BLUE}▶ Test: $test_name${NC}"
    
    if [ "$method" = "GET" ]; then
        response=$(curl -s -w "\n%{http_code}" "$API_BASE$endpoint")
    else
        response=$(curl -s -w "\n%{http_code}" -X "$method" "$API_BASE$endpoint" \
            -H "Content-Type: application/json" \
            -d "$data")
    fi
    
    http_code=$(echo "$response" | tail -n1)
    body=$(echo "$response" | sed '$d')
    
    if [ "$http_code" -eq 200 ] || [ "$http_code" -eq 201 ]; then
        echo -e "${GREEN}  ✅ PASSED${NC} (HTTP $http_code)"
        echo "  Response: $(echo "$body" | jq -r '.status // .message // "Success"' 2>/dev/null || echo "$body" | head -c 100)"
        ((TESTS_PASSED++))
    else
        echo -e "${RED}  ❌ FAILED${NC} (HTTP $http_code)"
        echo "  Error: $(echo "$body" | jq -r '.detail // .error // .' 2>/dev/null || echo "$body")"
        ((TESTS_FAILED++))
    fi
    echo ""
}

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  1. IAM AGENT TESTS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Test 1.1: Disable User Account
run_test "IAM - Disable User Account" \
    "/api/agents/iam/execute" \
    "POST" \
    '{
        "action_name": "disable_user_account",
        "params": {
            "username": "test.user@domain.local",
            "reason": "MCP Integration Test - Suspected compromise"
        },
        "incident_id": '$INCIDENT_ID'
    }'

# Test 1.2: Reset User Password
run_test "IAM - Reset User Password" \
    "/api/agents/iam/execute" \
    "POST" \
    '{
        "action_name": "reset_user_password",
        "params": {
            "username": "test.user2@domain.local",
            "reason": "MCP Integration Test - Password reset required",
            "force_change": true
        },
        "incident_id": '$INCIDENT_ID'
    }'

# Test 1.3: Remove User From Group
run_test "IAM - Remove User From Group" \
    "/api/agents/iam/execute" \
    "POST" \
    '{
        "action_name": "remove_user_from_group",
        "params": {
            "username": "test.user3@domain.local",
            "group_name": "Domain Admins",
            "reason": "MCP Integration Test - Unauthorized privileges"
        },
        "incident_id": '$INCIDENT_ID'
    }'

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  2. EDR AGENT TESTS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Test 2.1: Kill Process
run_test "EDR - Kill Process" \
    "/api/agents/edr/execute" \
    "POST" \
    '{
        "action_name": "kill_process",
        "params": {
            "hostname": "WORKSTATION-01",
            "process_name": "malware.exe",
            "pid": 4567,
            "reason": "MCP Integration Test - Malicious process detected"
        },
        "incident_id": '$INCIDENT_ID'
    }'

# Test 2.2: Quarantine File
run_test "EDR - Quarantine File" \
    "/api/agents/edr/execute" \
    "POST" \
    '{
        "action_name": "quarantine_file",
        "params": {
            "hostname": "WORKSTATION-02",
            "file_path": "C:\\\\Users\\\\Public\\\\suspicious.dll",
            "reason": "MCP Integration Test - Suspicious DLL detected"
        },
        "incident_id": '$INCIDENT_ID'
    }'

# Test 2.3: Isolate Host
run_test "EDR - Isolate Host" \
    "/api/agents/edr/execute" \
    "POST" \
    '{
        "action_name": "isolate_host",
        "params": {
            "hostname": "WORKSTATION-03",
            "isolation_level": "full",
            "reason": "MCP Integration Test - Ransomware activity detected"
        },
        "incident_id": '$INCIDENT_ID'
    }'

# Test 2.4: Collect Memory Dump
run_test "EDR - Collect Memory Dump" \
    "/api/agents/edr/execute" \
    "POST" \
    '{
        "action_name": "collect_memory_dump",
        "params": {
            "hostname": "WORKSTATION-04",
            "reason": "MCP Integration Test - Forensic analysis required"
        },
        "incident_id": '$INCIDENT_ID'
    }'

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  3. DLP AGENT TESTS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Test 3.1: Scan File for Sensitive Data
run_test "DLP - Scan File for Sensitive Data" \
    "/api/agents/dlp/execute" \
    "POST" \
    '{
        "action_name": "scan_file_for_sensitive_data",
        "params": {
            "file_path": "/shared/customer_data.xlsx",
            "pattern_types": ["ssn", "credit_card"],
            "reason": "MCP Integration Test - PII scan required"
        },
        "incident_id": '$INCIDENT_ID'
    }'

# Test 3.2: Block Upload
run_test "DLP - Block Upload" \
    "/api/agents/dlp/execute" \
    "POST" \
    '{
        "action_name": "block_upload",
        "params": {
            "upload_id": "upload_12345",
            "destination": "https://dropbox.com",
            "username": "test.user@domain.local",
            "reason": "MCP Integration Test - Unauthorized cloud upload"
        },
        "incident_id": '$INCIDENT_ID'
    }'

# Test 3.3: Quarantine Sensitive File
run_test "DLP - Quarantine Sensitive File" \
    "/api/agents/dlp/execute" \
    "POST" \
    '{
        "action_name": "quarantine_sensitive_file",
        "params": {
            "file_path": "/shared/api_keys.txt",
            "reason": "MCP Integration Test - API keys detected in file"
        },
        "incident_id": '$INCIDENT_ID'
    }'

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  4. AGENT ACTION QUERY TESTS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Test 4.1: Get All Agent Actions
run_test "Query - Get All Agent Actions" \
    "/api/agents/actions" \
    "GET"

# Test 4.2: Get Actions for Specific Incident
run_test "Query - Get Actions for Incident #$INCIDENT_ID" \
    "/api/agents/actions/$INCIDENT_ID" \
    "GET"

# Test 4.3: Filter by Agent Type (IAM)
run_test "Query - Filter by IAM Agent" \
    "/api/agents/actions?agent_type=iam&limit=10" \
    "GET"

# Test 4.4: Filter by Status (Success)
run_test "Query - Filter by Success Status" \
    "/api/agents/actions?status=success&limit=10" \
    "GET"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  5. ROLLBACK TESTS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# First, get a rollback ID from previous actions
echo -e "${YELLOW}  ℹ Getting rollback ID from previous actions...${NC}"
actions_response=$(curl -s "$API_BASE/api/agents/actions/$INCIDENT_ID")
rollback_id=$(echo "$actions_response" | jq -r '.[0].rollback_id // empty' 2>/dev/null)

if [ -n "$rollback_id" ] && [ "$rollback_id" != "null" ]; then
    echo -e "${GREEN}  ✓ Found rollback ID: $rollback_id${NC}"
    echo ""
    
    # Test 5.1: Rollback Action
    run_test "Rollback - Reverse Previous Action" \
        "/api/agents/rollback/$rollback_id" \
        "POST" \
        '{
            "reason": "MCP Integration Test - Testing rollback functionality"
        }'
else
    echo -e "${YELLOW}  ⚠ No rollback ID available (actions may not support rollback)${NC}"
    echo -e "${YELLOW}  ⚠ Skipping rollback test${NC}"
    echo ""
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  TEST SUMMARY"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

TOTAL_TESTS=$((TESTS_PASSED + TESTS_FAILED))
SUCCESS_RATE=$((TESTS_PASSED * 100 / TOTAL_TESTS))

echo "Total Tests Run: $TOTAL_TESTS"
echo -e "${GREEN}Tests Passed: $TESTS_PASSED${NC}"
if [ $TESTS_FAILED -gt 0 ]; then
    echo -e "${RED}Tests Failed: $TESTS_FAILED${NC}"
else
    echo "Tests Failed: $TESTS_FAILED"
fi
echo "Success Rate: ${SUCCESS_RATE}%"
echo ""

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "${GREEN}✅ ALL TESTS PASSED!${NC}"
    echo -e "${GREEN}🎉 MCP Agent Integration is working perfectly!${NC}"
    exit 0
else
    echo -e "${RED}❌ SOME TESTS FAILED!${NC}"
    echo -e "${YELLOW}⚠ Please check the errors above and fix the issues.${NC}"
    exit 1
fi



