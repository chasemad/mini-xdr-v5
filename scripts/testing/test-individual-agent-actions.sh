#!/bin/bash
# Individual Agent Action Testing Script
# Tests each agent action directly on Azure T-Pot honeypot

# Don't exit on errors - we want to run all tests
set +e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'
BOLD='\033[1m'

# Configuration
HONEYPOT_IP="74.235.242.205"
HONEYPOT_USER="azureuser"
HONEYPOT_SSH_KEY="$HOME/.ssh/mini-xdr-tpot-azure"
HONEYPOT_SSH_PORT="64295"
BACKEND_URL="http://localhost:8000"
TEST_IP="192.168.100.99"  # Safe test IP

# Results tracking
TESTS_PASSED=0
TESTS_FAILED=0
TESTS_WARNING=0

log() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

success() {
    echo -e "${GREEN}✅ $1${NC}"
    ((TESTS_PASSED++))
}

warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
    ((TESTS_WARNING++))
}

error() {
    echo -e "${RED}❌ $1${NC}"
    ((TESTS_FAILED++))
}

section() {
    echo ""
    echo -e "${BOLD}${CYAN}================================================================${NC}"
    echo -e "${BOLD}${CYAN}$1${NC}"
    echo -e "${BOLD}${CYAN}================================================================${NC}"
    echo ""
}

# Test SSH Connection
test_ssh_connection() {
    section "TEST 1: SSH Connection to Azure T-Pot"
    
    log "Testing SSH connection..."
    if ssh -i "$HONEYPOT_SSH_KEY" -p "$HONEYPOT_SSH_PORT" -o StrictHostKeyChecking=no -o ConnectTimeout=10 \
        "$HONEYPOT_USER@$HONEYPOT_IP" "echo 'Connection successful'" 2>/dev/null; then
        success "SSH connection successful"
        return 0
    else
        error "SSH connection failed"
        return 1
    fi
}

# Test Containment Agent - Block IP
test_block_ip_action() {
    section "TEST 2: Containment Agent - Block IP Action"
    
    log "Testing IP blocking capability..."
    
    # Test 1: Check iptables access (read-only)
    log "Checking iptables access..."
    if ssh -i "$HONEYPOT_SSH_KEY" -p "$HONEYPOT_SSH_PORT" -o StrictHostKeyChecking=no \
        "$HONEYPOT_USER@$HONEYPOT_IP" "sudo iptables -L INPUT -n | head -5" 2>/dev/null >/dev/null; then
        success "Can read iptables rules"
    else
        warning "Cannot read iptables rules (may need sudo config)"
    fi
    
    # Test 2: Test block_ip API endpoint
    log "Testing block_ip API endpoint..."
    response=$(curl -s -w "%{http_code}" -o /tmp/block_response.json -X POST "$BACKEND_URL/api/agents/orchestrate" \
        -H "Content-Type: application/json" \
        -d '{
            "query": "Block IP address 192.168.100.99",
            "agent_type": "containment",
            "context": {"action": "block_ip", "ip": "'$TEST_IP'"}
        }' 2>/dev/null)
    
    if [ "$response" = "200" ]; then
        success "Block IP API endpoint responded successfully"
        log "Response: $(cat /tmp/block_response.json | python3 -m json.tool 2>/dev/null || cat /tmp/block_response.json)"
    else
        error "Block IP API endpoint failed (HTTP $response)"
    fi
}

# Test Containment Agent - Isolate Host
test_isolate_host_action() {
    section "TEST 3: Containment Agent - Isolate Host Action"
    
    log "Testing host isolation capability..."
    
    response=$(curl -s -w "%{http_code}" -o /tmp/isolate_response.json -X POST "$BACKEND_URL/api/agents/orchestrate" \
        -H "Content-Type: application/json" \
        -d '{
            "query": "Isolate compromised host",
            "agent_type": "containment",
            "context": {"action": "isolate_host", "host": "test-host"}
        }' 2>/dev/null)
    
    if [ "$response" = "200" ]; then
        success "Isolate host API endpoint responded successfully"
        log "Response: $(cat /tmp/isolate_response.json | python3 -m json.tool 2>/dev/null || cat /tmp/isolate_response.json)"
    else
        error "Isolate host API endpoint failed (HTTP $response)"
    fi
}

# Test Forensics Agent - Collect Evidence
test_collect_evidence_action() {
    section "TEST 4: Forensics Agent - Collect Evidence"
    
    log "Testing evidence collection capability..."
    
    response=$(curl -s -w "%{http_code}" -o /tmp/forensics_response.json -X POST "$BACKEND_URL/api/agents/orchestrate" \
        -H "Content-Type: application/json" \
        -d '{
            "query": "Collect forensic evidence from recent attack",
            "agent_type": "forensics",
            "context": {"action": "collect_evidence"}
        }' 2>/dev/null)
    
    if [ "$response" = "200" ]; then
        success "Collect evidence API endpoint responded successfully"
        log "Response: $(cat /tmp/forensics_response.json | python3 -m json.tool 2>/dev/null || cat /tmp/forensics_response.json)"
    else
        error "Collect evidence API endpoint failed (HTTP $response)"
    fi
}

# Test Forensics Agent - Analyze Malware
test_analyze_malware_action() {
    section "TEST 5: Forensics Agent - Analyze Malware"
    
    log "Testing malware analysis capability..."
    
    response=$(curl -s -w "%{http_code}" -o /tmp/malware_response.json -X POST "$BACKEND_URL/api/agents/orchestrate" \
        -H "Content-Type: application/json" \
        -d '{
            "query": "Analyze malware sample",
            "agent_type": "forensics",
            "context": {"action": "analyze_malware"}
        }' 2>/dev/null)
    
    if [ "$response" = "200" ]; then
        success "Analyze malware API endpoint responded successfully"
        log "Response: $(cat /tmp/malware_response.json | python3 -m json.tool 2>/dev/null || cat /tmp/malware_response.json)"
    else
        error "Analyze malware API endpoint failed (HTTP $response)"
    fi
}

# Test Attribution Agent
test_attribution_agent() {
    section "TEST 6: Attribution Agent - Profile Threat Actor"
    
    log "Testing threat actor profiling..."
    
    response=$(curl -s -w "%{http_code}" -o /tmp/attribution_response.json -X POST "$BACKEND_URL/api/agents/orchestrate" \
        -H "Content-Type: application/json" \
        -d '{
            "query": "Profile threat actor from IP 192.168.100.99",
            "agent_type": "attribution",
            "context": {"action": "profile_threat_actor", "ip": "'$TEST_IP'"}
        }' 2>/dev/null)
    
    if [ "$response" = "200" ]; then
        success "Attribution agent API endpoint responded successfully"
        log "Response: $(cat /tmp/attribution_response.json | python3 -m json.tool 2>/dev/null || cat /tmp/attribution_response.json)"
    else
        error "Attribution agent API endpoint failed (HTTP $response)"
    fi
}

# Test Threat Hunting Agent
test_threat_hunting_agent() {
    section "TEST 7: Threat Hunting Agent - Hunt Similar Attacks"
    
    log "Testing threat hunting capability..."
    
    response=$(curl -s -w "%{http_code}" -o /tmp/hunting_response.json -X POST "$BACKEND_URL/api/agents/orchestrate" \
        -H "Content-Type: application/json" \
        -d '{
            "query": "Hunt for similar SSH brute force attacks",
            "agent_type": "threat_hunting",
            "context": {"action": "hunt_similar_attacks", "pattern": "ssh_brute_force"}
        }' 2>/dev/null)
    
    if [ "$response" = "200" ]; then
        success "Threat hunting agent API endpoint responded successfully"
        log "Response: $(cat /tmp/hunting_response.json | python3 -m json.tool 2>/dev/null || cat /tmp/hunting_response.json)"
    else
        error "Threat hunting agent API endpoint failed (HTTP $response)"
    fi
}

# Test Deception Agent
test_deception_agent() {
    section "TEST 8: Deception Agent - Deploy Honeypot"
    
    log "Testing honeypot deployment capability..."
    
    response=$(curl -s -w "%{http_code}" -o /tmp/deception_response.json -X POST "$BACKEND_URL/api/agents/orchestrate" \
        -H "Content-Type: application/json" \
        -d '{
            "query": "Deploy decoy honeypot service",
            "agent_type": "deception",
            "context": {"action": "deploy_honeypot"}
        }' 2>/dev/null)
    
    if [ "$response" = "200" ]; then
        success "Deception agent API endpoint responded successfully"
        log "Response: $(cat /tmp/deception_response.json | python3 -m json.tool 2>/dev/null || cat /tmp/deception_response.json)"
    else
        error "Deception agent API endpoint failed (HTTP $response)"
    fi
}

# Test Workflow Trigger Execution
test_workflow_triggers() {
    section "TEST 9: Workflow Trigger Execution"
    
    log "Testing workflow trigger system..."
    
    # Check if workflows are being evaluated
    log "Checking trigger evaluator status..."
    
    # Send test event that should trigger a workflow
    response=$(curl -s -w "%{http_code}" -o /tmp/trigger_response.json -X POST "$BACKEND_URL/ingest/multi" \
        -H "Content-Type: application/json" \
        -d '{
            "events": [
                {
                    "timestamp": "'$(date -u +"%Y-%m-%dT%H:%M:%S")'",
                    "eventid": "cowrie.login.failed",
                    "src_ip": "'$TEST_IP'",
                    "username": "admin",
                    "password": "password123",
                    "message": "Failed login attempt",
                    "source": "honeypot"
                }
            ],
            "source": "test_trigger"
        }' 2>/dev/null)
    
    if [ "$response" = "200" ]; then
        success "Event ingestion successful"
        log "Workflows should be automatically evaluated"
    else
        error "Event ingestion failed (HTTP $response)"
    fi
}

# Test Real Honeypot Data Flow
test_honeypot_data_flow() {
    section "TEST 10: Real Honeypot Data Flow"
    
    log "Testing data flow from Azure T-Pot to Mini-XDR..."
    
    # Check if Fluent Bit is running on T-Pot
    log "Checking Fluent Bit status on T-Pot..."
    if ssh -i "$HONEYPOT_SSH_KEY" -p "$HONEYPOT_SSH_PORT" -o StrictHostKeyChecking=no \
        "$HONEYPOT_USER@$HONEYPOT_IP" "systemctl is-active fluent-bit" 2>/dev/null | grep -q "active"; then
        success "Fluent Bit is running on T-Pot"
    else
        warning "Fluent Bit status unclear (may need verification)"
    fi
    
    # Check if events are being received
    log "Checking recent events in Mini-XDR..."
    response=$(curl -s "$BACKEND_URL/events?limit=5" 2>/dev/null)
    event_count=$(echo "$response" | python3 -c "import sys, json; print(len(json.load(sys.stdin)) if isinstance(json.load(sys.stdin), list) else 0)" 2>/dev/null || echo "0")
    
    if [ "$event_count" -gt 0 ]; then
        success "Events are being received ($event_count recent events)"
    else
        warning "No recent events found (honeypot may not have activity)"
    fi
}

# Main execution
main() {
    section "INDIVIDUAL AGENT ACTION TESTING"
    
    log "Backend URL: $BACKEND_URL"
    log "Azure Honeypot: $HONEYPOT_IP:$HONEYPOT_SSH_PORT"
    log "Test Time: $(date -u +"%Y-%m-%d %H:%M:%S UTC")"
    
    # Check backend health
    log "Checking backend health..."
    if curl -s "$BACKEND_URL/health" | grep -q "healthy"; then
        success "Backend is healthy"
    else
        error "Backend is not responding"
        exit 1
    fi
    
    # Run all tests
    test_ssh_connection || true
    test_block_ip_action || true
    test_isolate_host_action || true
    test_collect_evidence_action || true
    test_analyze_malware_action || true
    test_attribution_agent || true
    test_threat_hunting_agent || true
    test_deception_agent || true
    test_workflow_triggers || true
    test_honeypot_data_flow || true
    
    # Print summary
    section "TEST SUMMARY"
    
    TOTAL_TESTS=$((TESTS_PASSED + TESTS_FAILED + TESTS_WARNING))
    
    echo -e "${BOLD}Total Tests:${NC} $TOTAL_TESTS"
    echo -e "${GREEN}Passed:${NC} $TESTS_PASSED"
    echo -e "${YELLOW}Warnings:${NC} $TESTS_WARNING"
    echo -e "${RED}Failed:${NC} $TESTS_FAILED"
    
    if [ $TOTAL_TESTS -gt 0 ]; then
        PASS_RATE=$(echo "scale=1; $TESTS_PASSED * 100 / $TOTAL_TESTS" | bc)
        echo -e "\n${BOLD}Pass Rate:${NC} ${PASS_RATE}%"
    fi
    
    echo ""
    echo -e "${CYAN}================================================================${NC}"
    echo ""
    
    # Cleanup
    rm -f /tmp/*_response.json
    
    # Exit with status
    [ $TESTS_FAILED -eq 0 ]
}

main "$@"

