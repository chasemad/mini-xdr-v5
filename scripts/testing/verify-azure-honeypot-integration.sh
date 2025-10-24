#!/bin/bash
# =============================================================================
# Azure T-Pot Honeypot Integration Verification Script
# =============================================================================
# This script verifies that:
# 1. Azure T-Pot honeypot is accessible
# 2. Fluent Bit is forwarding logs to Mini-XDR
# 3. Mini-XDR is receiving and processing events
# 4. Detection models are working
# =============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ENV_FILE="$PROJECT_ROOT/backend/.env"

# Azure T-Pot Configuration
TPOT_IP="74.235.242.205"
TPOT_SSH_PORT="64295"
TPOT_WEB_PORT="64297"
TPOT_SSH_KEY="$HOME/.ssh/mini-xdr-tpot-azure"

# Mini-XDR Configuration
MINI_XDR_API="http://localhost:8000"

# Test results
TESTS_PASSED=0
TESTS_FAILED=0
TESTS_WARNING=0

# =============================================================================
# Utility Functions
# =============================================================================

print_header() {
    echo -e "\n${BLUE}${BOLD}╔════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}${BOLD}║${NC} $1"
    echo -e "${BLUE}${BOLD}╚════════════════════════════════════════════════════════════════════╝${NC}\n"
}

print_test() {
    echo -e "${YELLOW}[TEST]${NC} $1"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
    ((TESTS_PASSED++))
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
    ((TESTS_FAILED++))
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
    ((TESTS_WARNING++))
}

print_info() {
    echo -e "${CYAN}ℹ️  $1${NC}"
}

# =============================================================================
# Test 1: Azure T-Pot Connectivity
# =============================================================================

test_tpot_connectivity() {
    print_header "TEST 1: Azure T-Pot Connectivity"
    
    # Test 1.1: SSH Connectivity
    print_test "1.1 SSH Connectivity (Port $TPOT_SSH_PORT)"
    print_info "Testing connection to Azure T-Pot (this may take 10-15 seconds)..."
    if gtimeout 20 ssh -i "$TPOT_SSH_KEY" -p "$TPOT_SSH_PORT" \
        -o StrictHostKeyChecking=no \
        -o ConnectTimeout=15 \
        azureuser@"$TPOT_IP" "echo 'Connected'" 2>&1 | grep -q "Connected"; then
        print_success "SSH connection successful"
    else
        print_error "SSH connection failed - Check SSH key at $TPOT_SSH_KEY"
        print_info "Manual test: ssh -i $TPOT_SSH_KEY -p $TPOT_SSH_PORT azureuser@$TPOT_IP"
        return 1
    fi
    
    # Test 1.2: Web Interface Connectivity
    print_test "1.2 Web Interface Connectivity (Port $TPOT_WEB_PORT)"
    if gtimeout 5 curl -sSf -k "https://$TPOT_IP:$TPOT_WEB_PORT" &>/dev/null; then
        print_success "Web interface accessible"
    else
        print_warning "Web interface not accessible (may require VPN/firewall rules)"
    fi
    
    # Test 1.3: Honeypot Services Status
    print_test "1.3 T-Pot Honeypot Services Status"
    print_info "Checking running containers..."
    SERVICES_OUTPUT=$(gtimeout 20 ssh -i "$TPOT_SSH_KEY" -p "$TPOT_SSH_PORT" \
        -o StrictHostKeyChecking=no \
        -o ConnectTimeout=15 \
        azureuser@"$TPOT_IP" "docker ps --format '{{.Names}}' 2>/dev/null | grep -E '(cowrie|dionaea|suricata)' | wc -l" 2>/dev/null || echo "0")
    
    if [ "$SERVICES_OUTPUT" -ge 3 ]; then
        print_success "T-Pot services running ($SERVICES_OUTPUT containers)"
    else
        print_warning "Only $SERVICES_OUTPUT honeypot containers running (expected 3+)"
    fi
}

# =============================================================================
# Test 2: Fluent Bit Log Forwarding
# =============================================================================

test_fluent_bit_forwarding() {
    print_header "TEST 2: Fluent Bit Log Forwarding"
    
    # Test 2.1: Fluent Bit Service Status
    print_test "2.1 Fluent Bit Service Status on T-Pot"
    print_info "Checking Fluent Bit service..."
    FB_STATUS=$(gtimeout 20 ssh -i "$TPOT_SSH_KEY" -p "$TPOT_SSH_PORT" \
        -o StrictHostKeyChecking=no \
        -o ConnectTimeout=15 \
        azureuser@"$TPOT_IP" \
        "systemctl is-active fluent-bit 2>/dev/null || systemctl is-active td-agent-bit 2>/dev/null || echo 'inactive'" 2>/dev/null)
    
    if [ "$FB_STATUS" = "active" ]; then
        print_success "Fluent Bit service is running"
    else
        print_error "Fluent Bit service not running (status: $FB_STATUS)"
        print_info "To fix: SSH to T-Pot and run 'sudo systemctl start fluent-bit'"
        return 1
    fi
    
    # Test 2.2: Fluent Bit Configuration
    print_test "2.2 Fluent Bit Configuration Check"
    print_info "Checking Fluent Bit configuration files..."
    FB_CONFIG=$(gtimeout 20 ssh -i "$TPOT_SSH_KEY" -p "$TPOT_SSH_PORT" \
        -o StrictHostKeyChecking=no \
        -o ConnectTimeout=15 \
        azureuser@"$TPOT_IP" \
        "grep -l 'ingest/multi' /etc/fluent-bit/*.conf /opt/fluent-bit/etc/*.conf 2>/dev/null | head -1" 2>/dev/null || echo "")
    
    if [ -n "$FB_CONFIG" ]; then
        print_success "Fluent Bit configured to forward to Mini-XDR"
        print_info "Config file: $FB_CONFIG"
    else
        print_error "Fluent Bit not configured for Mini-XDR ingestion"
        print_info "Run: cd $PROJECT_ROOT && ./scripts/tpot-management/deploy-tpot-logging.sh"
        return 1
    fi
    
    # Test 2.3: Recent Log Activity
    print_test "2.3 T-Pot Log Activity Check"
    print_info "Checking for recent honeypot log files..."
    LOG_COUNT=$(gtimeout 20 ssh -i "$TPOT_SSH_KEY" -p "$TPOT_SSH_PORT" \
        -o StrictHostKeyChecking=no \
        -o ConnectTimeout=15 \
        azureuser@"$TPOT_IP" \
        "find /data/cowrie/log /data/dionaea/log /data/suricata/log -name '*.json' -mmin -10 2>/dev/null | wc -l" 2>/dev/null || echo "0")
    
    if [ "$LOG_COUNT" -gt 0 ]; then
        print_success "Recent honeypot log activity detected ($LOG_COUNT files modified in last 10min)"
    else
        print_warning "No recent honeypot log activity (honeypot may need traffic)"
    fi
}

# =============================================================================
# Test 3: Mini-XDR Event Ingestion
# =============================================================================

test_mini_xdr_ingestion() {
    print_header "TEST 3: Mini-XDR Event Ingestion"
    
    # Test 3.1: Backend Health
    print_test "3.1 Mini-XDR Backend Health"
    if HEALTH=$(curl -s "$MINI_XDR_API/health" 2>/dev/null); then
        STATUS=$(echo "$HEALTH" | jq -r '.status' 2>/dev/null || echo "unknown")
        if [ "$STATUS" = "healthy" ]; then
            print_success "Mini-XDR backend is healthy"
        else
            print_error "Mini-XDR backend unhealthy (status: $STATUS)"
            return 1
        fi
    else
        print_error "Mini-XDR backend not responding at $MINI_XDR_API"
        print_info "Start backend: cd $PROJECT_ROOT/backend && python main.py"
        return 1
    fi
    
    # Test 3.2: Recent Event Ingestion
    print_test "3.2 Recent Event Ingestion"
    EVENTS=$(curl -s "$MINI_XDR_API/events?limit=100" 2>/dev/null)
    EVENT_COUNT=$(echo "$EVENTS" | jq '. | length' 2>/dev/null || echo "0")
    
    if [ "$EVENT_COUNT" -gt 0 ]; then
        print_success "$EVENT_COUNT events in Mini-XDR database"
        
        # Check for recent events from T-Pot
        RECENT_TPOT=$(echo "$EVENTS" | jq '[.[] | select(.source_type == "tpot" or .source_type == "cowrie")] | length' 2>/dev/null || echo "0")
        if [ "$RECENT_TPOT" -gt 0 ]; then
            print_success "$RECENT_TPOT events from T-Pot honeypot"
        else
            print_warning "No events from T-Pot source (check Fluent Bit forwarding)"
        fi
    else
        print_warning "No events in Mini-XDR database (system may be new)"
    fi
    
    # Test 3.3: Ingestion Endpoint Accessibility
    print_test "3.3 Ingestion Endpoint Test"
    TEST_PAYLOAD='{"source_type":"test","hostname":"verification-test","events":[{"eventid":"test.verification","src_ip":"192.0.2.1","message":"Verification test"}]}'
    
    INGEST_RESPONSE=$(curl -s -X POST "$MINI_XDR_API/ingest/multi" \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer test-key" \
        -d "$TEST_PAYLOAD" 2>/dev/null || echo '{}')
    
    PROCESSED=$(echo "$INGEST_RESPONSE" | jq -r '.processed' 2>/dev/null || echo "0")
    if [ "$PROCESSED" -gt 0 ]; then
        print_success "Ingestion endpoint processing events correctly"
    else
        print_warning "Ingestion endpoint may have issues (processed: $PROCESSED)"
    fi
}

# =============================================================================
# Test 4: Detection Models
# =============================================================================

test_detection_models() {
    print_header "TEST 4: Detection Models"
    
    # Test 4.1: ML Model Status
    print_test "4.1 ML Model Status"
    ML_STATUS=$(curl -s "$MINI_XDR_API/api/ml/status" 2>/dev/null || echo '{}')
    MODEL_LOADED=$(echo "$ML_STATUS" | jq -r '.model_loaded' 2>/dev/null || echo "false")
    
    if [ "$MODEL_LOADED" = "true" ]; then
        print_success "ML detection model loaded"
        MODEL_TYPE=$(echo "$ML_STATUS" | jq -r '.model_type' 2>/dev/null || echo "unknown")
        print_info "Model type: $MODEL_TYPE"
    else
        print_warning "ML detection model not loaded (local ML fallback will be used)"
    fi
    
    # Test 4.2: Enhanced Threat Detector
    print_test "4.2 Enhanced Threat Detector Status"
    ENHANCED_STATUS=$(curl -s "$MINI_XDR_API/api/ml/enhanced/status" 2>/dev/null || echo '{}')
    ENHANCED_LOADED=$(echo "$ENHANCED_STATUS" | jq -r '.model_loaded' 2>/dev/null || echo "false")
    
    if [ "$ENHANCED_LOADED" = "true" ]; then
        print_success "Enhanced threat detector loaded"
    else
        print_warning "Enhanced detector not loaded (heuristic detection will be used)"
    fi
    
    # Test 4.3: Recent Incidents
    print_test "4.3 Incident Detection"
    INCIDENTS=$(curl -s "$MINI_XDR_API/incidents?limit=10" 2>/dev/null || echo '[]')
    INCIDENT_COUNT=$(echo "$INCIDENTS" | jq '. | length' 2>/dev/null || echo "0")
    
    if [ "$INCIDENT_COUNT" -gt 0 ]; then
        print_success "$INCIDENT_COUNT recent incidents detected"
        
        # Show incident summary
        echo "$INCIDENTS" | jq -r '.[] | "  • Incident #\(.id): \(.reason) (IP: \(.src_ip))"' 2>/dev/null | head -3
    else
        print_info "No incidents detected yet (system may need attack traffic)"
    fi
}

# =============================================================================
# Test 5: End-to-End Flow
# =============================================================================

test_end_to_end_flow() {
    print_header "TEST 5: End-to-End Flow Verification"
    
    print_test "5.1 Simulating SSH Brute Force to Azure Honeypot"
    print_info "Generating 5 failed SSH login attempts to $TPOT_IP..."
    
    # Generate some test traffic
    for i in {1..5}; do
        gtimeout 2 ssh -o StrictHostKeyChecking=no -o ConnectTimeout=2 \
            "testuser$i@$TPOT_IP" -p 22 2>/dev/null || true
        sleep 1
    done
    
    print_success "Test attacks completed"
    
    print_test "5.2 Waiting for Event Processing (15 seconds)"
    sleep 15
    
    print_test "5.3 Checking for New Events"
    NEW_EVENTS=$(curl -s "$MINI_XDR_API/events?limit=20" 2>/dev/null)
    RECENT_COUNT=$(echo "$NEW_EVENTS" | jq '[.[] | select(.ts | fromdateiso8601 > (now - 60))] | length' 2>/dev/null || echo "0")
    
    if [ "$RECENT_COUNT" -gt 0 ]; then
        print_success "$RECENT_COUNT new events detected in last minute"
        
        # Check if any are from our test IP
        FAILED_LOGINS=$(echo "$NEW_EVENTS" | jq '[.[] | select(.eventid == "cowrie.login.failed")] | length' 2>/dev/null || echo "0")
        if [ "$FAILED_LOGINS" -gt 0 ]; then
            print_success "SSH brute force events detected: $FAILED_LOGINS"
        fi
    else
        print_warning "No new events in last minute (log forwarding delay possible)"
    fi
    
    print_test "5.4 Checking for Incident Creation"
    NEW_INCIDENTS=$(curl -s "$MINI_XDR_API/incidents?limit=5" 2>/dev/null)
    RECENT_INCIDENTS=$(echo "$NEW_INCIDENTS" | jq '[.[] | select(.created_at | fromdateiso8601 > (now - 120))] | length' 2>/dev/null || echo "0")
    
    if [ "$RECENT_INCIDENTS" -gt 0 ]; then
        print_success "New incidents created: $RECENT_INCIDENTS"
    else
        print_info "No new incidents (detection threshold may not be met)"
    fi
}

# =============================================================================
# Summary and Recommendations
# =============================================================================

print_summary() {
    print_header "VERIFICATION SUMMARY"
    
    TOTAL_TESTS=$((TESTS_PASSED + TESTS_FAILED + TESTS_WARNING))
    
    echo -e "${BOLD}Test Results:${NC}"
    echo -e "  Total Tests: $TOTAL_TESTS"
    echo -e "  ${GREEN}Passed: $TESTS_PASSED${NC}"
    echo -e "  ${RED}Failed: $TESTS_FAILED${NC}"
    echo -e "  ${YELLOW}Warnings: $TESTS_WARNING${NC}"
    echo ""
    
    if [ $TESTS_FAILED -eq 0 ]; then
        echo -e "${GREEN}${BOLD}✅ AZURE HONEYPOT INTEGRATION VERIFIED!${NC}\n"
        echo -e "${CYAN}Your system is ready for comprehensive testing.${NC}"
        echo -e "${CYAN}Next steps:${NC}"
        echo -e "  1. Run comprehensive attack simulation:"
        echo -e "     ${YELLOW}./scripts/testing/test-comprehensive-honeypot-attacks.sh${NC}"
        echo -e "  2. Monitor live dashboard at:"
        echo -e "     ${YELLOW}http://localhost:3000${NC}"
        echo -e "  3. View T-Pot dashboard at:"
        echo -e "     ${YELLOW}https://$TPOT_IP:$TPOT_WEB_PORT${NC}"
    else
        echo -e "${RED}${BOLD}❌ VERIFICATION FAILED${NC}\n"
        echo -e "${YELLOW}Issues detected:${NC}"
        if [ $TESTS_FAILED -gt 0 ]; then
            echo -e "  • $TESTS_FAILED critical tests failed"
        fi
        if [ $TESTS_WARNING -gt 0 ]; then
            echo -e "  • $TESTS_WARNING warnings need attention"
        fi
        echo ""
        echo -e "${CYAN}Troubleshooting:${NC}"
        echo -e "  1. Check backend logs: ${YELLOW}tail -f $PROJECT_ROOT/backend/backend.log${NC}"
        echo -e "  2. Verify T-Pot services: ${YELLOW}ssh -i $TPOT_SSH_KEY -p $TPOT_SSH_PORT azureuser@$TPOT_IP 'docker ps'${NC}"
        echo -e "  3. Check Fluent Bit logs: ${YELLOW}ssh -i $TPOT_SSH_KEY -p $TPOT_SSH_PORT azureuser@$TPOT_IP 'sudo journalctl -u fluent-bit -n 50'${NC}"
    fi
    
    echo ""
}

# =============================================================================
# Main Execution
# =============================================================================

main() {
    echo -e "${BOLD}${BLUE}"
    echo "╔═══════════════════════════════════════════════════════════════════════╗"
    echo "║                                                                       ║"
    echo "║    AZURE T-POT HONEYPOT INTEGRATION VERIFICATION                     ║"
    echo "║                                                                       ║"
    echo "╚═══════════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}\n"
    
    print_info "Target: Azure T-Pot @ $TPOT_IP"
    print_info "Mini-XDR API: $MINI_XDR_API"
    echo ""
    
    # Run all tests (continue even if some fail)
    test_tpot_connectivity || true
    test_fluent_bit_forwarding || true
    test_mini_xdr_ingestion || true
    test_detection_models || true
    test_end_to_end_flow || true
    
    # Print summary
    print_summary
    
    # Exit with appropriate code
    if [ $TESTS_FAILED -eq 0 ]; then
        exit 0
    else
        exit 1
    fi
}

# Run main function
main "$@"

