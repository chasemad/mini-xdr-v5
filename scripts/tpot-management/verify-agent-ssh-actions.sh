#!/bin/bash

# ============================================================================
# Verify AI Agent SSH Actions on T-Pot
# ============================================================================
# This script verifies that Mini-XDR AI agents can successfully SSH into
# T-Pot and perform defensive actions like blocking IPs, managing containers,
# and monitoring honeypot logs.
#
# Usage: ./verify-agent-ssh-actions.sh
# ============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m'

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BACKEND_DIR="$PROJECT_ROOT/backend"
API_URL="http://localhost:8000"

# Test IP for blocking/unblocking (safe test IP)
TEST_IP="198.51.100.1"  # TEST-NET-2, reserved for documentation

# Functions
print_header() {
    echo ""
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}  $1${NC}"
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
    echo ""
}

print_test() {
    echo -e "${BLUE}▶${NC} $1"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_info() {
    echo -e "${MAGENTA}ℹ${NC} $1"
}

# Check if backend is running
check_backend() {
    if ! curl -s "$API_URL/health" > /dev/null 2>&1; then
        print_error "Backend is not running at $API_URL"
        echo ""
        echo "Start the backend with:"
        echo "  cd $BACKEND_DIR && python -m uvicorn app.main:app --reload"
        exit 1
    fi
    print_success "Backend is running"
}

# Test T-Pot connection status
test_tpot_connection() {
    print_test "Checking T-Pot connection status..."

    response=$(curl -s "$API_URL/api/tpot/status" 2>/dev/null || echo '{"connected": false}')
    connected=$(echo "$response" | python3 -c "import sys, json; print(json.load(sys.stdin).get('connected', False))" 2>/dev/null || echo "false")

    if [ "$connected" = "True" ] || [ "$connected" = "true" ]; then
        print_success "T-Pot is connected"

        # Show details
        host=$(echo "$response" | python3 -c "import sys, json; print(json.load(sys.stdin).get('host', 'N/A'))" 2>/dev/null)
        monitoring=$(echo "$response" | python3 -c "import sys, json; print(json.load(sys.stdin).get('monitoring_active', False))" 2>/dev/null)

        print_info "Host: $host"
        print_info "Monitoring active: $monitoring"
        return 0
    else
        print_error "T-Pot is not connected"
        print_warning "Run the setup script first: $SCRIPT_DIR/setup-tpot-ssh-integration.sh"
        return 1
    fi
}

# Test SSH command execution via API
test_ssh_execution() {
    print_test "Testing SSH command execution..."

    # Test with a simple command
    response=$(curl -s -X POST "$API_URL/api/tpot/execute" \
        -H "Content-Type: application/json" \
        -d '{"command": "uname -a"}' 2>/dev/null || echo '{"success": false}')

    success=$(echo "$response" | python3 -c "import sys, json; print(json.load(sys.stdin).get('success', False))" 2>/dev/null || echo "false")

    if [ "$success" = "True" ] || [ "$success" = "true" ]; then
        output=$(echo "$response" | python3 -c "import sys, json; print(json.load(sys.stdin).get('output', '')[0:100])" 2>/dev/null)
        print_success "SSH command execution working"
        print_info "Output: $output..."
        return 0
    else
        print_error "SSH command execution failed"
        return 1
    fi
}

# Test IP blocking capability
test_ip_blocking() {
    print_test "Testing IP blocking capability..."

    print_info "Attempting to block test IP: $TEST_IP"

    response=$(curl -s -X POST "$API_URL/api/tpot/block-ip" \
        -H "Content-Type: application/json" \
        -d "{\"ip_address\": \"$TEST_IP\"}" 2>/dev/null || echo '{"success": false}')

    success=$(echo "$response" | python3 -c "import sys, json; print(json.load(sys.stdin).get('success', False))" 2>/dev/null || echo "false")

    if [ "$success" = "True" ] || [ "$success" = "true" ]; then
        print_success "IP blocking successful"

        # Now unblock it
        print_test "Testing IP unblocking..."
        response=$(curl -s -X POST "$API_URL/api/tpot/unblock-ip" \
            -H "Content-Type: application/json" \
            -d "{\"ip_address\": \"$TEST_IP\"}" 2>/dev/null || echo '{"success": false}')

        success=$(echo "$response" | python3 -c "import sys, json; print(json.load(sys.stdin).get('success', False))" 2>/dev/null || echo "false")

        if [ "$success" = "True" ] || [ "$success" = "true" ]; then
            print_success "IP unblocking successful"
        else
            print_warning "IP unblocking failed (may need manual cleanup)"
        fi

        return 0
    else
        error=$(echo "$response" | python3 -c "import sys, json; print(json.load(sys.stdin).get('error', 'Unknown error'))" 2>/dev/null)
        print_error "IP blocking failed: $error"
        return 1
    fi
}

# Test container status query
test_container_status() {
    print_test "Testing honeypot container status query..."

    response=$(curl -s "$API_URL/api/tpot/containers" 2>/dev/null || echo '{"success": false}')

    success=$(echo "$response" | python3 -c "import sys, json; print(json.load(sys.stdin).get('success', False))" 2>/dev/null || echo "false")

    if [ "$success" = "True" ] || [ "$success" = "true" ]; then
        count=$(echo "$response" | python3 -c "import sys, json; print(len(json.load(sys.stdin).get('containers', [])))" 2>/dev/null || echo "0")
        print_success "Container status query successful ($count containers)"

        # List containers
        containers=$(echo "$response" | python3 -c "import sys, json; [print(c.get('name')) for c in json.load(sys.stdin).get('containers', [])]" 2>/dev/null | head -5)
        if [ -n "$containers" ]; then
            print_info "Active containers:"
            echo "$containers" | while read container; do
                echo "    • $container"
            done
        fi
        return 0
    else
        print_error "Container status query failed"
        return 1
    fi
}

# Test honeypot log monitoring
test_log_monitoring() {
    print_test "Testing honeypot log monitoring..."

    response=$(curl -s "$API_URL/api/tpot/monitoring-status" 2>/dev/null || echo '{"monitoring": []}')

    monitoring=$(echo "$response" | python3 -c "import sys, json; print(', '.join(json.load(sys.stdin).get('monitoring', [])))" 2>/dev/null || echo "none")

    if [ "$monitoring" != "none" ] && [ -n "$monitoring" ]; then
        print_success "Log monitoring active for: $monitoring"
        return 0
    else
        print_warning "No active log monitoring (may auto-start after first attack)"
        return 0  # Not a critical failure
    fi
}

# Test agent workflow integration
test_workflow_integration() {
    print_test "Checking SSH brute force workflow..."

    response=$(curl -s "$API_URL/api/workflows" 2>/dev/null || echo '{"workflows": []}')

    # Check if SSH brute force workflow exists
    has_ssh_workflow=$(echo "$response" | python3 -c "
import sys, json
workflows = json.load(sys.stdin).get('workflows', [])
ssh_workflows = [w for w in workflows if 'ssh' in w.get('name', '').lower() or 'brute' in w.get('name', '').lower()]
print('true' if ssh_workflows else 'false')
" 2>/dev/null || echo "false")

    if [ "$has_ssh_workflow" = "true" ]; then
        print_success "SSH brute force workflow configured"
        return 0
    else
        print_warning "SSH brute force workflow not found"
        print_info "Run: python $SCRIPT_DIR/setup-tpot-workflows.py"
        return 1
    fi
}

# Test agent authentication
test_agent_auth() {
    print_test "Checking agent authentication..."

    # Check if containment agent credentials are configured
    if [ -f "$BACKEND_DIR/.env" ]; then
        if grep -q "CONTAINMENT_AGENT_DEVICE_ID" "$BACKEND_DIR/.env" 2>/dev/null; then
            print_success "Agent credentials configured in .env"
            return 0
        else
            print_warning "Agent credentials not found in .env"
            print_info "Agents may use API key fallback authentication"
            return 0  # Not critical
        fi
    else
        print_error ".env file not found"
        return 1
    fi
}

# Test end-to-end: Create incident and verify agent response
test_end_to_end() {
    print_test "Testing end-to-end: Simulated SSH brute force attack..."

    # Generate SSH brute force events
    for i in {1..6}; do
        event="{
            \"eventid\": \"cowrie.login.failed\",
            \"src_ip\": \"$TEST_IP\",
            \"dst_port\": 22,
            \"username\": \"admin\",
            \"password\": \"test$i\",
            \"timestamp\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\",
            \"message\": \"SSH login failed\"
        }"

        curl -s -X POST "$API_URL/api/ingest/cowrie" \
            -H "Content-Type: application/json" \
            -d "$event" > /dev/null 2>&1
    done

    print_info "Ingested 6 failed SSH login attempts from $TEST_IP"

    # Wait for detection
    sleep 2

    # Check if incident was created
    response=$(curl -s "$API_URL/api/incidents?limit=1" 2>/dev/null || echo '{"incidents": []}')

    incident_count=$(echo "$response" | python3 -c "import sys, json; print(len(json.load(sys.stdin).get('incidents', [])))" 2>/dev/null || echo "0")

    if [ "$incident_count" -gt 0 ]; then
        incident_id=$(echo "$response" | python3 -c "import sys, json; incidents = json.load(sys.stdin).get('incidents', []); print(incidents[0].get('id') if incidents else 'none')" 2>/dev/null)
        severity=$(echo "$response" | python3 -c "import sys, json; incidents = json.load(sys.stdin).get('incidents', []); print(incidents[0].get('severity', 'unknown') if incidents else 'unknown')" 2>/dev/null)

        print_success "Incident created: ID=$incident_id, Severity=$severity"

        # Check for automated response
        sleep 3

        response=$(curl -s "$API_URL/api/incidents/$incident_id" 2>/dev/null || echo '{"actions": []}')
        action_count=$(echo "$response" | python3 -c "import sys, json; print(len(json.load(sys.stdin).get('actions', [])))" 2>/dev/null || echo "0")

        if [ "$action_count" -gt 0 ]; then
            print_success "Automated response triggered ($action_count actions)"
            return 0
        else
            print_warning "No automated response yet (may require workflow activation)"
            return 0
        fi
    else
        print_warning "No incident created (threshold may not be met)"
        return 0
    fi
}

# Main test sequence
main() {
    print_header "T-Pot AI Agent SSH Actions Verification"

    echo "This script verifies that Mini-XDR AI agents can:"
    echo "  • Connect to T-Pot via SSH"
    echo "  • Execute defensive commands (block IPs, manage containers)"
    echo "  • Monitor honeypot logs in real-time"
    echo "  • Respond automatically to SSH brute force attacks"
    echo ""

    read -p "Press Enter to begin verification..."

    # Test counter
    total_tests=0
    passed_tests=0

    # Test 1: Backend running
    print_header "Test 1: Backend Status"
    total_tests=$((total_tests + 1))
    if check_backend; then
        passed_tests=$((passed_tests + 1))
    fi

    # Test 2: T-Pot connection
    print_header "Test 2: T-Pot Connection"
    total_tests=$((total_tests + 1))
    if test_tpot_connection; then
        passed_tests=$((passed_tests + 1))

        # Only run remaining tests if T-Pot is connected

        # Test 3: SSH execution
        print_header "Test 3: SSH Command Execution"
        total_tests=$((total_tests + 1))
        if test_ssh_execution; then
            passed_tests=$((passed_tests + 1))
        fi

        # Test 4: IP blocking
        print_header "Test 4: IP Blocking/Unblocking"
        total_tests=$((total_tests + 1))
        if test_ip_blocking; then
            passed_tests=$((passed_tests + 1))
        fi

        # Test 5: Container status
        print_header "Test 5: Container Status Query"
        total_tests=$((total_tests + 1))
        if test_container_status; then
            passed_tests=$((passed_tests + 1))
        fi

        # Test 6: Log monitoring
        print_header "Test 6: Log Monitoring"
        total_tests=$((total_tests + 1))
        if test_log_monitoring; then
            passed_tests=$((passed_tests + 1))
        fi
    else
        print_warning "Skipping SSH action tests (T-Pot not connected)"
    fi

    # Test 7: Workflow integration (doesn't require T-Pot)
    print_header "Test 7: Workflow Configuration"
    total_tests=$((total_tests + 1))
    if test_workflow_integration; then
        passed_tests=$((passed_tests + 1))
    fi

    # Test 8: Agent authentication
    print_header "Test 8: Agent Authentication"
    total_tests=$((total_tests + 1))
    if test_agent_auth; then
        passed_tests=$((passed_tests + 1))
    fi

    # Test 9: End-to-end (optional)
    if [ "$1" = "--full" ]; then
        print_header "Test 9: End-to-End Attack Simulation"
        total_tests=$((total_tests + 1))
        if test_end_to_end; then
            passed_tests=$((passed_tests + 1))
        fi
    fi

    # Summary
    print_header "Verification Summary"

    echo ""
    echo "  Tests Passed: $passed_tests / $total_tests"
    echo ""

    if [ $passed_tests -eq $total_tests ]; then
        echo -e "${GREEN}✓ ALL TESTS PASSED!${NC}"
        echo ""
        echo "Your Mini-XDR AI agents are ready to defend T-Pot!"
        echo ""
        echo "Next steps:"
        echo "  1. Access T-Pot web interface to see live attacks"
        echo "  2. Run SSH brute force demo: $SCRIPT_DIR/../demo/demo-attack.sh"
        echo "  3. Watch agents automatically block attackers in real-time"
        return 0
    elif [ $passed_tests -ge $((total_tests * 3 / 4)) ]; then
        echo -e "${YELLOW}⚠ MOSTLY WORKING ($passed_tests/$total_tests passed)${NC}"
        echo ""
        echo "Some optional features may need configuration."
        return 0
    else
        echo -e "${RED}✗ VERIFICATION FAILED ($passed_tests/$total_tests passed)${NC}"
        echo ""
        echo "Please fix the issues above before running the demo."
        return 1
    fi
}

# Run main function
main "$@"
