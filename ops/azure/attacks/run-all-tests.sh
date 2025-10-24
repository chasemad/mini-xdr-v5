#!/bin/bash
# ============================================================================
# Mini-XDR Complete Attack Simulation Suite
# ============================================================================
# Runs all attack simulations against the mini corporate network
# to validate Mini-XDR detection and response capabilities
# ============================================================================

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
TERRAFORM_DIR="$PROJECT_ROOT/ops/azure/terraform"

echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║       Mini-XDR Attack Simulation Suite                         ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Get configuration
if [ -f "$TERRAFORM_DIR/terraform.tfstate" ]; then
    KEY_VAULT_NAME=$(terraform -chdir="$TERRAFORM_DIR" output -raw key_vault_name)
    APPGW_IP=$(terraform -chdir="$TERRAFORM_DIR" output -raw appgw_public_ip)
    API_KEY=$(az keyvault secret show --vault-name "$KEY_VAULT_NAME" --name "mini-xdr-api-key" --query value -o tsv)
    DC_IP=$(terraform -chdir="$TERRAFORM_DIR" output -raw domain_controller_private_ip)
else
    echo -e "${RED}❌ Terraform state not found. Deploy infrastructure first.${NC}"
    exit 1
fi

BACKEND_URL="https://$APPGW_IP"
REPORT_FILE="/tmp/mini-xdr-attack-test-report-$(date +%Y%m%d-%H%M%S).txt"

echo "Configuration:"
echo "  • Backend URL: $BACKEND_URL"
echo "  • Domain Controller: $DC_IP"
echo "  • Report: $REPORT_FILE"
echo ""

# Initialize report
cat > "$REPORT_FILE" << EOF
Mini-XDR Attack Simulation Report
Generated: $(date)
Backend: $BACKEND_URL
Domain Controller: $DC_IP

================================================
EOF

# Counter for results
TOTAL_TESTS=0
DETECTED_TESTS=0

# Function to run test and check detection
run_test() {
    local TEST_NAME="$1"
    local TEST_COMMAND="$2"
    local EXPECTED_DETECTION="$3"
    
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    
    echo ""
    echo -e "${YELLOW}[$TOTAL_TESTS] Testing: $TEST_NAME${NC}"
    echo "  Command: $TEST_COMMAND"
    
    # Run the test
    eval "$TEST_COMMAND" > /dev/null 2>&1 || true
    
    # Wait for detection
    sleep 5
    
    # Check if detected
    RECENT_INCIDENTS=$(curl -s -H "X-API-Key: $API_KEY" "$BACKEND_URL/api/incidents?limit=10")
    
    if echo "$RECENT_INCIDENTS" | grep -qi "$EXPECTED_DETECTION"; then
        echo -e "  ${GREEN}✓ DETECTED${NC}: $EXPECTED_DETECTION"
        echo "[$TOTAL_TESTS] $TEST_NAME: DETECTED ✓" >> "$REPORT_FILE"
        DETECTED_TESTS=$((DETECTED_TESTS + 1))
    else
        echo -e "  ${RED}✗ NOT DETECTED${NC}: $EXPECTED_DETECTION"
        echo "[$TOTAL_TESTS] $TEST_NAME: NOT DETECTED ✗" >> "$REPORT_FILE"
    fi
}

# ============================================================================
# Attack Simulations
# ============================================================================

echo -e "${BLUE}Starting attack simulations...${NC}"
echo ""

# Test 1: Brute Force Attack
run_test \
    "SSH Brute Force" \
    "for i in {1..10}; do ssh -o ConnectTimeout=1 admin@$DC_IP 2>/dev/null; done" \
    "Brute Force"

# Test 2: Port Scanning
run_test \
    "Network Reconnaissance" \
    "nmap -sT -p 1-1000 $DC_IP 2>/dev/null || nc -zv $DC_IP 80 443 3389 445 2>/dev/null" \
    "Reconnaissance"

# Test 3: Simulated Kerberos Attack (via API injection)
run_test \
    "Kerberos Attack Simulation" \
    "curl -s -X POST -H 'Content-Type: application/json' -H 'X-API-Key: $API_KEY' $BACKEND_URL/ingest/json -d '{\"src_ip\":\"$DC_IP\",\"event_type\":\"kerberos_attack\",\"severity\":\"high\"}'" \
    "Kerberos"

# Test 4: Simulated Lateral Movement
run_test \
    "Lateral Movement Simulation" \
    "curl -s -X POST -H 'Content-Type: application/json' -H 'X-API-Key: $API_KEY' $BACKEND_URL/ingest/json -d '{\"src_ip\":\"$DC_IP\",\"event_type\":\"lateral_movement\",\"severity\":\"high\",\"technique\":\"psexec\"}'" \
    "Lateral Movement"

# Test 5: Simulated Credential Theft
run_test \
    "Credential Theft Simulation" \
    "curl -s -X POST -H 'Content-Type: application/json' -H 'X-API-Key: $API_KEY' $BACKEND_URL/ingest/json -d '{\"src_ip\":\"$DC_IP\",\"event_type\":\"credential_theft\",\"severity\":\"critical\",\"technique\":\"lsass_dump\"}'" \
    "Credential"

# Test 6: Simulated Data Exfiltration
run_test \
    "Data Exfiltration Simulation" \
    "curl -s -X POST -H 'Content-Type: application/json' -H 'X-API-Key: $API_KEY' $BACKEND_URL/ingest/json -d '{\"src_ip\":\"$DC_IP\",\"event_type\":\"data_exfiltration\",\"severity\":\"high\",\"bytes\":1073741824}'" \
    "Exfiltration"

# Test 7: Web Application Attack
run_test \
    "SQL Injection Attempt" \
    "curl -s '$BACKEND_URL/?id=1%27%20OR%20%271%27=%271' 2>/dev/null" \
    "Web Attack"

# ============================================================================
# Generate Report
# ============================================================================

echo "" >> "$REPORT_FILE"
echo "================================================" >> "$REPORT_FILE"
echo "SUMMARY" >> "$REPORT_FILE"
echo "================================================" >> "$REPORT_FILE"
echo "Total Tests: $TOTAL_TESTS" >> "$REPORT_FILE"
echo "Detected: $DETECTED_TESTS" >> "$REPORT_FILE"
echo "Missed: $((TOTAL_TESTS - DETECTED_TESTS))" >> "$REPORT_FILE"
echo "Detection Rate: $(awk "BEGIN {printf \"%.1f\", ($DETECTED_TESTS/$TOTAL_TESTS)*100}")%" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"

# Display results
echo ""
echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║              Attack Simulation Results                         ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo "Total Tests: $TOTAL_TESTS"
echo "Detected: $DETECTED_TESTS"
echo "Missed: $((TOTAL_TESTS - DETECTED_TESTS))"
DETECTION_RATE=$(awk "BEGIN {printf \"%.1f\", ($DETECTED_TESTS/$TOTAL_TESTS)*100}")
echo "Detection Rate: ${DETECTION_RATE}%"
echo ""

if [ $DETECTED_TESTS -eq $TOTAL_TESTS ]; then
    echo -e "${GREEN}✓ ALL ATTACKS DETECTED! Perfect score! 🎯${NC}"
elif [ $DETECTED_TESTS -ge $((TOTAL_TESTS * 9 / 10)) ]; then
    echo -e "${GREEN}✓ Excellent detection rate (>90%)${NC}"
elif [ $DETECTED_TESTS -ge $((TOTAL_TESTS * 8 / 10)) ]; then
    echo -e "${YELLOW}⚠ Good detection rate (>80%), but could be better${NC}"
else
    echo -e "${RED}✗ Low detection rate (<80%), investigation needed${NC}"
fi

echo ""
echo "Detailed report saved to: $REPORT_FILE"
echo ""

# Display recent incidents
echo -e "${YELLOW}Recent Incidents:${NC}"
curl -s -H "X-API-Key: $API_KEY" "$BACKEND_URL/api/incidents?limit=10" | jq -r '.[] | "  • [\(.severity)] \(.attack_type) from \(.src_ip) - \(.timestamp)"' 2>/dev/null || echo "  (Unable to fetch incidents)"

echo ""
echo "View full dashboard at: https://$APPGW_IP"
echo ""

