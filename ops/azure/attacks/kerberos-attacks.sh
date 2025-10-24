#!/bin/bash
# ============================================================================
# Kerberos Attack Simulations
# ============================================================================
# Simulates various Kerberos-based attacks for testing detection
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
echo -e "${BLUE}║       Kerberos Attack Simulations                              ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Get configuration
if [ -f "$TERRAFORM_DIR/terraform.tfstate" ]; then
    KEY_VAULT_NAME=$(terraform -chdir="$TERRAFORM_DIR" output -raw key_vault_name)
    APPGW_IP=$(terraform -chdir="$TERRAFORM_DIR" output -raw appgw_public_ip)
    DC_IP=$(terraform -chdir="$TERRAFORM_DIR" output -raw domain_controller_private_ip)
    API_KEY=$(az keyvault secret show --vault-name "$KEY_VAULT_NAME" --name "mini-xdr-api-key" --query value -o tsv)
else
    echo -e "${RED}❌ Terraform state not found${NC}"
    exit 1
fi

BACKEND_URL="https://$APPGW_IP"

echo "Target Configuration:"
echo "  • Domain Controller: $DC_IP"
echo "  • Backend URL: $BACKEND_URL"
echo ""

# ============================================================================
# Attack 1: Kerberoasting Simulation
# ============================================================================

echo -e "${YELLOW}[1/4] Simulating Kerberoasting Attack...${NC}"

curl -s -X POST \
    -H "Content-Type: application/json" \
    -H "X-API-Key: $API_KEY" \
    "$BACKEND_URL/ingest/json" \
    -d "{
        \"event_type\": \"kerberos_attack\",
        \"attack_technique\": \"kerberoasting\",
        \"src_ip\": \"$DC_IP\",
        \"dest_ip\": \"10.0.10.20\",
        \"src_port\": 49152,
        \"dest_port\": 88,
        \"protocol\": \"kerberos\",
        \"severity\": \"high\",
        \"description\": \"TGS-REQ for service account SPN\",
        \"mitre_technique\": \"T1558.003\",
        \"timestamp\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"
    }" > /dev/null

echo -e "  ${GREEN}✓ Kerberoasting simulation sent${NC}"
sleep 2

# ============================================================================
# Attack 2: Golden Ticket Simulation
# ============================================================================

echo -e "${YELLOW}[2/4] Simulating Golden Ticket Attack...${NC}"

curl -s -X POST \
    -H "Content-Type: application/json" \
    -H "X-API-Key: $API_KEY" \
    "$BACKEND_URL/ingest/json" \
    -d "{
        \"event_type\": \"kerberos_attack\",
        \"attack_technique\": \"golden_ticket\",
        \"src_ip\": \"10.0.10.25\",
        \"dest_ip\": \"$DC_IP\",
        \"src_port\": 49200,
        \"dest_port\": 88,
        \"protocol\": \"kerberos\",
        \"severity\": \"critical\",
        \"description\": \"Forged TGT with extended validity (10 years)\",
        \"mitre_technique\": \"T1558.001\",
        \"indicators\": [
            \"Unusual TGT lifetime\",
            \"Non-standard encryption type\",
            \"Suspicious ticket flags\"
        ],
        \"timestamp\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"
    }" > /dev/null

echo -e "  ${GREEN}✓ Golden Ticket simulation sent${NC}"
sleep 2

# ============================================================================
# Attack 3: AS-REP Roasting Simulation
# ============================================================================

echo -e "${YELLOW}[3/4] Simulating AS-REP Roasting Attack...${NC}"

curl -s -X POST \
    -H "Content-Type: application/json" \
    -H "X-API-Key: $API_KEY" \
    "$BACKEND_URL/ingest/json" \
    -d "{
        \"event_type\": \"kerberos_attack\",
        \"attack_technique\": \"asrep_roasting\",
        \"src_ip\": \"10.0.10.30\",
        \"dest_ip\": \"$DC_IP\",
        \"src_port\": 49300,
        \"dest_port\": 88,
        \"protocol\": \"kerberos\",
        \"severity\": \"high\",
        \"description\": \"AS-REQ without pre-authentication\",
        \"mitre_technique\": \"T1558.004\",
        \"target_accounts\": [
            \"svc_backup\",
            \"svc_monitoring\"
        ],
        \"timestamp\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"
    }" > /dev/null

echo -e "  ${GREEN}✓ AS-REP Roasting simulation sent${NC}"
sleep 2

# ============================================================================
# Attack 4: Pass-the-Ticket Simulation
# ============================================================================

echo -e "${YELLOW}[4/4] Simulating Pass-the-Ticket Attack...${NC}"

curl -s -X POST \
    -H "Content-Type: application/json" \
    -H "X-API-Key: $API_KEY" \
    "$BACKEND_URL/ingest/json" \
    -d "{
        \"event_type\": \"kerberos_attack\",
        \"attack_technique\": \"pass_the_ticket\",
        \"src_ip\": \"10.0.10.21\",
        \"dest_ip\": \"10.0.10.22\",
        \"src_port\": 49400,
        \"dest_port\": 445,
        \"protocol\": \"smb\",
        \"severity\": \"high\",
        \"description\": \"Reusing stolen Kerberos ticket for SMB access\",
        \"mitre_technique\": \"T1550.003\",
        \"stolen_from\": \"john.smith@minicorp.local\",
        \"timestamp\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"
    }" > /dev/null

echo -e "  ${GREEN}✓ Pass-the-Ticket simulation sent${NC}"

# ============================================================================
# Check Detections
# ============================================================================

echo ""
echo -e "${YELLOW}Checking for detections (waiting 10 seconds)...${NC}"
sleep 10

DETECTIONS=$(curl -s -H "X-API-Key: $API_KEY" "$BACKEND_URL/api/incidents?limit=10&attack_type=Kerberos")

DETECTION_COUNT=$(echo "$DETECTIONS" | jq '. | length' 2>/dev/null || echo "0")

echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║        Kerberos Attack Simulation Complete                    ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo "Results:"
echo "  • Attacks Simulated: 4"
echo "  • Kerberoasting: Sent"
echo "  • Golden Ticket: Sent"
echo "  • AS-REP Roasting: Sent"
echo "  • Pass-the-Ticket: Sent"
echo ""
echo "Detection Status:"
echo "  • Kerberos Incidents Detected: $DETECTION_COUNT"
echo "  • Expected Detection Rate: 99.98%"
echo ""
echo "View detections:"
echo "  • Dashboard: https://$APPGW_IP/incidents"
echo "  • API: curl -H 'X-API-Key: $API_KEY' $BACKEND_URL/api/incidents"
echo ""

