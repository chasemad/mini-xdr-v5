#!/bin/bash
# ============================================================================
# Data Exfiltration Attack Simulations
# ============================================================================
# Simulates various data exfiltration techniques for testing detection
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
echo -e "${BLUE}║       Data Exfiltration Attack Simulations                     ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Get configuration
if [ -f "$TERRAFORM_DIR/terraform.tfstate" ]; then
    KEY_VAULT_NAME=$(terraform -chdir="$TERRAFORM_DIR" output -raw key_vault_name)
    APPGW_IP=$(terraform -chdir="$TERRAFORM_DIR" output -raw appgw_public_ip)
    API_KEY=$(az keyvault secret show --vault-name "$KEY_VAULT_NAME" --name "mini-xdr-api-key" --query value -o tsv)
else
    echo -e "${RED}❌ Terraform state not found${NC}"
    exit 1
fi

BACKEND_URL="https://$APPGW_IP"

echo "Target Configuration:"
echo "  • Backend URL: $BACKEND_URL"
echo ""

# ============================================================================
# Attack 1: Large File Transfer
# ============================================================================

echo -e "${YELLOW}[1/4] Simulating Large File Transfer...${NC}"

curl -s -X POST \
    -H "Content-Type: application/json" \
    -H "X-API-Key: $API_KEY" \
    "$BACKEND_URL/ingest/json" \
    -d "{
        \"event_type\": \"data_exfiltration\",
        \"attack_technique\": \"large_file_transfer\",
        \"src_ip\": \"10.0.10.30\",
        \"dest_ip\": \"8.8.8.8\",
        \"src_port\": 50000,
        \"dest_port\": 443,
        \"protocol\": \"https\",
        \"severity\": \"high\",
        \"description\": \"Large file upload to external server\",
        \"mitre_technique\": \"T1048.003\",
        \"bytes_transferred\": 5368709120,
        \"file_name\": \"customer_database_export.zip\",
        \"duration_seconds\": 180,
        \"timestamp\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"
    }" > /dev/null

echo -e "  ${GREEN}✓ Large file transfer simulation sent${NC}"
sleep 2

# ============================================================================
# Attack 2: Cloud Storage Upload
# ============================================================================

echo -e "${YELLOW}[2/4] Simulating Cloud Storage Upload...${NC}"

curl -s -X POST \
    -H "Content-Type: application/json" \
    -H "X-API-Key: $API_KEY" \
    "$BACKEND_URL/ingest/json" \
    -d "{
        \"event_type\": \"data_exfiltration\",
        \"attack_technique\": \"cloud_storage\",
        \"src_ip\": \"10.0.10.21\",
        \"dest_ip\": \"storage.googleapis.com\",
        \"src_port\": 50100,
        \"dest_port\": 443,
        \"protocol\": \"https\",
        \"severity\": \"high\",
        \"description\": \"Upload to public cloud storage\",
        \"mitre_technique\": \"T1567.002\",
        \"cloud_provider\": \"GCP\",
        \"bytes_transferred\": 2147483648,
        \"files_count\": 847,
        \"timestamp\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"
    }" > /dev/null

echo -e "  ${GREEN}✓ Cloud storage upload simulation sent${NC}"
sleep 2

# ============================================================================
# Attack 3: DNS Tunneling
# ============================================================================

echo -e "${YELLOW}[3/4] Simulating DNS Tunneling...${NC}"

# Simulate multiple DNS queries with data
for i in {1..10}; do
    curl -s -X POST \
        -H "Content-Type: application/json" \
        -H "X-API-Key: $API_KEY" \
        "$BACKEND_URL/ingest/json" \
        -d "{
            \"event_type\": \"data_exfiltration\",
            \"attack_technique\": \"dns_tunneling\",
            \"src_ip\": \"10.0.10.22\",
            \"dest_ip\": \"8.8.8.8\",
            \"src_port\": 50200,
            \"dest_port\": 53,
            \"protocol\": \"dns\",
            \"severity\": \"medium\",
            \"description\": \"Suspicious DNS query patterns (data tunneling)\",
            \"mitre_technique\": \"T1048.001\",
            \"query\": \"$(openssl rand -hex 32).exfil.attacker.com\",
            \"query_type\": \"TXT\",
            \"timestamp\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"
        }" > /dev/null
    
    sleep 0.5
done

echo -e "  ${GREEN}✓ DNS tunneling simulation sent (10 queries)${NC}"
sleep 2

# ============================================================================
# Attack 4: Database Dump Exfiltration
# ============================================================================

echo -e "${YELLOW}[4/4] Simulating Database Dump Exfiltration...${NC}"

curl -s -X POST \
    -H "Content-Type: application/json" \
    -H "X-API-Key: $API_KEY" \
    "$BACKEND_URL/ingest/json" \
    -d "{
        \"event_type\": \"data_exfiltration\",
        \"attack_technique\": \"database_dump\",
        \"src_ip\": \"10.0.10.40\",
        \"dest_ip\": \"185.220.101.1\",
        \"src_port\": 50300,
        \"dest_port\": 443,
        \"protocol\": \"https\",
        \"severity\": \"critical\",
        \"description\": \"Database dump uploaded to external server\",
        \"mitre_technique\": \"T1048.003\",
        \"database_type\": \"postgresql\",
        \"database_name\": \"customer_data\",
        \"tables_dumped\": 47,
        \"records_exfiltrated\": 1250000,
        \"bytes_transferred\": 8589934592,
        \"timestamp\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"
    }" > /dev/null

echo -e "  ${GREEN}✓ Database dump simulation sent${NC}"

# ============================================================================
# Check Detections
# ============================================================================

echo ""
echo -e "${YELLOW}Checking for detections (waiting 10 seconds)...${NC}"
sleep 10

DETECTIONS=$(curl -s -H "X-API-Key: $API_KEY" "$BACKEND_URL/api/incidents?limit=20" | \
    jq '[.[] | select(.attack_type | contains("Exfiltration"))] | length' 2>/dev/null || echo "0")

echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║        Data Exfiltration Simulation Complete                  ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo "Results:"
echo "  • Attacks Simulated: 4"
echo "  • Large File Transfer: 5 GB upload"
echo "  • Cloud Storage: 2 GB to GCP"
echo "  • DNS Tunneling: 10 queries"
echo "  • Database Dump: 8 GB (1.25M records)"
echo ""
echo "Detection Status:"
echo "  • Data Exfiltration Incidents: $DETECTIONS"
echo "  • Expected Detection Rate: 97.7%"
echo ""
echo "Expected Agent Response:"
echo "  • DLP Agent: Block unauthorized uploads"
echo "  • DLP Agent: Quarantine sensitive files"
echo "  • EDR Agent: Kill exfiltration process"
echo "  • Network: Block destination IPs"
echo ""

