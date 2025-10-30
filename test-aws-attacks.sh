#!/bin/bash
# AWS Attack Simulation Test
set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
RED='\033[0;31m'
NC='\033[0m'

# Configuration
API_KEY="c5ca0b95c5977306f18f49afca26adb882896dc4ec25cf69f22ef77f44e908ab"
MINI_XDR_API="http://k8s-minixdr-minixdri-dc5fc1df8b-1132128475.us-east-1.elb.amazonaws.com/api"
TOTAL_ATTACKS=0
ATTACKS_DETECTED=0

echo -e "${BLUE}═══════════════════════════════════════════════════${NC}"
echo -e "${BLUE} Mini-XDR AWS Attack Simulation Test${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════${NC}\n"

# Generate unique IPs
ATTACKER_IPS=("203.0.113.50" "198.51.100.75" "192.0.2.100")

echo -e "${YELLOW}Testing ${#ATTACKER_IPS[@]} attack scenarios${NC}\n"

# Function to ingest events
ingest_attack() {
    local attack_name=$1
    local attacker_ip=$2
    local events_json=$3

    echo -e "${CYAN}[Attack $((TOTAL_ATTACKS + 1))]${NC} $attack_name from $attacker_ip"

    local payload="{\"source_type\":\"cowrie\",\"hostname\":\"aws-test-honeypot\",\"events\":$events_json}"

    response=$(curl -s -X POST "$MINI_XDR_API/ingest/multi" \
        -H "Content-Type: application/json" \
        -H "x-api-key: $API_KEY" \
        -d "$payload" 2>/dev/null || echo '{"error":"failed"}')

    if echo "$response" | grep -q "error"; then
        echo -e "  ${RED}❌ Ingestion failed${NC}"
    else
        processed=$(echo "$response" | jq -r '.processed // 0' 2>/dev/null || echo "0")
        echo -e "  Ingested: $processed events"
    fi

    ((TOTAL_ATTACKS++))

    sleep 3

    # Check for incident
    incidents=$(curl -s -H "x-api-key: $API_KEY" "$MINI_XDR_API/incidents?src_ip=$attacker_ip&limit=1" 2>/dev/null || echo '[]')
    count=$(echo "$incidents" | jq '. | length' 2>/dev/null || echo "0")

    if [ "$count" -gt 0 ]; then
        echo -e "  ${GREEN}✅ Incident detected!${NC}"
        ((ATTACKS_DETECTED++))
    else
        echo -e "  ${YELLOW}⚠️  No incident created yet${NC}"
    fi

    echo ""
}

# Get baseline
echo -e "${BLUE}Baseline: Checking current incidents${NC}"
INCIDENTS_BEFORE=$(curl -s -H "x-api-key: $API_KEY" "$MINI_XDR_API/incidents" | jq 'length' || echo "0")
echo -e "Current incidents: $INCIDENTS_BEFORE\n"

# Attack 1: SSH Brute Force
echo -e "${YELLOW}═══ Test 1: SSH Brute Force (15 attempts) ═══${NC}"
EVENTS='['
for i in {1..15}; do
    [ $i -gt 1 ] && EVENTS+=','
    EVENTS+="{\"eventid\":\"cowrie.login.failed\",\"src_ip\":\"${ATTACKER_IPS[0]}\",\"dst_port\":22,\"message\":\"Failed SSH login\",\"timestamp\":\"$(date -u +%Y-%m-%dT%H:%M:%S.000Z)\",\"raw\":{\"username\":\"admin\",\"password\":\"pass$i\"}}"
done
EVENTS+=']'
ingest_attack "SSH Brute Force" "${ATTACKER_IPS[0]}" "$EVENTS"

# Attack 2: Port Scan
echo -e "${YELLOW}═══ Test 2: Port Scan (10 ports) ═══${NC}"
EVENTS='['
for port in 22 80 443 3306 5432 8080 8443 9200 3389 445; do
    [ "$port" != 22 ] && EVENTS+=','
    EVENTS+="{\"eventid\":\"connection.scan\",\"src_ip\":\"${ATTACKER_IPS[1]}\",\"dst_port\":$port,\"message\":\"Port scan detected\",\"timestamp\":\"$(date -u +%Y-%m-%dT%H:%M:%S.000Z)\",\"raw\":{\"scan_type\":\"syn\"}}"
done
EVENTS+=']'
ingest_attack "Port Scan" "${ATTACKER_IPS[1]}" "$EVENTS"

# Attack 3: Malware Download
echo -e "${YELLOW}═══ Test 3: Malware Activity ═══${NC}"
EVENTS='[
{"eventid":"cowrie.session.file_download","src_ip":"'${ATTACKER_IPS[2]}'","dst_port":22,"message":"Malware download","timestamp":"'$(date -u +%Y-%m-%dT%H:%M:%S.000Z)'","raw":{"url":"http://malware.com/trojan.exe","test_type":"malware"}},
{"eventid":"cowrie.command.input","src_ip":"'${ATTACKER_IPS[2]}'","dst_port":22,"message":"Malware execution","timestamp":"'$(date -u +%Y-%m-%dT%H:%M:%S.000Z)'","raw":{"input":"chmod +x malware && ./malware","test_type":"malware"}}
]'
ingest_attack "Malware Activity" "${ATTACKER_IPS[2]}" "$EVENTS"

# Final check
echo -e "${BLUE}═══ Waiting 5 seconds for detection pipeline ═══${NC}"
sleep 5

echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════${NC}"
echo -e "${BLUE} Test Results${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════${NC}\n"

INCIDENTS_AFTER=$(curl -s -H "x-api-key: $API_KEY" "$MINI_XDR_API/incidents" | jq 'length' || echo "0")
NEW_INCIDENTS=$((INCIDENTS_AFTER - INCIDENTS_BEFORE))

echo "Incidents before: $INCIDENTS_BEFORE"
echo "Incidents after:  $INCIDENTS_AFTER"
echo "New incidents:    $NEW_INCIDENTS"
echo "Total attacks:    $TOTAL_ATTACKS"
echo "Detected:         $ATTACKS_DETECTED"

if [ $NEW_INCIDENTS -gt 0 ]; then
    RATE=$(awk "BEGIN {printf \"%.1f\", ($NEW_INCIDENTS/$TOTAL_ATTACKS)*100}")
    echo -e "\nDetection Rate: ${GREEN}${RATE}%${NC}"

    echo -e "\n${YELLOW}Latest Incidents:${NC}"
    curl -s -H "x-api-key: $API_KEY" "$MINI_XDR_API/incidents" | jq -r '.[-5:] | .[] | "  🚨 \(.source_ip) - \(.reason)"' | tail -5

    if [ $NEW_INCIDENTS -ge 2 ]; then
        echo -e "\n${GREEN}✅ SUCCESS: Detection pipeline working!${NC}"
    else
        echo -e "\n${YELLOW}⚠️  PARTIAL: Some attacks detected${NC}"
    fi
else
    echo -e "\n${RED}❌ FAILURE: No new incidents detected${NC}"
    echo -e "${YELLOW}Possible issues:${NC}"
    echo "  - ML models need training"
    echo "  - Detection thresholds too high"
    echo "  - Pipeline not processing events"
fi

echo ""
