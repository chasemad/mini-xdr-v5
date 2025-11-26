#!/bin/bash
#
# 🎬 Mini-XDR Demo Attack - Simple Launcher
#
# This script runs attack demonstrations showcasing ML models and AI agents.
#
# Usage:
#   ./run-demo.sh                              # Full attack with delays
#   ./run-demo.sh --fast                       # Fast mode
#   ./run-demo.sh -t brute-force --fast        # SSH brute force attack
#   ./run-demo.sh -t web-attack                # Web application attack
#   ./run-demo.sh -t apt                       # Advanced Persistent Threat
#   ./run-demo.sh --list-types                 # Show all attack types
#
# Attack Types: full, brute-force, recon, apt, exfil, malware, web-attack
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo ""
echo -e "${CYAN}╔═══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║  ${GREEN}🎬 MINI-XDR COMPREHENSIVE ATTACK DEMONSTRATION${CYAN}              ║${NC}"
echo -e "${CYAN}╚═══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check if backend is running
echo -e "${YELLOW}Checking backend status...${NC}"
if ! curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo -e "${RED}❌ Backend is not running!${NC}"
    echo ""
    echo "Please start the servers first:"
    echo "  cd $PROJECT_ROOT"
    echo "  ./START_MINIXDR.sh"
    echo ""
    exit 1
fi
echo -e "${GREEN}✅ Backend is running${NC}"

# Check if frontend is running
echo -e "${YELLOW}Checking frontend status...${NC}"
if ! curl -s http://localhost:3000 > /dev/null 2>&1; then
    echo -e "${YELLOW}⚠️  Frontend might not be running (demo will still work)${NC}"
else
    echo -e "${GREEN}✅ Frontend is running${NC}"
fi

echo ""

# Run the Python demo script
cd "$PROJECT_ROOT"
python3 "$SCRIPT_DIR/run-full-demo-attack.py" "$@"
