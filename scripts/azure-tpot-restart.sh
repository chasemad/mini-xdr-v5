#!/bin/bash
# ========================================================================
# Azure TPOT Honeypot - RESTART Script
# Restarts the TPOT VM and services
# ========================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║           Azure TPOT Honeypot - RESTART                        ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

echo -e "${YELLOW}This will restart the TPOT VM and all services${NC}"
echo -e "${BLUE}Estimated time: 2-3 minutes${NC}"
echo ""

read -p "Continue? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}Restart cancelled${NC}"
    exit 0
fi

echo ""

# Stop TPOT
echo -e "${YELLOW}[1/2] Stopping TPOT...${NC}"
"$SCRIPT_DIR/azure-tpot-stop.sh"

echo ""
echo -e "${BLUE}Waiting 10 seconds before starting...${NC}"
sleep 10
echo ""

# Start TPOT
echo -e "${YELLOW}[2/2] Starting TPOT...${NC}"
"$SCRIPT_DIR/azure-tpot-start.sh"

echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                   RESTART COMPLETE                              ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${GREEN}✨ TPOT has been restarted!${NC}"

