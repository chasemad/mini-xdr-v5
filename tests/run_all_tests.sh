#!/bin/bash
# ========================================================================
# Run All Honeypot Tests - Quick Start Script
# ========================================================================

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
BOLD='\033[1m'
NC='\033[0m'

echo -e "${BOLD}${BLUE}"
cat << "EOF"
╔════════════════════════════════════════════════════════════════════════════╗
║                                                                            ║
║           MINI-XDR COMPREHENSIVE TESTING SUITE                            ║
║                                                                            ║
║  This script will:                                                         ║
║  1. Generate live attacks against your Azure honeypot                     ║
║  2. Wait for event processing                                             ║
║  3. Run comprehensive validation tests                                    ║
║  4. Generate detailed test report                                         ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}\n"

# Check prerequisites
echo -e "${YELLOW}Checking prerequisites...${NC}"

# Check if backend is running
if ! curl -s http://localhost:8000/health >/dev/null 2>&1; then
    echo -e "${RED}❌ Backend is not running!${NC}"
    echo -e "${YELLOW}Please start the backend first:${NC}"
    echo -e "  cd backend && source venv/bin/activate && uvicorn app.main:app --reload"
    exit 1
fi
echo -e "${GREEN}✅ Backend is running${NC}"

# Check Python dependencies
if ! python3 -c "import aiohttp" 2>/dev/null; then
    echo -e "${YELLOW}Installing Python dependencies...${NC}"
    pip3 install aiohttp
fi
echo -e "${GREEN}✅ Python dependencies OK${NC}"

# Check .env file
if [ ! -f "$(dirname "$0")/../backend/.env" ]; then
    echo -e "${RED}❌ .env file not found!${NC}"
    echo -e "${YELLOW}Please create backend/.env with your configuration${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Configuration file found${NC}"

echo ""

# Ask user which test mode
echo -e "${BOLD}${BLUE}Select Test Mode:${NC}"
echo -e "  ${YELLOW}1${NC} - Automated tests only (simulated attacks)"
echo -e "  ${YELLOW}2${NC} - Live honeypot attacks only"
echo -e "  ${YELLOW}3${NC} - Full suite (live attacks + validation) ${GREEN}[RECOMMENDED]${NC}"
echo -e "  ${YELLOW}4${NC} - Quick validation (check current state)"
echo ""
read -p "Enter choice [1-4]: " choice

case $choice in
    1)
        echo -e "\n${BLUE}Running automated tests with simulated attacks...${NC}\n"
        cd "$(dirname "$0")"
        python3 comprehensive_azure_honeypot_test.py
        ;;
    
    2)
        echo -e "\n${BLUE}Launching live honeypot attacks...${NC}\n"
        cd "$(dirname "$0")"
        bash live_honeypot_attack_suite.sh
        
        echo -e "\n${GREEN}${BOLD}Live attacks completed!${NC}"
        echo -e "${YELLOW}Wait 2-3 minutes for events to be processed, then run:${NC}"
        echo -e "  ${CYAN}python3 comprehensive_azure_honeypot_test.py${NC}"
        ;;
    
    3)
        echo -e "\n${BLUE}${BOLD}FULL TEST SUITE${NC}\n"
        cd "$(dirname "$0")"
        
        # Step 1: Live attacks
        echo -e "${YELLOW}[1/3] Launching live honeypot attacks...${NC}"
        bash live_honeypot_attack_suite.sh
        
        # Step 2: Wait for processing
        echo -e "\n${YELLOW}[2/3] Waiting for event processing...${NC}"
        for i in {180..1}; do
            printf "\r  ${CYAN}Time remaining: ${i} seconds...${NC}  "
            sleep 1
        done
        echo ""
        
        # Step 3: Validation
        echo -e "\n${YELLOW}[3/3] Running comprehensive validation tests...${NC}\n"
        python3 comprehensive_azure_honeypot_test.py
        
        echo -e "\n${GREEN}${BOLD}╔════════════════════════════════════════════════════╗${NC}"
        echo -e "${GREEN}${BOLD}║                                                    ║${NC}"
        echo -e "${GREEN}${BOLD}║        FULL TEST SUITE COMPLETED! 🎉               ║${NC}"
        echo -e "${GREEN}${BOLD}║                                                    ║${NC}"
        echo -e "${GREEN}${BOLD}╚════════════════════════════════════════════════════╝${NC}\n"
        ;;
    
    4)
        echo -e "\n${BLUE}Running quick validation...${NC}\n"
        cd "$(dirname "$0")"
        
        # Quick health checks
        echo -e "${YELLOW}System Health:${NC}"
        curl -s http://localhost:8000/health | jq '.'
        
        echo -e "\n${YELLOW}ML Model Status:${NC}"
        API_KEY=$(grep ^API_KEY ../backend/.env | cut -d'=' -f2)
        curl -s -H "x-api-key: $API_KEY" http://localhost:8000/api/ml/status | jq '.metrics'
        
        echo -e "\n${YELLOW}Incident Count:${NC}"
        INCIDENT_COUNT=$(curl -s http://localhost:8000/incidents | jq 'length')
        echo "  Total incidents: $INCIDENT_COUNT"
        
        echo -e "\n${YELLOW}Recent Incidents:${NC}"
        curl -s http://localhost:8000/incidents | jq '.[:3] | .[] | {id, threat_type, ml_confidence, severity}'
        
        echo -e "\n${GREEN}Quick validation complete!${NC}"
        echo -e "${YELLOW}For comprehensive testing, run option 3${NC}"
        ;;
    
    *)
        echo -e "${RED}Invalid choice${NC}"
        exit 1
        ;;
esac

echo ""
echo -e "${CYAN}📁 Test results saved in: $(dirname "$0")/test_results_*.json${NC}"
echo -e "${CYAN}📚 See TESTING_GUIDE.md for detailed documentation${NC}"
echo ""

