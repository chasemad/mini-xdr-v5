#!/bin/bash

# ============================================================================
# Complete T-Pot + Mini-XDR SSH Demo Setup
# ============================================================================
# This master script sets up everything needed for the SSH brute force demo
# where AI agents automatically defend your T-Pot honeypot
#
# What it does:
#   1. Configure T-Pot SSH connection
#   2. Test connectivity and defensive actions
#   3. Set up workflows
#   4. Start Mini-XDR backend
#   5. Provide demo commands
#
# Usage: ./SETUP_TPOT_DEMO.sh
# ============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
BOLD='\033[1m'
NC='\033[0m'

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BACKEND_DIR="$PROJECT_ROOT/backend"
FRONTEND_DIR="$PROJECT_ROOT/frontend"

# Functions
print_banner() {
    echo ""
    echo -e "${CYAN}${BOLD}╔═══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}${BOLD}║                                                               ║${NC}"
    echo -e "${CYAN}${BOLD}║          Mini-XDR + T-Pot SSH Demo Setup Wizard              ║${NC}"
    echo -e "${CYAN}${BOLD}║                                                               ║${NC}"
    echo -e "${CYAN}${BOLD}║      AI Agents Defending Your Honeypot via SSH 🛡️            ║${NC}"
    echo -e "${CYAN}${BOLD}║                                                               ║${NC}"
    echo -e "${CYAN}${BOLD}╚═══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

print_header() {
    echo ""
    echo -e "${MAGENTA}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${MAGENTA}${BOLD}  $1${NC}"
    echo -e "${MAGENTA}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
}

print_step() {
    echo -e "${BLUE}${BOLD}▶ Step $1:${NC} $2"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_info() {
    echo -e "${CYAN}ℹ${NC} $1"
}

print_command() {
    echo -e "${YELLOW}$ ${NC}$1"
}

# Check prerequisites
check_prerequisites() {
    print_header "Checking Prerequisites"

    local all_good=true

    # Check Python
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version)
        print_success "Python 3 installed: $PYTHON_VERSION"
    else
        print_error "Python 3 not found"
        all_good=false
    fi

    # Check Node.js
    if command -v node &> /dev/null; then
        NODE_VERSION=$(node --version)
        print_success "Node.js installed: $NODE_VERSION"
    else
        print_warning "Node.js not found (optional, needed for frontend)"
    fi

    # Check SSH
    if command -v ssh &> /dev/null; then
        print_success "SSH client installed"
    else
        print_error "SSH client not found"
        all_good=false
    fi

    # Check for sshpass (optional but recommended)
    if command -v sshpass &> /dev/null; then
        print_success "sshpass installed (recommended)"
    else
        print_warning "sshpass not installed (recommended for password auth)"
        print_info "Install with: brew install hudochenkov/sshpass/sshpass"
    fi

    # Check backend directory
    if [ -d "$BACKEND_DIR" ]; then
        print_success "Backend directory found"
    else
        print_error "Backend directory not found: $BACKEND_DIR"
        all_good=false
    fi

    if [ "$all_good" = false ]; then
        echo ""
        print_error "Prerequisites check failed. Please install missing components."
        exit 1
    fi

    echo ""
    print_success "All prerequisites met!"
}

# Step 1: Configure T-Pot SSH
configure_tpot_ssh() {
    print_header "Step 1: Configure T-Pot SSH Connection"

    echo "This will configure Mini-XDR to connect to your T-Pot honeypot via SSH"
    echo "so AI agents can automatically block attackers and manage honeypots."
    echo ""

    read -p "Press Enter to start configuration, or Ctrl+C to skip..."

    SETUP_SCRIPT="$SCRIPT_DIR/tpot-management/setup-tpot-ssh-integration.sh"

    if [ -f "$SETUP_SCRIPT" ]; then
        print_command "$SETUP_SCRIPT"
        "$SETUP_SCRIPT"

        if [ $? -eq 0 ]; then
            print_success "T-Pot SSH configuration complete"
            return 0
        else
            print_error "T-Pot SSH configuration failed"
            return 1
        fi
    else
        print_error "Setup script not found: $SETUP_SCRIPT"
        return 1
    fi
}

# Step 2: Verify configuration
verify_configuration() {
    print_header "Step 2: Verify T-Pot Configuration"

    echo "Testing T-Pot SSH connection and defensive capabilities..."
    echo ""

    # Check if backend is running
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        print_info "Backend is running, proceeding with verification..."

        VERIFY_SCRIPT="$SCRIPT_DIR/tpot-management/verify-agent-ssh-actions.sh"

        if [ -f "$VERIFY_SCRIPT" ]; then
            print_command "$VERIFY_SCRIPT"
            "$VERIFY_SCRIPT"

            if [ $? -eq 0 ]; then
                print_success "Verification passed!"
                return 0
            else
                print_warning "Some verification tests failed (this may be OK)"
                read -p "Continue anyway? [Y/n]: " CONTINUE
                if [ "$CONTINUE" = "n" ]; then
                    return 1
                fi
            fi
        else
            print_warning "Verification script not found, skipping..."
        fi
    else
        print_warning "Backend not running, skipping verification"
        print_info "You can verify later after starting the backend"
    fi

    return 0
}

# Step 3: Set up workflows
setup_workflows() {
    print_header "Step 3: Set Up SSH Brute Force Workflow"

    echo "Installing automated response workflow for SSH attacks..."
    echo ""

    # Check if backend is running
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        WORKFLOW_SCRIPT="$SCRIPT_DIR/tpot-management/setup-tpot-workflows.py"

        if [ -f "$WORKFLOW_SCRIPT" ]; then
            print_command "python3 $WORKFLOW_SCRIPT"
            cd "$BACKEND_DIR"
            python3 "$WORKFLOW_SCRIPT"

            if [ $? -eq 0 ]; then
                print_success "Workflows configured!"
                return 0
            else
                print_warning "Workflow setup had issues (may already exist)"
            fi
        else
            print_warning "Workflow script not found, skipping..."
        fi
    else
        print_warning "Backend not running, skipping workflow setup"
        print_info "You can set up workflows later with:"
        print_command "cd $BACKEND_DIR && python3 $WORKFLOW_SCRIPT"
    fi

    return 0
}

# Step 4: Start backend
start_backend() {
    print_header "Step 4: Start Mini-XDR Backend"

    # Check if already running
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        print_success "Backend is already running!"
        return 0
    fi

    echo "Starting backend server..."
    echo ""

    read -p "Start backend in new terminal? [Y/n]: " START_BACKEND

    if [ "$START_BACKEND" != "n" ]; then
        print_info "Opening new terminal for backend..."

        # macOS terminal
        if [[ "$OSTYPE" == "darwin"* ]]; then
            osascript -e "tell application \"Terminal\" to do script \"cd $BACKEND_DIR && python -m uvicorn app.main:app --reload\""
            print_success "Backend starting in new terminal window"
        else
            print_warning "Please start backend manually in another terminal:"
            print_command "cd $BACKEND_DIR && python -m uvicorn app.main:app --reload"
        fi

        # Wait for backend to start
        print_info "Waiting for backend to start..."
        for i in {1..30}; do
            if curl -s http://localhost:8000/health > /dev/null 2>&1; then
                print_success "Backend is running!"
                return 0
            fi
            sleep 1
            echo -n "."
        done
        echo ""
        print_warning "Backend may still be starting..."
    else
        print_info "Please start backend manually:"
        print_command "cd $BACKEND_DIR && python -m uvicorn app.main:app --reload"
    fi

    return 0
}

# Step 5: Demo commands
show_demo_commands() {
    print_header "🎯 Demo Ready!"

    echo ""
    echo -e "${GREEN}${BOLD}Your Mini-XDR + T-Pot SSH demo is configured!${NC}"
    echo ""
    echo -e "${CYAN}${BOLD}Access Points:${NC}"
    echo ""

    # Read T-Pot config from .env
    if [ -f "$BACKEND_DIR/.env" ]; then
        TPOT_IP=$(grep "^TPOT_HOST=" "$BACKEND_DIR/.env" | cut -d'=' -f2 || echo "203.0.113.42")
        TPOT_WEB_PORT=$(grep "^TPOT_WEB_PORT=" "$BACKEND_DIR/.env" | cut -d'=' -f2 || echo "64297")
        TPOT_SSH_PORT=$(grep "^TPOT_SSH_PORT=" "$BACKEND_DIR/.env" | cut -d'=' -f2 || echo "64295")
    else
        TPOT_IP="203.0.113.42"
        TPOT_WEB_PORT="64297"
        TPOT_SSH_PORT="64295"
    fi

    echo "  📊 Mini-XDR Dashboard:  http://localhost:3000"
    echo "  🔧 Backend API:         http://localhost:8000"
    echo "  🍯 T-Pot Web:          http://${TPOT_IP}:${TPOT_WEB_PORT}"
    echo "  🔌 T-Pot SSH:          ssh -p ${TPOT_SSH_PORT} admin@${TPOT_IP}"
    echo ""

    echo -e "${CYAN}${BOLD}Demo Attack Commands:${NC}"
    echo ""
    echo "  ${YELLOW}Option 1: Automated Demo Attack${NC}"
    print_command "$SCRIPT_DIR/demo/demo-attack.sh"
    echo ""

    echo "  ${YELLOW}Option 2: Manual SSH Brute Force${NC}"
    echo "  Generate SSH brute force from another machine:"
    print_command "for i in {1..10}; do ssh -p ${TPOT_SSH_PORT} admin@${TPOT_IP} 'wrong_\$i' 2>/dev/null; done"
    echo ""

    echo "  ${YELLOW}Option 3: Wait for Real Attacks${NC}"
    echo "  Just wait! The internet is constantly scanning."
    echo "  Watch attacks appear in T-Pot and Mini-XDR."
    echo ""

    echo -e "${CYAN}${BOLD}What to Watch:${NC}"
    echo ""
    echo "  1. 🎯 ${BOLD}T-Pot Web Interface${NC} - Live attack map"
    echo "     ${YELLOW}http://${TPOT_IP}:${TPOT_WEB_PORT}${NC}"
    echo ""
    echo "  2. 🛡️  ${BOLD}Mini-XDR Dashboard${NC} - Incidents and AI responses"
    echo "     ${YELLOW}http://localhost:3000${NC}"
    echo ""
    echo "  3. 📋 ${BOLD}Backend Logs${NC} - Real-time defensive actions"
    print_command "tail -f $BACKEND_DIR/backend.log | grep -i 'block\\|ssh\\|brute'"
    echo ""

    echo -e "${CYAN}${BOLD}Expected Behavior:${NC}"
    echo ""
    echo "  ✅ T-Pot detects SSH brute force (Cowrie honeypot)"
    echo "  ✅ Mini-XDR ingests attack events in real-time"
    echo "  ✅ After 6 failed attempts in 60s, incident created"
    echo "  ✅ AI agents automatically:"
    echo "      • Analyze the threat"
    echo "      • SSH into T-Pot"
    echo "      • Block attacker IP via UFW"
    echo "      • Log all actions"
    echo "      • Send notifications"
    echo ""

    echo -e "${CYAN}${BOLD}Verify Actions:${NC}"
    echo ""
    echo "  Check blocked IPs on T-Pot:"
    print_command "ssh -p ${TPOT_SSH_PORT} admin@${TPOT_IP} 'sudo ufw status | grep DENY'"
    echo ""

    echo "  Check Mini-XDR incidents:"
    print_command "curl http://localhost:8000/api/incidents | jq '.incidents[] | {id, severity, src_ip}'"
    echo ""

    echo -e "${CYAN}${BOLD}Troubleshooting:${NC}"
    echo ""
    echo "  • ${BOLD}Can't connect to T-Pot SSH:${NC}"
    echo "    Check firewall allows your IP: sudo ufw status"
    echo ""
    echo "  • ${BOLD}No incidents created:${NC}"
    echo "    Check backend logs: tail -f $BACKEND_DIR/backend.log"
    echo ""
    echo "  • ${BOLD}Agents can't block IPs:${NC}"
    echo "    Verify TPOT_API_KEY set in backend/.env"
    echo ""

    echo -e "${GREEN}${BOLD}Documentation:${NC}"
    echo "  📖 Full setup guide: $PROJECT_ROOT/TPOT_SSH_SETUP_GUIDE.md"
    echo ""

    echo -e "${MAGENTA}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}${BOLD}Ready to demonstrate AI-powered defense! 🚀${NC}"
    echo -e "${MAGENTA}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
}

# Main execution
main() {
    print_banner

    echo "This wizard will guide you through setting up the complete"
    echo "Mini-XDR + T-Pot SSH demo where AI agents automatically defend"
    echo "your honeypot by blocking attackers in real-time."
    echo ""
    echo "Prerequisites:"
    echo "  ✓ T-Pot running at 203.0.113.42 (or your IP)"
    echo "  ✓ Firewall allows 172.16.110.0/24 subnet"
    echo "  ✓ SSH access to T-Pot (port 64295)"
    echo "  ✓ T-Pot admin credentials"
    echo ""

    read -p "Press Enter to begin, or Ctrl+C to exit..."

    # Step 0: Prerequisites check
    check_prerequisites

    # Step 1: Configure T-Pot SSH
    if ! configure_tpot_ssh; then
        print_error "Setup failed at SSH configuration"
        exit 1
    fi

    # Step 2: Verify configuration
    verify_configuration

    # Step 3: Set up workflows
    setup_workflows

    # Step 4: Start backend
    start_backend

    # Step 5: Show demo commands
    show_demo_commands
}

# Run main function
main "$@"
