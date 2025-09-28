#!/bin/bash

# Mini-XDR Quick Start Helper
# Shows available commands and provides quick access

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

# Banner
show_banner() {
    echo -e "${CYAN}"
    echo "=================================="
    echo "    Mini-XDR Quick Start"
    echo "=================================="
    echo -e "${NC}"
}

# Show deployment status
check_deployment_status() {
    echo -e "${BLUE}📊 Checking Deployment Status...${NC}"
    echo ""
    
    # Check if stacks exist
    local backend_exists=false
    local frontend_exists=false
    
    if aws cloudformation describe-stacks --stack-name mini-xdr-backend --region us-east-1 >/dev/null 2>&1; then
        backend_exists=true
        echo -e "${GREEN}✅ Backend: Deployed${NC}"
    else
        echo -e "${RED}❌ Backend: Not deployed${NC}"
    fi
    
    if aws cloudformation describe-stacks --stack-name mini-xdr-frontend --region us-east-1 >/dev/null 2>&1; then
        frontend_exists=true
        echo -e "${GREEN}✅ Frontend: Deployed${NC}"
    else
        echo -e "${RED}❌ Frontend: Not deployed${NC}"
    fi
    
    # Check TPOT connectivity
    if ssh -i ~/.ssh/mini-xdr-tpot-key.pem -p 64295 -o ConnectTimeout=5 -o StrictHostKeyChecking=yes -o UserKnownHostsFile=~/.ssh/known_hosts \
           admin@34.193.101.171 "echo test" >/dev/null 2>&1; then
        echo -e "${GREEN}✅ TPOT: Accessible${NC}"
    else
        echo -e "${RED}❌ TPOT: Not accessible${NC}"
    fi
    
    echo ""
    
    if [ "$backend_exists" = true ] && [ "$frontend_exists" = true ]; then
        echo -e "${GREEN}🎉 System is fully deployed!${NC}"
        return 0
    else
        echo -e "${YELLOW}⚠️  System needs deployment${NC}"
        return 1
    fi
}

# Show quick commands
show_commands() {
    echo -e "${BLUE}🚀 Available Commands:${NC}"
    echo ""
    
    if check_deployment_status >/dev/null 2>&1; then
        echo -e "${GREEN}System Management:${NC}"
        echo "  ./aws-services-control.sh status     - Check system status"
        echo "  ./aws-services-control.sh start      - Start all services"
        echo "  ./aws-services-control.sh stop       - Stop services (save money)"
        echo "  ./aws-services-control.sh logs       - View backend logs"
        echo ""
        echo -e "${GREEN}Updates:${NC}"
        echo "  ./update-pipeline.sh frontend        - Deploy frontend changes"
        echo "  ./update-pipeline.sh backend         - Deploy backend changes"
        echo "  ./update-pipeline.sh both            - Deploy all changes"
        echo "  ./update-pipeline.sh quick           - Quick frontend update"
        echo ""
        echo -e "${GREEN}TPOT Security:${NC}"
        echo "  ./tpot-security-control.sh status    - Check TPOT security mode"
        echo "  ./tpot-security-control.sh testing   - Safe testing mode"
        echo "  ./tpot-security-control.sh live      - ⚠️  Live mode (real attacks)"
        echo "  ./tpot-security-control.sh lockdown  - Emergency shutdown"
    else
        echo -e "${YELLOW}Deployment Commands:${NC}"
        echo "  ./deploy-complete-aws-system.sh      - Deploy complete system"
        echo "  ./deploy-mini-xdr-aws.sh             - Deploy backend only"
        echo "  ./deploy-frontend-aws.sh             - Deploy frontend only"
    fi
    
    echo ""
}

# Show usage
show_usage() {
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  status      - Show deployment status"
    echo "  commands    - Show available commands"
    echo "  deploy      - Start complete deployment"
    echo "  start       - Start services (if deployed)"
    echo "  stop        - Stop services"
    echo "  logs        - Show logs"
    echo "  update      - Update all services"
    echo "  help        - Show this help"
    echo ""
}

# Main function
main() {
    show_banner
    
    case "${1:-commands}" in
        status)
            check_deployment_status
            ;;
        commands)
            show_commands
            ;;
        deploy)
            if check_deployment_status >/dev/null 2>&1; then
                echo -e "${YELLOW}System already deployed. Use 'update' to deploy changes.${NC}"
                exit 1
            fi
            echo -e "${BLUE}Starting complete deployment...${NC}"
            ./deploy-complete-aws-system.sh
            ;;
        start)
            if ! check_deployment_status >/dev/null 2>&1; then
                echo -e "${RED}System not deployed. Use 'deploy' first.${NC}"
                exit 1
            fi
            ./aws-services-control.sh start
            ;;
        stop)
            if ! check_deployment_status >/dev/null 2>&1; then
                echo -e "${RED}System not deployed.${NC}"
                exit 1
            fi
            ./aws-services-control.sh stop
            ;;
        logs)
            if ! check_deployment_status >/dev/null 2>&1; then
                echo -e "${RED}System not deployed.${NC}"
                exit 1
            fi
            ./aws-services-control.sh logs
            ;;
        update)
            if ! check_deployment_status >/dev/null 2>&1; then
                echo -e "${RED}System not deployed. Use 'deploy' first.${NC}"
                exit 1
            fi
            ./update-pipeline.sh both
            ;;
        help)
            show_usage
            ;;
        *)
            show_commands
            ;;
    esac
}

# Run main function
main "$@"
