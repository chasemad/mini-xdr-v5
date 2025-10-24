#!/bin/bash
#
# Port Forward Solution for Demo
# Use this to access your Mini-XDR demo while Azure LoadBalancer issues are resolved
#

set -e

# Color codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}Mini-XDR Port Forward Demo Access${NC}"
echo -e "${BLUE}======================================${NC}\n"

echo -e "${YELLOW}This will forward port 3000 on your local machine to the Mini-XDR frontend.${NC}"
echo -e "${YELLOW}Leave this running and access your demo at: http://localhost:3000${NC}\n"

echo -e "${GREEN}Starting port forward...${NC}"
echo -e "${GREEN}Press Ctrl+C to stop${NC}\n"

kubectl port-forward -n mini-xdr svc/mini-xdr-frontend-service 3000:3000


