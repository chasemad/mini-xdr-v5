#!/bin/bash
#
# Recreate LoadBalancer Service (Fallback Solution)
# Use this if NGINX Ingress doesn't work or you prefer direct LoadBalancer
#

set -e

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}Recreating LoadBalancer Service${NC}"
echo -e "${BLUE}======================================${NC}\n"

echo -e "${YELLOW}[Step 1] Backing up current LoadBalancer service...${NC}"
kubectl get svc mini-xdr-loadbalancer -n mini-xdr -o yaml > /tmp/mini-xdr-lb-backup-$(date +%Y%m%d_%H%M%S).yaml
echo -e "${GREEN}✓ Backup saved to /tmp/mini-xdr-lb-backup-$(date +%Y%m%d_%H%M%S).yaml${NC}\n"

echo -e "${YELLOW}[Step 2] Getting current external IP...${NC}"
CURRENT_IP=$(kubectl get svc mini-xdr-loadbalancer -n mini-xdr -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
echo "Current IP: $CURRENT_IP"
echo ""

echo -e "${YELLOW}[Step 3] Deleting existing LoadBalancer service...${NC}"
kubectl delete svc mini-xdr-loadbalancer -n mini-xdr
echo -e "${GREEN}✓ Service deleted${NC}\n"

echo "Waiting 10 seconds for Azure to clean up resources..."
sleep 10

echo -e "${YELLOW}[Step 4] Creating new LoadBalancer service with proper configuration...${NC}"

cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Service
metadata:
  name: mini-xdr-loadbalancer
  namespace: mini-xdr
  annotations:
    service.beta.kubernetes.io/azure-load-balancer-health-probe-request-path: /healthz
    service.beta.kubernetes.io/azure-load-balancer-health-probe-protocol: http
    service.beta.kubernetes.io/azure-load-balancer-health-probe-interval: "5"
    service.beta.kubernetes.io/azure-load-balancer-health-probe-num-of-probe: "2"
spec:
  type: LoadBalancer
  externalTrafficPolicy: Cluster  # Changed from Local to Cluster for better compatibility
  sessionAffinity: None
  selector:
    app: mini-xdr-frontend
  ports:
  - name: http
    protocol: TCP
    port: 80
    targetPort: 3000
  - name: https
    protocol: TCP
    port: 443
    targetPort: 3000
EOF

echo -e "${GREEN}✓ New LoadBalancer service created${NC}\n"

echo -e "${YELLOW}[Step 5] Waiting for new external IP assignment (2-5 minutes)...${NC}"

# Wait for external IP
TIMEOUT=600
ELAPSED=0
INTERVAL=5

while [ $ELAPSED -lt $TIMEOUT ]; do
    NEW_IP=$(kubectl get svc mini-xdr-loadbalancer -n mini-xdr -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "")
    
    if [ -n "$NEW_IP" ] && [ "$NEW_IP" != "null" ]; then
        echo -e "\n${GREEN}✓ New external IP assigned: $NEW_IP${NC}\n"
        break
    fi
    
    echo -n "."
    sleep $INTERVAL
    ELAPSED=$((ELAPSED + INTERVAL))
done

if [ -z "$NEW_IP" ] || [ "$NEW_IP" == "null" ]; then
    echo -e "\n${RED}✗ Timeout waiting for external IP${NC}"
    echo "Check service status:"
    echo "  kubectl get svc mini-xdr-loadbalancer -n mini-xdr"
    exit 1
fi

echo -e "${YELLOW}[Step 6] Waiting for Azure LoadBalancer rules to propagate...${NC}"
echo "Waiting 30 seconds..."
sleep 30

echo -e "${YELLOW}[Step 7] Verifying LoadBalancer backend configuration...${NC}"
RG="MC_mini-xdr-prod-rg_mini-xdr-aks_eastus"
LB_NAME="kubernetes"

echo "Checking LoadBalancer rules..."
az network lb rule list --resource-group "$RG" --lb-name "$LB_NAME" \
    --query "[].{Name:name, FrontendPort:frontendPort, BackendPort:backendPort, Protocol:protocol}" \
    --output table

echo ""
echo "Checking health probes..."
az network lb probe list --resource-group "$RG" --lb-name "$LB_NAME" \
    --query "[].{Name:name, Protocol:protocol, Port:port, Path:requestPath}" \
    --output table

echo ""
echo -e "${YELLOW}[Step 8] Testing new LoadBalancer endpoint...${NC}"
echo "Testing HTTP access to: http://$NEW_IP"

if curl -I -m 10 "http://$NEW_IP" 2>/dev/null | head -1; then
    echo -e "${GREEN}✓✓✓ SUCCESS! LoadBalancer is working!${NC}\n"
else
    echo -e "${YELLOW}⚠ Connection test inconclusive${NC}"
    echo "The LoadBalancer might still be propagating (can take up to 5 minutes)"
    echo ""
    echo "Manual test command:"
    echo "  curl -I http://$NEW_IP"
    echo ""
    echo "If still failing after 5 minutes:"
    echo "  1. Run diagnostics: ./scripts/azure/fix-loadbalancer-diagnostics.sh"
    echo "  2. Try NGINX Ingress: ./scripts/azure/deploy-nginx-ingress.sh"
    echo ""
fi

echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}LoadBalancer Recreation Complete!${NC}"
echo -e "${BLUE}======================================${NC}\n"

echo -e "${GREEN}New LoadBalancer IP: $NEW_IP${NC}"
echo "Previous IP was: $CURRENT_IP"
echo ""
echo -e "${YELLOW}Update your NSG if needed:${NC}"
echo "  az network nsg rule create \\"
echo "    --resource-group $RG \\"
echo "    --nsg-name aks-agentpool-10857568-nsg \\"
echo "    --name AllowHTTPFromMyIP \\"
echo "    --priority 130 \\"
echo "    --source-address-prefixes 24.11.0.176/32 \\"
echo "    --destination-port-ranges 80 443 \\"
echo "    --access Allow --protocol Tcp"
echo ""

# Save the new IP
echo "$NEW_IP" > /tmp/mini-xdr-new-lb-ip.txt
echo -e "${GREEN}New IP saved to: /tmp/mini-xdr-new-lb-ip.txt${NC}"


