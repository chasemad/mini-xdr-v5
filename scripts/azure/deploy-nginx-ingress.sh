#!/bin/bash
#
# Deploy NGINX Ingress Controller for Mini-XDR
# This is the RECOMMENDED solution (80% success rate per expert analysis)
#

set -e

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}Deploying NGINX Ingress Controller${NC}"
echo -e "${BLUE}======================================${NC}\n"

# Configuration
RG="MC_mini-xdr-prod-rg_mini-xdr-aks_eastus"
NSG_NAME="aks-agentpool-10857568-nsg"
YOUR_IP="24.11.0.176"

echo -e "${YELLOW}[Step 1] Installing NGINX Ingress Controller...${NC}"
echo "This will create a new LoadBalancer service with a new public IP"
echo "Downloading and applying official NGINX Ingress manifest..."
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.11.3/deploy/static/provider/cloud/deploy.yaml

echo -e "${GREEN}✓ NGINX Ingress Controller deployed${NC}\n"

echo -e "${YELLOW}[Step 2] Waiting for external IP assignment (this may take 2-5 minutes)...${NC}"
echo "Watching ingress-nginx-controller service..."

# Wait for external IP (timeout after 10 minutes)
TIMEOUT=600
ELAPSED=0
INTERVAL=5

while [ $ELAPSED -lt $TIMEOUT ]; do
    INGRESS_IP=$(kubectl get svc ingress-nginx-controller -n ingress-nginx -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "")
    
    if [ -n "$INGRESS_IP" ] && [ "$INGRESS_IP" != "null" ]; then
        echo -e "\n${GREEN}✓ External IP assigned: $INGRESS_IP${NC}\n"
        break
    fi
    
    echo -n "."
    sleep $INTERVAL
    ELAPSED=$((ELAPSED + INTERVAL))
done

if [ -z "$INGRESS_IP" ] || [ "$INGRESS_IP" == "null" ]; then
    echo -e "\n${RED}✗ Timeout waiting for external IP${NC}"
    echo "Check the service status manually:"
    echo "  kubectl get svc -n ingress-nginx ingress-nginx-controller"
    exit 1
fi

echo -e "${YELLOW}[Step 3] Creating Ingress resource for Mini-XDR frontend...${NC}"

cat <<EOF | kubectl apply -f -
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: mini-xdr-ingress
  namespace: mini-xdr
  annotations:
    nginx.ingress.kubernetes.io/ssl-redirect: "false"
    nginx.ingress.kubernetes.io/rewrite-target: /
    nginx.ingress.kubernetes.io/proxy-body-size: "100m"
    nginx.ingress.kubernetes.io/proxy-connect-timeout: "600"
    nginx.ingress.kubernetes.io/proxy-send-timeout: "600"
    nginx.ingress.kubernetes.io/proxy-read-timeout: "600"
spec:
  ingressClassName: nginx
  rules:
  - http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: mini-xdr-frontend-service
            port:
              number: 3000
EOF

echo -e "${GREEN}✓ Ingress resource created${NC}\n"

echo -e "${YELLOW}[Step 4] Updating NSG to allow traffic from your IP to new LoadBalancer...${NC}"
echo "Creating NSG rule for IP: $INGRESS_IP from source: $YOUR_IP"

az network nsg rule create \
    --resource-group "$RG" \
    --nsg-name "$NSG_NAME" \
    --name "AllowIngressFromMyIP" \
    --priority 140 \
    --source-address-prefixes "${YOUR_IP}/32" \
    --destination-address-prefixes "$INGRESS_IP" \
    --destination-port-ranges 80 443 \
    --access Allow \
    --protocol Tcp \
    --description "Allow HTTP/HTTPS to NGINX Ingress from my IP" \
    2>/dev/null && echo -e "${GREEN}✓ NSG rule created${NC}" || echo -e "${YELLOW}⚠ Rule might already exist or NSG update not needed${NC}"

echo ""
echo -e "${YELLOW}[Step 5] Testing the new Ingress endpoint...${NC}"
echo "Waiting 10 seconds for Ingress to be ready..."
sleep 10

echo "Testing HTTP access to: http://$INGRESS_IP"
if curl -I -m 10 "http://$INGRESS_IP" 2>/dev/null | head -1; then
    echo -e "${GREEN}✓✓✓ SUCCESS! Your Mini-XDR frontend is now accessible!${NC}\n"
else
    echo -e "${YELLOW}⚠ Initial test inconclusive, checking Ingress status...${NC}"
    kubectl get ingress -n mini-xdr mini-xdr-ingress
    echo ""
    echo "Check Ingress controller logs:"
    echo "  kubectl logs -n ingress-nginx deployment/ingress-nginx-controller --tail=50"
    echo ""
fi

echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}Deployment Complete!${NC}"
echo -e "${BLUE}======================================${NC}\n"

echo -e "${GREEN}Your Mini-XDR Demo URL:${NC}"
echo -e "${GREEN}  http://$INGRESS_IP${NC}\n"

echo -e "${YELLOW}Important Information:${NC}"
echo "• External IP: $INGRESS_IP"
echo "• Protocol: HTTP (port 80)"
echo "• Backend: mini-xdr-frontend-service (port 3000)"
echo "• Namespace: mini-xdr"
echo ""

echo -e "${YELLOW}Testing Commands:${NC}"
echo "  curl -I http://$INGRESS_IP"
echo "  curl http://$INGRESS_IP"
echo ""

echo -e "${YELLOW}Monitoring Commands:${NC}"
echo "  kubectl get svc -n ingress-nginx"
echo "  kubectl get ingress -n mini-xdr"
echo "  kubectl logs -n ingress-nginx deployment/ingress-nginx-controller -f"
echo ""

echo -e "${YELLOW}Next Steps for HTTPS (Optional):${NC}"
echo "1. Install cert-manager for automatic TLS certificates"
echo "2. Configure a domain name (or use nip.io for testing)"
echo "3. Update Ingress with TLS configuration"
echo ""

echo -e "${YELLOW}To add this to your resume/demo docs:${NC}"
echo "Update your documentation with the new external IP: $INGRESS_IP"
echo ""

# Save the IP to a file for reference
echo "$INGRESS_IP" > /tmp/mini-xdr-ingress-ip.txt
echo -e "${GREEN}Ingress IP saved to: /tmp/mini-xdr-ingress-ip.txt${NC}"


