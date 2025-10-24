#!/bin/bash

echo "🚀 Deploying Mini-XDR to AKS..."

# Set kubectl context
az aks get-credentials --resource-group mini-xdr-prod-rg --name mini-xdr-aks --overwrite-existing

# Create namespace
echo "📦 Creating namespace..."
kubectl apply -f namespace.yaml

# Create configmap
echo "⚙️  Creating configmap..."
kubectl apply -f configmap.yaml

# Deploy persistent volumes
echo "💾 Creating persistent volumes..."
kubectl apply -f persistent-volumes.yaml

# Deploy network policies for security
echo "🔒 Applying network security policies..."
kubectl apply -f network-policy.yaml

# Deploy backend
echo "🖥️  Deploying backend..."
kubectl apply -f backend-deployment.yaml

# Deploy frontend
echo "🌐 Deploying frontend..."
kubectl apply -f frontend-deployment.yaml

# Deploy loadbalancer for external access (IP-restricted)
echo "🚪 Creating LoadBalancer (restricted to 149.40.58.153)..."
kubectl apply -f loadbalancer-service.yaml

echo ""
echo "✅ Deployment complete!"
echo ""
echo "🔍 Checking pod status..."
kubectl get pods -n mini-xdr
echo ""
echo "⏳ Waiting for LoadBalancer IP assignment (this may take 2-3 minutes)..."
kubectl wait --for=jsonpath='{.status.loadBalancer.ingress}' service/mini-xdr-loadbalancer -n mini-xdr --timeout=300s 2>/dev/null || echo "Still provisioning..."
echo ""
echo "🌐 LoadBalancer Status:"
kubectl get svc -n mini-xdr mini-xdr-loadbalancer
echo ""
echo "🔐 Security Configuration:"
echo "  ✅ LoadBalancer restricted to IP: 149.40.58.153"
echo "  ✅ Network policies applied"
echo "  ✅ Pod-to-pod communication isolated"
echo ""
echo "📋 Access your deployment:"
EXTERNAL_IP=$(kubectl get svc mini-xdr-loadbalancer -n mini-xdr -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null)
if [ ! -z "$EXTERNAL_IP" ]; then
  echo "  🌐 Frontend: http://$EXTERNAL_IP"
  echo "  🔧 Backend API: http://$EXTERNAL_IP:8000"
else
  echo "  ⏳ LoadBalancer IP still provisioning... run: kubectl get svc -n mini-xdr"
fi
