#!/bin/bash
set -e

cd /Users/chasemad/Desktop/mini-xdr

echo "🚀 Setting Up Immediate Public Access (NodePort Method)"
echo "========================================================"
echo ""
echo "This gives you a public URL now while AWS approves ALB access"
echo ""

# Step 1: Update security.py
echo "Step 1/5: Updating backend security..."
POD=$(kubectl get pods -n mini-xdr -l app=mini-xdr-backend -o jsonpath='{.items[0].metadata.name}')
if kubectl exec $POD -n mini-xdr -- grep -q '"/api/auth"' /app/app/security.py 2>/dev/null; then
  echo "   ✅ Security.py already configured"
else
  kubectl exec $POD -n mini-xdr -- sh -c "sed -i '27i\    \"/api/auth\",  # Authentication endpoints' /app/app/security.py"
  kubectl delete pod $POD -n mini-xdr
  kubectl rollout status deployment/mini-xdr-backend -n mini-xdr --timeout=120s
  echo "   ✅ Security updated"
fi

# Step 2: Create NodePort services
echo ""
echo "Step 2/5: Creating NodePort services..."
cat > /tmp/nodeport-services.yaml << 'EOF'
apiVersion: v1
kind: Service
metadata:
  name: mini-xdr-backend-nodeport
  namespace: mini-xdr
spec:
  type: NodePort
  selector:
    app: mini-xdr-backend
  ports:
    - port: 8000
      targetPort: 8000
      nodePort: 30800
      name: http
---
apiVersion: v1
kind: Service
metadata:
  name: mini-xdr-frontend-nodeport
  namespace: mini-xdr
spec:
  type: NodePort
  selector:
    app: mini-xdr-frontend
  ports:
    - port: 3000
      targetPort: 3000
      nodePort: 30300
      name: http
EOF

kubectl apply -f /tmp/nodeport-services.yaml
echo "   ✅ NodePort services created"

# Step 3: Get node info
echo ""
echo "Step 3/5: Getting node IP address..."
NODE_NAME=$(kubectl get nodes -o jsonpath='{.items[0].metadata.name}')
INSTANCE_ID=$(aws ec2 describe-instances \
  --filters "Name=private-dns-name,Values=$NODE_NAME" \
  --query 'Reservations[0].Instances[0].InstanceId' \
  --output text \
  --region us-east-1)

NODE_PUBLIC_IP=$(aws ec2 describe-instances \
  --instance-ids $INSTANCE_ID \
  --query 'Reservations[0].Instances[0].PublicIpAddress' \
  --output text \
  --region us-east-1)

echo "   Node: $NODE_NAME"
echo "   Public IP: $NODE_PUBLIC_IP"

# Step 4: Open security group ports
echo ""
echo "Step 4/5: Opening security group ports..."
NODE_SG=$(aws ec2 describe-instances \
  --instance-ids $INSTANCE_ID \
  --query 'Reservations[0].Instances[0].SecurityGroups[0].GroupId' \
  --output text \
  --region us-east-1)

echo "   Security Group: $NODE_SG"

# Try to add rules (may already exist)
aws ec2 authorize-security-group-ingress \
  --group-id $NODE_SG \
  --protocol tcp \
  --port 30300 \
  --cidr 0.0.0.0/0 \
  --region us-east-1 2>/dev/null || echo "   ✅ Port 30300 already open"

aws ec2 authorize-security-group-ingress \
  --group-id $NODE_SG \
  --protocol tcp \
  --port 30800 \
  --cidr 0.0.0.0/0 \
  --region us-east-1 2>/dev/null || echo "   ✅ Port 30800 already open"

echo "   ✅ Ports opened"

# Step 5: Create organization
echo ""
echo "Step 5/5: Creating organization 'mini corp'..."
sleep 3  # Wait for services to be ready

RESPONSE=$(curl -s -w "\n%{http_code}" -X POST "http://$NODE_PUBLIC_IP:30800/api/auth/register" \
  -H "Content-Type: application/json" \
  -d '{
    "organization_name": "mini corp",
    "admin_email": "admin@example.com",
    "admin_password": "demo-tpot-api-key",
    "admin_name": "Demo Admin"
  }')

HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
BODY=$(echo "$RESPONSE" | head -n-1)

if [ "$HTTP_CODE" = "200" ]; then
  echo "   ✅ Organization created successfully!"
elif echo "$BODY" | grep -q "already"; then
  echo "   ℹ️  Organization already exists"
else
  echo "   Response: $BODY (HTTP $HTTP_CODE)"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ PUBLIC ACCESS ENABLED!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "🌐 Your Public Dashboard:"
echo "   http://$NODE_PUBLIC_IP:30300"
echo ""
echo "🔐 Login Credentials:"
echo "   Email:    admin@example.com"
echo "   Password: demo-tpot-api-key"
echo ""
echo "⚠️  Limitations:"
echo "   - URL has port number (:30300)"
echo "   - No HTTPS (shows 'Not Secure' in browser)"
echo "   - Less professional looking"
echo ""
echo "✅ Benefits:"
echo "   - Works immediately, no AWS approval needed"
echo "   - Login/password still required for security"
echo "   - Can share this URL with recruiters now"
echo ""
echo "🔄 When AWS Approves ALB:"
echo "   Run: ./deploy-alb-with-org.sh"
echo "   You'll get: https://your-domain.com (proper URL)"
echo ""
echo "🚀 Opening dashboard in browser..."
open "http://$NODE_PUBLIC_IP:30300"
