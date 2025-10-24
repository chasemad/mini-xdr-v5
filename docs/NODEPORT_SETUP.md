# Temporary NodePort Setup (While Waiting for ALB Approval)

## Quick Public Access via NodePort

This gives you a public URL immediately while AWS approves your load balancer request.

### Step 1: Create NodePort Services

```bash
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
```

### Step 2: Get Worker Node Public IP

```bash
# Get worker node external IP
NODE_IP=$(kubectl get nodes -o jsonpath='{.items[0].status.addresses[?(@.type=="ExternalIP")].address}')

# If above returns empty, try this:
NODE_IP=$(kubectl get nodes -o jsonpath='{.items[0].status.addresses[?(@.type=="InternalIP")].address}')

echo "Backend: http://$NODE_IP:30800"
echo "Frontend: http://$NODE_IP:30300"
```

### Step 3: Open Security Group Ports

```bash
# Get node security group
NODE_SG=$(aws ec2 describe-instances \
  --filters "Name=tag:eks:cluster-name,Values=mini-xdr-cluster" \
  --query 'Reservations[0].Instances[0].SecurityGroups[0].GroupId' \
  --output text \
  --region us-east-1)

# Allow access to NodePorts
aws ec2 authorize-security-group-ingress \
  --group-id $NODE_SG \
  --protocol tcp \
  --port 30300 \
  --cidr 0.0.0.0/0 \
  --region us-east-1

aws ec2 authorize-security-group-ingress \
  --group-id $NODE_SG \
  --protocol tcp \
  --port 30800 \
  --cidr 0.0.0.0/0 \
  --region us-east-1
```

### Step 4: Create Organization

```bash
# Get node IP
NODE_IP=$(kubectl get nodes -o jsonpath='{.items[0].status.addresses[?(@.type=="ExternalIP")].address}')

# Create organization
curl -X POST http://$NODE_IP:30800/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "organization_name": "mini corp",
    "admin_email": "chasemadrian@protonmail.com",
    "admin_password": "demo-tpot-api-key",
    "admin_name": "Chase Madison"
  }'

# Open frontend
echo "Dashboard: http://$NODE_IP:30300"
open "http://$NODE_IP:30300"
```

### Limitations

❌ **URL is not pretty:** `http://54.123.45.67:30300` instead of `http://xdr.yourcompany.com`  
❌ **No HTTPS:** Browser will show "Not Secure"  
❌ **Port numbers visible:** Looks less professional  
✅ **Works immediately:** No waiting for AWS approval  
✅ **Login still required:** Protected by authentication  

### When ALB is Approved

Once AWS enables load balancers, switch to the proper ALB setup:

```bash
# Remove NodePort services
kubectl delete service mini-xdr-backend-nodeport mini-xdr-frontend-nodeport -n mini-xdr

# Deploy ALB
./deploy-alb-with-org.sh
```

Then you'll have a proper URL with HTTPS support!


