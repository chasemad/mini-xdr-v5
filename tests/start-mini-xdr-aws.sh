#!/bin/bash
# Mini-XDR AWS Startup Script
# Starts all AWS resources and scales deployments
# Total startup time: ~8 minutes

set -e

REGION="us-east-1"
NAMESPACE="mini-xdr"
RDS_INSTANCE="mini-xdr-postgres"
REDIS_CLUSTER="mini-xdr-redis"
BACKEND_REPLICAS=2
FRONTEND_REPLICAS=3

echo "🚀 Mini-XDR AWS Startup Initiated"
echo "=================================="
echo "⏱️  Estimated time: 8 minutes"
echo ""

# 1. Start RDS instance
echo "🗄️  Starting RDS PostgreSQL instance..."
RDS_STATUS=$(aws rds describe-db-instances \
  --db-instance-identifier $RDS_INSTANCE \
  --region $REGION \
  --query 'DBInstances[0].DBInstanceStatus' \
  --output text 2>/dev/null || echo "not-found")

if [ "$RDS_STATUS" == "stopped" ]; then
  aws rds start-db-instance \
    --db-instance-identifier $RDS_INSTANCE \
    --region $REGION \
    --output text &>/dev/null
  echo "   ✅ RDS start initiated"
  echo "   ⏳ Waiting for RDS to become available (~5 minutes)..."
  
  # Wait for RDS to be available
  aws rds wait db-instance-available \
    --db-instance-identifier $RDS_INSTANCE \
    --region $REGION
  
  echo "   ✅ RDS is now available"
elif [ "$RDS_STATUS" == "available" ]; then
  echo "   ℹ️  RDS already running"
elif [ "$RDS_STATUS" == "starting" ]; then
  echo "   ℹ️  RDS already starting, waiting for available status..."
  aws rds wait db-instance-available \
    --db-instance-identifier $RDS_INSTANCE \
    --region $REGION
  echo "   ✅ RDS is now available"
else
  echo "   ⚠️  RDS status: $RDS_STATUS"
fi
echo ""

# 2. Check Redis cluster status
echo "🔴 Checking ElastiCache Redis status..."
REDIS_STATUS=$(aws elasticache describe-replication-groups \
  --region $REGION \
  --query "ReplicationGroups[?contains(ReplicationGroupId, '$REDIS_CLUSTER')].Status" \
  --output text 2>/dev/null || echo "not-found")

if [ "$REDIS_STATUS" == "available" ]; then
  echo "   ✅ Redis cluster already available"
elif [ "$REDIS_STATUS" == "not-found" ]; then
  echo "   ⚠️  Redis cluster not found - may need to recreate"
  echo "   ℹ️  Run './scripts/create-redis-cluster.sh' to recreate"
else
  echo "   ℹ️  Redis status: $REDIS_STATUS"
fi
echo ""

# 3. Scale EKS deployments
echo "📦 Scaling Kubernetes deployments..."
kubectl scale deployment mini-xdr-backend -n $NAMESPACE --replicas=$BACKEND_REPLICAS
kubectl scale deployment mini-xdr-frontend -n $NAMESPACE --replicas=$FRONTEND_REPLICAS
echo "   ✅ Backend scaled to $BACKEND_REPLICAS replicas"
echo "   ✅ Frontend scaled to $FRONTEND_REPLICAS replicas"
echo ""

# 4. Wait for pods to be ready
echo "⏳ Waiting for pods to become ready..."
echo "   Backend pods:"
kubectl wait --for=condition=ready pod \
  -l app=mini-xdr-backend \
  -n $NAMESPACE \
  --timeout=180s || echo "   ⚠️  Backend pods timeout (check manually)"

echo "   Frontend pods:"
kubectl wait --for=condition=ready pod \
  -l app=mini-xdr-frontend \
  -n $NAMESPACE \
  --timeout=180s || echo "   ⚠️  Frontend pods timeout (check manually)"
echo ""

# 5. Get endpoints
echo "🌐 Getting connection information..."
RDS_ENDPOINT=$(aws rds describe-db-instances \
  --db-instance-identifier $RDS_INSTANCE \
  --region $REGION \
  --query 'DBInstances[0].Endpoint.Address' \
  --output text 2>/dev/null || echo "not-available")

REDIS_ENDPOINT=$(aws elasticache describe-replication-groups \
  --region $REGION \
  --query "ReplicationGroups[?contains(ReplicationGroupId, '$REDIS_CLUSTER')].NodeGroups[0].PrimaryEndpoint.Address" \
  --output text 2>/dev/null || echo "not-available")

ALB_ADDRESS=$(kubectl get ingress -n $NAMESPACE -o jsonpath='{.items[0].status.loadBalancer.ingress[0].hostname}' 2>/dev/null || echo "not-configured")

echo ""
echo "✅ Startup Complete!"
echo "===================="
echo ""
echo "📊 System Status:"
POD_STATUS=$(kubectl get pods -n $NAMESPACE 2>/dev/null | grep -E "mini-xdr-(backend|frontend)" || echo "No pods found")
if [ "$POD_STATUS" != "No pods found" ]; then
  echo "$POD_STATUS" | while read line; do echo "   $line"; done
else
  echo "   ⚠️  No pods found - check namespace: $NAMESPACE"
fi
echo ""

echo "🔗 Endpoints:"
echo "   RDS:   $RDS_ENDPOINT:5432"
echo "   Redis: $REDIS_ENDPOINT:6379"
if [ "$ALB_ADDRESS" != "not-configured" ] && [ -n "$ALB_ADDRESS" ]; then
  echo "   ALB:   https://$ALB_ADDRESS"
else
  echo "   ALB:   Not configured (use port-forward)"
fi
echo ""

echo "🔐 Access Dashboard:"
echo "   Local access (port-forward):"
echo "   $ kubectl port-forward -n $NAMESPACE svc/mini-xdr-frontend-service 3000:3000 &"
echo "   $ kubectl port-forward -n $NAMESPACE svc/mini-xdr-backend-service 8000:8000 &"
echo "   $ open http://localhost:3000"
echo ""

echo "📋 Useful Commands:"
echo "   Check pods:    kubectl get pods -n $NAMESPACE"
echo "   View logs:     kubectl logs -f deployment/mini-xdr-backend -n $NAMESPACE"
echo "   Health check:  curl http://localhost:8000/health (after port-forward)"
echo ""

echo "🛑 To shutdown: ./stop-mini-xdr-aws.sh"
echo ""


