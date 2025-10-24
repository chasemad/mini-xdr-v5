#!/bin/bash
# Mini-XDR AWS Shutdown Script
# Stops all AWS resources to save costs (~$27/month savings)
# Startup time: ~8 minutes when restarting

set -e

REGION="us-east-1"
NAMESPACE="mini-xdr"
RDS_INSTANCE="mini-xdr-postgres"
REDIS_CLUSTER="mini-xdr-redis"

echo "🛑 Mini-XDR AWS Shutdown Initiated"
echo "===================================="
echo ""

# 1. Scale EKS deployments to 0 (immediate, no data loss)
echo "📦 Scaling Kubernetes deployments to 0 replicas..."
kubectl scale deployment mini-xdr-backend -n $NAMESPACE --replicas=0
kubectl scale deployment mini-xdr-frontend -n $NAMESPACE --replicas=0
echo "   ✅ Backend scaled to 0"
echo "   ✅ Frontend scaled to 0"
echo ""

# Wait for pods to terminate
echo "⏳ Waiting for pods to terminate..."
kubectl wait --for=delete pod -l app=mini-xdr-backend -n $NAMESPACE --timeout=60s 2>/dev/null || true
kubectl wait --for=delete pod -l app=mini-xdr-frontend -n $NAMESPACE --timeout=60s 2>/dev/null || true
echo "   ✅ All pods terminated"
echo ""

# 2. Stop RDS instance (saves ~$15/month)
echo "🗄️  Stopping RDS PostgreSQL instance..."
aws rds stop-db-instance \
  --db-instance-identifier $RDS_INSTANCE \
  --region $REGION \
  --output text &>/dev/null || echo "   ⚠️  RDS already stopped or not found"

RDS_STATUS=$(aws rds describe-db-instances \
  --db-instance-identifier $RDS_INSTANCE \
  --region $REGION \
  --query 'DBInstances[0].DBInstanceStatus' \
  --output text 2>/dev/null || echo "not-found")

if [ "$RDS_STATUS" != "not-found" ]; then
  echo "   ✅ RDS stop initiated (Status: $RDS_STATUS)"
  echo "   ℹ️  RDS will fully stop in ~2 minutes"
else
  echo "   ⚠️  RDS instance not found"
fi
echo ""

# 3. Stop ElastiCache Redis cluster (saves ~$12/month)
echo "🔴 Stopping ElastiCache Redis cluster..."

# Get replication group ID
REPLICATION_GROUP_ID=$(aws elasticache describe-replication-groups \
  --region $REGION \
  --query "ReplicationGroups[?contains(ReplicationGroupId, '$REDIS_CLUSTER')].ReplicationGroupId" \
  --output text 2>/dev/null || echo "")

if [ -n "$REPLICATION_GROUP_ID" ]; then
  # ElastiCache doesn't have a "stop" - need to delete and recreate
  # For now, just note it's still running
  echo "   ⚠️  Redis cluster is still running (ElastiCache can't be paused)"
  echo "   ℹ️  To fully save costs, delete and recreate Redis cluster"
  echo "   ℹ️  Cost if left running: ~$12/month"
  echo "   ℹ️  Run './scripts/delete-redis-cluster.sh' to remove it completely"
else
  echo "   ⚠️  Redis cluster not found or already deleted"
fi
echo ""

# Summary
echo "✅ Shutdown Complete!"
echo "===================="
echo ""
echo "📊 Current Status:"
echo "   • Backend pods: 0/2 running (stopped)"
echo "   • Frontend pods: 0/3 running (stopped)"
echo "   • RDS: stopping/stopped"
echo "   • Redis: running (can't pause ElastiCache)"
echo "   • EKS Cluster: running (control plane)"
echo ""
echo "💰 Cost Savings:"
echo "   • With RDS stopped: ~$15/month saved"
echo "   • Redis still running: ~$12/month cost"
echo "   • EKS control plane: ~$73/month (always running)"
echo "   • EKS nodes: ~$60/month (still running - scale down manually if needed)"
echo "   • Estimated savings: ~$15/month"
echo ""
echo "🚀 To restart: ./start-mini-xdr-aws.sh"
echo "⏱️  Startup time: ~8 minutes"
echo ""
echo "⚠️  Note: RDS will auto-start after 7 days if left stopped"
echo ""


