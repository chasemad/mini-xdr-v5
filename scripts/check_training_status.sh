#!/bin/bash
# Quick script to check Azure ML training status

echo "🔍 Checking Azure ML Training Status..."
echo "========================================"

JOB_ID="calm_frame_b9rlxztg0v"
WORKSPACE="mini-xdr-ml-workspace"
RESOURCE_GROUP="mini-xdr-ml-rg"

# Get status
STATUS=$(az ml job show \
  --name $JOB_ID \
  --workspace-name $WORKSPACE \
  --resource-group $RESOURCE_GROUP \
  --query "status" -o tsv 2>/dev/null)

echo "📊 Job ID: $JOB_ID"
echo "🔄 Status: $STATUS"

# Show creation time
CREATED=$(az ml job show \
  --name $JOB_ID \
  --workspace-name $WORKSPACE \
  --resource-group $RESOURCE_GROUP \
  --query "properties.creation_context.created_at" -o tsv 2>/dev/null)

echo "⏰ Started: $CREATED"

# Calculate elapsed time
if [ "$STATUS" = "Running" ]; then
    echo ""
    echo "✅ Training is RUNNING on Azure!"
    echo "🔗 Monitor at: https://ml.azure.com/runs/$JOB_ID"
    echo ""
    echo "⏱️  Estimated completion: 2-4 hours from start"
    echo "💰 Cost: ~\$0.20/hour (Standard_D4s_v3 CPU)"
    echo ""
    echo "💡 Tip: Check Azure ML Studio for real-time progress"
elif [ "$STATUS" = "Completed" ]; then
    echo ""
    echo "🎉 Training COMPLETED!"
    echo ""
    echo "📥 Download models with:"
    echo "  ./DOWNLOAD_TRAINED_MODELS.sh"
elif [ "$STATUS" = "Failed" ]; then
    echo ""
    echo "❌ Training FAILED"
    echo ""
    echo "📋 Check logs:"
    echo "  az ml job stream --name $JOB_ID --workspace-name $WORKSPACE --resource-group $RESOURCE_GROUP"
elif [ "$STATUS" = "Preparing" ] || [ "$STATUS" = "Starting" ]; then
    echo ""
    echo "🔄 Training is starting up..."
    echo "⏳ Usually takes 2-5 minutes to begin"
else
    echo ""
    echo "Status: $STATUS"
fi

echo ""
echo "========================================"

