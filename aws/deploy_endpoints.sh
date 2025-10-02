#!/bin/bash
#
# Deploy SageMaker Endpoints for All Models
#

set -e

echo "=========================================="
echo "🚀 DEPLOYING SAGEMAKER ENDPOINTS"
echo "=========================================="
echo ""

TIMESTAMP=$(date +%s)

# Model configurations
declare -A MODELS
MODELS[general]="mini-xdr-general-20250930-210140:ml.m5.large"
MODELS[ddos]="mini-xdr-ddos-20250930-210142:ml.t2.medium"
MODELS[bruteforce]="mini-xdr-bruteforce-20250930-210144:ml.t2.medium"
MODELS[webattack]="mini-xdr-webattack-20250930-210146:ml.t2.medium"

deploy_endpoint() {
    local model_name=$1
    local endpoint_name=$2
    local instance_type=$3

    echo "=========================================="
    echo "🚀 Deploying: $endpoint_name"
    echo "=========================================="
    echo "Model: $model_name"
    echo "Instance: $instance_type"
    echo ""

    config_name="${endpoint_name}-config-${TIMESTAMP}"

    # Create endpoint configuration
    echo "Creating endpoint configuration..."
    aws sagemaker create-endpoint-configuration \
        --endpoint-config-name "$config_name" \
        --production-variants \
            VariantName=primary,ModelName="$model_name",InstanceType="$instance_type",InitialInstanceCount=1 \
        --region us-east-1 2>&1 | head -5

    if [ $? -eq 0 ]; then
        echo "✅ Configuration created: $config_name"
    else
        echo "❌ Failed to create configuration"
        return 1
    fi

    # Check if endpoint exists
    echo ""
    echo "Checking if endpoint exists..."
    if aws sagemaker describe-endpoint --endpoint-name "$endpoint_name" --region us-east-1 >/dev/null 2>&1; then
        echo "Updating existing endpoint..."
        aws sagemaker update-endpoint \
            --endpoint-name "$endpoint_name" \
            --endpoint-config-name "$config_name" \
            --region us-east-1
    else
        echo "Creating new endpoint..."
        aws sagemaker create-endpoint \
            --endpoint-name "$endpoint_name" \
            --endpoint-config-name "$config_name" \
            --region us-east-1
    fi

    if [ $? -eq 0 ]; then
        echo "✅ Endpoint deployment initiated: $endpoint_name"
        echo "   (Takes 5-10 minutes to become InService)"
    else
        echo "❌ Failed to deploy endpoint"
        return 1
    fi

    echo ""
}

# Deploy all endpoints
deploy_endpoint "mini-xdr-general-20250930-210140" "mini-xdr-general-endpoint" "ml.m5.large"
deploy_endpoint "mini-xdr-ddos-20250930-210142" "mini-xdr-ddos-specialist" "ml.t2.medium"
deploy_endpoint "mini-xdr-bruteforce-20250930-210144" "mini-xdr-bruteforce-specialist" "ml.t2.medium"
deploy_endpoint "mini-xdr-webattack-20250930-210146" "mini-xdr-webattack-specialist" "ml.t2.medium"

echo ""
echo "=========================================="
echo "✅ ALL ENDPOINTS DEPLOYMENT INITIATED"
echo "=========================================="
echo ""
echo "Monitor endpoints with:"
echo "  aws sagemaker list-endpoints --region us-east-1"
echo ""
echo "Check specific endpoint status:"
echo "  aws sagemaker describe-endpoint --endpoint-name mini-xdr-general-endpoint"
echo ""
