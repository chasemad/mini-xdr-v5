#!/bin/bash
SECRET_NAME="$1"
REGION="${AWS_REGION:-us-east-1}"

if [ -z "$SECRET_NAME" ]; then
    echo "Usage: $0 <secret-name>"
    exit 1
fi

aws secretsmanager get-secret-value \
    --secret-id "$SECRET_NAME" \
    --query SecretString \
    --output text \
    --region "$REGION"
