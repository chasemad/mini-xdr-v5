#!/bin/bash
# AWS CodeBuild Setup Script for Mini-XDR
# Creates IAM roles and CodeBuild projects

set -e

AWS_REGION="us-east-1"
AWS_ACCOUNT_ID="116912495274"
PROJECT_NAME="mini-xdr"

echo "=== Setting up AWS CodeBuild for Mini-XDR ==="

# Step 1: Create IAM Service Role for CodeBuild
echo ""
echo "Step 1: Creating IAM Service Role..."

ROLE_NAME="${PROJECT_NAME}-codebuild-role"

# Check if role exists
if aws iam get-role --role-name $ROLE_NAME 2>/dev/null; then
    echo "✅ Role $ROLE_NAME already exists"
else
    echo "Creating IAM role..."
    
    # Create trust policy
    cat > /tmp/codebuild-trust-policy.json <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "codebuild.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF

    # Create role
    aws iam create-role \
        --role-name $ROLE_NAME \
        --assume-role-policy-document file:///tmp/codebuild-trust-policy.json \
        --description "Service role for Mini-XDR CodeBuild projects"

    # Attach policies
    echo "Attaching policies..."
    
    # Policy for ECR access
    aws iam attach-role-policy \
        --role-name $ROLE_NAME \
        --policy-arn arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryPowerUser

    # Create inline policy for CloudWatch Logs and S3
    cat > /tmp/codebuild-inline-policy.json <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:PutLogEvents"
      ],
      "Resource": "arn:aws:logs:${AWS_REGION}:${AWS_ACCOUNT_ID}:log-group:/aws/codebuild/${PROJECT_NAME}-*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject"
      ],
      "Resource": "arn:aws:s3:::codepipeline-${AWS_REGION}-*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "ssm:GetParameters"
      ],
      "Resource": "arn:aws:ssm:${AWS_REGION}:${AWS_ACCOUNT_ID}:parameter/${PROJECT_NAME}/*"
    }
  ]
}
EOF

    aws iam put-role-policy \
        --role-name $ROLE_NAME \
        --policy-name CodeBuildInlinePolicy \
        --policy-document file:///tmp/codebuild-inline-policy.json

    echo "✅ IAM role created successfully"
fi

ROLE_ARN="arn:aws:iam::${AWS_ACCOUNT_ID}:role/${ROLE_NAME}"
echo "Role ARN: $ROLE_ARN"

# Wait for role to propagate
echo "Waiting 10 seconds for IAM role to propagate..."
sleep 10

# Step 2: Create CodeBuild Project for Backend
echo ""
echo "Step 2: Creating CodeBuild project for Backend..."

BACKEND_PROJECT="${PROJECT_NAME}-backend-build"

if aws codebuild batch-get-projects --names $BACKEND_PROJECT --region $AWS_REGION 2>/dev/null | grep -q $BACKEND_PROJECT; then
    echo "✅ Backend project already exists"
else
    aws codebuild create-project \
        --name $BACKEND_PROJECT \
        --description "Builds Mini-XDR backend Docker image" \
        --source type=GITHUB,location=https://github.com/chasemad/mini-xdr-v2.git,buildspec=buildspec-backend.yml \
        --artifacts type=NO_ARTIFACTS \
        --environment type=LINUX_CONTAINER,image=aws/codebuild/standard:7.0,computeType=BUILD_GENERAL1_LARGE,privilegedMode=true \
        --service-role $ROLE_ARN \
        --region $AWS_REGION \
        --tags key=Project,value=mini-xdr key=Component,value=backend

    echo "✅ Backend CodeBuild project created"
fi

# Step 3: Create CodeBuild Project for Frontend
echo ""
echo "Step 3: Creating CodeBuild project for Frontend..."

FRONTEND_PROJECT="${PROJECT_NAME}-frontend-build"

if aws codebuild batch-get-projects --names $FRONTEND_PROJECT --region $AWS_REGION 2>/dev/null | grep -q $FRONTEND_PROJECT; then
    echo "✅ Frontend project already exists"
else
    aws codebuild create-project \
        --name $FRONTEND_PROJECT \
        --description "Builds Mini-XDR frontend Docker image" \
        --source type=GITHUB,location=https://github.com/chasemad/mini-xdr-v2.git,buildspec=buildspec-frontend.yml \
        --artifacts type=NO_ARTIFACTS \
        --environment type=LINUX_CONTAINER,image=aws/codebuild/standard:7.0,computeType=BUILD_GENERAL1_MEDIUM,privilegedMode=true \
        --service-role $ROLE_ARN \
        --region $AWS_REGION \
        --tags key=Project,value=mini-xdr key=Component,value=frontend

    echo "✅ Frontend CodeBuild project created"
fi

# Step 4: Set up GitHub webhooks
echo ""
echo "Step 4: Setting up GitHub webhooks..."
echo "⚠️  GitHub webhook setup requires GitHub personal access token"
echo "    You'll need to configure this manually in AWS Console or via API"

echo ""
echo "=== Setup Complete! ==="
echo ""
echo "📋 Next Steps:"
echo "1. Connect GitHub to CodeBuild:"
echo "   https://console.aws.amazon.com/codesuite/codebuild/projects/${BACKEND_PROJECT}/edit/source?region=${AWS_REGION}"
echo ""
echo "2. Configure webhook triggers (on push to main, on tag push)"
echo ""
echo "3. Start first build:"
echo "   aws codebuild start-build --project-name ${BACKEND_PROJECT}"
echo "   aws codebuild start-build --project-name ${FRONTEND_PROJECT}"
echo ""
echo "💰 Estimated Monthly Cost: \$5-10 (pay-per-minute)"
echo "🚀 Build Capacity: 100GB disk space, up to 15GB RAM"

