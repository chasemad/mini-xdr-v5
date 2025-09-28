#!/bin/bash

# Mini-XDR Frontend AWS Deployment Script
# Deploys Next.js frontend to S3 + CloudFront

set -euo pipefail

# Configuration
REGION="${AWS_REGION:-us-east-1}"
STACK_NAME="mini-xdr-frontend"
BACKEND_STACK_NAME="mini-xdr-backend"
PROJECT_DIR="/Users/chasemad/Desktop/mini-xdr"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[WARN] $1${NC}"
}

error() {
    echo -e "${RED}[ERROR] $1${NC}"
    exit 1
}

# Get backend information
get_backend_info() {
    log "Getting backend information..."
    
    if ! aws cloudformation describe-stacks --stack-name "$BACKEND_STACK_NAME" --region "$REGION" >/dev/null 2>&1; then
        error "Backend stack '$BACKEND_STACK_NAME' not found. Please deploy backend first."
    fi
    
    BACKEND_IP=$(aws cloudformation describe-stacks \
        --stack-name "$BACKEND_STACK_NAME" \
        --region "$REGION" \
        --query 'Stacks[0].Outputs[?OutputKey==`BackendPublicIP`].OutputValue' \
        --output text)
    
    if [ -z "$BACKEND_IP" ] || [ "$BACKEND_IP" = "None" ]; then
        error "Could not retrieve backend IP from CloudFormation stack"
    fi
    
    log "Backend IP: $BACKEND_IP"
}

# Create CloudFormation template for frontend
create_frontend_template() {
    log "Creating frontend CloudFormation template..."
    
    cat > "/tmp/mini-xdr-frontend.yaml" << EOF
AWSTemplateFormatVersion: '2010-09-09'
Description: 'Mini-XDR Frontend Infrastructure (S3 + CloudFront)'

Parameters:
  BackendIP:
    Type: String
    Default: ${BACKEND_IP}
    Description: Mini-XDR backend IP address

Resources:
  # S3 Bucket for Frontend Static Files
  FrontendBucket:
    Type: AWS::S3::Bucket
    Properties:
      BucketName: !Sub "mini-xdr-frontend-\${AWS::AccountId}-\${AWS::Region}"
      WebsiteConfiguration:
        IndexDocument: index.html
        ErrorDocument: error.html
      PublicAccessBlockConfiguration:
        BlockPublicAcls: false
        BlockPublicPolicy: false
        IgnorePublicAcls: false
        RestrictPublicBuckets: false
      CorsConfiguration:
        CorsRules:
          - AllowedHeaders: ['*']
            AllowedMethods: ['GET', 'HEAD']
            AllowedOrigins: ['*']
            MaxAge: 3000
      Tags:
        - Key: Name
          Value: mini-xdr-frontend
        - Key: Project
          Value: mini-xdr

  # Bucket Policy for Public Read Access
  FrontendBucketPolicy:
    Type: AWS::S3::BucketPolicy
    Properties:
      Bucket: !Ref FrontendBucket
      PolicyDocument:
        Statement:
          - Sid: PublicReadGetObject
            Effect: Allow
            Principal: '*'
            Action: 's3:GetObject'
            Resource: !Sub "\${FrontendBucket}/*"

  # CloudFront Origin Access Identity (for secure S3 access)
  CloudFrontOAI:
    Type: AWS::CloudFront::OriginAccessIdentity
    Properties:
      OriginAccessIdentityConfig:
        Comment: !Sub "OAI for Mini-XDR Frontend (\${AWS::StackName})"

  # CloudFront Distribution
  CloudFrontDistribution:
    Type: AWS::CloudFront::Distribution
    Properties:
      DistributionConfig:
        Origins:
          - Id: S3Origin
            DomainName: !GetAtt FrontendBucket.DomainName
            S3OriginConfig:
              OriginAccessIdentity: !Sub "origin-access-identity/cloudfront/\${CloudFrontOAI}"
        Enabled: true
        DefaultRootObject: index.html
        DefaultCacheBehavior:
          TargetOriginId: S3Origin
          ViewerProtocolPolicy: redirect-to-https
          AllowedMethods:
            - GET
            - HEAD
            - OPTIONS
          CachedMethods:
            - GET
            - HEAD
          ForwardedValues:
            QueryString: false
            Cookies:
              Forward: none
          Compress: true
          DefaultTTL: 86400
          MaxTTL: 31536000
        CustomErrorResponses:
          - ErrorCode: 404
            ResponseCode: 200
            ResponsePagePath: /index.html
          - ErrorCode: 403
            ResponseCode: 200
            ResponsePagePath: /index.html
        PriceClass: PriceClass_100
        ViewerCertificate:
          CloudFrontDefaultCertificate: true
      Tags:
        - Key: Name
          Value: mini-xdr-frontend-cdn
        - Key: Project
          Value: mini-xdr

Outputs:
  BucketName:
    Description: Name of the S3 bucket
    Value: !Ref FrontendBucket
    Export:
      Name: !Sub "\${AWS::StackName}-BucketName"
      
  BucketWebsiteURL:
    Description: S3 bucket website URL
    Value: !GetAtt FrontendBucket.WebsiteURL
    Export:
      Name: !Sub "\${AWS::StackName}-WebsiteURL"
      
  CloudFrontDomainName:
    Description: CloudFront distribution domain name
    Value: !GetAtt CloudFrontDistribution.DomainName
    Export:
      Name: !Sub "\${AWS::StackName}-CloudFrontURL"
      
  CloudFrontURL:
    Description: CloudFront distribution URL
    Value: !Sub "https://\${CloudFrontDistribution.DomainName}"
    
  BackendAPIURL:
    Description: Backend API URL for frontend configuration
    Value: !Sub "http://\${BackendIP}:8000"
EOF
}

# Deploy frontend infrastructure
deploy_frontend_stack() {
    log "Deploying frontend CloudFormation stack..."
    
    aws cloudformation deploy \
        --template-file "/tmp/mini-xdr-frontend.yaml" \
        --stack-name "$STACK_NAME" \
        --parameter-overrides \
            BackendIP="$BACKEND_IP" \
        --region "$REGION" || error "Frontend stack deployment failed"
    
    log "Frontend infrastructure deployed successfully!"
}

# Build frontend for production
build_frontend() {
    log "Building frontend for production..."
    
    cd "$PROJECT_DIR/frontend"
    
    # Create production environment file
    cat > ".env.production" << EOF
NEXT_PUBLIC_API_URL=http://${BACKEND_IP}:8000
NEXT_PUBLIC_WS_URL=ws://${BACKEND_IP}:8000/ws
NEXT_PUBLIC_ENV=aws-production
EOF
    
    # Install dependencies and build
    npm ci
    npm run build
    
    log "Frontend build completed!"
}

# Upload frontend to S3
upload_frontend() {
    log "Uploading frontend to S3..."
    
    # Get bucket name from CloudFormation
    local bucket_name
    bucket_name=$(aws cloudformation describe-stacks \
        --stack-name "$STACK_NAME" \
        --region "$REGION" \
        --query 'Stacks[0].Outputs[?OutputKey==`BucketName`].OutputValue' \
        --output text)
    
    if [ -z "$bucket_name" ] || [ "$bucket_name" = "None" ]; then
        error "Could not retrieve bucket name from CloudFormation stack"
    fi
    
    # Upload built files to S3
    cd "$PROJECT_DIR/frontend"
    aws s3 sync out/ "s3://$bucket_name/" --delete --region "$REGION"
    
    # Set proper content types
    aws s3 cp "s3://$bucket_name/" "s3://$bucket_name/" \
        --recursive \
        --metadata-directive REPLACE \
        --cache-control "public, max-age=31536000" \
        --region "$REGION"
    
    log "Frontend uploaded to S3 bucket: $bucket_name"
}

# Get frontend URLs
get_frontend_urls() {
    log "Getting frontend URLs..."
    
    local outputs
    outputs=$(aws cloudformation describe-stacks \
        --stack-name "$STACK_NAME" \
        --region "$REGION" \
        --query 'Stacks[0].Outputs' \
        --output json)
    
    CLOUDFRONT_URL=$(echo "$outputs" | jq -r '.[] | select(.OutputKey=="CloudFrontURL") | .OutputValue')
    WEBSITE_URL=$(echo "$outputs" | jq -r '.[] | select(.OutputKey=="BucketWebsiteURL") | .OutputValue')
    
    log "CloudFront URL: $CLOUDFRONT_URL"
    log "S3 Website URL: $WEBSITE_URL"
}

# Test frontend deployment
test_frontend() {
    log "Testing frontend deployment..."
    
    # Test CloudFront URL
    local retry_count=0
    local max_retries=5
    
    while ! curl -f "$CLOUDFRONT_URL" >/dev/null 2>&1; do
        retry_count=$((retry_count + 1))
        if [ $retry_count -gt $max_retries ]; then
            warn "CloudFront URL not accessible yet (propagation may take up to 15 minutes)"
            break
        fi
        log "Waiting for CloudFront propagation... ($retry_count/$max_retries)"
        sleep 30
    done
    
    # Test S3 website URL
    if curl -f "$WEBSITE_URL" >/dev/null 2>&1; then
        log "✅ Frontend is accessible via S3!"
    else
        warn "⚠️  S3 website URL not accessible"
    fi
    
    if [ $retry_count -le $max_retries ]; then
        log "✅ Frontend is accessible via CloudFront!"
    fi
}

# Create update script for future deployments
create_update_script() {
    log "Creating frontend update script..."
    
    local bucket_name
    bucket_name=$(aws cloudformation describe-stacks \
        --stack-name "$STACK_NAME" \
        --region "$REGION" \
        --query 'Stacks[0].Outputs[?OutputKey==`BucketName`].OutputValue' \
        --output text)
    
    cat > "/tmp/update-frontend-aws.sh" << EOF
#!/bin/bash
# Mini-XDR Frontend Update Script
# Quickly deploy frontend changes to AWS

set -euo pipefail

PROJECT_DIR="/Users/chasemad/Desktop/mini-xdr"
BUCKET_NAME="$bucket_name"
REGION="$REGION"

echo "🔄 Updating Mini-XDR Frontend on AWS..."

# Build frontend
cd "\$PROJECT_DIR/frontend"
echo "📦 Building frontend..."
npm run build

# Upload to S3
echo "⬆️  Uploading to S3..."
aws s3 sync out/ "s3://\$BUCKET_NAME/" --delete --region "\$REGION"

# Invalidate CloudFront cache
echo "🔄 Invalidating CloudFront cache..."
DISTRIBUTION_ID=\$(aws cloudfront list-distributions \
    --query "DistributionList.Items[?Origins.Items[0].DomainName=='$bucket_name.s3.amazonaws.com'].Id" \
    --output text)

if [ -n "\$DISTRIBUTION_ID" ] && [ "\$DISTRIBUTION_ID" != "None" ]; then
    aws cloudfront create-invalidation \
        --distribution-id "\$DISTRIBUTION_ID" \
        --paths "/*" >/dev/null
    echo "✅ CloudFront cache invalidated"
else
    echo "⚠️  Could not find CloudFront distribution for cache invalidation"
fi

echo "✅ Frontend update completed!"
echo "🌐 Access your updated frontend at: $CLOUDFRONT_URL"
EOF
    
    chmod +x "/tmp/update-frontend-aws.sh"
    cp "/tmp/update-frontend-aws.sh" "$HOME/update-frontend-aws.sh"
    
    log "Frontend update script created: $HOME/update-frontend-aws.sh"
}

# Main function
main() {
    log "Starting Mini-XDR frontend AWS deployment..."
    
    get_backend_info
    create_frontend_template
    deploy_frontend_stack
    build_frontend
    upload_frontend
    get_frontend_urls
    test_frontend
    create_update_script
    
    log "✅ Frontend deployment completed successfully!"
    log ""
    log "🌐 Frontend URLs:"
    log "   CloudFront: $CLOUDFRONT_URL"
    log "   S3 Direct: $WEBSITE_URL"
    log ""
    log "🔄 Update Script: $HOME/update-frontend-aws.sh"
    log ""
    log "Note: CloudFront propagation may take up to 15 minutes for global availability"
}

# Run main function
main "$@"
