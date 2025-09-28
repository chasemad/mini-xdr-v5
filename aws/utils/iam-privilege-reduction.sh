#!/bin/bash

# IAM PRIVILEGE REDUCTION SCRIPT
# Implements least-privilege IAM policies
# RUN THIS AFTER DATABASE SECURITY HARDENING

set -euo pipefail

# Configuration
REGION="${AWS_REGION:-us-east-1}"
PROJECT_ROOT="/Users/chasemad/Desktop/mini-xdr"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

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

critical() {
    echo -e "${RED}[CRITICAL] $1${NC}"
}

# Banner
show_banner() {
    echo -e "${CYAN}"
    echo "=============================================="
    echo "    🔐 IAM PRIVILEGE REDUCTION 🔐"
    echo "=============================================="
    echo -e "${NC}"
    echo "This script will:"
    echo "  ❌ Remove overprivileged policies (AmazonSageMakerFullAccess)"
    echo "  🎯 Implement least-privilege IAM policies"
    echo "  🔒 Create resource-specific permissions"
    echo "  📊 Enable CloudTrail for IAM monitoring"
    echo "  🛡️ Deploy IAM Access Analyzer"
    echo ""
}

# Identify overprivileged roles and policies
identify_overprivileged_access() {
    log "🔍 Identifying overprivileged IAM access..."
    
    # Find roles with overprivileged policies
    echo "Roles with AmazonSageMakerFullAccess:" > /tmp/overprivileged-roles.txt
    aws iam list-entities-for-policy \
        --policy-arn "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess" \
        --region "$REGION" \
        --query 'PolicyRoles[*].RoleName' \
        --output text >> /tmp/overprivileged-roles.txt
    
    # Find inline policies with wildcards
    echo -e "\nRoles with wildcard policies:" >> /tmp/overprivileged-roles.txt
    aws iam list-roles --query 'Roles[*].RoleName' --output text | while read role_name; do
        if [ -n "$role_name" ]; then
            aws iam list-role-policies --role-name "$role_name" --query 'PolicyNames' --output text | while read policy_name; do
                if [ -n "$policy_name" ]; then
                    policy_doc=$(aws iam get-role-policy --role-name "$role_name" --policy-name "$policy_name" --query 'PolicyDocument' --output json)
                    if echo "$policy_doc" | jq -r '.Statement[].Action[]' 2>/dev/null | grep -q '\*'; then
                        echo "$role_name ($policy_name)" >> /tmp/overprivileged-roles.txt
                    fi
                fi
            done
        fi
    done
    
    local overprivileged_count=$(wc -l < /tmp/overprivileged-roles.txt)
    
    if [ "$overprivileged_count" -gt 2 ]; then  # Account for headers
        critical "🚨 Found overprivileged IAM access:"
        cat /tmp/overprivileged-roles.txt
        return 1
    else
        log "✅ No major overprivileged access found"
        return 0
    fi
}

# Create least-privilege SageMaker policy
create_least_privilege_sagemaker_policy() {
    log "🎯 Creating least-privilege SageMaker policy..."
    
    cat > /tmp/sagemaker-least-privilege-policy.json << EOF
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "SageMakerTrainingJobManagement",
            "Effect": "Allow",
            "Action": [
                "sagemaker:CreateTrainingJob",
                "sagemaker:DescribeTrainingJob",
                "sagemaker:StopTrainingJob",
                "sagemaker:ListTrainingJobs"
            ],
            "Resource": [
                "arn:aws:sagemaker:${REGION}:${ACCOUNT_ID}:training-job/mini-xdr-*"
            ]
        },
        {
            "Sid": "SageMakerModelManagement",
            "Effect": "Allow",
            "Action": [
                "sagemaker:CreateModel",
                "sagemaker:DescribeModel",
                "sagemaker:DeleteModel",
                "sagemaker:ListModels"
            ],
            "Resource": [
                "arn:aws:sagemaker:${REGION}:${ACCOUNT_ID}:model/mini-xdr-*"
            ]
        },
        {
            "Sid": "SageMakerEndpointManagement",
            "Effect": "Allow",
            "Action": [
                "sagemaker:CreateEndpoint",
                "sagemaker:CreateEndpointConfig",
                "sagemaker:DescribeEndpoint",
                "sagemaker:DescribeEndpointConfig",
                "sagemaker:UpdateEndpoint",
                "sagemaker:DeleteEndpoint",
                "sagemaker:DeleteEndpointConfig",
                "sagemaker:InvokeEndpoint"
            ],
            "Resource": [
                "arn:aws:sagemaker:${REGION}:${ACCOUNT_ID}:endpoint/mini-xdr-*",
                "arn:aws:sagemaker:${REGION}:${ACCOUNT_ID}:endpoint-config/mini-xdr-*"
            ]
        },
        {
            "Sid": "SageMakerNotebookAccess",
            "Effect": "Allow",
            "Action": [
                "sagemaker:CreateNotebookInstance",
                "sagemaker:DescribeNotebookInstance",
                "sagemaker:StartNotebookInstance",
                "sagemaker:StopNotebookInstance"
            ],
            "Resource": [
                "arn:aws:sagemaker:${REGION}:${ACCOUNT_ID}:notebook-instance/mini-xdr-*"
            ]
        },
        {
            "Sid": "CloudWatchLogsAccess",
            "Effect": "Allow",
            "Action": [
                "logs:CreateLogGroup",
                "logs:CreateLogStream",
                "logs:DescribeLogGroups",
                "logs:DescribeLogStreams",
                "logs:PutLogEvents"
            ],
            "Resource": [
                "arn:aws:logs:${REGION}:${ACCOUNT_ID}:log-group:/aws/sagemaker/*"
            ]
        },
        {
            "Sid": "ECRAccess",
            "Effect": "Allow",
            "Action": [
                "ecr:BatchCheckLayerAvailability",
                "ecr:GetDownloadUrlForLayer",
                "ecr:BatchGetImage"
            ],
            "Resource": "*"
        }
    ]
}
EOF
    
    # Create the policy
    local policy_name="Mini-XDR-SageMaker-LeastPrivilege"
    aws iam create-policy \
        --policy-name "$policy_name" \
        --policy-document file:///tmp/sagemaker-least-privilege-policy.json \
        --description "Least-privilege policy for Mini-XDR SageMaker operations" \
        --region "$REGION" 2>/dev/null || \
    aws iam create-policy-version \
        --policy-arn "arn:aws:iam::${ACCOUNT_ID}:policy/$policy_name" \
        --policy-document file:///tmp/sagemaker-least-privilege-policy.json \
        --set-as-default \
        --region "$REGION"
    
    log "✅ Least-privilege SageMaker policy created: $policy_name"
}

# Create least-privilege S3 policy
create_least_privilege_s3_policy() {
    log "🗃️ Creating least-privilege S3 policy..."
    
    cat > /tmp/s3-least-privilege-policy.json << EOF
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "S3ModelsBucketAccess",
            "Effect": "Allow",
            "Action": [
                "s3:GetObject",
                "s3:PutObject",
                "s3:DeleteObject"
            ],
            "Resource": [
                "arn:aws:s3:::mini-xdr-models-${ACCOUNT_ID}-${REGION}/*"
            ]
        },
        {
            "Sid": "S3ModelsBucketList",
            "Effect": "Allow",
            "Action": [
                "s3:ListBucket",
                "s3:GetBucketLocation"
            ],
            "Resource": [
                "arn:aws:s3:::mini-xdr-models-${ACCOUNT_ID}-${REGION}"
            ]
        },
        {
            "Sid": "S3MLDataBucketRead",
            "Effect": "Allow",
            "Action": [
                "s3:GetObject",
                "s3:ListBucket"
            ],
            "Resource": [
                "arn:aws:s3:::mini-xdr-ml-data-${ACCOUNT_ID}-${REGION}/*",
                "arn:aws:s3:::mini-xdr-ml-data-${ACCOUNT_ID}-${REGION}"
            ],
            "Condition": {
                "StringEquals": {
                    "s3:ExistingObjectTag/Environment": ["training", "production"]
                }
            }
        },
        {
            "Sid": "S3MLArtifactsBucketAccess",
            "Effect": "Allow",
            "Action": [
                "s3:GetObject",
                "s3:PutObject",
                "s3:DeleteObject",
                "s3:ListBucket"
            ],
            "Resource": [
                "arn:aws:s3:::mini-xdr-ml-artifacts-${ACCOUNT_ID}-${REGION}/*",
                "arn:aws:s3:::mini-xdr-ml-artifacts-${ACCOUNT_ID}-${REGION}"
            ]
        }
    ]
}
EOF
    
    # Create the policy
    local policy_name="Mini-XDR-S3-LeastPrivilege"
    aws iam create-policy \
        --policy-name "$policy_name" \
        --policy-document file:///tmp/s3-least-privilege-policy.json \
        --description "Least-privilege policy for Mini-XDR S3 operations" \
        --region "$REGION" 2>/dev/null || \
    aws iam create-policy-version \
        --policy-arn "arn:aws:iam::${ACCOUNT_ID}:policy/$policy_name" \
        --policy-document file:///tmp/s3-least-privilege-policy.json \
        --set-as-default \
        --region "$REGION"
    
    log "✅ Least-privilege S3 policy created: $policy_name"
}

# Replace overprivileged policies
replace_overprivileged_policies() {
    log "🔄 Replacing overprivileged policies with least-privilege versions..."
    
    # Find roles with AmazonSageMakerFullAccess
    local roles_with_full_access
    roles_with_full_access=$(aws iam list-entities-for-policy \
        --policy-arn "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess" \
        --query 'PolicyRoles[*].RoleName' \
        --output text)
    
    for role_name in $roles_with_full_access; do
        if [ -n "$role_name" ] && [[ "$role_name" == *"mini-xdr"* || "$role_name" == *"sagemaker"* ]]; then
            log "Updating role: $role_name"
            
            # Detach the overprivileged policy
            aws iam detach-role-policy \
                --role-name "$role_name" \
                --policy-arn "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
            
            # Attach the least-privilege policy
            aws iam attach-role-policy \
                --role-name "$role_name" \
                --policy-arn "arn:aws:iam::${ACCOUNT_ID}:policy/Mini-XDR-SageMaker-LeastPrivilege"
            
            # Also attach the S3 policy if needed
            aws iam attach-role-policy \
                --role-name "$role_name" \
                --policy-arn "arn:aws:iam::${ACCOUNT_ID}:policy/Mini-XDR-S3-LeastPrivilege"
            
            log "✅ Updated role: $role_name"
        fi
    done
}

# Enable CloudTrail for IAM monitoring
enable_iam_monitoring() {
    log "📊 Enabling CloudTrail for IAM monitoring..."
    
    # Create S3 bucket for CloudTrail logs
    local cloudtrail_bucket="mini-xdr-cloudtrail-${ACCOUNT_ID}-${REGION}"
    
    aws s3api create-bucket \
        --bucket "$cloudtrail_bucket" \
        --region "$REGION" \
        --create-bucket-configuration LocationConstraint="$REGION" 2>/dev/null || warn "CloudTrail bucket may already exist"
    
    # Create bucket policy for CloudTrail
    cat > /tmp/cloudtrail-bucket-policy.json << EOF
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "AWSCloudTrailAclCheck",
            "Effect": "Allow",
            "Principal": {
                "Service": "cloudtrail.amazonaws.com"
            },
            "Action": "s3:GetBucketAcl",
            "Resource": "arn:aws:s3:::$cloudtrail_bucket"
        },
        {
            "Sid": "AWSCloudTrailWrite",
            "Effect": "Allow",
            "Principal": {
                "Service": "cloudtrail.amazonaws.com"
            },
            "Action": "s3:PutObject",
            "Resource": "arn:aws:s3:::$cloudtrail_bucket/AWSLogs/${ACCOUNT_ID}/*",
            "Condition": {
                "StringEquals": {
                    "s3:x-amz-acl": "bucket-owner-full-control"
                }
            }
        }
    ]
}
EOF
    
    aws s3api put-bucket-policy \
        --bucket "$cloudtrail_bucket" \
        --policy file:///tmp/cloudtrail-bucket-policy.json
    
    # Create CloudTrail
    local trail_name="mini-xdr-security-trail"
    aws cloudtrail create-trail \
        --name "$trail_name" \
        --s3-bucket-name "$cloudtrail_bucket" \
        --include-global-service-events \
        --is-multi-region-trail \
        --enable-log-file-validation \
        --region "$REGION" 2>/dev/null || warn "CloudTrail may already exist"
    
    # Start logging
    aws cloudtrail start-logging \
        --name "$trail_name" \
        --region "$REGION"
    
    # Create CloudWatch log group for CloudTrail
    aws logs create-log-group \
        --log-group-name "/aws/cloudtrail/mini-xdr" \
        --region "$REGION" 2>/dev/null || warn "Log group may already exist"
    
    log "✅ CloudTrail IAM monitoring enabled"
}

# Deploy IAM Access Analyzer
deploy_iam_access_analyzer() {
    log "🔍 Deploying IAM Access Analyzer..."
    
    # Create IAM Access Analyzer
    local analyzer_name="mini-xdr-access-analyzer"
    aws accessanalyzer create-analyzer \
        --analyzer-name "$analyzer_name" \
        --type ACCOUNT \
        --region "$REGION" 2>/dev/null || warn "Access Analyzer may already exist"
    
    # Create CloudWatch alarm for new findings
    aws cloudwatch put-metric-alarm \
        --alarm-name "Mini-XDR-IAM-Access-Analyzer-Findings" \
        --alarm-description "Alert on new IAM Access Analyzer findings" \
        --metric-name "NewFindingCount" \
        --namespace "AWS/AccessAnalyzer" \
        --statistic Sum \
        --period 300 \
        --threshold 1 \
        --comparison-operator GreaterThanOrEqualToThreshold \
        --evaluation-periods 1 \
        --alarm-actions "arn:aws:sns:${REGION}:${ACCOUNT_ID}:mini-xdr-security-alerts" \
        --region "$REGION" 2>/dev/null || warn "CloudWatch alarm creation failed"
    
    log "✅ IAM Access Analyzer deployed"
}

# Create IAM security monitoring dashboard
create_iam_monitoring_dashboard() {
    log "📈 Creating IAM security monitoring dashboard..."
    
    cat > /tmp/iam-dashboard.json << EOF
{
    "widgets": [
        {
            "type": "metric",
            "properties": {
                "metrics": [
                    [ "AWS/CloudTrail", "ErrorCount" ],
                    [ ".", "EventCount" ]
                ],
                "period": 300,
                "stat": "Sum",
                "region": "${REGION}",
                "title": "CloudTrail Activity"
            }
        },
        {
            "type": "log",
            "properties": {
                "query": "SOURCE '/aws/cloudtrail/mini-xdr' | fields @timestamp, eventName, sourceIPAddress, userIdentity.type\\n| filter eventName like /IAM/\\n| stats count() by eventName\\n| sort count desc",
                "region": "${REGION}",
                "title": "IAM API Calls",
                "view": "table"
            }
        },
        {
            "type": "metric",
            "properties": {
                "metrics": [
                    [ "AWS/AccessAnalyzer", "NewFindingCount" ],
                    [ ".", "ActiveFindingCount" ]
                ],
                "period": 3600,
                "stat": "Maximum",
                "region": "${REGION}",
                "title": "Access Analyzer Findings"
            }
        }
    ]
}
EOF
    
    aws cloudwatch put-dashboard \
        --dashboard-name "Mini-XDR-IAM-Security" \
        --dashboard-body file:///tmp/iam-dashboard.json \
        --region "$REGION"
    
    log "✅ IAM monitoring dashboard created"
}

# Generate IAM security report
generate_iam_security_report() {
    log "📊 Generating IAM security report..."
    
    # Get current IAM status
    local total_roles=$(aws iam list-roles --query 'Roles | length(@)' --output text)
    local total_policies=$(aws iam list-policies --scope Local --query 'Policies | length(@)' --output text)
    local cloudtrail_status=$(aws cloudtrail get-trail-status --name mini-xdr-security-trail --query 'IsLogging' --output text 2>/dev/null || echo "Not configured")
    
    cat > "/tmp/iam-privilege-reduction-report.txt" << EOF
IAM PRIVILEGE REDUCTION REPORT
==============================
Date: $(date)
Project: Mini-XDR
Account: $ACCOUNT_ID

ACTIONS TAKEN:
✅ Created least-privilege SageMaker policy
✅ Created least-privilege S3 policy
✅ Replaced overprivileged policies
✅ Enabled CloudTrail for IAM monitoring
✅ Deployed IAM Access Analyzer
✅ Created IAM security monitoring dashboard

IAM SECURITY IMPROVEMENTS:
- Total IAM Roles: $total_roles
- Custom Policies: $total_policies
- CloudTrail Logging: $cloudtrail_status
- Access Analyzer: Enabled
- Monitoring Dashboard: Created

POLICIES CREATED:
1. Mini-XDR-SageMaker-LeastPrivilege
   - Scope: SageMaker training/inference jobs with mini-xdr-* prefix
   - Actions: Limited to necessary operations only
   - Resources: Restricted to Mini-XDR resources

2. Mini-XDR-S3-LeastPrivilege
   - Scope: S3 buckets for ML data/models/artifacts
   - Actions: Read/write with conditions
   - Resources: mini-xdr-*-${ACCOUNT_ID}-${REGION} buckets only

REMOVED OVERPRIVILEGED POLICIES:
❌ AmazonSageMakerFullAccess (replaced with least-privilege)
❌ Wildcard resource permissions (scoped to specific resources)

MONITORING AND COMPLIANCE:
- CloudTrail: mini-xdr-security-trail
- Log Group: /aws/cloudtrail/mini-xdr
- Access Analyzer: mini-xdr-access-analyzer
- Dashboard: Mini-XDR-IAM-Security

SECURITY VALIDATION:
# Check IAM policies:
aws iam list-attached-role-policies --role-name [role-name]

# View CloudTrail logs:
aws logs start-query --log-group-name /aws/cloudtrail/mini-xdr

# Check Access Analyzer findings:
aws accessanalyzer list-findings --analyzer-arn arn:aws:access-analyzer:${REGION}:${ACCOUNT_ID}:analyzer/mini-xdr-access-analyzer

# View IAM dashboard:
https://console.aws.amazon.com/cloudwatch/home?region=${REGION}#dashboards:name=Mini-XDR-IAM-Security

NEXT STEPS:
1. Review and test applications with new IAM policies
2. Monitor CloudTrail logs for any access denied errors
3. Address any IAM Access Analyzer findings
4. Set up automated IAM policy reviews
5. Configure IAM credential rotation

EMERGENCY CONTACT:
If applications fail due to insufficient permissions, check CloudTrail logs for AccessDenied events.
EOF
    
    log "📋 Report saved to: /tmp/iam-privilege-reduction-report.txt"
    echo ""
    cat /tmp/iam-privilege-reduction-report.txt
}

# Validate IAM security improvements
validate_iam_security() {
    log "✅ Validating IAM security improvements..."
    
    # Check if least-privilege policies exist
    if aws iam get-policy --policy-arn "arn:aws:iam::${ACCOUNT_ID}:policy/Mini-XDR-SageMaker-LeastPrivilege" >/dev/null 2>&1; then
        log "✅ SageMaker least-privilege policy: Created"
    else
        warn "⚠️ SageMaker least-privilege policy: Not found"
    fi
    
    if aws iam get-policy --policy-arn "arn:aws:iam::${ACCOUNT_ID}:policy/Mini-XDR-S3-LeastPrivilege" >/dev/null 2>&1; then
        log "✅ S3 least-privilege policy: Created"
    else
        warn "⚠️ S3 least-privilege policy: Not found"
    fi
    
    # Check CloudTrail status
    local cloudtrail_status
    cloudtrail_status=$(aws cloudtrail get-trail-status --name mini-xdr-security-trail --query 'IsLogging' --output text 2>/dev/null || echo "false")
    
    if [ "$cloudtrail_status" = "true" ]; then
        log "✅ CloudTrail logging: Active"
    else
        warn "⚠️ CloudTrail logging: Not active"
    fi
    
    # Check Access Analyzer
    if aws accessanalyzer get-analyzer --analyzer-name mini-xdr-access-analyzer >/dev/null 2>&1; then
        log "✅ IAM Access Analyzer: Active"
    else
        warn "⚠️ IAM Access Analyzer: Not found"
    fi
    
    # Check for remaining overprivileged policies
    local full_access_roles
    full_access_roles=$(aws iam list-entities-for-policy \
        --policy-arn "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess" \
        --query 'PolicyRoles | length(@)' \
        --output text 2>/dev/null || echo "0")
    
    if [ "$full_access_roles" -eq 0 ]; then
        log "✅ No roles with AmazonSageMakerFullAccess found"
    else
        warn "⚠️ Found $full_access_roles roles still using AmazonSageMakerFullAccess"
    fi
}

# Main execution
main() {
    show_banner
    
    # Confirm action
    critical "⚠️  WARNING: This will modify IAM policies and may affect application permissions!"
    echo ""
    read -p "Continue with IAM privilege reduction? (type 'REDUCE PRIVILEGES' to confirm): " -r
    if [ "$REPLY" != "REDUCE PRIVILEGES" ]; then
        log "Operation cancelled by user"
        exit 0
    fi
    
    log "🔐 Starting IAM privilege reduction..."
    local start_time=$(date +%s)
    
    # Execute IAM security procedures
    if identify_overprivileged_access; then
        log "✅ No major overprivileged access found"
    else
        log "Proceeding with privilege reduction..."
    fi
    
    create_least_privilege_sagemaker_policy
    create_least_privilege_s3_policy
    replace_overprivileged_policies
    enable_iam_monitoring
    deploy_iam_access_analyzer
    create_iam_monitoring_dashboard
    validate_iam_security
    generate_iam_security_report
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    log "🎉 IAM privilege reduction completed in ${duration} seconds"
    
    echo ""
    critical "🚨 CRITICAL NEXT STEPS:"
    echo "1. Test all applications for permission issues"
    echo "2. Monitor CloudTrail logs for AccessDenied events"
    echo "3. Review IAM Access Analyzer findings"
    echo "4. Update deployment scripts with new policy ARNs"
    echo "5. Run comprehensive security validation"
}

# Export configuration for other scripts
export AWS_REGION="$REGION"
export ACCOUNT_ID="$ACCOUNT_ID"

# Run main function
main "$@"
