#!/bin/bash
# ============================================================================
# Seamless Onboarding Testing Script
# ============================================================================
# Automated testing script for seamless onboarding without Mini Corp network
# ============================================================================

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
ALB_URL="http://k8s-minixdr-minixdri-dc5fc1df8b-1132128475.us-east-1.elb.amazonaws.com"
AWS_REGION="us-east-1"
AWS_ACCOUNT_ID="116912495274"
TEST_ORG_EMAIL="test@minixdr.com"
TEST_ORG_PASSWORD="TestPassword123!"
EXTERNAL_ID="mini-xdr-test-org"

# Helper functions
log() { echo -e "[$(date '+%H:%M:%S')] $1"; }
success() { log "${GREEN}✅ $1${NC}"; }
warning() { log "${YELLOW}⚠️  $1${NC}"; }
error() { log "${RED}❌ ERROR: $1${NC}"; exit 1; }
info() { log "${BLUE}ℹ️  $1${NC}"; }
header() {
    echo
    log "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    log "${BLUE}🚀 $1${NC}"
    log "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo
}

# Check prerequisites
check_prerequisites() {
    header "CHECKING PREREQUISITES"

    # Check AWS CLI
    if ! command -v aws &> /dev/null; then
        error "AWS CLI not found. Install: https://aws.amazon.com/cli/"
    fi
    success "AWS CLI installed"

    # Check kubectl
    if ! command -v kubectl &> /dev/null; then
        error "kubectl not found. Install: https://kubernetes.io/docs/tasks/tools/"
    fi
    success "kubectl installed"

    # Check jq
    if ! command -v jq &> /dev/null; then
        error "jq not found. Install: brew install jq (macOS) or apt-get install jq (Linux)"
    fi
    success "jq installed"

    # Check AWS credentials
    if ! aws sts get-caller-identity &> /dev/null; then
        error "AWS credentials not configured. Run: aws configure"
    fi
    success "AWS credentials configured"

    # Check kubectl connectivity
    if ! kubectl get pods -n mini-xdr &> /dev/null; then
        error "kubectl cannot connect to EKS cluster. Run: aws eks update-kubeconfig --name mini-xdr-cluster --region us-east-1"
    fi
    success "kubectl connected to EKS cluster"

    # Check if backend is running
    if ! curl -s "$ALB_URL/health" &> /dev/null; then
        error "Backend is not reachable at $ALB_URL"
    fi
    success "Backend is reachable"
}

# Setup IAM roles
setup_iam_roles() {
    header "SETTING UP IAM ROLES"

    info "Creating Seamless Onboarding IAM role..."

    # Check if role already exists
    if aws iam get-role --role-name MiniXDR-SeamlessOnboarding-Test &> /dev/null; then
        warning "Role MiniXDR-SeamlessOnboarding-Test already exists. Skipping creation."
    else
        # Create trust policy
        cat > /tmp/trust-policy.json <<EOF
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": {"AWS": "arn:aws:iam::${AWS_ACCOUNT_ID}:root"},
    "Action": "sts:AssumeRole",
    "Condition": {
      "StringEquals": {"sts:ExternalId": "${EXTERNAL_ID}"}
    }
  }]
}
EOF

        # Create role
        aws iam create-role \
            --role-name MiniXDR-SeamlessOnboarding-Test \
            --assume-role-policy-document file:///tmp/trust-policy.json \
            --description "Mini-XDR Seamless Onboarding Test Role" &> /dev/null

        success "Created IAM role: MiniXDR-SeamlessOnboarding-Test"
    fi

    # Check if policy already exists
    POLICY_ARN="arn:aws:iam::${AWS_ACCOUNT_ID}:policy/MiniXDR-SeamlessOnboarding-Test-Policy"
    if aws iam get-policy --policy-arn "$POLICY_ARN" &> /dev/null; then
        warning "Policy already exists. Skipping creation."
    else
        # Create permissions policy
        cat > /tmp/permissions-policy.json <<'EOF'
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "AssetDiscovery",
      "Effect": "Allow",
      "Action": [
        "ec2:DescribeRegions",
        "ec2:DescribeInstances",
        "ec2:DescribeVpcs",
        "ec2:DescribeSubnets",
        "ec2:DescribeSecurityGroups",
        "rds:DescribeDBInstances"
      ],
      "Resource": "*"
    },
    {
      "Sid": "AgentDeployment",
      "Effect": "Allow",
      "Action": [
        "ssm:DescribeInstanceInformation",
        "ssm:SendCommand",
        "ssm:GetCommandInvocation",
        "ssm:ListCommandInvocations"
      ],
      "Resource": "*"
    }
  ]
}
EOF

        # Create policy
        aws iam create-policy \
            --policy-name MiniXDR-SeamlessOnboarding-Test-Policy \
            --policy-document file:///tmp/permissions-policy.json &> /dev/null

        success "Created IAM policy"
    fi

    # Attach policy to role
    aws iam attach-role-policy \
        --role-name MiniXDR-SeamlessOnboarding-Test \
        --policy-arn "$POLICY_ARN" &> /dev/null || true

    ROLE_ARN="arn:aws:iam::${AWS_ACCOUNT_ID}:role/MiniXDR-SeamlessOnboarding-Test"
    success "IAM role ARN: $ROLE_ARN"

    # Test assume role
    info "Testing AssumeRole..."
    if aws sts assume-role \
        --role-arn "$ROLE_ARN" \
        --role-session-name test-session \
        --external-id "$EXTERNAL_ID" &> /dev/null; then
        success "AssumeRole test successful"
    else
        error "Cannot assume role. Check IAM configuration."
    fi
}

# Setup EC2 instance profile
setup_ec2_profile() {
    header "SETTING UP EC2 INSTANCE PROFILE"

    # Check if role exists
    if aws iam get-role --role-name MiniXDR-Test-EC2-SSM &> /dev/null; then
        warning "EC2 role already exists. Skipping."
    else
        # Create role
        aws iam create-role \
            --role-name MiniXDR-Test-EC2-SSM \
            --assume-role-policy-document '{
                "Version": "2012-10-17",
                "Statement": [{
                    "Effect": "Allow",
                    "Principal": {"Service": "ec2.amazonaws.com"},
                    "Action": "sts:AssumeRole"
                }]
            }' &> /dev/null

        success "Created EC2 IAM role"
    fi

    # Attach SSM policy
    aws iam attach-role-policy \
        --role-name MiniXDR-Test-EC2-SSM \
        --policy-arn arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore &> /dev/null || true

    # Check if instance profile exists
    if aws iam get-instance-profile --instance-profile-name MiniXDR-Test-EC2-Profile &> /dev/null; then
        warning "Instance profile already exists. Skipping."
    else
        # Create instance profile
        aws iam create-instance-profile \
            --instance-profile-name MiniXDR-Test-EC2-Profile &> /dev/null

        # Add role to profile
        aws iam add-role-to-instance-profile \
            --instance-profile-name MiniXDR-Test-EC2-Profile \
            --role-name MiniXDR-Test-EC2-SSM &> /dev/null

        success "Created EC2 instance profile"
        info "Waiting 10s for profile to propagate..."
        sleep 10
    fi
}

# Launch test EC2 instances
launch_test_instances() {
    header "LAUNCHING TEST EC2 INSTANCES"

    # Check if instances already exist
    EXISTING_INSTANCES=$(aws ec2 describe-instances \
        --filters "Name=tag:Purpose,Values=seamless-onboarding-test" "Name=instance-state-name,Values=running,pending" \
        --query 'Reservations[].Instances[].InstanceId' \
        --output text)

    if [ -n "$EXISTING_INSTANCES" ]; then
        warning "Test instances already running:"
        aws ec2 describe-instances \
            --instance-ids $EXISTING_INSTANCES \
            --query 'Reservations[].Instances[].[InstanceId,State.Name,Tags[?Key==`Name`].Value|[0]]' \
            --output table
        return
    fi

    # Get default VPC and subnet
    DEFAULT_VPC=$(aws ec2 describe-vpcs --filters "Name=is-default,Values=true" --query 'Vpcs[0].VpcId' --output text)
    DEFAULT_SUBNET=$(aws ec2 describe-subnets --filters "Name=vpc-id,Values=$DEFAULT_VPC" --query 'Subnets[0].SubnetId' --output text)

    info "Using VPC: $DEFAULT_VPC"
    info "Using Subnet: $DEFAULT_SUBNET"

    # Launch instances
    info "Launching 3 t3.micro instances..."
    aws ec2 run-instances \
        --image-id ami-0c02fb55cc1f0c4c4 \
        --instance-type t3.micro \
        --count 3 \
        --iam-instance-profile Name=MiniXDR-Test-EC2-Profile \
        --subnet-id $DEFAULT_SUBNET \
        --tag-specifications \
            'ResourceType=instance,Tags=[{Key=Name,Value=mini-xdr-test},{Key=Purpose,Value=seamless-onboarding-test},{Key=Environment,Value=test}]' \
        --user-data '#!/bin/bash
yum update -y
yum install -y amazon-ssm-agent
systemctl enable amazon-ssm-agent
systemctl start amazon-ssm-agent' &> /dev/null

    # Wait for instances
    info "Waiting for instances to start..."
    INSTANCE_IDS=$(aws ec2 describe-instances \
        --filters "Name=tag:Purpose,Values=seamless-onboarding-test" "Name=instance-state-name,Values=pending,running" \
        --query 'Reservations[].Instances[].InstanceId' \
        --output text)

    aws ec2 wait instance-running --instance-ids $INSTANCE_IDS

    success "Instances launched:"
    aws ec2 describe-instances \
        --instance-ids $INSTANCE_IDS \
        --query 'Reservations[].Instances[].[InstanceId,State.Name,InstanceType]' \
        --output table
}

# Test seamless onboarding
test_onboarding() {
    header "TESTING SEAMLESS ONBOARDING"

    # Login
    info "Logging in as $TEST_ORG_EMAIL..."
    JWT_TOKEN=$(curl -s -X POST \
        "$ALB_URL/api/auth/login" \
        -H "Content-Type: application/json" \
        -d "{\"email\":\"$TEST_ORG_EMAIL\",\"password\":\"$TEST_ORG_PASSWORD\"}" \
        | jq -r '.access_token')

    if [ "$JWT_TOKEN" = "null" ] || [ -z "$JWT_TOKEN" ]; then
        error "Login failed! Make sure test organization exists."
    fi
    success "Login successful"

    # Start quick-start
    ROLE_ARN="arn:aws:iam::${AWS_ACCOUNT_ID}:role/MiniXDR-SeamlessOnboarding-Test"
    info "Starting seamless onboarding..."

    RESPONSE=$(curl -s -X POST \
        "$ALB_URL/api/onboarding/v2/quick-start" \
        -H "Authorization: Bearer $JWT_TOKEN" \
        -H "Content-Type: application/json" \
        -d "{
            \"provider\": \"aws\",
            \"credentials\": {
                \"role_arn\": \"$ROLE_ARN\",
                \"external_id\": \"$EXTERNAL_ID\"
            }
        }")

    echo "$RESPONSE" | jq '.'

    if echo "$RESPONSE" | jq -e '.status' | grep -q "initiated"; then
        success "Onboarding initiated!"
    else
        error "Failed to start onboarding"
    fi

    # Monitor progress
    info "Monitoring progress (30 seconds)..."
    for i in {1..6}; do
        sleep 5
        echo ""
        echo "--- Check $i/6 ---"
        curl -s -X GET \
            "$ALB_URL/api/onboarding/v2/progress" \
            -H "Authorization: Bearer $JWT_TOKEN" | jq '.overall_status, .overall_progress, .discovery.assets_found, .deployment.agents_deployed'
    done

    # Final results
    echo ""
    header "FINAL RESULTS"

    echo "=== Discovered Assets ==="
    curl -s -X GET \
        "$ALB_URL/api/onboarding/v2/assets" \
        -H "Authorization: Bearer $JWT_TOKEN" | jq '.total, .assets[] | select(.asset_type=="ec2") | {asset_id, region, agent_status}'

    echo ""
    echo "=== Deployment Summary ==="
    curl -s -X GET \
        "$ALB_URL/api/onboarding/v2/deployment/summary" \
        -H "Authorization: Bearer $JWT_TOKEN" | jq '.'

    echo ""
    echo "=== Validation Summary ==="
    curl -s -X GET \
        "$ALB_URL/api/onboarding/v2/validation/summary" \
        -H "Authorization: Bearer $JWT_TOKEN" | jq '.pass_rate, .checks'

    success "Testing complete!"
}

# Cleanup
cleanup() {
    header "CLEANUP"

    info "This will terminate test instances and delete IAM resources."
    read -p "Continue? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        info "Cleanup cancelled"
        return
    fi

    # Terminate instances
    INSTANCE_IDS=$(aws ec2 describe-instances \
        --filters "Name=tag:Purpose,Values=seamless-onboarding-test" \
        --query 'Reservations[].Instances[].InstanceId' \
        --output text)

    if [ -n "$INSTANCE_IDS" ]; then
        info "Terminating instances: $INSTANCE_IDS"
        aws ec2 terminate-instances --instance-ids $INSTANCE_IDS &> /dev/null
        success "Instances terminated"
    fi

    # Delete IAM resources
    info "Deleting IAM resources..."

    aws iam detach-role-policy \
        --role-name MiniXDR-SeamlessOnboarding-Test \
        --policy-arn "arn:aws:iam::${AWS_ACCOUNT_ID}:policy/MiniXDR-SeamlessOnboarding-Test-Policy" &> /dev/null || true

    aws iam delete-policy \
        --policy-arn "arn:aws:iam::${AWS_ACCOUNT_ID}:policy/MiniXDR-SeamlessOnboarding-Test-Policy" &> /dev/null || true

    aws iam delete-role --role-name MiniXDR-SeamlessOnboarding-Test &> /dev/null || true

    aws iam remove-role-from-instance-profile \
        --instance-profile-name MiniXDR-Test-EC2-Profile \
        --role-name MiniXDR-Test-EC2-SSM &> /dev/null || true

    aws iam delete-instance-profile --instance-profile-name MiniXDR-Test-EC2-Profile &> /dev/null || true

    aws iam detach-role-policy \
        --role-name MiniXDR-Test-EC2-SSM \
        --policy-arn arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore &> /dev/null || true

    aws iam delete-role --role-name MiniXDR-Test-EC2-SSM &> /dev/null || true

    success "Cleanup complete"
}

# Main menu
show_menu() {
    echo
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║     Seamless Onboarding Testing - Automated Script            ║"
    echo "╚════════════════════════════════════════════════════════════════╝"
    echo
    echo "1) Check Prerequisites"
    echo "2) Setup IAM Roles"
    echo "3) Setup EC2 Instance Profile"
    echo "4) Launch Test EC2 Instances"
    echo "5) Run Full Onboarding Test"
    echo "6) Cleanup (Terminate instances & delete IAM)"
    echo "7) Full Setup (Steps 1-5)"
    echo "8) Exit"
    echo
}

# Main script
main() {
    if [ $# -eq 0 ]; then
        # Interactive mode
        while true; do
            show_menu
            read -p "Select option: " choice
            case $choice in
                1) check_prerequisites ;;
                2) setup_iam_roles ;;
                3) setup_ec2_profile ;;
                4) launch_test_instances ;;
                5) test_onboarding ;;
                6) cleanup ;;
                7)
                    check_prerequisites
                    setup_iam_roles
                    setup_ec2_profile
                    launch_test_instances
                    test_onboarding
                    ;;
                8) exit 0 ;;
                *) error "Invalid option" ;;
            esac
        done
    else
        # Command line mode
        case $1 in
            check) check_prerequisites ;;
            iam) setup_iam_roles ;;
            profile) setup_ec2_profile ;;
            launch) launch_test_instances ;;
            test) test_onboarding ;;
            cleanup) cleanup ;;
            full)
                check_prerequisites
                setup_iam_roles
                setup_ec2_profile
                launch_test_instances
                test_onboarding
                ;;
            *) error "Unknown command: $1. Use: check|iam|profile|launch|test|cleanup|full" ;;
        esac
    fi
}

main "$@"
