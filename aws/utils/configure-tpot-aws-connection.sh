#!/bin/bash

# Configure TPOT to send data directly to AWS Mini-XDR
# This script updates TPOT's Fluent Bit configuration to send logs to AWS

set -euo pipefail

# Configuration
TPOT_HOST="34.193.101.171"
TPOT_SSH_PORT="64295"
TPOT_USER="admin"
KEY_NAME="${KEY_NAME:-mini-xdr-tpot-key}"
STACK_NAME="mini-xdr-backend"
REGION="${AWS_REGION:-us-east-1}"

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

# Get Mini-XDR backend IP from CloudFormation
get_backend_ip() {
    log "Getting Mini-XDR backend IP from CloudFormation..."
    
    if ! aws cloudformation describe-stacks --stack-name "$STACK_NAME" --region "$REGION" >/dev/null 2>&1; then
        error "Stack '$STACK_NAME' not found. Please deploy Mini-XDR backend first."
    fi
    
    BACKEND_IP=$(aws cloudformation describe-stacks \
        --stack-name "$STACK_NAME" \
        --region "$REGION" \
        --query 'Stacks[0].Outputs[?OutputKey==`BackendPublicIP`].OutputValue' \
        --output text)
    
    if [ -z "$BACKEND_IP" ] || [ "$BACKEND_IP" = "None" ]; then
        error "Could not retrieve backend IP from CloudFormation stack"
    fi
    
    log "Mini-XDR Backend IP: $BACKEND_IP"
}

# Test connectivity to Mini-XDR backend
test_backend_connectivity() {
    log "Testing connectivity to Mini-XDR backend..."
    
    local health_url="http://$BACKEND_IP:8000/health"
    if curl -f "$health_url" >/dev/null 2>&1; then
        log "✅ Mini-XDR backend is accessible"
    else
        error "❌ Cannot reach Mini-XDR backend at $health_url"
    fi
}

# Test TPOT connectivity
test_tpot_connectivity() {
    log "Testing connectivity to TPOT..."
    
    if ssh -i "~/.ssh/${KEY_NAME}.pem" -p "$TPOT_SSH_PORT" -o ConnectTimeout=10 -o StrictHostKeyChecking=yes -o UserKnownHostsFile=~/.ssh/known_hosts \
           "$TPOT_USER@$TPOT_HOST" "echo 'TPOT connection successful'" >/dev/null 2>&1; then
        log "✅ TPOT is accessible"
    else
        error "❌ Cannot connect to TPOT. Check SSH key and network connectivity."
    fi
}

# Create new Fluent Bit configuration
create_fluent_bit_config() {
    log "Creating new Fluent Bit configuration..."
    
    cat > "/tmp/fluent-bit-tpot-aws.conf" << EOF
[SERVICE]
    Flush         5
    Log_Level     info
    Daemon        off
    Parsers_File  parsers.conf
    HTTP_Server   On
    HTTP_Listen   0.0.0.0
    HTTP_Port     2020

# Cowrie SSH/Telnet Honeypot
[INPUT]
    Name              tail
    Path              /data/cowrie/log/cowrie.json
    Parser            json
    Tag               tpot.cowrie
    Refresh_Interval  5
    Read_from_Head    false
    Skip_Long_Lines   On
    Buffer_Chunk_Size 32k
    Buffer_Max_Size   64k

# Dionaea Multi-Protocol Honeypot  
[INPUT]
    Name              tail
    Path              /data/dionaea/log/dionaea.json
    Parser            json
    Tag               tpot.dionaea
    Refresh_Interval  5
    Read_from_Head    false
    Skip_Long_Lines   On

# Suricata Network IDS
[INPUT]
    Name              tail
    Path              /data/suricata/log/eve.json
    Parser            json
    Tag               tpot.suricata
    Refresh_Interval  5
    Read_from_Head    false
    Skip_Long_Lines   On

# Honeytrap Network Honeypot
[INPUT]
    Name              tail
    Path              /data/honeytrap/log/honeytrap.json
    Parser            json
    Tag               tpot.honeytrap
    Refresh_Interval  5
    Read_from_Head    false
    Skip_Long_Lines   On

# Elasticpot Elasticsearch Honeypot
[INPUT]
    Name              tail
    Path              /data/elasticpot/log/elasticpot.json
    Parser            json
    Tag               tpot.elasticpot
    Refresh_Interval  5
    Read_from_Head    false
    Skip_Long_Lines   On

# Mailoney SMTP Honeypot
[INPUT]
    Name              tail
    Path              /data/mailoney/log/commands.log
    Tag               tpot.mailoney
    Refresh_Interval  5
    Read_from_Head    false
    Skip_Long_Lines   On

# T-Pot System Logs
[INPUT]
    Name              tail
    Path              /var/log/tpot.log
    Tag               tpot.system
    Refresh_Interval  10
    Read_from_Head    false
    Skip_Long_Lines   On

# Output to AWS Mini-XDR Backend
[OUTPUT]
    Name  http
    Match tpot.*
    Host  $BACKEND_IP
    Port  8000
    URI   /ingest/multi
    Format json_stream
    Header Authorization Bearer tpot-honeypot-key
    Header Content-Type application/json
    Header X-Source tpot-aws
    json_date_key timestamp
    json_date_format iso8601
    Retry_Limit 5
    tls Off

# Backup output to local files (for debugging)
[OUTPUT]
    Name  file
    Match tpot.*
    Path  /data/backup/
    File  tpot-backup.log
    Format json_lines
EOF

    log "Fluent Bit configuration created"
}

# Backup existing TPOT configuration
backup_tpot_config() {
    log "Backing up existing TPOT configuration..."
    
    ssh -i "~/.ssh/${KEY_NAME}.pem" -p "$TPOT_SSH_PORT" -o StrictHostKeyChecking=yes -o UserKnownHostsFile=~/.ssh/known_hosts \
        "$TPOT_USER@$TPOT_HOST" << 'BACKUP_REMOTE'
        # Create backup directory
        sudo mkdir -p /opt/tpot/backup/$(date +%Y%m%d_%H%M%S)
        
        # Backup current Fluent Bit config
        if [ -f /opt/tpot/etc/fluent-bit/fluent-bit.conf ]; then
            sudo cp /opt/tpot/etc/fluent-bit/fluent-bit.conf /opt/tpot/backup/$(date +%Y%m%d_%H%M%S)/
            echo "Configuration backed up successfully"
        else
            echo "No existing Fluent Bit configuration found"
        fi
BACKUP_REMOTE
    
    log "Backup completed"
}

# Upload new configuration to TPOT
upload_tpot_config() {
    log "Uploading new configuration to TPOT..."
    
    # Upload the new configuration
    scp -i "~/.ssh/${KEY_NAME}.pem" -P "$TPOT_SSH_PORT" -o StrictHostKeyChecking=yes -o UserKnownHostsFile=~/.ssh/known_hosts \
        "/tmp/fluent-bit-tpot-aws.conf" "$TPOT_USER@$TPOT_HOST":/tmp/
    
    # Install the configuration
    ssh -i "~/.ssh/${KEY_NAME}.pem" -p "$TPOT_SSH_PORT" -o StrictHostKeyChecking=yes -o UserKnownHostsFile=~/.ssh/known_hosts \
        "$TPOT_USER@$TPOT_HOST" << 'INSTALL_CONFIG'
        # Install new configuration
        sudo cp /tmp/fluent-bit-tpot-aws.conf /opt/tpot/etc/fluent-bit/fluent-bit.conf
        sudo chmod 644 /opt/tpot/etc/fluent-bit/fluent-bit.conf
        sudo chown root:root /opt/tpot/etc/fluent-bit/fluent-bit.conf
        
        # Create backup directory for logs
        sudo mkdir -p /data/backup
        sudo chown tpot:tpot /data/backup
        
        echo "Configuration installed successfully"
INSTALL_CONFIG
    
    log "Configuration uploaded and installed"
}

# Restart TPOT services
restart_tpot_services() {
    log "Restarting TPOT services..."
    
    ssh -i "~/.ssh/${KEY_NAME}.pem" -p "$TPOT_SSH_PORT" -o StrictHostKeyChecking=yes -o UserKnownHostsFile=~/.ssh/known_hosts \
        "$TPOT_USER@$TPOT_HOST" << 'RESTART_SERVICES'
        # Restart Fluent Bit service
        sudo docker restart $(sudo docker ps -q --filter "name=fluent")
        
        # Check if restart was successful
        sleep 5
        if sudo docker ps | grep -q fluent; then
            echo "Fluent Bit restarted successfully"
        else
            echo "Warning: Fluent Bit may not have restarted properly"
            sudo docker ps
        fi
        
        # Show recent logs
        echo "Recent Fluent Bit logs:"
        sudo docker logs $(sudo docker ps -q --filter "name=fluent") --tail 20
RESTART_SERVICES
    
    log "TPOT services restarted"
}

# Test data flow
test_data_flow() {
    log "Testing data flow from TPOT to Mini-XDR..."
    
    # Generate some test events on TPOT
    ssh -i "~/.ssh/${KEY_NAME}.pem" -p "$TPOT_SSH_PORT" -o StrictHostKeyChecking=yes -o UserKnownHostsFile=~/.ssh/known_hosts \
        "$TPOT_USER@$TPOT_HOST" << 'GENERATE_TEST_DATA'
        # Generate a test SSH connection to Cowrie
        echo "Generating test data..."
        timeout 5 nc localhost 2222 || true
        
        # Wait for logs to be processed
        sleep 10
        
        echo "Test data generation completed"
GENERATE_TEST_DATA
    
    # Check if data is received by Mini-XDR
    log "Checking if data is received by Mini-XDR..."
    sleep 10
    
    local events_url="http://$BACKEND_IP:8000/events"
    local event_count
    event_count=$(curl -s "$events_url" | jq '. | length')
    
    log "Current event count in Mini-XDR: $event_count"
    
    if [ "$event_count" -gt 0 ]; then
        log "✅ Data flow is working! Events are being received."
    else
        warn "⚠️  No events received yet. This may be normal for a new deployment."
    fi
}

# Update security groups for TPOT communication
update_security_groups() {
    log "Updating security groups for TPOT communication..."
    
    # Get TPOT instance ID and security group
    local tpot_instance_id
    tpot_instance_id=$(aws ec2 describe-instances \
        --filters "Name=ip-address,Values=$TPOT_HOST" \
        --query 'Reservations[0].Instances[0].InstanceId' \
        --output text)
    
    if [ "$tpot_instance_id" = "None" ] || [ -z "$tpot_instance_id" ]; then
        warn "Could not find TPOT instance, skipping security group update"
        return
    fi
    
    local tpot_sg_id
    tpot_sg_id=$(aws ec2 describe-instances \
        --instance-ids "$tpot_instance_id" \
        --query 'Reservations[0].Instances[0].SecurityGroups[0].GroupId' \
        --output text)
    
    # Get Mini-XDR security group ID
    local backend_sg_id
    backend_sg_id=$(aws cloudformation describe-stacks \
        --stack-name "$STACK_NAME" \
        --region "$REGION" \
        --query 'Stacks[0].Outputs[?OutputKey==`SecurityGroupId`].OutputValue' \
        --output text 2>/dev/null || echo "")
    
    if [ -n "$backend_sg_id" ] && [ "$backend_sg_id" != "None" ]; then
        # Allow TPOT to access Mini-XDR backend
        aws ec2 authorize-security-group-ingress \
            --group-id "$backend_sg_id" \
            --protocol tcp \
            --port 8000 \
            --source-group "$tpot_sg_id" \
            --region "$REGION" 2>/dev/null || log "Security group rule may already exist"
        
        log "Security groups updated for TPOT → Mini-XDR communication"
    else
        warn "Could not find Mini-XDR security group, manual configuration may be needed"
    fi
}

# Create monitoring script
create_monitoring_script() {
    log "Creating monitoring script..."
    
    cat > "/tmp/monitor-tpot-connection.sh" << 'MONITOR_SCRIPT'
#!/bin/bash

# TPOT Connection Monitoring Script
# Run this on your local machine to monitor the data flow

BACKEND_IP="$1"
if [ -z "$BACKEND_IP" ]; then
    echo "Usage: $0 <backend-ip>"
    exit 1
fi

echo "Monitoring TPOT → Mini-XDR data flow..."
echo "Backend: $BACKEND_IP"
echo "Press Ctrl+C to stop"
echo ""

while true; do
    # Get current event count
    EVENTS=$(curl -s "http://$BACKEND_IP:8000/events" | jq '. | length' 2>/dev/null || echo "0")
    
    # Get health status
    HEALTH=$(curl -s "http://$BACKEND_IP:8000/health" | jq -r '.status' 2>/dev/null || echo "unknown")
    
    # Display status
    echo "$(date): Events: $EVENTS, Health: $HEALTH"
    
    sleep 30
done
MONITOR_SCRIPT
    
    chmod +x "/tmp/monitor-tpot-connection.sh"
    cp "/tmp/monitor-tpot-connection.sh" "$HOME/monitor-tpot-connection.sh"
    
    log "Monitoring script created: $HOME/monitor-tpot-connection.sh"
}

# Main function
main() {
    log "Starting TPOT → AWS Mini-XDR connection configuration..."
    
    get_backend_ip
    test_backend_connectivity
    test_tpot_connectivity
    backup_tpot_config
    create_fluent_bit_config
    upload_tpot_config
    restart_tpot_services
    update_security_groups
    test_data_flow
    create_monitoring_script
    
    log "✅ TPOT → AWS Mini-XDR connection configured successfully!"
    log ""
    log "🔗 Data Flow: TPOT ($TPOT_HOST) → Mini-XDR ($BACKEND_IP:8000)"
    log "📊 Monitor: $HOME/monitor-tpot-connection.sh $BACKEND_IP"
    log "🌐 API: http://$BACKEND_IP:8000/events"
    log ""
    log "The system is now configured for real attack data collection!"
    log "You should start seeing events from TPOT appear in Mini-XDR shortly."
}

# Run main function
main "$@"
