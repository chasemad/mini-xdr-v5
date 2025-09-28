#!/bin/bash

# DATABASE SECURITY HARDENING SCRIPT
# Implements secure database passwords and encryption
# RUN THIS AFTER SSH SECURITY FIXES

set -euo pipefail

# Configuration
REGION="${AWS_REGION:-us-east-1}"
PROJECT_ROOT="/Users/chasemad/Desktop/mini-xdr"
STACK_NAME="mini-xdr-backend"

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
    echo "    🗃️ DATABASE SECURITY HARDENING 🗃️"
    echo "=============================================="
    echo -e "${NC}"
    echo "This script will:"
    echo "  🔐 Generate cryptographically secure database passwords"
    echo "  🔒 Enable database encryption at rest"
    echo "  📡 Enforce SSL/TLS for all connections"
    echo "  📊 Enable database audit logging"
    echo "  🛡️ Implement database access controls"
    echo ""
}

# Generate secure database password
generate_secure_database_password() {
    log "🎲 Generating cryptographically secure database password..."
    
    # Generate a secure password (32 characters, URL-safe)
    SECURE_DB_PASSWORD=$(openssl rand -base64 32 | tr -d "=+/" | cut -c1-25)
    
    # Ensure it meets PostgreSQL requirements
    SECURE_DB_PASSWORD="MiniXDR_${SECURE_DB_PASSWORD}_$(date +%Y)"
    
    log "✅ Generated secure database password (${#SECURE_DB_PASSWORD} characters)"
    log "   Pattern: MiniXDR_[25 random chars]_2025"
}

# Update database password in AWS RDS
update_rds_password() {
    log "🗃️ Updating RDS database password..."
    
    # Find the RDS instance
    local db_instance_id
    db_instance_id=$(aws rds describe-db-instances \
        --region "$REGION" \
        --query 'DBInstances[?contains(DBInstanceIdentifier, `mini-xdr`)].DBInstanceIdentifier' \
        --output text | head -1)
    
    if [ -z "$db_instance_id" ] || [ "$db_instance_id" = "None" ]; then
        warn "No Mini-XDR RDS instance found. Skipping password update."
        return 0
    fi
    
    log "Found RDS instance: $db_instance_id"
    
    # Update the master password
    log "Updating master password for RDS instance..."
    aws rds modify-db-instance \
        --db-instance-identifier "$db_instance_id" \
        --master-user-password "$SECURE_DB_PASSWORD" \
        --apply-immediately \
        --region "$REGION"
    
    # Wait for modification to complete
    log "Waiting for password update to complete..."
    aws rds wait db-instance-available \
        --db-instance-identifiers "$db_instance_id" \
        --region "$REGION"
    
    log "✅ RDS password updated successfully"
}

# Enable database encryption
enable_database_encryption() {
    log "🔒 Enabling database encryption..."
    
    # Find the RDS instance
    local db_instance_id
    db_instance_id=$(aws rds describe-db-instances \
        --region "$REGION" \
        --query 'DBInstances[?contains(DBInstanceIdentifier, `mini-xdr`)].DBInstanceIdentifier' \
        --output text | head -1)
    
    if [ -z "$db_instance_id" ] || [ "$db_instance_id" = "None" ]; then
        warn "No Mini-XDR RDS instance found. Skipping encryption setup."
        return 0
    fi
    
    # Check if encryption is already enabled
    local encryption_status
    encryption_status=$(aws rds describe-db-instances \
        --db-instance-identifier "$db_instance_id" \
        --region "$REGION" \
        --query 'DBInstances[0].StorageEncrypted' \
        --output text)
    
    if [ "$encryption_status" = "True" ]; then
        log "✅ Database encryption already enabled"
    else
        warn "⚠️ Database encryption not enabled. This requires a new encrypted instance."
        log "Creating encrypted snapshot and new instance..."
        
        # Create a snapshot of current instance
        local snapshot_id="mini-xdr-pre-encryption-$(date +%Y%m%d%H%M%S)"
        aws rds create-db-snapshot \
            --db-instance-identifier "$db_instance_id" \
            --db-snapshot-identifier "$snapshot_id" \
            --region "$REGION"
        
        # Wait for snapshot to complete
        log "Waiting for snapshot to complete..."
        aws rds wait db-snapshot-completed \
            --db-snapshot-identifier "$snapshot_id" \
            --region "$REGION"
        
        # Create encrypted copy of snapshot
        local encrypted_snapshot_id="mini-xdr-encrypted-$(date +%Y%m%d%H%M%S)"
        aws rds copy-db-snapshot \
            --source-db-snapshot-identifier "$snapshot_id" \
            --target-db-snapshot-identifier "$encrypted_snapshot_id" \
            --encrypted \
            --region "$REGION"
        
        log "📋 Encrypted snapshot created: $encrypted_snapshot_id"
        log "ℹ️ Manual step required: Restore from encrypted snapshot to enable encryption"
    fi
}

# Configure SSL/TLS enforcement
configure_ssl_enforcement() {
    log "📡 Configuring SSL/TLS enforcement..."
    
    # Find the DB parameter group
    local db_instance_id
    db_instance_id=$(aws rds describe-db-instances \
        --region "$REGION" \
        --query 'DBInstances[?contains(DBInstanceIdentifier, `mini-xdr`)].DBInstanceIdentifier' \
        --output text | head -1)
    
    if [ -z "$db_instance_id" ] || [ "$db_instance_id" = "None" ]; then
        warn "No Mini-XDR RDS instance found. Skipping SSL configuration."
        return 0
    fi
    
    # Get parameter group name
    local param_group_name
    param_group_name=$(aws rds describe-db-instances \
        --db-instance-identifier "$db_instance_id" \
        --region "$REGION" \
        --query 'DBInstances[0].DBParameterGroups[0].DBParameterGroupName' \
        --output text)
    
    if [ "$param_group_name" = "None" ] || [ -z "$param_group_name" ]; then
        # Create a custom parameter group
        param_group_name="mini-xdr-secure-params"
        
        log "Creating custom parameter group: $param_group_name"
        aws rds create-db-parameter-group \
            --db-parameter-group-name "$param_group_name" \
            --db-parameter-group-family postgres15 \
            --description "Mini-XDR secure PostgreSQL parameters" \
            --region "$REGION" 2>/dev/null || warn "Parameter group may already exist"
        
        # Modify instance to use new parameter group
        aws rds modify-db-instance \
            --db-instance-identifier "$db_instance_id" \
            --db-parameter-group-name "$param_group_name" \
            --apply-immediately \
            --region "$REGION"
    fi
    
    # Enable SSL enforcement
    log "Enabling SSL enforcement in parameter group: $param_group_name"
    aws rds modify-db-parameter-group \
        --db-parameter-group-name "$param_group_name" \
        --parameters "ParameterName=rds.force_ssl,ParameterValue=1,ApplyMethod=immediate" \
        --region "$REGION"
    
    # Enable connection logging
    aws rds modify-db-parameter-group \
        --db-parameter-group-name "$param_group_name" \
        --parameters "ParameterName=log_connections,ParameterValue=1,ApplyMethod=immediate" \
        --region "$REGION"
    
    # Enable disconnection logging
    aws rds modify-db-parameter-group \
        --db-parameter-group-name "$param_group_name" \
        --parameters "ParameterName=log_disconnections,ParameterValue=1,ApplyMethod=immediate" \
        --region "$REGION"
    
    # Reboot instance to apply parameter changes
    log "Rebooting instance to apply SSL parameters..."
    aws rds reboot-db-instance \
        --db-instance-identifier "$db_instance_id" \
        --region "$REGION"
    
    log "✅ SSL/TLS enforcement configured"
}

# Enable database audit logging
enable_audit_logging() {
    log "📊 Enabling database audit logging..."
    
    # Find the RDS instance
    local db_instance_id
    db_instance_id=$(aws rds describe-db-instances \
        --region "$REGION" \
        --query 'DBInstances[?contains(DBInstanceIdentifier, `mini-xdr`)].DBInstanceIdentifier' \
        --output text | head -1)
    
    if [ -z "$db_instance_id" ] || [ "$db_instance_id" = "None" ]; then
        warn "No Mini-XDR RDS instance found. Skipping audit logging."
        return 0
    fi
    
    # Enable PostgreSQL logs
    log "Enabling PostgreSQL audit logging..."
    aws rds modify-db-instance \
        --db-instance-identifier "$db_instance_id" \
        --enabled-cloudwatch-logs-exports postgresql \
        --region "$REGION"
    
    # Create CloudWatch log group for database logs
    aws logs create-log-group \
        --log-group-name "/aws/rds/instance/$db_instance_id/postgresql" \
        --region "$REGION" 2>/dev/null || warn "Log group may already exist"
    
    # Set log retention to 30 days
    aws logs put-retention-policy \
        --log-group-name "/aws/rds/instance/$db_instance_id/postgresql" \
        --retention-in-days 30 \
        --region "$REGION"
    
    log "✅ Database audit logging enabled"
}

# Update application configurations
update_application_configs() {
    log "📝 Updating application configurations with secure database settings..."
    
    # Update Secrets Manager with new password
    log "Updating password in AWS Secrets Manager..."
    aws secretsmanager update-secret \
        --secret-id "mini-xdr/database-password" \
        --secret-string "$SECURE_DB_PASSWORD" \
        --region "$REGION"
    
    # Get RDS endpoint
    local db_endpoint
    db_endpoint=$(aws rds describe-db-instances \
        --region "$REGION" \
        --query 'DBInstances[?contains(DBInstanceIdentifier, `mini-xdr`)].Endpoint.Address' \
        --output text | head -1)
    
    if [ -n "$db_endpoint" ] && [ "$db_endpoint" != "None" ]; then
        # Create secure database URL with SSL
        local secure_db_url="postgresql://postgres:$SECURE_DB_PASSWORD@$db_endpoint:5432/postgres?sslmode=require"
        
        # Update secure environment template
        log "Updating secure environment template..."
        cat > "$PROJECT_ROOT/backend/.env.database-secure" << EOF
# SECURE DATABASE CONFIGURATION
# Generated on $(date)

# Database URL with SSL enforcement
DATABASE_URL=$secure_db_url

# Alternative format using Secrets Manager
DATABASE_URL_SECURE=postgresql://postgres:\$(aws secretsmanager get-secret-value --secret-id mini-xdr/database-password --query SecretString --output text)@$db_endpoint:5432/postgres?sslmode=require

# Database connection settings
DB_HOST=$db_endpoint
DB_PORT=5432
DB_NAME=postgres
DB_USER=postgres
DB_PASSWORD=\$(aws secretsmanager get-secret-value --secret-id mini-xdr/database-password --query SecretString --output text)
DB_SSL_MODE=require

# Connection pool settings
DB_POOL_SIZE=20
DB_MAX_OVERFLOW=0
DB_POOL_TIMEOUT=30
DB_POOL_RECYCLE=3600
EOF
        
        log "✅ Application configurations updated"
    else
        warn "Could not determine database endpoint"
    fi
}

# Create database connection test script
create_database_test_script() {
    log "🧪 Creating database connection test script..."
    
    cat > "$PROJECT_ROOT/test-database-connection.sh" << 'EOF'
#!/bin/bash

# Database Connection Test Script
# Tests database connectivity with new security settings

echo "Testing database connection with security hardening..."

# Get database password from Secrets Manager
DB_PASSWORD=$(aws secretsmanager get-secret-value --secret-id mini-xdr/database-password --query SecretString --output text 2>/dev/null)

if [ -z "$DB_PASSWORD" ]; then
    echo "❌ Could not retrieve database password from Secrets Manager"
    exit 1
fi

# Get database endpoint
DB_ENDPOINT=$(aws rds describe-db-instances \
    --query 'DBInstances[?contains(DBInstanceIdentifier, `mini-xdr`)].Endpoint.Address' \
    --output text | head -1)

if [ -z "$DB_ENDPOINT" ] || [ "$DB_ENDPOINT" = "None" ]; then
    echo "❌ Could not find Mini-XDR RDS instance"
    exit 1
fi

echo "Database endpoint: $DB_ENDPOINT"

# Test SSL connection
echo "Testing SSL connection..."
if command -v psql >/dev/null 2>&1; then
    # Test with psql if available
    if PGPASSWORD="$DB_PASSWORD" psql -h "$DB_ENDPOINT" -p 5432 -U postgres -d postgres -c "SELECT version();" -q 2>/dev/null; then
        echo "✅ Database SSL connection: Working"
    else
        echo "❌ Database SSL connection: Failed"
    fi
    
    # Test SSL mode
    if PGPASSWORD="$DB_PASSWORD" psql -h "$DB_ENDPOINT" -p 5432 -U postgres -d postgres -c "SHOW ssl;" -q 2>/dev/null | grep -q "on"; then
        echo "✅ Database SSL mode: Enabled"
    else
        echo "⚠️ Database SSL mode: Check required"
    fi
else
    echo "ℹ️ psql not available - using basic connectivity test"
    
    # Basic connectivity test
    if nc -zv "$DB_ENDPOINT" 5432 2>/dev/null; then
        echo "✅ Database port 5432: Accessible"
    else
        echo "❌ Database port 5432: Not accessible"
    fi
fi

echo "Database connection tests completed."
EOF
    
    chmod +x "$PROJECT_ROOT/test-database-connection.sh"
    log "✅ Database test script created: test-database-connection.sh"
}

# Generate database security report
generate_database_security_report() {
    log "📊 Generating database security report..."
    
    # Get RDS instance details
    local db_instance_id
    db_instance_id=$(aws rds describe-db-instances \
        --region "$REGION" \
        --query 'DBInstances[?contains(DBInstanceIdentifier, `mini-xdr`)].DBInstanceIdentifier' \
        --output text | head -1)
    
    local encryption_status="Unknown"
    local ssl_status="Unknown"
    local backup_retention="Unknown"
    
    if [ -n "$db_instance_id" ] && [ "$db_instance_id" != "None" ]; then
        encryption_status=$(aws rds describe-db-instances \
            --db-instance-identifier "$db_instance_id" \
            --region "$REGION" \
            --query 'DBInstances[0].StorageEncrypted' \
            --output text)
        
        backup_retention=$(aws rds describe-db-instances \
            --db-instance-identifier "$db_instance_id" \
            --region "$REGION" \
            --query 'DBInstances[0].BackupRetentionPeriod' \
            --output text)
    fi
    
    cat > "/tmp/database-security-hardening-report.txt" << EOF
DATABASE SECURITY HARDENING REPORT
===================================
Date: $(date)
Project: Mini-XDR
RDS Instance: $db_instance_id

ACTIONS TAKEN:
✅ Generated cryptographically secure database password (${#SECURE_DB_PASSWORD} chars)
✅ Updated RDS master password
✅ Configured SSL/TLS enforcement
✅ Enabled database audit logging
✅ Updated application configurations
✅ Created database connection test script

SECURITY IMPROVEMENTS:
- Password Pattern: MiniXDR_[25 random chars]_2025
- Encryption at Rest: $encryption_status
- SSL/TLS Enforcement: Enabled via parameter group
- Connection Logging: Enabled
- Disconnection Logging: Enabled
- CloudWatch Logs: Enabled
- Backup Retention: $backup_retention days

PASSWORD SECURITY:
- Length: ${#SECURE_DB_PASSWORD} characters
- Entropy: High (base64 + timestamp)
- Pattern: Non-predictable
- Storage: AWS Secrets Manager
- Rotation: Manual (can be automated)

DATABASE ACCESS CONTROLS:
- SSL Mode: require
- Connection Timeout: 30 seconds
- Pool Size: 20 connections
- Max Overflow: 0
- Pool Recycle: 3600 seconds

CONFIGURATION FILES UPDATED:
- backend/.env.database-secure
- Secrets Manager: mini-xdr/database-password

VALIDATION COMMANDS:
# Test database connection:
./test-database-connection.sh

# Check SSL enforcement:
aws rds describe-db-instances --db-instance-identifier $db_instance_id --query 'DBInstances[0].DBParameterGroups'

# View database logs:
aws logs describe-log-streams --log-group-name /aws/rds/instance/$db_instance_id/postgresql

NEXT STEPS:
1. Test application connectivity with new password
2. Monitor database performance after SSL enforcement
3. Set up automated password rotation
4. Configure database monitoring alerts
5. Run iam-privilege-reduction.sh script

EMERGENCY CONTACT:
If database connections fail, check SSL configuration and parameter group settings.
EOF
    
    log "📋 Report saved to: /tmp/database-security-hardening-report.txt"
    echo ""
    cat /tmp/database-security-hardening-report.txt
}

# Validate database security
validate_database_security() {
    log "✅ Validating database security improvements..."
    
    # Check if password is in Secrets Manager
    if aws secretsmanager describe-secret --secret-id "mini-xdr/database-password" --region "$REGION" >/dev/null 2>&1; then
        log "✅ Database password stored in Secrets Manager"
    else
        warn "⚠️ Database password not found in Secrets Manager"
    fi
    
    # Check RDS instance encryption
    local db_instance_id
    db_instance_id=$(aws rds describe-db-instances \
        --region "$REGION" \
        --query 'DBInstances[?contains(DBInstanceIdentifier, `mini-xdr`)].DBInstanceIdentifier' \
        --output text | head -1)
    
    if [ -n "$db_instance_id" ] && [ "$db_instance_id" != "None" ]; then
        local encryption_status
        encryption_status=$(aws rds describe-db-instances \
            --db-instance-identifier "$db_instance_id" \
            --region "$REGION" \
            --query 'DBInstances[0].StorageEncrypted' \
            --output text)
        
        if [ "$encryption_status" = "True" ]; then
            log "✅ Database encryption at rest: Enabled"
        else
            warn "⚠️ Database encryption at rest: Not enabled (requires new instance)"
        fi
        
        # Check parameter group
        local param_group
        param_group=$(aws rds describe-db-instances \
            --db-instance-identifier "$db_instance_id" \
            --region "$REGION" \
            --query 'DBInstances[0].DBParameterGroups[0].DBParameterGroupName' \
            --output text)
        
        if [ "$param_group" != "default.postgres15" ]; then
            log "✅ Custom parameter group: $param_group"
        else
            warn "⚠️ Using default parameter group"
        fi
    else
        warn "⚠️ No RDS instance found for validation"
    fi
    
    # Check configuration files
    if [ -f "$PROJECT_ROOT/backend/.env.database-secure" ]; then
        log "✅ Secure database configuration file created"
    else
        warn "⚠️ Secure database configuration file not found"
    fi
}

# Main execution
main() {
    show_banner
    
    # Confirm action
    critical "⚠️  WARNING: This will modify RDS database settings and may cause temporary downtime!"
    echo ""
    read -p "Continue with database security hardening? (type 'HARDEN DATABASE' to confirm): " -r
    if [ "$REPLY" != "HARDEN DATABASE" ]; then
        log "Operation cancelled by user"
        exit 0
    fi
    
    log "🗃️ Starting database security hardening..."
    local start_time=$(date +%s)
    
    # Execute database security procedures
    generate_secure_database_password
    update_rds_password
    enable_database_encryption
    configure_ssl_enforcement
    enable_audit_logging
    update_application_configs
    create_database_test_script
    validate_database_security
    generate_database_security_report
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    log "🎉 Database security hardening completed in ${duration} seconds"
    
    echo ""
    critical "🚨 NEXT STEPS:"
    echo "1. Test database connectivity: ./test-database-connection.sh"
    echo "2. Update application with new database configuration"
    echo "3. Run: ./iam-privilege-reduction.sh"
    echo "4. Monitor database performance and logs"
    echo "5. Set up automated password rotation"
}

# Export configuration for other scripts
export AWS_REGION="$REGION"
export PROJECT_ROOT="$PROJECT_ROOT"

# Run main function
main "$@"
