#!/bin/bash
# 🔑 Mini-XDR SSH Key Rotation System
# Automated SSH key rotation for enhanced security

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
AWS_REGION="${AWS_REGION:-us-east-1}"

# SSH Key Configuration
SSH_KEY_PATH="$HOME/.ssh/honey_key"
SSH_KEY_NEW_PATH="$HOME/.ssh/honey_key-new"
SSH_KEY_OLD_PATH="$HOME/.ssh/honey_key-old"

# T-Pot Configuration
TPOT_IP="${TPOT_IP:-34.193.101.171}"
TPOT_USER="${TPOT_USER:-tsec}"
TPOT_PORT="${TPOT_PORT:-22}"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

# Logging
log() {
    echo -e "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

success() {
    log "${GREEN}✅ $1${NC}"
}

warning() {
    log "${YELLOW}⚠️  $1${NC}"
}

error() {
    log "${RED}❌ ERROR: $1${NC}"
}

info() {
    log "${BLUE}ℹ️  $1${NC}"
}

# Help function
show_help() {
    echo -e "${BLUE}Mini-XDR SSH Key Rotation System${NC}"
    echo
    echo "USAGE:"
    echo "  $0 [COMMAND] [OPTIONS]"
    echo
    echo "COMMANDS:"
    echo "  rotate    Generate new SSH key and rotate with T-Pot"
    echo "  test      Test current SSH key connectivity"
    echo "  backup    Backup current SSH key"
    echo "  restore   Restore from backup"
    echo "  status    Show key status and last rotation"
    echo
    echo "OPTIONS:"
    echo "  --force   Skip confirmation prompts"
    echo "  --help    Show this help"
    echo
    echo "EXAMPLES:"
    echo "  $0 rotate              # Rotate SSH keys with confirmation"
    echo "  $0 rotate --force      # Rotate without confirmation"
    echo "  $0 test                # Test current key"
    echo "  $0 status              # Show key status"
}

# Test SSH connectivity
test_ssh_connectivity() {
    local key_path="$1"
    local description="$2"
    
    info "Testing SSH connectivity with $description..."
    
    if ssh -i "$key_path" \
           -o StrictHostKeyChecking=no \
           -o ConnectTimeout=10 \
           -o BatchMode=yes \
           "$TPOT_USER@$TPOT_IP" \
           -p "$TPOT_PORT" \
           "echo 'SSH test successful'" >/dev/null 2>&1; then
        success "SSH connectivity with $description: OK"
        return 0
    else
        error "SSH connectivity with $description: FAILED"
        return 1
    fi
}

# Generate new SSH key
generate_new_key() {
    info "Generating new SSH key pair..."
    
    # Remove old new key if it exists
    rm -f "$SSH_KEY_NEW_PATH" "$SSH_KEY_NEW_PATH.pub"
    
    # Generate new ed25519 key (more secure than RSA)
    ssh-keygen -t ed25519 \
               -f "$SSH_KEY_NEW_PATH" \
               -N "" \
               -C "mini-xdr-tpot-$(date +%Y%m%d)" \
               >/dev/null 2>&1
    
    chmod 600 "$SSH_KEY_NEW_PATH"
    chmod 644 "$SSH_KEY_NEW_PATH.pub"
    
    success "New SSH key generated: $SSH_KEY_NEW_PATH"
}

# Add new key to T-Pot authorized_keys
add_new_key_to_tpot() {
    info "Adding new SSH key to T-Pot authorized_keys..."
    
    local new_public_key=$(cat "$SSH_KEY_NEW_PATH.pub")
    
    # Add new key to authorized_keys (keeping old key for now)
    ssh -i "$SSH_KEY_PATH" \
        -o StrictHostKeyChecking=no \
        -p "$TPOT_PORT" \
        "$TPOT_USER@$TPOT_IP" \
        "echo '$new_public_key' >> ~/.ssh/authorized_keys" || {
        error "Failed to add new key to T-Pot"
        return 1
    }
    
    success "New SSH key added to T-Pot authorized_keys"
}

# Remove old key from T-Pot authorized_keys
remove_old_key_from_tpot() {
    info "Removing old SSH key from T-Pot authorized_keys..."
    
    local old_public_key=$(cat "$SSH_KEY_PATH.pub" 2>/dev/null || echo "")
    
    if [[ -n "$old_public_key" ]]; then
        # Remove old key from authorized_keys
        ssh -i "$SSH_KEY_NEW_PATH" \
            -o StrictHostKeyChecking=no \
            -p "$TPOT_PORT" \
            "$TPOT_USER@$TPOT_IP" \
            "grep -v '$old_public_key' ~/.ssh/authorized_keys > ~/.ssh/authorized_keys.tmp && mv ~/.ssh/authorized_keys.tmp ~/.ssh/authorized_keys" || {
            warning "Failed to remove old key from T-Pot (this is usually OK)"
        }
        
        success "Old SSH key removed from T-Pot authorized_keys"
    else
        warning "Could not read old public key for removal"
    fi
}

# Rotate SSH keys locally
rotate_keys_locally() {
    info "Rotating SSH keys locally..."
    
    # Backup current key
    if [[ -f "$SSH_KEY_PATH" ]]; then
        cp "$SSH_KEY_PATH" "$SSH_KEY_OLD_PATH"
        cp "$SSH_KEY_PATH.pub" "$SSH_KEY_OLD_PATH.pub" 2>/dev/null || true
        success "Current key backed up to: $SSH_KEY_OLD_PATH"
    fi
    
    # Move new key to current position
    mv "$SSH_KEY_NEW_PATH" "$SSH_KEY_PATH"
    mv "$SSH_KEY_NEW_PATH.pub" "$SSH_KEY_PATH.pub"
    
    success "SSH keys rotated successfully"
}

# Update key status
update_key_status() {
    local status_file="$HOME/.ssh/mini-xdr-key-status"
    
    cat > "$status_file" << EOF
# Mini-XDR SSH Key Status
LAST_ROTATION=$(date -u +"%Y-%m-%d %H:%M:%S UTC")
KEY_TYPE=ed25519
KEY_PATH=$SSH_KEY_PATH
TPOT_IP=$TPOT_IP
TPOT_USER=$TPOT_USER
TPOT_PORT=$TPOT_PORT
EOF

    success "Key status updated: $status_file"
}

# Show key status
show_key_status() {
    local status_file="$HOME/.ssh/mini-xdr-key-status"
    
    echo -e "${BLUE}Mini-XDR SSH Key Status:${NC}"
    echo
    
    if [[ -f "$status_file" ]]; then
        source "$status_file"
        echo -e "Last Rotation: ${GREEN}$LAST_ROTATION${NC}"
        echo -e "Key Type: $KEY_TYPE"
        echo -e "Key Path: $KEY_PATH"
        echo -e "T-Pot IP: $TPOT_IP"
        echo -e "T-Pot User: $TPOT_USER"
        echo -e "T-Pot Port: $TPOT_PORT"
    else
        echo -e "${YELLOW}No rotation history found${NC}"
    fi
    
    echo
    if [[ -f "$SSH_KEY_PATH" ]]; then
        echo -e "Current Key: ${GREEN}EXISTS${NC}"
        echo -e "Key Fingerprint: $(ssh-keygen -lf "$SSH_KEY_PATH" 2>/dev/null | awk '{print $2}' || echo 'Unknown')"
    else
        echo -e "Current Key: ${RED}NOT FOUND${NC}"
    fi
    
    if [[ -f "$SSH_KEY_OLD_PATH" ]]; then
        echo -e "Backup Key: ${GREEN}EXISTS${NC}"
    else
        echo -e "Backup Key: ${YELLOW}NOT FOUND${NC}"
    fi
}

# Main rotation function
rotate_ssh_keys() {
    local force_mode="$1"
    
    echo -e "${BLUE}Mini-XDR SSH Key Rotation${NC}"
    echo "=========================="
    
    # Check if current key exists
    if [[ ! -f "$SSH_KEY_PATH" ]]; then
        error "Current SSH key not found: $SSH_KEY_PATH"
        return 1
    fi
    
    # Test current key connectivity
    if ! test_ssh_connectivity "$SSH_KEY_PATH" "current key"; then
        error "Current SSH key is not working. Cannot proceed with rotation."
        return 1
    fi
    
    # Confirmation
    if [[ "$force_mode" != "true" ]]; then
        echo
        warning "This will rotate your SSH keys for T-Pot access."
        warning "Make sure you have tested the current key connectivity first."
        echo -n "Continue with SSH key rotation? (yes/no): "
        read -r confirmation
        if [[ "$confirmation" != "yes" ]]; then
            info "SSH key rotation cancelled."
            return 0
        fi
    fi
    
    # Rotation process
    generate_new_key || return 1
    add_new_key_to_tpot || return 1
    
    # Test new key before committing
    if test_ssh_connectivity "$SSH_KEY_NEW_PATH" "new key"; then
        remove_old_key_from_tpot
        rotate_keys_locally
        update_key_status
        
        success "🎉 SSH key rotation completed successfully!"
        warning "Old key backed up to: $SSH_KEY_OLD_PATH"
    else
        error "New key test failed. Rotation aborted."
        rm -f "$SSH_KEY_NEW_PATH" "$SSH_KEY_NEW_PATH.pub"
        return 1
    fi
}

# Backup current key
backup_key() {
    if [[ -f "$SSH_KEY_PATH" ]]; then
        cp "$SSH_KEY_PATH" "$SSH_KEY_OLD_PATH"
        cp "$SSH_KEY_PATH.pub" "$SSH_KEY_OLD_PATH.pub" 2>/dev/null || true
        success "SSH key backed up to: $SSH_KEY_OLD_PATH"
    else
        error "No current SSH key found to backup"
        return 1
    fi
}

# Restore from backup
restore_key() {
    if [[ -f "$SSH_KEY_OLD_PATH" ]]; then
        cp "$SSH_KEY_OLD_PATH" "$SSH_KEY_PATH"
        cp "$SSH_KEY_OLD_PATH.pub" "$SSH_KEY_PATH.pub" 2>/dev/null || true
        success "SSH key restored from backup"
        
        # Test restored key
        test_ssh_connectivity "$SSH_KEY_PATH" "restored key"
    else
        error "No backup SSH key found to restore"
        return 1
    fi
}

# Main function
main() {
    local command="${1:-status}"
    local force_mode="false"
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            rotate|test|backup|restore|status)
                command=$1
                ;;
            --force)
                force_mode="true"
                ;;
            --help|-h)
                show_help
                exit 0
                ;;
            *)
                error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
        shift
    done
    
    # Execute command
    case $command in
        rotate)
            rotate_ssh_keys "$force_mode"
            ;;
        test)
            if [[ -f "$SSH_KEY_PATH" ]]; then
                test_ssh_connectivity "$SSH_KEY_PATH" "current key"
            else
                error "SSH key not found: $SSH_KEY_PATH"
                exit 1
            fi
            ;;
        backup)
            backup_key
            ;;
        restore)
            restore_key
            ;;
        status)
            show_key_status
            ;;
        *)
            error "Unknown command: $command"
            show_help
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"
