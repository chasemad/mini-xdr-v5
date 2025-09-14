#!/bin/bash
# Mini-XDR Honeypot VM Startup Script
# Starts all honeypot services and tools automatically on VM boot
# Run this script on the AWS VM honeypot instance

set -e

# Configuration
LOG_FILE="/var/log/honeypot-startup.log"
SERVICES=("cowrie" "apache2" "fluent-bit" "ufw")
HONEYPOT_TOOLS=("cowrie" "apache2")

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Logging functions
log() {
    local message="[$(date +'%Y-%m-%d %H:%M:%S')] $1"
    echo -e "${BLUE}$message${NC}"
    echo "$message" >> "$LOG_FILE"
}

success() {
    local message="✅ $1"
    echo -e "${GREEN}$message${NC}"
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] SUCCESS: $1" >> "$LOG_FILE"
}

warning() {
    local message="⚠️  $1"
    echo -e "${YELLOW}$message${NC}"
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1" >> "$LOG_FILE"
}

error() {
    local message="❌ $1"
    echo -e "${RED}$message${NC}"
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1" >> "$LOG_FILE"
}

# Function to check if running as root/sudo
check_privileges() {
    if [ "$EUID" -ne 0 ]; then
        error "This script must be run as root or with sudo"
        echo "Usage: sudo $0"
        exit 1
    fi
}

# Function to wait for network connectivity
wait_for_network() {
    log "Waiting for network connectivity..."
    local max_attempts=30
    local attempt=0
    
    while [ $attempt -lt $max_attempts ]; do
        if ping -c 1 8.8.8.8 >/dev/null 2>&1; then
            success "Network connectivity established"
            return 0
        fi
        sleep 2
        attempt=$((attempt + 1))
        echo -n "."
    done
    
    warning "Network connectivity check failed after $max_attempts attempts"
    return 1
}

# Function to configure system settings
configure_system() {
    log "Configuring system settings..."
    
    # Set timezone to UTC for consistent logging
    timedatectl set-timezone UTC >/dev/null 2>&1 || true
    
    # Ensure hostname is set correctly
    if [ ! -f "/etc/hostname" ] || [ "$(cat /etc/hostname)" = "localhost" ]; then
        echo "mini-xdr-honeypot" > /etc/hostname
        hostnamectl set-hostname mini-xdr-honeypot >/dev/null 2>&1 || true
    fi
    
    # Update /etc/hosts
    if ! grep -q "mini-xdr-honeypot" /etc/hosts; then
        echo "127.0.0.1 mini-xdr-honeypot" >> /etc/hosts
    fi
    
    success "System settings configured"
}

# Function to start firewall
start_firewall() {
    log "Starting and configuring UFW firewall..."
    
    # Enable UFW if not already enabled
    if ! ufw status | grep -q "Status: active"; then
        ufw --force enable >/dev/null 2>&1
    fi
    
    # Ensure required ports are open
    local required_ports=("22" "80" "443" "21" "23" "3306" "22022")
    for port in "${required_ports[@]}"; do
        ufw allow "$port/tcp" >/dev/null 2>&1 || true
    done
    
    # Check UFW status
    if ufw status | grep -q "Status: active"; then
        success "UFW firewall is active and configured"
    else
        warning "UFW firewall may not be properly configured"
    fi
}

# Function to configure iptables for port redirection
configure_iptables() {
    log "Configuring iptables for port redirection..."
    
    # Check if iptables rules exist
    if ! iptables -t nat -L PREROUTING | grep -q "REDIRECT.*tcp dpt:22"; then
        log "Adding iptables rule to redirect port 22 to 2222..."
        iptables -t nat -A PREROUTING -p tcp --dport 22 -j REDIRECT --to-port 2222
        
        # Save iptables rules
        if command -v iptables-save >/dev/null 2>&1; then
            iptables-save > /etc/iptables/rules.v4 2>/dev/null || true
        fi
    fi
    
    success "Iptables configured for SSH honeypot redirection"
}

# Function to start Cowrie honeypot
start_cowrie() {
    log "Starting Cowrie SSH/Telnet honeypot..."
    
    # Check if Cowrie is installed
    if [ ! -d "/opt/cowrie" ]; then
        error "Cowrie not found at /opt/cowrie - please run honeypot setup first"
        return 1
    fi
    
    # Ensure cowrie user exists
    if ! id "cowrie" >/dev/null 2>&1; then
        error "Cowrie user does not exist"
        return 1
    fi
    
    # Ensure proper ownership
    chown -R cowrie:cowrie /opt/cowrie >/dev/null 2>&1 || true
    
    # Start Cowrie service
    systemctl daemon-reload
    systemctl enable cowrie >/dev/null 2>&1 || true
    
    if systemctl start cowrie >/dev/null 2>&1; then
        sleep 3
        if systemctl is-active --quiet cowrie; then
            success "Cowrie honeypot started successfully"
            
            # Verify Cowrie is listening on port 2222
            if netstat -tlnp | grep -q ":2222.*python"; then
                success "Cowrie is listening on port 2222"
            else
                warning "Cowrie may not be listening on port 2222"
            fi
        else
            error "Cowrie failed to start properly"
            return 1
        fi
    else
        error "Failed to start Cowrie service"
        return 1
    fi
}

# Function to start web honeypots
start_web_honeypots() {
    log "Starting web honeypots (Apache)..."
    
    # Start Apache
    systemctl enable apache2 >/dev/null 2>&1 || true
    if systemctl start apache2 >/dev/null 2>&1; then
        sleep 2
        if systemctl is-active --quiet apache2; then
            success "Apache web server started successfully"
            
            # Check if web honeypot files exist
            if [ -f "/var/www/html/login.php" ]; then
                success "Web honeypot login page is available"
            else
                warning "Web honeypot login page not found"
            fi
        else
            error "Apache failed to start properly"
            return 1
        fi
    else
        error "Failed to start Apache service"
        return 1
    fi
}

# Function to start Fluent Bit log forwarder
start_fluent_bit() {
    log "Starting Fluent Bit log forwarder..."
    
    # Check if Fluent Bit is installed
    if ! command -v fluent-bit >/dev/null 2>&1; then
        warning "Fluent Bit not installed - logs will not be forwarded"
        return 1
    fi
    
    # Check if configuration exists
    if [ ! -f "/etc/fluent-bit/fluent-bit.conf" ]; then
        warning "Fluent Bit configuration not found - logs will not be forwarded"
        return 1
    fi
    
    # Start Fluent Bit service
    systemctl enable fluent-bit >/dev/null 2>&1 || true
    if systemctl start fluent-bit >/dev/null 2>&1; then
        sleep 2
        if systemctl is-active --quiet fluent-bit; then
            success "Fluent Bit log forwarder started successfully"
        else
            warning "Fluent Bit may not be running properly"
        fi
    else
        warning "Failed to start Fluent Bit service - logs will not be forwarded"
    fi
}

# Function to verify honeypot services
verify_services() {
    log "Verifying honeypot services..."
    
    local all_good=true
    
    # Check each service
    for service in "${SERVICES[@]}"; do
        if systemctl is-active --quiet "$service"; then
            success "$service is running"
        else
            error "$service is not running"
            all_good=false
        fi
    done
    
    # Check specific honeypot functionality
    log "Checking honeypot-specific functionality..."
    
    # Check SSH honeypot (port 22 -> 2222 redirection)
    if netstat -tlnp | grep -q ":2222.*python"; then
        success "SSH honeypot is listening on port 2222"
    else
        error "SSH honeypot is not listening properly"
        all_good=false
    fi
    
    # Check web honeypot (port 80)
    if netstat -tlnp | grep -q ":80.*apache"; then
        success "Web honeypot is listening on port 80"
    else
        error "Web honeypot is not listening properly"
        all_good=false
    fi
    
    # Check log files exist and are being written
    local log_files=(
        "/opt/cowrie/var/log/cowrie/cowrie.json"
        "/var/log/apache2/access.log"
    )
    
    for log_file in "${log_files[@]}"; do
        if [ -f "$log_file" ]; then
            success "Log file exists: $log_file"
        else
            warning "Log file missing: $log_file"
        fi
    done
    
    return $all_good
}

# Function to create startup status file
create_status_file() {
    local status_file="/var/lib/honeypot-status"
    
    cat > "$status_file" << EOF
{
    "startup_time": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "hostname": "$(hostname)",
    "services": {
        "cowrie": "$(systemctl is-active cowrie 2>/dev/null || echo 'inactive')",
        "apache2": "$(systemctl is-active apache2 2>/dev/null || echo 'inactive')",
        "fluent-bit": "$(systemctl is-active fluent-bit 2>/dev/null || echo 'inactive')",
        "ufw": "$(ufw status | grep -q 'Status: active' && echo 'active' || echo 'inactive')"
    },
    "network": {
        "ssh_honeypot_port": "2222",
        "web_honeypot_port": "80",
        "management_port": "22022"
    },
    "last_updated": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF
    
    chmod 644 "$status_file"
    success "Status file created at $status_file"
}

# Function to setup automatic startup on boot
setup_auto_startup() {
    log "Setting up automatic startup on boot..."
    
    # Create systemd service for honeypot startup
    cat > /etc/systemd/system/honeypot-startup.service << EOF
[Unit]
Description=Mini-XDR Honeypot Startup Service
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
ExecStart=/opt/mini-xdr/honeypot-vm-startup.sh
RemainAfterExit=true
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF
    
    # Copy this script to /opt/mini-xdr/
    mkdir -p /opt/mini-xdr
    cp "$0" /opt/mini-xdr/honeypot-vm-startup.sh
    chmod +x /opt/mini-xdr/honeypot-vm-startup.sh
    
    # Enable the service
    systemctl daemon-reload
    systemctl enable honeypot-startup.service >/dev/null 2>&1
    
    success "Automatic startup configured"
}

# Function to display startup summary
show_startup_summary() {
    echo ""
    echo "=== 🍯 Mini-XDR Honeypot VM Startup Complete ==="
    echo ""
    echo "📊 Service Status:"
    for service in "${SERVICES[@]}"; do
        local status=$(systemctl is-active "$service" 2>/dev/null || echo "inactive")
        if [ "$status" = "active" ]; then
            echo "   ✅ $service: $status"
        else
            echo "   ❌ $service: $status"
        fi
    done
    
    echo ""
    echo "🔗 Network Ports:"
    echo "   • SSH Honeypot:    Port 22 (redirected to 2222)"
    echo "   • Web Honeypot:    Port 80 (HTTP)"
    echo "   • Web Honeypot:    Port 443 (HTTPS)"
    echo "   • SSH Management:  Port 22022"
    echo ""
    
    echo "📁 Log Files:"
    echo "   • Cowrie Logs:     /opt/cowrie/var/log/cowrie/cowrie.json"
    echo "   • Apache Logs:     /var/log/apache2/access.log"
    echo "   • Startup Log:     $LOG_FILE"
    echo "   • Status File:     /var/lib/honeypot-status"
    echo ""
    
    echo "🔧 Management:"
    echo "   • Stop All:        sudo /opt/mini-xdr/honeypot-vm-stop.sh"
    echo "   • Restart All:     sudo systemctl restart honeypot-startup"
    echo "   • Check Status:    sudo systemctl status honeypot-startup"
    echo "   • View Logs:       sudo journalctl -u honeypot-startup -f"
    echo ""
    
    local public_ip=$(curl -s -m 5 http://169.254.169.254/latest/meta-data/public-ipv4 2>/dev/null || echo "N/A")
    if [ "$public_ip" != "N/A" ]; then
        echo "🌐 Public Access:"
        echo "   • SSH Honeypot:    ssh admin@$public_ip"
        echo "   • Web Honeypot:    http://$public_ip/login.php"
        echo "   • SSH Management:  ssh -p 22022 ubuntu@$public_ip"
        echo ""
    fi
    
    echo "✅ Honeypot VM is ready for threat detection!"
}

# Main execution function
main() {
    echo "=== 🍯 Mini-XDR Honeypot VM Startup ==="
    echo "Starting all honeypot services and tools..."
    echo ""
    
    # Initialize log file
    touch "$LOG_FILE"
    chmod 644 "$LOG_FILE"
    
    # Check privileges
    check_privileges
    
    # Wait for network
    if ! wait_for_network; then
        warning "Continuing without network connectivity verification"
    fi
    
    echo ""
    
    # Configure system
    configure_system
    echo ""
    
    # Start firewall
    start_firewall
    echo ""
    
    # Configure iptables
    configure_iptables
    echo ""
    
    # Start honeypot services
    log "Starting honeypot services..."
    echo ""
    
    # Start Cowrie
    if start_cowrie; then
        echo ""
    else
        error "Critical: Cowrie honeypot failed to start"
        exit 1
    fi
    
    # Start web honeypots
    if start_web_honeypots; then
        echo ""
    else
        warning "Web honeypots failed to start - continuing"
    fi
    
    # Start log forwarder
    start_fluent_bit
    echo ""
    
    # Verify all services
    log "Verifying all services..."
    if verify_services; then
        success "All critical services are running"
    else
        warning "Some services may not be running properly"
    fi
    echo ""
    
    # Create status file
    create_status_file
    echo ""
    
    # Setup auto-startup (only if not already done)
    if [ ! -f "/etc/systemd/system/honeypot-startup.service" ]; then
        setup_auto_startup
        echo ""
    fi
    
    # Show summary
    show_startup_summary
    
    log "Honeypot VM startup completed successfully"
}

# Handle command line arguments
case "${1:-}" in
    --auto-startup)
        # Skip auto-startup setup when called from systemd
        setup_auto_startup() { :; }
        ;;
    --help|-h)
        echo "Usage: $0 [--auto-startup] [--help]"
        echo ""
        echo "Options:"
        echo "  --auto-startup    Skip setting up systemd service (used internally)"
        echo "  --help, -h        Show this help message"
        echo ""
        echo "This script starts all Mini-XDR honeypot services on an AWS VM."
        exit 0
        ;;
esac

# Run main function
main "$@"
