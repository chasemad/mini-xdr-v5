#!/bin/bash
# Mini-XDR Honeypot VM Stop Script
# Safely stops all honeypot services and tools
# Run this script on the AWS VM honeypot instance

set -e

# Configuration
LOG_FILE="/var/log/honeypot-shutdown.log"
SERVICES=("fluent-bit" "apache2" "cowrie")  # Stop in reverse order
CLEANUP_DIRS=("/tmp/honeypot-*" "/var/run/honeypot-*")

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

# Function to gracefully stop a service
stop_service() {
    local service_name=$1
    local timeout=${2:-30}
    
    log "Stopping $service_name service..."
    
    if ! systemctl is-active --quiet "$service_name"; then
        log "$service_name is not running"
        return 0
    fi
    
    # Try graceful stop first
    if systemctl stop "$service_name" >/dev/null 2>&1; then
        # Wait for service to stop
        local count=0
        while [ $count -lt $timeout ]; do
            if ! systemctl is-active --quiet "$service_name"; then
                success "$service_name stopped gracefully"
                return 0
            fi
            sleep 1
            count=$((count + 1))
        done
        
        warning "$service_name did not stop within $timeout seconds, forcing stop..."
        systemctl kill "$service_name" >/dev/null 2>&1 || true
        sleep 2
        
        if ! systemctl is-active --quiet "$service_name"; then
            success "$service_name force stopped"
            return 0
        else
            error "Failed to stop $service_name"
            return 1
        fi
    else
        error "Failed to initiate stop for $service_name"
        return 1
    fi
}

# Function to stop Cowrie honeypot specifically
stop_cowrie() {
    log "Stopping Cowrie SSH/Telnet honeypot..."
    
    # First try systemctl
    if stop_service "cowrie" 30; then
        # Verify no cowrie processes remain
        local cowrie_pids=$(pgrep -f "cowrie" 2>/dev/null || true)
        if [ -n "$cowrie_pids" ]; then
            warning "Found remaining Cowrie processes, terminating..."
            echo "$cowrie_pids" | xargs kill -TERM 2>/dev/null || true
            sleep 3
            
            # Force kill if still running
            cowrie_pids=$(pgrep -f "cowrie" 2>/dev/null || true)
            if [ -n "$cowrie_pids" ]; then
                warning "Force killing remaining Cowrie processes..."
                echo "$cowrie_pids" | xargs kill -KILL 2>/dev/null || true
            fi
        fi
        
        success "Cowrie honeypot stopped completely"
        return 0
    else
        # Manual cleanup if systemctl failed
        warning "Systemctl failed, attempting manual Cowrie cleanup..."
        
        if [ -f "/opt/cowrie/var/run/cowrie.pid" ]; then
            local cowrie_pid=$(cat /opt/cowrie/var/run/cowrie.pid 2>/dev/null || true)
            if [ -n "$cowrie_pid" ] && kill -0 "$cowrie_pid" 2>/dev/null; then
                log "Stopping Cowrie process $cowrie_pid..."
                kill -TERM "$cowrie_pid" 2>/dev/null || true
                sleep 3
                
                if kill -0 "$cowrie_pid" 2>/dev/null; then
                    warning "Force killing Cowrie process $cowrie_pid..."
                    kill -KILL "$cowrie_pid" 2>/dev/null || true
                fi
            fi
            rm -f /opt/cowrie/var/run/cowrie.pid
        fi
        
        # Kill any remaining cowrie processes
        pkill -f "cowrie" 2>/dev/null || true
        
        success "Cowrie manual cleanup completed"
        return 0
    fi
}

# Function to stop web honeypots
stop_web_honeypots() {
    log "Stopping web honeypots (Apache)..."
    
    if stop_service "apache2" 20; then
        # Verify no apache processes remain
        local apache_pids=$(pgrep -f "apache2" 2>/dev/null || true)
        if [ -n "$apache_pids" ]; then
            warning "Found remaining Apache processes, terminating..."
            echo "$apache_pids" | xargs kill -TERM 2>/dev/null || true
            sleep 2
            
            # Force kill if still running
            apache_pids=$(pgrep -f "apache2" 2>/dev/null || true)
            if [ -n "$apache_pids" ]; then
                warning "Force killing remaining Apache processes..."
                echo "$apache_pids" | xargs kill -KILL 2>/dev/null || true
            fi
        fi
        
        success "Web honeypots stopped"
        return 0
    else
        error "Failed to stop Apache web server"
        return 1
    fi
}

# Function to stop Fluent Bit log forwarder
stop_fluent_bit() {
    log "Stopping Fluent Bit log forwarder..."
    
    if ! systemctl is-active --quiet fluent-bit; then
        log "Fluent Bit is not running"
        return 0
    fi
    
    if stop_service "fluent-bit" 15; then
        success "Fluent Bit log forwarder stopped"
        return 0
    else
        warning "Failed to stop Fluent Bit gracefully, attempting force stop..."
        pkill -f "fluent-bit" 2>/dev/null || true
        success "Fluent Bit force stopped"
        return 0
    fi
}

# Function to clean up iptables rules
cleanup_iptables() {
    log "Cleaning up iptables rules..."
    
    # Remove SSH honeypot redirection rule
    if iptables -t nat -L PREROUTING | grep -q "REDIRECT.*tcp dpt:22"; then
        log "Removing SSH honeypot iptables redirection..."
        iptables -t nat -D PREROUTING -p tcp --dport 22 -j REDIRECT --to-port 2222 2>/dev/null || true
        
        # Save iptables rules
        if command -v iptables-save >/dev/null 2>&1; then
            iptables-save > /etc/iptables/rules.v4 2>/dev/null || true
        fi
        
        success "Iptables rules cleaned up"
    else
        log "No SSH honeypot iptables rules found"
    fi
}

# Function to disable firewall (optional)
disable_firewall() {
    local disable_ufw=${1:-false}
    
    if [ "$disable_ufw" = "true" ]; then
        log "Disabling UFW firewall..."
        ufw --force disable >/dev/null 2>&1 || true
        success "UFW firewall disabled"
    else
        log "Keeping UFW firewall enabled (use --disable-firewall to disable)"
    fi
}

# Function to cleanup temporary files
cleanup_temp_files() {
    log "Cleaning up temporary files..."
    
    # Clean up known temporary directories
    for cleanup_dir in "${CLEANUP_DIRS[@]}"; do
        if [ -d "$cleanup_dir" ]; then
            rm -rf "$cleanup_dir" 2>/dev/null || true
            log "Cleaned up directory: $cleanup_dir"
        fi
    done
    
    # Clean up log locks and temporary files
    find /var/log -name "*.lock" -delete 2>/dev/null || true
    find /tmp -name "*honeypot*" -delete 2>/dev/null || true
    find /var/run -name "*honeypot*" -delete 2>/dev/null || true
    
    success "Temporary files cleaned up"
}

# Function to verify services are stopped
verify_shutdown() {
    log "Verifying all services are stopped..."
    
    local all_stopped=true
    
    # Check each service
    for service in "${SERVICES[@]}"; do
        if systemctl is-active --quiet "$service"; then
            error "$service is still running"
            all_stopped=false
        else
            success "$service is stopped"
        fi
    done
    
    # Check for any remaining honeypot processes
    local remaining_processes=$(pgrep -f "(cowrie|apache2|fluent-bit)" 2>/dev/null || true)
    if [ -n "$remaining_processes" ]; then
        warning "Found remaining honeypot processes:"
        ps aux | grep -E "(cowrie|apache2|fluent-bit)" | grep -v grep || true
        all_stopped=false
    else
        success "No remaining honeypot processes found"
    fi
    
    # Check port usage
    local used_ports=$(netstat -tlnp 2>/dev/null | grep -E ":80|:2222|:443" || true)
    if [ -n "$used_ports" ]; then
        warning "Some honeypot ports may still be in use:"
        echo "$used_ports"
    else
        success "All honeypot ports are free"
    fi
    
    return $all_stopped
}

# Function to update status file
update_status_file() {
    local status_file="/var/lib/honeypot-status"
    
    if [ -f "$status_file" ]; then
        # Update existing status file
        local temp_file=$(mktemp)
        jq --arg timestamp "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
           --arg cowrie "$(systemctl is-active cowrie 2>/dev/null || echo 'inactive')" \
           --arg apache2 "$(systemctl is-active apache2 2>/dev/null || echo 'inactive')" \
           --arg fluent_bit "$(systemctl is-active fluent-bit 2>/dev/null || echo 'inactive')" \
           --arg ufw "$(ufw status | grep -q 'Status: active' && echo 'active' || echo 'inactive')" \
           '.services.cowrie = $cowrie | .services.apache2 = $apache2 | .services."fluent-bit" = $fluent_bit | .services.ufw = $ufw | .last_updated = $timestamp | .shutdown_time = $timestamp' \
           "$status_file" > "$temp_file" && mv "$temp_file" "$status_file" 2>/dev/null || true
        
        success "Status file updated"
    else
        warning "Status file not found - creating shutdown status"
        cat > "$status_file" << EOF
{
    "shutdown_time": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "hostname": "$(hostname)",
    "services": {
        "cowrie": "inactive",
        "apache2": "inactive",
        "fluent-bit": "inactive",
        "ufw": "$(ufw status | grep -q 'Status: active' && echo 'active' || echo 'inactive')"
    },
    "last_updated": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF
        chmod 644 "$status_file"
        success "Shutdown status file created"
    fi
}

# Function to disable auto-startup
disable_auto_startup() {
    local disable_auto=${1:-false}
    
    if [ "$disable_auto" = "true" ]; then
        log "Disabling automatic startup on boot..."
        
        if [ -f "/etc/systemd/system/honeypot-startup.service" ]; then
            systemctl disable honeypot-startup.service >/dev/null 2>&1 || true
            systemctl daemon-reload
            success "Automatic startup disabled"
        else
            log "No automatic startup service found"
        fi
    else
        log "Keeping automatic startup enabled (use --disable-auto-startup to disable)"
    fi
}

# Function to display shutdown summary
show_shutdown_summary() {
    echo ""
    echo "=== 🛑 Mini-XDR Honeypot VM Shutdown Complete ==="
    echo ""
    echo "📊 Final Service Status:"
    for service in cowrie apache2 fluent-bit ufw; do
        local status=$(systemctl is-active "$service" 2>/dev/null || echo "inactive")
        if [ "$status" = "inactive" ] || [ "$status" = "failed" ]; then
            echo "   ✅ $service: stopped"
        else
            echo "   ⚠️  $service: $status"
        fi
    done
    
    echo ""
    echo "🔗 Network Status:"
    local port_check=$(netstat -tlnp 2>/dev/null | grep -E ":80|:2222|:443" || true)
    if [ -z "$port_check" ]; then
        echo "   ✅ All honeypot ports are free"
    else
        echo "   ⚠️  Some ports may still be in use"
    fi
    
    echo ""
    echo "📁 Log Files Preserved:"
    echo "   • Cowrie Logs:     /opt/cowrie/var/log/cowrie/cowrie.json"
    echo "   • Apache Logs:     /var/log/apache2/access.log"
    echo "   • Shutdown Log:    $LOG_FILE"
    echo "   • Status File:     /var/lib/honeypot-status"
    echo ""
    
    echo "🔧 Management:"
    echo "   • Start All:       sudo /opt/mini-xdr/honeypot-vm-startup.sh"
    echo "   • Check Status:    sudo systemctl status honeypot-startup"
    echo "   • View Logs:       sudo journalctl -u honeypot-startup -f"
    echo ""
    
    if [ -f "/etc/systemd/system/honeypot-startup.service" ]; then
        if systemctl is-enabled honeypot-startup.service >/dev/null 2>&1; then
            echo "🔄 Auto-startup: Enabled (will start on next boot)"
        else
            echo "🔄 Auto-startup: Disabled"
        fi
    else
        echo "🔄 Auto-startup: Not configured"
    fi
    
    echo ""
    echo "✅ Honeypot VM is safely shut down!"
}

# Main execution function
main() {
    local disable_ufw=false
    local disable_auto=false
    local force_stop=false
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --disable-firewall)
                disable_ufw=true
                shift
                ;;
            --disable-auto-startup)
                disable_auto=true
                shift
                ;;
            --force)
                force_stop=true
                shift
                ;;
            --help|-h)
                echo "Usage: $0 [OPTIONS]"
                echo ""
                echo "Options:"
                echo "  --disable-firewall      Disable UFW firewall during shutdown"
                echo "  --disable-auto-startup  Disable automatic startup on boot"
                echo "  --force                 Force stop all services (skip graceful shutdown)"
                echo "  --help, -h              Show this help message"
                echo ""
                echo "This script safely stops all Mini-XDR honeypot services on an AWS VM."
                exit 0
                ;;
            *)
                error "Unknown option: $1"
                echo "Use --help for usage information"
                exit 1
                ;;
        esac
    done
    
    echo "=== 🛑 Mini-XDR Honeypot VM Shutdown ==="
    echo "Safely stopping all honeypot services and tools..."
    if [ "$force_stop" = "true" ]; then
        echo "⚡ Force mode enabled - services will be terminated immediately"
    fi
    echo ""
    
    # Initialize log file
    touch "$LOG_FILE"
    chmod 644 "$LOG_FILE"
    
    # Check privileges
    check_privileges
    
    # Stop services in reverse order
    log "Stopping honeypot services..."
    echo ""
    
    # Stop Fluent Bit first (log forwarder)
    stop_fluent_bit
    echo ""
    
    # Stop web honeypots
    stop_web_honeypots
    echo ""
    
    # Stop Cowrie (critical honeypot)
    stop_cowrie
    echo ""
    
    # Clean up iptables
    cleanup_iptables
    echo ""
    
    # Disable firewall if requested
    disable_firewall "$disable_ufw"
    echo ""
    
    # Clean up temporary files
    cleanup_temp_files
    echo ""
    
    # Disable auto-startup if requested
    disable_auto_startup "$disable_auto"
    echo ""
    
    # Verify shutdown
    log "Verifying shutdown..."
    if verify_shutdown; then
        success "All services stopped successfully"
    else
        warning "Some services may not have stopped properly"
    fi
    echo ""
    
    # Update status file
    update_status_file
    echo ""
    
    # Show summary
    show_shutdown_summary
    
    log "Honeypot VM shutdown completed"
}

# Run main function with all arguments
main "$@"
