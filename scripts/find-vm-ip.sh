#!/bin/bash
# Quick VM IP Discovery Script

echo "=== 🔍 Finding Your Honeypot VM ==="
echo ""

# Check if VM is on VMware NAT networks
echo "Scanning VMware NAT networks..."
for network in "192.168.238" "192.168.56" "192.168.168" "172.16.1" "172.16.5"; do
    echo "Scanning ${network}.0/24..."
    for i in {20..30} {100..110}; do
        ip="${network}.${i}"
        if ping -c 1 -W 500 "$ip" > /dev/null 2>&1; then
            echo "✅ Found responding host: $ip"
            
            # Test SSH ports
            for port in 22 22022 2222; do
                if nc -z -w 1 "$ip" "$port" 2>/dev/null; then
                    echo "   🔓 SSH port $port is open"
                    
                    # Test if it's our honeypot by checking for our SSH key
                    if timeout 5 ssh -p $port -i ~/.ssh/xdrops_id_ed25519 -o StrictHostKeyChecking=no -o ConnectTimeout=3 xdrops@$ip 'echo "XDR_TEST"' 2>/dev/null | grep -q "XDR_TEST"; then
                        echo "   🎯 THIS IS YOUR HONEYPOT VM!"
                        echo ""
                        echo "=== 🛠️ Fix Mini-XDR Configuration ==="
                        echo "Run these commands to update your configuration:"
                        echo ""
                        echo "# Update the backend configuration"
                        echo "sed -i '' 's/HONEYPOT_HOST=.*/HONEYPOT_HOST=$ip/' backend/.env"
                        echo "sed -i '' 's/HONEYPOT_SSH_PORT=.*/HONEYPOT_SSH_PORT=$port/' backend/.env"
                        echo ""
                        echo "# Restart Mini-XDR"
                        echo "./scripts/start-all.sh"
                        echo ""
                        echo "# Test the connection"
                        echo "ssh -p $port -i ~/.ssh/xdrops_id_ed25519 xdrops@$ip 'sudo ufw status'"
                        exit 0
                    fi
                fi
            done
        fi
    done
done

echo ""
echo "=== 🚨 VM Not Found - Manual Steps ==="
echo ""
echo "Your VM might be:"
echo "1. Powered off - Start it in VMware Fusion"
echo "2. On a different network - Check VMware network settings"
echo "3. Using DHCP - IP may have changed"
echo ""
echo "Manual troubleshooting:"
echo "1. Open VMware Fusion"
echo "2. Start your honeypot VM"
echo "3. In VM console, run: ip addr show"
echo "4. Note the VM's current IP address"
echo "5. Update backend/.env with the correct IP"
echo ""
echo "Recommended VMware network settings:"
echo "• Network Adapter: Bridged (Autodetect)"
echo "• Connect at Power On: ✓"
echo "• This will give your VM an IP on your main network (10.0.0.x)"
