#!/bin/bash
# VMware Fusion Networking Diagnostic and Fix Script

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo "=== 🔧 VMware Fusion Networking Diagnostic ==="
echo ""

# Check current network configuration
echo -e "${BLUE}🔍 Current Mac Network Configuration:${NC}"
echo "Your Mac IP: $(ifconfig en0 | grep 'inet ' | awk '{print $2}')"
echo "Target VM IP: 10.0.0.23"
echo ""

# Test basic connectivity
echo -e "${BLUE}🔍 Testing Basic Connectivity:${NC}"
if ping -c 1 -W 2000 10.0.0.23 > /dev/null 2>&1; then
    echo -e "${GREEN}✅ VM is reachable${NC}"
else
    echo -e "${RED}❌ VM is not reachable${NC}"
fi
echo ""

# Check VMware network services
echo -e "${BLUE}🔍 VMware Fusion Network Services:${NC}"
if pgrep -f vmnet > /dev/null; then
    echo -e "${GREEN}✅ VMware network services running${NC}"
else
    echo -e "${RED}❌ VMware network services not running${NC}"
fi

# Check vmnet interfaces
echo ""
echo -e "${BLUE}🔍 VMware Network Interfaces:${NC}"
for i in {0..9}; do
    if ifconfig vmnet$i > /dev/null 2>&1; then
        vmnet_ip=$(ifconfig vmnet$i | grep 'inet ' | awk '{print $2}')
        echo "vmnet$i: $vmnet_ip"
    fi
done
echo ""

echo -e "${BLUE}🔍 VMware Bridge Interfaces:${NC}"
for i in {100..110}; do
    if ifconfig bridge$i > /dev/null 2>&1; then
        bridge_ip=$(ifconfig bridge$i | grep 'inet ' | awk '{print $2}')
        if [ ! -z "$bridge_ip" ]; then
            echo "bridge$i: $bridge_ip"
        fi
    fi
done
echo ""

echo "=== 🛠️ Troubleshooting Steps ==="
echo ""
echo -e "${YELLOW}1. Check VM Power State:${NC}"
echo "   • Ensure your honeypot VM is powered on"
echo "   • Check VM console for any boot errors"
echo ""

echo -e "${YELLOW}2. VM Network Adapter Configuration:${NC}"
echo "   • Open VMware Fusion"
echo "   • Right-click your honeypot VM → Settings"
echo "   • Go to Network Adapter settings"
echo "   • Try these configurations in order:"
echo ""
echo "   Option A - Bridged Mode (Recommended):"
echo "     ✓ Connect directly to the physical network"
echo "     ✓ Autodetect your physical network interface"
echo "     ✓ VM gets IP from your router's DHCP"
echo ""
echo "   Option B - NAT Mode:"
echo "     ✓ Share the Mac's connection"
echo "     ✓ VM gets IP from VMware's DHCP (usually 172.16.x.x)"
echo ""

echo -e "${YELLOW}3. Fix VMware Network Services:${NC}"
echo "   If network services aren't running:"
echo "   sudo /Applications/VMware\\ Fusion.app/Contents/Library/vmnet-cli --stop"
echo "   sudo /Applications/VMware\\ Fusion.app/Contents/Library/vmnet-cli --start"
echo ""

echo -e "${YELLOW}4. VM Internal Network Configuration:${NC}"
echo "   SSH into your VM console (not from Mac) and run:"
echo "   • sudo ip addr show"
echo "   • sudo systemctl status networking"
echo "   • sudo dhclient -v"
echo ""

echo -e "${YELLOW}5. Update Mini-XDR Configuration:${NC}"
echo "   Once you get the VM's correct IP address:"
echo "   • Update backend/.env file:"
echo "   • Set HONEYPOT_HOST=<new_vm_ip>"
echo "   • Restart Mini-XDR: ./scripts/start-all.sh"
echo ""

echo "=== 🔍 Quick Scan for VMs ==="
echo "Scanning network for potential VM IPs..."
echo ""

# Scan common VMware network ranges
networks=("10.0.0" "192.168.1" "192.168.56" "192.168.238" "172.16.1")

for network in "${networks[@]}"; do
    echo "Scanning ${network}.0/24..."
    for i in {20..30}; do
        ip="${network}.${i}"
        if ping -c 1 -W 1000 "$ip" > /dev/null 2>&1; then
            echo -e "${GREEN}✅ Found responding host: $ip${NC}"
            
            # Test SSH on common ports
            for port in 22 22022 2222; do
                if nc -z -w 1 "$ip" "$port" 2>/dev/null; then
                    echo "   SSH port $port is open"
                fi
            done
        fi
    done
done

echo ""
echo "=== 📝 Next Steps ==="
echo "1. Try the VM network configuration options above"
echo "2. Note any responding IPs from the scan"
echo "3. Update your Mini-XDR configuration with the correct IP"
echo "4. Test connectivity with: ping <vm_ip>"
echo "5. Test SSH with: ssh -p 22022 -i ~/.ssh/xdrops_id_ed25519 xdrops@<vm_ip>"
echo ""
