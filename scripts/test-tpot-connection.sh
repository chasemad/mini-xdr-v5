#!/bin/bash

# Quick T-Pot Connection Test
# Run this after temporarily opening the firewall

echo "═══════════════════════════════════════════════════════════════"
echo "🔍 Testing T-Pot Connection from Mini-XDR"
echo "═══════════════════════════════════════════════════════════════"
echo ""

echo "Step 1: First, open T-Pot firewall temporarily"
echo "--------------------------------------------------------------"
echo "In your working terminal, run:"
echo ""
echo "  ssh -p 64295 luxieum@203.0.113.42"
echo "  sudo ufw allow 64295/tcp"
echo "  sudo ufw status | grep 64295"
echo ""
echo "Then come back and run this script again."
echo ""

read -p "Have you opened the firewall? [y/N]: " response

if [ "$response" != "y" ] && [ "$response" != "Y" ]; then
    echo "Please open the firewall first and run again!"
    exit 1
fi

echo ""
echo "Step 2: Testing SSH connection..."
echo "--------------------------------------------------------------"

if ssh -p 64295 -o ConnectTimeout=5 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null luxieum@203.0.113.42 "echo 'Success!'; hostname" 2>/dev/null | grep -q "Success"; then
    echo "✅ SSH connection works!"
    echo ""

    echo "Step 3: Restarting backend to connect T-Pot..."
    echo "--------------------------------------------------------------"

    # Kill backend
    pkill -f "uvicorn app.main" 2>/dev/null
    sleep 2

    # Restart backend
    cd $(cd "$(dirname "$0")/../.." .. pwd)/backend
    nohup ../venv/bin/python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000 > backend_tpot.log 2>&1 &

    echo "Backend restarting... waiting 15 seconds..."
    sleep 15

    echo ""
    echo "Step 4: Checking T-Pot connection status..."
    echo "--------------------------------------------------------------"

    curl -s http://localhost:8000/api/tpot/status | python3 -c "
import sys, json
data = json.load(sys.stdin)
status = data.get('status', 'unknown')
monitoring = data.get('monitoring_honeypots', [])

print(f'Status: {status.upper()}')
print(f'Monitoring: {', '.join(monitoring) if monitoring else 'None'}')
print('')

if status == 'connected':
    print('✅ T-Pot is CONNECTED and monitoring!')
    print('')
    print('You can now lock down the firewall:')
    print('  sudo ufw delete allow 64295/tcp')
    print('  sudo ufw allow from 172.16.110.1 to any port 64295')
else:
    print('⚠️  Still not connected. Checking logs...')
"

    echo ""
    echo "Backend logs:"
    tail -20 backend_tpot.log | grep -i "tpot\|connect\|ssh"

else
    echo "❌ SSH still not working from this environment"
    echo ""
    echo "Possible issues:"
    echo "  1. Firewall rule not applied yet"
    echo "  2. Different network routing in this terminal"
    echo "  3. SSH keys/config difference"
    echo ""
    echo "Try: ssh -vvv -p 64295 luxieum@203.0.113.42"
    exit 1
fi

echo ""
echo "═══════════════════════════════════════════════════════════════"
