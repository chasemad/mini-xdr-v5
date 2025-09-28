#!/bin/bash
# Test script to simulate SSH brute-force attack
# Run from Kali VM or attacker machine

set -e

HONEYPOT_IP=${1:-"10.0.0.23"}
ATTEMPTS=${2:-8}

echo "=== Mini-XDR Attack Simulation ==="
echo "Target: $HONEYPOT_IP:2222"
echo "Attempts: $ATTEMPTS"
echo ""

for i in $(seq 1 $ATTEMPTS); do
    echo "Attempt $i/$ATTEMPTS..."
    
    # Random username and password attempts
    USERS=("admin" "root" "user" "test" "guest" "oracle" "postgres")
    USER=${USERS[$RANDOM % ${#USERS[@]}]}
    
    # Use sshpass or timeout for failed attempts
    timeout 5 ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=yes -o UserKnownHostsFile=~/.ssh/known_hosts -p 2222 "$USER@$HONEYPOT_IP" exit 2>/dev/null || true
    
    sleep 1
done

echo ""
echo "Attack simulation complete!"
echo "Check Mini-XDR UI for incident detection:"
echo "http://10.0.0.123:3000/incidents"
