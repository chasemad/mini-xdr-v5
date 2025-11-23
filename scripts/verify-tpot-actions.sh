#!/bin/bash

# Verify T-Pot AI Agent Action Capabilities

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  T-Pot AI Agent Action Verification                          ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

TPOT_HOST="203.0.113.42"
TPOT_PORT="64295"
TPOT_USER="luxieum"

echo "Testing T-Pot connectivity and permissions..."
echo ""

# Test 1: SSH Connection
echo "Test 1: SSH Connection"
timeout 5 ssh -p $TPOT_PORT -o StrictHostKeyChecking=no $TPOT_USER@$TPOT_HOST "echo 'Connected'" 2>&1 | grep -q "Connected" && echo "  ✅ SSH connection working" || echo "  ❌ SSH connection failed"

# Test 2: Passwordless sudo for UFW
echo "Test 2: Passwordless sudo for UFW"
ssh -p $TPOT_PORT $TPOT_USER@$TPOT_HOST "sudo -n ufw status" 2>&1 | grep -q "Status" && echo "  ✅ UFW passwordless sudo working" || echo "  ❌ UFW requires password"

# Test 3: Docker access
echo "Test 3: Docker commands"
ssh -p $TPOT_PORT $TPOT_USER@$TPOT_HOST "sudo -n docker ps --format '{{.Names}}' | head -1" 2>&1 | grep -v "password" | grep -q "." && echo "  ✅ Docker commands working" || echo "  ❌ Docker requires password"

# Test 4: Actual IP block test
echo "Test 4: IP Blocking capability"
ssh -p $TPOT_PORT $TPOT_USER@$TPOT_HOST "sudo -n ufw deny from 1.2.3.4 2>&1" | grep -q "Rule added" && echo "  ✅ IP blocking working" || echo "  ⚠️  IP block may need verification"

# Check if IP was blocked
ssh -p $TPOT_PORT $TPOT_USER@$TPOT_HOST "sudo -n ufw status | grep 1.2.3.4" 2>&1 | grep -q "1.2.3.4" && echo "  ✅ IP 1.2.3.4 is blocked" || echo "  ⚠️  IP not in firewall rules"

# Cleanup test
echo "Test 5: Cleanup test block"
ssh -p $TPOT_PORT $TPOT_USER@$TPOT_HOST "sudo -n ufw status numbered | grep 1.2.3.4 | head -1 | awk '{print \$1}' | tr -d '[]'" 2>&1 > /tmp/rule_num
if [ -s /tmp/rule_num ]; then
    RULE_NUM=$(cat /tmp/rule_num)
    ssh -p $TPOT_PORT $TPOT_USER@$TPOT_HOST "echo 'y' | sudo -n ufw delete $RULE_NUM" 2>&1 | grep -q "Deleting" && echo "  ✅ Unblock working" || echo "  ⚠️  Unblock may need verification"
fi

echo ""
echo "══════════════════════════════════════════════════════════════"
echo ""

# Test via Mini-XDR API
echo "Testing via Mini-XDR API..."
echo ""

# Test block via API
echo "Blocking IP 9.9.9.9 via Mini-XDR API..."
RESULT=$(curl -s -X POST http://localhost:8000/api/tpot/firewall/block \
  -H "Content-Type: application/json" \
  -d '{"ip_address": "9.9.9.9"}' 2>/dev/null)

if echo "$RESULT" | grep -q '"success": true'; then
    echo "  ✅ API IP blocking working!"
    echo "  Response: $RESULT" | python3 -m json.tool 2>/dev/null || echo "$RESULT"
else
    echo "  ❌ API IP blocking failed"
    echo "  Response: $RESULT"
fi

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Verification Complete                                       ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "If all tests pass, your AI agents can now execute real actions!"
echo ""
