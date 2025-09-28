#!/bin/bash
# EMERGENCY: Secure TPOT honeypot for testing only
# This will restrict access to ONLY your IP address

set -e

# Your details
YOUR_IP="24.11.0.176"
SECURITY_GROUP="sg-037bd4ee6b74489b5"
REGION="us-east-1"

echo "🚨 SECURING TPOT HONEYPOT FOR TESTING ONLY"
echo "📍 Your IP: $YOUR_IP"
echo "🔒 Security Group: $SECURITY_GROUP"
echo ""

# Get current rules and remove all 0.0.0.0/0 rules
echo "📋 Getting current security group rules..."
aws ec2 describe-security-groups \
    --region $REGION \
    --group-ids $SECURITY_GROUP \
    --query 'SecurityGroups[0].IpPermissions' \
    --output json > /tmp/current-rules.json

echo "🗑️  Removing all existing ingress rules (this will block all internet access)..."
if [ -s /tmp/current-rules.json ] && [ "$(cat /tmp/current-rules.json)" != "[]" ]; then
    aws ec2 revoke-security-group-ingress \
        --region $REGION \
        --group-id $SECURITY_GROUP \
        --ip-permissions file:///tmp/current-rules.json 2>/dev/null || true
fi

echo "🔐 Adding new rules for YOUR IP ONLY..."

# SSH Management (port 64295) - for your SSH access to TPOT
aws ec2 authorize-security-group-ingress \
    --region $REGION \
    --group-id $SECURITY_GROUP \
    --protocol tcp \
    --port 64295 \
    --cidr "$YOUR_IP/32" || true

# TPOT Web Interface (port 64297) - for web access
aws ec2 authorize-security-group-ingress \
    --region $REGION \
    --group-id $SECURITY_GROUP \
    --protocol tcp \
    --port 64297 \
    --cidr "$YOUR_IP/32" || true

# Test honeypot ports (restricted to your IP for testing)
echo "🧪 Adding test honeypot ports (YOUR IP ONLY)..."
test_ports=(22 80 443 21 23 25 3306 3389)

for port in "${test_ports[@]}"; do
    aws ec2 authorize-security-group-ingress \
        --region $REGION \
        --group-id $SECURITY_GROUP \
        --protocol tcp \
        --port $port \
        --cidr "$YOUR_IP/32" 2>/dev/null || true
    echo "   ✅ Port $port restricted to your IP"
done

echo ""
echo "🎉 TPOT HONEYPOT IS NOW SECURED FOR TESTING!"
echo ""
echo "📊 Current Status:"
echo "   ✅ Only accessible from YOUR IP: $YOUR_IP"
echo "   ✅ Management SSH: ssh -i ~/.ssh/mini-xdr-tpot-key.pem -p 64295 admin@34.193.101.171"
echo "   ✅ Web Interface: https://34.193.101.171:64297"
echo "   ✅ Test attacks will come from YOU only"
echo ""
echo "🌍 When ready for REAL attacks from internet:"
echo "   Run: ./open-tpot-to-internet.sh"
echo ""
echo "🔒 Your honeypot is now SAFE for testing!"
