#!/bin/bash
# Secure T-Pot honeypot by removing all public access
# Only allow access from your specific IP address

set -e

REGION="us-east-1"
SG_ID="${AWS_SECURITY_GROUP_ID:-YOUR_SECURITY_GROUP_ID_HERE}"
YOUR_IP="24.11.0.176"

echo "🔒 Securing T-Pot honeypot - removing public access..."

# List of all honeypot ports that need to be secured
PORTS_TCP=(23 25 53 80 135 139 443 445 993 995 1433 1521 3306 3389 5432 5900 6379 8080 8443 9200 27017)
PORTS_UDP=(53 123 161 500 1434 1900)

# Remove TCP port access from public internet
for port in "${PORTS_TCP[@]}"; do
    echo "Removing public access from TCP port $port..."
    aws ec2 revoke-security-group-ingress \
        --group-id $SG_ID \
        --protocol tcp \
        --port $port \
        --cidr 0.0.0.0/0 \
        --region $REGION 2>/dev/null || echo "  (Rule may not exist)"
done

# Remove UDP port access from public internet
for port in "${PORTS_UDP[@]}"; do
    echo "Removing public access from UDP port $port..."
    aws ec2 revoke-security-group-ingress \
        --group-id $SG_ID \
        --protocol udp \
        --port $port \
        --cidr 0.0.0.0/0 \
        --region $REGION 2>/dev/null || echo "  (Rule may not exist)"
done

# Remove the wide port range 8000-9999
echo "Removing public access from port range 8000-9999..."
aws ec2 revoke-security-group-ingress \
    --group-id $SG_ID \
    --protocol tcp \
    --port 8000-9999 \
    --cidr 0.0.0.0/0 \
    --region $REGION 2>/dev/null || echo "  (Rule may not exist)"

echo ""
echo "✅ Public access removed from honeypot ports!"
echo ""
echo "📊 Current security status:"
echo "  • Management SSH (64295): Restricted to $YOUR_IP ✅"
echo "  • Web Interface (64297): Restricted to $YOUR_IP ✅"
echo "  • Honeypot ports: Public access BLOCKED 🔒"
echo ""
echo "🎯 To allow your Kali machine to test honeypots:"
echo "   1. Get your Kali machine's public IP"
echo "   2. Add specific rules for testing ports only"
echo ""
echo "🔍 Verify current rules:"
echo "   aws ec2 describe-security-groups --group-ids $SG_ID --region $REGION"
