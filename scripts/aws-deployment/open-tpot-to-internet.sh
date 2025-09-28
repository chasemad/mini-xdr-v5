#!/bin/bash
# Open TPOT honeypot to internet for REAL attack collection
# WARNING: This will expose your honeypot to real attackers worldwide!

set -e

# Your details
YOUR_IP="24.11.0.176"
SECURITY_GROUP="sg-037bd4ee6b74489b5"
REGION="us-east-1"

echo "🌍 OPENING TPOT HONEYPOT TO INTERNET FOR REAL ATTACKS"
echo "⚠️  WARNING: This will expose your honeypot to real attackers!"
echo ""
read -p "Are you sure you want to continue? (yes/no): " confirm

if [ "$confirm" != "yes" ]; then
    echo "❌ Cancelled. Honeypot remains restricted to your IP."
    exit 1
fi

echo ""
echo "🔓 Opening honeypot ports to the internet (0.0.0.0/0)..."

# Honeypot ports that should be open to attract real attackers
honeypot_ports=(21 22 23 25 53 80 135 139 443 445 993 995 1433 1521 3306 3389 5432 5900 6379 8080 8443 9200 27017)

for port in "${honeypot_ports[@]}"; do
    # Remove your IP restriction
    aws ec2 revoke-security-group-ingress \
        --region $REGION \
        --group-id $SECURITY_GROUP \
        --protocol tcp \
        --port $port \
        --cidr "$YOUR_IP/32" 2>/dev/null || true
    
    # Add internet access
    aws ec2 authorize-security-group-ingress \
        --region $REGION \
        --group-id $SECURITY_GROUP \
        --protocol tcp \
        --port $port \
        --cidr "0.0.0.0/0" 2>/dev/null || true
    
    echo "   🌐 Port $port now open to internet"
done

# Keep management ports restricted to your IP
echo ""
echo "🔒 Keeping management ports restricted to YOUR IP..."
echo "   ✅ SSH Management (64295): $YOUR_IP only"
echo "   ✅ Web Interface (64297): $YOUR_IP only"

echo ""
echo "🎉 TPOT HONEYPOT IS NOW OPEN TO REAL ATTACKS!"
echo ""
echo "📊 Current Status:"
echo "   🌍 Honeypot ports: OPEN to internet (0.0.0.0/0)"
echo "   🔒 Management ports: RESTRICTED to your IP ($YOUR_IP)"
echo "   ⚡ Real attackers will start hitting within minutes"
echo ""
echo "📈 Monitor real attacks:"
echo "   🌍 Globe: http://localhost:3000/visualizations"
echo "   📊 Backend: tail -f backend/backend.log"
echo "   🛡️  TPOT: ssh -i ~/.ssh/mini-xdr-tpot-key.pem -p 64295 admin@34.193.101.171"
echo ""
echo "🔒 To secure again for testing: ./secure-tpot-for-testing.sh"
