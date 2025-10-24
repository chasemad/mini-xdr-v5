#!/bin/bash
# ==============================================================================
# Generate Secure Keys for Mini-XDR Production Deployment
# ==============================================================================
# Generates cryptographically secure keys for:
# - JWT_SECRET_KEY (for JWT token signing)
# - ENCRYPTION_KEY (for sensitive data encryption)
# - API keys and secrets
# ==============================================================================

set -e

echo "======================================"
echo "Mini-XDR Secure Key Generation"
echo "======================================"
echo ""

# Check if openssl is available
if ! command -v openssl &> /dev/null; then
    echo "❌ Error: openssl is required but not installed"
    exit 1
fi

# Create secure directory for keys
KEYS_DIR="$(pwd)/.secure-keys"
mkdir -p "$KEYS_DIR"
chmod 700 "$KEYS_DIR"

KEYS_FILE="$KEYS_DIR/mini-xdr-secrets-$(date +%Y%m%d-%H%M%S).env"

echo "🔐 Generating cryptographically secure keys..."
echo ""

# Generate JWT Secret Key (512-bit hex)
JWT_SECRET=$(openssl rand -hex 64)
echo "✅ JWT_SECRET_KEY generated (128 characters)"

# Generate Encryption Key (256-bit base64)
ENCRYPTION_KEY=$(openssl rand -base64 32)
echo "✅ ENCRYPTION_KEY generated (32 bytes, base64 encoded)"

# Generate API Key for internal services
API_KEY=$(openssl rand -hex 32)
echo "✅ API_KEY generated (64 characters)"

# Generate Agent HMAC Key
AGENT_HMAC_KEY=$(openssl rand -hex 32)
echo "✅ AGENT_HMAC_KEY generated (64 characters)"

echo ""
echo "📝 Writing keys to: $KEYS_FILE"
echo ""

# Write to secure file
cat > "$KEYS_FILE" << EOF
# ==============================================================================
# Mini-XDR Production Secrets
# Generated: $(date -u +"%Y-%m-%d %H:%M:%S UTC")
# ==============================================================================
# ⚠️  CRITICAL: Keep this file secure. Never commit to version control.
# ==============================================================================

# JWT Token Signing Key (for user authentication)
JWT_SECRET_KEY=$JWT_SECRET

# Data Encryption Key (for sensitive data at rest)
ENCRYPTION_KEY=$ENCRYPTION_KEY

# API Key (for internal service authentication)
API_KEY=$API_KEY

# Agent HMAC Key (for agent authentication)
AGENT_HMAC_KEY=$AGENT_HMAC_KEY

# ==============================================================================
# Kubernetes Secret Creation Commands
# ==============================================================================
# Create secret in Kubernetes:
#
# kubectl create secret generic mini-xdr-secrets \\
#   --from-literal=JWT_SECRET_KEY="$JWT_SECRET" \\
#   --from-literal=ENCRYPTION_KEY="$ENCRYPTION_KEY" \\
#   --from-literal=API_KEY="$API_KEY" \\
#   --from-literal=AGENT_HMAC_KEY="$AGENT_HMAC_KEY" \\
#   --namespace mini-xdr
#
# Or update existing secret:
#
# kubectl delete secret mini-xdr-secrets -n mini-xdr
# kubectl create secret generic mini-xdr-secrets \\
#   --from-file=.env=$KEYS_FILE \\
#   --namespace mini-xdr
#
# ==============================================================================
# AWS Secrets Manager Commands
# ==============================================================================
# Store in AWS Secrets Manager (recommended):
#
# aws secretsmanager create-secret \\
#   --name mini-xdr/production/secrets \\
#   --description "Mini-XDR production secrets" \\
#   --secret-string file://$KEYS_FILE \\
#   --region us-east-1
#
# ==============================================================================

EOF

chmod 600 "$KEYS_FILE"

echo "✅ Keys generated successfully!"
echo ""
echo "📄 Keys file: $KEYS_FILE"
echo "🔒 Permissions: 600 (read/write for owner only)"
echo ""
echo "⚠️  IMPORTANT SECURITY NOTES:"
echo "  1. Store this file in a secure password manager"
echo "  2. Never commit this file to version control"
echo "  3. Rotate keys every 90 days"
echo "  4. Delete this file after deploying to production"
echo ""
echo "🚀 Next Steps:"
echo "  1. Review the keys in: $KEYS_FILE"
echo "  2. Deploy to Kubernetes or AWS Secrets Manager (see commands in file)"
echo "  3. Update application configuration to use new keys"
echo "  4. Test authentication after deployment"
echo "  5. Securely delete this file: shred -vfz -n 10 $KEYS_FILE"
echo ""
echo "======================================"
