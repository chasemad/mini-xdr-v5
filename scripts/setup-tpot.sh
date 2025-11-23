#!/bin/bash

# T-Pot Honeypot Integration Setup Script
# This script sets up the Mini-XDR to T-Pot connection

set -e

echo "======================================"
echo "T-Pot Integration Setup"
echo "======================================"
echo ""

# Check if we're in the correct directory
if [ ! -f "backend/requirements.txt" ]; then
    echo "❌ Error: Please run this script from the mini-xdr root directory"
    exit 1
fi

# Check for Python virtual environment
if [ ! -d "backend/venv" ]; then
    echo "❌ Error: Backend virtual environment not found at backend/venv"
    echo "Please create it first: cd backend && python3 -m venv venv"
    exit 1
fi

echo "✅ Found virtual environment"
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source backend/venv/bin/activate

# Install asyncssh dependency
echo "Installing asyncssh dependency..."
pip install asyncssh==2.14.2

echo "✅ Dependencies installed"
echo ""

# Check if .env file exists
if [ ! -f "backend/.env" ]; then
    echo "📝 Creating backend/.env file..."

    cat > backend/.env << 'EOF'
# Mini-XDR Backend Configuration

# API Configuration
API_HOST=127.0.0.1
API_PORT=8000
UI_ORIGIN=http://localhost:3000

# Database
DATABASE_URL=sqlite+aiosqlite:///./xdr.db

# T-Pot Honeypot Configuration
TPOT_HOST=24.11.0.176
TPOT_SSH_PORT=64295
TPOT_WEB_PORT=64297
TPOT_API_KEY=demo-tpot-api-key

HONEYPOT_HOST=24.11.0.176
HONEYPOT_USER=luxieum
HONEYPOT_SSH_KEY=~/.ssh/id_rsa
HONEYPOT_SSH_PORT=64295

# Detection Configuration
FAIL_WINDOW_SECONDS=60
FAIL_THRESHOLD=6
AUTO_CONTAIN=false
ALLOW_PRIVATE_IP_BLOCKING=true

# LLM Configuration (Optional)
LLM_PROVIDER=openai
# OPENAI_API_KEY=your-key-here
# OPENAI_MODEL=gpt-4

# Threat Intelligence (Optional)
# ABUSEIPDB_API_KEY=your-key-here
# VIRUSTOTAL_API_KEY=your-key-here
EOF

    echo "✅ Created backend/.env with T-Pot configuration"
    echo ""
else
    echo "⚠️  backend/.env already exists"
    echo "Checking if T-Pot configuration is present..."

    if ! grep -q "TPOT_HOST" backend/.env; then
        echo "Adding T-Pot configuration to existing .env..."
        cat >> backend/.env << 'EOF'

# T-Pot Honeypot Configuration
TPOT_HOST=24.11.0.176
TPOT_SSH_PORT=64295
TPOT_WEB_PORT=64297
TPOT_API_KEY=demo-tpot-api-key

HONEYPOT_HOST=24.11.0.176
HONEYPOT_USER=luxieum
HONEYPOT_SSH_KEY=~/.ssh/id_rsa
HONEYPOT_SSH_PORT=64295
EOF
        echo "✅ Added T-Pot configuration to .env"
    else
        echo "✅ T-Pot configuration already present"
    fi
    echo ""
fi

# Test SSH connectivity
echo "Testing T-Pot connectivity..."
echo "Attempting SSH connection to luxieum@24.11.0.176:64295"
echo "(This will timeout if your IP is not 172.16.110.1)"
echo ""

if timeout 5 ssh -p 64295 -o ConnectTimeout=5 -o StrictHostKeyChecking=no luxieum@24.11.0.176 "echo 'Connection successful'" 2>/dev/null; then
    echo "✅ SSH connection successful!"
    echo ""
else
    echo "⚠️  SSH connection failed or timed out"
    echo ""
    echo "Possible reasons:"
    echo "1. Your IP is not 172.16.110.1 (only this IP is allowed)"
    echo "2. Password authentication not configured"
    echo "3. Network connectivity issues"
    echo ""
    echo "To test manually:"
    echo "  ssh -p 64295 luxieum@24.11.0.176"
    echo "  Password: demo-tpot-api-key"
    echo ""
fi

# Display next steps
echo "======================================"
echo "Setup Complete!"
echo "======================================"
echo ""
echo "Next steps:"
echo ""
echo "1. Start the backend:"
echo "   cd backend"
echo "   python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000"
echo ""
echo "2. Start the frontend (in a new terminal):"
echo "   cd frontend"
echo "   npm run dev"
echo ""
echo "3. Access the Honeypot dashboard:"
echo "   http://localhost:3000/honeypot"
echo ""
echo "4. Check T-Pot Web UI (optional):"
echo "   https://24.11.0.176:64297"
echo "   Username: admin"
echo "   Password: TpotSecure2024!"
echo ""
echo "For detailed setup instructions, see:"
echo "   TPOT_SETUP.md"
echo "   TPOT_INTEGRATION_SUMMARY.md"
echo ""
echo "To generate test attacks:"
echo "   ssh root@24.11.0.176 -p 22  (try wrong password)"
echo "   curl http://24.11.0.176/admin"
echo "   nmap -p 1-100 24.11.0.176"
echo ""
echo "======================================"
