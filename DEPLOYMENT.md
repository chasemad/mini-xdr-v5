# Mini-XDR Deployment Guide

Complete deployment instructions for the Mini-XDR Security Detection & Response System.

## 🚀 Quick Start (Automated)

For the fastest deployment, use our automated startup script:

```bash
git clone https://github.com/chasemad/mini-xdr.git
cd mini-xdr
./scripts/start-all.sh
```

This script will:
- ✅ Check system requirements (Python 3.8+, Node.js 18+)
- ✅ Create Python virtual environment automatically
- ✅ Install all dependencies (Python + Node.js)
- ✅ Set up configuration files from templates
- ✅ Initialize database
- ✅ Test honeypot connectivity
- ✅ Start all services with health checks
- ✅ Verify system is working end-to-end

## 📋 Prerequisites

### Required Software
- **Python 3.8+** ([Download](https://python.org))
- **Node.js 18+** ([Download](https://nodejs.org))
- **SSH client** (pre-installed on macOS/Linux)
- **curl** (for API testing)

### Required Infrastructure
- **Honeypot VM** running Cowrie honeypot
- **SSH access** to honeypot for containment actions
- **Network connectivity** between XDR host and honeypot

## ⚙️ Configuration

### 1. Backend Configuration

Copy and edit the backend configuration:
```bash
cd backend
cp env.example .env
```

Edit `backend/.env` with your settings:
```bash
# Honeypot Connection (REQUIRED)
HONEYPOT_HOST=192.168.1.100        # Your honeypot VM IP
HONEYPOT_USER=xdrops               # SSH user for containment
HONEYPOT_SSH_KEY=~/.ssh/xdrops_id_ed25519  # SSH private key path
HONEYPOT_SSH_PORT=22022            # SSH port on honeypot

# API Security (RECOMMENDED)
API_KEY=your_secret_api_key_here   # Secure API access

# LLM Integration (OPTIONAL - for AI analysis)
OPENAI_API_KEY=sk-your-openai-key  # OpenAI API key
# OR
XAI_API_KEY=xai-your-x-api-key     # X.AI/Grok API key
```

### 2. Frontend Configuration

Copy and edit the frontend configuration:
```bash
cd frontend
cp env.local.example .env.local
```

Edit `frontend/.env.local`:
```bash
# API Connection
NEXT_PUBLIC_API_BASE=http://localhost:8000  # Backend API URL
NEXT_PUBLIC_API_KEY=your_secret_api_key_here  # Same as backend API_KEY
```

### 3. SSH Key Setup

Generate SSH keys for honeypot access:
```bash
# Generate key pair
ssh-keygen -t ed25519 -f ~/.ssh/xdrops_id_ed25519

# Copy public key to honeypot
ssh-copy-id -i ~/.ssh/xdrops_id_ed25519.pub -p 22022 xdrops@<honeypot-ip>

# Test connection
ssh -p 22022 -i ~/.ssh/xdrops_id_ed25519 xdrops@<honeypot-ip> sudo ufw status
```

## 🏗️ Manual Installation

If you prefer manual setup or the automated script fails:

### Backend Setup
```bash
cd backend

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Install MCP dependencies (for LLM integration)
npm install

# Initialize database
python -c "
import asyncio
from app.db import init_db
asyncio.run(init_db())
"

# Start backend
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Frontend Setup
```bash
cd frontend

# Install dependencies
npm install

# Start frontend
npm run dev
```

### MCP Server (Optional)
```bash
cd backend

# Start MCP server for LLM integration
npm run mcp
```

## 🍯 Honeypot Setup

### On Your Honeypot VM

1. **Install Mini-XDR honeypot components:**
```bash
# Copy setup script to honeypot
scp ops/honeypot-setup.sh user@honeypot-vm:

# Run setup
ssh user@honeypot-vm sudo ./honeypot-setup.sh
```

2. **Configure Cowrie with JSON logging**
3. **Set up Fluent Bit log forwarding:**
```bash
# Copy fluent-bit setup
scp ops/fluent-bit-install.sh user@honeypot-vm:
scp ops/fluent-bit.conf user@honeypot-vm:

# Install and configure
ssh user@honeypot-vm sudo ./fluent-bit-install.sh
```

## 🧪 Testing & Verification

### Automated Health Checks
The startup script performs comprehensive health checks:
- ✅ API endpoint testing
- ✅ Database connectivity
- ✅ SSH connectivity to honeypot
- ✅ UFW access verification
- ✅ Event ingestion testing
- ✅ LLM integration status

### Manual Testing

1. **Test Event Ingestion:**
```bash
curl -X POST http://localhost:8000/ingest/cowrie \
  -H 'Content-Type: application/json' \
  -d '{"eventid":"cowrie.login.failed","src_ip":"192.168.1.100"}'
```

2. **Test SSH Connectivity:**
```bash
curl http://localhost:8000/test/ssh
```

3. **Simulate Attack:**
```bash
# Copy and run attack simulation
scp ops/test-attack.sh user@kali-vm:
ssh user@kali-vm ./test-attack.sh <honeypot-ip> 8
```

## 🌐 Access Points

Once deployed, access your Mini-XDR system at:

- **Web Dashboard:** http://localhost:3000
- **API Documentation:** http://localhost:8000/docs
- **Backend API:** http://localhost:8000
- **Health Check:** http://localhost:8000/health

## 🔧 Troubleshooting

### Common Issues

1. **MCP Server "Not Running":**
   - This is normal - MCP servers run on-demand
   - Connect AI assistants via stdio to use MCP tools

2. **SSH Connection Failed:**
   - Verify honeypot VM is accessible
   - Check SSH key permissions: `chmod 600 ~/.ssh/xdrops_id_ed25519`
   - Test manual SSH connection

3. **Port Already in Use:**
   - The startup script automatically kills existing processes
   - Manually stop: `./scripts/stop-all.sh`

4. **Dependencies Missing:**
   - Re-run: `./scripts/start-all.sh` (auto-installs dependencies)
   - Manual install: see Manual Installation section

### Log Files
Check these files for detailed error information:
- Backend: `backend/backend.log`
- Frontend: `frontend/frontend.log`
- MCP: `backend/mcp.log`

## 🔒 Security Considerations

### Files Excluded from Git
The following sensitive files are automatically excluded:
- `backend/.env` (API keys, honeypot credentials)
- `frontend/.env.local` (API configuration)
- `*.log` (runtime logs)
- `*.db` (database files)
- SSH private keys

### Production Deployment
For production use:
1. Change default API keys
2. Use HTTPS for frontend access
3. Implement proper firewall rules
4. Use dedicated database (PostgreSQL)
5. Set up log rotation
6. Configure monitoring and alerting

## 📞 Support

- **Issues:** [GitHub Issues](https://github.com/chasemad/mini-xdr/issues)
- **Documentation:** This repository's README and guides
- **System Status:** Use `./scripts/system-status.sh`
