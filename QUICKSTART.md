# Mini-XDR Quick Start Guide

Get your Mini-XDR system running in minutes!

## 🚀 One-Command Setup

```bash
cd mini-xdr
./scripts/setup.sh
```

This script will:
- Set up Python virtual environment
- Install all dependencies (Python + Node.js)
- Initialize the SQLite database
- Create configuration files

## ⚙️ Configuration

### 1. Backend Configuration

Edit `backend/.env`:

```bash
# Required: Honeypot SSH access
HONEYPOT_HOST=10.0.0.23
HONEYPOT_USER=xdrops
HONEYPOT_SSH_KEY=~/.ssh/xdrops_id_ed25519
HONEYPOT_SSH_PORT=22022

# Optional: API security
API_KEY=your_secret_key

# Optional: LLM integration
LLM_PROVIDER=openai
OPENAI_API_KEY=your_openai_key
```

### 2. Frontend Configuration

Edit `frontend/.env.local`:

```bash
NEXT_PUBLIC_API_BASE=http://10.0.0.123:8000
NEXT_PUBLIC_API_KEY=your_secret_key
```

## 🏃‍♂️ Start All Services

```bash
./scripts/start-all.sh
```

This starts:
- Backend API server (port 8000)
- Frontend web UI (port 3000)
- MCP tools server (stdio)

## 🌐 Access Points

- **Web UI**: http://localhost:3000
- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

## 📡 Honeypot Setup

On your honeypot VM, run:

```bash
# Copy and run the setup script
scp ops/honeypot-setup.sh user@honeypot-vm:
ssh user@honeypot-vm sudo ./honeypot-setup.sh

# Generate SSH keys for XDR access
ssh-keygen -t ed25519 -f ~/.ssh/xdrops_id_ed25519

# Copy public key to honeypot
ssh-copy-id -i ~/.ssh/xdrops_id_ed25519.pub -p 22022 xdrops@honeypot-vm
```

## 🧪 Test the System

### 1. Manual Event Test

```bash
curl -X POST http://localhost:8000/ingest/cowrie \
  -H 'Content-Type: application/json' \
  -d '{"eventid":"cowrie.login.failed","src_ip":"192.168.1.100"}'
```

### 2. Simulate Attack

From an attacker machine:

```bash
# Copy test script
scp ops/test-attack.sh user@kali-vm:

# Run attack simulation
ssh user@kali-vm ./test-attack.sh 10.0.0.23 8
```

### 3. Verify Detection

Check the web UI at http://localhost:3000/incidents for new incidents.

## 🔧 Manual Operations

### Backend Only

```bash
cd backend
source .venv/bin/activate
uvicorn app.main:app --reload
```

### Frontend Only

```bash
cd frontend
npm run dev
```

### MCP Server Only

```bash
cd backend
npm run mcp
```

## 📊 Expected Workflow

1. **Attack occurs** → Cowrie logs to JSON file
2. **Fluent Bit** → Forwards events to Mini-XDR `/ingest/cowrie`
3. **Detection** → Sliding window analysis triggers incident
4. **Triage** → AI analysis provides severity and recommendations
5. **Response** → Manual or automatic IP blocking via UFW
6. **Monitoring** → SOC dashboard shows status and actions

## 🔍 Troubleshooting

### Backend Issues

```bash
# Check logs
cd backend && source .venv/bin/activate
python -c "from app.config import settings; print(settings.dict())"

# Test database
python -c "
import asyncio
from app.db import init_db
asyncio.run(init_db())
"
```

### SSH Connection Issues

```bash
# Test SSH access
ssh -p 22022 -i ~/.ssh/xdrops_id_ed25519 xdrops@10.0.0.23 sudo ufw status

# Check key permissions
chmod 600 ~/.ssh/xdrops_id_ed25519
chmod 644 ~/.ssh/xdrops_id_ed25519.pub
```

### Frontend Issues

```bash
# Check environment
cd frontend
cat .env.local

# Test API connectivity
curl http://localhost:8000/health
```

## 📈 Next Steps

1. **Configure Fluent Bit** on honeypot to forward Cowrie logs
2. **Set up LLM integration** for AI-powered triage
3. **Enable auto-contain** for automatic threat response
4. **Monitor dashboards** for security incidents
5. **Customize detections** for your environment

## 🆘 Support

- Check `README.md` for detailed documentation
- Review individual component READMEs in `backend/` and `frontend/`
- Examine operational scripts in `ops/`
- Test with provided simulation tools

Your Mini-XDR system is now ready to detect and respond to SSH brute-force attacks! 🛡️
