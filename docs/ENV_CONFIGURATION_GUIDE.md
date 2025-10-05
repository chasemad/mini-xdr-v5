# Mini-XDR Environment Configuration Guide

## 📋 Overview

All API keys and configuration are stored in `backend/.env` (no AWS Secrets Manager needed).

## 📍 Location

```
/Users/chasemad/Desktop/mini-xdr/backend/.env
```

## 🔑 Current Configuration

Your `.env` file contains all necessary API keys and configuration:

```bash
# View your current configuration
cd /Users/chasemad/Desktop/mini-xdr/backend
cat .env
```

## 🔧 Key Configuration Items

### Mini-XDR API Key
```bash
API_KEY=your-api-key-here
```

Used for:
- API authentication
- Workflow trigger management
- Incident API access

### LLM API Keys
```bash
OPENAI_API_KEY=your-openai-key
XAI_API_KEY=your-xai-key
```

Used for:
- AI-powered threat analysis
- Natural language workflow creation
- Automated attribution

### Threat Intel API Keys
```bash
ABUSEIPDB_API_KEY=your-abuseipdb-key
VIRUSTOTAL_API_KEY=your-virustotal-key
```

Used for:
- IP reputation lookups
- Malware analysis
- Threat intelligence enrichment

### T-Pot Configuration
```bash
TPOT_API_KEY=your-tpot-key
TPOT_HOST=your-tpot-ip-or-hostname
```

Used for:
- T-Pot honeypot integration
- Log forwarding authentication
- Event ingestion

## 🚀 Using Your Configuration

### 1. Access Workflows API

```bash
cd backend

# Get your API key
API_KEY=$(grep '^API_KEY=' .env | cut -d '=' -f2)

# List all workflow triggers
curl http://localhost:8000/api/triggers \
  -H "X-API-Key: $API_KEY" | jq

# Get trigger statistics
curl http://localhost:8000/api/triggers/stats/summary \
  -H "X-API-Key: $API_KEY" | jq
```

### 2. Test T-Pot Integration

```bash
cd backend

# Get T-Pot API key
TPOT_KEY=$(grep '^TPOT_API_KEY=' .env | cut -d '=' -f2)

# Test log ingestion
curl -X POST http://localhost:8000/ingest/multi \
  -H "Authorization: Bearer $TPOT_KEY" \
  -H "Content-Type: application/json" \
  -d '[{"test": "event"}]'
```

### 3. Verify LLM Configuration

```bash
cd backend

# Check if OpenAI key is set
grep '^OPENAI_API_KEY=' .env && echo "✅ OpenAI configured" || echo "⚠️ OpenAI not configured"

# Check if XAI key is set
grep '^XAI_API_KEY=' .env && echo "✅ XAI configured" || echo "⚠️ XAI not configured"
```

## 🔄 Updating Configuration

### Add or Update a Key

```bash
cd backend

# Edit .env file
nano .env

# Or append a new key
echo "NEW_API_KEY=your-new-key" >> .env

# Restart backend to pick up changes
pkill -f uvicorn
nohup uvicorn app.main:app --host 0.0.0.0 --port 8000 > backend.log 2>&1 &
```

### Verify Changes

```bash
# Check backend is running
curl http://localhost:8000/health

# Test with new configuration
# (use appropriate API endpoint)
```

## 🔒 Security Best Practices

### 1. Keep .env Private

```bash
# Verify .env is in .gitignore
cd /Users/chasemad/Desktop/mini-xdr
grep "\.env" .gitignore || echo "⚠️ Add .env to .gitignore!"
```

### 2. Use Strong API Keys

```bash
# Generate a strong API key
openssl rand -hex 32

# Update in .env
echo "API_KEY=$(openssl rand -hex 32)" >> backend/.env
```

### 3. Rotate Keys Regularly

```bash
# Backup old .env
cp backend/.env backend/.env.backup-$(date +%Y%m%d)

# Update keys
nano backend/.env

# Restart backend
pkill -f uvicorn && sleep 2
cd backend && nohup uvicorn app.main:app --host 0.0.0.0 --port 8000 > backend.log 2>&1 &
```

## 📝 Environment Variables Reference

### Core Settings
```bash
API_KEY=                    # Mini-XDR API authentication
DATABASE_URL=              # SQLite database path (default: sqlite+aiosqlite:///./xdr.db)
```

### LLM Configuration
```bash
OPENAI_API_KEY=            # OpenAI GPT-4 API key
OPENAI_MODEL=              # Model name (default: gpt-4)
XAI_API_KEY=               # X.AI Grok API key
XAI_MODEL=                 # Model name (default: grok-beta)
LLM_PROVIDER=              # Provider (openai or xai)
```

### Threat Intelligence
```bash
ABUSEIPDB_API_KEY=         # AbuseIPDB API key
VIRUSTOTAL_API_KEY=        # VirusTotal API key
```

### T-Pot Honeypot
```bash
TPOT_API_KEY=              # T-Pot authentication key
TPOT_HOST=                 # T-Pot IP or hostname
TPOT_SSH_PORT=             # SSH port (default: 64295)
TPOT_WEB_PORT=             # Web interface port (default: 64297)
```

### Detection Configuration
```bash
FAIL_THRESHOLD=            # Failed login threshold (default: 6)
FAIL_WINDOW_SECONDS=       # Detection window (default: 60)
AUTO_CONTAIN=              # Auto-containment (default: false)
```

### Agent Configuration
```bash
CONTAINMENT_AGENT_DEVICE_ID=      # Containment agent device ID
CONTAINMENT_AGENT_PUBLIC_ID=      # Containment agent public ID
CONTAINMENT_AGENT_HMAC_KEY=       # Containment agent HMAC key
CONTAINMENT_AGENT_SECRET=         # Containment agent secret

ATTRIBUTION_AGENT_DEVICE_ID=      # Attribution agent device ID
ATTRIBUTION_AGENT_PUBLIC_ID=      # Attribution agent public ID
ATTRIBUTION_AGENT_HMAC_KEY=       # Attribution agent HMAC key
ATTRIBUTION_AGENT_SECRET=         # Attribution agent secret

FORENSICS_AGENT_DEVICE_ID=        # Forensics agent device ID
FORENSICS_AGENT_PUBLIC_ID=        # Forensics agent public ID
FORENSICS_AGENT_HMAC_KEY=         # Forensics agent HMAC key
FORENSICS_AGENT_SECRET=           # Forensics agent secret
```

## 🧪 Testing Your Configuration

### Complete Configuration Test

```bash
#!/bin/bash
cd /Users/chasemad/Desktop/mini-xdr/backend

echo "🔍 Checking Mini-XDR Configuration"
echo ""

# Check backend health
echo "1. Backend Health:"
curl -s http://localhost:8000/health | jq '.status' && echo "✅ Backend running" || echo "❌ Backend not running"
echo ""

# Check API key
echo "2. API Key:"
if grep -q '^API_KEY=.\+' .env; then
    echo "✅ API key configured"
else
    echo "⚠️ API key not configured"
fi
echo ""

# Check T-Pot configuration
echo "3. T-Pot Configuration:"
if grep -q '^TPOT_API_KEY=.\+' .env; then
    echo "✅ T-Pot API key configured"
else
    echo "⚠️ T-Pot API key not configured"
fi
if grep -q '^TPOT_HOST=.\+' .env; then
    TPOT_HOST=$(grep '^TPOT_HOST=' .env | cut -d '=' -f2)
    echo "✅ T-Pot host: $TPOT_HOST"
else
    echo "⚠️ T-Pot host not configured"
fi
echo ""

# Check LLM configuration
echo "4. LLM Configuration:"
if grep -q '^OPENAI_API_KEY=.\+' .env; then
    echo "✅ OpenAI configured"
else
    echo "⚠️ OpenAI not configured"
fi
if grep -q '^XAI_API_KEY=.\+' .env; then
    echo "✅ XAI configured"
else
    echo "⚠️ XAI not configured"
fi
echo ""

# Check threat intel
echo "5. Threat Intelligence:"
if grep -q '^ABUSEIPDB_API_KEY=.\+' .env; then
    echo "✅ AbuseIPDB configured"
else
    echo "⚠️ AbuseIPDB not configured"
fi
if grep -q '^VIRUSTOTAL_API_KEY=.\+' .env; then
    echo "✅ VirusTotal configured"
else
    echo "⚠️ VirusTotal not configured"
fi
echo ""

# Check workflows
echo "6. Workflow Triggers:"
WORKFLOW_COUNT=$(sqlite3 xdr.db "SELECT COUNT(*) FROM workflow_triggers;" 2>/dev/null || echo "0")
echo "✅ $WORKFLOW_COUNT workflows configured"
echo ""

echo "✅ Configuration check complete!"
```

Save this as `check-config.sh` and run:
```bash
chmod +x check-config.sh
./check-config.sh
```

## 📚 Related Documentation

- **Workflow Setup:** `TPOT_WORKFLOWS_DEPLOYMENT_SUMMARY.md`
- **T-Pot Integration:** `scripts/tpot-management/TPOT_WORKFLOWS_GUIDE.md`
- **API Documentation:** `http://localhost:8000/docs`

## 🆘 Troubleshooting

### Backend Not Reading .env

```bash
# Check .env location
ls -la backend/.env

# Check backend is using correct directory
ps aux | grep uvicorn

# Restart backend from correct directory
cd backend
pkill -f uvicorn
nohup uvicorn app.main:app --host 0.0.0.0 --port 8000 > backend.log 2>&1 &
```

### API Key Not Working

```bash
# Verify API key format
cd backend
grep '^API_KEY=' .env

# Test with explicit key
API_KEY=$(grep '^API_KEY=' .env | cut -d '=' -f2)
curl http://localhost:8000/api/triggers \
  -H "X-API-Key: $API_KEY" -v
```

### T-Pot Not Connecting

```bash
# Check T-Pot configuration
cd backend
grep '^TPOT_' .env

# Test connectivity
TPOT_HOST=$(grep '^TPOT_HOST=' .env | cut -d '=' -f2)
ping -c 3 $TPOT_HOST
```

## 💡 Tips

1. **Backup your .env:** Always keep a backup before making changes
2. **Test changes:** Restart backend and test after any configuration changes
3. **Monitor logs:** Check `backend/backend.log` for configuration issues
4. **Use strong keys:** Generate secure random keys for production
5. **Document changes:** Keep notes on what each custom key is for

---

**Your configuration is stored locally in `backend/.env` - no AWS Secrets Manager needed!** ✅


