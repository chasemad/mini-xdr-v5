# 🛡️ Complete Mini-XDR Setup Guide

## Current Status
✅ Backend API running on port 8000
✅ Frontend dev server running  
✅ Virtual environments configured
✅ Database and ML models present
⏳ Python dependencies installing (wait for completion)

## **Phase 1: API Keys Setup**

### 1.1 OpenAI API Key (Required for AI Agents)
1. Go to https://platform.openai.com/api-keys
2. Click "Create new secret key"
3. Name it "Mini-XDR-System"
4. Copy the key (starts with sk-...)

### 1.2 AbuseIPDB API Key (Required for Threat Intelligence)
1. Go to https://www.abuseipdb.com/api
2. Register for free account
3. Go to API section
4. Copy your API key

### 1.3 VirusTotal API Key (Optional but Recommended)
1. Go to https://www.virustotal.com/gui/join-us
2. Register for free account
3. Go to your profile -> API Key
4. Copy the API key

## **Phase 2: Environment Configuration**

### 2.1 Backend Environment (.env)
```bash
# Navigate to backend directory
cd /Users/chasemad/Desktop/mini-xdr/backend

# Create .env file from template
cp env.example .env
```

Edit the .env file with your API keys:
```bash
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_KEY=your-secure-api-key-here

# Database
DATABASE_URL=sqlite+aiosqlite:///./xdr.db

# Detection Configuration
FAIL_WINDOW_SECONDS=60
FAIL_THRESHOLD=6
AUTO_CONTAIN=true

# Honeypot Configuration (Update with your actual honeypot details)
HONEYPOT_HOST=YOUR_HONEYPOT_IP
HONEYPOT_USER=xdrops
HONEYPOT_SSH_KEY=~/.ssh/xdrops_id_ed25519
HONEYPOT_SSH_PORT=22022

# LLM Configuration
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-your-openai-key-here
OPENAI_MODEL=gpt-4

# Threat Intelligence APIs
ABUSEIPDB_API_KEY=your-abuseipdb-key-here
VIRUSTOTAL_API_KEY=your-virustotal-key-here

# Enhanced ML Configuration
ML_MODELS_PATH=./models
ML_CONFIDENCE_THRESHOLD=0.8
AUTO_RETRAIN_ENABLED=true

# Agent Configuration
AGENT_API_KEY=secure-agent-key-here
AGENT_CONFIDENCE_THRESHOLD=0.7
MAX_AUTO_ACTIONS_PER_HOUR=10
```

### 2.2 Frontend Environment (.env.local)
```bash
# Navigate to frontend directory
cd /Users/chasemad/Desktop/mini-xdr/frontend

# Create .env.local file
cp env.local.example .env.local
```

Edit the .env.local file:
```bash
# API Configuration
NEXT_PUBLIC_API_BASE=http://localhost:8000
NEXT_PUBLIC_API_KEY=same-as-backend-api-key
```

## **Phase 3: Service Startup**

### 3.1 Wait for Dependencies Installation
Wait for the pip install process to complete (currently running at 99% CPU)

### 3.2 Restart Services with New Configuration
```bash
# Kill existing processes and restart
cd /Users/chasemad/Desktop/mini-xdr
./scripts/stop-all.sh
./scripts/start-all.sh
```

## **Phase 4: Verification Tests**

### 4.1 Backend Health Check
```bash
curl http://localhost:8000/health
# Expected: {"status": "healthy"}
```

### 4.2 AI Agent Test
```bash
curl -X POST http://localhost:8000/api/agents/orchestrate \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-api-key" \
  -d '{"query": "System status check"}'
```

### 4.3 Frontend Access
Open browser to: http://localhost:3000

### 4.4 Database Check
```bash
sqlite3 /Users/chasemad/Desktop/mini-xdr/xdr.db ".tables"
# Expected: List of tables including incidents, evidence, iocs, etc.
```

## **Phase 5: Honeypot Integration**

### 5.1 Verify Honeypot Connection
Update your honeypot VM's fluent-bit configuration to send logs to:
- URL: `http://YOUR_MINI_XDR_IP:8000/api/ingest/cowrie`
- Headers: `Authorization: Bearer your-api-key`

### 5.2 Test Log Ingestion
```bash
# Send test event to verify ingestion
curl -X POST http://localhost:8000/api/ingest/cowrie \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-api-key" \
  -d @test_events.json
```

## **Phase 6: End-to-End Testing**

### 6.1 Simulate Attack
```bash
# Run the test attack script
./ops/test-attack.sh
```

### 6.2 Verify Detection and Response
1. Check incidents page: http://localhost:3000/incidents
2. Verify AI agent responses in: http://localhost:3000/agents
3. Check containment actions were taken

## **Phase 7: Production Readiness**

### 7.1 SSL/TLS Setup (Production)
- Configure reverse proxy (nginx/apache)
- Obtain SSL certificates
- Update API_HOST to use HTTPS

### 7.2 Database Migration (Production)
- Consider PostgreSQL for production
- Update DATABASE_URL accordingly

### 7.3 Monitoring Setup
- Configure log aggregation
- Set up alerting for system health
- Monitor AI agent performance metrics

## **Troubleshooting**

### Common Issues:
1. **API Key Errors**: Verify all keys are correctly set in .env files
2. **Port Conflicts**: Ensure ports 8000 and 3000 are available
3. **Permission Issues**: Check file permissions for SSH keys and database
4. **Memory Issues**: Ensure adequate RAM for ML models (4GB+ recommended)

### Log Locations:
- Backend: `/Users/chasemad/Desktop/mini-xdr/backend/backend.log`
- Frontend: `/Users/chasemad/Desktop/mini-xdr/frontend/frontend.log`
- Database: SQLite browser or command line access

### Support Commands:
```bash
# Check system status
./scripts/system-status.sh

# View recent logs
tail -f backend/backend.log

# Test all components
./scripts/test-blocking-actions.sh
```
