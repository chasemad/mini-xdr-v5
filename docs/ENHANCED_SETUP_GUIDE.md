# Enhanced Mini-XDR Setup Guide
**Advanced Threat Detection & Response with AI Agents, ML Ensemble, and Autonomous Response**

This guide covers the manual configuration requirements for the Enhanced Mini-XDR system that cannot be automated during deployment.

## 🚀 Quick Start

### Prerequisites
- Python 3.12+
- Node.js 18+
- Docker & Docker Compose (for containerized deployment)
- Kubernetes cluster (for production scaling)
- SSH access to honeypot systems

### 1. Basic Installation

```bash
# Clone the repository
git clone <your-repo-url> mini-xdr
cd mini-xdr

# Backend setup
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Frontend setup
cd ../frontend
npm install
npm run build

# Create environment files
cp backend/env.example backend/.env
cp frontend/env.local.example frontend/env.local
```

### 2. Environment Configuration

#### Backend Configuration (`backend/.env`)
```bash
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_KEY=your-secure-api-key-here

# Database (SQLite for development, PostgreSQL for production)
DATABASE_URL=sqlite+aiosqlite:///./xdr.db
# DATABASE_URL=postgresql+asyncpg://user:password@localhost/mini_xdr

# Detection Configuration
FAIL_WINDOW_SECONDS=60
FAIL_THRESHOLD=6
AUTO_CONTAIN=true
ALLOW_PRIVATE_IP_BLOCKING=false

# Honeypot Configuration
HONEYPOT_HOST=10.0.0.23
HONEYPOT_USER=xdrops
HONEYPOT_SSH_KEY=~/.ssh/xdrops_id_ed25519
HONEYPOT_SSH_PORT=22022

# AI/LLM Configuration
LLM_PROVIDER=openai  # or xai
OPENAI_API_KEY=sk-your-openai-key-here
OPENAI_MODEL=gpt-4
XAI_API_KEY=your-xai-key-here
XAI_MODEL=grok-beta

# Threat Intelligence APIs (Optional)
ABUSEIPDB_API_KEY=your-abuseipdb-key-here
VIRUSTOTAL_API_KEY=your-virustotal-key-here
```

#### Frontend Configuration (`frontend/env.local`)
```bash
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_WS_URL=ws://localhost:8000/ws
```

## 🔧 Manual Configuration Requirements

### 1. SSH Key Setup for Honeypot Access

The system requires SSH key authentication to the honeypot for containment actions:

```bash
# Generate SSH key for honeypot access
ssh-keygen -t ed25519 -f ~/.ssh/xdrops_id_ed25519 -C "mini-xdr-ops"

# Copy public key to honeypot
ssh-copy-id -i ~/.ssh/xdrops_id_ed25519.pub xdrops@YOUR_HONEYPOT_IP

# Test connection
ssh -i ~/.ssh/xdrops_id_ed25519 xdrops@YOUR_HONEYPOT_IP "sudo ufw status"
```

### 2. Honeypot Configuration

#### Cowrie Honeypot Setup
```bash
# Install Cowrie (on honeypot system)
git clone https://github.com/cowrie/cowrie.git
cd cowrie
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Configure Cowrie for JSON logging
cp etc/cowrie.cfg.dist etc/cowrie.cfg

# Edit etc/cowrie.cfg
[output_jsonlog]
enabled = true
logfile = var/log/cowrie/cowrie.json
epoch_timestamp = false
```

#### Fluent Bit Log Forwarding (Optional)
```bash
# Install Fluent Bit on honeypot
curl https://raw.githubusercontent.com/fluent/fluent-bit/master/install.sh | sh

# Configure forwarding
sudo tee /etc/fluent-bit/fluent-bit.conf > /dev/null <<EOF
[INPUT]
    Name tail
    Path /opt/cowrie/var/log/cowrie/cowrie.json
    Parser json
    Tag cowrie

[OUTPUT]
    Name http
    Match *
    Host YOUR_MINI_XDR_HOST
    Port 8000
    URI /ingest/multi
    Format json
    Header Authorization Bearer YOUR_API_KEY
    Header Content-Type application/json
EOF
```

### 3. AI Agent Configuration

#### OpenAI API Setup
1. Create account at https://platform.openai.com/
2. Generate API key
3. Add to environment variables
4. Recommended models:
   - `gpt-4` for production
   - `gpt-3.5-turbo` for development

#### xAI (Grok) Setup (Alternative)
1. Create account at https://x.ai/
2. Generate API key
3. Set `LLM_PROVIDER=xai` in environment

### 4. Threat Intelligence APIs

#### AbuseIPDB (Free tier: 1000 queries/day)
1. Register at https://www.abuseipdb.com/
2. Generate API key
3. Add `ABUSEIPDB_API_KEY` to environment

#### VirusTotal (Free tier: 500 queries/day)
1. Register at https://www.virustotal.com/
2. Generate API key
3. Add `VIRUSTOTAL_API_KEY` to environment

### 5. Multi-Source Log Ingestion

#### Configure Log Sources
Create log source configurations:

```python
# backend/log_sources_config.py
LOG_SOURCES = [
    {
        "source_type": "cowrie",
        "hostname": "honeypot-01",
        "endpoint_url": "http://honeypot-01:8000/logs",
        "validation_key": "shared-secret-key"
    },
    {
        "source_type": "suricata",
        "hostname": "ids-01", 
        "endpoint_url": "http://ids-01:8000/logs",
        "validation_key": "shared-secret-key"
    }
]
```

#### Deploy Ingestion Agents
```bash
# On each log source system
python backend/app/agents/ingestion_agent.py \
    --config agent-config.json \
    --verbose

# agent-config.json example:
{
    "backend_url": "https://your-mini-xdr.com",
    "api_key": "your-agent-api-key",
    "source_type": "cowrie",
    "hostname": "honeypot-01",
    "log_paths": {
        "cowrie": "/var/log/cowrie/cowrie.json"
    },
    "batch_size": 50,
    "flush_interval": 30
}
```

### 6. ML Model Configuration

#### Initial Model Training
```bash
# After collecting initial data (recommended: 1000+ events)
curl -X POST http://localhost:8000/api/ml/retrain \
    -H "Content-Type: application/json" \
    -d '{"model_type": "ensemble"}'
```

#### Model Tuning Parameters
Edit configuration in UI or via API:
- Isolation Forest contamination: 0.1 (10% anomalies expected)
- LSTM sequence length: 10 events
- Detection thresholds: low=0.2, medium=0.5, high=0.8, critical=0.95

### 7. Containment Policy Configuration

#### Create Custom Policies
```yaml
# policies/custom_policies.yaml
policies:
  - name: "high_volume_ssh_attack"
    description: "Detect and contain high-volume SSH attacks"
    priority: 5
    conditions:
      event_count:
        min: 100
      threat_category: ["brute_force"]
      time_window: 300  # 5 minutes
    actions:
      block_ip:
        duration: 7200  # 2 hours
        immediate: true
      notify_analyst:
        urgency: "critical"
    agent_override: true
```

#### Load Policies
```bash
# Via API
curl -X POST http://localhost:8000/api/policies/load \
    -H "Content-Type: application/json" \
    -d @policies/custom_policies.yaml
```

## 🐳 Docker Deployment

### Development (Docker Compose)
```bash
# Create docker-compose.override.yml for local settings
version: '3.8'
services:
  backend:
    environment:
      - OPENAI_API_KEY=your-key-here
      - HONEYPOT_HOST=host.docker.internal
    volumes:
      - ~/.ssh/xdrops_id_ed25519:/app/.ssh/id_ed25519:ro

# Deploy
docker-compose up -d
```

### Production (Kubernetes)
```bash
# Build and deploy
./ops/deploy-k8s.sh --build --push --ingress

# You'll be prompted for:
# - OpenAI API Key
# - xAI API Key (optional)
# - Agent API Key
# - SSH key path
```

## 🔗 Network Access Requirements

### Firewall Rules
```bash
# On Mini-XDR host
sudo ufw allow 8000/tcp  # Backend API
sudo ufw allow 3000/tcp  # Frontend (if not using reverse proxy)
sudo ufw allow 22/tcp    # SSH for management

# On honeypot hosts
sudo ufw allow from MINI_XDR_IP to any port 22022  # SSH for containment
```

### Network Connectivity
- Mini-XDR → Honeypots: SSH (port 22022)
- Honeypots → Mini-XDR: HTTP/HTTPS (port 8000)
- Users → Mini-XDR: HTTPS (port 443 via ingress)
- Mini-XDR → Internet: HTTPS (443) for AI APIs and threat intel

## 🎯 Testing the Setup

### 1. Test Basic Functionality
```bash
# Health check
curl http://localhost:8000/health

# Test ingestion
curl -X POST http://localhost:8000/ingest/cowrie \
    -H "Content-Type: application/json" \
    -d '[{"src_ip": "1.2.3.4", "eventid": "cowrie.login.failed", "message": "test"}]'

# Test AI agent
curl -X POST http://localhost:8000/api/agents/orchestrate \
    -H "Content-Type: application/json" \
    -d '{"agent_type": "containment", "query": "Evaluate IP 1.2.3.4"}'
```

### 2. Test Containment
```bash
# Test SSH connectivity to honeypot
curl http://localhost:8000/test/ssh

# Test manual containment
curl -X POST http://localhost:8000/incidents/1/contain \
    -H "x-api-key: your-api-key"
```

### 3. Generate Test Data
```bash
# Use the provided test script
./ops/test-attack.sh

# Or create synthetic events
python scripts/generate_test_events.py --count 100 --attack-type brute_force
```

## 🛠 Troubleshooting

### Common Issues

#### "AI agents not initialized"
- Check LLM provider configuration
- Verify API keys are valid
- Check network connectivity to AI services

#### "SSH connection failed"
- Verify SSH key permissions: `chmod 600 ~/.ssh/xdrops_id_ed25519`
- Test manual SSH: `ssh -i ~/.ssh/xdrops_id_ed25519 user@honeypot`
- Check honeypot firewall rules

#### "ML models not training"
- Ensure sufficient training data (100+ events)
- Check for missing dependencies: `pip install torch xgboost`
- Review logs: `docker logs mini-xdr-backend`

#### "Ingestion agent connection failed"
- Verify API key configuration
- Check network connectivity
- Confirm backend endpoint is accessible

### Log Files
- Backend: `logs/mini-xdr.log`
- Agent: `logs/agent.log`
- Kubernetes: `kubectl logs -f deployment/mini-xdr-backend -n mini-xdr`

### Performance Monitoring
- Backend metrics: `http://localhost:8000/metrics`
- ML model status: `http://localhost:8000/api/ml/status`
- Agent status: Check UI at `/agents`

## 🔐 Security Considerations

### API Security
- Use strong API keys (32+ characters)
- Enable HTTPS in production
- Implement rate limiting
- Regular key rotation

### Network Security
- Use VPN/private networks when possible
- Implement network segmentation
- Monitor for unauthorized access
- Regular security updates

### Data Protection
- Encrypt sensitive data at rest
- Secure backup procedures
- Access logging and monitoring
- GDPR/compliance considerations

## 📊 Monitoring and Maintenance

### Regular Tasks
- Weekly: Review incidents and false positives
- Monthly: Retrain ML models with new data
- Quarterly: Update containment policies
- Annually: Security audit and penetration testing

### Monitoring Dashboards
- System health: `/health`
- ML analytics: `/analytics`
- Agent status: `/agents`
- Incident overview: `/incidents`

### Backup Procedures
```bash
# Database backup
sqlite3 xdr.db ".backup backup.db"

# Model backup
tar -czf models-backup.tar.gz models/

# Configuration backup
tar -czf config-backup.tar.gz policies/ .env
```

## 🚀 Advanced Features

### Distributed Deployment
- Multi-region clusters
- Edge computing nodes
- CDN integration
- Global load balancing

### Enterprise Integration
- SIEM integration (Splunk, QRadar)
- SOAR platforms (Phantom, Demisto)
- Ticketing systems (Jira, ServiceNow)
- Identity providers (LDAP, SAML)

### Custom Development
- Plugin architecture
- Custom detection rules
- Third-party integrations
- API extensions

---

For support and questions, refer to the main README.md or create an issue in the repository.
