# Mini-XDR: SSH Brute-Force Detection & Response System

A professional XDR (Extended Detection and Response) system tailored for home labs that detects SSH brute-force attacks on Cowrie honeypots and provides automated threat containment via firewall controls.

## 🎯 Features

- **Real-time Detection**: Sliding window threshold detection for SSH brute-force attacks
- **Automated Response**: Block/unblock IPs via UFW firewall on honeypot
- **AI-Powered Triage**: OpenAI/Grok integration for incident analysis and recommendations
- **SOC-Style UI**: Modern React/Next.js interface for incident management
- **MCP Integration**: Tools for LLM/agent workflows (human-led AI)
- **Scheduled Actions**: Automatic unblocking after specified time periods
- **Audit Trail**: Complete action logging with detailed results

## 🏗️ Architecture

```
┌─────────────────────┐    HTTP POST     ┌──────────────────────────┐
│ Honeypot VM         │    (JSON Events) │ Host Mac (Mini-XDR)      │
│ - Cowrie (2222)     │ ────────────────▶│ - FastAPI Backend         │
│ - UFW + nftables    │                  │ - SQLite Database         │
│ - Fluent Bit        │                  │ - Detection Engine        │
│                     │                  │ - Triage Worker           │
│ SSH Control ◀───────┼──────────────────│ - Next.js Frontend        │
│ (ufw commands)      │   Paramiko SSH   │ - MCP Tools Server        │
└─────────────────────┘                  └──────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Node.js 18+
- SSH access to honeypot VM with key-based authentication
- Cowrie honeypot with JSON logging enabled

### Backend Setup

```bash
cd mini-xdr/backend

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp env.example .env
# Edit .env with your settings

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
cd mini-xdr/frontend

# Install dependencies
npm install

# Configure environment
cp env.local .env.local
# Edit .env.local with API settings

# Start frontend
npm run dev
```

### MCP Tools Server

```bash
cd mini-xdr/backend

# Start MCP server (for LLM integration)
npm run mcp
```

## ⚙️ Configuration

### Backend Environment Variables

```bash
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_KEY=your_api_key_here

# Database
DATABASE_URL=sqlite+aiosqlite:///./xdr.db

# Detection Settings
FAIL_WINDOW_SECONDS=60
FAIL_THRESHOLD=6
AUTO_CONTAIN=false

# Honeypot Configuration
HONEYPOT_HOST=10.0.0.23
HONEYPOT_USER=xdrops
HONEYPOT_SSH_KEY=~/.ssh/xdrops_id_ed25519
HONEYPOT_SSH_PORT=22022

# LLM Configuration
LLM_PROVIDER=openai
OPENAI_API_KEY=your_openai_key
OPENAI_MODEL=gpt-4
XAI_API_KEY=your_xai_key
XAI_MODEL=grok-beta
```

### Frontend Environment Variables

```bash
NEXT_PUBLIC_API_BASE=http://10.0.0.123:8000
NEXT_PUBLIC_API_KEY=your_api_key_here
```

## 📊 Usage

### Web Interface

1. **Overview**: System health, auto-contain toggle, incident counts
2. **Incidents**: List of detected security incidents
3. **Incident Detail**: Full analysis, actions, and response controls

### API Endpoints

- `POST /ingest/cowrie` - Ingest Cowrie honeypot events
- `GET /incidents` - List all incidents
- `GET /incidents/{id}` - Get incident details
- `POST /incidents/{id}/contain` - Block source IP
- `POST /incidents/{id}/unblock` - Unblock source IP
- `POST /incidents/{id}/schedule_unblock?minutes=N` - Schedule unblock
- `GET /settings/auto_contain` - Get auto-contain setting
- `POST /settings/auto_contain` - Toggle auto-contain

### MCP Tools (for LLMs)

- `get_incidents()` - List incidents
- `get_incident(id)` - Get incident details
- `contain_incident(incident_id)` - Block IP
- `unblock_incident(incident_id)` - Unblock IP
- `schedule_unblock(incident_id, minutes)` - Schedule unblock
- `get_system_health()` - System status

## 🧪 Testing

### Manual Attack Simulation

From Kali VM (10.0.0.182):

```bash
# Generate multiple failed SSH attempts
for i in $(seq 1 8); do 
    ssh -p 2222 testuser@10.0.0.23 exit 2>/dev/null
done
```

### API Testing

```bash
# Test event ingestion
curl -X POST http://127.0.0.1:8000/ingest/cowrie \
  -H 'Content-Type: application/json' \
  -d '{"eventid":"cowrie.login.failed","src_ip":"10.0.0.182"}'

# List incidents
curl http://127.0.0.1:8000/incidents | jq

# Manual containment
curl -X POST http://127.0.0.1:8000/incidents/1/contain \
  -H 'x-api-key: your_key'
```

## 🔧 Operations

### Honeypot Setup Requirements

1. **Cowrie Configuration**: JSON logging enabled
2. **Fluent Bit**: Configured to POST to Mini-XDR ingest endpoint
3. **UFW**: Enabled with appropriate rules
4. **nftables**: DNAT redirect 22→2222 for Cowrie
5. **SSH User**: `xdrops` with key-only access and limited sudo

### Example Fluent Bit Config

```ini
[INPUT]
  Name tail
  Path /home/luxieum/cowrie/var/log/cowrie/cowrie.json
  Parser json

[OUTPUT]
  Name  http
  Host  10.0.0.123
  Port  8000
  URI   /ingest/cowrie
  Format json
```

### Example sudoers Entry

```
xdrops ALL=(ALL) NOPASSWD: /usr/sbin/ufw
```

## 🔒 Security Considerations

- Keep ingest endpoints network-restricted (VPN/allowlist)
- Use API keys for mutation operations
- SSH keys should be restricted to specific commands
- Private IP ranges are automatically protected from blocking
- All actions are logged with detailed audit trails

## 📈 Observability

- Backend logs: Request IDs, timing, errors
- Action audit trail: Every operation recorded
- Health checks: System status monitoring
- Metrics ready: Incidents/hour, MTTC, auto vs manual rates

## 🔄 Extensibility

The system is designed for modularity:

- **Detection**: Pluggable detectors via interface
- **Data Sources**: Additional ingest adapters
- **Playbooks**: Custom response actions
- **LLM Providers**: OpenAI, xAI, or custom models
- **UI Components**: Scalable React architecture

## 📄 License

MIT License - see LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📞 Support

For issues and questions:
- Create an issue in the repository
- Check the documentation in `/docs`
- Review the troubleshooting guide
