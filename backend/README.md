# Mini-XDR Backend

FastAPI-based backend for SSH brute-force detection and automated response.

## Features

- **Event Ingestion**: Cowrie JSON event processing
- **Detection Engine**: Sliding window threshold detection
- **Response Agent**: SSH/UFW remote command execution
- **Triage Worker**: AI-powered incident analysis
- **Background Tasks**: Scheduled unblock processing
- **API Endpoints**: RESTful incident management
- **MCP Tools**: LLM integration tools

## Setup

### Install Dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Configure Environment

```bash
cp env.example .env
```

Edit `.env` with your configuration:

```bash
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_KEY=optional_api_key

# Database
DATABASE_URL=sqlite+aiosqlite:///./xdr.db

# Detection
FAIL_WINDOW_SECONDS=60
FAIL_THRESHOLD=6
AUTO_CONTAIN=false

# Honeypot SSH Access
HONEYPOT_HOST=10.0.0.23
HONEYPOT_USER=xdrops
HONEYPOT_SSH_KEY=~/.ssh/xdrops_id_ed25519
HONEYPOT_SSH_PORT=22022

# LLM Integration
LLM_PROVIDER=openai
OPENAI_API_KEY=your_key
OPENAI_MODEL=gpt-4
```

### Initialize Database

```bash
python -c "
import asyncio
from app.db import init_db
asyncio.run(init_db())
"
```

### Run Server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## API Documentation

### Event Ingestion

```bash
POST /ingest/cowrie
Content-Type: application/json

{
  "eventid": "cowrie.login.failed",
  "src_ip": "10.0.0.182",
  "message": "login attempt",
  "timestamp": "2025-01-24T10:00:00Z"
}
```

### Incident Management

```bash
# List incidents
GET /incidents

# Get incident details
GET /incidents/{id}

# Contain incident (block IP)
POST /incidents/{id}/contain
Headers: x-api-key: your_key

# Unblock incident
POST /incidents/{id}/unblock
Headers: x-api-key: your_key

# Schedule unblock
POST /incidents/{id}/schedule_unblock?minutes=15
Headers: x-api-key: your_key
```

### Settings

```bash
# Get auto-contain setting
GET /settings/auto_contain

# Set auto-contain setting
POST /settings/auto_contain
Headers: x-api-key: your_key
Content-Type: application/json

true
```

## MCP Tools Server

The MCP (Model Context Protocol) server provides tools for LLM integration:

```bash
# Install TypeScript dependencies
npm install

# Run MCP server
npm run mcp
```

### Available Tools

- `get_incidents()` - List all incidents
- `get_incident(id)` - Get incident details
- `contain_incident(incident_id)` - Block source IP
- `unblock_incident(incident_id)` - Unblock source IP
- `schedule_unblock(incident_id, minutes)` - Schedule unblock
- `get_auto_contain_setting()` - Get current setting
- `set_auto_contain_setting(enabled)` - Toggle auto-contain
- `get_system_health()` - System status

## Architecture

### Detection Engine (`app/detect.py`)

- Sliding window threshold detection
- Configurable failure count and time window
- Automatic incident creation
- Support for multiple detector types

### Response Agent (`app/responder.py`)

- Paramiko SSH client for remote commands
- UFW firewall management
- IP validation and private IP protection
- Detailed command execution logging

### Triage Worker (`app/triager.py`)

- OpenAI/xAI integration for incident analysis
- Structured JSON output with severity and recommendations
- Fallback analysis when LLM unavailable
- Configurable model selection

### Background Scheduler

- APScheduler for recurring tasks
- Processes scheduled unblock actions
- Automatic status updates
- Error handling and retry logic

## Database Schema

### Events Table

```sql
CREATE TABLE events (
    id INTEGER PRIMARY KEY,
    ts DATETIME DEFAULT CURRENT_TIMESTAMP,
    src_ip VARCHAR(64) INDEXED,
    dst_ip VARCHAR(64),
    dst_port INTEGER,
    eventid VARCHAR(128) INDEXED,
    message TEXT,
    raw JSON
);
```

### Incidents Table

```sql
CREATE TABLE incidents (
    id INTEGER PRIMARY KEY,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    src_ip VARCHAR(64) INDEXED,
    reason VARCHAR(256),
    status VARCHAR(32) DEFAULT 'open',
    auto_contained BOOLEAN DEFAULT FALSE,
    triage_note JSON
);
```

### Actions Table

```sql
CREATE TABLE actions (
    id INTEGER PRIMARY KEY,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    incident_id INTEGER INDEXED,
    action VARCHAR(32),
    result VARCHAR(32),
    detail TEXT,
    params JSON,
    due_at DATETIME
);
```

## Testing

### Unit Tests

```bash
# Run detection tests
python -m pytest tests/test_detect.py

# Run responder tests
python -m pytest tests/test_responder.py

# Run API tests
python -m pytest tests/test_api.py
```

### Manual Testing

```bash
# Test event ingestion
curl -X POST http://localhost:8000/ingest/cowrie \
  -H 'Content-Type: application/json' \
  -d '{"eventid":"cowrie.login.failed","src_ip":"192.168.1.100"}'

# Verify detection triggered
curl http://localhost:8000/incidents

# Test containment
curl -X POST http://localhost:8000/incidents/1/contain \
  -H 'x-api-key: your_key'
```

## Monitoring

### Health Check

```bash
GET /health

{
  "status": "healthy",
  "timestamp": "2025-01-24T10:00:00Z",
  "auto_contain": false
}
```

### Logs

The application logs important events:

- Event ingestion statistics
- Detection triggers and thresholds
- SSH command execution results
- Triage analysis results
- Background task processing

### Metrics

Key metrics tracked:

- Events processed per minute
- Incidents created per hour
- Mean time to containment (MTTC)
- Auto vs manual containment rates
- SSH command success rates

## Troubleshooting

### Common Issues

1. **SSH Connection Failed**
   - Verify honeypot host/port
   - Check SSH key permissions
   - Confirm user has sudo access

2. **Detection Not Triggering**
   - Check threshold configuration
   - Verify event format
   - Review source IP extraction

3. **Triage Analysis Failed**
   - Verify API keys
   - Check model availability
   - Review provider configuration

4. **Database Errors**
   - Check SQLite file permissions
   - Verify database initialization
   - Review migration status
