# Comprehensive Azure Honeypot Testing Guide

## Overview

This testing suite validates that your Mini-XDR system correctly:
1. **Detects attacks** from the Azure T-Pot honeypot
2. **Classifies threats** using the ML model with accurate confidence scoring
3. **Executes agent responses** (Containment, Forensics, Attribution, Deception, Threat Hunting)
4. **Applies response actions** through workflows and tools
5. **Integrates all components** end-to-end

---

## Test Suite Components

### 1. `comprehensive_azure_honeypot_test.py`
**Purpose**: Automated testing of the entire system

**What it tests**:
- ✅ System health and ML model status
- ✅ Event ingestion and processing
- ✅ ML predictions with confidence scoring
- ✅ All 5+ agent responses
- ✅ Workflow creation and execution
- ✅ Response action tools
- ✅ Integration points
- ✅ Advanced attack scenarios

**Features**:
- Color-coded output
- Detailed pass/fail reporting
- JSON result export
- Comprehensive coverage of all attack types

### 2. `live_honeypot_attack_suite.sh`
**Purpose**: Generate real attack traffic against your Azure honeypot

**Attack types**:
1. SSH Brute Force (20 attempts)
2. Port Scanning (comprehensive)
3. Web Reconnaissance (12+ paths)
4. Telnet Brute Force
5. FTP Enumeration
6. SQL Injection attempts
7. Directory Traversal
8. DNS Enumeration
9. SMTP Probing
10. Multi-Stage APT simulation

---

## Prerequisites

### System Requirements
```bash
# Python dependencies
pip install aiohttp

# System tools (for live attacks)
sudo apt-get install -y nmap netcat telnet ftp curl sshpass

# Or on macOS:
brew install nmap netcat telnet curl
```

### Configuration
Ensure your `backend/.env` file has:
```bash
API_KEY=your-api-key
TPOT_HOST=your-azure-honeypot-ip
TPOT_SSH_PORT=64295
```

### Services Running
1. **Backend API**: `cd backend && uvicorn app.main:app --reload`
2. **Frontend** (optional): `cd frontend && npm run dev`
3. **Azure T-Pot**: Should be running and accessible

---

## Running the Tests

### Option 1: Full Automated Test Suite

```bash
# Navigate to tests directory
cd ./tests

# Run comprehensive test suite
python3 comprehensive_azure_honeypot_test.py
```

**Expected output**:
```
╔════════════════════════════════════════════════════════════════════════════╗
║                                                                            ║
║           COMPREHENSIVE AZURE HONEYPOT TESTING SUITE                      ║
║                                                                            ║
║  Testing: ML Models, Agents, Tools, Workflows, and Integrations           ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝

✅ Configuration loaded - T-Pot: 74.235.242.205

SECTION 1: SYSTEM HEALTH & CONFIGURATION
[TEST] 1.1 Backend Health Check
✅ Backend Health: PASSED Database: connected
[TEST] 1.2 ML Model Status
✅ ML Model Status: PASSED 3 models trained
...
```

**Test duration**: ~5-10 minutes

**Results**: Saved to `test_results_YYYYMMDD_HHMMSS.json`

### Option 2: Live Honeypot Attacks

```bash
# Navigate to tests directory
cd ./tests

# Run live attack suite
bash live_honeypot_attack_suite.sh
```

**What happens**:
1. Generates real attack traffic against your honeypot
2. Creates 10 different attack scenarios
3. Logs all activity for verification
4. Provides summary of attacks executed

**Wait 2-3 minutes** after attacks complete for:
- Events to be ingested
- ML models to classify threats
- Incidents to be created
- Agents to respond

Then run the automated test suite to verify detection.

### Option 3: Combined Testing (Recommended)

```bash
cd ./tests

# Step 1: Generate real attacks
bash live_honeypot_attack_suite.sh

# Step 2: Wait for processing
echo "Waiting 3 minutes for event processing..."
sleep 180

# Step 3: Verify detection and response
python3 comprehensive_azure_honeypot_test.py
```

---

## Test Sections Explained

### Section 1: System Health & Configuration
Tests basic connectivity and system status:
- Backend API health
- ML model loading and training
- Agent credential configuration
- SageMaker endpoint status

**Expected results**: All services should be healthy and responding

### Section 2: Honeypot Attack Simulation
Simulates various attack types:
- SSH Brute Force (203.0.113.50)
- Port Scanning (203.0.113.51)
- Malware Execution (203.0.113.52)
- APT File Download (203.0.113.53)

**Expected results**: All events should be ingested successfully

### Section 3: ML Model Predictions & Confidence Scoring
Validates ML model performance:
- Incident creation from attacks
- Confidence scores (should be >50% for real threats)
- Threat classification accuracy
- Anomaly score calculation

**Expected results**: 
- Multiple incidents created
- High confidence scores for clear attacks
- Accurate threat type classification

### Section 4: Agent Responses
Tests all agent responses:
1. **Containment Agent**: Block IP action
2. **Forensics Agent**: Evidence collection
3. **Attribution Agent**: Threat intel lookup
4. **Deception Agent**: Attacker profiling
5. **Threat Hunting Agent**: Pattern hunting

**Expected results**: All agents should respond and create actions

### Section 5: Workflow Execution & Tools
Tests response orchestration:
- NLP workflow creation
- Workflow listing and status
- Available response actions
- Investigation creation

**Expected results**: Workflows created and tools available

### Section 6: Integration Tests
Validates system integrations:
- Chat interface
- Alert system
- Dashboard metrics
- Event statistics

**Expected results**: All integrations responding

### Section 7: Advanced Scenarios
Tests complex attack patterns:
- Multi-stage attacks (recon → exploit → post-exploit)
- Coordinated multi-agent response
- Real-time honeypot connectivity

**Expected results**: Complex scenarios handled correctly

---

## Understanding Test Results

### Success Indicators ✅
```
✅ Backend Health: PASSED Database: connected
✅ ML Model Status: PASSED 3 models trained
✅ SSH Brute Force: PASSED 10 events ingested
✅ Incident Detection: PASSED 4 incidents detected
✅ ML Confidence: PASSED 3 incidents with >50% confidence
```

### Warning Indicators ⚠️
```
⚠️  Agent Credentials: WARNING Only 5 agents configured (expected 6)
⚠️  SageMaker Endpoint: WARNING Status: Updating
```
Warnings indicate non-critical issues that should be investigated.

### Failure Indicators ❌
```
❌ Backend Health: FAILED Connection refused
❌ ML Model Status: FAILED No models trained
```
Failures indicate critical issues requiring immediate attention.

---

## Verifying Specific Components

### Check ML Model Confidence Scores

```bash
# Get incidents with ML scores
curl -s http://localhost:8000/incidents | jq '.[] | {id, threat_type, ml_confidence, anomaly_score}'
```

**Expected output**:
```json
{
  "id": 1,
  "threat_type": "Brute Force Attack",
  "ml_confidence": 0.87,
  "anomaly_score": 0.92
}
```

### Check Agent Responses

```bash
# Get agent actions for an incident
curl -s http://localhost:8000/incidents/1 | jq '.actions'
```

**Expected output**: Multiple actions from different agents

### Check Workflows

```bash
# List all workflows
curl -s -H "x-api-key: $API_KEY" http://localhost:8000/api/workflows | jq '.workflows[] | {id, status, actions}'
```

### Check Event Processing

```bash
# Get recent events
curl -s http://localhost:8000/events | jq '.[] | {src_ip, eventid, timestamp}'
```

---

## Troubleshooting

### No Incidents Created

**Possible causes**:
1. Events not being ingested
2. ML model not trained
3. Detection thresholds too high

**Debug steps**:
```bash
# Check if events are being stored
curl -s http://localhost:8000/events | jq 'length'

# Check ML model status
curl -s -H "x-api-key: $API_KEY" http://localhost:8000/api/ml/status

# Check backend logs
tail -f backend/backend.log | grep "incident"
```

### Low ML Confidence Scores

**Possible causes**:
1. Insufficient training data
2. Model not properly loaded
3. Feature extraction issues

**Debug steps**:
```bash
# Check SageMaker endpoint
curl -s -H "x-api-key: $API_KEY" http://localhost:8000/api/ml/sagemaker/status

# Verify model predictions
tail -f backend/backend.log | grep "prediction"
```

### Agents Not Responding

**Possible causes**:
1. Agent credentials not configured
2. Agent API keys missing
3. Agent service issues

**Debug steps**:
```bash
# Check agent status
curl -s -H "x-api-key: $API_KEY" http://localhost:8000/api/agents/status

# Check agent credentials in database
cd backend
python3 -c "
from app.db import AsyncSessionLocal, init_db
from app.models import AgentCredential
from sqlalchemy import select
import asyncio

async def check():
    await init_db()
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(AgentCredential))
        creds = result.scalars().all()
        for cred in creds:
            print(f'{cred.role}: {cred.device_id}')

asyncio.run(check())
"
```

### Workflows Not Created

**Possible causes**:
1. NLP parser issues
2. Response engine not initialized
3. Invalid action types

**Debug steps**:
```bash
# Check available actions
curl -s -H "x-api-key: $API_KEY" http://localhost:8000/api/response-actions/available | jq '.actions | length'

# Test NLP parsing
curl -s -H "x-api-key: $API_KEY" -X POST http://localhost:8000/api/nlp/parse \
  -H "Content-Type: application/json" \
  -d '{"query": "Block IP 1.2.3.4"}' | jq
```

---

## Success Criteria

### ✅ System is Working Correctly If:

1. **ML Models**: 
   - At least 2 models trained
   - Confidence scores >50% for clear attacks
   - Threat types correctly classified

2. **Agents**:
   - All 6 agents configured
   - Agent actions created for incidents
   - Appropriate agent selected for each scenario

3. **Workflows**:
   - NLP commands create workflows
   - Workflows execute successfully
   - Multiple response actions available

4. **Integration**:
   - Events ingested from honeypot
   - Incidents created automatically
   - Dashboard shows accurate metrics
   - Chat interface responds

5. **End-to-End**:
   - Attack → Detection → Classification → Response
   - Multi-stage attacks detected
   - Coordinated responses executed

---

## Advanced Testing Scenarios

### Test 1: Verify ML Model Accuracy

```python
# Run this after generating attacks
import asyncio
from backend.app.sagemaker_client import sagemaker_client

async def test_ml():
    events = [{"src_ip": "1.2.3.4", "dst_port": 22, "eventid": "cowrie.login.failed"}]
    result = await sagemaker_client.detect_threats(events)
    print(f"Prediction: {result[0]['threat_type']}")
    print(f"Confidence: {result[0]['confidence']:.2%}")

asyncio.run(test_ml())
```

### Test 2: Verify Agent Coordination

```bash
# Create incident requiring multiple agents
curl -s -H "x-api-key: $API_KEY" -X POST http://localhost:8000/api/nlp/parse-and-execute \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Block the attacker, collect forensics, hunt for similar attacks, and profile the behavior",
    "incident_id": 1
  }' | jq '.workflow_id'

# Check if workflow created with multiple agent actions
curl -s -H "x-api-key: $API_KEY" http://localhost:8000/api/workflows | jq '.workflows[0].steps'
```

### Test 3: Verify Tool Execution

```bash
# Test containment tool
curl -s -H "x-api-key: $API_KEY" -X POST http://localhost:8000/incidents/1/actions/containment-block-ip

# Check if action executed
curl -s http://localhost:8000/incidents/1 | jq '.actions[] | select(.action_type=="block_ip")'
```

---

## Continuous Monitoring

### Real-Time Monitoring During Tests

```bash
# Terminal 1: Watch backend logs
tail -f backend/backend.log | grep -E "(incident|ML|agent|workflow)"

# Terminal 2: Watch events
watch -n 5 'curl -s http://localhost:8000/events | jq "length"'

# Terminal 3: Watch incidents
watch -n 5 'curl -s http://localhost:8000/incidents | jq "length"'

# Terminal 4: Run tests
python3 comprehensive_azure_honeypot_test.py
```

---

## Reporting Issues

If tests fail, provide:
1. Test results JSON file
2. Backend logs (`backend/backend.log`)
3. System status:
   ```bash
   curl -s http://localhost:8000/health | jq
   curl -s -H "x-api-key: $API_KEY" http://localhost:8000/api/ml/status | jq
   curl -s -H "x-api-key: $API_KEY" http://localhost:8000/api/agents/status | jq
   ```

---

## Next Steps After Testing

Once all tests pass:

1. **Production Deployment**:
   - Configure production API keys
   - Set up monitoring alerts
   - Enable auto-response workflows

2. **Tuning**:
   - Adjust ML confidence thresholds
   - Customize agent responses
   - Create custom workflows

3. **Monitoring**:
   - Set up continuous honeypot monitoring
   - Configure alert notifications
   - Review agent performance metrics

---

## Quick Reference

### Run Everything
```bash
# Full test cycle
cd ./tests
bash live_honeypot_attack_suite.sh && sleep 180 && python3 comprehensive_azure_honeypot_test.py
```

### Check Specific Component
```bash
# ML Model
curl -s -H "x-api-key: $API_KEY" http://localhost:8000/api/ml/status | jq

# Agents
curl -s -H "x-api-key: $API_KEY" http://localhost:8000/api/agents/status | jq

# Incidents
curl -s http://localhost:8000/incidents | jq 'length'

# Workflows
curl -s -H "x-api-key: $API_KEY" http://localhost:8000/api/workflows | jq '.workflows | length'
```

---

**Happy Testing! 🎉**

