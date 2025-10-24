# 🔐 Security & Testing Guide

## Overview

This guide covers:
1. ✅ **Security Audit** - Verify Azure TPOT isolation and home lab safety
2. 🐛 **Model Debugging** - Fix the 57% confidence issue
3. 🎯 **Attack Testing** - Test different attacks from different IPs
4. 🤖 **Agent Validation** - Verify AI agents respond correctly

---

## 🚨 SECURITY STATUS

### Current Configuration
- **Azure TPOT VM IP**: `74.235.242.205`
- **Authorized IP**: `24.11.0.176/32` (YOUR IP ONLY)
- **Home Lab**: ✅ **ISOLATED** (not exposed)
- **Backend**: Should run on `127.0.0.1` only

### What's Secured
1. ✅ **SSH Access (ports 22, 64295)**: Restricted to your IP
2. ✅ **Web Dashboard (port 64297)**: Restricted to your IP  
3. ✅ **Honeypot Ports**: Currently restricted to your IP for testing
4. ✅ **No Home Lab Exposure**: TPOT is on Azure, not your local network

### What's Safe to Do
- ✅ **Open TPOT to internet** - Your home lab is isolated
- ✅ **Test with real attacks** - Everything is in the cloud
- ✅ **Run comprehensive tests** - No security risk

---

## 🏃 QUICK START

### 1. Run All Tests at Once
```bash
./scripts/run-comprehensive-tests.sh
```

This will run:
1. Security audit
2. Model debugging
3. Attack scenario testing
4. Agent response validation

### 2. Run Individual Tests

#### Security Audit Only
```bash
./scripts/security-audit-comprehensive.sh
```

Checks:
- ✅ Local network exposure
- ✅ Azure NSG rules
- ✅ Environment configuration
- ✅ SSH key security
- ✅ Database security
- ✅ TPOT connectivity

#### Model Debugging Only
```bash
cd /Users/chasemad/Desktop/mini-xdr
python3 tests/test_model_confidence_debug.py
```

Diagnoses:
- 🔍 Model file integrity
- 🔍 Architecture compatibility
- 🔍 Feature extraction
- 🔍 Scaling issues
- 🔍 Weight initialization
- 🔍 Inference pipeline

#### Attack Scenario Testing Only
```bash
# Make sure backend is running first!
cd /Users/chasemad/Desktop/mini-xdr
python3 tests/test_comprehensive_attack_scenarios.py
```

Tests:
- 🎯 5 different attack types
- 🌐 Multiple IPs for same attack
- 🤖 Agent responses
- 🔌 MCP server integration
- 📊 Model classifications

---

## 🐛 FIXING THE 57% CONFIDENCE ISSUE

### Why This Happens

The model returning 57% confidence for all attacks indicates one of these issues:

1. **Model Architecture Mismatch**
   - Training architecture doesn't match inference architecture
   - Solution: Verify model metadata and reload

2. **Feature Scaling Problem**
   - Scaler not applied correctly
   - Features not normalized
   - Solution: Check scaler.pkl and retrain if needed

3. **Static Weights**
   - Model not properly trained
   - Weights stuck at initialization
   - Solution: Retrain model with fresh data

4. **Feature Extraction Bug**
   - All features have same value
   - NaN or Inf values in features
   - Solution: Debug feature extraction

### Diagnostic Steps

Run the model debugger:
```bash
python3 tests/test_model_confidence_debug.py
```

This will show you:
```
[1/7] Checking model files...
  ✅ threat_detector.pth - 1.12 MB
  ✅ scaler.pkl - 0.05 MB

[2/7] Checking model architecture...
  📋 Model Metadata:
     - Features: 79
     - Hidden dims: [512, 256, 128, 64]
     - Classes: 7
     - Accuracy: 72.67%

[3/7] Checking feature extraction...
  📊 Extracted 79 features
     Sample features:
       - event_count_1h: 10.0000
       - event_count_24h: 10.0000
  ✅ No NaN or Inf values

[4/7] Testing with synthetic attack data...
  🎯 SSH Brute Force:
     Confidence: 57.23%
     ⚠️  WARNING: Confidence is ~57% (suspicious!)

[5/7] Checking feature scaling...
  📊 Scaler type: StandardScaler
     - Mean range: [-0.5, 2.3]

[6/7] Checking model weights...
  ⚠️  WARNING: Weights look like they might be untrained!

[7/7] Testing with varied attack types...
  ❌ CRITICAL: Confidence doesn't change with input!
```

### Solutions

#### Solution 1: Retrain Model Locally
```bash
cd /Users/chasemad/Desktop/mini-xdr
python aws/train_local.py --epochs 50 --batch-size 256
```

This will:
- Load training data from `datasets/`
- Train on 79 features with 7 classes
- Save to `models/local_trained_enhanced/`
- Generate new scaler.pkl

#### Solution 2: Use AWS SageMaker Model
```bash
# Check SageMaker endpoints
aws sagemaker list-endpoints

# Configure backend to use SageMaker
# Edit backend/.env:
ML_BACKEND=sagemaker
SAGEMAKER_ENDPOINT=mini-xdr-threat-detector
```

#### Solution 3: Verify Feature Extraction
```python
# Test feature extraction with real events
from backend.app.deep_learning_models import deep_learning_manager
from backend.app.models import Event
from datetime import datetime

events = [...]  # Your events
features = deep_learning_manager._extract_features("192.168.1.100", events)
print(f"Features: {features}")
# Should show 79 features with varying values
```

---

## 🎯 ATTACK SCENARIO TESTING

### What Gets Tested

1. **Different Attack Types from Different IPs**
   ```
   203.0.113.10  → SSH Brute Force
   198.51.100.20 → DDoS Attack
   192.0.2.30    → Port Scan
   198.51.100.40 → Web Attack
   203.0.113.50  → Malware C2
   ```

2. **Same Attack from Multiple IPs**
   ```
   10.0.0.10 → SSH Brute Force → Incident #1
   10.0.0.11 → SSH Brute Force → Incident #2
   10.0.0.12 → SSH Brute Force → Incident #3
   ```

3. **Agent Responses**
   ```
   "Block this IP"                    → block_ip workflow
   "Investigate and gather forensics" → forensic investigation
   "Alert team and isolate host"      → multi-action workflow
   ```

### Expected Results

**Model Classification:**
```
Attack Type              Detected As              Confidence  Match
-----------------------------------------------------------------
SSH Brute Force          Brute Force Attack       72.5%       ✅
DDoS Attack              DDoS/DoS Attack          85.3%       ✅
Port Scan                Network Reconnaissance   68.9%       ✅
Web Attack               Web Application Attack   71.2%       ✅
Malware C2               Malware/Botnet          79.8%       ✅
```

**Confidence Variance:** Should be > 0.01 (not all the same)

**If All ~57%:** Model is stuck - retrain needed!

### Running Attack Tests

1. **Start Backend**
   ```bash
   cd backend
   uvicorn app.main:app --reload
   ```

2. **Run Tests**
   ```bash
   python3 tests/test_comprehensive_attack_scenarios.py
   ```

3. **Check Results**
   - View incidents: http://localhost:3000/incidents
   - Each attack should create a unique incident
   - Different IPs should create separate incidents
   - Confidence should vary between attacks

---

## 🤖 AGENT & MCP SERVER TESTING

### Test Agent Responses

```bash
# Via API
curl -X POST http://localhost:8000/api/agents/orchestrate \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Block this SSH brute force attack and alert the team",
    "incident_id": 1
  }'
```

Expected response:
```json
{
  "workflow_created": true,
  "workflow_id": "wf-123",
  "actions": ["block_ip", "alert_security_analysts"],
  "investigation_started": false
}
```

### Test MCP Server

1. **Health Check**
   ```bash
   curl http://localhost:8000/health
   ```

2. **Get Incidents**
   ```bash
   curl http://localhost:8000/incidents \
     -H "x-api-key: demo-minixdr-api-key"
   ```

3. **AI Analysis**
   ```bash
   curl -X POST http://localhost:8000/api/incidents/1/ai-analysis \
     -H "Content-Type: application/json" \
     -d '{"force_refresh": false}'
   ```

---

## 🚀 OPENING TPOT TO INTERNET

### ⚠️  Before You Do This

Run security audit first:
```bash
./scripts/security-audit-comprehensive.sh
```

Verify:
- ✅ Home lab is isolated (check output)
- ✅ Backend runs on localhost
- ✅ SSH keys are secure
- ✅ NSG rules are correct

### Opening to Internet

```bash
./scripts/open-azure-tpot-to-internet.sh
```

This will:
1. Update NSG rules to allow internet traffic to honeypots
2. Keep management ports (SSH, Web) restricted to your IP
3. Enable real attack capture

**After opening:**
- Honeypots will receive real attacks
- Attacks will be ingested into Mini-XDR
- Incidents will be created automatically
- Monitor: http://localhost:3000/incidents

### Closing Back Down

```bash
./scripts/secure-azure-tpot-testing.sh
```

This locks everything back down to your IP only.

---

## 📊 MONITORING AFTER TESTS

### View Incidents
```bash
# Via CLI
curl http://localhost:8000/incidents \
  -H "x-api-key: demo-minixdr-api-key" | jq .

# Via UI
open http://localhost:3000/incidents
```

### Check Model Performance
```bash
# Get incident with ML scores
curl http://localhost:8000/incidents/1 \
  -H "x-api-key: demo-minixdr-api-key" | jq .ml_anomaly_score
```

### View Agent Actions
```bash
# Get actions for incident
curl http://localhost:8000/incidents/1/actions \
  -H "x-api-key: demo-minixdr-api-key"
```

---

## 🔧 TROUBLESHOOTING

### Backend Won't Start
```bash
cd backend
source venv/bin/activate  # or: source ../venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

### Frontend Won't Start
```bash
cd frontend
npm install
npm run dev
```

### Can't Connect to TPOT
```bash
# Check your current IP
curl ifconfig.me

# Update NSG rules
./scripts/secure-azure-tpot-testing.sh

# Test connection
nc -zv 74.235.242.205 64295
```

### Model Returns 57% for Everything
```bash
# Run diagnostics
python3 tests/test_model_confidence_debug.py

# Retrain if needed
python aws/train_local.py

# Or use SageMaker
# Edit backend/.env: ML_BACKEND=sagemaker
```

### No Incidents Created
```bash
# Check event ingestion
curl -X POST http://localhost:8000/ingest/multi \
  -H "Content-Type: application/json" \
  -H "x-api-key: demo-minixdr-api-key" \
  -d '{"events": [...]}'

# Check logs
tail -f backend/backend.log
```

---

## 📝 SUMMARY CHECKLIST

Before opening TPOT to internet:

- [ ] Run security audit: `./scripts/security-audit-comprehensive.sh`
- [ ] Verify home lab is isolated (check audit output)
- [ ] Test model: `python3 tests/test_model_confidence_debug.py`
- [ ] Fix 57% issue if present (retrain model)
- [ ] Run attack tests: `python3 tests/test_comprehensive_attack_scenarios.py`
- [ ] Verify different attacks create different incidents
- [ ] Test agent responses work correctly
- [ ] Check MCP server endpoints
- [ ] Backend runs on 127.0.0.1 only
- [ ] Frontend works: http://localhost:3000

**When all checked:**
```bash
./scripts/open-azure-tpot-to-internet.sh
```

---

## 🎉 YOU'RE READY!

Your system is now:
- ✅ Secure (home lab isolated)
- ✅ Tested (model and agents working)
- ✅ Ready for production (TPOT can be opened)

**Next Steps:**
1. Open TPOT to internet
2. Monitor incoming attacks
3. Watch AI agents respond
4. Collect threat intelligence

**Happy Hunting! 🎯**

