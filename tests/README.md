# Azure Honeypot Comprehensive Testing Suite

## 🎯 Overview

This directory contains a complete testing infrastructure for validating your Mini-XDR Azure honeypot system, including ML models, AI agents, response tools, and workflows.

## 📁 Files in This Directory

| File | Purpose | Usage |
|------|---------|-------|
| `comprehensive_azure_honeypot_test.py` | **Main automated test suite** | `python3 comprehensive_azure_honeypot_test.py` |
| `live_honeypot_attack_suite.sh` | **Real attack generator** | `./live_honeypot_attack_suite.sh` |
| `run_all_tests.sh` | **Interactive test runner** | `./run_all_tests.sh` |
| `TESTING_GUIDE.md` | **Complete documentation** | Read for detailed instructions |
| `COMPREHENSIVE_TEST_SUMMARY.md` | **Implementation summary** | Overview of what was built |
| `README.md` | **This file** | Quick start guide |

## 🚀 Quick Start (3 Steps)

### 1. Prerequisites

```bash
# Ensure backend is running
cd ../backend
source venv/bin/activate
uvicorn app.main:app --reload &

# Install test dependencies
pip3 install aiohttp
```

### 2. Run Tests

```bash
# Option A: Interactive (easiest)
./run_all_tests.sh
# Then choose option 3

# Option B: Just validation
python3 comprehensive_azure_honeypot_test.py

# Option C: Full manual
./live_honeypot_attack_suite.sh  # Generate attacks
sleep 180                          # Wait for processing
python3 comprehensive_azure_honeypot_test.py  # Validate
```

### 3. Review Results

```bash
# Check console output for pass/fail
# View detailed JSON results
cat test_results_*.json | jq '.summary'
```

## 📊 What Gets Tested

### ✅ 35+ Automated Tests Covering:

1. **System Health** (4 tests)
   - Backend API, Database, ML Models, Agents

2. **Attack Detection** (4 tests)
   - SSH Brute Force, Port Scans, Malware, APT

3. **ML Predictions** (4 tests)
   - Incident creation, Confidence scores, Threat classification, Anomaly detection

4. **Agent Responses** (5 tests)
   - Containment, Forensics, Attribution, Deception, Threat Hunting

5. **Workflows & Tools** (4 tests)
   - NLP workflow creation, Response actions, Investigations

6. **Integrations** (4 tests)
   - Chat, Alerts, Dashboard, Events

7. **Advanced Scenarios** (3 tests)
   - Multi-stage attacks, Coordinated responses, Real-time connectivity

### 🎯 10 Live Attack Types:

1. SSH Brute Force (20 attempts)
2. Port Scanning (comprehensive)
3. Web Reconnaissance (12+ paths)
4. Telnet Brute Force
5. FTP Enumeration
6. SQL Injection
7. Directory Traversal
8. DNS Enumeration
9. SMTP Probing
10. Multi-Stage APT

## 📈 Success Criteria

**Minimum Passing**: 30/35 tests (85%)  
**Optimal**: 35/35 tests (100%)

**ML Confidence**: >60% average  
**Agents Responding**: 4+ out of 5  
**Workflows Created**: 1+ successful  

## 🔍 Verification Commands

```bash
# System status
curl -s http://localhost:8000/health | jq

# ML model status
curl -s -H "x-api-key: $API_KEY" http://localhost:8000/api/ml/status | jq

# Recent incidents
curl -s http://localhost:8000/incidents | jq '.[:5]'

# Agent status
curl -s -H "x-api-key: $API_KEY" http://localhost:8000/api/agents/status | jq

# Available workflows
curl -s -H "x-api-key: $API_KEY" http://localhost:8000/api/workflows | jq
```

## 📚 Documentation

- **Detailed Guide**: `TESTING_GUIDE.md` (comprehensive instructions)
- **Implementation Summary**: `COMPREHENSIVE_TEST_SUMMARY.md` (what was built)
- **Troubleshooting**: See TESTING_GUIDE.md sections on debugging

## 🎨 Output Colors

- 🟢 **Green ✅**: Test passed
- 🟡 **Yellow ⚠️**: Warning (non-critical)
- 🔴 **Red ❌**: Test failed
- 🔵 **Blue ℹ️**: Information
- 🟣 **Purple [ATTACK]**: Attack stage

## 🐛 Quick Troubleshooting

### Backend Not Running
```bash
cd ../backend
source venv/bin/activate
uvicorn app.main:app --reload
```

### No Incidents Created
```bash
# Check event ingestion
curl -s http://localhost:8000/events | jq 'length'

# Check detection logs
tail -f ../backend/backend.log | grep "incident"
```

### Agents Not Responding
```bash
# Verify agent credentials
cd ../scripts
./generate-agent-secrets-azure.sh
```

## 💡 Tips

1. **Run full suite first** to establish baseline
2. **Use live attacks** to test with real traffic
3. **Monitor logs** during tests: `tail -f ../backend/backend.log`
4. **Check T-Pot** for honeypot events: `https://$TPOT_HOST:64297`
5. **Review dashboard** at `http://localhost:3000`

## 🎯 Expected Timeline

- Live attacks: ~5-10 minutes
- Event processing: ~3 minutes
- Validation tests: ~5-10 minutes
- **Total**: ~20-25 minutes for full suite

## 📞 Getting Help

1. Check `TESTING_GUIDE.md` for detailed documentation
2. Review `COMPREHENSIVE_TEST_SUMMARY.md` for implementation details
3. Examine backend logs: `tail -f ../backend/backend.log`
4. Verify configuration: `cat ../backend/.env | grep -E "(API_KEY|TPOT_HOST)"`

## ✨ What Makes This Comprehensive

✅ **ML Model Validation**: Confidence scores, predictions, classifications  
✅ **Agent Testing**: All 5+ agents with real scenarios  
✅ **Tool Verification**: Response actions and workflows  
✅ **Attack Coverage**: 10+ different attack types  
✅ **Integration Tests**: End-to-end system validation  
✅ **Real-Time Testing**: Live attacks against actual honeypot  
✅ **Automated Reporting**: Detailed JSON results  

---

## 🚀 Ready to Test?

```bash
./run_all_tests.sh
```

Choose option 3 for the full suite and sit back! ☕

---

**Last Updated**: October 2025  
**Compatibility**: Mini-XDR v1.0+, Azure T-Pot Honeypot

