# 🚀 Azure Honeypot Testing - Quick Start

## One-Command Testing

```bash
cd /Users/chasemad/Desktop/mini-xdr/tests && ./run_all_tests.sh
```

Choose option **3** for full comprehensive testing.

---

## What Gets Tested

### ✅ ML Models
- Model loading and status
- Prediction accuracy
- **Confidence scoring** (>60% for clear threats)
- Threat classification

### ✅ All 5+ Agents
- **Containment Agent** → Block IP, Isolate Host
- **Forensics Agent** → Evidence Collection
- **Attribution Agent** → Threat Intel Lookup
- **Deception Agent** → Attacker Profiling
- **Threat Hunting Agent** → Pattern Detection

### ✅ Response Tools & Workflows
- NLP workflow creation
- Response action execution
- Multi-agent coordination
- Investigation capabilities

### ✅ 10 Attack Scenarios
1. SSH Brute Force
2. Port Scanning
3. Web Attacks
4. Telnet/FTP Attacks
5. SQL Injection
6. Directory Traversal
7. DNS Enumeration
8. SMTP Probing
9. Multi-Stage APT
10. Combined attacks

---

## Quick Commands

```bash
# Full automated test
cd tests && python3 comprehensive_azure_honeypot_test.py

# Generate live attacks
cd tests && ./live_honeypot_attack_suite.sh

# Check system status
curl -s http://localhost:8000/health | jq

# Check ML confidence scores
curl -s http://localhost:8000/incidents | jq '.[] | {id, threat_type, ml_confidence}'

# Check agent responses
curl -s http://localhost:8000/incidents/1 | jq '.actions'

# View workflows
curl -s -H "x-api-key: $API_KEY" http://localhost:8000/api/workflows | jq
```

---

## Success Criteria

| Metric | Minimum | Optimal |
|--------|---------|---------|
| Tests Passed | 30/35 (85%) | 35/35 (100%) |
| ML Confidence | >50% avg | >70% avg |
| Agents Responding | 4/5 | 5/5 |
| Incidents Created | 2+ | 4+ |
| Workflows Created | 1+ | 3+ |

---

## Expected Output

```
╔════════════════════════════════════════════════════════════════╗
║     COMPREHENSIVE AZURE HONEYPOT TESTING SUITE                ║
╚════════════════════════════════════════════════════════════════╝

SECTION 1: SYSTEM HEALTH & CONFIGURATION
[TEST] 1.1 Backend Health Check
✅ Backend Health: PASSED Database: connected
[TEST] 1.2 ML Model Status
✅ ML Model Status: PASSED 3 models trained
...

TEST SUMMARY
Total Tests: 35
Passed: 33
Failed: 0
Warnings: 2

🎉 ALL TESTS PASSED!
```

---

## Troubleshooting

### No incidents created?
```bash
# Check events
curl -s http://localhost:8000/events | jq 'length'

# Check logs
tail -f backend/backend.log | grep "incident"
```

### Low ML confidence?
```bash
# Check SageMaker
curl -s -H "x-api-key: $API_KEY" http://localhost:8000/api/ml/sagemaker/status | jq
```

### Agents not responding?
```bash
# Check agent status
curl -s -H "x-api-key: $API_KEY" http://localhost:8000/api/agents/status | jq
```

---

## Files Created

```
tests/
├── comprehensive_azure_honeypot_test.py  ← Main test suite (35+ tests)
├── live_honeypot_attack_suite.sh         ← Attack generator (10 types)
├── run_all_tests.sh                      ← Interactive runner
├── README.md                             ← Quick start
├── TESTING_GUIDE.md                      ← Complete guide
└── COMPREHENSIVE_TEST_SUMMARY.md         ← Implementation details
```

---

## Timeline

- **Live attacks**: 5-10 minutes
- **Processing wait**: 3 minutes
- **Validation**: 5-10 minutes
- **Total**: ~20-25 minutes

---

## Documentation

- **Quick Start**: `tests/README.md`
- **Detailed Guide**: `tests/TESTING_GUIDE.md`
- **Implementation**: `tests/COMPREHENSIVE_TEST_SUMMARY.md`
- **This File**: Quick reference for immediate use

---

## Support

1. Check `tests/TESTING_GUIDE.md` for detailed troubleshooting
2. Review backend logs: `tail -f backend/backend.log`
3. Verify configuration: `cat backend/.env | grep TPOT_HOST`

---

**Ready? Run this:**

```bash
cd /Users/chasemad/Desktop/mini-xdr/tests && ./run_all_tests.sh
```

Choose option **3** and let it run! ☕

---

**Last Updated**: October 2025  
**Version**: 1.0

