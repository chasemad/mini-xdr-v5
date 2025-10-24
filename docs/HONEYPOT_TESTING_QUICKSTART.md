# 🚀 Azure T-Pot Honeypot Testing - Quick Start Guide

This guide helps you verify and comprehensively test your Mini-XDR system's integration with the Azure T-Pot honeypot.

## 📋 Prerequisites

✅ **Azure T-Pot Honeypot** running at `74.235.242.205`
✅ **SSH Access** configured with key at `~/.ssh/mini-xdr-tpot-azure`
✅ **Mini-XDR Backend** running at `http://localhost:8000`
✅ **Fluent Bit** forwarding logs from T-Pot to Mini-XDR

---

## 🔍 Step 1: Verify Integration

First, verify that Azure T-Pot is properly connected and forwarding logs to Mini-XDR:

```bash
cd /Users/chasemad/Desktop/mini-xdr
chmod +x scripts/testing/verify-azure-honeypot-integration.sh
./scripts/testing/verify-azure-honeypot-integration.sh
```

### ✅ What This Checks:

1. **Azure T-Pot Connectivity** - SSH and web access
2. **Fluent Bit Log Forwarding** - Service status and configuration
3. **Mini-XDR Event Ingestion** - Events being received and processed
4. **Detection Models** - ML models loaded and ready
5. **End-to-End Flow** - Live attack simulation and detection

### Expected Output:

```
╔═══════════════════════════════════════════════════════════════════════╗
║    AZURE T-POT HONEYPOT INTEGRATION VERIFICATION                     ║
╚═══════════════════════════════════════════════════════════════════════╝

✅ SSH connection successful
✅ Fluent Bit service is running
✅ Mini-XDR backend is healthy
✅ ML detection model loaded
✅ AZURE HONEYPOT INTEGRATION VERIFIED!
```

---

## 🎯 Step 2: Comprehensive Attack Testing

Once verification passes, run the comprehensive attack simulation:

```bash
chmod +x scripts/testing/test-comprehensive-honeypot-attacks.sh
./scripts/testing/test-comprehensive-honeypot-attacks.sh
```

### 🚨 Attack Patterns Tested:

1. **SSH Brute Force** - 15 failed login attempts
2. **Password Spray** - 45 common passwords across multiple users
3. **SQL Injection** - 6 SQLi payloads
4. **Cross-Site Scripting (XSS)** - 4 XSS vectors
5. **Path Traversal** - Directory traversal attempts
6. **Malware Download** - Simulated malware C2 activity
7. **Data Exfiltration** - Credential theft and data upload
8. **Port Scan** - 16 ports scanned
9. **DDoS** - 150 rapid connection attempts
10. **APT Attack Chain** - Multi-stage persistent threat
11. **Credential Harvesting** - System file access attempts
12. **Ransomware** - File encryption simulation

### 📊 Test Features:

- **✅ Multiple Unique IPs** - 5-10 unique attacker sources per test
- **✅ Azure Blocking Verification** - SSH to T-Pot to verify iptables rules
- **✅ Detection Timing** - Measures average response time
- **✅ ML Confidence Scoring** - Verifies AI model predictions
- **✅ Comprehensive Reporting** - JSON and console output

---

## 🎨 Optional: Full Test Mode

For maximum coverage, run with the `--full-test` flag:

```bash
./scripts/testing/test-comprehensive-honeypot-attacks.sh --full-test
```

This generates **10 unique attacker IPs** and runs **all 12 attack patterns**.

---

## 📈 Understanding the Results

### Success Criteria

Your system should meet these thresholds:

| Metric | Target | Good |
|--------|--------|------|
| **Detection Accuracy** | >90% | ✅ 95%+ |
| **Average Response Time** | <5 seconds | ✅ <3s |
| **Blocking Effectiveness** | >80% | ✅ 90%+ |
| **False Positives** | 0 | ✅ 0 |

### Sample Output:

```
╔════════════════════════════════════════════════════════════════════════╗
║     🎯 MINI-XDR COMPREHENSIVE HONEYPOT ATTACK TEST                    ║
╚════════════════════════════════════════════════════════════════════════╝

📊 COMPREHENSIVE TEST RESULTS

Test Configuration:
  • Target: 74.235.242.205
  • Attacker IPs: 5 unique sources
  • Attack Patterns: 12 types
  • Duration: ~8 minutes

Detection Performance:
  • Total Attacks Launched: 12
  • Attacks Detected: 11 (91.7%)
  • Average Response Time: 2.3s
  • False Positives: 0 (0.0%)

Blocking Effectiveness:
  • IPs Blocked on Azure: 10
  • Blocking Success Rate: 83.3%

AI Agent Performance:
  • Automated Workflows Triggered: 15
  • ML Confidence Average: 78.5%

Success Criteria:
✅ Detection Accuracy: 91.7% (✓ >90%)
✅ Response Speed: 2.3s (✓ <5s)
✅ Blocking Effectiveness: 83.3% (✓ >80%)
✅ Zero False Positives (✓)

✅ PASS: System effectively detected and mitigated threats
```

---

## 🔧 Troubleshooting

### No Events Being Received

**Problem**: Mini-XDR shows 0 events from T-Pot

**Solutions**:
```bash
# 1. Check Fluent Bit service on T-Pot
ssh -i ~/.ssh/mini-xdr-tpot-azure -p 64295 azureuser@74.235.242.205 "systemctl status fluent-bit"

# 2. Check Fluent Bit logs
ssh -i ~/.ssh/mini-xdr-tpot-azure -p 64295 azureuser@74.235.242.205 "sudo journalctl -u fluent-bit -n 100"

# 3. Verify Mini-XDR backend is accessible from T-Pot
ssh -i ~/.ssh/mini-xdr-tpot-azure -p 64295 azureuser@74.235.242.205 "curl -v http://YOUR_PUBLIC_IP:8000/health"

# 4. Check Mini-XDR backend logs
cd /Users/chasemad/Desktop/mini-xdr/backend
tail -f backend.log | grep ingest
```

### Detection Not Working

**Problem**: Events received but no incidents created

**Solutions**:
```bash
# 1. Check detection model status
curl http://localhost:8000/api/ml/status | jq

# 2. Check recent events
curl http://localhost:8000/events?limit=10 | jq

# 3. Manually trigger detection
curl http://localhost:8000/api/detection/test

# 4. Review backend logs for errors
cd backend && tail -f backend.log | grep -i error
```

### IP Blocking Not Working

**Problem**: IPs not blocked on Azure T-Pot

**Solutions**:
```bash
# 1. Check if containment agent is configured
ssh -i ~/.ssh/mini-xdr-tpot-azure -p 64295 azureuser@74.235.242.205

# 2. Verify iptables rules
sudo iptables -L INPUT -n -v | head -20

# 3. Check if automated workflows are executing
curl http://localhost:8000/api/workflows | jq

# 4. Manually test IP blocking
curl -X POST http://localhost:8000/api/response/block-ip \
  -H "Content-Type: application/json" \
  -d '{"ip": "203.0.113.100", "reason": "test"}'
```

### SSH Connection Issues

**Problem**: Cannot SSH to Azure T-Pot

**Solutions**:
```bash
# 1. Verify SSH key permissions
chmod 600 ~/.ssh/mini-xdr-tpot-azure

# 2. Test SSH connection
ssh -vvv -i ~/.ssh/mini-xdr-tpot-azure -p 64295 azureuser@74.235.242.205

# 3. Check Azure NSG rules allow your IP
# Go to Azure Portal → T-Pot VM → Networking → Inbound Rules

# 4. Verify T-Pot VM is running
# Azure Portal → T-Pot VM → Overview → Status should be "Running"
```

---

## 📊 Viewing Live Results

### Mini-XDR Dashboard

```bash
# Start the frontend (if not already running)
cd /Users/chasemad/Desktop/mini-xdr/frontend
npm run dev

# Open in browser
open http://localhost:3000
```

Navigate to:
- **Incidents** - View detected threats and AI analysis
- **Events** - Real-time event stream
- **Intelligence** - Attack patterns and IOCs
- **Workflows** - Automated response actions

### T-Pot Dashboard

```bash
# Open T-Pot web interface
open https://74.235.242.205:64297
```

Login with your Azure T-Pot credentials to view:
- **Kibana** - Raw honeypot logs
- **Attack Map** - Geolocation of attackers
- **Suricata** - IDS alerts

---

## 🎯 What This Proves

After successful testing, you'll have evidence that your Mini-XDR system:

✅ **Detects** sophisticated multi-vector attacks from multiple sources
✅ **Classifies** threats accurately using ML models and AI agents
✅ **Responds** automatically within seconds
✅ **Blocks** malicious IPs on the Azure honeypot (verified)
✅ **Executes** proper containment workflows
✅ **Analyzes** attacks with AI-powered intelligence

---

## 📝 Detailed Test Report

After each test run, a detailed JSON report is saved:

```bash
# View the latest test report
ls -lt test_results_*.json | head -1 | xargs cat | jq
```

**Report includes**:
- Timestamp and configuration
- Detection metrics (accuracy, response times)
- Blocking effectiveness
- AI performance metrics
- Individual response times for each attack

---

## 🚀 Next Steps

Once testing is successful:

1. **Monitor Production Traffic**
   - Let the system run and observe real attacks
   - Review incidents daily
   - Tune detection thresholds as needed

2. **Integrate with SIEM**
   - Forward alerts to your existing SIEM
   - Set up notification channels (email, Slack, PagerDuty)

3. **Expand Honeypot Network**
   - Add more honeypots in different regions
   - Deploy different honeypot types (web, database, IoT)

4. **Fine-Tune ML Models**
   - Retrain with production data
   - Adjust confidence thresholds
   - Add custom threat patterns

---

## 📚 Additional Resources

- **Main README**: `/Users/chasemad/Desktop/mini-xdr/README.md`
- **Backend Logs**: `/Users/chasemad/Desktop/mini-xdr/backend/backend.log`
- **T-Pot Documentation**: [T-Pot Official Docs](https://github.com/telekom-security/tpotce)
- **Mini-XDR API Docs**: `http://localhost:8000/docs`

---

## 💬 Support

If you encounter issues:

1. Check the **Troubleshooting** section above
2. Review backend logs for error messages
3. Verify network connectivity between components
4. Ensure all services are running (backend, T-Pot, Fluent Bit)

---

**🎉 Happy Testing! Your AI-powered XDR system is ready to defend against real-world threats.**
