# T-Pot Azure Deployment - Complete Workflow Setup Guide

## 🎯 Overview

This guide covers the complete setup of automated workflow triggers for your T-Pot honeypot deployment on Azure. You'll have **17 comprehensive workflows** that automatically detect and respond to all attack types captured by T-Pot.

## 📋 What You're Getting

### Complete Threat Coverage

| Category | Workflows | Auto-Execute |
|----------|-----------|--------------|
| **SSH/Telnet Attacks** | 3 workflows | ✅ Yes |
| **Malware & Exploits** | 2 workflows | ✅ Yes |
| **Network IDS Alerts** | 1 workflow | ✅ Yes |
| **Database Attacks** | 1 workflow | ✅ Yes |
| **Port Scanning** | 1 workflow | ⚠️ Manual approval |
| **Advanced Threats** | 5 workflows (crypto, ransomware, exfil, botnets, DDoS) | ✅ Yes |
| **Web Attacks** | 2 workflows (SQL injection, XSS) | ⚠️ Manual approval |

**Total: 17 Automated Workflows**

### Response Actions

Each workflow can:
- 🚫 **Block attacker IPs** (30 mins to 7 days)
- 📋 **Create incident tickets** with full context
- 🤖 **Invoke AI agents** for attribution/forensics
- 📢 **Send notifications** (Slack, email, webhooks)
- 🛡️ **Apply rate limiting** during DDoS
- 🔒 **Isolate systems** for critical threats

---

## 🚀 Quick Setup (5 Minutes)

### One-Command Setup

```bash
cd /Users/chasemad/Desktop/mini-xdr
bash scripts/tpot-management/setup-all-tpot-workflows.sh
```

This script will:
1. ✅ Activate the Python virtual environment
2. ✅ Start the backend if needed
3. ✅ Create all 17 workflow triggers
4. ✅ Verify the configuration
5. ✅ Show you the summary

**That's it!** Your workflows are ready.

---

## 📖 Manual Setup (If Needed)

### Step 1: Activate Environment

```bash
cd /Users/chasemad/Desktop/mini-xdr/backend
source venv/bin/activate
```

### Step 2: Start Backend (if not running)

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Step 3: Run Setup Script

```bash
cd /Users/chasemad/Desktop/mini-xdr
python3 scripts/tpot-management/setup-tpot-workflows.py
```

### Step 4: Verify Configuration

```bash
bash scripts/tpot-management/verify-tpot-workflows.sh
```

---

## 🔍 View Your Workflows

### In the Web UI

1. Open: `http://localhost:3000/workflows`
2. Click the **"Auto Triggers"** tab
3. See all 17 workflows with their status

### Via API

```bash
curl http://localhost:8000/api/triggers \
  -H "X-API-Key: your-api-key" | jq
```

---

## 📊 Workflow Details

### Critical Auto-Execute Workflows (Immediate Response)

#### 1. SSH Brute Force Attack
- **Trigger:** 5 failed logins in 60 seconds
- **Actions:** Block IP (1 hour) → Create incident → AI attribution → Alert
- **Cooldown:** 60 seconds
- **Daily Limit:** 100

#### 2. Successful SSH Compromise
- **Trigger:** Any successful honeypot login
- **Actions:** Block IP (24 hours) → Critical incident → AI forensics → Alert
- **Cooldown:** 30 seconds
- **Daily Limit:** 50

#### 3. Malicious Command Execution
- **Trigger:** 3+ commands in 120 seconds
- **Actions:** Create incident → AI command analysis → Block IP (2 hours)
- **Cooldown:** 90 seconds
- **Daily Limit:** 75

#### 4. Malware Upload (Dionaea)
- **Trigger:** File upload to SMB honeypot
- **Actions:** Block IP (24 hours) → Critical incident → AI isolation → Alert
- **Cooldown:** 30 seconds
- **Daily Limit:** 100

#### 5. Ransomware Indicators
- **Trigger:** Ransomware behavior detected
- **Actions:** Block IP (7 days!) → Critical incident → Emergency isolation → Alert
- **Cooldown:** 30 seconds
- **Daily Limit:** 25

#### 6. Data Exfiltration
- **Trigger:** Data exfiltration patterns
- **Actions:** Block IP (24 hours) → Critical incident → AI analysis → Alert
- **Cooldown:** 30 seconds
- **Daily Limit:** 50

#### 7. Cryptomining Detection
- **Trigger:** Mining software/pool connections
- **Actions:** Block IP (24 hours) → High incident → AI termination → Alert
- **Cooldown:** 60 seconds
- **Daily Limit:** 50

#### 8. IoT Botnet Activity
- **Trigger:** IoT botnet patterns (Mirai, etc.)
- **Actions:** Block IP (24 hours) → High incident → AI campaign ID
- **Cooldown:** 120 seconds
- **Daily Limit:** 75

#### 9. DDoS Attack
- **Trigger:** 100+ connections in 10 seconds
- **Actions:** Critical incident → AI rate limiting → Alert
- **Cooldown:** 300 seconds (5 minutes)
- **Daily Limit:** 10

#### 10. SMB/CIFS Exploits
- **Trigger:** 3+ SMB connections in 120 seconds
- **Actions:** Block IP (1 hour) → High incident → AI exploit analysis
- **Cooldown:** 120 seconds
- **Daily Limit:** 60

#### 11. Suricata High-Severity Alerts
- **Trigger:** IDS alert with risk score ≥ 0.7
- **Actions:** Create incident → AI network analysis → Block IP (2 hours)
- **Cooldown:** 120 seconds
- **Daily Limit:** 100

#### 12. Elasticsearch Exploits
- **Trigger:** Elasticpot attack event
- **Actions:** Block IP (2 hours) → High incident → AI database analysis
- **Cooldown:** 120 seconds
- **Daily Limit:** 50

### Manual Approval Workflows (Requires Review)

#### 13. Network Service Scan
- **Trigger:** 10+ service connections in 60 seconds
- **Actions:** Create incident → AI scanner profiling
- **Why manual?** Port scans are common and usually low-severity
- **Cooldown:** 300 seconds
- **Daily Limit:** 30

#### 14. SQL Injection Attempt
- **Trigger:** SQL injection patterns detected
- **Actions:** Create incident → AI payload analysis → Block IP (2 hours)
- **Why manual?** Needs context review to avoid false positives
- **Cooldown:** 120 seconds
- **Daily Limit:** 50

#### 15. XSS Attack Attempt
- **Trigger:** Cross-site scripting patterns
- **Actions:** Create incident → AI XSS payload analysis
- **Why manual?** Common attack, needs validation
- **Cooldown:** 180 seconds
- **Daily Limit:** 40

---

## 🛡️ Safety Features

### Rate Limiting

**Cooldown Periods:**
- Critical threats: 30-60 seconds between triggers
- High threats: 60-120 seconds
- Medium threats: 120-300 seconds

**Why?** Prevents workflow spam during sustained attacks.

### Daily Limits

**Maximum Triggers per Day:**
- Critical workflows: 25-50 per day
- High workflows: 50-100 per day
- Medium workflows: 30-40 per day

**Why?** Prevents workflow exhaustion during massive campaigns.

### IP Block Durations

**Automatic IP Blocking:**
- Port scans: N/A (manual approval)
- Brute force: 1 hour
- Command execution: 2 hours
- Malware/exploits: 24 hours
- Ransomware: 7 days

**Block Levels:**
- Standard: Regular firewall block
- Aggressive: Enhanced blocking with upstream filtering

---

## 🎨 Customization

### Change Auto-Execute Settings

If you want to require approval for any workflow:

```bash
# Via API
curl -X PUT http://localhost:8000/api/triggers/{trigger_id} \
  -H "X-API-Key: your-key" \
  -H "Content-Type: application/json" \
  -d '{"auto_execute": false}'
```

### Adjust Thresholds

Edit `scripts/tpot-management/setup-tpot-workflows.py`:

```python
# Example: Make SSH brute force less sensitive
"conditions": {
    "event_type": "cowrie.login.failed",
    "threshold": 10,  # Changed from 5 to 10
    "window_seconds": 120,  # Changed from 60 to 120
}
```

Then re-run: `python3 scripts/tpot-management/setup-tpot-workflows.py`

### Enable/Disable Workflows

```bash
# Disable a workflow
curl -X POST http://localhost:8000/api/triggers/{id}/disable \
  -H "X-API-Key: your-key"

# Enable a workflow
curl -X POST http://localhost:8000/api/triggers/{id}/enable \
  -H "X-API-Key: your-key"
```

---

## 🧪 Testing Your Workflows

### Test SSH Brute Force

```bash
# From a test machine (WARNING: Your IP will be blocked!)
for i in {1..10}; do
  ssh root@YOUR_TPOT_IP
done
```

**Expected Result:**
- Workflow triggers after 5 failed attempts
- Your IP gets blocked for 1 hour
- Incident created with "SSH Brute Force Attack"
- Slack notification sent (if configured)

### Test Port Scan

```bash
# Use nmap
nmap -p 1-1000 YOUR_TPOT_IP
```

**Expected Result:**
- Workflow triggers after 10+ connections
- Incident created requiring manual approval
- No automatic IP block (manual approval needed)

### Verify Workflow Execution

```bash
# Check recent workflows
curl http://localhost:8000/api/workflows \
  -H "X-API-Key: your-key" | jq '.[] | {id, status, playbook_name}'

# Check trigger metrics
curl http://localhost:8000/api/triggers/stats/summary \
  -H "X-API-Key: your-key" | jq
```

---

## 🔗 Integration with Azure T-Pot

### Event Flow

```
Azure T-Pot → Fluent Bit → Mini-XDR Backend → Event Detection → 
Trigger Evaluation → Workflow Execution → Response Actions
```

### T-Pot Event Types Monitored

**Cowrie (SSH/Telnet):**
- `cowrie.login.failed`
- `cowrie.login.success`
- `cowrie.command.input`
- `cowrie.session.file_upload`

**Dionaea (Multi-Protocol):**
- `dionaea.connection.protocol.smb`
- `dionaea.connection.protocol.http`

**Suricata (IDS):**
- `suricata.alert`
- `suricata.flow`

**Elasticpot:**
- `elasticpot.attack`

**Honeytrap:**
- `honeytrap.connection`

**Pattern Matches:**
- `cryptomining`, `ransomware`, `data_exfiltration`, `iot_botnet`

### Configure Log Forwarding

After deploying T-Pot on Azure:

```bash
# SSH into your T-Pot instance
ssh -i ~/.ssh/mini-xdr-tpot-key.pem -p 64295 admin@YOUR_AZURE_IP

# Deploy log forwarding configuration
bash scripts/tpot-management/deploy-tpot-logging.sh
```

This configures Fluent Bit to forward all T-Pot logs to your Mini-XDR backend.

---

## 📈 Monitoring & Maintenance

### Check Workflow Performance

```bash
# View trigger statistics
curl http://localhost:8000/api/triggers/stats/summary \
  -H "X-API-Key: your-key" | jq
```

**Metrics Available:**
- Total triggers fired
- Success rate (%)
- Average response time (ms)
- Last triggered timestamp

### Review Blocked IPs

```bash
# Check blocked IPs
curl http://localhost:8000/api/containment/blocked-ips \
  -H "X-API-Key: your-key" | jq
```

### Monitor Backend Logs

```bash
# Watch for trigger evaluation
tail -f backend/backend.log | grep -E "trigger|workflow"

# Watch for workflow execution
tail -f backend/backend.log | grep "Executing workflow"
```

### Weekly Review

1. Check trigger success rates (should be >90%)
2. Review false positives (manual approval workflows)
3. Adjust thresholds if needed
4. Update threat patterns for new attacks

---

## 🆘 Troubleshooting

### Workflows Not Triggering

**Check:**
1. Backend running? `curl http://localhost:8000/health`
2. Triggers enabled? `./scripts/tpot-management/verify-tpot-workflows.sh`
3. Events being ingested? `tail -f backend/backend.log | grep ingest`
4. Trigger conditions met? Check event counts and thresholds

**Debug:**
```bash
# Check trigger evaluation
tail -f backend/backend.log | grep "Evaluating.*triggers"

# Check incident creation
curl http://localhost:8000/api/incidents \
  -H "X-API-Key: your-key" | jq
```

### Too Many False Positives

**Solutions:**
1. Increase thresholds (5 → 10 failed logins)
2. Extend time windows (60s → 120s)
3. Change auto_execute to false (require approval)
4. Adjust risk_score_min (0.7 → 0.8)

### Workflows Executing Slowly

**Check:**
- AI agent timeout settings (increase if needed)
- Database performance
- Network latency to external services

---

## 📚 Additional Documentation

- **Complete Workflow Details:** `scripts/tpot-management/TPOT_WORKFLOWS_GUIDE.md`
- **T-Pot Deployment Guide:** `ops/TPOT_DEPLOYMENT_GUIDE.md`
- **Workflow System Architecture:** `docs/WORKFLOW_SYSTEM_GUIDE.md`
- **Workflows vs Triggers:** `WORKFLOWS_VS_TRIGGERS_EXPLAINED.md`
- **API Documentation:** `http://localhost:8000/docs`

---

## ✅ Pre-Deployment Checklist

Before deploying T-Pot on Azure:

- [ ] Backend is running and healthy
- [ ] All 17 workflows are created and enabled
- [ ] Verification script passes all checks
- [ ] You've reviewed auto-execute settings
- [ ] You understand rate limiting and cooldowns
- [ ] API keys are configured
- [ ] You've tested with simulated attacks (optional)

---

## 🎉 Summary

You now have:

✅ **17 comprehensive workflows** covering all T-Pot attack types
✅ **12 auto-execute workflows** for immediate threat response
✅ **5 manual approval workflows** for careful review
✅ **AI-powered analysis** for attribution and forensics
✅ **Rate limiting** to prevent workflow exhaustion
✅ **Safety controls** with cooldowns and daily limits
✅ **Production-ready** configuration

**Your T-Pot deployment on Azure will be fully protected!** 🛡️

---

## 🚀 Next Steps

1. **Deploy T-Pot on Azure** (if not already done)
   - Follow: `ops/TPOT_DEPLOYMENT_GUIDE.md`

2. **Configure Log Forwarding**
   - Run: `bash scripts/tpot-management/deploy-tpot-logging.sh`

3. **Monitor Workflows**
   - UI: `http://localhost:3000/workflows` → "Auto Triggers" tab
   - Logs: `tail -f backend/backend.log | grep workflow`

4. **Test with Real Attacks**
   - Port scan from external host
   - Brute force SSH attempts
   - Verify workflows trigger and respond

5. **Tune Configuration**
   - Review first week of incidents
   - Adjust thresholds if needed
   - Enable/disable workflows based on your environment

---

**Questions? Issues?**
- Check the troubleshooting section above
- Review `TPOT_WORKFLOWS_GUIDE.md` for detailed explanations
- Run verification: `./scripts/tpot-management/verify-tpot-workflows.sh`

**Happy Honeypotting! 🍯🐝**


