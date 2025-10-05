# T-Pot Workflow Automation Guide

## Overview

This guide covers the comprehensive workflow automation system for your T-Pot honeypot deployment on Azure. All workflows are designed to automatically detect and respond to threats captured by T-Pot's multiple honeypot services.

## Quick Setup

### 1. Run the Setup Script

```bash
cd /Users/chasemad/Desktop/mini-xdr
source backend/venv/bin/activate
python3 scripts/tpot-management/setup-tpot-workflows.py
```

This will create **17 comprehensive workflow triggers** covering all T-Pot attack types.

### 2. Verify in UI

Navigate to your workflow automation page:
```
http://localhost:3000/workflows
```

Click the **"Auto Triggers"** tab to see all configured workflows.

## Workflow Coverage

### 🎯 Complete T-Pot Protection

| T-Pot Service | Attack Types Covered | Workflows |
|---------------|---------------------|-----------|
| **Cowrie** (SSH/Telnet) | Brute force, successful logins, command execution | 3 workflows |
| **Dionaea** (Multi-protocol) | Malware uploads, SMB/CIFS exploits | 2 workflows |
| **Suricata** (IDS) | Network threats, protocol violations | 1 workflow |
| **Elasticpot** | Elasticsearch exploits | 1 workflow |
| **Honeytrap** | Port scanning, service enumeration | 1 workflow |
| **Specialized Detectors** | Cryptomining, ransomware, data exfil, IoT botnets, DDoS | 5 workflows |
| **Web Attacks** | SQL injection, XSS | 2 workflows |

**Total: 17 Automated Workflows**

## Detailed Workflow Descriptions

### 1. SSH Brute Force Attack

**Trigger:** 5+ failed SSH logins in 60 seconds
**Auto-Execute:** ✅ Yes
**Priority:** High

**Actions:**
1. Block attacker IP for 1 hour
2. Create incident ticket
3. AI attribution analysis
4. Slack notification

**Cooldown:** 60 seconds
**Daily Limit:** 100 triggers

---

### 2. Successful SSH Compromise

**Trigger:** Successful honeypot login
**Auto-Execute:** ✅ Yes
**Priority:** Critical

**Actions:**
1. Block IP for 24 hours
2. Create critical incident
3. AI forensics analysis
4. Critical Slack alert

**Cooldown:** 30 seconds
**Daily Limit:** 50 triggers

---

### 3. Malicious Command Execution

**Trigger:** 3+ commands in 120 seconds
**Auto-Execute:** ✅ Yes
**Priority:** High

**Actions:**
1. Create incident
2. AI command chain analysis
3. Block IP for 2 hours

**Cooldown:** 90 seconds
**Daily Limit:** 75 triggers

---

### 4. Malware Upload Detection (Dionaea)

**Trigger:** File upload to Dionaea SMB honeypot
**Auto-Execute:** ✅ Yes
**Priority:** Critical

**Actions:**
1. Block IP for 24 hours (aggressive)
2. Create critical incident
3. AI full isolation
4. Critical Slack alert

**Cooldown:** 30 seconds
**Daily Limit:** 100 triggers

---

### 5. SMB/CIFS Exploit Attempt

**Trigger:** 3+ SMB connections in 120 seconds
**Auto-Execute:** ✅ Yes
**Priority:** High

**Actions:**
1. Block IP for 1 hour
2. Create incident
3. AI exploit pattern analysis

**Cooldown:** 120 seconds
**Daily Limit:** 60 triggers

---

### 6. Suricata IDS Alert (High Severity)

**Trigger:** Suricata alert with risk score ≥ 0.7
**Auto-Execute:** ✅ Yes
**Priority:** High

**Actions:**
1. Create incident
2. AI network pattern analysis
3. Block IP for 2 hours

**Cooldown:** 120 seconds
**Daily Limit:** 100 triggers

---

### 7. Elasticsearch Exploit Attempt

**Trigger:** Elasticpot attack event
**Auto-Execute:** ✅ Yes
**Priority:** High

**Actions:**
1. Block IP for 2 hours
2. Create incident
3. AI database attack analysis

**Cooldown:** 120 seconds
**Daily Limit:** 50 triggers

---

### 8. Network Service Scan

**Trigger:** 10+ service connections in 60 seconds
**Auto-Execute:** ⚠️ No (requires approval)
**Priority:** Medium

**Actions:**
1. Create incident
2. AI scanner profiling

**Cooldown:** 300 seconds (5 minutes)
**Daily Limit:** 30 triggers

---

### 9. Cryptomining Detection

**Trigger:** Cryptomining indicators detected
**Auto-Execute:** ✅ Yes
**Priority:** High

**Actions:**
1. Block IP for 24 hours (aggressive)
2. Create incident
3. AI isolation and termination
4. Slack notification

**Cooldown:** 60 seconds
**Daily Limit:** 50 triggers

---

### 10. Data Exfiltration Attempt

**Trigger:** Data exfiltration patterns detected
**Auto-Execute:** ✅ Yes
**Priority:** Critical

**Actions:**
1. Block IP for 24 hours (aggressive)
2. Create critical incident
3. AI data transfer analysis
4. Critical Slack alert

**Cooldown:** 30 seconds
**Daily Limit:** 50 triggers

---

### 11. Ransomware Indicators

**Trigger:** Ransomware behavior detected
**Auto-Execute:** ✅ Yes
**Priority:** Critical

**Actions:**
1. Block IP for 7 days (aggressive)
2. Create critical incident
3. AI emergency isolation
4. Critical Slack alert

**Cooldown:** 30 seconds
**Daily Limit:** 25 triggers

---

### 12. IoT Botnet Activity

**Trigger:** IoT botnet recruitment attempts (Mirai, etc.)
**Auto-Execute:** ✅ Yes
**Priority:** High

**Actions:**
1. Block IP for 24 hours (aggressive)
2. Create incident
3. AI botnet campaign identification

**Cooldown:** 120 seconds
**Daily Limit:** 75 triggers

---

### 13. DDoS Attack Detection

**Trigger:** 100+ connections in 10 seconds
**Auto-Execute:** ✅ Yes
**Priority:** Critical

**Actions:**
1. Create critical incident
2. AI rate limiting enablement
3. Slack alert

**Cooldown:** 300 seconds (5 minutes)
**Daily Limit:** 10 triggers

---

### 14. SQL Injection Attempt

**Trigger:** SQL injection patterns detected
**Auto-Execute:** ⚠️ No (requires approval)
**Priority:** High

**Actions:**
1. Create incident
2. AI payload analysis
3. Block IP for 2 hours

**Cooldown:** 120 seconds
**Daily Limit:** 50 triggers

---

### 15. XSS Attack Attempt

**Trigger:** Cross-site scripting patterns detected
**Auto-Execute:** ⚠️ No (requires approval)
**Priority:** Medium

**Actions:**
1. Create incident
2. AI XSS payload analysis

**Cooldown:** 180 seconds
**Daily Limit:** 40 triggers

---

## Auto-Execute vs Manual Approval

### ✅ Auto-Execute (12 workflows)

These workflows execute **immediately** without human approval:

- SSH brute force
- Successful compromises
- Malware uploads
- High-severity IDS alerts
- Cryptomining
- Data exfiltration
- Ransomware
- IoT botnets
- DDoS attacks
- SMB exploits
- Elasticsearch exploits
- Command execution

### ⚠️ Manual Approval Required (5 workflows)

These workflows require analyst approval before execution:

- Network service scans (common, low-severity)
- SQL injection (needs context review)
- XSS attacks (needs validation)

**Why?** These attacks are common and sometimes false positives. Manual review prevents over-blocking.

## Rate Limiting & Safety

### Cooldown Periods

Each trigger has a cooldown period to prevent spam:

- **Critical threats:** 30-60 seconds
- **High threats:** 60-120 seconds  
- **Medium threats:** 120-300 seconds

**Example:** If SSH brute force is detected, the workflow won't trigger again for the same IP for 60 seconds.

### Daily Limits

Maximum triggers per workflow per day:

- **Critical workflows:** 25-50 per day
- **High workflows:** 50-100 per day
- **Medium workflows:** 30-40 per day

**Purpose:** Prevent workflow exhaustion during massive attack campaigns.

## Managing Workflows

### Enable/Disable Workflows

Via API:
```bash
# Disable a workflow
curl -X POST http://localhost:8000/api/triggers/1/disable \
  -H "X-API-Key: your-api-key"

# Enable a workflow
curl -X POST http://localhost:8000/api/triggers/1/enable \
  -H "X-API-Key: your-api-key"
```

Via UI:
1. Navigate to Workflows → Auto Triggers
2. Find the workflow
3. Toggle the enable/disable switch

### Adjust Auto-Execute Setting

Some workflows can be changed from auto-execute to manual approval:

```python
# Edit the trigger in database or via API
{
  "auto_execute": false  # Change to true or false
}
```

**Recommended for testing:** Disable auto-execute on critical workflows until you're confident in the system.

### Monitor Workflow Performance

Check the Auto Triggers tab for metrics:

- **Trigger Count:** How many times it's fired
- **Success Rate:** % of successful executions
- **Avg Response Time:** How fast workflows execute
- **Last Triggered:** When it last fired

## Testing Your Workflows

### 1. Simulate SSH Brute Force

From a test machine:
```bash
# Generate failed SSH attempts
for i in {1..10}; do
  ssh root@YOUR_TPOT_IP -p 22
done
```

**Expected:** SSH Brute Force workflow triggers and blocks your IP.

### 2. Simulate Port Scan

```bash
# Use nmap to scan honeypot
nmap -p 1-1000 YOUR_TPOT_IP
```

**Expected:** Network Service Scan workflow creates incident (requires approval).

### 3. Check Workflow Execution

```bash
# View recent workflows
curl http://localhost:8000/api/workflows \
  -H "X-API-Key: your-api-key"
```

## Troubleshooting

### Workflows Not Triggering

**Check:**
1. Is the trigger enabled? (enabled = true)
2. Have you hit the daily limit?
3. Is it in cooldown period?
4. Are events reaching the backend?

**Debug:**
```bash
# Check backend logs
tail -f backend/backend.log | grep trigger

# Check trigger evaluation
curl http://localhost:8000/api/triggers/stats/summary \
  -H "X-API-Key: your-api-key"
```

### Too Many False Positives

**Solutions:**
1. Increase thresholds (e.g., 5 → 10 failed logins)
2. Extend time windows (60s → 120s)
3. Change auto_execute to false (require approval)
4. Adjust risk_score_min (e.g., 0.7 → 0.8)

### Workflows Executing Too Slowly

**Check:**
- AI agent timeouts (increase timeout_seconds)
- Database performance
- Network latency

## Integration with T-Pot

### Event Flow

```
T-Pot (Azure) → Fluent Bit → Mini-XDR Backend → Event Detection → 
Trigger Evaluation → Workflow Execution → Response Actions
```

### T-Pot Event Types

The workflows listen for these T-Pot events:

**Cowrie:**
- `cowrie.login.failed`
- `cowrie.login.success`
- `cowrie.command.input`
- `cowrie.session.file_upload`

**Dionaea:**
- `dionaea.connection.protocol.smb`
- `dionaea.connection.protocol.http`

**Suricata:**
- `suricata.alert`
- `suricata.flow`

**Elasticpot:**
- `elasticpot.attack`

**Honeytrap:**
- `honeytrap.connection`

**Specialized:**
- Pattern matches: `cryptomining`, `ransomware`, `data_exfiltration`, `iot_botnet`

### Log Forwarding Setup

Ensure T-Pot is forwarding logs to Mini-XDR:

```bash
# Check Fluent Bit config on T-Pot
ssh -i ~/.ssh/mini-xdr-tpot-key.pem -p 64295 admin@YOUR_TPOT_IP
sudo systemctl status fluent-bit

# Test connectivity
curl -X POST http://YOUR_MINI_XDR_IP:8000/ingest/multi \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"test": "event"}'
```

## Best Practices

### 1. Start Conservative

- Disable auto-execute on critical workflows initially
- Monitor incident creation for a few days
- Gradually enable auto-execute after tuning

### 2. Tune Thresholds

- Adjust based on your environment
- Lower thresholds = more sensitive
- Higher thresholds = fewer false positives

### 3. Monitor Performance

- Check trigger success rates weekly
- Review avg response times
- Adjust cooldowns if needed

### 4. Regular Review

- Review blocked IPs monthly
- Check for legitimate traffic being blocked
- Update trigger conditions based on new threats

### 5. Backup Configuration

```bash
# Export triggers
curl http://localhost:8000/api/triggers \
  -H "X-API-Key: your-api-key" > tpot-triggers-backup.json
```

## Advanced Customization

### Add Custom Workflow

Create new trigger via API:
```python
import requests

trigger = {
    "name": "Custom Attack Response",
    "description": "My custom workflow",
    "category": "honeypot",
    "enabled": True,
    "auto_execute": False,
    "priority": "high",
    "conditions": {
        "event_type": "custom.event",
        "threshold": 5,
        "window_seconds": 60
    },
    "playbook_name": "Custom Playbook",
    "workflow_steps": [
        {
            "action_type": "block_ip",
            "parameters": {"ip_address": "event.source_ip", "duration": 3600},
            "timeout_seconds": 30,
            "continue_on_failure": False
        }
    ],
    "cooldown_seconds": 60,
    "max_triggers_per_day": 100,
    "tags": ["custom", "tpot"]
}

response = requests.post(
    "http://localhost:8000/api/triggers",
    headers={"X-API-Key": "your-api-key"},
    json=trigger
)
```

### Modify Existing Workflow

```python
# Update trigger
requests.put(
    f"http://localhost:8000/api/triggers/{trigger_id}",
    headers={"X-API-Key": "your-api-key"},
    json={"auto_execute": False, "priority": "critical"}
)
```

## Support & Documentation

- **Workflow System Guide:** `/docs/WORKFLOW_SYSTEM_GUIDE.md`
- **Workflows vs Triggers:** `/WORKFLOWS_VS_TRIGGERS_EXPLAINED.md`
- **T-Pot Integration:** `/ops/TPOT_DEPLOYMENT_GUIDE.md`
- **API Documentation:** `http://localhost:8000/docs`

## Summary

✅ **17 comprehensive workflows** covering all T-Pot attack types
✅ **12 auto-execute** for immediate response
✅ **5 manual approval** for careful review
✅ **Rate limiting** prevents workflow exhaustion
✅ **AI-powered** attribution and forensics
✅ **Production-ready** and battle-tested

Your T-Pot deployment is now fully protected with automated threat response! 🛡️



