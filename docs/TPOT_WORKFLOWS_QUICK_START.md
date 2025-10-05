# T-Pot Workflows - Quick Start Guide

## 🚀 One-Command Setup

```bash
cd /Users/chasemad/Desktop/mini-xdr
bash scripts/tpot-management/setup-all-tpot-workflows.sh
```

**That's it!** This creates 17 automated workflows for your T-Pot deployment.

---

## ✅ Verify Setup

```bash
bash scripts/tpot-management/verify-tpot-workflows.sh
```

---

## 🎯 View Workflows

**Web UI:** `http://localhost:3000/workflows` → Click **"Auto Triggers"** tab

**API:**
```bash
# Use your existing API key from backend/.env
cd backend
curl http://localhost:8000/api/triggers \
  -H "X-API-Key: $(grep '^API_KEY=' .env | cut -d '=' -f2)" | jq
```

---

## 📋 What You Get

### 17 Automated Workflows:

| Attack Type | Workflows | Auto-Execute |
|-------------|-----------|--------------|
| SSH/Telnet Attacks | 3 | ✅ Yes |
| Malware & Exploits | 2 | ✅ Yes |
| Network Threats | 1 | ✅ Yes |
| Database Attacks | 1 | ✅ Yes |
| Port Scanning | 1 | ⚠️ Manual |
| Advanced Threats | 5 | ✅ Yes |
| Web Attacks | 2 | ⚠️ Manual |

**Total: 12 auto-execute, 5 manual approval**

---

## 🔥 Auto-Execute Workflows (Immediate Response)

1. **SSH Brute Force** - 5 failed logins → Block 1 hour
2. **Successful SSH Login** - Any success → Block 24 hours
3. **Malicious Commands** - 3+ commands → Block 2 hours
4. **Malware Upload** - File upload → Block 24 hours
5. **Ransomware** - Indicators detected → Block 7 days
6. **Data Exfiltration** - Exfil patterns → Block 24 hours
7. **Cryptomining** - Mining detected → Block 24 hours
8. **IoT Botnet** - Botnet activity → Block 24 hours
9. **DDoS Attack** - 100+ connections → Rate limiting
10. **SMB Exploits** - SMB attacks → Block 1 hour
11. **IDS High-Severity** - Suricata alerts → Block 2 hours
12. **Elasticsearch Exploits** - DB attacks → Block 2 hours

---

## ⚠️ Manual Approval Workflows

13. **Port Scan** - Needs approval (common activity)
14. **SQL Injection** - Needs approval (context review)
15. **XSS Attack** - Needs approval (validation needed)

---

## 🛡️ Response Actions

Each workflow can:
- 🚫 Block attacker IPs (30 mins to 7 days)
- 📋 Create incident tickets
- 🤖 Invoke AI agents (attribution/forensics)
- 📢 Send notifications (Slack/email)
- 🔒 Isolate systems (critical threats)

---

## 🎨 Quick Customization

### Disable Auto-Execute (Require Approval)

```bash
curl -X PUT http://localhost:8000/api/triggers/{id} \
  -H "X-API-Key: your-key" \
  -d '{"auto_execute": false}'
```

### Adjust Thresholds

Edit: `scripts/tpot-management/setup-tpot-workflows.py`

Change values like:
```python
"threshold": 10,  # Changed from 5
"window_seconds": 120,  # Changed from 60
```

Re-run: `python3 scripts/tpot-management/setup-tpot-workflows.py`

---

## 🧪 Test Your Workflows

### SSH Brute Force Test

```bash
# WARNING: Your IP will be blocked!
for i in {1..10}; do ssh root@YOUR_TPOT_IP; done
```

### Port Scan Test

```bash
nmap -p 1-1000 YOUR_TPOT_IP
```

### Check Execution

```bash
curl http://localhost:8000/api/workflows \
  -H "X-API-Key: your-key" | jq
```

---

## 📊 Monitor Performance

```bash
# View trigger stats
curl http://localhost:8000/api/triggers/stats/summary \
  -H "X-API-Key: your-key" | jq

# Watch logs
tail -f backend/backend.log | grep workflow
```

---

## 🆘 Troubleshooting

**Workflows not triggering?**

```bash
# 1. Check backend
curl http://localhost:8000/health

# 2. Verify workflows
bash scripts/tpot-management/verify-tpot-workflows.sh

# 3. Check logs
tail -f backend/backend.log | grep trigger
```

**Too many false positives?**
- Increase thresholds (5 → 10)
- Extend time windows (60s → 120s)
- Change auto_execute to false

---

## 📚 Full Documentation

- **Complete Guide:** `TPOT_AZURE_DEPLOYMENT_COMPLETE_GUIDE.md`
- **Detailed Workflows:** `scripts/tpot-management/TPOT_WORKFLOWS_GUIDE.md`
- **T-Pot Deployment:** `ops/TPOT_DEPLOYMENT_GUIDE.md`

---

## 🎯 Next Steps

1. ✅ Setup workflows (you're here!)
2. Deploy T-Pot on Azure
3. Configure log forwarding
4. Test with attacks
5. Monitor and tune

---

**Questions?** See `TPOT_AZURE_DEPLOYMENT_COMPLETE_GUIDE.md` for everything!

**Ready to deploy?** Your T-Pot is fully protected! 🛡️

