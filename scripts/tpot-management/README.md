# T-Pot Management Scripts

Management scripts for T-Pot honeypot deployment and integration with Mini-XDR.

## Quick Start: Setup Workflows

### 1️⃣ Setup All T-Pot Workflows (Recommended)

```bash
cd /Users/chasemad/Desktop/mini-xdr
source backend/venv/bin/activate
python3 scripts/tpot-management/setup-tpot-workflows.py
```

This creates **17 comprehensive workflows** for:
- SSH brute force attacks
- Malware uploads
- Ransomware detection
- Data exfiltration
- Cryptomining
- IoT botnets
- DDoS attacks
- Web attacks (SQL injection, XSS)
- And more!

### 2️⃣ Verify Configuration

```bash
./scripts/tpot-management/verify-tpot-workflows.sh
```

Checks that all workflows are properly configured and enabled.

### 3️⃣ View in UI

Navigate to: `http://localhost:3000/workflows`

Click the **"Auto Triggers"** tab to see all your workflows.

---

## Available Scripts

### Workflow Management

| Script | Purpose | Usage |
|--------|---------|-------|
| `setup-tpot-workflows.py` | Create/update all T-Pot workflows | `python3 setup-tpot-workflows.py` |
| `verify-tpot-workflows.sh` | Verify workflows are configured | `./verify-tpot-workflows.sh` |
| `TPOT_WORKFLOWS_GUIDE.md` | Comprehensive workflow documentation | Read for details |

### T-Pot Deployment

| Script | Purpose | Usage |
|--------|---------|-------|
| `setup-tpot-integration.sh` | Initial T-Pot integration setup | `./setup-tpot-integration.sh` |
| `deploy-tpot-logging.sh` | Configure log forwarding | `./deploy-tpot-logging.sh` |
| `start-secure-tpot.sh` | Start T-Pot with secure config | `./start-secure-tpot.sh` |
| `secure-tpot.sh` | Apply security hardening | `./secure-tpot.sh` |
| `kali-access.sh` | Configure Kali attack testing | `./kali-access.sh` |

---

## Workflow Coverage

✅ **Cowrie SSH/Telnet**: Brute force, successful logins, command execution
✅ **Dionaea Multi-Protocol**: Malware uploads, SMB exploits
✅ **Suricata IDS**: High-severity network alerts
✅ **Elasticpot**: Elasticsearch exploits
✅ **Honeytrap**: Port scanning, reconnaissance
✅ **Specialized Detectors**: Cryptomining, ransomware, data exfil, botnets, DDoS
✅ **Web Attacks**: SQL injection, XSS

### Auto-Execute vs Manual Approval

- **12 workflows** auto-execute immediately (critical threats)
- **5 workflows** require manual approval (common/lower-severity)

See `TPOT_WORKFLOWS_GUIDE.md` for complete details.

---

## Workflow Actions

Each workflow can perform these actions:

1. **Block IP** - Automatically block attacker IPs
2. **Create Incident** - Generate incident tickets
3. **AI Analysis** - Invoke AI agents for attribution/forensics
4. **Notifications** - Send Slack/email alerts
5. **Rate Limiting** - Apply traffic controls
6. **Isolation** - Quarantine affected systems

---

## Configuration

### API Key

Set your Mini-XDR API key:
```bash
export MINI_XDR_API_KEY="your-api-key-here"
```

### T-Pot Connection

T-Pot configuration: `/Users/chasemad/Desktop/mini-xdr/config/tpot/tpot-config.json`

### Database

Workflows are stored in: `xdr.db` (workflow_triggers table)

---

## Troubleshooting

### Workflows not triggering?

1. Check if backend is running: `curl http://localhost:8000/health`
2. Verify triggers are enabled: `./verify-tpot-workflows.sh`
3. Check logs: `tail -f backend/backend.log | grep trigger`

### Too many false positives?

Edit thresholds in workflows:
- Increase `threshold` values (e.g., 5 → 10)
- Extend `window_seconds` (e.g., 60 → 120)
- Change `auto_execute` to `false`

### Need to customize workflows?

1. Edit `setup-tpot-workflows.py`
2. Modify the `TPOT_TRIGGERS` array
3. Re-run the setup script

Or use the API:
```bash
curl -X PUT http://localhost:8000/api/triggers/{id} \
  -H "X-API-Key: your-key" \
  -H "Content-Type: application/json" \
  -d '{"auto_execute": false, "priority": "high"}'
```

---

## Testing Workflows

### Test SSH Brute Force

```bash
# From a test machine (will get blocked!)
for i in {1..10}; do
  ssh root@YOUR_TPOT_IP
done
```

### Test Port Scan

```bash
nmap -p 1-1000 YOUR_TPOT_IP
```

### Check Workflow Execution

```bash
curl http://localhost:8000/api/workflows \
  -H "X-API-Key: your-key" | jq
```

---

## Safety Features

### Rate Limiting
- **Cooldown periods** prevent workflow spam (30-300 seconds)
- **Daily limits** cap maximum triggers per day (25-100)

### Manual Approval
- Port scans and web attacks require approval
- Prevents over-blocking from common activities

### IP Block Durations
- **Low threats**: 1 hour
- **Medium threats**: 2 hours
- **High threats**: 24 hours
- **Critical threats**: 7 days

---

## Documentation

- **Complete Workflow Guide**: `TPOT_WORKFLOWS_GUIDE.md` (read this!)
- **T-Pot Deployment**: `../../ops/TPOT_DEPLOYMENT_GUIDE.md`
- **Workflow System**: `../../docs/WORKFLOW_SYSTEM_GUIDE.md`
- **API Docs**: `http://localhost:8000/docs`

---

## Azure Deployment

### Deploy T-Pot on Azure

```bash
# Use Azure deployment scripts
cd ../../ops
# Follow TPOT_DEPLOYMENT_GUIDE.md for Azure setup
```

### Configure Log Forwarding

After T-Pot is deployed:
```bash
# SSH into T-Pot instance
ssh -i ~/.ssh/mini-xdr-tpot-key.pem -p 64295 admin@YOUR_AZURE_IP

# Configure Fluent Bit to forward to Mini-XDR
# See deploy-tpot-logging.sh for details
```

---

## Support

For issues or questions:
1. Check `TPOT_WORKFLOWS_GUIDE.md` for detailed explanations
2. Run `./verify-tpot-workflows.sh` for diagnostics
3. Review backend logs: `tail -f backend/backend.log`
4. Check API docs: `http://localhost:8000/docs`

---

## Summary

✅ **17 automated workflows** covering all T-Pot attack types
✅ **AI-powered** attribution and forensics analysis
✅ **Rate-limited** to prevent exhaustion
✅ **Production-ready** with safety controls
✅ **Fully documented** with comprehensive guides

**Your T-Pot deployment will be fully protected! 🛡️**
