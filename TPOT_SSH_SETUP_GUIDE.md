# T-Pot SSH Integration Setup Guide

## Overview

This guide explains how to configure Mini-XDR AI agents to SSH into your T-Pot honeypot for automated defensive actions.

## What This Enables

Once configured, your AI agents will be able to:

1. **Monitor T-Pot in Real-Time** - Ingest attacks from Cowrie, Dionaea, Suricata, and other honeypots
2. **Execute Defensive Actions via SSH** - Block malicious IPs using UFW firewall
3. **Manage Honeypot Containers** - Start/stop specific honeypot services
4. **Analyze Attack Patterns** - Query Elasticsearch for threat intelligence
5. **Respond Automatically** - Take action based on configured workflows

## Prerequisites

✅ T-Pot honeypot running and accessible (you have this at 203.0.113.42)
✅ SSH access enabled on T-Pot (port 64295)
✅ Firewall configured to allow your Mini-XDR host (done: 172.16.110.0/24)
✅ T-Pot admin credentials

## Quick Setup (5 Minutes)

### Step 1: Run the Configuration Script

```bash
cd /Users/chasemad/Desktop/mini-xdr
./scripts/tpot-management/setup-tpot-ssh-integration.sh
```

This interactive script will:
- Prompt for your T-Pot IP (203.0.113.42)
- Configure SSH authentication (password or key)
- Test the SSH connection
- Update backend `.env` configuration
- Create helper scripts

**When prompted:**
- T-Pot IP: `203.0.113.42`
- SSH Port: `64295` (default)
- Web Port: `64297` (default)
- Username: `admin` (or your T-Pot admin username)
- Authentication: Choose password (recommended for T-Pot)

### Step 2: Verify Configuration

```bash
./scripts/tpot-management/verify-agent-ssh-actions.sh
```

This will test:
- ✅ Backend connectivity
- ✅ T-Pot SSH connection
- ✅ IP blocking capability
- ✅ Container management
- ✅ Log monitoring
- ✅ Workflow configuration

### Step 3: Start Mini-XDR

```bash
cd backend
python -m uvicorn app.main:app --reload
```

The backend will automatically:
1. Connect to T-Pot via SSH
2. Set up tunnels for Elasticsearch and Kibana
3. Start monitoring honeypot logs
4. Enable real-time attack ingestion

### Step 4: Run Demo Attack

```bash
./scripts/demo/demo-attack.sh
```

Or manually trigger SSH brute force:

```bash
# From another terminal, simulate attack on T-Pot
for i in {1..10}; do
    ssh -p 64295 admin@203.0.113.42 "wrong_password_$i" 2>/dev/null
done
```

Watch Mini-XDR:
1. Detect the brute force attack
2. Create an incident
3. Automatically SSH into T-Pot
4. Block the attacker's IP via UFW
5. Log all actions

## Configuration Details

### Environment Variables (backend/.env)

After running the setup script, these are configured:

```bash
# T-Pot Connection
TPOT_HOST=203.0.113.42
TPOT_SSH_PORT=64295
TPOT_WEB_PORT=64297
TPOT_API_KEY=<your-ssh-password>

# SSH Configuration
HONEYPOT_HOST=203.0.113.42
HONEYPOT_USER=admin
HONEYPOT_SSH_PORT=64295
HONEYPOT_SSH_KEY=~/.ssh/id_rsa
```

### SSH Access Methods

**Method 1: Password Authentication (Recommended for T-Pot)**
- More secure for sudo operations
- T-Pot uses password-authenticated sudo for UFW
- Configure with `TPOT_API_KEY` in `.env`

**Method 2: SSH Key Authentication**
- Faster, no password prompts
- May require passwordless sudo configuration on T-Pot
- Configure with `HONEYPOT_SSH_KEY` in `.env`

### Defensive Actions Available

Once connected, agents can execute:

#### 1. Block IP Address
```python
# Via T-Pot connector (password-authenticated sudo)
result = await connector.block_ip("192.0.2.1")
# Executes: echo 'password' | sudo -S ufw deny from 192.0.2.1
```

#### 2. Unblock IP Address
```python
result = await connector.unblock_ip("192.0.2.1")
```

#### 3. Query Active Blocks
```python
result = await connector.get_active_blocks()
# Returns: {"blocked_ips": ["192.0.2.1", ...], "count": 5}
```

#### 4. Container Management
```python
# Stop a honeypot
result = await connector.stop_honeypot_container("cowrie")

# Start a honeypot
result = await connector.start_honeypot_container("cowrie")

# Get container status
result = await connector.get_container_status()
```

#### 5. Execute Custom Commands
```python
result = await connector.execute_command("docker ps")
```

## Workflows

### SSH Brute Force Detection Workflow

**Trigger:** 6+ failed SSH login attempts in 60 seconds

**Actions:**
1. **Containment Agent**: Block source IP on T-Pot firewall (via SSH)
2. **Attribution Agent**: Lookup IP reputation (AbuseIPDB, VirusTotal)
3. **Forensics Agent**: Collect attack artifacts (usernames, passwords tried)
4. **Deception Agent**: Deploy additional SSH honeypot if attacker sophisticated
5. **Notification**: Alert SOC via Slack/email

**Setup Workflow:**
```bash
cd scripts/tpot-management
python setup-tpot-workflows.py
```

## Monitoring

### Real-Time Log Ingestion

T-Pot connector monitors these honeypot logs:

| Honeypot | Log Path | Attack Types |
|----------|----------|--------------|
| **Cowrie** | `/home/luxieum/tpotce/data/cowrie/log/cowrie.json` | SSH/Telnet brute force |
| **Dionaea** | `/home/luxieum/tpotce/data/dionaea/log/dionaea.json` | Malware, SMB exploits |
| **Suricata** | `/home/luxieum/tpotce/data/suricata/log/eve.json` | Network attacks, IDS alerts |
| **Wordpot** | `/home/luxieum/tpotce/data/wordpot/logs/wordpot.json` | WordPress attacks |
| **Elasticpot** | `/home/luxieum/tpotce/data/elasticpot/log/elasticpot.json` | Elasticsearch exploits |

Logs are:
1. Tailed in real-time via SSH
2. Parsed and normalized
3. Ingested into Mini-XDR event database
4. Analyzed by ML models
5. Triggering workflows when thresholds met

### Access T-Pot Web Interface

Access from your Mini-XDR host (172.16.110.0/24):

- **HTTP**: http://203.0.113.42:64297
- **HTTPS**: https://203.0.113.42:64294
- **Kibana**: http://203.0.113.42:64296
- **Elasticsearch**: http://203.0.113.42:64298

Login with your T-Pot admin credentials.

## Troubleshooting

### Issue: "Connection refused" to T-Pot

**Cause**: Firewall not allowing your IP

**Fix**:
1. Check your current IP: `curl ifconfig.me`
2. On T-Pot, verify firewall rules:
   ```bash
   ssh -p 64295 admin@203.0.113.42
   sudo ufw status numbered
   ```
3. Ensure your IP/subnet is allowed:
   ```bash
   sudo ufw allow from 172.16.110.0/24 to any port 64295
   ```

### Issue: "Permission denied" for UFW commands

**Cause**: Sudo requires password

**Fix**: Use password authentication method
```bash
# In backend/.env
TPOT_API_KEY=your_tpot_password
```

The connector will use: `echo 'password' | sudo -S ufw deny from <ip>`

### Issue: SSH connection works but no logs ingested

**Cause**: Log files may not exist yet or need sudo access

**Check**:
```bash
ssh -p 64295 admin@203.0.113.42
sudo ls -la /home/luxieum/tpotce/data/cowrie/log/
```

**Fix**: Wait for first attack, or generate test traffic

### Issue: Backend can't reach T-Pot from different network

**Cause**: Firewall rules are subnet-specific

**Solution**:
- VPN into same network as T-Pot
- Or update T-Pot firewall to allow your external IP
- Or use SSH port forwarding:
  ```bash
  ssh -L 64295:localhost:64295 -L 64297:localhost:64297 user@jump-host
  ```

## Testing

### Test 1: Manual SSH Connection

```bash
ssh -p 64295 admin@203.0.113.42
```

Should connect successfully with your credentials.

### Test 2: Test UFW Block (Manual)

```bash
ssh -p 64295 admin@203.0.113.42
echo 'your_password' | sudo -S ufw deny from 198.51.100.1
sudo ufw status | grep 198.51.100.1
echo 'your_password' | sudo -S ufw delete <rule_number>
```

### Test 3: API Endpoint Test

```bash
# Backend must be running
curl http://localhost:8000/api/tpot/status

# Should return:
{
  "connected": true,
  "host": "203.0.113.42",
  "monitoring_active": true,
  ...
}
```

### Test 4: Block IP via API

```bash
curl -X POST http://localhost:8000/api/tpot/block-ip \
  -H "Content-Type: application/json" \
  -d '{"ip_address": "198.51.100.1"}'

# Verify on T-Pot
ssh -p 64295 admin@203.0.113.42 "sudo ufw status | grep 198.51.100.1"

# Unblock
curl -X POST http://localhost:8000/api/tpot/unblock-ip \
  -H "Content-Type: application/json" \
  -d '{"ip_address": "198.51.100.1"}'
```

### Test 5: Full End-to-End Test

```bash
./scripts/tpot-management/verify-agent-ssh-actions.sh --full
```

This runs a complete test including:
1. Connection verification
2. SSH execution
3. IP blocking/unblocking
4. Simulated SSH brute force
5. Incident creation
6. Automated response

## Security Considerations

1. **Password in .env**: The `.env` file contains sensitive credentials. Ensure:
   - `.env` is in `.gitignore` (it is)
   - File permissions: `chmod 600 backend/.env`
   - Never commit to version control

2. **SSH Key Protection**: If using key auth:
   - `chmod 600 ~/.ssh/id_rsa`
   - Use passphrase-protected keys

3. **Firewall Rules**: T-Pot firewall should only allow:
   - Your Mini-XDR host IP/subnet
   - Not 0.0.0.0/0 (entire internet)

4. **Sudo Access**: Password-authenticated sudo is actually MORE secure because:
   - Each command requires password
   - Not passwordless sudo (which would be less secure)
   - Commands are logged

## Next Steps

1. ✅ **Setup Complete** - Run configuration script
2. ✅ **Verify** - Run verification script
3. ✅ **Start Backend** - Launch Mini-XDR
4. ✅ **Generate Attack** - Run demo or wait for real attacks
5. ✅ **Watch Magic** - See AI agents defend in real-time

## Demo Commands

```bash
# Setup
./scripts/tpot-management/setup-tpot-ssh-integration.sh

# Verify
./scripts/tpot-management/verify-agent-ssh-actions.sh

# Start backend
cd backend && python -m uvicorn app.main:app --reload

# (In another terminal) Start frontend
cd frontend && npm run dev

# (In another terminal) Run attack
./scripts/demo/demo-attack.sh

# Watch logs
tail -f backend/backend.log | grep -i "block\|ssh\|brute"
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/tpot/status` | GET | T-Pot connection status |
| `/api/tpot/connect` | POST | Manually connect to T-Pot |
| `/api/tpot/disconnect` | POST | Disconnect from T-Pot |
| `/api/tpot/block-ip` | POST | Block IP address |
| `/api/tpot/unblock-ip` | POST | Unblock IP address |
| `/api/tpot/active-blocks` | GET | List blocked IPs |
| `/api/tpot/containers` | GET | List honeypot containers |
| `/api/tpot/execute` | POST | Execute SSH command |
| `/api/tpot/monitoring-status` | GET | Log monitoring status |

## Support

If you encounter issues:

1. Check logs: `tail -f backend/backend.log`
2. Test SSH manually: `ssh -p 64295 admin@203.0.113.42`
3. Verify firewall: Check T-Pot UFW rules
4. Run verification: `./scripts/tpot-management/verify-agent-ssh-actions.sh`

---

**You're now ready to demonstrate real-time AI-powered threat defense with T-Pot! 🚀**
