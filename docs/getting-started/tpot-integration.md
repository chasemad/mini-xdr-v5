# T-Pot Honeypot Integration Setup Guide

This guide will help you connect your Mini-XDR system to the T-Pot honeypot infrastructure for real-time attack monitoring and automated response.

## Overview

Your T-Pot honeypot is deployed at **24.11.0.176** and provides:
- **Multiple honeypot types**: Cowrie (SSH/Telnet), Dionaea (Malware), Suricata (IDS), and more
- **Real-time attack telemetry**: JSON logs from all honeypot sensors
- **Elasticsearch integration**: Aggregated attack data for analysis
- **Defensive actions**: Firewall blocking, container management

## Connection Information

### T-Pot Server Details
```
Host: 24.11.0.176
SSH Port: 64295
SSH User: luxieum
Allowed IP: 172.16.110.1 (only this IP can connect)
```

### Authentication
```
SSH Password: demo-tpot-api-key
Web UI URL: https://24.11.0.176:64297
Web UI Username: admin
Web UI Password: TpotSecure2024!
```

### Internal Services (via SSH tunnel)
```
Elasticsearch: localhost:64298 (tunneled)
Kibana: localhost:64296 (tunneled)
Attack Map: localhost:64299 (tunneled)
```

## Setup Instructions

### Step 1: Configure Environment Variables

1. Copy the T-Pot configuration template:
```bash
cd ./backend
cp .env.tpot .env
```

2. Update `.env` with your credentials (already pre-filled):
```bash
TPOT_HOST=24.11.0.176
TPOT_SSH_PORT=64295
TPOT_API_KEY=demo-tpot-api-key
HONEYPOT_USER=luxieum
```

### Step 2: Install Required Dependencies

```bash
cd ./backend
source venv/bin/activate  # or activate your virtual environment
pip install asyncssh==2.14.2
```

### Step 3: Test Connection

Test SSH connectivity to T-Pot:
```bash
ssh -p 64295 luxieum@24.11.0.176
# Use password: demo-tpot-api-key
```

If successful, you should see the T-Pot shell prompt.

### Step 4: Verify Log Access

Once connected via SSH, verify you can read honeypot logs:
```bash
sudo tail -f /home/luxieum/tpotce/data/cowrie/log/cowrie.json
sudo tail -f /home/luxieum/tpotce/data/suricata/log/eve.json
```

You should see JSON-formatted attack logs streaming in real-time.

### Step 5: Start Mini-XDR Backend

Start the backend which will automatically connect to T-Pot:
```bash
cd ./backend
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

You should see in the logs:
```
INFO:     T-Pot honeypot monitoring...
INFO:     ✅ Successfully connected to T-Pot at 24.11.0.176
INFO:     ✅ Elasticsearch tunnel: localhost:64298
INFO:     ✅ Kibana tunnel: localhost:64296
INFO:     ✅ Started monitoring: cowrie
INFO:     ✅ Started monitoring: suricata
INFO:     ✅ T-Pot monitoring initialized successfully
```

### Step 6: Access T-Pot Dashboard in UI

1. Start the frontend:
```bash
cd ./frontend
npm run dev
```

2. Open your browser to: http://localhost:3000

3. Log in and navigate to **Honeypot** in the sidebar

4. You should see:
   - Connection status: **Connected**
   - Active honeypots monitoring
   - Recent attacks from T-Pot
   - Container status
   - Blocked IPs

## Architecture

### Data Flow

```
[Attacker] → [T-Pot Honeypots] → [Logs] → [Mini-XDR Connector] → [Database] → [ML Detection] → [AI Agents] → [Response Actions]
```

1. **Attackers** hit T-Pot honeypots (SSH, HTTP, etc.)
2. **Honeypots** log all attack activity to JSON files
3. **Mini-XDR Connector** tails log files via SSH and ingests events
4. **ML Models** analyze attacks for patterns and anomalies
5. **AI Agents** decide on response actions
6. **Response Actions** executed on T-Pot (block IPs, stop containers, etc.)

### Monitored Honeypot Types

| Honeypot | Service | Log Path |
|----------|---------|----------|
| Cowrie | SSH/Telnet | `/home/luxieum/tpotce/data/cowrie/log/cowrie.json` |
| Dionaea | Malware | `/home/luxieum/tpotce/data/dionaea/log/dionaea.json` |
| Suricata | Network IDS | `/home/luxieum/tpotce/data/suricata/log/eve.json` |
| WordPot | WordPress | `/home/luxieum/tpotce/data/wordpot/logs/wordpot.json` |
| ElasticPot | Elasticsearch | `/home/luxieum/tpotce/data/elasticpot/log/elasticpot.json` |
| RedisHoneypot | Redis | `/home/luxieum/tpotce/data/redishoneypot/log/redishoneypot.log` |
| Mailoney | SMTP | `/home/luxieum/tpotce/data/mailoney/log/commands.log` |
| SentryPeer | VoIP | `/home/luxieum/tpotce/data/sentrypeer/log/sentrypeer.json` |

## Available Actions

### Via API/UI

1. **View Status**: GET `/api/tpot/status`
2. **Recent Attacks**: GET `/api/tpot/attacks/recent?minutes=5`
3. **Block IP**: POST `/api/tpot/firewall/block`
4. **Unblock IP**: POST `/api/tpot/firewall/unblock`
5. **Stop Container**: POST `/api/tpot/containers/stop`
6. **Start Container**: POST `/api/tpot/containers/start`
7. **Query Elasticsearch**: POST `/api/tpot/elasticsearch/query`

### Via SSH (Manual)

```bash
# View firewall blocks
sudo ufw status

# Block an IP
sudo ufw deny from 1.2.3.4

# View containers
docker ps

# Stop/start honeypot
docker stop cowrie
docker start cowrie

# View logs
sudo tail -f /home/luxieum/tpotce/data/cowrie/log/cowrie.json
```

## Testing the Integration

### 1. Generate Test Attacks

From your Mac (IP must be 172.16.110.1), generate attacks:

```bash
# SSH brute force
hydra -l root -P /path/to/passwords.txt ssh://24.11.0.176

# Port scan
nmap -p- 24.11.0.176

# Web attack
nikto -h http://24.11.0.176
```

### 2. Verify Detection

- Check the **Honeypot** dashboard in UI
- Look for new incidents in **Incidents** page
- Verify ML models scored the attacks
- Check if AI agents triggered responses

### 3. Verify Actions

```bash
# SSH into T-Pot
ssh -p 64295 luxieum@24.11.0.176

# Check if IPs were blocked by Mini-XDR
sudo ufw status | grep DENY

# View recent firewall changes
sudo tail -20 /var/log/ufw.log
```

## Troubleshooting

### Connection Issues

**Problem**: Cannot connect to T-Pot
```
ERROR: Connection timeout to T-Pot at 24.11.0.176:64295
```

**Solution**:
1. Verify your IP is 172.16.110.1:
   ```bash
   curl ifconfig.me
   ```
2. Test SSH manually:
   ```bash
   ssh -p 64295 -v luxieum@24.11.0.176
   ```
3. Check firewall allows your IP on T-Pot

### Authentication Failures

**Problem**: Permission denied
```
ERROR: Failed to connect to T-Pot: Permission denied
```

**Solution**:
1. Verify password in `.env`: `TPOT_API_KEY=demo-tpot-api-key`
2. Try SSH manually to test credentials
3. Check if SSH key authentication is required

### No Logs Appearing

**Problem**: Connected but no attacks showing
```
INFO: Connected but no events ingested
```

**Solution**:
1. Verify log files exist:
   ```bash
   ssh luxieum@24.11.0.176 -p 64295
   ls -la /home/luxieum/tpotce/data/cowrie/log/
   ```
2. Check file permissions:
   ```bash
   sudo cat /home/luxieum/tpotce/data/cowrie/log/cowrie.json
   ```
3. Generate test traffic to honeypots

### Elasticsearch Tunnel Issues

**Problem**: Cannot query Elasticsearch
```
ERROR: Elasticsearch tunnel not established
```

**Solution**:
1. Verify SSH tunnels in logs:
   ```
   INFO: ✅ Elasticsearch tunnel: localhost:64298
   ```
2. Test tunnel manually:
   ```bash
   curl http://localhost:64298/_cluster/health
   ```
3. Restart backend to re-establish tunnels

## Security Considerations

### Firewall Rules
- T-Pot only accepts connections from **172.16.110.1**
- All other IPs are blocked by default
- Mini-XDR can add additional blocks via UFW

### Authentication
- SSH password stored in environment variable (encrypted in production)
- Web UI uses basic auth (admin/TpotSecure2024!)
- API requests require JWT token

### Data Privacy
- Attack logs contain attacker IPs, credentials, commands
- Malware samples stored in `/home/luxieum/tpotce/data/dionaea/binaries/`
- All data stays within your infrastructure

## Performance

### Expected Load
- **Log ingestion**: ~100-1000 events/hour (depends on attack volume)
- **SSH connections**: 1 persistent connection + tunnels
- **CPU impact**: Minimal (log tailing is lightweight)
- **Network**: ~1-10 MB/hour for log streaming

### Scaling
- Can monitor multiple T-Pot instances by configuring additional connectors
- Elasticsearch provides buffering for high-volume attacks
- ML detection runs asynchronously to avoid blocking ingestion

## Next Steps

1. ✅ Connect to T-Pot and verify monitoring
2. ✅ Generate test attacks to validate detection
3. ✅ Review incidents in UI and verify ML scoring
4. ✅ Test defensive actions (block IP, stop container)
5. ✅ Configure alert thresholds and response policies
6. ✅ Set up automated response workflows
7. ✅ Monitor effectiveness metrics in Analytics

## Support

For issues or questions:
1. Check logs: `tail -f ./backend/backend.log`
2. Review T-Pot logs: `ssh luxieum@24.11.0.176 -p 64295 "sudo journalctl -u tpot -f"`
3. Test connectivity: Use the **T-Pot Status** endpoint in Swagger docs at http://localhost:8000/docs

---

**Status**: Ready for Production Testing ✅

Your T-Pot integration is fully configured and ready to use. Start the backend and navigate to the Honeypot dashboard to begin monitoring real-world attacks!
