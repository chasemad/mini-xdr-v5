# 🚀 Mini-XDR Quick Start Guide

## ⚡ Start Everything (One Command)

```bash
cd /Users/chasemad/Desktop/mini-xdr
./scripts/start-all.sh
```

This automatically:
- Starts backend (port 8000)
- Starts frontend (port 3000)
- Loads all secrets from Azure
- Connects to T-Pot honeypot
- Activates all 6 agents

---

## 📊 Check System Status

```bash
# Full system test
./scripts/final-azure-test.sh

# Quick health check
curl http://localhost:8000/health

# View incidents
curl http://localhost:8000/incidents | jq .
```

---

## 🍯 T-Pot Access

**Web Interface:** https://74.235.242.205:64297  
**Username:** `tsec`  
**Password:** `minixdrtpot2025`

**SSH Access:**
```bash
ssh -i ~/.ssh/mini-xdr-tpot-azure azureuser@74.235.242.205 -p 64295
```

**View Logs:**
```bash
# Cowrie SSH honeypot
ssh -i ~/.ssh/mini-xdr-tpot-azure azureuser@74.235.242.205 -p 64295 "sudo docker logs -f cowrie"

# All containers
ssh -i ~/.ssh/mini-xdr-tpot-azure azureuser@74.235.242.205 -p 64295 "sudo docker ps"
```

---

## 🔐 Azure Secrets

**List all secrets:**
```bash
az keyvault secret list --vault-name minixdrchasemad --query "[].name" -o tsv
```

**Get a secret:**
```bash
az keyvault secret show --vault-name minixdrchasemad --name mini-xdr-api-key --query value -o tsv
```

**Sync secrets to .env:**
```bash
./scripts/sync-secrets-from-azure.sh minixdrchasemad
```

---

## 🤖 Test Agents

**Send test event:**
```bash
API_KEY=$(grep ^API_KEY backend/.env | cut -d'=' -f2)
curl -H "x-api-key: $API_KEY" -H "Content-Type: application/json" \
  -X POST http://localhost:8000/ingest/multi \
  -d '{"source_type":"cowrie","hostname":"test","events":[{"eventid":"cowrie.login.failed","src_ip":"192.168.1.100","timestamp":"'$(date -u +%Y-%m-%dT%H:%M:%SZ)'"}]}'
```

**Check agent credentials:**
```bash
cd backend && source venv/bin/activate
python3 -c "
from app.db import AsyncSessionLocal, init_db
from app.models import AgentCredential
from sqlalchemy import select
import asyncio
async def check():
    await init_db()
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(AgentCredential))
        print(f'{len(result.scalars().all())} agent credentials configured')
asyncio.run(check())
"
```

---

## 🎯 Simulate Attack

```bash
./test-honeypot-attack.sh
```

Then check for new incidents:
```bash
curl http://localhost:8000/incidents | jq '.[] | {id, source_ip, reason}'
```

---

## 🛠️ Common Tasks

### Restart Backend
```bash
pkill -f "uvicorn.*app.entrypoint"
cd backend && source venv/bin/activate
uvicorn app.entrypoint:app --reload
```

### View Logs
```bash
tail -f backend/logs/backend.log
```

### Check ML Models
```bash
API_KEY=$(grep ^API_KEY backend/.env | cut -d'=' -f2)
curl -H "x-api-key: $API_KEY" http://localhost:8000/api/ml/status | jq .
```

### Stop VM (Save $)
```bash
az vm deallocate --resource-group mini-xdr-rg --name mini-xdr-tpot
```

### Start VM
```bash
az vm start --resource-group mini-xdr-rg --name mini-xdr-tpot
```

---

## 📱 URLs

| Service | URL |
|---------|-----|
| Frontend Dashboard | http://localhost:3000 |
| Backend API | http://localhost:8000 |
| API Documentation | http://localhost:8000/docs |
| T-Pot Web | https://74.235.242.205:64297 |

---

## ✅ Verification Checklist

```bash
# 1. Backend running?
curl http://localhost:8000/health

# 2. Agents configured?
grep -i "AGENT_DEVICE_ID" backend/.env | wc -l  # Should be 6+

# 3. T-Pot reachable?
ssh -i ~/.ssh/mini-xdr-tpot-azure azureuser@74.235.242.205 -p 64295 "echo OK"

# 4. Azure connected?
az keyvault secret list --vault-name minixdrchasemad --query "[].name" | wc -l  # Should be 31

# 5. ML models loaded?
API_KEY=$(grep ^API_KEY backend/.env | cut -d'=' -f2)
curl -s -H "x-api-key: $API_KEY" http://localhost:8000/api/ml/status | jq '.metrics.models_trained'  # Should be 12+
```

---

## 🆘 Troubleshooting

### Backend not starting?
```bash
# Check logs
tail -50 backend/logs/backend.log

# Verify .env
ls -la backend/.env

# Re-sync secrets
./scripts/sync-secrets-from-azure.sh minixdrchasemad
```

### Can't connect to T-Pot?
```bash
# Check VM status
az vm show -d --resource-group mini-xdr-rg --name mini-xdr-tpot --query powerState

# Start if stopped
az vm start --resource-group mini-xdr-rg --name mini-xdr-tpot

# Test SSH
ssh -v -i ~/.ssh/mini-xdr-tpot-azure azureuser@74.235.242.205 -p 64295
```

### Agents not working?
```bash
# Regenerate credentials
./scripts/generate-agent-secrets-azure.sh minixdrchasemad

# Sync to .env
./scripts/sync-secrets-from-azure.sh minixdrchasemad

# Restart backend
pkill -f uvicorn && cd backend && source venv/bin/activate && uvicorn app.entrypoint:app --reload
```

---

## 📚 Documentation

- **This Guide:** Quick start commands
- **AZURE_DEPLOYMENT_SUCCESS.md:** Complete verification report
- **DEPLOYMENT_COMPLETE.md:** Initial setup summary
- **AZURE_STATUS_REPORT.md:** Technical details
- **docs/TPOT_AZURE_DEPLOYMENT_COMPLETE_GUIDE.md:** Detailed guide

---

## 🎉 All Systems Operational!

```
✅ Backend:  Healthy
✅ Agents:   7 configured
✅ T-Pot:    36 containers running
✅ Azure:    31 secrets stored
✅ ML:       12 models trained
✅ Ready:    YES! 🚀
```

**Start detecting threats now!** 🛡️


