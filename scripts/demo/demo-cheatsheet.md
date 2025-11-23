# Mini-XDR Demo Cheat Sheet

**Quick reference for 3-4 minute hiring manager demo**

---

## 🚀 Quick Start (Before Recording)

```bash
# 1. Start services
cd /Users/chasemad/Desktop/mini-xdr
docker-compose up -d

# 2. Run pre-demo setup
./scripts/demo/pre-demo-setup.sh

# 3. Set T-Pot IP
export TPOT_IP="24.11.0.176"
```

---

## 📋 Demo Flow (3-4 Minutes)

### Minute 1: Intro + Live Attack
**Say:** "This is Mini-XDR - 12 AI agents, 97.98% ML accuracy, natural language control, processing real T-Pot attacks in under 2 seconds."

**Show:** Dashboard at http://localhost:3000

**Do:** Run attack in terminal
```bash
./scripts/demo/demo-attack.sh
```

**Point out:** Incidents appearing in real-time

---

### Minute 2: AI Agent Analysis
**Say:** "12 AI agents are analyzing - Attribution profiling the attacker, Containment recommending actions, Forensics collecting evidence."

**Do:** Click on high-severity incident

**Show:**
- ML scores (0-100)
- Threat classification
- Attack timeline
- AI analysis

**Quick API demo:**
```bash
INCIDENT_ID=1
curl -X POST http://localhost:8000/api/incidents/$INCIDENT_ID/ai-analysis | jq
```

---

### Minute 3: Copilot + Response
**Say:** "Natural language control - I'll block the attacker with plain English."

**Do:** Open copilot (http://localhost:3000/agents)

**Type:**
```
"Block this IP for 24 hours and alert the security team"
```

**Show:**
- AI confirmation dialog
- Click "Approve & Execute"
- Workflow executes
- Results displayed

**Or via terminal:**
```bash
curl -X POST http://localhost:8000/incidents/$INCIDENT_ID/contain \
  -H "Content-Type: application/json" \
  -d '{"duration_hours": 24, "reason": "Demo containment"}'
```

---

### Minute 4: Workflow + Closing
**Say:** "Visual workflow automation for complex playbooks."

**Show:** Workflow designer (if available) or:
```bash
curl http://localhost:8000/api/response/workflows | jq '.[0]'
```

**Closing:** "Enterprise XDR with 12 AI agents, 5 ML models, natural language, visual automation, T-Pot integration - FastAPI, React 19, PyTorch, LangChain. 846K+ events processed, 97.98% accuracy, sub-2-second response."

---

## 🎯 Key Talking Points

### Architecture
- **Backend:** FastAPI, PostgreSQL, Redis, 70+ Python deps
- **ML:** PyTorch, XGBoost, SHAP/LIME explainability
- **AI:** 12 LangChain agents with autonomous decisions
- **Frontend:** Next.js 15, React 19, Three.js 3D viz

### Metrics
- **97.98%** detection accuracy
- **<2 seconds** response time
- **<5%** false positive rate
- **12** specialized AI agents
- **846,073+** events processed

### Differentiators
1. 12 AI agents vs traditional alerting
2. Natural language interface
3. Visual workflow designer
4. Explainable AI (SHAP/LIME)
5. Real-time T-Pot integration
6. 99% local (no cloud dependency)

---

## 📍 Important URLs

| Page | URL |
|------|-----|
| Dashboard | http://localhost:3000 |
| Incidents | http://localhost:3000/incidents |
| AI Copilot | http://localhost:3000/agents |
| Visualizations | http://localhost:3000/visualizations |
| API Docs | http://localhost:8000/docs |
| Health Check | http://localhost:8000/health |

---

## 🛠️ Quick Commands Reference

### System Status
```bash
# Services
docker-compose ps

# Backend health
curl http://localhost:8000/health | jq

# ML models
curl http://localhost:8000/api/ml/status | jq

# AI agents
curl http://localhost:8000/api/agents/status | jq

# T-Pot status
curl http://localhost:8000/api/tpot/status | jq
```

### Attack Simulation
```bash
# Full attack suite
./scripts/demo/demo-attack.sh

# Quick SSH brute force
for i in {1..5}; do
  sshpass -p "admin123" ssh -o ConnectTimeout=2 root@24.11.0.176 &
done

# Web scanning
curl http://24.11.0.176/admin
curl http://24.11.0.176/wp-admin
curl "http://24.11.0.176/login?user=admin' OR 1=1--"
```

### Incident Management
```bash
# List incidents
curl http://localhost:8000/incidents | jq

# Get incident details
curl http://localhost:8000/incidents/1 | jq

# AI analysis
curl -X POST http://localhost:8000/api/incidents/1/ai-analysis | jq

# Block IP
curl -X POST http://localhost:8000/incidents/1/contain \
  -H "Content-Type: application/json" \
  -d '{"duration_hours": 24}'

# Check containment
curl http://localhost:8000/incidents/1/block-status | jq
```

### Workflows
```bash
# List workflows
curl http://localhost:8000/api/response/workflows | jq

# Execute workflow
curl -X POST http://localhost:8000/api/response/workflows/execute \
  -H "Content-Type: application/json" \
  -d '{"workflow_id": 1, "incident_id": 1}'
```

---

## 🚨 Troubleshooting

### T-Pot Not Connected?
```bash
# Use manual event injection
curl -X POST http://localhost:8000/ingest/multi \
  -H "Content-Type: application/json" \
  -d '{
    "source_type": "cowrie",
    "hostname": "demo-honeypot",
    "events": [{
      "eventid": "cowrie.login.failed",
      "src_ip": "203.0.113.100",
      "dst_port": 22,
      "username": "root",
      "password": "admin123",
      "timestamp": "'$(date -u +%Y-%m-%dT%H:%M:%S)'Z"
    }]
  }'
```

### No Incidents Showing?
```bash
# Check backend logs
docker-compose logs backend | tail -50

# Check incident count
curl http://localhost:8000/incidents | jq 'length'

# Restart backend
docker-compose restart backend
```

### ML Models Not Loaded?
```bash
# Check model files
ls -lh models/

# Check status
curl http://localhost:8000/api/ml/status | jq '.models_loaded'
```

---

## 💡 Pro Tips

1. **Pre-record the attack** if network is unreliable
2. **Have backup manual event injection** ready
3. **Keep terminal and browser side-by-side** for split screen
4. **Practice the copilot interaction** - it's the wow factor
5. **End on the dashboard** showing metrics and status
6. **Keep it under 4 minutes** - hiring managers are busy

---

## 🎬 Alternative 3-Minute Speed Run

**If time is very limited:**

1. **Minute 1:** Dashboard + attack simulation + incidents appearing
2. **Minute 2:** Click incident + AI analysis + copilot "Block this IP for 24 hours"
3. **Minute 3:** Show containment result + quick workflow API demo + closing

**Skip:** Detailed workflow designer, extensive API demos, visualization deep-dive

**Focus:** AI agents, natural language, real-time response

---

## 📝 Post-Demo Q&A Prep

**Expected Questions:**

1. **"How long did this take to build?"**
   - 6 months of focused development

2. **"What's the tech stack?"**
   - Backend: FastAPI, PostgreSQL, Redis
   - ML: PyTorch, XGBoost, scikit-learn
   - AI: LangChain, OpenAI, 12 custom agents
   - Frontend: Next.js 15, React 19, Three.js

3. **"Can it scale?"**
   - Yes! Kubernetes deployment handles 10K+ events/sec
   - Distributed architecture with Kafka/Redis

4. **"What about false positives?"**
   - <5% with continuous learning
   - Rollback Agent learns from mistakes
   - Confidence scoring on all decisions

5. **"Cloud or on-prem?"**
   - 99% local operation
   - Optional cloud integrations (AWS, Azure)
   - Complete data privacy

---

**Good luck with your demo! 🚀**
