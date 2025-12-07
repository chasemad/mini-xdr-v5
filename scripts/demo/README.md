# Mini-XDR Demo Scripts

This directory contains everything you need to deliver a professional demo of Mini-XDR to hiring managers.

## 📁 Files

- **`pre-demo-setup.sh`** - Automated setup script to prepare your system before recording
- **`demo-attack.sh`** - Coordinated attack simulation against T-Pot honeypot
- **`demo-cheatsheet.md`** - Quick reference guide for the demo flow
- **`/demo-video.plan.md`** (root) - Complete 3-4 minute demo script with narration

## 🚀 Quick Start

### 1. Run Pre-Demo Setup

This checks all services, verifies connectivity, and opens browser windows:

```bash
cd .
./scripts/demo/pre-demo-setup.sh
```

**What it does:**
- ✅ Checks required tools (docker, curl, jq, nmap, sshpass)
- ✅ Starts Docker services
- ✅ Verifies backend health
- ✅ Checks ML models loaded
- ✅ Verifies AI agents active
- ✅ Tests T-Pot connectivity
- ✅ Opens browser windows for demo

### 2. Review the Demo Script

Read the complete demo plan with narration, commands, and talking points:

```bash
less demo-video.plan.md
```

### 3. Run Attack Simulation

During your demo, run this to simulate real attacks against T-Pot:

```bash
export TPOT_IP="24.11.0.176"
./scripts/demo/demo-attack.sh
```

**Generates:**
- SSH brute force attacks (5 attempts)
- Web vulnerability scanning (10 paths)
- SQL injection testing (4 endpoints)
- Port reconnaissance (9 ports)

### 4. Reference the Cheat Sheet

Keep this open during recording for quick command reference:

```bash
open scripts/demo/demo-cheatsheet.md
```

## 🎯 Demo Flow Overview

### 3-4 Minute Structure

**Minute 1: Introduction + Live Attack**
- Show dashboard
- Run attack simulation
- Watch incidents appear in real-time

**Minute 2: AI Agent Analysis**
- Click on high-severity incident
- Show ML scores and threat classification
- Demonstrate AI agent analysis

**Minute 3: Natural Language Copilot**
- Open AI Copilot
- Type: "Block this IP for 24 hours and alert the security team"
- Show automatic workflow execution

**Minute 4: Workflow + Closing**
- Show visual workflow designer or API
- Summarize key capabilities
- End on dashboard showing system metrics

## 📋 Pre-Demo Checklist

Before you start recording:

- [ ] All Docker services running (`docker-compose ps`)
- [ ] Backend health check passes (`curl http://localhost:8000/health`)
- [ ] ML models loaded (5/5)
- [ ] AI agents active (12/12)
- [ ] T-Pot reachable on port 22
- [ ] Browser windows open (dashboard, incidents, copilot)
- [ ] Terminal ready with demo-attack.sh
- [ ] Cheat sheet open for reference

## 🛠️ Troubleshooting

### Services Not Starting

```bash
# Check Docker daemon
docker info

# View logs
docker-compose logs backend | tail -50

# Restart services
docker-compose down && docker-compose up -d
```

### T-Pot Not Reachable

```bash
# Test connectivity
ping 24.11.0.176
telnet 24.11.0.176 22

# Use manual event injection instead
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

### No Incidents Appearing

```bash
# Check T-Pot status
curl http://localhost:8000/api/tpot/status | jq

# Check backend logs
docker-compose logs backend | grep -i incident

# Verify database
curl http://localhost:8000/incidents | jq 'length'
```

### ML Models Not Loading

```bash
# Verify model files exist
ls -lh models/

# Check model status
curl http://localhost:8000/api/ml/status | jq

# Restart backend to reload
docker-compose restart backend
```

## 🎬 Recording Tips

1. **Use 1920x1080 resolution** for clear visibility
2. **Enable "Do Not Disturb"** to avoid notifications
3. **Close unnecessary applications** for clean screen
4. **Use split screen** - terminal on left, browser on right
5. **Speak clearly and confidently** - you built something amazing!
6. **Keep under 4 minutes** - hiring managers are busy
7. **Practice once or twice** to smooth out transitions
8. **Have backup plans** - manual event injection, pre-recorded terminal

## 📊 Key Metrics to Highlight

- **97.98%** ML detection accuracy
- **<2 seconds** autonomous response time
- **12** specialized AI agents
- **5** ensemble ML models
- **846,073+** cybersecurity events processed
- **<5%** false positive rate
- **10,000+** events/second scalability

## 🎯 Talking Points

### For Technical Hiring Managers

**Architecture:**
- Modern async Python with FastAPI
- Ensemble ML: PyTorch + XGBoost + scikit-learn
- 12 LangChain-powered AI agents
- Next.js 15 + React 19 frontend
- Real-time T-Pot honeypot integration

**Achievements:**
- Built from scratch in 6 months
- Full-stack: backend, ML, AI agents, frontend
- Enterprise-grade security (HMAC, JWT, audit trails)
- Production-ready (Docker, Kubernetes, monitoring)
- Explainable AI (SHAP/LIME for interpretability)

**Differentiators:**
- Natural language interface vs traditional SIEM
- Autonomous decision-making vs manual workflows
- Visual automation designer vs scripting
- Real-time honeypot integration
- 99% local operation (privacy-first)

## 📞 Support

If you encounter issues:

1. Check the troubleshooting section above
2. Review backend logs: `docker-compose logs backend`
3. Check the main documentation: `docs/`
4. Verify system health: `curl http://localhost:8000/health`

## 🎉 You've Got This!

You built an enterprise-grade XDR platform with AI agents, ML models, natural language control, and real-time threat detection. That's impressive engineering work!

**Show it off with confidence! 🚀**

---

*For the complete demo script with narration and detailed walkthrough, see `/demo-video.plan.md`*
