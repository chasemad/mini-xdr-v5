# 🎬 Mini-XDR Demo Package - Complete Index

**Everything you need to deliver a professional hiring manager demo**

---

## 📦 What's Included

This demo package contains a complete set of scripts, documentation, and reference materials for showcasing Mini-XDR's capabilities in a 3-4 minute professional demo video.

---

## 📂 File Structure

```
mini-xdr/
├── demo-video.plan.md                          # Master demo script with narration
├── DEMO-PACKAGE-INDEX.md                       # This file - package overview
│
└── scripts/demo/
    ├── README.md                               # Demo scripts overview
    ├── DEMO-WALKTHROUGH.md                     # Complete step-by-step walkthrough
    ├── demo-cheatsheet.md                      # Detailed command reference
    ├── QUICK-REFERENCE.txt                     # 1-page printable cheat sheet
    │
    ├── pre-demo-setup.sh                       # Automated setup script ⚡
    ├── demo-attack.sh                          # T-Pot attack simulation ⚡
    ├── manual-event-injection.sh               # Backup event injection ⚡
    └── validate-demo-ready.sh                  # Readiness validation ⚡

    ⚡ = Executable scripts (chmod +x applied)
```

---

## 🚀 Quick Start (2 Commands!)

```bash
# 1. Run automated setup
./scripts/demo/pre-demo-setup.sh

# 2. Validate you're ready
./scripts/demo/validate-demo-ready.sh

# If all green: Start recording! 🎬
```

---

## 📚 Documentation Guide

### For First-Time Demo

**Start here:**
1. Read `DEMO-WALKTHROUGH.md` - Complete step-by-step guide
2. Run `pre-demo-setup.sh` - Automated setup
3. Review `QUICK-REFERENCE.txt` - Keep open during recording
4. Practice once with `demo-attack.sh`
5. Record your demo!

### For Quick Reference

**During recording:**
- Keep `QUICK-REFERENCE.txt` open on second monitor
- Or glance at `demo-cheatsheet.md` for commands
- Main script narration in `demo-video.plan.md`

### For Troubleshooting

**If something goes wrong:**
1. Check `scripts/demo/README.md` - Troubleshooting section
2. Run `validate-demo-ready.sh` - Identify issues
3. Use `manual-event-injection.sh` - Backup if T-Pot down

---

## 🎯 Demo Content Overview

### What the Demo Showcases

**Minute 1: Architecture + Live Attack**
- Dashboard overview
- 12 AI agents, 5 ML models
- Real-time T-Pot attack simulation
- Incidents appearing in real-time

**Minute 2: AI Agent Analysis**
- Click high-severity incident
- ML ensemble scores (97.98% accuracy)
- 12 AI agents analyzing threat
- Threat intelligence enrichment

**Minute 3: Natural Language Copilot**
- Chat interface interaction
- "Block this IP for 24 hours"
- Automatic workflow execution
- Real containment action

**Minute 4: Workflow + Closing**
- Visual workflow automation
- System metrics summary
- Tech stack highlight
- Call to action

---

## 📋 Pre-Demo Checklist

### System Requirements

- [ ] Docker 20.10+ running
- [ ] 16GB RAM available
- [ ] All services started
- [ ] ML models loaded (5/5)
- [ ] AI agents active (12/12)
- [ ] T-Pot reachable OR manual injection ready

### Recording Requirements

- [ ] Screen recording software ready
- [ ] Microphone working
- [ ] 1920x1080 resolution
- [ ] Do Not Disturb enabled
- [ ] Browser tabs open
- [ ] Terminal ready
- [ ] Backup commands ready

### Run These Commands

```bash
# Automated setup
./scripts/demo/pre-demo-setup.sh

# Validation
./scripts/demo/validate-demo-ready.sh

# Should see: ✨ DEMO READY! ✨
```

---

## 🎬 Demo Scripts

### 1. Pre-Demo Setup (`pre-demo-setup.sh`)

**What it does:**
- Checks all required tools (docker, curl, jq, nmap)
- Starts Docker services
- Verifies backend health
- Checks ML models loaded
- Verifies AI agents active
- Tests T-Pot connectivity
- Opens browser windows

**Run it:**
```bash
./scripts/demo/pre-demo-setup.sh
```

**Expected output:**
```
✓ docker installed
✓ Backend API is healthy
✓ ML models loaded: 5
✓ AI agents active: 12
✓ T-Pot is reachable
```

---

### 2. Demo Attack Simulation (`demo-attack.sh`)

**What it does:**
- SSH brute force (5 attempts)
- Web vulnerability scanning (10 paths)
- SQL injection testing (4 endpoints)
- Port reconnaissance (9 ports)

**Run it during demo:**
```bash
export TPOT_IP="24.11.0.176"
./scripts/demo/demo-attack.sh
```

**Expected results:**
- Multiple incidents created in Mini-XDR
- High ML risk scores (80-100)
- AI agent analysis triggered
- Source IP flagged

---

### 3. Manual Event Injection (`manual-event-injection.sh`)

**When to use:**
- T-Pot is not reachable
- Network issues during demo
- As backup plan

**What it does:**
- Injects 5 pre-crafted attack events
- Directly via Mini-XDR API
- No T-Pot connection needed

**Run it:**
```bash
./scripts/demo/manual-event-injection.sh
```

**Events injected:**
- SSH brute force (5 attempts)
- SQL injection attempt
- Port scanning activity
- Malware download attempt
- Path traversal attack

---

### 4. Readiness Validation (`validate-demo-ready.sh`)

**What it checks:**
- System tools installed
- Docker services running
- API health
- ML models loaded
- AI agents active
- T-Pot connectivity
- Demo files exist

**Run it:**
```bash
./scripts/demo/validate-demo-ready.sh
```

**Possible outcomes:**
- ✨ **DEMO READY!** - All checks passed
- ⚠️ **DEMO READY (WITH WARNINGS)** - Can proceed with backup plans
- ❌ **NOT READY FOR DEMO** - Fix issues first

---

## 📖 Documentation Files

### 1. Master Demo Script (`demo-video.plan.md`)

**Complete demo script with:**
- Full narration for each phase
- All commands to run
- Screen action instructions
- Timing for each phase
- Key talking points
- Technical architecture details
- Post-demo Q&A prep

**Use this for:** Full detailed reference

---

### 2. Complete Walkthrough (`scripts/demo/DEMO-WALKTHROUGH.md`)

**Step-by-step guide with:**
- Pre-recording setup
- Phase-by-phase instructions
- Exact narration scripts
- Troubleshooting solutions
- Post-demo discussion points
- Do's and don'ts

**Use this for:** First-time demo preparation

---

### 3. Demo Cheat Sheet (`scripts/demo/demo-cheatsheet.md`)

**Quick reference with:**
- 3-4 minute demo flow
- All commands in one place
- Important URLs
- Quick troubleshooting
- API examples
- Key metrics

**Use this for:** Quick lookup during demo

---

### 4. Quick Reference Card (`scripts/demo/QUICK-REFERENCE.txt`)

**1-page printable reference:**
- Minute-by-minute breakdown
- Essential commands only
- Key metrics to mention
- Troubleshooting one-liners
- Post-demo Q&A

**Use this for:** Keep visible during recording

---

### 5. Demo Scripts README (`scripts/demo/README.md`)

**Overview documentation:**
- File descriptions
- Quick start guide
- Pre-demo checklist
- Troubleshooting guide
- Recording tips

**Use this for:** Understanding the demo package

---

## 🎯 Key Metrics to Highlight

### Performance Metrics
- **97.98%** ML detection accuracy
- **<2 seconds** autonomous response time
- **<5%** false positive rate
- **10,000+** events/second scalability

### System Capabilities
- **12** specialized AI agents
- **5** ensemble ML models
- **846,073+** events processed
- **99%** local operation (no cloud)

### Technical Stack
- **Backend:** FastAPI, PostgreSQL, Redis, 70+ Python deps
- **ML:** PyTorch, XGBoost, scikit-learn, SHAP/LIME
- **AI:** 12 LangChain agents, OpenAI integration
- **Frontend:** Next.js 15, React 19, Three.js

---

## 🎬 Recording Tips

### Setup
1. Use 1920x1080 resolution
2. Enable "Do Not Disturb"
3. Close unnecessary apps
4. Split screen: browser left, terminal right
5. Keep QUICK-REFERENCE.txt visible

### During Recording
1. Speak clearly and confidently
2. Show enthusiasm for your work
3. Keep narration moving (don't pause too long)
4. Point out key features as you show them
5. Stay under 4 minutes

### If Something Goes Wrong
1. Don't panic or apologize
2. Use backup commands
3. Acknowledge and continue
4. Or pause recording, fix, and resume

---

## 🆘 Troubleshooting Quick Guide

### T-Pot Not Working?
```bash
# Use manual event injection
./scripts/demo/manual-event-injection.sh
```

### No Incidents Appearing?
```bash
# Check logs
docker-compose logs backend | tail -50

# Restart backend
docker-compose restart backend

# Verify incidents
curl http://localhost:8000/incidents | jq 'length'
```

### ML Models Not Loaded?
```bash
# Check model files
ls -lh models/

# Restart backend
docker-compose restart backend

# Verify status
curl http://localhost:8000/api/ml/status | jq
```

### Services Not Running?
```bash
# Check status
docker-compose ps

# Restart all
docker-compose restart

# View logs
docker-compose logs -f
```

---

## 🎯 Success Criteria

### Your demo is successful if you show:

✅ **Real-time Detection**
- Live attack simulation
- Incidents created in real-time
- ML scoring in action

✅ **AI Agent Orchestration**
- 12 agents analyzing threats
- Autonomous decision-making
- Confidence scores and reasoning

✅ **Natural Language Control**
- Copilot interaction
- Plain English commands
- Automatic workflow execution

✅ **Technical Competence**
- Modern tech stack
- Scalable architecture
- Production-ready features

✅ **Business Value**
- Sub-2-second response
- 97.98% accuracy
- Reduced analyst workload

---

## 📞 Need Help?

### Resources

1. **Complete Walkthrough:** `scripts/demo/DEMO-WALKTHROUGH.md`
2. **Troubleshooting:** `scripts/demo/README.md`
3. **Quick Commands:** `scripts/demo/QUICK-REFERENCE.txt`
4. **Validation:** Run `./scripts/demo/validate-demo-ready.sh`

### Common Issues

- **T-Pot connection fails** → Use manual event injection
- **Copilot slow** → Use terminal commands instead
- **Models not loading** → Restart backend, wait 30 seconds
- **No incidents** → Check backend logs, verify ingestion

---

## 🎉 You're Ready!

You've built an enterprise-grade XDR platform that demonstrates:
- Full-stack development skills
- ML/AI engineering expertise
- System architecture capabilities
- Security engineering knowledge
- Product development experience

**Show it with confidence! 🚀**

---

## 📝 Final Steps

1. ✅ Read `DEMO-WALKTHROUGH.md` thoroughly
2. ✅ Run `./scripts/demo/pre-demo-setup.sh`
3. ✅ Run `./scripts/demo/validate-demo-ready.sh`
4. ✅ Practice once with attack simulation
5. ✅ Keep `QUICK-REFERENCE.txt` visible
6. ✅ Start recording!

---

**Good luck with your hiring manager! You've got this! 🎬✨**

*For detailed documentation, see `demo-video.plan.md` and `scripts/demo/DEMO-WALKTHROUGH.md`*
