# T-Pot + Mini-XDR Quick Start Guide

## 🎯 Your T-Pot Configuration

```
T-Pot IP:        203.0.113.42
SSH Port:        64295
Web Interface:   http://203.0.113.42:64297
HTTPS Interface: https://203.0.113.42:64294
Allowed Subnet:  172.16.110.0/24
```

## 🚀 Quick Setup (5 Minutes)

### One Command Setup

```bash
cd /Users/chasemad/Desktop/mini-xdr
./scripts/SETUP_TPOT_DEMO.sh
```

This interactive wizard does everything:
- ✅ Configure T-Pot SSH connection
- ✅ Test SSH and defensive capabilities
- ✅ Set up automated workflows
- ✅ Start Mini-XDR backend
- ✅ Provide demo commands

### Manual Commands

If you prefer step-by-step:

```bash
# 1. Configure SSH
./scripts/tpot-management/setup-tpot-ssh-integration.sh

# 2. Start Backend
cd backend
python -m uvicorn app.main:app --reload

# 3. Verify Everything Works
./scripts/tpot-management/verify-agent-ssh-actions.sh

# 4. Run Demo Attack
./scripts/demo/demo-attack.sh
```

## 📋 What You'll Need

When running setup, have ready:
- ✓ T-Pot IP: `203.0.113.42`
- ✓ SSH Port: `64295`
- ✓ Username: `admin` (or your T-Pot admin user)
- ✓ Password: Your T-Pot SSH password

## 🎬 Demo Attack Options

### Option 1: Automated Demo

```bash
./scripts/demo/demo-attack.sh
```

### Option 2: Manual SSH Brute Force

```bash
# From your Mac or another machine
for i in {1..10}; do
    ssh -p 64295 admin@203.0.113.42 "wrong_$i" 2>/dev/null
    echo "Attack attempt $i"
done
```

### Option 3: Wait for Real Attacks

Just wait! T-Pot is constantly scanned by the internet.

## 👀 What to Watch

### 1. T-Pot Attack Map
```
http://203.0.113.42:64297
```
See live attacks from around the world

### 2. Mini-XDR Dashboard
```
http://localhost:3000
```
Watch AI agents respond in real-time

### 3. Backend Logs
```bash
tail -f backend/backend.log | grep -i "block\|ssh\|brute"
```

Look for:
```
INFO - SSH brute-force check: 192.0.2.1 has 6 failures
INFO - Using T-Pot connector to block 192.0.2.1
INFO - ✅ Successfully blocked 192.0.2.1 on T-Pot firewall
```

### 4. Firewall Blocks on T-Pot
```bash
ssh -p 64295 admin@203.0.113.42 "sudo ufw status | grep DENY"
```

See IPs blocked by AI agents

## 🎯 Expected Workflow

```
1. Attacker hits T-Pot SSH (Cowrie honeypot)
   ↓
2. After 6 failed logins in 60 seconds
   ↓
3. Mini-XDR creates incident
   ↓
4. AI agents analyze threat
   ↓
5. Containment agent SSHs into T-Pot
   ↓
6. Executes: sudo ufw deny from <attacker-ip>
   ↓
7. Attacker blocked!
```

All happening automatically in 2-5 seconds!

## ✅ Verification Checklist

- [ ] Can SSH into T-Pot manually: `ssh -p 64295 admin@203.0.113.42`
- [ ] Backend is running: `curl http://localhost:8000/health`
- [ ] T-Pot connected: `curl http://localhost:8000/api/tpot/status | jq`
- [ ] Can block test IP via API
- [ ] SSH brute force workflow exists

Run full verification:
```bash
./scripts/tpot-management/verify-agent-ssh-actions.sh
```

## 🐛 Troubleshooting

### "Connection refused" to T-Pot

**Problem**: Can't SSH to T-Pot

**Fix**: Ensure firewall allows your IP
```bash
# On T-Pot
sudo ufw allow from YOUR_IP to any port 64295
```

### "Permission denied" for UFW

**Problem**: Sudo commands failing

**Fix**: Ensure password in backend/.env
```bash
# backend/.env
TPOT_API_KEY=your_tpot_password
```

### No incidents created

**Problem**: Attacks not triggering responses

**Debug**:
```bash
# Check T-Pot connection
curl http://localhost:8000/api/tpot/status

# Check events ingested
curl http://localhost:8000/api/events?limit=10

# Check backend logs
tail -f backend/backend.log
```

### Backend not running

**Fix**:
```bash
cd backend
python -m uvicorn app.main:app --reload
```

## 📚 Documentation

- **Full Setup Guide**: [`TPOT_SSH_SETUP_GUIDE.md`](./TPOT_SSH_SETUP_GUIDE.md)
- **Demo Walkthrough**: [`docs/demo/tpot-ssh-demo.md`](./docs/demo/tpot-ssh-demo.md)
- **Main README**: [`README.md`](./README.md)

## 🔥 Quick Demo Script

Perfect for showing to others:

```bash
# Terminal 1: Backend logs
cd backend && tail -f backend.log | grep -i "block\|brute"

# Terminal 2: Run attack
for i in {1..10}; do
    ssh -p 64295 admin@203.0.113.42 "wrong_$i" 2>/dev/null
    echo "Attack $i"
    sleep 1
done

# Watch Terminal 1 for automatic blocking!

# Verify block
ssh -p 64295 admin@203.0.113.42 "sudo ufw status | grep DENY | tail -1"
```

## 🎓 What This Demonstrates

✨ **Real-time Threat Detection**
- Monitoring live honeypot data via SSH

✨ **AI-Powered Analysis**
- Multiple agents collaborating on threat assessment

✨ **Automated Response**
- No human in the loop - agents take action

✨ **SSH-Based Control**
- Agents execute defensive commands on remote systems

✨ **Complete Audit Trail**
- Every action logged with full context

✨ **Production-Ready**
- Password-authenticated sudo for security
- Error handling and retry logic
- Scalable architecture

## 🚀 After the Demo

Once you've verified everything works:

1. **Leave it running** - Capture real attacks
2. **Monitor dashboards** - Watch live threats
3. **Review incidents** - Learn attack patterns
4. **Tune workflows** - Adjust thresholds as needed
5. **Add honeypots** - Scale to multiple T-Pot instances

## 🎉 You're Ready!

Your Mini-XDR AI agents are now configured to defend your T-Pot honeypot!

**Next step**: Run the setup wizard:
```bash
./scripts/SETUP_TPOT_DEMO.sh
```

---

**Questions? Issues?**
- Check logs: `backend/backend.log`
- Run verification: `./scripts/tpot-management/verify-agent-ssh-actions.sh`
- Review full guide: `TPOT_SSH_SETUP_GUIDE.md`

**Happy defending! 🛡️**
