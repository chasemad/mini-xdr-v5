# 🔒 Secure T-Pot Honeypot - Ready for Testing

## Current Status: **FULLY SECURED & CONFIGURED**

✅ **T-Pot Instance**: Stopped (Maximum security)  
✅ **Security Groups**: Locked down to your IP only  
✅ **Mini-XDR Integration**: Configured with secure API key  
✅ **Log Forwarding**: Ready to deploy  
✅ **Management Scripts**: Created and ready  

---

## 🚀 When You're Ready to Test

### Step 1: Start Mini-XDR Backend
```bash
cd backend && python -m app.main
```

### Step 2: Start T-Pot Honeypot (Secure)
```bash
./start-secure-tpot.sh
```
- This will start T-Pot with all security restrictions in place
- Only your IP can access management ports
- All honeypot ports remain blocked from public internet

### Step 3: Get Your Local IP for Log Forwarding
```bash
# Get your local network IP (for log forwarding)
ifconfig | grep "inet " | grep -v 127.0.0.1 | awk '{print $2}' | head -1
```

### Step 4: Deploy Log Forwarding
```bash
# Replace with actual IPs
./deploy-tpot-logging.sh 34.193.101.171 YOUR_LOCAL_IP
```

### Step 5: Test with Kali Machine
```bash
# Get Kali's public IP first
curl -s -4 icanhazip.com  # Run this on Kali

# Allow Kali to access specific honeypot ports
./kali-access.sh add KALI_PUBLIC_IP 22 80 443 3306

# Test from Kali
ssh admin@34.193.101.171        # Should connect to SSH honeypot
curl http://34.193.101.171/     # Should hit web honeypot

# Remove access when done
./kali-access.sh remove KALI_PUBLIC_IP 22 80 443 3306
```

---

## 🔑 Security Configuration

### T-Pot API Key
```
GENERATE_NEW_SECURE_API_KEY_64_CHARS
```
*This key is used for secure log forwarding from T-Pot to Mini-XDR*

### Access Restrictions
- **Management SSH (64295)**: Your IP only
- **Web Interface (64297)**: Your IP only  
- **Honeypot Services**: Blocked by default, selectively enabled for testing

### Log Sources Configured
- **Cowrie**: SSH/Telnet honeypot logs
- **Dionaea**: Multi-protocol honeypot logs
- **Suricata**: Network intrusion detection
- **Honeytrap**: Network honeypot logs
- **Elasticpot**: Elasticsearch honeypot logs
- **Heralding**: Multi-service honeypot logs

---

## 📊 Testing Workflow

### 1. Controlled Attack Testing
```bash
# From Kali machine (after enabling access)
nmap -sS 34.193.101.171 -p 22,80,443
ssh admin@34.193.101.171
curl -A "BadBot/1.0" http://34.193.101.171/admin
```

### 2. Monitor in Mini-XDR
- Check `/ingest/multi` endpoint for incoming logs
- Verify ML detection on honeypot events
- Test agent responses to detected threats

### 3. Verify Log Flow
```bash
# Check T-Pot log forwarding status
ssh -i ~/.ssh/mini-xdr-tpot-key.pem -p 64295 admin@34.193.101.171 \
    "sudo systemctl status tpot-fluent-bit"
```

---

## 🛠️ Management Commands

### T-Pot Management
```bash
./start-secure-tpot.sh                    # Start T-Pot securely
./kali-access.sh status                   # Check access rules
./kali-access.sh add IP ports...          # Add testing access
./kali-access.sh remove IP ports...       # Remove access
```

### T-Pot Direct Access (When Running)
```bash
# SSH to T-Pot
ssh -i ~/.ssh/mini-xdr-tpot-key.pem -p 64295 admin@TPOT_IP

# Check T-Pot status
sudo systemctl status tpot.service
sudo docker ps

# Check log forwarding
sudo systemctl status tpot-fluent-bit
```

---

## 🔍 Verification Checklist

- [ ] Mini-XDR backend running on port 8000
- [ ] T-Pot started with `start-secure-tpot.sh`
- [ ] Log forwarding deployed with correct local IP
- [ ] Kali machine IP allowed for specific ports
- [ ] Test attacks generating logs in Mini-XDR
- [ ] ML models detecting honeypot events
- [ ] Agents responding to threats appropriately

---

## ⚠️ Security Reminders

1. **Always remove Kali access** when testing is complete
2. **Monitor AWS costs** - stop T-Pot when not needed
3. **Check security group rules** regularly
4. **Rotate API keys** periodically
5. **Review logs** for any unexpected access

---

## 🎯 Current Configuration Files

- `config/tpot/tpot-config.json` - T-Pot integration settings
- `config/tpot/fluent-bit-tpot.conf` - Log forwarding configuration  
- `backend/.env` - Updated with T-Pot API key
- `TPOT_SECURITY_STATUS.md` - Security documentation

**Status**: Ready for secure controlled testing! 🚀
