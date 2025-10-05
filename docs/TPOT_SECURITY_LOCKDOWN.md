# 🔒 Azure T-Pot Security Lockdown - COMPLETE

**Date:** October 5, 2025  
**Status:** ✅ **SECURED FOR TESTING**  
**Security Audit:** **PASSED** ✨

---

## 🎯 Security Assessment Summary

### BEFORE Lockdown (VULNERABLE 🚨)
```
❌ SSH (22, 64295):    Open to entire internet (*)
❌ Web UI (64297):     Open to entire internet (*)
❌ Honeypot Ports:     Open to entire internet (*)
   - Ports: 21, 23, 25, 80, 110, 143, 443, 445, 1433, 3306, 3389, 5432, 8080
```

### AFTER Lockdown (SECURED ✅)
```
✅ SSH (22, 64295):    Restricted to 24.11.0.176/32 only
✅ Web UI (64297):     Restricted to 24.11.0.176/32 only
✅ Honeypot Ports:     Restricted to 24.11.0.176/32 only
   - Ports: 21, 23, 25, 80, 110, 143, 443, 445, 1433, 3306, 3389, 5432, 8080
   - Rule: allow-honeypot-ports-restricted
```

---

## 📋 Current Network Security Group Rules

| Rule Name | Priority | Source IP | Destination Ports | Protocol | Status |
|-----------|----------|-----------|-------------------|----------|--------|
| `allow-ssh-your-ip-v4` | 100 | `24.11.0.176/32` | 22, 64295 | TCP | ✅ Secure |
| `allow-tpot-web-v4` | 200 | `24.11.0.176/32` | 64297 | TCP | ✅ Secure |
| `allow-honeypot-ports-restricted` | 300 | `24.11.0.176/32` | 21, 23, 25, 80, 110, 143, 443, 445, 1433, 3306, 3389, 5432, 8080 | ALL | ✅ Secure |

**Resource Group:** `mini-xdr-rg`  
**NSG Name:** `mini-xdr-tpotNSG`  
**T-Pot VM IP:** `74.235.242.205`  
**Your Authorized IP:** `24.11.0.176`

---

## 🔐 What Was Secured

### 1. **Management Access (CRITICAL)**
- **SSH Port 22:** Standard SSH - restricted to your IP only
- **SSH Port 64295:** T-Pot management SSH - restricted to your IP only
- **Web Port 64297:** T-Pot web dashboard - restricted to your IP only

**Risk Before:** Anyone could have accessed your T-Pot management interfaces  
**Risk After:** Only you can access from your specific IP address

### 2. **Honeypot Services (IMPORTANT)**
Secured the following honeypot ports (previously open to internet):
- **Port 21:** FTP honeypot
- **Port 23:** Telnet honeypot
- **Port 25:** SMTP honeypot
- **Port 80:** HTTP honeypot
- **Port 110:** POP3 honeypot
- **Port 143:** IMAP honeypot
- **Port 443:** HTTPS honeypot
- **Port 445:** SMB honeypot
- **Port 1433:** MSSQL honeypot
- **Port 3306:** MySQL honeypot
- **Port 3389:** RDP honeypot
- **Port 5432:** PostgreSQL honeypot
- **Port 8080:** HTTP-Alt honeypot

**Risk Before:** Honeypots were capturing real internet traffic (could overwhelm system during testing)  
**Risk After:** Only you can test honeypots from your machine

---

## 🛡️ Security Posture

### Current Mode: **TESTING/DEVELOPMENT** 🧪

```
┌─────────────────────────────────────────────────────────┐
│                     INTERNET                            │
│                    (BLOCKED 🔒)                         │
└────────────────────┬────────────────────────────────────┘
                     │
                     │ All traffic blocked
                     │ except from your IP
                     ▼
┌─────────────────────────────────────────────────────────┐
│              Azure Network Security Group               │
│         ✅ Firewall Rules: Your IP Only                 │
└────────────────────┬────────────────────────────────────┘
                     │
                     │ Allows: 24.11.0.176/32
                     ▼
┌─────────────────────────────────────────────────────────┐
│          T-Pot Honeypot VM (74.235.242.205)            │
│                                                         │
│  ✅ SSH Management (64295)                              │
│  ✅ Web Dashboard (64297)                               │
│  ✅ 36 Honeypot Containers                              │
│  ✅ All services secured                                │
└─────────────────────────────────────────────────────────┘
```

---

## 🚀 Management Scripts

### 1. Check Security Status
```bash
./scripts/check-azure-tpot-security.sh
```
**Purpose:** Quick security audit showing current NSG rules and access status

### 2. Secure for Testing (Current Mode)
```bash
./scripts/secure-azure-tpot-testing.sh
```
**Purpose:** Lock down all honeypot ports to your IP only  
**When to use:** During development and testing

### 3. Open to Internet (Production Mode)
```bash
./scripts/open-azure-tpot-to-internet.sh
```
**Purpose:** Open honeypot ports to internet for real threat capture  
**When to use:** When ready to go live and capture real attacks  
⚠️ **WARNING:** Only use when ready for production

---

## 🧪 Testing Your Honeypots

### From Your Machine (Works Now ✅)
```bash
# Test SSH honeypot
ssh root@74.235.242.205

# Test HTTP honeypot
curl http://74.235.242.205

# Test HTTPS honeypot
curl -k https://74.235.242.205

# Test Telnet honeypot
telnet 74.235.242.205 23

# Test MySQL honeypot
mysql -h 74.235.242.205 -u admin -p

# Run comprehensive test
./test-honeypot-attack.sh
```

### From Internet (Blocked 🔒)
All attempts from other IPs will be **dropped by Azure NSG**  
No response, no connection - complete blackhole

---

## 📊 Access Control Matrix

| Service Type | Port(s) | Your IP | Internet | Purpose |
|-------------|---------|---------|----------|---------|
| SSH Management | 22, 64295 | ✅ ALLOW | 🔒 DENY | System administration |
| Web Dashboard | 64297 | ✅ ALLOW | 🔒 DENY | T-Pot web interface |
| FTP Honeypot | 21 | ✅ ALLOW | 🔒 DENY | Testing only |
| Telnet Honeypot | 23 | ✅ ALLOW | 🔒 DENY | Testing only |
| SMTP Honeypot | 25 | ✅ ALLOW | 🔒 DENY | Testing only |
| HTTP Honeypot | 80 | ✅ ALLOW | 🔒 DENY | Testing only |
| POP3 Honeypot | 110 | ✅ ALLOW | 🔒 DENY | Testing only |
| IMAP Honeypot | 143 | ✅ ALLOW | 🔒 DENY | Testing only |
| HTTPS Honeypot | 443 | ✅ ALLOW | 🔒 DENY | Testing only |
| SMB Honeypot | 445 | ✅ ALLOW | 🔒 DENY | Testing only |
| MSSQL Honeypot | 1433 | ✅ ALLOW | 🔒 DENY | Testing only |
| MySQL Honeypot | 3306 | ✅ ALLOW | 🔒 DENY | Testing only |
| RDP Honeypot | 3389 | ✅ ALLOW | 🔒 DENY | Testing only |
| PostgreSQL Honeypot | 5432 | ✅ ALLOW | 🔒 DENY | Testing only |
| HTTP-Alt Honeypot | 8080 | ✅ ALLOW | 🔒 DENY | Testing only |

---

## 🔄 Workflow: Testing to Production

### Phase 1: Testing (Current ✅)
```bash
# Already done - you're here!
./scripts/secure-azure-tpot-testing.sh
```
- All ports restricted to your IP
- Safe for development and testing
- No risk of overwhelming attacks

### Phase 2: Pre-Production Verification
```bash
# Test all honeypots work
./test-honeypot-attack.sh

# Verify backend is ingesting events
curl http://localhost:8000/incidents | jq .

# Check T-Pot logs
ssh -i ~/.ssh/mini-xdr-tpot-azure azureuser@74.235.242.205 -p 64295
sudo docker logs cowrie
```

### Phase 3: Go Live (When Ready)
```bash
# Open to internet for real attacks
./scripts/open-azure-tpot-to-internet.sh
```
**This will:**
- Remove IP restrictions from honeypot ports
- Keep management ports (SSH, Web) secured to your IP
- Begin capturing real attacks from around the world

### Phase 4: Monitor Production
```bash
# Watch T-Pot web interface
open https://74.235.242.205:64297

# Monitor Mini-XDR frontend
open http://localhost:3000/incidents

# View live logs
ssh -i ~/.ssh/mini-xdr-tpot-azure azureuser@74.235.242.205 -p 64295
sudo docker logs -f cowrie
```

---

## 🎯 IP Address Changes

### If Your IP Changes
Your current authorized IP is `24.11.0.176`. If your IP changes:

1. **Check your new IP:**
   ```bash
   curl ifconfig.me
   ```

2. **Update NSG rules:**
   ```bash
   ./scripts/secure-azure-tpot-testing.sh
   ```
   The script will detect your new IP and update all rules automatically

3. **Verify:**
   ```bash
   ./scripts/check-azure-tpot-security.sh
   ```

---

## 🔍 Manual Security Verification

### View All NSG Rules
```bash
az network nsg rule list \
  --resource-group mini-xdr-rg \
  --nsg-name mini-xdr-tpotNSG \
  --output table
```

### View Specific Rule
```bash
# Check honeypot rule
az network nsg rule show \
  --resource-group mini-xdr-rg \
  --nsg-name mini-xdr-tpotNSG \
  --name allow-honeypot-ports-restricted
```

### Test Connectivity
```bash
# From your machine (should work ✅)
nc -zv 74.235.242.205 22
nc -zv 74.235.242.205 64295
nc -zv 74.235.242.205 64297
nc -zv 74.235.242.205 80

# From other IPs (should timeout 🔒)
# Use a VPN or different network to verify blocking
```

---

## 📈 Cost Optimization

### Current Configuration
- **VM Size:** Standard_B2s (2 vCPU, 4GB RAM)
- **Public IP:** Standard SKU (static)
- **Storage:** 30GB Standard SSD
- **Estimated Monthly Cost:** $40-65 USD

### Cost During Testing Phase
Since honeypots are restricted to your IP only:
- **Minimal bandwidth usage** (only your test traffic)
- **Lower CPU/memory usage** (no internet-scale attacks)
- **Same base VM cost** (~$40/month)

### Cost in Production Phase
When opened to internet:
- **Increased bandwidth** (real attack traffic)
- **Higher CPU usage** (processing attacks)
- **More storage** (logs and attack data)
- **Estimated:** $50-75/month depending on attack volume

---

## 🛡️ Additional Security Recommendations

### 1. **Enable Azure Security Center** (Optional)
```bash
az security pricing create \
  --name VirtualMachines \
  --tier Standard
```

### 2. **Set Up Log Analytics** (Optional)
- Monitor NSG flow logs
- Track access patterns
- Alert on suspicious activity

### 3. **Configure Azure Monitor Alerts**
```bash
# Alert if IP is changed without authorization
# Alert if NSG rules are modified
# Alert on high bandwidth usage
```

### 4. **Regular Security Audits**
```bash
# Run weekly
./scripts/check-azure-tpot-security.sh

# Verify no unauthorized rule changes
az network nsg rule list \
  --resource-group mini-xdr-rg \
  --nsg-name mini-xdr-tpotNSG \
  --output table
```

### 5. **Key Management**
```bash
# Rotate SSH keys every 90 days
ssh-keygen -t rsa -b 4096 -f ~/.ssh/mini-xdr-tpot-azure-new

# Update Azure VM
az vm user update \
  --resource-group mini-xdr-rg \
  --name mini-xdr-tpot \
  --username azureuser \
  --ssh-key-value "$(cat ~/.ssh/mini-xdr-tpot-azure-new.pub)"
```

---

## 📞 Quick Reference

### T-Pot Access
- **Web UI:** https://74.235.242.205:64297
- **Username:** `tsec`
- **Password:** `minixdrtpot2025`
- **SSH:** `ssh -i ~/.ssh/mini-xdr-tpot-azure azureuser@74.235.242.205 -p 64295`

### Azure Resources
- **Resource Group:** `mini-xdr-rg`
- **VM Name:** `mini-xdr-tpot`
- **NSG Name:** `mini-xdr-tpotNSG`
- **Region:** `eastus`

### Your Configuration
- **Authorized IP:** `24.11.0.176`
- **SSH Key:** `~/.ssh/mini-xdr-tpot-azure`

---

## ✅ Security Checklist

- [x] SSH access restricted to your IP only
- [x] Web interface restricted to your IP only
- [x] Honeypot ports restricted to your IP only
- [x] No public internet access to any services
- [x] NSG rules verified and documented
- [x] Management scripts created and tested
- [x] Security audit scripts in place
- [x] Testing workflow documented
- [x] Production deployment procedure documented
- [x] IP change procedure documented

---

## 🎉 Summary

**Your Azure T-Pot honeypot is now FULLY SECURED for testing!**

✨ **What was accomplished:**
- Removed dangerous public internet access
- Restricted all services to your IP only (24.11.0.176)
- Created management scripts for easy security control
- Documented complete security posture
- Established testing → production workflow

🔒 **Security Status:**
- Management interfaces: **SECURED** ✅
- Honeypot services: **SECURED** ✅
- Internet access: **BLOCKED** ✅
- Ready for: **TESTING** ✅

🚀 **Next Steps:**
1. Test honeypots from your machine: `./test-honeypot-attack.sh`
2. Verify Mini-XDR ingestion: `http://localhost:3000/incidents`
3. When ready, go live: `./scripts/open-azure-tpot-to-internet.sh`

**Your system is secure and ready for testing! 🛡️**

---

*Last Updated: October 5, 2025*  
*Security Review: Passed ✅*  
*Mode: Testing/Development 🧪*

