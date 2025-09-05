# 🚨 Mini-XDR Attack Testing Guide

This guide provides safe and effective ways to test your Mini-XDR system's detection and response capabilities using realistic attack scenarios.

## ⚠️ IMPORTANT DISCLAIMER

**ONLY USE THESE TOOLS AGAINST:**
- Systems you own
- Systems you have explicit written permission to test
- Lab environments set up for testing

**NEVER USE AGAINST:**
- Production systems without permission
- Third-party systems
- Any system you don't own

## 🎯 Attack Scripts Overview

### 1. Simple Attack Test (`simple_attack_test.py`)
**Best for:** Quick testing and demonstrations
- **Duration:** ~2-3 minutes
- **Attack Types:** Web attacks, brute force, directory traversal, reconnaissance
- **Intensity:** Moderate (good for initial testing)

### 2. Comprehensive Attack Simulation (`attack_simulation.py`)
**Best for:** Extended testing and stress testing
- **Duration:** Configurable (default 5 minutes)
- **Attack Types:** Multi-threaded comprehensive attacks
- **Intensity:** Configurable (low/medium/high)

## 🛠️ Setup Instructions

### Prerequisites on Kali Linux

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python dependencies
pip3 install requests urllib3

# Download the attack scripts
wget https://raw.githubusercontent.com/your-repo/mini-xdr/main/simple_attack_test.py
wget https://raw.githubusercontent.com/your-repo/mini-xdr/main/attack_simulation.py

# Make scripts executable
chmod +x simple_attack_test.py attack_simulation.py
```

## 🎮 Running the Tests

### Option 1: Simple Attack Test (Recommended for first test)

```bash
# Basic usage against your target
python3 simple_attack_test.py 192.168.1.100

# If your Mini-XDR is running on a different port
python3 simple_attack_test.py your-server-ip
```

### Option 2: Comprehensive Attack Simulation

```bash
# Basic usage
python3 attack_simulation.py --target 192.168.1.100

# High intensity, longer duration
python3 attack_simulation.py --target 192.168.1.100 --intensity high --duration 600

# Against different port
python3 attack_simulation.py --target 192.168.1.100 --port 8080 --intensity medium --duration 300

# Low intensity for testing
python3 attack_simulation.py --target 192.168.1.100 --intensity low --duration 120
```

## 📊 What to Expect in Mini-XDR

### Incidents That Should Be Created:
1. **SQL Injection Attacks**
   - Risk Score: High (70-90%)
   - Severity: High
   - IOCs: Malicious payloads in parameters

2. **Brute Force Attacks**
   - Risk Score: Medium-High (60-80%)
   - Severity: Medium-High
   - IOCs: Multiple failed login attempts

3. **Directory Traversal**
   - Risk Score: High (75-95%)
   - Severity: High
   - IOCs: Path traversal patterns

4. **Reconnaissance Activity**
   - Risk Score: Medium (50-70%)
   - Severity: Medium
   - IOCs: Scanning patterns, tool signatures

### Threat Intelligence Hits:
- **Malicious User Agents:** sqlmap, Nikto, DirBuster, Nmap, etc.
- **Known Attack Patterns:** SQL injection signatures
- **Scanning Tools:** gobuster, Hydra signatures

## 🛡️ Testing SOC Response Actions

Once incidents are created, test these SOC actions:

### 1. Block IP
```
Action: socBlockIP()
Expected: IP address gets blocked from further access
Test: Try accessing from the same IP - should be blocked
```

### 2. Isolate Host
```
Action: socIsolateHost()
Expected: Target host gets isolated from network
Test: Network connectivity should be restricted
```

### 3. Threat Intelligence Lookup
```
Action: socThreatIntelLookup()
Expected: Returns threat intel data about the attacking IP
Test: Should show malicious indicators and reputation data
```

### 4. Hunt Similar Attacks
```
Action: socHuntSimilarAttacks()
Expected: Finds other incidents with similar patterns
Test: Should correlate related attack incidents
```

### 5. Reset Passwords
```
Action: socResetPasswords()
Expected: Forces password reset for affected accounts
Test: User accounts should require password change
```

## 🔍 Monitoring and Verification

### Check These Locations:

1. **Mini-XDR Dashboard** (`http://localhost:3000`)
   - Active Incidents tab
   - Threat Overview metrics
   - Recent Activity section

2. **Backend Logs** (`backend/backend.log`)
   ```bash
   tail -f backend/backend.log
   ```

3. **Database** (Check incidents table)
   ```bash
   sqlite3 backend/xdr.db "SELECT * FROM incidents ORDER BY created_at DESC LIMIT 10;"
   ```

## 🎯 Attack Scenarios by Use Case

### Scenario 1: Web Application Security Test
```bash
# Focus on web attacks
python3 simple_attack_test.py your-target-ip
```
**Expected:** SQL injection, XSS, directory traversal incidents

### Scenario 2: Brute Force Detection Test
```bash
# Run comprehensive with focus on authentication
python3 attack_simulation.py --target your-target-ip --intensity medium --duration 300
```
**Expected:** Multiple brute force incidents, account lockout triggers

### Scenario 3: Threat Intelligence Test
```bash
# Use known malicious indicators
python3 attack_simulation.py --target your-target-ip --intensity low --duration 180
```
**Expected:** Threat intel hits on user agents and patterns

### Scenario 4: SOC Response Test
```bash
# Generate incidents then test response actions
python3 simple_attack_test.py your-target-ip
# Then use SOC dashboard to execute response actions
```

## 🚦 Traffic Light System

### 🟢 GREEN (Safe Testing)
- Internal lab environment
- VM-based targets
- Isolated test networks
- Systems you own

### 🟡 YELLOW (Proceed with Caution)
- Development environments
- Staging systems
- Always get permission first

### 🔴 RED (DO NOT TEST)
- Production systems
- Third-party systems
- Systems without explicit permission
- Public internet targets

## 🔧 Troubleshooting

### Common Issues:

1. **No Incidents Created**
   - Check Mini-XDR backend is running
   - Verify target IP is reachable
   - Check logs for errors

2. **Connection Refused**
   - Target system may be down
   - Firewall blocking connections
   - Wrong IP/port specified

3. **Low Detection Rate**
   - Increase attack intensity
   - Check ML model is loaded
   - Verify detection rules are active

4. **SOC Actions Not Working**
   - Check API endpoints are responding
   - Verify permissions and authentication
   - Check backend logs for errors

## 📈 Performance Metrics

### Expected Response Times:
- **Incident Creation:** < 30 seconds
- **ML Analysis:** < 60 seconds
- **Threat Intel Lookup:** < 10 seconds
- **SOC Action Execution:** < 5 seconds

### Success Indicators:
- ✅ Incidents appear in dashboard
- ✅ Risk scores are calculated
- ✅ Threat intel hits are recorded
- ✅ SOC actions execute successfully
- ✅ Toast notifications appear
- ✅ Real-time updates work

## 🎉 Next Steps

After successful testing:

1. **Review Incident Details**
   - Click on incidents to see full analysis
   - Check IOC extraction
   - Verify timeline accuracy

2. **Test AI Chat**
   - Ask questions about incidents
   - Request analysis and recommendations
   - Test context-aware responses

3. **Explore Advanced Features**
   - Multi-tab incident investigation
   - Attack timeline visualization
   - Compromise assessment

4. **Customize Detection Rules**
   - Adjust ML thresholds
   - Add custom IOC patterns
   - Configure alert policies

## 📞 Support

If you encounter issues:
1. Check the logs first
2. Verify network connectivity
3. Test with simple attacks before complex ones
4. Review the Mini-XDR documentation

**Remember: Always test responsibly and only against authorized targets!** 🛡️
