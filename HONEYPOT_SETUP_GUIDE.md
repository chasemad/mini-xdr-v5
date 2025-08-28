# 🍯 Complete Honeypot VM Setup Guide for Mini-XDR Integration

## 📋 Prerequisites
- Honeypot VM with Ubuntu 20.04/22.04 LTS
- SSH access to honeypot VM as root or sudo user
- Network connectivity between XDR host (10.0.0.123) and honeypot (10.0.0.23)
- Mini-XDR system already set up and running

## 🎯 Overview
This guide will help you:
1. Set up Cowrie SSH honeypot with JSON logging
2. Configure secure SSH access for XDR system
3. Install and configure the Mini-XDR ingestion agent
4. Set up automated log forwarding
5. Test the complete integration

---

## 📦 Phase 1: Basic System Setup (15 minutes)

### Step 1: Update System and Install Dependencies
```bash
# SSH into your honeypot VM
ssh your-user@honeypot-ip

# Update system
sudo apt update && sudo apt upgrade -y

# Install essential packages
sudo apt install -y \
    python3 \
    python3-pip \
    python3-venv \
    git \
    curl \
    wget \
    ufw \
    nftables \
    net-tools \
    htop \
    jq
```

### Step 2: Configure Firewall and Port Redirection
```bash
# Enable UFW firewall
sudo ufw --force enable

# Allow necessary ports
sudo ufw allow 22022/tcp comment "SSH management"
sudo ufw allow 2222/tcp comment "Cowrie honeypot"
sudo ufw allow 22/tcp comment "Redirected SSH"
sudo ufw allow from 10.0.0.123 comment "XDR system access"

# Setup nftables for port redirection (22 -> 2222)
sudo tee /etc/nftables.conf << 'EOF'
#!/usr/sbin/nft -f

flush ruleset

table inet nat {
    chain prerouting {
        type nat hook prerouting priority 0; policy accept;
        tcp dport 22 redirect to :2222
    }
}
EOF

# Enable and start nftables
sudo systemctl enable nftables
sudo systemctl start nftables
```

---

## 🍯 Phase 2: Cowrie Honeypot Installation (20 minutes)

### Step 3: Create Cowrie User and Environment
```bash
# Create dedicated cowrie user
sudo adduser --disabled-password cowrie
sudo su - cowrie

# Create Python virtual environment
python3 -m venv cowrie-env
source cowrie-env/bin/activate

# Install required Python packages
pip install --upgrade pip
pip install twisted cryptography pyopenssl service_identity bcrypt
```

### Step 4: Install and Configure Cowrie
```bash
# Still as cowrie user
cd /home/cowrie
git clone https://github.com/cowrie/cowrie.git
cd cowrie

# Install Cowrie dependencies
pip install -r requirements.txt

# Copy and configure Cowrie
cp etc/cowrie.cfg.dist etc/cowrie.cfg

# Edit configuration for JSON logging
cat >> etc/cowrie.cfg << 'EOF'

[honeypot]
hostname = server01
log_path = var/log/cowrie
download_path = var/lib/cowrie/downloads
state_path = var/lib/cowrie
etc_path = honeyfs/etc
contents_path = honeyfs

[ssh]
enabled = true
listen_endpoints = tcp:2222:interface=0.0.0.0
version = SSH-2.0-OpenSSH_6.0p1 Debian-4+deb7u2
rsa_public_key = etc/ssh_host_rsa_key.pub
rsa_private_key = etc/ssh_host_rsa_key
dsa_public_key = etc/ssh_host_dsa_key.pub
dsa_private_key = etc/ssh_host_dsa_key

[output_jsonlog]
enabled = true
logfile = var/log/cowrie/cowrie.json
epoch_timestamp = true

[output_textlog]
enabled = false
EOF

# Generate SSH host keys
ssh-keygen -t rsa -b 2048 -f etc/ssh_host_rsa_key -N ""
ssh-keygen -t dsa -b 1024 -f etc/ssh_host_dsa_key -N ""

# Create log directories
mkdir -p var/log/cowrie var/lib/cowrie/downloads
```

### Step 5: Create Cowrie Systemd Service
```bash
# Exit cowrie user, back to your regular user
exit

# Create systemd service file
sudo tee /etc/systemd/system/cowrie.service << 'EOF'
[Unit]
Description=Cowrie SSH Honeypot
Documentation=https://cowrie.readthedocs.io
After=network.target

[Service]
Type=forking
User=cowrie
Group=cowrie
ExecStart=/home/cowrie/cowrie-env/bin/python /home/cowrie/cowrie/bin/cowrie start
ExecStop=/home/cowrie/cowrie-env/bin/python /home/cowrie/cowrie/bin/cowrie stop
WorkingDirectory=/home/cowrie/cowrie
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF

# Enable and start Cowrie
sudo systemctl daemon-reload
sudo systemctl enable cowrie
sudo systemctl start cowrie

# Check status
sudo systemctl status cowrie
```

---

## 🔐 Phase 3: XDR SSH Access Setup (10 minutes)

### Step 6: Configure SSH Access for XDR System
```bash
# Create xdrops user for XDR operations
sudo useradd -m -s /bin/bash xdrops

# Create SSH directory
sudo mkdir -p /home/xdrops/.ssh
sudo chmod 700 /home/xdrops/.ssh

# Add the XDR system's public key
# Replace with your actual public key from the XDR host
sudo tee /home/xdrops/.ssh/authorized_keys << 'EOF'
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIPvpS9tZDSnYx9WZyymXagulQLnxIdxXtwOTzAYgwWUL chasemad@Chases-MacBook-Pro.local
EOF

sudo chmod 600 /home/xdrops/.ssh/authorized_keys
sudo chown -R xdrops:xdrops /home/xdrops/.ssh

# Configure sudo permissions for UFW
sudo tee /etc/sudoers.d/xdrops-ufw << 'EOF'
# Allow xdrops user to manage UFW rules only
xdrops ALL=(ALL) NOPASSWD: /usr/sbin/ufw
EOF

# Configure SSH for management access on port 22022
sudo sed -i '/^#Port 22/a Port 22022' /etc/ssh/sshd_config
sudo systemctl restart sshd
```

---

## 🤖 Phase 4: Mini-XDR Agent Installation (15 minutes)

### Step 7: Install Mini-XDR Ingestion Agent
```bash
# Create directory for XDR agent
sudo mkdir -p /opt/mini-xdr
sudo chown xdrops:xdrops /opt/mini-xdr

# Switch to xdrops user
sudo su - xdrops

# Create Python virtual environment for agent
cd /opt/mini-xdr
python3 -m venv agent-env
source agent-env/bin/activate

# Install required Python packages
pip install --upgrade pip
pip install aiohttp aiofiles cryptography
```

### Step 8: Download and Configure Ingestion Agent
```bash
# Still as xdrops user in /opt/mini-xdr
# Copy the ingestion agent from your XDR system
# On XDR host, run: scp backend/app/agents/ingestion_agent.py xdrops@10.0.0.23:/opt/mini-xdr/

# For now, create the agent file directly:
cat > ingestion_agent.py << 'AGENT_EOF'
# Copy the entire content from the ingestion_agent.py file
# This is a complex file, so we'll download it from the XDR system
AGENT_EOF

# Create agent configuration
cat > agent-config.json << 'CONFIG_EOF'
{
  "backend_url": "http://10.0.0.123:8000",
  "api_key": "honeypot-agent-key-12345",
  "source_type": "cowrie",
  "hostname": "honeypot-01",
  "log_paths": {
    "cowrie": "/home/cowrie/cowrie/var/log/cowrie/cowrie.json"
  },
  "batch_size": 25,
  "flush_interval": 15,
  "max_retries": 3,
  "retry_delay": 5,
  "validate_ssl": false,
  "compress_data": true
}
CONFIG_EOF
```

### Step 9: Create Agent Systemd Service
```bash
# Exit xdrops user
exit

# Create systemd service for the agent
sudo tee /etc/systemd/system/mini-xdr-agent.service << 'EOF'
[Unit]
Description=Mini-XDR Ingestion Agent
Documentation=https://github.com/mini-xdr
After=network.target cowrie.service
Requires=cowrie.service

[Service]
Type=simple
User=xdrops
Group=xdrops
ExecStart=/opt/mini-xdr/agent-env/bin/python /opt/mini-xdr/ingestion_agent.py --config /opt/mini-xdr/agent-config.json --verbose
WorkingDirectory=/opt/mini-xdr
Restart=always
RestartSec=10
Environment=PYTHONPATH=/opt/mini-xdr

[Install]
WantedBy=multi-user.target
EOF
```

---

## 🔄 Phase 5: Alternative Log Forwarding (Fluent Bit) (10 minutes)

### Step 10: Install Fluent Bit (Alternative to Agent)
```bash
# Install Fluent Bit repository
curl https://raw.githubusercontent.com/fluent/fluent-bit/master/install.sh | sh

# Create Fluent Bit configuration
sudo mkdir -p /etc/fluent-bit

sudo tee /etc/fluent-bit/fluent-bit.conf << 'EOF'
[SERVICE]
    Flush        1
    Log_Level    info
    Daemon       off
    HTTP_Server  On
    HTTP_Listen  0.0.0.0
    HTTP_Port    2020

[INPUT]
    Name              tail
    Path              /home/cowrie/cowrie/var/log/cowrie/cowrie.json*
    Parser            json
    Tag               cowrie
    Refresh_Interval  1
    Read_from_Head    false

[OUTPUT]
    Name  http
    Match cowrie
    Host  10.0.0.123
    Port  8000
    URI   /ingest/multi
    Format json
    Header Authorization Bearer honeypot-agent-key-12345
    Retry_Limit 5
EOF

# Create Fluent Bit systemd service
sudo tee /etc/systemd/system/fluent-bit-xdr.service << 'EOF'
[Unit]
Description=Fluent Bit XDR Log Forwarder
Documentation=https://fluentbit.io/
Requires=network.target
After=network.target cowrie.service

[Service]
Type=simple
ExecStart=/opt/fluent-bit/bin/fluent-bit -c /etc/fluent-bit/fluent-bit.conf
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF
```

---

## ✅ Phase 6: Service Management and Testing (10 minutes)

### Step 11: Start All Services
```bash
# Reload systemd and start services
sudo systemctl daemon-reload

# Start Cowrie honeypot
sudo systemctl enable cowrie
sudo systemctl start cowrie

# Choose ONE of the following options:

# Option A: Start Mini-XDR Agent (recommended)
sudo systemctl enable mini-xdr-agent
sudo systemctl start mini-xdr-agent

# Option B: Start Fluent Bit (alternative)
# sudo systemctl enable fluent-bit-xdr
# sudo systemctl start fluent-bit-xdr

# Check service status
sudo systemctl status cowrie
sudo systemctl status mini-xdr-agent  # or fluent-bit-xdr
```

### Step 12: Verify Services and Connectivity
```bash
# Check if Cowrie is listening
sudo netstat -tlnp | grep :2222

# Check UFW status
sudo ufw status numbered

# Test log generation (simulate attack)
ssh root@localhost -p 2222
# (This will fail but create logs)

# Check Cowrie logs
sudo tail -f /home/cowrie/cowrie/var/log/cowrie/cowrie.json

# Check agent logs
sudo journalctl -u mini-xdr-agent -f

# Test network connectivity to XDR
curl -v http://10.0.0.123:8000/health
```

---

## 🧪 Phase 7: Integration Testing (15 minutes)

### Step 13: Copy Agent from XDR System
```bash
# On your XDR host (10.0.0.123), copy the agent to the honeypot:
cd /Users/chasemad/Desktop/mini-xdr
scp -i ~/.ssh/xdrops_id_ed25519 -P 22022 backend/app/agents/ingestion_agent.py xdrops@10.0.0.23:/opt/mini-xdr/

# Test SSH connectivity from XDR to honeypot
ssh -i ~/.ssh/xdrops_id_ed25519 -p 22022 xdrops@10.0.0.23 'sudo ufw status'
```

### Step 14: Test Log Forwarding
```bash
# On honeypot, generate test attack data
for i in {1..5}; do
  ssh -o ConnectTimeout=5 root@localhost -p 2222 2>/dev/null &
done

# Check logs are being generated
sudo tail -10 /home/cowrie/cowrie/var/log/cowrie/cowrie.json

# On XDR system, check if events are being received
curl http://localhost:8000/incidents | jq .
```

### Step 15: Test XDR Containment
```bash
# On XDR system, test manual containment
curl -X POST http://localhost:8000/contain \
  -H "Content-Type: application/json" \
  -d '{"ip": "192.168.1.100", "reason": "test containment"}'

# On honeypot, verify UFW rule was added
ssh -i ~/.ssh/xdrops_id_ed25519 -p 22022 xdrops@10.0.0.23 'sudo ufw status numbered'
```

---

## 🔧 Troubleshooting Guide

### Common Issues and Solutions

#### 1. Cowrie Won't Start
```bash
# Check logs
sudo journalctl -u cowrie -n 50

# Check if port 2222 is available
sudo netstat -tlnp | grep :2222

# Check permissions
sudo ls -la /home/cowrie/cowrie/var/log/cowrie/
```

#### 2. Agent Connection Issues
```bash
# Test network connectivity
telnet 10.0.0.123 8000

# Check agent logs
sudo journalctl -u mini-xdr-agent -n 50

# Verify configuration
cat /opt/mini-xdr/agent-config.json
```

#### 3. SSH Access Problems
```bash
# Check SSH service
sudo systemctl status sshd

# Verify SSH configuration
sudo sshd -T | grep -E "(Port|PubkeyAuthentication)"

# Check UFW rules
sudo ufw status numbered
```

#### 4. Log Forwarding Issues
```bash
# Check if logs are being generated
sudo tail -f /home/cowrie/cowrie/var/log/cowrie/cowrie.json

# Test manual log forwarding
curl -X POST http://10.0.0.123:8000/ingest/multi \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer honeypot-agent-key-12345" \
  -d '{"source_type": "cowrie", "hostname": "test", "events": [{"test": "event"}]}'
```

---

## 📊 Monitoring Commands

### Service Status Checks
```bash
# All services status
sudo systemctl status cowrie mini-xdr-agent

# View real-time logs
sudo journalctl -f -u cowrie -u mini-xdr-agent

# Network connections
sudo netstat -tlnp | grep -E "(2222|22022)"

# Resource usage
htop
```

### Performance Monitoring
```bash
# Log file sizes
du -sh /home/cowrie/cowrie/var/log/cowrie/

# Event rates
tail -f /home/cowrie/cowrie/var/log/cowrie/cowrie.json | grep -c 'login'

# Agent statistics (if using Mini-XDR agent)
sudo journalctl -u mini-xdr-agent | grep "Agent stats"
```

---

## 🎯 Success Verification Checklist

- [ ] Cowrie honeypot is running on port 2222
- [ ] SSH management access works on port 22022
- [ ] Port redirection (22 → 2222) is working
- [ ] XDR system can SSH to honeypot as xdrops user
- [ ] UFW commands work via SSH from XDR system
- [ ] Logs are being generated in `/home/cowrie/cowrie/var/log/cowrie/cowrie.json`
- [ ] Ingestion agent or Fluent Bit is forwarding logs
- [ ] XDR system is receiving events from honeypot
- [ ] Containment actions work (UFW rules added remotely)

---

## 🚀 Next Steps

Once setup is complete:

1. **Generate test traffic**: Use the attack simulation scripts
2. **Monitor the XDR dashboard**: Check http://localhost:3000
3. **Test AI agents**: Use the agents interface to analyze honeypot data
4. **Tune detection**: Adjust thresholds based on honeypot activity
5. **Set up monitoring**: Configure alerts for honeypot events

The honeypot should now be fully integrated with your Mini-XDR system!
