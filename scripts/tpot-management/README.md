# 🍯 T-Pot Honeypot Management Scripts

This directory contains all scripts for managing your T-Pot honeypot deployment on AWS, including security configuration, log forwarding, and access control.

## Scripts Overview

### 🚀 Deployment & Startup

#### `setup-tpot-integration.sh`
**Complete T-Pot integration setup**
- **Purpose**: Prepare Mini-XDR for T-Pot honeypot integration
- **Features**: API key generation, Fluent Bit config, environment setup
- **Usage**: `./setup-tpot-integration.sh`
- **Run**: Once during initial setup

#### `start-secure-tpot.sh`
**Secure T-Pot startup script**
- **Purpose**: Start T-Pot honeypot with all security measures in place
- **Features**: AWS instance startup, status monitoring, security verification
- **Usage**: `./start-secure-tpot.sh`
- **Run**: Whenever you want to start the honeypot

### 🔒 Security Management

#### `secure-tpot.sh`
**T-Pot security hardening script**
- **Purpose**: Remove all public internet access to honeypot services
- **Features**: AWS security group modification, access restriction
- **Usage**: `./secure-tpot.sh`
- **Run**: Already executed - honeypot is secured

#### `kali-access.sh`
**Kali machine access control**
- **Purpose**: Manage selective access for controlled testing
- **Features**: Add/remove IP access, port-specific controls, status checking
- **Usage**: `./kali-access.sh [add|remove|status] [kali-ip] [ports...]`
- **Run**: Before and after testing sessions

### 📡 Log Management

#### `deploy-tpot-logging.sh`
**Log forwarding deployment**
- **Purpose**: Deploy Fluent Bit configuration to T-Pot for log forwarding
- **Features**: Fluent Bit installation, systemd service creation, log pipeline setup
- **Usage**: `./deploy-tpot-logging.sh <tpot-ip> <local-ip>`
- **Run**: After T-Pot startup to enable log forwarding

## Usage Guide

### Initial Setup (One-time)
```bash
# 1. Set up Mini-XDR integration
./setup-tpot-integration.sh

# 2. Security is already configured (honeypot is locked down)
# No need to run secure-tpot.sh again
```

### Regular Operations

#### Starting T-Pot for Testing
```bash
# 1. Start the honeypot securely
./start-secure-tpot.sh

# 2. Deploy log forwarding (get your local IP first)
ifconfig | grep "inet " | grep -v 127.0.0.1 | awk '{print $2}' | head -1
./deploy-tpot-logging.sh 34.193.101.171 YOUR_LOCAL_IP

# 3. Allow your Kali machine for testing
curl -s -4 icanhazip.com  # Run this on Kali to get IP
./kali-access.sh add KALI_IP 22 80 443 3306
```

#### After Testing
```bash
# Remove Kali access
./kali-access.sh remove KALI_IP 22 80 443 3306

# Check security status
./kali-access.sh status

# Stop T-Pot (optional - saves AWS costs)
aws ec2 stop-instances --region us-east-1 --instance-ids $(aws ec2 describe-instances --region us-east-1 --filters "Name=tag:Name,Values=mini-xdr-tpot-honeypot" --query 'Reservations[0].Instances[0].InstanceId' --output text)
```

### Access Control Examples

#### Grant Kali Access for Specific Tests
```bash
# SSH and web honeypots only
./kali-access.sh add 203.0.113.10 22 80 443

# Database testing
./kali-access.sh add 203.0.113.10 3306 5432 6379

# Multiple services
./kali-access.sh add 203.0.113.10 22 23 25 80 443 3306 3389

# Check what's currently allowed
./kali-access.sh status
```

#### Remove Access
```bash
# Remove specific ports
./kali-access.sh remove 203.0.113.10 22 80 443

# Remove all access (run status first to see current rules)
./kali-access.sh status
# Then remove each rule manually
```

## Configuration Files

### Generated Configurations
- **`config/tpot/tpot-config.json`**: T-Pot integration settings
- **`config/tpot/fluent-bit-tpot.conf`**: Log forwarding configuration
- **`backend/.env`**: Updated with T-Pot API key

### Current Settings
- **T-Pot IP**: 34.193.101.171
- **SSH Management Port**: 64295
- **Web Interface Port**: 64297
- **Security Group**: sg-037bd4ee6b74489b5
- **API Key**: 6c49b95dd921e0003ce159e6b3c0b6eb4e126fc2b19a1530a0f72a4a9c0c1eee

## Security Architecture

### Current Security Posture
```
Internet → AWS Security Groups → T-Pot Instance
          ↓ (BLOCKED by default)
          ✅ Management: Your IP only (64295, 64297)
          🔒 Honeypots: Blocked from public
          🎯 Testing: Selective Kali access only
```

### Network Flow
```
Kali Machine → (Controlled Access) → T-Pot Honeypots
T-Pot Logs → Fluent Bit → Your Mini-XDR (Port 8000)
```

## Troubleshooting

### Common Issues

#### T-Pot Won't Start
```bash
# Check instance status
aws ec2 describe-instances --region us-east-1 --filters "Name=tag:Name,Values=mini-xdr-tpot-honeypot"

# Check if already running
./start-secure-tpot.sh  # Will show current status
```

#### Can't Access T-Pot Management
```bash
# Verify your IP is allowed
./kali-access.sh status

# Test SSH access
ssh -i ~/.ssh/mini-xdr-tpot-key.pem -p 64295 admin@34.193.101.171
```

#### Log Forwarding Not Working
```bash
# Check Fluent Bit status on T-Pot
ssh -i ~/.ssh/mini-xdr-tpot-key.pem -p 64295 admin@34.193.101.171 "sudo systemctl status tpot-fluent-bit"

# Verify Mini-XDR is listening
netstat -tulnp | grep :8000

# Check logs
ssh -i ~/.ssh/mini-xdr-tpot-key.pem -p 64295 admin@34.193.101.171 "sudo journalctl -u tpot-fluent-bit -f"
```

#### Kali Access Not Working
```bash
# Verify security group rules
aws ec2 describe-security-groups --group-ids sg-037bd4ee6b74489b5 --region us-east-1

# Test from Kali
curl -m 5 http://34.193.101.171/  # Should work if access granted
nmap -p 22,80,443 34.193.101.171  # Should show open if access granted
```

## Maintenance

### Regular Tasks
- **Monitor AWS costs**: T-Pot costs ~$50-80/month when running
- **Rotate API keys**: Update T-Pot API key monthly
- **Review security groups**: Ensure no unintended access
- **Update T-Pot**: SSH to instance and update containers

### Backup Important Files
- SSH keys: `~/.ssh/mini-xdr-tpot-key.pem`
- Configuration: `config/tpot/`
- Environment: `backend/.env` (T-Pot section)

---

**Security Status**: 🔒 **FULLY SECURED**  
**Public Access**: ❌ **BLOCKED**  
**Management Access**: ✅ **YOUR IP ONLY**  
**Testing Access**: 🎯 **CONTROLLED VIA SCRIPTS**

**Last Updated**: September 16, 2025  
**Maintained by**: Mini-XDR Security Team


