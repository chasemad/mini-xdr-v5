# Mini-XDR AWS Honeypot Management Guide

Complete guide for managing AWS VM honeypots with automatic startup/stop capabilities.

## 🚀 Quick Start

### 1. Deploy New Honeypot
```bash
# Run the enhanced setup script
./ops/aws-honeypot-enhanced-setup.sh

# This will:
# - Create VPC infrastructure
# - Launch EC2 instance with honeypot tools
# - Configure automatic startup on boot
# - Create local management scripts
# - Set up log forwarding to your XDR
```

### 2. Manage Existing Honeypot
```bash
# Check status
~/.mini-xdr/honeypot-management/status.sh

# Start services
~/.mini-xdr/honeypot-management/start.sh

# Stop services
~/.mini-xdr/honeypot-management/stop.sh

# Restart services
~/.mini-xdr/honeypot-management/restart.sh

# Connect for manual management
~/.mini-xdr/honeypot-management/connect.sh
```

## 📋 Available Scripts

### Deployment Scripts
| Script | Purpose | Usage |
|--------|---------|-------|
| `aws-honeypot-enhanced-setup.sh` | Complete AWS infrastructure deployment | `./ops/aws-honeypot-enhanced-setup.sh` |
| `aws-honeypot-setup.sh` | Basic multi-region deployment | `./ops/aws-honeypot-setup.sh` |
| `aws-private-honeypot-setup.sh` | Private honeypot (your IP only) | `./ops/aws-private-honeypot-setup.sh` |

### VM Management Scripts (On Honeypot)
| Script | Purpose | Location |
|--------|---------|----------|
| `honeypot-vm-startup.sh` | Start all honeypot services | `/opt/mini-xdr/honeypot-vm-startup.sh` |
| `honeypot-vm-stop.sh` | Stop all honeypot services | `/opt/mini-xdr/honeypot-vm-stop.sh` |
| `honeypot-status.sh` | Check service status | `/opt/mini-xdr/honeypot-status.sh` |

### Local Management Scripts
| Script | Purpose | Location |
|--------|---------|----------|
| `connect.sh` | SSH to honeypot | `~/.mini-xdr/honeypot-management/connect.sh` |
| `status.sh` | Remote status check | `~/.mini-xdr/honeypot-management/status.sh` |
| `start.sh` | Remote start services | `~/.mini-xdr/honeypot-management/start.sh` |
| `stop.sh` | Remote stop services | `~/.mini-xdr/honeypot-management/stop.sh` |
| `restart.sh` | Remote restart services | `~/.mini-xdr/honeypot-management/restart.sh` |

## 🛠️ Honeypot Services

### Core Services
- **Cowrie SSH Honeypot** - Port 22 (redirected to 2222)
- **Apache Web Honeypot** - Port 80/443
- **Fluent Bit Log Forwarder** - Sends logs to your XDR
- **UFW Firewall** - Manages access rules

### Additional Honeypot Ports
- **FTP**: Port 21
- **Telnet**: Port 23
- **SMTP**: Port 25
- **DNS**: Port 53
- **POP3**: Port 110
- **IMAP**: Port 143
- **MySQL**: Port 3306
- **PostgreSQL**: Port 5432
- **MSSQL**: Port 1433

## 🔧 VM Operations

### Starting Honeypot Services
```bash
# On the VM (requires sudo)
sudo /opt/mini-xdr/honeypot-vm-startup.sh

# Remotely from your machine
~/.mini-xdr/honeypot-management/start.sh
```

### Stopping Honeypot Services
```bash
# On the VM (requires sudo)
sudo /opt/mini-xdr/honeypot-vm-stop.sh

# With options
sudo /opt/mini-xdr/honeypot-vm-stop.sh --disable-firewall --disable-auto-startup --force

# Remotely from your machine
~/.mini-xdr/honeypot-management/stop.sh
```

### Checking Status
```bash
# On the VM
sudo /opt/mini-xdr/honeypot-status.sh

# Remotely from your machine
~/.mini-xdr/honeypot-management/status.sh

# Check individual services
systemctl status cowrie
systemctl status apache2
systemctl status fluent-bit
```

## 🔄 Automatic Startup

The honeypot is configured with automatic startup on VM boot via systemd:

### Service Configuration
```bash
# Check auto-startup status
systemctl status honeypot-startup.service

# Enable auto-startup
sudo systemctl enable honeypot-startup.service

# Disable auto-startup
sudo systemctl disable honeypot-startup.service

# View startup logs
sudo journalctl -u honeypot-startup.service -f
```

### Manual Service Control
```bash
# Start all honeypot services
sudo systemctl start honeypot-startup

# Stop all honeypot services
sudo systemctl stop honeypot-startup

# Restart all honeypot services
sudo systemctl restart honeypot-startup
```

## 📊 Monitoring & Logs

### Log Locations
```bash
# Cowrie SSH honeypot logs
/opt/cowrie/var/log/cowrie/cowrie.json
/opt/cowrie/var/log/cowrie/cowrie.log

# Web honeypot logs
/var/log/web-honeypot.log

# Apache logs
/var/log/apache2/access.log
/var/log/apache2/error.log

# System logs
/var/log/honeypot-startup.log
/var/log/honeypot-shutdown.log

# Status file
/var/lib/honeypot-status
```

### Real-time Monitoring
```bash
# Monitor Cowrie activity
tail -f /opt/cowrie/var/log/cowrie/cowrie.json

# Monitor web attacks
tail -f /var/log/web-honeypot.log

# Monitor all honeypot logs
sudo journalctl -f -u cowrie -u apache2 -u fluent-bit

# Monitor startup service
sudo journalctl -u honeypot-startup.service -f
```

### Status Checking
```bash
# Quick status overview
/opt/mini-xdr/honeypot-status.sh

# Detailed service status
systemctl status cowrie apache2 fluent-bit ufw

# Network port status
netstat -tlnp | grep -E ":22|:80|:443|:2222|:22022"

# Process status
ps aux | grep -E "(cowrie|apache2|fluent-bit)"
```

## 🔐 Security & Access

### SSH Access
```bash
# Management access (port 22022)
ssh -i ~/.ssh/mini-xdr-honeypot-key.pem -p 22022 ubuntu@<PUBLIC_IP>

# Honeypot access (port 22 - for testing)
ssh admin@<PUBLIC_IP>  # This goes to Cowrie honeypot
```

### Web Access
```bash
# Main login honeypot
http://<PUBLIC_IP>/login.php

# WordPress admin simulation
http://<PUBLIC_IP>/wp-admin/

# phpMyAdmin simulation
http://<PUBLIC_IP>/phpmyadmin/

# API endpoint
http://<PUBLIC_IP>/api/v1.php
```

### Firewall Management
```bash
# Check firewall status
sudo ufw status numbered

# Add custom rule
sudo ufw allow from <IP> to any port <PORT>

# Remove rule
sudo ufw delete <RULE_NUMBER>

# Reset firewall (caution!)
sudo ufw --force reset
```

## 🚨 Troubleshooting

### Common Issues

#### Services Won't Start
```bash
# Check service status
systemctl status cowrie apache2 fluent-bit

# Check service logs
sudo journalctl -u cowrie -n 50
sudo journalctl -u apache2 -n 50

# Check port conflicts
netstat -tlnp | grep -E ":2222|:80|:443"

# Restart individual services
sudo systemctl restart cowrie
sudo systemctl restart apache2
```

#### SSH Access Issues
```bash
# Check SSH service
systemctl status sshd

# Check SSH configuration
sudo sshd -t

# Check firewall rules
sudo ufw status numbered

# Check key permissions
ls -la ~/.ssh/mini-xdr-honeypot-key.pem
chmod 600 ~/.ssh/mini-xdr-honeypot-key.pem
```

#### Log Forwarding Issues
```bash
# Check Fluent Bit status
systemctl status fluent-bit

# Check Fluent Bit configuration
sudo cat /etc/fluent-bit/fluent-bit.conf

# Test log forwarding
curl -X POST http://YOUR_XDR_IP:8000/ingest/multi \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer demo-honeypot-key' \
  -d '{"test": "message"}'
```

#### Auto-startup Issues
```bash
# Check startup service
systemctl status honeypot-startup.service

# Check startup service logs
sudo journalctl -u honeypot-startup.service -n 100

# Test startup script manually
sudo /opt/mini-xdr/honeypot-vm-startup.sh

# Re-enable auto-startup
sudo systemctl daemon-reload
sudo systemctl enable honeypot-startup.service
```

### Recovery Procedures

#### Complete Service Reset
```bash
# Stop all services
sudo /opt/mini-xdr/honeypot-vm-stop.sh --force

# Clean up processes
sudo pkill -f cowrie
sudo pkill -f apache2
sudo pkill -f fluent-bit

# Restart all services
sudo /opt/mini-xdr/honeypot-vm-startup.sh

# Re-enable auto-startup
sudo systemctl enable honeypot-startup.service
```

#### Restore Default Configuration
```bash
# Backup current config
sudo cp /opt/cowrie/etc/cowrie.cfg /opt/cowrie/etc/cowrie.cfg.backup

# Restore default Cowrie config
cd /opt/cowrie
sudo cp etc/cowrie.cfg.dist etc/cowrie.cfg
sudo chown cowrie:cowrie etc/cowrie.cfg

# Restart services
sudo systemctl restart cowrie
```

## 🔄 Updates & Maintenance

### Updating Honeypot Scripts
```bash
# Update startup script
sudo wget -O /opt/mini-xdr/honeypot-vm-startup.sh \
  https://raw.githubusercontent.com/your-repo/mini-xdr/main/ops/honeypot-vm-startup.sh
sudo chmod +x /opt/mini-xdr/honeypot-vm-startup.sh

# Update stop script
sudo wget -O /opt/mini-xdr/honeypot-vm-stop.sh \
  https://raw.githubusercontent.com/your-repo/mini-xdr/main/ops/honeypot-vm-stop.sh
sudo chmod +x /opt/mini-xdr/honeypot-vm-stop.sh
```

### System Updates
```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Update Cowrie
cd /opt/cowrie
sudo -u cowrie git pull
sudo -u cowrie ./cowrie-env/bin/pip install --upgrade -r requirements.txt
sudo systemctl restart cowrie
```

### Log Rotation
```bash
# Check log rotation config
sudo cat /etc/logrotate.d/honeypot

# Test log rotation
sudo logrotate -d /etc/logrotate.d/honeypot

# Force log rotation
sudo logrotate -f /etc/logrotate.d/honeypot
```

## 📈 Performance Tuning

### Resource Monitoring
```bash
# Check system resources
htop
df -h
free -h

# Check service resource usage
systemctl status cowrie apache2 fluent-bit
```

### Optimization Settings
```bash
# Adjust log retention (edit /etc/logrotate.d/honeypot)
# Change "rotate 30" to desired number of days

# Adjust Fluent Bit flush interval (edit /etc/fluent-bit/fluent-bit.conf)
# Change "Flush 5" to desired seconds

# Optimize Cowrie performance (edit /opt/cowrie/etc/cowrie.cfg)
# Adjust [honeypot] settings as needed
```

## 🆘 Emergency Procedures

### Complete Shutdown
```bash
# Emergency stop all services
sudo /opt/mini-xdr/honeypot-vm-stop.sh --force --disable-auto-startup

# Disable all honeypot services
sudo systemctl disable cowrie apache2 fluent-bit honeypot-startup

# Shutdown VM
sudo shutdown -h now
```

### Instance Recovery
```bash
# If SSH access is lost, use AWS Systems Manager Session Manager
# Or connect via EC2 Instance Connect in AWS Console

# Check system status
systemctl --failed
journalctl -p err -n 50

# Restart networking
sudo systemctl restart networking

# Reset firewall if needed
sudo ufw --force reset
sudo ufw allow 22022/tcp
sudo ufw --force enable
```

## 📞 Support

For issues with the honeypot management system:

1. Check this guide first
2. Review log files in `/var/log/`
3. Test individual components
4. Use the troubleshooting procedures above
5. Check AWS CloudWatch for instance metrics

## 🔗 Related Documentation

- [Mini-XDR Setup Guide](../docs/SETUP_GUIDE.md)
- [Attack Testing Guide](../docs/ATTACK_TESTING_GUIDE.md)
- [Deployment Guide](../docs/DEPLOYMENT.md)
- [Cowrie Documentation](https://cowrie.readthedocs.io/)
- [AWS EC2 Documentation](https://docs.aws.amazon.com/ec2/)

---

**⚠️ Important Notes:**
- Always use `sudo` for service management commands
- Keep your SSH keys secure and properly permissioned
- Monitor AWS costs regularly
- Test changes in a development environment first
- Backup important configurations before making changes
