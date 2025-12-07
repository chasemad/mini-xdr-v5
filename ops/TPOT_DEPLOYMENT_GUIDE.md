# T-Pot Honeypot AWS Deployment Guide

## Overview

This guide provides instructions for deploying a T-Pot honeypot on AWS using the automated deployment script. T-Pot is a comprehensive honeypot platform that includes multiple honeypot services for advanced threat detection and analysis.

## Prerequisites

### System Requirements
- **AWS CLI** installed and configured
- **Sufficient AWS permissions** for EC2, VPC, and IAM operations
- **Instance Type**: t3.xlarge minimum (8GB RAM, 128GB SSD)
- **Your public IP address** for management access

### Mini-XDR Integration
- Mini-XDR backend running and ready to receive logs
- Port 8000 open on your Mini-XDR instance for log ingestion

## Quick Start

### 1. Run the Deployment Script
```bash
cd ./ops
./aws-tpot-honeypot-setup.sh
```

### 2. Wait for Installation
The script will:
- Create VPC infrastructure
- Launch EC2 instance with Debian 12
- Install T-Pot automatically (15-30 minutes)
- Configure security groups
- Set up log forwarding

### 3. Access T-Pot
After installation completes:
```bash
# Get credentials
~/.mini-xdr/tpot-management/get-credentials.sh

# Access web interface
~/.mini-xdr/tpot-management/web-access.sh

# SSH access
ssh -i ~/.ssh/mini-xdr-tpot-key.pem -p 64295 tpot@YOUR_INSTANCE_IP
```

## T-Pot Components

### Included Honeypots
- **Cowrie**: SSH/Telnet honeypot
- **Dionaea**: Multi-protocol honeypot (HTTP, FTP, TFTP, etc.)
- **Suricata**: Network IDS/IPS
- **Elasticpot**: Elasticsearch honeypot
- **Honeytrap**: Network honeypot
- **Mailoney**: SMTP honeypot
- **Rdpy**: RDP honeypot
- **Tanner**: Web application honeypot

### Monitoring & Analysis
- **Elasticsearch**: Log storage and indexing
- **Kibana**: Data visualization and analysis
- **Logstash**: Log processing and enrichment

## Network Configuration

### Security Groups
The deployment creates comprehensive security group rules:

#### Management Access (Restricted to Your IP)
- **Port 64295**: SSH management
- **Port 64297**: T-Pot web interface

#### Honeypot Services (Open to Internet)
- **Port 22**: SSH honeypot
- **Port 80/443**: Web honeypots
- **Port 21**: FTP honeypot
- **Port 25**: SMTP honeypot
- **Port 3306**: MySQL honeypot
- **Port 3389**: RDP honeypot
- **Ports 8000-9999**: Additional services

## Management Scripts

### Local Management Commands
```bash
# Check status
~/.mini-xdr/tpot-management/status.sh

# Connect via SSH
~/.mini-xdr/tpot-management/connect.sh

# Restart services
~/.mini-xdr/tpot-management/restart.sh

# Get login credentials
~/.mini-xdr/tpot-management/get-credentials.sh

# Open web interface
~/.mini-xdr/tpot-management/web-access.sh
```

### Remote Management Commands
```bash
# On the T-Pot instance
sudo /opt/mini-xdr-tpot/tpot-status.sh
sudo /opt/mini-xdr-tpot/tpot-restart.sh
sudo /opt/mini-xdr-tpot/tpot-stop.sh
sudo /opt/mini-xdr-tpot/tpot-start.sh
```

## Log Integration with Mini-XDR

### Fluent Bit Configuration
Logs are automatically forwarded to your Mini-XDR instance via Fluent Bit:

```
T-Pot Logs → Fluent Bit → Mini-XDR (Port 8000)
```

### Log Sources
- **Suricata**: Network intrusion detection alerts
- **Cowrie**: SSH/Telnet interaction logs
- **Dionaea**: Multi-protocol attack logs
- **Logstash**: Processed and enriched logs

### Mini-XDR Backend Configuration
Add to your Mini-XDR `.env` file:
```env
TPOT_HOST=YOUR_INSTANCE_IP
TPOT_SSH_PORT=64295
TPOT_WEB_PORT=64297
TPOT_SSH_KEY=~/.ssh/mini-xdr-tpot-key.pem
```

## Monitoring and Analysis

### Web Interface Access
1. Open: `https://YOUR_INSTANCE_IP:64297/`
2. Login with credentials from: `~/.mini-xdr/tpot-management/get-credentials.sh`
3. Navigate to Kibana for log analysis

### Key Dashboards
- **Attack Overview**: Real-time attack statistics
- **Geolocation**: Attack source mapping
- **Honeypot Activity**: Service-specific metrics
- **Network Analysis**: Traffic patterns and anomalies

## Testing the Deployment

### Basic Connectivity Tests
```bash
# Test SSH honeypot
ssh admin@YOUR_INSTANCE_IP

# Test web honeypot
curl http://YOUR_INSTANCE_IP/

# Test management access
ssh -i ~/.ssh/mini-xdr-tpot-key.pem -p 64295 tpot@YOUR_INSTANCE_IP
```

### Attack Simulation
```bash
# From your Mini-XDR system
python3 attack_simulation.py --target YOUR_INSTANCE_IP
```

## Troubleshooting

### Common Issues

#### T-Pot Installation Failed
```bash
# Check installation logs
ssh -i ~/.ssh/mini-xdr-tpot-key.pem -p 22 admin@YOUR_INSTANCE_IP
sudo tail -f /var/log/cloud-init-output.log
```

#### Services Not Starting
```bash
# Check Docker status
sudo systemctl status docker
sudo docker ps -a

# Restart T-Pot services
sudo /opt/mini-xdr-tpot/tpot-restart.sh
```

#### Web Interface Not Accessible
```bash
# Check if installation is complete
sudo docker ps | grep nginx

# Verify security group allows your IP
# Check AWS console or run deployment script again
```

#### Log Forwarding Issues
```bash
# Check Fluent Bit status
sudo systemctl status fluent-bit

# Test Mini-XDR connectivity
curl -X POST http://YOUR_MINI_XDR_IP:8000/ingest/multi \
  -H "Authorization: Bearer demo-tpot-api-key" \
  -H "Content-Type: application/json" \
  -d '{"test": "connectivity"}'
```

## Maintenance

### Regular Tasks
- **Monitor disk usage**: T-Pot generates significant logs
- **Update T-Pot**: Regular security updates
- **Review attack data**: Analyze captured threats
- **Backup configurations**: Save custom settings

### Log Rotation
Automatic log rotation is configured for:
- 30-day retention for honeypot logs
- Compressed storage for older logs
- Automatic cleanup of old data

## Security Considerations

### Access Control
- Management ports restricted to your IP only
- Strong auto-generated passwords
- SSH key-based authentication
- Regular security updates

### Data Privacy
- Logs contain attack data and may include sensitive information
- Ensure compliance with local data protection regulations
- Consider data anonymization for long-term storage

## Cost Optimization

### Instance Sizing
- **t3.xlarge**: ~$50-80/month (recommended)
- **t3.large**: ~$25-40/month (minimum viable)
- Additional costs: EIP (~$4/month), data transfer

### Cost Monitoring
```bash
# Check instance costs in AWS console
# Set up billing alerts
# Monitor data transfer costs
```

## Advanced Configuration

### Custom Honeypot Services
Edit `/opt/tpot/tpotce/docker-compose.yml` to:
- Enable/disable specific honeypots
- Modify port configurations
- Add custom honeypot containers

### Log Analysis Enhancement
- Configure additional Kibana dashboards
- Set up automated alerting
- Integrate with external SIEM systems

## Support and Resources

### Documentation
- [T-Pot Official Documentation](https://github.com/telekom-security/tpotce)
- [Mini-XDR Integration Guide](./ENHANCED_MCP_GUIDE.md)

### Community
- T-Pot GitHub Issues
- Security honeypot communities
- Mini-XDR project discussions

---

## Deployment Summary

The T-Pot deployment script provides:
- ✅ **Automated AWS infrastructure setup**
- ✅ **Complete T-Pot installation**
- ✅ **Security group configuration**
- ✅ **Log forwarding to Mini-XDR**
- ✅ **Management scripts and tools**
- ✅ **Auto-start capabilities**
- ✅ **Comprehensive monitoring**

Run `./aws-tpot-honeypot-setup.sh` to begin deployment!
