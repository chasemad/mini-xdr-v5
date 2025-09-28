# ☁️ AWS Deployment Scripts

Scripts for managing AWS-specific deployments, including T-Pot honeypot security and internet exposure controls.

## Scripts Overview

### 🔒 T-Pot Security Management

#### `secure-tpot-for-testing.sh`
**Secure T-Pot for controlled testing**
- **Purpose**: Restrict T-Pot access to your IP only for safe testing
- **Usage**: `./secure-tpot-for-testing.sh`
- **Features**: Emergency security lockdown, removes all public access, controlled testing setup

#### `open-tpot-to-internet.sh`
**Open T-Pot to real internet attacks**
- **Purpose**: Expose honeypot to real attackers worldwide for live data collection
- **Usage**: `./open-tpot-to-internet.sh`
- **Features**: Internet exposure, real attack collection, security warnings

## Security Models

### 🔒 Secure Testing Mode (Default)
```
Internet → AWS Security Groups → BLOCKED
Your IP → AWS Security Groups → ✅ ALLOWED
Kali IP → AWS Security Groups → 🎯 CONTROLLED (via kali-access.sh)
```

### 🌍 Internet Collection Mode  
```
Internet → AWS Security Groups → ✅ OPEN (honeypot ports)
Management → AWS Security Groups → 🔒 YOUR IP ONLY
```

## Usage Examples

### Safe Testing (Recommended)
```bash
# Ensure honeypot is locked down for testing
./aws-deployment/secure-tpot-for-testing.sh

# Verify security status
aws ec2 describe-security-groups --group-ids sg-037bd4ee6b74489b5

# Allow specific Kali access for controlled testing
cd ../tpot-management
./kali-access.sh add KALI_IP 22 80 443
```

### Live Attack Collection (Advanced)
```bash
# ⚠️ WARNING: This exposes honeypot to real attackers!
./aws-deployment/open-tpot-to-internet.sh

# Monitor real attacks
tail -f ../../backend/backend.log

# View on globe visualization
# http://localhost:3000/visualizations

# Return to secure mode when done
./secure-tpot-for-testing.sh
```

## Security Architecture

### Current Configuration
- **T-Pot Instance**: i-0a1b2c3d4e5f6g7h8
- **Security Group**: sg-037bd4ee6b74489b5
- **Management SSH**: Port 64295 (your IP only)
- **Web Interface**: Port 64297 (your IP only)
- **Honeypot Ports**: 21, 22, 23, 25, 80, 443, etc.

### AWS Resources
- **Region**: us-east-1
- **Instance Type**: t3.medium
- **Operating System**: Debian-based T-Pot
- **Storage**: 50GB EBS volume

## Cost Management

### Running Costs
- **T-Pot Instance**: ~$25-40/month when running
- **EBS Storage**: ~$5/month (persistent)
- **Data Transfer**: Variable based on attack volume

### Cost Optimization
```bash
# Stop instance when not testing (saves ~80% of costs)
aws ec2 stop-instances --instance-ids i-0a1b2c3d4e5f6g7h8

# Start when needed
cd ../tpot-management && ./start-secure-tpot.sh

# Monitor costs
aws ce get-cost-and-usage --time-period Start=2025-09-01,End=2025-09-30 --granularity MONTHLY --metrics BlendedCost
```

## Deployment Scenarios

### 📋 Development/Testing
- Use `secure-tpot-for-testing.sh`
- Allow only your IP and test IPs
- Focus on controlled attack testing

### 🌍 Research/Collection
- Use `open-tpot-to-internet.sh`
- Monitor costs and attack volume
- Collect real-world threat intelligence

### 🎓 Training/Demos
- Use secure mode with specific access
- Show real attack data from previous collections
- Demonstrate containment capabilities

## Integration with Mini-XDR

### Data Flow
```
Real Attackers → T-Pot → Fluent Bit → Mini-XDR → ML Models → Incidents → Dashboard
```

### Log Sources
- **Cowrie**: SSH/Telnet honeypot
- **Dionaea**: Multi-protocol honeypot  
- **Suricata**: Network IDS
- **Honeytrap**: Network honeypot
- **ElasticPot**: Elasticsearch honeypot

## Safety Guidelines

### ⚠️ Important Warnings
- **Only expose to internet when actively monitoring**
- **Real attacks generate significant log volume**
- **Monitor AWS costs when exposed to internet**
- **Always return to secure mode after testing**

### Best Practices
- Schedule internet exposure for specific timeframes
- Monitor attack volume and costs
- Keep management access restricted to your IP
- Regular security group audits
- Rotate API keys monthly

---

**Status**: Production Ready  
**Security Level**: High  
**Last Updated**: September 27, 2025  
**Maintained by**: Mini-XDR AWS Team
