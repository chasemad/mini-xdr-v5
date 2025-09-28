# 📂 Mini-XDR Scripts Directory

This directory contains all operational scripts for the Mini-XDR system, organized by function for easy management and maintenance.

## Directory Structure

```
scripts/
├── 🚀 start-all.sh              # Start complete Mini-XDR system
├── 🛑 stop-all.sh               # Stop all Mini-XDR services  
├── 📊 system-status.sh          # Check system health and status
├── 📋 README.md                 # This documentation
├── 🔐 auth/                     # Authentication & security scripts
├── 📊 datasets/                 # Dataset download & processing
├── 🧠 ml-training/              # Machine learning & training
├── 🏗️ infrastructure/           # VM, networking & infrastructure
├── 🧪 testing/                  # Testing & validation scripts
├── ☁️ aws-deployment/           # AWS-specific deployment scripts
├── 🚨 attack-simulation/        # Attack testing and simulation
├── 🍯 tpot-management/          # T-Pot honeypot management
└── 🛠️ system-maintenance/       # System maintenance & troubleshooting
```

## Quick Start Guide

### 🚀 System Operations
```bash
# Start the complete Mini-XDR system
./start-all.sh

# Check system status
./system-status.sh

# Stop all services
./stop-all.sh
```

### 🚨 Attack Testing
```bash
# Quick attack test
cd attack-simulation
./quick_attack.sh 192.168.1.100

# Advanced attack chain simulation
cd testing
./simulate-advanced-attack-chain.sh
```

### 🧠 ML Training & Datasets
```bash
# Download and train with real datasets
cd datasets && python3 download-real-datasets.py --download-all
cd ../ml-training && python3 train-with-real-datasets.py

# Generate training data and optimize
cd ml-training
python3 generate-training-data.py --mode comprehensive
python3 optimize-training.py --mode continuous
```

### 🍯 T-Pot Honeypot Management
```bash
cd tpot-management

# Start T-Pot securely
./start-secure-tpot.sh

# Allow Kali testing access
./kali-access.sh add KALI_IP 22 80 443

# Deploy log forwarding
./deploy-tpot-logging.sh 34.193.101.171 YOUR_LOCAL_IP
```

### 🔐 Authentication & Security
```bash
# Generate agent credentials
cd auth && python3 mint_agent_cred.py

# Send authenticated requests  
python3 auth/send_signed_request.py --path /api/ml/status --method GET

# Secure homelab
./auth/homelab_lockdown.sh --apply
```

### 🛠️ System Maintenance
```bash
cd system-maintenance

# Fix dependency issues
./fix_dependencies.sh
```

## Script Categories

### 🔐 Authentication & Security (`auth/`)
**Purpose**: Manage authentication, credentials, and security configurations
- `agent_auth.py` - HMAC authentication utilities for agents
- `mint_agent_cred.py` - Generate new agent credentials
- `send_signed_request.py` - Send authenticated API requests
- `homelab_lockdown.sh` - Network security lockdown for homelab

### 📊 Dataset Management (`datasets/`)
**Purpose**: Download, process, and convert cybersecurity datasets
- `download-*-datasets.py` - Various dataset downloaders (CICIDS2017, real-world, etc.)
- `enhanced-cicids-processor.py` - Enhanced CICIDS2017 processing
- `enhanced-threat-feeds.py` - Live threat intelligence downloader
- `process-cicids2017-ml.py` - Official CICIDS2017 MachineLearningCSV processor

### 🧠 ML Training (`ml-training/`)
**Purpose**: Train and optimize machine learning models
- `massive-dataset-trainer.py` - Train with ALL available datasets
- `train-with-real-datasets.py` - Enhanced training with real-world data
- `generate-training-data.py` - Synthetic training data generator
- `optimize-training.py` - Training optimization and scheduling
- `import-historical-data.py` - Import existing logs for training

### 🏗️ Infrastructure (`infrastructure/`)
**Purpose**: VM management, networking, and infrastructure setup
- `find-vm-ip.sh` - VM IP discovery script
- `fix-vmware-networking.sh` - VMware networking diagnostics
- `setup.sh` - Complete Mini-XDR system setup
- `ssh-*.sh` - SSH connectivity utilities for various environments
- `setup-*-relay.sh` - AWS relay setup scripts

### 🧪 Testing & Validation (`testing/`)
**Purpose**: Test detection capabilities and validate system functionality
- `simple-test-adaptive.sh` - Basic adaptive detection testing
- `simulate-advanced-attack-chain.sh` - Multi-phase APT-style attack simulation
- `verify_ip_blocks.py` - IP block verification on honeypot

### ☁️ AWS Deployment (`aws-deployment/`)
**Purpose**: AWS-specific deployment and security management
- `secure-tpot-for-testing.sh` - Lock down T-Pot for safe testing
- `open-tpot-to-internet.sh` - Expose T-Pot to real internet attacks

### 🚨 Attack Simulation (`attack-simulation/`)
**Purpose**: Test Mini-XDR detection and response capabilities
- `attack_simulation.py` - Comprehensive multi-vector attack simulator
- `simple_attack_test.py` - Quick focused attack validation
- `multi_ip_attack.sh` - Advanced multi-source attack simulation
- `simple_multi_ip_attack.sh` - Quick multi-IP attack test
- `quick_attack.sh` - Rapid attack sequence for immediate testing

### 🍯 T-Pot Management (`tpot-management/`)
**Purpose**: Manage T-Pot honeypot deployment and security
- `setup-tpot-integration.sh` - Complete T-Pot integration setup
- `start-secure-tpot.sh` - Secure T-Pot startup script
- `secure-tpot.sh` - T-Pot security hardening (already applied)
- `kali-access.sh` - Kali machine access control
- `deploy-tpot-logging.sh` - Log forwarding deployment

### 🛠️ System Maintenance (`system-maintenance/`)
**Purpose**: Maintain and troubleshoot Mini-XDR system
- `fix_dependencies.sh` - Phase 2B dependencies fix and installation

## Common Workflows

### 🔄 Daily Operations
```bash
# 1. Start system
./start-all.sh

# 2. Check status
./system-status.sh

# 3. Run attack tests
cd attack-simulation && ./quick_attack.sh localhost

# 4. Check results in dashboard at http://localhost:3000
```

### 🧠 ML Training Workflow
```bash
# 1. Download real datasets
cd datasets && python3 download-real-datasets.py --download-all

# 2. Train enhanced models
cd ../ml-training && python3 train-with-real-datasets.py

# 3. Optimize training
python3 optimize-training.py --mode continuous --duration 30

# 4. Test adaptive detection
cd ../testing && ./simple-test-adaptive.sh
```

### 🧪 T-Pot Testing Session
```bash
cd tpot-management

# 1. Start T-Pot
./start-secure-tpot.sh

# 2. Deploy logging
./deploy-tpot-logging.sh 34.193.101.171 $(ifconfig | grep "inet " | grep -v 127.0.0.1 | awk '{print $2}' | head -1)

# 3. Allow Kali access
./kali-access.sh add $(curl -s -4 icanhazip.com) 22 80 443

# 4. Run attacks from Kali
cd ../attack-simulation
python3 simple_attack_test.py 34.193.101.171

# 5. Test advanced detection
cd ../testing && ./simulate-advanced-attack-chain.sh

# 6. Remove access when done
cd ../tpot-management
./kali-access.sh remove $(curl -s -4 icanhazip.com) 22 80 443
```

### 🔧 Troubleshooting Session
```bash
# 1. Fix dependencies
cd system-maintenance && ./fix_dependencies.sh

# 2. Fix infrastructure issues
cd ../infrastructure && ./fix-vmware-networking.sh

# 3. Restart system
cd .. && ./stop-all.sh && ./start-all.sh

# 4. Verify functionality
./system-status.sh

# 5. Test detection capabilities
cd testing && ./simple-test-adaptive.sh
```

## Security Considerations

### 🔒 T-Pot Security Status
- **Public Access**: ❌ BLOCKED (all honeypot ports secured)
- **Management**: ✅ YOUR IP ONLY (SSH/Web interface)
- **Testing Access**: 🎯 CONTROLLED (via kali-access.sh)

### ⚠️ Attack Script Safety
- Only use against systems you own or have permission to test
- Attack scripts generate real malicious traffic
- Always inform security teams before testing
- Follow responsible disclosure practices

### 🛡️ Best Practices
- Always remove Kali access after testing
- Monitor AWS costs when T-Pot is running
- Regularly review security group rules
- Keep API keys secure and rotate them monthly

## Integration Points

### 📊 Mini-XDR Dashboard
- **URL**: http://localhost:3000
- **SOC Interface**: Real-time incident monitoring
- **Analytics**: ML model performance and drift detection
- **3D Visualization**: Interactive threat landscape

### 📡 Log Flow Architecture
```
Attack Scripts → Mini-XDR → Incident Detection
T-Pot Honeypot → Fluent Bit → Mini-XDR → ML Analysis
External Threats → T-Pot → Log Processing → SOC Dashboard
```

### 🤖 AI Agent Integration
- **Detection**: Automated threat identification
- **Analysis**: ML-powered incident analysis  
- **Response**: Autonomous containment actions
- **Learning**: Continuous model improvement

## Support and Documentation

### 📚 Detailed Documentation
- Each script directory contains detailed README.md
- Individual scripts have built-in help (`--help` flag)
- Configuration files include inline documentation

### 🆘 Getting Help
- Check script-specific README files
- Use `--help` flag on Python scripts
- Review logs in `/var/log/mini-xdr/`
- Check system status with `./system-status.sh`

### 🐛 Issue Reporting
- Include output of `./system-status.sh`
- Provide relevant log excerpts
- Describe steps to reproduce
- Include system configuration details

---

## 🗂️ Navigation Guide

- **🚀 Core Operations**: Root directory (`start-all.sh`, `stop-all.sh`, `system-status.sh`)
- **🔐 Security & Auth**: `auth/` - Credentials, HMAC signing, security lockdown
- **📊 Data Management**: `datasets/` - Download, process cybersecurity datasets
- **🧠 ML & Training**: `ml-training/` - Model training, optimization, data generation
- **🏗️ Infrastructure**: `infrastructure/` - VM setup, networking, deployment setup
- **🧪 Testing**: `testing/` - Detection testing, validation, verification
- **☁️ AWS**: `aws-deployment/` - AWS-specific T-Pot security management
- **🚨 Attack Testing**: `attack-simulation/` - Multi-vector attack simulations
- **🍯 Honeypot Mgmt**: `tpot-management/` - T-Pot deployment and control
- **🛠️ Maintenance**: `system-maintenance/` - Dependencies and troubleshooting

Each subdirectory contains detailed README.md files with specific usage instructions.

---

**Organization Status**: ✅ **COMPLETELY REORGANIZED**  
**Root Directory**: 🧹 **CLEANED & STRUCTURED**  
**Script Locations**: 📁 **CATEGORIZED BY PURPOSE**  
**AWS Deployment**: 🚀 **READY**

**Last Updated**: September 27, 2025  
**Maintained by**: Mini-XDR Operations Team