# 🏗️ Infrastructure Management Scripts

Scripts for managing infrastructure, virtual machines, networking, and deployment setup for Mini-XDR.

## Scripts Overview

### 🖥️ Virtual Machine Management

#### `find-vm-ip.sh`
**VM IP discovery script**
- **Purpose**: Automatically discover honeypot VM IP addresses
- **Usage**: `./find-vm-ip.sh`
- **Features**: VMware network scanning, SSH key testing, configuration updates

#### `fix-vmware-networking.sh`
**VMware networking diagnostic**
- **Purpose**: Diagnose and fix VMware Fusion networking issues
- **Usage**: `./fix-vmware-networking.sh`
- **Features**: Network interface detection, connectivity testing, troubleshooting guide

### 🔗 SSH & Connectivity

#### `ssh-native.sh`
**Native Terminal SSH execution**
- **Purpose**: Execute SSH commands through macOS Terminal for Cursor compatibility
- **Usage**: `./ssh-native.sh <ssh_arguments>`
- **Features**: Terminal integration, networking issue workarounds

#### `ssh-wrapper.sh`
**SSH wrapper for Cursor terminal**
- **Purpose**: SSH compatibility wrapper for Cursor's integrated terminal
- **Usage**: Direct SSH execution with proper environment setup
- **Features**: SSH key loading, environment variable export

### ☁️ AWS Infrastructure

#### `setup-aws-mini-xdr-relay.sh`
**AWS relay setup for T-Pot connectivity**
- **Purpose**: Create AWS-based relay to enable T-Pot connectivity
- **Usage**: `./setup-aws-mini-xdr-relay.sh`
- **Features**: AWS instance creation, security group setup, relay service deployment

#### `setup-relay-fixed.sh`
**Fixed AWS relay setup**
- **Purpose**: AWS relay setup with proper VPC and IPv4 handling
- **Usage**: `./setup-relay-fixed.sh`
- **Features**: VPC configuration, IPv4 handling, improved networking

### 🛠️ System Setup

#### `setup.sh`
**Mini-XDR complete setup script**
- **Purpose**: Complete system setup and dependency installation
- **Usage**: `./setup.sh`
- **Features**: Virtual environment setup, dependency installation, database initialization

## Usage Examples

### VM Discovery & Setup
```bash
# Find your honeypot VM
./infrastructure/find-vm-ip.sh

# Fix VMware networking issues
./infrastructure/fix-vmware-networking.sh

# Update configuration with discovered IP
# (Script will provide specific commands)
```

### SSH Connectivity
```bash
# Test SSH with native terminal (if Cursor has issues)
./infrastructure/ssh-native.sh -p 22022 -i ~/.ssh/key user@host

# Use SSH wrapper for improved compatibility
./infrastructure/ssh-wrapper.sh -p 22022 -i ~/.ssh/key user@host
```

### AWS Relay Setup
```bash
# Set up AWS relay for T-Pot
./infrastructure/setup-relay-fixed.sh

# Check relay status
aws ec2 describe-instances --filters "Name=tag:Name,Values=mini-xdr-relay"
```

### Initial System Setup
```bash
# Complete Mini-XDR setup
./infrastructure/setup.sh

# This will:
# - Install Python/Node dependencies
# - Create virtual environments
# - Initialize database
# - Set up configuration files
```

## Configuration Files

### Generated Configurations
- **SSH Keys**: `~/.ssh/` directory
- **Environment Files**: `backend/.env`, `frontend/.env.local`
- **AWS Configurations**: Various temporary files

### Network Settings
- **VMware Networks**: 192.168.238.x, 192.168.56.x, 172.16.x.x
- **Default Honeypot**: 10.0.0.23:22022
- **AWS Relay**: Auto-configured based on deployment

## Troubleshooting

### Common VM Issues
```bash
# VM not found
./find-vm-ip.sh  # Scans common network ranges

# Networking problems
./fix-vmware-networking.sh  # Diagnoses VMware issues
```

### SSH Connection Issues
```bash
# Use native terminal if Cursor has networking issues
./ssh-native.sh -p 22022 -i ~/.ssh/key user@host

# Or use the wrapper
./ssh-wrapper.sh -p 22022 -i ~/.ssh/key user@host
```

### AWS Relay Issues
```bash
# Check if relay is running
aws ec2 describe-instances --filters "Name=tag:Name,Values=mini-xdr-relay"

# SSH to relay for debugging
ssh -i ~/.ssh/mini-xdr-tpot-key.pem ubuntu@RELAY_IP
```

## Integration Points

### With Main System
- **VM Discovery**: Updates backend/.env automatically
- **SSH Setup**: Provides keys for responder module
- **AWS Integration**: Enables T-Pot connectivity

### With Other Scripts
- **Auth Scripts**: Use agent_auth.py for authentication
- **T-Pot Scripts**: Use SSH utilities for connectivity
- **Training Scripts**: Rely on proper setup for data flow

---

**Status**: Production Ready  
**Last Updated**: September 27, 2025  
**Maintained by**: Mini-XDR Infrastructure Team
