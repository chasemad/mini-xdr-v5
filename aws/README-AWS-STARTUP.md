# 🛡️ Mini-XDR AWS Startup Guide

## Quick Commands

```bash
# Check status of all instances
./start-mini-xdr-aws.sh status

# Start all instances in testing mode (safe)
./start-mini-xdr-aws.sh testing

# Start in LIVE mode (requires confirmation)
./start-mini-xdr-aws.sh live

# Stop all instances safely
./start-mini-xdr-aws.sh stop

# Development - backend only
./start-mini-xdr-aws.sh --backend-only

# Dry run to see what would happen
./start-mini-xdr-aws.sh --dry-run testing
```

## Current Infrastructure

| Instance | ID | Public IP | Purpose | Status |
|----------|----|-----------|---------| -------|
| Backend | i-0f0bcdd3762243393 | 98.81.155.222 | Main API & ML Engine | ✅ Running |
| Relay | i-0394a885327642dee | 18.204.222.38 | Data forwarding | ✅ Running |
| T-Pot | i-091156c8c15b7ece4 | 34.193.101.171 | Honeypot | 🛑 Stopped (Safe) |

## Access Information

- **🎯 Frontend Dashboard**: http://98.81.155.222:3000 (Main UI)
- **🔧 Backend API**: http://98.81.155.222:8000
- **📊 Health Check**: http://98.81.155.222:8000/health
- **📋 API Documentation**: http://98.81.155.222:8000/docs
- **🍯 T-Pot SSH**: `ssh -i ~/.ssh/mini-xdr-tpot-key.pem -p 64295 admin@34.193.101.171`

## Safety Features

### Testing Mode (Default)
- T-Pot configured for limited exposure
- Safe for development and testing
- All security validations enabled
- No real internet attacks

### Live Mode
- **⚠️ DANGER**: Exposes T-Pot to real attacks
- Requires explicit confirmation
- Enhanced monitoring needed
- Emergency shutdown procedures ready

## Script Options

| Option | Description |
|--------|-------------|
| `--help` | Show help message |
| `--dry-run` | Simulate without making changes |
| `--force` | Skip confirmation prompts |
| `--backend-only` | Start only backend instance |
| `--verbose` | Enable detailed logging |
| `--skip-security-check` | Skip security validation (not recommended) |

## Security Validations

The script automatically checks:
- ✅ Security group configurations
- ✅ SSH key accessibility
- ✅ AWS credential validity
- ✅ No overly permissive 0.0.0.0/0 rules
- ✅ Instance connectivity

## Connectivity Tests

Automatic testing includes:
- ✅ Backend API health endpoint
- ✅ T-Pot SSH accessibility
- ✅ Relay server connectivity
- ✅ Service startup verification

## Example Workflow

### 1. Development Testing
```bash
# Check current status
./start-mini-xdr-aws.sh status

# Start backend only for development
./start-mini-xdr-aws.sh --backend-only

# Full testing environment
./start-mini-xdr-aws.sh testing
```

### 2. Production Deployment
```bash
# Dry run first
./start-mini-xdr-aws.sh --dry-run live

# Real deployment (requires confirmation)
./start-mini-xdr-aws.sh live
```

### 3. Emergency Shutdown
```bash
# Stop everything immediately
./start-mini-xdr-aws.sh stop
```

## Log Files

All operations are logged to `/tmp/mini-xdr-aws-startup-YYYYMMDD-HHMMSS.log`

## Troubleshooting

### Instance Won't Start
```bash
# Check AWS instance status directly
aws ec2 describe-instances --instance-ids i-0f0bcdd3762243393

# Force start with detailed logging
./start-mini-xdr-aws.sh --verbose --force testing
```

### Connectivity Issues
```bash
# Skip security checks if needed
./start-mini-xdr-aws.sh --skip-security-check testing

# Check logs
tail -f /tmp/mini-xdr-aws-startup-*.log
```

### T-Pot Problems
```bash
# Start without T-Pot
./start-mini-xdr-aws.sh --backend-only

# Manual T-Pot connection test
ssh -i ~/.ssh/mini-xdr-tpot-key.pem -p 64295 admin@34.193.101.171
```

## Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   T-Pot         │───►│  Mini-XDR       │───►│  AI Agents      │
│  (Stopped)      │    │   Backend       │    │  Orchestrator   │
│ 34.193.101.171  │    │ 98.81.155.222   │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Attack Data    │    │   ML Engine     │    │  Response       │
│   Collection    │    │ + Detection     │    │  Actions        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Support

For issues or questions:
1. Check the log files first
2. Try `--dry-run` mode to test
3. Use `--verbose` for detailed output
4. Verify AWS credentials and permissions