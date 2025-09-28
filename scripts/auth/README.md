# 🔐 Authentication & Security Scripts

Scripts for managing authentication, credentials, and security configurations in Mini-XDR.

## Scripts Overview

### 🔑 Credential Management

#### `agent_auth.py`
**HMAC authentication utilities for agents**
- **Purpose**: Load agent credentials and build signed request headers
- **Usage**: Imported by other scripts for API authentication
- **Features**: HMAC signing, credential loading from environment

#### `mint_agent_cred.py`
**Generate new agent credentials**
- **Purpose**: Create new device credentials for HMAC authentication
- **Usage**: `python3 mint_agent_cred.py [ttl_days]`
- **Features**: UUID generation, HMAC key creation, database storage

#### `send_signed_request.py`
**Send authenticated API requests**
- **Purpose**: Make signed HTTP requests to Mini-XDR API
- **Usage**: `python3 send_signed_request.py --path /api/endpoint [options]`
- **Features**: HMAC signing, multiple HTTP methods, JSON payloads

### 🔒 Security Configuration

#### `homelab_lockdown.sh`
**Network security lockdown for homelab**
- **Purpose**: Restrict network access to loopback interface only
- **Usage**: `./homelab_lockdown.sh [--apply]`
- **Features**: Cross-platform firewall rules, dry-run mode, security hardening

## Usage Examples

### Generate Agent Credentials
```bash
# Create new agent credentials (90-day expiry)
python3 auth/mint_agent_cred.py

# Create credentials with custom TTL
python3 auth/mint_agent_cred.py 30
```

### Send Authenticated Requests
```bash
# Test API endpoint
python3 auth/send_signed_request.py --path /health --method GET

# Send event data
python3 auth/send_signed_request.py --path /ingest/multi --body '{"events":[...]}'

# Use body from file
python3 auth/send_signed_request.py --path /api/ml/retrain --body-file event_data.json
```

### Secure Homelab
```bash
# Preview security changes (dry run)
./auth/homelab_lockdown.sh

# Apply security lockdown
./auth/homelab_lockdown.sh --apply
```

## Integration

These scripts are used throughout the Mini-XDR system:
- **Agent Authentication**: Used by all API clients
- **Credential Management**: For setting up new agents
- **Security**: For production deployment hardening

---

**Status**: Production Ready  
**Last Updated**: September 27, 2025  
**Maintained by**: Mini-XDR Security Team
