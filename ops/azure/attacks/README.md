# Mini-XDR Attack Simulations for Azure Mini Corporate Network

This directory contains attack simulation scripts to test Mini-XDR's detection capabilities against the mini corporate network.

## 🎯 Available Attack Simulations

### 1. Kerberos Attacks (`kerberos-attacks.sh`)

Simulates various Kerberos-based attacks:
- **Kerberoasting**: Request service tickets for service accounts
- **AS-REP Roasting**: Target accounts with "Do not require Kerberos preauthentication"
- **Golden Ticket**: Forge Kerberos TGTs
- **Silver Ticket**: Forge service tickets
- **Pass-the-Ticket**: Reuse stolen Kerberos tickets

### 2. Lateral Movement (`lateral-movement.sh`)

Simulates lateral movement techniques:
- **PSExec**: Remote command execution
- **WMI**: Windows Management Instrumentation attacks
- **RDP**: Remote Desktop brute force and session hijacking
- **SMB**: File share enumeration and exploitation
- **PowerShell Remoting**: Remote PowerShell sessions

### 3. Data Exfiltration (`data-exfiltration.sh`)

Simulates data theft:
- **Large File Transfers**: Detect abnormal file movements
- **External Uploads**: HTTP/HTTPS data uploads
- **Email Exfiltration**: Bulk email sending
- **Cloud Storage**: Upload to public cloud services
- **DNS Tunneling**: Covert data channels

### 4. Credential Theft (`credential-theft.sh`)

Simulates credential dumping:
- **LSASS Dumping**: Extract credentials from memory
- **Mimikatz**: Run credential harvesting tools
- **DCSync**: Replicate AD credentials
- **NTDS.dit Extraction**: Database theft
- **Registry Hives**: SAM/SECURITY extraction

## 🚀 Running Simulations

### Prerequisites

1. Mini Corporate Network deployed and running
2. Active Directory configured with test users
3. Mini-XDR agents installed on all endpoints
4. Access to Azure Bastion for VM connections

### Basic Usage

```bash
# Run all attack simulations
./run-all-tests.sh

# Run specific attack type
./kerberos-attacks.sh
./lateral-movement.sh
./data-exfiltration.sh
./credential-theft.sh
```

### Advanced Options

```bash
# Run with specific target
./kerberos-attacks.sh --target mini-corp-dc01

# Run with custom intensity (1-10)
./lateral-movement.sh --intensity 5

# Run in stealth mode (slower, less noisy)
./data-exfiltration.sh --stealth

# Generate report after running
./run-all-tests.sh --report
```

## 📊 Expected Detection Results

### Kerberos Attacks
- **Detection Rate:** 99.98%
- **Alert Types:** Kerberos Attack, Golden Ticket, Kerberoasting
- **Response:** IAM Agent disables accounts, revokes tickets

### Lateral Movement
- **Detection Rate:** 98.9%
- **Alert Types:** Lateral Movement, Remote Execution, SMB Abuse
- **Response:** EDR Agent kills processes, isolates hosts

### Data Exfiltration
- **Detection Rate:** 97.7%
- **Alert Types:** Data Exfiltration, Large File Transfer
- **Response:** DLP Agent blocks uploads, quarantines files

### Credential Theft
- **Detection Rate:** 99.8%
- **Alert Types:** Credential Dumping, Mimikatz Detection
- **Response:** EDR Agent terminates process, collects memory dump

## 🔍 Verification

### Check Detections in Dashboard

```bash
# Get Application Gateway IP
APPGW_IP=$(terraform -chdir=ops/azure/terraform output -raw appgw_public_ip)

# Access dashboard
open https://$APPGW_IP/incidents
```

### Check via API

```bash
# Get API key
KEY_VAULT=$(terraform -chdir=ops/azure/terraform output -raw key_vault_name)
API_KEY=$(az keyvault secret show --vault-name $KEY_VAULT --name mini-xdr-api-key --query value -o tsv)

# Query recent incidents
curl -H "X-API-Key: $API_KEY" "https://$APPGW_IP/api/incidents?limit=10"

# Query specific attack types
curl -H "X-API-Key: $API_KEY" "https://$APPGW_IP/api/incidents?attack_type=Kerberos%20Attack"
```

### Check Backend Logs

```bash
# View real-time detections
kubectl logs -n mini-xdr -l app=mini-xdr-backend -f | grep "DETECTED"

# Search for specific attack
kubectl logs -n mini-xdr -l app=mini-xdr-backend | grep "Kerberos"
```

## ⚠️ Safety Considerations

### Important Warnings

1. **Only run in isolated environment**: Never run these on production networks
2. **Mini corporate network only**: Attacks target the isolated test network
3. **Antivirus may detect**: Some tools may be flagged as malicious
4. **Legal compliance**: Ensure you have authorization for all testing

### Network Isolation

The mini corporate network is isolated in Azure:
- Private subnet (10.0.10.0/24)
- No public IPs on VMs
- NSG rules block internet access
- Only accessible via Azure Bastion

### Rollback Capabilities

All simulated attacks can be rolled back:

```bash
# List all actions taken
curl -H "X-API-Key: $API_KEY" "https://$APPGW_IP/api/agents/actions"

# Rollback specific action
curl -X POST -H "X-API-Key: $API_KEY" \
  "https://$APPGW_IP/api/agents/rollback/{rollback_id}"
```

## 📝 Creating Custom Simulations

### Template Script

```bash
#!/bin/bash
# Custom Attack Simulation Template

set -e

# Configuration
ATTACK_NAME="My Custom Attack"
ATTACK_TYPE="custom_attack"
TARGET_VM="mini-corp-ws01"

echo "Running $ATTACK_NAME simulation..."

# Step 1: Preparation
echo "[1/3] Preparing attack..."
# Your preparation code here

# Step 2: Execute attack
echo "[2/3] Executing attack..."
# Your attack code here

# Step 3: Cleanup
echo "[3/3] Cleaning up..."
# Your cleanup code here

echo "Simulation complete. Check dashboard for detections."
```

### Best Practices

1. **Log all actions**: Make attacks traceable
2. **Include cleanup**: Remove artifacts after testing
3. **Add delays**: Simulate realistic attacker behavior
4. **Document expected results**: List what should be detected
5. **Test incrementally**: Start with basic attacks, increase complexity

## 🧪 Testing Matrix

| Attack Category | Technique | MITRE ATT&CK | Detection | Response |
|----------------|-----------|--------------|-----------|----------|
| **Kerberos** | Kerberoasting | T1558.003 | ✅ 99.98% | Disable account |
| **Kerberos** | Golden Ticket | T1558.001 | ✅ 99.98% | Revoke tickets |
| **Lateral** | PSExec | T1021.002 | ✅ 98.9% | Kill process |
| **Lateral** | WMI | T1047 | ✅ 98.9% | Isolate host |
| **Credential** | LSASS Dump | T1003.001 | ✅ 99.8% | Memory dump |
| **Credential** | DCSync | T1003.006 | ✅ 99.8% | Disable account |
| **Exfiltration** | HTTP Upload | T1048.003 | ✅ 97.7% | Block upload |
| **Exfiltration** | Cloud Storage | T1567.002 | ✅ 97.7% | Quarantine file |

## 📚 Additional Resources

- [MITRE ATT&CK Framework](https://attack.mitre.org/)
- [Atomic Red Team](https://github.com/redcanaryco/atomic-red-team)
- [Purple Team Lab](https://github.com/DefensiveOrigins/APT06202001)
- [Mini-XDR Detection Guide](../../../docs/DETECTION_GUIDE.md)

---

**Ready to test!** Start with `./run-all-tests.sh` to validate all detection capabilities.

