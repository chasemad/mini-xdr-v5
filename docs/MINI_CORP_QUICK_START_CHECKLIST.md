# 🚀 Mini Corp Quick Start Checklist

**Use this checklist to track progress through the 3-week deployment**

---

## Week 1: ML Model Enhancement ✅ Days 1-7

### Day 1-2: Data Collection
- [ ] Create `/datasets/windows_ad_datasets/` directory
- [ ] Download ADFA-LD dataset (Windows attacks)
- [ ] Download OpTC dataset (DARPA lateral movement)
- [ ] Download Mordor datasets (Kerberos attacks)
- [ ] Download EVTX attack samples (Mimikatz, PSExec)
- [ ] Verify: 11,000+ samples downloaded

### Day 3: Data Conversion
- [ ] Create `scripts/data-processing/convert_windows_datasets.py`
- [ ] Convert ADFA-LD to Mini-XDR format
- [ ] Convert OpTC to Mini-XDR format
- [ ] Convert Mordor to Mini-XDR format
- [ ] Convert EVTX samples to Mini-XDR format
- [ ] Verify: All datasets in consistent JSON format

### Day 4: Dataset Merging
- [ ] Merge honeypot data (988 existing) + Windows data (11,000 new)
- [ ] Balance class distribution (12,000 total samples)
- [ ] Create training/validation split (80/20)
- [ ] Save to `datasets/combined_enterprise_training.json`
- [ ] Verify: 12 attack classes present

### Day 5: Model Training
- [ ] Train 13-class enterprise model (3-4 hours)
- [ ] Train Kerberos specialist model (20 min)
- [ ] Train lateral movement specialist (20 min)
- [ ] Train credential theft specialist (20 min)
- [ ] Train insider threat specialist (20 min)
- [ ] Verify: All models ≥85% accuracy

### Day 6: Model Integration
- [ ] Create `backend/app/feature_extractor.py`
- [ ] Update `backend/app/ml_engine.py` for 13 classes
- [ ] Add Windows event feature extraction
- [ ] Deploy models to `models/enterprise/`
- [ ] Test model loading and inference
- [ ] Verify: Models load successfully

### Day 7: End-to-End Testing
- [ ] Create test script `scripts/testing/test_enterprise_detection.py`
- [ ] Test Kerberos attack detection
- [ ] Test lateral movement detection
- [ ] Test credential theft detection
- [ ] Test normal traffic (no false positives)
- [ ] Verify: 100% detection, <5% false positives

**Week 1 Complete:** ML models ready for corporate threats ✅

---

## Week 2: Enterprise Agent Development ✅ Days 8-14

### Day 8: IAM Agent - Setup
- [ ] Create `backend/app/agents/iam_agent.py`
- [ ] Implement IAMAgent class (base structure)
- [ ] Add LDAP connection to Active Directory
- [ ] Add KerberosAnalyzer class
- [ ] Test AD connectivity
- [ ] Verify: Can connect to AD and query users

### Day 9: IAM Agent - Detection
- [ ] Implement `analyze_authentication_event()`
- [ ] Implement `detect_kerberos_attack()`
- [ ] Implement `detect_privilege_escalation()`
- [ ] Add impossible travel detection
- [ ] Add off-hours access detection
- [ ] Add brute force detection
- [ ] Verify: All detection methods functional

### Day 10: IAM Agent - Response
- [ ] Implement `disable_user_account()`
- [ ] Implement `revoke_kerberos_tickets()`
- [ ] Implement `quarantine_user()`
- [ ] Implement `reset_user_password()`
- [ ] Implement `enforce_mfa()`
- [ ] Add API endpoints to `main.py`
- [ ] Create test script `scripts/testing/test_iam_agent.sh`
- [ ] Verify: All actions work against test AD

### Day 11: EDR Agent - Setup
- [ ] Create `backend/app/agents/edr_agent.py`
- [ ] Implement EDRAgent class
- [ ] Add Windows remote execution (WinRM/PowerShell)
- [ ] Add process monitoring logic
- [ ] Test remote command execution
- [ ] Verify: Can execute commands on Windows endpoints

### Day 12: EDR Agent - Actions
- [ ] Implement `kill_process()`
- [ ] Implement `quarantine_file()`
- [ ] Implement `collect_memory_dump()`
- [ ] Implement `isolate_host()` (network level)
- [ ] Implement `analyze_process_behavior()`
- [ ] Add API endpoints to `main.py`
- [ ] Create test script `scripts/testing/test_edr_agent.sh`
- [ ] Verify: All actions work against test Windows VM

### Day 13: DLP Agent
- [ ] Create `backend/app/agents/dlp_agent.py`
- [ ] Implement DLPAgent class
- [ ] Add data classification engine
- [ ] Add exfiltration detection
- [ ] Add file monitoring
- [ ] Implement `block_upload()`
- [ ] Implement `quarantine_sensitive_file()`
- [ ] Add API endpoints
- [ ] Verify: Can detect and block data exfiltration

### Day 14: Compliance & Remediation Agents
- [ ] Create `backend/app/agents/compliance_agent.py`
- [ ] Implement ComplianceAgent class
- [ ] Add audit trail collection
- [ ] Add compliance report generation
- [ ] Create `backend/app/agents/remediation_agent.py`
- [ ] Implement RemediationAgent class
- [ ] Add system restoration capabilities
- [ ] Add patch management
- [ ] Verify: All 5 new agents operational

**Week 2 Complete:** Enterprise agents built and tested ✅

---

## Week 3: Infrastructure & Integration ✅ Days 15-21

### Day 15: Azure Infrastructure - Planning
- [ ] Review `scripts/mini-corp/deploy-mini-corp-azure.sh`
- [ ] Verify Azure subscription and permissions
- [ ] Plan IP addressing (10.100.0.0/16)
- [ ] Design network security groups
- [ ] Estimate costs (~$300/month)
- [ ] Create resource group: `mini-corp-rg`

### Day 16: Azure Infrastructure - Deployment
- [ ] Deploy Virtual Network (10.100.0.0/16)
- [ ] Create 4 subnets (management, servers, workstations, security)
- [ ] Deploy DC01 (Domain Controller) - 10.100.1.1
- [ ] Deploy FS01 (File Server) - 10.100.1.2
- [ ] Deploy WEB01 (Web Server) - 10.100.1.3
- [ ] Deploy DB01 (Database) - 10.100.1.4
- [ ] Deploy WS01-03 (Workstations) - 10.100.2.1-3
- [ ] Deploy XDR-COLLECTOR (Ubuntu) - 10.100.3.10
- [ ] Configure Network Security Groups
- [ ] Set up VPN access
- [ ] Verify: All VMs running and accessible

### Day 17: Active Directory Setup
- [ ] Promote DC01 to Domain Controller
- [ ] Create domain: minicorp.local
- [ ] Configure DNS
- [ ] Create OUs (Users, Computers, Security)
- [ ] Create test user accounts (10-15 users)
- [ ] Create security groups (Domain Admins, Users, Quarantine)
- [ ] Join all Windows VMs to domain
- [ ] Verify: Domain functional, all systems joined

### Day 18: Windows Agent Deployment
- [ ] Download Sysmon (SwiftOnSecurity config)
- [ ] Deploy Sysmon to all Windows VMs
- [ ] Download and configure Winlogbeat
- [ ] Deploy Winlogbeat to all Windows VMs
- [ ] Configure log forwarding to XDR-COLLECTOR
- [ ] Install OSQuery on all endpoints
- [ ] Test log collection
- [ ] Verify: Logs flowing to Mini-XDR backend

### Day 19: Workflow Configuration
- [ ] Create `scripts/mini-corp/setup-mini-corp-workflows.py`
- [ ] Add Kerberos Golden Ticket workflow
- [ ] Add Kerberos Silver Ticket workflow
- [ ] Add DCSync attack workflow
- [ ] Add Pass-the-hash detection workflow
- [ ] Add Mimikatz response workflow
- [ ] Add PSExec lateral movement workflow
- [ ] Add WMI lateral movement workflow
- [ ] Add RDP brute force workflow
- [ ] Add PowerShell abuse workflow
- [ ] Add registry persistence workflow
- [ ] Add scheduled task abuse workflow
- [ ] Add service creation workflow
- [ ] Add NTDS.dit theft workflow
- [ ] Add Group Policy abuse workflow
- [ ] Add insider exfiltration workflow
- [ ] Run workflow setup script
- [ ] Verify: All 15+ workflows active in database

### Day 20: Attack Simulation Testing
- [ ] Create `tests/mini-corp/test_kerberos_attacks.py`
- [ ] Simulate Golden Ticket attack
- [ ] Verify: Detection, incident, workflow, action
- [ ] Simulate lateral movement (PSExec)
- [ ] Verify: Detection, incident, workflow, action
- [ ] Simulate credential theft (Mimikatz simulation)
- [ ] Verify: Detection, incident, workflow, action
- [ ] Simulate data exfiltration
- [ ] Verify: Detection, incident, workflow, action
- [ ] Simulate privilege escalation
- [ ] Verify: Detection, incident, workflow, action
- [ ] Test normal user activity (no false positives)
- [ ] Verify: No incidents created for normal behavior

### Day 21: Final Validation & Documentation
- [ ] Run comprehensive test suite
- [ ] Check detection coverage (target: 95%+)
- [ ] Check action success rate (target: 90%+)
- [ ] Verify UI shows all activity
- [ ] Verify audit trail complete
- [ ] Review all security controls
- [ ] Test disaster recovery
- [ ] Document any issues/limitations
- [ ] Create operational runbook
- [ ] Verify: System production-ready

**Week 3 Complete:** Mini Corp operational and integrated ✅

---

## Configuration Requirements

### Environment Variables (backend/.env)
```bash
# Active Directory
AD_SERVER=10.100.1.1
AD_DOMAIN=minicorp.local
AD_ADMIN_USER=xdr-admin
AD_ADMIN_PASSWORD=<store-in-key-vault>

# Azure Configuration
AZURE_SUBSCRIPTION_ID=<your-subscription-id>
AZURE_RESOURCE_GROUP=mini-corp-rg
AZURE_VNET=mini-corp-vnet

# XDR Collector
XDR_COLLECTOR_IP=10.100.3.10
XDR_COLLECTOR_USER=xdradmin

# Windows Remote Management
WINRM_USER=<domain-admin>
WINRM_PASSWORD=<store-in-key-vault>

# ML Models
ENTERPRISE_MODEL_PATH=models/enterprise/model.pt
SPECIALIST_MODELS_PATH=models/specialists/

# Monitoring
LOG_STORAGE_PATH=/var/log/mini-xdr/
EVIDENCE_STORAGE_PATH=/var/evidence/mini-xdr/
```

### Azure Resources Needed
- [ ] Azure subscription with Owner/Contributor role
- [ ] Resource Group: mini-corp-rg
- [ ] Virtual Network: mini-corp-vnet (10.100.0.0/16)
- [ ] 8 Virtual Machines (1 Ubuntu, 7 Windows)
- [ ] Network Security Groups (4 NSGs)
- [ ] VPN Gateway (for secure access)
- [ ] Azure Key Vault (for secrets)
- [ ] Storage Account (for logs/evidence)

### Software Dependencies
```bash
# Backend (Python)
pip install ldap3  # Active Directory
pip install pywinrm  # Windows remote management
pip install smbprotocol  # SMB access
pip install pypsrp  # PowerShell remoting

# Windows Agents
- Sysmon v15+ (SwiftOnSecurity config)
- Winlogbeat 8.0+
- OSQuery 5.0+
- PowerShell 5.1+
```

---

## Success Criteria

### Phase 1 (ML Models)
- ✅ 12,000+ training samples
- ✅ 13-class model ≥85% accuracy
- ✅ 4 specialist models ≥90% accuracy
- ✅ <5% false positive rate
- ✅ <2 second detection latency

### Phase 2 (Agents)
- ✅ 5 new agents implemented (IAM, EDR, DLP, Compliance, Remediation)
- ✅ All API endpoints functional
- ✅ Agent actions tested and working
- ✅ Documentation complete

### Phase 3 (Infrastructure)
- ✅ 8 VMs deployed and configured
- ✅ Active Directory operational
- ✅ All agents installed and reporting
- ✅ 15+ workflows active
- ✅ All attacks detected and responded to
- ✅ UI showing complete visibility
- ✅ Audit trail complete

---

## Daily Progress Log

**Use this to track daily progress:**

```
Day 1 (Date: ____): 
Status: 
Completed: 
Blocked: 
Next: 

Day 2 (Date: ____):
Status: 
Completed: 
Blocked: 
Next: 

[Continue for all 21 days]
```

---

## Emergency Contacts / Resources

### Documentation
- Main Plan: `docs/MINI_CORP_ENTERPRISE_DEPLOYMENT_PLAN.md`
- This Checklist: `docs/MINI_CORP_QUICK_START_CHECKLIST.md`
- Handoff Summary: `docs/SESSION_HANDOFF_WORKFLOW_TESTING.md`

### Key Scripts
- ML Training: `aws/train_enterprise_model.py`
- Azure Deployment: `scripts/mini-corp/deploy-mini-corp-azure.sh`
- Agent Testing: `scripts/testing/test_*_agent.sh`
- Attack Simulation: `tests/mini-corp/test_*.py`

### Azure Resources
- Subscription Portal: https://portal.azure.com
- Cost Management: https://portal.azure.com/#blade/Microsoft_Azure_CostManagement
- Resource Group: `mini-corp-rg`

---

## Quick Commands Reference

```bash
# Check backend status
curl http://localhost:8000/health

# Check ML models loaded
curl http://localhost:8000/ml/status

# Test agent
curl -X POST http://localhost:8000/agents/iam/disable-user \
  -H "X-API-Key: your-key" \
  -d '{"username": "test.user", "reason": "test"}'

# View recent incidents
curl http://localhost:8000/incidents?limit=10

# Deploy Azure infrastructure
cd scripts/mini-corp
./deploy-mini-corp-azure.sh

# Test detection
python3 scripts/testing/test_enterprise_detection.py --attack kerberos
```

---

**Last Updated:** October 6, 2025  
**Status:** Ready to Begin  
**Next Action:** Start Day 1 - Download Windows/AD datasets


