# 🤖 Agent Capability Audit & Enhancement Plan

**Date:** October 6, 2025  
**Status:** Analysis Complete - Ready for Implementation

---

## ✅ EXISTING AGENTS & THEIR CAPABILITIES

### 1. **ContainmentAgent** (Comprehensive ✅)
**File:** `backend/app/agents/containment_agent.py`
**Status:** Production-ready, very feature-rich

**Current Capabilities:**
- ✅ Block IP addresses (UFW/iptables)
- ✅ Isolate hosts (network segmentation)
- ✅ Honeypot-specific isolation & redirection
- ✅ Enhanced monitoring for suspicious IPs
- ✅ Rate limiting
- ✅ Password resets
- ✅ Deploy WAF rules
- ✅ Traffic capture
- ✅ Hunt for similar attacks
- ✅ Threat intelligence lookups
- ✅ Notify analysts
- ✅ Rollback actions (basic)
- ✅ LangChain/AI orchestration
- ✅ Playbook execution

**What It Can Do:**
```python
# Network containment
- block_ip(ip, duration)
- isolate_host(hostname, level="strict|partial")
- enable_enhanced_monitoring(target_ip)
- apply_rate_limiting(target_ip)

# Honeypot-specific
- redirect_to_isolated_honeypot(target_ip)
- execute_honeypot_isolation(target_ip)

# Defensive actions
- deploy_waf_rules(target_ip)
- execute_password_reset(target_ip)
- capture_traffic(target_ip)

# Intelligence & hunting
- threat_intel_lookup(target_ip)
- hunt_similar_attacks(target_ip)
```

**Enhancement Needed:**
- Add integration with IAM agent for AD-specific actions
- Add integration with EDR agent for endpoint actions
- Enhance rollback to use new RollbackAgent capabilities

---

### 2. **RollbackAgent** ✅ (Already Exists!)
**File:** `backend/app/agents/containment_agent.py` (lines 2122-2675)
**Status:** Sophisticated AI-powered rollback system

**Current Capabilities:**
- ✅ False positive detection (multi-factor analysis)
- ✅ Temporal pattern analysis (business hours, regularity)
- ✅ Behavioral pattern analysis (entropy, tool detection)
- ✅ Threat intel consistency checking
- ✅ Impact assessment before rollback
- ✅ AI-powered rollback evaluation (using LLM)
- ✅ Learning from rollback decisions
- ✅ Execute rollbacks with comprehensive logging

**What It Can Do:**
```python
# Comprehensive rollback evaluation
result = await rollback_agent.evaluate_for_rollback(
    incident=incident,
    hours_since_action=12.5,
    db_session=db
)

# Returns:
{
    "should_rollback": true/false,
    "confidence": 0.0-1.0,
    "reasoning": "Detailed AI-generated reasoning",
    "fp_indicators": ["temporal_pattern", "behavioral_consistency"],
    "impact_assessment": {...},
    "ai_evaluation": {...}
}

# Execute rollback
await rollback_agent.execute_rollback(incident, rollback_decision, db)
```

**Key Features:**
- Analyzes temporal patterns (business hours vs off-hours)
- Checks for regular intervals (scheduled scans)
- Calculates command entropy
- Detects legitimate tools usage
- Cross-references threat intelligence
- Assesses impact of rollback
- Uses AI for final evaluation
- Learns from decisions for continuous improvement

**Enhancement Needed:**
- Extend to handle IAM/EDR/DLP specific rollbacks
- Add database persistence for rollback tracking
- UI integration for manual rollback triggers

---

### 3. **ThreatHuntingAgent** ✅
**File:** `backend/app/agents/containment_agent.py` (lines 1624-2120)
**Status:** Production-ready

**Capabilities:**
- ✅ Proactive threat hunting
- ✅ AI-generated hunt hypotheses
- ✅ Hunt for lateral movement
- ✅ Hunt for credential stuffing
- ✅ Hunt for persistence mechanisms
- ✅ Hunt for data exfiltration
- ✅ Hunt for C2 communication
- ✅ Pattern-based hunting
- ✅ Finding correlation

---

### 4. **ForensicsAgent** ✅
**File:** `backend/app/agents/forensics_agent.py`
**Status:** Production-ready

**Capabilities:**
- ✅ Traffic capture (tcpdump/pcap analysis)
- ✅ Evidence collection (logs, network, files, memory, system state)
- ✅ Forensic case management
- ✅ Chain of custody tracking
- ✅ Evidence integrity verification
- ✅ Timeline reconstruction
- ✅ Pattern analysis (malware, network, credentials, persistence)
- ✅ Risk assessment
- ✅ AI-powered forensic analysis
- ✅ Generate forensic reports

---

### 5. **AttributionAgent** ✅
**File:** `backend/app/agents/attribution_agent.py`
**Status:** Production-ready

**Capabilities:**
- ✅ IP reputation analysis
- ✅ Infrastructure analysis (ASN, geolocation, DNS)
- ✅ TTP analysis (MITRE ATT&CK mapping)
- ✅ Behavioral pattern analysis
- ✅ Campaign correlation
- ✅ Threat actor attribution
- ✅ Actor signature matching
- ✅ New actor cluster identification

---

### 6. **DeceptionAgent** ✅
**File:** `backend/app/agents/deception_agent.py`
**Status:** Production-ready

**Capabilities:**
- ✅ Deploy honeypots (Cowrie, Dionaea, Conpot, ElasticHoney, Web)
- ✅ Manage honeypot lifecycle (start, stop, remove)
- ✅ Analyze attacker behavior
- ✅ Calculate sophistication levels
- ✅ Adaptive honeypot deployment
- ✅ Create deception scenarios
- ✅ Evaluate honeypot effectiveness
- ✅ AI-powered deception strategy

---

## ❌ MISSING AGENTS (MUST CREATE)

### 1. **IAM Agent (Identity & Access Management)** 🔴 CRITICAL
**Status:** ❌ Does NOT exist
**Priority:** P0 (Critical for Active Directory security)

**Why We Need It:**
- Active Directory is the #1 corporate attack target (90% of breaches involve AD compromise)
- Current ContainmentAgent has no AD-specific capabilities
- Can't manage Windows authentication, Kerberos, or user accounts
- No way to respond to credential theft, Golden Ticket, or privilege escalation attacks

**Required Capabilities:**
```python
# User account management
- disable_user_account(username, reason) → Disable compromised AD user
- enable_user_account(username) → Re-enable after investigation
- quarantine_user(username) → Move to security group with restricted access
- reset_user_password(username, force_change=True) → Reset compromised password

# Kerberos security
- revoke_kerberos_tickets(username) → Force re-authentication
- reset_krbtgt_password() → Respond to Golden Ticket attacks

# Group membership
- remove_from_privileged_group(username, group) → Remove from Domain Admins, etc.
- add_to_group(username, group) → Restore legitimate access

# Detection
- detect_kerberos_attack(event) → Detect Golden/Silver Ticket
- detect_privilege_escalation(event) → Detect suspicious group additions
- detect_impossible_travel(event) → Detect geo-anomaly logins
- detect_off_hours_access(event) → Detect suspicious timing
```

**Integration Requirements:**
- LDAP connection to Active Directory
- PowerShell remoting to Domain Controllers
- Event Log parsing (Security log, Kerberos events)

**Rollback Support:**
- Capture userAccountControl before disable
- Capture group memberships before changes
- Store original password hash (for auditing)
- Rollback → Re-enable account, restore groups

---

### 2. **EDR Agent (Endpoint Detection & Response)** 🔴 CRITICAL
**Status:** ❌ Does NOT exist
**Priority:** P0 (Critical for Windows endpoint security)

**Why We Need It:**
- No way to take action on Windows endpoints
- Can't kill malicious processes (Mimikatz, ransomware)
- Can't quarantine malware files
- Can't isolate compromised Windows hosts
- ContainmentAgent only does network-level isolation

**Required Capabilities:**
```python
# Process management
- kill_process(hostname, process_name/pid) → Terminate malware
- suspend_process(hostname, pid) → Freeze for analysis
- analyze_process_behavior(hostname, pid) → Check for injection, etc.

# File operations
- quarantine_file(hostname, file_path) → Move to quarantine
- delete_file(hostname, file_path) → Remove malware
- restore_file(hostname, file_path) → Rollback quarantine

# Memory forensics
- collect_memory_dump(hostname) → Full memory capture
- scan_memory(hostname, yara_rules) → Memory-resident malware

# Host isolation
- isolate_host(hostname, level="strict|partial") → Network isolation
- un_isolate_host(hostname) → Restore connectivity

# Registry/Persistence
- delete_registry_key(hostname, key_path) → Remove persistence
- disable_scheduled_task(hostname, task_name) → Block malicious tasks
- disable_service(hostname, service_name) → Stop malicious services

# Detection
- detect_process_injection(event) → Detect code injection
- detect_lolbin_abuse(event) → Detect LOLBin misuse
- detect_powershell_abuse(event) → Detect suspicious PowerShell
```

**Integration Requirements:**
- WinRM/PowerShell remoting to Windows hosts
- Sysmon event parsing
- Windows Event Log integration
- File system access (SMB or remote PowerShell)

**Rollback Support:**
- Store process state before kill (for forensics)
- Track quarantined file original location
- Backup registry keys before deletion
- Rollback → Restore files, re-enable services, undo registry changes

---

### 3. **DLP Agent (Data Loss Prevention)** 🟠 HIGH
**Status:** ❌ Does NOT exist
**Priority:** P1 (High - needed for data protection)

**Why We Need It:**
- No way to detect sensitive data exfiltration
- Can't scan files for PII, credit cards, API keys
- Can't block unauthorized uploads
- No monitoring of USB devices, email attachments

**Required Capabilities:**
```python
# Data classification
- scan_file(file_path) → Detect PII, credit cards, SSNs, API keys, etc.
- classify_data(content) → Categorize data sensitivity
- calculate_sensitivity_score(file_path) → Risk scoring

# Blocking/Prevention
- block_upload(hostname, process_name, destination) → Stop data upload
- block_email_attachment(email_id, attachment) → Prevent email exfiltration
- block_usb_device(hostname, device_id) → Prevent USB exfiltration

# Monitoring
- monitor_large_file_transfers(threshold_mb) → Detect bulk exfiltration
- monitor_database_exports(hostname, db_name) → Track data dumps
- monitor_cloud_uploads(hostname, destination) → Track cloud exfiltration

# Detection
- detect_data_exfiltration(event) → Identify exfiltration attempts
- detect_unusual_file_access(event) → Insider threat detection
```

**Integration Requirements:**
- File system access (for scanning)
- Network traffic monitoring (for uploads)
- Endpoint agents (for USB/email monitoring)
- Pattern matching (regex for PII, credit cards, etc.)

**Rollback Support:**
- Track blocked uploads (for false positive analysis)
- Store blocked file metadata
- Rollback → Unblock destination, restore access

---

## 🔧 ENHANCEMENTS TO EXISTING AGENTS

### ContainmentAgent Enhancements
**Add to existing agent:**

```python
# Add IAM integration
async def _execute_ad_containment(self, username: str, action: str) -> Dict:
    """Delegate to IAM agent for AD-specific actions"""
    from .iam_agent import iam_agent
    return await iam_agent.execute_action(action, {"username": username})

# Add EDR integration
async def _execute_endpoint_containment(self, hostname: str, action: str) -> Dict:
    """Delegate to EDR agent for endpoint-specific actions"""
    from .edr_agent import edr_agent
    return await edr_agent.execute_action(action, {"hostname": hostname})

# Enhanced orchestration
async def orchestrate_enterprise_response(self, incident: Incident) -> Dict:
    """
    Orchestrate response using all available agents:
    - ContainmentAgent → Network-level actions
    - IAM Agent → Active Directory actions
    - EDR Agent → Endpoint actions
    - DLP Agent → Data protection actions
    """
    # Determine required actions based on threat type
    # Coordinate multi-agent response
    # Track all actions for rollback
```

### RollbackAgent Enhancements
**Add support for new agent types:**

```python
# Extend rollback capabilities
async def _execute_rollback_impl(self, rollback_data: Dict) -> Dict:
    """Execute rollback for any agent type"""
    
    agent_type = rollback_data['agent_type']
    
    if agent_type == "iam":
        from .iam_agent import iam_agent
        return await iam_agent.rollback_action(rollback_data['rollback_id'])
    
    elif agent_type == "edr":
        from .edr_agent import edr_agent
        return await edr_agent.rollback_action(rollback_data['rollback_id'])
    
    elif agent_type == "dlp":
        from .dlp_agent import dlp_agent
        return await dlp_agent.rollback_action(rollback_data['rollback_id'])
    
    elif agent_type == "containment":
        # Existing rollback logic for network containment
        return await self._rollback_network_containment(rollback_data)
```

---

## 📊 IMPLEMENTATION PRIORITY

### Phase 1: Critical Agents (Week 1)
**Must create before deploying Mini Corp infrastructure**

1. ✅ **Fix ML Errors** (Day 1)
   - Already documented in ML_FIXES_AND_AGENT_FRAMEWORK.md
   - ml_feature_extractor.py already exists ✅

2. 🔴 **Create IAM Agent** (Days 2-4)
   - Priority: P0 (CRITICAL)
   - File: `backend/app/agents/iam_agent.py`
   - ~800 lines of code
   - Test with mock AD first, then real AD

3. 🔴 **Create EDR Agent** (Days 5-6)
   - Priority: P0 (CRITICAL)
   - File: `backend/app/agents/edr_agent.py`
   - ~600 lines of code
   - Test with Windows test VM

4. 🟠 **Create DLP Agent** (Day 7)
   - Priority: P1 (HIGH)
   - File: `backend/app/agents/dlp_agent.py`
   - ~400 lines of code
   - Start with file scanning, add network monitoring later

### Phase 2: Integration & Testing (Week 2)
5. **Enhance ContainmentAgent** (Day 8)
   - Add IAM/EDR/DLP integration
   - Multi-agent orchestration

6. **Enhance RollbackAgent** (Day 9)
   - Add IAM/EDR/DLP rollback support
   - Database persistence

7. **Add ActionLog Database Model** (Day 10)
   - Track all agent actions
   - Store rollback data
   - Chain of custody

8. **Create API Endpoints** (Day 11)
   - `/api/agents/iam/execute`
   - `/api/agents/edr/execute`
   - `/api/agents/dlp/execute`
   - `/api/actions/rollback/{rollback_id}`

9. **Build Frontend UI** (Days 12-13)
   - Action management on incident page
   - Rollback confirmation modal
   - Action timeline view

10. **End-to-End Testing** (Day 14)
    - Test each agent individually
    - Test multi-agent orchestration
    - Test rollback functionality
    - Test UI workflows

---

## 🎯 DECISION: CREATE 3 NEW AGENTS

**Based on audit, we need to CREATE:**
1. ❌ IAM Agent (doesn't exist)
2. ❌ EDR Agent (doesn't exist)
3. ❌ DLP Agent (doesn't exist)

**Based on audit, we need to ENHANCE:**
1. ✅ ContainmentAgent (add integration with new agents)
2. ✅ RollbackAgent (extend to support new agent types)

**Already have (no changes needed):**
1. ✅ ThreatHuntingAgent (fully capable)
2. ✅ ForensicsAgent (fully capable)
3. ✅ AttributionAgent (fully capable)
4. ✅ DeceptionAgent (fully capable)

---

## 🚀 NEXT STEPS

1. ✅ Update TODOs to reflect actual needs
2. ⏳ Create IAM Agent with rollback support
3. ⏳ Create EDR Agent with rollback support
4. ⏳ Create DLP Agent with rollback support
5. ⏳ Add ActionLog database model
6. ⏳ Create API endpoints
7. ⏳ Build frontend UI
8. ⏳ Test complete workflow

---

**Status:** Analysis complete - ready to implement 3 new agents
**Good News:** RollbackAgent already exists and is sophisticated!
**Good News:** ContainmentAgent is comprehensive for network-level actions!
**Action Required:** Create IAM, EDR, and DLP agents to fill enterprise capability gaps

