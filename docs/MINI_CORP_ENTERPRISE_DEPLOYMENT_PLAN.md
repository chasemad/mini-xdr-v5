# 🏢 Mini Corp Enterprise XDR Deployment Plan - Option B (Full Suite)

**Created:** October 6, 2025  
**Target Completion:** 3 weeks (21 days)  
**Deployment Strategy:** Build → Train → Deploy → Test  
**Status:** 🔴 **NOT STARTED** - Ready to Begin

---

## 📋 Executive Summary

This document provides a **complete implementation plan** for deploying enterprise-grade XDR capabilities to monitor and protect the "Mini Corp" simulated corporate network on Azure. This plan follows **Option B: Full Enterprise Suite** for comprehensive 95% threat coverage.

### 🎯 Critical Decision: Infrastructure Deployment Timing

**❌ DO NOT deploy Mini Corp Azure infrastructure yet**

**Why Wait?**
1. **ML Models Need Retraining** - Current models trained on network attacks (DDoS, web attacks), NOT Windows/AD attacks (Kerberos, DCSync, lateral movement)
2. **Agents Need Development** - IAM, EDR, and DLP agents must be built before deployment
3. **No Detection = Wasted Cost** - Running VMs without proper detection/response is inefficient
4. **Testing Requirements** - Need local development environment first

**When to Deploy Infrastructure:** Week 3, Day 15 (after agents built and models retrained)

---

## 🔍 Current State Analysis

### ✅ What's Working (Honeypot-Focused)

**ML Models Trained (12 models):**
- ✅ DDoS/DoS Detection (100% accuracy)
- ✅ Brute Force Detection (94.7% accuracy)
- ✅ Web Attack Detection (79.7% accuracy)
- ✅ Network Reconnaissance (covered)
- ✅ Malware/Botnet (covered)
- ✅ General 7-class classifier (86.8% accuracy)

**Training Data:**
- ✅ CICIDS2017 (239 samples) - Network attacks
- ✅ UNSW-NB15 (180 samples) - Various attacks
- ✅ KDD Cup 1999 (186 samples) - Classic attacks
- ✅ Honeypot logs (81 samples) - SSH/Telnet
- ✅ Threat intel (200 samples) - Known bad IPs

**Total Training Samples:** 988 samples (network-focused)

**Existing Agents (5):**
1. ✅ ContainmentAgent - IP blocking, host isolation, firewall rules
2. ✅ ForensicsAgent - Evidence collection, malware analysis
3. ✅ AttributionAgent - Threat actor profiling
4. ✅ ThreatHuntingAgent - Pattern hunting
5. ✅ DeceptionAgent - Honeypot deployment

**Existing Workflows (25 total):**
- ✅ 15 T-Pot honeypot workflows
- ✅ 7 Mini Corp workflows (basic)
- ✅ 3 Default workflows

### ❌ Critical Gaps for Mini Corp

**Missing ML Training Data:**
- ❌ Windows Event Logs (authentication, process creation, registry)
- ❌ Active Directory attacks (Kerberos, Golden Ticket, DCSync)
- ❌ Endpoint behavior (process injection, LOLBins, PowerShell abuse)
- ❌ Lateral movement (PSExec, WMI, RDP abuse)
- ❌ Data exfiltration from corporate endpoints
- ❌ Insider threat patterns

**Missing Agents:**
- ❌ IAM Agent (Active Directory management)
- ❌ EDR Agent (Endpoint detection & response)
- ❌ DLP Agent (Data loss prevention)
- ❌ Compliance Agent (Audit & governance)
- ❌ Remediation Agent (Automated recovery)

**Missing Integration:**
- ❌ Windows Event Log ingestion
- ❌ Sysmon event parsing
- ❌ PowerShell transcript collection
- ❌ Active Directory LDAP queries
- ❌ Windows remote execution (WinRM/PowerShell)

---

## 🎯 Phase-by-Phase Implementation Plan

## **PHASE 1: ML Model Enhancement (Week 1: Days 1-7)**

### Priority: 🔴 **CRITICAL** - Must Complete Before Infrastructure Deployment

### Goal
Retrain ML models with Windows/AD attack patterns to ensure detection capability for corporate threats.

---

### **Task 1.1: Collect Windows/AD Training Data (Days 1-2)**

**Objective:** Gather 10,000+ samples of Windows and Active Directory attack patterns

#### Subtask 1.1.1: Download Public Datasets
**Priority:** P0 (Blocking)

**Actions:**
```bash
# Create dataset collection directory
mkdir -p /Users/chasemad/Desktop/mini-xdr/datasets/windows_ad_datasets

# Download ADFA-LD (UNSW Windows dataset)
# URL: https://cloudstor.aarnet.edu.au/plus/s/DS3zdEq3gqzqEOT
# Contains: Windows system call traces, attacks, normal behavior

# Download OpTC Dataset (DARPA)
# URL: https://github.com/FiveDirections/OpTC-data
# Contains: Windows endpoint attacks, lateral movement

# Download Mordor Datasets (APT simulations)
# URL: https://github.com/OTRF/Security-Datasets
# Contains: Kerberos attacks, Pass-the-hash, Golden Ticket, DCSync

# Download Windows Event Log samples
# Source: https://github.com/sbousseaden/EVTX-ATTACK-SAMPLES
# Contains: Real attack event logs (Mimikatz, PSExec, etc.)
```

**Expected Datasets:**
| Dataset | Samples | Attack Types |
|---------|---------|-------------|
| ADFA-LD | ~5,000 | System call anomalies, privilege escalation |
| OpTC | ~3,000 | Lateral movement, C2, exfiltration |
| Mordor | ~2,000 | Kerberos attacks, credential theft |
| EVTX Samples | ~1,000 | Windows-specific attacks |
| **TOTAL** | **11,000+** | **Comprehensive Windows coverage** |

**Deliverable:** Downloaded datasets in `/Users/chasemad/Desktop/mini-xdr/datasets/windows_ad_datasets/`

---

#### Subtask 1.1.2: Convert to Mini-XDR Format
**Priority:** P0 (Blocking)

**Create Conversion Script:**
```bash
# File: /Users/chasemad/Desktop/mini-xdr/scripts/data-processing/convert_windows_datasets.py
```

**Script Requirements:**
- Parse Windows Event Log XML/JSON
- Extract 79 features matching current model input
- Map to attack categories:
  - Class 7: Kerberos Attacks (new)
  - Class 8: Lateral Movement (new)
  - Class 9: Credential Theft (new)
  - Class 10: Privilege Escalation (new)
  - Class 11: Data Exfiltration (refined)
  - Class 12: Insider Threats (new)

**Expected Output:**
```json
{
  "events": [
    {
      "src_ip": "10.100.2.1",
      "dst_ip": "10.100.1.1",
      "src_port": 49158,
      "dst_port": 88,
      "protocol": "tcp",
      "bytes_sent": 1024,
      "bytes_received": 512,
      "duration": 0.5,
      "label": "kerberos_attack",
      "attack_category": "credential_theft",
      "event_type": "golden_ticket",
      "severity": "critical",
      "indicators": ["AS-REQ with suspicious encryption", "Ticket lifetime 10 years"]
    }
  ]
}
```

**Deliverable:** Converted datasets ready for ML training

---

### **Task 1.2: Retrain ML Models with Corporate Data (Days 3-5)**

**Objective:** Retrain all 12 models with combined honeypot + corporate attack data

#### Subtask 1.2.1: Merge Training Datasets
**Priority:** P0 (Blocking)

**Actions:**
```bash
cd /Users/chasemad/Desktop/mini-xdr

# Merge datasets
python3 scripts/data-processing/merge_training_data.py \
  --honeypot-data datasets/real_datasets/ \
  --windows-data datasets/windows_ad_datasets/ \
  --output datasets/combined_enterprise_training.json

# Expected output:
# - Total samples: 12,000+ (988 existing + 11,000 new)
# - 12 attack classes
# - Balanced distribution
```

**Class Distribution Target:**
```
Class 0:  Normal Traffic              - 3,000 samples (25%)
Class 1:  DDoS/DoS                    - 1,500 samples (12.5%)
Class 2:  Network Reconnaissance      - 1,000 samples (8.3%)
Class 3:  Brute Force                 - 1,000 samples (8.3%)
Class 4:  Web Application Attacks     - 800 samples (6.7%)
Class 5:  Malware/Botnet              - 800 samples (6.7%)
Class 6:  Advanced Persistent Threats - 500 samples (4.2%)
Class 7:  Kerberos Attacks           - 900 samples (7.5%) ← NEW
Class 8:  Lateral Movement           - 800 samples (6.7%) ← NEW
Class 9:  Credential Theft           - 900 samples (7.5%) ← NEW
Class 10: Privilege Escalation       - 700 samples (5.8%) ← NEW
Class 11: Data Exfiltration          - 600 samples (5.0%) ← NEW
Class 12: Insider Threats            - 500 samples (4.2%) ← NEW
-----------------------------------------------------------
TOTAL:                                 12,000 samples (100%)
```

---

#### Subtask 1.2.2: Train Enhanced Multi-Class Model
**Priority:** P0 (Blocking)

**Training Configuration:**
```python
# File: /Users/chasemad/Desktop/mini-xdr/aws/train_enterprise_model.py

MODEL_CONFIG = {
    "name": "mini-xdr-enterprise-v3",
    "classes": 13,  # 0=normal + 12 attack types
    "architecture": "XDREnterpriseDetector",
    "layers": [79, 512, 256, 128, 64, 13],
    "dropout": 0.3,
    "learning_rate": 0.001,
    "batch_size": 64,
    "epochs": 50,
    "early_stopping": 10,
    "validation_split": 0.2
}

TARGET_ACCURACY = 0.85  # 85% minimum
TARGET_PRECISION = 0.80  # 80% per-class
```

**Training Commands:**
```bash
# Local training for development
cd /Users/chasemad/Desktop/mini-xdr
source ml-training-env/bin/activate

python3 aws/train_enterprise_model.py \
  --data datasets/combined_enterprise_training.json \
  --output models/enterprise/ \
  --epochs 50 \
  --gpu  # If available

# Expected training time: 20-30 minutes
```

**Success Criteria:**
- ✅ Overall accuracy ≥ 85%
- ✅ Per-class precision ≥ 80%
- ✅ F1-score ≥ 0.80
- ✅ Low false positive rate (<5%)

**Deliverable:** Trained model saved to `models/enterprise/model.pt`

---

#### Subtask 1.2.3: Train Specialist Models for Critical Threats
**Priority:** P1 (High)

**Specialist Models to Train:**

1. **Kerberos Attack Specialist (Binary)**
   - Detects: Golden Ticket, Silver Ticket, Kerberoasting, AS-REP roasting
   - Target accuracy: 95%+
   - Training time: ~10 minutes

2. **Lateral Movement Specialist (Binary)**
   - Detects: PSExec, WMI, RDP abuse, SMB exploitation
   - Target accuracy: 92%+
   - Training time: ~10 minutes

3. **Credential Theft Specialist (Binary)**
   - Detects: Mimikatz, password dumping, NTDS.dit theft, DCSync
   - Target accuracy: 95%+
   - Training time: ~10 minutes

4. **Insider Threat Specialist (Binary)**
   - Detects: Unusual data access, off-hours activity, privilege abuse
   - Target accuracy: 85%+
   - Training time: ~12 minutes

**Training Script:**
```bash
# Train all specialists
python3 aws/train_specialist_models.py \
  --data datasets/combined_enterprise_training.json \
  --specialists kerberos,lateral_movement,credential_theft,insider \
  --output models/specialists/
```

**Deliverable:** 4 specialist models in `models/specialists/`

---

#### Subtask 1.2.4: Validate Model Performance
**Priority:** P0 (Blocking)

**Validation Tests:**
```bash
# Test with holdout dataset
python3 scripts/testing/validate_enterprise_models.py \
  --models models/enterprise/,models/specialists/ \
  --test-data datasets/test/windows_ad_holdout.json \
  --output validation_report.json

# Generate confusion matrix
# Check per-class metrics
# Identify weak spots
```

**Validation Checklist:**
- [ ] Overall accuracy ≥ 85%
- [ ] Kerberos attack detection ≥ 95%
- [ ] Lateral movement detection ≥ 92%
- [ ] Credential theft detection ≥ 95%
- [ ] False positive rate ≤ 5%
- [ ] Inference time ≤ 50ms per event

**Deliverable:** Validation report confirming model readiness

---

### **Task 1.3: Integrate Models into Backend (Days 6-7)**

**Objective:** Deploy new models to Mini-XDR backend and test detection pipeline

#### Subtask 1.3.1: Update ML Engine
**Priority:** P0 (Blocking)

**File:** `/Users/chasemad/Desktop/mini-xdr/backend/app/ml_engine.py`

**Changes Required:**
```python
# Add new model loader
class EnterpriseMLDetector:
    def __init__(self):
        self.general_model = self._load_model("models/enterprise/model.pt")
        self.kerberos_specialist = self._load_model("models/specialists/kerberos.pt")
        self.lateral_movement_specialist = self._load_model("models/specialists/lateral_movement.pt")
        self.credential_theft_specialist = self._load_model("models/specialists/credential_theft.pt")
        self.insider_specialist = self._load_model("models/specialists/insider.pt")
        
        # Load scaler
        self.scaler = joblib.load("models/enterprise/scaler.pkl")
        
        # Class mapping for 13 classes
        self.class_names = [
            "normal", "ddos", "reconnaissance", "brute_force",
            "web_attack", "malware", "apt", "kerberos_attack",
            "lateral_movement", "credential_theft", "privilege_escalation",
            "data_exfiltration", "insider_threat"
        ]
    
    async def detect_threat(self, event_features: Dict) -> Dict:
        """
        Multi-model ensemble detection
        Returns: threat_type, confidence, specialist_results
        """
        # Step 1: General model classification
        general_result = self._classify(self.general_model, event_features)
        
        # Step 2: If high-risk class, run specialist
        if general_result['class'] in ['kerberos_attack', 'lateral_movement', 
                                       'credential_theft', 'insider_threat']:
            specialist_result = self._run_specialist(general_result['class'], event_features)
            return self._merge_results(general_result, specialist_result)
        
        return general_result
```

**Deliverable:** Updated ML engine with 13-class detection

---

#### Subtask 1.3.2: Add Feature Extraction for Windows Events
**Priority:** P0 (Blocking)

**File:** `/Users/chasemad/Desktop/mini-xdr/backend/app/feature_extractor.py`

**Create Feature Extractor:**
```python
class WindowsEventFeatureExtractor:
    """
    Extracts 79 features from Windows events for ML detection
    Handles: Event Logs, Sysmon, PowerShell, Network flows
    """
    
    def extract_features(self, event: Event) -> List[float]:
        """
        Returns 79-dimensional feature vector matching training format
        """
        features = []
        
        # Network features (20 features)
        features.extend(self._extract_network_features(event))
        
        # Process features (15 features)
        features.extend(self._extract_process_features(event))
        
        # Authentication features (12 features)
        features.extend(self._extract_auth_features(event))
        
        # File system features (10 features)
        features.extend(self._extract_file_features(event))
        
        # Registry features (8 features)
        features.extend(self._extract_registry_features(event))
        
        # Kerberos features (8 features)
        features.extend(self._extract_kerberos_features(event))
        
        # Behavioral features (6 features)
        features.extend(self._extract_behavioral_features(event))
        
        return features  # Total: 79 features
```

**Deliverable:** Feature extractor supporting Windows events

---

#### Subtask 1.3.3: Test End-to-End Detection
**Priority:** P0 (Blocking)

**Test Script:**
```bash
# File: scripts/testing/test_enterprise_detection.py

# Test 1: Kerberos Golden Ticket attack
python3 scripts/testing/test_enterprise_detection.py \
  --attack-type kerberos_golden_ticket \
  --expected-detection true

# Test 2: Lateral movement via PSExec
python3 scripts/testing/test_enterprise_detection.py \
  --attack-type lateral_movement_psexec \
  --expected-detection true

# Test 3: Credential dumping (Mimikatz)
python3 scripts/testing/test_enterprise_detection.py \
  --attack-type credential_dump_mimikatz \
  --expected-detection true

# Test 4: Normal corporate traffic
python3 scripts/testing/test_enterprise_detection.py \
  --attack-type normal_ad_authentication \
  --expected-detection false
```

**Success Criteria:**
- [ ] All attack patterns detected (100%)
- [ ] Normal traffic not flagged as attack
- [ ] Detection latency < 2 seconds
- [ ] Incident created automatically
- [ ] Confidence scores > 0.80

**Deliverable:** Passing end-to-end tests

---

### **Phase 1 Completion Checklist**

- [ ] Downloaded 11,000+ Windows/AD attack samples
- [ ] Converted datasets to Mini-XDR format
- [ ] Merged with existing training data (12,000+ total)
- [ ] Trained 13-class enterprise model (85%+ accuracy)
- [ ] Trained 4 specialist models (90%+ accuracy)
- [ ] Validated all models with test data
- [ ] Integrated models into backend
- [ ] Created Windows feature extractor
- [ ] Passed end-to-end detection tests
- [ ] Documented model performance

**Phase 1 Output:**
- ✅ ML models ready for corporate threat detection
- ✅ Backend integrated and tested
- ✅ Detection pipeline operational

**Estimated Duration:** 7 days (1 week)

---

## **PHASE 2: Enterprise Agent Development (Week 2: Days 8-14)**

### Priority: 🟠 **HIGH** - Core Capability Expansion

### Goal
Build 5 new enterprise agents with Windows/AD management capabilities.

---

### **Task 2.1: IAM Agent Development (Days 8-10)**

**Objective:** Create Identity & Access Management agent for Active Directory

#### Priority: P0 (CRITICAL) - Active Directory is the #1 corporate attack target

---

#### Subtask 2.1.1: Create IAM Agent Base Class
**Priority:** P0

**File:** `/Users/chasemad/Desktop/mini-xdr/backend/app/agents/iam_agent.py`

**Agent Structure:**
```python
"""
Identity & Access Management (IAM) Agent
Specialized in Active Directory security and authentication monitoring
"""

import asyncio
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import ldap3
from ldap3 import Server, Connection, ALL

from ..models import Event, Incident, Action
from ..config import settings

logger = logging.getLogger(__name__)


class IAMAgent:
    """
    AI Agent for Identity & Access Management in corporate environments
    
    Capabilities:
    - Active Directory monitoring and management
    - Kerberos attack detection and response
    - Authentication anomaly detection
    - User account lifecycle management
    - Privilege escalation prevention
    """
    
    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        self.agent_id = "iam_agent_v1"
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Active Directory configuration
        self.ad_server = settings.ad_server  # e.g., "10.100.1.1"
        self.ad_domain = settings.ad_domain  # e.g., "minicorp.local"
        self.ad_user = settings.ad_admin_user
        self.ad_password = settings.ad_admin_password
        
        # LDAP connection
        self.ldap_server = None
        self.ldap_conn = None
        
        # Kerberos monitoring
        self.kerberos_analyzer = KerberosAnalyzer()
        
        # Authentication tracker
        self.auth_sessions = {}
        self.failed_auth_tracker = {}
        
        # Quarantine group
        self.quarantine_group_dn = "CN=Quarantine,OU=Security,DC=minicorp,DC=local"
    
    async def initialize(self):
        """Initialize LDAP connection to Active Directory"""
        try:
            self.ldap_server = Server(self.ad_server, get_info=ALL)
            self.ldap_conn = Connection(
                self.ldap_server,
                user=f"{self.ad_domain}\\{self.ad_user}",
                password=self.ad_password,
                auto_bind=True
            )
            self.logger.info("✅ Connected to Active Directory")
        except Exception as e:
            self.logger.error(f"❌ Failed to connect to AD: {e}")
    
    # ==================== DETECTION METHODS ====================
    
    async def analyze_authentication_event(self, event: Event) -> Optional[Dict]:
        """
        Analyze authentication event for anomalies
        Detects: Impossible travel, off-hours access, brute force, etc.
        """
        analysis = {
            "event_id": event.id,
            "username": event.metadata.get("username"),
            "source_ip": event.src_ip,
            "timestamp": event.timestamp,
            "anomalies": [],
            "risk_score": 0.0,
            "recommended_actions": []
        }
        
        # Check 1: Impossible travel
        if await self._detect_impossible_travel(event):
            analysis["anomalies"].append("impossible_travel")
            analysis["risk_score"] += 0.3
        
        # Check 2: Off-hours access
        if await self._detect_off_hours_access(event):
            analysis["anomalies"].append("off_hours_access")
            analysis["risk_score"] += 0.2
        
        # Check 3: Brute force pattern
        if await self._detect_brute_force_pattern(event):
            analysis["anomalies"].append("brute_force")
            analysis["risk_score"] += 0.4
        
        # Check 4: Service account abuse
        if await self._detect_service_account_abuse(event):
            analysis["anomalies"].append("service_account_abuse")
            analysis["risk_score"] += 0.3
        
        # Recommend actions based on risk
        if analysis["risk_score"] >= 0.7:
            analysis["recommended_actions"] = [
                "disable_user_account",
                "revoke_all_sessions",
                "alert_security_team"
            ]
        elif analysis["risk_score"] >= 0.4:
            analysis["recommended_actions"] = [
                "enforce_mfa",
                "monitor_user_activity"
            ]
        
        return analysis if analysis["anomalies"] else None
    
    async def detect_kerberos_attack(self, event: Event) -> Optional[Dict]:
        """
        Detect Kerberos-based attacks
        Types: Golden Ticket, Silver Ticket, Kerberoasting, AS-REP roasting
        """
        kerberos_data = event.metadata.get("kerberos", {})
        
        # Analyze Kerberos ticket
        analysis = await self.kerberos_analyzer.analyze_ticket(kerberos_data)
        
        if analysis["is_malicious"]:
            return {
                "attack_type": analysis["attack_type"],
                "confidence": analysis["confidence"],
                "indicators": analysis["indicators"],
                "affected_user": analysis["username"],
                "recommended_actions": [
                    "revoke_kerberos_tickets",
                    "reset_krbtgt_password",  # For Golden Ticket
                    "disable_affected_accounts",
                    "investigate_dc_compromise"
                ]
            }
        
        return None
    
    async def detect_privilege_escalation(self, event: Event) -> Optional[Dict]:
        """
        Detect privilege escalation attempts
        Monitors: Group membership changes, privilege grants, ACL modifications
        """
        # Check for suspicious group additions
        if event.event_type == "group_membership_changed":
            group_dn = event.metadata.get("group_dn")
            user_dn = event.metadata.get("user_dn")
            
            # Check if privileged group
            if self._is_privileged_group(group_dn):
                return {
                    "escalation_type": "group_membership",
                    "group": group_dn,
                    "user": user_dn,
                    "risk_score": 0.8,
                    "recommended_actions": [
                        "remove_from_privileged_group",
                        "investigate_who_added_user",
                        "audit_recent_changes"
                    ]
                }
        
        return None
    
    # ==================== RESPONSE ACTIONS ====================
    
    async def disable_user_account(self, username: str, reason: str) -> Dict[str, Any]:
        """
        Disable Active Directory user account
        
        Args:
            username: Username to disable (samAccountName)
            reason: Reason for disabling (for audit log)
        
        Returns:
            Action result with success/failure
        """
        try:
            self.logger.info(f"🔒 Disabling AD user: {username} (Reason: {reason})")
            
            # Search for user
            user_dn = await self._get_user_dn(username)
            if not user_dn:
                return {
                    "success": False,
                    "error": f"User {username} not found in AD"
                }
            
            # Disable account (set userAccountControl flag)
            # UAC flag 0x0002 = ACCOUNTDISABLE
            modify_result = self.ldap_conn.modify(
                user_dn,
                {'userAccountControl': [(ldap3.MODIFY_REPLACE, ['514'])]}  # 512 + 2
            )
            
            if modify_result:
                self.logger.info(f"✅ Successfully disabled user: {username}")
                
                # Add to audit log
                await self._log_iam_action({
                    "action": "disable_user_account",
                    "username": username,
                    "reason": reason,
                    "timestamp": datetime.now().isoformat(),
                    "actor": "iam_agent"
                })
                
                return {
                    "success": True,
                    "action": "disable_user_account",
                    "username": username,
                    "user_dn": user_dn,
                    "message": f"Account {username} has been disabled",
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {
                    "success": False,
                    "error": f"LDAP modify failed: {self.ldap_conn.result}"
                }
        
        except Exception as e:
            self.logger.error(f"❌ Failed to disable user {username}: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def revoke_kerberos_tickets(self, username: str) -> Dict[str, Any]:
        """
        Revoke all Kerberos tickets for a user
        Forces re-authentication
        """
        try:
            # On Domain Controller, run: klist purge for user
            dc_host = self.ad_server
            
            command = f"powershell -Command \"Invoke-Command -ComputerName {dc_host} -ScriptBlock {{ klist purge -li 0x3e7 }}\""
            
            # Execute via responder agent
            from ..responder import ResponderAgent
            responder = ResponderAgent(ssh_host=dc_host, ssh_user=self.ad_user)
            
            status, stdout, stderr = await responder.execute_command(command, timeout=30)
            
            if status == "success":
                return {
                    "success": True,
                    "action": "revoke_kerberos_tickets",
                    "username": username,
                    "message": "Kerberos tickets purged, user must re-authenticate"
                }
            else:
                return {
                    "success": False,
                    "error": stderr
                }
        
        except Exception as e:
            self.logger.error(f"Failed to revoke Kerberos tickets: {e}")
            return {"success": False, "error": str(e)}
    
    async def quarantine_user(self, username: str, reason: str) -> Dict[str, Any]:
        """
        Move user to quarantine security group
        Restricts access while maintaining account for investigation
        """
        try:
            self.logger.info(f"🚨 Quarantining user: {username}")
            
            user_dn = await self._get_user_dn(username)
            
            # Add to Quarantine group
            add_result = self.ldap_conn.modify(
                self.quarantine_group_dn,
                {'member': [(ldap3.MODIFY_ADD, [user_dn])]}
            )
            
            # Remove from all other groups (except Domain Users)
            await self._remove_from_privileged_groups(user_dn)
            
            if add_result:
                return {
                    "success": True,
                    "action": "quarantine_user",
                    "username": username,
                    "quarantine_group": self.quarantine_group_dn,
                    "message": f"User {username} has been quarantined"
                }
            else:
                return {"success": False, "error": self.ldap_conn.result}
        
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def reset_user_password(self, username: str, force_change: bool = True) -> Dict[str, Any]:
        """
        Reset user password (generates random secure password)
        Optionally force password change on next login
        """
        try:
            import secrets
            import string
            
            # Generate secure random password
            alphabet = string.ascii_letters + string.digits + string.punctuation
            new_password = ''.join(secrets.choice(alphabet) for _ in range(16))
            
            user_dn = await self._get_user_dn(username)
            
            # Reset password
            modify_result = self.ldap_conn.extend.microsoft.modify_password(
                user_dn,
                new_password
            )
            
            if force_change:
                # Set pwdLastSet to 0 (force change on next login)
                self.ldap_conn.modify(
                    user_dn,
                    {'pwdLastSet': [(ldap3.MODIFY_REPLACE, ['0'])]}
                )
            
            if modify_result:
                # Store password temporarily for admin notification
                # (In production, use secure password vault)
                return {
                    "success": True,
                    "action": "reset_password",
                    "username": username,
                    "temporary_password": new_password,  # Send to admin securely
                    "force_change_required": force_change,
                    "message": "Password has been reset"
                }
            else:
                return {"success": False, "error": "Password reset failed"}
        
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def enforce_mfa(self, username: str) -> Dict[str, Any]:
        """
        Enforce MFA for user account
        Sets AD attribute requiring multi-factor authentication
        """
        try:
            user_dn = await self._get_user_dn(username)
            
            # Set MFA required attribute
            # (Depends on your MFA solution - Azure AD, Duo, etc.)
            modify_result = self.ldap_conn.modify(
                user_dn,
                {'msDS-User-Account-Control-Computed': [(ldap3.MODIFY_REPLACE, ['1'])]}
            )
            
            return {
                "success": True if modify_result else False,
                "action": "enforce_mfa",
                "username": username,
                "message": "MFA enforcement enabled"
            }
        
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    # ==================== HELPER METHODS ====================
    
    async def _get_user_dn(self, username: str) -> Optional[str]:
        """Get user Distinguished Name from username"""
        search_base = f"DC={self.ad_domain.replace('.', ',DC=')}"
        search_filter = f"(samAccountName={username})"
        
        self.ldap_conn.search(
            search_base=search_base,
            search_filter=search_filter,
            attributes=['distinguishedName']
        )
        
        if self.ldap_conn.entries:
            return str(self.ldap_conn.entries[0].distinguishedName)
        return None
    
    def _is_privileged_group(self, group_dn: str) -> bool:
        """Check if group is privileged (Domain Admins, etc.)"""
        privileged_groups = [
            "Domain Admins",
            "Enterprise Admins",
            "Schema Admins",
            "Administrators",
            "Account Operators",
            "Backup Operators"
        ]
        return any(group in group_dn for group in privileged_groups)
    
    async def _detect_impossible_travel(self, event: Event) -> bool:
        """Detect if user logged in from geographically impossible location"""
        # Implementation: Check if same user logged in from different geo location
        # within impossible timeframe
        return False  # Placeholder
    
    async def _detect_off_hours_access(self, event: Event) -> bool:
        """Detect authentication outside business hours"""
        hour = event.timestamp.hour
        # Business hours: 8am - 6pm
        return hour < 8 or hour > 18
    
    async def _detect_brute_force_pattern(self, event: Event) -> bool:
        """Detect authentication brute force pattern"""
        username = event.metadata.get("username")
        src_ip = event.src_ip
        
        key = f"{username}:{src_ip}"
        
        if key not in self.failed_auth_tracker:
            self.failed_auth_tracker[key] = []
        
        # Add failed attempt
        if event.event_type == "authentication_failed":
            self.failed_auth_tracker[key].append(datetime.now())
            
            # Remove attempts older than 5 minutes
            cutoff = datetime.now() - timedelta(minutes=5)
            self.failed_auth_tracker[key] = [
                t for t in self.failed_auth_tracker[key] if t > cutoff
            ]
            
            # Check threshold: 5 failures in 5 minutes
            if len(self.failed_auth_tracker[key]) >= 5:
                return True
        
        return False
    
    async def _detect_service_account_abuse(self, event: Event) -> bool:
        """Detect service account used for interactive login"""
        username = event.metadata.get("username", "")
        logon_type = event.metadata.get("logon_type")
        
        # Service accounts typically have "svc-" prefix
        is_service_account = username.startswith("svc-") or username.startswith("service-")
        
        # Interactive logon type = 2, Remote Desktop = 10
        is_interactive = logon_type in ["2", "10"]
        
        return is_service_account and is_interactive
    
    async def _remove_from_privileged_groups(self, user_dn: str):
        """Remove user from all privileged groups"""
        # Get current group memberships
        self.ldap_conn.search(
            search_base=user_dn,
            search_filter="(objectClass=user)",
            attributes=['memberOf']
        )
        
        if self.ldap_conn.entries:
            groups = self.ldap_conn.entries[0].memberOf
            
            for group_dn in groups:
                if self._is_privileged_group(str(group_dn)):
                    # Remove from group
                    self.ldap_conn.modify(
                        str(group_dn),
                        {'member': [(ldap3.MODIFY_DELETE, [user_dn])]}
                    )
    
    async def _log_iam_action(self, action_data: Dict):
        """Log IAM action to audit trail"""
        # Store in database for compliance
        # In production: Also send to SIEM
        self.logger.info(f"IAM Action: {action_data}")


class KerberosAnalyzer:
    """Analyzes Kerberos tickets for attacks"""
    
    async def analyze_ticket(self, kerberos_data: Dict) -> Dict:
        """
        Analyze Kerberos ticket for malicious indicators
        
        Checks for:
        - Abnormally long ticket lifetime (Golden Ticket)
        - Suspicious encryption types
        - Forged PAC data
        - Service ticket anomalies (Silver Ticket)
        """
        indicators = []
        is_malicious = False
        attack_type = None
        confidence = 0.0
        
        # Check 1: Ticket lifetime
        ticket_lifetime = kerberos_data.get("ticket_lifetime", 600)
        if ticket_lifetime > 86400:  # > 24 hours
            indicators.append("abnormal_ticket_lifetime")
            confidence += 0.4
            attack_type = "golden_ticket"
            is_malicious = True
        
        # Check 2: Encryption type
        enc_type = kerberos_data.get("encryption_type")
        if enc_type in ["DES", "RC4"]:  # Weak encryption
            indicators.append("weak_encryption")
            confidence += 0.2
        
        # Check 3: PAC validation
        if not kerberos_data.get("pac_validated", True):
            indicators.append("invalid_pac")
            confidence += 0.3
            is_malicious = True
        
        # Check 4: Service ticket anomalies
        if kerberos_data.get("is_service_ticket"):
            if self._detect_silver_ticket(kerberos_data):
                indicators.append("silver_ticket_indicators")
                confidence += 0.5
                attack_type = "silver_ticket"
                is_malicious = True
        
        return {
            "is_malicious": is_malicious,
            "attack_type": attack_type,
            "confidence": min(confidence, 1.0),
            "indicators": indicators,
            "username": kerberos_data.get("username")
        }
    
    def _detect_silver_ticket(self, kerberos_data: Dict) -> bool:
        """Detect Silver Ticket attack indicators"""
        # Silver ticket: forged service ticket without contacting DC
        # Indicators: Missing PAC signature, unusual service account usage
        return (
            not kerberos_data.get("pac_signature_valid", True) or
            kerberos_data.get("service_account_suspicious", False)
        )


# Export agent
__all__ = ['IAMAgent', 'KerberosAnalyzer']
```

**Deliverable:** Full IAM Agent with AD integration (1,200+ lines)

---

#### Subtask 2.1.2: Add IAM Agent API Endpoints
**Priority:** P0

**File:** `/Users/chasemad/Desktop/mini-xdr/backend/app/main.py`

**Add Endpoints:**
```python
# IAM Agent endpoints
@app.post("/agents/iam/disable-user")
async def iam_disable_user(
    username: str,
    reason: str,
    _api_key: str = Depends(verify_api_key),
    db: AsyncSession = Depends(get_db)
):
    """Disable Active Directory user account"""
    from .agents.iam_agent import IAMAgent
    
    agent = IAMAgent()
    await agent.initialize()
    
    result = await agent.disable_user_account(username, reason)
    
    # Log action
    await log_action(db, "iam_disable_user", result)
    
    return result

@app.post("/agents/iam/revoke-tickets")
async def iam_revoke_tickets(
    username: str,
    _api_key: str = Depends(verify_api_key)
):
    """Revoke all Kerberos tickets for user"""
    from .agents.iam_agent import IAMAgent
    
    agent = IAMAgent()
    await agent.initialize()
    
    return await agent.revoke_kerberos_tickets(username)

@app.post("/agents/iam/quarantine-user")
async def iam_quarantine_user(
    username: str,
    reason: str,
    _api_key: str = Depends(verify_api_key)
):
    """Quarantine user account"""
    from .agents.iam_agent import IAMAgent
    
    agent = IAMAgent()
    await agent.initialize()
    
    return await agent.quarantine_user(username, reason)

@app.post("/agents/iam/reset-password")
async def iam_reset_password(
    username: str,
    force_change: bool = True,
    _api_key: str = Depends(verify_api_key)
):
    """Reset user password"""
    from .agents.iam_agent import IAMAgent
    
    agent = IAMAgent()
    await agent.initialize()
    
    return await agent.reset_user_password(username, force_change)

@app.post("/agents/iam/analyze-auth")
async def iam_analyze_auth(
    event_id: int,
    _api_key: str = Depends(verify_api_key),
    db: AsyncSession = Depends(get_db)
):
    """Analyze authentication event for anomalies"""
    from .agents.iam_agent import IAMAgent
    
    # Get event
    event = await db.get(Event, event_id)
    if not event:
        raise HTTPException(404, "Event not found")
    
    agent = IAMAgent()
    await agent.initialize()
    
    analysis = await agent.analyze_authentication_event(event)
    
    return analysis or {"message": "No anomalies detected"}
```

**Deliverable:** IAM Agent integrated into API

---

#### Subtask 2.1.3: Test IAM Agent
**Priority:** P0

**Test Script:**
```bash
# File: scripts/testing/test_iam_agent.sh

#!/bin/bash

echo "🧪 Testing IAM Agent"

API_BASE="http://localhost:8000"
API_KEY="your-api-key-here"

# Test 1: Disable user account
echo "Test 1: Disable user account"
curl -X POST "$API_BASE/agents/iam/disable-user" \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "username": "test.user",
    "reason": "Suspected credential compromise"
  }'

# Test 2: Revoke Kerberos tickets
echo "Test 2: Revoke Kerberos tickets"
curl -X POST "$API_BASE/agents/iam/revoke-tickets" \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "username": "test.user"
  }'

# Test 3: Quarantine user
echo "Test 3: Quarantine user"
curl -X POST "$API_BASE/agents/iam/quarantine-user" \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "username": "compromised.user",
    "reason": "Kerberos Golden Ticket detected"
  }'

# Test 4: Reset password
echo "Test 4: Reset password"
curl -X POST "$API_BASE/agents/iam/reset-password" \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "username": "test.user",
    "force_change": true
  }'

echo "✅ IAM Agent tests complete"
```

**Success Criteria:**
- [ ] All API endpoints respond
- [ ] AD connection successful
- [ ] User disable/enable works
- [ ] Password reset functional
- [ ] Kerberos revocation works
- [ ] Quarantine group operations work

**Deliverable:** Passing IAM Agent tests

---

### **Task 2.2: EDR Agent Development (Days 11-12)**

**Objective:** Create Endpoint Detection & Response agent for Windows endpoints

#### Priority: P0 (CRITICAL) - Endpoint security is essential

---

#### Subtask 2.2.1: Create EDR Agent
**Priority:** P0

**File:** `/Users/chasemad/Desktop/mini-xdr/backend/app/agents/edr_agent.py`

**Key Capabilities:**
- Process monitoring and termination
- File quarantine and restoration
- Memory dump collection
- Registry modification detection
- PowerShell/CMD abuse detection
- Host isolation (network level)

**Agent Structure:**
```python
"""
Endpoint Detection & Response (EDR) Agent
Monitors and responds to endpoint threats on Windows systems
"""

class EDRAgent:
    """
    AI Agent for Endpoint Detection & Response
    
    Capabilities:
    - Process monitoring and termination
    - File system integrity monitoring
    - Registry change detection
    - Memory analysis
    - Host network isolation
    - Malware quarantine
    """
    
    def __init__(self, llm_client=None):
        self.agent_id = "edr_agent_v1"
        # Windows agent configurations
        self.agents = {}  # hostname -> agent connection
    
    async def kill_process(self, hostname: str, process_name: str) -> Dict:
        """Terminate malicious process on Windows endpoint"""
        # Implementation
    
    async def quarantine_file(self, hostname: str, file_path: str) -> Dict:
        """Move suspicious file to quarantine location"""
        # Implementation
    
    async def collect_memory_dump(self, hostname: str) -> Dict:
        """Collect memory dump for forensic analysis"""
        # Implementation
    
    async def isolate_host(self, hostname: str, level: str = "strict") -> Dict:
        """Isolate endpoint from network"""
        # Implementation
    
    async def analyze_process_behavior(self, hostname: str, pid: int) -> Dict:
        """Analyze process behavior for malicious indicators"""
        # Implementation
```

**Deliverable:** EDR Agent (600+ lines) - [DETAILED CODE AVAILABLE ON REQUEST]

---

### **Task 2.3: DLP Agent Development (Day 13)**

**Objective:** Create Data Loss Prevention agent

---

### **Task 2.4: Compliance Agent Development (Day 14)**

**Objective:** Create Compliance & Governance agent for audit trails

---

### **Task 2.5: Remediation Agent Development (Day 14)**

**Objective:** Create Automated Remediation agent for recovery

---

### **Phase 2 Completion Checklist**

- [ ] IAM Agent created and tested (3 days)
- [ ] EDR Agent created and tested (2 days)
- [ ] DLP Agent created and tested (1 day)
- [ ] Compliance Agent created and tested (1 day)
- [ ] Remediation Agent created and tested (1 day)
- [ ] All agents integrated into backend API
- [ ] Agent orchestration tested
- [ ] Documentation completed

**Phase 2 Output:**
- ✅ 5 new enterprise agents operational
- ✅ API endpoints for all agent actions
- ✅ Agent coordination working

**Estimated Duration:** 7 days (1 week)

---

## **PHASE 3: Infrastructure & Integration (Week 3: Days 15-21)**

### Priority: 🟢 **MEDIUM** - Now safe to deploy infrastructure

### Goal
Deploy Mini Corp Azure infrastructure and integrate with agents

---

### **Task 3.1: Deploy Mini Corp Azure Infrastructure (Days 15-16)**

**Objective:** Create secure corporate network on Azure

#### NOW SAFE TO DEPLOY (Agents and models ready)

**Network Architecture:**
```yaml
Azure Resource Group: mini-corp-rg
Location: East US

Virtual Network: mini-corp-vnet (10.100.0.0/16)
  Subnets:
    - management-subnet: 10.100.0.0/24 (VPN only)
    - server-subnet: 10.100.1.0/24
    - workstation-subnet: 10.100.2.0/24
    - security-subnet: 10.100.3.0/24

Virtual Machines:
  1. DC01 (Domain Controller)
     - Size: Standard_D2s_v3
     - OS: Windows Server 2022
     - IP: 10.100.1.1
     - Role: Active Directory, DNS
  
  2. FS01 (File Server)
     - Size: Standard_D2s_v3
     - OS: Windows Server 2022
     - IP: 10.100.1.2
     - Role: File shares, data storage
  
  3. WEB01 (Web Server)
     - Size: Standard_B2s
     - OS: Windows Server 2022
     - IP: 10.100.1.3
     - Role: IIS web server
  
  4. DB01 (Database)
     - Size: Standard_D2s_v3
     - OS: Windows Server 2022
     - IP: 10.100.1.4
     - Role: SQL Server
  
  5-7. WS01-03 (Workstations)
     - Size: Standard_B1s
     - OS: Windows 10/11
     - IPs: 10.100.2.1-3
     - Role: Employee workstations
  
  8. XDR-COLLECTOR (Monitoring)
     - Size: Standard_B2s
     - OS: Ubuntu 22.04
     - IP: 10.100.3.10
     - Role: Log collection, agent management

Network Security Groups:
  - management-nsg: Only your IP via VPN
  - internal-nsg: Internal traffic only
  - monitoring-nsg: XDR collector access
```

**Deployment Script:**
```bash
# File: scripts/mini-corp/deploy-mini-corp-azure.sh

#!/bin/bash

az group create --name mini-corp-rg --location eastus

# Create VNet
az network vnet create \
  --resource-group mini-corp-rg \
  --name mini-corp-vnet \
  --address-prefix 10.100.0.0/16

# Create subnets
az network vnet subnet create \
  --resource-group mini-corp-rg \
  --vnet-name mini-corp-vnet \
  --name server-subnet \
  --address-prefix 10.100.1.0/24

# Create VMs (repeat for each)
az vm create \
  --resource-group mini-corp-rg \
  --name DC01 \
  --image Win2022Datacenter \
  --size Standard_D2s_v3 \
  --vnet-name mini-corp-vnet \
  --subnet server-subnet \
  --private-ip-address 10.100.1.1 \
  --admin-username azureadmin \
  --generate-ssh-keys

# Continue for all VMs...
```

**Deliverable:** Mini Corp infrastructure running on Azure

---

### **Task 3.2: Deploy Windows Agents (Days 17-18)**

**Objective:** Install monitoring agents on all Windows endpoints

**Agents to Deploy:**
1. **Sysmon** - Process, network, file monitoring
2. **Winlogbeat** - Windows Event Log forwarding
3. **OSQuery** - Endpoint inventory and queries

**Deployment Script:**
```powershell
# File: scripts/mini-corp/deploy-windows-agents.ps1

# Deploy to all Windows VMs
$computers = @("DC01", "FS01", "WEB01", "DB01", "WS01", "WS02", "WS03")

foreach ($computer in $computers) {
    # Install Sysmon
    Invoke-Command -ComputerName $computer -ScriptBlock {
        # Download and install Sysmon
        # Configure with SwiftOnSecurity config
    }
    
    # Install Winlogbeat
    Invoke-Command -ComputerName $computer -ScriptBlock {
        # Install Winlogbeat
        # Configure forwarding to XDR collector
    }
    
    # Install OSQuery
    Invoke-Command -ComputerName $computer -ScriptBlock {
        # Install OSQuery
        # Configure queries
    }
}
```

---

### **Task 3.3: Configure Mini Corp Workflows (Day 19)**

**Objective:** Create 15+ Mini Corp specific workflows

**Workflows to Create:**
1. Kerberos Golden Ticket Response
2. Kerberos Silver Ticket Response
3. DCSync Attack Containment
4. Pass-the-Hash Detection
5. Mimikatz Credential Dump Response
6. PSExec Lateral Movement
7. WMI Lateral Movement
8. RDP Brute Force
9. Suspicious PowerShell Execution
10. Registry Persistence Detection
11. Scheduled Task Abuse
12. Service Creation Abuse
13. NTDS.dit Theft Response
14. Group Policy Abuse
15. Insider Data Exfiltration

---

### **Task 3.4: End-to-End Testing (Days 20-21)**

**Objective:** Validate complete detection and response pipeline

**Attack Simulations:**
```bash
# Test 1: Kerberos Golden Ticket
python3 tests/mini-corp/test_golden_ticket_attack.py

# Test 2: Lateral Movement
python3 tests/mini-corp/test_lateral_movement.py

# Test 3: Data Exfiltration
python3 tests/mini-corp/test_data_exfiltration.py

# Test 4: Credential Theft
python3 tests/mini-corp/test_credential_theft.py

# Test 5: Privilege Escalation
python3 tests/mini-corp/test_privilege_escalation.py
```

**Success Criteria:**
- [ ] All attacks detected (100%)
- [ ] Incidents created automatically
- [ ] Workflows trigger correctly
- [ ] Agents execute actions
- [ ] Actions succeed (>90%)
- [ ] UI displays all activity
- [ ] Audit trail complete

---

## 📊 **Success Metrics**

### Detection Coverage
- [ ] 95%+ detection rate for all 12 attack types
- [ ] <5% false positive rate
- [ ] <2 second detection latency
- [ ] 100% attack attribution

### Response Effectiveness
- [ ] 90%+ action success rate
- [ ] <30 second response time
- [ ] 100% audit trail coverage
- [ ] Zero unauthorized actions

### UI/UX Requirements
- [ ] All events visible in timeline
- [ ] Incidents show complete context
- [ ] Actions logged with status
- [ ] Real-time updates
- [ ] Audit trail searchable/filterable

---

## 🔐 **Security Checklist**

Before going live:
- [ ] All credentials in Azure Key Vault
- [ ] No hardcoded secrets
- [ ] HTTPS enforced
- [ ] Azure AD authentication
- [ ] NSGs properly configured
- [ ] VPN access only
- [ ] MFA enforced
- [ ] Logging enabled
- [ ] Backups configured
- [ ] Disaster recovery tested

---

## 📁 **Files to Create (Summary)**

### Phase 1 (ML Models)
1. `datasets/windows_ad_datasets/` - Training data
2. `scripts/data-processing/convert_windows_datasets.py` - Converter
3. `aws/train_enterprise_model.py` - Training script
4. `models/enterprise/model.pt` - Trained model
5. `backend/app/feature_extractor.py` - Feature extraction
6. `scripts/testing/test_enterprise_detection.py` - Tests

### Phase 2 (Agents)
7. `backend/app/agents/iam_agent.py` - IAM Agent (1,200 lines)
8. `backend/app/agents/edr_agent.py` - EDR Agent (600 lines)
9. `backend/app/agents/dlp_agent.py` - DLP Agent (400 lines)
10. `backend/app/agents/compliance_agent.py` - Compliance (300 lines)
11. `backend/app/agents/remediation_agent.py` - Remediation (400 lines)
12. `backend/app/main.py` - API endpoints (add 50+ lines)
13. `scripts/testing/test_iam_agent.sh` - IAM tests
14. `scripts/testing/test_edr_agent.sh` - EDR tests

### Phase 3 (Infrastructure)
15. `scripts/mini-corp/deploy-mini-corp-azure.sh` - Azure deployment
16. `scripts/mini-corp/deploy-windows-agents.ps1` - Agent deployment
17. `scripts/mini-corp/setup-mini-corp-workflows.py` - Workflow setup
18. `tests/mini-corp/test_*.py` - Attack simulation tests

---

## 🎯 **Priority Order (Critical Path)**

**MUST DO FIRST (Blocking):**
1. Task 1.1: Collect Windows training data (Days 1-2)
2. Task 1.2: Retrain ML models (Days 3-5)
3. Task 1.3: Integrate models (Days 6-7)

**MUST DO SECOND:**
4. Task 2.1: Build IAM Agent (Days 8-10)
5. Task 2.2: Build EDR Agent (Days 11-12)

**CAN DO LAST:**
6. Task 2.3-2.5: Other agents (Days 13-14)
7. Task 3.1-3.4: Infrastructure & testing (Days 15-21)

---

## 📞 **Quick Start Commands**

```bash
# Week 1: ML Training
cd /Users/chasemad/Desktop/mini-xdr
python3 scripts/data-processing/download_windows_datasets.py
python3 scripts/data-processing/convert_windows_datasets.py
python3 aws/train_enterprise_model.py --epochs 50

# Week 2: Agent Development
# Create agents in backend/app/agents/
# Test with: ./scripts/testing/test_iam_agent.sh

# Week 3: Infrastructure
az login
./scripts/mini-corp/deploy-mini-corp-azure.sh
./scripts/mini-corp/deploy-windows-agents.ps1
python3 tests/mini-corp/test_all_attacks.py
```

---

## ✅ **Final Deliverables**

**At the end of 3 weeks, you will have:**

1. ✅ **ML Models** - 13-class detector + 4 specialists (85%+ accuracy)
2. ✅ **5 New Agents** - IAM, EDR, DLP, Compliance, Remediation
3. ✅ **Azure Infrastructure** - 8 VMs running Mini Corp network
4. ✅ **15+ Workflows** - Corporate-specific detection & response
5. ✅ **Complete Testing** - All attacks simulated and detected
6. ✅ **UI Integration** - Full visibility and audit trail
7. ✅ **Production Ready** - Secure, monitored, documented

---

## 📝 **Handoff Notes**

**Current Status:** Ready to begin Phase 1

**Next Action:** Download Windows/AD attack datasets

**Estimated Effort:** 3 weeks (21 days) full-time

**Budget:** ~$300-500/month Azure costs (can be reduced with auto-shutdown)

**Risk Level:** LOW - All components architected, just need implementation

**Confidence Level:** HIGH - Clear path, proven technology, experienced team

---

**Document Version:** 1.0  
**Created:** October 6, 2025  
**Last Updated:** October 6, 2025  
**Status:** 🟢 Ready for Implementation

---


