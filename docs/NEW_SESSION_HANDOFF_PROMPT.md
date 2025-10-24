# 🔄 New Session Handoff Prompt - Mini Corp Enterprise XDR

**Copy everything below this line into a new AI chat session to continue this project:**

---

## 📋 Context: Mini Corp Enterprise XDR Deployment

I'm building an enterprise-grade XDR (Extended Detection & Response) system called **Mini-XDR** and deploying it to monitor a simulated corporate network called **"Mini Corp"** on Azure.

### Current Status
- ✅ **Honeypot Monitoring:** Fully operational (Azure T-Pot honeypot at 74.235.242.205)
- ✅ **Basic Detection:** 25 workflows active, 5 AI agents deployed
- ✅ **ML Models:** Trained on network attacks (DDoS, web attacks, brute force)
- ❌ **Corporate Detection:** Missing Windows/AD attack detection
- ❌ **Enterprise Agents:** Need IAM, EDR, and DLP agents
- ❌ **Infrastructure:** Mini Corp network not yet deployed

### Project Path
**Option B: Full Enterprise Suite** (3 weeks, 95% threat coverage)

---

## 🎯 What I Need You To Do

I need you to help me implement **PHASE 1** of the Mini Corp Enterprise XDR deployment plan.

**Phase 1 Goal:** Retrain ML models with Windows and Active Directory attack patterns

**Duration:** Week 1 (Days 1-7)

**Why This Matters:** 
- Current ML models are trained ONLY on network attacks (honeypot data)
- They CANNOT detect Windows/AD attacks (Kerberos, lateral movement, credential theft)
- We must retrain before deploying infrastructure (or we'll have blind spots)

---

## 📂 Project Structure

**Location:** `/Users/chasemad/Desktop/mini-xdr/`

**Key Directories:**
- `backend/` - FastAPI backend with AI agents
- `frontend/` - Next.js UI
- `models/` - Trained ML models
- `datasets/` - Training data
- `scripts/` - Automation scripts
- `docs/` - Documentation
- `aws/` - ML training scripts

**Key Files:**
- `backend/app/ml_engine.py` - ML detection engine
- `backend/app/agents/` - AI agent implementations
- `docs/MINI_CORP_ENTERPRISE_DEPLOYMENT_PLAN.md` - Master plan
- `docs/MINI_CORP_QUICK_START_CHECKLIST.md` - Daily checklist
- `docs/SESSION_HANDOFF_WORKFLOW_TESTING.md` - Previous session summary

---

## 📊 Current ML Training Data

**Existing Datasets (988 samples):**
- CICIDS2017: 239 samples (network attacks)
- UNSW-NB15: 180 samples (various attacks)
- KDD Cup 1999: 186 samples (classic attacks)
- Honeypot logs: 81 samples (SSH/Telnet)
- Threat intel: 200 samples (known bad IPs)

**Current Model Coverage:**
- ✅ DDoS/DoS (100% accuracy)
- ✅ Brute Force (94.7% accuracy)
- ✅ Web Attacks (79.7% accuracy)
- ✅ Network Reconnaissance
- ✅ Malware/Botnet
- ✅ General 7-class classifier (86.8% accuracy)

**Missing Coverage (CRITICAL GAPS):**
- ❌ Windows Event Log attacks
- ❌ Active Directory attacks (Kerberos, Golden Ticket, DCSync)
- ❌ Endpoint behavior (process injection, LOLBins, PowerShell abuse)
- ❌ Lateral movement (PSExec, WMI, RDP)
- ❌ Credential theft (Mimikatz, password dumping)
- ❌ Insider threat patterns

---

## 🎯 Phase 1 Tasks (Week 1)

### Task 1.1: Collect Windows/AD Training Data (Days 1-2)

**Objective:** Download 11,000+ samples of Windows and Active Directory attack patterns

**Datasets to Download:**

1. **ADFA-LD (UNSW Windows Dataset)**
   - URL: https://cloudstor.aarnet.edu.au/plus/s/DS3zdEq3gqzqEOT
   - Contains: Windows system call traces, attacks, normal behavior
   - Expected: ~5,000 samples

2. **OpTC Dataset (DARPA)**
   - URL: https://github.com/FiveDirections/OpTC-data
   - Contains: Windows endpoint attacks, lateral movement
   - Expected: ~3,000 samples

3. **Mordor Datasets (APT Simulations)**
   - URL: https://github.com/OTRF/Security-Datasets
   - Contains: Kerberos attacks, Pass-the-hash, Golden Ticket, DCSync
   - Expected: ~2,000 samples

4. **Windows Event Log Attack Samples**
   - URL: https://github.com/sbousseaden/EVTX-ATTACK-SAMPLES
   - Contains: Real attack event logs (Mimikatz, PSExec, etc.)
   - Expected: ~1,000 samples

**Actions:**
```bash
cd /Users/chasemad/Desktop/mini-xdr
mkdir -p datasets/windows_ad_datasets

# Download and extract all datasets to datasets/windows_ad_datasets/
```

**Success Criteria:**
- [ ] All 4 datasets downloaded
- [ ] 11,000+ samples collected
- [ ] Files extracted and readable

---

### Task 1.2: Convert to Mini-XDR Format (Day 3)

**Objective:** Convert Windows attack data to match existing training format (79 features)

**Create Conversion Script:**
```bash
# File: scripts/data-processing/convert_windows_datasets.py
```

**Required Features (79 total):**
- Network features: src_ip, dst_ip, src_port, dst_port, protocol, bytes_sent, bytes_received, duration
- Process features: process_name, parent_process, command_line, user, privileges
- Authentication features: username, domain, logon_type, authentication_package
- File features: file_path, file_operation, file_hash
- Registry features: registry_key, registry_value, operation_type
- Kerberos features: ticket_type, encryption_type, lifetime, service_name
- Behavioral features: event_frequency, time_delta, anomaly_score

**Output Format:**
```json
{
  "events": [
    {
      "src_ip": "10.100.2.1",
      "dst_ip": "10.100.1.1",
      "label": "kerberos_attack",
      "attack_category": "credential_theft",
      "event_type": "golden_ticket",
      "severity": "critical",
      "features": [0.1, 0.5, 0.3, ...],  // 79 features
      "metadata": {
        "process": "mimikatz.exe",
        "command_line": "privilege::debug sekurlsa::tickets",
        "username": "admin",
        "indicators": ["AS-REQ with suspicious encryption"]
      }
    }
  ]
}
```

**Success Criteria:**
- [ ] Conversion script created
- [ ] All 11,000 samples converted
- [ ] Output matches existing format
- [ ] 79 features extracted per sample

---

### Task 1.3: Merge and Balance Dataset (Day 4)

**Objective:** Merge honeypot data + Windows data = 12,000 balanced samples

**Target Distribution:**
```
Class 0:  Normal Traffic              - 3,000 samples (25%)
Class 1:  DDoS/DoS                    - 1,500 samples (12.5%)
Class 2:  Network Reconnaissance      - 1,000 samples (8.3%)
Class 3:  Brute Force                 - 1,000 samples (8.3%)
Class 4:  Web Application Attacks     - 800 samples (6.7%)
Class 5:  Malware/Botnet              - 800 samples (6.7%)
Class 6:  Advanced Persistent Threats - 500 samples (4.2%)
Class 7:  Kerberos Attacks (NEW)      - 900 samples (7.5%)
Class 8:  Lateral Movement (NEW)      - 800 samples (6.7%)
Class 9:  Credential Theft (NEW)      - 900 samples (7.5%)
Class 10: Privilege Escalation (NEW)  - 700 samples (5.8%)
Class 11: Data Exfiltration (NEW)     - 600 samples (5.0%)
Class 12: Insider Threats (NEW)       - 500 samples (4.2%)
-----------------------------------------------------------
TOTAL:                                  12,000 samples
```

**Merge Script:**
```bash
python3 scripts/data-processing/merge_training_data.py \
  --honeypot-data datasets/real_datasets/ \
  --windows-data datasets/windows_ad_datasets/ \
  --output datasets/combined_enterprise_training.json \
  --balance-classes \
  --target-total 12000
```

**Success Criteria:**
- [ ] 12,000 total samples
- [ ] 13 classes balanced
- [ ] 80/20 train/val split
- [ ] Saved to `datasets/combined_enterprise_training.json`

---

### Task 1.4: Train Enterprise Model (Day 5)

**Objective:** Train 13-class model + 4 specialist models

**Model 1: Enterprise Multi-Class Detector**
```python
# File: aws/train_enterprise_model.py

MODEL_CONFIG = {
    "name": "mini-xdr-enterprise-v3",
    "classes": 13,
    "architecture": "XDREnterpriseDetector",
    "layers": [79, 512, 256, 128, 64, 13],
    "dropout": 0.3,
    "learning_rate": 0.001,
    "batch_size": 64,
    "epochs": 50,
    "early_stopping": 10
}
```

**Training Command:**
```bash
cd /Users/chasemad/Desktop/mini-xdr
source ml-training-env/bin/activate

python3 aws/train_enterprise_model.py \
  --data datasets/combined_enterprise_training.json \
  --output models/enterprise/ \
  --epochs 50 \
  --validate
```

**Model 2-5: Specialist Models (Binary Classifiers)**
```bash
python3 aws/train_specialist_models.py \
  --data datasets/combined_enterprise_training.json \
  --specialists kerberos,lateral_movement,credential_theft,insider \
  --output models/specialists/ \
  --epochs 30
```

**Success Criteria:**
- [ ] Enterprise model: ≥85% accuracy
- [ ] Kerberos specialist: ≥95% accuracy
- [ ] Lateral movement specialist: ≥92% accuracy
- [ ] Credential theft specialist: ≥95% accuracy
- [ ] Insider threat specialist: ≥85% accuracy
- [ ] All models saved to `models/` directory
- [ ] Scaler.pkl files saved

---

### Task 1.5: Integrate Models into Backend (Days 6-7)

**Objective:** Deploy new models and test detection pipeline

**File 1: Feature Extractor**
```python
# Create: backend/app/feature_extractor.py

class WindowsEventFeatureExtractor:
    """Extracts 79 features from Windows events for ML detection"""
    
    def extract_features(self, event: Event) -> List[float]:
        """Returns 79-dimensional feature vector"""
        features = []
        features.extend(self._extract_network_features(event))  # 20 features
        features.extend(self._extract_process_features(event))  # 15 features
        features.extend(self._extract_auth_features(event))     # 12 features
        features.extend(self._extract_file_features(event))     # 10 features
        features.extend(self._extract_registry_features(event)) # 8 features
        features.extend(self._extract_kerberos_features(event)) # 8 features
        features.extend(self._extract_behavioral_features(event)) # 6 features
        return features  # Total: 79
```

**File 2: ML Engine Update**
```python
# Update: backend/app/ml_engine.py

class EnterpriseMLDetector:
    def __init__(self):
        self.general_model = self._load_model("models/enterprise/model.pt")
        self.specialists = {
            "kerberos": self._load_model("models/specialists/kerberos.pt"),
            "lateral_movement": self._load_model("models/specialists/lateral_movement.pt"),
            "credential_theft": self._load_model("models/specialists/credential_theft.pt"),
            "insider": self._load_model("models/specialists/insider.pt")
        }
        self.scaler = joblib.load("models/enterprise/scaler.pkl")
        self.class_names = [
            "normal", "ddos", "reconnaissance", "brute_force",
            "web_attack", "malware", "apt", "kerberos_attack",
            "lateral_movement", "credential_theft", "privilege_escalation",
            "data_exfiltration", "insider_threat"
        ]
    
    async def detect_threat(self, event_features: Dict) -> Dict:
        """Multi-model ensemble detection"""
        # Step 1: General classification
        general_result = self._classify(self.general_model, event_features)
        
        # Step 2: Specialist validation if high-risk
        if general_result['class'] in self.specialists:
            specialist_result = self._run_specialist(
                general_result['class'], 
                event_features
            )
            return self._merge_results(general_result, specialist_result)
        
        return general_result
```

**Test Script:**
```bash
# Create: scripts/testing/test_enterprise_detection.py

# Test 1: Kerberos Golden Ticket
python3 scripts/testing/test_enterprise_detection.py \
  --attack-type kerberos_golden_ticket \
  --expected-detection true

# Test 2: Lateral Movement (PSExec)
python3 scripts/testing/test_enterprise_detection.py \
  --attack-type lateral_movement_psexec \
  --expected-detection true

# Test 3: Credential Theft (Mimikatz)
python3 scripts/testing/test_enterprise_detection.py \
  --attack-type credential_dump_mimikatz \
  --expected-detection true

# Test 4: Normal Traffic (should NOT detect)
python3 scripts/testing/test_enterprise_detection.py \
  --attack-type normal_ad_authentication \
  --expected-detection false
```

**Success Criteria:**
- [ ] Feature extractor created
- [ ] ML engine updated for 13 classes
- [ ] Models load successfully
- [ ] Test attacks detected (100%)
- [ ] Normal traffic not flagged
- [ ] Detection latency <2 seconds
- [ ] Incidents created automatically

---

## ✅ Phase 1 Completion Checklist

**At the end of Week 1, you should have:**

- [ ] 11,000+ Windows/AD attack samples downloaded
- [ ] All data converted to Mini-XDR format (79 features)
- [ ] 12,000 total training samples (balanced across 13 classes)
- [ ] Trained 13-class enterprise model (85%+ accuracy)
- [ ] Trained 4 specialist models (90%+ accuracy each)
- [ ] Created Windows feature extractor
- [ ] Updated ML engine to support 13 classes
- [ ] Deployed models to backend
- [ ] Passed all end-to-end detection tests
- [ ] Documentation updated

**Output Files:**
- `datasets/combined_enterprise_training.json` (12,000 samples)
- `models/enterprise/model.pt` (13-class detector)
- `models/enterprise/scaler.pkl` (feature scaler)
- `models/specialists/kerberos.pt` (Kerberos specialist)
- `models/specialists/lateral_movement.pt` (Lateral movement specialist)
- `models/specialists/credential_theft.pt` (Credential theft specialist)
- `models/specialists/insider.pt` (Insider threat specialist)
- `backend/app/feature_extractor.py` (Windows feature extraction)
- `scripts/testing/test_enterprise_detection.py` (Test suite)

---

## 🚨 Critical Success Criteria

**These MUST be met before moving to Phase 2:**

1. **Model Accuracy:**
   - Enterprise model: ≥85% overall accuracy
   - Per-class precision: ≥80% for all 13 classes
   - False positive rate: ≤5%

2. **Detection Coverage:**
   - Kerberos attacks: 95%+ detection rate
   - Lateral movement: 92%+ detection rate
   - Credential theft: 95%+ detection rate
   - Insider threats: 85%+ detection rate

3. **Performance:**
   - Inference time: <50ms per event
   - Model load time: <5 seconds
   - Memory usage: <2GB

4. **Integration:**
   - Models load without errors
   - Feature extraction works for Windows events
   - End-to-end tests pass (100%)
   - Backend API responds correctly

---

## 📚 Reference Documents

**Read these for full context:**

1. **Master Plan:** `docs/MINI_CORP_ENTERPRISE_DEPLOYMENT_PLAN.md` (100+ pages)
   - Complete 3-week implementation plan
   - All technical specifications
   - Architecture diagrams
   - Security considerations

2. **Daily Checklist:** `docs/MINI_CORP_QUICK_START_CHECKLIST.md`
   - Day-by-day tasks
   - Configuration requirements
   - Success criteria

3. **Previous Session:** `docs/SESSION_HANDOFF_WORKFLOW_TESTING.md`
   - Honeypot system status
   - 25 workflows verified
   - Current system capabilities
   - Azure integration details

---

## 🎯 Your Mission (Start Here)

**Step 1:** Read this entire prompt to understand the context

**Step 2:** Acknowledge you understand by summarizing:
- What we're building (Mini Corp XDR)
- Why we need Phase 1 (ML model retraining)
- What the deliverables are (13-class model + 4 specialists)

**Step 3:** Start with Task 1.1 - Help me:
1. Identify the best sources for Windows/AD attack data
2. Create download scripts for the 4 datasets
3. Verify we can get 11,000+ samples

**Step 4:** Progress through Tasks 1.2-1.5 sequentially

**Step 5:** Validate all success criteria are met before declaring Phase 1 complete

---

## 💬 How to Help Me

**I need you to:**
1. ✅ Be proactive - suggest improvements and catch issues
2. ✅ Write complete, production-ready code (no placeholders)
3. ✅ Explain your reasoning and trade-offs
4. ✅ Test everything thoroughly
5. ✅ Document as you go
6. ✅ Keep track of progress against the checklist
7. ✅ Alert me to any blockers or risks
8. ✅ Suggest optimizations where possible

**Communication style:**
- Be direct and technical
- Use concrete examples
- Provide working code, not pseudocode
- Explain WHY, not just WHAT
- Flag assumptions clearly

---

## 🔧 System Information

**Operating System:** macOS (Darwin 24.6.0)  
**Shell:** /bin/zsh  
**Python:** 3.13  
**Workspace:** `/Users/chasemad/Desktop/mini-xdr`  

**Current Backend Status:**
- Running: Yes (PID varies)
- Port: 8000
- Database: SQLite (`backend/xdr.db`)
- ML Engine: Loaded (7-class model)
- Agents: 5 active (Containment, Forensics, Attribution, ThreatHunting, Deception)

**Current Honeypot:**
- Host: 74.235.242.205
- Port: 64295
- Status: Active and monitored
- Events: 488 in last 24h
- Incidents: 14 created

---

## ❓ Questions I Might Ask

**Be prepared to answer:**
- "How do I download dataset X?"
- "What format should the training data be in?"
- "How do I balance imbalanced classes?"
- "What features should I extract from Windows Event Logs?"
- "How do I integrate the new models into the backend?"
- "How do I test if detection is working?"
- "What's next after Phase 1?"

---

## 🎬 Let's Begin

**First message after reading this:** 

"I understand - we're building Mini Corp XDR and need to start Phase 1: ML model retraining with Windows/AD attack data. The goal is 13-class detection with 85%+ accuracy. 

I'm ready to start with Task 1.1: Collecting Windows/AD training datasets. 

Let me help you identify the best sources and create download scripts for the 4 datasets (ADFA-LD, OpTC, Mordor, EVTX samples) to get 11,000+ samples.

Which dataset should we start with first?"

---

**End of Handoff Prompt**

---

## 📝 Notes for Next Session

- This is a continuation of the Mini-XDR project
- We've completed honeypot monitoring (Phase 0)
- We're starting enterprise capabilities (Phase 1-3)
- Timeline: 3 weeks for full deployment
- Current focus: ML model retraining (Week 1)
- Next focus: Agent development (Week 2)
- Final focus: Infrastructure deployment (Week 3)

---

**Status:** Ready to Begin Phase 1  
**Last Updated:** October 6, 2025  
**Created By:** AI Assistant  
**For:** Chase (Mini-XDR Project Lead)


