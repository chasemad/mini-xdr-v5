# 🧪 Local Testing Plan - No Azure Required

**Prerequisites:** ✅ Frontend and Backend running locally  
**Time Required:** 45-60 minutes  
**Azure/T-Pot Required:** ❌ NO - All tests run in simulation mode

---

## ✅ What You Have Running

- ✅ Backend: `http://localhost:8000`
- ✅ Frontend: `http://localhost:3000`
- ✅ Database: SQLite (local)
- ✅ Models: Both network ensemble + Windows specialist loaded

**What You DON'T Need:**
- ❌ Azure VM
- ❌ Mini-corp network
- ❌ T-Pot honeypot
- ❌ Active Directory
- ❌ Real Windows endpoints

**Why:** All agents work in **simulation mode** - they mimic actions without touching real infrastructure.

---

## 🎯 Test Plan Overview

| Test # | Component | Time | Requires Azure? |
|--------|-----------|------|-----------------|
| 1 | Backend Health | 2 min | ❌ NO |
| 2 | Model Loading | 2 min | ❌ NO |
| 3 | ML Detection | 5 min | ❌ NO |
| 4 | Agent Actions (IAM/EDR/DLP) | 10 min | ❌ NO |
| 5 | Unified UI Display | 10 min | ❌ NO |
| 6 | Rollback Functionality | 5 min | ❌ NO |
| 7 | Auto-Refresh | 5 min | ❌ NO |
| 8 | Action Detail Modal | 5 min | ❌ NO |
| **Total** | | **~45 min** | **❌ NO AZURE** |

---

## 🧪 Test 1: Backend Health Check (2 minutes)

### Objective
Verify backend is running and all systems are accessible.

### Steps
```bash
# Open new terminal tab
cd /Users/chasemad/Desktop/mini-xdr

# Test 1.1: Backend is responding
curl http://localhost:8000/health

# Expected: {"status":"healthy"}

# Test 1.2: API docs accessible
curl http://localhost:8000/docs

# Expected: HTML response with API documentation

# Test 1.3: Check if models are loaded
curl http://localhost:8000/api/ml/models/status

# Expected: Both network_model and windows_specialist loaded
```

### Success Criteria
- [ ] Backend responds to health check
- [ ] API docs accessible
- [ ] Both models reported as loaded

---

## 🧪 Test 2: Model Loading Verification (2 minutes)

### Objective
Confirm both the network ensemble (4.436M) and Windows specialist (390K) are loaded.

### Steps
```bash
# Test 2.1: Python verification
python3 << 'EOF'
from backend.app.ensemble_ml_detector import EnsembleMLDetector
detector = EnsembleMLDetector()
print(f"✅ Network Model: {detector.network_model is not None}")
print(f"✅ Windows Specialist: {detector.windows_specialist is not None}")
print(f"📊 Total Classes: {len(detector.windows_classes)}")
EOF
```

### Expected Output
```
✅ Network Model: True
✅ Windows Specialist: True
📊 Total Classes: 13
```

### Success Criteria
- [ ] Network model loaded (4.436M samples)
- [ ] Windows specialist loaded (390K samples)
- [ ] 13 detection classes available

---

## 🧪 Test 3: ML Detection Testing (5 minutes)

### Objective
Test that the ML models can detect different attack types.

### Steps
```bash
# Test 3.1: Create test detection script
cat > test_detection.py << 'EOF'
import numpy as np
import sys
sys.path.insert(0, '/Users/chasemad/Desktop/mini-xdr')
from backend.app.ensemble_ml_detector import EnsembleMLDetector

detector = EnsembleMLDetector()

# Test cases with 79-dimensional feature vectors
test_cases = [
    ("Normal Traffic", np.random.randn(79) * 0.1),  # Low variance = normal
    ("DDoS Attack", np.random.randn(79) * 5.0),     # High variance = attack
    ("Windows Attack", np.concatenate([
        np.random.randn(59) * 0.5,  # Normal network features
        np.random.randn(20) * 10.0  # Anomalous Windows features
    ]))
]

print("\n" + "="*80)
print("🧪 ML DETECTION TESTS")
print("="*80)

for name, features in test_cases:
    import asyncio
    result = asyncio.run(detector.detect_threat(features))
    print(f"\n{name}:")
    print(f"  Threat Type: {result['threat_type']}")
    print(f"  Confidence: {result['confidence']:.2%}")
    print(f"  Model Used: {result['model_used']}")

print("\n" + "="*80)
print("✅ Detection tests complete!")
print("="*80)
EOF

python3 test_detection.py
```

### Success Criteria
- [ ] Script runs without errors
- [ ] All three test cases return predictions
- [ ] Confidence scores are between 0-1
- [ ] Both models (network + windows) are used

---

## 🧪 Test 4: Agent Actions Testing (10 minutes)

### Objective
Test IAM, EDR, and DLP agent actions in simulation mode.

### Steps

#### Test 4.1: IAM Agent - Disable User Account
```bash
curl -X POST http://localhost:8000/api/agents/iam/execute \
  -H "Content-Type: application/json" \
  -d '{
    "action_name": "disable_user_account",
    "params": {
      "username": "test.user@domain.local",
      "reason": "Testing IAM agent - simulated compromise"
    },
    "incident_id": 1
  }'
```

**Expected Response:**
```json
{
  "action_id": "iam_action_...",
  "status": "success",
  "message": "IAM action executed successfully",
  "result": {
    "simulated": true,
    "action": "disable_user_account",
    "username": "test.user@domain.local"
  },
  "rollback_id": "rollback_..."
}
```

#### Test 4.2: EDR Agent - Quarantine File
```bash
curl -X POST http://localhost:8000/api/agents/edr/execute \
  -H "Content-Type: application/json" \
  -d '{
    "action_name": "quarantine_file",
    "params": {
      "host": "WORKSTATION-01",
      "file_path": "C:\\Users\\user\\malware.exe",
      "reason": "Testing EDR agent - simulated malware"
    },
    "incident_id": 1
  }'
```

#### Test 4.3: DLP Agent - Scan File
```bash
curl -X POST http://localhost:8000/api/agents/dlp/execute \
  -H "Content-Type: application/json" \
  -d '{
    "action_name": "scan_file",
    "params": {
      "file_path": "/tmp/test_document.txt",
      "reason": "Testing DLP agent - simulated sensitive data"
    },
    "incident_id": 1
  }'
```

#### Test 4.4: Query All Agent Actions
```bash
curl http://localhost:8000/api/agents/actions/1
```

**Expected:** List of all 3 actions with status, timestamps, and rollback IDs

### Success Criteria
- [ ] IAM action executes successfully
- [ ] EDR action executes successfully
- [ ] DLP action executes successfully
- [ ] All actions have rollback_id
- [ ] All actions stored in database
- [ ] Query returns all 3 actions

---

## 🧪 Test 5: Unified UI Display (10 minutes)

### Objective
Verify the unified action panel shows ALL action types (manual, workflow, agent).

### Steps

#### Step 5.1: Navigate to Frontend
1. Open browser: `http://localhost:3000`
2. Navigate to any incident (or create a new one)
3. Scroll to "Unified Response Actions" section

#### Step 5.2: Verify Display
Check that you see:
- [ ] Section header: "Response Actions" or "Unified Response Actions"
- [ ] All 3 agent actions from Test 4 (IAM, EDR, DLP)
- [ ] Agent-specific colors:
  - IAM actions in **Blue** 🔵
  - EDR actions in **Purple** 🟣
  - DLP actions in **Green** 🟢
- [ ] Status badges (Success ✅ or Failed ❌)
- [ ] Timestamps ("Xm ago" format)
- [ ] Clickable rows
- [ ] Rollback buttons (if applicable)

#### Step 5.3: Visual Inspection
Take a screenshot or note:
- [ ] No duplicate sections
- [ ] Actions in chronological order
- [ ] Clear visual hierarchy
- [ ] No console errors in browser DevTools

### Success Criteria
- [ ] Unified panel is visible
- [ ] All agent actions displayed
- [ ] Color coding correct
- [ ] No duplicate sections
- [ ] Layout is clean and professional

---

## 🧪 Test 6: Rollback Functionality (5 minutes)

### Objective
Verify that agent actions can be rolled back successfully.

### Steps

#### Step 6.1: Get Rollback ID
From Test 4, get the `rollback_id` from the IAM action response.

Example: `rollback_abc123def456`

#### Step 6.2: Execute Rollback via API
```bash
# Replace ROLLBACK_ID with actual value
curl -X POST http://localhost:8000/api/agents/rollback/ROLLBACK_ID \
  -H "Content-Type: application/json"
```

**Expected Response:**
```json
{
  "status": "success",
  "message": "Action rolled back successfully",
  "rollback_result": {
    "action": "enable_user_account",
    "username": "test.user@domain.local",
    "simulated": true
  }
}
```

#### Step 6.3: Verify in UI
1. Refresh the incident page
2. Find the IAM action
3. Check status changed to "Rolled Back" 🔄
4. Check "Rolled back Xm ago" timestamp appears

#### Step 6.4: Test UI Rollback Button
1. Execute another agent action (any type)
2. Click the rollback button in the UI
3. Confirm in the dialog
4. Verify status updates immediately

### Success Criteria
- [ ] API rollback succeeds
- [ ] UI updates to show "rolled_back" status
- [ ] Timestamp displayed correctly
- [ ] UI rollback button works
- [ ] Confirmation dialog appears
- [ ] Page refreshes automatically

---

## 🧪 Test 7: Auto-Refresh Testing (5 minutes)

### Objective
Verify the unified panel auto-refreshes every 5 seconds.

### Steps

#### Step 7.1: Setup
1. Open browser to incident page
2. Open browser DevTools (F12)
3. Go to Network tab
4. Filter for XHR/Fetch requests

#### Step 7.2: Execute New Action
In another terminal, execute a new agent action:
```bash
curl -X POST http://localhost:8000/api/agents/iam/execute \
  -H "Content-Type: application/json" \
  -d '{
    "action_name": "reset_password",
    "params": {
      "username": "auto.refresh.test@domain.local",
      "reason": "Testing auto-refresh functionality"
    },
    "incident_id": 1
  }'
```

#### Step 7.3: Observe Auto-Refresh
- [ ] Wait and watch the UI (don't manually refresh)
- [ ] Within 5 seconds, new action should appear
- [ ] Check Network tab shows GET request to `/api/agents/actions/1`
- [ ] Action appears without page reload

### Success Criteria
- [ ] New action appears within 5 seconds
- [ ] No manual refresh needed
- [ ] Network requests occur every 5 seconds
- [ ] UI updates smoothly without flickering

---

## 🧪 Test 8: Action Detail Modal (5 minutes)

### Objective
Verify clicking an action opens a detailed modal with full information.

### Steps

#### Step 8.1: Click Action
1. In the unified panel, click any agent action row
2. Modal should open

#### Step 8.2: Verify Modal Content
Check that modal displays:
- [ ] Action header with agent type badge (IAM/EDR/DLP)
- [ ] Action name (e.g., "Disable User Account")
- [ ] Execution timestamp
- [ ] Status badge (Success/Failed/Rolled Back)
- [ ] Parameters section:
  - All input parameters displayed
  - Formatted as key-value pairs
  - JSON prettified if complex
- [ ] Results section:
  - Action results displayed
  - Error messages (if failed)
- [ ] Rollback section (if applicable):
  - Rollback ID shown
  - Rollback button available
  - Or "Already rolled back" if executed
- [ ] Related events (if any)
- [ ] Close button (X) works

#### Step 8.3: Test Interactions
- [ ] Click outside modal to close
- [ ] Use ESC key to close
- [ ] Click rollback button (if available)
- [ ] Verify confirmation dialog

### Success Criteria
- [ ] Modal opens on click
- [ ] All information displayed correctly
- [ ] Close mechanisms work
- [ ] Rollback button functional
- [ ] Layout is clean and readable

---

## 📊 Complete Test Checklist

### Backend Tests
- [ ] Backend health check passes
- [ ] API docs accessible
- [ ] Both models loaded (network + Windows)
- [ ] 13 detection classes available
- [ ] ML detection working
- [ ] IAM agent actions execute
- [ ] EDR agent actions execute
- [ ] DLP agent actions execute
- [ ] Actions stored in database
- [ ] Rollback API works

### Frontend Tests
- [ ] Frontend loads without errors
- [ ] Incident page accessible
- [ ] Unified action panel visible
- [ ] All agent actions displayed
- [ ] Color coding correct (Blue/Purple/Green)
- [ ] Status badges showing
- [ ] Timestamps formatted correctly
- [ ] No duplicate sections
- [ ] Auto-refresh works (5 seconds)
- [ ] Click opens detail modal
- [ ] Modal displays complete information
- [ ] Modal close mechanisms work
- [ ] Rollback button works
- [ ] Confirmation dialogs appear

### Integration Tests
- [ ] Agent actions → Database → UI flow works
- [ ] Rollback updates everywhere (API, DB, UI)
- [ ] Real-time updates work end-to-end
- [ ] No console errors
- [ ] No network errors
- [ ] Performance acceptable (<2s loads)

---

## 🐛 Troubleshooting

### Issue: Backend not responding
```bash
# Check if backend is running
ps aux | grep uvicorn

# Check logs
tail -f backend/backend.log

# Restart if needed
cd backend && source venv/bin/activate && uvicorn app.main:app --reload
```

### Issue: Frontend not loading
```bash
# Check if frontend is running
ps aux | grep next

# Check for errors
cd frontend && npm run dev

# Check browser console for errors (F12)
```

### Issue: Models not loading
```bash
# Verify model files exist
ls -lh models/windows_specialist_13class/
ls -lh models/local_trained_enhanced/general/

# Test model loading
python3 -c "
from backend.app.ensemble_ml_detector import EnsembleMLDetector
detector = EnsembleMLDetector()
print('Network:', detector.network_model is not None)
print('Windows:', detector.windows_specialist is not None)
"
```

### Issue: Agent actions failing
```bash
# Check if agents are in simulation mode (expected)
python3 -c "
from backend.app.agents.iam_agent import IAMAgent
agent = IAMAgent()
print('IAM simulation mode:', agent.ldap_connection is None)
"
```

### Issue: UI not updating
1. Open browser DevTools (F12)
2. Check Console for JavaScript errors
3. Check Network tab for failed requests
4. Hard refresh: Ctrl+Shift+R (Windows/Linux) or Cmd+Shift+R (Mac)

---

## 🎯 Success Criteria Summary

### Minimum Passing Score: 30/35 checks

**Critical (Must Pass):**
- ✅ Both models loaded
- ✅ Agent actions execute
- ✅ Actions stored in database
- ✅ Unified UI displays actions
- ✅ Rollback functionality works

**Important (Should Pass):**
- ✅ Auto-refresh works
- ✅ Modal displays details
- ✅ Color coding correct
- ✅ No errors in console

**Nice-to-Have (Optional):**
- Performance optimizations
- Visual polish
- Additional error handling

---

## ⏱️ Time Breakdown

| Phase | Duration | Can Skip? |
|-------|----------|-----------|
| Setup verification | 5 min | No |
| Backend tests | 10 min | No |
| Agent action tests | 10 min | No |
| UI tests | 15 min | No |
| Rollback tests | 5 min | No |
| Edge cases | 10 min | Yes |
| **Total Minimum** | **45 min** | |
| **Total Complete** | **60 min** | |

---

## 🎉 What Happens After Testing

### If All Tests Pass (Expected):
1. ✅ Mark system as **Production Ready**
2. ✅ Update documentation with test results
3. ✅ Create deployment package
4. ✅ Plan staging deployment (optional)
5. ✅ Plan production rollout

### If Some Tests Fail:
1. Document which tests failed
2. Analyze root cause
3. Fix issues
4. Re-run failed tests
5. Continue when passing

---

## 📝 Test Results Template

After testing, create a file `TEST_RESULTS.md`:

```markdown
# Test Results - [Date]

## Overall Status: [PASS/FAIL]

### Backend Tests: [X/10] Passed
- [ ] Models loaded
- [ ] Detection working
- [ ] Agents executing
...

### Frontend Tests: [X/15] Passed
- [ ] UI displays correctly
- [ ] Auto-refresh working
- [ ] Modal functional
...

### Integration Tests: [X/10] Passed
- [ ] End-to-end flow
- [ ] Rollback working
...

## Issues Found:
1. [Issue description]
   - Impact: High/Medium/Low
   - Fix: [What needs to be done]

## Recommendations:
- [Recommendation 1]
- [Recommendation 2]
```

---

## 🚀 Ready to Test!

You have everything you need:
- ✅ Frontend running
- ✅ Backend running
- ✅ No Azure required
- ✅ Test plan ready

**Start with Test 1 and work your way through!**

Questions to start:
1. "Run backend health check" - I'll guide you through it
2. "Test agent actions" - I'll provide the curl commands
3. "Verify UI" - I'll tell you what to look for

**What would you like to test first?** 🧪

