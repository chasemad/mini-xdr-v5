# Manual End-to-End Test Guide
## Chat → Workflow → Investigation Integration

### Prerequisites
1. ✅ Backend running on `http://localhost:8000`
2. ✅ Frontend running on `http://localhost:3000`
3. ✅ At least one incident in the system

---

## Test Scenario 1: Workflow Creation from Incident Chat

### Steps:
1. Navigate to: `http://localhost:3000/incidents/incident/8` (or any incident ID)
2. Open the AI Chat sidebar (right panel)
3. Type one of these commands:

**Test 1.1 - Block IP:**
```
Block IP 192.0.2.100 and send alert to security team
```

**Expected Result:**
- ✅ AI responds with "Workflow Created Successfully!"
- ✅ Shows workflow ID and DB ID
- ✅ Green toast notification appears: "Workflow Created"
- ✅ Workflow appears in incident detail workflows section
- ✅ Message shows approval status

**Test 1.2 - Isolate Host:**
```
Isolate this host and terminate suspicious processes
```

**Expected Result:**
- ✅ Workflow created with 2 actions: isolate_host, terminate_process
- ✅ Shows approval requirement (likely yes for destructive actions)

**Test 1.3 - Identity Protection:**
```
Reset passwords and enable MFA for compromised accounts
```

**Expected Result:**
- ✅ Workflow with identity protection actions
- ✅ Shows priority and approval status

---

## Test Scenario 2: Investigation Trigger

### Steps:
1. On the same incident page chat
2. Type investigation commands:

**Test 2.1 - Basic Investigation:**
```
Investigate this attack pattern and check for similar incidents
```

**Expected Result:**
- ✅ AI responds with "Investigation Initiated"
- ✅ Shows Case ID (format: `inv_XXXXXXXXXXXX`)
- ✅ Shows evidence count and event analysis
- ✅ Blue toast notification: "Investigation Started"
- ✅ Investigation action appears in incident actions list

**Test 2.2 - Forensics Analysis:**
```
Analyze the events and run deep forensics on this incident
```

**Expected Result:**
- ✅ Investigation started
- ✅ Event analysis breakdown shown
- ✅ Time span and event types displayed

---

## Test Scenario 3: Cross-Page Workflow Sync

### Steps:
1. Go to: `http://localhost:3000/workflows`
2. Select incident #8 from the incident selector
3. In the "Natural Language" tab, type:
```
Block this IP and isolate the host
```
4. Click "Create Workflow"
5. Navigate to: `http://localhost:3000/incidents/incident/8`

**Expected Result:**
- ✅ Workflow appears in incident detail page
- ✅ Workflow shows correct name and status
- ✅ Can execute workflow from incident page

---

## Test Scenario 4: Different Attack Types

Test different attack scenarios with appropriate responses:

### SSH Brute Force Attack:
```
Block this SSH brute force attack and alert the security team
```
**Expected Actions:** block_ip, alert_security_analysts

### DDoS Attack:
```
Deploy firewall rules to mitigate this DDoS attack
```
**Expected Actions:** deploy_firewall_rules

### Malware Detection:
```
Isolate infected host, terminate malicious processes, and capture forensics
```
**Expected Actions:** isolate_host, terminate_process, capture_network_traffic

### Data Exfiltration:
```
Block the IP, revoke user sessions, and encrypt sensitive data
```
**Expected Actions:** block_ip, revoke_user_sessions, encrypt_sensitive_data

### Phishing Attack:
```
Quarantine the email and block the sender
```
**Expected Actions:** quarantine_email, block_sender

---

## Test Scenario 5: Combined Workflow + Investigation

### Steps:
1. On incident page chat, type:
```
Block this IP, isolate the host, and investigate the attack pattern
```

**Expected Result:**
- ✅ Workflow created with block_ip and isolate_host
- ✅ Investigation started separately
- ✅ Two toast notifications (one for workflow, one for investigation)
- ✅ Both workflow and investigation action appear in incident

---

## Test Scenario 6: Approval Workflow

### Steps:
1. Type a command that requires approval:
```
Terminate all processes and delete malicious files
```

**Expected Result:**
- ✅ Workflow created
- ✅ Shows "⚠️ Requires approval before execution"
- ✅ `approval_required: true` in workflow
- ✅ Cannot execute without approval

---

## Verification Checklist

After running tests, verify:

### Frontend Indicators:
- [ ] Toast notifications appear for workflows
- [ ] Toast notifications appear for investigations
- [ ] Chat messages show formatted responses
- [ ] Incident data refreshes after workflow creation
- [ ] Investigation actions appear in action history

### Backend Logs:
Check backend terminal for:
```
INFO: Workflow creation from chat...
INFO: Investigation case created...
```

### Database Verification:
```bash
# Check workflows were created
sqlite3 backend/xdr.db "SELECT id, workflow_id, playbook_name, status FROM response_workflows ORDER BY created_at DESC LIMIT 5;"

# Check investigation actions
sqlite3 backend/xdr.db "SELECT id, action, result, detail FROM actions WHERE action='forensic_investigation' ORDER BY created_at DESC LIMIT 5;"
```

---

## Expected Performance

- **Workflow Creation:** < 2 seconds
- **Investigation Trigger:** < 3 seconds  
- **Chat Response:** < 1 second
- **UI Refresh:** Immediate (< 500ms)

---

## Troubleshooting

### Workflow Not Created:
- Check if query contains action keywords (block, isolate, alert, etc.)
- Verify NLP parser can extract IP addresses or targets
- Check backend logs for errors

### Investigation Not Triggered:
- Ensure query contains investigation keywords (investigate, analyze, examine, etc.)
- Verify incident has events for analysis
- Check forensics agent initialization

### No Toast Notifications:
- Check browser console for errors
- Verify response contains `workflow_created` or `investigation_started` flags
- Ensure showToast function is working

### Workflows Not Syncing:
- Check database foreign key constraints
- Verify incident_id matches between workflow and incident
- Test workflow endpoint directly: `GET /api/response/workflows`

---

## Success Criteria

✅ **100% Success** if ALL of the following work:
1. Workflow creation from chat with action keywords
2. Investigation trigger from chat with investigation keywords  
3. Toast notifications for both workflows and investigations
4. Incident data refreshes after creation
5. Actions appear in incident history
6. Different attack types trigger appropriate workflows
7. Approval requirements are correctly identified
8. Cross-page workflow sync works

---

## Test Results Template

```
Date: ___________
Tester: ___________

Scenario 1 (Workflow Creation): PASS / FAIL
Scenario 2 (Investigation): PASS / FAIL
Scenario 3 (Cross-Page Sync): PASS / FAIL
Scenario 4 (Attack Types): PASS / FAIL
Scenario 5 (Combined): PASS / FAIL
Scenario 6 (Approval): PASS / FAIL

Overall Status: PASS / PARTIAL / FAIL
Notes: ___________
```


