# Workflow Types: Automatic vs. On-Demand

## 🔄 **Type 1: AUTOMATIC TRIGGERS (Always Running)**

### How It Works Right Now

```
Honeypot Event ───▶ Event Ingestion ───▶ Incident Detection ───▶ TRIGGER EVALUATION (AUTOMATIC)
                    (Cowrie logs)         (Pattern matching)      (Workflows execute automatically!)
                                                                          │
                                                                          ▼
                                                                   ┌─────────────────┐
                                                                   │ Trigger Rules:  │
                                                                   │ • SSH Brute     │
                                                                   │ • SQL Injection │
                                                                   │ • Malware       │
                                                                   └─────────────────┘
                                                                          │
                                                                          ▼
                                                              ┌──────────────────────┐
                                                              │ Workflow Executes!   │
                                                              │ 1. Block IP          │
                                                              │ 2. Create incident   │
                                                              │ 3. Alert team        │
                                                              └──────────────────────┘
```

### Code Flow (backend/app/main.py:925-940)

```python
# When incident is created from events...
if incident:
    # ✅ THIS IS AUTOMATIC - Runs for every incident!
    executed_workflows = await trigger_evaluator.evaluate_triggers_for_incident(
        db, incident, recent_events
    )

    if executed_workflows:
        logger.info(f"✓ Executed {len(executed_workflows)} workflows for incident #{incident.id}")
```

### Example: SSH Brute Force Trigger (ALWAYS ACTIVE)

**Trigger Configuration** (Created once, runs forever):
```json
{
  "name": "SSH Brute Force Detection",
  "enabled": true,           // ✅ Always watching!
  "auto_execute": true,      // ✅ Runs automatically!
  "conditions": {
    "event_type": "cowrie.login.failed",
    "threshold": 6,          // Trigger if ≥6 failed logins
    "window_seconds": 60     // Within 60 seconds
  },
  "workflow_steps": [
    {"action_type": "block_ip", "parameters": {...}},
    {"action_type": "send_notification", "parameters": {...}}
  ]
}
```

**What Happens Automatically:**
1. ⚡ **Event Stream**: Honeypot logs 8 failed SSH logins from 203.0.113.50
2. ⚡ **Incident Created**: System creates Incident #23
3. ⚡ **Trigger Evaluation**: `trigger_evaluator.evaluate_triggers_for_incident()` runs
4. ⚡ **Condition Match**: "SSH Brute Force Detection" trigger conditions met
5. ⚡ **Workflow Executes**: IP blocked, notification sent (NO HUMAN INTERVENTION)

**This is 100% automated and always running!**

---

## 🎯 **Type 2: ON-DEMAND WORKFLOWS (Case-by-Case)**

These are for **one-off situations** that don't match automatic trigger patterns.

### Current Methods

#### 2a. Manual API Creation
```python
POST /api/response/workflows/create
{
  "incident_id": 23,
  "playbook_name": "Custom Response",
  "steps": [
    {"action_type": "block_ip", "parameters": {"ip_address": "198.51.100.23"}},
    {"action_type": "isolate_host", "parameters": {"host_id": "web-server-01"}}
  ],
  "auto_execute": true
}
```

**Use Case**: Security analyst sees unusual activity and manually creates a response.

#### 2b. NLP-Based Creation
```python
POST /api/workflows/nlp/create
{
  "text": "Block IP 198.51.100.23 and isolate web-server-01",
  "incident_id": 23,
  "auto_execute": true
}
```

**Use Case**: Same as above, but using natural language instead of JSON.

#### 2c. Template-Based
```python
GET /api/workflows/templates
# Returns: "Malware Response", "DDoS Mitigation", etc.

# User selects template, fills in parameters, creates workflow
```

---

## 🆕 **What We Can Add**

### Enhancement 1: Quick Action Commands (Slack-bot style)

**Purpose**: Execute immediate actions without creating a full workflow.

```
Current:  "Block IP 1.2.3.4" → Creates workflow → Review → Execute (3 steps)
New:      "Block IP 1.2.3.4" → DONE! (1 step, instant)
```

#### Implementation

**New Endpoint**: `/api/actions/execute` (Immediate execution)

```python
@app.post("/api/actions/execute")
async def execute_immediate_action(request: ImmediateActionRequest):
    """Execute a single action immediately without creating a workflow"""

    # Validate action type
    if request.action_type not in ALLOWED_IMMEDIATE_ACTIONS:
        raise HTTPException(400, "Action requires workflow approval")

    # Execute immediately
    result = await response_engine.execute_single_action(
        action_type=request.action_type,
        parameters=request.parameters
    )

    return {"success": True, "result": result}
```

**Allowed Immediate Actions** (Safe, non-destructive):
- ✅ `block_ip` (temporary blocks only, <24h)
- ✅ `unblock_ip`
- ✅ `send_notification`
- ✅ `threat_intel_lookup`
- ❌ `terminate_process` (requires workflow + approval)
- ❌ `delete_files` (requires workflow + approval)

**Frontend Example**:
```typescript
// Quick action button in UI
<Button onClick={() => quickBlockIP("198.51.100.23")}>
  ⚡ Block IP Now
</Button>

const quickBlockIP = async (ip: string) => {
  const result = await fetch('/api/actions/execute', {
    method: 'POST',
    body: JSON.stringify({
      action_type: "block_ip",
      parameters: { ip_address: ip, duration: 3600 }
    })
  })

  // Done! No workflow created, instant execution
  toast.success("IP blocked!")
}
```

**Use Cases**:
- "Someone is attacking right now - block them IMMEDIATELY"
- "Quick threat intel lookup on this IP"
- "Send urgent alert to on-call team"

---

### Enhancement 2: NLP for Creating Automatic Triggers

**Purpose**: Use natural language to CREATE new automatic triggers (not just one-off workflows).

```
Current:  NLP creates one-off workflow for specific incident
New:      NLP creates permanent trigger that runs forever
```

#### Implementation

**New Endpoint**: `/api/triggers/nlp/create`

```python
@app.post("/api/triggers/nlp/create")
async def create_trigger_from_natural_language(request: NLPTriggerRequest):
    """
    Create a permanent automatic trigger from natural language

    Example:
    "Set up a trigger to automatically block any IP with more than 5
     failed SSH logins within 60 seconds"
    """

    # Parse the rule
    trigger_intent = await parse_trigger_rule(request.text)

    # Create trigger in database
    trigger = WorkflowTrigger(
        name=trigger_intent.name,
        enabled=True,
        auto_execute=trigger_intent.auto_execute,
        conditions=trigger_intent.conditions,
        workflow_steps=trigger_intent.actions
    )

    db.add(trigger)
    await db.commit()

    return {"trigger_id": trigger.id, "message": "Trigger created and active!"}
```

**User Examples**:

```
User: "Create a trigger to automatically block IPs with more than 10
       failed login attempts in 5 minutes"

System: ✅ Created trigger "Auto-block brute force"
        • Condition: ≥10 failed logins within 300 seconds
        • Action: Block IP for 1 hour
        • Status: Active and monitoring
```

```
User: "Set up automatic ransomware response: If malware detected with
       risk score >0.8, isolate the host and alert the team"

System: ✅ Created trigger "Ransomware Auto-Response"
        • Condition: event_type=malware AND risk_score≥0.8
        • Actions: 1) Isolate host  2) Alert security team
        • Status: Active and monitoring
```

**Key Difference from On-Demand Workflows**:
- **On-Demand**: "Block IP 1.2.3.4" → Executes once, right now
- **Automatic Trigger**: "Block IPs with >10 failed logins" → Runs forever, matches any IP

---

## 📊 **Comparison Chart**

| Feature | Automatic Triggers | On-Demand Workflows | Quick Actions |
|---------|-------------------|---------------------|---------------|
| **Setup** | One-time (permanent) | Every incident | No setup needed |
| **Execution** | Automatic (when conditions match) | Manual (analyst decides) | Instant (one command) |
| **Use Case** | Known attack patterns | Custom responses | Emergency actions |
| **Scope** | Matches any incident meeting conditions | Specific to one incident | Single action |
| **Examples** | "Block all SSH brute force" | "Respond to this specific ransomware" | "Block this IP now" |
| **Status** | ✅ Already built! | ✅ Already built! | 🆕 New feature |

---

## 🎯 **Recommended UI Flow**

### Workflows Page - New Layout

```
┌────────────────────────────────────────────────────────────┐
│  Workflow Automation                                       │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  [🔄 Automatic Triggers] [⚡ Quick Actions] [📋 Workflows] │
│  ─────────────────────                                     │
│                                                            │
│  ┌─────────────────────────────────────────────────────┐  │
│  │ 🔄 Automatic Triggers (Always Running)             │  │
│  ├─────────────────────────────────────────────────────┤  │
│  │                                                     │  │
│  │  ✅ SSH Brute Force Detection                      │  │
│  │     ├─ Condition: ≥6 failed logins in 60s          │  │
│  │     ├─ Actions: Block IP → Alert team              │  │
│  │     └─ Status: 🟢 Active • 47 triggers today       │  │
│  │                                                     │  │
│  │  ✅ SQL Injection Response                         │  │
│  │     ├─ Condition: SQL pattern detected             │  │
│  │     ├─ Actions: Analyze payload → Block IP         │  │
│  │     └─ Status: 🟢 Active • 3 triggers today        │  │
│  │                                                     │  │
│  │  [+ Create New Trigger] [💬 Use Natural Language]  │  │
│  └─────────────────────────────────────────────────────┘  │
│                                                            │
│  ┌─────────────────────────────────────────────────────┐  │
│  │ ⚡ Quick Actions (Instant Execution)                │  │
│  ├─────────────────────────────────────────────────────┤  │
│  │                                                     │  │
│  │  💬 "Block IP 198.51.100.23"                       │  │
│  │     [⚡ Execute Now]                                │  │
│  │                                                     │  │
│  │  Or use quick commands:                            │  │
│  │  [🚫 Block IP] [✅ Unblock IP] [🔍 Threat Lookup] │  │
│  │                                                     │  │
│  └─────────────────────────────────────────────────────┘  │
│                                                            │
│  ┌─────────────────────────────────────────────────────┐  │
│  │ 📋 Recent Workflows (Case-by-Case)                 │  │
│  ├─────────────────────────────────────────────────────┤  │
│  │                                                     │  │
│  │  Workflow #127 - Manual Incident Response          │  │
│  │  ├─ Incident: #23 (198.51.100.45)                  │  │
│  │  ├─ Status: ✅ Completed (4/4 steps)               │  │
│  │  └─ Created: Manual (Security Analyst)             │  │
│  │                                                     │  │
│  │  Workflow #126 - NLP: Block attacker               │  │
│  │  ├─ Incident: #22 (203.0.113.50)                   │  │
│  │  ├─ Status: ✅ Completed (2/2 steps)               │  │
│  │  └─ Created: Natural Language                      │  │
│  │                                                     │  │
│  │  [Create New Workflow]                             │  │
│  └─────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────┘
```

---

## 💡 **Summary**

### What You Have Now ✅
1. **Automatic Triggers** - Always running, execute workflows when conditions match
   - Created once, runs forever
   - Example: SSH brute force trigger blocks IPs automatically
   - Status: **Fully implemented and working!**

2. **On-Demand Workflows** - Create custom workflows for specific incidents
   - Manual API, NLP, or template-based
   - Executes once for that specific incident
   - Status: **Fully implemented and working!**

### What We Can Add 🆕
1. **Quick Actions** - Instant execution without workflow creation
   - "Block this IP now" → Done in 1 second
   - No workflow overhead for simple actions
   - Status: **New feature to build**

2. **NLP for Creating Triggers** - Use natural language to set up permanent automatic triggers
   - "Create a trigger for brute force attacks" → Sets up permanent rule
   - Status: **New feature to build**

3. **Conversational Chat** - Multi-turn dialogue for building workflows/triggers
   - Ask clarifying questions
   - Guide users through complex setups
   - Status: **New feature to build**

---

**The key insight**: You already have #1 (automatic/always running)! The NLP/API methods are for #2 (one-off custom responses). We can add #3 and #4 to make the system even more powerful and user-friendly.

Does this clarify the difference? Want me to build any of the new features?
