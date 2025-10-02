# How NLP Workflow Creation Actually Works

## 🎯 **Current State: NLP Creates ONE-OFF Workflows**

### What We Have Now
```
Analyst types: "Block IP 192.168.1.100"
   ↓
NLP parses text with regex patterns
   ↓
Creates workflow for THIS incident only
   ↓
Workflow executes once → Done
```

**Key Point**: This creates a **temporary workflow** for a specific incident, NOT a permanent trigger!

### What We DON'T Have Yet (But You Want)
```
Analyst types: "Create a trigger to always block IPs with >10 failed logins"
   ↓
NLP parses trigger conditions
   ↓
Creates PERMANENT trigger in database
   ↓
Trigger runs forever, matches any future incident
```

---

## 🔍 **Deep Dive: How NLP Parsing Works**

### Step 1: Pattern Matching (Primary Method)

**Code**: `backend/app/nlp_workflow_parser.py:46-83`

```python
self.action_patterns = {
    # Network actions
    r'\b(block|ban|blacklist)\s+(?:ip\s+)?(\d+\.\d+\.\d+\.\d+)': ('block_ip', 'network'),
    r'\b(isolate|quarantine)\s+(?:the\s+)?host': ('isolate_host', 'endpoint'),
    r'\b(investigate|analyze)\s+': ('investigate_behavior', 'forensics'),
    # ... 20+ more patterns
}
```

**Example Input**: "Block IP 192.168.1.100 and isolate the host"

**Regex Matching**:
1. Pattern `r'\b(block|ban)\s+(?:ip\s+)?(\d+\.\d+\.\d+\.\d+)'` matches "Block IP 192.168.1.100"
   - Captures: verb="block", ip="192.168.1.100"
   - Maps to: `block_ip` action type

2. Pattern `r'\b(isolate|quarantine)\s+(?:the\s+)?host'` matches "isolate the host"
   - Maps to: `isolate_host` action type

**Output**:
```json
{
  "actions": [
    {
      "action_type": "block_ip",
      "category": "network",
      "parameters": {
        "ip_address": "192.168.1.100",
        "reason": "NLP workflow: Block IP 192.168.1.100 and isolate the host"
      }
    },
    {
      "action_type": "isolate_host",
      "category": "endpoint",
      "parameters": {
        "reason": "NLP workflow: Block IP 192.168.1.100 and isolate the host"
      }
    }
  ]
}
```

### Step 2: OpenAI Fallback (When Regex Fails)

**Code**: `backend/app/nlp_workflow_parser.py:205-258`

```python
# Only called if pattern matching finds ZERO actions
if len(actions) == 0 and self.openai_api_key:
    prompt = f"""You are a cybersecurity response automation system.
    Convert this natural language request into a structured list of response actions.

    User request: "{text}"

    Available action types:
    - block_ip, unblock_ip, deploy_firewall_rules
    - isolate_host, un_isolate_host, terminate_process
    - reset_passwords, revoke_user_sessions, disable_user_account
    ...

    Respond in JSON format:
    {{"actions": [{{"action_type": "...", "category": "...", "reason": "..."}}]}}
    """

    response = await openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
```

**When Used**:
- Complex phrasing: "I need you to prevent further unauthorized access from that suspicious actor"
- Ambiguous requests: "Respond to the ransomware"
- Misspellings: "Blokc the attackr"

**Limitation**: Only triggers as fallback (not primary method)

---

## 🛠️ **Action Type Registry: The 68 Available Tools**

### Where Actions Are Registered

**Code**: `backend/app/advanced_response_engine.py:54-422`

```python
class ResponseEngine:
    def __init__(self):
        self.action_registry = {
            # Network Actions (7 actions)
            "block_ip": {
                "category": ActionCategory.NETWORK,
                "name": "IP Blocking",
                "description": "Block IP address using firewall rules",
                "parameters": ["ip_address", "duration", "block_level"],
                "safety_level": "medium",
                "rollback_supported": True,
                "estimated_duration": 30
            },
            "deploy_firewall_rules": {
                "category": ActionCategory.NETWORK,
                "name": "Firewall Rule Deployment",
                "description": "Deploy custom firewall rules",
                "parameters": ["rule_set", "target_zone"],
                "safety_level": "high",
                "rollback_supported": True,
                "estimated_duration": 120
            },

            # Endpoint Actions (15 actions)
            "isolate_host_advanced": {...},
            "terminate_process": {...},
            "collect_memory_dump": {...},

            # Identity Actions (8 actions)
            "reset_passwords": {...},
            "revoke_user_sessions": {...},
            "enforce_mfa": {...},

            # ... 68 total actions across 10 categories
        }
```

### The 10 Action Categories

1. **Network** (7 actions): block_ip, deploy_firewall_rules, traffic_redirection, etc.
2. **Endpoint** (15 actions): isolate_host, terminate_process, scan_endpoint, etc.
3. **Identity** (8 actions): reset_passwords, revoke_sessions, disable_accounts, etc.
4. **Forensics** (12 actions): collect_logs, memory_dump, threat_intel_lookup, etc.
5. **Data** (6 actions): backup_data, encrypt_data, check_db_integrity, etc.
6. **Email** (4 actions): quarantine_email, block_sender, analyze_headers, etc.
7. **Cloud** (5 actions): deploy_waf, update_security_groups, enable_dlp, etc.
8. **Compliance** (4 actions): generate_audit_report, enforce_policies, etc.
9. **Communication** (4 actions): alert_analysts, create_ticket, send_notification, etc.
10. **Malware** (3 actions): isolate_file, submit_to_sandbox, update_signatures, etc.

---

## ❌ **Current Limitations**

### Limitation 1: Only Maps to Existing Actions

**Problem**: If analyst needs an action that doesn't exist, NLP can't create it.

**Example**:
```
Analyst: "Restart the web server and clear cache"

NLP Parser: ❌ No pattern matches "restart web server"
           ❌ No pattern matches "clear cache"

Result: Empty workflow (fails)
```

**Fix Needed**: Either add those action types OR suggest similar actions

### Limitation 2: Cannot Extract Complex Parameters

**Problem**: NLP extracts simple parameters (IPs, hosts) but not complex configs.

**Example**:
```
Analyst: "Deploy firewall rule to block traffic from 192.168.1.0/24
          to 10.0.0.0/8 on ports 443, 8080, and 8443 using TCP protocol"

Current NLP:
  ✅ Detects: deploy_firewall_rules action
  ❌ Extracts: Only "192.168.1.0/24" as source_ip
  ❌ Missing: subnet mask, destination, ports, protocol

Result: Incomplete workflow
```

**Fix Needed**: More sophisticated parameter extraction

### Limitation 3: No Context Memory

**Problem**: Each NLP request is independent (no conversation history).

**Example**:
```
Request 1: "Block IP 192.168.1.100"
Request 2: "Also isolate the host"  ← System doesn't know which IP!

Result: Fails (no context)
```

**Fix Needed**: Conversational chat interface with session memory

### Limitation 4: Cannot Create Multi-Condition Triggers

**Problem**: NLP creates workflows, not triggers with conditions.

**Example**:
```
Analyst: "Create a trigger to block IPs with >10 failed logins
          within 5 minutes, but only if they're from external IPs"

Current NLP:
  ✅ Creates workflow with: block_ip action
  ❌ Cannot create: trigger with conditions
  ❌ Cannot specify: threshold, time window, IP filtering

Result: Creates one-off workflow instead of permanent trigger
```

**Fix Needed**: NEW NLP parser specifically for trigger creation

### Limitation 5: No Tool Discovery

**Problem**: Analyst doesn't know what actions are available.

**Example**:
```
Analyst: "Can I automatically rotate credentials?"

System: ❌ No help available
        ❌ Doesn't suggest similar actions

Reality: We HAVE "reset_passwords" and "revoke_user_sessions" actions
```

**Fix Needed**: Action discovery and suggestions

---

## ✅ **What Happens When Workflow is Created**

### Flow: NLP Text → Executable Workflow

```
1. TEXT INPUT
   "Block IP 192.168.1.100 and send alert"

2. NLP PARSING (nlp_workflow_parser.py)
   ↓
   Regex matches: "block" → block_ip
                  "send alert" → send_notification
   ↓
   Creates WorkflowIntent:
   {
     "actions": [
       {"action_type": "block_ip", "parameters": {"ip_address": "192.168.1.100"}},
       {"action_type": "send_notification", "parameters": {"channel": "slack"}}
     ],
     "priority": "medium",
     "confidence": 0.85
   }

3. WORKFLOW CREATION (nlp_workflow_routes.py:172)
   ↓
   Creates ResponseWorkflow in database:
   {
     "workflow_id": "nlp_abc123",
     "incident_id": 8,
     "steps": [...actions from above...],
     "status": "pending"
   }

4. ACTION VALIDATION (advanced_response_engine.py:791)
   ↓
   Checks if action types exist in registry:
   ✅ "block_ip" found in action_registry
   ✅ "send_notification" found in action_registry

5. EXECUTION (advanced_response_engine.py:970)
   ↓
   For each action:
     - Validate parameters
     - Route to execution handler
     - Execute via system commands/APIs

   block_ip:
     → Calls _execute_block_ip_action()
     → Runs: sudo iptables -I INPUT -s 192.168.1.100 -j DROP
     → Returns: {"success": true, "detail": "IP blocked"}

   send_notification:
     → Calls _execute_send_notification_action()
     → Sends Slack message (or logs if Slack not configured)
     → Returns: {"success": true, "detail": "Notification sent"}

6. COMPLETION
   ↓
   Workflow status → "completed"
   All actions logged to database
   Analyst sees results in UI
```

---

## 🔧 **How to Add New Actions/Tools**

### Scenario: Analyst Needs "Restart Service" Action

Currently: ❌ Not available

Let's add it:

#### Step 1: Register Action Type

**File**: `backend/app/advanced_response_engine.py`

```python
# Add to action_registry (around line 100)
"restart_service": {
    "category": ActionCategory.ENDPOINT,
    "name": "Restart Service",
    "description": "Restart a system service (Apache, Nginx, etc.)",
    "parameters": ["service_name", "host_id", "force"],
    "safety_level": "high",  # High because it affects availability
    "rollback_supported": False,  # Can't "undo" a restart
    "estimated_duration": 30
},
```

#### Step 2: Add NLP Pattern

**File**: `backend/app/nlp_workflow_parser.py`

```python
# Add to action_patterns (around line 58)
r'\brestart\s+(?:the\s+)?(\w+)\s+service': ('restart_service', 'endpoint'),
r'\breboot\s+service\s+(\w+)': ('restart_service', 'endpoint'),
```

Now NLP will recognize:
- "Restart the apache service" ✅
- "Reboot service nginx" ✅

#### Step 3: Implement Execution Handler

**File**: `backend/app/advanced_response_engine.py`

```python
async def _execute_restart_service_action(self, params: Dict[str, Any]) -> Dict[str, Any]:
    """Execute restart service action"""
    try:
        service_name = params.get("service_name")
        host_id = params.get("host_id", "localhost")
        force = params.get("force", False)

        self.logger.info(f"Restarting service {service_name} on {host_id}")

        # In production, this would use SSH/Ansible to restart service
        # For now, simulate
        if force:
            command = f"sudo systemctl restart {service_name} --force"
        else:
            command = f"sudo systemctl restart {service_name}"

        # Execute command (simplified)
        result = {
            "success": True,
            "detail": f"Service {service_name} restarted successfully",
            "command": command,
            "host": host_id
        }

        return result

    except Exception as e:
        return {"success": False, "error": str(e)}
```

#### Step 4: Add to Execution Router

**File**: `backend/app/advanced_response_engine.py` (around line 993)

```python
# Execute the action based on type
if action_type in ["block_ip", "block_ip_advanced"]:
    result = await self._execute_block_ip_action(action_params)
elif action_type == "restart_service":  # ← ADD THIS
    result = await self._execute_restart_service_action(action_params)
elif action_type == "create_incident":
    result = await self._execute_create_incident_action(action_params)
# ... rest of router
```

#### Step 5: Test It!

```bash
curl -X POST http://localhost:8000/api/workflows/nlp/create \
  -H "Content-Type: application/json" \
  -H "x-api-key: YOUR_API_KEY" \
  -d '{
    "text": "Restart the apache service",
    "incident_id": 8,
    "auto_execute": true
  }'
```

**Result**:
```json
{
  "success": true,
  "workflow_id": "nlp_xyz789",
  "message": "Workflow created and executing with 1 actions",
  "actions_created": 1
}
```

---

## 🎯 **What You Need for NLP Trigger Creation (New Feature)**

### Current: NLP Creates Workflows
```python
# nlp_workflow_routes.py:127
@router.post("/api/workflows/nlp/create")
async def create_workflow_from_natural_language(request):
    intent = await parse_workflow_from_natural_language(request.text)

    # Creates ONE-OFF workflow
    workflow = ResponseWorkflow(
        workflow_id=f"nlp_{uuid.uuid4().hex[:12]}",
        steps=intent.actions  # Actions to execute
    )
```

### New: NLP Creates Triggers
```python
# NEW FILE: nlp_trigger_routes.py
@router.post("/api/triggers/nlp/create")
async def create_trigger_from_natural_language(request):
    # Different parsing - extract RULES not just actions
    trigger_intent = await parse_trigger_from_natural_language(request.text)

    # Creates PERMANENT trigger
    trigger = WorkflowTrigger(
        name=trigger_intent.name,
        enabled=True,
        conditions=trigger_intent.conditions,  # ← Rules to match
        workflow_steps=trigger_intent.actions   # ← Actions to execute when matched
    )
```

### Key Differences

| Workflow NLP | Trigger NLP |
|-------------|------------|
| Parses: Actions | Parses: Conditions + Actions |
| Creates: ResponseWorkflow | Creates: WorkflowTrigger |
| Executes: Once | Executes: Every time conditions match |
| Example: "Block IP 1.2.3.4" | Example: "Block IPs with >10 failed logins" |

### New NLP Patterns Needed for Triggers

```python
# NEW FILE: backend/app/nlp_trigger_parser.py

class TriggerIntent:
    """Represents a parsed trigger rule"""
    def __init__(self):
        self.trigger_name: str = ""
        self.conditions: Dict[str, Any] = {}  # ← NEW: Trigger conditions
        self.actions: List[Dict[str, Any]] = []
        self.priority: str = "medium"
        self.auto_execute: bool = False

class NLPTriggerParser:
    """Parse natural language into trigger rules"""

    def __init__(self):
        # Condition patterns
        self.condition_patterns = {
            # Threshold conditions
            r'(more than|greater than|>|>=)\s*(\d+)\s+(failed logins?|login attempts?)':
                {'type': 'threshold', 'field': 'failed_logins'},

            r'(more than|>)\s*(\d+)\s+(events?|requests?)\s+(?:within|in)\s+(\d+)\s+(seconds?|minutes?|hours?)':
                {'type': 'threshold_window'},

            # Risk score conditions
            r'risk score\s+(above|over|>|>=)\s+([\d.]+)':
                {'type': 'risk_score_min'},

            # Event type conditions
            r'(?:when|if)\s+(\w+)\s+(?:is detected|occurs|happens)':
                {'type': 'event_type'},

            # Pattern matching
            r'(?:contains|matches|includes)\s+["\'](.+?)["\']':
                {'type': 'pattern_match'}
        }

    async def parse(self, text: str) -> TriggerIntent:
        """Parse natural language into trigger intent"""
        intent = TriggerIntent()

        # Extract trigger name
        intent.trigger_name = self._generate_trigger_name(text)

        # Extract conditions
        intent.conditions = self._extract_conditions(text)

        # Extract actions (reuse existing workflow parser)
        intent.actions = self._extract_actions(text)

        # Determine if auto-execute is safe
        intent.auto_execute = self._is_auto_execute_safe(intent)

        return intent

    def _extract_conditions(self, text: str) -> Dict[str, Any]:
        """Extract trigger conditions from text"""
        conditions = {}

        # Example: "more than 10 failed logins within 5 minutes"
        threshold_match = re.search(
            r'(more than|>|>=)\s*(\d+)\s+(failed logins?)\s+(?:within|in)\s+(\d+)\s+(minutes?)',
            text.lower()
        )

        if threshold_match:
            conditions['event_type'] = 'cowrie.login.failed'
            conditions['threshold'] = int(threshold_match.group(2))
            conditions['window_seconds'] = int(threshold_match.group(4)) * 60

        # Example: "risk score above 0.8"
        risk_match = re.search(r'risk score\s+(above|over|>|>=)\s+([\d.]+)', text.lower())
        if risk_match:
            conditions['risk_score_min'] = float(risk_match.group(2))

        # Example: "when SQL injection is detected"
        event_match = re.search(r'(?:when|if)\s+([\w\s]+?)\s+(?:is detected|occurs)', text.lower())
        if event_match:
            event_type = event_match.group(1).strip()
            if 'sql injection' in event_type:
                conditions['pattern_match'] = 'sql injection'

        return conditions
```

### Example Usage

```python
# User input
text = "Create a trigger to automatically block IPs with more than 10 failed logins within 5 minutes"

# Parse trigger
parser = NLPTriggerParser()
trigger_intent = await parser.parse(text)

# Result
{
  "trigger_name": "Auto-block brute force attacks",
  "conditions": {
    "event_type": "cowrie.login.failed",
    "threshold": 10,
    "window_seconds": 300
  },
  "actions": [
    {
      "action_type": "block_ip",
      "parameters": {
        "ip_address": "event.source_ip",  # ← Variable, filled at runtime
        "duration": 3600,
        "block_level": "standard"
      }
    }
  ],
  "auto_execute": true
}

# Create permanent trigger
trigger = WorkflowTrigger(
    name=trigger_intent.trigger_name,
    enabled=True,
    conditions=trigger_intent.conditions,
    workflow_steps=trigger_intent.actions,
    auto_execute=trigger_intent.auto_execute
)

# Save to database
db.add(trigger)
await db.commit()

# Now this trigger runs FOREVER!
```

---

## 📋 **Checklist: Ensuring Analysts Have What They Need**

### 1. Action Coverage Audit

**Current**: 68 actions across 10 categories

**Questions to Ask**:
- ✅ Can we block/unblock IPs? → YES
- ✅ Can we isolate hosts? → YES
- ✅ Can we reset passwords? → YES
- ✅ Can we collect forensic data? → YES
- ❌ Can we restart services? → NO (need to add)
- ❌ Can we clear cache? → NO (need to add)
- ❌ Can we rotate certificates? → NO (need to add)

**Action**: Create a survey or interview analysts to identify missing actions.

### 2. Tool Integration

**What We Support**:
- ✅ Firewall (iptables)
- ✅ Email (Slack, email)
- ✅ Cloud (AWS WAF, Security Groups)
- ❌ SIEM integration (Splunk, ELK)
- ❌ Ticketing (Jira, ServiceNow)
- ❌ Endpoint management (CrowdStrike, Carbon Black)

**Action**: Prioritize integrations based on analyst needs.

### 3. Parameter Flexibility

**Current Limitations**:
- Simple parameters only (IP, duration, host)
- No complex configs (port ranges, CIDR blocks, regex)

**Example Gap**:
```
Analyst: "Block traffic from 192.168.0.0/16 to ports 22, 80, 443"

Current: ❌ Can only extract single IP
Needed: ✅ Extract subnet, port list
```

**Action**: Enhance parameter extraction for complex rules.

### 4. Safety Mechanisms

**What We Have**:
- ✅ Approval required for critical actions
- ✅ Rollback support for some actions
- ✅ Action validation before execution

**What We Need**:
- ❌ Dry-run mode (preview what would happen)
- ❌ Automatic rollback on failure
- ❌ Action dependencies (do A before B)

**Action**: Add dry-run mode and better rollback.

### 5. Documentation & Discovery

**Current Problem**: Analysts don't know what actions exist.

**Solutions**:
1. **Action Browser in UI**
   ```
   [Search Actions: "restart"]

   Results:
   ✅ restart_service - Restart a system service
   ✅ restart_host - Reboot a host
   ❌ restart_database - Not available (request feature)
   ```

2. **Auto-suggestions**
   ```
   User types: "Can I clear cache?"

   System: "No 'clear_cache' action found. Did you mean:
            • restart_service (clears cache on restart)
            • flush_dns_cache
            • clear_browser_data"
   ```

3. **Action Templates**
   ```
   "Block IP Address"
   Parameters:
   • ip_address: [Required] The IP to block
   • duration: [Optional, default=3600] Block duration in seconds
   • block_level: [Optional, default=standard] standard|strict|temporary

   Example: "Block IP 192.168.1.100 for 24 hours"
   ```

---

## 🎯 **Next Steps**

### Priority 1: Add NLP Trigger Creation (What You Want!)

**Deliverables**:
1. ✅ New endpoint: `/api/triggers/nlp/create`
2. ✅ NLP parser for trigger conditions (not just actions)
3. ✅ UI for creating triggers with natural language
4. ✅ Test with examples like "Block IPs with >10 failed logins"

**Time Estimate**: 4-6 hours

### Priority 2: Add Missing Common Actions

**Based on typical analyst needs**:
1. `restart_service` - Restart Apache, Nginx, etc.
2. `clear_cache` - Clear application/DNS cache
3. `rotate_credentials` - Rotate API keys, passwords
4. `create_jira_ticket` - Auto-create tickets
5. `snapshot_vm` - Create VM snapshot before action

**Time Estimate**: 2-3 hours (for all 5)

### Priority 3: Action Discovery UI

**Deliverables**:
1. ✅ Browse all 68+ actions
2. ✅ Search and filter
3. ✅ See examples and parameters
4. ✅ Test actions in sandbox mode

**Time Estimate**: 3-4 hours

### Priority 4: Conversational Chat Interface

**Deliverables**:
1. ✅ Multi-turn conversation
2. ✅ Clarifying questions
3. ✅ Session memory
4. ✅ Action suggestions

**Time Estimate**: 6-8 hours

---

**Should I start building #1 (NLP Trigger Creation)? That's what you asked for!**
