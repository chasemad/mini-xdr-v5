# Workflows vs Triggers - System Explained

## ✅ Auto Triggers Tab Now Working!

The Auto Triggers tab is now fully functional and will display all your automatic workflow triggers.

---

## Understanding the Two Systems

### 1. **Response Workflows** (What NLP Creates)

**Created by:** Natural Language chat, Templates, Designer
**Purpose:** One-time workflow execution for a specific incident
**Stored in:** `ResponseWorkflow` table
**Shown in:** "Executor" tab on Workflows page

**Example:**
```
"Block IP 192.168.1.100 for incident #5"
```

**Characteristics:**
- ✅ Created instantly from natural language
- ✅ Executes once for a specific situation
- ✅ Tied to a specific incident (optional)
- ✅ Shows progress and completion status
- ⚠️ Does NOT automatically trigger on future events

---

### 2. **Workflow Triggers** (Automatic/Recurring)

**Created by:** Automations page (/automations) or trigger API
**Purpose:** Automatic workflow that fires when conditions are met
**Stored in:** `WorkflowTrigger` table
**Shown in:** "Auto Triggers" tab AND /automations page

**Example:**
```
Trigger: "When brute_force detected → Block IP + Alert Team"
Conditions: event_type = "brute_force"
Auto-Execute: Yes/No
```

**Characteristics:**
- ✅ Automatically fires when conditions match
- ✅ Recurring - fires every time conditions are met
- ✅ Can be enabled/disabled
- ✅ Can require approval or auto-execute
- ✅ Has cooldown period to prevent spam

---

## How They Work Together

```
┌─────────────────────────────────────────────┐
│         NLP Workflow Creation               │
│                                             │
│  User: "Block IP 192.168.1.100"            │
│                                             │
│  ✓ Creates ResponseWorkflow (ID: 3)        │
│  ✓ Shows in Executor tab                   │
│  ✓ Executes once                            │
│  ✓ Status: pending → running → completed   │
└─────────────────────────────────────────────┘
                      │
                      │ (Optional)
                      │ Convert to Trigger
                      ▼
┌─────────────────────────────────────────────┐
│         Automatic Trigger Creation          │
│                                             │
│  Name: "Block Brute Force IPs"             │
│  Conditions: event_type = "brute_force"    │
│  Actions: block_ip, alert_team             │
│                                             │
│  ✓ Creates WorkflowTrigger (ID: 1)         │
│  ✓ Shows in Auto Triggers tab              │
│  ✓ Fires automatically when triggered      │
│  ✓ Can be edited/paused/deleted            │
└─────────────────────────────────────────────┘
```

---

## What You're Seeing Now

### Workflows Page Tabs:

1. **Natural Language** - Create one-time workflows via NLP
   - Type: "Block IP 192.168.1.100"
   - Click "Parse" → "Create Workflow"
   - Result: Creates ResponseWorkflow

2. **Designer** - Visual workflow builder (one-time workflows)

3. **Templates** - Pre-built playbook templates (one-time workflows)

4. **Executor** - Monitor running workflows
   - Shows: ResponseWorkflows (one-time executions)
   - Your malware workflow IS here!
   - Current count: **4 Total, 4 Active**

5. **Auto Triggers** ✨ **NOW IMPLEMENTED!**
   - Shows: WorkflowTriggers (automatic/recurring)
   - Currently shows: **1 trigger** (Test_Auto_Execute_Toggle)
   - Can pause/resume/edit triggers
   - Click "Manage Triggers" → goes to /automations page

6. **Analytics** - Workflow performance metrics

---

## Current State of Your System

### Workflows (One-Time Executions):
✅ **4 workflows in database:**
1. NLP Workflow: "Create a malware response workflow..." (incident #5)
2. NLP Workflow: "Block IP 192.168.1.100"
3. test_playbook (incident #5)
4. test_playbook (incident #1)

**Location:** Executor tab on http://localhost:3000/workflows

---

### Triggers (Automatic/Recurring):
✅ **1 trigger in database:**
1. "Test_Auto_Execute_Toggle"
   - Category: test
   - Priority: high
   - Enabled: ✅ Yes
   - Auto-Execute: ❌ No (requires approval)
   - Conditions: event_type = "brute_force"
   - Executions: 2 runs, 100% success rate

**Location:** Auto Triggers tab on http://localhost:3000/workflows

---

## How to Use Each System

### Creating One-Time Workflows (ResponseWorkflow):

**Option 1: NLP Chat**
1. Go to http://localhost:3000/workflows
2. Click "Natural Language" tab
3. Type: "Block IP 192.168.1.100 and isolate host"
4. Click ⚙️ **Parse**
5. Review the generated workflow
6. Click **Create Workflow**
7. ✅ Workflow appears in "Executor" tab

**Option 2: Templates**
1. Click "Templates" tab
2. Choose a template (e.g., "Emergency Containment")
3. Click "Use Template"
4. Select incident (optional)
5. ✅ Workflow created and shown in Executor

**Option 3: Designer**
1. Click "Designer" tab
2. Drag and drop actions
3. Save workflow
4. ✅ Workflow created

---

### Creating Automatic Triggers (WorkflowTrigger):

**Option 1: Automations Page** (Recommended)
1. Go to http://localhost:3000/automations
2. Click **+ Create Trigger** or use NLP Suggestions tab
3. Set conditions: event_type, threshold, patterns
4. Add workflow steps
5. Set auto_execute: true/false
6. Save trigger
7. ✅ Trigger shows in Auto Triggers tab

**Option 2: API**
```bash
curl -X POST "http://localhost:8000/api/triggers/" \
  -H "x-api-key: YOUR_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Auto Block Brute Force",
    "conditions": {"event_type": "brute_force"},
    "auto_execute": false,
    "workflow_steps": [...]
  }'
```

---

## Why Your Malware Workflow Isn't in Auto Triggers

Your malware workflow created via NLP is a **ResponseWorkflow** (one-time execution), NOT a **WorkflowTrigger** (automatic).

**It's in the right place!** → Executor tab

**To make it automatic:**
1. Go to /automations page
2. Look for "NLP Suggestions" tab
3. Your NLP workflows might be there as suggestions
4. Click "Approve" to convert to a trigger
5. OR manually create a new trigger with the same actions

---

## Quick Reference

| Feature | ResponseWorkflow | WorkflowTrigger |
|---------|------------------|-----------------|
| **Purpose** | One-time execution | Automatic/recurring |
| **Created via** | NLP, Templates, Designer | Automations page, API |
| **Shown in** | Executor tab | Auto Triggers tab + /automations |
| **Triggers when** | Manually created | Conditions match |
| **Can be edited** | No (just executed) | Yes (/automations page) |
| **Has conditions** | No | Yes (event_type, patterns, etc.) |
| **Auto-execute** | Always executes | Configurable |
| **Recurring** | No (one-time) | Yes (fires repeatedly) |

---

## What's New in Auto Triggers Tab

✅ **Displays all WorkflowTriggers** from database
✅ **Shows trigger status** (Active/Paused, Auto-Execute/Manual)
✅ **Pause/Resume buttons** - Toggle triggers on/off
✅ **Edit button** - Opens /automations page for editing
✅ **Refresh button** - Manually reload triggers
✅ **"Manage Triggers" button** - Go to full automations page

---

## Testing Your Auto Triggers Tab

1. **Go to:** http://localhost:3000/workflows
2. **Click:** "Auto Triggers" tab
3. **You should see:** 1 trigger listed:
   - Name: "Test_Auto_Execute_Toggle"
   - Status: Active, Manual Approval
   - Priority: High
   - 2 executions, 100% success
4. **Try clicking:** Pause button (should disable trigger)
5. **Try clicking:** Edit button (opens /automations page)

---

## Next Steps

### Option 1: Keep workflows separate
- Use NLP for **one-time responses** to specific incidents
- Use /automations page for **automatic triggers**

### Option 2: Add "Convert to Trigger" button
- I can add a button on workflows in Executor tab
- Clicking it converts ResponseWorkflow → WorkflowTrigger
- Would allow easy conversion

### Option 3: NLP can create triggers directly
- Update NLP parser to detect "automatically" or "every time"
- Example: "Automatically block IPs whenever brute force is detected"
- Parser creates WorkflowTrigger instead of ResponseWorkflow

---

## Summary

✅ **Your malware workflow IS working** - it's in the Executor tab where one-time workflows belong
✅ **Auto Triggers tab NOW shows triggers** - currently showing your test trigger
✅ **Two separate systems** - ResponseWorkflows (one-time) vs WorkflowTriggers (automatic)
✅ **Both systems fully functional** - just serving different purposes

**The confusion was:**
- You expected NLP workflows to appear in Auto Triggers
- But NLP creates ResponseWorkflows (one-time), not Triggers (automatic)
- Now Auto Triggers tab shows actual WorkflowTriggers correctly

**Everything is working as designed!** 🎉

---

**Created:** October 2, 2025
**Status:** ✅ Auto Triggers tab fully implemented and functional
