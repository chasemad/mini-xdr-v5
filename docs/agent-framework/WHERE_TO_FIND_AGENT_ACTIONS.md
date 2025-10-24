# 👀 Where to Find Agent Actions in UI

**Current Location:** Incident #14  
**Status:** ✅ 3 Agent actions created and ready to display

---

## 📍 **Exact Location in Your UI**

### **Step 1: Find the "Unified Response Actions" Section**

On your Incident #14 page, scroll down to find this section:

```
┌─────────────────────────────────────────────────────────────┐
│ 🛡️ Unified Response Actions                                 │
│ (41 total • 0 manual • 41 workflow • 0 agent)    [Refresh]  │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  📋 Create Incident            WORKFLOW • HONEYPOT...   15h │
│  🤖 Invoke AI Agent           WORKFLOW • HONEYPOT...   15h │
│  📧 Send Notification         WORKFLOW • HONEYPOT...   15h │
│  ... (more workflow actions)                                │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### **Step 2: Refresh to See Agent Actions**

Click the **[Refresh]** button in the top-right of this section, OR wait 5 seconds for auto-refresh.

### **Step 3: After Refresh, You'll See**

```
┌─────────────────────────────────────────────────────────────┐
│ 🛡️ Unified Response Actions                                 │
│ (44 total • 0 manual • 41 workflow • 3 agent)    [Refresh]  │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  👤 Disable User Account      IAM AGENT              Just now│
│     Status: ✅ Success                    [🔄 Rollback]      │
│     username: compromised.user@domain.local                  │
│                                                              │
│  🖥️ Quarantine File           EDR AGENT              Just now│
│     Status: ✅ Success                    [🔄 Rollback]      │
│     hostname: HONEYPOT-01                                    │
│     file_path: C:\malicious\payload.exe                      │
│                                                              │
│  🔒 Scan File                 DLP AGENT              Just now│
│     Status: ✅ Success                    [🔄 Rollback]      │
│     file_path: /var/log/honeypot/exfiltrated_data.log       │
│                                                              │
│  📋 Create Incident            WORKFLOW • HONEYPOT...   15h │
│  🤖 Invoke AI Agent           WORKFLOW • HONEYPOT...   15h │
│  ... (more workflow actions)                                │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 🎨 **Visual Indicators**

### **Color Coding (You'll See These):**

1. **IAM AGENT** badge = **Blue** background 🔵
2. **EDR AGENT** badge = **Purple** background 🟣
3. **DLP AGENT** badge = **Green** background 🟢

### **What Each Action Shows:**

```
┌─────────────────────────────────────────────────────┐
│ 👤 Disable User Account        IAM AGENT     Just now│  ← Icon & Name
│     Status: ✅ Success              [🔄 Rollback]    │  ← Status Badge
│     username: compromised.user@domain.local          │  ← Parameters
│     reason: Honeypot compromise detected...          │  ← Details
└─────────────────────────────────────────────────────┘
   │           │                  │              │
   │           │                  │              └── Rollback Button (clickable)
   │           │                  └────────────── Agent Type Badge (colored)
   │           └────────────────────────────────── Status (success/failed)
   └───────────────────────────────────────────────── Action Icon
```

---

## 🖱️ **Interactions Available**

### **Click on Any Action Row:**
Opens a **detailed modal** showing:
- Full execution timeline
- All input parameters (JSON formatted)
- Complete results
- Related events (within 5 min window)
- Rollback button (if applicable)

### **Click Rollback Button:**
1. Opens confirmation dialog
2. Asks: "Are you sure you want to rollback?"
3. If confirmed, executes rollback immediately
4. Updates status to "Rolled Back 🔄"
5. Shows rollback timestamp

### **Hover Effects:**
- Row highlights on hover
- Border color changes
- Cursor changes to pointer

---

## 📊 **What the Counter Shows**

In the section header, you'll see:

```
(44 total • 0 manual • 41 workflow • 3 agent)
```

**Breakdown:**
- **44 total** = All response actions combined
- **0 manual** = Quick actions executed by user
- **41 workflow** = Automated workflow actions (already visible)
- **3 agent** = Our new IAM, EDR, DLP actions ← **NEW!**

---

## ⏰ **Auto-Refresh Behavior**

The agent actions section auto-refreshes **every 5 seconds** without page reload.

**You'll see:**
- Loading indicator (subtle)
- Smooth updates (no flicker)
- New actions appear automatically
- Counts update in real-time

**If you execute more agent actions in terminal, they'll appear within 5 seconds!**

---

## 🧪 **Test the Auto-Refresh** (Optional)

Want to see the magic? Keep your browser open and run this:

```bash
curl -X POST http://localhost:8000/api/agents/iam/execute \
  -H "Content-Type: application/json" \
  -d '{
    "action_name": "reset_password",
    "params": {
      "username": "watch.me.appear@domain.local",
      "reason": "Testing real-time auto-refresh"
    },
    "incident_id": 14
  }'
```

**Watch your browser** - within 5 seconds, you'll see a 4th agent action appear:

```
🔑 Reset Password          IAM AGENT              Just now
   Status: ✅ Success                [🔄 Rollback]
```

---

## 🐛 **If You Don't See Agent Actions**

### **Checklist:**

1. **Are you on Incident #14?**
   - Check top of page: "Incident #14"
   - If not, navigate back to Incident #14

2. **Did you refresh?**
   - Click [Refresh] button, OR
   - Wait 5 seconds for auto-refresh, OR
   - Hard refresh: Cmd+Shift+R (Mac) / Ctrl+Shift+R (Win)

3. **Check browser console (F12 → Console)**
   - Should see: No errors
   - Should see: Network requests to `/api/agents/actions/14`

4. **Verify actions exist in database:**
   ```bash
   curl http://localhost:8000/api/agents/actions/14
   ```
   Should return JSON with 3 actions

### **Still Not Seeing Them?**

Check the section name - it might be called:
- "Unified Response Actions" ← **Most likely**
- "Response Actions & Status"
- "Action History"

Look for a section with this icon: 🛡️ or ⚡ or 📋

---

## 📸 **What Success Looks Like**

### **Before Refresh:**
- Counter shows: `(41 total • 0 manual • 41 workflow • 0 agent)`
- Only workflow actions visible

### **After Refresh:**
- Counter shows: `(44 total • 0 manual • 41 workflow • 3 agent)`
- Three new colored badges: IAM (Blue), EDR (Purple), DLP (Green)
- Rollback buttons visible on each
- "Just now" timestamps

---

## 🎯 **Quick Reference**

| What | Where | How |
|------|-------|-----|
| **Find Actions** | Scroll down on Incident #14 | Look for "Unified Response Actions" |
| **See New Actions** | Click [Refresh] button | OR wait 5 seconds |
| **Open Details** | Click any action row | Modal opens |
| **Rollback** | Click [Rollback] button | Confirmation dialog |
| **Color Codes** | IAM=Blue, EDR=Purple, DLP=Green | Visual badge |

---

## ✅ **Success Criteria**

You'll know it's working when you see:

- [x] Counter updates from "0 agent" to "3 agent"
- [x] Three new action rows at the top (newest first)
- [x] Color-coded badges (Blue/Purple/Green)
- [x] Rollback buttons on each agent action
- [x] Parameters displayed inline
- [x] "Just now" timestamps
- [x] Clickable rows (cursor changes)

---

**Ready? Click Refresh or Wait 5 Seconds!** ⏰

The agent actions are created and waiting in the database. Your UI is fully configured. You just need to refresh to see them! 🎉

