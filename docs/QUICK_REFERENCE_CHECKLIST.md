# POLLING ISSUE - QUICK REFERENCE CHECKLIST

## Problem Statement
NaturalLanguageInput component loses `parsedWorkflow` local state when polling triggers parent re-render every 45 seconds.

---

## Files & Line Numbers

| File | Key Lines | What's There |
|------|-----------|--------------|
| `DataService.ts` | 30-141 | `hasDataChanged()` - intelligent comparison |
| | 147-180 | `refreshIncidents()` - polling + callback |
| | 186-218 | `refreshWorkflows()` - polling + callback |
| | 273-298 | `startPeriodicRefresh()` - 45s interval |
| `AppContext.tsx` | 100-185 | Reducer (creates new state objects) |
| | 102-107 | SET_INCIDENTS action |
| | 127-132 | SET_WORKFLOWS action |
| | 198-239 | Provider component |
| `WorkflowsPage.tsx` | 215-218 | `handleWorkflowCreated` (useCallback) |
| | 225-236 | `selectedIncidentObject` (useMemo) |
| | 646-655 | Renders NaturalLanguageInput |
| `NaturalLanguageInput.tsx` | 69-75 | Wrapped in memo() |
| | 76 | Only uses `dispatch`, not `state` |
| | 77-80 | Local state: `parsedWorkflow` |
| | 495-499 | Memo comparison function |

---

## Fixes Applied (In Order)

- [x] **Fix #1:** Intelligent change detection (DataService)
- [x] **Fix #2:** React.memo wrapper (NaturalLanguageInput)
- [x] **Fix #3:** Remove state dependencies (NaturalLanguageInput)
- [x] **Fix #4:** Memoize callbacks (WorkflowsPage)
- [x] **Fix #5:** Memoize incident object (WorkflowsPage)
- [x] **Fix #6:** Increase polling interval to 45s
- [ ] **Status:** Still having issues ❌

---

## Top 3 Hypotheses (Prioritized)

### 🔴 #1: AppContext Reducer Creates New State Unnecessarily
**File:** `AppContext.tsx:102-107`
```typescript
lastUpdated: { ...state.lastUpdated, incidents: Date.now() }  // ← Always changes!
```
**Fix:** Check if data identical before creating new state object

### 🔴 #2: Callback Fires Even When hasDataChanged() Returns False
**File:** `DataService.ts:173`
```typescript
if (this.hasDataChanged(...)) {
  this.callbacks.onIncidentsUpdate?.(incidents)  // ← Being called outside if?
}
```
**Fix:** Add logging to verify if statement logic

### 🟡 #3: Memo Comparison Not Preventing Re-renders
**File:** `NaturalLanguageInput.tsx:495-499`
```typescript
return prevProps.selectedIncidentId === nextProps.selectedIncidentId  // ← Working?
```
**Fix:** Add logging to see what props are actually changing

---

## Quick Debugging Commands

### Add This Logging First
```typescript
// DataService.ts line 173 (refreshIncidents)
const hasChanged = this.hasDataChanged(this.lastIncidents, incidents, 'incidents')
console.log('[DataService] hasChanged:', hasChanged, 'Will call callback:', hasChanged)
if (hasChanged) {
  console.log('[DataService] ✅ CALLING onIncidentsUpdate')
  this.callbacks.onIncidentsUpdate?.(incidents)
} else {
  console.log('[DataService] ⏭️  SKIPPING onIncidentsUpdate')
}

// AppContext.tsx line 102 (SET_INCIDENTS reducer)
case 'SET_INCIDENTS':
  console.log('[Reducer] SET_INCIDENTS called, same data?', state.incidents === action.payload)
  if (state.incidents === action.payload) {
    console.log('[Reducer] ⏭️  RETURNING SAME STATE')
    return state
  }
  console.log('[Reducer] ✅ CREATING NEW STATE')
  return { ...state, incidents: action.payload, lastUpdated: {...} }

// NaturalLanguageInput.tsx line 495 (memo comparison)
}, (prevProps, nextProps) => {
  const skip = prevProps.selectedIncidentId === nextProps.selectedIncidentId
  console.log('[Memo]', skip ? '⏭️  SKIPPING' : '✅ RENDERING', {
    idChanged: prevProps.selectedIncidentId !== nextProps.selectedIncidentId,
    callbackChanged: prevProps.onWorkflowCreated !== nextProps.onWorkflowCreated,
    incidentChanged: prevProps.selectedIncident !== nextProps.selectedIncident
  })
  return skip
})
```

### Watch Logs in Real-Time
```bash
# Terminal 1: Watch frontend logs
tail -f /tmp/frontend.log | grep "GET /workflows"

# Terminal 2: Browser console at http://localhost:3000/workflows
# Filter by: "DataService" or "Reducer" or "Memo"
```

### Test Command
```bash
# Should see pattern every 45 seconds:
# GET /workflows 200 in XXms
# [DataService] hasChanged: false
# [DataService] ⏭️  SKIPPING onIncidentsUpdate
# [Memo] ⏭️  SKIPPING render
```

---

## Expected Console Output (When Fixed)

### Good Output (No Changes)
```
[DataService] hasChanged: false
[DataService] ⏭️  SKIPPING onIncidentsUpdate
[DataService] hasChanged: false
[DataService] ⏭️  SKIPPING onWorkflowsUpdate
(No Reducer logs)
(No Memo logs)
```

### Good Output (With Changes)
```
[DataService] hasChanged: true
[DataService] ✅ CALLING onIncidentsUpdate
[Reducer] ✅ CREATING NEW STATE
[Memo] ⏭️  SKIPPING render (if incident ID didn't change)
```

### Bad Output (Current State)
```
[DataService] hasChanged: false
[DataService] ⏭️  SKIPPING onIncidentsUpdate
[Reducer] ✅ CREATING NEW STATE  ← WHY IS THIS HAPPENING?
[Memo] ✅ RENDERING  ← Component re-rendering anyway!
```

---

## Test Protocol (30 Second Version)

1. Open http://localhost:3000/workflows + Console (F12)
2. Select incident #1
3. NLP chat: "Block IP 1.2.3.4 and send alert"
4. Click Parse → see parsed workflow
5. Start timer, wait 50 seconds
6. **Check:** Parsed workflow still visible? ✅ / ❌
7. **Check:** Console shows "SKIPPING" logs? ✅ / ❌
8. **Check:** No "Component rendered" logs? ✅ / ❌

---

## Immediate Action Items

1. [ ] Add logging to DataService.ts line 173 (verify hasChanged logic)
2. [ ] Add logging to AppContext.tsx line 102 (verify reducer behavior)
3. [ ] Add logging to NaturalLanguageInput.tsx line 495 (verify memo)
4. [ ] Run test protocol and collect logs for 1 minute
5. [ ] Analyze which component is causing re-renders
6. [ ] Apply appropriate fix based on findings

---

## Most Likely Fix (Based on Analysis)

**Problem:** AppContext reducer always creates new state because `lastUpdated` timestamp always changes

**Solution:**
```typescript
// AppContext.tsx line 102-107
case 'SET_INCIDENTS':
  // Don't create new state if data is identical
  if (state.incidents === action.payload) {
    return state  // ← Prevents re-renders!
  }
  return {
    ...state,
    incidents: action.payload,
    lastUpdated: { ...state.lastUpdated, incidents: Date.now() }
  }
```

**Alternative:** Remove `lastUpdated` field entirely if not being used

---

## Emergency Workarounds (If Nothing Else Works)

### Option 1: Move State to Context
```typescript
// Store parsedWorkflow in AppContext instead of local state
// Won't be lost during re-renders but breaks encapsulation
```

### Option 2: Disable Polling During Interaction
```typescript
// Stop polling when user is typing/reviewing workflow
// Resume after workflow accepted/rejected
```

### Option 3: Manual Refresh Only
```typescript
// Remove automatic polling
// Add "Refresh" button for manual updates
```

### Option 4: Use localStorage
```typescript
// Persist parsedWorkflow to localStorage
// Restore on component mount
// Ugly but functional
```

---

## Success Criteria

- [ ] User can review parsed workflow for 2+ minutes without it disappearing
- [ ] No UI flicker during polling
- [ ] Console logs show "SKIPPING" when no changes detected
- [ ] Polling still detects real changes (new incidents, status updates)
- [ ] All existing functionality still works

---

## Key Contacts / Context

- **Project:** Mini-XDR Security Platform
- **Frontend:** Next.js 15.5.0 on localhost:3000
- **Backend:** FastAPI on localhost:8000
- **State:** React Context API (WebSocket disabled)
- **Polling:** 45 second interval
- **Issue Duration:** Multiple attempted fixes, still persisting
- **User Impact:** Cannot review AI-parsed workflows before accepting

---

## Related Files

- `/tmp/POLLING_ISSUE_HANDOFF.md` - Complete technical documentation
- `/tmp/NEW_CHAT_PROMPT.txt` - Prompt for new chat session
- `/tmp/polling_fix_final.md` - Previous fix attempt documentation
- `/tmp/polling_fix_summary.md` - Original analysis

---

Last Updated: Current debugging session
Status: In progress, needs systematic logging approach
