# POLLING ISSUE - COMPLETE HANDOFF DOCUMENT

## EXECUTIVE SUMMARY

**Problem:** The workflows page in the Mini-XDR frontend has a disruptive "refresh" behavior that happens every 30-45 seconds. This causes:
1. UI flicker/visual disruption
2. Loss of user state in the NLP workflow chat (parsed workflows disappear before user can accept them)
3. Poor user experience during workflow creation

**Status:** Multiple fixes attempted, but user reports issue still occurs.

**Critical User Quote:** "its still doing it and if i dont accept the workflow after the chat prompt it gets removed when it does the poll/refresh thing"

---

## SYSTEM ARCHITECTURE

### Technology Stack
- **Frontend:** Next.js 15.5.0 (React)
- **Backend:** FastAPI (Python)
- **State Management:** React Context API
- **Data Fetching:** Polling-based (WebSocket disabled due to connection leak)

### Key Components Flow
```
DataService (polling every 45s)
    ↓ (fetches incidents & workflows)
AppContext (global state via React Context)
    ↓ (provides state to all components)
WorkflowsPage (main page)
    ↓ (renders)
NaturalLanguageInput (NLP chat component)
    ↓ (local state: parsedWorkflow)
User interacts, parses workflow
    ↓ (45s passes)
DataService polls again
    ↓ (even if no changes, React re-renders)
Component re-renders → LOCAL STATE LOST ❌
```

---

## FILES INVOLVED

### 1. `/Users/chasemad/Desktop/mini-xdr/frontend/app/services/DataService.ts`
**Purpose:** Centralized service for polling backend data

**Key Methods:**
- `hasDataChanged(oldData, newData, dataType)` - Lines 30-141
  - Compares old vs new data to determine if UI update needed
  - Returns true = trigger state update, false = skip update

- `refreshIncidents()` - Lines 147-180
  - Polls `/api/incidents`
  - Calls `hasDataChanged()` to check if update needed
  - Only calls `onIncidentsUpdate()` if changes detected

- `refreshWorkflows()` - Lines 186-218
  - Polls `/api/response/workflows`
  - Calls `hasDataChanged()` to check if update needed
  - Only calls `onWorkflowsUpdate()` if changes detected

- `startPeriodicRefresh(intervalMs = 45000)` - Lines 273-298
  - Sets up interval to poll every 45 seconds
  - Runs in background continuously

**Current State:**
- Modified to include intelligent change detection
- Logs to console when changes detected
- Should only trigger state updates on significant changes

**Lines Modified:**
- 30-141: Complete rewrite of `hasDataChanged()` with type-specific logic
- 173: Added dataType parameter to method call
- 208: Added dataType parameter to method call
- 273-284: Updated interval to 45s with documentation

### 2. `/Users/chasemad/Desktop/mini-xdr/frontend/app/contexts/AppContext.tsx`
**Purpose:** Global state management using React Context

**Key State:**
```typescript
interface AppState {
  incidents: Incident[]
  workflows: Workflow[]
  selectedIncident: number | null
  // ... other fields
}
```

**Key Reducer Actions:**
- `SET_INCIDENTS` - Line 102-107: Creates NEW state object with updated incidents
- `SET_WORKFLOWS` - Line 127-132: Creates NEW state object with updated workflows

**CRITICAL ISSUE:** Every reducer action creates a NEW state object reference
- Even if data hasn't changed, new object = React thinks state changed
- This triggers re-renders in ALL components using `useAppContext()`

**Lines of Interest:**
- 100-185: Reducer function (creates new state objects)
- 198-239: Provider component
- 201-232: WebSocket code commented out (disabled due to connection leak)

### 3. `/Users/chasemad/Desktop/mini-xdr/frontend/app/workflows/page.tsx`
**Purpose:** Main workflows page container

**Key Functionality:**
- Sets up DataService on mount (Line 123-160)
- Starts periodic refresh if WebSocket not connected (Line 152-155)
- Renders NaturalLanguageInput component (Line 646-655)

**Lines Modified:**
- 10: Added `useMemo`, `useCallback` imports
- 215-218: Memoized `handleWorkflowCreated` callback
- 225-236: Memoized `selectedIncidentObject` to prevent reference changes
- 652-653: Pass memoized props to NaturalLanguageInput

**Current State:**
- Uses memoization to create stable prop references
- Should prevent child components from re-rendering unnecessarily

### 4. `/Users/chasemad/Desktop/mini-xdr/frontend/app/components/NaturalLanguageInput.tsx`
**Purpose:** NLP chat interface for creating workflows

**Critical Local State:**
```typescript
const [parsedWorkflow, setParsedWorkflow] = useState<ParsedWorkflow | null>(null)
```

**User Flow:**
1. User types natural language request
2. User clicks "Parse" button
3. Component calls NLP parser
4. Sets `parsedWorkflow` state with result
5. User reviews parsed workflow
6. **PROBLEM:** If polling happens here, component re-renders and state is lost
7. User clicks "Create Workflow" - but workflow is gone!

**Lines Modified:**
- 8: Added `memo` import
- 36: Added `selectedIncident` prop to avoid state lookup
- 69-75: Wrapped entire component in `React.memo()`
- 76: Changed from `const { state, dispatch }` to just `const { dispatch }`
- 83-85: Added debug logging to track re-renders
- 117: Use `selectedIncident` prop instead of `state.incidents.find()`
- 322-325: Use `selectedIncident` prop instead of state lookup
- 495-499: Custom memo comparison function

**Memo Strategy:**
```typescript
}, (prevProps, nextProps) => {
  // Only re-render if selectedIncidentId changes
  return prevProps.selectedIncidentId === nextProps.selectedIncidentId
})
```

**Current State:**
- Wrapped in React.memo to prevent re-renders
- No longer depends on global state (only dispatch)
- Should preserve local state during parent re-renders

---

## FIXES ATTEMPTED (IN ORDER)

### Fix #1: Basic Change Detection
**What:** Added simple JSON string comparison
**Where:** DataService.ts, hasDataChanged()
**Result:** FAILED - Too sensitive, every tiny change triggered update

### Fix #2: Intelligent Change Detection
**What:** Rewrote hasDataChanged() with type-specific logic
- Workflows: Only update if status, step, or progress > 10% changes
- Incidents: Only update if status, escalation, or risk > 0.1 changes
- Uses Map-based ID comparison for efficiency
- Logs all changes to console

**Where:** DataService.ts lines 30-141
**Result:** IMPROVED but INCOMPLETE - State updates less frequent, but still causing re-renders

### Fix #3: Increased Polling Interval
**What:** Changed from 30s → 45s
**Where:** DataService.ts line 285
**Result:** MINIMAL IMPROVEMENT - Less frequent but still disruptive

### Fix #4: Component Memoization
**What:** Wrapped NaturalLanguageInput in React.memo()
**Where:** NaturalLanguageInput.tsx lines 69-75, 495-499
**Result:** UNKNOWN - User still reports issue

### Fix #5: Removed State Dependencies
**What:** Component no longer reads from global state, only uses dispatch
**Where:** NaturalLanguageInput.tsx line 76, 117, 322-325
**Result:** UNKNOWN - Should prevent re-renders from state changes

### Fix #6: Stable Props with Memoization
**What:** Memoized callback functions and incident object in parent
**Where:** WorkflowsPage.tsx lines 215-236
**Result:** UNKNOWN - Should prevent prop reference changes

---

## CURRENT HYPOTHESIS: WHY FIXES AREN'T WORKING

### Theory #1: React.memo Not Preventing Re-renders
**Why:** The memo comparison function might not be working correctly
**Evidence Needed:** Check browser console for `[NaturalLanguageInput] Component rendered/re-rendered` logs
**Test:**
```javascript
// In browser console during polling
// Should NOT see this log if memo is working:
[NaturalLanguageInput] Component rendered/re-rendered
```

### Theory #2: Memoization Dependencies Wrong
**Why:** `useMemo` deps include `state.incidents.length` which might change
**Location:** WorkflowsPage.tsx line 236
```typescript
}, [state.selectedIncident, state.incidents.length])  // ← This might be the problem
```
**Issue:** If incidents array changes (even without new incidents), length might trigger recalculation

### Theory #3: AppContext Provider Re-rendering
**Why:** The entire AppContext Provider might be re-rendering, forcing all children to re-render
**Location:** AppContext.tsx line 198-239
**Evidence Needed:** Check if provider value is recreating object reference

### Theory #4: Callback References Still Changing
**Why:** `handleWorkflowCreated` or other callbacks might not be properly memoized
**Location:** WorkflowsPage.tsx line 215-218
**Evidence Needed:** Use React DevTools Profiler to see what props are changing

### Theory #5: Next.js Fast Refresh Interfering
**Why:** Development mode hot reload might be causing re-renders
**Evidence:** Log shows "⚠ Fast Refresh had to perform a full reload due to a runtime error"
**Test:** Build production version and test there

### Theory #6: Polling Still Triggering State Updates
**Why:** hasDataChanged() might have edge cases that return true incorrectly
**Evidence Needed:** Check console logs during polling
```
[DataService] No significant workflow changes detected  ← Good
[DataService] Workflow w123 status changed: ...  ← Unexpected?
```

---

## SYSTEMATIC DEBUGGING APPROACH

### Step 1: Verify Polling Frequency
```bash
# Watch the logs to see how often polling happens
tail -f /tmp/frontend.log | grep "GET /workflows"

# Expected: One GET request every 45 seconds
# If faster: Interval not being respected
# If slower: Polling might be paused
```

### Step 2: Verify Change Detection
```javascript
// Open browser console (F12) on http://localhost:3000/workflows
// Wait for 45 seconds and observe logs

// EXPECTED LOGS (when nothing changes):
[DataService] No significant workflow changes detected
[DataService] No significant incident changes detected

// BAD LOGS (false positives):
[DataService] Workflow abc status changed: running → running  ← Same status!
[DataService] Workflow abc progress changed: 45.0% → 45.0%  ← Same progress!
```

### Step 3: Verify Component Re-render Prevention
```javascript
// In browser console, watch for this log:
[NaturalLanguageInput] Component rendered/re-rendered

// EXPECTED: Only see this log when:
// - User types in textarea
// - User clicks Parse button
// - User changes selected incident

// BAD: See this log during polling (every 45s)
```

### Step 4: Check Memo Comparison Function
Add more detailed logging to NaturalLanguageInput.tsx:

```typescript
}, (prevProps, nextProps) => {
  console.log('[NaturalLanguageInput] Memo comparison:', {
    prevId: prevProps.selectedIncidentId,
    nextId: nextProps.selectedIncidentId,
    equal: prevProps.selectedIncidentId === nextProps.selectedIncidentId
  })
  return prevProps.selectedIncidentId === nextProps.selectedIncidentId
})
```

**Expected:** Should log `equal: true` during polling
**Bad:** Logs `equal: false` even when ID hasn't changed

### Step 5: Use React DevTools Profiler
```
1. Install React DevTools browser extension
2. Open DevTools → Profiler tab
3. Click "Record" button
4. Wait 45 seconds (for polling to happen)
5. Click "Stop" button
6. Examine which components re-rendered and why

Look for:
- NaturalLanguageInput in the flame graph
- "Why did this render?" section shows prop/state changes
```

### Step 6: Check AppContext Provider Value
Add logging to AppContext.tsx:

```typescript
export function AppProvider({ children }: AppProviderProps) {
  const [state, dispatch] = useReducer(appReducer, initialState)

  // Add this debug code:
  const contextValue = useMemo(() => {
    console.log('[AppContext] Creating new context value')
    return { state, dispatch }
  }, [state, dispatch])

  return (
    <AppContext.Provider value={contextValue}>
      {children}
    </AppContext.Provider>
  )
}
```

**Expected:** Log should NOT appear during polling (if change detection works)
**Bad:** Log appears every 45 seconds (context recreating)

### Step 7: Test State Reference Stability
Add to WorkflowsPage.tsx:

```typescript
useEffect(() => {
  console.log('[WorkflowsPage] State reference changed', {
    incidentsLength: state.incidents.length,
    workflowsLength: state.workflows.length,
    selectedIncident: state.selectedIncident
  })
}, [state])  // Will log whenever state object reference changes
```

**Expected:** Should log ONLY when significant changes occur
**Bad:** Logs every 45 seconds even when "No significant changes detected"

### Step 8: Verify Callbacks Are Stable
Add to WorkflowsPage.tsx:

```typescript
useEffect(() => {
  console.log('[WorkflowsPage] handleWorkflowCreated reference changed')
}, [handleWorkflowCreated])
```

**Expected:** Should only log once on component mount
**Bad:** Logs multiple times (callback recreating)

### Step 9: Check DataService Change Detection Logic
Add detailed logging to DataService.ts hasDataChanged():

```typescript
private hasDataChanged(oldData: any[], newData: any[], dataType: 'incidents' | 'workflows'): boolean {
  console.log(`[DataService] hasDataChanged called for ${dataType}`, {
    oldLength: oldData.length,
    newLength: newData.length,
    oldData: JSON.stringify(oldData).substring(0, 100),
    newData: JSON.stringify(newData).substring(0, 100)
  })

  // ... rest of logic
}
```

**Look for:**
- Are old/new data actually different?
- Is the comparison logic correct?
- Are there edge cases (empty arrays, null values, etc.)?

### Step 10: Test Production Build
Development mode has extra overhead and debugging features that might cause issues:

```bash
cd /Users/chasemad/Desktop/mini-xdr/frontend
npm run build
npm start  # Runs production build

# Then test at http://localhost:3000/workflows
```

**If problem goes away:** Development mode issue (Fast Refresh, hot reload, etc.)
**If problem persists:** Real logic issue in the code

---

## POTENTIAL ROOT CAUSES (PRIORITIZED)

### 1. hasDataChanged() Returns False But State Still Updates (HIGH PRIORITY)
**Symptom:** Console shows "No significant changes" but component still re-renders
**Root Cause:** The callback is being called even when hasDataChanged() returns false

**Check this code:**
```typescript
// DataService.ts line 173
if (this.hasDataChanged(this.lastIncidents, incidents, 'incidents')) {
  this.lastIncidents = incidents
  this.callbacks.onIncidentsUpdate?.(incidents)  // ← Should only call if true
}
```

**Possible Bug:** Callback is being called outside this if statement somewhere

**How to Verify:**
```typescript
// Add this logging:
const hasChanged = this.hasDataChanged(this.lastIncidents, incidents, 'incidents')
console.log('[DataService] hasChanged result:', hasChanged)
if (hasChanged) {
  console.log('[DataService] Calling onIncidentsUpdate callback')
  this.lastIncidents = incidents
  this.callbacks.onIncidentsUpdate?.(incidents)
} else {
  console.log('[DataService] Skipping callback - no changes')
}
```

### 2. AppContext Reducer Creates New State Even When Data Unchanged (HIGH PRIORITY)
**Symptom:** hasDataChanged() works, but reducer still creates new state object

**Check this code:**
```typescript
// AppContext.tsx line 102-107
case 'SET_INCIDENTS':
  return {
    ...state,
    incidents: action.payload,  // ← New array reference
    lastUpdated: { ...state.lastUpdated, incidents: Date.now() }  // ← Always changes!
  }
```

**Root Cause:** `lastUpdated` timestamp ALWAYS changes, creating new state reference

**Possible Fix:**
```typescript
case 'SET_INCIDENTS':
  // Only update if data actually different
  if (state.incidents === action.payload) {
    return state  // ← Return same state object!
  }
  return {
    ...state,
    incidents: action.payload,
    lastUpdated: { ...state.lastUpdated, incidents: Date.now() }
  }
```

### 3. React.memo Comparison Function Not Working (MEDIUM PRIORITY)
**Symptom:** Component re-renders despite memo

**Check this code:**
```typescript
// NaturalLanguageInput.tsx line 495-499
}, (prevProps, nextProps) => {
  return prevProps.selectedIncidentId === nextProps.selectedIncidentId
})
```

**Possible Issue:** Comparison returns wrong value, or props are different

**How to Debug:**
```typescript
}, (prevProps, nextProps) => {
  const shouldSkipRender = prevProps.selectedIncidentId === nextProps.selectedIncidentId
  console.log('[NaturalLanguageInput] Memo comparison:', {
    prevId: prevProps.selectedIncidentId,
    nextId: nextProps.selectedIncidentId,
    shouldSkipRender,
    onWorkflowCreatedChanged: prevProps.onWorkflowCreated !== nextProps.onWorkflowCreated,
    selectedIncidentChanged: prevProps.selectedIncident !== nextProps.selectedIncident
  })
  return shouldSkipRender
})
```

**Note:** If `onWorkflowCreated` or `selectedIncident` props are changing, memo won't help

### 4. useMemo Dependencies Too Broad (MEDIUM PRIORITY)
**Symptom:** Memoized values recalculate even when they shouldn't

**Check this code:**
```typescript
// WorkflowsPage.tsx line 236
}, [state.selectedIncident, state.incidents.length])
```

**Issue:** `state.incidents.length` might change even when no new incidents
- Example: Incident status changes, array recreated, same length

**Better Approach:**
```typescript
}, [state.selectedIncident])  // Only depend on ID

// OR use a ref to track if incidents array actually changed
const prevIncidentsRef = useRef(state.incidents)
useEffect(() => {
  prevIncidentsRef.current = state.incidents
}, [state.incidents])
```

### 5. Multiple DataService Instances (LOW PRIORITY)
**Symptom:** Multiple polling intervals running simultaneously

**How to Check:**
```bash
# In browser console, count how many times this appears in 45 seconds:
[DataService] No significant workflow changes detected

# Expected: 1 time
# Bad: 2+ times (multiple services polling)
```

**Root Cause:** Component remounting creates new service but doesn't cleanup old one

**Check:** WorkflowsPage.tsx line 157-159
```typescript
return () => {
  service.cleanup()  // ← Is this being called?
}
```

### 6. DataService lastIncidents/lastWorkflows Not Persisting (LOW PRIORITY)
**Symptom:** hasDataChanged() always returns true on first comparison

**Check this code:**
```typescript
// DataService.ts lines 19-20
private lastIncidents: any[] = []
private lastWorkflows: any[] = []
```

**Issue:** If service recreates, these reset to empty arrays

**How to Verify:**
```typescript
async refreshIncidents() {
  console.log('[DataService] refreshIncidents called, lastIncidents:', this.lastIncidents.length)
  // Should be > 0 after first fetch
}
```

---

## RECOMMENDED NEXT STEPS

### Immediate Actions (Do First)

1. **Add Comprehensive Logging**
   - Add the logging from Step 1 in "Systematic Debugging" section
   - Run for 5 minutes and collect all logs
   - Analyze the exact sequence of events

2. **Verify hasDataChanged() is Actually Preventing Callbacks**
   - Add logging before/inside the if statement
   - Confirm callback is NOT called when hasDataChanged() returns false

3. **Check AppContext Reducer lastUpdated Field**
   - This timestamp always changes, forcing new state object
   - Consider removing it or only updating when data changes

### Secondary Actions (If Above Don't Work)

4. **Test React.memo with Logging**
   - Add detailed logging to memo comparison function
   - Verify it's actually blocking re-renders

5. **Use React DevTools Profiler**
   - Record a 1-minute session including polling
   - Examine why NaturalLanguageInput is re-rendering

6. **Test Production Build**
   - Rule out development mode issues

### Nuclear Options (Last Resort)

7. **Debounce State Updates**
   ```typescript
   // In AppContext, debounce SET_INCIDENTS action
   const debouncedDispatch = useMemo(() =>
     debounce((action) => dispatch(action), 500)
   , [])
   ```

8. **Move Parsed Workflow to Global State**
   - Store in AppContext instead of local state
   - Won't be lost during re-renders
   - But defeats purpose of component encapsulation

9. **Disable Polling Entirely**
   ```typescript
   // Comment out in WorkflowsPage.tsx
   // service.startPeriodicRefresh()
   ```
   - Manual refresh only
   - User clicks button to refresh
   - No automatic updates

---

## TESTING PROTOCOL

### Manual Test Case
```
1. Open http://localhost:3000/workflows
2. Open browser console (F12)
3. Select incident #1 from dropdown
4. Type in NLP chat: "Block IP 192.168.1.100 on the honeypot and send alert"
5. Click "Parse" button
6. Observe parsed workflow appears
7. Start timer
8. Wait 50 seconds (ensure polling happens at least once)
9. VERIFY: Parsed workflow still visible
10. VERIFY: Console shows "No significant changes detected"
11. VERIFY: Console does NOT show "Component rendered/re-rendered" during polling
12. Click "Create Workflow" button
13. VERIFY: Workflow is created successfully
```

**Success Criteria:**
- ✅ Parsed workflow visible for entire 50 seconds
- ✅ No UI flicker or disruption
- ✅ Console shows only DataService logs, not component re-render logs
- ✅ Workflow creation succeeds

**Failure Indicators:**
- ❌ Parsed workflow disappears at 45-second mark
- ❌ UI flickers or "jumps"
- ❌ Console shows "Component rendered/re-rendered" at 45 seconds
- ❌ Workflow creation fails or shows error

### Automated Test (Optional)
```typescript
// Create a test file: __tests__/polling.test.tsx

import { render, screen, waitFor } from '@testing-library/react'
import { act } from 'react-dom/test-utils'
import NaturalLanguageInput from '../app/components/NaturalLanguageInput'

test('parsed workflow persists during parent re-renders', async () => {
  const { rerender } = render(
    <NaturalLanguageInput
      selectedIncidentId={1}
      selectedIncident={{ id: 1, status: 'open' }}
    />
  )

  // User parses workflow
  const textarea = screen.getByPlaceholderText(/describe your response/i)
  fireEvent.change(textarea, { target: { value: 'Block IP 1.2.3.4' } })
  fireEvent.click(screen.getByText('Parse'))

  await waitFor(() => {
    expect(screen.getByText(/Generated Workflow/i)).toBeInTheDocument()
  })

  // Simulate parent re-render (like what happens during polling)
  rerender(
    <NaturalLanguageInput
      selectedIncidentId={1}  // Same ID
      selectedIncident={{ id: 1, status: 'open' }}  // Same incident
    />
  )

  // Verify parsed workflow still visible
  expect(screen.getByText(/Generated Workflow/i)).toBeInTheDocument()
})
```

---

## KEY FILES TO EXAMINE (FULL PATHS)

```
/Users/chasemad/Desktop/mini-xdr/frontend/app/services/DataService.ts
/Users/chasemad/Desktop/mini-xdr/frontend/app/contexts/AppContext.tsx
/Users/chasemad/Desktop/mini-xdr/frontend/app/workflows/page.tsx
/Users/chasemad/Desktop/mini-xdr/frontend/app/components/NaturalLanguageInput.tsx
```

## BACKEND ENDPOINTS (For Reference)

```
GET http://localhost:8000/api/incidents
GET http://localhost:8000/api/response/workflows
```

## LOGS TO MONITOR

```bash
# Frontend dev server
tail -f /tmp/frontend.log

# Browser console
Open http://localhost:3000/workflows
Press F12 → Console tab
Filter: "DataService" or "NaturalLanguageInput"
```

---

## CONTACT CONTEXT

**Project:** Mini-XDR (Security Operations Center Platform)
**User Issue:** "its still doing it and if i dont accept the workflow after the chat prompt it gets removed when it does the poll/refresh thing"
**Environment:** Development mode, Next.js 15.5.0, localhost:3000
**Backend:** Running on localhost:8000

---

## ADDITIONAL DEBUGGING COMMANDS

### Check if multiple services running
```bash
# Should show only 1 node process on port 3000
lsof -ti:3000
```

### Check backend is responding
```bash
curl http://localhost:8000/api/incidents
curl http://localhost:8000/api/response/workflows
```

### Watch polling in real-time
```bash
# In one terminal
tail -f /tmp/frontend.log | grep -E "(GET /workflows|Compiled)"

# Should see pattern every 45 seconds:
# GET /workflows 200 in XXms
```

### Check for memory leaks
```javascript
// In browser console
performance.memory
// Run this before and after several polling cycles
// If heapSize keeps growing: memory leak
```

---

## SUCCESS CRITERIA FOR COMPLETE FIX

1. ✅ User can parse NLP workflow and review it for 2-3 minutes without it disappearing
2. ✅ No visible UI flicker or "refresh" during polling
3. ✅ Console shows clear logging of when/why updates occur
4. ✅ Polling still detects real changes (new incidents, workflow status changes)
5. ✅ Component re-renders only when user interacts or significant data changes
6. ✅ All existing functionality works (workflow creation, execution, etc.)

---

## QUESTIONS TO ANSWER

1. Is `hasDataChanged()` actually preventing the callback from being invoked?
2. Is the AppContext reducer creating new state objects even when data unchanged?
3. Is React.memo actually preventing component re-renders?
4. Are the memoized props (callbacks, incident object) staying stable?
5. Is there a development mode issue that wouldn't occur in production?
6. Are there multiple DataService instances polling simultaneously?

---

## FINAL NOTES

- WebSocket is intentionally disabled (connection leak issue)
- Polling is necessary for real-time updates
- User needs smooth experience during workflow review/creation
- The fix must balance responsiveness (detecting changes) with stability (no disruption)
- Production readiness depends on solving this issue

**Last Known State:** All fixes applied but user still reports issue persisting. Need to add comprehensive logging and follow systematic debugging approach to identify root cause.
