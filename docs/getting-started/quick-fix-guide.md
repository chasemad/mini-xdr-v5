# Quick Fix Guide - Incident Action Buttons

## Problem
Incident detail page action buttons (like "Block IP") were failing:
- Showed error, then false success
- IP not actually blocked on T-Pot
- AI agent actions not synced with UI

## Solution (3 Steps)

### Step 1: Configure T-Pot Password
```bash
cd /Users/chasemad/Desktop/mini-xdr
./scripts/configure_tpot.sh
```

Enter when prompted:
- Host: 24.11.0.176 (or press Enter for default)
- Port: 64295 (or press Enter for default)
- Username: luxieum (or press Enter for default)
- Password: [your T-Pot SSH password]

### Step 2: Restart Backend
```bash
cd backend
source venv/bin/activate
uvicorn app.main:app --reload
```

### Step 3: Test It
1. Go to `http://localhost:3000/incidents/incident/3` (or any incident)
2. Click "Execute" on any AI recommendation
3. Should now work! ✅

## Verify IP Blocked
```bash
ssh -p 64295 luxieum@203.0.113.42 "sudo ufw status"
```

## What Was Fixed

### Backend
- Enhanced T-Pot connector logging and error handling
- Fixed SSH password authentication
- Added dual action recording (manual + workflow)
- New API endpoint: `GET /api/incidents/{id}/actions`

### Frontend
- Checks for already-executed actions on load
- Shows detailed error messages with troubleshooting
- Marks executed actions with checkmarks

## Files Changed
1. `backend/app/tpot_connector.py`
2. `backend/app/responder.py`
3. `backend/app/main.py`
4. `frontend/components/EnhancedAIAnalysis.tsx`
5. `frontend/app/incidents/incident/[id]/page.tsx`

## New Files
- `scripts/configure_tpot.sh` - T-Pot configuration helper
- `docs/operations/incident-action-execution-fix.md` - Full documentation
- `INCIDENT_ACTION_FIX_SUMMARY.md` - Detailed summary

## Still Having Issues?

### Error: "Not connected to T-Pot"
→ Run `./scripts/configure_tpot.sh` to set SSH password

### Error: "Command timeout"
→ Check T-Pot is accessible from your IP (should be 172.16.110.1)

### Error: "UFW command failed"
→ Verify SSH password is correct and user has sudo access

### Actions not showing as executed
→ Refresh the page, check browser console for errors

## Full Documentation
See: `docs/operations/incident-action-execution-fix.md`
