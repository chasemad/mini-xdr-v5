# Quick Restart Guide

## Problem
Backend server needs to be restarted to load:
1. New `/api/incidents/{id}/actions` endpoint (was returning 404)
2. All the fixes I made to T-Pot connector and action execution

Frontend needs restart to:
1. Clear cached API key
2. Pick up new environment variables

## Quick Solution (Automated)

### Option 1: Use the Restart Script
```bash
cd /Users/chasemad/Desktop/mini-xdr
./RESTART_SERVERS.sh
```

This will:
- Stop existing backend/frontend servers
- Start backend with all new code
- Start frontend with updated API key
- Show you the URLs and log locations

### Option 2: Manual Restart

#### Stop Everything First
```bash
# Kill backend
lsof -Pi :8000 -sTCP:LISTEN -t | xargs kill

# Kill frontend
lsof -Pi :3000 -sTCP:LISTEN -t | xargs kill
```

#### Start Backend
```bash
cd /Users/chasemad/Desktop/mini-xdr/backend
source venv/bin/activate
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Wait for: `✅ Successfully connected to T-Pot at 24.11.0.176`

#### Start Frontend (in new terminal)
```bash
cd /Users/chasemad/Desktop/mini-xdr/frontend
npm run dev
```

## After Restart

1. **Go to**: http://localhost:3000/incidents/incident/3
2. **Hard refresh**: Cmd+Shift+R (Mac) or Ctrl+Shift+R (Windows/Linux)
3. **Click "Execute"** on any AI recommendation
4. **Should work now!** ✅

## What to Look For

### Backend Logs Should Show:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
✅ Successfully connected to T-Pot at 24.11.0.176
✅ Connection test successful
```

### Browser Console Should NOT Show:
- ❌ 404 errors for `/api/incidents/3/actions`
- ❌ 401 Unauthorized errors
- ✅ Should see 200 OK responses

### When You Execute an Action:
- Loading spinner appears
- Either success message OR detailed error (not "Invalid API key")
- Action appears in Response Timeline
- Refreshing page shows checkmark on executed action

## Troubleshooting

### Backend Won't Start
```bash
# Check what's on port 8000
lsof -Pi :8000 -sTCP:LISTEN

# View backend logs
tail -f /Users/chasemad/Desktop/mini-xdr/backend/logs/server.log
```

### Frontend Won't Start
```bash
# Check what's on port 3000
lsof -Pi :3000 -sTCP:LISTEN

# View frontend logs
tail -f /Users/chasemad/Desktop/mini-xdr/frontend/logs/server.log
```

### Still Getting 401 Errors
1. Verify API key matches:
   ```bash
   # Backend
   cd backend && grep "^API_KEY=" .env

   # Frontend
   cd frontend && grep "^NEXT_PUBLIC_API_KEY=" env.local
   ```

2. Hard refresh browser (Cmd+Shift+R)
3. Clear browser cache
4. Try incognito/private window

### Still Getting 404 for /actions Endpoint
- Backend wasn't restarted after my code changes
- Use restart script above
- Check backend logs for startup errors

## Complete Restart Checklist

- [ ] Stop backend server (port 8000)
- [ ] Stop frontend server (port 3000)
- [ ] Start backend server
- [ ] Wait for T-Pot connection message
- [ ] Start frontend server
- [ ] Hard refresh browser
- [ ] Test action execution
- [ ] Verify in Response Timeline
- [ ] Check T-Pot firewall rules

## Files Changed That Require Restart

**Backend (requires restart):**
- `backend/app/main.py` - New endpoint + dual action recording
- `backend/app/tpot_connector.py` - Enhanced logging
- `backend/app/responder.py` - Better error handling

**Frontend (requires restart):**
- `frontend/env.local` - Updated API key
- `frontend/components/EnhancedAIAnalysis.tsx` - Already-executed check
- `frontend/app/incidents/incident/[id]/page.tsx` - Better errors

## Quick Commands Reference

```bash
# Restart everything
./RESTART_SERVERS.sh

# Just restart backend
cd backend && source venv/bin/activate && uvicorn app.main:app --reload

# Just restart frontend
cd frontend && npm run dev

# Check if servers are running
lsof -Pi :8000 -sTCP:LISTEN  # Backend
lsof -Pi :3000 -sTCP:LISTEN  # Frontend

# View logs
tail -f backend/logs/server.log
tail -f frontend/logs/server.log
```
