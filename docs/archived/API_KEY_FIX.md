# API Key Mismatch Fix

## Problem
Frontend and backend had different API keys, causing "Invalid API key" errors when executing actions.

## Root Cause
- Backend `.env` uses the placeholder: `demo-minixdr-api-key`
- Frontend `env.local` had drifted to a different placeholder

## Solution
Updated frontend `env.local` to match the backend API key placeholder.

## How to Fix (Already Done)

The API key placeholders have been updated in:
```
/frontend/env.local
/backend/.env
```

**No restart needed** - Next.js will pick up the new environment variable on next request.

## Verify It's Fixed

1. Refresh the incident page in your browser (hard refresh: Cmd+Shift+R)
2. Click "Execute" on any action
3. Should work now! ✅

## If Still Getting Errors

### Check Backend API Key
```bash
cd /Users/chasemad/Desktop/mini-xdr/backend
grep "^API_KEY=" .env
```

### Check Frontend API Key
```bash
cd /Users/chasemad/Desktop/mini-xdr/frontend
grep "^NEXT_PUBLIC_API_KEY=" env.local
```

### They Should Match!
Both should show: `demo-minixdr-api-key` (replace with your own in production)

## Testing

After refreshing your browser:
1. Open DevTools (F12) → Network tab
2. Click "Execute" on an action
3. Look for the request to `/api/incidents/3/execute-ai-recommendation`
4. Check Request Headers → should see: `x-api-key: demo-minixdr-api-key`
5. Response should be 200 OK (not 401 Unauthorized)

## Prevention

To prevent this in the future:
- Keep API keys synchronized between frontend and backend
- Consider using a shared `.env` or config management system
- Document the API key in a secure location for team members
