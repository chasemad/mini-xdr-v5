# 🔑 Azure API Key Sync - Fixed!

**Issue:** Frontend getting 401 Unauthorized after moving secrets to Azure  
**Root Cause:** Frontend `.env.local` had old API key  
**Status:** ✅ **FIXED!**

---

## What Happened

After syncing secrets from Azure Key Vault to the backend, the frontend still had the old API key in its `.env.local` file, causing authentication failures:

```
Backend API Key:  788cf45e96f1f65a97407a6cc1e0ea84751ee5088c26c9b8bc1b81860b86018f ✅
Frontend API Key: demo-minixdr-api-key ❌
Result: 401 Unauthorized
```

---

## The Fix

### 1. Updated Frontend Configuration
```bash
# Updated /Users/chasemad/Desktop/mini-xdr/frontend/.env.local
NEXT_PUBLIC_API_KEY=788cf45e96f1f65a97407a6cc1e0ea84751ee5088c26c9b8bc1b81860b86018f
```

### 2. Restarted Frontend
```bash
pkill -f "next dev"
cd frontend && npm run dev
```

### 3. Created Auto-Sync Script
Created `scripts/sync-frontend-api-key.sh` to automatically sync frontend API key from backend.

### 4. Updated Main Sync Script
Modified `scripts/sync-secrets-from-azure.sh` to automatically update both backend and frontend.

---

## Prevention - Future Sync Process

### Recommended Flow (One Command):
```bash
# This now syncs BOTH backend and frontend automatically!
./scripts/sync-secrets-from-azure.sh minixdrchasemad
```

Then restart both services:
```bash
# Restart backend
pkill -f uvicorn
cd backend && source venv/bin/activate && uvicorn app.entrypoint:app --reload

# Restart frontend  
pkill -f "next dev"
cd frontend && npm run dev
```

### Or Use Start-All Script (Easiest):
```bash
# This handles everything automatically
./scripts/start-all.sh
```

---

## Manual Sync (If Needed)

If you ever need to manually sync just the frontend API key:
```bash
./scripts/sync-frontend-api-key.sh
```

This will:
1. Read API key from backend `.env`
2. Update frontend `.env.local`
3. Backup old configuration
4. Show restart instructions

---

## Environment Files

### Backend: `/backend/.env`
Source of truth for API keys (synced from Azure Key Vault)
```bash
API_KEY=788cf45e96f1f65a97407a6cc1e0ea84751ee5088c26c9b8bc1b81860b86018f
TPOT_HOST=74.235.242.205
# ... other secrets from Azure
```

### Frontend: `/frontend/.env.local`
Must match backend API key for authentication
```bash
NEXT_PUBLIC_API_KEY=788cf45e96f1f65a97407a6cc1e0ea84751ee5088c26c9b8bc1b81860b86018f
NEXT_PUBLIC_API_BASE=http://localhost:8000
```

---

## Verification

Test that everything works:

### 1. Check Backend
```bash
curl -H "x-api-key: 788cf45e96f1f65a97407a6cc1e0ea84751ee5088c26c9b8bc1b81860b86018f" \
  http://localhost:8000/health
```
**Expected:** `{"status":"healthy",...}`

### 2. Check Frontend
```bash
# Open browser
open http://localhost:3000

# Check browser console - should see no 401 errors
# Navigate to incidents page - should load successfully
```

### 3. Check Frontend → Backend Communication
```bash
# In browser, navigate to an incident
# URL: http://localhost:3000/incidents/incident/1

# Should load without authentication errors
```

---

## Common Issues & Solutions

### Issue: Still getting 401 errors
**Solution:**
```bash
# 1. Verify frontend has correct key
cat frontend/.env.local | grep NEXT_PUBLIC_API_KEY

# 2. Verify backend has correct key
cat backend/.env | grep "^API_KEY="

# 3. If they don't match, resync
./scripts/sync-frontend-api-key.sh

# 4. Hard restart frontend (clear cache)
pkill -f "next dev"
cd frontend && rm -rf .next
npm run dev
```

### Issue: Frontend not picking up new env vars
**Solution:**
```bash
# Next.js caches env vars - need full restart
pkill -f "next dev"
cd frontend
rm -rf .next  # Clear Next.js cache
npm run dev
```

### Issue: Lost API key after Azure sync
**Solution:**
```bash
# Re-sync from Azure (includes frontend sync now)
./scripts/sync-secrets-from-azure.sh minixdrchasemad

# Restart both services
./scripts/start-all.sh
```

---

## Updated Scripts

### 1. `sync-secrets-from-azure.sh` (Enhanced)
Now automatically syncs to both backend and frontend:
- ✅ Syncs backend `.env` from Azure Key Vault
- ✅ Syncs frontend `.env.local` with matching API key
- ✅ Creates backups of both files
- ✅ Shows restart instructions

### 2. `sync-frontend-api-key.sh` (New)
Dedicated script for frontend API key sync:
- Reads API key from backend `.env`
- Updates frontend `.env.local`
- Backs up old configuration
- Checks if frontend is running

### 3. `start-all.sh` (Already Working)
Full system startup with automatic environment setup

---

## Best Practices

### When Syncing Secrets:
1. ✅ Always use `sync-secrets-from-azure.sh` (syncs both)
2. ✅ Restart both backend and frontend after sync
3. ✅ Use `start-all.sh` for full system restart
4. ✅ Verify with a test API call

### When Rotating Keys:
1. Update in Azure Key Vault
2. Run `sync-secrets-from-azure.sh`
3. Restart all services
4. Test authentication

### When Deploying:
1. Ensure Azure Key Vault is accessible
2. Run sync script
3. Use `start-all.sh` for clean startup
4. Verify all services authenticate correctly

---

## Quick Reference

```bash
# Full sync from Azure (backend + frontend)
./scripts/sync-secrets-from-azure.sh minixdrchasemad

# Sync only frontend API key
./scripts/sync-frontend-api-key.sh

# Start entire system (handles everything)
./scripts/start-all.sh

# Restart backend
pkill -f uvicorn && cd backend && source venv/bin/activate && uvicorn app.entrypoint:app --reload

# Restart frontend
pkill -f "next dev" && cd frontend && npm run dev

# Test authentication
API_KEY=$(cat backend/.env | grep "^API_KEY=" | cut -d'=' -f2)
curl -H "x-api-key: $API_KEY" http://localhost:8000/health
```

---

## ✅ Status: RESOLVED

- ✅ Frontend `.env.local` updated with correct API key
- ✅ Frontend restarted and working
- ✅ Auto-sync script created
- ✅ Main sync script enhanced
- ✅ Authentication working end-to-end
- ✅ No more 401 errors!

**Your system is fully operational again!** 🎉

---

*Created: October 5, 2025*  
*Issue Resolution Time: 5 minutes*  
*Scripts Created: 2*


