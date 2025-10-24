# 🚀 Fresh Incident Test Instructions

**Date**: October 7, 2025  
**Purpose**: Clear mock incidents and run fresh attack against T-Pot honeypot

---

## ✅ **Pre-Flight Checklist**

All data flows verified as using real database data:
- ✅ Incident data from Incident table
- ✅ Events from Event table (T-Pot logs)
- ✅ AI analysis from OpenAI/xAI
- ✅ ML predictions from ensemble models
- ✅ Actions from Action/ActionLog/AdvancedResponseAction tables
- ✅ Real-time updates working

---

## 🗑️ **Step 1: Clear Mock Incidents**

### Option A: Use the Cleanup Script (Recommended)

```bash
cd /Users/chasemad/Desktop/mini-xdr
./scripts/clear_all_incidents.sh
```

**What it does**:
- Lists current incident count
- Asks for confirmation
- Deletes all incidents via API
- Shows summary of deleted/failed

### Option B: Manual API Deletion

```bash
# List all incidents
curl http://localhost:8000/incidents \
  -H "x-api-key: demo-minixdr-api-key"

# Delete specific incident (repeat for each ID)
curl -X DELETE http://localhost:8000/incidents/14 \
  -H "x-api-key: demo-minixdr-api-key"
```

### Verify Deletion

1. **Check incidents page**: http://localhost:3000/incidents
2. Should show "No incidents found" or empty list
3. **Check database directly** (optional):
   ```bash
   cd backend
   source venv/bin/activate
   python3 -c "from app.database import get_db; from app.models import Incident; from sqlalchemy import select; import asyncio; print('Checking incidents...'); exec('async def check(): async for db in get_db(): result = await db.execute(select(Incident)); print(f\"Found {len(result.scalars().all())} incidents\"); asyncio.run(check())')"
   ```

---

## 🎯 **Step 2: Run Fresh Attack Against T-Pot**

### Prerequisites

1. **T-Pot honeypot is running** at your configured IP
2. **Backend is running**: `cd backend && source venv/bin/activate && python -m app.main`
3. **Frontend is running**: `cd frontend && npm run dev`
4. **Log ingestion is active** (automatic if backend running)

### Attack Options

#### Option A: Automated Attack Script

```bash
cd /Users/chasemad/Desktop/mini-xdr
./test-honeypot-attack.sh
```

This script will:
- Launch SSH brute force attack against T-Pot
- Generate real honeypot logs
- Trigger incident creation
- Execute containment

#### Option B: Manual Attack (More Control)

```bash
# SSH brute force attack
for i in {1..50}; do
    ssh admin@<TPOT_IP> -p 22 2>/dev/null &
done

# Wait a few seconds between batches
sleep 5

for i in {1..50}; do
    ssh root@<TPOT_IP> -p 22 2>/dev/null &
done
```

Replace `<TPOT_IP>` with your T-Pot honeypot IP address.

#### Option C: Nmap Scan (Lighter Attack)

```bash
nmap -sV -p 22,23,80,443,3306,8080 <TPOT_IP>
```

---

## 👀 **Step 3: Watch Real-Time Incident Creation**

### Monitor Backend Logs

```bash
cd /Users/chasemad/Desktop/mini-xdr/backend
tail -f backend.log
```

**Look for**:
- Event ingestion from T-Pot
- ML model predictions
- Incident creation
- AI triage
- Agent orchestration

### Monitor Frontend

1. **Open incidents page**: http://localhost:3000/incidents
2. **Watch for new incident** (should appear within 1-2 minutes)
3. **Click on incident** to see enterprise UI
4. **Observe**:
   - Threat Status Bar updates
   - AI analysis generates
   - Real events in timeline
   - ML scores displayed

---

## 🎬 **Expected Flow**

### Timeline (from attack to enterprise UI)

```
00:00 - Launch attack against T-Pot
        ↓
00:30 - T-Pot logs attack attempts
        ↓
01:00 - Backend ingests T-Pot logs → Event table
        ↓
01:30 - ML models analyze events
        ↓
02:00 - Incident created with ML predictions
        ↓
02:30 - AI triage (GPT-4) analyzes incident
        ↓
03:00 - Agent orchestrator (if configured) acts
        ↓
03:30 - Incident appears in frontend
        ↓
04:00 - Click incident → Enterprise UI loads
        ↓
04:30 - All components display real data:
        • Threat Status Bar
        • AI Analysis with recommendations
        • Unified Response Timeline
        • Tactical Decision Center
        • Event Timeline
        • IOCs & Evidence
        • ML Analysis
```

**Total time**: 3-5 minutes from attack to full enterprise UI

---

## 🧪 **Step 4: Test Enterprise UI Features**

Once incident appears:

### 1. **Threat Status Bar**
- ✅ Shows "Attack: ACTIVE"
- ✅ Shows containment status
- ✅ Shows agent count (may be 0 initially)
- ✅ Shows confidence from ML models

### 2. **AI Analysis** (Left Column)
- ✅ Click "Refresh" if not loaded
- ✅ Read AI-generated summary
- ✅ See severity assessment
- ✅ View recommendations
- ✅ Click "Execute" on a recommendation
- ✅ Watch action appear in timeline

### 3. **Response Timeline** (Right Column)
- ✅ Filter actions by type
- ✅ Sort by newest/oldest/status
- ✅ Expand action cards
- ✅ Click "View Full Details"
- ✅ Test rollback (if available)

### 4. **Tactical Decision Center**
- ✅ Click "Contain Now" → Should block IP
- ✅ Click "Hunt Threats" → Should search similar
- ✅ Click "Escalate" → Shows alert

### 5. **Detailed Tabs**
- ✅ **Attack Timeline**: Shows real T-Pot events
- ✅ **IOCs & Evidence**: Shows extracted IPs, domains, hashes
- ✅ **ML Analysis**: Shows ensemble model scores
- ✅ **Forensics**: (placeholder for future)

---

## 📊 **Step 5: Verify Real Data**

### Check Each Component

1. **Threat Status Bar**
   - Source IP matches attack IP? ✅
   - Duration calculates correctly? ✅
   - Threat category from ML model? ✅

2. **AI Analysis**
   - Summary mentions real attack type? ✅
   - Confidence score from ML model? ✅
   - Recommendations make sense? ✅

3. **Response Timeline**
   - Manual actions (if you executed any)? ✅
   - Agent actions (if agents ran)? ✅
   - Workflow actions (if workflows ran)? ✅

4. **Event Timeline**
   - Shows real T-Pot log entries? ✅
   - Event IDs match honeypot types? ✅
   - Timestamps are recent? ✅

5. **IOCs**
   - IP addresses from attack? ✅
   - Domains/hashes if detected? ✅

6. **ML Scores**
   - Ensemble model scores present? ✅
   - Scores match incident severity? ✅

---

## 🎯 **Success Criteria**

### ✅ **Test is Successful If**:

1. Incident created within 3-5 minutes of attack
2. All enterprise UI components display real data
3. AI analysis generates (may take 10-20 seconds)
4. Events show real T-Pot logs
5. ML scores populated
6. Actions execute and appear in timeline
7. Real-time updates work (try executing action, watch it appear)
8. No hardcoded or placeholder data visible

### ❌ **Troubleshooting**

**Incident not appearing?**
- Check backend logs for errors
- Verify T-Pot logs are being ingested
- Check Event table: `SELECT COUNT(*) FROM events;`
- Verify ML models are loaded

**AI analysis not generating?**
- Check OpenAI API key is set
- Check backend logs for API errors
- Try manual refresh button

**Actions not showing?**
- Check Action/ActionLog tables
- Verify API endpoints are working
- Check browser console for errors

---

## 🔄 **Step 6: Repeat Test (Optional)**

To test multiple incidents:

1. **Keep first incident** for comparison
2. **Run another attack** (different type if possible)
3. **Watch for new incident**
4. **Compare** how enterprise UI handles different attack types

---

## 📝 **Quick Command Reference**

```bash
# Clear all incidents
./scripts/clear_all_incidents.sh

# Run attack
./test-honeypot-attack.sh

# Watch backend logs
cd backend && tail -f backend.log

# Check incident count
curl http://localhost:8000/incidents \
  -H "x-api-key: demo-minixdr-api-key" \
  | python3 -c "import sys, json; print(f\"Found {len(json.load(sys.stdin))} incidents\")"

# Open frontend
open http://localhost:3000/incidents
```

---

## 🎉 **Expected Results**

After completing all steps, you should have:

✅ Clean database with only fresh, real incidents  
✅ Enterprise UI displaying 100% real data  
✅ AI analysis from actual OpenAI/xAI API  
✅ ML predictions from trained models  
✅ Real T-Pot events in timeline  
✅ Working action execution  
✅ Real-time updates functional  
✅ Professional, enterprise-grade UI  

---

**Ready to proceed?** Run the cleanup script when you're ready!

```bash
cd /Users/chasemad/Desktop/mini-xdr
./scripts/clear_all_incidents.sh
```

Then launch an attack and watch the magic happen! 🚀

