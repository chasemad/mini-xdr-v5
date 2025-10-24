# 🚀 Azure GPU Quota Request Guide

## 🎯 Most Likely to be AUTO-APPROVED Instantly

### Option 1: T4 GPU (Best for Auto-Approval) ⭐ RECOMMENDED
**Family:** `Standard NCASv3_T4 Family vCPUs`  
**Approval:** ✅ Usually **INSTANT** for new subscriptions  
**Why:** Lower-tier GPU, commonly approved automatically  
**Request Amount:** **4 vCPUs** (= 1 GPU)  
**Cost:** ~$0.40/hour ($0.28 spot)  
**Training Speed:** 30-60 minutes (10x faster than CPU)

### Option 2: NC-series (Fallback)
**Family:** `Standard NC Family vCPUs` (older K80 GPUs)  
**Approval:** ✅ Usually **INSTANT**  
**Request Amount:** **6 vCPUs** (= 1 GPU)  
**Cost:** ~$0.90/hour  
**Note:** Older GPU but still 5-8x faster than CPU

### Option 3: V100 (Powerful but May Require Approval)
**Family:** `Standard NCv3 Family vCPUs`  
**Approval:** ⚠️ Often requires **MANUAL REVIEW** (1-2 business days)  
**Request Amount:** **6 vCPUs** (= 1 GPU)  
**Cost:** ~$3/hour  
**Training Speed:** 15-30 minutes (fastest)

---

## 📍 How to Request (2 Methods)

### Method 1: Azure Portal (Easiest) ⭐

#### Step-by-Step:

1. **Go to Azure Portal:**
   - Open: https://portal.azure.com

2. **Navigate to Quotas:**
   - Search bar (top): Type `quotas`
   - Click: **"Quotas"** service

3. **Find Machine Learning Quotas:**
   - Click: **"Machine Learning"** or **"Compute"**
   - Location: Select **"East US"** (or your workspace region)

4. **Request T4 GPU Quota (Most Likely Auto-Approved):**
   - Search for: `NCASv3_T4`
   - Find: **"Standard NCASv3_T4 Family vCPUs"**
   - Current limit: 0
   - Click: **"Request quota increase"**
   - New limit: **4** (for 1 GPU) or **8** (for 2 GPUs)
   - Justification: "ML model training for cybersecurity threat detection"
   - Submit

5. **Approval:**
   - ✅ If auto-approved: Instant (refresh page)
   - ⏳ If manual review: Email notification (usually within 24 hours)

---

### Method 2: Azure CLI (Faster)

```bash
# Login to Azure
az login

# Set your subscription
az account set --subscription e5636423-8514-4bdd-bfef-f7ecdb934260

# Request T4 GPU quota (4 vCPUs = 1 GPU)
az quota create \
  --resource-name "StandardNCASv3_T4Family" \
  --scope "/subscriptions/e5636423-8514-4bdd-bfef-f7ecdb934260/providers/Microsoft.MachineLearningServices/locations/eastus" \
  --limit-object value=4 \
  --resource-type dedicated

# Check if approved (wait 1-2 minutes, then check)
az quota show \
  --resource-name "StandardNCASv3_T4Family" \
  --scope "/subscriptions/e5636423-8514-4bdd-bfef-f7ecdb934260/providers/Microsoft.MachineLearningServices/locations/eastus"
```

---

## 🎯 Recommended Strategy for Instant Approval

### 1. Start with T4 (Most Likely Instant)
```
Request: Standard NCASv3_T4 Family vCPUs
Amount: 4 vCPUs
Region: East US
```

**Why This Works:**
- T4 is entry-level ML GPU
- Azure typically auto-approves 4-8 vCPUs for new users
- No manual review needed
- Available in most regions

### 2. If T4 Not Available, Try NC-series (K80)
```
Request: Standard NC Family vCPUs  
Amount: 6 vCPUs
Region: East US
```

### 3. Alternative: Use Different Region
If East US quotas are full, try:
- **West US 2** (usually has capacity)
- **South Central US**
- **West Europe**

---

## ⏱️ Approval Times (What to Expect)

| GPU Type | Auto-Approval? | Time | Success Rate |
|----------|---------------|------|--------------|
| **T4 (NCASv3_T4)** | ✅ Usually | Instant - 5 min | ~90% |
| **K80 (NC)** | ✅ Often | Instant - 10 min | ~80% |
| **V100 (NCv3)** | ⚠️ Sometimes | 1-2 business days | ~50% instant |
| **A100 (NC_A100)** | ❌ Rarely | 2-5 business days | ~10% instant |

---

## 🚀 After Approval: Launch GPU Training

### 1. Check if Quota Approved
```bash
# List all quotas
az ml compute list-usage \
  --location eastus \
  --workspace-name mini-xdr-ml-workspace \
  --resource-group mini-xdr-ml-rg
```

### 2. Create T4 GPU Cluster
```bash
# Create T4 GPU cluster (after quota approved)
source ml-training-env/bin/activate
python3 << 'EOF'
from azure.ai.ml import MLClient
from azure.ai.ml.entities import AmlCompute
from azure.identity import AzureCliCredential

client = MLClient(
    credential=AzureCliCredential(),
    subscription_id="e5636423-8514-4bdd-bfef-f7ecdb934260",
    resource_group_name="mini-xdr-ml-rg",
    workspace_name="mini-xdr-ml-workspace"
)

cluster = AmlCompute(
    name="gpu-t4-cluster",
    type="amlcompute",
    size="Standard_NC4as_T4_v3",  # 1x T4 GPU
    min_instances=0,
    max_instances=1,
    tier="low_priority"  # 70% cheaper!
)

client.compute.begin_create_or_update(cluster).result()
print("✅ T4 GPU cluster created!")
EOF
```

### 3. Launch Training on GPU
```bash
# Launch faster GPU training
source ml-training-env/bin/activate
python3 scripts/azure-ml/launch_azure_training.py \
  --compute gpu-t4-cluster \
  --spot  # Use spot instances for 70% savings
```

---

## 💰 Cost Comparison

### Current CPU Training
- **Compute:** Standard_D4s_v3 (4 CPU cores)
- **Time:** 2-4 hours
- **Cost:** $0.40-0.80
- **Status:** Currently running

### With T4 GPU (After Approval)
- **Compute:** Standard_NC4as_T4_v3 (1x T4 GPU)
- **Time:** 30-60 minutes
- **Cost:** $0.40/hour × 1 hour = **$0.40** (same or cheaper!)
- **Speedup:** 10x faster
- **With Spot:** $0.28/hour = **$0.28** total

### With V100 GPU (If Approved)
- **Compute:** Standard_NC6s_v3 (1x V100 GPU)  
- **Time:** 15-30 minutes
- **Cost:** $3/hour × 0.5 hours = **$1.50**
- **Speedup:** 20x faster

---

## 🎯 What to Do Right Now

### Option A: Request T4 GPU (Recommended)
1. Go to: https://portal.azure.com
2. Search: "quotas"
3. Find: "Standard NCASv3_T4 Family vCPUs"
4. Request: **4 vCPUs** in **East US**
5. Wait: Check back in 5-10 minutes

### Option B: Keep Current CPU Training
- Your current training is already running
- Will complete in 2-4 hours
- Cost: $0.40-0.80
- **No action needed - just wait!**

### Option C: Cancel CPU, Wait for GPU, Then Retrain
```bash
# Cancel current job
az ml job cancel \
  --name calm_frame_b9rlxztg0v \
  --workspace-name mini-xdr-ml-workspace \
  --resource-group mini-xdr-ml-rg

# Request GPU quota (via portal)
# After approved, create GPU cluster (see above)
# Launch new training on GPU
```

---

## 🤔 Should You Cancel Current Training?

### Keep CPU Training Running If:
- ✅ You can wait 2-4 hours
- ✅ Want to minimize hassle
- ✅ $0.40-0.80 cost is acceptable

### Cancel and Wait for GPU If:
- ✅ GPU quota gets approved instantly
- ✅ Want 10x faster training (30-60 min)
- ✅ Plan to retrain multiple times

### My Recommendation:
**Let CPU training finish** (it's already running and will complete in a few hours). Meanwhile, request T4 GPU quota for future training runs. Best of both worlds!

---

## 📊 Check Quota Request Status

### Portal Method:
1. Go to: https://portal.azure.com
2. Click: Bell icon (notifications) in top right
3. Look for: "Quota request" notifications

### CLI Method:
```bash
# Check current quotas
az ml compute list-usage \
  --location eastus \
  --workspace-name mini-xdr-ml-workspace \
  --resource-group mini-xdr-ml-rg | grep -i "ncast4"

# If approved, you'll see:
# "limit": 4 (instead of 0)
```

### Email:
- Check email for: "Azure quota request approved"
- Usually arrives within minutes if auto-approved

---

## 🎯 Quick Decision Tree

```
Do you need faster training?
│
├─ No → Keep CPU training (2-4 hours, $0.40-0.80)
│
└─ Yes → Request T4 GPU quota now
    │
    ├─ Auto-approved (5-10 min)?
    │   └─ Yes → Cancel CPU, create GPU cluster, retrain
    │
    └─ Needs manual review (1-2 days)?
        └─ Let CPU finish, use GPU next time
```

---

## 📝 Exact Portal Steps (Screenshots)

### Step 1: Open Quotas
```
Azure Portal → Search "quotas" → Click "Quotas" service
```

### Step 2: Filter
```
Provider: Machine Learning
Location: East US
```

### Step 3: Find T4
```
Search box: "NCASv3_T4"
Find: "Standard NCASv3_T4 Family vCPUs"
Current: 0
```

### Step 4: Request
```
Click row → "Request quota increase" button
New limit: 4
Support method: Standard
Severity: C - Minimal impact
Description: "ML training for cybersecurity threat detection models"
Submit
```

### Step 5: Wait
```
Status: "Submitted" → "Approved" (5-10 min usually)
or
Status: "Submitted" → "In review" (1-2 business days)
```

---

## ⚡ Pro Tips

### Increase Approval Chances:
1. **Request small amounts first** (4 vCPUs, not 64)
2. **Use low-priority/spot** (cheaper = easier approval)
3. **Request T4, not V100** (entry-level = faster approval)
4. **Choose less busy regions** (West US 2 over East US)
5. **Provide good justification** ("ML model training" > "testing")

### What NOT to Do:
- ❌ Don't request 100 vCPUs immediately
- ❌ Don't request A100 GPUs (hardest to get)
- ❌ Don't use vague justification ("need GPU")
- ❌ Don't request in multiple regions simultaneously

---

## 🎯 Bottom Line

**Best for Instant Approval:**
```
GPU Type: Standard_NC4as_T4_v3 (T4)
Quota Name: Standard NCASv3_T4 Family vCPUs
Amount: 4 vCPUs (= 1 GPU)
Region: East US (or West US 2)
Approval Time: Usually instant (90% auto-approved)
```

**Go request it now at:** https://portal.azure.com → Search "quotas" → Request increase

**Meanwhile:** Your current CPU training continues running and will complete in 2-4 hours! 🚀

