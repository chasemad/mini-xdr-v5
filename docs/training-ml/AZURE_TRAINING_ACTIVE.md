# 🚀 Azure ML Training - ACTIVE JOB

**Status:** ✅ **TRAINING IN PROGRESS**  
**Started:** October 6, 2025 at 22:44 UTC  
**Job ID:** `calm_frame_b9rlxztg0v`

---

## 📊 Training Details

### Data Uploaded ✅
- **Size:** 586.43 MB
- **Upload Time:** 31 seconds  
- **Samples:** 4,008,000+ events
- **Classes:** 13 attack classes
- **Includes:** 
  - Existing datasets (CICIDS2017, UNSW-NB15, KDD Cup)
  - New Windows/AD attack data (Mordor, EVTX, OpTC, APT29, Atomic Red Team)

### Training Configuration
- **Compute:** cpu-cluster (Standard_D4s_v3 - 4 cores, 16GB RAM)
- **Model:** 13-class threat detector + specialist models
- **Epochs:** 50
- **Batch Size:** 512
- **Environment:** mini-xdr-training-env:1 (PyTorch + CUDA ready)

### Expected Performance
- **Accuracy:** 85-92%
- **Training Time:** 2-4 hours (CPU cluster)
- **Classes Detected:**
  - 0: Normal Traffic
  - 1: DDoS/DoS
  - 2: Reconnaissance  
  - 3: Brute Force
  - 4: Web Attacks
  - 5: Malware/Botnet
  - 6: APT
  - 7: **Kerberos Attacks (NEW)**
  - 8: **Lateral Movement (NEW)**
  - 9: **Credential Theft (NEW)**
  - 10: **Privilege Escalation (NEW)**
  - 11: **Data Exfiltration (NEW)**
  - 12: **Insider Threats (NEW)**

---

## 🔗 Monitor Your Training

### Azure ML Studio (Recommended)
**URL:** https://ml.azure.com/runs/calm_frame_b9rlxztg0v?wsid=/subscriptions/e5636423-8514-4bdd-bfef-f7ecdb934260/resourcegroups/mini-xdr-ml-rg/workspaces/mini-xdr-ml-workspace&tid=564be7e4-0c14-4433-b0c6-8c6c5e381d7f

**What you'll see:**
- ✅ Real-time training logs
- ✅ Loss and accuracy curves
- ✅ CPU utilization
- ✅ Estimated completion time
- ✅ Model artifacts when complete

### Azure CLI
```bash
# Check job status
az ml job show \
  --name calm_frame_b9rlxztg0v \
  --workspace-name mini-xdr-ml-workspace \
  --resource-group mini-xdr-ml-rg

# Stream live logs
az ml job stream \
  --name calm_frame_b9rlxztg0v \
  --workspace-name mini-xdr-ml-workspace \
  --resource-group mini-xdr-ml-rg
```

---

## 💰 Cost Tracking

### Current Job
- **Compute:** Standard_D4s_v3 (4 vCPUs, 16GB RAM)
- **Cost:** ~$0.20/hour
- **Estimated Duration:** 2-4 hours
- **Estimated Total Cost:** $0.40-0.80

### Note About GPU Quota
Your Azure subscription has 0 GPU quota, so training is running on CPU. To get GPU access (10-50x faster):

1. **Request Quota Increase:**
   ```bash
   # Go to: Azure Portal → Subscriptions → Usage + quotas
   # Search for: "Standard NCASv3_T4 Family vCPUs" or "Standard NCv3 Family vCPUs"
   # Request: 4-8 vCPUs
   ```

2. **Typical GPU Training:**
   - T4 GPU: ~$0.40/hour, finishes in 30-60 min = $0.20-0.40
   - V100 GPU: ~$3/hour, finishes in 15-30 min = $0.75-1.50

---

## 📥 What Happens When Training Completes

### Automatic Downloads
Azure ML will automatically:
1. ✅ Save trained model to workspace storage
2. ✅ Generate metrics and performance reports
3. ✅ Create confusion matrix and visualizations
4. ✅ Package everything for download

### Manual Download (After Completion)
```bash
# Download model artifacts
az ml job download \
  --name calm_frame_b9rlxztg0v \
  --workspace-name mini-xdr-ml-workspace \
  --resource-group mini-xdr-ml-rg \
  --download-path ./models/azure_trained/

# Files you'll get:
# - best_model.pth (trained model)
# - scaler.pkl (feature scaler)
# - metrics.json (performance metrics)
# - training_logs.txt (complete logs)
```

---

## 🔄 After Training Completes

### Step 1: Verify Model Performance
```bash
# Check metrics
cat models/azure_trained/metrics.json

# Look for:
# - accuracy >= 0.85
# - f1_score >= 0.80
# - per_class_precision >= 0.75
```

### Step 2: Deploy to Backend
```bash
# Copy trained models to production location
cp models/azure_trained/best_model.pth models/enterprise/model.pt
cp models/azure_trained/scaler.pkl models/enterprise/scaler.pkl

# Update backend to use new models
# (backend will auto-detect and load on restart)
```

### Step 3: Test Detection
```bash
# Test with sample attacks
python3 scripts/testing/test_enterprise_detection.py

# Verify all attack types detected
```

### Step 4: Clean Up Azure Resources (Optional)
```bash
# Delete compute cluster to stop all costs
az ml compute delete \
  --name cpu-cluster \
  --workspace-name mini-xdr-ml-workspace \
  --resource-group mini-xdr-ml-rg \
  --yes

# Or delete entire resource group
az group delete \
  --name mini-xdr-ml-rg \
  --yes
```

---

## 📊 Training Progress Checklist

- [x] Azure ML workspace created
- [x] Compute cluster provisioned
- [x] Training data uploaded (586MB)
- [x] Training job submitted
- [ ] Training in progress (2-4 hours)
- [ ] Model training complete
- [ ] Model downloaded
- [ ] Model deployed to backend
- [ ] Detection tests passed
- [ ] Production ready

---

## 🐛 Troubleshooting

### If Training Fails
1. **Check logs in Azure ML Studio** (link above)
2. **Common issues:**
   - Out of memory: Reduce batch size in training script
   - Data loading error: Check data format
   - Timeout: Increase max runtime

### If Training is Too Slow
**Current:** CPU cluster (2-4 hours)

**Options to speed up:**
1. Request GPU quota increase (10-50x faster)
2. Use spot instances (same speed, 70% cheaper)
3. Increase CPU cluster size (2x cores = ~1.5x faster)

### If You Need to Cancel
```bash
# Cancel the training job
az ml job cancel \
  --name calm_frame_b9rlxztg0v \
  --workspace-name mini-xdr-ml-workspace \
  --resource-group mini-xdr-ml-rg
```

---

## 📞 Support & Resources

### Azure ML Documentation
- Job monitoring: https://learn.microsoft.com/azure/machine-learning/how-to-track-monitor-analyze-runs
- Download outputs: https://learn.microsoft.com/azure/machine-learning/how-to-use-pipeline#download-outputs

### Project Documentation
- Training guide: `AZURE_ML_TRAINING_QUICKSTART.md`
- Full plan: `docs/MINI_CORP_ENTERPRISE_DEPLOYMENT_PLAN.md`
- Session handoff: `docs/SESSION_HANDOFF_WORKFLOW_TESTING.md`

### Workspace Info
```json
{
  "subscription_id": "e5636423-8514-4bdd-bfef-f7ecdb934260",
  "resource_group": "mini-xdr-ml-rg",
  "workspace_name": "mini-xdr-ml-workspace",
  "location": "eastus",
  "job_id": "calm_frame_b9rlxztg0v"
}
```

---

## ✅ Success Criteria

Training is successful when:
- ✅ Job completes without errors
- ✅ Model accuracy ≥ 85%
- ✅ F1 score ≥ 0.80
- ✅ All 13 classes have precision > 0.75
- ✅ False positive rate < 5%
- ✅ Model files downloaded successfully

---

## 🎯 Next Steps After Successful Training

1. **Validate Model** (Day 1)
   - Download and test model
   - Verify performance metrics
   - Test with sample attacks

2. **Deploy to Production** (Day 2)
   - Integrate into backend
   - Run end-to-end tests
   - Monitor detection accuracy

3. **Deploy Mini Corp** (Week 3)
   - NOW SAFE: Models trained with corporate attack data
   - Follow: `docs/MINI_CORP_ENTERPRISE_DEPLOYMENT_PLAN.md`
   - Deploy Azure infrastructure
   - Configure workflows

---

**Training Status:** 🟢 IN PROGRESS  
**Estimated Completion:** 2-4 hours from start (22:44 UTC + 2-4h = ~00:44-02:44 UTC)  
**Monitor:** https://ml.azure.com (link above)

**You can safely close this terminal - training runs in the cloud! 🚀**

