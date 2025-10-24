# 🚀 Azure ML Fast Training - Quick Start Guide

**Goal:** Train Mini-XDR models 10-50x faster using Azure GPU compute on your 4M+ events

**Status:** ✅ Datasets Downloaded | 📋 Ready for Azure Setup

---

## 📊 What We Have

### Downloaded Datasets ✅
- ✅ Mordor (Kerberos, Golden Ticket, DCSync) - 2,000 samples
- ✅ EVTX Samples (Mimikatz, PSExec, PowerShell) - 1,000 samples  
- ✅ OpTC (Lateral movement, C2, exfiltration) - 3,000 samples
- ✅ APT29 (Advanced Persistent Threats) - 500 samples
- ✅ Atomic Red Team (MITRE ATT&CK) - 1,500 samples
- ✅ Existing datasets (CICIDS2017, UNSW-NB15, etc.) - 4M+ events

**Total:** 4,008,000+ training samples ready!

---

## 🎯 Fast Setup (3 Steps)

### Step 1: Convert Windows Datasets (5 minutes)

```bash
cd /Users/chasemad/Desktop/mini-xdr

# Convert Windows/AD datasets to Mini-XDR format (79 features, 13 classes)
python3 scripts/data-processing/convert_windows_datasets.py
```

**Output:** `datasets/windows_converted/windows_ad_converted.json` (~8,000 samples)

### Step 2: Setup Azure ML Workspace (10 minutes)

**Prerequisites:**
- Azure account with active subscription
- Azure CLI installed: `brew install azure-cli`
- Login: `az login`

```bash
# Set your Azure subscription ID
export AZURE_SUBSCRIPTION_ID="your-subscription-id-here"

# Create Azure ML workspace with GPU compute
python3 scripts/azure-ml/setup_azure_ml_workspace.py \
  --subscription-id $AZURE_SUBSCRIPTION_ID \
  --resource-group mini-xdr-ml-rg \
  --workspace mini-xdr-ml-workspace \
  --location eastus
```

**This creates:**
- ✅ Azure ML workspace
- ✅ GPU compute clusters (V100 or T4)
- ✅ Training environment (PyTorch + CUDA)
- ✅ Configuration file

### Step 3: Launch Training (2 minutes to submit, 30-120 min to complete)

```bash
# Launch GPU training job
python3 scripts/azure-ml/launch_azure_training.py \
  --compute gpu-cluster-t4  \
  --spot  # Use spot instances for 70% cost savings
```

**Training will:**
- ✅ Upload your 4M+ events to Azure
- ✅ Train on GPU (10-50x faster than local)
- ✅ Save trained models
- ✅ Download models back to your machine

---

## 💰 Cost Breakdown

### Option 1: T4 GPU (Recommended for most users)
- **Compute:** Standard_NC4as_T4_v3 (1x T4 GPU)
- **Cost:** ~$0.40/hour (spot instances)
- **Training time:** 1-2 hours
- **Total cost:** $0.40-0.80

### Option 2: V100 GPU (Fastest)
- **Compute:** Standard_NC6s_v3 (1x V100 GPU)
- **Cost:** ~$3/hour (dedicated)
- **Training time:** 30-60 minutes
- **Total cost:** $1.50-3.00

### Cost Comparison
- **Local CPU:** FREE but takes 20-40 hours
- **Azure T4 GPU:** $0.80 and finishes in 1-2 hours ✅ **Best value**
- **Azure V100 GPU:** $3 and finishes in 30-60 min (fastest)

**Recommendation:** Start with T4 spot instances ($0.80 total)

---

## 📈 Expected Performance

### Training Metrics
- **Accuracy:** 85-92% (13-class detection)
- **F1 Score:** 0.82-0.90
- **Classes detected:**
  - 0: Normal traffic
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

### Speed Comparison
| Method | Hardware | Time | Cost |
|--------|----------|------|------|
| Local CPU | MacBook | 20-40 hours | $0 |
| Local GPU | M1/M2 | 4-8 hours | $0 |
| Azure T4 | Cloud GPU | 1-2 hours | $0.80 |
| Azure V100 | Cloud GPU | 30-60 min | $3 |

---

## 🔍 Monitoring Your Training

### Option 1: Azure ML Studio (Web UI)
After launching training, you'll get a URL like:
```
https://ml.azure.com/runs/mini-xdr-fast-training_...
```

Open in browser to see:
- ✅ Real-time training metrics
- ✅ Loss curves
- ✅ GPU utilization
- ✅ Estimated completion time

### Option 2: Azure CLI
```bash
# List all training jobs
az ml job list \
  --workspace-name mini-xdr-ml-workspace \
  --resource-group mini-xdr-ml-rg

# Get job status
az ml job show \
  --name <job-name> \
  --workspace-name mini-xdr-ml-workspace \
  --resource-group mini-xdr-ml-rg

# Stream logs
az ml job stream \
  --name <job-name> \
  --workspace-name mini-xdr-ml-workspace \
  --resource-group mini-xdr-ml-rg
```

---

## 📦 What You Get After Training

### Downloaded Files
```
models/
├── azure_trained/
│   ├── best_model.pth          # Trained PyTorch model
│   ├── scaler.pkl              # Feature scaler
│   ├── metrics.json            # Performance metrics
│   └── confusion_matrix.png    # Visualization
```

### Metrics File Example
```json
{
  "accuracy": 0.89,
  "f1_score": 0.87,
  "training_time_minutes": 45,
  "num_samples": 4008000,
  "num_classes": 13,
  "per_class_precision": {
    "kerberos_attack": 0.94,
    "lateral_movement": 0.91,
    "credential_theft": 0.93
  }
}
```

---

## 🛠️ Troubleshooting

### Issue: "Azure subscription not found"
```bash
# Login to Azure
az login

# List subscriptions
az account list --output table

# Set correct subscription
az account set --subscription "your-subscription-id"
```

### Issue: "Compute quota exceeded"
**Solution:** Request quota increase or use different region
```bash
# Check quota
az ml compute list-usage \
  --location eastus \
  --workspace-name mini-xdr-ml-workspace \
  --resource-group mini-xdr-ml-rg
```

### Issue: "Training job fails"
1. Check logs in Azure ML Studio
2. Verify data uploaded correctly
3. Try with smaller dataset first
4. Use CPU cluster as fallback

### Issue: "Cost concerns"
**Solutions:**
- Use spot instances (70% cheaper)
- Set max training time: `--max-runtime-minutes 120`
- Use auto-shutdown: `--idle-time-before-scale-down 300`
- Delete compute when done: `az ml compute delete ...`

---

## 🎓 Advanced Options

### Use Different Compute
```bash
# V100 GPU (fastest)
python3 scripts/azure-ml/launch_azure_training.py \
  --compute gpu-cluster-v100

# CPU (cheapest, slowest)
python3 scripts/azure-ml/launch_azure_training.py \
  --compute cpu-cluster
```

### Custom Hyperparameters
Edit `scripts/azure-ml/azure_train.py` and modify:
```python
--batch-size 512      # Larger = faster but needs more GPU memory
--epochs 50           # More epochs = better accuracy but longer training
--learning-rate 0.001 # Lower = more stable, higher = faster convergence
```

### Multi-GPU Training
For datasets >10M events:
```bash
# Use 4x V100 GPUs (4x faster)
python3 scripts/azure-ml/launch_azure_training.py \
  --compute gpu-cluster-v100 \
  --num-nodes 4
```

---

## 📋 Checklist

Before starting:
- [ ] Azure account created
- [ ] Azure CLI installed and logged in
- [ ] Subscription ID set: `export AZURE_SUBSCRIPTION_ID=...`
- [ ] Datasets downloaded (✅ Already done!)
- [ ] Datasets converted to Mini-XDR format

After training:
- [ ] Models downloaded from Azure
- [ ] Metrics reviewed (accuracy >85%)
- [ ] Models deployed to backend
- [ ] Test detection with sample attacks
- [ ] Delete Azure compute to stop costs

---

## 🚀 Next Steps After Training

1. **Integrate Models into Backend**
   ```bash
   # Copy trained models
   cp models/azure_trained/* models/enterprise/
   
   # Update backend ML engine
   # Edit: backend/app/ml_engine.py
   ```

2. **Test Detection**
   ```bash
   python3 scripts/testing/test_enterprise_detection.py
   ```

3. **Deploy to Production**
   ```bash
   # Restart backend with new models
   cd backend
   python3 app/main.py
   ```

4. **Start Mini Corp Infrastructure**
   - Now safe to deploy (models trained!)
   - See: `docs/MINI_CORP_ENTERPRISE_DEPLOYMENT_PLAN.md`

---

## 💡 Tips for Success

1. **Start Small:** Test with 10% of data first to verify everything works
2. **Use Spot Instances:** 70% cheaper, perfect for training
3. **Monitor Costs:** Set spending alerts in Azure portal
4. **Save Checkpoints:** Training saves best model automatically
5. **Compare Results:** Run local training on small sample to verify Azure results
6. **Clean Up:** Delete compute resources after training to stop costs

---

## 📞 Support

**Issues?** Check:
- Azure ML Studio logs
- `scripts/azure-ml/workspace_config.json` 
- Azure subscription limits
- Network connectivity

**Questions about:**
- Azure setup → Azure ML documentation
- Model training → `docs/MINI_CORP_ENTERPRISE_DEPLOYMENT_PLAN.md`
- Integration → `docs/SESSION_HANDOFF_WORKFLOW_TESTING.md`

---

**Ready to train 10-50x faster? Start with Step 1! 🚀**

