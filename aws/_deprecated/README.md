# 🗄️ Deprecated AWS Scripts

**⚠️ These scripts are no longer actively used but preserved for reference.**

## 📁 Organization

### `deployment-scripts/`
Legacy deployment and infrastructure setup scripts replaced by `start-mini-xdr-aws-v2.sh`:

- **`deploy-automated-production.sh`** - Old automated production deployment
- **`deploy-complete-aws-ml-system.sh`** - Legacy complete ML system deployment
- **`deploy-mini-xdr-code.sh`** - Old code deployment approach
- **`deploy-secure-mini-xdr.sh`** - Legacy secure deployment
- **`deploy-secure-ml-production.sh`** - Old secure ML production deployment

**Replacement**: `../start-mini-xdr-aws-v2.sh` handles all deployment needs

### `ml-training-legacy/`
Old ML training approaches replaced by proper PyTorch deep learning:

- **`pytorch_train.py`** - "Fake" PyTorch script (actually traditional ML)

**Replacement**: `../pytorch_deep_learning_train.py` for real deep learning

### `utility-scripts/`
*(Currently empty - utilities moved to `../utils/` and actively maintained)*

## 🚀 Current Active Scripts

**In parent `aws/` directory:**
- `start-mini-xdr-aws-v2.sh` - Main infrastructure control
- `pytorch_deep_learning_train.py` - Deep learning training
- `pytorch_sequential_train.py` - Sequential/temporal models
- `launch-deep-learning-gpu.py` - Training launcher
- `monitor-training.py` - Training monitor

## 📝 Migration Notes

- All functionality from deprecated scripts has been consolidated
- Current scripts provide better error handling, security, and features
- These scripts are kept for reference and emergency fallback only

**Last Updated**: September 27, 2025