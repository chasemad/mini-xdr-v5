# 🧠 Deprecated ML Training Scripts

**Status**: ⚠️ DEPRECATED - Use proper deep learning scripts instead

## Scripts in this folder:

### `pytorch_train.py`
- **Purpose**: "PyTorch" training (actually traditional ML)
- **Issues**: 
  - Used scikit-learn Isolation Forest, not neural networks
  - Only trained on 200K samples (9% of data)
  - No GPU acceleration despite PyTorch name
- **Deprecated**: September 2025
- **Replacement**: `../../pytorch_deep_learning_train.py`

## What was wrong:
```python
# Old "pytorch_train.py" - NOT actually PyTorch!
from sklearn.ensemble import IsolationForest  # Traditional ML
model = IsolationForest()  # No neural networks
```

## Proper replacement:
```python
# New pytorch_deep_learning_train.py - REAL PyTorch!
import torch.nn as nn
model = XDRThreatDetector()  # Actual neural networks
model.to(device)  # GPU acceleration
```

## Migration:
```bash
# Old (fake PyTorch):
python pytorch_train.py  # 6 min, 200K samples, traditional ML

# New (real deep learning):
python pytorch_deep_learning_train.py  # 45 min, 2.2M samples, neural nets
```
