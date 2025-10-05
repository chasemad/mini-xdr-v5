#!/bin/bash
# Quick-start script for enhanced training with full dataset
# Uses ALL real data (4M+ samples) + synthetic supplement

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  Mini-XDR Enhanced ML Training - Full Dataset                 ║"
echo "║  Real Data: 4M+ samples | Synthetic: 10% supplement           ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo

# Check if we're in the right directory
if [ ! -d "datasets" ]; then
    echo "❌ Error: datasets directory not found"
    echo "   Please run this script from /Users/chasemad/Desktop/mini-xdr"
    exit 1
fi

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: python3 not found"
    exit 1
fi

# Check dependencies
echo "📦 Checking dependencies..."
python3 -c "import torch, numpy, pandas, sklearn" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ Missing dependencies. Installing..."
    pip3 install torch numpy pandas scikit-learn joblib
fi

# Check hardware
echo "🔍 Detecting hardware..."
python3 -c "
import torch
if torch.cuda.is_available():
    print('   ✅ CUDA GPU detected')
elif torch.backends.mps.is_available():
    print('   ✅ Apple Silicon GPU detected (MPS)')
else:
    print('   ⚠️  Using CPU (training will be slower)')
"

# Show dataset info
echo
echo "📊 Dataset Summary:"
echo "   Current: 1.6M samples (only partial CICIDS2017)"
echo "   Available: 4M+ samples (ALL real data)"
echo "   Enhancement: Adding synthetic data as 10% supplement"
echo

# Training options
echo "🎯 Training Configuration:"
echo "   Models: General + 3 Specialists (DDoS, BruteForce, WebAttacks)"
echo "   Epochs: 50 (vs 30 before)"
echo "   Batch Size: 256 (vs 512 before - better generalization)"
echo "   Learning Rate: 0.0005 (vs 0.001 before - more stable)"
echo "   Improvements:"
echo "     ✅ Focal loss for class imbalance"
echo "     ✅ Data augmentation (Gaussian noise + feature dropout)"
echo "     ✅ Cosine annealing LR schedule"
echo "     ✅ Gradient clipping"
echo "     ✅ Early stopping (patience=15)"
echo

# Ask for confirmation
read -p "🚀 Start training? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Training cancelled."
    exit 0
fi

# Start training
echo
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  Starting Enhanced Training...                                 ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo

START_TIME=$(date +%s)

python3 aws/train_enhanced_full_dataset.py \
    --data-dir datasets \
    --models general ddos brute_force web_attacks \
    --epochs 50 \
    --batch-size 256 \
    --learning-rate 0.0005 \
    --patience 15 \
    --synthetic-ratio 0.1

TRAINING_EXIT_CODE=$?

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
MINUTES=$((DURATION / 60))
SECONDS=$((DURATION % 60))

echo
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  Training Complete!                                            ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo
echo "⏱️  Total time: ${MINUTES}m ${SECONDS}s"
echo
echo "📁 Models saved to: models/local_trained_enhanced/"
echo "📊 Training logs: training_enhanced.log"
echo

if [ $TRAINING_EXIT_CODE -eq 0 ]; then
    echo "✅ Training successful!"
    echo
    echo "Next steps:"
    echo "  1. Review results: cat models/local_trained_enhanced/training_summary.json"
    echo "  2. Compare to old models: python3 compare_models.py"
    echo "  3. Test new models: python3 test_backend_integration_formats.py"
    echo "  4. Deploy if better: cp -r models/local_trained_enhanced/* models/local_trained/"
else
    echo "❌ Training failed with exit code $TRAINING_EXIT_CODE"
    echo "   Check training_enhanced.log for details"
fi


