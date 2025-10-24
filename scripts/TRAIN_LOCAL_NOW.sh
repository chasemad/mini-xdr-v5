#!/bin/bash
# Quick Local Training - Start Immediately
# Uses all your 4M+ events + Windows data

echo "🚀 Starting Local Mini-XDR Training"
echo "===================================="
echo ""
echo "📊 Training Data:"
echo "   - 4M+ existing events (CICIDS2017, UNSW-NB15, KDD)"
echo "   - 8,000+ Windows/AD attacks (Mordor, EVTX, OpTC, APT29)"
echo "   - Total: 4,008,000+ samples"
echo ""
echo "🧠 Model: 13-class threat detector"
echo "⏱️  Time: 4-8 hours (overnight training)"
echo "💰 Cost: FREE (local compute)"
echo ""
echo "Starting in 5 seconds... (Ctrl+C to cancel)"
sleep 5

cd /Users/chasemad/Desktop/mini-xdr

# Activate virtual environment
source ml-training-env/bin/activate

# Run training
python3 aws/train_enhanced_full_dataset.py \
  --data datasets/real_datasets \
  --output models/enterprise \
  --epochs 50 \
  --batch-size 256 \
  --learning-rate 0.001

echo ""
echo "🎉 Training complete!"
echo "📊 Check results in: models/enterprise/"

