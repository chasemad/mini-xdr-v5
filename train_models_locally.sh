#!/bin/bash

# Quick Start: Train Mini-XDR ML Models Locally
# This script trains all 4 models (general + 3 specialists) on your machine

set -e

echo "======================================================================"
echo "Mini-XDR Local Model Training"
echo "======================================================================"
echo ""

# Check if training data exists
if [ ! -f "aws/training_data/training_features_20250929_062520.npy" ]; then
    echo "❌ Error: Training data not found!"
    echo "   Expected: aws/training_data/training_features_20250929_062520.npy"
    echo ""
    echo "   Please ensure training data is in aws/training_data/"
    exit 1
fi

echo "✅ Training data found (1.6M samples)"
echo ""

# Check Python environment
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: python3 not found"
    exit 1
fi

echo "Checking dependencies..."

# Check for required packages
python3 -c "import torch" 2>/dev/null || {
    echo "❌ PyTorch not found. Installing..."
    pip3 install torch torchvision torchaudio
}

python3 -c "import sklearn" 2>/dev/null || {
    echo "❌ scikit-learn not found. Installing..."
    pip3 install scikit-learn
}

python3 -c "import pandas" 2>/dev/null || {
    echo "❌ pandas not found. Installing..."
    pip3 install pandas
}

python3 -c "import numpy" 2>/dev/null || {
    echo "❌ numpy not found. Installing..."
    pip3 install numpy
}

echo "✅ All dependencies installed"
echo ""

# Detect hardware
echo "Detecting hardware..."
python3 << 'EOF'
import torch
import platform

print(f"OS: {platform.system()} {platform.release()}")
print(f"Python: {platform.python_version()}")
print(f"PyTorch: {torch.__version__}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)} (CUDA)")
    print("⚡ Training will use GPU acceleration")
elif torch.backends.mps.is_available():
    print("GPU: Apple Silicon (MPS)")
    print("⚡ Training will use GPU acceleration")
else:
    print("CPU: Will use CPU (slower)")
    print("⏱️  Estimated time: 1-2 hours per model")
EOF

echo ""
echo "======================================================================"
echo "Starting Training"
echo "======================================================================"
echo ""
echo "Training 4 models:"
echo "  1. General (7-class): Normal, DDoS, Recon, BruteForce, WebAttack, Malware, APT"
echo "  2. DDoS Specialist (binary)"
echo "  3. Brute Force Specialist (binary)"
echo "  4. Web Attack Specialist (binary)"
echo ""
echo "Output directory: models/local_trained/"
echo ""

# Prompt user
read -p "Continue with training? (y/n) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Training cancelled."
    exit 0
fi

# Run training
echo ""
echo "🚀 Starting training... (this may take a while)"
echo ""

python3 aws/train_local.py \
    --data-dir aws/training_data \
    --output-dir models/local_trained \
    --models general ddos brute_force web_attacks \
    --epochs 30 \
    --batch-size 512 \
    --learning-rate 0.001 \
    --patience 10 \
    --device auto

echo ""
echo "======================================================================"
echo "✅ Training Complete!"
echo "======================================================================"
echo ""
echo "Models saved to: models/local_trained/"
echo ""
echo "Next steps:"
echo "  1. Test models: python3 aws/local_inference.py"
echo "  2. Update backend to use local models (see INTEGRATION.md)"
echo "  3. Deploy to production"
echo ""
echo "Model files:"
find models/local_trained -name "threat_detector.pth" -o -name "model_metadata.json"
echo ""


