#!/bin/bash
#
# Master Training Script - Train All Models with Proper Scaler Saving
# This script trains the general model + all specialist models
#

set -e  # Exit on error

echo "=========================================="
echo "🚀 MINI-XDR MODEL TRAINING SUITE"
echo "=========================================="
echo ""

# Configuration
DATA_DIR="/tmp/training_data"
OUTPUT_BASE="/tmp/models"
TRAINING_SCRIPT="/Users/chasemad/Desktop/mini-xdr/aws/train_specialist_model.py"

# Check if data exists
if [ ! -d "$DATA_DIR" ] || [ -z "$(ls -A $DATA_DIR/*.csv 2>/dev/null)" ]; then
    echo "❌ Training data not found in $DATA_DIR"
    echo "   Downloading data from S3..."
    mkdir -p "$DATA_DIR"
    aws s3 sync s3://mini-xdr-ml-data-bucket-675076709589/data/comprehensive-train/ "$DATA_DIR/" \
        --exclude "*" --include "train_chunk_*.csv"
fi

echo "✅ Training data ready:"
ls -lh "$DATA_DIR"/*.csv | wc -l | xargs echo "   Files:"
du -sh "$DATA_DIR" | awk '{print "   Size: " $1}'
echo ""

# Function to train a model
train_model() {
    local model_type=$1
    local model_name=$2
    local epochs=${3:-50}

    echo "=========================================="
    echo "🎯 Training: $model_name"
    echo "=========================================="
    echo "Type: $model_type"
    echo "Epochs: $epochs"
    echo "Data: $DATA_DIR"
    echo ""

    output_dir="$OUTPUT_BASE/$model_type"
    mkdir -p "$output_dir"

    echo "⏱️  Start time: $(date)"

    python3 "$TRAINING_SCRIPT" \
        --data-dir "$DATA_DIR" \
        --output-dir "$output_dir" \
        --specialist-type "$model_type" \
        --epochs "$epochs" \
        --batch-size 256 \
        --learning-rate 0.001 \
        --patience 15 \
        --hidden-dims "512,256,128,64" \
        --dropout-rate 0.3

    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ $model_name training completed successfully!"
        echo "   Output: $output_dir"
        echo "   Files:"
        ls -lh "$output_dir"
        echo ""
    else
        echo ""
        echo "❌ $model_name training failed!"
        echo ""
        return 1
    fi

    echo "⏱️  End time: $(date)"
    echo ""
}

# Create output directory
mkdir -p "$OUTPUT_BASE"

# Training sequence
echo "📋 Training Plan:"
echo "   1. General Purpose Model (7 classes)"
echo "   2. DDoS/DoS Specialist (binary)"
echo "   3. Brute Force Specialist (binary)"
echo "   4. Web Attack Specialist (binary)"
echo ""
read -p "Press Enter to start training (or Ctrl+C to cancel)..."
echo ""

# 1. Train General Purpose Model (fixes scaler issue)
train_model "general" "General Purpose Model" 50

# 2. Train DDoS/DoS Specialist
train_model "ddos" "DDoS/DoS Specialist" 40

# 3. Train Brute Force Specialist
train_model "brute_force" "Brute Force Specialist" 40

# 4. Train Web Attack Specialist
train_model "web_attacks" "Web Attack Specialist" 40

echo ""
echo "=========================================="
echo "✅ ALL MODELS TRAINED SUCCESSFULLY!"
echo "=========================================="
echo ""
echo "📊 Summary:"
echo "   General Purpose Model: $OUTPUT_BASE/general/"
echo "   DDoS/DoS Specialist:   $OUTPUT_BASE/ddos/"
echo "   Brute Force Specialist: $OUTPUT_BASE/brute_force/"
echo "   Web Attack Specialist:  $OUTPUT_BASE/web_attacks/"
echo ""
echo "📦 Next steps:"
echo "   1. Package models for SageMaker"
echo "   2. Deploy to endpoints"
echo "   3. Test with end-to-end simulation"
echo ""
echo "🎉 Training complete!"