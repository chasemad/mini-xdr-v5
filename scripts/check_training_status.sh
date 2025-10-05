#!/bin/bash
# Quick training status checker

echo "======================================================================"
echo "Mini-XDR Training Status"
echo "======================================================================"
echo ""

# Check if training is running
if ps aux | grep -v grep | grep "train_local.py" > /dev/null; then
    echo "✅ Training is RUNNING"
    echo ""
    
    # Show current progress
    echo "Last 15 lines of training log:"
    echo "----------------------------------------------------------------------"
    tail -15 training.log
    echo "----------------------------------------------------------------------"
    echo ""
    
    # Show which model is training
    if grep -q "TRAINING GENERAL MODEL" training.log | tail -1; then
        current_model=$(grep "TRAINING.*MODEL" training.log | tail -1 | awk '{print $2}')
        echo "Currently training: $current_model model"
    fi
    
    # Show progress
    epochs_done=$(grep -c "Epoch \[" training.log)
    echo "Training epochs completed so far: $epochs_done"
    
else
    echo "⚠️  Training is NOT running"
    echo ""
    
    # Check if it completed
    if grep -q "✅ All models trained successfully!" training.log 2>/dev/null; then
        echo "✅ Training COMPLETED successfully!"
        echo ""
        echo "Results:"
        tail -30 training.log | grep -E "(general|ddos|brute_force|web_attacks).*Accuracy"
        echo ""
        echo "Models saved to: models/local_trained/"
        echo ""
        echo "Next steps:"
        echo "  1. Test models: python3 aws/local_inference.py"
        echo "  2. Check results: cat models/local_trained/training_summary.json"
    else
        echo "❌ Training may have failed. Check training.log for errors:"
        echo ""
        tail -30 training.log
    fi
fi

echo ""
echo "======================================================================"
echo ""
echo "Commands:"
echo "  Watch live:     watch -n 2 tail -20 training.log"
echo "  Full log:       less training.log"
echo "  Stop training:  pkill -f train_local.py"
echo ""


