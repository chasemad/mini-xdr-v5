#!/usr/bin/env python3
"""
Fix the scaler by creating one from the training data
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

def create_scaler_from_training_data():
    """Create scaler from the comprehensive training data"""
    print("🔧 Creating scaler from training data...")

    # Load a sample of training data to fit scaler
    try:
        # Load first training chunk
        chunk_path = "/tmp/claude/train_chunk_000.csv"
        df = pd.read_csv(chunk_path, header=None)
        print(f"✅ Loaded training data: {df.shape}")

        # Features are all columns except last (which is labels)
        X = df.iloc[:, :-1].values  # 79 features
        print(f"📊 Features shape: {X.shape}")

        # Create and fit scaler
        scaler = StandardScaler()
        scaler.fit(X)
        print(f"✅ Scaler fitted with {X.shape[0]:,} samples")

        # Save the proper scaler
        scaler_path = "/Users/chasemad/Desktop/mini-xdr/models/scaler.pkl"
        joblib.dump(scaler, scaler_path)
        print(f"💾 Saved scaler to {scaler_path}")

        # Test the scaler
        test_sample = X[0:1]  # First sample
        scaled = scaler.transform(test_sample)
        print(f"🧪 Test transform - Original range: [{X[0].min():.3f}, {X[0].max():.3f}]")
        print(f"🧪 Test transform - Scaled range: [{scaled[0].min():.3f}, {scaled[0].max():.3f}]")

        return True

    except Exception as e:
        print(f"❌ Failed to create scaler: {e}")
        return False

if __name__ == "__main__":
    create_scaler_from_training_data()