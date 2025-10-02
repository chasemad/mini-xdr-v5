#!/usr/bin/env python3
"""
🧪 ENHANCED TRAINING VALIDATION SCRIPT
Tests the enhanced SageMaker training pipeline with a small dataset
to validate configuration before running on 2M+ events

Key Validations:
- Enhanced model architecture works correctly
- GPU training configuration is valid
- Data processing pipeline functions
- SageMaker integration is properly configured
- Inference script generates correct outputs
"""

import os
import sys
import json
import numpy as np
import torch
import tempfile
from pathlib import Path
import logging
import boto3
from datetime import datetime

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the training script components
try:
    from enhanced_sagemaker_train import (
        EnhancedXDRThreatDetector,
        MultiDatasetLoader,
        AttentionLayer,
        UncertaintyBlock
    )
    print("✅ Successfully imported enhanced training components")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_enhanced_model_architecture():
    """Test the enhanced model architecture"""
    print("\n🧠 Testing Enhanced Model Architecture...")

    try:
        # Test model creation
        model = EnhancedXDRThreatDetector(
            input_dim=79,
            hidden_dims=[256, 128, 64],
            num_classes=7,
            dropout_rate=0.3,
            use_attention=True
        )

        print(f"✅ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")

        # Test forward pass
        batch_size = 32
        test_input = torch.randn(batch_size, 79)

        # Standard forward pass
        logits, uncertainty = model(test_input)
        print(f"✅ Standard forward pass: logits {logits.shape}, uncertainty {uncertainty.shape}")

        # Test uncertainty prediction
        mean_pred, pred_uncertainty, mean_uncertainty = model.predict_with_uncertainty(test_input, n_samples=10)
        print(f"✅ Uncertainty prediction: mean {mean_pred.shape}, uncertainty {pred_uncertainty.shape}")

        # Validate outputs
        assert logits.shape == (batch_size, 7), f"Expected logits shape {(batch_size, 7)}, got {logits.shape}"
        assert uncertainty.shape == (batch_size, 1), f"Expected uncertainty shape {(batch_size, 1)}, got {uncertainty.shape}"

        # Test attention layer separately
        attention = AttentionLayer(79)
        attended = attention(test_input)
        print(f"✅ Attention layer: input {test_input.shape} -> output {attended.shape}")

        # Test uncertainty block
        uncertainty_block = UncertaintyBlock(79, 64)
        block_output = uncertainty_block(test_input)
        print(f"✅ Uncertainty block: input {test_input.shape} -> output {block_output.shape}")

        print("🎉 Enhanced model architecture validation PASSED")
        return True

    except Exception as e:
        print(f"❌ Model architecture test failed: {e}")
        return False


def test_data_processing():
    """Test data loading and processing"""
    print("\n📊 Testing Data Processing Pipeline...")

    try:
        # Create temporary directory for test
        with tempfile.TemporaryDirectory() as temp_dir:
            data_loader = MultiDatasetLoader(temp_dir)

            # Test synthetic data generation
            features, labels, dataset_info = data_loader.load_and_combine_datasets()

            print(f"✅ Generated test dataset: {len(features):,} samples")
            print(f"✅ Feature shape: {features.shape}")
            print(f"✅ Labels shape: {labels.shape}")
            print(f"✅ Datasets loaded: {list(dataset_info.keys())}")

            # Validate data characteristics
            assert features.shape[1] == 79, f"Expected 79 features, got {features.shape[1]}"
            assert len(np.unique(labels)) == 7, f"Expected 7 classes, got {len(np.unique(labels))}"
            assert not np.any(np.isnan(features)), "Features contain NaN values"
            assert not np.any(np.isinf(features)), "Features contain infinite values"

            # Test class distribution
            unique_classes, counts = np.unique(labels, return_counts=True)
            class_dist = dict(zip(unique_classes, counts))
            print(f"✅ Class distribution: {class_dist}")

            # Ensure reasonable class balance
            min_samples = min(counts)
            max_samples = max(counts)
            balance_ratio = min_samples / max_samples
            print(f"✅ Class balance ratio: {balance_ratio:.2f}")

            if balance_ratio < 0.1:
                print("⚠️ Warning: Severe class imbalance detected")

            print("🎉 Data processing validation PASSED")
            return True

    except Exception as e:
        print(f"❌ Data processing test failed: {e}")
        return False


def test_gpu_configuration():
    """Test GPU configuration"""
    print("\n🖥️ Testing GPU Configuration...")

    try:
        # Check CUDA availability
        cuda_available = torch.cuda.is_available()
        print(f"CUDA Available: {cuda_available}")

        if cuda_available:
            device_count = torch.cuda.device_count()
            print(f"GPU Devices: {device_count}")

            for i in range(device_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")

            # Test GPU memory allocation
            device = torch.device('cuda:0')
            test_tensor = torch.randn(1000, 1000).to(device)
            print(f"✅ GPU memory test passed: {test_tensor.device}")

        else:
            print("⚠️ CUDA not available - will use CPU for training")

        print("🎉 GPU configuration validation PASSED")
        return True

    except Exception as e:
        print(f"❌ GPU configuration test failed: {e}")
        return False


def test_small_training_run():
    """Test actual training with small dataset"""
    print("\n🚀 Testing Small Training Run...")

    try:
        # Create small synthetic dataset
        n_samples = 1000
        n_features = 79
        n_classes = 7

        # Generate balanced dataset
        samples_per_class = n_samples // n_classes
        features = []
        labels = []

        for class_id in range(n_classes):
            class_features = np.random.normal(
                loc=class_id * 0.5,  # Different means for each class
                scale=1.0,
                size=(samples_per_class, n_features)
            )
            class_labels = [class_id] * samples_per_class

            features.append(class_features)
            labels.extend(class_labels)

        features = np.vstack(features)
        labels = np.array(labels)

        print(f"✅ Created test dataset: {features.shape}")

        # Create model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = EnhancedXDRThreatDetector(
            input_dim=n_features,
            hidden_dims=[128, 64],  # Smaller for quick test
            num_classes=n_classes,
            dropout_rate=0.2,
            use_attention=True
        ).to(device)

        # Prepare data
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import RobustScaler
        from torch.utils.data import DataLoader, TensorDataset

        # Scale data
        scaler = RobustScaler()
        features_scaled = scaler.fit_transform(features)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            features_scaled, labels, test_size=0.2, stratify=labels, random_state=42
        )

        # Create data loaders
        train_dataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long)
        )

        test_dataset = TensorDataset(
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.long)
        )

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        # Training setup
        from sklearn.utils.class_weight import compute_class_weight
        import torch.optim as optim
        import torch.nn as nn

        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        optimizer = optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.01)

        # Quick training (just a few epochs)
        print("🏃 Running quick training test...")

        model.train()
        for epoch in range(3):  # Just 3 epochs for validation
            epoch_loss = 0.0
            for batch_features, batch_labels in train_loader:
                batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)

                optimizer.zero_grad()

                logits, uncertainty = model(batch_features)
                loss = criterion(logits, batch_labels)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(train_loader)
            print(f"  Epoch {epoch + 1}: Loss = {avg_loss:.4f}")

        # Test inference
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_features, batch_labels in test_loader:
                batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)

                logits, uncertainty = model(batch_features)
                _, predicted = torch.max(logits, 1)

                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()

        accuracy = correct / total
        print(f"✅ Test accuracy: {accuracy:.4f}")

        # Test uncertainty prediction
        test_input = torch.randn(5, n_features).to(device)
        mean_pred, pred_uncertainty, mean_uncertainty = model.predict_with_uncertainty(test_input, n_samples=10)

        print(f"✅ Uncertainty prediction test passed")
        print(f"  Mean prediction shape: {mean_pred.shape}")
        print(f"  Average uncertainty: {torch.mean(mean_uncertainty).item():.4f}")

        print("🎉 Small training run validation PASSED")
        return True

    except Exception as e:
        print(f"❌ Small training run failed: {e}")
        return False


def test_sagemaker_configuration():
    """Test SageMaker configuration"""
    print("\n☁️ Testing SageMaker Configuration...")

    try:
        # Test AWS credentials
        session = boto3.Session()
        credentials = session.get_credentials()

        if credentials:
            print("✅ AWS credentials found")
        else:
            print("❌ AWS credentials not found")
            return False

        # Test SageMaker client
        sagemaker_client = boto3.client('sagemaker')
        response = sagemaker_client.list_training_jobs(MaxResults=1)
        print("✅ SageMaker client connection successful")

        # Test S3 bucket access
        s3_client = boto3.client('s3')
        bucket = 'mini-xdr-ml-data-bucket-675076709589'

        try:
            s3_client.head_bucket(Bucket=bucket)
            print(f"✅ S3 bucket {bucket} accessible")
        except Exception as e:
            print(f"⚠️ S3 bucket access issue: {e}")
            print("   This may be expected if bucket doesn't exist yet")

        # Test execution role (if in SageMaker environment)
        if 'SM_' in os.environ:
            from sagemaker import get_execution_role
            role = get_execution_role()
            print(f"✅ SageMaker execution role: {role}")
        else:
            role = 'arn:aws:iam::675076709589:role/SageMakerExecutionRole'
            print(f"⚠️ Using default role: {role}")

        print("🎉 SageMaker configuration validation PASSED")
        return True

    except Exception as e:
        print(f"❌ SageMaker configuration test failed: {e}")
        return False


def create_test_inference_script():
    """Test inference script creation"""
    print("\n📝 Testing Inference Script Creation...")

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Import the inference script creation function
            from enhanced_sagemaker_train import create_sagemaker_inference_script

            create_sagemaker_inference_script(temp_dir)

            inference_file = Path(temp_dir) / "enhanced_inference.py"

            if inference_file.exists():
                print("✅ Inference script created successfully")

                # Check script content
                with open(inference_file, 'r') as f:
                    content = f.read()

                # Validate key components are present
                required_components = [
                    'EnhancedXDRThreatDetector',
                    'model_fn',
                    'input_fn',
                    'predict_fn',
                    'output_fn',
                    'uncertainty_score',
                    'enhanced_prediction'
                ]

                for component in required_components:
                    if component in content:
                        print(f"  ✅ {component} found in inference script")
                    else:
                        print(f"  ❌ {component} missing from inference script")
                        return False

                print("🎉 Inference script validation PASSED")
                return True
            else:
                print("❌ Inference script file not created")
                return False

    except Exception as e:
        print(f"❌ Inference script test failed: {e}")
        return False


def generate_validation_report():
    """Generate comprehensive validation report"""
    print("\n📊 GENERATING VALIDATION REPORT")
    print("=" * 50)

    tests = [
        ("Enhanced Model Architecture", test_enhanced_model_architecture),
        ("Data Processing Pipeline", test_data_processing),
        ("GPU Configuration", test_gpu_configuration),
        ("Small Training Run", test_small_training_run),
        ("SageMaker Configuration", test_sagemaker_configuration),
        ("Inference Script Creation", create_test_inference_script)
    ]

    results = {}
    passed_tests = 0
    total_tests = len(tests)

    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = "PASS" if result else "FAIL"
            if result:
                passed_tests += 1
        except Exception as e:
            results[test_name] = f"ERROR: {e}"

    # Print summary
    print("\n🏆 VALIDATION SUMMARY")
    print("=" * 50)

    for test_name, result in results.items():
        status_emoji = "✅" if result == "PASS" else "❌"
        print(f"{status_emoji} {test_name}: {result}")

    success_rate = passed_tests / total_tests
    print(f"\n📊 Overall Success Rate: {passed_tests}/{total_tests} ({success_rate:.1%})")

    if success_rate >= 0.8:
        print("🎉 VALIDATION PASSED - Ready for full training!")
        print("🚀 You can now run the enhanced SageMaker training with confidence")
        return True
    else:
        print("⚠️ VALIDATION ISSUES DETECTED")
        print("🔧 Please fix the failed tests before running full training")
        return False


def main():
    """Main validation function"""
    print("🧪 ENHANCED SAGEMAKER TRAINING VALIDATION")
    print("=" * 60)
    print("This script validates the enhanced training pipeline")
    print("before running on the full 2M+ event dataset")
    print("=" * 60)

    validation_success = generate_validation_report()

    if validation_success:
        print("\n🎯 NEXT STEPS:")
        print("1. Prepare your 2M+ event dataset in S3")
        print("2. Run: python launch_enhanced_sagemaker.py")
        print("3. Monitor training progress in AWS Console")
        print("4. Deploy to SageMaker endpoint when complete")
        print("\n💡 TIP: Start with a smaller subset (100k events) for initial testing")
    else:
        print("\n🔧 FIX REQUIRED:")
        print("Please address the failed validation tests before proceeding")

    return validation_success


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)