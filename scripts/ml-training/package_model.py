#!/usr/bin/env python3
"""
Package ML models and inference code for SageMaker deployment
"""

import tarfile
import os
import shutil
import json
from pathlib import Path

def create_model_package():
    """Create model.tar.gz package for SageMaker"""

    # Create temporary directory structure
    temp_dir = Path("/tmp/mini-xdr-model")
    temp_dir.mkdir(exist_ok=True)

    # Create code directory
    code_dir = temp_dir / "code"
    code_dir.mkdir(exist_ok=True)

    # Copy inference script
    inference_src = Path("/Users/chasemad/Desktop/mini-xdr/scripts/ml-training/inference.py")
    inference_dst = code_dir / "inference.py"
    shutil.copy2(inference_src, inference_dst)

    # Create requirements.txt
    requirements = """
torch>=2.0.0
scikit-learn>=1.3.0
numpy>=1.21.0
"""

    with open(code_dir / "requirements.txt", "w") as f:
        f.write(requirements.strip())

    # Create dummy model files (will be replaced with actual trained models)
    models_dir = temp_dir / "models"
    models_dir.mkdir(exist_ok=True)

    # Create placeholder files
    with open(models_dir / "model_info.json", "w") as f:
        json.dump({
            "model_type": "threat_detection",
            "version": "1.0.0",
            "created_at": "2025-09-27",
            "features": 79,
            "models": ["isolation_forest", "lstm_classifier"]
        }, f, indent=2)

    # Create the tar.gz archive
    output_path = Path("/Users/chasemad/Desktop/mini-xdr/models/model.tar.gz")
    output_path.parent.mkdir(exist_ok=True)

    with tarfile.open(output_path, "w:gz") as tar:
        tar.add(temp_dir, arcname=".")

    # Clean up temp directory
    shutil.rmtree(temp_dir)

    print(f"✅ Created model package: {output_path}")
    return output_path

if __name__ == "__main__":
    create_model_package()