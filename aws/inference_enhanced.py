"""
SageMaker Inference Script for Enhanced XDR Threat Detection Model
Handles model loading, input processing, and prediction for the enhanced 79-feature 7-class model
with attention mechanisms and uncertainty quantification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import logging
from pathlib import Path
import joblib
import os

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class AttentionLayer(nn.Module):
    """Attention mechanism for feature relationship learning"""

    def __init__(self, input_dim: int, attention_dim: int = 64):
        super().__init__()
        self.attention_dim = attention_dim
        self.query = nn.Linear(input_dim, attention_dim)
        self.key = nn.Linear(input_dim, attention_dim)
        self.value = nn.Linear(input_dim, attention_dim)
        self.output = nn.Linear(attention_dim, input_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        batch_size = x.size(0)

        q = self.query(x).unsqueeze(1)
        k = self.key(x).unsqueeze(1)
        v = self.value(x).unsqueeze(1)

        attention_weights = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.attention_dim), dim=-1)
        attended = torch.matmul(attention_weights, v).squeeze(1)
        output = self.output(attended)
        output = self.dropout(output)

        return output + x  # Residual connection


class UncertaintyBlock(nn.Module):
    """Monte Carlo Dropout for uncertainty estimation"""

    def __init__(self, input_dim: int, output_dim: int, dropout_rate: float = 0.3):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.batch_norm = nn.BatchNorm1d(output_dim)

    def forward(self, x):
        x = self.linear(x)
        x = self.batch_norm(x)
        x = F.relu(x)
        x = self.dropout(x)
        return x


class EnhancedXDRThreatDetector(nn.Module):
    """
    Enhanced threat detection model with:
    - Attention mechanisms
    - Uncertainty quantification
    - Skip connections
    - Feature interactions
    """

    def __init__(self, input_dim: int = 79, hidden_dims: list = [512, 256, 128, 64],
                 num_classes: int = 7, dropout_rate: float = 0.3, use_attention: bool = True):
        super().__init__()

        self.input_dim = input_dim
        self.num_classes = num_classes
        self.use_attention = use_attention

        # Feature interaction layer
        self.feature_interaction = nn.Linear(input_dim, input_dim)

        # Attention mechanism
        if use_attention:
            self.attention = AttentionLayer(input_dim, attention_dim=64)

        # Enhanced feature extraction layers
        layers = []
        prev_dim = input_dim

        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(UncertaintyBlock(prev_dim, hidden_dim, dropout_rate))
            prev_dim = hidden_dim

        self.feature_extractor = nn.ModuleList(layers)

        # Skip connections
        self.skip_connections = nn.ModuleList([
            nn.Linear(input_dim, hidden_dims[-1]),
            nn.Linear(hidden_dims[0], hidden_dims[-1])
        ])

        # Classification head
        self.classifier = nn.Linear(prev_dim, num_classes)

        # Uncertainty estimation head
        self.uncertainty_head = nn.Linear(prev_dim, 1)

        # Dropout for uncertainty
        self.mc_dropout = nn.Dropout(dropout_rate)

    def forward(self, x, return_features: bool = False):
        batch_size = x.size(0)

        # Feature interactions
        x_interact = torch.relu(self.feature_interaction(x))
        x = x + x_interact  # Residual

        # Attention mechanism
        if self.use_attention:
            x_attended = self.attention(x)
            x = x_attended

        # Store for skip connections
        x_input = x
        x_mid = None

        # Feature extraction with uncertainty blocks
        for i, layer in enumerate(self.feature_extractor):
            x = layer(x)
            if i == 0:
                x_mid = x

        # Skip connections
        skip1 = torch.relu(self.skip_connections[0](x_input))
        skip2 = torch.relu(self.skip_connections[1](x_mid))
        x = x + skip1 + skip2

        # Classification and uncertainty
        features = x
        logits = self.classifier(features)
        uncertainty = torch.sigmoid(self.uncertainty_head(self.mc_dropout(features)))

        if return_features:
            return logits, uncertainty, features

        return logits, uncertainty


def model_fn(model_dir):
    """
    Load the PyTorch model for SageMaker inference
    This function is called by SageMaker to load the model
    """
    logger.info(f"Loading enhanced model from {model_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    try:
        # Load model metadata
        metadata_path = os.path.join(model_dir, "model_metadata.json")
        metadata = {}
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            logger.info(f"Loaded model metadata: {metadata}")
        else:
            logger.warning("No metadata found, using defaults")
            metadata = {
                "features": 79,
                "num_classes": 7,
                "hidden_dims": [512, 256, 128, 64],
                "dropout_rate": 0.3,
                "use_attention": True
            }

        # Try different possible model file names
        model_files = [
            "enhanced_threat_detector.pth",
            "threat_detector.pth",
            "model.pth",
            "pytorch_model.bin"
        ]

        model_path = None
        for filename in model_files:
            path = os.path.join(model_dir, filename)
            if os.path.exists(path):
                model_path = path
                logger.info(f"Found model file: {filename}")
                break

        if model_path is None:
            # List all files in model_dir for debugging
            files = os.listdir(model_dir)
            logger.error(f"No model file found in {model_dir}. Available files: {files}")
            raise FileNotFoundError(f"No model file found in {model_dir}")

        # Create model architecture matching training
        features = metadata.get('features', 79)
        num_classes = metadata.get('num_classes', 7)
        hidden_dims = metadata.get('hidden_dims', [512, 256, 128, 64])
        dropout_rate = metadata.get('dropout_rate', 0.3)
        use_attention = metadata.get('use_attention', True)

        model = EnhancedXDRThreatDetector(
            input_dim=features,
            hidden_dims=hidden_dims,
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            use_attention=use_attention
        ).to(device)

        # Load trained weights
        checkpoint = torch.load(model_path, map_location=device)

        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                # Assume the dict itself is the state dict
                model.load_state_dict(checkpoint)
        else:
            # Direct state dict
            model.load_state_dict(checkpoint)

        model.eval()

        logger.info(f"Successfully loaded enhanced threat detection model")
        logger.info(f"Model config: {features} features, {num_classes} classes, attention={use_attention}")

        # Load scaler for feature preprocessing
        scaler = None
        scaler_path = os.path.join(model_dir, "scaler.pkl")
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            logger.info("Loaded feature scaler")
        else:
            logger.warning("No scaler found - features should be pre-scaled")

        # Return model and preprocessing components as a dict
        return {
            'model': model,
            'scaler': scaler,
            'device': device,
            'metadata': metadata
        }

    except Exception as e:
        logger.error(f"Failed to load model: {e}", exc_info=True)
        raise


def input_fn(request_body, request_content_type):
    """
    Process input data before prediction
    Expected format: {"instances": [[79 float values]]}
    """
    logger.info(f"Processing input with content type: {request_content_type}")

    try:
        if request_content_type == 'application/json':
            input_data = json.loads(request_body)

            # Extract instances from the input
            if 'instances' in input_data:
                instances = input_data['instances']
            else:
                # Fallback: assume the request body is directly the instances
                instances = input_data

            # Convert to numpy array
            if isinstance(instances, list):
                data = np.array(instances, dtype=np.float32)
            else:
                raise ValueError("Input instances must be a list")

            # Validate input shape
            if data.ndim == 1:
                data = data.reshape(1, -1)  # Single instance

            expected_features = 79
            if data.shape[1] != expected_features:
                raise ValueError(f"Expected {expected_features} features, got {data.shape[1]}")

            logger.info(f"Processed input shape: {data.shape}")
            return data

        else:
            raise ValueError(f"Unsupported content type: {request_content_type}")

    except Exception as e:
        logger.error(f"Input processing failed: {e}")
        raise


def predict_fn(input_data, model_dict):
    """
    Make predictions using the loaded model with uncertainty quantification
    """
    logger.info(f"Making prediction for input shape: {input_data.shape}")

    try:
        model = model_dict['model']
        scaler = model_dict['scaler']
        device = model_dict['device']

        # Apply feature scaling if available
        if scaler is not None:
            input_data = scaler.transform(input_data)
            logger.info("Applied feature scaling")
        else:
            logger.info("No scaler - using pre-normalized data directly")

        # Convert to tensor
        input_tensor = torch.tensor(input_data, dtype=torch.float32).to(device)

        # Make prediction
        with torch.no_grad():
            logits, uncertainty = model(input_tensor)

            # Apply softmax to get class probabilities
            probabilities = torch.softmax(logits, dim=1)

            # Convert to numpy for JSON serialization
            predictions = probabilities.cpu().numpy().tolist()
            uncertainties = uncertainty.cpu().numpy().tolist()

        logger.info(f"Generated predictions for {len(predictions)} instances")

        # Return predictions with uncertainty scores
        results = []
        for i, (pred, unc) in enumerate(zip(predictions, uncertainties)):
            results.append({
                "probabilities": pred,
                "uncertainty": unc[0] if isinstance(unc, list) else unc,
                "predicted_class": int(np.argmax(pred)),
                "confidence": float(max(pred))
            })

        return results

    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise


def output_fn(predictions, accept):
    """
    Format output for SageMaker response
    """
    logger.info(f"Formatting output with accept type: {accept}")

    try:
        if accept == 'application/json':
            # Return predictions in the format expected by the client
            # Flatten for backward compatibility if needed
            if predictions and isinstance(predictions[0], dict):
                # New format with uncertainty
                response = {"predictions": [p["probabilities"] for p in predictions]}
            else:
                # Fallback to simple format
                response = {"predictions": predictions}

            return json.dumps(response)
        else:
            raise ValueError(f"Unsupported accept type: {accept}")

    except Exception as e:
        logger.error(f"Output formatting failed: {e}")
        raise


# Optional: Handler for direct testing
if __name__ == "__main__":
    # Test the inference functions locally
    print("Testing inference functions locally...")

    # Test model loading
    model_dict = model_fn(".")
    print(f"Model loaded successfully: {model_dict['metadata']}")

    # Test input processing
    test_input = json.dumps({"instances": [[0.1] * 79]})
    processed_input = input_fn(test_input, "application/json")
    print(f"Input processed: {processed_input.shape}")

    # Test prediction
    predictions = predict_fn(processed_input, model_dict)
    print(f"Predictions: {predictions}")

    # Test output formatting
    output = output_fn(predictions, "application/json")
    print(f"Output: {output}")