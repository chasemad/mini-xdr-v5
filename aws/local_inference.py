#!/usr/bin/env python3
"""
Local Inference Client for Mini-XDR
Replaces SageMaker client for local model inference
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import time

logger = logging.getLogger(__name__)


# Model architecture (must match training)
class AttentionLayer(nn.Module):
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
        attention_weights = torch.softmax(
            torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.attention_dim), 
            dim=-1
        )
        attended = torch.matmul(attention_weights, v).squeeze(1)
        output = self.output(attended)
        output = self.dropout(output)
        return output + x


class UncertaintyBlock(nn.Module):
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


class ThreatDetector(nn.Module):
    def __init__(self, input_dim: int = 79, hidden_dims: list = [512, 256, 128, 64],
                 num_classes: int = 7, dropout_rate: float = 0.3, use_attention: bool = True):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.use_attention = use_attention

        self.feature_interaction = nn.Linear(input_dim, input_dim)

        if use_attention:
            self.attention = AttentionLayer(input_dim, attention_dim=64)

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(UncertaintyBlock(prev_dim, hidden_dim, dropout_rate))
            prev_dim = hidden_dim

        self.feature_extractor = nn.ModuleList(layers)

        self.skip_connections = nn.ModuleList([
            nn.Linear(input_dim, hidden_dims[-1]),
            nn.Linear(hidden_dims[0], hidden_dims[-1])
        ])

        self.classifier = nn.Linear(prev_dim, num_classes)
        self.uncertainty_head = nn.Linear(prev_dim, 1)
        self.mc_dropout = nn.Dropout(dropout_rate)

    def forward(self, x, return_features: bool = False):
        x_interact = torch.relu(self.feature_interaction(x))
        x = x + x_interact

        if self.use_attention:
            x_attended = self.attention(x)
            x = x_attended

        x_input = x
        x_mid = None

        for i, layer in enumerate(self.feature_extractor):
            x = layer(x)
            if i == 0:
                x_mid = x

        skip1 = torch.relu(self.skip_connections[0](x_input))
        skip2 = torch.relu(self.skip_connections[1](x_mid))
        x = x + skip1 + skip2

        features = x
        logits = self.classifier(features)
        uncertainty = torch.sigmoid(self.uncertainty_head(self.mc_dropout(features)))

        if return_features:
            return logits, uncertainty, features

        return logits, uncertainty


class LocalMLClient:
    """
    Local ML inference client - drop-in replacement for SageMaker client
    """
    
    def __init__(self, model_dir: str = "models/local_trained"):
        self.model_dir = Path(model_dir)
        self.device = self._get_device()
        self.models = {}
        self.metadata = {}
        
        # Load all available models
        self._load_models()
        
    def _get_device(self):
        """Auto-detect best available device"""
        if torch.cuda.is_available():
            device = torch.device('cuda')
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
            logger.info("Using Apple Silicon GPU (MPS)")
        else:
            device = torch.device('cpu')
            logger.info("Using CPU for inference")
        return device
    
    def _load_models(self):
        """Load all trained models"""
        model_types = ['general', 'ddos', 'brute_force', 'web_attacks']
        
        for model_type in model_types:
            model_path = self.model_dir / model_type / 'threat_detector.pth'
            metadata_path = self.model_dir / model_type / 'model_metadata.json'
            
            if model_path.exists() and metadata_path.exists():
                try:
                    # Load metadata
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    
                    # Create model
                    model = ThreatDetector(
                        input_dim=metadata['features'],
                        hidden_dims=metadata['hidden_dims'],
                        num_classes=metadata['num_classes'],
                        dropout_rate=metadata['dropout_rate'],
                        use_attention=metadata['use_attention']
                    ).to(self.device)
                    
                    # Load weights
                    model.load_state_dict(torch.load(model_path, map_location=self.device))
                    model.eval()
                    
                    self.models[model_type] = model
                    self.metadata[model_type] = metadata
                    
                    logger.info(f"✅ Loaded {model_type} model (accuracy: {metadata['best_val_accuracy']*100:.2f}%)")
                    
                except Exception as e:
                    logger.error(f"Failed to load {model_type} model: {e}")
        
        if not self.models:
            logger.warning("No models loaded! Run train_local.py first.")
    
    async def health_check(self) -> bool:
        """Check if models are loaded and ready"""
        return len(self.models) > 0
    
    def _extract_features_from_event(self, event: Dict) -> np.ndarray:
        """
        Extract 79 features from a network event
        This is a simplified version - adapt to your actual feature extraction
        """
        # Initialize 79 features with zeros
        features = np.zeros(79, dtype=np.float32)
        
        # Extract basic features from event
        # TODO: Implement proper feature extraction matching your training data
        # This is a placeholder - you'll need to adapt this to your actual features
        
        # Example features (indices 0-10):
        features[0] = hash(event.get('src_ip', '')) % 1000 / 1000.0  # Normalized IP hash
        features[1] = event.get('dst_port', 0) / 65535.0  # Normalized port
        features[2] = len(event.get('message', '')) / 100.0  # Message length
        features[3] = 1.0 if 'failed' in event.get('eventid', '') else 0.0
        features[4] = 1.0 if 'login' in event.get('eventid', '') else 0.0
        
        # Add more feature extraction as needed...
        
        return features
    
    async def detect_threats(self, events: List[Dict]) -> List[Dict]:
        """
        Detect threats from network events
        Compatible with SageMaker client interface
        """
        if not self.models:
            logger.warning("No models loaded, cannot perform detection")
            return []
        
        if not events:
            return []
        
        try:
            # Use general model for classification
            if 'general' not in self.models:
                logger.error("General model not loaded")
                return []
            
            model = self.models['general']
            class_names = self.metadata['general']['class_names']
            
            # Extract features from events
            features_list = []
            for event in events:
                # Check if features are pre-extracted (passed from backend)
                if 'features' in event and event['features'] is not None:
                    features = np.array(event['features'], dtype=np.float32)
                    if len(features) != 79:
                        logger.warning(f"Expected 79 features, got {len(features)}. Padding/truncating.")
                        if len(features) < 79:
                            features = np.pad(features, (0, 79 - len(features)), 'constant')
                        else:
                            features = features[:79]
                else:
                    # Fall back to simple extraction
                    features = self._extract_features_from_event(event)
                features_list.append(features)
            
            # Convert to tensor
            features_tensor = torch.FloatTensor(features_list).to(self.device)
            
            # Run inference
            with torch.no_grad():
                logits, uncertainty = model(features_tensor)
                probabilities = torch.softmax(logits, dim=1)
                predicted_classes = torch.argmax(logits, dim=1)
            
            # Format results
            results = []
            for i, event in enumerate(events):
                pred_class = predicted_classes[i].item()
                confidence = probabilities[i][pred_class].item()
                uncertainty_score = uncertainty[i].item()
                
                # ALWAYS check specialist models (they have 94-99% accuracy)
                # Don't rely only on general model (66% accuracy)
                specialist_scores = {}
                specialist_predictions = {}
                
                # Check all specialists
                if 'ddos' in self.models:
                    specialist_scores['ddos'] = self._get_specialist_score(
                        features_list[i], 'ddos'
                    )
                    specialist_predictions['ddos'] = specialist_scores['ddos']
                
                if 'brute_force' in self.models:
                    specialist_scores['brute_force'] = self._get_specialist_score(
                        features_list[i], 'brute_force'
                    )
                    specialist_predictions['brute_force'] = specialist_scores['brute_force']
                
                if 'web_attacks' in self.models:
                    specialist_scores['web_attacks'] = self._get_specialist_score(
                        features_list[i], 'web_attacks'
                    )
                    specialist_predictions['web_attacks'] = specialist_scores['web_attacks']
                
                # Use specialist scores to override general model if they detect an attack
                max_specialist_score = max(specialist_scores.values()) if specialist_scores else 0.0
                max_specialist_name = max(specialist_scores, key=specialist_scores.get) if specialist_scores else None
                
                # If any specialist has high confidence of attack, use that
                if max_specialist_score > 0.7:
                    # Override general model prediction
                    class_mapping = {
                        'ddos': 1,
                        'brute_force': 3,
                        'web_attacks': 4
                    }
                    pred_class = class_mapping.get(max_specialist_name, pred_class)
                    confidence = max_specialist_score
                    logger.info(f"Specialist override: {max_specialist_name} with {max_specialist_score:.3f} confidence")
                
                result = {
                    'event_id': event.get('id', i),
                    'src_ip': event.get('src_ip', 'unknown'),
                    'predicted_class': class_names[pred_class],
                    'predicted_class_id': pred_class,
                    'confidence': confidence,
                    'uncertainty': uncertainty_score,
                    'anomaly_score': 1.0 - confidence if pred_class != 0 else 0.0,
                    'probabilities': probabilities[i].cpu().numpy().tolist(),
                    'specialist_scores': specialist_scores,
                    'is_attack': pred_class != 0,
                    'threat_level': self._calculate_threat_level(confidence, pred_class)
                }
                
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _get_specialist_score(self, features: np.ndarray, specialist_type: str) -> float:
        """Get prediction from specialist model"""
        try:
            model = self.models[specialist_type]
            features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                logits, _ = model(features_tensor)
                probabilities = torch.softmax(logits, dim=1)
                # Return probability of attack (class 1)
                return probabilities[0][1].item()
        except Exception as e:
            logger.error(f"Specialist {specialist_type} inference failed: {e}")
            return 0.0
    
    def _calculate_threat_level(self, confidence: float, pred_class: int) -> str:
        """Calculate threat level based on confidence and class"""
        if pred_class == 0:  # Normal
            return "none"
        
        if confidence >= 0.9:
            return "critical"
        elif confidence >= 0.7:
            return "high"
        elif confidence >= 0.5:
            return "medium"
        else:
            return "low"
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get status of all loaded models"""
        status = {
            'models_loaded': len(self.models),
            'device': str(self.device),
            'models': {}
        }
        
        for model_type, metadata in self.metadata.items():
            status['models'][model_type] = {
                'accuracy': metadata.get('best_val_accuracy', 0),
                'num_classes': metadata.get('num_classes', 0),
                'training_date': metadata.get('training_date', 'unknown')
            }
        
        return status


# Global instance
local_ml_client = LocalMLClient()


async def test_local_inference():
    """Test the local inference client"""
    # Create some test events
    test_events = [
        {
            'id': 1,
            'src_ip': '192.168.1.100',
            'dst_port': 22,
            'eventid': 'cowrie.login.failed',
            'message': 'Failed login attempt',
            'timestamp': '2025-10-02T10:00:00Z'
        },
        {
            'id': 2,
            'src_ip': '10.0.0.50',
            'dst_port': 80,
            'eventid': 'http.request',
            'message': 'Normal HTTP request',
            'timestamp': '2025-10-02T10:00:01Z'
        }
    ]
    
    client = LocalMLClient()
    
    # Check health
    is_healthy = await client.health_check()
    print(f"Client healthy: {is_healthy}")
    
    if is_healthy:
        # Get status
        status = client.get_model_status()
        print(f"\nModel Status:")
        print(json.dumps(status, indent=2))
        
        # Run inference
        print(f"\nRunning inference on {len(test_events)} events...")
        results = await client.detect_threats(test_events)
        
        print(f"\nResults:")
        for result in results:
            print(f"  Event {result['event_id']}: {result['predicted_class']} "
                  f"(confidence: {result['confidence']:.3f}, "
                  f"threat: {result['threat_level']})")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_local_inference())

