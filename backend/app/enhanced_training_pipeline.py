"""
🚀 STRATEGIC ML TRAINING ENHANCEMENT PIPELINE
Focus: Quality over Quantity for optimal threat detection

Key Strategies:
1. Hard Example Mining: Focus on challenging cases
2. Class Balance Optimization: Ensure proper representation
3. Adversarial Training: Robust against evasion
4. Active Learning: Train on most informative examples
5. Data Quality: Clean, validate, and enhance existing data
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime, timezone
import joblib
from pathlib import Path
import asyncio
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight
import json

from .enhanced_threat_detector import EnhancedXDRThreatDetector, FeatureEnhancer
from .models import Event
from .deep_learning_models import deep_learning_manager
from sqlalchemy import select, and_, func
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


class StrategicDataEnhancer:
    """
    Strategic data enhancement focused on quality improvements:
    - Hard example mining
    - Data cleaning and validation
    - Synthetic hard examples generation
    - Class imbalance correction
    """

    def __init__(self):
        self.noise_threshold = 0.1
        self.hard_example_threshold = 0.7  # Confidence threshold for hard examples

    async def enhance_training_data(
        self,
        db: AsyncSession,
        existing_model: EnhancedXDRThreatDetector = None,
        max_samples_per_class: int = 10000
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Strategic data enhancement pipeline
        Returns: features, labels, enhancement_stats
        """

        logger.info("Starting strategic data enhancement pipeline...")

        # Step 1: Extract high-quality base dataset
        features, labels = await self._extract_base_dataset(db, max_samples_per_class)

        # Step 2: Clean and validate data
        features, labels, cleaning_stats = self._clean_and_validate_data(features, labels)

        # Step 3: Identify hard examples using existing model
        hard_examples = None
        if existing_model:
            hard_examples = self._identify_hard_examples(features, labels, existing_model)

        # Step 4: Balance classes strategically
        features, labels, balance_stats = self._strategic_class_balancing(
            features, labels, hard_examples
        )

        # Step 5: Generate synthetic hard examples
        synthetic_features, synthetic_labels = self._generate_synthetic_hard_examples(
            features, labels, num_synthetic=min(1000, len(features) // 10)
        )

        # Combine original and synthetic data
        if len(synthetic_features) > 0:
            features = np.vstack([features, synthetic_features])
            labels = np.concatenate([labels, synthetic_labels])

        # Step 6: Final quality validation
        features, labels = self._final_quality_check(features, labels)

        enhancement_stats = {
            "original_samples": cleaning_stats["original_samples"],
            "cleaned_samples": cleaning_stats["cleaned_samples"],
            "hard_examples_identified": len(hard_examples) if hard_examples is not None else 0,
            "synthetic_examples_added": len(synthetic_features),
            "final_samples": len(features),
            "class_distribution": dict(zip(*np.unique(labels, return_counts=True))),
            "quality_score": self._calculate_quality_score(features, labels),
            "cleaning_stats": cleaning_stats,
            "balance_stats": balance_stats
        }

        logger.info(f"Data enhancement completed: {enhancement_stats}")
        return features, labels, enhancement_stats

    async def _extract_base_dataset(
        self,
        db: AsyncSession,
        max_samples_per_class: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract high-quality base dataset from events"""

        logger.info("Extracting base dataset from events...")

        # Get diverse events from different time periods and IPs
        query = """
        WITH ranked_events AS (
            SELECT *,
                   ROW_NUMBER() OVER (
                       PARTITION BY src_ip, DATE(ts)
                       ORDER BY ts DESC
                   ) as day_rank
            FROM events
            WHERE ts > NOW() - INTERVAL '30 days'
              AND src_ip IS NOT NULL
              AND eventid IS NOT NULL
        )
        SELECT src_ip, eventid, ts, message, raw, dst_port
        FROM ranked_events
        WHERE day_rank <= 5  -- Max 5 events per IP per day for diversity
        ORDER BY ts DESC
        LIMIT 100000
        """

        result = await db.execute(query)
        events = result.fetchall()

        # Group events by source IP for feature extraction
        events_by_ip = {}
        for event in events:
            src_ip = event.src_ip
            if src_ip not in events_by_ip:
                events_by_ip[src_ip] = []

            # Convert to Event object
            event_obj = Event(
                src_ip=event.src_ip,
                dst_port=event.dst_port,
                eventid=event.eventid,
                message=event.message,
                raw=event.raw,
                ts=event.ts
            )
            events_by_ip[src_ip].append(event_obj)

        # Extract features for each IP
        all_features = []
        all_labels = []

        for src_ip, ip_events in events_by_ip.items():
            try:
                # Extract 79-dimensional features
                feature_dict = deep_learning_manager._extract_features(src_ip, ip_events)
                feature_vector = list(feature_dict.values())

                # Ensure exactly 79 features
                if len(feature_vector) != 79:
                    feature_vector = feature_vector[:79] + [0.0] * (79 - len(feature_vector))

                # Generate label based on event characteristics
                label = self._generate_ground_truth_label(ip_events)

                all_features.append(feature_vector)
                all_labels.append(label)

            except Exception as e:
                logger.warning(f"Failed to extract features for {src_ip}: {e}")
                continue

        features = np.array(all_features, dtype=np.float32)
        labels = np.array(all_labels, dtype=np.int64)

        logger.info(f"Extracted base dataset: {len(features)} samples, {len(np.unique(labels))} classes")
        return features, labels

    def _generate_ground_truth_label(self, events: List[Event]) -> int:
        """Generate ground truth labels based on event characteristics"""

        # Analyze event patterns to determine threat type
        failed_logins = len([e for e in events if "failed" in e.eventid.lower()])
        success_logins = len([e for e in events if "success" in e.eventid.lower()])
        unique_ports = len(set(e.dst_port for e in events if e.dst_port))
        event_rate = len(events) / max((events[-1].ts - events[0].ts).total_seconds() / 60, 1) if len(events) > 1 else 0

        # Decision rules based on domain expertise
        if failed_logins >= 10 and success_logins == 0:
            return 3  # Brute Force Attack

        if unique_ports >= 5 and event_rate > 2:
            return 2  # Network Reconnaissance

        if event_rate > 10:
            return 1  # DDoS/DoS Attack

        # Check for web attack patterns
        web_attacks = 0
        for event in events:
            if event.message:
                message = event.message.lower()
                if any(pattern in message for pattern in ['sql', 'xss', 'injection', '<script>']):
                    web_attacks += 1

        if web_attacks >= 2:
            return 4  # Web Application Attack

        # Check for malware indicators
        malware_indicators = 0
        for event in events:
            if event.message:
                message = event.message.lower()
                if any(pattern in message for pattern in ['malware', 'virus', 'trojan', 'backdoor']):
                    malware_indicators += 1

        if malware_indicators >= 1:
            return 5  # Malware/Botnet

        # Check for APT-like behavior (sophisticated, persistent)
        if (len(events) > 20 and
            unique_ports >= 3 and
            event_rate < 1 and  # Slow and stealthy
            len(set(e.eventid for e in events)) >= 5):  # Diverse tactics
            return 6  # Advanced Persistent Threat

        return 0  # Normal

    def _clean_and_validate_data(
        self,
        features: np.ndarray,
        labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Clean and validate training data"""

        original_samples = len(features)
        logger.info(f"Starting data cleaning on {original_samples} samples...")

        # Remove samples with NaN or infinite values
        valid_mask = np.isfinite(features).all(axis=1)
        features = features[valid_mask]
        labels = labels[valid_mask]

        # Remove duplicate samples
        unique_indices = []
        seen_features = set()
        for i, feature_vector in enumerate(features):
            feature_hash = hash(tuple(feature_vector.round(6)))
            if feature_hash not in seen_features:
                seen_features.add(feature_hash)
                unique_indices.append(i)

        features = features[unique_indices]
        labels = labels[unique_indices]

        # Remove extreme outliers (features beyond 5 standard deviations)
        z_scores = np.abs((features - np.mean(features, axis=0)) / (np.std(features, axis=0) + 1e-8))
        outlier_mask = (z_scores < 5).all(axis=1)
        features = features[outlier_mask]
        labels = labels[outlier_mask]

        # Validate label consistency
        valid_labels = (labels >= 0) & (labels < 7)
        features = features[valid_labels]
        labels = labels[valid_labels]

        cleaned_samples = len(features)
        cleaning_stats = {
            "original_samples": original_samples,
            "cleaned_samples": cleaned_samples,
            "removed_invalid": original_samples - cleaned_samples,
            "cleaning_ratio": cleaned_samples / max(original_samples, 1)
        }

        logger.info(f"Data cleaning completed: {cleaned_samples}/{original_samples} samples retained")
        return features, labels, cleaning_stats

    def _identify_hard_examples(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        model: EnhancedXDRThreatDetector
    ) -> List[int]:
        """Identify hard examples using existing model predictions"""

        logger.info("Identifying hard examples...")

        model.eval()
        hard_examples = []

        with torch.no_grad():
            # Process in batches
            batch_size = 256
            for i in range(0, len(features), batch_size):
                batch_features = features[i:i + batch_size]
                batch_labels = labels[i:i + batch_size]

                # Convert to tensor
                input_tensor = torch.tensor(batch_features, dtype=torch.float32)

                # Get predictions with uncertainty
                probabilities, _, uncertainty = model.predict_with_uncertainty(input_tensor, n_samples=10)

                # Identify hard examples
                for j, (prob, true_label, unc) in enumerate(zip(probabilities, batch_labels, uncertainty)):
                    predicted_class = torch.argmax(prob).item()
                    confidence = prob[true_label].item()

                    # Hard example criteria
                    if (confidence < self.hard_example_threshold or  # Low confidence on true label
                            predicted_class != true_label or  # Misclassified
                            torch.mean(unc).item() > 0.3):  # High uncertainty
                        hard_examples.append(i + j)

        logger.info(f"Identified {len(hard_examples)} hard examples ({len(hard_examples) / len(features) * 100:.1f}%)")
        return hard_examples

    def _strategic_class_balancing(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        hard_examples: List[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Strategic class balancing that preserves hard examples"""

        logger.info("Performing strategic class balancing...")

        unique_classes, class_counts = np.unique(labels, return_counts=True)
        target_samples = min(max(class_counts) // 2, 5000)  # Target balanced size

        balanced_features = []
        balanced_labels = []
        balance_stats = {}

        for class_label in unique_classes:
            class_mask = labels == class_label
            class_indices = np.where(class_mask)[0]

            # Prioritize hard examples for this class
            if hard_examples:
                class_hard_examples = [idx for idx in hard_examples if idx in class_indices]
                class_easy_examples = [idx for idx in class_indices if idx not in class_hard_examples]
            else:
                class_hard_examples = []
                class_easy_examples = class_indices.tolist()

            # Sample strategy: keep all hard examples + sample easy examples
            selected_indices = class_hard_examples.copy()

            # Add easy examples to reach target
            remaining_needed = target_samples - len(selected_indices)
            if remaining_needed > 0 and class_easy_examples:
                if len(class_easy_examples) <= remaining_needed:
                    selected_indices.extend(class_easy_examples)
                else:
                    # Randomly sample easy examples
                    np.random.seed(42)
                    sampled_easy = np.random.choice(class_easy_examples, remaining_needed, replace=False)
                    selected_indices.extend(sampled_easy.tolist())

            # Add to balanced dataset
            balanced_features.append(features[selected_indices])
            balanced_labels.extend([class_label] * len(selected_indices))

            balance_stats[f"class_{class_label}"] = {
                "original": len(class_indices),
                "hard_examples": len(class_hard_examples),
                "selected": len(selected_indices)
            }

        balanced_features = np.vstack(balanced_features)
        balanced_labels = np.array(balanced_labels)

        logger.info(f"Class balancing completed: {len(balanced_features)} samples")
        return balanced_features, balanced_labels, balance_stats

    def _generate_synthetic_hard_examples(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        num_synthetic: int = 1000
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic hard examples using adversarial techniques"""

        if num_synthetic == 0:
            return np.array([]), np.array([])

        logger.info(f"Generating {num_synthetic} synthetic hard examples...")

        synthetic_features = []
        synthetic_labels = []

        # For each class, generate synthetic examples
        unique_classes = np.unique(labels)
        samples_per_class = num_synthetic // len(unique_classes)

        for class_label in unique_classes:
            class_mask = labels == class_label
            class_features = features[class_mask]

            if len(class_features) < 2:
                continue

            for _ in range(samples_per_class):
                # SMOTE-like generation: interpolate between existing samples
                idx1, idx2 = np.random.choice(len(class_features), 2, replace=False)
                sample1, sample2 = class_features[idx1], class_features[idx2]

                # Generate synthetic sample
                alpha = np.random.random()
                synthetic_sample = alpha * sample1 + (1 - alpha) * sample2

                # Add small amount of noise for diversity
                noise = np.random.normal(0, 0.01, size=synthetic_sample.shape)
                synthetic_sample += noise

                synthetic_features.append(synthetic_sample)
                synthetic_labels.append(class_label)

        synthetic_features = np.array(synthetic_features)
        synthetic_labels = np.array(synthetic_labels)

        logger.info(f"Generated {len(synthetic_features)} synthetic examples")
        return synthetic_features, synthetic_labels

    def _final_quality_check(self, features: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Final quality validation"""

        # Remove any remaining invalid samples
        valid_mask = np.isfinite(features).all(axis=1) & (labels >= 0) & (labels < 7)
        features = features[valid_mask]
        labels = labels[valid_mask]

        # Ensure reasonable feature ranges
        features = np.clip(features, -10, 10)

        return features, labels

    def _calculate_quality_score(self, features: np.ndarray, labels: np.ndarray) -> float:
        """Calculate overall dataset quality score"""

        try:
            # Class balance score
            unique_classes, class_counts = np.unique(labels, return_counts=True)
            balance_score = 1.0 - np.std(class_counts) / np.mean(class_counts)

            # Feature completeness score
            completeness_score = 1.0 - np.mean(np.isnan(features))

            # Feature variance score (features should have reasonable variance)
            variance_score = np.mean([
                1.0 if 0.01 < np.var(features[:, i]) < 100 else 0.5
                for i in range(features.shape[1])
            ])

            quality_score = (balance_score + completeness_score + variance_score) / 3.0
            return min(max(quality_score, 0.0), 1.0)

        except Exception as e:
            logger.error(f"Quality score calculation failed: {e}")
            return 0.5


class EnhancedTrainingPipeline:
    """Complete training pipeline for enhanced threat detection model"""

    def __init__(self):
        self.data_enhancer = StrategicDataEnhancer()
        self.feature_enhancer = FeatureEnhancer()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    async def train_enhanced_model(
        self,
        db: AsyncSession,
        model_save_path: str,
        existing_model_path: str = None
    ) -> Dict[str, Any]:
        """Complete enhanced model training pipeline"""

        logger.info("Starting enhanced model training pipeline...")

        # Load existing model if available
        existing_model = None
        if existing_model_path and Path(existing_model_path).exists():
            existing_model = EnhancedXDRThreatDetector()
            try:
                state_dict = torch.load(existing_model_path, map_location=self.device)
                existing_model.load_state_dict(state_dict)
                existing_model.eval()
                logger.info("Loaded existing model for hard example mining")
            except Exception as e:
                logger.warning(f"Failed to load existing model: {e}")
                existing_model = None

        # Step 1: Strategic data enhancement
        features, labels, enhancement_stats = await self.data_enhancer.enhance_training_data(
            db, existing_model, max_samples_per_class=5000
        )

        # Step 2: Feature enhancement
        enhanced_features = self.feature_enhancer.enhance_features(features, labels, fit=True)

        # Step 3: Train-validation split
        X_train, X_val, y_train, y_val = train_test_split(
            enhanced_features, labels, test_size=0.2, stratify=labels, random_state=42
        )

        # Step 4: Train enhanced model
        model = EnhancedXDRThreatDetector(
            input_dim=enhanced_features.shape[1],
            hidden_dims=[512, 256, 128, 64],
            num_classes=len(np.unique(labels)),
            dropout_rate=0.3,
            use_attention=True
        ).to(self.device)

        training_stats = self._train_model(
            model, X_train, y_train, X_val, y_val
        )

        # Step 5: Save model and components
        save_path = Path(model_save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save enhanced model
        torch.save(model.state_dict(), save_path / "enhanced_threat_detector.pth")

        # Save feature enhancer
        enhancer_data = {
            'important_features': self.feature_enhancer.important_features,
            'scaler': self.feature_enhancer.scaler
        }
        joblib.dump(enhancer_data, save_path / "feature_enhancer.pkl")

        # Save metadata
        metadata = {
            "model_type": "EnhancedXDRThreatDetector",
            "features": enhanced_features.shape[1],
            "num_classes": len(np.unique(labels)),
            "training_samples": len(X_train),
            "validation_samples": len(X_val),
            "enhancement_stats": enhancement_stats,
            "training_stats": training_stats,
            "timestamp": datetime.now().isoformat()
        }

        with open(save_path / "enhanced_model_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        results = {
            "success": True,
            "model_path": str(save_path),
            "training_samples": len(X_train),
            "validation_accuracy": training_stats["best_val_accuracy"],
            "final_loss": training_stats["final_loss"],
            "enhancement_stats": enhancement_stats,
            "training_stats": training_stats
        }

        logger.info(f"Enhanced model training completed successfully: {results}")
        return results

    def _train_model(
        self,
        model: EnhancedXDRThreatDetector,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> Dict[str, Any]:
        """Train the enhanced model"""

        logger.info("Starting model training...")

        # Calculate class weights for imbalanced classes
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(self.device)

        # Create data loaders
        train_dataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long)
        )

        val_dataset = TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.long)
        )

        # Weighted sampling for balanced training
        sample_weights = [class_weights[label] for label in y_train]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

        train_loader = DataLoader(train_dataset, batch_size=128, sampler=sampler)
        val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

        # Loss function and optimizer
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

        # Training loop
        best_val_accuracy = 0.0
        patience_counter = 0
        max_patience = 15
        num_epochs = 100

        training_history = {
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": []
        }

        for epoch in range(num_epochs):
            # Training phase
            model.train()
            train_loss = 0.0

            for batch_features, batch_labels in train_loader:
                batch_features, batch_labels = batch_features.to(self.device), batch_labels.to(self.device)

                optimizer.zero_grad()

                # Forward pass
                logits, uncertainty = model(batch_features)

                # Main classification loss
                class_loss = criterion(logits, batch_labels)

                # Uncertainty regularization (encourage calibrated uncertainty)
                uncertainty_loss = torch.mean(uncertainty)

                # Combined loss
                total_loss = class_loss + 0.1 * uncertainty_loss

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                train_loss += total_loss.item()

            # Validation phase
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0

            with torch.no_grad():
                for batch_features, batch_labels in val_loader:
                    batch_features, batch_labels = batch_features.to(self.device), batch_labels.to(self.device)

                    logits, uncertainty = model(batch_features)
                    loss = criterion(logits, batch_labels)

                    val_loss += loss.item()

                    # Accuracy calculation
                    _, predicted = torch.max(logits, 1)
                    total += batch_labels.size(0)
                    correct += (predicted == batch_labels).sum().item()

            # Calculate metrics
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            val_accuracy = correct / total

            training_history["train_loss"].append(avg_train_loss)
            training_history["val_loss"].append(avg_val_loss)
            training_history["val_accuracy"].append(val_accuracy)

            # Learning rate scheduling
            scheduler.step(avg_val_loss)

            # Early stopping
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= max_patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch + 1}/{num_epochs}: "
                    f"Train Loss: {avg_train_loss:.4f}, "
                    f"Val Loss: {avg_val_loss:.4f}, "
                    f"Val Acc: {val_accuracy:.4f}"
                )

        # Final validation metrics
        model.eval()
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch_features, batch_labels in val_loader:
                batch_features = batch_features.to(self.device)

                logits, _ = model(batch_features)
                predictions = torch.argmax(logits, dim=1).cpu().numpy()

                all_predictions.extend(predictions)
                all_labels.extend(batch_labels.numpy())

        # Calculate detailed metrics
        f1_macro = f1_score(all_labels, all_predictions, average='macro')
        f1_weighted = f1_score(all_labels, all_predictions, average='weighted')

        training_stats = {
            "best_val_accuracy": best_val_accuracy,
            "final_loss": training_history["val_loss"][-1],
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            "epochs_trained": len(training_history["train_loss"]),
            "training_history": training_history
        }

        logger.info(f"Training completed - Best Val Accuracy: {best_val_accuracy:.4f}, F1 Macro: {f1_macro:.4f}")
        return training_stats


# Global training pipeline instance
enhanced_training_pipeline = EnhancedTrainingPipeline()