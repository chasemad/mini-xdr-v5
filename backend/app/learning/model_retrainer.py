"""
Model Retrainer - Automated Model Retraining Pipeline

This module implements the core retraining logic that uses Council-corrected
samples to continuously improve ML models. It integrates all Phase 2 enhancements:
- Data balancing (SMOTE/ADASYN)
- Weighted loss functions (Focal Loss)
- Threshold optimization
- Model validation and deployment

Flow:
1. Load training samples from training_collector
2. Balance dataset using SMOTE/ADASYN
3. Split into train/validation sets
4. Train models with weighted loss
5. Calibrate probabilities (temperature scaling)
6. Optimize per-class thresholds
7. Validate performance (must exceed baseline)
8. Save and deploy new models
9. Mark samples as used
10. Update metrics

Goal: Improve accuracy from 72.7% → 85%+ through continuous learning.

Usage:
```python
from app.learning import model_retrainer

# Trigger retraining (typically called by scheduler)
result = await model_retrainer.retrain_models(job_id="manual_20250121")

# Check results
print(f"Success: {result['success']}")
print(f"New Accuracy: {result['new_accuracy']:.2%}")
print(f"Improvement: {result['improvement']:.2%}")
```
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sqlalchemy.ext.asyncio import AsyncSession
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


class ModelRetrainer:
    """
    Automated model retraining pipeline.

    This class orchestrates the entire retraining process, from loading
    training data to deploying improved models.
    """

    def __init__(
        self,
        models_path: str = "./models",
        min_accuracy_improvement: float = 0.01,  # 1% minimum improvement
        validation_split: float = 0.2,
        batch_size: int = 128,
        epochs: int = 20,
        learning_rate: float = 0.001,
        early_stopping_patience: int = 5,
    ):
        """
        Initialize model retrainer.

        Args:
            models_path: Path to save retrained models
            min_accuracy_improvement: Minimum accuracy improvement to deploy
            validation_split: Fraction of data for validation
            batch_size: Training batch size
            epochs: Maximum training epochs
            learning_rate: Learning rate for optimizer
            early_stopping_patience: Epochs to wait for improvement
        """
        self.models_path = Path(models_path)
        self.models_path.mkdir(parents=True, exist_ok=True)

        self.min_accuracy_improvement = min_accuracy_improvement
        self.validation_split = validation_split
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.early_stopping_patience = early_stopping_patience

        # Retraining history
        self.history = []

        logger.info(
            f"ModelRetrainer initialized: "
            f"models_path={models_path}, "
            f"min_improvement={min_accuracy_improvement:.1%}"
        )

    async def retrain_models(
        self,
        job_id: Optional[str] = None,
        force_deploy: bool = False,
    ) -> Dict[str, Any]:
        """
        Execute complete retraining pipeline.

        Args:
            job_id: Unique job identifier
            force_deploy: Deploy even if improvement < threshold

        Returns:
            Dictionary with retraining results
        """
        if job_id is None:
            job_id = f"retrain_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

        start_time = datetime.now(timezone.utc)
        logger.info(f"Starting retraining job: {job_id}")

        result = {
            "job_id": job_id,
            "success": False,
            "start_time": start_time.isoformat(),
            "end_time": None,
            "duration_seconds": None,
            "baseline_accuracy": None,
            "new_accuracy": None,
            "improvement": None,
            "deployed": False,
            "error": None,
        }

        try:
            # Step 1: Load training data
            logger.info(f"[{job_id}] Step 1/9: Loading training data...")
            X, y, db_session = await self._load_training_data()

            if len(X) == 0:
                raise ValueError("No training data available")

            logger.info(
                f"[{job_id}] Loaded {len(X)} samples, " f"features shape: {X.shape}"
            )

            # Step 2: Balance dataset
            logger.info(f"[{job_id}] Step 2/9: Balancing dataset...")
            X_balanced, y_balanced = await self._balance_dataset(X, y)

            logger.info(
                f"[{job_id}] Balanced to {len(X_balanced)} samples, "
                f"class distribution: {np.bincount(y_balanced)}"
            )

            # Step 3: Split into train/validation
            logger.info(f"[{job_id}] Step 3/9: Splitting data...")
            X_train, X_val, y_train, y_val = train_test_split(
                X_balanced,
                y_balanced,
                test_size=self.validation_split,
                random_state=42,
                stratify=y_balanced,
            )

            logger.info(f"[{job_id}] Train: {len(X_train)}, Val: {len(X_val)}")

            # Step 4: Get baseline accuracy (current model)
            logger.info(f"[{job_id}] Step 4/9: Measuring baseline accuracy...")
            baseline_accuracy = await self._get_baseline_accuracy(X_val, y_val)
            result["baseline_accuracy"] = baseline_accuracy

            logger.info(f"[{job_id}] Baseline accuracy: {baseline_accuracy:.2%}")

            # Step 5: Train new model
            logger.info(f"[{job_id}] Step 5/9: Training new model...")
            new_model, training_stats = await self._train_model(
                X_train, y_train, X_val, y_val
            )

            logger.info(
                f"[{job_id}] Training complete: "
                f"best_val_accuracy={training_stats['best_accuracy']:.2%}, "
                f"epochs={training_stats['epochs_trained']}"
            )

            # Step 6: Calibrate probabilities
            logger.info(f"[{job_id}] Step 6/9: Calibrating probabilities...")
            temperature = await self._calibrate_probabilities(new_model, X_val, y_val)

            logger.info(f"[{job_id}] Optimal temperature: {temperature:.3f}")

            # Step 7: Optimize thresholds
            logger.info(f"[{job_id}] Step 7/9: Optimizing thresholds...")
            thresholds = await self._optimize_thresholds(
                new_model, X_val, y_val, temperature
            )

            logger.info(
                f"[{job_id}] Optimized thresholds: "
                f"{', '.join([f'{t:.3f}' for t in thresholds.values()])}"
            )

            # Step 8: Validate new model
            logger.info(f"[{job_id}] Step 8/9: Validating new model...")
            new_accuracy = await self._validate_model(
                new_model, X_val, y_val, temperature, thresholds
            )
            result["new_accuracy"] = new_accuracy

            improvement = new_accuracy - baseline_accuracy
            result["improvement"] = improvement

            logger.info(
                f"[{job_id}] New accuracy: {new_accuracy:.2%} "
                f"(improvement: {improvement:+.2%})"
            )

            # Step 9: Deploy if improvement meets threshold
            if improvement >= self.min_accuracy_improvement or force_deploy:
                logger.info(f"[{job_id}] Step 9/9: Deploying new model...")

                await self._deploy_model(new_model, temperature, thresholds, job_id)

                result["deployed"] = True
                logger.info(f"[{job_id}] Model deployed successfully")

            else:
                logger.warning(
                    f"[{job_id}] Improvement {improvement:.2%} below threshold "
                    f"{self.min_accuracy_improvement:.2%}, not deploying"
                )
                result["deployed"] = False

            # Mark samples as used
            if db_session:
                await self._mark_samples_used(db_session)

            # Success
            result["success"] = True

        except Exception as e:
            logger.error(f"[{job_id}] Retraining failed: {e}", exc_info=True)
            result["error"] = str(e)

        finally:
            end_time = datetime.now(timezone.utc)
            result["end_time"] = end_time.isoformat()
            result["duration_seconds"] = (end_time - start_time).total_seconds()

            # Save result to history
            self.history.append(result)
            await self._save_retraining_log(result)

            logger.info(
                f"[{job_id}] Retraining complete: "
                f"success={result['success']}, "
                f"duration={result['duration_seconds']:.1f}s"
            )

        return result

    async def _load_training_data(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[AsyncSession]]:
        """
        Load training data from training_collector.

        Returns:
            (X, y, db_session) tuple
        """
        from ..database import get_db_session
        from .training_collector import training_collector

        # Get database session
        db_session = None
        async with get_db_session() as db:
            # Load training data
            X, y = await training_collector.load_training_data(db=db)
            db_session = db

        return X, y, db_session

    async def _balance_dataset(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Balance dataset using SMOTE/ADASYN.

        Args:
            X: Feature matrix
            y: Labels

        Returns:
            (X_balanced, y_balanced) tuple
        """
        from .data_augmentation import balance_dataset

        X_balanced, y_balanced = balance_dataset(
            X,
            y,
            strategy="auto",  # Auto-select SMOTE or ADASYN
        )

        return X_balanced, y_balanced

    async def _get_baseline_accuracy(
        self, X_val: np.ndarray, y_val: np.ndarray
    ) -> float:
        """
        Get baseline accuracy from current production model.

        Args:
            X_val: Validation features
            y_val: Validation labels

        Returns:
            Baseline accuracy
        """
        try:
            from ..deep_learning_models import deep_learning_manager

            # Get current model
            model = deep_learning_manager.general_model

            if model is None:
                logger.warning("No baseline model available, using 0.727")
                return 0.727  # Known baseline from Phase 1

            # Evaluate
            model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X_val)
                y_tensor = torch.LongTensor(y_val)

                outputs = model(X_tensor)
                _, predictions = torch.max(outputs, 1)

                accuracy = (predictions == y_tensor).float().mean().item()

            return accuracy

        except Exception as e:
            logger.warning(f"Failed to get baseline accuracy: {e}")
            return 0.727  # Default to known baseline

    async def _train_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Tuple[nn.Module, Dict[str, Any]]:
        """
        Train new model with weighted loss.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels

        Returns:
            (model, training_stats) tuple
        """
        from ..deep_learning_models import ThreatClassifier
        from .weighted_loss import FocalLoss, calculate_class_weights

        # Calculate class weights
        class_weights = calculate_class_weights(y_train, method="inverse_frequency")

        # Create model
        n_features = X_train.shape[1]
        n_classes = len(np.unique(y_train))

        model = ThreatClassifier(
            input_size=n_features,
            num_classes=n_classes,
            hidden_sizes=[256, 128, 64],
            dropout=0.3,
        )

        # Create loss function (Focal Loss)
        criterion = FocalLoss(
            alpha=torch.FloatTensor(class_weights), gamma=2.0, label_smoothing=0.05
        )

        # Create optimizer
        optimizer = torch.optim.Adam(
            model.parameters(), lr=self.learning_rate, weight_decay=1e-5
        )

        # Create data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train), torch.LongTensor(y_train)
        )
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )

        val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        # Training loop
        best_accuracy = 0.0
        best_model_state = None
        epochs_without_improvement = 0

        training_history = {
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": [],
        }

        for epoch in range(self.epochs):
            # Training phase
            model.train()
            train_loss = 0.0

            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validation phase
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0

            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()

                    _, predictions = torch.max(outputs, 1)
                    correct += (predictions == batch_y).sum().item()
                    total += batch_y.size(0)

            val_loss /= len(val_loader)
            val_accuracy = correct / total

            training_history["train_loss"].append(train_loss)
            training_history["val_loss"].append(val_loss)
            training_history["val_accuracy"].append(val_accuracy)

            logger.debug(
                f"Epoch {epoch+1}/{self.epochs}: "
                f"train_loss={train_loss:.4f}, "
                f"val_loss={val_loss:.4f}, "
                f"val_accuracy={val_accuracy:.4f}"
            )

            # Save best model
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                best_model_state = model.state_dict().copy()
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            # Early stopping
            if epochs_without_improvement >= self.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

        # Restore best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        training_stats = {
            "best_accuracy": best_accuracy,
            "epochs_trained": epoch + 1,
            "history": training_history,
        }

        return model, training_stats

    async def _calibrate_probabilities(
        self, model: nn.Module, X_val: np.ndarray, y_val: np.ndarray
    ) -> float:
        """
        Calibrate model probabilities using temperature scaling.

        Args:
            model: Trained model
            X_val: Validation features
            y_val: Validation labels

        Returns:
            Optimal temperature
        """
        from .weighted_loss import TemperatureScaling

        model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_val)
            y_tensor = torch.LongTensor(y_val)
            logits = model(X_tensor)

        # Fit temperature scaling
        temp_scaler = TemperatureScaling()
        temp_scaler.fit(logits, y_tensor)

        return temp_scaler.temperature.item()

    async def _optimize_thresholds(
        self, model: nn.Module, X_val: np.ndarray, y_val: np.ndarray, temperature: float
    ) -> Dict[int, float]:
        """
        Optimize per-class thresholds.

        Args:
            model: Trained model
            X_val: Validation features
            y_val: Validation labels
            temperature: Calibration temperature

        Returns:
            Dictionary of optimal thresholds per class
        """
        from .threshold_optimizer import ThresholdOptimizer

        # Get calibrated probabilities
        model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_val)
            logits = model(X_tensor)

            # Apply temperature scaling
            scaled_logits = logits / temperature
            probabilities = torch.softmax(scaled_logits, dim=1)

        # Optimize thresholds
        optimizer = ThresholdOptimizer(metric="f1", search_method="grid")
        thresholds = optimizer.optimize(probabilities.numpy(), y_val)

        return thresholds

    async def _validate_model(
        self,
        model: nn.Module,
        X_val: np.ndarray,
        y_val: np.ndarray,
        temperature: float,
        thresholds: Dict[int, float],
    ) -> float:
        """
        Validate model with calibration and thresholds.

        Args:
            model: Trained model
            X_val: Validation features
            y_val: Validation labels
            temperature: Calibration temperature
            thresholds: Per-class thresholds

        Returns:
            Validation accuracy
        """
        model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_val)

            # Get calibrated probabilities
            logits = model(X_tensor)
            scaled_logits = logits / temperature
            probabilities = torch.softmax(scaled_logits, dim=1)

            # Apply per-class thresholds
            predictions = []
            for probs in probabilities:
                # Check each class against its threshold
                above_threshold = [
                    (i, probs[i].item())
                    for i in range(len(probs))
                    if probs[i].item() >= thresholds.get(i, 0.5)
                ]

                if above_threshold:
                    # Choose class with highest probability among those above threshold
                    pred_class = max(above_threshold, key=lambda x: x[1])[0]
                else:
                    # If none above threshold, choose highest probability
                    pred_class = torch.argmax(probs).item()

                predictions.append(pred_class)

            predictions = np.array(predictions)
            accuracy = (predictions == y_val).mean()

        # Log detailed metrics
        logger.info("\nValidation Metrics:")
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info("\nClassification Report:")
        logger.info(classification_report(y_val, predictions))

        return accuracy

    async def _deploy_model(
        self,
        model: nn.Module,
        temperature: float,
        thresholds: Dict[int, float],
        job_id: str,
    ):
        """
        Deploy new model to production.

        Args:
            model: Trained model
            temperature: Calibration temperature
            thresholds: Per-class thresholds
            job_id: Job identifier
        """
        # Save model
        model_filename = self.models_path / f"general_model_{job_id}.pt"
        torch.save(model.state_dict(), model_filename)

        # Save metadata
        metadata = {
            "job_id": job_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "temperature": temperature,
            "thresholds": {str(k): float(v) for k, v in thresholds.items()},
        }

        metadata_filename = self.models_path / f"general_model_{job_id}_metadata.json"
        with open(metadata_filename, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Model saved: {model_filename}")
        logger.info(f"Metadata saved: {metadata_filename}")

        # Update production model
        from ..deep_learning_models import deep_learning_manager

        deep_learning_manager.general_model = model
        deep_learning_manager.set_temperature(temperature)
        deep_learning_manager.set_per_class_thresholds(thresholds)

        logger.info("Production model updated")

    async def _mark_samples_used(self, db: AsyncSession):
        """Mark training samples as used."""
        from .training_collector import training_collector

        await training_collector.mark_samples_used(db)
        logger.info("Training samples marked as used")

    async def _save_retraining_log(self, result: Dict[str, Any]):
        """Save retraining log to file."""
        log_filename = self.models_path / f"retrain_log_{result['job_id']}.json"

        with open(log_filename, "w") as f:
            json.dump(result, f, indent=2)

        logger.info(f"Retraining log saved: {log_filename}")

    def get_history(self, limit: int = 10) -> list:
        """Get retraining history."""
        return self.history[-limit:]


# Global singleton instance
model_retrainer = ModelRetrainer()


__all__ = [
    "ModelRetrainer",
    "model_retrainer",
]
