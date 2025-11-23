"""
Training Data Collector - Continuous Learning from Council Corrections

This module collects incidents with Council verdicts and stores them as labeled
training data for model retraining. This enables the self-improving ML system.

Flow:
1. Incident detected → ML prediction (72.7% accuracy)
2. Council verification → Gemini Judge provides correct label
3. TrainingCollector stores (features, correct_label, ml_prediction)
4. When 1000+ samples collected → Trigger retraining
5. Retrained model deployed → Higher accuracy → Fewer Council calls

Goal: Improve 72.7% → 85%+ through continuous learning from Council corrections.

Usage:
```python
from app.learning import training_collector

# After Council verdict
await training_collector.collect_sample(
    features=feature_vector,
    ml_prediction="Brute Force Attack",
    council_verdict="FALSE_POSITIVE",
    correct_label="Normal",
    incident_id=incident.id
)

# Check if retraining needed
if training_collector.should_trigger_retrain():
    await training_collector.trigger_retrain()
```
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sqlalchemy import and_, desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


class TrainingCollector:
    """
    Collects training samples from Council corrections for model retraining.

    Storage Strategy:
    - Database: Metadata and labels (small, queryable)
    - S3/Local: Feature vectors (large, bulk access)

    Retraining Triggers:
    - 1000+ new samples collected
    - 7 days since last retrain
    - Council override rate > 15% (model drift)
    """

    def __init__(
        self,
        storage_path: str = "./data/training_samples",
        batch_size: int = 100,
        retrain_threshold: int = 1000,
        retrain_interval_days: int = 7,
        override_rate_threshold: float = 0.15,
    ):
        """
        Initialize training collector.

        Args:
            storage_path: Path to store feature vectors
            batch_size: Batch size for writing to disk
            retrain_threshold: Minimum samples before retraining
            retrain_interval_days: Days between retrains
            override_rate_threshold: Council override rate to trigger retrain
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.batch_size = batch_size
        self.retrain_threshold = retrain_threshold
        self.retrain_interval_days = retrain_interval_days
        self.override_rate_threshold = override_rate_threshold

        # In-memory buffer (flushed to disk periodically)
        self.sample_buffer: List[Dict[str, Any]] = []

        # Statistics
        self.stats = {
            "total_collected": 0,
            "council_overrides": 0,
            "last_retrain": None,
            "pending_samples": 0,
        }

        logger.info(
            f"TrainingCollector initialized: "
            f"threshold={retrain_threshold} samples, "
            f"interval={retrain_interval_days} days"
        )

    async def collect_sample(
        self,
        features: np.ndarray,
        ml_prediction: str,
        ml_confidence: float,
        council_verdict: str,
        correct_label: str,
        incident_id: int,
        db: Optional[AsyncSession] = None,
    ) -> bool:
        """
        Collect a training sample from a Council-verified incident.

        Args:
            features: 79-dimensional feature vector
            ml_prediction: ML model's prediction
            ml_confidence: ML model's confidence
            council_verdict: Council's verdict (CONFIRM, OVERRIDE, UNCERTAIN)
            correct_label: Correct label (from Council or analyst)
            incident_id: Database incident ID
            db: Database session (optional, for storing metadata)

        Returns:
            True if sample collected successfully
        """
        try:
            # Create sample record
            sample = {
                "incident_id": incident_id,
                "features": features.tolist()
                if isinstance(features, np.ndarray)
                else features,
                "ml_prediction": ml_prediction,
                "ml_confidence": ml_confidence,
                "council_verdict": council_verdict,
                "correct_label": correct_label,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "was_override": council_verdict == "OVERRIDE",
            }

            # Add to buffer
            self.sample_buffer.append(sample)

            # Update statistics
            self.stats["total_collected"] += 1
            self.stats["pending_samples"] += 1
            if council_verdict == "OVERRIDE":
                self.stats["council_overrides"] += 1

            # Store in database if session provided
            if db:
                await self._store_in_database(sample, db)

            # Flush buffer if full
            if len(self.sample_buffer) >= self.batch_size:
                await self._flush_buffer()

            logger.debug(
                f"Collected training sample {self.stats['total_collected']}: "
                f"ML={ml_prediction}, Council={council_verdict}, "
                f"Correct={correct_label}"
            )

            return True

        except Exception as e:
            logger.error(f"Failed to collect training sample: {e}")
            return False

    async def _store_in_database(self, sample: Dict[str, Any], db: AsyncSession):
        """
        Store sample metadata in database.

        Note: We import models here to avoid circular dependencies.
        """
        try:
            # Lazy import to avoid circular dependency
            from ..models import TrainingSample

            db_sample = TrainingSample(
                incident_id=sample["incident_id"],
                ml_prediction=sample["ml_prediction"],
                ml_confidence=sample["ml_confidence"],
                council_verdict=sample["council_verdict"],
                correct_label=sample["correct_label"],
                was_override=sample["was_override"],
                features_stored=True,  # Indicates features are in file storage
            )

            db.add(db_sample)
            await db.commit()

        except Exception as e:
            logger.error(f"Failed to store sample in database: {e}")
            # Continue even if DB storage fails (features still in buffer)

    async def _flush_buffer(self):
        """Flush in-memory buffer to disk storage."""
        if not self.sample_buffer:
            return

        try:
            # Generate filename with timestamp
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            filename = self.storage_path / f"training_batch_{timestamp}.json"

            # Write buffer to file
            with open(filename, "w") as f:
                json.dump(self.sample_buffer, f)

            logger.info(f"Flushed {len(self.sample_buffer)} samples to {filename}")

            # Clear buffer
            self.sample_buffer = []

        except Exception as e:
            logger.error(f"Failed to flush sample buffer: {e}")

    async def should_trigger_retrain(
        self, db: Optional[AsyncSession] = None
    ) -> Tuple[bool, str]:
        """
        Check if retraining should be triggered.

        Triggers:
        1. Collected samples >= threshold (e.g., 1000)
        2. Days since last retrain >= interval (e.g., 7 days)
        3. Council override rate > threshold (e.g., 15%) - indicates model drift

        Args:
            db: Database session (optional, for querying sample count)

        Returns:
            (should_retrain, reason) tuple
        """
        reasons = []

        # Check sample count
        if db:
            from ..models import TrainingSample

            count_query = select(func.count(TrainingSample.id)).where(
                TrainingSample.used_for_training == False
            )
            result = await db.execute(count_query)
            pending_count = result.scalar()
        else:
            pending_count = self.stats["pending_samples"]

        if pending_count >= self.retrain_threshold:
            reasons.append(
                f"Sample threshold reached: {pending_count}/{self.retrain_threshold}"
            )

        # Check time since last retrain
        if self.stats["last_retrain"]:
            last_retrain = datetime.fromisoformat(self.stats["last_retrain"])
            days_since = (datetime.now(timezone.utc) - last_retrain).days

            if days_since >= self.retrain_interval_days:
                reasons.append(
                    f"Time interval reached: {days_since}/{self.retrain_interval_days} days"
                )

        # Check override rate (model drift indicator)
        if self.stats["total_collected"] > 100:  # Need minimum samples
            override_rate = (
                self.stats["council_overrides"] / self.stats["total_collected"]
            )

            if override_rate > self.override_rate_threshold:
                reasons.append(
                    f"High override rate: {override_rate:.1%} > {self.override_rate_threshold:.1%} (model drift)"
                )

        should_retrain = len(reasons) > 0

        if should_retrain:
            reason = "; ".join(reasons)
            logger.info(f"Retraining triggered: {reason}")
            return True, reason
        else:
            return False, "No trigger conditions met"

    async def trigger_retrain(self):
        """
        Trigger the retraining process.

        This method creates a retraining task but doesn't block.
        The actual retraining happens in the background.
        """
        try:
            # Flush any pending samples
            await self._flush_buffer()

            # Import retrainer
            from .model_retrainer import model_retrainer

            # Schedule retraining (non-blocking)
            logger.info("Scheduling model retraining task...")
            asyncio.create_task(model_retrainer.retrain_models())

            # Update last retrain timestamp
            self.stats["last_retrain"] = datetime.now(timezone.utc).isoformat()

            logger.info("Retraining task scheduled successfully")

        except Exception as e:
            logger.error(f"Failed to trigger retraining: {e}")

    async def load_training_data(
        self, db: Optional[AsyncSession] = None, limit: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load collected training data for retraining.

        Args:
            db: Database session (optional)
            limit: Maximum samples to load

        Returns:
            (X, y) tuple where X is features, y is labels
        """
        all_samples = []

        try:
            # Load from file storage
            sample_files = sorted(self.storage_path.glob("training_batch_*.json"))

            for sample_file in sample_files:
                with open(sample_file, "r") as f:
                    samples = json.load(f)
                    all_samples.extend(samples)

            # Add buffer samples
            all_samples.extend(self.sample_buffer)

            if not all_samples:
                logger.warning("No training samples available")
                return np.array([]), np.array([])

            # Apply limit if specified
            if limit and len(all_samples) > limit:
                all_samples = all_samples[-limit:]  # Most recent samples

            # Extract features and labels
            X = np.array([s["features"] for s in all_samples])
            y = np.array(
                [self._label_to_class_id(s["correct_label"]) for s in all_samples]
            )

            logger.info(
                f"Loaded {len(all_samples)} training samples: "
                f"features shape={X.shape}, labels shape={y.shape}"
            )

            return X, y

        except Exception as e:
            logger.error(f"Failed to load training data: {e}")
            return np.array([]), np.array([])

    def _label_to_class_id(self, label: str) -> int:
        """
        Convert string label to class ID.

        Class mapping (CICIDS2017):
        0: Normal
        1: DDoS/DoS Attack
        2: Network Reconnaissance / PortScan
        3: Brute Force Attack
        4: Web Application Attack
        5: Malware/Botnet
        6: Advanced Persistent Threat / Infiltration
        """
        label_map = {
            "normal": 0,
            "benign": 0,
            "false_positive": 0,
            "ddos": 1,
            "dos": 1,
            "ddos/dos attack": 1,
            "portscan": 2,
            "network reconnaissance": 2,
            "reconnaissance": 2,
            "brute force": 3,
            "brute force attack": 3,
            "web attack": 4,
            "web application attack": 4,
            "botnet": 5,
            "malware": 5,
            "malware/botnet": 5,
            "apt": 6,
            "infiltration": 6,
            "advanced persistent threat": 6,
        }

        normalized_label = label.lower().strip()
        return label_map.get(normalized_label, 0)  # Default to Normal if unknown

    def get_statistics(self) -> Dict[str, Any]:
        """Get collection statistics."""
        override_rate = (
            self.stats["council_overrides"] / self.stats["total_collected"]
            if self.stats["total_collected"] > 0
            else 0.0
        )

        return {
            **self.stats,
            "override_rate": override_rate,
            "buffer_size": len(self.sample_buffer),
            "storage_path": str(self.storage_path),
        }

    async def mark_samples_used(
        self, db: AsyncSession, sample_ids: Optional[List[int]] = None
    ):
        """
        Mark samples as used for training (prevents reuse).

        Args:
            db: Database session
            sample_ids: Specific sample IDs to mark (or all if None)
        """
        try:
            from ..models import TrainingSample

            if sample_ids:
                # Mark specific samples
                query = select(TrainingSample).where(TrainingSample.id.in_(sample_ids))
            else:
                # Mark all unused samples
                query = select(TrainingSample).where(
                    TrainingSample.used_for_training == False
                )

            result = await db.execute(query)
            samples = result.scalars().all()

            for sample in samples:
                sample.used_for_training = True

            await db.commit()

            # Reset pending counter
            self.stats["pending_samples"] = 0

            logger.info(f"Marked {len(samples)} samples as used for training")

        except Exception as e:
            logger.error(f"Failed to mark samples as used: {e}")


# Global singleton instance
training_collector = TrainingCollector()


# Convenience functions
async def collect_council_correction(
    features: np.ndarray,
    ml_prediction: str,
    ml_confidence: float,
    council_verdict: str,
    correct_label: str,
    incident_id: int,
    db: Optional[AsyncSession] = None,
) -> bool:
    """
    Quick function to collect a training sample from Council correction.

    Args:
        features: Feature vector
        ml_prediction: ML model's prediction
        ml_confidence: ML model's confidence
        council_verdict: Council verdict
        correct_label: Correct label
        incident_id: Incident ID
        db: Database session

    Returns:
        True if collected successfully
    """
    return await training_collector.collect_sample(
        features=features,
        ml_prediction=ml_prediction,
        ml_confidence=ml_confidence,
        council_verdict=council_verdict,
        correct_label=correct_label,
        incident_id=incident_id,
        db=db,
    )


__all__ = [
    "TrainingCollector",
    "training_collector",
    "collect_council_correction",
]
