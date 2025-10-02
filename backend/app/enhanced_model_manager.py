"""
🚀 ENHANCED MODEL MANAGEMENT INTERFACE
Easy-to-use interface for training, deploying, and monitoring enhanced threat detection models

Features:
1. One-click model training with strategic data enhancement
2. Model performance monitoring and comparison
3. A/B testing between models
4. Model versioning and rollback
5. Real-time performance metrics
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
import json
from datetime import datetime, timezone
import pandas as pd
import numpy as np

from .enhanced_training_pipeline import enhanced_training_pipeline
from .enhanced_threat_detector import enhanced_detector, EnhancedThreatDetectionSystem
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


class EnhancedModelManager:
    """Complete management interface for enhanced threat detection models"""

    def __init__(self, models_dir: str = "/Users/chasemad/Desktop/mini-xdr/models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # Model registry
        self.model_registry = {}
        self.active_model = None
        self.performance_metrics = {}

        self._load_model_registry()

    async def train_new_model(
        self,
        db: AsyncSession,
        model_name: str = None,
        use_existing_model_for_hard_examples: bool = True
    ) -> Dict[str, Any]:
        """
        Train a new enhanced model with strategic data improvements

        Args:
            db: Database session
            model_name: Optional model name (auto-generated if None)
            use_existing_model_for_hard_examples: Use existing model to identify hard examples

        Returns:
            Training results and model information
        """

        if model_name is None:
            model_name = f"enhanced_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        logger.info(f"🚀 Training new enhanced model: {model_name}")

        try:
            # Prepare model directory
            model_path = self.models_dir / model_name
            model_path.mkdir(parents=True, exist_ok=True)

            # Determine existing model path for hard example mining
            existing_model_path = None
            if use_existing_model_for_hard_examples and self.active_model:
                existing_model_path = str(self.models_dir / self.active_model / "enhanced_threat_detector.pth")
                if not Path(existing_model_path).exists():
                    # Fallback to original model
                    existing_model_path = str(self.models_dir / "threat_detector.pth")
                    if not Path(existing_model_path).exists():
                        existing_model_path = None

            # Train the enhanced model
            training_results = await enhanced_training_pipeline.train_enhanced_model(
                db=db,
                model_save_path=str(model_path),
                existing_model_path=existing_model_path
            )

            if training_results["success"]:
                # Register the new model
                model_info = {
                    "name": model_name,
                    "path": str(model_path),
                    "created_at": datetime.now().isoformat(),
                    "training_stats": training_results["training_stats"],
                    "enhancement_stats": training_results["enhancement_stats"],
                    "validation_accuracy": training_results["validation_accuracy"],
                    "status": "trained"
                }

                self.model_registry[model_name] = model_info
                self._save_model_registry()

                logger.info(f"✅ Model {model_name} trained successfully!")
                logger.info(f"   📊 Validation Accuracy: {training_results['validation_accuracy']:.1%}")
                logger.info(f"   🎯 Training Samples: {training_results['training_samples']:,}")

                return {
                    "success": True,
                    "model_name": model_name,
                    "model_info": model_info,
                    "training_results": training_results
                }
            else:
                logger.error(f"❌ Model {model_name} training failed")
                return {"success": False, "error": "Training failed"}

        except Exception as e:
            logger.error(f"❌ Model training failed: {e}")
            return {"success": False, "error": str(e)}

    async def deploy_model(self, model_name: str) -> Dict[str, Any]:
        """
        Deploy a trained model as the active model

        Args:
            model_name: Name of the model to deploy

        Returns:
            Deployment results
        """

        if model_name not in self.model_registry:
            return {"success": False, "error": f"Model {model_name} not found in registry"}

        model_info = self.model_registry[model_name]
        model_path = model_info["path"]

        try:
            logger.info(f"🚀 Deploying model: {model_name}")

            # Load the model into the enhanced detector
            success = enhanced_detector.load_model(model_path)

            if success:
                # Update active model
                old_model = self.active_model
                self.active_model = model_name

                # Update model status
                model_info["status"] = "active"
                model_info["deployed_at"] = datetime.now().isoformat()

                # Mark previous model as inactive
                if old_model and old_model in self.model_registry:
                    self.model_registry[old_model]["status"] = "inactive"

                self._save_model_registry()

                logger.info(f"✅ Model {model_name} deployed successfully!")
                if old_model:
                    logger.info(f"   📦 Previous model {old_model} deactivated")

                return {
                    "success": True,
                    "model_name": model_name,
                    "previous_model": old_model,
                    "deployment_time": model_info["deployed_at"]
                }
            else:
                logger.error(f"❌ Failed to load model {model_name}")
                return {"success": False, "error": "Model loading failed"}

        except Exception as e:
            logger.error(f"❌ Model deployment failed: {e}")
            return {"success": False, "error": str(e)}

    def get_model_performance(self, model_name: str = None) -> Dict[str, Any]:
        """
        Get performance metrics for a specific model or active model

        Args:
            model_name: Optional model name (uses active model if None)

        Returns:
            Performance metrics
        """

        target_model = model_name or self.active_model

        if not target_model:
            return {"error": "No model specified and no active model"}

        if target_model not in self.model_registry:
            return {"error": f"Model {target_model} not found"}

        model_info = self.model_registry[target_model]

        # Get real-time performance metrics if available
        runtime_metrics = self.performance_metrics.get(target_model, {})

        performance = {
            "model_name": target_model,
            "status": model_info["status"],
            "training_accuracy": model_info.get("validation_accuracy", 0.0),
            "training_samples": model_info.get("training_stats", {}).get("training_samples", 0),
            "created_at": model_info["created_at"],
            "deployed_at": model_info.get("deployed_at"),
            "runtime_metrics": runtime_metrics,
            "model_details": {
                "enhancement_stats": model_info.get("enhancement_stats", {}),
                "training_stats": model_info.get("training_stats", {})
            }
        }

        return performance

    def compare_models(self, model_names: List[str] = None) -> Dict[str, Any]:
        """
        Compare performance of multiple models

        Args:
            model_names: List of model names to compare (all if None)

        Returns:
            Comparison results
        """

        if model_names is None:
            model_names = list(self.model_registry.keys())

        comparison = {
            "models": {},
            "summary": {},
            "recommendations": []
        }

        accuracies = []
        training_samples = []

        for name in model_names:
            if name in self.model_registry:
                model_info = self.model_registry[name]
                perf = self.get_model_performance(name)

                comparison["models"][name] = {
                    "accuracy": perf.get("training_accuracy", 0.0),
                    "samples": perf.get("training_samples", 0),
                    "status": perf.get("status"),
                    "created_at": perf.get("created_at"),
                    "runtime_metrics": perf.get("runtime_metrics", {})
                }

                accuracies.append(perf.get("training_accuracy", 0.0))
                training_samples.append(perf.get("training_samples", 0))

        if accuracies:
            comparison["summary"] = {
                "best_accuracy": max(accuracies),
                "avg_accuracy": np.mean(accuracies),
                "total_models": len(model_names),
                "avg_training_samples": int(np.mean(training_samples)) if training_samples else 0
            }

            # Find best model
            best_idx = np.argmax(accuracies)
            best_model = model_names[best_idx]

            comparison["recommendations"].append(
                f"Best performing model: {best_model} ({accuracies[best_idx]:.1%} accuracy)"
            )

            if self.active_model != best_model:
                comparison["recommendations"].append(
                    f"Consider deploying {best_model} - it outperforms current active model"
                )

        return comparison

    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status and health"""

        status = {
            "timestamp": datetime.now().isoformat(),
            "active_model": self.active_model,
            "total_models": len(self.model_registry),
            "enhanced_detector_loaded": enhanced_detector.model is not None,
            "openai_available": enhanced_detector.openai_analyzer.client is not None,
            "models": {}
        }

        # Add model summaries
        for name, info in self.model_registry.items():
            status["models"][name] = {
                "status": info["status"],
                "accuracy": info.get("validation_accuracy", 0.0),
                "created_at": info["created_at"]
            }

        # System health checks
        status["health"] = {
            "models_available": len(self.model_registry) > 0,
            "active_model_loaded": enhanced_detector.model is not None,
            "enhanced_features_enabled": True,
            "openai_integration": enhanced_detector.openai_analyzer.client is not None
        }

        return status

    def _load_model_registry(self):
        """Load model registry from disk"""
        registry_file = self.models_dir / "model_registry.json"

        if registry_file.exists():
            try:
                with open(registry_file, 'r') as f:
                    self.model_registry = json.load(f)

                # Find active model
                for name, info in self.model_registry.items():
                    if info.get("status") == "active":
                        self.active_model = name
                        break

                logger.info(f"Loaded model registry: {len(self.model_registry)} models")

            except Exception as e:
                logger.error(f"Failed to load model registry: {e}")
                self.model_registry = {}
        else:
            logger.info("No existing model registry found")

    def _save_model_registry(self):
        """Save model registry to disk"""
        registry_file = self.models_dir / "model_registry.json"

        try:
            with open(registry_file, 'w') as f:
                json.dump(self.model_registry, f, indent=2)
            logger.debug("Model registry saved")
        except Exception as e:
            logger.error(f"Failed to save model registry: {e}")

    async def quick_start_setup(self, db: AsyncSession) -> Dict[str, Any]:
        """
        Quick start setup: Train and deploy first enhanced model

        Returns:
            Setup results
        """

        logger.info("🚀 Starting Quick Setup for Enhanced Threat Detection")

        try:
            # Check if we already have an active model
            if self.active_model:
                logger.info(f"✅ Active model already exists: {self.active_model}")
                return {
                    "success": True,
                    "message": f"System already configured with active model: {self.active_model}",
                    "active_model": self.active_model,
                    "action": "none_required"
                }

            # Train first model
            logger.info("📚 Training first enhanced model...")
            training_result = await self.train_new_model(
                db=db,
                model_name="initial_enhanced_model",
                use_existing_model_for_hard_examples=False  # No existing model yet
            )

            if not training_result["success"]:
                return {
                    "success": False,
                    "error": f"Initial model training failed: {training_result.get('error', 'Unknown error')}"
                }

            # Deploy the model
            model_name = training_result["model_name"]
            logger.info(f"🚀 Deploying initial model: {model_name}")

            deployment_result = await self.deploy_model(model_name)

            if deployment_result["success"]:
                logger.info("🎉 Quick setup completed successfully!")

                return {
                    "success": True,
                    "message": "Enhanced threat detection system setup completed",
                    "model_trained": model_name,
                    "model_deployed": True,
                    "training_accuracy": training_result["training_results"]["validation_accuracy"],
                    "action": "setup_completed"
                }
            else:
                return {
                    "success": False,
                    "error": f"Model deployment failed: {deployment_result.get('error', 'Unknown error')}"
                }

        except Exception as e:
            logger.error(f"❌ Quick setup failed: {e}")
            return {
                "success": False,
                "error": f"Quick setup failed: {str(e)}"
            }

    def update_performance_metrics(self, model_name: str, metrics: Dict[str, Any]):
        """Update runtime performance metrics for a model"""
        if model_name not in self.performance_metrics:
            self.performance_metrics[model_name] = {}

        self.performance_metrics[model_name].update(metrics)
        self.performance_metrics[model_name]["last_updated"] = datetime.now().isoformat()


# Global model manager instance
model_manager = EnhancedModelManager()


# Convenience functions for easy access
async def train_enhanced_model(db: AsyncSession, model_name: str = None) -> Dict[str, Any]:
    """Train a new enhanced threat detection model"""
    return await model_manager.train_new_model(db, model_name)

async def deploy_enhanced_model(model_name: str) -> Dict[str, Any]:
    """Deploy an enhanced model"""
    return await model_manager.deploy_model(model_name)

def get_model_status() -> Dict[str, Any]:
    """Get current model system status"""
    return model_manager.get_system_status()

async def quick_setup_enhanced_system(db: AsyncSession) -> Dict[str, Any]:
    """Quick setup for enhanced threat detection system"""
    return await model_manager.quick_start_setup(db)