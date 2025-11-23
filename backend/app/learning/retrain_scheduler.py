"""
Retrain Scheduler - Background Task for Automated Model Retraining

This module runs a background scheduler that periodically checks if model
retraining should be triggered based on Council corrections. When trigger
conditions are met, it initiates the retraining pipeline.

Flow:
1. Background task runs every N minutes (default: 60)
2. Checks training_collector.should_trigger_retrain()
3. If triggered, calls model_retrainer.retrain_models()
4. Logs results and updates metrics

Trigger Conditions:
- 1000+ new labeled samples collected
- 7 days since last retrain
- Council override rate > 15% (model drift)

Usage:
```python
from app.learning import retrain_scheduler

# Start scheduler (typically in FastAPI startup)
await retrain_scheduler.start()

# Stop scheduler (typically in FastAPI shutdown)
await retrain_scheduler.stop()

# Check status
status = retrain_scheduler.get_status()
print(f"Scheduler active: {status['active']}")
print(f"Last check: {status['last_check']}")
print(f"Last retrain: {status['last_retrain']}")
```
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


class RetrainScheduler:
    """
    Background scheduler for automated model retraining.

    This scheduler periodically checks if retraining should be triggered
    and initiates the retraining pipeline when conditions are met.
    """

    def __init__(
        self,
        check_interval_minutes: int = 60,
        enable_scheduler: bool = True,
        max_retrain_duration_hours: int = 4,
    ):
        """
        Initialize the retrain scheduler.

        Args:
            check_interval_minutes: How often to check for retrain triggers
            enable_scheduler: Enable/disable scheduler (useful for testing)
            max_retrain_duration_hours: Maximum time for retraining job
        """
        self.check_interval_minutes = check_interval_minutes
        self.enable_scheduler = enable_scheduler
        self.max_retrain_duration_hours = max_retrain_duration_hours

        # Scheduler instance
        self.scheduler: Optional[AsyncIOScheduler] = None

        # State tracking
        self.is_active = False
        self.last_check: Optional[datetime] = None
        self.last_retrain: Optional[datetime] = None
        self.retrain_in_progress = False
        self.current_job_id: Optional[str] = None

        # Statistics
        self.stats = {
            "total_checks": 0,
            "total_retrains_triggered": 0,
            "total_retrains_completed": 0,
            "total_retrains_failed": 0,
            "last_error": None,
        }

        logger.info(
            f"RetrainScheduler initialized: "
            f"check_interval={check_interval_minutes}min, "
            f"enabled={enable_scheduler}"
        )

    async def start(self):
        """Start the background scheduler."""
        if not self.enable_scheduler:
            logger.warning("Scheduler disabled, not starting")
            return

        if self.is_active:
            logger.warning("Scheduler already running")
            return

        try:
            # Create scheduler
            self.scheduler = AsyncIOScheduler()

            # Add periodic check job
            self.scheduler.add_job(
                self._check_and_retrain,
                trigger=IntervalTrigger(minutes=self.check_interval_minutes),
                id="retrain_check",
                name="Check for Model Retraining",
                replace_existing=True,
                max_instances=1,  # Prevent concurrent checks
            )

            # Start scheduler
            self.scheduler.start()
            self.is_active = True

            logger.info(
                f"Retrain scheduler started: checking every "
                f"{self.check_interval_minutes} minutes"
            )

            # Run initial check after 5 minutes
            self.scheduler.add_job(
                self._check_and_retrain,
                trigger="date",
                run_date=datetime.now(timezone.utc) + timedelta(minutes=5),
                id="initial_check",
                name="Initial Retrain Check",
            )

        except Exception as e:
            logger.error(f"Failed to start retrain scheduler: {e}")
            self.is_active = False
            raise

    async def stop(self):
        """Stop the background scheduler."""
        if not self.is_active:
            logger.warning("Scheduler not running")
            return

        try:
            if self.scheduler:
                self.scheduler.shutdown(wait=True)
                self.scheduler = None

            self.is_active = False
            logger.info("Retrain scheduler stopped")

        except Exception as e:
            logger.error(f"Error stopping scheduler: {e}")

    async def _check_and_retrain(self):
        """
        Periodic check for retraining triggers.

        This method is called by the scheduler at regular intervals.
        """
        self.last_check = datetime.now(timezone.utc)
        self.stats["total_checks"] += 1

        logger.debug(f"Running retrain check #{self.stats['total_checks']}")

        # Skip if retraining already in progress
        if self.retrain_in_progress:
            logger.info("Retraining already in progress, skipping check")
            return

        try:
            # Import here to avoid circular dependency
            from ..database import get_db_session
            from .training_collector import training_collector

            # Check if retraining should be triggered
            async with get_db_session() as db:
                (
                    should_retrain,
                    reason,
                ) = await training_collector.should_trigger_retrain(db)

            if should_retrain:
                logger.info(f"Retraining triggered: {reason}")
                self.stats["total_retrains_triggered"] += 1

                # Start retraining (non-blocking)
                await self._start_retrain_job(reason)
            else:
                logger.debug(f"No retraining needed: {reason}")

        except Exception as e:
            logger.error(f"Error during retrain check: {e}", exc_info=True)
            self.stats["last_error"] = str(e)

    async def _start_retrain_job(self, reason: str):
        """
        Start a retraining job in the background.

        Args:
            reason: Reason for triggering retraining
        """
        self.retrain_in_progress = True
        self.current_job_id = (
            f"retrain_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        )

        logger.info(f"Starting retraining job: {self.current_job_id}")

        try:
            # Import retrainer
            from .model_retrainer import model_retrainer

            # Start retraining (with timeout)
            retrain_task = asyncio.create_task(
                model_retrainer.retrain_models(job_id=self.current_job_id)
            )

            # Wait with timeout
            timeout_seconds = self.max_retrain_duration_hours * 3600
            await asyncio.wait_for(retrain_task, timeout=timeout_seconds)

            # Success
            self.last_retrain = datetime.now(timezone.utc)
            self.stats["total_retrains_completed"] += 1

            logger.info(f"Retraining job {self.current_job_id} completed successfully")

        except asyncio.TimeoutError:
            logger.error(
                f"Retraining job {self.current_job_id} exceeded timeout "
                f"({self.max_retrain_duration_hours}h)"
            )
            self.stats["total_retrains_failed"] += 1
            self.stats["last_error"] = "Timeout"

        except Exception as e:
            logger.error(
                f"Retraining job {self.current_job_id} failed: {e}", exc_info=True
            )
            self.stats["total_retrains_failed"] += 1
            self.stats["last_error"] = str(e)

        finally:
            self.retrain_in_progress = False
            self.current_job_id = None

    async def manual_trigger(self, reason: str = "Manual trigger") -> Dict[str, Any]:
        """
        Manually trigger retraining (e.g., via API endpoint).

        Args:
            reason: Reason for manual trigger

        Returns:
            Status dictionary
        """
        if self.retrain_in_progress:
            return {
                "success": False,
                "message": "Retraining already in progress",
                "job_id": self.current_job_id,
            }

        logger.info(f"Manual retraining triggered: {reason}")
        self.stats["total_retrains_triggered"] += 1

        # Start retraining in background
        asyncio.create_task(self._start_retrain_job(reason))

        return {
            "success": True,
            "message": "Retraining job started",
            "reason": reason,
            "job_id": self.current_job_id,
        }

    def get_status(self) -> Dict[str, Any]:
        """Get scheduler status and statistics."""
        return {
            "active": self.is_active,
            "retrain_in_progress": self.retrain_in_progress,
            "current_job_id": self.current_job_id,
            "check_interval_minutes": self.check_interval_minutes,
            "last_check": self.last_check.isoformat() if self.last_check else None,
            "last_retrain": self.last_retrain.isoformat()
            if self.last_retrain
            else None,
            "next_check": self._get_next_check_time(),
            "statistics": self.stats,
        }

    def _get_next_check_time(self) -> Optional[str]:
        """Calculate next scheduled check time."""
        if not self.scheduler or not self.is_active:
            return None

        job = self.scheduler.get_job("retrain_check")
        if job and job.next_run_time:
            return job.next_run_time.isoformat()

        return None

    async def force_stop_retrain(self) -> Dict[str, Any]:
        """
        Force stop current retraining job (emergency use only).

        Returns:
            Status dictionary
        """
        if not self.retrain_in_progress:
            return {
                "success": False,
                "message": "No retraining job in progress",
            }

        logger.warning(f"Force stopping retraining job: {self.current_job_id}")

        # Reset state (the actual task will continue in background)
        job_id = self.current_job_id
        self.retrain_in_progress = False
        self.current_job_id = None

        return {
            "success": True,
            "message": f"Retraining job {job_id} marked as stopped",
            "note": "Background task may still be running",
        }


# Global singleton instance
retrain_scheduler = RetrainScheduler()


# Convenience functions for FastAPI integration
async def start_retrain_scheduler():
    """
    Start the retrain scheduler.

    Call this in FastAPI's startup event:
    ```python
    @app.on_event("startup")
    async def startup():
        await start_retrain_scheduler()
    ```
    """
    await retrain_scheduler.start()


async def stop_retrain_scheduler():
    """
    Stop the retrain scheduler.

    Call this in FastAPI's shutdown event:
    ```python
    @app.on_event("shutdown")
    async def shutdown():
        await stop_retrain_scheduler()
    ```
    """
    await retrain_scheduler.stop()


def get_scheduler_status() -> Dict[str, Any]:
    """Get scheduler status (for API endpoints)."""
    return retrain_scheduler.get_status()


async def trigger_manual_retrain(reason: str = "Manual trigger") -> Dict[str, Any]:
    """Manually trigger retraining (for API endpoints)."""
    return await retrain_scheduler.manual_trigger(reason)


__all__ = [
    "RetrainScheduler",
    "retrain_scheduler",
    "start_retrain_scheduler",
    "stop_retrain_scheduler",
    "get_scheduler_status",
    "trigger_manual_retrain",
]
