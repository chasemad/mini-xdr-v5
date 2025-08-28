"""
Continuous Learning Pipeline for Adaptive Detection
Continuously learns and adapts detection models in the background
"""
import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional, List
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, func

from .models import Event, Incident
from .adaptive_detection import behavioral_analyzer
from .baseline_engine import baseline_engine
from .ml_engine import ml_detector, prepare_training_data_from_events
from .db import AsyncSessionLocal

logger = logging.getLogger(__name__)


class ContinuousLearningPipeline:
    """Continuously learns and adapts detection models"""
    
    def __init__(self):
        self.learning_schedule = {
            'baseline_update': 3600,      # Update baselines every hour
            'model_retrain': 86400,       # Retrain ML models daily  
            'pattern_refresh': 1800,      # Refresh patterns every 30min
            'sensitivity_adjust': 7200    # Adjust sensitivity every 2 hours
        }
        
        self.running = False
        self.tasks = []
        
        # Learning configuration
        self.min_events_for_training = 100
        self.max_training_events = 10000
        self.training_history_days = 14
        
        # Performance tracking
        self.learning_metrics = {
            'last_baseline_update': None,
            'last_model_retrain': None,
            'last_pattern_refresh': None,
            'training_events_used': 0,
            'successful_updates': 0,
            'failed_updates': 0
        }
        
    async def start_learning_loop(self):
        """Start continuous learning background tasks"""
        if self.running:
            logger.warning("Learning pipeline already running")
            return
        
        self.running = True
        logger.info("Starting continuous learning pipeline")
        
        # Schedule all background tasks
        self.tasks = [
            asyncio.create_task(self._baseline_update_loop()),
            asyncio.create_task(self._model_retrain_loop()),
            asyncio.create_task(self._pattern_update_loop()),
            asyncio.create_task(self._sensitivity_adjustment_loop())
        ]
        
        logger.info(f"Started {len(self.tasks)} learning tasks")
    
    async def stop_learning_loop(self):
        """Stop all learning tasks"""
        if not self.running:
            return
        
        self.running = False
        
        # Cancel all tasks
        for task in self.tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)
        
        logger.info("Stopped continuous learning pipeline")
    
    async def _baseline_update_loop(self):
        """Continuously update behavioral baselines"""
        while self.running:
            try:
                async with AsyncSessionLocal() as db:
                    await baseline_engine.update_baselines(db)
                    await db.commit()
                
                self.learning_metrics['last_baseline_update'] = datetime.now(timezone.utc)
                self.learning_metrics['successful_updates'] += 1
                logger.info("Baselines updated successfully")
                
                await asyncio.sleep(self.learning_schedule['baseline_update'])
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.learning_metrics['failed_updates'] += 1
                logger.error(f"Baseline update error: {e}")
                await asyncio.sleep(60)  # Retry in 1 minute
    
    async def _model_retrain_loop(self):
        """Continuously retrain ML models with new data"""
        while self.running:
            try:
                async with AsyncSessionLocal() as db:
                    success = await self._retrain_ml_models(db)
                    if success:
                        self.learning_metrics['last_model_retrain'] = datetime.now(timezone.utc)
                        self.learning_metrics['successful_updates'] += 1
                        logger.info("ML models retrained successfully")
                    else:
                        self.learning_metrics['failed_updates'] += 1
                        logger.warning("ML model retraining failed")
                
                await asyncio.sleep(self.learning_schedule['model_retrain'])
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.learning_metrics['failed_updates'] += 1
                logger.error(f"Model retraining error: {e}")
                await asyncio.sleep(3600)  # Retry in 1 hour
    
    async def _pattern_update_loop(self):
        """Continuously update behavioral patterns"""
        while self.running:
            try:
                async with AsyncSessionLocal() as db:
                    await self._update_behavioral_patterns(db)
                
                self.learning_metrics['last_pattern_refresh'] = datetime.now(timezone.utc)
                self.learning_metrics['successful_updates'] += 1
                logger.info("Behavioral patterns updated successfully")
                
                await asyncio.sleep(self.learning_schedule['pattern_refresh'])
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.learning_metrics['failed_updates'] += 1
                logger.error(f"Pattern update error: {e}")
                await asyncio.sleep(300)  # Retry in 5 minutes
    
    async def _sensitivity_adjustment_loop(self):
        """Automatically adjust detection sensitivity based on performance"""
        while self.running:
            try:
                async with AsyncSessionLocal() as db:
                    await self._adjust_detection_sensitivity(db)
                
                logger.info("Detection sensitivity adjusted")
                await asyncio.sleep(self.learning_schedule['sensitivity_adjust'])
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Sensitivity adjustment error: {e}")
                await asyncio.sleep(1800)  # Retry in 30 minutes
    
    async def _retrain_ml_models(self, db: AsyncSession) -> bool:
        """Retrain ML models with recent clean data"""
        try:
            # Get training data from recent events
            training_events = await self._get_training_events(db)
            
            if len(training_events) < self.min_events_for_training:
                logger.warning(f"Insufficient training data: {len(training_events)} events")
                return False
            
            # Prepare training data
            training_data = await prepare_training_data_from_events(training_events)
            
            if not training_data:
                logger.warning("No valid training data prepared")
                return False
            
            self.learning_metrics['training_events_used'] = len(training_events)
            
            # Train ensemble models
            results = await ml_detector.train_models(training_data)
            
            # Check if any models trained successfully
            successful_models = sum(1 for success in results.values() if success)
            
            if successful_models > 0:
                logger.info(f"Successfully trained {successful_models} ML models")
                return True
            else:
                logger.warning("No ML models trained successfully")
                return False
                
        except Exception as e:
            logger.error(f"ML model retraining failed: {e}")
            return False
    
    async def _get_training_events(self, db: AsyncSession) -> List[Event]:
        """Get clean training events from recent history"""
        
        # Get events from the last N days, excluding incident periods
        window_start = datetime.now(timezone.utc) - timedelta(days=self.training_history_days)
        
        # Get incident IPs to exclude
        incident_query = select(Incident.src_ip.distinct()).where(
            Incident.created_at >= window_start
        )
        incident_result = await db.execute(incident_query)
        incident_ips = {row[0] for row in incident_result}
        
        # Get clean events (not from incident IPs)
        events_query = select(Event).where(
            and_(
                Event.ts >= window_start,
                ~Event.src_ip.in_(incident_ips) if incident_ips else True
            )
        ).order_by(Event.ts.desc()).limit(self.max_training_events)
        
        result = await db.execute(events_query)
        events = result.scalars().all()
        
        logger.info(f"Retrieved {len(events)} clean training events, excluded {len(incident_ips)} incident IPs")
        return events
    
    async def _update_behavioral_patterns(self, db: AsyncSession):
        """Update behavioral pattern thresholds based on recent performance"""
        
        # Get recent incident accuracy
        recent_incidents = await self._get_recent_incidents(db, hours=24)
        
        if not recent_incidents:
            return
        
        # Analyze false positive rate
        false_positives = 0
        true_positives = 0
        
        for incident in recent_incidents:
            # Simple heuristic: incidents that were quickly dismissed are likely false positives
            if incident.status == "dismissed" and hasattr(incident, 'created_at'):
                time_to_dismiss = datetime.now(timezone.utc) - incident.created_at
                if time_to_dismiss.total_seconds() < 3600:  # Dismissed within 1 hour
                    false_positives += 1
                else:
                    true_positives += 1
            elif incident.status in ["contained", "open"]:
                true_positives += 1
        
        total_incidents = false_positives + true_positives
        
        if total_incidents > 5:  # Need minimum sample size
            false_positive_rate = false_positives / total_incidents
            
            # Adjust behavioral analyzer threshold
            if false_positive_rate > 0.3:  # Too many false positives
                behavioral_analyzer.adaptive_threshold = min(0.8, behavioral_analyzer.adaptive_threshold + 0.05)
                logger.info(f"Increased behavioral threshold to {behavioral_analyzer.adaptive_threshold:.2f} (FPR: {false_positive_rate:.2f})")
            elif false_positive_rate < 0.1:  # Very few false positives, can be more aggressive
                behavioral_analyzer.adaptive_threshold = max(0.4, behavioral_analyzer.adaptive_threshold - 0.05)
                logger.info(f"Decreased behavioral threshold to {behavioral_analyzer.adaptive_threshold:.2f} (FPR: {false_positive_rate:.2f})")
    
    async def _get_recent_incidents(self, db: AsyncSession, hours: int = 24) -> List[Incident]:
        """Get recent incidents for analysis"""
        window_start = datetime.now(timezone.utc) - timedelta(hours=hours)
        
        query = select(Incident).where(
            Incident.created_at >= window_start
        ).order_by(Incident.created_at.desc())
        
        result = await db.execute(query)
        return result.scalars().all()
    
    async def _adjust_detection_sensitivity(self, db: AsyncSession):
        """Adjust detection sensitivity based on system load and accuracy"""
        
        # Get recent system performance metrics
        recent_incidents = await self._get_recent_incidents(db, hours=6)
        
        incident_rate = len(recent_incidents) / 6  # Incidents per hour
        
        # Adjust baseline engine sensitivity
        if incident_rate > 10:  # High incident rate, reduce sensitivity
            baseline_engine.adjust_sensitivity("low")
            logger.info("Reduced detection sensitivity due to high incident rate")
        elif incident_rate < 2:  # Low incident rate, increase sensitivity
            baseline_engine.adjust_sensitivity("high")
            logger.info("Increased detection sensitivity due to low incident rate")
        else:
            baseline_engine.adjust_sensitivity("medium")
    
    def get_learning_status(self) -> Dict[str, Any]:
        """Get current learning pipeline status"""
        return {
            'running': self.running,
            'active_tasks': len([t for t in self.tasks if not t.done()]),
            'learning_metrics': self.learning_metrics,
            'schedule': self.learning_schedule,
            'behavioral_threshold': getattr(behavioral_analyzer, 'adaptive_threshold', 0.6),
            'baseline_status': baseline_engine.get_baseline_status(),
            'ml_model_status': ml_detector.get_model_status()
        }
    
    async def force_learning_update(self) -> Dict[str, bool]:
        """Force an immediate learning update (for testing/manual triggers)"""
        results = {}
        
        try:
            async with AsyncSessionLocal() as db:
                # Force baseline update
                await baseline_engine.update_baselines(db)
                results['baseline_update'] = True
                
                # Force ML model retraining
                results['model_retrain'] = await self._retrain_ml_models(db)
                
                # Force pattern update
                await self._update_behavioral_patterns(db)
                results['pattern_update'] = True
                
                await db.commit()
                
        except Exception as e:
            logger.error(f"Forced learning update failed: {e}")
            results['error'] = str(e)
        
        return results


# Global learning pipeline instance
learning_pipeline = ContinuousLearningPipeline()
