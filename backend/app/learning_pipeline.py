"""
Enhanced Continuous Learning Pipeline for Adaptive Detection
Continuously learns and adapts detection models with real-time adaptation,
ensemble optimization, and explainable AI integration
"""
import asyncio
import logging
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional, List
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, func

from .models import Event, Incident
from .adaptive_detection import behavioral_analyzer
from .baseline_engine import baseline_engine
from .ml_engine import ml_detector, prepare_training_data_from_events
from .db import AsyncSessionLocal

# Phase 2B: Advanced ML Integration
try:
    from .online_learning import online_learning_engine, adapt_models_with_new_data
    from .ensemble_optimizer import create_optimized_ensemble, meta_learning_optimizer
    from .model_versioning import model_registry, ab_test_manager, performance_monitor
    from .explainable_ai import explainable_ai
    ADVANCED_ML_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Advanced ML features not available: {e}")
    ADVANCED_ML_AVAILABLE = False
    online_learning_engine = None
    model_registry = None

logger = logging.getLogger(__name__)


class ContinuousLearningPipeline:
    """Enhanced continuously learning and adapting detection models with real-time adaptation"""
    
    def __init__(self):
        self.learning_schedule = {
            'baseline_update': 3600,      # Update baselines every hour
            'model_retrain': 86400,       # Retrain ML models daily  
            'pattern_refresh': 1800,      # Refresh patterns every 30min
            'sensitivity_adjust': 7200,   # Adjust sensitivity every 2 hours
            'online_adaptation': 300,     # Online adaptation every 5 minutes
            'ensemble_optimization': 21600, # Ensemble optimization every 6 hours
            'model_performance_check': 900  # Performance monitoring every 15 minutes
        }
        
        self.running = False
        self.tasks = []
        
        # Learning configuration
        self.min_events_for_training = 100
        self.max_training_events = 10000
        self.training_history_days = 14
        
        # Phase 2B: Advanced ML configuration
        self.online_adaptation_enabled = ADVANCED_ML_AVAILABLE
        self.ensemble_optimization_enabled = ADVANCED_ML_AVAILABLE
        self.explainable_ai_enabled = ADVANCED_ML_AVAILABLE
        self.min_online_adaptation_events = 50
        
        # Performance tracking
        self.learning_metrics = {
            'last_baseline_update': None,
            'last_model_retrain': None,
            'last_pattern_refresh': None,
            'last_online_adaptation': None,
            'last_ensemble_optimization': None,
            'last_performance_check': None,
            'training_events_used': 0,
            'successful_updates': 0,
            'failed_updates': 0,
            'online_adaptations': 0,
            'ensemble_optimizations': 0,
            'drift_detections': 0
        }
        
        # Advanced ML tracking
        self.active_ab_tests = []
        self.model_performance_alerts = []
        self.drift_detection_history = []
        
    async def start_learning_loop(self):
        """Start enhanced continuous learning background tasks"""
        if self.running:
            logger.warning("Learning pipeline already running")
            return
        
        self.running = True
        logger.info("Starting enhanced continuous learning pipeline")
        
        # Initialize advanced ML components
        if ADVANCED_ML_AVAILABLE:
            await self._initialize_advanced_ml()
        
        # Schedule all background tasks
        self.tasks = [
            asyncio.create_task(self._baseline_update_loop()),
            asyncio.create_task(self._model_retrain_loop()),
            asyncio.create_task(self._pattern_update_loop()),
            asyncio.create_task(self._sensitivity_adjustment_loop())
        ]
        
        # Add Phase 2B advanced ML tasks
        if self.online_adaptation_enabled:
            self.tasks.append(asyncio.create_task(self._online_adaptation_loop()))
        
        if self.ensemble_optimization_enabled:
            self.tasks.append(asyncio.create_task(self._ensemble_optimization_loop()))
        
        if ADVANCED_ML_AVAILABLE:
            self.tasks.append(asyncio.create_task(self._model_performance_monitoring_loop()))
        
        logger.info(f"Started {len(self.tasks)} learning tasks (Phase 2B features: {ADVANCED_ML_AVAILABLE})")
    
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
    
    # Phase 2B: Advanced ML Methods
    
    async def _initialize_advanced_ml(self):
        """Initialize advanced ML components"""
        try:
            if online_learning_engine:
                logger.info("Initializing online learning engine...")
                # Additional initialization if needed
            
            if explainable_ai:
                logger.info("Initializing explainable AI components...")
                # Additional initialization if needed
            
            logger.info("Advanced ML components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize advanced ML components: {e}")
    
    async def _online_adaptation_loop(self):
        """Real-time model adaptation loop"""
        while self.running:
            try:
                async with AsyncSessionLocal() as db:
                    # Get recent events for online adaptation
                    recent_events = await self._get_recent_events_for_adaptation(db)
                    
                    if len(recent_events) >= self.min_online_adaptation_events:
                        # Perform online adaptation
                        adaptation_result = await adapt_models_with_new_data(recent_events)
                        
                        if adaptation_result.get('success'):
                            self.learning_metrics['last_online_adaptation'] = datetime.now(timezone.utc)
                            self.learning_metrics['online_adaptations'] += 1
                            
                            logger.info(f"Online adaptation completed: {adaptation_result.get('samples_processed')} samples")
                        else:
                            logger.warning(f"Online adaptation failed: {adaptation_result.get('error', 'Unknown error')}")
                
                await asyncio.sleep(self.learning_schedule['online_adaptation'])
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Online adaptation loop error: {e}")
                await asyncio.sleep(60)  # Retry in 1 minute
    
    async def _ensemble_optimization_loop(self):
        """Ensemble model optimization loop"""
        while self.running:
            try:
                async with AsyncSessionLocal() as db:
                    # Get training data for ensemble optimization
                    training_events = await self._get_training_events(db)
                    
                    if len(training_events) >= self.min_events_for_training:
                        # Optimize ensemble models
                        success = await self._optimize_ensemble_models(training_events)
                        
                        if success:
                            self.learning_metrics['last_ensemble_optimization'] = datetime.now(timezone.utc)
                            self.learning_metrics['ensemble_optimizations'] += 1
                            logger.info("Ensemble optimization completed")
                
                await asyncio.sleep(self.learning_schedule['ensemble_optimization'])
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Ensemble optimization loop error: {e}")
                await asyncio.sleep(3600)  # Retry in 1 hour
    
    async def _model_performance_monitoring_loop(self):
        """Model performance monitoring loop"""
        while self.running:
            try:
                if model_registry and performance_monitor:
                    # Check performance of all production models
                    production_models = model_registry.get_production_models()
                    
                    for model_version in production_models:
                        health_summary = performance_monitor.get_model_health_summary(
                            model_version.model_id, model_version.version
                        )
                        
                        if health_summary['status'] in ['unhealthy', 'degraded']:
                            alert = {
                                'timestamp': datetime.now(timezone.utc),
                                'model_id': model_version.model_id,
                                'version': model_version.version,
                                'status': health_summary['status'],
                                'accuracy': health_summary.get('accuracy', 0),
                                'error_rate': health_summary.get('error_rate', 0)
                            }
                            
                            self.model_performance_alerts.append(alert)
                            logger.warning(f"Model performance alert: {model_version.model_id} v{model_version.version} is {health_summary['status']}")
                    
                    # Clean old alerts (keep last 100)
                    if len(self.model_performance_alerts) > 100:
                        self.model_performance_alerts = self.model_performance_alerts[-100:]
                
                self.learning_metrics['last_performance_check'] = datetime.now(timezone.utc)
                await asyncio.sleep(self.learning_schedule['model_performance_check'])
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Model performance monitoring error: {e}")
                await asyncio.sleep(900)  # Retry in 15 minutes
    
    async def _get_recent_events_for_adaptation(self, db: AsyncSession) -> List[Event]:
        """Get recent events for online adaptation"""
        # Get events from the last hour
        window_start = datetime.now(timezone.utc) - timedelta(hours=1)
        
        query = select(Event).where(
            Event.ts >= window_start
        ).order_by(Event.ts.desc()).limit(1000)  # Limit for performance
        
        result = await db.execute(query)
        events = result.scalars().all()
        
        return events
    
    async def _optimize_ensemble_models(self, training_events: List[Event]) -> bool:
        """Optimize ensemble models using meta-learning"""
        try:
            if not meta_learning_optimizer:
                return False
            
            # Prepare training data
            training_data = await prepare_training_data_from_events(training_events)
            
            if not training_data or 'X' not in training_data or 'y' not in training_data:
                logger.warning("No valid training data for ensemble optimization")
                return False
            
            X, y = training_data['X'], training_data['y']
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]
            
            # Get recommended ensemble configuration
            ensemble_config = meta_learning_optimizer.recommend_ensemble_config(X, y)
            
            # Create optimized ensemble
            ensemble = create_optimized_ensemble(X, y, feature_names, ensemble_config)
            
            # Register the new ensemble model
            if model_registry:
                model_version = model_registry.register_model(
                    model=ensemble,
                    model_id="optimized_ensemble",
                    version=f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    model_type="ensemble",
                    algorithm=ensemble_config.method.value,
                    hyperparameters={'config': ensemble_config.__dict__},
                    training_data_hash=str(hash(str(X.tobytes()))),
                    description=f"Auto-optimized ensemble using {ensemble_config.method.value}"
                )
                
                logger.info(f"Registered optimized ensemble: {model_version.model_id} v{model_version.version}")
            
            return True
            
        except Exception as e:
            logger.error(f"Ensemble optimization failed: {e}")
            return False
    
    async def create_ab_test(self, model_a_id: str, model_a_version: str,
                           model_b_id: str, model_b_version: str,
                           test_name: str, description: str = "") -> Optional[str]:
        """Create A/B test for model comparison"""
        
        if not ab_test_manager:
            logger.error("A/B test manager not available")
            return None
        
        try:
            test_id = ab_test_manager.create_ab_test(
                name=test_name,
                description=description or f"Comparing {model_a_id} v{model_a_version} vs {model_b_id} v{model_b_version}",
                model_a_id=model_a_id,
                model_a_version=model_a_version,
                model_b_id=model_b_id,
                model_b_version=model_b_version,
                traffic_split=0.5,
                success_metric='f1_score',
                min_sample_size=100,
                max_duration_hours=168  # 1 week
            )
            
            # Start the test
            if ab_test_manager.start_ab_test(test_id):
                self.active_ab_tests.append({
                    'test_id': test_id,
                    'name': test_name,
                    'started_at': datetime.now(timezone.utc),
                    'model_a': f"{model_a_id} v{model_a_version}",
                    'model_b': f"{model_b_id} v{model_b_version}"
                })
                
                logger.info(f"Created and started A/B test: {test_id}")
                return test_id
            else:
                logger.error(f"Failed to start A/B test: {test_id}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to create A/B test: {e}")
            return None
    
    def get_enhanced_learning_status(self) -> Dict[str, Any]:
        """Get enhanced learning pipeline status with Phase 2B features"""
        base_status = self.get_learning_status()
        
        if ADVANCED_ML_AVAILABLE:
            # Add Phase 2B status information
            base_status.update({
                'phase_2b_features': {
                    'online_adaptation_enabled': self.online_adaptation_enabled,
                    'ensemble_optimization_enabled': self.ensemble_optimization_enabled,
                    'explainable_ai_enabled': self.explainable_ai_enabled,
                    'online_adaptations': self.learning_metrics['online_adaptations'],
                    'ensemble_optimizations': self.learning_metrics['ensemble_optimizations'],
                    'drift_detections': self.learning_metrics['drift_detections'],
                    'last_online_adaptation': self.learning_metrics['last_online_adaptation'],
                    'last_ensemble_optimization': self.learning_metrics['last_ensemble_optimization'],
                    'last_performance_check': self.learning_metrics['last_performance_check']
                },
                'active_ab_tests': len(self.active_ab_tests),
                'model_performance_alerts': len(self.model_performance_alerts),
                'drift_detection_history': len(self.drift_detection_history)
            })
            
            # Add model registry status
            if model_registry:
                production_models = model_registry.get_production_models()
                base_status['production_models'] = len(production_models)
            
            # Add online learning engine status
            if online_learning_engine:
                base_status['online_learning_status'] = online_learning_engine.get_drift_status()
        
        return base_status
    
    async def explain_recent_prediction(self, incident_id: int, 
                                      context: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Generate explanation for a recent prediction/incident"""
        
        if not self.explainable_ai_enabled or not explainable_ai:
            logger.warning("Explainable AI not available")
            return None
        
        try:
            async with AsyncSessionLocal() as db:
                # Get the incident
                incident_query = select(Incident).where(Incident.id == incident_id)
                result = await db.execute(incident_query)
                incident = result.scalar_one_or_none()
                
                if not incident:
                    logger.error(f"Incident {incident_id} not found")
                    return None
                
                # Get recent events for this IP
                events_query = select(Event).where(
                    and_(
                        Event.src_ip == incident.src_ip,
                        Event.ts <= incident.created_at,
                        Event.ts >= incident.created_at - timedelta(hours=1)
                    )
                ).order_by(Event.ts.desc()).limit(10)
                
                result = await db.execute(events_query)
                events = result.scalars().all()
                
                if not events:
                    logger.warning(f"No events found for incident {incident_id}")
                    return None
                
                # Prepare feature data (simplified)
                feature_data = {}
                feature_names = ['event_count', 'unique_ports', 'avg_message_length', 'anomaly_score']
                
                feature_data['event_count'] = len(events)
                feature_data['unique_ports'] = len(set(e.dst_port for e in events if e.dst_port))
                feature_data['avg_message_length'] = np.mean([len(e.message or '') for e in events])
                feature_data['anomaly_score'] = np.mean([e.anomaly_score or 0 for e in events])
                
                # Generate explanation
                from .explainable_ai import explain_threat_prediction
                
                explanation = await explain_threat_prediction(
                    model_id="threat_detection",
                    model_version="current",
                    instance_data=feature_data,
                    feature_names=feature_names,
                    user_context=context or {
                        'incident_id': incident_id,
                        'source_ip': incident.src_ip,
                        'incident_type': incident.reason
                    }
                )
                
                return {
                    'incident_id': incident_id,
                    'explanation_id': explanation.explanation_id,
                    'prediction': explanation.prediction,
                    'confidence': explanation.confidence,
                    'summary': explanation.narrative_summary,
                    'technical_details': explanation.technical_details,
                    'top_features': [
                        {
                            'name': attr.feature_name,
                            'value': attr.feature_value,
                            'importance': attr.attribution_score,
                            'description': attr.description
                        }
                        for attr in explanation.feature_attributions[:5]
                    ],
                    'counterfactuals': [
                        {
                            'changes': cf.feature_changes,
                            'summary': cf.change_summary,
                            'feasibility': cf.feasibility_score
                        }
                        for cf in explanation.counterfactuals[:3]
                    ]
                }
                
        except Exception as e:
            logger.error(f"Failed to generate explanation for incident {incident_id}: {e}")
            return None


# Global learning pipeline instance
learning_pipeline = ContinuousLearningPipeline()
