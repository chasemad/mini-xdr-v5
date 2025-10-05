"""
Effectiveness Tracker for Mini-XDR
Real-time tracking of response effectiveness with machine learning optimization.
"""

import logging
import asyncio
import json
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
import statistics

from sqlalchemy import select, and_, func, desc, update
from sqlalchemy.ext.asyncio import AsyncSession

from .models import (
    ResponseWorkflow, AdvancedResponseAction, ResponseImpactMetrics,
    Incident, Event
)
from .config import settings

logger = logging.getLogger(__name__)


@dataclass
class EffectivenessMetric:
    """Individual effectiveness metric"""
    metric_id: str
    action_type: str
    success_rate: float
    confidence_score: float
    sample_size: int
    last_updated: datetime
    trend: str  # improving, stable, declining


@dataclass
class ActionPerformance:
    """Performance data for a specific action"""
    action_type: str
    total_executions: int
    successful_executions: int
    failed_executions: int
    average_duration: float
    success_rate: float
    confidence_interval: Tuple[float, float]
    effectiveness_trend: str
    last_execution: Optional[datetime]


class EffectivenessTracker:
    """
    Real-time effectiveness tracking system that monitors response performance
    and provides continuous optimization feedback.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Tracking configuration
        self.tracking_config = {
            "min_sample_size": 3,
            "confidence_level": 0.95,
            "trend_window_days": 30,
            "effectiveness_threshold": 0.8,
            "update_frequency_minutes": 5
        }
        
        # Real-time metrics storage
        self.live_metrics: Dict[str, EffectivenessMetric] = {}
        self.performance_history: Dict[str, List[ActionPerformance]] = defaultdict(list)
        
        # Tracking state
        self.last_update = datetime.now(timezone.utc)
        self.tracking_active = False
        
    async def start_tracking(self, db_session: AsyncSession):
        """Start real-time effectiveness tracking"""
        try:
            self.tracking_active = True
            
            # Initialize baseline metrics
            await self._initialize_baseline_metrics(db_session)
            
            # Start tracking loop
            asyncio.create_task(self._tracking_loop(db_session))
            
            self.logger.info("Effectiveness tracking started")
            
        except Exception as e:
            self.logger.error(f"Failed to start effectiveness tracking: {e}")
    
    async def stop_tracking(self):
        """Stop effectiveness tracking"""
        self.tracking_active = False
        self.logger.info("Effectiveness tracking stopped")
    
    async def track_workflow_execution(
        self,
        workflow_id: str,
        execution_results: Dict[str, Any],
        db_session: AsyncSession
    ) -> Dict[str, Any]:
        """Track effectiveness of a workflow execution in real-time"""
        
        try:
            # Get workflow details
            workflow_result = await db_session.execute(
                select(ResponseWorkflow).where(ResponseWorkflow.workflow_id == workflow_id)
            )
            workflow = workflow_result.scalars().first()
            
            if not workflow:
                return {"success": False, "error": "Workflow not found"}
            
            # Track individual action effectiveness
            action_tracking = []
            for result in execution_results.get("results", []):
                action_effectiveness = await self._track_action_effectiveness(
                    result, workflow, db_session
                )
                action_tracking.append(action_effectiveness)
            
            # Update workflow effectiveness
            workflow_effectiveness = await self._track_workflow_effectiveness(
                workflow, execution_results, db_session
            )
            
            # Update real-time learning
            learning_updates = await self._update_real_time_learning(
                workflow, action_tracking, db_session
            )
            
            # Generate effectiveness insights
            insights = await self._generate_effectiveness_insights(
                workflow, action_tracking, learning_updates
            )
            
            return {
                "success": True,
                "workflow_id": workflow_id,
                "action_tracking": action_tracking,
                "workflow_effectiveness": workflow_effectiveness,
                "learning_updates": learning_updates,
                "effectiveness_insights": insights,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to track workflow effectiveness: {e}")
            return {"success": False, "error": str(e)}
    
    async def get_real_time_effectiveness(
        self,
        action_type: Optional[str] = None,
        timeframe_hours: int = 24
    ) -> Dict[str, Any]:
        """Get real-time effectiveness metrics"""
        
        try:
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=timeframe_hours)
            
            if action_type:
                # Get specific action effectiveness
                if action_type in self.live_metrics:
                    metric = self.live_metrics[action_type]
                    if metric.last_updated > cutoff_time:
                        return {
                            "success": True,
                            "action_type": action_type,
                            "effectiveness": {
                                "success_rate": metric.success_rate,
                                "confidence_score": metric.confidence_score,
                                "sample_size": metric.sample_size,
                                "trend": metric.trend,
                                "last_updated": metric.last_updated.isoformat()
                            }
                        }
            
            # Get all current effectiveness metrics
            current_metrics = {}
            for action_type, metric in self.live_metrics.items():
                if metric.last_updated > cutoff_time:
                    current_metrics[action_type] = {
                        "success_rate": metric.success_rate,
                        "confidence_score": metric.confidence_score,
                        "sample_size": metric.sample_size,
                        "trend": metric.trend,
                        "last_updated": metric.last_updated.isoformat()
                    }
            
            # Calculate overall system effectiveness
            overall_effectiveness = await self._calculate_overall_effectiveness()
            
            return {
                "success": True,
                "timeframe_hours": timeframe_hours,
                "action_metrics": current_metrics,
                "overall_effectiveness": overall_effectiveness,
                "tracking_status": "active" if self.tracking_active else "inactive",
                "last_update": self.last_update.isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get real-time effectiveness: {e}")
            return {"success": False, "error": str(e)}
    
    async def get_effectiveness_trends(
        self,
        db_session: AsyncSession,
        days_back: int = 30,
        action_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Get effectiveness trends over time"""
        
        try:
            start_time = datetime.now(timezone.utc) - timedelta(days=days_back)
            
            # Get historical data
            query = select(ResponseWorkflow).where(
                and_(
                    ResponseWorkflow.created_at >= start_time,
                    ResponseWorkflow.status == "completed"
                )
            ).order_by(ResponseWorkflow.created_at)
            
            workflows_result = await db_session.execute(query)
            workflows = workflows_result.scalars().all()
            
            # Get actions if specific types requested
            if action_types:
                actions_result = await db_session.execute(
                    select(AdvancedResponseAction).where(
                        and_(
                            AdvancedResponseAction.workflow_id.in_([w.id for w in workflows]),
                            AdvancedResponseAction.action_type.in_(action_types)
                        )
                    )
                )
                actions = actions_result.scalars().all()
            else:
                actions_result = await db_session.execute(
                    select(AdvancedResponseAction).where(
                        AdvancedResponseAction.workflow_id.in_([w.id for w in workflows])
                    )
                )
                actions = actions_result.scalars().all()
            
            # Calculate trends by time periods
            trends = await self._calculate_effectiveness_trends_detailed(workflows, actions)
            
            # Calculate action-specific trends
            action_trends = await self._calculate_action_trends(actions, action_types)
            
            # Generate trend predictions
            predictions = await self._generate_trend_predictions(trends, action_trends)
            
            return {
                "success": True,
                "timeframe_days": days_back,
                "trends": trends,
                "action_trends": action_trends,
                "predictions": predictions,
                "trend_quality": self._assess_trend_data_quality(workflows, actions),
                "analysis_timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get effectiveness trends: {e}")
            return {"success": False, "error": str(e)}
    
    async def _initialize_baseline_metrics(self, db_session: AsyncSession):
        """Initialize baseline effectiveness metrics from historical data"""
        
        try:
            # Get last 30 days of data for baseline
            start_time = datetime.now(timezone.utc) - timedelta(days=30)
            
            # Get completed workflows
            workflows_result = await db_session.execute(
                select(ResponseWorkflow).where(
                    and_(
                        ResponseWorkflow.created_at >= start_time,
                        ResponseWorkflow.status == "completed"
                    )
                )
            )
            workflows = workflows_result.scalars().all()
            
            # Get actions
            if workflows:
                actions_result = await db_session.execute(
                    select(AdvancedResponseAction).where(
                        AdvancedResponseAction.workflow_id.in_([w.id for w in workflows])
                    )
                )
                actions = actions_result.scalars().all()
            else:
                actions = []
            
            # Calculate baseline metrics for each action type
            action_stats = defaultdict(lambda: {"total": 0, "successful": 0, "durations": []})
            
            for action in actions:
                stats = action_stats[action.action_type]
                stats["total"] += 1
                
                if action.status == "completed":
                    stats["successful"] += 1
                
                if action.completed_at and action.created_at:
                    duration = (action.completed_at - action.created_at).total_seconds() * 1000
                    stats["durations"].append(duration)
            
            # Create baseline metrics
            for action_type, stats in action_stats.items():
                if stats["total"] >= self.tracking_config["min_sample_size"]:
                    success_rate = stats["successful"] / stats["total"]
                    avg_duration = statistics.mean(stats["durations"]) if stats["durations"] else 0.0
                    
                    self.live_metrics[action_type] = EffectivenessMetric(
                        metric_id=f"baseline_{action_type}",
                        action_type=action_type,
                        success_rate=success_rate,
                        confidence_score=min(stats["total"] / 20.0, 1.0),
                        sample_size=stats["total"],
                        last_updated=datetime.now(timezone.utc),
                        trend="stable"
                    )
            
            self.logger.info(f"Initialized baseline metrics for {len(self.live_metrics)} action types")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize baseline metrics: {e}")
    
    async def _tracking_loop(self, db_session: AsyncSession):
        """Main tracking loop for continuous effectiveness monitoring"""
        
        while self.tracking_active:
            try:
                # Update effectiveness metrics
                await self._update_effectiveness_metrics(db_session)
                
                # Update performance history
                await self._update_performance_history(db_session)
                
                # Clean up old data
                await self._cleanup_old_tracking_data()
                
                self.last_update = datetime.now(timezone.utc)
                
                # Wait for next update cycle
                await asyncio.sleep(self.tracking_config["update_frequency_minutes"] * 60)
                
            except Exception as e:
                self.logger.error(f"Tracking loop error: {e}")
                await asyncio.sleep(60)  # Wait 1 minute on error
    
    async def _track_action_effectiveness(
        self,
        action_result: Dict[str, Any],
        workflow: ResponseWorkflow,
        db_session: AsyncSession
    ) -> Dict[str, Any]:
        """Track effectiveness of individual action execution"""
        
        action_type = action_result.get("action_type")
        success = action_result.get("success", False)
        
        # Update live metrics
        if action_type in self.live_metrics:
            metric = self.live_metrics[action_type]
            
            # Update success rate with exponential moving average
            alpha = 0.1  # Learning rate
            new_success_rate = (1 - alpha) * metric.success_rate + alpha * (1.0 if success else 0.0)
            
            # Update metric
            metric.success_rate = new_success_rate
            metric.sample_size += 1
            metric.last_updated = datetime.now(timezone.utc)
            metric.confidence_score = min(metric.sample_size / 20.0, 1.0)
            
            # Update trend
            metric.trend = self._calculate_trend_direction(metric, success)
        else:
            # Create new metric
            self.live_metrics[action_type] = EffectivenessMetric(
                metric_id=f"live_{action_type}_{datetime.now().timestamp()}",
                action_type=action_type,
                success_rate=1.0 if success else 0.0,
                confidence_score=0.1,
                sample_size=1,
                last_updated=datetime.now(timezone.utc),
                trend="stable"
            )
        
        return {
            "action_type": action_type,
            "execution_success": success,
            "updated_success_rate": self.live_metrics[action_type].success_rate,
            "sample_size": self.live_metrics[action_type].sample_size,
            "confidence": self.live_metrics[action_type].confidence_score,
            "trend": self.live_metrics[action_type].trend
        }
    
    async def _track_workflow_effectiveness(
        self,
        workflow: ResponseWorkflow,
        execution_results: Dict[str, Any],
        db_session: AsyncSession
    ) -> Dict[str, Any]:
        """Track effectiveness of entire workflow"""
        
        # Calculate workflow-level metrics
        total_actions = len(execution_results.get("results", []))
        successful_actions = len([r for r in execution_results.get("results", []) if r.get("success")])
        
        workflow_success_rate = successful_actions / total_actions if total_actions > 0 else 0.0
        
        # Update workflow in database
        workflow.success_rate = workflow_success_rate
        workflow.completed_at = datetime.now(timezone.utc)
        
        await db_session.commit()
        
        # Track playbook effectiveness
        playbook_name = workflow.playbook_name
        if playbook_name:
            await self._update_playbook_effectiveness(playbook_name, workflow_success_rate)
        
        return {
            "workflow_id": workflow.workflow_id,
            "playbook_name": playbook_name,
            "success_rate": workflow_success_rate,
            "total_actions": total_actions,
            "successful_actions": successful_actions,
            "execution_time_ms": workflow.execution_time_ms,
            "effectiveness_grade": self._grade_effectiveness(workflow_success_rate)
        }
    
    async def _update_real_time_learning(
        self,
        workflow: ResponseWorkflow,
        action_tracking: List[Dict[str, Any]],
        db_session: AsyncSession
    ) -> Dict[str, Any]:
        """Update real-time learning based on execution results"""
        
        learning_updates = {
            "models_updated": [],
            "thresholds_adjusted": [],
            "patterns_discovered": [],
            "optimization_opportunities": []
        }
        
        # Update action success thresholds
        for tracking in action_tracking:
            action_type = tracking["action_type"]
            success_rate = tracking["updated_success_rate"]
            
            # Adjust confidence thresholds based on performance
            if success_rate > 0.9 and tracking["sample_size"] > 10:
                learning_updates["thresholds_adjusted"].append({
                    "action_type": action_type,
                    "adjustment": "increased_automation_threshold",
                    "new_threshold": 0.7,  # Lower threshold for high-performing actions
                    "reason": "High success rate enables more automation"
                })
            elif success_rate < 0.6:
                learning_updates["thresholds_adjusted"].append({
                    "action_type": action_type,
                    "adjustment": "decreased_automation_threshold",
                    "new_threshold": 0.9,  # Higher threshold for poor-performing actions
                    "reason": "Low success rate requires more careful selection"
                })
        
        # Discover new patterns
        if workflow.success_rate and workflow.success_rate > 0.9:
            workflow_pattern = {
                "pattern_type": "high_success_workflow",
                "playbook_name": workflow.playbook_name,
                "success_rate": workflow.success_rate,
                "action_sequence": [t["action_type"] for t in action_tracking],
                "discovery_time": datetime.now(timezone.utc).isoformat()
            }
            learning_updates["patterns_discovered"].append(workflow_pattern)
        
        # Identify optimization opportunities
        slow_actions = [t for t in action_tracking if t.get("execution_time_ms", 0) > 600000]  # > 10 min
        if slow_actions:
            learning_updates["optimization_opportunities"].extend([
                {
                    "type": "performance_optimization",
                    "action_type": action["action_type"],
                    "issue": "slow_execution",
                    "recommendation": "parameter_tuning"
                }
                for action in slow_actions
            ])
        
        return learning_updates
    
    async def _generate_effectiveness_insights(
        self,
        workflow: ResponseWorkflow,
        action_tracking: List[Dict[str, Any]],
        learning_updates: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate actionable insights from effectiveness tracking"""
        
        insights = {
            "workflow_insights": [],
            "action_insights": [],
            "optimization_insights": [],
            "trend_insights": []
        }
        
        # Workflow-level insights
        if workflow.success_rate:
            if workflow.success_rate > 0.9:
                insights["workflow_insights"].append({
                    "type": "high_performance",
                    "message": f"Workflow '{workflow.playbook_name}' achieved exceptional success rate",
                    "recommendation": "Consider this workflow as a template for similar incidents"
                })
            elif workflow.success_rate < 0.6:
                insights["workflow_insights"].append({
                    "type": "performance_concern",
                    "message": f"Workflow '{workflow.playbook_name}' has low success rate",
                    "recommendation": "Review action sequence and parameters for optimization"
                })
        
        # Action-level insights
        for tracking in action_tracking:
            action_type = tracking["action_type"]
            success_rate = tracking["updated_success_rate"]
            sample_size = tracking["sample_size"]
            
            if success_rate > 0.95 and sample_size > 10:
                insights["action_insights"].append({
                    "type": "high_reliability",
                    "action_type": action_type,
                    "message": f"{action_type} shows very high reliability",
                    "recommendation": "Increase automation confidence for this action"
                })
            elif success_rate < 0.5 and sample_size > 5:
                insights["action_insights"].append({
                    "type": "reliability_issue",
                    "action_type": action_type,
                    "message": f"{action_type} has concerning failure rate",
                    "recommendation": "Investigate failure causes and improve implementation"
                })
        
        # Optimization insights from learning updates
        if learning_updates["optimization_opportunities"]:
            insights["optimization_insights"] = learning_updates["optimization_opportunities"]
        
        # Trend insights
        improving_actions = [t for t in action_tracking if t["trend"] == "improving"]
        declining_actions = [t for t in action_tracking if t["trend"] == "declining"]
        
        if improving_actions:
            insights["trend_insights"].append({
                "type": "positive_trend",
                "message": f"{len(improving_actions)} actions showing improvement",
                "actions": [a["action_type"] for a in improving_actions]
            })
        
        if declining_actions:
            insights["trend_insights"].append({
                "type": "negative_trend",
                "message": f"{len(declining_actions)} actions showing decline",
                "actions": [a["action_type"] for a in declining_actions],
                "recommendation": "Immediate review required for declining actions"
            })
        
        return insights
    
    async def _update_effectiveness_metrics(self, db_session: AsyncSession):
        """Update effectiveness metrics from recent executions"""
        
        # Get recent actions (last update cycle)
        cutoff_time = self.last_update
        
        recent_actions_result = await db_session.execute(
            select(AdvancedResponseAction).where(
                AdvancedResponseAction.created_at > cutoff_time
            )
        )
        recent_actions = recent_actions_result.scalars().all()
        
        # Update metrics for each action type
        action_updates = defaultdict(lambda: {"total": 0, "successful": 0})
        
        for action in recent_actions:
            action_type = action.action_type
            action_updates[action_type]["total"] += 1
            
            if action.status == "completed":
                action_updates[action_type]["successful"] += 1
        
        # Apply updates to live metrics
        for action_type, updates in action_updates.items():
            if action_type in self.live_metrics:
                metric = self.live_metrics[action_type]
                
                # Update with new data
                total_new = metric.sample_size + updates["total"]
                successful_new = metric.sample_size * metric.success_rate + updates["successful"]
                
                metric.success_rate = successful_new / total_new if total_new > 0 else 0.0
                metric.sample_size = total_new
                metric.last_updated = datetime.now(timezone.utc)
                
                # Update trend
                recent_success_rate = updates["successful"] / updates["total"] if updates["total"] > 0 else 0.0
                metric.trend = self._update_trend_direction(metric, recent_success_rate)
    
    async def _update_performance_history(self, db_session: AsyncSession):
        """Update performance history for trending analysis"""
        
        current_time = datetime.now(timezone.utc)
        
        # Create performance snapshot for each tracked action
        for action_type, metric in self.live_metrics.items():
            performance = ActionPerformance(
                action_type=action_type,
                total_executions=metric.sample_size,
                successful_executions=int(metric.sample_size * metric.success_rate),
                failed_executions=int(metric.sample_size * (1 - metric.success_rate)),
                average_duration=0.0,  # Would calculate from recent executions
                success_rate=metric.success_rate,
                confidence_interval=self._calculate_confidence_interval(metric),
                effectiveness_trend=metric.trend,
                last_execution=current_time
            )
            
            # Add to history (keep last 100 data points)
            history = self.performance_history[action_type]
            history.append(performance)
            if len(history) > 100:
                history.pop(0)
    
    async def _cleanup_old_tracking_data(self):
        """Clean up old tracking data to prevent memory bloat"""
        
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=24)
        
        # Clean up old metrics
        expired_metrics = [
            action_type for action_type, metric in self.live_metrics.items()
            if metric.last_updated < cutoff_time
        ]
        
        for action_type in expired_metrics:
            del self.live_metrics[action_type]
        
        # Clean up old performance history
        for action_type in list(self.performance_history.keys()):
            history = self.performance_history[action_type]
            recent_history = [p for p in history if p.last_execution and p.last_execution > cutoff_time]
            
            if recent_history:
                self.performance_history[action_type] = recent_history
            else:
                del self.performance_history[action_type]
    
    async def _calculate_overall_effectiveness(self) -> Dict[str, Any]:
        """Calculate overall system effectiveness"""
        
        if not self.live_metrics:
            return {"overall_score": 0.0, "grade": "No Data"}
        
        # Weighted average based on sample sizes
        total_weight = sum(metric.sample_size for metric in self.live_metrics.values())
        weighted_success = sum(
            metric.success_rate * metric.sample_size 
            for metric in self.live_metrics.values()
        )
        
        overall_success_rate = weighted_success / total_weight if total_weight > 0 else 0.0
        
        # Calculate confidence
        min_confidence = min(metric.confidence_score for metric in self.live_metrics.values())
        avg_confidence = statistics.mean([metric.confidence_score for metric in self.live_metrics.values()])
        
        # Assess trends
        improving_count = len([m for m in self.live_metrics.values() if m.trend == "improving"])
        declining_count = len([m for m in self.live_metrics.values() if m.trend == "declining"])
        
        overall_trend = "improving" if improving_count > declining_count else "declining" if declining_count > improving_count else "stable"
        
        return {
            "overall_score": overall_success_rate,
            "grade": self._grade_effectiveness(overall_success_rate),
            "confidence": avg_confidence,
            "min_confidence": min_confidence,
            "trend": overall_trend,
            "tracked_actions": len(self.live_metrics),
            "improving_actions": improving_count,
            "declining_actions": declining_count,
            "total_sample_size": sum(metric.sample_size for metric in self.live_metrics.values())
        }
    
    # Helper methods
    def _calculate_trend_direction(self, metric: EffectivenessMetric, new_success: bool) -> str:
        """Calculate trend direction based on recent performance"""
        
        # Simple trend calculation based on recent vs historical performance
        recent_impact = 0.1  # Weight for new data point
        historical_rate = metric.success_rate
        new_rate = historical_rate * (1 - recent_impact) + (1.0 if new_success else 0.0) * recent_impact
        
        if new_rate > historical_rate + 0.05:
            return "improving"
        elif new_rate < historical_rate - 0.05:
            return "declining"
        else:
            return "stable"
    
    def _update_trend_direction(self, metric: EffectivenessMetric, recent_success_rate: float) -> str:
        """Update trend direction based on recent performance"""
        
        if recent_success_rate > metric.success_rate + 0.1:
            return "improving"
        elif recent_success_rate < metric.success_rate - 0.1:
            return "declining"
        else:
            return metric.trend  # Keep existing trend
    
    def _calculate_confidence_interval(self, metric: EffectivenessMetric) -> Tuple[float, float]:
        """Calculate confidence interval for success rate"""
        
        # Simplified confidence interval calculation
        z_score = 1.96  # 95% confidence
        n = metric.sample_size
        p = metric.success_rate
        
        if n == 0:
            return (0.0, 0.0)
        
        margin_error = z_score * ((p * (1 - p)) / n) ** 0.5
        
        return (
            max(0.0, p - margin_error),
            min(1.0, p + margin_error)
        )
    
    def _grade_effectiveness(self, success_rate: float) -> str:
        """Convert success rate to effectiveness grade"""
        if success_rate >= 0.95: return "Excellent"
        elif success_rate >= 0.85: return "Good"
        elif success_rate >= 0.75: return "Satisfactory"
        elif success_rate >= 0.65: return "Needs Improvement"
        else: return "Poor"
    
    async def _update_playbook_effectiveness(self, playbook_name: str, success_rate: float):
        """Update effectiveness tracking for specific playbook"""
        
        # In a full implementation, this would update playbook performance metrics
        # For now, just log the information
        self.logger.info(f"Playbook '{playbook_name}' executed with {success_rate:.1%} success rate")
    
    async def _calculate_effectiveness_trends_detailed(
        self, 
        workflows: List[ResponseWorkflow], 
        actions: List[AdvancedResponseAction]
    ) -> Dict[str, Any]:
        """Calculate detailed effectiveness trends"""
        
        # Group by time periods (daily)
        daily_metrics = defaultdict(lambda: {"workflows": [], "actions": []})
        
        for workflow in workflows:
            if workflow.created_at:
                date_key = workflow.created_at.date().isoformat()
                daily_metrics[date_key]["workflows"].append(workflow)
        
        for action in actions:
            if action.created_at:
                date_key = action.created_at.date().isoformat()
                daily_metrics[date_key]["actions"].append(action)
        
        # Calculate daily effectiveness
        daily_effectiveness = []
        for date_str, data in sorted(daily_metrics.items()):
            date_workflows = data["workflows"]
            date_actions = data["actions"]
            
            if date_workflows:
                daily_success_rates = [w.success_rate for w in date_workflows if w.success_rate is not None]
                daily_avg_success = statistics.mean(daily_success_rates) if daily_success_rates else 0.0
                
                daily_effectiveness.append({
                    "date": date_str,
                    "success_rate": daily_avg_success,
                    "workflow_count": len(date_workflows),
                    "action_count": len(date_actions)
                })
        
        return {
            "daily_effectiveness": daily_effectiveness,
            "trend_analysis": self._analyze_daily_trends(daily_effectiveness),
            "data_quality": "high" if len(daily_effectiveness) >= 7 else "medium" if len(daily_effectiveness) >= 3 else "low"
        }
    
    async def _calculate_action_trends(
        self, 
        actions: List[AdvancedResponseAction], 
        action_types: Optional[List[str]]
    ) -> Dict[str, Any]:
        """Calculate trends for specific action types"""
        
        action_trends = {}
        
        # Group actions by type
        actions_by_type = defaultdict(list)
        for action in actions:
            if not action_types or action.action_type in action_types:
                actions_by_type[action.action_type].append(action)
        
        # Calculate trends for each action type
        for action_type, type_actions in actions_by_type.items():
            if len(type_actions) >= 5:  # Minimum for trend analysis
                # Sort by creation time
                sorted_actions = sorted(type_actions, key=lambda a: a.created_at or datetime.min)
                
                # Calculate success rate over time
                success_rates = []
                window_size = max(3, len(sorted_actions) // 5)
                
                for i in range(window_size, len(sorted_actions) + 1):
                    window = sorted_actions[i-window_size:i]
                    successful = len([a for a in window if a.status == "completed"])
                    success_rate = successful / len(window)
                    success_rates.append(success_rate)
                
                # Calculate trend
                trend_data = self._calculate_linear_trend(success_rates)
                
                action_trends[action_type] = {
                    "total_executions": len(type_actions),
                    "success_rate_trend": trend_data,
                    "current_success_rate": success_rates[-1] if success_rates else 0.0,
                    "trend_confidence": trend_data.get("confidence", "low")
                }
        
        return action_trends
    
    async def _generate_trend_predictions(
        self, 
        trends: Dict[str, Any], 
        action_trends: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate predictions based on trend analysis"""
        
        predictions = {
            "overall_predictions": {},
            "action_predictions": {},
            "confidence": "medium"
        }
        
        # Overall system predictions
        if trends.get("daily_effectiveness"):
            recent_trend = trends["daily_effectiveness"][-7:] if len(trends["daily_effectiveness"]) >= 7 else trends["daily_effectiveness"]
            if recent_trend:
                avg_recent_success = statistics.mean([d["success_rate"] for d in recent_trend])
                predictions["overall_predictions"] = {
                    "predicted_success_rate_24h": avg_recent_success,
                    "predicted_trend": "stable",  # Simplified
                    "confidence": 0.6
                }
        
        # Action-specific predictions
        for action_type, trend_data in action_trends.items():
            current_rate = trend_data["current_success_rate"]
            trend_direction = trend_data["success_rate_trend"]["direction"]
            
            if trend_direction == "improving":
                predicted_rate = min(current_rate + 0.05, 1.0)
            elif trend_direction == "declining":
                predicted_rate = max(current_rate - 0.05, 0.0)
            else:
                predicted_rate = current_rate
            
            predictions["action_predictions"][action_type] = {
                "predicted_success_rate": predicted_rate,
                "trend_direction": trend_direction,
                "confidence": trend_data["trend_confidence"]
            }
        
        return predictions
    
    def _assess_trend_data_quality(
        self, 
        workflows: List[ResponseWorkflow], 
        actions: List[AdvancedResponseAction]
    ) -> Dict[str, Any]:
        """Assess quality of trend data"""
        
        return {
            "workflow_sample_size": len(workflows),
            "action_sample_size": len(actions),
            "time_span_days": (
                (max(w.created_at for w in workflows if w.created_at) - 
                 min(w.created_at for w in workflows if w.created_at)).days
                if workflows else 0
            ),
            "data_completeness": len([w for w in workflows if w.success_rate is not None]) / len(workflows) if workflows else 0,
            "quality_grade": "High" if len(workflows) >= 50 else "Medium" if len(workflows) >= 20 else "Low"
        }
    
    def _analyze_daily_trends(self, daily_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze daily effectiveness trends"""
        
        if len(daily_data) < 3:
            return {"trend": "insufficient_data"}
        
        success_rates = [d["success_rate"] for d in daily_data]
        trend = self._calculate_linear_trend(success_rates)
        
        return {
            "linear_trend": trend,
            "volatility": statistics.stdev(success_rates) if len(success_rates) > 1 else 0.0,
            "best_day": max(daily_data, key=lambda d: d["success_rate"]) if daily_data else None,
            "worst_day": min(daily_data, key=lambda d: d["success_rate"]) if daily_data else None
        }


# Global instance
effectiveness_tracker = EffectivenessTracker()


async def get_effectiveness_tracker() -> EffectivenessTracker:
    """Get the global effectiveness tracker instance"""
    return effectiveness_tracker







