"""
Response Analytics Engine for Mini-XDR
Comprehensive analytics calculation engine for response effectiveness,
performance monitoring, and optimization insights.
"""

import logging
import asyncio
import json
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import statistics

from sqlalchemy import select, and_, func, desc, asc, text
from sqlalchemy.ext.asyncio import AsyncSession

from .models import (
    Incident, ResponseWorkflow, AdvancedResponseAction, 
    ResponseImpactMetrics, Event, Action
)
from .config import settings

logger = logging.getLogger(__name__)


@dataclass
class ResponseMetrics:
    """Response performance metrics"""
    total_workflows: int
    successful_workflows: int
    average_response_time: float
    success_rate: float
    false_positive_rate: float
    mean_time_to_containment: float
    cost_effectiveness_score: float


@dataclass
class EffectivenessAnalysis:
    """Response effectiveness analysis"""
    action_effectiveness: Dict[str, float]
    workflow_effectiveness: Dict[str, float]
    trend_analysis: Dict[str, Any]
    improvement_recommendations: List[str]
    comparative_analysis: Dict[str, Any]


@dataclass
class TrendData:
    """Trend analysis data point"""
    timestamp: datetime
    success_rate: float
    response_time: float
    incident_volume: int
    effectiveness_score: float


class AnalyticsTimeframe(str, Enum):
    LAST_24H = "last_24h"
    LAST_7D = "last_7d"
    LAST_30D = "last_30d"
    LAST_90D = "last_90d"
    CUSTOM = "custom"


class MetricType(str, Enum):
    EFFECTIVENESS = "effectiveness"
    PERFORMANCE = "performance"
    BUSINESS_IMPACT = "business_impact"
    COMPLIANCE = "compliance"
    TRENDS = "trends"


class ResponseAnalyticsEngine:
    """
    Comprehensive analytics calculation engine for response performance,
    effectiveness monitoring, and optimization insights.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Analytics configuration
        self.analytics_config = {
            "effectiveness_threshold": 0.8,
            "performance_baseline": 300,  # 5 minutes
            "trend_sample_size": 50,
            "statistical_confidence": 0.95,
            "outlier_threshold": 2.0  # Standard deviations
        }
        
        # Metric calculation weights
        self.metric_weights = {
            "success_rate": 0.35,
            "response_time": 0.25,
            "false_positive_rate": 0.20,
            "business_impact": 0.15,
            "compliance_score": 0.05
        }
        
        # Caching for expensive calculations
        self.analytics_cache = {}
        self.cache_ttl = 300  # 5 minutes
    
    async def calculate_comprehensive_metrics(
        self,
        db_session: AsyncSession,
        timeframe: AnalyticsTimeframe = AnalyticsTimeframe.LAST_7D,
        workflow_id: Optional[str] = None,
        incident_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive response metrics for specified timeframe
        """
        try:
            # Define time range
            time_range = self._get_time_range(timeframe)
            
            # Get base data
            base_data = await self._get_base_analytics_data(
                db_session, time_range, workflow_id, incident_id
            )
            
            if not base_data["workflows"]:
                return {
                    "success": True,
                    "metrics": self._get_empty_metrics(),
                    "message": "No data available for specified criteria"
                }
            
            # Calculate core metrics in parallel
            metrics_tasks = [
                self._calculate_response_metrics(base_data),
                self._calculate_effectiveness_analysis(base_data, db_session),
                self._calculate_trend_analysis(base_data, db_session),
                self._calculate_business_impact_metrics(base_data),
                self._calculate_compliance_metrics(base_data)
            ]
            
            results = await asyncio.gather(*metrics_tasks, return_exceptions=True)
            
            # Combine results
            comprehensive_metrics = {
                "success": True,
                "timeframe": timeframe.value,
                "analysis_timestamp": datetime.now(timezone.utc).isoformat(),
                "data_quality": self._assess_data_quality(base_data),
                "response_metrics": results[0] if not isinstance(results[0], Exception) else {},
                "effectiveness_analysis": results[1] if not isinstance(results[1], Exception) else {},
                "trend_analysis": results[2] if not isinstance(results[2], Exception) else {},
                "business_impact": results[3] if not isinstance(results[3], Exception) else {},
                "compliance_metrics": results[4] if not isinstance(results[4], Exception) else {},
                "optimization_opportunities": await self._identify_optimization_opportunities(base_data),
                "executive_summary": self._generate_executive_summary(results),
                "recommendations": await self._generate_analytics_recommendations(results, base_data)
            }
            
            return comprehensive_metrics
            
        except Exception as e:
            self.logger.error(f"Failed to calculate comprehensive metrics: {e}")
            return {"success": False, "error": str(e)}
    
    async def _get_base_analytics_data(
        self,
        db_session: AsyncSession,
        time_range: Tuple[datetime, datetime],
        workflow_id: Optional[str] = None,
        incident_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """Get base data for analytics calculations"""
        
        start_time, end_time = time_range
        
        # Build base queries
        workflow_query = select(ResponseWorkflow).where(
            and_(
                ResponseWorkflow.created_at >= start_time,
                ResponseWorkflow.created_at <= end_time
            )
        )
        
        if workflow_id:
            workflow_query = workflow_query.where(ResponseWorkflow.workflow_id == workflow_id)
        
        if incident_id:
            workflow_query = workflow_query.where(ResponseWorkflow.incident_id == incident_id)
        
        # Execute queries
        workflows_result = await db_session.execute(workflow_query)
        workflows = workflows_result.scalars().all()
        
        # Get related actions
        workflow_ids = [w.id for w in workflows]
        if workflow_ids:
            actions_result = await db_session.execute(
                select(AdvancedResponseAction).where(
                    AdvancedResponseAction.workflow_id.in_(workflow_ids)
                )
            )
            actions = actions_result.scalars().all()
            
            # Get impact metrics
            metrics_result = await db_session.execute(
                select(ResponseImpactMetrics).where(
                    ResponseImpactMetrics.workflow_id.in_(workflow_ids)
                )
            )
            impact_metrics = metrics_result.scalars().all()
        else:
            actions = []
            impact_metrics = []
        
        # Get related incidents
        incident_ids = list(set(w.incident_id for w in workflows))
        if incident_ids:
            incidents_result = await db_session.execute(
                select(Incident).where(Incident.id.in_(incident_ids))
            )
            incidents = incidents_result.scalars().all()
        else:
            incidents = []
        
        return {
            "workflows": workflows,
            "actions": actions,
            "impact_metrics": impact_metrics,
            "incidents": incidents,
            "time_range": time_range,
            "total_records": len(workflows)
        }
    
    async def _calculate_response_metrics(self, base_data: Dict[str, Any]) -> ResponseMetrics:
        """Calculate core response performance metrics"""
        
        workflows = base_data["workflows"]
        actions = base_data["actions"]
        impact_metrics = base_data["impact_metrics"]
        
        if not workflows:
            return ResponseMetrics(0, 0, 0.0, 0.0, 0.0, 0.0, 0.0)
        
        # Core calculations
        total_workflows = len(workflows)
        successful_workflows = len([w for w in workflows if w.success_rate and w.success_rate > 0.8])
        
        # Response time calculations
        response_times = [w.execution_time_ms for w in workflows if w.execution_time_ms]
        average_response_time = statistics.mean(response_times) if response_times else 0.0
        
        # Success rate calculation
        success_rates = [w.success_rate for w in workflows if w.success_rate is not None]
        overall_success_rate = statistics.mean(success_rates) if success_rates else 0.0
        
        # False positive calculation (from impact metrics)
        false_positives = sum(m.false_positives for m in impact_metrics)
        total_responses = sum(m.attacks_blocked + m.false_positives for m in impact_metrics)
        false_positive_rate = false_positives / total_responses if total_responses > 0 else 0.0
        
        # Mean time to containment (simplified)
        containment_times = [w.execution_time_ms for w in workflows if w.status == "completed"]
        mean_time_to_containment = statistics.mean(containment_times) if containment_times else 0.0
        
        # Cost effectiveness (simplified calculation)
        total_cost = sum(m.cost_impact_usd for m in impact_metrics)
        attacks_blocked = sum(m.attacks_blocked for m in impact_metrics)
        cost_effectiveness = attacks_blocked / max(total_cost, 1) if total_cost > 0 else attacks_blocked
        
        return ResponseMetrics(
            total_workflows=total_workflows,
            successful_workflows=successful_workflows,
            average_response_time=average_response_time,
            success_rate=overall_success_rate,
            false_positive_rate=false_positive_rate,
            mean_time_to_containment=mean_time_to_containment,
            cost_effectiveness_score=cost_effectiveness
        )
    
    async def _calculate_effectiveness_analysis(
        self, 
        base_data: Dict[str, Any], 
        db_session: AsyncSession
    ) -> EffectivenessAnalysis:
        """Calculate detailed effectiveness analysis"""
        
        workflows = base_data["workflows"]
        actions = base_data["actions"]
        
        # Action effectiveness analysis
        action_stats = {}
        for action in actions:
            action_type = action.action_type
            if action_type not in action_stats:
                action_stats[action_type] = {"total": 0, "successful": 0}
            
            action_stats[action_type]["total"] += 1
            if action.status == "completed":
                action_stats[action_type]["successful"] += 1
        
        action_effectiveness = {}
        for action_type, stats in action_stats.items():
            if stats["total"] >= 3:  # Minimum sample size
                effectiveness = stats["successful"] / stats["total"]
                action_effectiveness[action_type] = effectiveness
        
        # Workflow effectiveness by playbook
        workflow_stats = {}
        for workflow in workflows:
            playbook = workflow.playbook_name
            if playbook not in workflow_stats:
                workflow_stats[playbook] = {"total": 0, "successful": 0, "success_rates": []}
            
            workflow_stats[playbook]["total"] += 1
            if workflow.success_rate:
                workflow_stats[playbook]["success_rates"].append(workflow.success_rate)
                if workflow.success_rate > 0.8:
                    workflow_stats[playbook]["successful"] += 1
        
        workflow_effectiveness = {}
        for playbook, stats in workflow_stats.items():
            if stats["total"] >= 2:
                avg_success = statistics.mean(stats["success_rates"]) if stats["success_rates"] else 0.0
                workflow_effectiveness[playbook] = avg_success
        
        # Trend analysis
        trend_analysis = await self._calculate_effectiveness_trends(workflows)
        
        # Improvement recommendations
        improvement_recommendations = self._generate_improvement_recommendations(
            action_effectiveness, workflow_effectiveness
        )
        
        # Comparative analysis
        comparative_analysis = await self._calculate_comparative_analysis(workflows, db_session)
        
        return EffectivenessAnalysis(
            action_effectiveness=action_effectiveness,
            workflow_effectiveness=workflow_effectiveness,
            trend_analysis=trend_analysis,
            improvement_recommendations=improvement_recommendations,
            comparative_analysis=comparative_analysis
        )
    
    async def _calculate_trend_analysis(
        self, 
        base_data: Dict[str, Any], 
        db_session: AsyncSession
    ) -> Dict[str, Any]:
        """Calculate trend analysis for response metrics"""
        
        workflows = base_data["workflows"]
        time_range = base_data["time_range"]
        
        # Create time buckets
        start_time, end_time = time_range
        duration = end_time - start_time
        bucket_size = duration / 24  # 24 time points
        
        trend_buckets = []
        current_time = start_time
        
        while current_time < end_time:
            bucket_end = current_time + bucket_size
            bucket_workflows = [
                w for w in workflows 
                if w.created_at and current_time <= w.created_at < bucket_end
            ]
            
            if bucket_workflows:
                success_rates = [w.success_rate for w in bucket_workflows if w.success_rate]
                response_times = [w.execution_time_ms for w in bucket_workflows if w.execution_time_ms]
                
                trend_point = TrendData(
                    timestamp=current_time,
                    success_rate=statistics.mean(success_rates) if success_rates else 0.0,
                    response_time=statistics.mean(response_times) if response_times else 0.0,
                    incident_volume=len(bucket_workflows),
                    effectiveness_score=self._calculate_bucket_effectiveness(bucket_workflows)
                )
                trend_buckets.append(trend_point)
            
            current_time = bucket_end
        
        # Calculate trend statistics
        if len(trend_buckets) >= 2:
            success_trend = self._calculate_linear_trend([t.success_rate for t in trend_buckets])
            response_time_trend = self._calculate_linear_trend([t.response_time for t in trend_buckets])
            volume_trend = self._calculate_linear_trend([float(t.incident_volume) for t in trend_buckets])
        else:
            success_trend = {"slope": 0.0, "direction": "stable"}
            response_time_trend = {"slope": 0.0, "direction": "stable"}
            volume_trend = {"slope": 0.0, "direction": "stable"}
        
        return {
            "trend_data": [
                {
                    "timestamp": t.timestamp.isoformat(),
                    "success_rate": t.success_rate,
                    "response_time": t.response_time,
                    "incident_volume": t.incident_volume,
                    "effectiveness_score": t.effectiveness_score
                }
                for t in trend_buckets
            ],
            "success_rate_trend": success_trend,
            "response_time_trend": response_time_trend,
            "incident_volume_trend": volume_trend,
            "trend_quality": self._assess_trend_quality(trend_buckets),
            "predictions": self._generate_trend_predictions(trend_buckets)
        }
    
    async def _calculate_business_impact_metrics(self, base_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate business impact metrics"""
        
        impact_metrics = base_data["impact_metrics"]
        workflows = base_data["workflows"]
        
        # Cost impact analysis
        total_cost_impact = sum(m.cost_impact_usd for m in impact_metrics)
        total_downtime = sum(m.downtime_minutes for m in impact_metrics)
        systems_affected = sum(m.systems_affected for m in impact_metrics)
        users_affected = sum(m.users_affected for m in impact_metrics)
        
        # ROI calculation (attacks blocked vs cost)
        attacks_blocked = sum(m.attacks_blocked for m in impact_metrics)
        estimated_attack_cost = attacks_blocked * 10000  # $10K per blocked attack
        roi = (estimated_attack_cost - total_cost_impact) / max(total_cost_impact, 1) if total_cost_impact > 0 else 0
        
        # Efficiency metrics
        avg_cost_per_response = total_cost_impact / len(workflows) if workflows else 0
        avg_downtime_per_incident = total_downtime / len(workflows) if workflows else 0
        
        return {
            "total_cost_impact_usd": total_cost_impact,
            "total_downtime_minutes": total_downtime,
            "systems_affected": systems_affected,
            "users_affected": users_affected,
            "estimated_roi": roi,
            "avg_cost_per_response": avg_cost_per_response,
            "avg_downtime_per_incident": avg_downtime_per_incident,
            "cost_effectiveness_grade": self._grade_cost_effectiveness(roi),
            "business_impact_trends": await self._calculate_business_impact_trends(base_data)
        }
    
    async def _calculate_compliance_metrics(self, base_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate compliance-related metrics"""
        
        impact_metrics = base_data["impact_metrics"]
        workflows = base_data["workflows"]
        
        # Compliance impact distribution
        compliance_distribution = {"none": 0, "low": 0, "medium": 0, "high": 0, "critical": 0}
        for metric in impact_metrics:
            impact_level = metric.compliance_impact or "none"
            if impact_level in compliance_distribution:
                compliance_distribution[impact_level] += 1
        
        # Response time compliance (regulatory requirements)
        gdpr_compliant = len([w for w in workflows if w.execution_time_ms and w.execution_time_ms <= 72 * 3600 * 1000])  # 72 hours
        sox_compliant = len([w for w in workflows if w.execution_time_ms and w.execution_time_ms <= 24 * 3600 * 1000])  # 24 hours
        
        total_workflows = len(workflows)
        
        return {
            "compliance_impact_distribution": compliance_distribution,
            "gdpr_compliance_rate": gdpr_compliant / total_workflows if total_workflows > 0 else 0,
            "sox_compliance_rate": sox_compliant / total_workflows if total_workflows > 0 else 0,
            "regulatory_metrics": {
                "gdpr_72h_compliance": gdpr_compliant,
                "sox_24h_compliance": sox_compliant,
                "total_assessments": total_workflows
            },
            "audit_readiness_score": self._calculate_audit_readiness(workflows, impact_metrics),
            "compliance_trends": await self._calculate_compliance_trends(base_data)
        }
    
    def _get_time_range(self, timeframe: AnalyticsTimeframe) -> Tuple[datetime, datetime]:
        """Get time range for analytics timeframe"""
        
        end_time = datetime.now(timezone.utc)
        
        if timeframe == AnalyticsTimeframe.LAST_24H:
            start_time = end_time - timedelta(hours=24)
        elif timeframe == AnalyticsTimeframe.LAST_7D:
            start_time = end_time - timedelta(days=7)
        elif timeframe == AnalyticsTimeframe.LAST_30D:
            start_time = end_time - timedelta(days=30)
        elif timeframe == AnalyticsTimeframe.LAST_90D:
            start_time = end_time - timedelta(days=90)
        else:
            start_time = end_time - timedelta(days=7)  # Default
        
        return start_time, end_time
    
    def _get_empty_metrics(self) -> Dict[str, Any]:
        """Return empty metrics structure"""
        return {
            "response_metrics": ResponseMetrics(0, 0, 0.0, 0.0, 0.0, 0.0, 0.0),
            "effectiveness_analysis": EffectivenessAnalysis({}, {}, {}, [], {}),
            "trend_analysis": {"trend_data": []},
            "business_impact": {},
            "compliance_metrics": {}
        }
    
    def _assess_data_quality(self, base_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess the quality of analytics data"""
        
        workflows = base_data["workflows"]
        actions = base_data["actions"]
        
        completeness_score = 0.0
        quality_factors = []
        
        # Sample size quality
        if len(workflows) >= 50:
            completeness_score += 0.3
            quality_factors.append("Large sample size")
        elif len(workflows) >= 20:
            completeness_score += 0.2
            quality_factors.append("Adequate sample size")
        elif len(workflows) >= 5:
            completeness_score += 0.1
            quality_factors.append("Small sample size")
        
        # Data completeness
        complete_workflows = len([w for w in workflows if w.success_rate is not None and w.execution_time_ms])
        if complete_workflows == len(workflows):
            completeness_score += 0.3
            quality_factors.append("Complete workflow data")
        elif complete_workflows > len(workflows) * 0.8:
            completeness_score += 0.2
            quality_factors.append("Mostly complete data")
        
        # Recency quality
        recent_workflows = len([
            w for w in workflows 
            if w.created_at and w.created_at > datetime.now(timezone.utc) - timedelta(days=7)
        ])
        if recent_workflows > len(workflows) * 0.5:
            completeness_score += 0.2
            quality_factors.append("Recent data available")
        
        # Diversity quality (different playbooks/actions)
        unique_playbooks = len(set(w.playbook_name for w in workflows))
        unique_actions = len(set(a.action_type for a in actions))
        if unique_playbooks >= 5 and unique_actions >= 10:
            completeness_score += 0.2
            quality_factors.append("Diverse response patterns")
        
        return {
            "quality_score": min(completeness_score, 1.0),
            "quality_grade": self._score_to_grade(completeness_score),
            "quality_factors": quality_factors,
            "sample_size": len(workflows),
            "completeness_percentage": (complete_workflows / len(workflows)) * 100 if workflows else 0,
            "data_span_days": (max(w.created_at for w in workflows if w.created_at) - 
                             min(w.created_at for w in workflows if w.created_at)).days if workflows else 0
        }
    
    async def _calculate_effectiveness_trends(self, workflows: List[ResponseWorkflow]) -> Dict[str, Any]:
        """Calculate effectiveness trends over time"""
        
        if len(workflows) < 5:
            return {"insufficient_data": True, "trend": "stable"}
        
        # Sort workflows by creation time
        sorted_workflows = sorted(workflows, key=lambda w: w.created_at or datetime.min)
        
        # Calculate rolling averages
        window_size = max(5, len(workflows) // 5)
        rolling_averages = []
        
        for i in range(window_size, len(sorted_workflows) + 1):
            window = sorted_workflows[i-window_size:i]
            success_rates = [w.success_rate for w in window if w.success_rate is not None]
            if success_rates:
                avg_success = statistics.mean(success_rates)
                rolling_averages.append({
                    "timestamp": window[-1].created_at.isoformat() if window[-1].created_at else None,
                    "success_rate": avg_success,
                    "sample_size": len(success_rates)
                })
        
        # Calculate trend direction
        if len(rolling_averages) >= 2:
            first_half = rolling_averages[:len(rolling_averages)//2]
            second_half = rolling_averages[len(rolling_averages)//2:]
            
            first_avg = statistics.mean([ra["success_rate"] for ra in first_half])
            second_avg = statistics.mean([ra["success_rate"] for ra in second_half])
            
            improvement = second_avg - first_avg
            
            if improvement > 0.1:
                trend_direction = "improving"
            elif improvement < -0.1:
                trend_direction = "declining"
            else:
                trend_direction = "stable"
        else:
            trend_direction = "stable"
            improvement = 0.0
        
        return {
            "rolling_averages": rolling_averages,
            "trend_direction": trend_direction,
            "improvement_rate": improvement,
            "window_size": window_size,
            "confidence": min(len(rolling_averages) / 10.0, 1.0)
        }
    
    def _generate_improvement_recommendations(
        self, 
        action_effectiveness: Dict[str, float], 
        workflow_effectiveness: Dict[str, float]
    ) -> List[str]:
        """Generate improvement recommendations based on analytics"""
        
        recommendations = []
        
        # Action-based recommendations
        if action_effectiveness:
            best_actions = sorted(action_effectiveness.items(), key=lambda x: x[1], reverse=True)
            worst_actions = sorted(action_effectiveness.items(), key=lambda x: x[1])
            
            if best_actions and best_actions[0][1] > 0.9:
                recommendations.append(f"Consider using {best_actions[0][0]} more frequently (98% success rate)")
            
            if worst_actions and worst_actions[0][1] < 0.5:
                recommendations.append(f"Review {worst_actions[0][0]} implementation (low success rate)")
        
        # Workflow-based recommendations
        if workflow_effectiveness:
            best_workflows = sorted(workflow_effectiveness.items(), key=lambda x: x[1], reverse=True)
            
            if best_workflows and best_workflows[0][1] > 0.9:
                recommendations.append(f"Use '{best_workflows[0][0]}' as template for similar incidents")
        
        # General recommendations
        recommendations.extend([
            "Consider implementing automated rollback for failed actions",
            "Add more granular success criteria for better measurement",
            "Implement A/B testing for response strategy optimization"
        ])
        
        return recommendations
    
    async def _calculate_comparative_analysis(
        self, 
        workflows: List[ResponseWorkflow], 
        db_session: AsyncSession
    ) -> Dict[str, Any]:
        """Calculate comparative analysis vs industry benchmarks"""
        
        if not workflows:
            return {"insufficient_data": True}
        
        # Calculate our metrics
        our_success_rate = statistics.mean([w.success_rate for w in workflows if w.success_rate])
        our_response_time = statistics.mean([w.execution_time_ms for w in workflows if w.execution_time_ms])
        
        # Industry benchmarks (typical enterprise SOC metrics)
        industry_benchmarks = {
            "success_rate": 0.75,  # 75% industry average
            "response_time_ms": 1800000,  # 30 minutes
            "false_positive_rate": 0.15,  # 15%
            "automation_rate": 0.40  # 40%
        }
        
        # Calculate comparisons
        success_comparison = (our_success_rate / industry_benchmarks["success_rate"] - 1) * 100
        time_comparison = (1 - our_response_time / industry_benchmarks["response_time_ms"]) * 100
        
        return {
            "our_metrics": {
                "success_rate": our_success_rate,
                "avg_response_time_ms": our_response_time
            },
            "industry_benchmarks": industry_benchmarks,
            "comparisons": {
                "success_rate_vs_industry": success_comparison,
                "response_time_vs_industry": time_comparison
            },
            "competitive_position": self._assess_competitive_position(success_comparison, time_comparison),
            "benchmark_sources": ["SANS SOC Survey 2024", "Ponemon Institute", "NIST Cybersecurity Framework"]
        }
    
    async def _identify_optimization_opportunities(self, base_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify optimization opportunities from analytics"""
        
        opportunities = []
        workflows = base_data["workflows"]
        actions = base_data["actions"]
        
        # Identify slow workflows
        slow_workflows = [w for w in workflows if w.execution_time_ms and w.execution_time_ms > 1800000]  # > 30 min
        if slow_workflows:
            opportunities.append({
                "type": "performance_optimization",
                "priority": "high",
                "description": f"{len(slow_workflows)} workflows are taking longer than 30 minutes",
                "potential_improvement": "30-50% faster response times",
                "implementation": "Action parallelization and timeout optimization"
            })
        
        # Identify low-success actions
        action_stats = {}
        for action in actions:
            action_type = action.action_type
            if action_type not in action_stats:
                action_stats[action_type] = {"total": 0, "successful": 0}
            action_stats[action_type]["total"] += 1
            if action.status == "completed":
                action_stats[action_type]["successful"] += 1
        
        low_success_actions = [
            action for action, stats in action_stats.items()
            if stats["total"] >= 3 and stats["successful"] / stats["total"] < 0.6
        ]
        
        if low_success_actions:
            opportunities.append({
                "type": "effectiveness_improvement",
                "priority": "medium",
                "description": f"Actions with low success rates: {', '.join(low_success_actions)}",
                "potential_improvement": "15-25% higher success rates",
                "implementation": "Parameter tuning and safety condition improvements"
            })
        
        # Identify automation opportunities
        manual_workflows = [w for w in workflows if not w.auto_executed]
        if len(manual_workflows) > len(workflows) * 0.7:
            opportunities.append({
                "type": "automation_opportunity",
                "priority": "medium",
                "description": f"{len(manual_workflows)} workflows could be automated",
                "potential_improvement": "60-80% faster response times",
                "implementation": "Confidence threshold tuning and safety validation"
            })
        
        return opportunities
    
    def _generate_executive_summary(self, analytics_results: List[Any]) -> Dict[str, Any]:
        """Generate executive summary of analytics"""
        
        try:
            response_metrics = analytics_results[0] if len(analytics_results) > 0 and not isinstance(analytics_results[0], Exception) else None
            effectiveness = analytics_results[1] if len(analytics_results) > 1 and not isinstance(analytics_results[1], Exception) else None
            trends = analytics_results[2] if len(analytics_results) > 2 and not isinstance(analytics_results[2], Exception) else None
            
            if not response_metrics:
                return {"summary": "Insufficient data for executive summary"}
            
            # Key insights
            key_insights = []
            
            if response_metrics.success_rate > 0.8:
                key_insights.append("High response effectiveness achieved")
            elif response_metrics.success_rate < 0.6:
                key_insights.append("Response effectiveness needs improvement")
            
            if response_metrics.average_response_time < 300000:  # < 5 minutes
                key_insights.append("Fast response times maintained")
            elif response_metrics.average_response_time > 1800000:  # > 30 minutes
                key_insights.append("Response times exceed target thresholds")
            
            # Performance grade
            overall_score = (
                response_metrics.success_rate * 0.4 +
                (1 - min(response_metrics.false_positive_rate, 0.5) / 0.5) * 0.3 +
                (1 - min(response_metrics.average_response_time / 1800000, 1.0)) * 0.3
            )
            
            performance_grade = self._score_to_grade(overall_score)
            
            return {
                "performance_grade": performance_grade,
                "overall_score": overall_score,
                "key_insights": key_insights,
                "total_workflows_analyzed": response_metrics.total_workflows,
                "summary_period": "Last 7 days",
                "recommendation": self._get_executive_recommendation(overall_score, key_insights)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to generate executive summary: {e}")
            return {"summary": f"Executive summary generation failed: {str(e)}"}
    
    async def _generate_analytics_recommendations(
        self, 
        analytics_results: List[Any], 
        base_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate actionable recommendations from analytics"""
        
        recommendations = []
        
        try:
            response_metrics = analytics_results[0] if len(analytics_results) > 0 else None
            
            if response_metrics and isinstance(response_metrics, ResponseMetrics):
                # Performance recommendations
                if response_metrics.average_response_time > 600000:  # > 10 minutes
                    recommendations.append({
                        "category": "performance",
                        "priority": "high",
                        "title": "Optimize Response Times",
                        "description": "Average response time exceeds 10 minutes",
                        "action": "Implement action parallelization and optimize timeouts",
                        "expected_impact": "30-50% faster responses"
                    })
                
                # Effectiveness recommendations
                if response_metrics.success_rate < 0.8:
                    recommendations.append({
                        "category": "effectiveness",
                        "priority": "high",
                        "title": "Improve Success Rates",
                        "description": f"Success rate is {response_metrics.success_rate:.1%}",
                        "action": "Review failed actions and improve error handling",
                        "expected_impact": "15-25% higher success rates"
                    })
                
                # Cost optimization
                if response_metrics.cost_effectiveness_score < 1.0:
                    recommendations.append({
                        "category": "cost_optimization",
                        "priority": "medium",
                        "title": "Optimize Cost Effectiveness",
                        "description": "Response costs exceed optimal thresholds",
                        "action": "Implement automated action selection and resource optimization",
                        "expected_impact": "20-30% cost reduction"
                    })
        
            # Add strategic recommendations
            recommendations.extend([
                {
                    "category": "strategic",
                    "priority": "medium",
                    "title": "Implement Predictive Analytics",
                    "description": "Proactive threat response capabilities",
                    "action": "Deploy ML models for attack prediction",
                    "expected_impact": "40-60% reduction in successful attacks"
                },
                {
                    "category": "automation",
                    "priority": "medium",
                    "title": "Increase Automation Coverage",
                    "description": "Many workflows still require manual intervention",
                    "action": "Expand confidence thresholds for automated responses",
                    "expected_impact": "50-70% faster incident resolution"
                }
            ])
        
        except Exception as e:
            self.logger.error(f"Failed to generate recommendations: {e}")
        
        return recommendations
    
    # Helper methods
    def _calculate_bucket_effectiveness(self, workflows: List[ResponseWorkflow]) -> float:
        """Calculate effectiveness score for a time bucket"""
        if not workflows:
            return 0.0
        
        success_rates = [w.success_rate for w in workflows if w.success_rate is not None]
        return statistics.mean(success_rates) if success_rates else 0.0
    
    def _calculate_linear_trend(self, values: List[float]) -> Dict[str, Any]:
        """Calculate linear trend for a series of values"""
        if len(values) < 2:
            return {"slope": 0.0, "direction": "stable", "r_squared": 0.0}
        
        # Simple linear regression
        n = len(values)
        x = list(range(n))
        
        x_mean = statistics.mean(x)
        y_mean = statistics.mean(values)
        
        numerator = sum((x[i] - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
        
        slope = numerator / denominator if denominator != 0 else 0.0
        
        # Determine direction
        if slope > 0.01:
            direction = "improving"
        elif slope < -0.01:
            direction = "declining"
        else:
            direction = "stable"
        
        # Calculate R-squared (simplified)
        y_pred = [y_mean + slope * (x[i] - x_mean) for i in range(n)]
        ss_res = sum((values[i] - y_pred[i]) ** 2 for i in range(n))
        ss_tot = sum((values[i] - y_mean) ** 2 for i in range(n))
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
        
        return {
            "slope": slope,
            "direction": direction,
            "r_squared": max(0.0, r_squared),
            "confidence": "high" if r_squared > 0.7 else "medium" if r_squared > 0.4 else "low"
        }
    
    def _assess_trend_quality(self, trend_buckets: List[TrendData]) -> Dict[str, Any]:
        """Assess the quality of trend analysis"""
        
        return {
            "data_points": len(trend_buckets),
            "quality": "high" if len(trend_buckets) >= 20 else "medium" if len(trend_buckets) >= 10 else "low",
            "coverage": "complete" if len(trend_buckets) >= 24 else "partial",
            "reliability": "reliable" if len(trend_buckets) >= 15 else "limited"
        }
    
    def _generate_trend_predictions(self, trend_buckets: List[TrendData]) -> Dict[str, Any]:
        """Generate predictions based on trends"""
        
        if len(trend_buckets) < 5:
            return {"prediction": "insufficient_data"}
        
        # Predict next period performance
        recent_success_rates = [t.success_rate for t in trend_buckets[-5:]]
        recent_response_times = [t.response_time for t in trend_buckets[-5:]]
        
        predicted_success = statistics.mean(recent_success_rates)
        predicted_response_time = statistics.mean(recent_response_times)
        
        return {
            "predicted_success_rate": predicted_success,
            "predicted_response_time": predicted_response_time,
            "confidence": 0.7,  # Medium confidence for simple predictions
            "prediction_window": "next_24_hours"
        }
    
    async def _calculate_business_impact_trends(self, base_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate business impact trends"""
        
        impact_metrics = base_data["impact_metrics"]
        
        if not impact_metrics:
            return {"trend": "no_data"}
        
        # Sort by timestamp
        sorted_metrics = sorted(impact_metrics, key=lambda m: m.created_at)
        
        # Calculate trends
        cost_trend = [m.cost_impact_usd for m in sorted_metrics]
        downtime_trend = [m.downtime_minutes for m in sorted_metrics]
        
        return {
            "cost_trend": self._calculate_linear_trend(cost_trend),
            "downtime_trend": self._calculate_linear_trend(downtime_trend),
            "impact_distribution": self._calculate_impact_distribution(sorted_metrics)
        }
    
    def _calculate_audit_readiness(
        self, 
        workflows: List[ResponseWorkflow], 
        impact_metrics: List[ResponseImpactMetrics]
    ) -> float:
        """Calculate audit readiness score"""
        
        readiness_score = 0.0
        
        # Documentation completeness
        documented_workflows = len([w for w in workflows if w.execution_log])
        if workflows:
            readiness_score += (documented_workflows / len(workflows)) * 0.4
        
        # Compliance tracking
        compliance_tracked = len([m for m in impact_metrics if m.compliance_impact != "none"])
        if impact_metrics:
            readiness_score += (compliance_tracked / len(impact_metrics)) * 0.3
        
        # Response time compliance
        compliant_responses = len([w for w in workflows if w.execution_time_ms and w.execution_time_ms <= 86400000])  # 24h
        if workflows:
            readiness_score += (compliant_responses / len(workflows)) * 0.3
        
        return min(readiness_score, 1.0)
    
    async def _calculate_compliance_trends(self, base_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate compliance-related trends"""
        
        impact_metrics = base_data["impact_metrics"]
        
        if not impact_metrics:
            return {"trend": "no_data"}
        
        # Sort by timestamp
        sorted_metrics = sorted(impact_metrics, key=lambda m: m.created_at)
        
        # Calculate compliance violation rates over time
        violation_rates = []
        window_size = max(5, len(sorted_metrics) // 5)
        
        for i in range(window_size, len(sorted_metrics) + 1):
            window = sorted_metrics[i-window_size:i]
            violations = len([m for m in window if m.compliance_impact in ["high", "critical"]])
            violation_rate = violations / len(window)
            violation_rates.append(violation_rate)
        
        return {
            "violation_rate_trend": self._calculate_linear_trend(violation_rates),
            "current_violation_rate": violation_rates[-1] if violation_rates else 0.0,
            "compliance_grade": self._grade_compliance_performance(violation_rates)
        }
    
    # Utility methods
    def _score_to_grade(self, score: float) -> str:
        """Convert numeric score to letter grade"""
        if score >= 0.9: return "A"
        elif score >= 0.8: return "B"
        elif score >= 0.7: return "C"
        elif score >= 0.6: return "D"
        else: return "F"
    
    def _grade_cost_effectiveness(self, roi: float) -> str:
        """Grade cost effectiveness based on ROI"""
        if roi >= 5.0: return "Excellent"
        elif roi >= 2.0: return "Good"
        elif roi >= 1.0: return "Adequate"
        elif roi >= 0.0: return "Poor"
        else: return "Negative"
    
    def _assess_competitive_position(self, success_comparison: float, time_comparison: float) -> str:
        """Assess competitive position vs industry"""
        
        overall_advantage = (success_comparison + time_comparison) / 2
        
        if overall_advantage > 20:
            return "Industry Leading"
        elif overall_advantage > 10:
            return "Above Average"
        elif overall_advantage > -10:
            return "Industry Average"
        else:
            return "Below Average"
    
    def _calculate_impact_distribution(self, metrics: List[ResponseImpactMetrics]) -> Dict[str, Any]:
        """Calculate distribution of business impacts"""
        
        cost_buckets = {"low": 0, "medium": 0, "high": 0}
        downtime_buckets = {"low": 0, "medium": 0, "high": 0}
        
        for metric in metrics:
            # Cost distribution
            if metric.cost_impact_usd <= 1000:
                cost_buckets["low"] += 1
            elif metric.cost_impact_usd <= 10000:
                cost_buckets["medium"] += 1
            else:
                cost_buckets["high"] += 1
            
            # Downtime distribution
            if metric.downtime_minutes <= 15:
                downtime_buckets["low"] += 1
            elif metric.downtime_minutes <= 60:
                downtime_buckets["medium"] += 1
            else:
                downtime_buckets["high"] += 1
        
        return {
            "cost_distribution": cost_buckets,
            "downtime_distribution": downtime_buckets
        }
    
    def _grade_compliance_performance(self, violation_rates: List[float]) -> str:
        """Grade compliance performance"""
        if not violation_rates:
            return "No Data"
        
        avg_violation_rate = statistics.mean(violation_rates)
        
        if avg_violation_rate <= 0.05:
            return "Excellent"
        elif avg_violation_rate <= 0.10:
            return "Good"
        elif avg_violation_rate <= 0.20:
            return "Adequate"
        else:
            return "Poor"
    
    def _get_executive_recommendation(self, overall_score: float, insights: List[str]) -> str:
        """Get executive recommendation based on performance"""
        
        if overall_score > 0.8:
            return "Maintain current performance and consider expanding automation"
        elif overall_score > 0.6:
            return "Focus on improving response effectiveness and reducing false positives"
        else:
            return "Immediate optimization required - review response strategies and tooling"


# Global instance
response_analytics_engine = ResponseAnalyticsEngine()


async def get_analytics_engine() -> ResponseAnalyticsEngine:
    """Get the global analytics engine instance"""
    return response_analytics_engine





