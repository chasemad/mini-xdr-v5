"""
Learning Response Engine for Mini-XDR
Adaptive response engine that learns from execution results and continuously improves.
"""

import logging
import asyncio
import json
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from sqlalchemy import select, and_, func, desc
from sqlalchemy.ext.asyncio import AsyncSession

from .models import (
    Incident, ResponseWorkflow, AdvancedResponseAction, 
    ResponseImpactMetrics, ResponsePlaybook
)
from .ai_response_advisor import get_ai_advisor, RecommendationConfidence, ResponseStrategy
from .context_analyzer import get_context_analyzer
from .response_optimizer import get_response_optimizer, OptimizationStrategy

logger = logging.getLogger(__name__)


@dataclass
class LearningMetrics:
    """Metrics for adaptive learning performance"""
    total_workflows: int
    successful_workflows: int
    average_improvement: float
    learning_velocity: float
    adaptation_score: float
    confidence_accuracy: float


class LearningResponseEngine:
    """
    Adaptive response engine that learns from execution results and continuously improves
    response strategies using AI and machine learning.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.ai_advisor = None
        self.context_analyzer = None
        self.response_optimizer = None
        self.initialized = False
        
        # Learning configuration
        self.learning_config = {
            "min_samples_for_learning": 5,
            "confidence_threshold": 0.7,
            "adaptation_rate": 0.1,
            "feedback_weight": 0.3,
            "historical_weight": 0.7,
            "max_learning_history_days": 90
        }
        
        # Learning state
        self.learning_history = []
        self.performance_trends = {}
        self.adaptation_patterns = {}
        
    async def initialize(self):
        """Initialize the learning response engine"""
        try:
            # Initialize AI components
            self.ai_advisor = await get_ai_advisor()
            self.context_analyzer = await get_context_analyzer()
            self.response_optimizer = await get_response_optimizer()
            
            self.initialized = True
            self.logger.info("Learning Response Engine initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Learning Response Engine: {e}")
    
    async def generate_adaptive_recommendations(
        self,
        incident_id: int,
        db_session: AsyncSession,
        user_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate adaptive response recommendations that improve over time
        """
        try:
            if not self.initialized:
                await self.initialize()
            
            # Step 1: Get comprehensive context analysis
            context_result = await self.context_analyzer.analyze_comprehensive_context(
                incident_id, db_session, include_predictions=True
            )
            
            if not context_result.get("success"):
                return context_result
            
            context_analysis = context_result["context_analysis"]
            
            # Step 2: Get AI-powered recommendations
            recommendations_result = await self.ai_advisor.get_response_recommendations(
                incident_id, db_session, user_context
            )
            
            if not recommendations_result.get("success"):
                return recommendations_result
            
            # Step 3: Apply learning-based optimization
            optimized_result = await self.response_optimizer.optimize_response_strategy(
                incident_id, 
                recommendations_result["recommendations"],
                db_session,
                OptimizationStrategy.EFFECTIVENESS
            )
            
            if not optimized_result.get("success"):
                # Fall back to original recommendations
                optimized_recommendations = recommendations_result["recommendations"]
            else:
                optimized_recommendations = optimized_result["optimized_recommendations"]
            
            # Step 4: Apply adaptive learning adjustments
            adaptive_recommendations = await self._apply_adaptive_learning(
                optimized_recommendations, context_analysis, db_session
            )
            
            # Step 5: Generate comprehensive response plan
            response_plan = await self._generate_comprehensive_response_plan(
                incident_id, adaptive_recommendations, context_analysis, db_session
            )
            
            return {
                "success": True,
                "incident_id": incident_id,
                "adaptive_recommendations": adaptive_recommendations,
                "response_plan": response_plan,
                "context_analysis": context_analysis,
                "optimization_applied": optimized_result.get("success", False),
                "learning_insights": await self._get_learning_insights(incident_id, db_session),
                "confidence_metrics": self._calculate_confidence_metrics(adaptive_recommendations),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to generate adaptive recommendations: {e}")
            return {"success": False, "error": str(e)}
    
    async def _apply_adaptive_learning(
        self,
        recommendations: List[Dict[str, Any]],
        context_analysis: Dict[str, Any],
        db_session: AsyncSession
    ) -> List[Dict[str, Any]]:
        """Apply adaptive learning adjustments to recommendations"""
        
        # Get learning history for similar contexts
        learning_data = await self._get_relevant_learning_data(context_analysis, db_session)
        
        adapted_recommendations = []
        
        for rec in recommendations:
            # Apply learning adjustments
            adapted_rec = await self._adapt_recommendation(rec, learning_data, context_analysis)
            adapted_recommendations.append(adapted_rec)
        
        # Add learned actions if confidence is high
        learned_actions = await self._get_learned_actions(context_analysis, learning_data)
        for learned_action in learned_actions:
            if learned_action["confidence"] > self.learning_config["confidence_threshold"]:
                adapted_recommendations.append(learned_action)
        
        # Sort by adapted confidence scores
        adapted_recommendations.sort(key=lambda x: x.get("confidence", 0.0), reverse=True)
        
        return adapted_recommendations
    
    async def _adapt_recommendation(
        self,
        recommendation: Dict[str, Any],
        learning_data: Dict[str, Any],
        context_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Adapt a single recommendation based on learning data"""
        
        adapted = recommendation.copy()
        action_type = recommendation.get("action_type")
        
        # Adjust confidence based on historical performance
        historical_performance = learning_data.get("action_performance", {}).get(action_type, {})
        if historical_performance:
            historical_success = historical_performance.get("success_rate", 0.5)
            original_confidence = recommendation.get("confidence", 0.5)
            
            # Weighted average of original confidence and historical performance
            adapted_confidence = (
                original_confidence * (1 - self.learning_config["historical_weight"]) +
                historical_success * self.learning_config["historical_weight"]
            )
            adapted["confidence"] = adapted_confidence
            adapted["learning_adjusted"] = True
            adapted["historical_basis"] = {
                "sample_size": historical_performance.get("sample_size", 0),
                "success_rate": historical_success,
                "avg_duration": historical_performance.get("avg_duration", 0)
            }
        
        # Adjust parameters based on learned optimal values
        parameter_learning = learning_data.get("parameter_optimization", {}).get(action_type, {})
        if parameter_learning:
            adapted["parameters"] = {
                **adapted.get("parameters", {}),
                **parameter_learning.get("optimal_parameters", {})
            }
            adapted["parameter_learning_applied"] = True
        
        # Adjust timing based on learned patterns
        timing_learning = learning_data.get("timing_optimization", {}).get(action_type, {})
        if timing_learning:
            adapted["optimal_timing"] = timing_learning.get("optimal_delay", 0)
            adapted["timing_learning_applied"] = True
        
        return adapted
    
    async def _get_learned_actions(
        self,
        context_analysis: Dict[str, Any],
        learning_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Get additional actions learned from similar contexts"""
        
        learned_actions = []
        
        # Get threat context
        threat_context = context_analysis.get("threat_context", {})
        threat_category = threat_context.get("threat_category", "unknown")
        
        # Find learned patterns for this threat category
        category_learning = learning_data.get("category_patterns", {}).get(threat_category, {})
        
        for action_type, learning_info in category_learning.get("emergent_actions", {}).items():
            if learning_info.get("confidence", 0.0) > self.learning_config["confidence_threshold"]:
                learned_actions.append({
                    "action_type": action_type,
                    "confidence": learning_info["confidence"],
                    "priority": learning_info.get("priority", 5),
                    "learned": True,
                    "learning_basis": {
                        "sample_size": learning_info.get("sample_size", 0),
                        "success_rate": learning_info.get("success_rate", 0.0),
                        "emergence_trend": learning_info.get("trend", "stable")
                    },
                    "parameters": learning_info.get("optimal_parameters", {}),
                    "rationale": f"Learned action with {learning_info['confidence']:.1%} confidence"
                })
        
        return learned_actions
    
    async def _generate_comprehensive_response_plan(
        self,
        incident_id: int,
        recommendations: List[Dict[str, Any]],
        context_analysis: Dict[str, Any],
        db_session: AsyncSession
    ) -> Dict[str, Any]:
        """Generate a comprehensive response plan with multiple execution strategies"""
        
        # Create primary response plan
        primary_plan = {
            "strategy": "primary",
            "actions": recommendations[:5],  # Top 5 recommendations
            "estimated_duration": sum(r.get("estimated_duration", 300) for r in recommendations[:5]),
            "confidence": sum(r.get("confidence", 0.0) for r in recommendations[:5]) / min(len(recommendations), 5),
            "approval_required": any(r.get("approval_required", False) for r in recommendations[:5])
        }
        
        # Create fallback plan
        fallback_plan = {
            "strategy": "fallback",
            "actions": await self._create_fallback_actions(recommendations, context_analysis),
            "trigger_conditions": ["primary_plan_failure", "high_risk_detected"],
            "confidence": 0.6
        }
        
        # Create emergency plan
        emergency_plan = {
            "strategy": "emergency",
            "actions": await self._create_emergency_actions(context_analysis),
            "trigger_conditions": ["critical_escalation", "immediate_threat"],
            "confidence": 0.8,
            "auto_execute": True
        }
        
        # Determine recommended execution strategy
        threat_context = context_analysis.get("threat_context", {})
        response_strategy = self._determine_execution_strategy(threat_context, primary_plan)
        
        return {
            "recommended_strategy": response_strategy,
            "primary_plan": primary_plan,
            "fallback_plan": fallback_plan,
            "emergency_plan": emergency_plan,
            "execution_guidelines": self._generate_execution_guidelines(
                recommendations, context_analysis
            ),
            "monitoring_requirements": self._generate_monitoring_requirements(
                recommendations, context_analysis
            ),
            "success_criteria": self._define_success_criteria(
                recommendations, context_analysis
            )
        }
    
    async def _get_relevant_learning_data(
        self,
        context_analysis: Dict[str, Any],
        db_session: AsyncSession
    ) -> Dict[str, Any]:
        """Get relevant learning data for the current context"""
        
        # Extract context features for similarity matching
        threat_context = context_analysis.get("threat_context", {})
        threat_category = threat_context.get("threat_category", "unknown")
        severity_score = threat_context.get("severity_score", 0.5)
        
        # Get historical workflows for similar contexts
        similar_workflows = await self._get_similar_historical_workflows(
            threat_category, severity_score, db_session
        )
        
        # Analyze performance patterns
        learning_data = {
            "action_performance": await self._analyze_action_performance(similar_workflows, db_session),
            "parameter_optimization": await self._analyze_parameter_performance(similar_workflows),
            "timing_optimization": await self._analyze_timing_patterns(similar_workflows),
            "category_patterns": await self._analyze_category_patterns(similar_workflows),
            "context_similarity": self._calculate_context_similarity(similar_workflows, threat_context),
            "sample_quality": self._assess_sample_quality(similar_workflows)
        }
        
        return learning_data
    
    async def _get_similar_historical_workflows(
        self,
        threat_category: str,
        severity_score: float,
        db_session: AsyncSession
    ) -> List[ResponseWorkflow]:
        """Get historically similar workflows for learning"""
        
        # Query for similar workflows
        result = await db_session.execute(
            select(ResponseWorkflow)
            .join(Incident)
            .where(
                and_(
                    ResponseWorkflow.status == "completed",
                    ResponseWorkflow.created_at >= datetime.now(timezone.utc) - timedelta(
                        days=self.learning_config["max_learning_history_days"]
                    ),
                    # Add similarity criteria here based on available incident fields
                    Incident.status.in_(["contained", "dismissed"])
                )
            )
            .order_by(ResponseWorkflow.created_at.desc())
            .limit(100)
        )
        
        return result.scalars().all()
    
    async def _analyze_action_performance(
        self, 
        workflows: List[ResponseWorkflow], 
        db_session: AsyncSession
    ) -> Dict[str, Any]:
        """Analyze performance of individual actions across workflows"""
        
        action_stats = {}
        
        for workflow in workflows:
            if not workflow.steps:
                continue
                
            workflow_success = workflow.success_rate or 0.0
            
            for step in workflow.steps:
                action_type = step.get("action_type")
                if not action_type:
                    continue
                
                if action_type not in action_stats:
                    action_stats[action_type] = {
                        "total_uses": 0,
                        "successful_uses": 0,
                        "total_duration": 0,
                        "success_rates": [],
                        "durations": []
                    }
                
                stats = action_stats[action_type]
                stats["total_uses"] += 1
                stats["success_rates"].append(workflow_success)
                
                if workflow_success > 0.7:
                    stats["successful_uses"] += 1
                
                # Estimate duration (simplified)
                estimated_duration = step.get("timeout_seconds", 300) * 1000
                stats["durations"].append(estimated_duration)
                stats["total_duration"] += estimated_duration
        
        # Calculate performance metrics
        performance_data = {}
        for action_type, stats in action_stats.items():
            if stats["total_uses"] >= self.learning_config["min_samples_for_learning"]:
                performance_data[action_type] = {
                    "success_rate": stats["successful_uses"] / stats["total_uses"],
                    "avg_duration": stats["total_duration"] / stats["total_uses"],
                    "sample_size": stats["total_uses"],
                    "confidence": min(stats["total_uses"] / 20.0, 1.0),  # Confidence based on sample size
                    "trend": self._calculate_performance_trend(stats["success_rates"])
                }
        
        return performance_data
    
    async def _analyze_parameter_performance(
        self, 
        workflows: List[ResponseWorkflow]
    ) -> Dict[str, Any]:
        """Analyze optimal parameters for different actions"""
        
        parameter_data = {}
        
        for workflow in workflows:
            if not workflow.steps or not workflow.success_rate or workflow.success_rate < 0.8:
                continue
            
            for step in workflow.steps:
                action_type = step.get("action_type")
                parameters = step.get("parameters", {})
                
                if action_type and parameters:
                    if action_type not in parameter_data:
                        parameter_data[action_type] = {"successful_parameters": []}
                    
                    parameter_data[action_type]["successful_parameters"].append(parameters)
        
        # Find optimal parameters
        optimal_parameters = {}
        for action_type, data in parameter_data.items():
            if len(data["successful_parameters"]) >= 3:
                optimal_parameters[action_type] = {
                    "optimal_parameters": self._extract_optimal_parameters(
                        data["successful_parameters"]
                    ),
                    "sample_size": len(data["successful_parameters"]),
                    "confidence": min(len(data["successful_parameters"]) / 10.0, 1.0)
                }
        
        return optimal_parameters
    
    async def _analyze_timing_patterns(self, workflows: List[ResponseWorkflow]) -> Dict[str, Any]:
        """Analyze optimal timing patterns for actions"""
        
        timing_data = {}
        
        for workflow in workflows:
            if not workflow.steps or not workflow.success_rate or workflow.success_rate < 0.8:
                continue
            
            # Analyze timing between actions
            for i, step in enumerate(workflow.steps):
                action_type = step.get("action_type")
                if action_type and i > 0:
                    previous_action = workflow.steps[i-1].get("action_type")
                    
                    timing_key = f"{previous_action}->{action_type}"
                    if timing_key not in timing_data:
                        timing_data[timing_key] = {"delays": [], "success_rates": []}
                    
                    # Estimate delay (simplified)
                    estimated_delay = 60  # 1 minute default
                    timing_data[timing_key]["delays"].append(estimated_delay)
                    timing_data[timing_key]["success_rates"].append(workflow.success_rate)
        
        # Calculate optimal timings
        optimal_timings = {}
        for timing_key, data in timing_data.items():
            if len(data["delays"]) >= 3:
                avg_delay = sum(data["delays"]) / len(data["delays"])
                avg_success = sum(data["success_rates"]) / len(data["success_rates"])
                
                if avg_success > 0.8:
                    action_type = timing_key.split("->")[1]
                    optimal_timings[action_type] = {
                        "optimal_delay": avg_delay,
                        "confidence": min(len(data["delays"]) / 10.0, 1.0),
                        "success_rate": avg_success
                    }
        
        return optimal_timings
    
    async def _analyze_category_patterns(self, workflows: List[ResponseWorkflow]) -> Dict[str, Any]:
        """Analyze patterns specific to threat categories"""
        
        category_patterns = {}
        
        # Group workflows by threat category (simplified)
        for workflow in workflows:
            # In a full implementation, this would get threat category from the incident
            category = "general"  # Simplified
            
            if category not in category_patterns:
                category_patterns[category] = {
                    "workflows": [],
                    "common_actions": {},
                    "emergent_actions": {},
                    "success_patterns": []
                }
            
            category_patterns[category]["workflows"].append(workflow)
            
            # Track action usage in successful workflows
            if workflow.success_rate and workflow.success_rate > 0.8 and workflow.steps:
                for step in workflow.steps:
                    action_type = step.get("action_type")
                    if action_type:
                        if action_type not in category_patterns[category]["common_actions"]:
                            category_patterns[category]["common_actions"][action_type] = 0
                        category_patterns[category]["common_actions"][action_type] += 1
        
        # Identify emergent successful patterns
        for category, data in category_patterns.items():
            total_workflows = len(data["workflows"])
            if total_workflows >= 5:
                for action_type, count in data["common_actions"].items():
                    usage_rate = count / total_workflows
                    if usage_rate > 0.6:  # Used in 60%+ of successful workflows
                        data["emergent_actions"][action_type] = {
                            "confidence": usage_rate,
                            "sample_size": count,
                            "success_rate": 0.8,  # Simplified
                            "trend": "emerging" if usage_rate > 0.8 else "stable"
                        }
        
        return category_patterns
    
    def _calculate_context_similarity(
        self, 
        workflows: List[ResponseWorkflow], 
        current_context: Dict[str, Any]
    ) -> float:
        """Calculate similarity between current context and historical contexts"""
        
        if not workflows:
            return 0.0
        
        # Simplified similarity calculation
        # In a full implementation, this would use vector similarity or clustering
        
        current_severity = current_context.get("severity_score", 0.5)
        current_category = current_context.get("threat_category", "unknown")
        
        similarity_scores = []
        for workflow in workflows:
            # Simplified similarity based on available data
            workflow_similarity = 0.5  # Base similarity
            
            # Add more sophisticated similarity calculations here
            similarity_scores.append(workflow_similarity)
        
        return sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0.0
    
    def _assess_sample_quality(self, workflows: List[ResponseWorkflow]) -> Dict[str, Any]:
        """Assess the quality of the learning sample"""
        
        if not workflows:
            return {"quality_score": 0.0, "sample_size": 0}
        
        # Calculate quality metrics
        successful_workflows = len([w for w in workflows if w.success_rate and w.success_rate > 0.7])
        completed_workflows = len([w for w in workflows if w.status == "completed"])
        
        quality_score = 0.0
        
        # Sample size quality
        if len(workflows) >= 20:
            quality_score += 0.4
        elif len(workflows) >= 10:
            quality_score += 0.3
        elif len(workflows) >= 5:
            quality_score += 0.2
        
        # Success rate quality
        if completed_workflows > 0:
            success_rate = successful_workflows / completed_workflows
            quality_score += success_rate * 0.4
        
        # Recency quality
        recent_workflows = len([
            w for w in workflows 
            if w.created_at and w.created_at > datetime.now(timezone.utc) - timedelta(days=30)
        ])
        recency_score = min(recent_workflows / len(workflows), 1.0) * 0.2
        quality_score += recency_score
        
        return {
            "quality_score": quality_score,
            "sample_size": len(workflows),
            "successful_workflows": successful_workflows,
            "recency_score": recency_score,
            "recommendation": "high_quality" if quality_score > 0.7 else "medium_quality" if quality_score > 0.4 else "low_quality"
        }
    
    async def _create_fallback_actions(
        self, 
        primary_recommendations: List[Dict[str, Any]], 
        context_analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Create fallback actions for when primary plan fails"""
        
        fallback_actions = []
        
        # Conservative blocking action
        fallback_actions.append({
            "action_type": "block_ip_advanced",
            "confidence": 0.8,
            "priority": 1,
            "parameters": {
                "duration": 7200,  # 2 hours
                "block_level": "comprehensive"
            },
            "rationale": "Conservative blocking as fallback measure"
        })
        
        # Evidence collection
        fallback_actions.append({
            "action_type": "memory_dump_collection",
            "confidence": 0.7,
            "priority": 2,
            "parameters": {
                "dump_type": "minimal",
                "encryption": True
            },
            "rationale": "Preserve evidence for investigation"
        })
        
        return fallback_actions
    
    async def _create_emergency_actions(self, context_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create emergency response actions for critical situations"""
        
        emergency_actions = []
        
        # Immediate isolation
        emergency_actions.append({
            "action_type": "isolate_host_advanced",
            "confidence": 0.9,
            "priority": 1,
            "parameters": {
                "isolation_level": "strict",
                "monitoring": "comprehensive"
            },
            "rationale": "Emergency isolation to prevent damage"
        })
        
        # Immediate blocking
        emergency_actions.append({
            "action_type": "block_ip_advanced",
            "confidence": 0.9,
            "priority": 2,
            "parameters": {
                "duration": 86400,  # 24 hours
                "block_level": "comprehensive",
                "geo_restrictions": True
            },
            "rationale": "Emergency blocking to stop attack"
        })
        
        return emergency_actions
    
    def _determine_execution_strategy(
        self, 
        threat_context: Dict[str, Any], 
        primary_plan: Dict[str, Any]
    ) -> str:
        """Determine the recommended execution strategy"""
        
        severity = threat_context.get("severity_score", 0.5)
        confidence = primary_plan.get("confidence", 0.0)
        
        if severity > 0.8 and confidence > 0.8:
            return "immediate_execution"
        elif severity > 0.6 or confidence > 0.7:
            return "guided_execution"
        else:
            return "manual_approval"
    
    def _generate_execution_guidelines(
        self, 
        recommendations: List[Dict[str, Any]], 
        context_analysis: Dict[str, Any]
    ) -> List[str]:
        """Generate execution guidelines for analysts"""
        
        guidelines = [
            "Monitor execution progress closely for any unexpected results",
            "Be prepared to execute fallback plan if primary actions fail",
            "Document all execution results for continuous learning"
        ]
        
        # Add context-specific guidelines
        threat_context = context_analysis.get("threat_context", {})
        if threat_context.get("severity_score", 0) > 0.8:
            guidelines.append("Consider escalating to senior analyst due to high severity")
        
        if any(rec.get("approval_required", False) for rec in recommendations):
            guidelines.append("Obtain required approvals before executing high-impact actions")
        
        return guidelines
    
    def _generate_monitoring_requirements(
        self, 
        recommendations: List[Dict[str, Any]], 
        context_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate monitoring requirements for the response plan"""
        
        return {
            "real_time_monitoring": True,
            "success_criteria_tracking": True,
            "impact_measurement": True,
            "rollback_triggers": [
                "success_rate_below_50_percent",
                "unexpected_system_impact",
                "compliance_violation_detected"
            ],
            "escalation_triggers": [
                "multiple_action_failures",
                "new_attack_vectors_detected",
                "critical_system_compromise"
            ],
            "reporting_schedule": "immediate_alerts_plus_hourly_summary"
        }
    
    def _define_success_criteria(
        self, 
        recommendations: List[Dict[str, Any]], 
        context_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Define success criteria for the response plan"""
        
        return {
            "primary_success": {
                "attack_stopped": True,
                "systems_protected": True,
                "minimal_business_impact": True
            },
            "secondary_success": {
                "evidence_collected": True,
                "lessons_learned": True,
                "processes_improved": True
            },
            "success_thresholds": {
                "action_success_rate": 0.8,
                "overall_workflow_success": 0.85,
                "response_time_under": 1800,  # 30 minutes
                "false_positive_rate_under": 0.1
            },
            "measurement_methods": {
                "attack_volume_reduction": "monitor_event_rate",
                "system_health": "endpoint_monitoring",
                "business_impact": "service_availability_metrics"
            }
        }
    
    async def _get_learning_insights(
        self, 
        incident_id: int, 
        db_session: AsyncSession
    ) -> Dict[str, Any]:
        """Get current learning insights for the incident"""
        
        # Get recent learning data
        recent_learning = await self._get_recent_learning_data(db_session)
        
        return {
            "learning_maturity": self._assess_learning_maturity(recent_learning),
            "adaptation_velocity": self._calculate_adaptation_velocity(recent_learning),
            "recommendation_accuracy": self._calculate_recommendation_accuracy(recent_learning),
            "improvement_trends": self._analyze_improvement_trends(recent_learning),
            "learning_gaps": self._identify_learning_gaps(recent_learning)
        }
    
    def _calculate_confidence_metrics(self, recommendations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate confidence metrics for recommendations"""
        
        if not recommendations:
            return {"overall_confidence": 0.0, "distribution": {}}
        
        confidences = [rec.get("confidence", 0.0) for rec in recommendations]
        
        return {
            "overall_confidence": sum(confidences) / len(confidences),
            "min_confidence": min(confidences),
            "max_confidence": max(confidences),
            "high_confidence_count": len([c for c in confidences if c > 0.8]),
            "low_confidence_count": len([c for c in confidences if c < 0.5]),
            "distribution": {
                "very_high": len([c for c in confidences if c >= 0.9]),
                "high": len([c for c in confidences if 0.7 <= c < 0.9]),
                "medium": len([c for c in confidences if 0.5 <= c < 0.7]),
                "low": len([c for c in confidences if 0.3 <= c < 0.5]),
                "very_low": len([c for c in confidences if c < 0.3])
            }
        }
    
    async def learn_from_workflow_execution(
        self,
        workflow_id: str,
        execution_results: Dict[str, Any],
        analyst_feedback: Optional[Dict[str, Any]],
        db_session: AsyncSession
    ) -> Dict[str, Any]:
        """Learn from workflow execution and analyst feedback"""
        
        try:
            # Record learning event
            learning_event = {
                "workflow_id": workflow_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "execution_results": execution_results,
                "analyst_feedback": analyst_feedback,
                "learning_score": self._calculate_learning_score(execution_results, analyst_feedback)
            }
            
            self.learning_history.append(learning_event)
            
            # Update learning models based on results
            model_updates = await self._update_learning_models(learning_event, db_session)
            
            # Analyze learning progress
            learning_progress = self._analyze_learning_progress()
            
            return {
                "success": True,
                "learning_event_id": len(self.learning_history),
                "learning_score": learning_event["learning_score"],
                "model_updates": model_updates,
                "learning_progress": learning_progress,
                "adaptation_recommendations": await self._generate_adaptation_recommendations(learning_event)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to learn from workflow execution: {e}")
            return {"success": False, "error": str(e)}
    
    # Helper methods (simplified implementations)
    def _calculate_performance_trend(self, success_rates: List[float]) -> str:
        """Calculate performance trend"""
        if len(success_rates) < 2:
            return "stable"
        
        recent_avg = sum(success_rates[-5:]) / min(len(success_rates), 5)
        older_avg = sum(success_rates[:-5]) / max(len(success_rates) - 5, 1)
        
        if recent_avg > older_avg + 0.1:
            return "improving"
        elif recent_avg < older_avg - 0.1:
            return "declining"
        else:
            return "stable"
    
    def _extract_optimal_parameters(self, parameter_sets: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract optimal parameters from successful executions"""
        # Simplified implementation - would use statistical analysis in production
        return parameter_sets[0] if parameter_sets else {}
    
    async def _get_recent_learning_data(self, db_session: AsyncSession) -> Dict[str, Any]:
        """Get recent learning data for analysis"""
        return {"workflows_analyzed": len(self.learning_history)}  # Simplified
    
    def _assess_learning_maturity(self, learning_data: Dict[str, Any]) -> str:
        """Assess the maturity of the learning system"""
        workflows_analyzed = learning_data.get("workflows_analyzed", 0)
        
        if workflows_analyzed > 100:
            return "mature"
        elif workflows_analyzed > 50:
            return "developing"
        elif workflows_analyzed > 10:
            return "initial"
        else:
            return "nascent"
    
    def _calculate_adaptation_velocity(self, learning_data: Dict[str, Any]) -> float:
        """Calculate how quickly the system is adapting"""
        return 0.5  # Simplified
    
    def _calculate_recommendation_accuracy(self, learning_data: Dict[str, Any]) -> float:
        """Calculate accuracy of previous recommendations"""
        return 0.8  # Simplified
    
    def _analyze_improvement_trends(self, learning_data: Dict[str, Any]) -> List[str]:
        """Analyze improvement trends"""
        return ["Increasing action effectiveness", "Decreasing response times"]  # Simplified
    
    def _identify_learning_gaps(self, learning_data: Dict[str, Any]) -> List[str]:
        """Identify areas needing more learning data"""
        return ["Advanced persistent threat response", "Insider threat detection"]  # Simplified
    
    def _calculate_learning_score(
        self, 
        execution_results: Dict[str, Any], 
        analyst_feedback: Optional[Dict[str, Any]]
    ) -> float:
        """Calculate learning score from execution and feedback"""
        
        base_score = execution_results.get("success_rate", 0.0)
        
        if analyst_feedback:
            feedback_score = analyst_feedback.get("effectiveness_rating", 0.5)
            # Weight execution results and analyst feedback
            return base_score * 0.7 + feedback_score * 0.3
        
        return base_score
    
    async def _update_learning_models(
        self, 
        learning_event: Dict[str, Any], 
        db_session: AsyncSession
    ) -> Dict[str, Any]:
        """Update learning models based on new learning event"""
        
        return {
            "models_updated": ["action_effectiveness", "parameter_optimization"],
            "improvement_detected": learning_event["learning_score"] > 0.7
        }
    
    def _analyze_learning_progress(self) -> Dict[str, Any]:
        """Analyze overall learning progress"""
        
        if not self.learning_history:
            return {"progress": "no_data"}
        
        recent_scores = [event["learning_score"] for event in self.learning_history[-10:]]
        avg_recent_score = sum(recent_scores) / len(recent_scores)
        
        return {
            "progress": "improving" if avg_recent_score > 0.7 else "stable",
            "average_learning_score": avg_recent_score,
            "total_learning_events": len(self.learning_history),
            "learning_velocity": "moderate"
        }
    
    async def _generate_adaptation_recommendations(self, learning_event: Dict[str, Any]) -> List[str]:
        """Generate recommendations for system adaptation"""
        
        recommendations = []
        
        if learning_event["learning_score"] > 0.8:
            recommendations.append("Consider increasing automation level for similar incidents")
        elif learning_event["learning_score"] < 0.5:
            recommendations.append("Review and refine response strategies for this incident type")
        
        return recommendations


# Global instance
learning_response_engine = LearningResponseEngine()


async def get_learning_engine() -> LearningResponseEngine:
    """Get the global learning response engine instance"""
    if not learning_response_engine.initialized:
        await learning_response_engine.initialize()
    return learning_response_engine





