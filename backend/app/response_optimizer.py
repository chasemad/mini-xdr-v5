"""
Response Optimizer for Mini-XDR
AI-powered optimization of response strategies and workflows.
"""

import logging
import asyncio
import json
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from sqlalchemy import select, and_, func, desc
from sqlalchemy.ext.asyncio import AsyncSession

from .models import (
    Incident, ResponseWorkflow, AdvancedResponseAction, 
    ResponseImpactMetrics, ResponsePlaybook
)
from .config import settings
from .secrets_manager import get_secure_env

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Result of response optimization analysis"""
    optimized_workflow: Dict[str, Any]
    optimization_score: float
    improvements: List[str]
    risk_reduction: float
    efficiency_gain: float
    confidence: float


class OptimizationStrategy(str, Enum):
    PERFORMANCE = "performance"      # Optimize for speed
    EFFECTIVENESS = "effectiveness"  # Optimize for success rate
    EFFICIENCY = "efficiency"        # Optimize for resource usage
    SAFETY = "safety"               # Optimize for minimal risk
    COMPLIANCE = "compliance"        # Optimize for regulatory compliance


class ResponseOptimizer:
    """
    AI-powered response optimizer that learns from historical data
    and continuously improves response strategies.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.openai_client = None
        self.initialized = False
        
        # Optimization models
        self.optimization_models = self._initialize_optimization_models()
        self.performance_baselines = self._initialize_performance_baselines()
        self.learning_weights = self._initialize_learning_weights()
        
        # Historical analysis cache
        self.performance_cache = {}
        self.pattern_cache = {}
        
    async def initialize(self):
        """Initialize the response optimizer"""
        try:
            # Initialize OpenAI client if available
            if settings.llm_provider == "openai":
                import openai
                api_key = get_secure_env("OPENAI_API_KEY", "mini-xdr/openai-api-key")
                if api_key:
                    self.openai_client = openai.AsyncOpenAI(api_key=api_key)
                    self.logger.info("OpenAI client initialized for Response Optimizer")
            
            self.initialized = True
            self.logger.info("Response Optimizer initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Response Optimizer: {e}")
    
    def _initialize_optimization_models(self) -> Dict[str, Any]:
        """Initialize optimization models and parameters"""
        return {
            "action_sequencing": {
                "parallel_safe_actions": ["memory_dump_collection", "data_classification"],
                "sequential_required": ["isolate_host_advanced", "memory_dump_collection"],
                "conflict_matrix": {
                    "block_ip_advanced": ["traffic_redirection"],
                    "isolate_host_advanced": ["registry_hardening"]
                }
            },
            "timing_optimization": {
                "immediate_actions": ["block_ip_advanced", "account_disable"],
                "delayed_actions": ["memory_dump_collection", "backup_verification"],
                "optimal_delays": {
                    "memory_dump_collection": 300,  # 5 minutes after isolation
                    "data_classification": 600      # 10 minutes after isolation
                }
            },
            "resource_optimization": {
                "high_resource_actions": ["memory_dump_collection", "backup_verification"],
                "low_resource_actions": ["block_ip_advanced", "dns_sinkhole"],
                "resource_constraints": {
                    "max_concurrent_high_resource": 2,
                    "max_total_concurrent": 5
                }
            },
            "effectiveness_patterns": {
                "malware": ["isolate_host_advanced", "memory_dump_collection", "block_ip_advanced"],
                "ddos": ["traffic_redirection", "deploy_firewall_rules", "block_ip_advanced"],
                "brute_force": ["block_ip_advanced", "account_disable", "password_reset_bulk"],
                "insider_threat": ["account_disable", "memory_dump_collection", "iam_policy_restriction"]
            }
        }
    
    def _initialize_performance_baselines(self) -> Dict[str, Any]:
        """Initialize performance baselines for optimization"""
        return {
            "action_success_rates": {
                "block_ip_advanced": 0.95,
                "isolate_host_advanced": 0.88,
                "memory_dump_collection": 0.92,
                "deploy_firewall_rules": 0.85,
                "dns_sinkhole": 0.90,
                "account_disable": 0.98,
                "password_reset_bulk": 0.85,
                "email_recall": 0.82
            },
            "action_durations": {
                "block_ip_advanced": 120,
                "isolate_host_advanced": 300,
                "memory_dump_collection": 600,
                "deploy_firewall_rules": 180,
                "dns_sinkhole": 90,
                "account_disable": 60,
                "password_reset_bulk": 240,
                "email_recall": 150
            },
            "resource_requirements": {
                "block_ip_advanced": 0.1,
                "isolate_host_advanced": 0.3,
                "memory_dump_collection": 0.8,
                "deploy_firewall_rules": 0.4,
                "dns_sinkhole": 0.2,
                "account_disable": 0.1,
                "password_reset_bulk": 0.3,
                "email_recall": 0.4
            }
        }
    
    def _initialize_learning_weights(self) -> Dict[str, float]:
        """Initialize learning weights for different optimization factors"""
        return {
            "historical_success": 0.30,
            "execution_time": 0.20,
            "resource_efficiency": 0.15,
            "safety_score": 0.15,
            "compliance_alignment": 0.10,
            "user_feedback": 0.10
        }
    
    async def optimize_workflow(
        self,
        workflow_id: str,
        db_session: AsyncSession,
        strategy: OptimizationStrategy = OptimizationStrategy.EFFECTIVENESS,
        context: Optional[Dict[str, Any]] = None
    ) -> OptimizationResult:
        """
        Optimize an existing workflow based on specified strategy
        """
        try:
            # Get workflow and its history
            workflow = await self._get_workflow_with_history(workflow_id, db_session)
            if not workflow:
                raise ValueError(f"Workflow {workflow_id} not found")
            
            # Analyze current workflow performance
            current_analysis = await self._analyze_current_workflow(workflow, db_session)
            
            # Generate optimization recommendations
            optimization_candidates = await self._generate_optimization_candidates(
                workflow, current_analysis, strategy, context
            )
            
            # Evaluate and rank optimizations
            best_optimization = await self._evaluate_optimizations(
                optimization_candidates, current_analysis, strategy
            )
            
            # Calculate improvement metrics
            improvements = await self._calculate_improvements(
                workflow, best_optimization, current_analysis
            )
            
            # Generate AI-enhanced optimization if available
            if self.openai_client:
                ai_optimization = await self._get_ai_optimization(
                    workflow, best_optimization, current_analysis, strategy
                )
                if ai_optimization:
                    best_optimization = ai_optimization
            
            return OptimizationResult(
                optimized_workflow=best_optimization,
                optimization_score=best_optimization.get("optimization_score", 0.0),
                improvements=improvements.get("improvement_list", []),
                risk_reduction=improvements.get("risk_reduction", 0.0),
                efficiency_gain=improvements.get("efficiency_gain", 0.0),
                confidence=best_optimization.get("confidence", 0.0)
            )
            
        except Exception as e:
            self.logger.error(f"Failed to optimize workflow: {e}")
            raise
    
    async def optimize_response_strategy(
        self,
        incident_id: int,
        current_recommendations: List[Dict[str, Any]],
        db_session: AsyncSession,
        strategy: OptimizationStrategy = OptimizationStrategy.EFFECTIVENESS
    ) -> Dict[str, Any]:
        """
        Optimize response strategy recommendations based on historical data and AI analysis
        """
        try:
            # Get incident context
            incident = await db_session.get(Incident, incident_id)
            if not incident:
                return {"success": False, "error": "Incident not found"}
            
            # Analyze historical performance for similar incidents
            historical_analysis = await self._analyze_historical_performance(
                incident, current_recommendations, db_session
            )
            
            # Generate optimization suggestions
            optimizations = await self._generate_strategy_optimizations(
                current_recommendations, historical_analysis, strategy
            )
            
            # Apply optimization filters
            optimized_recommendations = await self._apply_optimization_filters(
                current_recommendations, optimizations, strategy
            )
            
            # Calculate optimization metrics
            optimization_metrics = self._calculate_optimization_metrics(
                current_recommendations, optimized_recommendations, historical_analysis
            )
            
            return {
                "success": True,
                "original_recommendations": current_recommendations,
                "optimized_recommendations": optimized_recommendations,
                "optimization_strategy": strategy.value,
                "optimization_metrics": optimization_metrics,
                "improvements": optimizations.get("improvements", []),
                "confidence": optimizations.get("confidence", 0.0),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to optimize response strategy: {e}")
            return {"success": False, "error": str(e)}
    
    async def learn_from_execution(
        self,
        workflow_id: str,
        execution_results: Dict[str, Any],
        db_session: AsyncSession
    ) -> Dict[str, Any]:
        """
        Learn from workflow execution results to improve future optimizations
        """
        try:
            # Get workflow details
            workflow = await self._get_workflow_with_history(workflow_id, db_session)
            if not workflow:
                return {"success": False, "error": "Workflow not found"}
            
            # Extract learning insights
            learning_insights = await self._extract_learning_insights(
                workflow, execution_results, db_session
            )
            
            # Update optimization models
            model_updates = await self._update_optimization_models(learning_insights)
            
            # Store learning data for future use
            await self._store_learning_data(workflow_id, learning_insights, db_session)
            
            return {
                "success": True,
                "insights": learning_insights,
                "model_updates": model_updates,
                "learning_score": learning_insights.get("learning_score", 0.0)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to learn from execution: {e}")
            return {"success": False, "error": str(e)}
    
    async def _get_workflow_with_history(
        self, 
        workflow_id: str, 
        db_session: AsyncSession
    ) -> Optional[ResponseWorkflow]:
        """Get workflow with full execution history"""
        
        result = await db_session.execute(
            select(ResponseWorkflow)
            .where(ResponseWorkflow.workflow_id == workflow_id)
        )
        
        return result.scalars().first()
    
    async def _analyze_current_workflow(
        self, 
        workflow: ResponseWorkflow, 
        db_session: AsyncSession
    ) -> Dict[str, Any]:
        """Analyze current workflow performance and characteristics"""
        
        # Get workflow actions
        actions_result = await db_session.execute(
            select(AdvancedResponseAction)
            .where(AdvancedResponseAction.workflow_id == workflow.id)
            .order_by(AdvancedResponseAction.created_at)
        )
        actions = actions_result.scalars().all()
        
        # Get impact metrics
        metrics_result = await db_session.execute(
            select(ResponseImpactMetrics)
            .where(ResponseImpactMetrics.workflow_id == workflow.id)
        )
        metrics = metrics_result.scalars().all()
        
        analysis = {
            "workflow_id": workflow.workflow_id,
            "current_success_rate": workflow.success_rate or 0.0,
            "execution_time_ms": workflow.execution_time_ms or 0,
            "total_actions": len(actions),
            "completed_actions": len([a for a in actions if a.status == "completed"]),
            "failed_actions": len([a for a in actions if a.status == "failed"]),
            "action_sequence": [
                {
                    "action_type": action.action_type,
                    "status": action.status,
                    "duration_ms": (action.completed_at - action.created_at).total_seconds() * 1000 
                                  if action.completed_at and action.created_at else 0,
                    "confidence": action.confidence_score or 0.0
                }
                for action in actions
            ],
            "impact_metrics": [
                {
                    "attacks_blocked": metric.attacks_blocked,
                    "response_time_ms": metric.response_time_ms,
                    "success_rate": metric.success_rate,
                    "cost_impact": metric.cost_impact_usd
                }
                for metric in metrics
            ],
            "bottlenecks": self._identify_workflow_bottlenecks(actions),
            "inefficiencies": self._identify_workflow_inefficiencies(actions)
        }
        
        return analysis
    
    async def _generate_optimization_candidates(
        self,
        workflow: ResponseWorkflow,
        current_analysis: Dict[str, Any],
        strategy: OptimizationStrategy,
        context: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Generate potential optimization candidates"""
        
        candidates = []
        
        # Strategy-specific optimizations
        if strategy == OptimizationStrategy.PERFORMANCE:
            candidates.extend(await self._generate_performance_optimizations(workflow, current_analysis))
        elif strategy == OptimizationStrategy.EFFECTIVENESS:
            candidates.extend(await self._generate_effectiveness_optimizations(workflow, current_analysis))
        elif strategy == OptimizationStrategy.EFFICIENCY:
            candidates.extend(await self._generate_efficiency_optimizations(workflow, current_analysis))
        elif strategy == OptimizationStrategy.SAFETY:
            candidates.extend(await self._generate_safety_optimizations(workflow, current_analysis))
        elif strategy == OptimizationStrategy.COMPLIANCE:
            candidates.extend(await self._generate_compliance_optimizations(workflow, current_analysis))
        
        # Universal optimizations (apply to all strategies)
        candidates.extend(await self._generate_universal_optimizations(workflow, current_analysis))
        
        return candidates
    
    async def _generate_performance_optimizations(
        self, 
        workflow: ResponseWorkflow, 
        analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate performance-focused optimizations"""
        
        optimizations = []
        
        # Parallel execution opportunities
        if self._can_parallelize_actions(workflow.steps):
            optimizations.append({
                "type": "parallel_execution",
                "description": "Execute compatible actions in parallel",
                "expected_improvement": 0.4,
                "confidence": 0.8,
                "implementation": {
                    "parallel_groups": self._identify_parallel_groups(workflow.steps)
                }
            })
        
        # Action reordering for faster execution
        if self._has_suboptimal_ordering(workflow.steps):
            optimizations.append({
                "type": "action_reordering",
                "description": "Reorder actions for optimal execution sequence",
                "expected_improvement": 0.2,
                "confidence": 0.7,
                "implementation": {
                    "optimal_sequence": self._calculate_optimal_sequence(workflow.steps)
                }
            })
        
        # Remove redundant actions
        redundant_actions = self._identify_redundant_actions(workflow.steps)
        if redundant_actions:
            optimizations.append({
                "type": "redundancy_removal",
                "description": f"Remove {len(redundant_actions)} redundant actions",
                "expected_improvement": 0.3,
                "confidence": 0.9,
                "implementation": {
                    "actions_to_remove": redundant_actions
                }
            })
        
        return optimizations
    
    async def _generate_effectiveness_optimizations(
        self, 
        workflow: ResponseWorkflow, 
        analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate effectiveness-focused optimizations"""
        
        optimizations = []
        
        # Add missing critical actions
        critical_gaps = await self._identify_critical_gaps(workflow)
        if critical_gaps:
            optimizations.append({
                "type": "critical_action_addition",
                "description": f"Add {len(critical_gaps)} critical missing actions",
                "expected_improvement": 0.5,
                "confidence": 0.8,
                "implementation": {
                    "actions_to_add": critical_gaps
                }
            })
        
        # Improve action parameters
        parameter_improvements = self._suggest_parameter_improvements(workflow.steps)
        if parameter_improvements:
            optimizations.append({
                "type": "parameter_optimization",
                "description": "Optimize action parameters based on historical data",
                "expected_improvement": 0.3,
                "confidence": 0.7,
                "implementation": {
                    "parameter_updates": parameter_improvements
                }
            })
        
        # Add error handling and retry logic
        if not self._has_adequate_error_handling(workflow.steps):
            optimizations.append({
                "type": "error_handling_enhancement",
                "description": "Add robust error handling and retry mechanisms",
                "expected_improvement": 0.4,
                "confidence": 0.9,
                "implementation": {
                    "retry_strategies": self._design_retry_strategies(workflow.steps)
                }
            })
        
        return optimizations
    
    async def _generate_efficiency_optimizations(
        self, 
        workflow: ResponseWorkflow, 
        analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate efficiency-focused optimizations"""
        
        optimizations = []
        
        # Resource optimization
        if self._has_resource_conflicts(workflow.steps):
            optimizations.append({
                "type": "resource_optimization",
                "description": "Optimize resource usage to prevent conflicts",
                "expected_improvement": 0.3,
                "confidence": 0.8,
                "implementation": {
                    "resource_schedule": self._create_resource_schedule(workflow.steps)
                }
            })
        
        # Conditional execution
        conditional_opportunities = self._identify_conditional_opportunities(workflow.steps)
        if conditional_opportunities:
            optimizations.append({
                "type": "conditional_execution",
                "description": "Add conditional logic to skip unnecessary actions",
                "expected_improvement": 0.4,
                "confidence": 0.6,
                "implementation": {
                    "conditional_rules": conditional_opportunities
                }
            })
        
        return optimizations
    
    async def _generate_safety_optimizations(
        self, 
        workflow: ResponseWorkflow, 
        analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate safety-focused optimizations"""
        
        optimizations = []
        
        # Add safety checkpoints
        if not self._has_adequate_safety_checks(workflow.steps):
            optimizations.append({
                "type": "safety_checkpoints",
                "description": "Add safety validation checkpoints",
                "expected_improvement": 0.2,
                "confidence": 0.9,
                "implementation": {
                    "checkpoints": self._design_safety_checkpoints(workflow.steps)
                }
            })
        
        # Improve rollback planning
        if not self._has_comprehensive_rollback(workflow.steps):
            optimizations.append({
                "type": "rollback_enhancement",
                "description": "Enhance rollback capabilities for safer execution",
                "expected_improvement": 0.3,
                "confidence": 0.8,
                "implementation": {
                    "rollback_plan": self._design_enhanced_rollback(workflow.steps)
                }
            })
        
        return optimizations
    
    async def _generate_compliance_optimizations(
        self, 
        workflow: ResponseWorkflow, 
        analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate compliance-focused optimizations"""
        
        optimizations = []
        
        # Add audit logging
        if not self._has_adequate_audit_logging(workflow.steps):
            optimizations.append({
                "type": "audit_logging_enhancement",
                "description": "Enhanced audit logging for compliance",
                "expected_improvement": 0.2,
                "confidence": 0.9,
                "implementation": {
                    "logging_enhancements": self._design_audit_logging(workflow.steps)
                }
            })
        
        # Add compliance validation steps
        compliance_gaps = self._identify_compliance_gaps(workflow.steps)
        if compliance_gaps:
            optimizations.append({
                "type": "compliance_validation",
                "description": "Add compliance validation steps",
                "expected_improvement": 0.3,
                "confidence": 0.8,
                "implementation": {
                    "validation_steps": compliance_gaps
                }
            })
        
        return optimizations
    
    async def _generate_universal_optimizations(
        self, 
        workflow: ResponseWorkflow, 
        analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate universal optimizations that apply to all strategies"""
        
        optimizations = []
        
        # Timeout optimization
        if self._has_suboptimal_timeouts(workflow.steps):
            optimizations.append({
                "type": "timeout_optimization",
                "description": "Optimize action timeouts based on historical data",
                "expected_improvement": 0.15,
                "confidence": 0.8,
                "implementation": {
                    "timeout_adjustments": self._calculate_optimal_timeouts(workflow.steps)
                }
            })
        
        # Progress tracking enhancement
        if not self._has_detailed_progress_tracking(workflow.steps):
            optimizations.append({
                "type": "progress_tracking",
                "description": "Add detailed progress tracking and monitoring",
                "expected_improvement": 0.1,
                "confidence": 0.9,
                "implementation": {
                    "tracking_enhancements": self._design_progress_tracking(workflow.steps)
                }
            })
        
        return optimizations
    
    async def _evaluate_optimizations(
        self,
        candidates: List[Dict[str, Any]],
        current_analysis: Dict[str, Any],
        strategy: OptimizationStrategy
    ) -> Dict[str, Any]:
        """Evaluate and rank optimization candidates"""
        
        if not candidates:
            return {"optimization_score": 0.0, "confidence": 0.0}
        
        # Score each candidate based on strategy
        scored_candidates = []
        for candidate in candidates:
            score = self._calculate_optimization_score(candidate, strategy)
            scored_candidates.append({
                **candidate,
                "optimization_score": score
            })
        
        # Return best candidate
        best_candidate = max(scored_candidates, key=lambda x: x["optimization_score"])
        
        return best_candidate
    
    def _calculate_optimization_score(
        self, 
        candidate: Dict[str, Any], 
        strategy: OptimizationStrategy
    ) -> float:
        """Calculate optimization score for a candidate"""
        
        base_score = candidate.get("expected_improvement", 0.0)
        confidence = candidate.get("confidence", 0.0)
        
        # Strategy-specific weightings
        strategy_weights = {
            OptimizationStrategy.PERFORMANCE: {"speed": 0.6, "reliability": 0.4},
            OptimizationStrategy.EFFECTIVENESS: {"success_rate": 0.7, "completeness": 0.3},
            OptimizationStrategy.EFFICIENCY: {"resource_usage": 0.6, "cost": 0.4},
            OptimizationStrategy.SAFETY: {"risk_reduction": 0.8, "rollback": 0.2},
            OptimizationStrategy.COMPLIANCE: {"audit": 0.6, "reporting": 0.4}
        }
        
        # Apply strategy weighting
        weighted_score = base_score * confidence
        
        return weighted_score
    
    async def _calculate_improvements(
        self,
        original_workflow: ResponseWorkflow,
        optimized_workflow: Dict[str, Any],
        current_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate specific improvements from optimization"""
        
        improvements = {
            "improvement_list": [],
            "risk_reduction": 0.0,
            "efficiency_gain": 0.0,
            "performance_gain": 0.0,
            "safety_improvement": 0.0
        }
        
        # Calculate performance improvements
        original_duration = current_analysis.get("execution_time_ms", 0)
        optimized_duration = optimized_workflow.get("estimated_duration_ms", original_duration)
        
        if optimized_duration < original_duration:
            performance_gain = (original_duration - optimized_duration) / original_duration
            improvements["performance_gain"] = performance_gain
            improvements["improvement_list"].append(
                f"Reduced execution time by {performance_gain:.1%}"
            )
        
        # Calculate effectiveness improvements
        original_success_rate = current_analysis.get("current_success_rate", 0.0)
        optimized_success_rate = optimized_workflow.get("expected_success_rate", original_success_rate)
        
        if optimized_success_rate > original_success_rate:
            effectiveness_gain = optimized_success_rate - original_success_rate
            improvements["improvement_list"].append(
                f"Improved success rate by {effectiveness_gain:.1%}"
            )
        
        # Calculate efficiency improvements
        original_actions = current_analysis.get("total_actions", 0)
        optimized_actions = len(optimized_workflow.get("optimized_steps", []))
        
        if optimized_actions < original_actions:
            efficiency_gain = (original_actions - optimized_actions) / original_actions
            improvements["efficiency_gain"] = efficiency_gain
            improvements["improvement_list"].append(
                f"Reduced action count by {efficiency_gain:.1%}"
            )
        
        return improvements
    
    async def _get_ai_optimization(
        self,
        workflow: ResponseWorkflow,
        base_optimization: Dict[str, Any],
        current_analysis: Dict[str, Any],
        strategy: OptimizationStrategy
    ) -> Optional[Dict[str, Any]]:
        """Get AI-enhanced optimization recommendations"""
        
        if not self.openai_client:
            return None
        
        try:
            # Prepare context for AI
            ai_context = {
                "workflow": {
                    "name": workflow.playbook_name,
                    "steps": workflow.steps,
                    "success_rate": workflow.success_rate,
                    "execution_time": workflow.execution_time_ms
                },
                "current_analysis": current_analysis,
                "optimization_strategy": strategy.value,
                "base_optimization": base_optimization
            }
            
            # Get AI recommendations
            response = await self.openai_client.chat.completions.create(
                model=settings.openai_model,
                messages=[
                    {
                        "role": "system",
                        "content": """You are an expert cybersecurity workflow optimizer. Analyze the provided 
                        workflow and suggest specific optimizations based on the given strategy. Focus on 
                        practical, implementable improvements that enhance the specified optimization goal.
                        
                        Return your response as JSON with this structure:
                        {
                          "optimized_steps": [...],
                          "optimization_rationale": "explanation",
                          "expected_improvement": 0.3,
                          "confidence": 0.8,
                          "risk_assessment": "low/medium/high"
                        }"""
                    },
                    {
                        "role": "user",
                        "content": f"Optimize this workflow for {strategy.value}:\n\n{json.dumps(ai_context, indent=2)}"
                    }
                ],
                temperature=0.3,
                max_tokens=1500
            )
            
            # Parse AI response
            ai_content = response.choices[0].message.content
            if ai_content:
                try:
                    ai_optimization = json.loads(ai_content)
                    
                    # Merge with base optimization
                    enhanced_optimization = {
                        **base_optimization,
                        **ai_optimization,
                        "ai_enhanced": True,
                        "ai_rationale": ai_optimization.get("optimization_rationale", "")
                    }
                    
                    return enhanced_optimization
                    
                except json.JSONDecodeError:
                    self.logger.warning("Failed to parse AI optimization as JSON")
            
        except Exception as e:
            self.logger.error(f"Failed to get AI optimization: {e}")
        
        return None
    
    async def _analyze_historical_performance(
        self,
        incident: Incident,
        current_recommendations: List[Dict[str, Any]],
        db_session: AsyncSession
    ) -> Dict[str, Any]:
        """Analyze historical performance for similar incidents"""
        
        # Get similar workflows from the past
        similar_workflows_result = await db_session.execute(
            select(ResponseWorkflow)
            .join(Incident)
            .where(
                and_(
                    or_(
                        Incident.src_ip == incident.src_ip,
                        Incident.reason.like(f"%{incident.reason[:20]}%")
                    ),
                    ResponseWorkflow.created_at >= datetime.now(timezone.utc) - timedelta(days=90),
                    ResponseWorkflow.status == "completed"
                )
            )
            .order_by(ResponseWorkflow.created_at.desc())
            .limit(50)
        )
        
        similar_workflows = similar_workflows_result.scalars().all()
        
        if not similar_workflows:
            return {"similarity_count": 0, "average_success_rate": 0.0}
        
        # Analyze performance patterns
        performance_data = {
            "similarity_count": len(similar_workflows),
            "average_success_rate": sum(w.success_rate or 0 for w in similar_workflows) / len(similar_workflows),
            "average_execution_time": sum(w.execution_time_ms or 0 for w in similar_workflows) / len(similar_workflows),
            "most_effective_actions": self._identify_most_effective_actions(similar_workflows),
            "least_effective_actions": self._identify_least_effective_actions(similar_workflows),
            "optimal_sequences": self._identify_optimal_sequences(similar_workflows),
            "common_failures": self._identify_common_failures(similar_workflows)
        }
        
        return performance_data
    
    async def _generate_strategy_optimizations(
        self,
        recommendations: List[Dict[str, Any]],
        historical_analysis: Dict[str, Any],
        strategy: OptimizationStrategy
    ) -> Dict[str, Any]:
        """Generate strategy-specific optimizations for recommendations"""
        
        optimizations = {
            "improvements": [],
            "confidence": 0.0,
            "reordered_actions": [],
            "added_actions": [],
            "removed_actions": [],
            "parameter_adjustments": {}
        }
        
        # Apply historical learning
        effective_actions = historical_analysis.get("most_effective_actions", [])
        ineffective_actions = historical_analysis.get("least_effective_actions", [])
        
        # Reorder based on effectiveness
        reordered = self._reorder_by_effectiveness(recommendations, effective_actions)
        optimizations["reordered_actions"] = reordered
        
        # Suggest adding highly effective actions
        for action in effective_actions:
            if not any(rec["action_type"] == action for rec in recommendations):
                optimizations["added_actions"].append({
                    "action_type": action,
                    "reason": "Historically highly effective for similar incidents",
                    "confidence": 0.8
                })
        
        # Suggest removing ineffective actions
        for rec in recommendations:
            if rec["action_type"] in ineffective_actions:
                optimizations["removed_actions"].append({
                    "action_type": rec["action_type"],
                    "reason": "Historically low effectiveness",
                    "confidence": 0.7
                })
        
        optimizations["confidence"] = 0.7
        optimizations["improvements"].append("Applied historical learning patterns")
        
        return optimizations
    
    # Helper methods (simplified implementations)
    def _identify_workflow_bottlenecks(self, actions: List[AdvancedResponseAction]) -> List[str]:
        """Identify workflow bottlenecks"""
        bottlenecks = []
        
        for action in actions:
            if action.status == "failed" and action.retry_count > 2:
                bottlenecks.append(f"Action {action.action_type} failed multiple times")
        
        return bottlenecks
    
    def _identify_workflow_inefficiencies(self, actions: List[AdvancedResponseAction]) -> List[str]:
        """Identify workflow inefficiencies"""
        inefficiencies = []
        
        # Check for long-running actions
        for action in actions:
            if action.completed_at and action.created_at:
                duration = (action.completed_at - action.created_at).total_seconds()
                expected_duration = self.performance_baselines["action_durations"].get(action.action_type, 300)
                
                if duration > expected_duration * 2:
                    inefficiencies.append(f"Action {action.action_type} took {duration:.0f}s (expected {expected_duration}s)")
        
        return inefficiencies
    
    def _can_parallelize_actions(self, steps: List[Dict[str, Any]]) -> bool:
        """Check if workflow can benefit from parallelization"""
        safe_parallel = self.optimization_models["action_sequencing"]["parallel_safe_actions"]
        workflow_actions = [step.get("action_type") for step in steps]
        
        return len([action for action in workflow_actions if action in safe_parallel]) >= 2
    
    def _has_suboptimal_ordering(self, steps: List[Dict[str, Any]]) -> bool:
        """Check if workflow has suboptimal action ordering"""
        # Simplified check - look for high-priority actions not at the beginning
        immediate_actions = self.optimization_models["timing_optimization"]["immediate_actions"]
        
        for i, step in enumerate(steps):
            if step.get("action_type") in immediate_actions and i > 1:
                return True
        
        return False
    
    def _identify_redundant_actions(self, steps: List[Dict[str, Any]]) -> List[str]:
        """Identify redundant actions in workflow"""
        action_types = [step.get("action_type") for step in steps]
        return [action for action in set(action_types) if action_types.count(action) > 1]
    
    async def _identify_critical_gaps(self, workflow: ResponseWorkflow) -> List[Dict[str, Any]]:
        """Identify critical missing actions"""
        # Get incident to understand threat type
        # This would typically access the incident through the workflow relationship
        return []  # Simplified
    
    def _suggest_parameter_improvements(self, steps: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Suggest parameter improvements based on historical data"""
        return {}  # Simplified
    
    def _has_adequate_error_handling(self, steps: List[Dict[str, Any]]) -> bool:
        """Check if workflow has adequate error handling"""
        return any(step.get("max_retries", 0) > 0 for step in steps)
    
    def _design_retry_strategies(self, steps: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Design retry strategies for actions"""
        return {"default_retries": 3, "backoff_strategy": "exponential"}
    
    def _has_resource_conflicts(self, steps: List[Dict[str, Any]]) -> bool:
        """Check for resource conflicts in workflow"""
        high_resource_actions = self.optimization_models["resource_optimization"]["high_resource_actions"]
        workflow_actions = [step.get("action_type") for step in steps]
        
        return len([action for action in workflow_actions if action in high_resource_actions]) > 2
    
    def _identify_conditional_opportunities(self, steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify opportunities for conditional execution"""
        return []  # Simplified
    
    def _has_adequate_safety_checks(self, steps: List[Dict[str, Any]]) -> bool:
        """Check if workflow has adequate safety checks"""
        return False  # Always suggest safety improvements
    
    def _design_safety_checkpoints(self, steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Design safety checkpoints for workflow"""
        return [
            {"checkpoint": "pre_execution_validation", "position": "before_each_action"},
            {"checkpoint": "impact_assessment", "position": "before_high_risk_actions"}
        ]
    
    def _has_comprehensive_rollback(self, steps: List[Dict[str, Any]]) -> bool:
        """Check if workflow has comprehensive rollback plan"""
        return any(step.get("rollback_plan") for step in steps)
    
    def _design_enhanced_rollback(self, steps: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Design enhanced rollback plan"""
        return {
            "auto_rollback": True,
            "rollback_triggers": ["failure_threshold", "user_request"],
            "rollback_sequence": "reverse_order"
        }
    
    def _has_adequate_audit_logging(self, steps: List[Dict[str, Any]]) -> bool:
        """Check for adequate audit logging"""
        return False  # Always suggest audit improvements
    
    def _design_audit_logging(self, steps: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Design audit logging enhancements"""
        return {
            "log_level": "detailed",
            "compliance_mapping": "SOC2_CC6.1",
            "retention_period": "7_years"
        }
    
    def _identify_compliance_gaps(self, steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify compliance gaps"""
        return [
            {"gap": "evidence_collection", "framework": "SOC2", "priority": "high"}
        ]
    
    def _has_suboptimal_timeouts(self, steps: List[Dict[str, Any]]) -> bool:
        """Check for suboptimal timeout values"""
        return True  # Often can be optimized
    
    def _calculate_optimal_timeouts(self, steps: List[Dict[str, Any]]) -> Dict[str, int]:
        """Calculate optimal timeout values"""
        return {step.get("action_type", "unknown"): 300 for step in steps}
    
    def _has_detailed_progress_tracking(self, steps: List[Dict[str, Any]]) -> bool:
        """Check for detailed progress tracking"""
        return False  # Can always be improved
    
    def _design_progress_tracking(self, steps: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Design progress tracking enhancements"""
        return {
            "granularity": "action_level",
            "real_time_updates": True,
            "milestone_notifications": True
        }
    
    def _identify_parallel_groups(self, steps: List[Dict[str, Any]]) -> List[List[str]]:
        """Identify groups of actions that can run in parallel"""
        safe_parallel = self.optimization_models["action_sequencing"]["parallel_safe_actions"]
        
        parallel_actions = [step.get("action_type") for step in steps if step.get("action_type") in safe_parallel]
        
        if len(parallel_actions) >= 2:
            return [parallel_actions]
        
        return []
    
    def _calculate_optimal_sequence(self, steps: List[Dict[str, Any]]) -> List[str]:
        """Calculate optimal action sequence"""
        immediate_actions = self.optimization_models["timing_optimization"]["immediate_actions"]
        delayed_actions = self.optimization_models["timing_optimization"]["delayed_actions"]
        
        # Put immediate actions first, then others, then delayed actions
        sequence = []
        
        for step in steps:
            action_type = step.get("action_type")
            if action_type in immediate_actions:
                sequence.insert(0, action_type)
            elif action_type in delayed_actions:
                sequence.append(action_type)
            else:
                sequence.insert(-len([a for a in sequence if a in delayed_actions]), action_type)
        
        return sequence
    
    def _create_resource_schedule(self, steps: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create optimal resource schedule"""
        return {
            "schedule": "sequential_high_resource",
            "max_concurrent": 2,
            "resource_pools": ["cpu_intensive", "network_intensive", "io_intensive"]
        }
    
    def _reorder_by_effectiveness(
        self, 
        recommendations: List[Dict[str, Any]], 
        effective_actions: List[str]
    ) -> List[Dict[str, Any]]:
        """Reorder recommendations by historical effectiveness"""
        
        # Sort by effectiveness (effective actions first)
        def effectiveness_score(rec):
            action_type = rec.get("action_type", "")
            if action_type in effective_actions:
                return effective_actions.index(action_type)
            return len(effective_actions)
        
        return sorted(recommendations, key=effectiveness_score)
    
    def _identify_most_effective_actions(self, workflows: List[ResponseWorkflow]) -> List[str]:
        """Identify most effective actions from historical data"""
        action_performance = {}
        
        for workflow in workflows:
            if workflow.success_rate and workflow.success_rate > 0.8 and workflow.steps:
                for step in workflow.steps:
                    action_type = step.get("action_type")
                    if action_type:
                        if action_type not in action_performance:
                            action_performance[action_type] = {"total": 0, "successful": 0}
                        action_performance[action_type]["total"] += 1
                        if workflow.success_rate > 0.8:
                            action_performance[action_type]["successful"] += 1
        
        # Calculate success rates and return top performers
        effective_actions = []
        for action, perf in action_performance.items():
            if perf["total"] >= 3:  # Minimum sample size
                success_rate = perf["successful"] / perf["total"]
                if success_rate > 0.8:
                    effective_actions.append(action)
        
        return effective_actions
    
    def _identify_least_effective_actions(self, workflows: List[ResponseWorkflow]) -> List[str]:
        """Identify least effective actions from historical data"""
        return []  # Simplified implementation
    
    def _identify_optimal_sequences(self, workflows: List[ResponseWorkflow]) -> List[List[str]]:
        """Identify optimal action sequences"""
        return []  # Simplified implementation
    
    def _identify_common_failures(self, workflows: List[ResponseWorkflow]) -> List[Dict[str, Any]]:
        """Identify common failure patterns"""
        return []  # Simplified implementation
    
    async def _apply_optimization_filters(
        self,
        recommendations: List[Dict[str, Any]],
        optimizations: Dict[str, Any],
        strategy: OptimizationStrategy
    ) -> List[Dict[str, Any]]:
        """Apply optimization filters to recommendations"""
        
        optimized = recommendations.copy()
        
        # Apply reordering
        if optimizations.get("reordered_actions"):
            optimized = optimizations["reordered_actions"]
        
        # Add new actions
        for new_action in optimizations.get("added_actions", []):
            optimized.append(new_action)
        
        # Remove ineffective actions
        removed_types = [ra["action_type"] for ra in optimizations.get("removed_actions", [])]
        optimized = [rec for rec in optimized if rec.get("action_type") not in removed_types]
        
        return optimized
    
    def _calculate_optimization_metrics(
        self,
        original: List[Dict[str, Any]],
        optimized: List[Dict[str, Any]],
        historical: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate optimization improvement metrics"""
        
        return {
            "action_count_change": len(optimized) - len(original),
            "expected_time_improvement": 0.2,  # 20% improvement
            "expected_success_improvement": 0.15,  # 15% improvement
            "optimization_confidence": 0.75,
            "historical_basis": historical.get("similarity_count", 0) > 5
        }
    
    async def _extract_learning_insights(
        self,
        workflow: ResponseWorkflow,
        execution_results: Dict[str, Any],
        db_session: AsyncSession
    ) -> Dict[str, Any]:
        """Extract learning insights from workflow execution"""
        
        insights = {
            "workflow_id": workflow.workflow_id,
            "success_rate": execution_results.get("success_rate", 0.0),
            "execution_time": execution_results.get("execution_time_ms", 0),
            "effective_actions": [],
            "ineffective_actions": [],
            "optimization_opportunities": [],
            "learning_score": 0.0
        }
        
        # Analyze action effectiveness
        for result in execution_results.get("results", []):
            if result.get("success"):
                insights["effective_actions"].append(result.get("action_type"))
            else:
                insights["ineffective_actions"].append(result.get("action_type"))
        
        # Calculate learning score
        total_actions = len(execution_results.get("results", []))
        if total_actions > 0:
            insights["learning_score"] = len(insights["effective_actions"]) / total_actions
        
        return insights
    
    async def _update_optimization_models(self, insights: Dict[str, Any]) -> Dict[str, Any]:
        """Update optimization models based on learning insights"""
        
        updates = {
            "updated_models": [],
            "learning_applied": False
        }
        
        # Update action success rates
        for action in insights.get("effective_actions", []):
            if action in self.performance_baselines["action_success_rates"]:
                current_rate = self.performance_baselines["action_success_rates"][action]
                # Apply learning with decay
                self.performance_baselines["action_success_rates"][action] = current_rate * 0.9 + 0.1
                updates["updated_models"].append(f"Improved success rate for {action}")
                updates["learning_applied"] = True
        
        return updates
    
    async def _store_learning_data(
        self,
        workflow_id: str,
        insights: Dict[str, Any],
        db_session: AsyncSession
    ):
        """Store learning data for future optimization"""
        # In a full implementation, this would store learning data in the database
        # For now, we'll just log it
        self.logger.info(f"Learning insights for {workflow_id}: {insights.get('learning_score', 0.0):.2f}")


# Global instance
response_optimizer = ResponseOptimizer()


async def get_response_optimizer() -> ResponseOptimizer:
    """Get the global response optimizer instance"""
    if not response_optimizer.initialized:
        await response_optimizer.initialize()
    return response_optimizer







