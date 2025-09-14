"""
Advanced Multi-Agent Coordination Hub for Mini-XDR
Handles sophisticated agent collaboration, conflict resolution, and decision optimization
"""
import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
from collections import defaultdict, deque
import statistics

from ..models import Incident, Event, Action
from ..config import settings

logger = logging.getLogger(__name__)


class CoordinationStrategy(Enum):
    """Strategies for multi-agent coordination"""
    SEQUENTIAL = "sequential"          # Agents execute in sequence
    PARALLEL = "parallel"             # Agents execute simultaneously
    HIERARCHICAL = "hierarchical"     # Priority-based execution
    CONSENSUS = "consensus"           # Consensus-based decision making
    COMPETITIVE = "competitive"       # Best result wins
    COLLABORATIVE = "collaborative"   # Agents share information and collaborate


class DecisionMethod(Enum):
    """Methods for combining agent decisions"""
    MAJORITY_VOTE = "majority_vote"
    WEIGHTED_AVERAGE = "weighted_average"
    CONFIDENCE_WEIGHTED = "confidence_weighted"
    EXPERT_OVERRIDE = "expert_override"
    ENSEMBLE_STACKING = "ensemble_stacking"


class ConflictResolutionStrategy(Enum):
    """Strategies for resolving agent conflicts"""
    HIGHEST_CONFIDENCE = "highest_confidence"
    MOST_EXPERIENCED = "most_experienced"
    CONSENSUS_BUILDING = "consensus_building"
    HUMAN_ESCALATION = "human_escalation"
    META_ANALYSIS = "meta_analysis"


@dataclass
class AgentCapability:
    """Defines what an agent can do"""
    name: str
    domain_expertise: List[str]
    confidence_threshold: float
    execution_time_estimate: float
    resource_requirements: Dict[str, Any]
    dependencies: List[str]
    success_rate: float = 0.0
    last_updated: datetime = None


@dataclass
class CoordinationContext:
    """Context for coordination decisions"""
    incident_severity: str
    time_constraints: Optional[timedelta]
    resource_availability: Dict[str, float]
    stakeholder_priority: str
    automation_level: str
    risk_tolerance: float
    compliance_requirements: List[str]


@dataclass
class AgentDecisionNode:
    """Represents a decision point in agent coordination"""
    agent_id: str
    decision_type: str
    confidence: float
    supporting_evidence: List[str]
    alternative_options: List[Dict[str, Any]]
    execution_cost: float
    potential_impact: float
    risk_factors: List[str]
    timestamp: datetime


@dataclass
class CoordinationPlan:
    """Plan for coordinating multiple agents"""
    plan_id: str
    strategy: CoordinationStrategy
    execution_order: List[str]
    decision_points: List[AgentDecisionNode]
    fallback_options: List[str]
    estimated_duration: float
    resource_allocation: Dict[str, float]
    success_probability: float


class AgentPerformanceTracker:
    """Tracks and analyzes agent performance over time"""
    
    def __init__(self):
        self.performance_history = defaultdict(list)
        self.success_rates = defaultdict(float)
        self.response_times = defaultdict(list)
        self.collaboration_scores = defaultdict(dict)
        self.specialization_metrics = defaultdict(dict)
    
    def record_performance(
        self, 
        agent_id: str, 
        task_type: str, 
        success: bool,
        execution_time: float,
        confidence_score: float,
        quality_score: float
    ):
        """Record an agent's performance on a specific task"""
        record = {
            'timestamp': datetime.utcnow(),
            'task_type': task_type,
            'success': success,
            'execution_time': execution_time,
            'confidence_score': confidence_score,
            'quality_score': quality_score
        }
        
        self.performance_history[agent_id].append(record)
        
        # Update running averages
        recent_records = self.performance_history[agent_id][-100:]  # Last 100 records
        success_count = sum(1 for r in recent_records if r['success'])
        self.success_rates[agent_id] = success_count / len(recent_records)
        
        self.response_times[agent_id].append(execution_time)
        if len(self.response_times[agent_id]) > 100:
            self.response_times[agent_id] = self.response_times[agent_id][-100:]
    
    def get_agent_expertise_score(self, agent_id: str, domain: str) -> float:
        """Calculate agent's expertise score in a specific domain"""
        if agent_id not in self.performance_history:
            return 0.5  # Default neutral score
        
        domain_records = [
            r for r in self.performance_history[agent_id]
            if domain.lower() in r.get('task_type', '').lower()
        ]
        
        if not domain_records:
            return 0.5
        
        success_rate = sum(r['success'] for r in domain_records) / len(domain_records)
        avg_quality = statistics.mean(r['quality_score'] for r in domain_records)
        avg_confidence = statistics.mean(r['confidence_score'] for r in domain_records)
        
        # Combined expertise score
        expertise_score = (success_rate * 0.4 + avg_quality * 0.3 + avg_confidence * 0.3)
        return min(max(expertise_score, 0.0), 1.0)
    
    def recommend_agent_for_task(
        self, 
        task_type: str, 
        available_agents: List[str],
        urgency: str = "medium"
    ) -> List[Tuple[str, float]]:
        """Recommend the best agents for a specific task"""
        recommendations = []
        
        for agent_id in available_agents:
            if agent_id not in self.performance_history:
                score = 0.3  # Low score for unknown agents
            else:
                expertise = self.get_agent_expertise_score(agent_id, task_type)
                success_rate = self.success_rates.get(agent_id, 0.5)
                
                # Factor in response time for urgent tasks
                if urgency == "high" and agent_id in self.response_times:
                    avg_response_time = statistics.mean(self.response_times[agent_id])
                    time_factor = max(0, 1 - (avg_response_time / 60))  # Penalty for slow response
                    score = (expertise * 0.6 + success_rate * 0.3 + time_factor * 0.1)
                else:
                    score = (expertise * 0.7 + success_rate * 0.3)
                
                score = min(max(score, 0.0), 1.0)
            
            recommendations.append((agent_id, score))
        
        # Sort by score descending
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations


class DecisionTree:
    """Decision tree for agent coordination logic"""
    
    def __init__(self):
        self.rules = []
        self.default_strategy = CoordinationStrategy.PARALLEL
    
    def add_rule(
        self, 
        conditions: Dict[str, Any], 
        action: CoordinationStrategy,
        priority: int = 1
    ):
        """Add a coordination rule"""
        self.rules.append({
            'conditions': conditions,
            'action': action,
            'priority': priority,
            'created_at': datetime.utcnow()
        })
        
        # Sort by priority
        self.rules.sort(key=lambda x: x['priority'], reverse=True)
    
    def evaluate(self, context: CoordinationContext) -> CoordinationStrategy:
        """Evaluate context and return appropriate coordination strategy"""
        
        for rule in self.rules:
            if self._match_conditions(rule['conditions'], context):
                return rule['action']
        
        return self.default_strategy
    
    def _match_conditions(self, conditions: Dict[str, Any], context: CoordinationContext) -> bool:
        """Check if context matches rule conditions"""
        
        for key, expected_value in conditions.items():
            context_value = getattr(context, key, None)
            
            if isinstance(expected_value, list):
                if context_value not in expected_value:
                    return False
            elif isinstance(expected_value, dict):
                # Handle range conditions
                if 'min' in expected_value and context_value < expected_value['min']:
                    return False
                if 'max' in expected_value and context_value > expected_value['max']:
                    return False
            else:
                if context_value != expected_value:
                    return False
        
        return True


class ConflictResolver:
    """Resolves conflicts between agent recommendations"""
    
    def __init__(self, performance_tracker: AgentPerformanceTracker):
        self.performance_tracker = performance_tracker
        self.resolution_history = deque(maxlen=1000)
    
    async def resolve_conflicts(
        self,
        conflicting_decisions: List[AgentDecisionNode],
        context: CoordinationContext,
        resolution_strategy: ConflictResolutionStrategy = ConflictResolutionStrategy.HIGHEST_CONFIDENCE
    ) -> AgentDecisionNode:
        """Resolve conflicts between agent decisions"""
        
        if len(conflicting_decisions) == 1:
            return conflicting_decisions[0]
        
        if not conflicting_decisions:
            raise ValueError("No decisions provided for conflict resolution")
        
        resolution_start = datetime.utcnow()
        
        try:
            if resolution_strategy == ConflictResolutionStrategy.HIGHEST_CONFIDENCE:
                resolved = max(conflicting_decisions, key=lambda d: d.confidence)
            
            elif resolution_strategy == ConflictResolutionStrategy.MOST_EXPERIENCED:
                # Use agent with highest expertise in this domain
                agent_scores = []
                for decision in conflicting_decisions:
                    expertise = self.performance_tracker.get_agent_expertise_score(
                        decision.agent_id, decision.decision_type
                    )
                    agent_scores.append((decision, expertise))
                
                resolved = max(agent_scores, key=lambda x: x[1])[0]
            
            elif resolution_strategy == ConflictResolutionStrategy.CONSENSUS_BUILDING:
                resolved = await self._build_consensus(conflicting_decisions, context)
            
            elif resolution_strategy == ConflictResolutionStrategy.META_ANALYSIS:
                resolved = await self._meta_analysis_resolution(conflicting_decisions, context)
            
            else:
                # Default to highest confidence
                resolved = max(conflicting_decisions, key=lambda d: d.confidence)
            
            # Record the resolution
            resolution_record = {
                'timestamp': datetime.utcnow(),
                'strategy': resolution_strategy,
                'num_conflicts': len(conflicting_decisions),
                'resolution_time': (datetime.utcnow() - resolution_start).total_seconds(),
                'resolved_agent': resolved.agent_id,
                'resolved_confidence': resolved.confidence
            }
            self.resolution_history.append(resolution_record)
            
            logger.info(f"Resolved conflict between {len(conflicting_decisions)} agents using {resolution_strategy.value}")
            
            return resolved
            
        except Exception as e:
            logger.error(f"Conflict resolution failed: {e}")
            # Fallback to highest confidence
            return max(conflicting_decisions, key=lambda d: d.confidence)
    
    async def _build_consensus(
        self, 
        decisions: List[AgentDecisionNode], 
        context: CoordinationContext
    ) -> AgentDecisionNode:
        """Build consensus among conflicting decisions"""
        
        # Group similar decisions
        decision_groups = defaultdict(list)
        for decision in decisions:
            decision_groups[decision.decision_type].append(decision)
        
        # Find the group with highest combined confidence
        best_group = None
        best_score = 0
        
        for decision_type, group in decision_groups.items():
            combined_confidence = sum(d.confidence for d in group) / len(group)
            group_size_bonus = len(group) * 0.1  # Bonus for agreement
            total_score = combined_confidence + group_size_bonus
            
            if total_score > best_score:
                best_score = total_score
                best_group = group
        
        if best_group:
            # Return decision with highest confidence from best group
            return max(best_group, key=lambda d: d.confidence)
        
        # Fallback
        return max(decisions, key=lambda d: d.confidence)
    
    async def _meta_analysis_resolution(
        self, 
        decisions: List[AgentDecisionNode], 
        context: CoordinationContext
    ) -> AgentDecisionNode:
        """Use meta-analysis to resolve conflicts"""
        
        # Calculate composite scores for each decision
        scored_decisions = []
        
        for decision in decisions:
            # Base confidence score
            score = decision.confidence * 0.4
            
            # Agent expertise bonus
            expertise = self.performance_tracker.get_agent_expertise_score(
                decision.agent_id, decision.decision_type
            )
            score += expertise * 0.3
            
            # Evidence quality bonus
            evidence_score = len(decision.supporting_evidence) * 0.05
            score += min(evidence_score, 0.2)  # Cap at 0.2
            
            # Risk-adjusted score
            risk_penalty = len(decision.risk_factors) * 0.02
            score -= min(risk_penalty, 0.1)  # Cap penalty at 0.1
            
            # Context alignment bonus
            if context.incident_severity == "critical" and decision.potential_impact > 0.8:
                score += 0.1  # Bonus for high-impact decisions on critical incidents
            
            scored_decisions.append((decision, score))
        
        # Return decision with highest composite score
        return max(scored_decisions, key=lambda x: x[1])[0]


class AdvancedCoordinationHub:
    """
    Advanced Multi-Agent Coordination Hub
    Orchestrates complex agent interactions with sophisticated decision-making
    """
    
    def __init__(self):
        self.hub_id = "coordination_hub_v2"
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Core components
        self.performance_tracker = AgentPerformanceTracker()
        self.decision_tree = DecisionTree()
        self.conflict_resolver = ConflictResolver(self.performance_tracker)
        
        # Agent registry
        self.agent_capabilities: Dict[str, AgentCapability] = {}
        self.active_agents: Set[str] = set()
        self.agent_load: Dict[str, int] = defaultdict(int)
        
        # Coordination state
        self.active_coordinations: Dict[str, CoordinationPlan] = {}
        self.coordination_history: deque = deque(maxlen=1000)
        
        # Performance metrics
        self.stats = {
            "coordinations_executed": 0,
            "conflicts_resolved": 0,
            "average_coordination_time": 0.0,
            "success_rate": 0.0,
            "last_activity": datetime.utcnow()
        }
        
        # Initialize default coordination rules
        self._initialize_coordination_rules()
    
    def _initialize_coordination_rules(self):
        """Initialize default coordination rules"""
        
        # Critical incidents require hierarchical coordination
        self.decision_tree.add_rule(
            conditions={'incident_severity': 'critical'},
            action=CoordinationStrategy.HIERARCHICAL,
            priority=10
        )
        
        # High automation level enables parallel execution
        self.decision_tree.add_rule(
            conditions={'automation_level': 'high'},
            action=CoordinationStrategy.PARALLEL,
            priority=8
        )
        
        # Time constraints require competitive approach
        self.decision_tree.add_rule(
            conditions={'time_constraints': {'max': timedelta(minutes=5)}},
            action=CoordinationStrategy.COMPETITIVE,
            priority=9
        )
        
        # Low risk tolerance requires consensus
        self.decision_tree.add_rule(
            conditions={'risk_tolerance': {'max': 0.3}},
            action=CoordinationStrategy.CONSENSUS,
            priority=7
        )
    
    def register_agent(
        self, 
        agent_id: str, 
        capabilities: AgentCapability
    ):
        """Register an agent with its capabilities"""
        self.agent_capabilities[agent_id] = capabilities
        self.active_agents.add(agent_id)
        self.logger.info(f"Registered agent {agent_id} with capabilities: {capabilities.domain_expertise}")
    
    def unregister_agent(self, agent_id: str):
        """Unregister an agent"""
        if agent_id in self.agent_capabilities:
            del self.agent_capabilities[agent_id]
        self.active_agents.discard(agent_id)
        if agent_id in self.agent_load:
            del self.agent_load[agent_id]
        self.logger.info(f"Unregistered agent {agent_id}")
    
    async def coordinate_agents(
        self,
        incident: Incident,
        required_capabilities: List[str],
        context: CoordinationContext,
        max_agents: int = 5
    ) -> Dict[str, Any]:
        """
        Coordinate multiple agents for incident response
        
        Args:
            incident: The incident requiring coordination
            required_capabilities: List of required agent capabilities
            context: Coordination context
            max_agents: Maximum number of agents to coordinate
            
        Returns:
            Coordination results with agent outputs and decisions
        """
        
        coordination_start = datetime.utcnow()
        coordination_id = f"coord_{incident.id}_{int(coordination_start.timestamp())}"
        
        try:
            self.logger.info(f"Starting agent coordination {coordination_id} for incident {incident.id}")
            
            # 1. Select appropriate agents
            selected_agents = await self._select_agents(
                required_capabilities, context, max_agents
            )
            
            if not selected_agents:
                return {
                    "success": False,
                    "error": "No suitable agents available for coordination",
                    "coordination_id": coordination_id
                }
            
            # 2. Determine coordination strategy
            strategy = self.decision_tree.evaluate(context)
            
            # 3. Create coordination plan
            plan = await self._create_coordination_plan(
                coordination_id, selected_agents, strategy, incident, context
            )
            
            self.active_coordinations[coordination_id] = plan
            
            # 4. Execute coordination based on strategy
            results = await self._execute_coordination(plan, incident, context)
            
            # 5. Process and resolve conflicts if any
            if len(results.get('agent_decisions', [])) > 1:
                resolved_decision = await self.conflict_resolver.resolve_conflicts(
                    results['agent_decisions'],
                    context,
                    ConflictResolutionStrategy.META_ANALYSIS
                )
                results['final_decision'] = resolved_decision
                self.stats["conflicts_resolved"] += 1
            
            # 6. Update performance metrics
            execution_time = (datetime.utcnow() - coordination_start).total_seconds()
            await self._update_performance_metrics(
                selected_agents, results, execution_time, True
            )
            
            # 7. Clean up
            if coordination_id in self.active_coordinations:
                del self.active_coordinations[coordination_id]
            
            self.stats["coordinations_executed"] += 1
            self.stats["last_activity"] = datetime.utcnow()
            
            # Record coordination history
            coordination_record = {
                'coordination_id': coordination_id,
                'incident_id': incident.id,
                'strategy': strategy.value,
                'agents_involved': [agent_id for agent_id, _ in selected_agents],
                'execution_time': execution_time,
                'success': results.get('success', True),
                'timestamp': coordination_start
            }
            self.coordination_history.append(coordination_record)
            
            self.logger.info(f"Completed coordination {coordination_id} in {execution_time:.2f}s")
            
            return {
                "success": True,
                "coordination_id": coordination_id,
                "strategy": strategy.value,
                "agents_involved": [agent_id for agent_id, _ in selected_agents],
                "execution_time": execution_time,
                "results": results,
                "performance_summary": self._generate_performance_summary(results)
            }
            
        except Exception as e:
            self.logger.error(f"Coordination {coordination_id} failed: {e}")
            
            # Clean up on failure
            if coordination_id in self.active_coordinations:
                del self.active_coordinations[coordination_id]
            
            return {
                "success": False,
                "coordination_id": coordination_id,
                "error": str(e),
                "partial_results": {}
            }
    
    async def _select_agents(
        self, 
        required_capabilities: List[str], 
        context: CoordinationContext,
        max_agents: int
    ) -> List[Tuple[str, float]]:
        """Select the best agents for the coordination"""
        
        candidate_agents = []
        
        for agent_id in self.active_agents:
            capabilities = self.agent_capabilities.get(agent_id)
            if not capabilities:
                continue
            
            # Check if agent has required capabilities
            capability_match = any(
                cap in capabilities.domain_expertise 
                for cap in required_capabilities
            )
            
            if not capability_match:
                continue
            
            # Calculate agent suitability score
            base_score = capabilities.success_rate
            
            # Factor in current load
            load_penalty = self.agent_load.get(agent_id, 0) * 0.1
            load_adjusted_score = max(0, base_score - load_penalty)
            
            # Get performance-based recommendations
            task_type = required_capabilities[0] if required_capabilities else "general"
            performance_recommendations = self.performance_tracker.recommend_agent_for_task(
                task_type, [agent_id], context.stakeholder_priority
            )
            
            if performance_recommendations:
                performance_score = performance_recommendations[0][1]
                final_score = (load_adjusted_score * 0.6 + performance_score * 0.4)
            else:
                final_score = load_adjusted_score
            
            candidate_agents.append((agent_id, final_score))
        
        # Sort by score and select top agents
        candidate_agents.sort(key=lambda x: x[1], reverse=True)
        selected_agents = candidate_agents[:max_agents]
        
        self.logger.info(f"Selected {len(selected_agents)} agents from {len(candidate_agents)} candidates")
        
        return selected_agents
    
    async def _create_coordination_plan(
        self,
        coordination_id: str,
        selected_agents: List[Tuple[str, float]],
        strategy: CoordinationStrategy,
        incident: Incident,
        context: CoordinationContext
    ) -> CoordinationPlan:
        """Create a detailed coordination plan"""
        
        agent_ids = [agent_id for agent_id, _ in selected_agents]
        
        # Determine execution order based on strategy
        if strategy == CoordinationStrategy.HIERARCHICAL:
            # Order by agent scores (highest first)
            execution_order = [agent_id for agent_id, score in 
                             sorted(selected_agents, key=lambda x: x[1], reverse=True)]
        elif strategy == CoordinationStrategy.SEQUENTIAL:
            # Order by dependencies and capabilities
            execution_order = self._calculate_dependency_order(agent_ids)
        else:
            # Parallel or competitive - order doesn't matter much
            execution_order = agent_ids
        
        # Estimate resource requirements
        resource_allocation = {}
        total_cpu = 0
        total_memory = 0
        
        for agent_id in agent_ids:
            capabilities = self.agent_capabilities.get(agent_id)
            if capabilities and capabilities.resource_requirements:
                cpu_req = capabilities.resource_requirements.get('cpu', 0.1)
                memory_req = capabilities.resource_requirements.get('memory', 0.1)
                resource_allocation[agent_id] = {'cpu': cpu_req, 'memory': memory_req}
                total_cpu += cpu_req
                total_memory += memory_req
        
        # Estimate execution duration
        if strategy == CoordinationStrategy.PARALLEL:
            # Maximum of all agent execution times
            estimated_duration = max(
                self.agent_capabilities[agent_id].execution_time_estimate
                for agent_id in agent_ids
                if agent_id in self.agent_capabilities
            )
        else:
            # Sum of all agent execution times
            estimated_duration = sum(
                self.agent_capabilities[agent_id].execution_time_estimate
                for agent_id in agent_ids
                if agent_id in self.agent_capabilities
            )
        
        # Calculate success probability
        agent_success_rates = [
            self.agent_capabilities[agent_id].success_rate
            for agent_id in agent_ids
            if agent_id in self.agent_capabilities
        ]
        
        if strategy == CoordinationStrategy.PARALLEL:
            # All agents need to succeed
            success_probability = 1.0
            for rate in agent_success_rates:
                success_probability *= rate
        else:
            # Average success rate
            success_probability = sum(agent_success_rates) / len(agent_success_rates) if agent_success_rates else 0.5
        
        return CoordinationPlan(
            plan_id=coordination_id,
            strategy=strategy,
            execution_order=execution_order,
            decision_points=[],
            fallback_options=self._generate_fallback_options(agent_ids, context),
            estimated_duration=estimated_duration,
            resource_allocation=resource_allocation,
            success_probability=success_probability
        )
    
    async def _execute_coordination(
        self, 
        plan: CoordinationPlan, 
        incident: Incident, 
        context: CoordinationContext
    ) -> Dict[str, Any]:
        """Execute the coordination plan"""
        
        results = {
            'strategy': plan.strategy.value,
            'agents_executed': [],
            'agent_results': {},
            'agent_decisions': [],
            'execution_details': {},
            'success': True
        }
        
        try:
            if plan.strategy == CoordinationStrategy.PARALLEL:
                results = await self._execute_parallel_coordination(plan, incident, context, results)
            elif plan.strategy == CoordinationStrategy.SEQUENTIAL:
                results = await self._execute_sequential_coordination(plan, incident, context, results)
            elif plan.strategy == CoordinationStrategy.HIERARCHICAL:
                results = await self._execute_hierarchical_coordination(plan, incident, context, results)
            elif plan.strategy == CoordinationStrategy.COMPETITIVE:
                results = await self._execute_competitive_coordination(plan, incident, context, results)
            else:
                # Default to collaborative
                results = await self._execute_collaborative_coordination(plan, incident, context, results)
            
        except Exception as e:
            self.logger.error(f"Coordination execution failed: {e}")
            results['success'] = False
            results['error'] = str(e)
        
        return results
    
    async def _execute_parallel_coordination(
        self, plan: CoordinationPlan, incident: Incident, context: CoordinationContext, results: Dict
    ) -> Dict[str, Any]:
        """Execute agents in parallel"""
        
        # Create tasks for all agents
        tasks = []
        for agent_id in plan.execution_order:
            task = asyncio.create_task(
                self._execute_single_agent(agent_id, incident, context)
            )
            tasks.append((agent_id, task))
        
        # Wait for all tasks to complete
        for agent_id, task in tasks:
            try:
                agent_result = await task
                results['agents_executed'].append(agent_id)
                results['agent_results'][agent_id] = agent_result
                
                # Track decision if provided
                if 'decision' in agent_result:
                    results['agent_decisions'].append(agent_result['decision'])
                
            except Exception as e:
                self.logger.error(f"Agent {agent_id} failed in parallel execution: {e}")
                results['agent_results'][agent_id] = {'error': str(e), 'success': False}
        
        return results
    
    async def _execute_sequential_coordination(
        self, plan: CoordinationPlan, incident: Incident, context: CoordinationContext, results: Dict
    ) -> Dict[str, Any]:
        """Execute agents sequentially, passing context between them"""
        
        shared_context = {'incident': incident, 'context': context}
        
        for agent_id in plan.execution_order:
            try:
                # Add results from previous agents to context
                shared_context['previous_results'] = results['agent_results']
                
                agent_result = await self._execute_single_agent(
                    agent_id, incident, context, shared_context
                )
                
                results['agents_executed'].append(agent_id)
                results['agent_results'][agent_id] = agent_result
                
                # Track decision if provided
                if 'decision' in agent_result:
                    results['agent_decisions'].append(agent_result['decision'])
                
            except Exception as e:
                self.logger.error(f"Agent {agent_id} failed in sequential execution: {e}")
                results['agent_results'][agent_id] = {'error': str(e), 'success': False}
                # Continue with next agent despite failure
        
        return results
    
    async def _execute_hierarchical_coordination(
        self, plan: CoordinationPlan, incident: Incident, context: CoordinationContext, results: Dict
    ) -> Dict[str, Any]:
        """Execute agents hierarchically based on priority/expertise"""
        
        # Execute in order, but stop early if high-confidence decision is reached
        for agent_id in plan.execution_order:
            try:
                agent_result = await self._execute_single_agent(agent_id, incident, context)
                
                results['agents_executed'].append(agent_id)
                results['agent_results'][agent_id] = agent_result
                
                # Check if this agent provided a high-confidence decision
                if 'decision' in agent_result and 'confidence' in agent_result['decision']:
                    decision = agent_result['decision']
                    results['agent_decisions'].append(decision)
                    
                    # If high confidence and this is a high-priority agent, we can stop
                    if decision['confidence'] > 0.9 and len(results['agents_executed']) >= 1:
                        self.logger.info(f"Stopping hierarchical execution early due to high confidence from {agent_id}")
                        break
                
            except Exception as e:
                self.logger.error(f"Agent {agent_id} failed in hierarchical execution: {e}")
                results['agent_results'][agent_id] = {'error': str(e), 'success': False}
        
        return results
    
    async def _execute_competitive_coordination(
        self, plan: CoordinationPlan, incident: Incident, context: CoordinationContext, results: Dict
    ) -> Dict[str, Any]:
        """Execute agents competitively - fastest/best result wins"""
        
        # Start all agents but use the first successful high-quality result
        tasks = []
        for agent_id in plan.execution_order:
            task = asyncio.create_task(
                self._execute_single_agent(agent_id, incident, context)
            )
            tasks.append((agent_id, task))
        
        # Wait for first successful completion or all completions
        completed = 0
        winning_result = None
        winning_agent = None
        
        for agent_id, task in tasks:
            try:
                agent_result = await task
                completed += 1
                
                results['agents_executed'].append(agent_id)
                results['agent_results'][agent_id] = agent_result
                
                # Check if this is a winning result
                if (not winning_result and agent_result.get('success', True) and 
                    'decision' in agent_result and 
                    agent_result['decision'].get('confidence', 0) > 0.7):
                    
                    winning_result = agent_result
                    winning_agent = agent_id
                    results['agent_decisions'].append(agent_result['decision'])
                    
                    # Cancel remaining tasks for efficiency
                    for other_agent, other_task in tasks:
                        if other_agent != agent_id and not other_task.done():
                            other_task.cancel()
                    
                    break
                
            except Exception as e:
                self.logger.error(f"Agent {agent_id} failed in competitive execution: {e}")
                results['agent_results'][agent_id] = {'error': str(e), 'success': False}
        
        if winning_result:
            self.logger.info(f"Agent {winning_agent} won the competitive execution")
        
        return results
    
    async def _execute_collaborative_coordination(
        self, plan: CoordinationPlan, incident: Incident, context: CoordinationContext, results: Dict
    ) -> Dict[str, Any]:
        """Execute agents collaboratively with information sharing"""
        
        # Start with parallel execution but allow information sharing
        shared_state = {
            'findings': [],
            'hypotheses': [],
            'evidence': [],
            'recommendations': []
        }
        
        # Execute in rounds to allow collaboration
        for round_num in range(2):  # Two rounds for collaboration
            round_tasks = []
            
            for agent_id in plan.execution_order:
                enhanced_context = {
                    'incident': incident,
                    'context': context,
                    'shared_state': shared_state,
                    'round': round_num
                }
                
                task = asyncio.create_task(
                    self._execute_single_agent(agent_id, incident, context, enhanced_context)
                )
                round_tasks.append((agent_id, task))
            
            # Collect results from this round
            for agent_id, task in round_tasks:
                try:
                    agent_result = await task
                    
                    if agent_id not in results['agents_executed']:
                        results['agents_executed'].append(agent_id)
                    
                    results['agent_results'][agent_id] = agent_result
                    
                    # Update shared state with new findings
                    if 'findings' in agent_result:
                        shared_state['findings'].extend(agent_result['findings'])
                    if 'decision' in agent_result:
                        results['agent_decisions'].append(agent_result['decision'])
                    
                except Exception as e:
                    self.logger.error(f"Agent {agent_id} failed in collaborative execution round {round_num}: {e}")
                    results['agent_results'][agent_id] = {'error': str(e), 'success': False}
        
        return results
    
    async def _execute_single_agent(
        self, 
        agent_id: str, 
        incident: Incident, 
        context: CoordinationContext,
        additional_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute a single agent (placeholder - would integrate with actual agents)"""
        
        # This is a placeholder - in the real implementation, this would call the actual agent
        execution_start = datetime.utcnow()
        
        # Simulate agent execution
        await asyncio.sleep(0.1)  # Simulate processing time
        
        execution_time = (datetime.utcnow() - execution_start).total_seconds()
        
        # Record performance
        success = True  # Would be determined by actual agent execution
        confidence_score = 0.8  # Would come from agent
        quality_score = 0.75  # Would be calculated based on results
        
        self.performance_tracker.record_performance(
            agent_id=agent_id,
            task_type=context.incident_severity,
            success=success,
            execution_time=execution_time,
            confidence_score=confidence_score,
            quality_score=quality_score
        )
        
        # Return mock result structure
        return {
            'success': success,
            'agent_id': agent_id,
            'execution_time': execution_time,
            'decision': AgentDecisionNode(
                agent_id=agent_id,
                decision_type="mock_decision",
                confidence=confidence_score,
                supporting_evidence=["evidence_1", "evidence_2"],
                alternative_options=[],
                execution_cost=0.1,
                potential_impact=0.7,
                risk_factors=[],
                timestamp=datetime.utcnow()
            ),
            'findings': [
                {
                    'type': 'mock_finding',
                    'description': f'Mock finding from {agent_id}',
                    'confidence': confidence_score
                }
            ]
        }
    
    def _calculate_dependency_order(self, agent_ids: List[str]) -> List[str]:
        """Calculate execution order based on agent dependencies"""
        
        ordered = []
        remaining = set(agent_ids)
        
        while remaining:
            # Find agents with no unmet dependencies
            ready_agents = []
            for agent_id in remaining:
                capabilities = self.agent_capabilities.get(agent_id)
                if not capabilities or not capabilities.dependencies:
                    ready_agents.append(agent_id)
                else:
                    # Check if all dependencies are already ordered
                    unmet_deps = set(capabilities.dependencies) - set(ordered)
                    if not unmet_deps:
                        ready_agents.append(agent_id)
            
            if not ready_agents:
                # No agents ready - break circular dependencies by adding arbitrary agent
                ready_agents.append(next(iter(remaining)))
            
            # Add ready agents to order
            ordered.extend(ready_agents)
            remaining -= set(ready_agents)
        
        return ordered
    
    def _generate_fallback_options(
        self, 
        primary_agents: List[str], 
        context: CoordinationContext
    ) -> List[str]:
        """Generate fallback options if primary coordination fails"""
        
        fallbacks = []
        
        # Alternative agents for the same capabilities
        for agent_id in primary_agents:
            capabilities = self.agent_capabilities.get(agent_id)
            if capabilities:
                for alt_agent_id, alt_capabilities in self.agent_capabilities.items():
                    if (alt_agent_id != agent_id and 
                        any(cap in alt_capabilities.domain_expertise 
                            for cap in capabilities.domain_expertise)):
                        fallbacks.append(alt_agent_id)
        
        # Emergency escalation procedures
        if context.incident_severity == 'critical':
            fallbacks.extend(['human_escalation', 'emergency_containment'])
        
        return list(set(fallbacks))  # Remove duplicates
    
    async def _update_performance_metrics(
        self, 
        agents: List[Tuple[str, float]], 
        results: Dict[str, Any], 
        execution_time: float,
        success: bool
    ):
        """Update overall performance metrics"""
        
        # Update agent load tracking
        for agent_id, _ in agents:
            if agent_id in self.agent_load:
                self.agent_load[agent_id] = max(0, self.agent_load[agent_id] - 1)
        
        # Update running averages
        total_coordinations = self.stats["coordinations_executed"]
        current_avg_time = self.stats["average_coordination_time"]
        
        new_avg_time = ((current_avg_time * total_coordinations + execution_time) / 
                       (total_coordinations + 1))
        self.stats["average_coordination_time"] = new_avg_time
        
        # Update success rate
        current_success_rate = self.stats["success_rate"]
        new_success_rate = ((current_success_rate * total_coordinations + (1.0 if success else 0.0)) / 
                           (total_coordinations + 1))
        self.stats["success_rate"] = new_success_rate
    
    def _generate_performance_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a performance summary for the coordination"""
        
        summary = {
            'agents_participated': len(results.get('agents_executed', [])),
            'successful_agents': sum(1 for r in results.get('agent_results', {}).values() 
                                   if r.get('success', True)),
            'decisions_generated': len(results.get('agent_decisions', [])),
            'conflicts_detected': len(results.get('agent_decisions', [])) > 1,
            'overall_success': results.get('success', True)
        }
        
        # Calculate average confidence if decisions available
        decisions = results.get('agent_decisions', [])
        if decisions:
            confidences = [d.confidence for d in decisions if hasattr(d, 'confidence')]
            if confidences:
                summary['average_confidence'] = sum(confidences) / len(confidences)
        
        return summary
    
    def get_coordination_status(self) -> Dict[str, Any]:
        """Get comprehensive status of the coordination hub"""
        
        return {
            'hub_id': self.hub_id,
            'active_agents': len(self.active_agents),
            'registered_capabilities': len(self.agent_capabilities),
            'active_coordinations': len(self.active_coordinations),
            'coordination_rules': len(self.decision_tree.rules),
            'performance_stats': self.stats.copy(),
            'agent_load_distribution': dict(self.agent_load),
            'recent_coordinations': list(self.coordination_history)[-10:],
            'conflict_resolution_stats': {
                'total_resolutions': len(self.conflict_resolver.resolution_history),
                'recent_resolution_times': [
                    r['resolution_time'] for r in 
                    list(self.conflict_resolver.resolution_history)[-10:]
                ]
            }
        }


# Global coordination hub instance
coordination_hub = AdvancedCoordinationHub()


async def get_coordination_hub() -> AdvancedCoordinationHub:
    """Get the global coordination hub instance"""
    return coordination_hub
