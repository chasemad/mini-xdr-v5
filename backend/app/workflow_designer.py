"""
Visual Workflow Designer Engine for Mini-XDR
Backend engine for visual workflow creation, validation, and template management.
"""

import logging
import asyncio
import json
import uuid
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from sqlalchemy import select, and_, func, desc, or_
from sqlalchemy.ext.asyncio import AsyncSession

from .models import (
    ResponseWorkflow, AdvancedResponseAction, ResponsePlaybook,
    Incident, ResponseApproval
)
from .config import settings
from .advanced_response_engine import get_response_engine

logger = logging.getLogger(__name__)


class NodeType(str, Enum):
    START = "start"
    END = "end"
    ACTION = "action"
    CONDITION = "condition"
    WAIT = "wait"
    APPROVAL = "approval"
    PARALLEL = "parallel"
    MERGE = "merge"


class ValidationSeverity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class WorkflowNode:
    """Visual workflow node representation"""
    id: str
    type: NodeType
    position: Dict[str, float]
    data: Dict[str, Any]
    connections: List[str] = None
    
    def __post_init__(self):
        if self.connections is None:
            self.connections = []


@dataclass
class WorkflowEdge:
    """Visual workflow edge representation"""
    id: str
    source: str
    target: str
    type: str = "default"
    data: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.data is None:
            self.data = {}


@dataclass
class ValidationResult:
    """Workflow validation result"""
    valid: bool
    errors: List[Dict[str, Any]]
    warnings: List[Dict[str, Any]]
    suggestions: List[Dict[str, Any]]
    execution_plan: Optional[List[Dict[str, Any]]] = None


class VisualWorkflowDesigner:
    """
    Visual workflow designer engine for creating, validating, and managing
    complex response workflows with conditional logic and branching.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Validation rules
        self.validation_rules = self._initialize_validation_rules()
        
        # Node compatibility matrix
        self.compatibility_matrix = self._initialize_compatibility_matrix()
        
        # Template patterns
        self.template_patterns = self._initialize_template_patterns()
        
    def _initialize_validation_rules(self) -> Dict[str, Any]:
        """Initialize workflow validation rules"""
        return {
            "required_nodes": {
                "start": {"min": 1, "max": 1},
                "end": {"min": 1, "max": 1},
                "action": {"min": 1, "max": 50}
            },
            "node_constraints": {
                "condition": {
                    "requires_branches": True,
                    "min_outputs": 2,
                    "max_outputs": 5
                },
                "parallel": {
                    "min_parallel_actions": 2,
                    "max_parallel_actions": 10
                },
                "approval": {
                    "requires_human_input": True,
                    "timeout_required": True
                }
            },
            "workflow_constraints": {
                "max_total_nodes": 100,
                "max_workflow_depth": 20,
                "max_execution_time": 7200,  # 2 hours
                "max_parallel_branches": 5
            },
            "safety_rules": {
                "high_risk_actions_require_approval": True,
                "max_consecutive_failures": 3,
                "rollback_plan_required": True
            }
        }
    
    def _initialize_compatibility_matrix(self) -> Dict[str, Any]:
        """Initialize node type compatibility matrix"""
        return {
            "start": {
                "can_connect_to": ["action", "condition", "wait", "approval"],
                "cannot_connect_to": ["start", "end"]
            },
            "action": {
                "can_connect_to": ["action", "condition", "wait", "approval", "end", "parallel", "merge"],
                "cannot_connect_to": ["start"]
            },
            "condition": {
                "can_connect_to": ["action", "condition", "wait", "approval", "end"],
                "cannot_connect_to": ["start"],
                "requires_multiple_outputs": True
            },
            "wait": {
                "can_connect_to": ["action", "condition", "approval", "end"],
                "cannot_connect_to": ["start", "wait"]
            },
            "approval": {
                "can_connect_to": ["action", "condition", "end"],
                "cannot_connect_to": ["start", "approval"]
            },
            "parallel": {
                "can_connect_to": ["action", "condition"],
                "cannot_connect_to": ["start", "end", "parallel"],
                "requires_merge": True
            },
            "merge": {
                "can_connect_to": ["action", "condition", "end"],
                "cannot_connect_to": ["start", "parallel"],
                "requires_multiple_inputs": True
            }
        }
    
    def _initialize_template_patterns(self) -> Dict[str, Any]:
        """Initialize common workflow template patterns"""
        return {
            "linear_response": {
                "description": "Sequential action execution",
                "pattern": ["start", "action", "action", "action", "end"],
                "use_cases": ["simple_containment", "basic_investigation"]
            },
            "conditional_response": {
                "description": "Response with decision points",
                "pattern": ["start", "action", "condition", ["action_a", "action_b"], "end"],
                "use_cases": ["adaptive_response", "threat_specific_actions"]
            },
            "approval_workflow": {
                "description": "High-risk actions requiring approval",
                "pattern": ["start", "action", "approval", "action", "end"],
                "use_cases": ["destructive_actions", "compliance_required"]
            },
            "parallel_execution": {
                "description": "Concurrent action execution",
                "pattern": ["start", "parallel", ["action_1", "action_2", "action_3"], "merge", "end"],
                "use_cases": ["forensics_collection", "multi_vector_response"]
            },
            "investigation_workflow": {
                "description": "Evidence collection and analysis",
                "pattern": ["start", "action", "wait", "condition", "action", "end"],
                "use_cases": ["forensic_investigation", "malware_analysis"]
            }
        }
    
    async def validate_visual_workflow(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        db_session: AsyncSession
    ) -> ValidationResult:
        """
        Comprehensive validation of visual workflow design
        """
        
        try:
            # Convert to internal representation
            workflow_nodes = [WorkflowNode(**node) for node in nodes]
            workflow_edges = [WorkflowEdge(**edge) for edge in edges]
            
            # Validation tasks
            validation_tasks = [
                self._validate_basic_structure(workflow_nodes, workflow_edges),
                self._validate_node_compatibility(workflow_nodes, workflow_edges),
                self._validate_execution_logic(workflow_nodes, workflow_edges),
                self._validate_safety_requirements(workflow_nodes, workflow_edges),
                self._validate_performance_constraints(workflow_nodes, workflow_edges)
            ]
            
            # Run validations in parallel
            validation_results = await asyncio.gather(*validation_tasks, return_exceptions=True)
            
            # Combine validation results
            all_errors = []
            all_warnings = []
            all_suggestions = []
            
            for result in validation_results:
                if isinstance(result, Exception):
                    all_errors.append({
                        "severity": ValidationSeverity.ERROR,
                        "message": f"Validation error: {str(result)}",
                        "node_id": None
                    })
                elif isinstance(result, dict):
                    all_errors.extend(result.get("errors", []))
                    all_warnings.extend(result.get("warnings", []))
                    all_suggestions.extend(result.get("suggestions", []))
            
            # Generate execution plan if valid
            execution_plan = None
            if len(all_errors) == 0:
                execution_plan = await self._generate_execution_plan(workflow_nodes, workflow_edges)
            
            # Validate against backend action registry
            backend_validation = await self._validate_against_backend(workflow_nodes, db_session)
            all_warnings.extend(backend_validation.get("warnings", []))
            
            is_valid = len(all_errors) == 0
            
            return ValidationResult(
                valid=is_valid,
                errors=all_errors,
                warnings=all_warnings,
                suggestions=all_suggestions,
                execution_plan=execution_plan
            )
            
        except Exception as e:
            self.logger.error(f"Failed to validate visual workflow: {e}")
            return ValidationResult(
                valid=False,
                errors=[{
                    "severity": ValidationSeverity.ERROR,
                    "message": f"Validation system error: {str(e)}",
                    "node_id": None
                }],
                warnings=[],
                suggestions=[]
            )
    
    async def create_workflow_from_visual(
        self,
        incident_id: int,
        playbook_name: str,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        metadata: Dict[str, Any],
        db_session: AsyncSession
    ) -> Dict[str, Any]:
        """
        Create executable workflow from visual design
        """
        
        try:
            # Validate visual workflow
            validation = await self.validate_visual_workflow(nodes, edges, db_session)
            
            if not validation.valid:
                return {
                    "success": False,
                    "error": "Workflow validation failed",
                    "validation_errors": validation.errors
                }
            
            # Convert visual design to executable steps
            executable_steps = await self._convert_to_executable_steps(
                nodes, edges, validation.execution_plan
            )
            
            # Create workflow using advanced response engine
            response_engine = await get_response_engine()
            
            workflow_result = await response_engine.create_workflow(
                incident_id=incident_id,
                playbook_name=playbook_name,
                steps=executable_steps,
                auto_execute=metadata.get("auto_execute", False),
                priority=metadata.get("priority", "medium"),
                db_session=db_session
            )
            
            if workflow_result.get("success"):
                # Store visual design metadata
                await self._store_visual_metadata(
                    workflow_result["workflow_id"],
                    nodes,
                    edges,
                    metadata,
                    db_session
                )
                
                return {
                    "success": True,
                    "workflow_id": workflow_result["workflow_id"],
                    "workflow_db_id": workflow_result["workflow_db_id"],
                    "validation_result": validation,
                    "executable_steps": len(executable_steps),
                    "estimated_duration": sum(step.get("timeout_seconds", 300) for step in executable_steps),
                    "visual_metadata_stored": True
                }
            else:
                return workflow_result
                
        except Exception as e:
            self.logger.error(f"Failed to create workflow from visual design: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def create_template_from_workflow(
        self,
        workflow_id: str,
        template_name: str,
        template_description: str,
        category: str,
        db_session: AsyncSession
    ) -> Dict[str, Any]:
        """
        Create reusable template from successful workflow
        """
        
        try:
            # Get workflow details
            workflow_result = await db_session.execute(
                select(ResponseWorkflow).where(ResponseWorkflow.workflow_id == workflow_id)
            )
            workflow = workflow_result.scalars().first()
            
            if not workflow:
                return {"success": False, "error": "Workflow not found"}
            
            # Only create templates from successful workflows
            if not workflow.success_rate or workflow.success_rate < 0.8:
                return {
                    "success": False,
                    "error": "Only successful workflows (>80% success rate) can be converted to templates"
                }
            
            # Create template
            template = ResponsePlaybook(
                name=template_name,
                description=template_description,
                category=category,
                steps=workflow.steps,
                estimated_duration_minutes=workflow.execution_time_ms // 60000 if workflow.execution_time_ms else 30,
                status="active",
                times_used=0,
                success_rate=workflow.success_rate,
                created_by="workflow_designer",
                tags=["visual_workflow", "auto_generated"],
                compliance_frameworks=["SOC2", "ISO27001"]  # Default frameworks
            )
            
            db_session.add(template)
            await db_session.commit()
            await db_session.refresh(template)
            
            return {
                "success": True,
                "template_id": template.id,
                "template_name": template_name,
                "based_on_workflow": workflow_id,
                "success_rate": workflow.success_rate,
                "estimated_duration": template.estimated_duration_minutes
            }
            
        except Exception as e:
            self.logger.error(f"Failed to create template from workflow: {e}")
            return {"success": False, "error": str(e)}
    
    async def get_workflow_templates(
        self,
        category: Optional[str] = None,
        db_session: AsyncSession = None
    ) -> Dict[str, Any]:
        """
        Get available workflow templates with usage statistics
        """
        
        try:
            # Build query
            query = select(ResponsePlaybook).where(ResponsePlaybook.status == "active")
            
            if category and category != "all":
                query = query.where(ResponsePlaybook.category == category)
            
            query = query.order_by(ResponsePlaybook.success_rate.desc(), ResponsePlaybook.times_used.desc())
            
            # Execute query
            result = await db_session.execute(query)
            templates = result.scalars().all()
            
            # Format response
            template_data = []
            for template in templates:
                template_data.append({
                    "id": template.id,
                    "name": template.name,
                    "description": template.description,
                    "category": template.category,
                    "steps": template.steps,
                    "estimated_duration_minutes": template.estimated_duration_minutes,
                    "times_used": template.times_used,
                    "success_rate": template.success_rate,
                    "created_at": template.created_at.isoformat() if template.created_at else None,
                    "tags": template.tags,
                    "compliance_frameworks": template.compliance_frameworks,
                    "difficulty": self._assess_template_difficulty(template.steps),
                    "threat_types": self._extract_threat_types(template.steps)
                })
            
            return {
                "success": True,
                "templates": template_data,
                "total_count": len(template_data),
                "categories": list(set(t["category"] for t in template_data if t["category"]))
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get workflow templates: {e}")
            return {"success": False, "error": str(e)}
    
    async def _validate_basic_structure(
        self, 
        nodes: List[WorkflowNode], 
        edges: List[WorkflowEdge]
    ) -> Dict[str, Any]:
        """Validate basic workflow structure"""
        
        errors = []
        warnings = []
        suggestions = []
        
        # Check required nodes
        start_nodes = [n for n in nodes if n.type == NodeType.START]
        end_nodes = [n for n in nodes if n.type == NodeType.END]
        action_nodes = [n for n in nodes if n.type == NodeType.ACTION]
        
        if len(start_nodes) == 0:
            errors.append({
                "severity": ValidationSeverity.ERROR,
                "message": "Workflow must have exactly one start node",
                "node_id": None
            })
        elif len(start_nodes) > 1:
            errors.append({
                "severity": ValidationSeverity.ERROR,
                "message": "Workflow cannot have multiple start nodes",
                "node_id": None
            })
        
        if len(end_nodes) == 0:
            errors.append({
                "severity": ValidationSeverity.ERROR,
                "message": "Workflow must have at least one end node",
                "node_id": None
            })
        
        if len(action_nodes) == 0:
            errors.append({
                "severity": ValidationSeverity.ERROR,
                "message": "Workflow must contain at least one action",
                "node_id": None
            })
        
        # Check for orphaned nodes
        connected_nodes = set()
        for edge in edges:
            connected_nodes.add(edge.source)
            connected_nodes.add(edge.target)
        
        orphaned_nodes = [n for n in nodes if n.id not in connected_nodes and n.type not in [NodeType.START, NodeType.END]]
        if orphaned_nodes:
            warnings.append({
                "severity": ValidationSeverity.WARNING,
                "message": f"{len(orphaned_nodes)} nodes are not connected to the workflow",
                "node_id": [n.id for n in orphaned_nodes]
            })
        
        # Check workflow complexity
        if len(nodes) > self.validation_rules["workflow_constraints"]["max_total_nodes"]:
            errors.append({
                "severity": ValidationSeverity.ERROR,
                "message": f"Workflow exceeds maximum node count ({self.validation_rules['workflow_constraints']['max_total_nodes']})",
                "node_id": None
            })
        
        if len(action_nodes) > 20:
            warnings.append({
                "severity": ValidationSeverity.WARNING,
                "message": "Large workflows (>20 actions) may be difficult to manage",
                "node_id": None
            })
        
        # Suggest optimizations
        if len(action_nodes) > 10:
            suggestions.append({
                "severity": ValidationSeverity.INFO,
                "message": "Consider breaking this into smaller, focused workflows",
                "node_id": None
            })
        
        return {
            "errors": errors,
            "warnings": warnings,
            "suggestions": suggestions
        }
    
    async def _validate_node_compatibility(
        self, 
        nodes: List[WorkflowNode], 
        edges: List[WorkflowEdge]
    ) -> Dict[str, Any]:
        """Validate node type compatibility and connections"""
        
        errors = []
        warnings = []
        suggestions = []
        
        # Build adjacency lists
        outgoing = {node.id: [] for node in nodes}
        incoming = {node.id: [] for node in nodes}
        
        for edge in edges:
            outgoing[edge.source].append(edge.target)
            incoming[edge.target].append(edge.source)
        
        # Check each node's connections
        for node in nodes:
            node_rules = self.compatibility_matrix.get(node.type.value, {})
            
            # Check outgoing connections
            outgoing_nodes = [next(n for n in nodes if n.id == target_id) for target_id in outgoing[node.id]]
            for target_node in outgoing_nodes:
                allowed_types = node_rules.get("can_connect_to", [])
                forbidden_types = node_rules.get("cannot_connect_to", [])
                
                if allowed_types and target_node.type.value not in allowed_types:
                    errors.append({
                        "severity": ValidationSeverity.ERROR,
                        "message": f"{node.type.value} cannot connect to {target_node.type.value}",
                        "node_id": node.id
                    })
                
                if target_node.type.value in forbidden_types:
                    errors.append({
                        "severity": ValidationSeverity.ERROR,
                        "message": f"{node.type.value} cannot connect to {target_node.type.value}",
                        "node_id": node.id
                    })
            
            # Check node-specific requirements
            if node.type == NodeType.CONDITION:
                if len(outgoing[node.id]) < 2:
                    errors.append({
                        "severity": ValidationSeverity.ERROR,
                        "message": "Condition nodes must have at least 2 outgoing connections",
                        "node_id": node.id
                    })
            
            if node.type == NodeType.PARALLEL:
                if len(outgoing[node.id]) < 2:
                    errors.append({
                        "severity": ValidationSeverity.ERROR,
                        "message": "Parallel nodes must have at least 2 outgoing connections",
                        "node_id": node.id
                    })
            
            if node.type == NodeType.MERGE:
                if len(incoming[node.id]) < 2:
                    errors.append({
                        "severity": ValidationSeverity.ERROR,
                        "message": "Merge nodes must have at least 2 incoming connections",
                        "node_id": node.id
                    })
        
        return {
            "errors": errors,
            "warnings": warnings,
            "suggestions": suggestions
        }
    
    async def _validate_execution_logic(
        self, 
        nodes: List[WorkflowNode], 
        edges: List[WorkflowEdge]
    ) -> Dict[str, Any]:
        """Validate workflow execution logic"""
        
        errors = []
        warnings = []
        suggestions = []
        
        # Check for cycles
        if self._has_cycles(nodes, edges):
            errors.append({
                "severity": ValidationSeverity.ERROR,
                "message": "Workflow contains circular dependencies",
                "node_id": None
            })
        
        # Check reachability
        start_nodes = [n for n in nodes if n.type == NodeType.START]
        end_nodes = [n for n in nodes if n.type == NodeType.END]
        
        if start_nodes and end_nodes:
            for end_node in end_nodes:
                if not self._is_reachable(start_nodes[0], end_node, nodes, edges):
                    errors.append({
                        "severity": ValidationSeverity.ERROR,
                        "message": f"End node {end_node.id} is not reachable from start",
                        "node_id": end_node.id
                    })
        
        # Check for dead ends (nodes with no path to end)
        action_nodes = [n for n in nodes if n.type == NodeType.ACTION]
        for action_node in action_nodes:
            if end_nodes and not any(self._is_reachable(action_node, end_node, nodes, edges) for end_node in end_nodes):
                warnings.append({
                    "severity": ValidationSeverity.WARNING,
                    "message": f"Action {action_node.id} has no path to workflow completion",
                    "node_id": action_node.id
                })
        
        # Check for approval requirements
        high_risk_actions = [n for n in nodes if n.type == NodeType.ACTION and n.data.get("safety_level") == "high"]
        for action_node in high_risk_actions:
            # Check if there's an approval node before this action
            has_approval = self._has_approval_before_node(action_node, nodes, edges)
            if not has_approval:
                warnings.append({
                    "severity": ValidationSeverity.WARNING,
                    "message": f"High-risk action {action_node.id} should have approval gate",
                    "node_id": action_node.id
                })
        
        return {
            "errors": errors,
            "warnings": warnings,
            "suggestions": suggestions
        }
    
    async def _validate_safety_requirements(
        self, 
        nodes: List[WorkflowNode], 
        edges: List[WorkflowEdge]
    ) -> Dict[str, Any]:
        """Validate safety requirements and risk mitigation"""
        
        errors = []
        warnings = []
        suggestions = []
        
        # Check for rollback planning
        destructive_actions = [
            n for n in nodes 
            if n.type == NodeType.ACTION and not n.data.get("rollback_supported", True)
        ]
        
        if destructive_actions:
            warnings.append({
                "severity": ValidationSeverity.WARNING,
                "message": f"{len(destructive_actions)} actions don't support rollback",
                "node_id": [n.id for n in destructive_actions]
            })
        
        # Check for safety intervals
        consecutive_high_risk = self._find_consecutive_high_risk_actions(nodes, edges)
        if consecutive_high_risk:
            suggestions.append({
                "severity": ValidationSeverity.INFO,
                "message": "Consider adding wait nodes between high-risk actions",
                "node_id": consecutive_high_risk
            })
        
        # Check for error handling
        action_nodes = [n for n in nodes if n.type == NodeType.ACTION]
        nodes_without_error_handling = [
            n for n in action_nodes 
            if not self._has_error_handling(n, nodes, edges)
        ]
        
        if len(nodes_without_error_handling) > len(action_nodes) * 0.5:
            suggestions.append({
                "severity": ValidationSeverity.INFO,
                "message": "Consider adding conditional error handling for failed actions",
                "node_id": None
            })
        
        return {
            "errors": errors,
            "warnings": warnings,
            "suggestions": suggestions
        }
    
    async def _validate_performance_constraints(
        self, 
        nodes: List[WorkflowNode], 
        edges: List[WorkflowEdge]
    ) -> Dict[str, Any]:
        """Validate performance and resource constraints"""
        
        errors = []
        warnings = []
        suggestions = []
        
        # Calculate estimated execution time
        action_nodes = [n for n in nodes if n.type == NodeType.ACTION]
        total_duration = sum(n.data.get("estimated_duration", 300) for n in action_nodes)
        
        if total_duration > self.validation_rules["workflow_constraints"]["max_execution_time"]:
            errors.append({
                "severity": ValidationSeverity.ERROR,
                "message": f"Workflow duration ({total_duration}s) exceeds maximum allowed time",
                "node_id": None
            })
        
        # Check for resource conflicts
        high_resource_actions = [
            n for n in action_nodes 
            if n.data.get("action_type") in ["memory_dump_collection", "disk_imaging", "vulnerability_patching"]
        ]
        
        if len(high_resource_actions) > 3:
            warnings.append({
                "severity": ValidationSeverity.WARNING,
                "message": "Multiple high-resource actions may cause performance issues",
                "node_id": [n.id for n in high_resource_actions]
            })
        
        # Suggest parallelization opportunities
        parallelizable_actions = self._identify_parallelizable_actions(action_nodes, edges)
        if parallelizable_actions:
            suggestions.append({
                "severity": ValidationSeverity.INFO,
                "message": f"Actions {', '.join(parallelizable_actions)} could be executed in parallel",
                "node_id": parallelizable_actions
            })
        
        return {
            "errors": errors,
            "warnings": warnings,
            "suggestions": suggestions
        }
    
    async def _validate_against_backend(
        self, 
        nodes: List[WorkflowNode], 
        db_session: AsyncSession
    ) -> Dict[str, Any]:
        """Validate against backend action registry"""
        
        warnings = []
        
        try:
            response_engine = await get_response_engine()
            available_actions = response_engine.get_available_actions()
            
            action_nodes = [n for n in nodes if n.type == NodeType.ACTION]
            
            for node in action_nodes:
                action_type = node.data.get("actionType") or node.data.get("action_type")
                if action_type and action_type not in available_actions.get("actions", {}):
                    warnings.append({
                        "severity": ValidationSeverity.WARNING,
                        "message": f"Action type '{action_type}' not found in registry",
                        "node_id": node.id
                    })
        
        except Exception as e:
            warnings.append({
                "severity": ValidationSeverity.WARNING,
                "message": f"Backend validation unavailable: {str(e)}",
                "node_id": None
            })
        
        return {"warnings": warnings}
    
    async def _generate_execution_plan(
        self, 
        nodes: List[WorkflowNode], 
        edges: List[WorkflowEdge]
    ) -> List[Dict[str, Any]]:
        """Generate optimized execution plan from visual workflow"""
        
        # Topological sort to determine execution order
        execution_order = self._topological_sort(nodes, edges)
        
        execution_plan = []
        
        for node_id in execution_order:
            node = next(n for n in nodes if n.id == node_id)
            
            if node.type == NodeType.ACTION:
                step = {
                    "step_id": len(execution_plan) + 1,
                    "node_id": node.id,
                    "action_type": node.data.get("actionType") or node.data.get("action_type"),
                    "parameters": node.data.get("parameters", {}),
                    "timeout_seconds": node.data.get("estimated_duration", 300),
                    "continue_on_failure": False,
                    "max_retries": 3,
                    "node_type": "action",
                    "position": node.position
                }
                execution_plan.append(step)
                
            elif node.type == NodeType.CONDITION:
                step = {
                    "step_id": len(execution_plan) + 1,
                    "node_id": node.id,
                    "condition": node.data.get("condition", "success"),
                    "branches": self._get_condition_branches(node, nodes, edges),
                    "node_type": "condition",
                    "position": node.position
                }
                execution_plan.append(step)
                
            elif node.type == NodeType.WAIT:
                step = {
                    "step_id": len(execution_plan) + 1,
                    "node_id": node.id,
                    "wait_duration": node.data.get("duration", "30s"),
                    "node_type": "wait",
                    "position": node.position
                }
                execution_plan.append(step)
                
            elif node.type == NodeType.APPROVAL:
                step = {
                    "step_id": len(execution_plan) + 1,
                    "node_id": node.id,
                    "approval_type": node.data.get("approval_type", "manual"),
                    "approver": node.data.get("approver", "manager"),
                    "timeout_minutes": node.data.get("timeout_minutes", 60),
                    "node_type": "approval",
                    "position": node.position
                }
                execution_plan.append(step)
        
        return execution_plan
    
    async def _convert_to_executable_steps(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        execution_plan: Optional[List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """Convert visual workflow to executable steps"""
        
        executable_steps = []
        
        # Use execution plan if available, otherwise simple conversion
        if execution_plan:
            for step in execution_plan:
                if step.get("node_type") == "action":
                    executable_steps.append({
                        "action_type": step["action_type"],
                        "parameters": step["parameters"],
                        "timeout_seconds": step["timeout_seconds"],
                        "continue_on_failure": step["continue_on_failure"],
                        "max_retries": step["max_retries"]
                    })
        else:
            # Simple conversion for action nodes only
            action_nodes = [n for n in nodes if n.get("type") == "actionNode"]
            for node in action_nodes:
                executable_steps.append({
                    "action_type": node["data"].get("actionType") or node["data"].get("action_type"),
                    "parameters": node["data"].get("parameters", {}),
                    "timeout_seconds": node["data"].get("estimated_duration", 300),
                    "continue_on_failure": False,
                    "max_retries": 3
                })
        
        return executable_steps
    
    async def _store_visual_metadata(
        self,
        workflow_id: str,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        metadata: Dict[str, Any],
        db_session: AsyncSession
    ):
        """Store visual workflow metadata for future editing"""
        
        try:
            # Update workflow with visual metadata
            workflow_result = await db_session.execute(
                select(ResponseWorkflow).where(ResponseWorkflow.workflow_id == workflow_id)
            )
            workflow = workflow_result.scalars().first()
            
            if workflow:
                # Store visual data in workflow's execution_log
                visual_metadata = {
                    "visual_design": {
                        "nodes": nodes,
                        "edges": edges,
                        "metadata": metadata,
                        "created_with": "visual_designer",
                        "designer_version": "2.0"
                    }
                }
                
                if workflow.execution_log:
                    workflow.execution_log.update(visual_metadata)
                else:
                    workflow.execution_log = visual_metadata
                
                await db_session.commit()
                
        except Exception as e:
            self.logger.error(f"Failed to store visual metadata: {e}")
    
    # Helper methods
    def _has_cycles(self, nodes: List[WorkflowNode], edges: List[WorkflowEdge]) -> bool:
        """Check for cycles in workflow graph"""
        visited = set()
        rec_stack = set()
        
        # Build adjacency list
        adj_list = {node.id: [] for node in nodes}
        for edge in edges:
            adj_list[edge.source].append(edge.target)
        
        def dfs(node_id: str) -> bool:
            if node_id in rec_stack:
                return True
            if node_id in visited:
                return False
            
            visited.add(node_id)
            rec_stack.add(node_id)
            
            for neighbor in adj_list.get(node_id, []):
                if dfs(neighbor):
                    return True
            
            rec_stack.remove(node_id)
            return False
        
        for node in nodes:
            if node.id not in visited:
                if dfs(node.id):
                    return True
        
        return False
    
    def _is_reachable(
        self, 
        start_node: WorkflowNode, 
        target_node: WorkflowNode, 
        nodes: List[WorkflowNode], 
        edges: List[WorkflowEdge]
    ) -> bool:
        """Check if target node is reachable from start node"""
        
        # Build adjacency list
        adj_list = {node.id: [] for node in nodes}
        for edge in edges:
            adj_list[edge.source].append(edge.target)
        
        # BFS to check reachability
        queue = [start_node.id]
        visited = set([start_node.id])
        
        while queue:
            current = queue.pop(0)
            
            if current == target_node.id:
                return True
            
            for neighbor in adj_list.get(current, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        return False
    
    def _topological_sort(self, nodes: List[WorkflowNode], edges: List[WorkflowEdge]) -> List[str]:
        """Perform topological sort for execution order"""
        
        # Build adjacency list and in-degree count
        adj_list = {node.id: [] for node in nodes}
        in_degree = {node.id: 0 for node in nodes}
        
        for edge in edges:
            adj_list[edge.source].append(edge.target)
            in_degree[edge.target] += 1
        
        # Find nodes with no incoming edges
        queue = [node_id for node_id, degree in in_degree.items() if degree == 0]
        result = []
        
        while queue:
            current = queue.pop(0)
            result.append(current)
            
            # Reduce in-degree for neighbors
            for neighbor in adj_list[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        return result
    
    def _has_approval_before_node(
        self, 
        node: WorkflowNode, 
        nodes: List[WorkflowNode], 
        edges: List[WorkflowEdge]
    ) -> bool:
        """Check if there's an approval node before given node"""
        
        # Simple check for approval in path (could be more sophisticated)
        approval_nodes = [n.id for n in nodes if n.type == NodeType.APPROVAL]
        
        # Check if any approval node can reach this node
        for approval_id in approval_nodes:
            approval_node = next(n for n in nodes if n.id == approval_id)
            if self._is_reachable(approval_node, node, nodes, edges):
                return True
        
        return False
    
    def _find_consecutive_high_risk_actions(
        self, 
        nodes: List[WorkflowNode], 
        edges: List[WorkflowEdge]
    ) -> List[str]:
        """Find consecutive high-risk actions that might need separation"""
        
        high_risk_nodes = [
            n.id for n in nodes 
            if n.type == NodeType.ACTION and n.data.get("safety_level") == "high"
        ]
        
        consecutive = []
        
        # Build adjacency list
        adj_list = {node.id: [] for node in nodes}
        for edge in edges:
            adj_list[edge.source].append(edge.target)
        
        # Check for consecutive high-risk actions
        for node_id in high_risk_nodes:
            for neighbor_id in adj_list.get(node_id, []):
                if neighbor_id in high_risk_nodes:
                    consecutive.extend([node_id, neighbor_id])
        
        return list(set(consecutive))
    
    def _has_error_handling(
        self, 
        node: WorkflowNode, 
        nodes: List[WorkflowNode], 
        edges: List[WorkflowEdge]
    ) -> bool:
        """Check if node has error handling logic"""
        
        # Look for condition nodes connected to this action
        adj_list = {node.id: [] for node in nodes}
        for edge in edges:
            adj_list[edge.source].append(edge.target)
        
        connected_nodes = adj_list.get(node.id, [])
        condition_nodes = [
            n for n in nodes 
            if n.id in connected_nodes and n.type == NodeType.CONDITION
        ]
        
        return len(condition_nodes) > 0
    
    def _identify_parallelizable_actions(
        self, 
        action_nodes: List[WorkflowNode], 
        edges: List[WorkflowEdge]
    ) -> List[str]:
        """Identify actions that could be executed in parallel"""
        
        # Simple heuristic: actions with same dependencies and no conflicts
        parallelizable = []
        
        # Actions that are typically safe to parallelize
        safe_parallel_actions = [
            "memory_dump_collection", "log_preservation", "data_classification",
            "threat_intelligence_enrichment", "network_packet_capture"
        ]
        
        parallel_candidates = [
            n.id for n in action_nodes 
            if n.data.get("action_type") in safe_parallel_actions
        ]
        
        if len(parallel_candidates) >= 2:
            parallelizable = parallel_candidates[:3]  # Limit to 3 for simplicity
        
        return parallelizable
    
    def _get_condition_branches(
        self, 
        condition_node: WorkflowNode, 
        nodes: List[WorkflowNode], 
        edges: List[WorkflowEdge]
    ) -> Dict[str, List[str]]:
        """Get the branches for a condition node"""
        
        # Build adjacency list
        adj_list = {node.id: [] for node in nodes}
        for edge in edges:
            adj_list[edge.source].append(edge.target)
        
        branches = {
            "success": [],
            "failure": []
        }
        
        # Get connected nodes
        connected = adj_list.get(condition_node.id, [])
        
        # Simple assignment (in production, would use edge labels)
        if len(connected) >= 2:
            branches["success"] = connected[:len(connected)//2]
            branches["failure"] = connected[len(connected)//2:]
        
        return branches
    
    def _assess_template_difficulty(self, steps: List[Dict[str, Any]]) -> str:
        """Assess difficulty level of template"""
        
        if not steps:
            return "beginner"
        
        complexity_score = 0
        
        # Add complexity for number of steps
        complexity_score += min(len(steps) / 10, 1.0) * 30
        
        # Add complexity for high-risk actions
        high_risk_actions = [
            s for s in steps 
            if s.get("action_type") in ["deploy_firewall_rules", "system_hardening", "vulnerability_patching"]
        ]
        complexity_score += len(high_risk_actions) * 20
        
        # Add complexity for conditional logic
        # (simplified - would check for actual conditional steps)
        if len(steps) > 5:
            complexity_score += 20
        
        if complexity_score >= 70:
            return "advanced"
        elif complexity_score >= 40:
            return "intermediate"
        else:
            return "beginner"
    
    def _extract_threat_types(self, steps: List[Dict[str, Any]]) -> List[str]:
        """Extract likely threat types this template addresses"""
        
        threat_mappings = {
            "isolate_host_advanced": ["malware", "insider_threat"],
            "memory_dump_collection": ["malware", "apt"],
            "block_ip_advanced": ["brute_force", "ddos"],
            "email_recall": ["phishing"],
            "deploy_firewall_rules": ["ddos", "network_attack"],
            "account_disable": ["insider_threat", "credential_compromise"]
        }
        
        threat_types = set()
        for step in steps:
            action_type = step.get("action_type")
            if action_type in threat_mappings:
                threat_types.update(threat_mappings[action_type])
        
        return list(threat_types)


# Global instance
visual_workflow_designer = VisualWorkflowDesigner()


async def get_workflow_designer() -> VisualWorkflowDesigner:
    """Get the global workflow designer instance"""
    return visual_workflow_designer












