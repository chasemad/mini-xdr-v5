"""
Policy Engine for AI-Driven Containment Decisions
Manages YAML-based containment policies with dynamic evaluation
"""
import yaml
import json
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from .models import ContainmentPolicy, Incident
from .config import settings

logger = logging.getLogger(__name__)


@dataclass
class PolicyMatch:
    """Result of policy evaluation"""
    policy_id: str
    policy_name: str
    matched: bool
    confidence: float
    actions: Dict[str, Any]
    conditions_met: List[str]
    conditions_failed: List[str]
    override_allowed: bool


class PolicyEngine:
    """Engine for evaluating and managing containment policies"""
    
    def __init__(self, policy_dir: str = "policies"):
        self.policy_dir = Path(policy_dir)
        self.policy_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)
        self.cached_policies = {}
        self.cache_timestamp = None
        
        # Ensure default policies exist
        self._ensure_default_policies()
    
    def _ensure_default_policies(self):
        """Create default policy files if they don't exist"""
        default_policy_file = self.policy_dir / "default_policies.yaml"
        
        if not default_policy_file.exists():
            default_policies = {
                "policies": [
                    {
                        "name": "high_risk_ssh_brute_force",
                        "description": "High-risk SSH brute force attacks requiring immediate containment",
                        "priority": 10,
                        "status": "active",
                        "conditions": {
                            "risk_score": {"min": 0.8},
                            "threat_category": ["brute_force", "password_spray"],
                            "escalation_level": ["high", "critical"],
                            "event_count": {"min": 50}
                        },
                        "actions": {
                            "block_ip": {"duration": 3600, "immediate": True},
                            "isolate_host": {"level": "hard"},
                            "notify_analyst": {
                                "message": "Critical SSH attack detected - immediate response required",
                                "urgency": "high"
                            },
                            "escalate": True
                        },
                        "agent_override": True,
                        "escalation_threshold": 0.9,
                        "cooldown_period": 1800
                    },
                    {
                        "name": "medium_risk_reconnaissance", 
                        "description": "Medium-risk reconnaissance activities",
                        "priority": 50,
                        "status": "active",
                        "conditions": {
                            "risk_score": {"min": 0.4, "max": 0.7},
                            "threat_category": ["reconnaissance", "port_scan"],
                            "event_count": {"min": 10, "max": 50}
                        },
                        "actions": {
                            "block_ip": {"duration": 900, "immediate": False},
                            "notify_analyst": {
                                "message": "Reconnaissance activity detected",
                                "urgency": "medium"
                            },
                            "rate_limit": {"max_connections": 5, "window": 300}
                        },
                        "agent_override": True,
                        "escalation_threshold": 0.7,
                        "cooldown_period": 600
                    },
                    {
                        "name": "low_risk_monitoring",
                        "description": "Low-risk activities requiring monitoring only",
                        "priority": 100,
                        "status": "active",
                        "conditions": {
                            "risk_score": {"max": 0.3},
                            "threat_category": ["unknown", "benign"]
                        },
                        "actions": {
                            "monitor_only": True,
                            "log_enhanced": True,
                            "notify_analyst": {
                                "message": "Low-risk activity monitored",
                                "urgency": "low"
                            }
                        },
                        "agent_override": False,
                        "escalation_threshold": 0.5,
                        "cooldown_period": 300
                    },
                    {
                        "name": "threat_intel_malicious",
                        "description": "Known malicious IPs from threat intelligence",
                        "priority": 5,
                        "status": "active",
                        "conditions": {
                            "threat_intel": {
                                "is_malicious": True,
                                "confidence": {"min": 0.7}
                            }
                        },
                        "actions": {
                            "block_ip": {"duration": 7200, "immediate": True},
                            "blacklist": {"permanent": True},
                            "notify_analyst": {
                                "message": "Known malicious IP detected",
                                "urgency": "high"
                            }
                        },
                        "agent_override": True,
                        "escalation_threshold": 0.95,
                        "cooldown_period": 3600
                    },
                    {
                        "name": "ml_high_anomaly",
                        "description": "High ML anomaly scores indicating unusual behavior",
                        "priority": 20,
                        "status": "active",
                        "conditions": {
                            "ml_score": {"min": 0.9},
                            "risk_score": {"min": 0.6}
                        },
                        "actions": {
                            "block_ip": {"duration": 1800, "immediate": False},
                            "enhance_monitoring": True,
                            "collect_additional_data": True,
                            "notify_analyst": {
                                "message": "High ML anomaly detected - requires investigation",
                                "urgency": "medium"
                            }
                        },
                        "agent_override": True,
                        "escalation_threshold": 0.8,
                        "cooldown_period": 900
                    }
                ]
            }
            
            with open(default_policy_file, 'w') as f:
                yaml.dump(default_policies, f, default_flow_style=False, indent=2)
            
            self.logger.info(f"Created default policies file: {default_policy_file}")
    
    async def evaluate_policies(
        self, 
        incident: Incident,
        context: Dict[str, Any],
        db: Optional[AsyncSession] = None
    ) -> List[PolicyMatch]:
        """
        Evaluate all policies against an incident and context
        
        Args:
            incident: The incident to evaluate
            context: Additional context (ML scores, threat intel, etc.)
            db: Database session for policy queries
            
        Returns:
            List of policy matches ordered by priority
        """
        policies = await self._load_policies(db)
        matches = []
        
        for policy in policies:
            if policy.get('status') != 'active':
                continue
                
            match = await self._evaluate_single_policy(policy, incident, context)
            if match.matched:
                matches.append(match)
        
        # Sort by priority (lower number = higher priority)
        matches.sort(key=lambda m: self._get_policy_priority(m.policy_name, policies))
        
        return matches
    
    async def _evaluate_single_policy(
        self, 
        policy: Dict[str, Any], 
        incident: Incident, 
        context: Dict[str, Any]
    ) -> PolicyMatch:
        """Evaluate a single policy against incident and context"""
        
        policy_name = policy['name']
        conditions = policy.get('conditions', {})
        actions = policy.get('actions', {})
        
        conditions_met = []
        conditions_failed = []
        
        # Evaluate each condition
        for condition_name, condition_value in conditions.items():
            if await self._evaluate_condition(condition_name, condition_value, incident, context):
                conditions_met.append(condition_name)
            else:
                conditions_failed.append(condition_name)
        
        # Policy matches if all conditions are met
        matched = len(conditions_failed) == 0 and len(conditions_met) > 0
        
        # Calculate confidence based on how many conditions matched
        total_conditions = len(conditions)
        confidence = len(conditions_met) / max(total_conditions, 1) if total_conditions > 0 else 0.0
        
        return PolicyMatch(
            policy_id=policy.get('id', policy_name),
            policy_name=policy_name,
            matched=matched,
            confidence=confidence,
            actions=actions,
            conditions_met=conditions_met,
            conditions_failed=conditions_failed,
            override_allowed=policy.get('agent_override', True)
        )
    
    async def _evaluate_condition(
        self, 
        condition_name: str, 
        condition_value: Any, 
        incident: Incident, 
        context: Dict[str, Any]
    ) -> bool:
        """Evaluate a single condition"""
        
        try:
            if condition_name == 'risk_score':
                actual_value = incident.risk_score or context.get('risk_score', 0.0)
                return self._check_numeric_condition(actual_value, condition_value)
            
            elif condition_name == 'threat_category':
                actual_value = incident.threat_category
                return self._check_list_condition(actual_value, condition_value)
            
            elif condition_name == 'escalation_level':
                actual_value = incident.escalation_level
                return self._check_list_condition(actual_value, condition_value)
            
            elif condition_name == 'event_count':
                actual_value = context.get('event_count', 0)
                return self._check_numeric_condition(actual_value, condition_value)
            
            elif condition_name == 'ml_score':
                actual_value = context.get('ml_score', 0.0)
                return self._check_numeric_condition(actual_value, condition_value)
            
            elif condition_name == 'threat_intel':
                return self._check_threat_intel_condition(condition_value, context)
            
            elif condition_name == 'time_of_day':
                return self._check_time_condition(condition_value)
            
            elif condition_name == 'source_type':
                actual_value = context.get('source_type', 'unknown')
                return self._check_list_condition(actual_value, condition_value)
            
            else:
                self.logger.warning(f"Unknown condition: {condition_name}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error evaluating condition {condition_name}: {e}")
            return False
    
    def _check_numeric_condition(self, actual_value: float, condition: Any) -> bool:
        """Check numeric conditions (min, max, exact)"""
        if isinstance(condition, (int, float)):
            return actual_value >= condition
        
        elif isinstance(condition, dict):
            if 'min' in condition and actual_value < condition['min']:
                return False
            if 'max' in condition and actual_value > condition['max']:
                return False
            if 'exact' in condition and actual_value != condition['exact']:
                return False
            return True
        
        return False
    
    def _check_list_condition(self, actual_value: str, condition: Any) -> bool:
        """Check list/string conditions"""
        if isinstance(condition, str):
            return actual_value == condition
        elif isinstance(condition, list):
            return actual_value in condition
        return False
    
    def _check_threat_intel_condition(self, condition: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """Check threat intelligence conditions"""
        intel_data = context.get('threat_intel', {})
        
        for key, expected in condition.items():
            if key == 'is_malicious':
                if intel_data.get('is_malicious', False) != expected:
                    return False
            elif key == 'confidence':
                intel_confidence = intel_data.get('confidence', 0.0)
                if not self._check_numeric_condition(intel_confidence, expected):
                    return False
            elif key == 'risk_score':
                intel_risk = intel_data.get('risk_score', 0.0)
                if not self._check_numeric_condition(intel_risk, expected):
                    return False
            elif key == 'category':
                intel_category = intel_data.get('category', 'unknown')
                if not self._check_list_condition(intel_category, expected):
                    return False
        
        return True
    
    def _check_time_condition(self, condition: Dict[str, Any]) -> bool:
        """Check time-based conditions"""
        now = datetime.now(timezone.utc)
        
        if 'hour_range' in condition:
            hour_range = condition['hour_range']
            start_hour, end_hour = hour_range
            current_hour = now.hour
            
            if start_hour <= end_hour:
                return start_hour <= current_hour <= end_hour
            else:  # Overnight range (e.g., 22-6)
                return current_hour >= start_hour or current_hour <= end_hour
        
        if 'weekday' in condition:
            # 0 = Monday, 6 = Sunday
            allowed_weekdays = condition['weekday']
            if isinstance(allowed_weekdays, int):
                allowed_weekdays = [allowed_weekdays]
            return now.weekday() in allowed_weekdays
        
        return True
    
    async def _load_policies(self, db: Optional[AsyncSession] = None) -> List[Dict[str, Any]]:
        """Load policies from database and files"""
        policies = []
        
        # Load from database if available
        if db:
            db_policies = await self._load_policies_from_db(db)
            policies.extend(db_policies)
        
        # Load from YAML files
        file_policies = self._load_policies_from_files()
        policies.extend(file_policies)
        
        return policies
    
    async def _load_policies_from_db(self, db: AsyncSession) -> List[Dict[str, Any]]:
        """Load policies from database"""
        try:
            query = select(ContainmentPolicy).where(
                ContainmentPolicy.status == "active"
            ).order_by(ContainmentPolicy.priority)
            
            result = await db.execute(query)
            db_policies = result.scalars().all()
            
            policies = []
            for policy in db_policies:
                policy_dict = {
                    'id': policy.id,
                    'name': policy.name,
                    'description': policy.description,
                    'priority': policy.priority,
                    'status': policy.status,
                    'conditions': policy.conditions,
                    'actions': policy.actions,
                    'agent_override': policy.agent_override,
                    'escalation_threshold': policy.escalation_threshold
                }
                policies.append(policy_dict)
            
            return policies
            
        except Exception as e:
            self.logger.error(f"Failed to load policies from database: {e}")
            return []
    
    def _load_policies_from_files(self) -> List[Dict[str, Any]]:
        """Load policies from YAML files"""
        policies = []
        
        for policy_file in self.policy_dir.glob("*.yaml"):
            try:
                with open(policy_file, 'r') as f:
                    policy_data = yaml.safe_load(f)
                
                if 'policies' in policy_data:
                    policies.extend(policy_data['policies'])
                else:
                    # Single policy file
                    policies.append(policy_data)
                    
                self.logger.debug(f"Loaded policies from {policy_file}")
                
            except Exception as e:
                self.logger.error(f"Failed to load policy file {policy_file}: {e}")
        
        return policies
    
    def _get_policy_priority(self, policy_name: str, policies: List[Dict[str, Any]]) -> int:
        """Get priority for a policy (lower = higher priority)"""
        for policy in policies:
            if policy['name'] == policy_name:
                return policy.get('priority', 100)
        return 100
    
    async def create_policy_from_template(
        self, 
        template_name: str, 
        parameters: Dict[str, Any],
        db: AsyncSession
    ) -> ContainmentPolicy:
        """Create a new policy from a template"""
        
        templates = {
            'ip_blocklist': {
                'name': f"blocklist_{parameters.get('name', 'custom')}",
                'description': f"Block specific IPs: {parameters.get('description', '')}",
                'conditions': {
                    'src_ip': parameters.get('ip_list', [])
                },
                'actions': {
                    'block_ip': {'duration': parameters.get('duration', 3600), 'immediate': True}
                }
            },
            'high_volume_attack': {
                'name': f"high_volume_{parameters.get('threshold', 100)}",
                'description': f"High volume attack detection (>{parameters.get('threshold', 100)} events)",
                'conditions': {
                    'event_count': {'min': parameters.get('threshold', 100)},
                    'risk_score': {'min': 0.5}
                },
                'actions': {
                    'block_ip': {'duration': parameters.get('duration', 1800)},
                    'notify_analyst': {'urgency': 'high'}
                }
            }
        }
        
        if template_name not in templates:
            raise ValueError(f"Unknown template: {template_name}")
        
        template = templates[template_name]
        
        # Create database policy
        policy = ContainmentPolicy(
            name=template['name'],
            description=template['description'],
            priority=parameters.get('priority', 50),
            conditions=template['conditions'],
            actions=template['actions'],
            agent_override=parameters.get('agent_override', True),
            escalation_threshold=parameters.get('escalation_threshold', 0.8)
        )
        
        db.add(policy)
        await db.commit()
        
        self.logger.info(f"Created policy from template {template_name}: {policy.name}")
        return policy
    
    def export_policies_to_yaml(self, output_file: str, policies: List[Dict[str, Any]]):
        """Export policies to YAML file"""
        policy_data = {'policies': policies}
        
        with open(output_file, 'w') as f:
            yaml.dump(policy_data, f, default_flow_style=False, indent=2)
        
        self.logger.info(f"Exported {len(policies)} policies to {output_file}")


# Global policy engine instance
policy_engine = PolicyEngine()
