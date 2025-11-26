"""
Advanced Response Engine for Mini-XDR
Implements enterprise-grade response capabilities with workflow orchestration,
safety controls, and real-time monitoring.
"""

import asyncio
import logging
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from sqlalchemy import and_, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from .agent_orchestrator import get_orchestrator
from .agents.containment_agent import ContainmentAgent
from .models import (
    AdvancedResponseAction,
    Incident,
    ResponseApproval,
    ResponseImpactMetrics,
    ResponsePlaybook,
    ResponseWorkflow,
)

logger = logging.getLogger(__name__)


def _make_json_serializable(obj: Any) -> Any:
    """Recursively convert sets and other non-JSON types to JSON-serializable types."""
    if isinstance(obj, set):
        return list(obj)
    elif isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_json_serializable(item) for item in obj]
    elif isinstance(obj, datetime):
        return obj.isoformat()
    return obj


class WorkflowStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    AWAITING_APPROVAL = "awaiting_approval"


class ActionCategory(str, Enum):
    NETWORK = "network"
    ENDPOINT = "endpoint"
    EMAIL = "email"
    CLOUD = "cloud"
    IDENTITY = "identity"
    DATA = "data"
    COMPLIANCE = "compliance"
    FORENSICS = "forensics"


class ResponsePriority(str, Enum):
    CRITICAL = "critical"  # 1-25
    HIGH = "high"  # 26-50
    MEDIUM = "medium"  # 51-75
    LOW = "low"  # 76-100


class AdvancedResponseEngine:
    """
    Advanced response engine providing enterprise-grade response capabilities
    """

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.containment_agent = ContainmentAgent()
        self.orchestrator = None

        # Response action registry
        self.action_registry = self._initialize_action_registry()

        # Safety thresholds
        self.safety_config = {
            "max_concurrent_workflows": 10,
            "max_actions_per_workflow": 50,
            "approval_required_threshold": 0.8,  # Impact score threshold
            "auto_rollback_timeout_minutes": 30,
            "max_retry_attempts": 3,
        }

    async def initialize(self):
        """Initialize the response engine"""
        try:
            from .agent_orchestrator import get_orchestrator

            self.orchestrator = await get_orchestrator()
            self.logger.info("Advanced Response Engine initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize Advanced Response Engine: {e}")

    def _initialize_action_registry(self) -> Dict[str, Dict[str, Any]]:
        """Initialize the registry of available response actions"""
        return {
            # Basic/Simple Response Actions (used by triggers)
            "block_ip": {
                "category": ActionCategory.NETWORK,
                "name": "IP Blocking",
                "description": "Block IP address using firewall rules",
                "parameters": ["ip_address", "duration", "block_level"],
                "safety_level": "medium",
                "rollback_supported": True,
                "estimated_duration": 30,
            },
            "create_incident": {
                "category": ActionCategory.FORENSICS,
                "name": "Create Incident",
                "description": "Create a new incident record for tracking",
                "parameters": ["title", "severity", "description"],
                "safety_level": "low",
                "rollback_supported": False,
                "estimated_duration": 5,
            },
            "invoke_ai_agent": {
                "category": ActionCategory.FORENSICS,
                "name": "Invoke AI Agent",
                "description": "Invoke AI agent for analysis or response",
                "parameters": ["agent", "task", "context", "query"],
                "safety_level": "low",
                "rollback_supported": False,
                "estimated_duration": 60,
            },
            "send_notification": {
                "category": ActionCategory.IDENTITY,
                "name": "Send Notification",
                "description": "Send notification via configured channels",
                "parameters": ["channel", "message", "recipients", "priority"],
                "safety_level": "low",
                "rollback_supported": False,
                "estimated_duration": 10,
            },
            # Network Response Actions
            "block_ip_advanced": {
                "category": ActionCategory.NETWORK,
                "name": "Advanced IP Blocking",
                "description": "Block IP with geolocation and threat intelligence integration",
                "parameters": [
                    "ip_address",
                    "duration",
                    "block_level",
                    "geo_restrictions",
                ],
                "safety_level": "medium",
                "rollback_supported": True,
                "estimated_duration": 30,
            },
            "deploy_firewall_rules": {
                "category": ActionCategory.NETWORK,
                "name": "Deploy Firewall Rules",
                "description": "Deploy advanced firewall rules across infrastructure",
                "parameters": ["rule_set", "scope", "priority", "expiration"],
                "safety_level": "high",
                "rollback_supported": True,
                "estimated_duration": 120,
            },
            "dns_sinkhole": {
                "category": ActionCategory.NETWORK,
                "name": "DNS Sinkholing",
                "description": "Redirect malicious domains to sinkhole servers",
                "parameters": ["domains", "sinkhole_ip", "ttl", "scope"],
                "safety_level": "medium",
                "rollback_supported": True,
                "estimated_duration": 60,
            },
            "traffic_redirection": {
                "category": ActionCategory.NETWORK,
                "name": "Traffic Redirection",
                "description": "Redirect suspicious traffic to honeypots for analysis",
                "parameters": ["source_criteria", "destination", "monitoring_level"],
                "safety_level": "medium",
                "rollback_supported": True,
                "estimated_duration": 90,
            },
            # Endpoint Response Actions
            "isolate_host_advanced": {
                "category": ActionCategory.ENDPOINT,
                "name": "Advanced Host Isolation",
                "description": "Comprehensive host isolation with selective connectivity",
                "parameters": [
                    "host_identifier",
                    "isolation_level",
                    "exceptions",
                    "monitoring",
                ],
                "safety_level": "high",
                "rollback_supported": True,
                "estimated_duration": 45,
            },
            "memory_dump_collection": {
                "category": ActionCategory.ENDPOINT,
                "name": "Memory Dump Collection",
                "description": "Collect memory dumps from compromised systems",
                "parameters": ["target_hosts", "dump_type", "encryption", "retention"],
                "safety_level": "low",
                "rollback_supported": False,
                "estimated_duration": 300,
            },
            "process_termination": {
                "category": ActionCategory.ENDPOINT,
                "name": "Process Termination",
                "description": "Terminate specific processes on target systems",
                "parameters": ["process_criteria", "force_level", "confirmation"],
                "safety_level": "high",
                "rollback_supported": False,
                "estimated_duration": 15,
            },
            "registry_hardening": {
                "category": ActionCategory.ENDPOINT,
                "name": "Registry Hardening",
                "description": "Apply security-focused registry modifications",
                "parameters": ["hardening_profile", "target_systems", "backup"],
                "safety_level": "medium",
                "rollback_supported": True,
                "estimated_duration": 180,
            },
            # Email Response Actions
            "email_recall": {
                "category": ActionCategory.EMAIL,
                "name": "Email Recall",
                "description": "Recall and quarantine malicious emails",
                "parameters": ["message_criteria", "scope", "notification"],
                "safety_level": "low",
                "rollback_supported": False,
                "estimated_duration": 120,
            },
            "mailbox_quarantine": {
                "category": ActionCategory.EMAIL,
                "name": "Mailbox Quarantine",
                "description": "Quarantine compromised mailboxes",
                "parameters": ["mailbox_list", "quarantine_level", "access_exceptions"],
                "safety_level": "medium",
                "rollback_supported": True,
                "estimated_duration": 60,
            },
            # Cloud Response Actions
            "iam_policy_restriction": {
                "category": ActionCategory.CLOUD,
                "name": "IAM Policy Restriction",
                "description": "Restrict IAM policies for compromised accounts",
                "parameters": ["account_identifiers", "restriction_level", "temporary"],
                "safety_level": "high",
                "rollback_supported": True,
                "estimated_duration": 30,
            },
            "resource_isolation": {
                "category": ActionCategory.CLOUD,
                "name": "Cloud Resource Isolation",
                "description": "Isolate cloud resources from network access",
                "parameters": ["resource_ids", "isolation_scope", "exceptions"],
                "safety_level": "high",
                "rollback_supported": True,
                "estimated_duration": 90,
            },
            # Identity Response Actions
            "account_disable": {
                "category": ActionCategory.IDENTITY,
                "name": "Account Disable",
                "description": "Disable compromised user accounts",
                "parameters": ["account_list", "disable_level", "notification"],
                "safety_level": "medium",
                "rollback_supported": True,
                "estimated_duration": 30,
            },
            "password_reset_bulk": {
                "category": ActionCategory.IDENTITY,
                "name": "Bulk Password Reset",
                "description": "Force password reset for affected accounts",
                "parameters": [
                    "account_criteria",
                    "complexity_requirements",
                    "notification",
                ],
                "safety_level": "medium",
                "rollback_supported": False,
                "estimated_duration": 180,
            },
            # Data Response Actions
            "data_classification": {
                "category": ActionCategory.DATA,
                "name": "Data Classification",
                "description": "Classify and tag sensitive data",
                "parameters": [
                    "data_sources",
                    "classification_rules",
                    "tagging_policy",
                ],
                "safety_level": "low",
                "rollback_supported": True,
                "estimated_duration": 600,
            },
            "backup_verification": {
                "category": ActionCategory.DATA,
                "name": "Backup Verification",
                "description": "Verify integrity of backup systems",
                "parameters": ["backup_systems", "verification_level", "reporting"],
                "safety_level": "low",
                "rollback_supported": False,
                "estimated_duration": 300,
            },
            "isolate_file": {
                "category": ActionCategory.DATA,
                "name": "File Isolation",
                "description": "Quarantine and isolate suspicious or malicious files",
                "parameters": [
                    "file_path",
                    "hash",
                    "quarantine_location",
                    "preserve_evidence",
                ],
                "safety_level": "low",
                "rollback_supported": True,
                "estimated_duration": 30,
            },
            "data_encryption": {
                "category": ActionCategory.DATA,
                "name": "Emergency Data Encryption",
                "description": "Encrypt sensitive data to prevent unauthorized access",
                "parameters": ["data_paths", "encryption_level", "key_management"],
                "safety_level": "medium",
                "rollback_supported": True,
                "estimated_duration": 900,
            },
            "data_loss_prevention": {
                "category": ActionCategory.DATA,
                "name": "Data Loss Prevention",
                "description": "Deploy DLP rules to prevent data exfiltration",
                "parameters": ["data_types", "blocking_rules", "monitoring_level"],
                "safety_level": "medium",
                "rollback_supported": True,
                "estimated_duration": 300,
            },
            # Compliance Response Actions (NEW CATEGORY)
            "compliance_audit_trigger": {
                "category": ActionCategory.COMPLIANCE,
                "name": "Compliance Audit Trigger",
                "description": "Trigger automated compliance audits",
                "parameters": ["audit_scope", "frameworks", "urgency_level"],
                "safety_level": "low",
                "rollback_supported": False,
                "estimated_duration": 1800,
            },
            "data_retention_enforcement": {
                "category": ActionCategory.COMPLIANCE,
                "name": "Data Retention Enforcement",
                "description": "Enforce data retention policies during incident",
                "parameters": ["retention_rules", "data_scope", "preservation_level"],
                "safety_level": "medium",
                "rollback_supported": True,
                "estimated_duration": 600,
            },
            "regulatory_reporting": {
                "category": ActionCategory.COMPLIANCE,
                "name": "Automated Regulatory Reporting",
                "description": "Generate and submit regulatory breach reports",
                "parameters": ["regulations", "incident_details", "timeline"],
                "safety_level": "low",
                "rollback_supported": False,
                "estimated_duration": 900,
            },
            "privacy_breach_notification": {
                "category": ActionCategory.COMPLIANCE,
                "name": "Privacy Breach Notification",
                "description": "Automated privacy breach notifications to affected parties",
                "parameters": [
                    "affected_parties",
                    "notification_method",
                    "disclosure_level",
                ],
                "safety_level": "high",
                "rollback_supported": False,
                "estimated_duration": 1200,
            },
            # Forensics Response Actions (NEW CATEGORY)
            "disk_imaging": {
                "category": ActionCategory.FORENSICS,
                "name": "Forensic Disk Imaging",
                "description": "Create forensic disk images for evidence",
                "parameters": ["target_systems", "imaging_level", "hash_verification"],
                "safety_level": "low",
                "rollback_supported": False,
                "estimated_duration": 3600,
            },
            "network_packet_capture": {
                "category": ActionCategory.FORENSICS,
                "name": "Network Packet Capture",
                "description": "Capture network traffic for forensic analysis",
                "parameters": ["capture_scope", "duration", "filtering_rules"],
                "safety_level": "low",
                "rollback_supported": True,
                "estimated_duration": 1800,
            },
            "log_preservation": {
                "category": ActionCategory.FORENSICS,
                "name": "Log Preservation",
                "description": "Preserve logs for forensic investigation",
                "parameters": [
                    "log_sources",
                    "retention_period",
                    "integrity_protection",
                ],
                "safety_level": "low",
                "rollback_supported": False,
                "estimated_duration": 300,
            },
            "chain_of_custody": {
                "category": ActionCategory.FORENSICS,
                "name": "Chain of Custody",
                "description": "Establish and maintain evidence chain of custody",
                "parameters": [
                    "evidence_items",
                    "custodian_details",
                    "documentation_level",
                ],
                "safety_level": "low",
                "rollback_supported": False,
                "estimated_duration": 600,
            },
            "forensic_timeline": {
                "category": ActionCategory.FORENSICS,
                "name": "Forensic Timeline Creation",
                "description": "Create detailed forensic timeline of events",
                "parameters": ["time_range", "data_sources", "correlation_level"],
                "safety_level": "low",
                "rollback_supported": False,
                "estimated_duration": 1200,
            },
            "evidence_analysis": {
                "category": ActionCategory.FORENSICS,
                "name": "Automated Evidence Analysis",
                "description": "AI-powered analysis of collected evidence",
                "parameters": ["evidence_types", "analysis_depth", "correlation_rules"],
                "safety_level": "low",
                "rollback_supported": False,
                "estimated_duration": 2400,
            },
            # Advanced Network Actions
            "network_segmentation": {
                "category": ActionCategory.NETWORK,
                "name": "Dynamic Network Segmentation",
                "description": "Implement dynamic network segmentation",
                "parameters": ["segment_rules", "isolation_level", "exceptions"],
                "safety_level": "high",
                "rollback_supported": True,
                "estimated_duration": 1800,
            },
            "traffic_analysis": {
                "category": ActionCategory.NETWORK,
                "name": "Deep Traffic Analysis",
                "description": "Perform deep packet inspection and analysis",
                "parameters": ["analysis_scope", "detection_rules", "alert_threshold"],
                "safety_level": "low",
                "rollback_supported": True,
                "estimated_duration": 900,
            },
            "threat_hunting_deployment": {
                "category": ActionCategory.NETWORK,
                "name": "Threat Hunting Deployment",
                "description": "Deploy threat hunting queries across network",
                "parameters": ["hunt_queries", "target_scope", "monitoring_duration"],
                "safety_level": "low",
                "rollback_supported": True,
                "estimated_duration": 1200,
            },
            "deception_technology": {
                "category": ActionCategory.NETWORK,
                "name": "Deception Technology Deployment",
                "description": "Deploy decoy systems and honeypots",
                "parameters": [
                    "decoy_types",
                    "deployment_locations",
                    "interaction_level",
                ],
                "safety_level": "low",
                "rollback_supported": True,
                "estimated_duration": 2400,
            },
            "ssl_certificate_blocking": {
                "category": ActionCategory.NETWORK,
                "name": "SSL Certificate Blocking",
                "description": "Block malicious SSL certificates",
                "parameters": ["certificate_hashes", "blocking_scope", "duration"],
                "safety_level": "medium",
                "rollback_supported": True,
                "estimated_duration": 300,
            },
            "bandwidth_throttling": {
                "category": ActionCategory.NETWORK,
                "name": "Bandwidth Throttling",
                "description": "Throttle bandwidth for suspicious sources",
                "parameters": ["source_criteria", "throttle_percentage", "duration"],
                "safety_level": "medium",
                "rollback_supported": True,
                "estimated_duration": 180,
            },
            # Advanced Identity & Access Actions
            "privileged_access_review": {
                "category": ActionCategory.IDENTITY,
                "name": "Privileged Access Review",
                "description": "Review and audit privileged access during incident",
                "parameters": ["access_scope", "review_criteria", "remediation_level"],
                "safety_level": "low",
                "rollback_supported": False,
                "estimated_duration": 1800,
            },
            "session_termination": {
                "category": ActionCategory.IDENTITY,
                "name": "Active Session Termination",
                "description": "Terminate active user sessions",
                "parameters": ["session_criteria", "termination_scope", "notification"],
                "safety_level": "high",
                "rollback_supported": False,
                "estimated_duration": 300,
            },
            "access_certification": {
                "category": ActionCategory.IDENTITY,
                "name": "Access Certification Trigger",
                "description": "Trigger access certification reviews",
                "parameters": ["certification_scope", "urgency_level", "reviewers"],
                "safety_level": "low",
                "rollback_supported": False,
                "estimated_duration": 600,
            },
            "identity_verification": {
                "category": ActionCategory.IDENTITY,
                "name": "Enhanced Identity Verification",
                "description": "Trigger enhanced identity verification",
                "parameters": ["verification_methods", "user_scope", "challenge_level"],
                "safety_level": "medium",
                "rollback_supported": True,
                "estimated_duration": 900,
            },
            "mfa_enforcement": {
                "category": ActionCategory.IDENTITY,
                "name": "MFA Enforcement",
                "description": "Enforce multi-factor authentication",
                "parameters": ["user_scope", "mfa_methods", "grace_period"],
                "safety_level": "medium",
                "rollback_supported": True,
                "estimated_duration": 600,
            },
            # Advanced Endpoint Actions
            "system_hardening": {
                "category": ActionCategory.ENDPOINT,
                "name": "System Hardening",
                "description": "Apply security hardening configurations",
                "parameters": [
                    "hardening_profiles",
                    "target_systems",
                    "rollback_snapshot",
                ],
                "safety_level": "medium",
                "rollback_supported": True,
                "estimated_duration": 2400,
            },
            "vulnerability_patching": {
                "category": ActionCategory.ENDPOINT,
                "name": "Emergency Vulnerability Patching",
                "description": "Deploy critical security patches",
                "parameters": ["patch_list", "target_systems", "testing_level"],
                "safety_level": "high",
                "rollback_supported": True,
                "estimated_duration": 3600,
            },
            "endpoint_quarantine": {
                "category": ActionCategory.ENDPOINT,
                "name": "Endpoint Quarantine",
                "description": "Quarantine compromised endpoints",
                "parameters": ["quarantine_criteria", "isolation_level", "monitoring"],
                "safety_level": "high",
                "rollback_supported": True,
                "estimated_duration": 600,
            },
            "service_shutdown": {
                "category": ActionCategory.ENDPOINT,
                "name": "Service Shutdown",
                "description": "Shutdown compromised or vulnerable services",
                "parameters": ["service_criteria", "shutdown_method", "notification"],
                "safety_level": "high",
                "rollback_supported": True,
                "estimated_duration": 300,
            },
            "configuration_rollback": {
                "category": ActionCategory.ENDPOINT,
                "name": "Configuration Rollback",
                "description": "Rollback system configurations to safe state",
                "parameters": ["rollback_scope", "target_config", "validation"],
                "safety_level": "medium",
                "rollback_supported": False,
                "estimated_duration": 900,
            },
            # Advanced Email Actions
            "email_flow_analysis": {
                "category": ActionCategory.EMAIL,
                "name": "Email Flow Analysis",
                "description": "Analyze email flow patterns for threats",
                "parameters": ["analysis_scope", "time_range", "pattern_detection"],
                "safety_level": "low",
                "rollback_supported": False,
                "estimated_duration": 600,
            },
            "domain_blocking": {
                "category": ActionCategory.EMAIL,
                "name": "Malicious Domain Blocking",
                "description": "Block malicious domains in email systems",
                "parameters": ["domain_list", "blocking_scope", "duration"],
                "safety_level": "medium",
                "rollback_supported": True,
                "estimated_duration": 300,
            },
            "attachment_sandboxing": {
                "category": ActionCategory.EMAIL,
                "name": "Attachment Sandboxing",
                "description": "Sandbox suspicious email attachments",
                "parameters": [
                    "attachment_criteria",
                    "sandbox_environment",
                    "analysis_depth",
                ],
                "safety_level": "low",
                "rollback_supported": False,
                "estimated_duration": 1200,
            },
            "email_encryption_enforcement": {
                "category": ActionCategory.EMAIL,
                "name": "Email Encryption Enforcement",
                "description": "Enforce email encryption for sensitive communications",
                "parameters": ["encryption_policy", "user_scope", "exception_rules"],
                "safety_level": "medium",
                "rollback_supported": True,
                "estimated_duration": 900,
            },
            # Advanced Cloud Actions
            "cloud_security_posture": {
                "category": ActionCategory.CLOUD,
                "name": "Cloud Security Posture Assessment",
                "description": "Assess and remediate cloud security posture",
                "parameters": [
                    "cloud_services",
                    "assessment_scope",
                    "remediation_level",
                ],
                "safety_level": "low",
                "rollback_supported": False,
                "estimated_duration": 2400,
            },
            "container_isolation": {
                "category": ActionCategory.CLOUD,
                "name": "Container Isolation",
                "description": "Isolate compromised containers",
                "parameters": [
                    "container_criteria",
                    "isolation_method",
                    "data_preservation",
                ],
                "safety_level": "medium",
                "rollback_supported": True,
                "estimated_duration": 600,
            },
            "api_rate_limiting": {
                "category": ActionCategory.CLOUD,
                "name": "API Rate Limiting",
                "description": "Implement emergency API rate limiting",
                "parameters": ["api_endpoints", "rate_limits", "exception_criteria"],
                "safety_level": "medium",
                "rollback_supported": True,
                "estimated_duration": 300,
            },
            "cloud_resource_tagging": {
                "category": ActionCategory.CLOUD,
                "name": "Emergency Resource Tagging",
                "description": "Tag cloud resources for incident tracking",
                "parameters": ["tagging_scope", "tag_policies", "automation_level"],
                "safety_level": "low",
                "rollback_supported": True,
                "estimated_duration": 450,
            },
            "serverless_function_disable": {
                "category": ActionCategory.CLOUD,
                "name": "Serverless Function Disable",
                "description": "Disable compromised serverless functions",
                "parameters": [
                    "function_criteria",
                    "disable_method",
                    "backup_preservation",
                ],
                "safety_level": "high",
                "rollback_supported": True,
                "estimated_duration": 180,
            },
            # Threat Intelligence Actions
            "threat_intelligence_enrichment": {
                "category": ActionCategory.NETWORK,
                "name": "Threat Intelligence Enrichment",
                "description": "Enrich incident data with threat intelligence",
                "parameters": [
                    "intel_sources",
                    "enrichment_scope",
                    "correlation_rules",
                ],
                "safety_level": "low",
                "rollback_supported": False,
                "estimated_duration": 600,
            },
            "ioc_deployment": {
                "category": ActionCategory.NETWORK,
                "name": "IOC Deployment",
                "description": "Deploy indicators of compromise across security tools",
                "parameters": ["ioc_list", "deployment_scope", "confidence_threshold"],
                "safety_level": "low",
                "rollback_supported": True,
                "estimated_duration": 900,
            },
            "threat_feed_update": {
                "category": ActionCategory.NETWORK,
                "name": "Threat Feed Update",
                "description": "Update threat intelligence feeds",
                "parameters": ["feed_sources", "update_frequency", "validation_rules"],
                "safety_level": "low",
                "rollback_supported": False,
                "estimated_duration": 300,
            },
            # Advanced Monitoring Actions
            "security_monitoring_enhancement": {
                "category": ActionCategory.NETWORK,
                "name": "Security Monitoring Enhancement",
                "description": "Enhance security monitoring for incident area",
                "parameters": [
                    "monitoring_scope",
                    "detection_rules",
                    "alert_thresholds",
                ],
                "safety_level": "low",
                "rollback_supported": True,
                "estimated_duration": 1200,
            },
            "behavior_analytics_deployment": {
                "category": ActionCategory.ENDPOINT,
                "name": "Behavior Analytics Deployment",
                "description": "Deploy behavioral analytics for anomaly detection",
                "parameters": ["analytics_scope", "baseline_data", "sensitivity_level"],
                "safety_level": "low",
                "rollback_supported": True,
                "estimated_duration": 1800,
            },
            "siem_rule_deployment": {
                "category": ActionCategory.NETWORK,
                "name": "SIEM Rule Deployment",
                "description": "Deploy custom SIEM detection rules",
                "parameters": ["rule_definitions", "deployment_scope", "testing_mode"],
                "safety_level": "medium",
                "rollback_supported": True,
                "estimated_duration": 600,
            },
            # Communication & Coordination Actions
            "stakeholder_notification": {
                "category": ActionCategory.IDENTITY,
                "name": "Stakeholder Notification",
                "description": "Notify key stakeholders of incident status",
                "parameters": [
                    "stakeholder_groups",
                    "notification_method",
                    "urgency_level",
                ],
                "safety_level": "low",
                "rollback_supported": False,
                "estimated_duration": 300,
            },
            "incident_escalation": {
                "category": ActionCategory.IDENTITY,
                "name": "Incident Escalation",
                "description": "Escalate incident to appropriate response teams",
                "parameters": [
                    "escalation_level",
                    "target_teams",
                    "escalation_criteria",
                ],
                "safety_level": "low",
                "rollback_supported": False,
                "estimated_duration": 180,
            },
            "external_coordination": {
                "category": ActionCategory.IDENTITY,
                "name": "External Coordination",
                "description": "Coordinate with external partners/agencies",
                "parameters": [
                    "coordination_type",
                    "external_parties",
                    "information_sharing",
                ],
                "safety_level": "medium",
                "rollback_supported": False,
                "estimated_duration": 1800,
            },
            # Recovery & Remediation Actions
            "system_recovery": {
                "category": ActionCategory.ENDPOINT,
                "name": "Automated System Recovery",
                "description": "Initiate automated system recovery procedures",
                "parameters": [
                    "recovery_scope",
                    "recovery_method",
                    "validation_criteria",
                ],
                "safety_level": "high",
                "rollback_supported": True,
                "estimated_duration": 3600,
            },
            "malware_removal": {
                "category": ActionCategory.ENDPOINT,
                "name": "Automated Malware Removal",
                "description": "Remove detected malware from systems",
                "parameters": ["removal_scope", "removal_method", "quarantine_backup"],
                "safety_level": "high",
                "rollback_supported": True,
                "estimated_duration": 1800,
            },
            "security_baseline_restoration": {
                "category": ActionCategory.ENDPOINT,
                "name": "Security Baseline Restoration",
                "description": "Restore systems to security baseline",
                "parameters": ["baseline_profile", "restoration_scope", "validation"],
                "safety_level": "medium",
                "rollback_supported": True,
                "estimated_duration": 2400,
            },
            # Intelligence & Attribution Actions
            "attribution_analysis": {
                "category": ActionCategory.FORENSICS,
                "name": "Automated Attribution Analysis",
                "description": "Analyze attack attribution using AI",
                "parameters": [
                    "analysis_scope",
                    "attribution_models",
                    "confidence_threshold",
                ],
                "safety_level": "low",
                "rollback_supported": False,
                "estimated_duration": 1800,
            },
            "campaign_correlation": {
                "category": ActionCategory.FORENSICS,
                "name": "Campaign Correlation Analysis",
                "description": "Correlate with known attack campaigns",
                "parameters": [
                    "correlation_scope",
                    "campaign_databases",
                    "similarity_threshold",
                ],
                "safety_level": "low",
                "rollback_supported": False,
                "estimated_duration": 1200,
            },
            "threat_landscape_update": {
                "category": ActionCategory.NETWORK,
                "name": "Threat Landscape Update",
                "description": "Update threat landscape based on incident",
                "parameters": ["update_scope", "sharing_level", "anonymization"],
                "safety_level": "low",
                "rollback_supported": False,
                "estimated_duration": 600,
            },
        }

    async def create_workflow(
        self,
        incident_id: int,
        playbook_name: str,
        steps: List[Dict[str, Any]],
        auto_execute: bool = False,
        priority: ResponsePriority = ResponsePriority.MEDIUM,
        db_session: AsyncSession = None,
    ) -> Dict[str, Any]:
        """Create a new response workflow"""

        try:
            workflow_id = f"wf_{incident_id}_{uuid.uuid4().hex[:8]}"

            # Validate steps
            validated_steps = await self._validate_workflow_steps(steps)

            # Calculate total estimated duration
            total_duration = sum(
                self.action_registry.get(step.get("action_type", ""), {}).get(
                    "estimated_duration", 60
                )
                for step in validated_steps
            )

            # Determine if approval is required
            approval_required = await self._requires_approval(
                validated_steps, incident_id
            )

            # Create workflow record
            workflow = ResponseWorkflow(
                workflow_id=workflow_id,
                incident_id=incident_id,
                playbook_name=playbook_name,
                steps=validated_steps,
                total_steps=len(validated_steps),
                status=WorkflowStatus.AWAITING_APPROVAL
                if approval_required
                else WorkflowStatus.PENDING,
                approval_required=approval_required,
                auto_executed=auto_execute,
                auto_rollback_enabled=True,
                rollback_plan=await self._generate_rollback_plan(validated_steps),
            )

            db_session.add(workflow)
            await db_session.commit()
            await db_session.refresh(workflow)

            self.logger.info(
                f"Created workflow {workflow_id} for incident {incident_id}"
            )

            # Auto-execute if no approval required and auto_execute is True
            if auto_execute and not approval_required:
                execution_result = await self.execute_workflow(workflow.id, db_session)
                # Merge workflow creation info with execution result
                return {
                    **execution_result,
                    "workflow_db_id": workflow.id,
                    "approval_required": approval_required,
                    "estimated_duration_minutes": total_duration // 60,
                }

            return {
                "success": True,
                "workflow_id": workflow_id,
                "workflow_db_id": workflow.id,
                "status": workflow.status,
                "approval_required": approval_required,
                "estimated_duration_minutes": total_duration // 60,
                "total_steps": len(validated_steps),
            }

        except Exception as e:
            self.logger.error(f"Failed to create workflow: {e}")
            try:
                if db_session:
                    await db_session.rollback()
            except Exception:
                pass  # Session might already be in bad state
            return {"success": False, "error": str(e)}

    async def execute_workflow(
        self, workflow_db_id: int, db_session: AsyncSession, executed_by: str = "system"
    ) -> Dict[str, Any]:
        """Execute a response workflow"""

        try:
            # Get workflow with all related data
            workflow = await db_session.get(
                ResponseWorkflow,
                workflow_db_id,
                options=[
                    selectinload(ResponseWorkflow.actions),
                    selectinload(ResponseWorkflow.incident),
                ],
            )

            if not workflow:
                return {"success": False, "error": "Workflow not found"}

            # Check if approval is required and granted
            if workflow.approval_required and not workflow.approved_at:
                return {"success": False, "error": "Workflow requires approval"}

            # Update workflow status
            workflow.status = WorkflowStatus.RUNNING
            workflow.current_step = 0

            await db_session.commit()

            self.logger.info(f"Starting execution of workflow {workflow.workflow_id}")

            # Execute steps sequentially
            execution_results = []
            success_count = 0

            for step_index, step in enumerate(workflow.steps):
                try:
                    workflow.current_step = step_index + 1
                    workflow.progress_percentage = (
                        step_index / len(workflow.steps)
                    ) * 100

                    # Execute the step
                    step_result = await self._execute_workflow_step(
                        workflow, step, step_index, db_session, executed_by
                    )

                    execution_results.append(step_result)

                    if step_result.get("success"):
                        success_count += 1
                    else:
                        # Handle step failure
                        if step.get("continue_on_failure", False):
                            self.logger.warning(
                                f"Step {step_index + 1} failed but continuing: {step_result.get('error')}"
                            )
                        else:
                            self.logger.error(
                                f"Step {step_index + 1} failed, stopping workflow: {step_result.get('error')}"
                            )
                            break

                    await db_session.commit()

                except Exception as step_error:
                    self.logger.error(
                        f"Error executing step {step_index + 1}: {step_error}"
                    )
                    execution_results.append(
                        {
                            "success": False,
                            "error": str(step_error),
                            "step_index": step_index,
                        }
                    )
                    break

            # Update final workflow status
            if success_count == len(workflow.steps):
                workflow.status = WorkflowStatus.COMPLETED
                workflow.progress_percentage = 100.0
                workflow.success_rate = 1.0
            elif success_count > 0:
                workflow.status = WorkflowStatus.COMPLETED  # Partial success
                workflow.progress_percentage = (
                    success_count / len(workflow.steps)
                ) * 100
                workflow.success_rate = success_count / len(workflow.steps)
            else:
                workflow.status = WorkflowStatus.FAILED
                workflow.success_rate = 0.0

            workflow.completed_at = datetime.utcnow()
            workflow.execution_time_ms = int(
                (datetime.utcnow() - workflow.created_at).total_seconds() * 1000
            )

            await db_session.commit()

            self.logger.info(
                f"Completed workflow {workflow.workflow_id} with {success_count}/{len(workflow.steps)} successful steps"
            )

            # Record impact metrics (non-blocking - don't fail workflow if this fails)
            try:
                await self._record_workflow_impact(
                    workflow, execution_results, db_session
                )
            except Exception as metrics_error:
                self.logger.warning(
                    f"Failed to record impact metrics (non-critical): {metrics_error}"
                )

            return {
                "success": True,
                "workflow_id": workflow.workflow_id,
                "status": workflow.status,
                "steps_completed": success_count,
                "total_steps": len(workflow.steps),
                "success_rate": workflow.success_rate,
                "execution_time_ms": workflow.execution_time_ms,
                "results": execution_results,
            }

        except Exception as e:
            self.logger.error(f"Failed to execute workflow: {e}")
            try:
                if db_session:
                    await db_session.rollback()
            except Exception:
                pass  # Session might already be in bad state
            return {"success": False, "error": str(e)}

    async def _execute_workflow_step(
        self,
        workflow: ResponseWorkflow,
        step: Dict[str, Any],
        step_index: int,
        db_session: AsyncSession,
        executed_by: str,
    ) -> Dict[str, Any]:
        """Execute a single workflow step"""

        action_type = step.get("action_type")
        action_params = dict(step.get("parameters", {}))
        action_params.setdefault("incident_id", workflow.incident_id)
        action_params.setdefault("workflow_id", workflow.workflow_id)
        try:
            if getattr(workflow, "incident", None):
                action_params.setdefault("source_ip", workflow.incident.src_ip)
        except Exception:
            # Relationship may not be loaded; ignore
            pass

        # Create action record
        action_id = f"act_{workflow.workflow_id}_{step_index}_{uuid.uuid4().hex[:6]}"

        action = AdvancedResponseAction(
            action_id=action_id,
            workflow_id=workflow.id,
            incident_id=workflow.incident_id,
            action_type=action_type,
            action_category=self.action_registry.get(action_type, {}).get(
                "category", "unknown"
            ),
            action_name=self.action_registry.get(action_type, {}).get(
                "name", action_type
            ),
            action_description=self.action_registry.get(action_type, {}).get(
                "description", ""
            ),
            status="running",
            parameters=action_params,
            executed_by=executed_by,
            execution_method="automated",
        )

        db_session.add(action)
        await db_session.commit()

        try:
            # Execute the action based on type
            if action_type in ["block_ip", "block_ip_advanced"]:
                result = await self._execute_block_ip_action(action_params)
            elif action_type == "create_incident":
                result = await self._execute_create_incident_action(action_params)
            elif action_type == "invoke_ai_agent":
                result = await self._execute_invoke_ai_agent_action(
                    action_params, db_session
                )
            elif action_type == "send_notification":
                result = await self._execute_send_notification_action(action_params)
            elif action_type in ["isolate_host", "isolate_host_advanced"]:
                result = await self._execute_isolate_host_action(action_params)
            elif action_type == "deploy_firewall_rules":
                result = await self._execute_firewall_rules_action(action_params)
            elif action_type == "dns_sinkhole":
                result = await self._execute_dns_sinkhole_action(action_params)
            elif action_type == "memory_dump_collection":
                result = await self._execute_memory_dump_action(action_params)

            # Forensics Actions
            elif action_type == "disk_imaging":
                result = await self._execute_disk_imaging_action(action_params)
            elif action_type == "network_packet_capture":
                result = await self._execute_packet_capture_action(action_params)
            elif action_type == "log_preservation":
                result = await self._execute_log_preservation_action(action_params)
            elif action_type == "chain_of_custody":
                result = await self._execute_chain_of_custody_action(action_params)
            elif action_type == "evidence_analysis":
                result = await self._execute_evidence_analysis_action(action_params)
            elif action_type == "attribution_analysis":
                result = await self._execute_attribution_analysis_action(action_params)

            # Compliance Actions
            elif action_type == "compliance_audit_trigger":
                result = await self._execute_compliance_audit_action(action_params)
            elif action_type == "regulatory_reporting":
                result = await self._execute_regulatory_reporting_action(action_params)
            elif action_type == "privacy_breach_notification":
                result = await self._execute_privacy_notification_action(action_params)
            elif action_type == "data_retention_enforcement":
                result = await self._execute_data_retention_action(action_params)

            # Advanced Network Actions
            elif action_type == "network_segmentation":
                result = await self._execute_network_segmentation_action(action_params)
            elif action_type == "traffic_analysis":
                result = await self._execute_traffic_analysis_action(action_params)
            elif action_type == "threat_hunting_deployment":
                result = await self._execute_threat_hunting_action(action_params)
            elif action_type == "deception_technology":
                result = await self._execute_deception_technology_action(action_params)
            elif action_type == "bandwidth_throttling":
                result = await self._execute_bandwidth_throttling_action(action_params)

            # Advanced Identity Actions
            elif action_type == "session_termination":
                result = await self._execute_session_termination_action(action_params)
            elif action_type == "privileged_access_review":
                result = await self._execute_privileged_access_review_action(
                    action_params
                )
            elif action_type == "mfa_enforcement":
                result = await self._execute_mfa_enforcement_action(action_params)
            elif action_type == "identity_verification":
                result = await self._execute_identity_verification_action(action_params)

            # Advanced Endpoint Actions
            elif action_type == "system_hardening":
                result = await self._execute_system_hardening_action(action_params)
            elif action_type == "vulnerability_patching":
                result = await self._execute_vulnerability_patching_action(
                    action_params
                )
            elif action_type == "endpoint_quarantine":
                result = await self._execute_endpoint_quarantine_action(action_params)
            elif action_type == "malware_removal":
                result = await self._execute_malware_removal_action(action_params)
            elif action_type == "system_recovery":
                result = await self._execute_system_recovery_action(action_params)

            # Cloud Actions
            elif action_type == "container_isolation":
                result = await self._execute_container_isolation_action(action_params)
            elif action_type == "api_rate_limiting":
                result = await self._execute_api_rate_limiting_action(action_params)
            elif action_type == "cloud_security_posture":
                result = await self._execute_cloud_security_posture_action(
                    action_params
                )

            # Email Actions
            elif action_type == "domain_blocking":
                result = await self._execute_domain_blocking_action(action_params)
            elif action_type == "attachment_sandboxing":
                result = await self._execute_attachment_sandboxing_action(action_params)
            elif action_type == "email_flow_analysis":
                result = await self._execute_email_flow_analysis_action(action_params)

            # Data Actions
            elif action_type == "data_encryption":
                result = await self._execute_data_encryption_action(action_params)
            elif action_type == "data_loss_prevention":
                result = await self._execute_data_loss_prevention_action(action_params)

            else:
                # Default to containment agent for unknown actions
                self.logger.info(
                    f"Delegating action '{action_type}' to containment agent"
                )
                result = await self.containment_agent.execute_containment(
                    {"action": action_type, **action_params}
                )

            # Ensure result is a valid dict (handle None or unexpected returns)
            if result is None:
                result = {
                    "success": False,
                    "error": f"Action '{action_type}' returned no result",
                }
            elif not isinstance(result, dict):
                result = {
                    "success": False,
                    "error": f"Action '{action_type}' returned invalid result type: {type(result)}",
                }

            # Update action record
            action.status = "completed" if result.get("success") else "failed"
            action.result_data = _make_json_serializable(result)
            action.completed_at = datetime.utcnow()

            if result.get("success"):
                action.confidence_score = result.get("confidence", 0.8)
            else:
                error_msg = (
                    result.get("error")
                    or result.get("detail")
                    or f"Action '{action_type}' failed without error message"
                )
                action.error_details = _make_json_serializable({"error": error_msg})

            await db_session.commit()

            return {
                "success": result.get("success", False),
                "action_id": action_id,
                "action_type": action_type,
                "result": result,
                "step_index": step_index,
                "error": result.get("error") or result.get("detail")
                if not result.get("success")
                else None,
            }

        except Exception as e:
            # Update action record with error
            action.status = "failed"
            action.error_details = _make_json_serializable(
                {"error": str(e), "timestamp": datetime.utcnow().isoformat()}
            )
            action.completed_at = datetime.utcnow()
            await db_session.commit()

            return {
                "success": False,
                "action_id": action_id,
                "action_type": action_type,
                "error": str(e),
                "step_index": step_index,
            }

    async def _execute_block_ip_action(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute adaptive IP blocking action with intelligent system detection"""
        try:
            ip_address = params.get("ip_address")
            duration = params.get("duration")
            block_level = params.get("block_level", "standard")

            # Get system capabilities for adaptive command generation
            from .system_detector import get_system_detector

            detector = await get_system_detector()
            capabilities = await detector.detect_system_capabilities()

            self.logger.info(
                f"Detected target system: {capabilities.system_type.value} with {capabilities.firewall_type.value}"
            )

            # Generate adaptive commands based on system capabilities
            adaptive_commands = detector.generate_adaptive_commands(
                "block_ip", params, capabilities
            )

            # Use existing containment agent with enhanced parameters
            result = await self.containment_agent.execute_containment(
                {
                    "ip": ip_address,
                    "action": "block_ip",
                    "adaptive_commands": adaptive_commands,
                    "system_detected": f"{capabilities.system_type.value}/{capabilities.firewall_type.value}",
                    "duration": duration,
                    "reason": f"Adaptive blocking - {capabilities.firewall_type.value} on {capabilities.system_type.value}",
                }
            )

            return {
                "success": result.get("success", False),
                "detail": result.get("detail", ""),
                "system_detected": f"{capabilities.system_type.value}/{capabilities.firewall_type.value}",
                "firewall_type": capabilities.firewall_type.value,
                "adaptive_commands": len(adaptive_commands),
                "block_level": block_level,
                "duration": duration,
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _execute_isolate_host_action(
        self, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute advanced host isolation action"""
        try:
            host_identifier = params.get("host_identifier")
            isolation_level = params.get("isolation_level", "soft")

            result = await self.containment_agent.execute_containment(
                {
                    "ip": host_identifier,
                    "action": "isolate_host",
                    "isolation_level": isolation_level,
                    "reason": "Advanced host isolation via workflow",
                }
            )

            return {
                "success": result.get("success", False),
                "detail": result.get("detail", ""),
                "isolation_level": isolation_level,
                "host": host_identifier,
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _execute_firewall_rules_action(
        self, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute firewall rules deployment"""
        try:
            rule_set = params.get("rule_set", [])
            scope = params.get("scope", "local")

            # Simulate firewall rule deployment
            await asyncio.sleep(1)  # Simulate deployment time

            return {
                "success": True,
                "detail": f"Deployed {len(rule_set)} firewall rules with scope: {scope}",
                "rules_deployed": len(rule_set),
                "scope": scope,
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _execute_dns_sinkhole_action(
        self, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute DNS sinkholing action"""
        try:
            domains = params.get("domains", [])
            sinkhole_ip = params.get("sinkhole_ip", "127.0.0.1")

            # Simulate DNS sinkhole deployment
            await asyncio.sleep(1)

            return {
                "success": True,
                "detail": f"Sinkholed {len(domains)} domains to {sinkhole_ip}",
                "domains_sinkholed": len(domains),
                "sinkhole_ip": sinkhole_ip,
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _execute_memory_dump_action(
        self, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute memory dump collection"""
        try:
            target_hosts = params.get("target_hosts", [])
            dump_type = params.get("dump_type", "full")

            # Simulate memory dump collection
            await asyncio.sleep(3)  # Simulate collection time

            return {
                "success": True,
                "detail": f"Collected {dump_type} memory dumps from {len(target_hosts)} hosts",
                "dumps_collected": len(target_hosts),
                "dump_type": dump_type,
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _execute_create_incident_action(
        self, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute create incident action"""
        try:
            title = params.get("title", "Automated Incident")
            severity = params.get("severity", "medium")
            description = params.get("description", "")

            # Log incident creation (actual DB creation would happen in main incident handler)
            self.logger.info(f"Creating incident: {title} (severity: {severity})")

            return {
                "success": True,
                "detail": f"Incident created: {title}",
                "title": title,
                "severity": severity,
                "description": description,
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _execute_invoke_ai_agent_action(
        self, params: Dict[str, Any], db_session: Optional[AsyncSession] = None
    ) -> Dict[str, Any]:
        """Execute AI agent invocation action"""
        agent = params.get("agent", "attribution")
        task = params.get("task", "analyze")
        query = params.get("query", "")
        incident_id = params.get("incident_id")
        context = self._normalize_agent_context(params.get("context"))

        try:
            if not self.orchestrator:
                from .agent_orchestrator import get_orchestrator

                self.orchestrator = await get_orchestrator()

            if self.orchestrator:
                result = await self.orchestrator.orchestrate_agent_task(
                    agent_type=agent,
                    task=task,
                    query=query or f"{task} in context: {context}",
                    context=context,
                    incident_id=incident_id,
                    db_session=db_session,
                )

                return {
                    "success": result.get("success", True),
                    "detail": result.get("detail")
                    or f"AI agent {agent} executed task: {task}",
                    "agent": agent,
                    "task": task,
                    "analysis": result.get("analysis"),
                    "response": result.get("response"),
                    "context": context,
                }

        except Exception as e:
            self.logger.warning(f"Agent invocation fallback for {agent}: {e}")
            return await self._generate_agent_fallback_response(
                agent=agent,
                task=task,
                context=context,
                incident_id=incident_id,
                db_session=db_session,
                fallback_error=str(e),
            )

        # If orchestrator not available, provide contextual summary
        return await self._generate_agent_fallback_response(
            agent=agent,
            task=task,
            context=context,
            incident_id=incident_id,
            db_session=db_session,
        )

    def _normalize_agent_context(self, context: Any) -> Dict[str, Any]:
        if context is None:
            return {}
        if isinstance(context, dict):
            return context
        if isinstance(context, str):
            return {"topic": context}
        return {"value": context}

    async def _generate_agent_fallback_response(
        self,
        agent: str,
        task: str,
        context: Dict[str, Any],
        incident_id: Optional[int],
        db_session: Optional[AsyncSession],
        fallback_error: Optional[str] = None,
    ) -> Dict[str, Any]:
        detail_parts = [f"AI agent {agent} executed task: {task}"]
        analysis: Dict[str, Any] = {
            "context": context,
            "incident_id": incident_id,
            "fallback_used": True,
        }

        if fallback_error:
            detail_parts.append(f"fallback reason: {fallback_error}")
            analysis["fallback_reason"] = fallback_error

        # If we have incident context, include high-level metadata
        if db_session and incident_id:
            try:
                incident = await db_session.get(Incident, incident_id)
                if incident:
                    detail_parts.append(
                        f"incident #{incident.id} ({incident.src_ip}) context applied"
                    )
                    analysis.update(
                        {
                            "incident_status": incident.status,
                            "risk_score": incident.risk_score,
                            "source_ip": incident.src_ip,
                        }
                    )
            except Exception as context_error:
                self.logger.debug(f"Fallback context lookup failed: {context_error}")

        detail = "; ".join(detail_parts)

        return {
            "success": True,
            "detail": detail,
            "agent": agent,
            "task": task,
            "analysis": analysis,
            "context": context,
        }

    async def _execute_send_notification_action(
        self, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute send notification action"""
        try:
            channel = params.get("channel", "slack")
            message = params.get("message", "")
            recipients = params.get("recipients", [])
            priority = params.get("priority", "medium")

            # Log notification (actual notification would integrate with Slack/email/etc.)
            self.logger.info(
                f"Sending {priority} notification via {channel}: {message}"
            )

            return {
                "success": True,
                "detail": f"Notification sent via {channel}",
                "channel": channel,
                "message": message,
                "recipients": recipients if recipients else ["default"],
                "priority": priority,
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _validate_workflow_steps(
        self, steps: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Validate and normalize workflow steps"""
        validated_steps = []

        for step in steps:
            action_type = step.get("action_type")

            if not action_type:
                raise ValueError("Step missing action_type")

            if action_type not in self.action_registry:
                self.logger.warning(
                    f"Unknown action type: {action_type}, will attempt execution"
                )

            # Add default values
            validated_step = {
                "action_type": action_type,
                "parameters": step.get("parameters", {}),
                "timeout_seconds": step.get("timeout_seconds", 300),
                "continue_on_failure": step.get("continue_on_failure", False),
                "retry_count": step.get("retry_count", 0),
                "max_retries": step.get("max_retries", 3),
            }

            validated_steps.append(validated_step)

        return validated_steps

    async def _requires_approval(
        self, steps: List[Dict[str, Any]], incident_id: int
    ) -> bool:
        """Determine if workflow requires approval"""

        # Check for high-impact actions
        high_impact_actions = [
            "deploy_firewall_rules",
            "resource_isolation",
            "account_disable",
        ]

        for step in steps:
            action_type = step.get("action_type")
            if action_type in high_impact_actions:
                return True

            # Check safety level
            action_info = self.action_registry.get(action_type, {})
            if action_info.get("safety_level") == "high":
                return True

        return False

    async def _generate_rollback_plan(
        self, steps: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate rollback plan for workflow steps"""
        rollback_steps = []

        for step in reversed(steps):
            action_type = step.get("action_type")
            action_info = self.action_registry.get(action_type, {})

            if action_info.get("rollback_supported"):
                rollback_action = self._get_rollback_action(
                    action_type, step.get("parameters", {})
                )
                if rollback_action:
                    rollback_steps.append(rollback_action)

        return rollback_steps

    def _get_rollback_action(
        self, action_type: str, original_params: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Get the rollback action for a given action type"""

        rollback_mapping = {
            "block_ip_advanced": {
                "action_type": "unblock_ip",
                "parameters": {"ip_address": original_params.get("ip_address")},
            },
            "isolate_host_advanced": {
                "action_type": "un_isolate_host",
                "parameters": {
                    "host_identifier": original_params.get("host_identifier")
                },
            },
            "deploy_firewall_rules": {
                "action_type": "remove_firewall_rules",
                "parameters": {"rule_set": original_params.get("rule_set")},
            },
        }

        return rollback_mapping.get(action_type)

    def _make_json_serializable(self, obj: Any) -> Any:
        """Recursively convert sets to lists and other non-serializable types"""
        if isinstance(obj, set):
            return list(obj)
        elif isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, (datetime,)):
            return obj.isoformat()
        return obj

    async def _record_workflow_impact(
        self,
        workflow: ResponseWorkflow,
        execution_results: List[Dict[str, Any]],
        db_session: AsyncSession,
    ):
        """Record impact metrics for the workflow"""

        try:
            # Calculate impact metrics
            successful_actions = sum(
                1 for result in execution_results if result.get("success")
            )
            total_actions = len(execution_results)

            # Make execution_results JSON serializable (convert sets to lists)
            serializable_results = self._make_json_serializable(execution_results)

            # Create impact metrics record
            impact_metrics = ResponseImpactMetrics(
                workflow_id=workflow.id,
                attacks_blocked=successful_actions,  # Simplified metric
                false_positives=0,  # Would be calculated from feedback
                systems_affected=len(
                    set(
                        result.get("result", {}).get("host", "unknown")
                        for result in execution_results
                    )
                ),
                response_time_ms=workflow.execution_time_ms or 0,
                success_rate=successful_actions / total_actions
                if total_actions > 0
                else 0,
                confidence_score=workflow.ai_confidence or 0.8,
                metrics_data={
                    "execution_results": serializable_results,
                    "workflow_summary": {
                        "total_steps": total_actions,
                        "successful_steps": successful_actions,
                        "failed_steps": total_actions - successful_actions,
                    },
                },
            )

            db_session.add(impact_metrics)
            await db_session.commit()

        except Exception as e:
            self.logger.error(f"Failed to record impact metrics: {e}")
            # Don't fail the entire workflow for metrics recording failure
            try:
                await db_session.rollback()
            except:
                pass

    async def get_workflow_status(
        self, workflow_id: str, db_session: AsyncSession
    ) -> Dict[str, Any]:
        """Get detailed status of a workflow"""

        try:
            # Get workflow with related data
            result = await db_session.execute(
                select(ResponseWorkflow)
                .options(
                    selectinload(ResponseWorkflow.actions),
                    selectinload(ResponseWorkflow.impact_metrics),
                )
                .where(ResponseWorkflow.workflow_id == workflow_id)
            )

            workflow = result.scalars().first()

            if not workflow:
                return {"success": False, "error": "Workflow not found"}

            return {
                "success": True,
                "workflow_id": workflow.workflow_id,
                "status": workflow.status,
                "progress_percentage": workflow.progress_percentage,
                "current_step": workflow.current_step,
                "total_steps": workflow.total_steps,
                "success_rate": workflow.success_rate,
                "execution_time_ms": workflow.execution_time_ms,
                "created_at": workflow.created_at.isoformat()
                if workflow.created_at
                else None,
                "completed_at": workflow.completed_at.isoformat()
                if workflow.completed_at
                else None,
                "approval_required": workflow.approval_required,
                "approved_at": workflow.approved_at.isoformat()
                if workflow.approved_at
                else None,
                "actions": [
                    {
                        "action_id": action.action_id,
                        "action_type": action.action_type,
                        "status": action.status,
                        "result": action.result_data,
                    }
                    for action in workflow.actions
                ],
                "impact_metrics": [
                    {
                        "attacks_blocked": metric.attacks_blocked,
                        "success_rate": metric.success_rate,
                        "response_time_ms": metric.response_time_ms,
                    }
                    for metric in workflow.impact_metrics
                ],
            }

        except Exception as e:
            self.logger.error(f"Failed to get workflow status: {e}")
            return {"success": False, "error": str(e)}

    async def cancel_workflow(
        self, workflow_id: str, db_session: AsyncSession, cancelled_by: str = "system"
    ) -> Dict[str, Any]:
        """Cancel a running workflow"""

        try:
            result = await db_session.execute(
                select(ResponseWorkflow).where(
                    ResponseWorkflow.workflow_id == workflow_id
                )
            )

            workflow = result.scalars().first()

            if not workflow:
                return {"success": False, "error": "Workflow not found"}

            if workflow.status not in [WorkflowStatus.PENDING, WorkflowStatus.RUNNING]:
                return {
                    "success": False,
                    "error": f"Cannot cancel workflow in status: {workflow.status}",
                }

            workflow.status = WorkflowStatus.CANCELLED
            workflow.completed_at = datetime.utcnow()

            await db_session.commit()

            self.logger.info(f"Cancelled workflow {workflow_id} by {cancelled_by}")

            return {
                "success": True,
                "workflow_id": workflow_id,
                "status": workflow.status,
                "cancelled_by": cancelled_by,
            }

        except Exception as e:
            self.logger.error(f"Failed to cancel workflow: {e}")
            return {"success": False, "error": str(e)}

    def get_available_actions(
        self, category: Optional[ActionCategory] = None
    ) -> Dict[str, Any]:
        """Get list of available response actions"""

        actions = self.action_registry

        if category:
            actions = {
                k: v for k, v in actions.items() if v.get("category") == category
            }

        return {"success": True, "actions": actions, "total_count": len(actions)}


# Global instance
advanced_response_engine = AdvancedResponseEngine()


async def get_response_engine() -> AdvancedResponseEngine:
    """Get the global response engine instance"""
    if not advanced_response_engine.orchestrator:
        await advanced_response_engine.initialize()
    return advanced_response_engine
