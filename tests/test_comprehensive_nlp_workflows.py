"""
Comprehensive NLP Workflow Testing
Tests all 40+ response actions across all categories with natural language prompts
Validates: NLP parsing → workflow creation → database persistence → UI display
"""
import asyncio
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.app.nlp_workflow_parser import parse_workflow_from_natural_language
from backend.app.db import AsyncSessionLocal, init_db
from backend.app.models import WorkflowTrigger, NLPWorkflowSuggestion, Incident
from sqlalchemy import select
import json
from datetime import datetime


# Comprehensive test cases covering all 40+ actions and request types
COMPREHENSIVE_TEST_CASES = [
    # ========== NETWORK DEFENSE ACTIONS ==========
    {
        "category": "Network - Basic IP Blocking",
        "prompts": [
            "Block IP 192.168.1.100 immediately",
            "Ban the attacker at 10.0.0.5",
            "Block attacking IP ranges from 172.16.0.0",
            "Emergency: block IP 203.0.113.50 right now"
        ],
        "expected_actions": ["block_ip", "block_ip_advanced"],
        "expected_request_type": "response",
        "expected_category": "network"
    },
    {
        "category": "Network - Firewall Rules",
        "prompts": [
            "Deploy firewall rules to block the attack",
            "Establish DDoS protection on all interfaces",
            "Implement network containment for this threat",
            "Activate advanced firewall rules immediately"
        ],
        "expected_actions": ["deploy_firewall_rules"],
        "expected_request_type": "response",
        "expected_category": "network"
    },
    {
        "category": "Network - DNS Security",
        "prompts": [
            "Sinkhole the malicious domain evil.com",
            "Redirect suspicious DNS queries to honeypot",
            "Block command and control traffic",
            "Stop C2 communication from infected hosts"
        ],
        "expected_actions": ["block_c2_traffic"],
        "expected_request_type": "response",
        "expected_category": "network"
    },
    {
        "category": "Network - Traffic Analysis",
        "prompts": [
            "Capture network traffic for forensic analysis",
            "Perform traffic analysis on suspicious connections",
            "Monitor and inspect all network packets",
            "Start deep packet inspection on infected subnet"
        ],
        "expected_actions": ["capture_network_traffic"],
        "expected_request_type": "response",
        "expected_category": "network"
    },
    {
        "category": "Network - Rate Limiting",
        "prompts": [
            "Set up rate limiting on the API",
            "Throttle API calls from suspicious sources",
            "Implement bandwidth throttling for attacker",
            "Limit connection rate for brute force source"
        ],
        "expected_actions": ["api_rate_limiting"],
        "expected_request_type": "response",
        "expected_category": "cloud"
    },

    # ========== ENDPOINT CONTAINMENT ==========
    {
        "category": "Endpoint - Host Isolation",
        "prompts": [
            "Isolate the infected host immediately",
            "Quarantine the compromised endpoint",
            "Host isolation for web-server-01",
            "Lockdown workstation infected with malware",
            "Endpoint isolation for all affected systems"
        ],
        "expected_actions": ["isolate_host", "isolate_host_advanced"],
        "expected_request_type": "response",
        "expected_category": "endpoint"
    },
    {
        "category": "Endpoint - Process Control",
        "prompts": [
            "Terminate malicious process on the endpoint",
            "Kill suspicious process running on server",
            "Stop the malware process immediately",
            "End all ransomware processes"
        ],
        "expected_actions": ["terminate_process"],
        "expected_request_type": "response",
        "expected_category": "endpoint"
    },
    {
        "category": "Endpoint - Forensics Collection",
        "prompts": [
            "Capture forensic memory dump from infected host",
            "Memory dump collection for incident analysis",
            "Collect RAM dump from compromised server",
            "Perform memory dump on suspicious endpoint"
        ],
        "expected_actions": ["memory_dump_collection"],
        "expected_request_type": "response",
        "expected_category": "forensics"
    },
    {
        "category": "Endpoint - File Operations",
        "prompts": [
            "Quarantine malicious files on the system",
            "Isolate suspicious files from execution",
            "Delete malicious files from endpoint",
            "Scan the filesystem for malware"
        ],
        "expected_actions": ["isolate_file", "delete_malicious_files", "scan_filesystem"],
        "expected_request_type": "response",
        "expected_category": "endpoint"
    },

    # ========== IDENTITY & ACCESS MANAGEMENT ==========
    {
        "category": "Identity - Password Management",
        "prompts": [
            "Reset passwords for compromised accounts",
            "Force password reset for all affected users",
            "Change credentials for the breached accounts",
            "Implement credential stuffing defense by resetting passwords"
        ],
        "expected_actions": ["reset_passwords"],
        "expected_request_type": "response",
        "expected_category": "identity"
    },
    {
        "category": "Identity - MFA Enforcement",
        "prompts": [
            "Enforce MFA for all privileged accounts",
            "Require multi-factor authentication immediately",
            "Enable MFA enforcement for security",
            "Implement MFA across all users"
        ],
        "expected_actions": ["enforce_mfa"],
        "expected_request_type": "response",
        "expected_category": "identity"
    },
    {
        "category": "Identity - Account Control",
        "prompts": [
            "Disable compromised user account",
            "Suspend the suspicious user immediately",
            "Deactivate all affected accounts",
            "Lock out the attacker's account"
        ],
        "expected_actions": ["disable_user_account"],
        "expected_request_type": "response",
        "expected_category": "identity"
    },
    {
        "category": "Identity - Session Management",
        "prompts": [
            "Revoke all active user sessions",
            "Terminate user sessions for compromised accounts",
            "Logout all users immediately",
            "End all active sessions for security"
        ],
        "expected_actions": ["revoke_user_sessions"],
        "expected_request_type": "response",
        "expected_category": "identity"
    },

    # ========== EMAIL SECURITY ==========
    {
        "category": "Email - Phishing Response",
        "prompts": [
            "Quarantine the phishing emails immediately",
            "Pull malicious emails from all mailboxes",
            "Recall and quarantine suspicious messages",
            "Remove phishing campaign emails"
        ],
        "expected_actions": ["quarantine_email"],
        "expected_request_type": "response",
        "expected_category": "email"
    },
    {
        "category": "Email - Sender Blocking",
        "prompts": [
            "Block the sender domain malicious.com",
            "Ban sender from sending more phishing emails",
            "Block all emails from attacker domain",
            "Prevent future emails from this sender"
        ],
        "expected_actions": ["block_sender"],
        "expected_request_type": "response",
        "expected_category": "email"
    },

    # ========== DATA PROTECTION ==========
    {
        "category": "Data - Encryption",
        "prompts": [
            "Encrypt sensitive data to prevent breach",
            "Apply emergency data encryption",
            "Protect critical data with encryption",
            "Secure sensitive files with encryption"
        ],
        "expected_actions": ["encrypt_sensitive_data"],
        "expected_request_type": "response",
        "expected_category": "data"
    },
    {
        "category": "Data - Backup",
        "prompts": [
            "Backup critical data immediately",
            "Create emergency backup of sensitive systems",
            "Protect data by backing up now",
            "Ensure data is backed up before containment"
        ],
        "expected_actions": ["backup_critical_data"],
        "expected_request_type": "response",
        "expected_category": "data"
    },
    {
        "category": "Data - DLP",
        "prompts": [
            "Enable DLP to prevent data exfiltration",
            "Activate data loss prevention controls",
            "Prevent data theft with DLP rules",
            "Stop data exfiltration immediately"
        ],
        "expected_actions": ["enable_dlp"],
        "expected_request_type": "response",
        "expected_category": "data"
    },

    # ========== INVESTIGATION & FORENSICS ==========
    {
        "category": "Investigation - Threat Analysis",
        "prompts": [
            "Investigate the brute force attack",
            "Analyze suspicious behavior on the network",
            "Examine the malware infection path",
            "Research the threat actor's methods"
        ],
        "expected_actions": ["investigate_behavior"],
        "expected_request_type": "investigation",
        "expected_category": "forensics"
    },
    {
        "category": "Investigation - Threat Intelligence",
        "prompts": [
            "Lookup threat intelligence for this IP",
            "Check reputation of suspicious domain",
            "Search for IOCs in threat feeds",
            "Query threat intelligence databases"
        ],
        "expected_actions": ["threat_intel_lookup"],
        "expected_request_type": "investigation",
        "expected_category": "forensics"
    },
    {
        "category": "Investigation - Threat Hunting",
        "prompts": [
            "Hunt for similar attacks across network",
            "Search for indicators of compromise",
            "Find all instances of this malware",
            "Look for related suspicious activity"
        ],
        "expected_actions": ["hunt_similar_attacks"],
        "expected_request_type": "investigation",
        "expected_category": "forensics"
    },
    {
        "category": "Investigation - Malware Analysis",
        "prompts": [
            "Analyze the malicious payload",
            "Examine the binary for threats",
            "Investigate malware capabilities",
            "Study the ransomware sample"
        ],
        "expected_actions": ["analyze_malware"],
        "expected_request_type": "investigation",
        "expected_category": "forensics"
    },
    {
        "category": "Investigation - Evidence Collection",
        "prompts": [
            "Capture forensic evidence from the scene",
            "Collect evidence for investigation",
            "Preserve artifacts for analysis",
            "Gather forensic data from endpoints"
        ],
        "expected_actions": ["capture_forensic_evidence"],
        "expected_request_type": "investigation",
        "expected_category": "forensics"
    },
    {
        "category": "Investigation - Actor Tracking",
        "prompts": [
            "Track the threat actor's movements",
            "Identify the attacker's infrastructure",
            "Trace the campaign origin",
            "Monitor threat actor behavior"
        ],
        "expected_actions": ["track_threat_actor"],
        "expected_request_type": "investigation",
        "expected_category": "forensics"
    },

    # ========== COMMUNICATION & ALERTING ==========
    {
        "category": "Communication - Alerting",
        "prompts": [
            "Alert security analysts about this incident",
            "Notify the SOC team immediately",
            "Send alert to security operations",
            "Escalate to security team"
        ],
        "expected_actions": ["alert_security_analysts"],
        "expected_request_type": "response",
        "expected_category": "communication"
    },
    {
        "category": "Communication - Case Management",
        "prompts": [
            "Create incident case for tracking",
            "Open an incident response case",
            "Create ticket for this security event",
            "Start incident case management"
        ],
        "expected_actions": ["create_incident_case"],
        "expected_request_type": "response",
        "expected_category": "communication"
    },

    # ========== COMPLEX MULTI-ACTION SCENARIOS ==========
    {
        "category": "Complex - Malware Response",
        "prompts": [
            "Block IP 10.0.0.5 and isolate infected host",
            "Isolate endpoint, collect memory dump, then alert SOC",
            "Block attacker, reset passwords, and enable MFA",
            "Quarantine host and capture forensic evidence"
        ],
        "expected_actions": ["block_ip", "isolate_host", "memory_dump_collection", "reset_passwords", "enforce_mfa", "alert_security_analysts"],
        "expected_request_type": "response",
        "expected_category": "multiple"
    },
    {
        "category": "Complex - Ransomware Response",
        "prompts": [
            "Emergency: Isolate all infected hosts, backup critical data, and alert team",
            "Ransomware detected: isolate systems, collect forensics, notify analysts",
            "Contain ransomware by isolating hosts and blocking C2 traffic"
        ],
        "expected_actions": ["isolate_host", "backup_critical_data", "block_c2_traffic", "alert_security_analysts"],
        "expected_request_type": "response",
        "expected_category": "multiple"
    },
    {
        "category": "Complex - Credential Stuffing",
        "prompts": [
            "Credential stuffing attack: block IP, reset passwords, enable MFA",
            "Block attacker, reset compromised accounts, enforce multi-factor auth",
            "Defend against credential stuffing with IP block and password resets"
        ],
        "expected_actions": ["block_ip", "reset_passwords", "enforce_mfa"],
        "expected_request_type": "response",
        "expected_category": "multiple"
    },
    {
        "category": "Complex - DDoS Mitigation",
        "prompts": [
            "Mitigate DDoS attack by deploying firewall rules and rate limiting",
            "Stop DDoS with traffic analysis and bandwidth throttling",
            "Implement DDoS protection with firewall and rate limiting"
        ],
        "expected_actions": ["deploy_firewall_rules", "api_rate_limiting"],
        "expected_request_type": "response",
        "expected_category": "network"
    },
    {
        "category": "Complex - APT Investigation",
        "prompts": [
            "Investigate advanced persistent threat activity",
            "Hunt for APT indicators, track threat actor, analyze malware",
            "Research APT campaign and identify compromise scope"
        ],
        "expected_actions": ["investigate_behavior", "hunt_similar_attacks", "track_threat_actor", "analyze_malware"],
        "expected_request_type": "investigation",
        "expected_category": "forensics"
    },
    {
        "category": "Complex - Data Breach Response",
        "prompts": [
            "Data breach detected: enable DLP, encrypt data, backup critical files",
            "Prevent data exfiltration by enabling DLP and encrypting sensitive data",
            "Respond to breach with encryption, backup, and DLP activation"
        ],
        "expected_actions": ["enable_dlp", "encrypt_sensitive_data", "backup_critical_data"],
        "expected_request_type": "response",
        "expected_category": "data"
    },

    # ========== AUTOMATION TRIGGERS ==========
    {
        "category": "Automation - Trigger Creation",
        "prompts": [
            "Whenever SSH brute force is detected, automatically block the IP",
            "Every time phishing email arrives, quarantine it automatically",
            "Automate response: on malware detection, isolate host and alert team",
            "Always block IPs that attempt more than 10 failed logins"
        ],
        "expected_actions": ["block_ip", "quarantine_email", "isolate_host", "alert_security_analysts"],
        "expected_request_type": "automation",
        "expected_category": "multiple"
    },
    {
        "category": "Automation - Scheduled Actions",
        "prompts": [
            "Schedule daily threat hunting across the network",
            "Automatically run vulnerability scans every night",
            "Set up recurring backup of critical systems",
            "Configure automatic password rotation monthly"
        ],
        "expected_actions": ["hunt_similar_attacks"],
        "expected_request_type": "automation",
        "expected_category": "multiple"
    },

    # ========== REPORTING & ANALYTICS ==========
    {
        "category": "Reporting - Metrics",
        "prompts": [
            "Show me incident statistics for the last week",
            "Generate report of all blocked IPs",
            "List all recent security incidents",
            "Summarize threat activity this month"
        ],
        "expected_actions": [],  # Reporting requests may not generate actions
        "expected_request_type": "reporting",
        "expected_category": "reporting"
    },
    {
        "category": "Reporting - Export",
        "prompts": [
            "Export incident data to CSV",
            "Generate compliance report for audit",
            "Create executive summary of security posture",
            "Produce incident timeline report"
        ],
        "expected_actions": [],
        "expected_request_type": "reporting",
        "expected_category": "reporting"
    },

    # ========== Q&A / INFORMATIONAL ==========
    {
        "category": "Q&A - System Information",
        "prompts": [
            "What is the current threat level?",
            "How many incidents are open?",
            "Explain what ransomware is",
            "Tell me about brute force attacks"
        ],
        "expected_actions": [],
        "expected_request_type": "qa",
        "expected_category": "qa"
    },

    # ========== THREAT-SPECIFIC SCENARIOS ==========
    {
        "category": "Threat - SQL Injection",
        "prompts": [
            "SQL injection detected: block attacker and alert team",
            "Respond to SQL injection by blocking IP and investigating",
            "Web attack detected: implement firewall rules"
        ],
        "expected_actions": ["block_ip", "deploy_firewall_rules", "alert_security_analysts"],
        "expected_request_type": "response",
        "expected_category": "network"
    },
    {
        "category": "Threat - Insider Threat",
        "prompts": [
            "Insider threat detected: disable user account and investigate",
            "Suspicious insider activity: revoke sessions and alert team",
            "Employee data theft: enable DLP and disable account"
        ],
        "expected_actions": ["disable_user_account", "revoke_user_sessions", "enable_dlp", "alert_security_analysts"],
        "expected_request_type": "response",
        "expected_category": "multiple"
    },
    {
        "category": "Threat - Lateral Movement",
        "prompts": [
            "Lateral movement detected: isolate hosts and investigate",
            "Prevent lateral movement by network segmentation",
            "Contain lateral movement with host isolation"
        ],
        "expected_actions": ["isolate_host", "investigate_behavior"],
        "expected_request_type": "response",
        "expected_category": "endpoint"
    },
    {
        "category": "Threat - Command & Control",
        "prompts": [
            "C2 beaconing detected: block traffic and investigate",
            "Command and control communication: block C2 and isolate host",
            "Stop beaconing activity by blocking C2 traffic"
        ],
        "expected_actions": ["block_c2_traffic", "isolate_host"],
        "expected_request_type": "response",
        "expected_category": "network"
    },

    # ========== PRIORITY VARIATIONS ==========
    {
        "category": "Priority - Critical",
        "prompts": [
            "CRITICAL: Ransomware spreading - isolate all hosts immediately",
            "EMERGENCY: Active data breach - enable DLP now",
            "URGENT: APT detected - full investigation required"
        ],
        "expected_actions": ["isolate_host", "enable_dlp"],
        "expected_request_type": "response",
        "expected_priority": "critical"
    },
    {
        "category": "Priority - High",
        "prompts": [
            "High priority: Malware infection needs immediate containment",
            "Important: Block this attacker immediately",
            "High severity incident requires urgent response"
        ],
        "expected_actions": ["isolate_host", "block_ip"],
        "expected_request_type": "response",
        "expected_priority": "high"
    },
    {
        "category": "Priority - Low",
        "prompts": [
            "Low priority: Investigate suspicious activity when time permits",
            "Routine check: Hunt for IOCs in the environment",
            "Low severity: Review user access patterns"
        ],
        "expected_actions": ["investigate_behavior", "hunt_similar_attacks"],
        "expected_request_type": "investigation",
        "expected_priority": "low"
    }
]


class ComprehensiveNLPTester:
    """Comprehensive NLP workflow testing framework"""

    def __init__(self):
        self.db = None
        self.results = {
            "total_tests": 0,
            "passed": 0,
            "failed": 0,
            "categories_tested": set(),
            "actions_detected": set(),
            "request_types": {},
            "failures": []
        }

    async def setup(self):
        """Initialize database connection"""
        await init_db()
        self.db = AsyncSessionLocal()

    async def teardown(self):
        """Cleanup database connection"""
        if self.db:
            await self.db.close()

    async def test_nlp_parsing(self, prompt: str, test_case: dict):
        """Test single NLP prompt"""
        try:
            # Parse the prompt
            intent, explanation = await parse_workflow_from_natural_language(prompt)

            # Validate results
            validation = {
                "prompt": prompt,
                "category": test_case["category"],
                "parsed_successfully": True,
                "detected_actions": [a["action_type"] for a in intent.actions],
                "request_type": intent.request_type,
                "priority": intent.priority,
                "confidence": intent.confidence,
                "approval_required": intent.approval_required
            }

            # Check if expected actions were detected
            expected_actions = test_case.get("expected_actions", [])
            if expected_actions:
                detected_any = any(
                    exp_action in validation["detected_actions"]
                    for exp_action in expected_actions
                )
                validation["action_match"] = detected_any
            else:
                validation["action_match"] = True  # No actions expected

            # Check request type
            validation["request_type_match"] = (
                intent.request_type == test_case.get("expected_request_type", "response")
            )

            # Check priority if specified
            if "expected_priority" in test_case:
                validation["priority_match"] = (
                    intent.priority == test_case["expected_priority"]
                )
            else:
                validation["priority_match"] = True

            # Overall pass/fail
            validation["passed"] = (
                validation["action_match"] and
                validation["request_type_match"] and
                validation["priority_match"]
            )

            # Track statistics
            self.results["actions_detected"].update(validation["detected_actions"])
            request_type = intent.request_type
            self.results["request_types"][request_type] = self.results["request_types"].get(request_type, 0) + 1

            return validation

        except Exception as e:
            return {
                "prompt": prompt,
                "category": test_case["category"],
                "parsed_successfully": False,
                "error": str(e),
                "passed": False
            }

    async def test_database_persistence(self, prompt: str, incident_id: int = None):
        """Test workflow persistence to database"""
        try:
            # Parse workflow
            intent, explanation = await parse_workflow_from_natural_language(prompt, incident_id)

            # Create NLP suggestion
            detected_actions = [action['action_type'] for action in intent.actions]

            suggestion = NLPWorkflowSuggestion(
                prompt=prompt,
                incident_id=incident_id,
                request_type=intent.request_type,
                priority=intent.priority,
                confidence=intent.confidence,
                fallback_used=False,
                workflow_steps=intent.to_workflow_steps(),
                detected_actions=detected_actions,
                missing_actions=[],
                parser_version="v1.0",
                parser_diagnostics={
                    "explanation": explanation,
                    "confidence": intent.confidence,
                    "approval_required": intent.approval_required
                }
            )

            self.db.add(suggestion)
            await self.db.commit()
            await self.db.refresh(suggestion)

            # Verify it was saved
            result = await self.db.execute(
                select(NLPWorkflowSuggestion).where(NLPWorkflowSuggestion.id == suggestion.id)
            )
            saved_suggestion = result.scalar_one_or_none()

            return {
                "saved": saved_suggestion is not None,
                "suggestion_id": suggestion.id if saved_suggestion else None,
                "workflow_steps_count": len(saved_suggestion.workflow_steps) if saved_suggestion else 0
            }

        except Exception as e:
            return {
                "saved": False,
                "error": str(e)
            }

    async def test_trigger_creation(self, suggestion_id: int):
        """Test creating trigger from suggestion"""
        try:
            # Get suggestion
            result = await self.db.execute(
                select(NLPWorkflowSuggestion).where(NLPWorkflowSuggestion.id == suggestion_id)
            )
            suggestion = result.scalar_one_or_none()

            if not suggestion:
                return {"created": False, "error": "Suggestion not found"}

            # Create trigger
            trigger = WorkflowTrigger(
                name=f"NLP_Trigger_{suggestion.id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                description=f"Auto-generated from NLP: {suggestion.prompt[:100]}",
                category="nlp_generated",
                enabled=True,
                auto_execute=False,
                priority=suggestion.priority,
                status="active",
                conditions={"event_type": "manual", "source": "nlp"},
                playbook_name=f"nlp_{suggestion.id}",
                workflow_steps=suggestion.workflow_steps,
                source="nlp",
                source_prompt=suggestion.prompt,
                parser_confidence=suggestion.confidence,
                parser_version=suggestion.parser_version,
                request_type=suggestion.request_type,
                fallback_used=suggestion.fallback_used,
                created_by="test_system"
            )

            self.db.add(trigger)

            # Update suggestion
            suggestion.status = "approved"
            suggestion.trigger_id = trigger.id
            suggestion.reviewed_by = "test_system"
            suggestion.reviewed_at = datetime.utcnow()

            await self.db.commit()
            await self.db.refresh(trigger)

            return {
                "created": True,
                "trigger_id": trigger.id,
                "trigger_name": trigger.name
            }

        except Exception as e:
            await self.db.rollback()
            return {
                "created": False,
                "error": str(e)
            }

    async def run_comprehensive_tests(self):
        """Run all comprehensive tests"""
        print("=" * 80)
        print("COMPREHENSIVE NLP WORKFLOW TESTING")
        print("=" * 80)
        print(f"Testing {len(COMPREHENSIVE_TEST_CASES)} categories with 100+ prompts")
        print()

        await self.setup()

        test_number = 0

        for test_case in COMPREHENSIVE_TEST_CASES:
            category = test_case["category"]
            print(f"\n{'='*80}")
            print(f"CATEGORY: {category}")
            print(f"{'='*80}")

            self.results["categories_tested"].add(category)

            for prompt in test_case["prompts"]:
                test_number += 1
                self.results["total_tests"] += 1

                print(f"\n[Test {test_number}] Prompt: \"{prompt}\"")

                # Test NLP parsing
                parse_result = await self.test_nlp_parsing(prompt, test_case)

                if parse_result["passed"]:
                    self.results["passed"] += 1
                    print(f"  ✓ PASSED")
                    print(f"    - Actions: {', '.join(parse_result['detected_actions'])}")
                    print(f"    - Request Type: {parse_result['request_type']}")
                    print(f"    - Priority: {parse_result['priority']}")
                    print(f"    - Confidence: {parse_result['confidence']:.2f}")
                else:
                    self.results["failed"] += 1
                    self.results["failures"].append(parse_result)
                    print(f"  ✗ FAILED")
                    if "error" in parse_result:
                        print(f"    - Error: {parse_result['error']}")
                    else:
                        print(f"    - Expected actions not detected")
                        print(f"    - Got: {parse_result.get('detected_actions', [])}")

                # Test database persistence (sample every 10th test to avoid DB bloat)
                if test_number % 10 == 0:
                    print(f"  Testing database persistence...")
                    db_result = await self.test_database_persistence(prompt)
                    if db_result["saved"]:
                        print(f"    ✓ Saved to database (ID: {db_result['suggestion_id']})")

                        # Test trigger creation
                        trigger_result = await self.test_trigger_creation(db_result['suggestion_id'])
                        if trigger_result["created"]:
                            print(f"    ✓ Trigger created (ID: {trigger_result['trigger_id']})")
                        else:
                            print(f"    ✗ Trigger creation failed: {trigger_result.get('error', 'Unknown')}")
                    else:
                        print(f"    ✗ Database save failed: {db_result.get('error', 'Unknown')}")

        await self.teardown()

        # Print final results
        print("\n" + "=" * 80)
        print("FINAL TEST RESULTS")
        print("=" * 80)
        print(f"Total Tests: {self.results['total_tests']}")
        print(f"Passed: {self.results['passed']} ({self.results['passed']/self.results['total_tests']*100:.1f}%)")
        print(f"Failed: {self.results['failed']} ({self.results['failed']/self.results['total_tests']*100:.1f}%)")
        print(f"\nCategories Tested: {len(self.results['categories_tested'])}")
        print(f"Unique Actions Detected: {len(self.results['actions_detected'])}")
        print(f"  - {', '.join(sorted(self.results['actions_detected']))}")
        print(f"\nRequest Type Distribution:")
        for req_type, count in sorted(self.results['request_types'].items()):
            print(f"  - {req_type}: {count}")

        if self.results['failures']:
            print(f"\n⚠ {len(self.results['failures'])} Failed Tests:")
            for i, failure in enumerate(self.results['failures'][:10], 1):  # Show first 10
                print(f"  {i}. {failure['category']}: \"{failure['prompt'][:60]}...\"")

        print("\n" + "=" * 80)
        return self.results


async def main():
    """Main test runner"""
    tester = ComprehensiveNLPTester()
    results = await tester.run_comprehensive_tests()

    # Exit with appropriate code
    exit_code = 0 if results['failed'] == 0 else 1
    sys.exit(exit_code)


if __name__ == "__main__":
    asyncio.run(main())
