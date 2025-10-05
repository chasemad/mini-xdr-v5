#!/usr/bin/env python3
"""
Setup Comprehensive T-Pot Workflow Triggers
Creates automatic response workflows for all T-Pot honeypot attack types
"""
import asyncio
import sys
import os
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "backend"))

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import select
from app.models import WorkflowTrigger
from app.config import settings

# Color codes for output
GREEN = '\033[0;32m'
BLUE = '\033[0;34m'
YELLOW = '\033[1;33m'
RED = '\033[0;31m'
NC = '\033[0m'

def log(msg):
    print(f"{BLUE}[INFO]{NC} {msg}")

def success(msg):
    print(f"{GREEN}✅ {msg}{NC}")

def warning(msg):
    print(f"{YELLOW}⚠️  {msg}{NC}")

def error(msg):
    print(f"{RED}❌ {msg}{NC}")


# Define all T-Pot workflow triggers
TPOT_TRIGGERS = [
    # =============================================================================
    # COWRIE SSH/TELNET HONEYPOT TRIGGERS
    # =============================================================================
    {
        "name": "T-Pot: SSH Brute Force Attack",
        "description": "Detect and block SSH brute force attacks from T-Pot Cowrie honeypot",
        "category": "honeypot",
        "enabled": True,
        "auto_execute": True,
        "priority": "high",
        "conditions": {
            "event_type": "cowrie.login.failed",
            "threshold": 5,
            "window_seconds": 60,
            "source": "honeypot"
        },
        "playbook_name": "SSH Brute Force Response",
        "workflow_steps": [
            {
                "action_type": "block_ip",
                "parameters": {
                    "ip_address": "event.source_ip",
                    "duration": 3600,
                    "block_level": "aggressive"
                },
                "timeout_seconds": 30,
                "continue_on_failure": False
            },
            {
                "action_type": "create_incident",
                "parameters": {
                    "title": "SSH Brute Force Attack Detected",
                    "severity": "high",
                    "description": "Multiple failed SSH login attempts from {source_ip} on T-Pot Cowrie honeypot"
                },
                "timeout_seconds": 10,
                "continue_on_failure": False
            },
            {
                "action_type": "invoke_ai_agent",
                "parameters": {
                    "agent": "attribution",
                    "task": "profile_threat_actor",
                    "context": "ssh_brute_force"
                },
                "timeout_seconds": 60,
                "continue_on_failure": True
            },
            {
                "action_type": "send_notification",
                "parameters": {
                    "channel": "slack",
                    "message": "🚨 SSH brute force blocked: {source_ip} - {threshold} attempts in {window}s"
                },
                "timeout_seconds": 10,
                "continue_on_failure": True
            }
        ],
        "cooldown_seconds": 60,
        "max_triggers_per_day": 100,
        "tags": ["tpot", "cowrie", "ssh", "brute-force"]
    },
    
    {
        "name": "T-Pot: Successful SSH Compromise",
        "description": "Alert on successful login attempts (potential compromise indicators)",
        "category": "honeypot",
        "enabled": True,
        "auto_execute": True,
        "priority": "critical",
        "conditions": {
            "event_type": "cowrie.login.success",
            "source": "honeypot"
        },
        "playbook_name": "Honeypot Compromise Response",
        "workflow_steps": [
            {
                "action_type": "block_ip",
                "parameters": {
                    "ip_address": "event.source_ip",
                    "duration": 86400,
                    "block_level": "aggressive"
                },
                "timeout_seconds": 30,
                "continue_on_failure": False
            },
            {
                "action_type": "create_incident",
                "parameters": {
                    "title": "Successful SSH Login on Honeypot",
                    "severity": "critical",
                    "description": "Attacker successfully logged into honeypot from {source_ip}"
                },
                "timeout_seconds": 10,
                "continue_on_failure": False
            },
            {
                "action_type": "invoke_ai_agent",
                "parameters": {
                    "agent": "forensics",
                    "task": "capture_session_details",
                    "context": "honeypot_compromise"
                },
                "timeout_seconds": 120,
                "continue_on_failure": True
            },
            {
                "action_type": "send_notification",
                "parameters": {
                    "channel": "slack",
                    "message": "🔴 CRITICAL: Successful honeypot login from {source_ip}"
                },
                "timeout_seconds": 10,
                "continue_on_failure": True
            }
        ],
        "cooldown_seconds": 30,
        "max_triggers_per_day": 50,
        "tags": ["tpot", "cowrie", "ssh", "compromise", "critical"]
    },

    {
        "name": "T-Pot: Malicious Command Execution",
        "description": "Detect suspicious commands executed on Cowrie honeypot",
        "category": "honeypot",
        "enabled": True,
        "auto_execute": True,
        "priority": "high",
        "conditions": {
            "event_type": "cowrie.command.input",
            "threshold": 3,
            "window_seconds": 120,
            "source": "honeypot"
        },
        "playbook_name": "Malicious Command Response",
        "workflow_steps": [
            {
                "action_type": "create_incident",
                "parameters": {
                    "title": "Malicious Commands Detected",
                    "severity": "high",
                    "description": "Multiple suspicious commands executed from {source_ip}"
                },
                "timeout_seconds": 10,
                "continue_on_failure": False
            },
            {
                "action_type": "invoke_ai_agent",
                "parameters": {
                    "agent": "forensics",
                    "task": "analyze_command_chain",
                    "context": "command_execution"
                },
                "timeout_seconds": 90,
                "continue_on_failure": True
            },
            {
                "action_type": "block_ip",
                "parameters": {
                    "ip_address": "event.source_ip",
                    "duration": 7200
                },
                "timeout_seconds": 30,
                "continue_on_failure": False
            }
        ],
        "cooldown_seconds": 90,
        "max_triggers_per_day": 75,
        "tags": ["tpot", "cowrie", "commands", "execution"]
    },

    # =============================================================================
    # DIONAEA MULTI-PROTOCOL HONEYPOT TRIGGERS
    # =============================================================================
    {
        "name": "T-Pot: Malware Upload Detection (Dionaea)",
        "description": "Detect malware payloads uploaded to Dionaea honeypot",
        "category": "honeypot",
        "enabled": True,
        "auto_execute": True,
        "priority": "critical",
        "conditions": {
            "event_type": "dionaea.connection.protocol.smb",
            "pattern_match": "upload",
            "source": "honeypot"
        },
        "playbook_name": "Malware Containment",
        "workflow_steps": [
            {
                "action_type": "block_ip",
                "parameters": {
                    "ip_address": "event.source_ip",
                    "duration": 86400,
                    "block_level": "aggressive"
                },
                "timeout_seconds": 30,
                "continue_on_failure": False
            },
            {
                "action_type": "create_incident",
                "parameters": {
                    "title": "Malware Upload Detected (Dionaea)",
                    "severity": "critical",
                    "description": "Malware payload uploaded from {source_ip}"
                },
                "timeout_seconds": 10,
                "continue_on_failure": False
            },
            {
                "action_type": "invoke_ai_agent",
                "parameters": {
                    "agent": "containment",
                    "task": "full_isolation",
                    "context": "malware_upload"
                },
                "timeout_seconds": 120,
                "continue_on_failure": True
            },
            {
                "action_type": "send_notification",
                "parameters": {
                    "channel": "slack",
                    "message": "🦠 CRITICAL: Malware upload from {source_ip} (Dionaea)"
                },
                "timeout_seconds": 10,
                "continue_on_failure": True
            }
        ],
        "cooldown_seconds": 30,
        "max_triggers_per_day": 100,
        "tags": ["tpot", "dionaea", "malware", "critical"]
    },

    {
        "name": "T-Pot: SMB/CIFS Exploit Attempt",
        "description": "Detect SMB/CIFS exploitation attempts on Dionaea",
        "category": "honeypot",
        "enabled": True,
        "auto_execute": True,
        "priority": "high",
        "conditions": {
            "event_type": "dionaea.connection.protocol.smb",
            "threshold": 3,
            "window_seconds": 120,
            "source": "honeypot"
        },
        "playbook_name": "SMB Exploit Response",
        "workflow_steps": [
            {
                "action_type": "block_ip",
                "parameters": {
                    "ip_address": "event.source_ip",
                    "duration": 3600
                },
                "timeout_seconds": 30,
                "continue_on_failure": False
            },
            {
                "action_type": "create_incident",
                "parameters": {
                    "title": "SMB/CIFS Exploit Attempt",
                    "severity": "high",
                    "description": "SMB exploitation attempts from {source_ip}"
                },
                "timeout_seconds": 10,
                "continue_on_failure": False
            },
            {
                "action_type": "invoke_ai_agent",
                "parameters": {
                    "agent": "attribution",
                    "task": "analyze_exploit_pattern",
                    "context": "smb_exploit"
                },
                "timeout_seconds": 90,
                "continue_on_failure": True
            }
        ],
        "cooldown_seconds": 120,
        "max_triggers_per_day": 60,
        "tags": ["tpot", "dionaea", "smb", "exploit"]
    },

    # =============================================================================
    # SURICATA IDS TRIGGERS
    # =============================================================================
    {
        "name": "T-Pot: Suricata IDS Alert (High Severity)",
        "description": "Respond to high-severity Suricata IDS alerts",
        "category": "honeypot",
        "enabled": True,
        "auto_execute": True,
        "priority": "high",
        "conditions": {
            "event_type": "suricata.alert",
            "risk_score_min": 0.7,
            "source": "honeypot"
        },
        "playbook_name": "IDS Alert Response",
        "workflow_steps": [
            {
                "action_type": "create_incident",
                "parameters": {
                    "title": "Suricata IDS High-Severity Alert",
                    "severity": "high",
                    "description": "High-severity network threat from {source_ip}"
                },
                "timeout_seconds": 10,
                "continue_on_failure": False
            },
            {
                "action_type": "invoke_ai_agent",
                "parameters": {
                    "agent": "forensics",
                    "task": "analyze_network_pattern",
                    "context": "ids_alert"
                },
                "timeout_seconds": 90,
                "continue_on_failure": True
            },
            {
                "action_type": "block_ip",
                "parameters": {
                    "ip_address": "event.source_ip",
                    "duration": 7200
                },
                "timeout_seconds": 30,
                "continue_on_failure": False
            }
        ],
        "cooldown_seconds": 120,
        "max_triggers_per_day": 100,
        "tags": ["tpot", "suricata", "ids", "network"]
    },

    # =============================================================================
    # ELASTICPOT TRIGGERS
    # =============================================================================
    {
        "name": "T-Pot: Elasticsearch Exploit Attempt",
        "description": "Detect Elasticsearch exploitation attempts on Elasticpot",
        "category": "honeypot",
        "enabled": True,
        "auto_execute": True,
        "priority": "high",
        "conditions": {
            "event_type": "elasticpot.attack",
            "source": "honeypot"
        },
        "playbook_name": "Database Exploit Response",
        "workflow_steps": [
            {
                "action_type": "block_ip",
                "parameters": {
                    "ip_address": "event.source_ip",
                    "duration": 7200
                },
                "timeout_seconds": 30,
                "continue_on_failure": False
            },
            {
                "action_type": "create_incident",
                "parameters": {
                    "title": "Elasticsearch Exploit Attempt",
                    "severity": "high",
                    "description": "Elasticsearch exploitation from {source_ip}"
                },
                "timeout_seconds": 10,
                "continue_on_failure": False
            },
            {
                "action_type": "invoke_ai_agent",
                "parameters": {
                    "agent": "attribution",
                    "task": "analyze_database_attack",
                    "context": "elasticsearch_exploit"
                },
                "timeout_seconds": 90,
                "continue_on_failure": True
            }
        ],
        "cooldown_seconds": 120,
        "max_triggers_per_day": 50,
        "tags": ["tpot", "elasticpot", "database", "exploit"]
    },

    # =============================================================================
    # HONEYTRAP TRIGGERS
    # =============================================================================
    {
        "name": "T-Pot: Network Service Scan",
        "description": "Detect network service scanning on Honeytrap",
        "category": "honeypot",
        "enabled": True,
        "auto_execute": False,  # Requires approval (common activity)
        "priority": "medium",
        "conditions": {
            "event_type": "honeytrap.connection",
            "threshold": 10,
            "window_seconds": 60,
            "source": "honeypot"
        },
        "playbook_name": "Port Scan Response",
        "workflow_steps": [
            {
                "action_type": "create_incident",
                "parameters": {
                    "title": "Network Service Scan Detected",
                    "severity": "medium",
                    "description": "Multiple service connections from {source_ip}"
                },
                "timeout_seconds": 10,
                "continue_on_failure": False
            },
            {
                "action_type": "invoke_ai_agent",
                "parameters": {
                    "agent": "attribution",
                    "task": "profile_scanner",
                    "context": "port_scan"
                },
                "timeout_seconds": 60,
                "continue_on_failure": True
            }
        ],
        "cooldown_seconds": 300,
        "max_triggers_per_day": 30,
        "tags": ["tpot", "honeytrap", "scan", "reconnaissance"]
    },

    # =============================================================================
    # SPECIALIZED THREAT DETECTORS
    # =============================================================================
    {
        "name": "T-Pot: Cryptomining Detection",
        "description": "Detect cryptocurrency mining activity",
        "category": "honeypot",
        "enabled": True,
        "auto_execute": True,
        "priority": "high",
        "conditions": {
            "pattern_match": "cryptomining",
            "source": "honeypot"
        },
        "playbook_name": "Cryptomining Response",
        "workflow_steps": [
            {
                "action_type": "block_ip",
                "parameters": {
                    "ip_address": "event.source_ip",
                    "duration": 86400,
                    "block_level": "aggressive"
                },
                "timeout_seconds": 30,
                "continue_on_failure": False
            },
            {
                "action_type": "create_incident",
                "parameters": {
                    "title": "Cryptomining Activity Detected",
                    "severity": "high",
                    "description": "Cryptocurrency mining detected from {source_ip}"
                },
                "timeout_seconds": 10,
                "continue_on_failure": False
            },
            {
                "action_type": "invoke_ai_agent",
                "parameters": {
                    "agent": "containment",
                    "task": "isolate_and_terminate",
                    "context": "cryptomining"
                },
                "timeout_seconds": 120,
                "continue_on_failure": True
            },
            {
                "action_type": "send_notification",
                "parameters": {
                    "channel": "slack",
                    "message": "⛏️ Cryptomining blocked: {source_ip}"
                },
                "timeout_seconds": 10,
                "continue_on_failure": True
            }
        ],
        "cooldown_seconds": 60,
        "max_triggers_per_day": 50,
        "tags": ["tpot", "cryptomining", "resource-abuse"]
    },

    {
        "name": "T-Pot: Data Exfiltration Attempt",
        "description": "Detect potential data exfiltration activity",
        "category": "honeypot",
        "enabled": True,
        "auto_execute": True,
        "priority": "critical",
        "conditions": {
            "pattern_match": "data_exfiltration",
            "source": "honeypot"
        },
        "playbook_name": "Data Exfiltration Response",
        "workflow_steps": [
            {
                "action_type": "block_ip",
                "parameters": {
                    "ip_address": "event.source_ip",
                    "duration": 86400,
                    "block_level": "aggressive"
                },
                "timeout_seconds": 30,
                "continue_on_failure": False
            },
            {
                "action_type": "create_incident",
                "parameters": {
                    "title": "Data Exfiltration Detected",
                    "severity": "critical",
                    "description": "Potential data exfiltration from {source_ip}"
                },
                "timeout_seconds": 10,
                "continue_on_failure": False
            },
            {
                "action_type": "invoke_ai_agent",
                "parameters": {
                    "agent": "forensics",
                    "task": "analyze_data_transfer",
                    "context": "data_exfiltration"
                },
                "timeout_seconds": 120,
                "continue_on_failure": True
            },
            {
                "action_type": "send_notification",
                "parameters": {
                    "channel": "slack",
                    "message": "🚨 CRITICAL: Data exfiltration attempt from {source_ip}"
                },
                "timeout_seconds": 10,
                "continue_on_failure": True
            }
        ],
        "cooldown_seconds": 30,
        "max_triggers_per_day": 50,
        "tags": ["tpot", "exfiltration", "critical"]
    },

    {
        "name": "T-Pot: Ransomware Indicators",
        "description": "Detect ransomware preparation and execution behaviors",
        "category": "honeypot",
        "enabled": True,
        "auto_execute": True,
        "priority": "critical",
        "conditions": {
            "pattern_match": "ransomware",
            "source": "honeypot"
        },
        "playbook_name": "Ransomware Emergency Response",
        "workflow_steps": [
            {
                "action_type": "block_ip",
                "parameters": {
                    "ip_address": "event.source_ip",
                    "duration": 604800,  # 7 days
                    "block_level": "aggressive"
                },
                "timeout_seconds": 30,
                "continue_on_failure": False
            },
            {
                "action_type": "create_incident",
                "parameters": {
                    "title": "Ransomware Activity Detected",
                    "severity": "critical",
                    "description": "Ransomware indicators from {source_ip}"
                },
                "timeout_seconds": 10,
                "continue_on_failure": False
            },
            {
                "action_type": "invoke_ai_agent",
                "parameters": {
                    "agent": "containment",
                    "task": "emergency_isolation",
                    "context": "ransomware"
                },
                "timeout_seconds": 120,
                "continue_on_failure": True
            },
            {
                "action_type": "send_notification",
                "parameters": {
                    "channel": "slack",
                    "message": "🔴 CRITICAL ALERT: Ransomware detected from {source_ip}"
                },
                "timeout_seconds": 10,
                "continue_on_failure": True
            }
        ],
        "cooldown_seconds": 30,
        "max_triggers_per_day": 25,
        "tags": ["tpot", "ransomware", "critical", "emergency"]
    },

    {
        "name": "T-Pot: IoT Botnet Activity",
        "description": "Detect IoT botnet recruitment attempts (Mirai, etc.)",
        "category": "honeypot",
        "enabled": True,
        "auto_execute": True,
        "priority": "high",
        "conditions": {
            "pattern_match": "iot_botnet",
            "source": "honeypot"
        },
        "playbook_name": "IoT Botnet Response",
        "workflow_steps": [
            {
                "action_type": "block_ip",
                "parameters": {
                    "ip_address": "event.source_ip",
                    "duration": 86400,
                    "block_level": "aggressive"
                },
                "timeout_seconds": 30,
                "continue_on_failure": False
            },
            {
                "action_type": "create_incident",
                "parameters": {
                    "title": "IoT Botnet Activity Detected",
                    "severity": "high",
                    "description": "IoT botnet recruitment attempt from {source_ip}"
                },
                "timeout_seconds": 10,
                "continue_on_failure": False
            },
            {
                "action_type": "invoke_ai_agent",
                "parameters": {
                    "agent": "attribution",
                    "task": "identify_botnet_campaign",
                    "context": "iot_botnet"
                },
                "timeout_seconds": 90,
                "continue_on_failure": True
            }
        ],
        "cooldown_seconds": 120,
        "max_triggers_per_day": 75,
        "tags": ["tpot", "botnet", "iot", "mirai"]
    },

    {
        "name": "T-Pot: DDoS Attack Detection",
        "description": "Detect distributed denial of service attacks",
        "category": "honeypot",
        "enabled": True,
        "auto_execute": True,
        "priority": "critical",
        "conditions": {
            "threshold": 100,
            "window_seconds": 10,
            "source": "honeypot"
        },
        "playbook_name": "DDoS Mitigation",
        "workflow_steps": [
            {
                "action_type": "create_incident",
                "parameters": {
                    "title": "DDoS Attack Detected",
                    "severity": "critical",
                    "description": "High-volume attack traffic detected"
                },
                "timeout_seconds": 10,
                "continue_on_failure": False
            },
            {
                "action_type": "invoke_ai_agent",
                "parameters": {
                    "agent": "containment",
                    "task": "enable_rate_limiting",
                    "context": "ddos_attack"
                },
                "timeout_seconds": 60,
                "continue_on_failure": True
            },
            {
                "action_type": "send_notification",
                "parameters": {
                    "channel": "slack",
                    "message": "⚠️ DDoS attack in progress - rate limiting engaged"
                },
                "timeout_seconds": 10,
                "continue_on_failure": True
            }
        ],
        "cooldown_seconds": 300,  # 5 minutes between alerts
        "max_triggers_per_day": 10,
        "tags": ["tpot", "ddos", "volumetric", "critical"]
    },

    # =============================================================================
    # WEB APPLICATION ATTACKS
    # =============================================================================
    {
        "name": "T-Pot: SQL Injection Attempt",
        "description": "Detect SQL injection attempts on web honeypots",
        "category": "honeypot",
        "enabled": True,
        "auto_execute": False,  # Requires approval
        "priority": "high",
        "conditions": {
            "pattern_match": "sql_injection",
            "source": "honeypot"
        },
        "playbook_name": "SQL Injection Response",
        "workflow_steps": [
            {
                "action_type": "create_incident",
                "parameters": {
                    "title": "SQL Injection Attempt",
                    "severity": "high",
                    "description": "SQL injection detected from {source_ip}"
                },
                "timeout_seconds": 10,
                "continue_on_failure": False
            },
            {
                "action_type": "invoke_ai_agent",
                "parameters": {
                    "agent": "forensics",
                    "task": "analyze_injection_payload",
                    "context": "sql_injection"
                },
                "timeout_seconds": 90,
                "continue_on_failure": True
            },
            {
                "action_type": "block_ip",
                "parameters": {
                    "ip_address": "event.source_ip",
                    "duration": 7200
                },
                "timeout_seconds": 30,
                "continue_on_failure": False
            }
        ],
        "cooldown_seconds": 120,
        "max_triggers_per_day": 50,
        "tags": ["tpot", "web", "sql-injection"]
    },

    {
        "name": "T-Pot: XSS Attack Attempt",
        "description": "Detect cross-site scripting attempts",
        "category": "honeypot",
        "enabled": True,
        "auto_execute": False,
        "priority": "medium",
        "conditions": {
            "pattern_match": "xss",
            "source": "honeypot"
        },
        "playbook_name": "XSS Response",
        "workflow_steps": [
            {
                "action_type": "create_incident",
                "parameters": {
                    "title": "XSS Attack Attempt",
                    "severity": "medium",
                    "description": "Cross-site scripting attempt from {source_ip}"
                },
                "timeout_seconds": 10,
                "continue_on_failure": False
            },
            {
                "action_type": "invoke_ai_agent",
                "parameters": {
                    "agent": "forensics",
                    "task": "analyze_xss_payload",
                    "context": "xss_attack"
                },
                "timeout_seconds": 60,
                "continue_on_failure": True
            }
        ],
        "cooldown_seconds": 180,
        "max_triggers_per_day": 40,
        "tags": ["tpot", "web", "xss"]
    },
]


async def create_or_update_triggers(session: AsyncSession):
    """Create or update all T-Pot workflow triggers"""
    created = 0
    updated = 0
    skipped = 0

    for trigger_data in TPOT_TRIGGERS:
        try:
            # Check if trigger already exists
            result = await session.execute(
                select(WorkflowTrigger).where(
                    WorkflowTrigger.name == trigger_data["name"]
                )
            )
            existing = result.scalar_one_or_none()

            if existing:
                # Update existing trigger
                log(f"Updating trigger: {trigger_data['name']}")
                for key, value in trigger_data.items():
                    if key != "name":  # Don't update the name
                        setattr(existing, key, value)
                updated += 1
            else:
                # Create new trigger
                log(f"Creating trigger: {trigger_data['name']}")
                trigger = WorkflowTrigger(**trigger_data)
                session.add(trigger)
                created += 1

        except Exception as e:
            error(f"Error processing trigger '{trigger_data['name']}': {e}")
            skipped += 1
            continue

    # Commit all changes
    await session.commit()
    
    return created, updated, skipped


async def main():
    """Main setup function"""
    print(f"\n{BLUE}{'='*70}{NC}")
    print(f"{BLUE}T-Pot Workflow Trigger Setup{NC}")
    print(f"{BLUE}{'='*70}{NC}\n")

    log("Connecting to database...")
    engine = create_async_engine(settings.database_url, echo=False)

    async_session = sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )

    try:
        async with async_session() as session:
            log(f"Creating/updating {len(TPOT_TRIGGERS)} T-Pot workflow triggers...")
            created, updated, skipped = await create_or_update_triggers(session)

            print(f"\n{BLUE}{'='*70}{NC}")
            success(f"Setup complete!")
            print(f"\n{GREEN}📊 Summary:{NC}")
            print(f"  • Created:  {created} new triggers")
            print(f"  • Updated:  {updated} existing triggers")
            print(f"  • Skipped:  {skipped} triggers (errors)")
            print(f"  • Total:    {len(TPOT_TRIGGERS)} triggers configured")
            
            print(f"\n{GREEN}🎯 Trigger Categories:{NC}")
            categories = {}
            for trigger in TPOT_TRIGGERS:
                cat = trigger.get("category", "unknown")
                categories[cat] = categories.get(cat, 0) + 1
            for cat, count in categories.items():
                print(f"  • {cat}: {count}")

            print(f"\n{GREEN}🚀 Auto-Execute Status:{NC}")
            auto_count = sum(1 for t in TPOT_TRIGGERS if t.get("auto_execute"))
            manual_count = len(TPOT_TRIGGERS) - auto_count
            print(f"  • Auto-execute: {auto_count}")
            print(f"  • Manual approval: {manual_count}")

            print(f"\n{YELLOW}💡 Next Steps:{NC}")
            print(f"  1. Review triggers in workflow automation UI")
            print(f"  2. Adjust auto_execute settings as needed")
            print(f"  3. Test with simulated attacks")
            print(f"  4. Monitor trigger performance")
            print(f"\n{BLUE}{'='*70}{NC}\n")

    except Exception as e:
        error(f"Setup failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        await engine.dispose()

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))



