#!/usr/bin/env python3
"""Seed Mini-Corp production response workflows"""

import asyncio
import sys
from pathlib import Path

# Ensure backend module is importable when running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "backend"))

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import select

from app.models import WorkflowTrigger
from app.config import settings

BLUE = "\033[0;34m"
GREEN = "\033[0;32m"
YELLOW = "\033[1;33m"
RED = "\033[0;31m"
NC = "\033[0m"


def log(message: str) -> None:
    print(f"{BLUE}[INFO]{NC} {message}")


def success(message: str) -> None:
    print(f"{GREEN}✅ {message}{NC}")


def warning(message: str) -> None:
    print(f"{YELLOW}⚠️  {message}{NC}")


def error(message: str) -> None:
    print(f"{RED}❌ {message}{NC}")


MINI_CORP_TRIGGERS = [
    {
        "name": "Mini Corp: Ransomware Containment",
        "description": "Immediate containment of ransomware activity within production network",
        "category": "mini_corp",
        "enabled": True,
        "auto_execute": True,
        "priority": "critical",
        "conditions": {
            "threat_category": "ransomware",
            "escalation_level_min": "high",
            "risk_score_min": 0.55
        },
        "playbook_name": "Mini Corp Ransomware Response",
        "workflow_steps": [
            {
                "action_type": "block_ip_advanced",
                "parameters": {
                    "ip_address": "{source_ip}",
                    "duration": 604800,
                    "block_level": "aggressive"
                },
                "timeout_seconds": 45,
                "continue_on_failure": False
            },
            {
                "action_type": "isolate_host_advanced",
                "parameters": {
                    "host_identifier": "{source_ip}",
                    "isolation_level": "strict",
                    "monitoring": "packet-capture"
                },
                "timeout_seconds": 120,
                "continue_on_failure": False
            },
            {
                "action_type": "memory_dump_collection",
                "parameters": {
                    "target_hosts": ["{source_ip}"],
                    "dump_type": "full",
                    "retention": "30d"
                },
                "timeout_seconds": 300,
                "continue_on_failure": True
            },
            {
                "action_type": "deploy_firewall_rules",
                "parameters": {
                    "rule_set": [
                        {"action": "block", "ip": "{source_ip}", "protocol": "*"},
                        {"action": "monitor", "ip": "{source_ip}", "destination": "forensics-tap"}
                    ],
                    "scope": "edge",
                    "priority": "critical",
                    "expiration": 86400
                },
                "timeout_seconds": 90,
                "continue_on_failure": True
            },
            {
                "action_type": "reset_passwords",
                "parameters": {
                    "ip": "{source_ip}",
                    "reason": "Automated ransomware response"
                },
                "timeout_seconds": 120,
                "continue_on_failure": True
            },
            {
                "action_type": "invoke_ai_agent",
                "parameters": {
                    "agent": "containment",
                    "task": "emergency_isolation",
                    "context": "mini_corp_ransomware"
                },
                "timeout_seconds": 120,
                "continue_on_failure": True
            },
            {
                "action_type": "send_notification",
                "parameters": {
                    "channel": "slack",
                    "priority": "high",
                    "message": "🔴 Mini Corp: Automated ransomware containment executed for {source_ip}"
                },
                "timeout_seconds": 20,
                "continue_on_failure": True
            }
        ],
        "cooldown_seconds": 120,
        "max_triggers_per_day": 10,
        "tags": ["mini_corp", "ransomware", "critical", "auto"]
    },
    {
        "name": "Mini Corp: Data Exfiltration Response",
        "description": "Contain potential data exfiltration from production systems",
        "category": "mini_corp",
        "enabled": True,
        "auto_execute": True,
        "priority": "critical",
        "conditions": {
            "threat_category": "data_exfiltration",
            "escalation_level_min": "high",
            "risk_score_min": 0.6
        },
        "playbook_name": "Mini Corp Data Exfiltration Response",
        "workflow_steps": [
            {
                "action_type": "block_ip_advanced",
                "parameters": {
                    "ip_address": "{source_ip}",
                    "duration": 172800,
                    "block_level": "aggressive"
                },
                "timeout_seconds": 45,
                "continue_on_failure": False
            },
            {
                "action_type": "deploy_firewall_rules",
                "parameters": {
                    "rule_set": [
                        {"action": "block", "ip": "{source_ip}", "protocol": "*"},
                        {"action": "mirror", "ip": "{source_ip}", "destination": "forensics-tap"}
                    ],
                    "scope": "edge",
                    "priority": "critical",
                    "expiration": 43200
                },
                "timeout_seconds": 90,
                "continue_on_failure": True
            },
            {
                "action_type": "dns_sinkhole",
                "parameters": {
                    "domains": ["exfil-mini-corp.local", "{source_ip}"],
                    "sinkhole_ip": "10.50.50.50",
                    "ttl": 60,
                    "scope": "global"
                },
                "timeout_seconds": 45,
                "continue_on_failure": True
            },
            {
                "action_type": "memory_dump_collection",
                "parameters": {
                    "target_hosts": ["{source_ip}"],
                    "dump_type": "network-focused",
                    "retention": "14d"
                },
                "timeout_seconds": 240,
                "continue_on_failure": True
            },
            {
                "action_type": "capture_traffic",
                "parameters": {
                    "ip": "{source_ip}",
                    "reason": "Data exfiltration investigation"
                },
                "timeout_seconds": 120,
                "continue_on_failure": True
            },
            {
                "action_type": "invoke_ai_agent",
                "parameters": {
                    "agent": "forensics",
                    "task": "analyze_data_transfer",
                    "context": "mini_corp_data_exfiltration"
                },
                "timeout_seconds": 180,
                "continue_on_failure": True
            },
            {
                "action_type": "send_notification",
                "parameters": {
                    "channel": "slack",
                    "priority": "high",
                    "message": "🚨 Mini Corp: Data exfiltration safeguards deployed for {source_ip}"
                },
                "timeout_seconds": 20,
                "continue_on_failure": True
            }
        ],
        "cooldown_seconds": 180,
        "max_triggers_per_day": 10,
        "tags": ["mini_corp", "data_exfiltration", "critical", "auto"]
    },
    {
        "name": "Mini Corp: Privilege Escalation Investigation",
        "description": "Respond to suspected privilege escalation events",
        "category": "mini_corp",
        "enabled": True,
        "auto_execute": False,
        "priority": "high",
        "conditions": {
            "pattern_match": "privilege escalation",
            "risk_score_min": 0.55
        },
        "playbook_name": "Mini Corp Privilege Escalation",
        "workflow_steps": [
            {
                "action_type": "capture_traffic",
                "parameters": {
                    "ip": "{source_ip}",
                    "reason": "Privilege escalation investigation"
                },
                "timeout_seconds": 120,
                "continue_on_failure": True
            },
            {
                "action_type": "hunt_similar_attacks",
                "parameters": {
                    "ip": "{source_ip}",
                    "reason": "Search for linked privilege escalation attempts"
                },
                "timeout_seconds": 120,
                "continue_on_failure": True
            },
            {
                "action_type": "invoke_ai_agent",
                "parameters": {
                    "agent": "forensics",
                    "task": "analyze_privilege_escalation",
                    "context": "mini_corp_privilege_escalation"
                },
                "timeout_seconds": 180,
                "continue_on_failure": True
            },
            {
                "action_type": "send_notification",
                "parameters": {
                    "channel": "slack",
                    "priority": "medium",
                    "message": "⚠️ Mini Corp: Privilege escalation investigation workflow queued for {source_ip}"
                },
                "timeout_seconds": 20,
                "continue_on_failure": True
            }
        ],
        "cooldown_seconds": 300,
        "max_triggers_per_day": 8,
        "tags": ["mini_corp", "privilege_escalation", "investigation"]
    },
    {
        "name": "Mini Corp: Lateral Movement Containment",
        "description": "Contain suspected lateral movement inside the production network",
        "category": "mini_corp",
        "enabled": True,
        "auto_execute": True,
        "priority": "high",
        "conditions": {
            "pattern_match": "lateral movement",
            "risk_score_min": 0.5
        },
        "playbook_name": "Mini Corp Lateral Movement",
        "workflow_steps": [
            {
                "action_type": "block_ip_advanced",
                "parameters": {
                    "ip_address": "{source_ip}",
                    "duration": 86400,
                    "block_level": "aggressive"
                },
                "timeout_seconds": 45,
                "continue_on_failure": False
            },
            {
                "action_type": "isolate_host_advanced",
                "parameters": {
                    "host_identifier": "{source_ip}",
                    "isolation_level": "enhanced"
                },
                "timeout_seconds": 120,
                "continue_on_failure": False
            },
            {
                "action_type": "hunt_similar_attacks",
                "parameters": {
                    "ip": "{source_ip}",
                    "reason": "Lateral movement containment"
                },
                "timeout_seconds": 120,
                "continue_on_failure": True
            },
            {
                "action_type": "invoke_ai_agent",
                "parameters": {
                    "agent": "containment",
                    "task": "correlate_hosts",
                    "context": "mini_corp_lateral_movement"
                },
                "timeout_seconds": 150,
                "continue_on_failure": True
            },
            {
                "action_type": "send_notification",
                "parameters": {
                    "channel": "slack",
                    "priority": "high",
                    "message": "⚠️ Mini Corp: Lateral movement containment enacted for {source_ip}"
                },
                "timeout_seconds": 20,
                "continue_on_failure": True
            }
        ],
        "cooldown_seconds": 240,
        "max_triggers_per_day": 12,
        "tags": ["mini_corp", "lateral_movement", "auto"]
    },
    {
        "name": "Mini Corp: Web Application Attack",
        "description": "Respond to high-risk web application attacks detected on Mini Corp services",
        "category": "mini_corp",
        "enabled": True,
        "auto_execute": True,
        "priority": "high",
        "conditions": {
            "pattern_match": "sql injection",
            "risk_score_min": 0.5
        },
        "playbook_name": "Mini Corp Web Attack Response",
        "workflow_steps": [
            {
                "action_type": "deploy_waf_rules",
                "parameters": {
                    "ip": "{source_ip}",
                    "reason": "Mini Corp WAF hardening"
                },
                "timeout_seconds": 90,
                "continue_on_failure": True
            },
            {
                "action_type": "block_ip",
                "parameters": {
                    "ip_address": "{source_ip}",
                    "duration": 86400,
                    "block_level": "standard"
                },
                "timeout_seconds": 30,
                "continue_on_failure": False
            },
            {
                "action_type": "capture_traffic",
                "parameters": {
                    "ip": "{source_ip}",
                    "reason": "Collect malicious payloads"
                },
                "timeout_seconds": 90,
                "continue_on_failure": True
            },
            {
                "action_type": "invoke_ai_agent",
                "parameters": {
                    "agent": "forensics",
                    "task": "analyze_web_attack",
                    "context": "mini_corp_web_application"
                },
                "timeout_seconds": 150,
                "continue_on_failure": True
            },
            {
                "action_type": "send_notification",
                "parameters": {
                    "channel": "slack",
                    "priority": "medium",
                    "message": "🛡️ Mini Corp: Web application defense engaged for {source_ip}"
                },
                "timeout_seconds": 20,
                "continue_on_failure": True
            }
        ],
        "cooldown_seconds": 120,
        "max_triggers_per_day": 25,
        "tags": ["mini_corp", "web", "sql_injection"]
    },
    {
        "name": "Mini Corp: Credential Compromise",
        "description": "Automated response to suspected credential compromise or account takeover",
        "category": "mini_corp",
        "enabled": True,
        "auto_execute": False,
        "priority": "high",
        "conditions": {
            "pattern_match": "credential",
            "risk_score_min": 0.45
        },
        "playbook_name": "Mini Corp Credential Protection",
        "workflow_steps": [
            {
                "action_type": "reset_passwords",
                "parameters": {
                    "ip": "{source_ip}",
                    "reason": "Credential compromise mitigation"
                },
                "timeout_seconds": 120,
                "continue_on_failure": True
            },
            {
                "action_type": "threat_intel_lookup",
                "parameters": {
                    "ip": "{source_ip}",
                    "reason": "Assess reputation for credential compromise"
                },
                "timeout_seconds": 90,
                "continue_on_failure": True
            },
            {
                "action_type": "invoke_ai_agent",
                "parameters": {
                    "agent": "forensics",
                    "task": "analyze_account_activity",
                    "context": "mini_corp_credential_compromise"
                },
                "timeout_seconds": 150,
                "continue_on_failure": True
            },
            {
                "action_type": "send_notification",
                "parameters": {
                    "channel": "slack",
                    "priority": "medium",
                    "message": "🔐 Mini Corp: Credential compromise playbook executed for {source_ip}"
                },
                "timeout_seconds": 20,
                "continue_on_failure": True
            }
        ],
        "cooldown_seconds": 300,
        "max_triggers_per_day": 15,
        "tags": ["mini_corp", "credential", "account"]
    },
    {
        "name": "Mini Corp: DDoS Mitigation",
        "description": "Protect Mini Corp perimeter from distributed denial-of-service attacks",
        "category": "mini_corp",
        "enabled": True,
        "auto_execute": True,
        "priority": "critical",
        "conditions": {
            "threat_category": "ddos",
            "escalation_level_min": "high"
        },
        "playbook_name": "Mini Corp DDoS Mitigation",
        "workflow_steps": [
            {
                "action_type": "deploy_firewall_rules",
                "parameters": {
                    "rule_set": [
                        {"action": "rate_limit", "ip": "{source_ip}", "pps": 150},
                        {"action": "block", "ip": "{source_ip}", "protocol": "udp"}
                    ],
                    "scope": "edge",
                    "priority": "critical",
                    "expiration": 3600
                },
                "timeout_seconds": 90,
                "continue_on_failure": True
            },
            {
                "action_type": "block_ip",
                "parameters": {
                    "ip_address": "{source_ip}",
                    "duration": 43200,
                    "block_level": "aggressive"
                },
                "timeout_seconds": 30,
                "continue_on_failure": False
            },
            {
                "action_type": "invoke_ai_agent",
                "parameters": {
                    "agent": "containment",
                    "task": "enable_rate_limiting",
                    "context": "mini_corp_ddos"
                },
                "timeout_seconds": 120,
                "continue_on_failure": True
            },
            {
                "action_type": "send_notification",
                "parameters": {
                    "channel": "slack",
                    "priority": "high",
                    "message": "⚡ Mini Corp: DDoS mitigation controls deployed for {source_ip}"
                },
                "timeout_seconds": 20,
                "continue_on_failure": True
            }
        ],
        "cooldown_seconds": 180,
        "max_triggers_per_day": 8,
        "tags": ["mini_corp", "ddos", "auto"]
    },
]


async def create_or_update_triggers(session: AsyncSession):
    created = 0
    updated = 0
    skipped = 0

    for trigger_data in MINI_CORP_TRIGGERS:
        try:
            result = await session.execute(
                select(WorkflowTrigger).where(
                    WorkflowTrigger.name == trigger_data["name"]
                )
            )
            existing = result.scalar_one_or_none()

            if existing:
                log(f"Updating trigger: {trigger_data['name']}")
                for key, value in trigger_data.items():
                    if key != "name":
                        setattr(existing, key, value)
                updated += 1
            else:
                log(f"Creating trigger: {trigger_data['name']}")
                trigger = WorkflowTrigger(**trigger_data)
                session.add(trigger)
                created += 1

        except Exception as exc:  # pragma: no cover
            error(f"Error processing trigger '{trigger_data['name']}': {exc}")
            skipped += 1
            continue

    await session.commit()
    return created, updated, skipped


async def main() -> int:
    print(f"\n{BLUE}{'='*70}{NC}")
    print(f"{BLUE}Mini Corp Workflow Trigger Setup{NC}")
    print(f"{BLUE}{'='*70}{NC}\n")

    db_url = settings.database_url
    log(f"Database: {db_url}")

    engine = create_async_engine(db_url, echo=False)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    try:
        async with async_session() as session:
            log(f"Creating/updating {len(MINI_CORP_TRIGGERS)} Mini Corp workflow triggers...")
            created, updated, skipped = await create_or_update_triggers(session)

            print(f"\n{BLUE}{'='*70}{NC}")
            success("Setup complete!")
            print(f"\n{GREEN}📊 Summary:{NC}")
            print(f"  • Created:  {created} new triggers")
            print(f"  • Updated:  {updated} existing triggers")
            print(f"  • Skipped:  {skipped} triggers (errors)")
            print(f"  • Total:    {len(MINI_CORP_TRIGGERS)} triggers configured")

            print(f"\n{GREEN}🚀 Auto-Execute Status:{NC}")
            auto_count = sum(1 for t in MINI_CORP_TRIGGERS if t.get("auto_execute"))
            manual_count = len(MINI_CORP_TRIGGERS) - auto_count
            print(f"  • Auto-execute: {auto_count}")
            print(f"  • Manual approval: {manual_count}")

            print(f"\n{YELLOW}💡 Next Steps:{NC}")
            print("  1. Review Mini Corp triggers in the workflow automation UI")
            print("  2. Adjust auto_execute and priorities as needed")
            print("  3. Test with simulated Mini Corp attack scenarios")
            print("  4. Monitor trigger execution metrics")
            print(f"\n{BLUE}{'='*70}{NC}\n")

    except Exception as exc:  # pragma: no cover
        error(f"Setup failed: {exc}")
        return 1
    finally:
        await engine.dispose()

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
