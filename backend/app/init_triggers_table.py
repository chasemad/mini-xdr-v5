"""
Initialize workflow_triggers table and seed default honeypot triggers
"""
import asyncio
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text
from app.models import Base, WorkflowTrigger
from app.config import settings

async def init_database():
    """Create workflow_triggers table if it doesn't exist"""
    engine = create_async_engine(settings.database_url, echo=True)

    async with engine.begin() as conn:
        # Create all tables (will skip existing ones)
        await conn.run_sync(Base.metadata.create_all)

    print("✓ Database tables initialized successfully")

    # Create session for seeding
    async_session = sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )

    async with async_session() as session:
        # Check if triggers already exist
        result = await session.execute(text("SELECT COUNT(*) FROM workflow_triggers"))
        count = result.scalar()

        if count > 0:
            print(f"✓ Found {count} existing triggers in database")
            return

        print("Seeding default honeypot workflow triggers...")

        # Default Trigger 1: SSH Brute Force Attack
        ssh_trigger = WorkflowTrigger(
            name="SSH Brute Force Detection",
            description="Automatically respond to SSH brute force attacks from honeypot",
            category="honeypot",
            enabled=True,
            auto_execute=True,
            priority="high",
            conditions={
                "event_type": "cowrie.login.failed",
                "threshold": 6,
                "window_seconds": 60,
                "source": "honeypot"
            },
            playbook_name="SSH Brute Force Response",
            workflow_steps=[
                {
                    "action_type": "block_ip",
                    "parameters": {
                        "ip_address": "event.source_ip",
                        "duration": 3600,
                        "block_level": "standard"
                    },
                    "timeout_seconds": 30,
                    "continue_on_failure": False
                },
                {
                    "action_type": "create_incident",
                    "parameters": {
                        "title": "SSH Brute Force Attack",
                        "severity": "high",
                        "description": "Multiple failed SSH login attempts detected from {source_ip}"
                    },
                    "timeout_seconds": 10,
                    "continue_on_failure": False
                },
                {
                    "action_type": "invoke_ai_agent",
                    "parameters": {
                        "agent": "attribution",
                        "task": "analyze_threat_actor",
                        "context": "ssh_brute_force"
                    },
                    "timeout_seconds": 60,
                    "continue_on_failure": True
                },
                {
                    "action_type": "send_notification",
                    "parameters": {
                        "channel": "slack",
                        "message": "🚨 SSH brute force attack blocked from {source_ip}"
                    },
                    "timeout_seconds": 10,
                    "continue_on_failure": True
                }
            ],
            cooldown_seconds=60,
            max_triggers_per_day=100,
            tags=["honeypot", "ssh", "brute-force", "auto-block"]
        )

        # Default Trigger 2: SQL Injection Attempt
        sql_trigger = WorkflowTrigger(
            name="SQL Injection Detection",
            description="Detect and respond to SQL injection attempts",
            category="honeypot",
            enabled=True,
            auto_execute=False,  # Requires approval
            priority="high",
            conditions={
                "event_type": "webhoneypot.request",
                "pattern_match": "(union|select|insert|update|delete|drop).*from",
                "source": "honeypot"
            },
            playbook_name="SQL Injection Response",
            workflow_steps=[
                {
                    "action_type": "analyze_payload",
                    "parameters": {"payload_field": "request.body", "analysis_type": "sql_injection"},
                    "timeout_seconds": 30,
                    "continue_on_failure": False
                },
                {
                    "action_type": "create_incident",
                    "parameters": {
                        "title": "SQL Injection Attempt",
                        "severity": "high",
                        "description": "SQL injection pattern detected from {source_ip}"
                    },
                    "timeout_seconds": 10,
                    "continue_on_failure": False
                },
                {
                    "action_type": "invoke_ai_agent",
                    "parameters": {
                        "agent": "forensics",
                        "task": "analyze_attack_pattern",
                        "context": "sql_injection"
                    },
                    "timeout_seconds": 90,
                    "continue_on_failure": True
                },
                {
                    "action_type": "block_ip",
                    "parameters": {"source": "event.source_ip", "duration_seconds": 7200},
                    "timeout_seconds": 30,
                    "continue_on_failure": False
                }
            ],
            cooldown_seconds=120,
            max_triggers_per_day=50,
            tags=["honeypot", "web", "sql-injection", "manual-approval"]
        )

        # Default Trigger 3: Malware Payload Detection
        malware_trigger = WorkflowTrigger(
            name="Malware Payload Detection",
            description="Automatically contain and analyze malware uploads",
            category="honeypot",
            enabled=True,
            auto_execute=True,
            priority="critical",
            conditions={
                "event_type": "file.upload.detected",
                "risk_score_min": 0.8,
                "source": "honeypot"
            },
            playbook_name="Malware Containment",
            workflow_steps=[
                {
                    "action_type": "isolate_file",
                    "parameters": {"file_path": "event.file_path", "quarantine": True},
                    "timeout_seconds": 30,
                    "continue_on_failure": False
                },
                {
                    "action_type": "block_ip",
                    "parameters": {"source": "event.source_ip", "duration_seconds": 86400},
                    "timeout_seconds": 30,
                    "continue_on_failure": False
                },
                {
                    "action_type": "create_incident",
                    "parameters": {
                        "title": "Malware Payload Detected",
                        "severity": "critical",
                        "description": "High-confidence malware detected from {source_ip}"
                    },
                    "timeout_seconds": 10,
                    "continue_on_failure": False
                },
                {
                    "action_type": "invoke_ai_agent",
                    "parameters": {
                        "agent": "containment",
                        "task": "full_isolation",
                        "context": "malware_detected"
                    },
                    "timeout_seconds": 120,
                    "continue_on_failure": True
                }
            ],
            cooldown_seconds=30,
            max_triggers_per_day=200,
            tags=["honeypot", "malware", "auto-containment", "critical"]
        )

        # Add all triggers to session
        session.add(ssh_trigger)
        session.add(sql_trigger)
        session.add(malware_trigger)

        await session.commit()

        print("✓ Seeded 3 default honeypot workflow triggers:")
        print("  1. SSH Brute Force Detection (auto-execute)")
        print("  2. SQL Injection Detection (requires approval)")
        print("  3. Malware Payload Detection (auto-execute)")

    await engine.dispose()

if __name__ == "__main__":
    asyncio.run(init_database())