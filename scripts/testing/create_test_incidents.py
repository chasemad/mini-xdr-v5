#!/usr/bin/env python3
"""
Create Test Incidents for Mini-XDR Demo

This script creates realistic test incidents to demonstrate the UI/UX
functionality of the Mini-XDR system.
"""

import asyncio
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Add backend to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "backend"))

from app.db import Base
from app.models import Action, Event, Incident
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

# Database URL - use environment variable or default
DATABASE_URL = "postgresql+asyncpg://xdr_user:xdr_secure_pass_2024@mini-xdr-db.c8iy4xqknzvo.us-east-1.rds.amazonaws.com:5432/mini_xdr_db"


async def create_test_incidents(db: AsyncSession):
    """Create realistic test incidents"""

    print("🔨 Creating test incidents for Mini-XDR...")

    # Get current time
    now = datetime.now(timezone.utc)

    incidents_data = []
    events_data = []
    actions_data = []

    # ========================================
    # Incident 1: Critical SSH Brute Force Attack
    # ========================================
    incident1_time = now - timedelta(hours=2)
    incident1_data = {
        "src_ip": "45.142.214.123",
        "reason": "SSH brute-force: 47 failed login attempts in 60s",
        "status": "open",
        "auto_contained": False,
        "created_at": incident1_time,
        "escalation_level": "high",
        "risk_score": 0.85,
        "threat_category": "brute_force",
        "containment_confidence": 0.82,
        "containment_method": "ai_agent",
        "agent_confidence": 0.85,
        "triage_note": {
            "summary": "Aggressive SSH brute-force attack detected from known malicious IP",
            "severity": "high",
            "recommendation": "Immediate containment recommended. IP is on multiple threat intelligence lists.",
            "rationale": [
                "47 failed login attempts in 60 seconds exceeds threshold by 7.8x",
                "Source IP matches AbuseIPDB blacklist (confidence: 98%)",
                "Geographic anomaly: Connection from Russia targeting US infrastructure",
                "Attack pattern matches known botnet behavior (Mirai variant)",
                "Multiple SSH protocol violations detected",
            ],
        },
        "ensemble_scores": {
            "ml_detector": 0.89,
            "threat_intel": 0.98,
            "behavioral_analysis": 0.76,
            "anomaly_detection": 0.83,
        },
    }
    incidents_data.append(incident1_data)

    # Create associated events for incident 1
    for i in range(15):
        event_time = incident1_time - timedelta(seconds=60 - (i * 4))
        events_data.append(
            {
                "src_ip": "45.142.214.123",
                "dst_ip": "10.0.1.50",
                "dst_port": 22,
                "eventid": "cowrie.login.failed",
                "message": f"Failed SSH login attempt #{i+1}",
                "ts": event_time,
                "source_type": "cowrie",
                "hostname": "honeypot-01",
                "anomaly_score": 0.85,
                "raw": {
                    "username": ["admin", "root", "ubuntu", "test"][i % 4],
                    "password": "********",
                    "protocol": "SSH-2.0",
                    "auth_method": "password",
                },
            }
        )

    # ========================================
    # Incident 2: Medium-Severity Web Attack (SQL Injection)
    # ========================================
    incident2_time = now - timedelta(hours=4)
    incident2_data = {
        "src_ip": "103.252.200.45",
        "reason": "Web attack detected: SQL injection (12), Admin scan (8) (20 indicators in 10min)",
        "status": "open",
        "auto_contained": False,
        "created_at": incident2_time,
        "escalation_level": "medium",
        "risk_score": 0.62,
        "threat_category": "web_attack",
        "containment_confidence": 0.68,
        "containment_method": "ml_driven",
        "agent_confidence": 0.70,
        "triage_note": {
            "summary": "Automated web vulnerability scanner detected targeting application endpoints",
            "severity": "medium",
            "recommendation": "Monitor and prepare for containment. Verify WAF rules are active.",
            "rationale": [
                "Multiple SQL injection patterns detected in HTTP parameters",
                "Admin panel scanning activity observed",
                "User-Agent indicates automated scanning tool (sqlmap)",
                "Attack sophistication level: Medium",
                "No successful exploitation detected yet",
            ],
        },
        "ensemble_scores": {
            "ml_detector": 0.64,
            "threat_intel": 0.45,
            "behavioral_analysis": 0.71,
            "signature_match": 0.88,
        },
    }
    incidents_data.append(incident2_data)

    # Create associated events for incident 2
    sql_payloads = ["' OR '1'='1", "admin' --", "1' UNION SELECT NULL--", "' AND 1=1--"]
    for i in range(20):
        event_time = incident2_time - timedelta(minutes=10 - (i * 0.5))
        events_data.append(
            {
                "src_ip": "103.252.200.45",
                "dst_ip": "10.0.2.100",
                "dst_port": 80,
                "eventid": "web.sql_injection",
                "message": f"SQL injection attempt detected in parameter 'id'",
                "ts": event_time,
                "source_type": "suricata",
                "hostname": "web-server-01",
                "anomaly_score": 0.72,
                "raw": {
                    "method": "GET",
                    "uri": f"/api/user?id={sql_payloads[i % 4]}",
                    "user_agent": "sqlmap/1.7.2#stable",
                    "payload_type": "sql_injection",
                },
            }
        )

    # ========================================
    # Incident 3: Critical Ransomware Detection
    # ========================================
    incident3_time = now - timedelta(minutes=30)
    incident3_data = {
        "src_ip": "10.0.5.42",
        "reason": "ML-detected anomaly: Ransomware-like behavior (confidence: 94%)",
        "status": "contained",
        "auto_contained": True,
        "created_at": incident3_time,
        "escalation_level": "critical",
        "risk_score": 0.94,
        "threat_category": "malware",
        "containment_confidence": 0.92,
        "containment_method": "ai_agent",
        "agent_confidence": 0.94,
        "agent_id": "containment_orchestrator_v1",
        "triage_note": {
            "summary": "Critical: AI detected ransomware-like file encryption behavior. Host isolated.",
            "severity": "critical",
            "recommendation": "Host has been automatically isolated. Initiate incident response playbook immediately.",
            "rationale": [
                "Rapid file modification detected: 1,247 files in 3 minutes",
                "File extensions changed to .locked pattern",
                "Shadow copy deletion commands executed",
                "Process spawned from suspicious parent (phishing email attachment)",
                "Lateral movement attempts blocked",
                "AI confidence: 94% (above auto-containment threshold)",
            ],
        },
        "ensemble_scores": {
            "ml_detector": 0.96,
            "behavioral_analysis": 0.94,
            "edr_signals": 0.91,
            "file_integrity": 0.89,
        },
        "agent_actions": [
            {
                "timestamp": (incident3_time + timedelta(seconds=30)).isoformat(),
                "action": "isolate_host",
                "agent": "containment_orchestrator_v1",
                "status": "success",
                "details": "Host 10.0.5.42 isolated from network",
            },
            {
                "timestamp": (incident3_time + timedelta(seconds=45)).isoformat(),
                "action": "snapshot_memory",
                "agent": "forensics_agent_v1",
                "status": "success",
                "details": "Memory dump captured for forensic analysis",
            },
        ],
    }
    incidents_data.append(incident3_data)

    # Create associated action for incident 3
    actions_data.append(
        {
            "created_at": incident3_time + timedelta(seconds=30),
            "action": "isolate",
            "result": "success",
            "detail": "Host 10.0.5.42 automatically isolated due to ransomware detection",
            "params": {
                "auto_contained": True,
                "reason": "ransomware_detection",
                "confidence": 0.94,
            },
        }
    )

    # ========================================
    # Incident 4: Low-Severity Port Scan
    # ========================================
    incident4_time = now - timedelta(hours=6)
    incident4_data = {
        "src_ip": "198.51.100.77",
        "reason": "Network reconnaissance: Port scan detected across 45 ports",
        "status": "dismissed",
        "auto_contained": False,
        "created_at": incident4_time,
        "escalation_level": "low",
        "risk_score": 0.35,
        "threat_category": "reconnaissance",
        "containment_confidence": 0.25,
        "containment_method": "rule_based",
        "agent_confidence": 0.40,
        "triage_note": {
            "summary": "Low-priority port scan from research network. Likely legitimate security testing.",
            "severity": "low",
            "recommendation": "Monitor only. IP belongs to university research network.",
            "rationale": [
                "Sequential port scanning detected (ports 1-1000)",
                "Scan speed: Low intensity (2 ports/second)",
                "Source IP belongs to .edu domain",
                "No exploit attempts following scan",
                "WHOIS indicates security research organization",
            ],
        },
        "ensemble_scores": {
            "ml_detector": 0.42,
            "threat_intel": 0.15,
            "behavioral_analysis": 0.38,
        },
    }
    incidents_data.append(incident4_data)

    # ========================================
    # Incident 5: Credential Stuffing Attack
    # ========================================
    incident5_time = now - timedelta(hours=1)
    incident5_data = {
        "src_ip": "203.0.113.88",
        "reason": "Credential stuffing attack: 156 login attempts with known leaked credentials",
        "status": "open",
        "auto_contained": False,
        "created_at": incident5_time,
        "escalation_level": "high",
        "risk_score": 0.79,
        "threat_category": "credential_stuffing",
        "containment_confidence": 0.81,
        "containment_method": "ai_agent",
        "agent_confidence": 0.79,
        "triage_note": {
            "summary": "Credential stuffing attack using leaked database from recent breach",
            "severity": "high",
            "recommendation": "Block IP and reset passwords for targeted accounts. Enable MFA.",
            "rationale": [
                "156 login attempts across 89 unique accounts",
                "Credentials match patterns from recent data breach (RockYou2024)",
                "Geographic anomaly: Multiple countries within 5 minutes",
                "Using residential proxy network to evade detection",
                "2 successful authentications detected before blocking",
            ],
        },
        "ensemble_scores": {
            "ml_detector": 0.82,
            "threat_intel": 0.88,
            "behavioral_analysis": 0.74,
            "credential_analysis": 0.91,
        },
    }
    incidents_data.append(incident5_data)

    # Create incidents in database
    created_incidents = []
    print(f"\n📋 Creating {len(incidents_data)} test incidents...")
    for i, inc_data in enumerate(incidents_data, 1):
        incident = Incident(**inc_data)
        db.add(incident)
        await db.flush()  # Get the ID
        created_incidents.append(incident)
        print(f"  ✅ Incident #{incident.id}: {inc_data['reason'][:60]}...")

    # Create events
    print(f"\n📊 Creating {len(events_data)} test events...")
    for event_data in events_data:
        event = Event(**event_data)
        db.add(event)

    # Link actions to incidents
    if actions_data:
        print(f"\n⚡ Creating {len(actions_data)} test actions...")
        for action_data in actions_data:
            action_data["incident_id"] = created_incidents[
                2
            ].id  # Link to ransomware incident
            action = Action(**action_data)
            db.add(action)

    # Commit all changes
    await db.commit()

    print("\n✅ Test incidents created successfully!")
    print("\n📈 Summary:")
    print(f"  • Incidents: {len(incidents_data)}")
    print(f"  • Events: {len(events_data)}")
    print(f"  • Actions: {len(actions_data)}")
    print("\n🌐 Access your Mini-XDR dashboard to view the incidents:")
    print(
        "  http://k8s-minixdr-minixdri-dc5fc1df8b-1132128475.us-east-1.elb.amazonaws.com"
    )

    return created_incidents


async def main():
    """Main execution function"""
    try:
        # Create async engine
        engine = create_async_engine(DATABASE_URL, echo=False)

        # Create async session
        async_session = sessionmaker(
            engine, class_=AsyncSession, expire_on_commit=False
        )

        async with async_session() as session:
            # Check if incidents already exist
            result = await session.execute(select(Incident))
            existing = result.scalars().all()

            if existing:
                print(
                    f"⚠️  Warning: {len(existing)} incident(s) already exist in database."
                )
                print("Proceeding will add more test incidents.")
                response = input("Continue? (y/N): ")
                if response.lower() != "y":
                    print("Aborted.")
                    return

            # Create test incidents
            incidents = await create_test_incidents(session)

            print("\n✨ Done! Test incidents are ready for demo.")

    except Exception as e:
        print(f"\n❌ Error creating test incidents: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
