#!/usr/bin/env python3
"""
Create Test Incidents for Mini-XDR Demo - Simple Version

This script creates realistic test incidents directly via SQL to demonstrate
the UI/UX functionality of the Mini-XDR system.
"""

import asyncio
import json
import sys
from datetime import datetime, timedelta, timezone

import asyncpg

# Database connection parameters
DB_CONFIG = {
    "host": "mini-xdr-postgres.ccnkck4wij3r.us-east-1.rds.amazonaws.com",
    "port": 5432,
    "database": "xdrdb",
    "user": "xdradmin",
    "password": "MiniXDR2025!Secure#Prod",
}


async def create_test_incidents():
    """Create realistic test incidents"""

    print("🔨 Creating test incidents for Mini-XDR...")

    # Connect to database
    conn = await asyncpg.connect(**DB_CONFIG)

    try:
        # Get current time
        now = datetime.now(timezone.utc)

        # Check existing incidents
        existing_count = await conn.fetchval("SELECT COUNT(*) FROM incidents")
        if existing_count > 0:
            print(
                f"⚠️  Warning: {existing_count} incident(s) already exist in database."
            )
            print("Proceeding will add more test incidents.\n")

        incidents_created = []

        # ========================================
        # Incident 1: Critical SSH Brute Force Attack
        # ========================================
        print("\n📋 Creating Incident 1: Critical SSH Brute Force Attack...")
        incident1_time = now - timedelta(hours=2)

        incident1_id = await conn.fetchval(
            """
            INSERT INTO incidents (
                src_ip, reason, status, auto_contained, created_at,
                escalation_level, risk_score, threat_category,
                containment_confidence, containment_method,
                agent_confidence, triage_note, ensemble_scores
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13
            ) RETURNING id
        """,
            "45.142.214.123",
            "SSH brute-force: 47 failed login attempts in 60s",
            "open",
            False,
            incident1_time,
            "high",
            0.85,
            "brute_force",
            0.82,
            "ai_agent",
            0.85,
            json.dumps(
                {
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
                }
            ),
            json.dumps(
                {
                    "ml_detector": 0.89,
                    "threat_intel": 0.98,
                    "behavioral_analysis": 0.76,
                    "anomaly_detection": 0.83,
                }
            ),
        )

        incidents_created.append(incident1_id)
        print(f"  ✅ Incident #{incident1_id} created")

        # Create events for incident 1
        print("  📊 Creating 15 associated events...")
        for i in range(15):
            event_time = incident1_time - timedelta(seconds=60 - (i * 4))
            await conn.execute(
                """
                INSERT INTO events (
                    src_ip, dst_ip, dst_port, eventid, message, ts,
                    source_type, hostname, anomaly_score, raw
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            """,
                "45.142.214.123",
                "10.0.1.50",
                22,
                "cowrie.login.failed",
                f"Failed SSH login attempt #{i+1}",
                event_time,
                "cowrie",
                "honeypot-01",
                0.85,
                json.dumps(
                    {
                        "username": ["admin", "root", "ubuntu", "test"][i % 4],
                        "password": "********",
                        "protocol": "SSH-2.0",
                        "auth_method": "password",
                    }
                ),
            )

        # ========================================
        # Incident 2: Medium-Severity Web Attack (SQL Injection)
        # ========================================
        print("\n📋 Creating Incident 2: Medium-Severity SQL Injection Attack...")
        incident2_time = now - timedelta(hours=4)

        incident2_id = await conn.fetchval(
            """
            INSERT INTO incidents (
                src_ip, reason, status, auto_contained, created_at,
                escalation_level, risk_score, threat_category,
                containment_confidence, containment_method,
                agent_confidence, triage_note, ensemble_scores
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13
            ) RETURNING id
        """,
            "103.252.200.45",
            "Web attack detected: SQL injection (12), Admin scan (8) (20 indicators in 10min)",
            "open",
            False,
            incident2_time,
            "medium",
            0.62,
            "web_attack",
            0.68,
            "ml_driven",
            0.70,
            json.dumps(
                {
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
                }
            ),
            json.dumps(
                {
                    "ml_detector": 0.64,
                    "threat_intel": 0.45,
                    "behavioral_analysis": 0.71,
                    "signature_match": 0.88,
                }
            ),
        )

        incidents_created.append(incident2_id)
        print(f"  ✅ Incident #{incident2_id} created")

        # ========================================
        # Incident 3: Critical Ransomware Detection
        # ========================================
        print("\n📋 Creating Incident 3: Critical Ransomware Detection...")
        incident3_time = now - timedelta(minutes=30)

        incident3_id = await conn.fetchval(
            """
            INSERT INTO incidents (
                src_ip, reason, status, auto_contained, created_at,
                escalation_level, risk_score, threat_category,
                containment_confidence, containment_method,
                agent_confidence, agent_id, triage_note,
                ensemble_scores, agent_actions
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15
            ) RETURNING id
        """,
            "10.0.5.42",
            "ML-detected anomaly: Ransomware-like behavior (confidence: 94%)",
            "contained",
            True,
            incident3_time,
            "critical",
            0.94,
            "malware",
            0.92,
            "ai_agent",
            0.94,
            "containment_orchestrator_v1",
            json.dumps(
                {
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
                }
            ),
            json.dumps(
                {
                    "ml_detector": 0.96,
                    "behavioral_analysis": 0.94,
                    "edr_signals": 0.91,
                    "file_integrity": 0.89,
                }
            ),
            json.dumps(
                [
                    {
                        "timestamp": (
                            incident3_time + timedelta(seconds=30)
                        ).isoformat(),
                        "action": "isolate_host",
                        "agent": "containment_orchestrator_v1",
                        "status": "success",
                        "details": "Host 10.0.5.42 isolated from network",
                    },
                    {
                        "timestamp": (
                            incident3_time + timedelta(seconds=45)
                        ).isoformat(),
                        "action": "snapshot_memory",
                        "agent": "forensics_agent_v1",
                        "status": "success",
                        "details": "Memory dump captured for forensic analysis",
                    },
                ]
            ),
        )

        incidents_created.append(incident3_id)
        print(f"  ✅ Incident #{incident3_id} created (AUTO-CONTAINED)")

        # Create action for incident 3
        print("  ⚡ Creating containment action...")
        await conn.execute(
            """
            INSERT INTO actions (
                incident_id, created_at, action, result, detail, params
            ) VALUES ($1, $2, $3, $4, $5, $6)
        """,
            incident3_id,
            incident3_time + timedelta(seconds=30),
            "isolate",
            "success",
            "Host 10.0.5.42 automatically isolated due to ransomware detection",
            json.dumps(
                {
                    "auto_contained": True,
                    "reason": "ransomware_detection",
                    "confidence": 0.94,
                }
            ),
        )

        # ========================================
        # Incident 4: Low-Severity Port Scan
        # ========================================
        print("\n📋 Creating Incident 4: Low-Severity Port Scan...")
        incident4_time = now - timedelta(hours=6)

        incident4_id = await conn.fetchval(
            """
            INSERT INTO incidents (
                src_ip, reason, status, auto_contained, created_at,
                escalation_level, risk_score, threat_category,
                containment_confidence, containment_method,
                agent_confidence, triage_note, ensemble_scores
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13
            ) RETURNING id
        """,
            "198.51.100.77",
            "Network reconnaissance: Port scan detected across 45 ports",
            "dismissed",
            False,
            incident4_time,
            "low",
            0.35,
            "reconnaissance",
            0.25,
            "rule_based",
            0.40,
            json.dumps(
                {
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
                }
            ),
            json.dumps(
                {"ml_detector": 0.42, "threat_intel": 0.15, "behavioral_analysis": 0.38}
            ),
        )

        incidents_created.append(incident4_id)
        print(f"  ✅ Incident #{incident4_id} created (DISMISSED)")

        # ========================================
        # Incident 5: Credential Stuffing Attack
        # ========================================
        print("\n📋 Creating Incident 5: Credential Stuffing Attack...")
        incident5_time = now - timedelta(hours=1)

        incident5_id = await conn.fetchval(
            """
            INSERT INTO incidents (
                src_ip, reason, status, auto_contained, created_at,
                escalation_level, risk_score, threat_category,
                containment_confidence, containment_method,
                agent_confidence, triage_note, ensemble_scores
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13
            ) RETURNING id
        """,
            "203.0.113.88",
            "Credential stuffing attack: 156 login attempts with known leaked credentials",
            "open",
            False,
            incident5_time,
            "high",
            0.79,
            "credential_stuffing",
            0.81,
            "ai_agent",
            0.79,
            json.dumps(
                {
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
                }
            ),
            json.dumps(
                {
                    "ml_detector": 0.82,
                    "threat_intel": 0.88,
                    "behavioral_analysis": 0.74,
                    "credential_analysis": 0.91,
                }
            ),
        )

        incidents_created.append(incident5_id)
        print(f"  ✅ Incident #{incident5_id} created")

        print("\n✅ Test incidents created successfully!")
        print("\n📈 Summary:")
        print(f"  • Total incidents created: {len(incidents_created)}")
        print(f"  • Incident IDs: {incidents_created}")
        print(f"  • Severity breakdown:")
        print(f"    - Critical: 1 (auto-contained)")
        print(f"    - High: 2")
        print(f"    - Medium: 1")
        print(f"    - Low: 1 (dismissed)")
        print("\n🌐 Access your Mini-XDR dashboard to view the incidents:")
        print(
            "  http://k8s-minixdr-minixdri-dc5fc1df8b-1132128475.us-east-1.elb.amazonaws.com"
        )

        return incidents_created

    finally:
        await conn.close()


async def main():
    """Main execution function"""
    try:
        await create_test_incidents()
        print("\n✨ Done! Test incidents are ready for demo.")

    except Exception as e:
        print(f"\n❌ Error creating test incidents: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
