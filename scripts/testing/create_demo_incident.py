#!/usr/bin/env python3
"""
Create Demo Incident - Direct Database Insert
Bypasses ML detection to showcase Phase 2 features
"""

import asyncio
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../backend"))

from app.db import AsyncSessionLocal
from app.models import Incident


async def create_demo_incident():
    """Create a demo incident with full Phase 2 metadata"""

    print("=" * 80)
    print("🎯 CREATING DEMO INCIDENT")
    print("=" * 80)
    print()

    # Incident data with Phase 2 enhancements
    incident_data = {
        "src_ip": "203.0.113.66",
        "reason": "Malware/Botnet C2 Communication (ML Confidence: 87.5%)",
        "status": "open",
        "escalation_level": "high",
        "risk_score": 0.875,
        "threat_category": "malware_botnet",
        # Phase 1: ML Detection
        "ml_confidence": 0.875,
        "containment_method": "ml_driven",
        "containment_confidence": 0.875,
        # Phase 1: Council Data
        "council_verdict": "THREAT",
        "council_reasoning": "High confidence malware detection. Multiple C2 beaconing patterns observed with 15-minute intervals. Lateral movement attempts to 10 internal hosts via SMB. Matches APT29 TTP patterns.",
        "council_confidence": 0.92,
        "routing_path": json.dumps(
            ["ml_predict", "council_verify", "gemini_judge", "threat_confirmed"]
        ),
        "api_calls_made": json.dumps(["gemini", "grok", "openai"]),
        "processing_time_ms": 1250,
        # Detailed AI analysis
        "gemini_analysis": json.dumps(
            {
                "reasoning": "Command and control communication pattern detected. Regular beaconing to known malicious domains every 15 minutes. High probability of successful compromise.",
                "confidence": 0.92,
                "threat_indicators": [
                    "c2_beaconing",
                    "lateral_movement",
                    "data_exfiltration",
                ],
                "recommended_actions": [
                    "isolate_host",
                    "block_c2_domains",
                    "forensic_analysis",
                ],
            }
        ),
        "grok_intel": json.dumps(
            {
                "threat_actor": "APT29 (Cozy Bear)",
                "campaign": "SolarWinds Supply Chain Attack",
                "ttps": [
                    "T1071.001 - Web Protocols",
                    "T1041 - Exfiltration Over C2",
                    "T1570 - Lateral Tool Transfer",
                ],
                "iocs_matched": 8,
                "threat_score": 9.2,
            }
        ),
        "openai_remediation": json.dumps(
            {
                "automated_response": [
                    "Block C2 domains: evil-c2-server-0.com, evil-c2-server-1.com, evil-c2-server-2.com",
                    "Isolate host 203.0.113.66 from network",
                    "Capture memory dump for forensic analysis",
                    "Reset credentials for accessed accounts",
                    "Deploy honeytokens on lateral movement targets",
                ],
                "manual_steps": [
                    "Review firewall logs for additional C2 traffic",
                    "Check EDR for process execution tree",
                    "Analyze PCAP for data exfiltration volumes",
                    "Coordinate with SOC for threat hunting",
                ],
            }
        ),
        # Phase 2: Enhanced triage note with advanced features
        "triage_note": json.dumps(
            {
                "summary": "Malware/Botnet C2 Communication detected from 203.0.113.66",
                "severity": "high",
                "confidence": 0.92,
                "anomaly_score": 0.875,
                "threat_class": 5,
                "event_count": 50,
                "recommendation": "Immediate containment and forensic analysis required. APT-level threat actor identified.",
                "rationale": [
                    "ML model classified with 87.5% confidence as Malware/Botnet",
                    "Council verification increased confidence to 92%",
                    "Phase 2 advanced features (100D) extracted successfully",
                    "Attribution: APT29 (Cozy Bear) with high confidence",
                    "Containment actions automatically executed with 92% effectiveness",
                ],
                "indicators": {
                    "enhanced_model_prediction": {
                        "class_probabilities": {
                            "normal": 0.01,
                            "ddos": 0.02,
                            "reconnaissance": 0.03,
                            "brute_force": 0.02,
                            "web_attack": 0.05,
                            "malware_botnet": 0.875,
                            "apt": 0.005,
                        },
                        "uncertainty_score": 0.125,
                        "explanation": "High C2 beaconing frequency with lateral movement patterns",
                        "feature_importance": {
                            "c2_beaconing_frequency": 0.32,
                            "lateral_movement_attempts": 0.28,
                            "data_exfil_volume": 0.18,
                            "unique_c2_domains": 0.12,
                            "smb_connection_attempts": 0.10,
                        },
                        "openai_enhanced": False,
                    },
                    "phase2_advanced_features": {
                        "feature_count": 100,
                        "features_extracted": True,
                        "feature_dimensions": "100D (79 base + 21 advanced)",
                        "threat_intel_score": 0.92,
                        "behavioral_anomaly": 0.88,
                        "network_graph_centrality": 0.75,
                    },
                },
                "council_verified": True,
                "council_verdict": "THREAT",
                "agents": {
                    "attribution": {
                        "threat_actor": "APT29",
                        "confidence": 0.78,
                        "tactics": [
                            "initial_access",
                            "lateral_movement",
                            "exfiltration",
                        ],
                        "techniques": ["T1071.001", "T1041", "T1570"],
                        "iocs_identified": 12,
                    },
                    "containment": {
                        "actions_taken": [
                            "isolate_host",
                            "block_c2_domains",
                            "firewall_rule_update",
                        ],
                        "effectiveness": 0.92,
                        "status": "active",
                        "systems_isolated": 1,
                    },
                    "forensics": {
                        "evidence_collected": [
                            "memory_dump",
                            "network_pcap",
                            "process_tree",
                        ],
                        "timeline_events": 45,
                        "suspicious_processes": [
                            "svchost.exe",
                            "rundll32.exe",
                            "powershell.exe",
                        ],
                        "files_analyzed": 8,
                    },
                    "deception": {
                        "honeytokens_deployed": 3,
                        "attacker_interactions": 0,
                        "intelligence_gathered": [],
                    },
                },
            }
        ),
    }

    async with AsyncSessionLocal() as db:
        print("💾 Creating incident in database...")
        incident = Incident(**incident_data)
        db.add(incident)
        await db.commit()
        await db.refresh(incident)

        print("✅ Demo incident created successfully!")
        print()
        print("=" * 80)
        print("📊 INCIDENT DETAILS")
        print("=" * 80)
        print()
        print(f"🆔 Incident ID: #{incident.id}")
        print(f"🌐 Source IP: {incident.src_ip}")
        print(f"⚠️  Severity: {incident.escalation_level.upper()}")
        print(f"📈 ML Confidence: {incident.ml_confidence:.1%}")
        print(f"🏛️  Council Verdict: {incident.council_verdict}")
        print(f"🤖 Council Confidence: {incident.council_confidence:.1%}")
        print()
        print("🎨 Phase 2 Features Included:")
        print("  ✅ 100-dimensional advanced features")
        print("  ✅ Feature store caching metadata")
        print("  ✅ Council of Models analysis")
        print("  ✅ AI Agent coordination")
        print("  ✅ Attribution (APT29)")
        print("  ✅ Containment actions")
        print("  ✅ Forensics evidence")
        print("  ✅ Deception intelligence")
        print()
        print("=" * 80)
        print()
        print(f"🌐 View in Browser:")
        print(f"   Dashboard: http://localhost:3000")
        print(
            f"   Incident Details: http://localhost:3000/incidents/incident/{incident.id}"
        )
        print()
        print("=" * 80)


if __name__ == "__main__":
    asyncio.run(create_demo_incident())
