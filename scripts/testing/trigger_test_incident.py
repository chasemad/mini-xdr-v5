#!/usr/bin/env python3
"""
Trigger Test Incident Script
Creates synthetic events to trigger ML models → Council → Agents → Incident creation
"""

import asyncio
import os
import sys
from datetime import datetime, timezone

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../backend"))

from app.db import AsyncSessionLocal
from app.intelligent_detection import intelligent_detector
from app.models import Event
from sqlalchemy.ext.asyncio import AsyncSession


async def create_test_events():
    """Create synthetic malware/botnet events"""

    test_ip = "203.0.113.50"  # Test IP (RFC 5737)

    events = []
    base_time = datetime.now(timezone.utc)

    # Create 25 suspicious events simulating C2 beaconing
    for i in range(25):
        event = Event(
            ts=base_time,
            src_ip=test_ip,
            dst_ip="8.8.8.8",  # Suspicious DNS queries
            dst_port=53,  # DNS
            eventid=f"cowrie.dns.query.{i}",
            message=f"Suspicious DNS query to known C2 domain: evil-c2-server-{i % 3}.com",
            source_type="cowrie",
            raw={"query": f"evil-c2-server-{i % 3}.com", "type": "dns"},
        )
        events.append(event)

    # Add some HTTP POST requests (data exfiltration pattern)
    for i in range(15):
        event = Event(
            ts=base_time,
            src_ip=test_ip,
            dst_ip="93.184.216.34",  # Example.com IP
            dst_port=443,
            eventid=f"cowrie.http.post.{i}",
            message=f"HTTP POST to suspicious domain: data-exfil-{i % 2}.net - possible data exfiltration",
            source_type="cowrie",
            raw={
                "url": f"https://data-exfil-{i % 2}.net/upload",
                "method": "POST",
                "size": 1024 * (i + 1),
            },
        )
        events.append(event)

    # Add some lateral movement attempts
    for i in range(10):
        event = Event(
            ts=base_time,
            src_ip=test_ip,
            dst_ip=f"192.168.1.{10 + i}",
            dst_port=445,  # SMB
            eventid=f"cowrie.smb.connect.{i}",
            message=f"SMB connection attempt to internal host 192.168.1.{10 + i} - lateral movement detected",
            source_type="cowrie",
            anomaly_score=0.8 + (i * 0.01),
            raw={
                "dest_host": f"192.168.1.{10 + i}",
                "protocol": "smb",
                "action": "connect",
            },
        )
        events.append(event)

    return events


async def trigger_incident():
    """Main function to trigger incident creation"""

    print("=" * 80)
    print("🧪 TEST INCIDENT TRIGGER")
    print("=" * 80)
    print()

    print("📝 Creating synthetic malware/botnet events...")
    events = await create_test_events()
    print(f"✅ Created {len(events)} test events")
    print()

    # Get database session
    async with AsyncSessionLocal() as db:
        # Save events to database
        print("💾 Saving events to database...")
        for event in events:
            db.add(event)
        await db.commit()
        print("✅ Events saved")
        print()

        # Trigger intelligent detection
        print("🤖 Triggering Intelligent Detection Engine...")
        print("   This will:")
        print("   1. Extract features (79D base + 21D advanced = 100D)")
        print("   2. Check Feature Store cache")
        print("   3. Run ML classification (3-model ensemble)")
        print(
            "   4. Apply Phase 2 enhancements (temperature scaling, per-class thresholds)"
        )
        print("   5. Route through Council if confidence 50-90%")
        print("   6. Collect training sample")
        print("   7. Trigger agent coordination")
        print("   8. Create incident")
        print()

        test_ip = events[0].src_ip
        result = await intelligent_detector.analyze_and_create_incidents(
            db=db, src_ip=test_ip, events=events
        )

        print("=" * 80)
        print("📊 DETECTION RESULTS")
        print("=" * 80)
        print()

        if result.get("incident_created"):
            incident_id = result.get("incident_id")
            print(f"✅ Incident Created: #{incident_id}")
            print()
            print(f"🎯 Threat Type: {result.get('threat_type', 'Unknown')}")
            print(f"📈 ML Confidence: {result.get('confidence', 0):.1%}")
            print(f"⚠️  Severity: {result.get('severity', 'Unknown')}")
            print()

            # Check if Council was involved
            classification = result.get("classification", {})
            if "council_verified" in str(classification):
                print("🏛️  Council Verification: ✅ VERIFIED")

            print()
            print(
                f"🌐 View Incident: http://localhost:3000/incidents/incident/{incident_id}"
            )
            print()
        else:
            print(f"❌ No Incident Created")
            print(f"   Reason: {result.get('reason', 'Unknown')}")
            print()

        print("=" * 80)
        print()
        print("Next Steps:")
        print("1. Open http://localhost:3000 to see the dashboard")
        print("2. Click on the new incident to see the 6-tab analysis")
        print("3. Review Council verdict, Agent decisions, and Phase 2 features")
        print()


if __name__ == "__main__":
    asyncio.run(trigger_incident())
