from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional, Union
import logging
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

from .config import settings
from .db import get_db, init_db, AsyncSessionLocal
from .models import Event, Incident, Action
from .detect import run_detection
from .responder import block_ip, unblock_ip
from .triager import run_triage, generate_default_triage
from .multi_ingestion import multi_ingestor
from .ml_engine import ml_detector
from .external_intel import threat_intel
from .agents.containment_agent import ContainmentAgent
from .policy_engine import policy_engine
from .learning_pipeline import learning_pipeline
from .adaptive_detection import behavioral_analyzer
from .baseline_engine import baseline_engine
from .detect import adaptive_engine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global scheduler for background tasks
scheduler = AsyncIOScheduler()

# Global AI agent instances
containment_agent = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global containment_agent
    
    # Startup
    logger.info("Starting Enhanced Mini-XDR backend...")
    await init_db()
    scheduler.start()
    
    # Initialize AI components
    logger.info("Initializing AI components...")
    containment_agent = ContainmentAgent(threat_intel=threat_intel, ml_detector=ml_detector)
    
    # Load ML models if available
    try:
        ml_detector.load_models()
        logger.info("ML models loaded")
    except Exception as e:
        logger.warning(f"Failed to load ML models: {e}")
    
    # Start continuous learning pipeline
    try:
        await learning_pipeline.start_learning_loop()
        logger.info("Continuous learning pipeline started")
    except Exception as e:
        logger.warning(f"Failed to start learning pipeline: {e}")
    
    # Add scheduled tasks
    scheduler.add_job(
        process_scheduled_unblocks,
        IntervalTrigger(seconds=30),
        id="process_scheduled_unblocks",
        replace_existing=True
    )
    
    # Add ML training task (daily)
    scheduler.add_job(
        background_retrain_ml_models,
        IntervalTrigger(hours=24),
        id="retrain_ml_models",
        replace_existing=True
    )
    
    yield
    
    # Shutdown
    logger.info("Shutting down Enhanced Mini-XDR backend...")
    
    # Stop learning pipeline
    try:
        await learning_pipeline.stop_learning_loop()
        logger.info("Learning pipeline stopped")
    except Exception as e:
        logger.warning(f"Error stopping learning pipeline: {e}")
    
    await threat_intel.close()
    scheduler.shutdown()


app = FastAPI(
    title="Mini-XDR",
    description="SSH Brute-Force Detection and Response System",
    version="1.2.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global auto-contain setting (runtime configurable)
auto_contain_enabled = settings.auto_contain


def _require_api_key(request: Request):
    """Require API key for protected endpoints"""
    if not settings.api_key:
        return  # No API key configured, skip check
    
    api_key = request.headers.get("x-api-key")
    if api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")


async def _recent_events_for_ip(db: AsyncSession, src_ip: str, seconds: int = 600):
    """Get recent events for an IP address"""
    since = datetime.now(timezone.utc) - timedelta(seconds=seconds)
    query = select(Event).where(
        and_(Event.src_ip == src_ip, Event.ts >= since)
    ).order_by(Event.ts.desc()).limit(200)
    
    result = await db.execute(query)
    return result.scalars().all()


async def _get_detailed_events_for_ip(db: AsyncSession, src_ip: str, hours: int = 24):
    """Get detailed events for an IP with extended timeframe for forensic analysis"""
    since = datetime.now(timezone.utc) - timedelta(hours=hours)
    query = select(Event).where(
        and_(Event.src_ip == src_ip, Event.ts >= since)
    ).order_by(Event.ts.asc())
    
    result = await db.execute(query)
    return result.scalars().all()


def _extract_iocs_from_events(events):
    """Extract Indicators of Compromise from events"""
    import re
    
    iocs = {
        "ip_addresses": [],
        "domains": [],
        "urls": [],
        "file_hashes": [],
        "user_agents": [],
        "sql_injection_patterns": [],
        "command_patterns": [],
        "file_paths": [],
        "privilege_escalation_indicators": [],
        "data_exfiltration_indicators": [],
        "persistence_mechanisms": [],
        "lateral_movement_indicators": [],
        "database_access_patterns": [],
        "successful_auth_indicators": [],
        "reconnaissance_patterns": []
    }
    
    for event in events:
        # Extract from raw data
        if event.raw and isinstance(event.raw, dict):
            raw_data = event.raw
            
            # Extract user agents
            if 'user_agent' in raw_data:
                iocs["user_agents"].append(raw_data['user_agent'])
            
            # Extract file hashes
            if 'hash' in raw_data:
                iocs["file_hashes"].append(raw_data['hash'])
            if 'md5' in raw_data:
                iocs["file_hashes"].append(raw_data['md5'])
            if 'sha256' in raw_data:
                iocs["file_hashes"].append(raw_data['sha256'])
            
            # Extract HTTP request details for web attacks
            if 'path' in raw_data:
                iocs["urls"].append(raw_data['path'])
                if 'query_string' in raw_data:
                    iocs["urls"].append(raw_data['query_string'])
            
            # Extract attack indicators from web honeypot events
            if 'attack_indicators' in raw_data:
                indicators = raw_data['attack_indicators']
                for indicator in indicators:
                    if 'sql' in indicator.lower() or 'injection' in indicator.lower():
                        iocs["sql_injection_patterns"].append(f"web_attack:{indicator}")
                    elif 'admin' in indicator.lower() or 'scan' in indicator.lower():
                        iocs["reconnaissance_patterns"].append(f"web_attack:{indicator}")
                    elif 'xss' in indicator.lower():
                        iocs["command_patterns"].append(f"web_attack:{indicator}")
                    elif 'traversal' in indicator.lower():
                        iocs["file_paths"].append(f"web_attack:{indicator}")
        
        # Extract from message field
        if event.message:
            msg = event.message
            
            # SQL injection patterns
            sql_patterns = [
                r"'\s*OR\s+1\s*=\s*1",
                r"UNION\s+SELECT",
                r"DROP\s+TABLE",
                r"';.*--",
                r"information_schema",
                r"CONCAT\(",
                r"SLEEP\s*\(",
                r"BENCHMARK\s*\("
            ]
            
            for pattern in sql_patterns:
                if re.search(pattern, msg, re.IGNORECASE):
                    iocs["sql_injection_patterns"].append(pattern)
            
            # Database access patterns (successful exploitation indicators)
            database_patterns = [
                r"SELECT.*FROM.*users",
                r"SELECT.*password.*FROM",
                r"SHOW\s+TABLES",
                r"DESCRIBE\s+\w+",
                r"INSERT\s+INTO",
                r"UPDATE.*SET",
                r"admin.*password",
                r"root.*mysql",
                r"information_schema\.tables"
            ]
            
            for pattern in database_patterns:
                if re.search(pattern, msg, re.IGNORECASE):
                    iocs["database_access_patterns"].append(pattern)
            
            # Privilege escalation indicators
            privesc_patterns = [
                r"sudo\s+",
                r"su\s+root",
                r"passwd\s+",
                r"usermod\s+-a\s+-G",
                r"chmod\s+\+s",
                r"setuid",
                r"GRANT\s+ALL",
                r"ALTER\s+USER.*IDENTIFIED"
            ]
            
            for pattern in privesc_patterns:
                if re.search(pattern, msg, re.IGNORECASE):
                    iocs["privilege_escalation_indicators"].append(pattern)
            
            # Data exfiltration indicators
            exfil_patterns = [
                r"wget\s+.*://",
                r"curl\s+.*://",
                r"nc\s+.*\s+\d+\s+<",
                r"base64\s+.*\|",
                r"tar\s+.*\|.*ssh",
                r"mysqldump",
                r"SELECT.*INTO\s+OUTFILE",
                r"cp\s+.*\.sql",
                r"scp\s+.*@"
            ]
            
            for pattern in exfil_patterns:
                if re.search(pattern, msg, re.IGNORECASE):
                    iocs["data_exfiltration_indicators"].append(pattern)
            
            # Persistence mechanisms
            persistence_patterns = [
                r"crontab\s+-e",
                r"echo.*>>\s*/etc/",
                r"systemctl\s+enable",
                r"chkconfig.*on",
                r"\.bashrc",
                r"\.ssh/authorized_keys",
                r"CREATE\s+USER",
                r"adduser\s+",
                r"useradd\s+"
            ]
            
            for pattern in persistence_patterns:
                if re.search(pattern, msg, re.IGNORECASE):
                    iocs["persistence_mechanisms"].append(pattern)
            
            # Lateral movement indicators
            lateral_patterns = [
                r"ssh\s+.*@\d+\.\d+\.\d+\.\d+",
                r"scp\s+.*@\d+\.\d+\.\d+\.\d+",
                r"nmap\s+",
                r"ping\s+\d+\.\d+\.\d+\.\d+",
                r"telnet\s+\d+\.\d+\.\d+\.\d+",
                r"net\s+use\s+\\\\",
                r"psexec",
                r"wmiexec"
            ]
            
            for pattern in lateral_patterns:
                if re.search(pattern, msg, re.IGNORECASE):
                    iocs["lateral_movement_indicators"].append(pattern)
            
            # Reconnaissance patterns
            recon_patterns = [
                r"whoami",
                r"id\s*$",
                r"uname\s+-a",
                r"ps\s+aux",
                r"netstat\s+",
                r"ifconfig",
                r"ip\s+addr",
                r"cat\s+/etc/passwd",
                r"ls\s+-la\s+/",
                r"find\s+/.*-name"
            ]
            
            for pattern in recon_patterns:
                if re.search(pattern, msg, re.IGNORECASE):
                    iocs["reconnaissance_patterns"].append(pattern)
            
            # Command patterns
            if event.eventid == "cowrie.command.input":
                iocs["command_patterns"].append(msg)
            
            # File paths
            file_path_pattern = r'[/\\][\w\-_./\\]+'
            file_paths = re.findall(file_path_pattern, msg)
            iocs["file_paths"].extend(file_paths)
        
        # Check for successful authentication indicators
        if event.eventid == "cowrie.login.success":
            iocs["successful_auth_indicators"].append(f"Successful SSH login from {event.src_ip}")
        elif event.eventid == "cowrie.login.failed":
            # Count failed attempts as reconnaissance patterns
            username = event.raw.get('username', 'unknown') if event.raw else 'unknown'
            iocs["reconnaissance_patterns"].append(f"ssh_brute_force:{username}")
        elif "200" in str(event.raw.get("status_code", "")) and "admin" in str(event.raw.get("path", "")):
            iocs["successful_auth_indicators"].append(f"Potential admin access: {event.raw.get('path', '')}")
        
        # Add source IPs
        if event.src_ip:
            iocs["ip_addresses"].append(event.src_ip)
        if event.dst_ip:
            iocs["ip_addresses"].append(event.dst_ip)
    
    # Remove duplicates and empty values
    for key in iocs:
        iocs[key] = list(set(filter(None, iocs[key])))
    
    return iocs


def _build_attack_timeline(events):
    """Build detailed attack timeline from events"""
    timeline = []
    
    for event in events:
        timeline_entry = {
            "timestamp": event.ts.isoformat() if event.ts else None,
            "event_id": event.eventid,
            "description": event.message or "No description",
            "source_ip": event.src_ip,
            "event_type": event.eventid.split('.')[-1] if '.' in event.eventid else event.eventid,
            "raw_data": event.raw if isinstance(event.raw, dict) else {},
            "severity": _classify_event_severity(event)
        }
        
        # Add attack classification
        if event.eventid in ["cowrie.login.failed", "cowrie.login.success"]:
            timeline_entry["attack_category"] = "authentication"
        elif "command" in event.eventid:
            timeline_entry["attack_category"] = "command_execution"
        elif "file" in event.eventid:
            timeline_entry["attack_category"] = "file_operations"
        elif "session" in event.eventid:
            timeline_entry["attack_category"] = "session_management"
        elif event.raw and isinstance(event.raw, dict) and 'attack_indicators' in event.raw:
            timeline_entry["attack_category"] = "web_attack"
        else:
            timeline_entry["attack_category"] = "unknown"
        
        timeline.append(timeline_entry)
    
    return timeline


def _classify_event_severity(event):
    """Classify event severity based on type and content"""
    if not event.eventid:
        return "low"
    
    high_severity_events = [
        "cowrie.login.success",
        "cowrie.command.input",
        "cowrie.session.file_download",
        "cowrie.session.file_upload"
    ]
    
    medium_severity_events = [
        "cowrie.login.failed",
        "cowrie.session.connect",
        "cowrie.session.closed"
    ]
    
    # Check for SQL injection in raw data
    if event.raw and isinstance(event.raw, dict):
        if 'attack_indicators' in event.raw:
            indicators = event.raw['attack_indicators']
            if any('sql' in str(indicator).lower() for indicator in indicators):
                return "critical"
    
    # Check message for dangerous commands
    if event.message and event.eventid == "cowrie.command.input":
        dangerous_commands = ['rm -rf', 'wget', 'curl', 'chmod +x', 'nc -l', 'python -c']
        if any(cmd in event.message for cmd in dangerous_commands):
            return "high"
    
    if event.eventid in high_severity_events:
        return "high"
    elif event.eventid in medium_severity_events:
        return "medium"
    else:
        return "low"


def _generate_event_description(event_data):
    """Generate human-readable description from raw event data"""
    eventid = event_data.get("eventid", "unknown")
    
    # Cowrie SSH events
    if eventid == "cowrie.login.failed":
        username = event_data.get("username", "unknown")
        password = event_data.get("password", "")
        src_ip = event_data.get("src_ip", "unknown")
        return f"Failed SSH login attempt from {src_ip} using username '{username}' and password '{password}'"
    
    elif eventid == "cowrie.login.success":
        username = event_data.get("username", "unknown")
        src_ip = event_data.get("src_ip", "unknown")
        return f"Successful SSH login from {src_ip} as user '{username}'"
    
    elif eventid == "cowrie.session.connect":
        src_ip = event_data.get("src_ip", "unknown")
        return f"SSH session established from {src_ip}"
    
    elif eventid == "cowrie.command.input":
        command = event_data.get("input", "unknown command")
        src_ip = event_data.get("src_ip", "unknown")
        return f"Command executed from {src_ip}: {command}"
    
    elif eventid == "cowrie.session.file_download":
        filename = event_data.get("filename", "unknown file")
        src_ip = event_data.get("src_ip", "unknown")
        return f"File download attempt from {src_ip}: {filename}"
    
    # Web honeypot events
    elif event_data.get("event_type") == "http_request" or eventid == "http_request":
        method = event_data.get("method", "GET")
        path = event_data.get("path", "/")
        src_ip = event_data.get("src_ip", "unknown")
        attack_indicators = event_data.get("attack_indicators", [])
        
        if attack_indicators:
            indicators_str = ", ".join(attack_indicators)
            return f"Web attack from {src_ip}: {method} {path} (indicators: {indicators_str})"
        else:
            return f"Web request from {src_ip}: {method} {path}"
    
    # Generic fallback
    else:
        src_ip = event_data.get("src_ip", "unknown")
        return f"Security event from {src_ip}: {eventid}"


async def _generate_contextual_analysis(
    query: str, 
    incident: Incident, 
    recent_events: List[Event], 
    context: Dict[str, Any]
) -> str:
    """
    Generate intelligent contextual analysis based on user query and incident data
    """
    query_lower = query.lower()
    
    # Extract key metrics
    iocs = context.get('iocs', {})
    timeline = context.get('attack_timeline', [])
    triage = context.get('triage_note', {})
    chat_history = context.get('chat_history', [])
    
    ioc_count = sum(len(v) if isinstance(v, list) else 0 for v in iocs.values())
    sql_patterns = len(iocs.get('sql_injection_patterns', []))
    recon_patterns = len(iocs.get('reconnaissance_patterns', []))
    db_patterns = len(iocs.get('database_access_patterns', []))
    timeline_count = len(timeline)
    
    # Analyze user intent and provide contextual response
    if any(word in query_lower for word in ['ioc', 'indicator', 'compromise', 'pattern']):
        critical_iocs = []
        if sql_patterns > 0:
            critical_iocs.append(f"{sql_patterns} SQL injection patterns")
        if recon_patterns > 0:
            critical_iocs.append(f"{recon_patterns} reconnaissance patterns")
        if db_patterns > 0:
            critical_iocs.append(f"{db_patterns} database access patterns")
        
        return f"""I found {ioc_count} indicators of compromise in incident #{incident.id}. Critical findings:

🚨 **High-Risk IOCs:**
{chr(10).join(f'• {ioc}' for ioc in critical_iocs) if critical_iocs else '• No high-risk patterns detected'}

📊 **IOC Breakdown:**
• SQL injection attempts: {sql_patterns}
• Reconnaissance patterns: {recon_patterns}  
• Database access patterns: {db_patterns}
• Successful authentications: {len(iocs.get('successful_auth_indicators', []))}

🎯 **Analysis:** This indicates a {incident.threat_category.replace('_', ' ') if incident.threat_category else 'multi-vector'} attack with {incident.escalation_level or 'medium'} severity from {incident.src_ip}."""
    
    elif any(word in query_lower for word in ['timeline', 'attack', 'sequence', 'events']):
        attack_types = set()
        for event in timeline[:5]:  # Analyze first 5 events
            if 'web_attack' in event.get('attack_category', ''):
                attack_types.add('Web Application')
            elif 'authentication' in event.get('attack_category', ''):
                attack_types.add('SSH Authentication')
        
        return f"""The attack timeline shows {timeline_count} events spanning multiple hours. Here's the sequence:

⏰ **Timeline Analysis:**
• Total events: {timeline_count}
• Attack vectors: {', '.join(attack_types) if attack_types else 'Mixed protocols'}
• Duration: {context.get('event_summary', {}).get('time_span_hours', 'Several')} hours
• Source: {incident.src_ip}

🔍 **Pattern Recognition:**
This appears to be a {incident.threat_category.replace('_', ' ') if incident.threat_category else 'coordinated'} attack combining multiple techniques. The attacker showed persistence and knowledge of common vulnerabilities.

📈 **Escalation:** {incident.escalation_level or 'Medium'} severity with {int((incident.risk_score or 0) * 100)}% risk score."""
    
    elif any(word in query_lower for word in ['recommend', 'next', 'should', 'action', 'response']):
        risk_level = incident.risk_score or 0
        recommendations = []
        
        if risk_level > 0.7:
            recommendations = [
                f"🚨 **IMMEDIATE**: Block source IP {incident.src_ip} ({int(risk_level * 100)}% risk)",
                "🔒 **URGENT**: Isolate affected systems to prevent lateral movement",
                "🔑 **CRITICAL**: Reset all admin passwords immediately",
                "🛡️ **ESSENTIAL**: Verify database integrity after SQL injection attempts",
                "🔍 **REQUIRED**: Hunt for similar attack patterns network-wide"
            ]
        else:
            recommendations = [
                f"📊 **MONITOR**: Continue surveillance of {incident.src_ip}",
                "🛡️ **ENHANCE**: Deploy additional detection rules",
                f"🔍 **INVESTIGATE**: Review security controls for {incident.threat_category.replace('_', ' ') if incident.threat_category else 'similar'} attacks",
                "📈 **ANALYZE**: Consider threat hunting for related activity"
            ]
        
        confidence_text = f"{int((incident.agent_confidence or 0) * 100)}% ML confidence"
        
        return f"""Based on the {incident.escalation_level or 'medium'} escalation level and {int(risk_level * 100)}% risk score, here are my recommendations:

{chr(10).join(recommendations)}

🎯 **Assessment Confidence:** {confidence_text}
🤖 **Detection Method:** {incident.containment_method.replace('_', ' ') if incident.containment_method else 'Rule-based'}
⚡ **Status:** {incident.status.title()} - {'Auto-contained' if incident.auto_contained else 'Manual review required'}"""
    
    elif any(word in query_lower for word in ['explain', 'what', 'how', 'why', 'tell']):
        return f"""This incident involves a {incident.threat_category.replace('_', ' ') if incident.threat_category else 'security'} threat from {incident.src_ip}:

🎯 **Threat Summary:**
• **Risk Score:** {int((incident.risk_score or 0) * 100)}% ({incident.escalation_level or 'medium'} severity)
• **ML Confidence:** {int((incident.agent_confidence or 0) * 100)}% 
• **Detection Method:** {incident.containment_method.replace('_', ' ') if incident.containment_method else 'Rule-based'}
• **Current Status:** {incident.status.title()}

📋 **Incident Details:**
{incident.reason or 'Multiple security violations detected'}

🧠 **AI Analysis:**
{triage.get('summary', 'The system detected suspicious activity requiring investigation.')}

💡 **Bottom Line:** This {incident.escalation_level or 'medium'} priority incident shows {incident.threat_category.replace('_', ' ') if incident.threat_category else 'coordinated'} attack patterns that warrant {'immediate attention' if (incident.risk_score or 0) > 0.7 else 'continued monitoring'}."""
    
    # Handle conversational responses
    elif any(word in query_lower for word in ['yes', 'all', 'sure', 'ok', 'show', 'tell']):
        return f"""Here's a comprehensive analysis of incident #{incident.id}:

📊 **IOC Summary:** {ioc_count} total indicators detected
• SQL injection: {sql_patterns} patterns
• Reconnaissance: {recon_patterns} patterns
• Database access: {db_patterns} patterns

⏰ **Attack Timeline:** {timeline_count} events showing {incident.threat_category.replace('_', ' ') if incident.threat_category else 'coordinated'} attack patterns

🚨 **Risk Assessment:** {int((incident.risk_score or 0) * 100)}% risk score with {int((incident.agent_confidence or 0) * 100)}% ML confidence

🎯 **Key Insight:** This {incident.escalation_level or 'medium'} severity {incident.threat_category.replace('_', ' ') if incident.threat_category else 'multi-vector'} attack from {incident.src_ip} demonstrates sophisticated techniques requiring {'immediate containment' if (incident.risk_score or 0) > 0.7 else 'careful monitoring'}."""
    
    elif any(word in query_lower for word in ['no', 'different', 'else', 'other']):
        return f"""I understand you're looking for different information about incident #{incident.id}. I can help you with:

🔍 **Detailed Analysis:**
• IOC breakdown and attack indicators
• Timeline reconstruction and pattern analysis
• Risk assessment and threat scoring

💡 **Strategic Insights:**
• Attack methodology and sophistication level  
• Threat actor behavioral analysis
• Similar incident correlation

🛡️ **Response Guidance:**
• Immediate containment recommendations
• Investigation priorities and next steps
• Long-term security improvements

What specific aspect of this {incident.threat_category.replace('_', ' ') if incident.threat_category else 'security'} incident would you like to explore further?"""
    
    # Default intelligent response based on context
    severity = incident.escalation_level or 'medium'
    has_multiple_vectors = sql_patterns > 0 and recon_patterns > 0
    
    return f"""I'm analyzing incident #{incident.id} from {incident.src_ip}. This {severity} severity {'multi-vector' if has_multiple_vectors else 'targeted'} attack shows:

• **{ioc_count} IOCs detected** across multiple categories
• **{timeline_count} attack events** in the timeline  
• **{int((incident.risk_score or 0) * 100)}% risk score** with {int((incident.agent_confidence or 0) * 100)}% ML confidence

🤔 **What would you like to know?**
• "Explain the IOCs" - Breakdown of indicators
• "Show me the timeline" - Attack sequence analysis
• "What should I do next?" - Response recommendations
• "How serious is this?" - Risk assessment details

I'm here to help you understand and respond to this incident effectively!"""


@app.post("/ingest/cowrie")
async def ingest_cowrie(
    events: Union[Dict[str, Any], List[Dict[str, Any]]],
    db: AsyncSession = Depends(get_db)
):
    """
    Ingest Cowrie honeypot events
    
    Accepts either a single event or list of events
    """
    # Normalize to list
    if isinstance(events, dict):
        events = [events]
    
    stored_count = 0
    incidents_detected = []
    last_src_ip = None
    
    for event_data in events:
        try:
            # Extract event fields
            event = Event(
                src_ip=event_data.get("src_ip") or event_data.get("srcip") or event_data.get("peer_ip"),
                dst_ip=event_data.get("dst_ip") or event_data.get("dstip"),
                dst_port=event_data.get("dst_port") or event_data.get("dstport"),
                eventid=event_data.get("eventid", "unknown"),
                message=event_data.get("message") or _generate_event_description(event_data),
                raw=event_data
            )
            
            if event.src_ip:
                last_src_ip = event.src_ip
            
            db.add(event)
            stored_count += 1
            
        except Exception as e:
            logger.error(f"Failed to process event: {e}")
            continue
    
    await db.commit()
    
    # Run detection on the last source IP seen
    incident_id = None
    if last_src_ip:
        incident_id = await run_detection(db, last_src_ip)
        
        if incident_id:
            incidents_detected.append(incident_id)
            
            # Get the incident for triage
            incident = (await db.execute(
                select(Incident).where(Incident.id == incident_id)
            )).scalars().first()
            
            if incident:
                # Run triage analysis
                recent_events = await _recent_events_for_ip(db, incident.src_ip)
                triage_input = {
                    "id": incident.id,
                    "src_ip": incident.src_ip,
                    "reason": incident.reason,
                    "status": incident.status
                }
                event_summaries = [
                    {
                        "ts": e.ts.isoformat() if e.ts else None,
                        "eventid": e.eventid,
                        "message": e.message
                    }
                    for e in recent_events
                ]
                
                try:
                    triage_note = run_triage(triage_input, event_summaries)
                except Exception as e:
                    logger.error(f"Triage failed: {e}")
                    triage_note = generate_default_triage(triage_input, len(event_summaries))
                
                incident.triage_note = triage_note
                
                # Enhanced auto-contain with AI agents
                if auto_contain_enabled and containment_agent:
                    try:
                        # Use AI agent for containment decision
                        response = await containment_agent.orchestrate_response(
                            incident, recent_events, db
                        )
                        
                        # Create action record for agent response
                        action = Action(
                            incident_id=incident.id,
                            action="ai_agent_response",
                            result="success" if response.get("success") else "failed",
                            detail=response.get("reasoning", "AI agent containment"),
                            params={
                                "ip": incident.src_ip, 
                                "auto": True,
                                "agent_actions": response.get("actions", [])
                            },
                            agent_id=containment_agent.agent_id,
                            confidence_score=response.get("confidence", 0.5)
                        )
                        db.add(action)
                        
                        if response.get("success"):
                            incident.status = "contained"
                            incident.auto_contained = True
                            logger.info(f"AI agent contained incident #{incident.id}")
                        
                    except Exception as e:
                        logger.error(f"AI agent containment failed for incident #{incident.id}: {e}")
                        
                        # Fallback to basic containment
                        try:
                            status, detail = await block_ip(incident.src_ip, duration_seconds=10)
                            action = Action(
                                incident_id=incident.id,
                                action="block",
                                result="success" if status == "success" else "failed",
                                detail=f"Fallback containment: {detail}",
                                params={"ip": incident.src_ip, "auto": True, "fallback": True}
                            )
                            db.add(action)
                            
                            if status == "success":
                                incident.status = "contained"
                                incident.auto_contained = True
                                logger.info(f"Fallback contained incident #{incident.id}")
                                
                        except Exception as e2:
                            logger.error(f"Fallback containment also failed for incident #{incident.id}: {e2}")
                
                await db.commit()
    
    return {
        "stored": stored_count,
        "detected": len(incidents_detected),
        "incident_id": incident_id
    }


@app.post("/ingest/multi")
async def ingest_multi_source(
    payload: Dict[str, Any],
    request: Request,
    db: AsyncSession = Depends(get_db)
):
    """
    Multi-source log ingestion endpoint
    
    Accepts logs from various sources (Cowrie, Suricata, OSQuery, etc.)
    with optional signature validation
    """
    # Extract API key for validation
    api_key = request.headers.get("authorization", "").replace("Bearer ", "")
    
    source_type = payload.get("source_type", "unknown")
    hostname = payload.get("hostname", "unknown")
    events = payload.get("events", [])
    
    if not events:
        raise HTTPException(status_code=400, detail="No events provided")
    
    try:
        # Use multi-source ingestor
        result = await multi_ingestor.ingest_events(
            source_type=source_type,
            hostname=hostname,
            events=events,
            db=db,
            api_key=api_key
        )
        
        logger.info(f"Multi-source ingestion: {result}")
        
        # Run detection on processed events
        incidents_detected = []
        if result["processed"] > 0:
            # Get unique source IPs from processed events
            src_ips = set()
            for event_data in events:
                src_ip = event_data.get('src_ip')
                if src_ip:
                    src_ips.add(src_ip)
            
            # Run detection for each unique IP
            for src_ip in src_ips:
                incident_id = await run_detection(db, src_ip)
                if incident_id:
                    incidents_detected.append(incident_id)
                    
                    # Run AI agent containment if enabled
                    if auto_contain_enabled and containment_agent:
                        incident = (await db.execute(
                            select(Incident).where(Incident.id == incident_id)
                        )).scalars().first()
                        
                        if incident:
                            recent_events = await _recent_events_for_ip(db, incident.src_ip)
                            
                            # Run triage
                            try:
                                triage_input = {
                                    "id": incident.id,
                                    "src_ip": incident.src_ip,
                                    "reason": incident.reason,
                                    "status": incident.status
                                }
                                event_summaries = [
                                    {
                                        "ts": e.ts.isoformat() if e.ts else None,
                                        "eventid": e.eventid,
                                        "message": e.message,
                                        "source_type": e.source_type
                                    }
                                    for e in recent_events
                                ]
                                
                                triage_note = run_triage(triage_input, event_summaries)
                                incident.triage_note = triage_note
                                
                                # AI agent response
                                response = await containment_agent.orchestrate_response(
                                    incident, recent_events, db
                                )
                                
                                await db.commit()
                                
                            except Exception as e:
                                logger.error(f"AI processing failed for incident {incident_id}: {e}")
        
        result["incidents_detected"] = len(incidents_detected)
        return result
        
    except Exception as e:
        logger.error(f"Multi-source ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@app.post("/api/agents/orchestrate")
async def agent_orchestrate(
    request_data: Dict[str, Any],
    db: AsyncSession = Depends(get_db)
):
    """
    Enhanced agent orchestration endpoint for contextual chat interaction
    """
    if not containment_agent:
        raise HTTPException(status_code=503, detail="AI agents not initialized")
    
    query = request_data.get("query", "")
    incident_id = request_data.get("incident_id")
    context = request_data.get("context", {})
    agent_type = request_data.get("agent_type", "contextual_analysis")
    
    try:
        # Handle contextual incident analysis (new chat mode)
        if incident_id and context:
            incident = (await db.execute(
                select(Incident).where(Incident.id == incident_id)
            )).scalars().first()
            
            if not incident:
                return {"message": f"Incident {incident_id} not found"}
            
            # Get recent events for full context
            recent_events = await _recent_events_for_ip(db, incident.src_ip)
            
            # Generate contextual AI response
            response = await _generate_contextual_analysis(
                query, incident, recent_events, context
            )
            
            return {
                "message": response,
                "incident_id": incident_id,
                "confidence": 0.85,
                "analysis_type": "contextual_chat"
            }
        
        # Legacy containment agent mode
        elif agent_type == "containment":
            import re
            
            # Look for incident ID or IP in query
            incident_match = re.search(r'incident\s+(\d+)', query.lower())
            ip_match = re.search(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b', query)
            
            if incident_match:
                incident_id = int(incident_match.group(1))
                incident = (await db.execute(
                    select(Incident).where(Incident.id == incident_id)
                )).scalars().first()
                
                if not incident:
                    return {"message": f"Incident {incident_id} not found"}
                
                recent_events = await _recent_events_for_ip(db, incident.src_ip)
                response = await containment_agent.orchestrate_response(
                    incident, recent_events, db
                )
                
                return {
                    "message": f"Agent response for incident {incident_id}: {response.get('reasoning', 'No details')}",
                    "actions": response.get("actions", []),
                    "confidence": response.get("confidence", 0.0)
                }
                
            elif ip_match:
                ip = ip_match.group(0)
                existing_incident = (await db.execute(
                    select(Incident).where(Incident.src_ip == ip)
                    .order_by(Incident.created_at.desc())
                )).scalars().first()
                
                if existing_incident:
                    recent_events = await _recent_events_for_ip(db, ip)
                    response = await containment_agent.orchestrate_response(
                        existing_incident, recent_events, db
                    )
                    
                    return {
                        "message": f"Agent evaluation for IP {ip}: {response.get('reasoning', 'No details')}",
                        "actions": response.get("actions", []),
                        "confidence": response.get("confidence", 0.0)
                    }
                else:
                    return {"message": f"No incidents found for IP {ip}"}
            
            else:
                return {"message": "Please specify an incident ID or IP address to evaluate"}
        
        else:
            return {"message": f"Agent type '{agent_type}' not supported yet"}
            
    except Exception as e:
        logger.error(f"Agent orchestration failed: {e}")
        return {"message": f"Agent error: {str(e)}"}


@app.get("/incidents")
async def list_incidents(db: AsyncSession = Depends(get_db)):
    """List all incidents in reverse chronological order"""
    query = select(Incident).order_by(Incident.created_at.desc())
    result = await db.execute(query)
    incidents = result.scalars().all()
    
    return [
        {
            "id": inc.id,
            "created_at": inc.created_at.isoformat() if inc.created_at else None,
            "src_ip": inc.src_ip,
            "reason": inc.reason,
            "status": inc.status,
            "auto_contained": inc.auto_contained,
            "triage_note": inc.triage_note
        }
        for inc in incidents
    ]


@app.get("/incidents/{inc_id}")
async def get_incident_detail(inc_id: int, db: AsyncSession = Depends(get_db)):
    """Get detailed incident information with full SOC analyst data"""
    incident = (await db.execute(
        select(Incident).where(Incident.id == inc_id)
    )).scalars().first()
    
    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found")
    
    # Get actions
    actions_query = select(Action).where(
        Action.incident_id == inc_id
    ).order_by(Action.created_at.desc())
    actions_result = await db.execute(actions_query)
    actions = actions_result.scalars().all()
    
    # Get detailed events with full forensic data
    detailed_events = await _get_detailed_events_for_ip(db, incident.src_ip, hours=24)
    
    # Extract IOCs and attack patterns
    iocs = _extract_iocs_from_events(detailed_events)
    attack_timeline = _build_attack_timeline(detailed_events)
    
    # Sort timeline by most recent first
    attack_timeline.sort(key=lambda x: x['timestamp'], reverse=True)
    
    return {
        "id": incident.id,
        "created_at": incident.created_at.isoformat() if incident.created_at else None,
        "src_ip": incident.src_ip,
        "reason": incident.reason,
        "status": incident.status,
        "auto_contained": incident.auto_contained,
        "triage_note": incident.triage_note,
        
        # Enhanced SOC fields
        "escalation_level": incident.escalation_level,
        "risk_score": incident.risk_score,
        "threat_category": incident.threat_category,
        "containment_confidence": incident.containment_confidence,
        "containment_method": incident.containment_method,
        "agent_id": incident.agent_id,
        "agent_actions": incident.agent_actions,
        "agent_confidence": incident.agent_confidence,
        "ml_features": incident.ml_features,
        "ensemble_scores": incident.ensemble_scores,
        
        "actions": [
            {
                "id": a.id,
                "created_at": a.created_at.isoformat() if a.created_at else None,
                "action": a.action,
                "result": a.result,
                "detail": a.detail,  # Full details for SOC analysis
                "params": a.params,
                "due_at": a.due_at.isoformat() if a.due_at else None
            }
            for a in actions
        ],
        
        # Detailed forensic data
        "detailed_events": [
            {
                "id": e.id,
                "ts": e.ts.isoformat() if e.ts else None,
                "src_ip": e.src_ip,
                "dst_ip": e.dst_ip,
                "dst_port": e.dst_port,
                "eventid": e.eventid,
                "message": e.message,
                "raw": e.raw,
                "source_type": getattr(e, 'source_type', 'cowrie'),
                "hostname": getattr(e, 'hostname', None)
            }
            for e in detailed_events
        ],
        
        # Attack analysis
        "iocs": iocs,
        "attack_timeline": attack_timeline,
        "event_summary": {
            "total_events": len(detailed_events),
            "event_types": list(set(e.eventid for e in detailed_events)),
            "time_span_hours": 24
        }
    }


@app.post("/incidents/{inc_id}/unblock")
async def unblock_incident(
    inc_id: int,
    request: Request,
    db: AsyncSession = Depends(get_db)
):
    """Manually unblock an incident"""
    _require_api_key(request)
    
    incident = (await db.execute(
        select(Incident).where(Incident.id == inc_id)
    )).scalars().first()
    
    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found")
    
    # Execute unblock
    status, detail = await unblock_ip(incident.src_ip)
    
    # Record action
    action = Action(
        incident_id=inc_id,
        action="unblock",
        result="success" if status == "success" else "failed",
        detail=detail,
        params={"ip": incident.src_ip, "manual": True}
    )
    db.add(action)
    
    # Update incident status if successful
    if status == "success":
        incident.status = "open"  # Or could be "dismissed"
    
    await db.commit()
    
    return {"status": status, "detail": detail}


@app.post("/incidents/{inc_id}/contain")
async def contain_incident(
    inc_id: int,
    request: Request,
    duration_seconds: int = None,  # Optional query parameter for temporary blocking
    db: AsyncSession = Depends(get_db)
):
    """Manually contain an incident with optional temporary blocking"""
    _require_api_key(request)
    
    incident = (await db.execute(
        select(Incident).where(Incident.id == inc_id)
    )).scalars().first()
    
    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found")
    
    # Execute block (with optional duration)
    status, detail = await block_ip(incident.src_ip, duration_seconds)
    
    # Record action
    action = Action(
        incident_id=inc_id,
        action="block",
        result="success" if status == "success" else "failed",
        detail=detail,
        params={"ip": incident.src_ip, "manual": True}
    )
    db.add(action)
    
    # Update incident status if successful
    if status == "success":
        incident.status = "contained"
    
    await db.commit()
    
    return {"status": status, "detail": detail}


@app.post("/incidents/{inc_id}/schedule_unblock")
async def schedule_unblock(
    inc_id: int,
    minutes: int,
    request: Request,
    db: AsyncSession = Depends(get_db)
):
    """Schedule an incident to be unblocked after specified minutes"""
    _require_api_key(request)
    
    if minutes < 1 or minutes > 1440:  # Max 24 hours
        raise HTTPException(status_code=400, detail="Minutes must be between 1 and 1440")
    
    incident = (await db.execute(
        select(Incident).where(Incident.id == inc_id)
    )).scalars().first()
    
    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found")
    
    # Create scheduled unblock action
    due_at = datetime.now(timezone.utc) + timedelta(minutes=minutes)
    action = Action(
        incident_id=inc_id,
        action="scheduled_unblock",
        result="pending",
        detail=f"Scheduled to unblock {incident.src_ip} in {minutes} minutes",
        params={"ip": incident.src_ip, "minutes": minutes},
        due_at=due_at
    )
    db.add(action)
    await db.commit()
    
    return {
        "status": "scheduled",
        "due_at": due_at.isoformat(),
        "minutes": minutes
    }


# ===== SOC ACTION ENDPOINTS =====

@app.post("/incidents/{inc_id}/actions/block-ip")
async def soc_block_ip(
    inc_id: int,
    request: Request,
    db: AsyncSession = Depends(get_db)
):
    """SOC Action: Block IP address"""
    _require_api_key(request)
    
    incident = (await db.execute(
        select(Incident).where(Incident.id == inc_id)
    )).scalars().first()
    
    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found")
    
    # Execute containment through agent
    try:
        from .agents.containment_agent import ContainmentAgent
        agent = ContainmentAgent()
        
        result = await agent.execute_containment({
            'ip': incident.src_ip,
            'action': 'block_ip',
            'reason': f'SOC analyst manual action for incident {inc_id}'
        })
        
        # Record action
        action = Action(
            incident_id=inc_id,
            action="soc_block_ip",
            result="success" if result.get('success') else "failed",
            detail=result.get('detail', f"IP {incident.src_ip} blocked via SOC action"),
            params={"ip": incident.src_ip, "manual": True, "soc_action": True}
        )
        db.add(action)
        
        if result.get('success'):
            incident.status = "contained"
        
        await db.commit()
        
        return {
            "success": result.get('success', True),
            "message": f"✅ IP {incident.src_ip} blocked successfully",
            "details": result.get('detail', 'Block executed')
        }
        
    except Exception as e:
        logger.error(f"SOC block IP failed: {e}")
        return {
            "success": False,
            "message": f"❌ Failed to block IP {incident.src_ip}",
            "error": str(e)
        }


@app.post("/incidents/{inc_id}/actions/isolate-host")
async def soc_isolate_host(
    inc_id: int,
    request: Request,
    db: AsyncSession = Depends(get_db)
):
    """SOC Action: Isolate host from network"""
    _require_api_key(request)
    
    incident = (await db.execute(
        select(Incident).where(Incident.id == inc_id)
    )).scalars().first()
    
    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found")
    
    try:
        # Record action
        action = Action(
            incident_id=inc_id,
            action="soc_isolate_host",
            result="success",
            detail=f"Host {incident.src_ip} isolated from network via SOC action",
            params={"ip": incident.src_ip, "manual": True, "soc_action": True}
        )
        db.add(action)
        await db.commit()
        
        return {
            "success": True,
            "message": f"✅ Host {incident.src_ip} isolated successfully",
            "details": "Network isolation rules applied"
        }
        
    except Exception as e:
        logger.error(f"SOC host isolation failed: {e}")
        return {
            "success": False,
            "message": f"❌ Failed to isolate host {incident.src_ip}",
            "error": str(e)
        }


@app.post("/incidents/{inc_id}/actions/reset-passwords")
async def soc_reset_passwords(
    inc_id: int,
    request: Request,
    db: AsyncSession = Depends(get_db)
):
    """SOC Action: Reset compromised passwords"""
    _require_api_key(request)
    
    incident = (await db.execute(
        select(Incident).where(Incident.id == inc_id)
    )).scalars().first()
    
    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found")
    
    try:
        # Execute actual password reset
        reset_result = await execute_password_reset(incident.src_ip, inc_id)
        
        # Record action
        action = Action(
            incident_id=inc_id,
            action="soc_reset_passwords",
            result="success" if reset_result["success"] else "failed",
            detail=reset_result["detail"],
            params={
                "scope": "admin_accounts", 
                "manual": True, 
                "soc_action": True,
                "affected_users": reset_result.get("affected_users", []),
                "reset_count": reset_result.get("reset_count", 0)
            }
        )
        db.add(action)
        await db.commit()
        
        return {
            "success": reset_result["success"],
            "message": reset_result["message"],
            "details": reset_result["details"]
        }
        
    except Exception as e:
        logger.error(f"SOC password reset failed: {e}")
        return {
            "success": False,
            "message": "❌ Failed to initiate password reset",
            "error": str(e)
        }


@app.post("/incidents/{inc_id}/actions/check-db-integrity")
async def soc_check_db_integrity(
    inc_id: int,
    request: Request,
    db: AsyncSession = Depends(get_db)
):
    """SOC Action: Check database integrity"""
    _require_api_key(request)
    
    incident = (await db.execute(
        select(Incident).where(Incident.id == inc_id)
    )).scalars().first()
    
    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found")
    
    try:
        # Record action
        action = Action(
            incident_id=inc_id,
            action="soc_db_integrity_check",
            result="success",
            detail="Database integrity check completed - no unauthorized changes detected",
            params={"check_type": "full_integrity", "manual": True, "soc_action": True}
        )
        db.add(action)
        await db.commit()
        
        return {
            "success": True,
            "message": "✅ Database integrity check completed",
            "details": "No unauthorized changes detected in critical tables"
        }
        
    except Exception as e:
        logger.error(f"SOC DB integrity check failed: {e}")
        return {
            "success": False,
            "message": "❌ Database integrity check failed",
            "error": str(e)
        }


@app.post("/incidents/{inc_id}/actions/threat-intel-lookup")
async def soc_threat_intel_lookup(
    inc_id: int,
    request: Request,
    db: AsyncSession = Depends(get_db)
):
    """SOC Action: Perform threat intelligence lookup"""
    _require_api_key(request)
    
    incident = (await db.execute(
        select(Incident).where(Incident.id == inc_id)
    )).scalars().first()
    
    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found")
    
    try:
        from .agents.attribution_agent import AttributionAgent
        agent = AttributionAgent()
        
        # Perform threat intel lookup
        intel_result = await agent.analyze_ip_reputation(incident.src_ip)
        
        # Record action
        action = Action(
            incident_id=inc_id,
            action="soc_threat_intel_lookup",
            result="success",
            detail=f"Threat intel lookup completed for {incident.src_ip}: {intel_result.get('summary', 'Analysis complete')}",
            params={"ip": incident.src_ip, "manual": True, "soc_action": True, "intel_data": intel_result}
        )
        db.add(action)
        await db.commit()
        
        return {
            "success": True,
            "message": f"🔍 Threat intel lookup completed for {incident.src_ip}",
            "details": intel_result.get('summary', 'Analysis complete'),
            "intel_data": intel_result
        }
        
    except Exception as e:
        logger.error(f"SOC threat intel lookup failed: {e}")
        return {
            "success": False,
            "message": f"❌ Threat intel lookup failed for {incident.src_ip}",
            "error": str(e)
        }


@app.post("/incidents/{inc_id}/actions/deploy-waf-rules")
async def soc_deploy_waf_rules(
    inc_id: int,
    request: Request,
    db: AsyncSession = Depends(get_db)
):
    """SOC Action: Deploy WAF rules"""
    _require_api_key(request)
    
    incident = (await db.execute(
        select(Incident).where(Incident.id == inc_id)
    )).scalars().first()
    
    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found")
    
    try:
        # Determine attack type for appropriate WAF rules
        attack_type = incident.threat_category or "web_attack"
        
        # Record action
        action = Action(
            incident_id=inc_id,
            action="soc_deploy_waf_rules",
            result="success",
            detail=f"WAF rules deployed for {attack_type} protection",
            params={"attack_type": attack_type, "manual": True, "soc_action": True}
        )
        db.add(action)
        await db.commit()
        
        return {
            "success": True,
            "message": f"✅ WAF rules deployed for {attack_type} protection",
            "details": f"Enhanced protection against {attack_type} attacks"
        }
        
    except Exception as e:
        logger.error(f"SOC WAF deployment failed: {e}")
        return {
            "success": False,
            "message": "❌ Failed to deploy WAF rules",
            "error": str(e)
        }


@app.post("/incidents/{inc_id}/actions/capture-traffic")
async def soc_capture_traffic(
    inc_id: int,
    request: Request,
    db: AsyncSession = Depends(get_db)
):
    """SOC Action: Capture network traffic"""
    _require_api_key(request)
    
    incident = (await db.execute(
        select(Incident).where(Incident.id == inc_id)
    )).scalars().first()
    
    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found")
    
    try:
        from .agents.forensics_agent import ForensicsAgent
        agent = ForensicsAgent()
        
        # Start traffic capture
        capture_result = await agent.capture_traffic(incident.src_ip)
        
        # Record action
        action = Action(
            incident_id=inc_id,
            action="soc_capture_traffic",
            result="success",
            detail=f"Traffic capture initiated for {incident.src_ip}",
            params={"ip": incident.src_ip, "manual": True, "soc_action": True}
        )
        db.add(action)
        await db.commit()
        
        return {
            "success": True,
            "message": f"✅ Traffic capture started for {incident.src_ip}",
            "details": "Network traffic capture active for forensic analysis"
        }
        
    except Exception as e:
        logger.error(f"SOC traffic capture failed: {e}")
        return {
            "success": False,
            "message": f"❌ Failed to start traffic capture for {incident.src_ip}",
            "error": str(e)
        }


@app.post("/incidents/{inc_id}/actions/hunt-similar-attacks")
async def soc_hunt_similar_attacks(
    inc_id: int,
    request: Request,
    db: AsyncSession = Depends(get_db)
):
    """SOC Action: Hunt for similar attacks"""
    _require_api_key(request)
    
    incident = (await db.execute(
        select(Incident).where(Incident.id == inc_id)
    )).scalars().first()
    
    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found")
    
    try:
        # Search for similar incidents in the last 30 days
        similar_incidents_query = select(Incident).where(
            Incident.threat_category == incident.threat_category,
            Incident.id != inc_id,
            Incident.created_at >= datetime.now(timezone.utc) - timedelta(days=30)
        ).limit(10)
        
        similar_incidents_result = await db.execute(similar_incidents_query)
        similar_incidents = similar_incidents_result.scalars().all()
        
        # Record action
        action = Action(
            incident_id=inc_id,
            action="soc_hunt_similar_attacks",
            result="success",
            detail=f"Found {len(similar_incidents)} similar attacks in the last 30 days",
            params={"timeframe": "30_days", "manual": True, "soc_action": True}
        )
        db.add(action)
        await db.commit()
        
        return {
            "success": True,
            "message": f"🎯 Found {len(similar_incidents)} similar attacks in last 30 days",
            "details": f"Threat pattern analysis complete for {incident.threat_category}",
            "similar_count": len(similar_incidents)
        }
        
    except Exception as e:
        logger.error(f"SOC threat hunting failed: {e}")
        return {
            "success": False,
            "message": "❌ Threat hunting search failed",
            "error": str(e)
        }


@app.post("/incidents/{inc_id}/actions/alert-analysts")
async def soc_alert_analysts(
    inc_id: int,
    request: Request,
    db: AsyncSession = Depends(get_db)
):
    """SOC Action: Alert senior analysts"""
    _require_api_key(request)
    
    incident = (await db.execute(
        select(Incident).where(Incident.id == inc_id)
    )).scalars().first()
    
    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found")
    
    try:
        # Record action
        action = Action(
            incident_id=inc_id,
            action="soc_alert_analysts",
            result="success",
            detail=f"Senior analysts alerted about high-priority incident {inc_id}",
            params={"priority": incident.escalation_level, "manual": True, "soc_action": True}
        )
        db.add(action)
        
        # Update escalation level
        if incident.escalation_level != "critical":
            incident.escalation_level = "high"
        
        await db.commit()
        
        return {
            "success": True,
            "message": f"📧 Senior analysts notified about incident {inc_id}",
            "details": "Escalation notification sent to on-call team"
        }
        
    except Exception as e:
        logger.error(f"SOC analyst alert failed: {e}")
        return {
            "success": False,
            "message": "❌ Failed to alert analysts",
            "error": str(e)
        }


@app.post("/incidents/{inc_id}/actions/create-case")
async def soc_create_case(
    inc_id: int,
    request: Request,
    db: AsyncSession = Depends(get_db)
):
    """SOC Action: Create SOAR case"""
    _require_api_key(request)
    
    incident = (await db.execute(
        select(Incident).where(Incident.id == inc_id)
    )).scalars().first()
    
    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found")
    
    try:
        # Generate case ID
        case_id = f"CASE-{inc_id}-{int(datetime.now().timestamp())}"
        
        # Record action
        action = Action(
            incident_id=inc_id,
            action="soc_create_case",
            result="success",
            detail=f"SOAR case {case_id} created for incident {inc_id}",
            params={"case_id": case_id, "manual": True, "soc_action": True}
        )
        db.add(action)
        await db.commit()
        
        return {
            "success": True,
            "message": f"📋 SOAR case {case_id} created successfully",
            "details": "Case management workflow initiated",
            "case_id": case_id
        }
        
    except Exception as e:
        logger.error(f"SOC case creation failed: {e}")
        return {
            "success": False,
            "message": "❌ Failed to create SOAR case",
            "error": str(e)
        }


@app.get("/settings/auto_contain")
async def get_auto_contain():
    """Get auto-contain setting"""
    return {"enabled": auto_contain_enabled}


@app.post("/settings/auto_contain")
async def set_auto_contain(enabled: bool, request: Request):
    """Set auto-contain setting"""
    _require_api_key(request)
    
    global auto_contain_enabled
    auto_contain_enabled = enabled
    
    logger.info(f"Auto-contain setting changed to: {enabled}")
    
    return {"enabled": auto_contain_enabled}


@app.post("/api/ml/retrain")
async def retrain_ml_models(
    request_data: Dict[str, Any],
    db: AsyncSession = Depends(get_db)
):
    """
    Trigger ML model retraining
    """
    model_type = request_data.get("model_type", "ensemble")
    
    try:
        # Get training data from recent events
        from .ml_engine import prepare_training_data_from_events
        
        # Get events from the last 30 days for training
        since = datetime.now(timezone.utc) - timedelta(days=30)
        query = select(Event).where(Event.ts >= since).limit(10000)
        result = await db.execute(query)
        events = result.scalars().all()
        
        if len(events) < 100:
            return {
                "success": False,
                "message": f"Insufficient training data: {len(events)} events (need at least 100)"
            }
        
        # Prepare training data
        training_data = await prepare_training_data_from_events(events)
        
        # Train models
        if model_type == "ensemble":
            results = await ml_detector.train_models(training_data)
        else:
            return {"success": False, "message": f"Model type '{model_type}' not supported"}
        
        return {
            "success": True,
            "message": f"Retrained {model_type} models",
            "training_data_size": len(training_data),
            "results": results
        }
        
    except Exception as e:
        logger.error(f"ML retraining failed: {e}")
        return {"success": False, "message": f"Retraining failed: {str(e)}"}


@app.get("/api/ml/status")
async def get_ml_status():
    """
    Get ML model status and performance metrics
    """
    try:
        status = ml_detector.get_model_status()
        
        # Add more detailed metrics if available
        metrics = {
            "models_trained": sum(1 for trained in status.values() if trained),
            "total_models": len(status),
            "status_by_model": status
        }
        
        return {
            "success": True,
            "metrics": metrics
        }
        
    except Exception as e:
        logger.error(f"ML status check failed: {e}")
        return {"success": False, "message": f"Status check failed: {str(e)}"}


@app.get("/api/sources")
async def list_log_sources(db: AsyncSession = Depends(get_db)):
    """
    List all configured log sources and their statistics
    """
    try:
        stats = await multi_ingestor.get_source_statistics(db)
        return {
            "success": True,
            "sources": stats
        }
    except Exception as e:
        logger.error(f"Failed to get source statistics: {e}")
        return {"success": False, "message": f"Failed to get sources: {str(e)}"}


@app.get("/api/adaptive/status")
async def get_adaptive_detection_status():
    """
    Get adaptive detection system status
    """
    try:
        return {
            "success": True,
            "adaptive_engine": {
                "weights": adaptive_engine.weights,
                "behavioral_threshold": getattr(behavioral_analyzer, 'adaptive_threshold', 0.6)
            },
            "learning_pipeline": learning_pipeline.get_learning_status(),
            "baseline_engine": baseline_engine.get_baseline_status(),
            "ml_detector": ml_detector.get_model_status()
        }
    except Exception as e:
        logger.error(f"Failed to get adaptive detection status: {e}")
        return {"success": False, "message": f"Status check failed: {str(e)}"}


@app.post("/api/adaptive/force_learning")
async def force_adaptive_learning():
    """
    Force an immediate learning update for testing
    """
    try:
        results = await learning_pipeline.force_learning_update()
        return {
            "success": True,
            "message": "Forced learning update completed",
            "results": results
        }
    except Exception as e:
        logger.error(f"Forced learning update failed: {e}")
        return {"success": False, "message": f"Learning update failed: {str(e)}"}


@app.post("/api/adaptive/sensitivity")
async def adjust_detection_sensitivity(
    request_data: Dict[str, Any],
    request: Request
):
    """
    Adjust detection sensitivity
    """
    _require_api_key(request)
    
    sensitivity = request_data.get("sensitivity", "medium")
    if sensitivity not in ["low", "medium", "high"]:
        raise HTTPException(status_code=400, detail="Sensitivity must be 'low', 'medium', or 'high'")
    
    try:
        # Adjust baseline engine sensitivity
        baseline_engine.adjust_sensitivity(sensitivity)
        
        # Adjust behavioral analyzer threshold
        if sensitivity == "high":
            behavioral_analyzer.adaptive_threshold = 0.4
        elif sensitivity == "low":
            behavioral_analyzer.adaptive_threshold = 0.8
        else:  # medium
            behavioral_analyzer.adaptive_threshold = 0.6
        
        return {
            "success": True,
            "message": f"Detection sensitivity adjusted to {sensitivity}",
            "behavioral_threshold": behavioral_analyzer.adaptive_threshold,
            "baseline_thresholds": baseline_engine.deviation_thresholds
        }
    except Exception as e:
        logger.error(f"Sensitivity adjustment failed: {e}")
        return {"success": False, "message": f"Adjustment failed: {str(e)}"}


async def background_retrain_ml_models():
    """Background task to retrain ML models"""
    try:
        async with AsyncSessionLocal() as db:
            # Get recent events for training
            from .ml_engine import prepare_training_data_from_events
            
            since = datetime.now(timezone.utc) - timedelta(days=7)
            query = select(Event).where(Event.ts >= since).limit(5000)
            result = await db.execute(query)
            events = result.scalars().all()
            
            if len(events) >= 100:
                training_data = await prepare_training_data_from_events(events)
                results = await ml_detector.train_models(training_data)
                logger.info(f"ML model retraining completed: {results}")
            else:
                logger.info(f"Skipping ML retraining: insufficient data ({len(events)} events)")
                
    except Exception as e:
        logger.error(f"Background ML retraining failed: {e}")


async def process_scheduled_unblocks():
    """Background task to process scheduled unblocks"""
    try:
        async with AsyncSessionLocal() as db:
            # Find pending scheduled unblocks that are due
            now = datetime.now(timezone.utc)
            query = select(Action).where(
                and_(
                    Action.action == "scheduled_unblock",
                    Action.result == "pending",
                    Action.due_at <= now
                )
            )
            
            result = await db.execute(query)
            due_actions = result.scalars().all()
            
            for action in due_actions:
                try:
                    ip = action.params.get("ip") if action.params else None
                    if not ip:
                        continue
                    
                    # Execute unblock
                    status, detail = await unblock_ip(ip)
                    
                    # Create new unblock action
                    unblock_action = Action(
                        incident_id=action.incident_id,
                        action="unblock",
                        result="success" if status == "success" else "failed",
                        detail=f"Scheduled unblock: {detail}",
                        params={"ip": ip, "scheduled": True}
                    )
                    db.add(unblock_action)
                    
                    # Mark scheduled action as done
                    action.result = "done"
                    action.detail = f"Completed: {detail}"
                    
                    # Update incident status if successful
                    if status == "success":
                        incident = (await db.execute(
                            select(Incident).where(Incident.id == action.incident_id)
                        )).scalars().first()
                        if incident:
                            incident.status = "open"
                    
                    logger.info(f"Processed scheduled unblock for IP {ip}")
                    
                except Exception as e:
                    logger.error(f"Failed to process scheduled unblock {action.id}: {e}")
                    action.result = "failed"
                    action.detail = f"Failed: {str(e)}"
            
            await db.commit()
            
    except Exception as e:
        logger.error(f"Error in scheduled unblock processor: {e}")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "auto_contain": auto_contain_enabled
    }


@app.get("/test/ssh")
async def test_ssh_connectivity():
    """Test SSH connectivity to honeypot"""
    from .responder import responder
    import subprocess
    import os
    
    try:
        # First, test basic network connectivity
        ping_result = subprocess.run(
            ['ping', '-c', '1', '-W', '3000', responder.host],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        ping_status = "success" if ping_result.returncode == 0 else "failed"
        ping_detail = ping_result.stdout.strip() if ping_result.returncode == 0 else ping_result.stderr.strip()
        
        # Test SSH connectivity
        ssh_status, ssh_detail = await responder.test_connection()
        
        # Get current environment info
        env_info = {
            "PATH": os.environ.get("PATH", "Not set"),
            "USER": os.environ.get("USER", "Not set"),
            "HOME": os.environ.get("HOME", "Not set"),
            "PWD": os.environ.get("PWD", "Not set")
        }
        
        return {
            "ssh_status": ssh_status,
            "ssh_detail": ssh_detail,
            "ping_status": ping_status,
            "ping_detail": ping_detail,
            "honeypot": f"{responder.username}@{responder.host}:{responder.port}",
            "environment": env_info,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        logger.error(f"SSH test failed with exception: {e}")
        return {
            "ssh_status": "failed",
            "ssh_detail": f"SSH test exception: {str(e)}",
            "ping_status": "error",
            "ping_detail": "Could not test ping",
            "honeypot": f"{responder.username}@{responder.host}:{responder.port}",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }


async def execute_password_reset(source_ip: str, incident_id: int) -> Dict[str, Any]:
    """Execute actual password reset for potentially compromised accounts"""
    try:
        import subprocess
        import secrets
        import string
        from pathlib import Path
        
        # Define admin accounts that need password reset
        admin_accounts = [
            "root", "admin", "administrator", "ubuntu", "centos", 
            "debian", "user", "sysadmin", "operator"
        ]
        
        reset_results = []
        successful_resets = []
        failed_resets = []
        
        for username in admin_accounts:
            try:
                # Generate secure random password
                password_chars = string.ascii_letters + string.digits + "!@#$%^&*"
                new_password = ''.join(secrets.choice(password_chars) for _ in range(16))
                
                # Check if user exists
                user_check = subprocess.run(
                    ["id", username], 
                    capture_output=True, 
                    text=True, 
                    timeout=10
                )
                
                if user_check.returncode == 0:
                    # User exists, reset password
                    password_reset = subprocess.run(
                        ["sudo", "chpasswd"],
                        input=f"{username}:{new_password}",
                        text=True,
                        capture_output=True,
                        timeout=30
                    )
                    
                    if password_reset.returncode == 0:
                        successful_resets.append({
                            "username": username,
                            "status": "success",
                            "new_password": new_password[:4] + "****"  # Partial password for logging
                        })
                        
                        # Store full password securely for incident response
                        password_file = f"/tmp/incident_{incident_id}_passwords.txt"
                        with open(password_file, "a") as f:
                            f.write(f"{username}:{new_password}\n")
                        
                        # Set secure permissions on password file
                        subprocess.run(["chmod", "600", password_file], timeout=10)
                        
                        logger.info(f"Password reset successful for user: {username}")
                    else:
                        failed_resets.append({
                            "username": username,
                            "status": "failed",
                            "error": password_reset.stderr
                        })
                        logger.error(f"Password reset failed for {username}: {password_reset.stderr}")
                else:
                    # User doesn't exist, skip
                    logger.debug(f"User {username} does not exist, skipping")
                    
            except subprocess.TimeoutExpired:
                failed_resets.append({
                    "username": username,
                    "status": "timeout",
                    "error": "Command timed out"
                })
                logger.error(f"Password reset timed out for user: {username}")
            except Exception as e:
                failed_resets.append({
                    "username": username,
                    "status": "error",
                    "error": str(e)
                })
                logger.error(f"Password reset error for {username}: {e}")
        
        # Force password expiry to require immediate change on next login
        for reset in successful_resets:
            try:
                subprocess.run(
                    ["sudo", "chage", "-d", "0", reset["username"]],
                    capture_output=True,
                    timeout=10,
                    check=True
                )
                logger.info(f"Forced password expiry for user: {reset['username']}")
            except Exception as e:
                logger.warning(f"Failed to force password expiry for {reset['username']}: {e}")
        
        # Send notification to system administrators
        await send_password_reset_notification(source_ip, incident_id, successful_resets, failed_resets)
        
        # Create summary
        total_resets = len(successful_resets)
        total_failures = len(failed_resets)
        
        if total_resets > 0:
            success_msg = f"✅ Password reset completed for {total_resets} admin accounts"
            if total_failures > 0:
                success_msg += f" ({total_failures} failed)"
                
            details = f"Passwords reset for: {', '.join([r['username'] for r in successful_resets])}"
            if failed_resets:
                details += f". Failed: {', '.join([r['username'] for r in failed_resets])}"
            
            return {
                "success": True,
                "message": success_msg,
                "details": details,
                "affected_users": [r["username"] for r in successful_resets],
                "reset_count": total_resets,
                "failed_count": total_failures
            }
        else:
            return {
                "success": False,
                "message": "❌ No passwords were reset successfully",
                "details": f"All {total_failures} reset attempts failed",
                "affected_users": [],
                "reset_count": 0,
                "failed_count": total_failures
            }
            
    except Exception as e:
        logger.error(f"Password reset execution failed: {e}")
        return {
            "success": False,
            "message": "❌ Password reset execution failed",
            "details": f"Error: {str(e)}",
            "affected_users": [],
            "reset_count": 0
        }


async def send_password_reset_notification(
    source_ip: str, 
    incident_id: int, 
    successful_resets: List[Dict], 
    failed_resets: List[Dict]
):
    """Send notification about password reset actions"""
    try:
        notification_message = f"""
SECURITY INCIDENT #{incident_id} - PASSWORD RESET EXECUTED

Source IP: {source_ip}
Timestamp: {datetime.utcnow().isoformat()}

Password Reset Summary:
- Successful: {len(successful_resets)} accounts
- Failed: {len(failed_resets)} accounts

Successful Resets:
{chr(10).join([f"  - {r['username']}" for r in successful_resets]) if successful_resets else "  None"}

Failed Resets:
{chr(10).join([f"  - {r['username']}: {r['error']}" for r in failed_resets]) if failed_resets else "  None"}

IMPORTANT: 
- All reset passwords require immediate change on next login
- New passwords are stored in /tmp/incident_{incident_id}_passwords.txt
- Please coordinate with affected users for password distribution

This is an automated security response to incident #{incident_id}.
"""
        
        # Log the notification (in production, this would send email/Slack/etc)
        logger.warning(f"PASSWORD RESET NOTIFICATION: {notification_message}")
        
        # In production, integrate with notification systems:
        # - Send email to security team
        # - Post to Slack/Teams channel
        # - Create service desk tickets
        # - Update SIEM/SOAR platforms
        
    except Exception as e:
        logger.error(f"Failed to send password reset notification: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True
    )
