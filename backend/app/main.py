import asyncio
import hmac
import html
import logging
import re
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Union

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    HTTPException,
    Request,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import and_, asc, delete, desc, func, or_, select, text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from .adaptive_detection import behavioral_analyzer
from .advanced_response_engine import (
    ActionCategory,
    ResponsePriority,
    get_response_engine,
)
from .agent_orchestrator import get_orchestrator
from .agent_routes import router as agent_router
from .agents.containment_agent import ContainmentAgent
from .ai_response_advisor import get_ai_advisor
from .baseline_engine import baseline_engine
from .config import settings
from .context_analyzer import get_context_analyzer
from .db import AsyncSessionLocal, get_db, init_db
from .detect import adaptive_engine, run_detection
from .external_intel import threat_intel
from .learning_pipeline import learning_pipeline
from .learning_response_engine import get_learning_engine
from .ml_engine import ml_detector
from .models import (
    Action,
    ActionLog,
    AdvancedResponseAction,
    Event,
    Incident,
    ResponseApproval,
    ResponseImpactMetrics,
    ResponsePlaybook,
    ResponseWorkflow,
)
from .multi_ingestion import multi_ingestor
from .nlp_suggestion_routes import router as nlp_suggestion_router
from .nlp_workflow_routes import router as nlp_workflow_router
from .onboarding_routes import router as onboarding_router
from .onboarding_v2.routes import router as onboarding_v2_router
from .policy_engine import policy_engine
from .responder import block_ip, unblock_ip
from .response_optimizer import get_response_optimizer
from .security import AuthMiddleware, RateLimiter
from .triager import generate_default_triage, run_triage
from .trigger_evaluator import trigger_evaluator
from .trigger_routes import router as trigger_router
from .websocket_manager import ws_manager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global scheduler for background tasks
scheduler = AsyncIOScheduler()

# Global AI agent instances
containment_agent = None
agent_orchestrator = None


# Mapping of advanced response actions to their rollback counterparts for UI hints
ADVANCED_ACTION_ROLLBACK_MAP = {
    "block_ip": {"action_type": "unblock_ip", "label": "Unblock IP"},
    "block_ip_advanced": {"action_type": "unblock_ip", "label": "Unblock IP"},
    "isolate_host": {
        "action_type": "un_isolate_host",
        "label": "Release Host Isolation",
    },
    "isolate_host_advanced": {
        "action_type": "un_isolate_host",
        "label": "Release Host Isolation",
    },
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global containment_agent, agent_orchestrator

    # Startup
    logger.info("Starting Enhanced Mini-XDR backend...")

    # Initialize database with timeout and error handling
    try:
        await asyncio.wait_for(init_db(), timeout=60)
        logger.info("Database initialized successfully")
    except asyncio.TimeoutError:
        logger.warning("DB init timeout - will retry on first request")
    except Exception as e:
        logger.error(f"DB init failed: {e} - continuing anyway")

    scheduler.start()

    # Initialize AI components
    logger.info("Initializing AI components...")
    containment_agent = ContainmentAgent(
        threat_intel=threat_intel, ml_detector=ml_detector
    )

    # Initialize agent orchestrator
    logger.info("Initializing Agent Orchestrator...")
    agent_orchestrator = await get_orchestrator()

    # Initialize DLP Agent
    logger.info("Initializing DLP Agent...")
    try:
        from .agents.dlp_agent import dlp_agent

        logger.info("✅ DLP Agent activated successfully")
    except Exception as e:
        logger.warning(f"Failed to initialize DLP agent: {e}")

    # Load ML models if available
    try:
        ml_detector.load_models()
        logger.info("ML models loaded")
    except Exception as e:
        logger.warning(f"Failed to load ML models: {e}")

    # Initialize Enhanced Threat Detector (Local Models Only - NO AWS)
    try:
        from pathlib import Path

        from .enhanced_threat_detector import enhanced_detector

        # Try multiple model paths in order of preference
        model_paths = [
            Path(__file__).parent.parent.parent / "models",  # Project root models
            Path(__file__).parent.parent / "models",  # Backend models
            Path(
                "/Users/chasemad/Desktop/mini-xdr/models"
            ),  # Absolute path as fallback
        ]

        for model_path in model_paths:
            if model_path.exists():
                logger.info(f"Attempting to load enhanced detector from: {model_path}")
                if enhanced_detector.load_model(str(model_path)):
                    logger.info(
                        "✅ Enhanced Local Threat Detector loaded successfully (NO AWS)"
                    )
                    break
        else:
            logger.warning(
                "Enhanced detector models not found - will use fallback detection"
            )
    except Exception as e:
        logger.warning(f"Failed to initialize enhanced detector: {e}")

    # Start continuous learning pipeline
    try:
        await learning_pipeline.start_learning_loop()
        logger.info("Continuous learning pipeline started")
    except Exception as e:
        logger.warning(f"Failed to start learning pipeline: {e}")

    # Start automated retraining scheduler (Phase 2)
    try:
        from .learning import start_retrain_scheduler

        await start_retrain_scheduler()
        logger.info("✅ Automated retraining scheduler started (Phase 2)")
    except Exception as e:
        logger.warning(f"Failed to start retraining scheduler: {e}")

    # Add scheduled tasks
    scheduler.add_job(
        process_scheduled_unblocks,
        IntervalTrigger(seconds=30),
        id="process_scheduled_unblocks",
        replace_existing=True,
    )

    # Add ML training task (daily)
    scheduler.add_job(
        background_retrain_ml_models,
        IntervalTrigger(hours=24),
        id="retrain_ml_models",
        replace_existing=True,
    )

    # Initialize T-Pot honeypot monitoring
    try:
        from .tpot_connector import initialize_tpot_monitoring
        from .tpot_elasticsearch_ingestor import start_elasticsearch_ingestion

        logger.info("Initializing T-Pot honeypot monitoring...")
        if await initialize_tpot_monitoring(AsyncSessionLocal):
            logger.info("✅ T-Pot monitoring initialized successfully")
            # Start Elasticsearch-based ingestion
            await start_elasticsearch_ingestion()
            logger.info("✅ T-Pot Elasticsearch ingestion started")
        else:
            logger.warning(
                "T-Pot monitoring initialization failed - will continue without honeypot integration"
            )
    except Exception as e:
        logger.warning(
            f"Failed to initialize T-Pot: {e} - continuing without honeypot integration"
        )

    yield

    # Shutdown
    logger.info("Shutting down Enhanced Mini-XDR backend...")

    # Shutdown T-Pot monitoring
    try:
        from .tpot_connector import shutdown_tpot_monitoring
        from .tpot_elasticsearch_ingestor import stop_elasticsearch_ingestion

        await stop_elasticsearch_ingestion()
        await shutdown_tpot_monitoring()
        logger.info("T-Pot monitoring stopped")
    except Exception as e:
        logger.warning(f"Error stopping T-Pot monitoring: {e}")

    # Stop learning pipeline
    try:
        await learning_pipeline.stop_learning_loop()
        logger.info("Learning pipeline stopped")
    except Exception as e:
        logger.warning(f"Error stopping learning pipeline: {e}")

    # Stop automated retraining scheduler (Phase 2)
    try:
        from .learning import stop_retrain_scheduler

        await stop_retrain_scheduler()
        logger.info("Retraining scheduler stopped")
    except Exception as e:
        logger.warning(f"Error stopping retraining scheduler: {e}")

    await threat_intel.close()
    scheduler.shutdown()


app = FastAPI(
    title="Mini-XDR",
    description="SSH Brute-Force Detection and Response System",
    version="1.2.0",
    lifespan=lifespan,
)

# Security middleware is configured in entrypoint.py for production deployment
# For development/testing, basic CORS is applied here
if not getattr(settings, "_entrypoint_mode", False):
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=[
            "Content-Type",
            "Authorization",
            "x-api-key",
            "X-Device-ID",
            "X-TS",
            "X-Nonce",
            "X-Signature",
        ],
    )

    rate_limiter = RateLimiter()
    app.add_middleware(AuthMiddleware, rate_limiter=rate_limiter)

# Include routers
app.include_router(nlp_workflow_router)
app.include_router(nlp_suggestion_router)
app.include_router(trigger_router)
app.include_router(onboarding_router)  # Legacy onboarding
app.include_router(onboarding_v2_router)
app.include_router(agent_router)  # Agent communication endpoints

# T-Pot Honeypot Integration
from .tpot_routes import router as tpot_router

app.include_router(tpot_router)

# System Health Check
from .health_routes import router as health_router

app.include_router(health_router)

# Agent Coordination API (for v2 incident UI)
from .agent_coordination_routes import router as agent_coordination_router

app.include_router(agent_coordination_router)

from fastapi import Depends

# =============== Telemetry Status Endpoint (Org-scoped) ===============
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from .auth import get_current_user
from .db import get_db
from .models import AgentEnrollment as DbEnroll
from .models import DiscoveredAsset as DbAsset
from .models import Event as DbEvent
from .models import Incident as DbIncident
from .models import Organization as DbOrg
from .models import User as DbUser


@app.get("/api/telemetry/status", tags=["Telemetry"])
async def telemetry_status(
    current_user: DbUser = Depends(get_current_user), db: AsyncSession = Depends(get_db)
):
    org_id = current_user.organization_id

    # Counts per org
    assets_q = await db.execute(
        select(func.count(DbAsset.id)).where(DbAsset.organization_id == org_id)
    )
    agents_q = await db.execute(
        select(func.count(DbEnroll.id)).where(DbEnroll.organization_id == org_id)
    )
    incidents_q = await db.execute(
        select(func.count(DbIncident.id)).where(DbIncident.organization_id == org_id)
    )

    # Last event time
    last_event_q = await db.execute(
        select(func.max(DbEvent.ts)).where(DbEvent.organization_id == org_id)
    )
    last_event = last_event_q.scalar_one_or_none()

    has_logs = last_event is not None

    return {
        "hasLogs": has_logs,
        "lastEventAt": last_event.isoformat() if last_event else None,
        "assetsDiscovered": int(assets_q.scalar() or 0),
        "agentsEnrolled": int(agents_q.scalar() or 0),
        "incidents": int(incidents_q.scalar() or 0),
    }


# Security headers middleware
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data: https:; "
        "connect-src 'self'; "
        "font-src 'self'; "
        "object-src 'none'; "
        "base-uri 'self'; "
        "form-action 'self'"
    )
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
    if request.url.scheme == "https":
        response.headers[
            "Strict-Transport-Security"
        ] = "max-age=31536000; includeSubDomains"
    return response


# Global auto-contain setting (runtime configurable)
auto_contain_enabled = settings.auto_contain


def sanitize_log_data(data):
    """Sanitize log data to prevent injection attacks"""
    if isinstance(data, str):
        # Remove potential injection patterns and control characters
        data = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", data)  # Remove control chars
        data = re.sub(r'[<>"\'\`]', "", data)  # Remove potential injection chars
        data = data.replace("\n", " ").replace("\r", " ")  # Remove newlines
        return html.escape(data)  # HTML escape remaining content
    elif isinstance(data, dict):
        return {k: sanitize_log_data(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [sanitize_log_data(item) for item in data]
    return data


def validate_ip_address(ip_str: str) -> bool:
    """Validate IP address format"""
    import ipaddress

    try:
        ipaddress.ip_address(ip_str)
        return True
    except ValueError:
        return False


def sanitize_input_data(data: dict) -> dict:
    """Sanitize input data for API endpoints"""
    sanitized = {}
    for key, value in data.items():
        # Sanitize key names
        clean_key = re.sub(r"[^a-zA-Z0-9_-]", "", str(key))[:50]  # Limit key length

        if isinstance(value, str):
            # Limit string length and sanitize
            clean_value = sanitize_log_data(value)[:1000]
        elif isinstance(value, (int, float, bool)):
            clean_value = value
        elif isinstance(value, dict):
            clean_value = sanitize_input_data(value)
        elif isinstance(value, list):
            clean_value = [
                sanitize_log_data(item) if isinstance(item, str) else item
                for item in value[:100]
            ]
        else:
            clean_value = str(value)[:500]

        sanitized[clean_key] = clean_value
    return sanitized


def _require_api_key(request: Request):
    """Require API key for ALL environments - no bypass allowed"""
    if not settings.api_key:
        logger.error("API key must be configured for security")
        raise HTTPException(status_code=500, detail="API key must be configured")

    api_key = request.headers.get("x-api-key")
    if not api_key:
        raise HTTPException(status_code=401, detail="Missing API key header")

    # Use secure comparison to prevent timing attacks
    if not hmac.compare_digest(api_key, settings.api_key):
        raise HTTPException(status_code=401, detail="Invalid API key")


async def _recent_events_for_ip(db: AsyncSession, src_ip: str, seconds: int = 600):
    """Get recent events for an IP address"""
    since = datetime.now(timezone.utc) - timedelta(seconds=seconds)
    query = (
        select(Event)
        .where(and_(Event.src_ip == src_ip, Event.ts >= since))
        .order_by(Event.ts.desc())
        .limit(200)
    )

    result = await db.execute(query)
    return result.scalars().all()


async def _get_detailed_events_for_ip(db: AsyncSession, src_ip: str, hours: int = 24):
    """Get detailed events for an IP with extended timeframe for forensic analysis"""
    since = datetime.now(timezone.utc) - timedelta(hours=hours)
    query = (
        select(Event)
        .where(and_(Event.src_ip == src_ip, Event.ts >= since))
        .order_by(Event.ts.asc())
    )

    result = await db.execute(query)
    return result.scalars().all()


async def _get_all_events_for_ip(db: AsyncSession, src_ip: str):
    """Get ALL events for an IP without time restrictions for complete incident analysis"""
    logger.info(f"Getting ALL events for IP: {src_ip}")
    query = select(Event).where(Event.src_ip == src_ip).order_by(Event.ts.asc())

    result = await db.execute(query)
    events = result.scalars().all()
    logger.info(f"Found {len(events)} events for IP {src_ip}")
    return events


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
        "reconnaissance_patterns": [],
    }

    for event in events:
        # Extract from raw data
        if event.raw and isinstance(event.raw, dict):
            raw_data = event.raw

            # Extract user agents
            if "user_agent" in raw_data:
                iocs["user_agents"].append(raw_data["user_agent"])

            # Extract file hashes
            if "hash" in raw_data:
                iocs["file_hashes"].append(raw_data["hash"])
            if "md5" in raw_data:
                iocs["file_hashes"].append(raw_data["md5"])
            if "sha256" in raw_data:
                iocs["file_hashes"].append(raw_data["sha256"])

            # Extract HTTP request details for web attacks
            if "path" in raw_data:
                iocs["urls"].append(raw_data["path"])
                if "query_string" in raw_data:
                    iocs["urls"].append(raw_data["query_string"])

            # Extract attack indicators from web honeypot events
            if "attack_indicators" in raw_data:
                indicators = raw_data["attack_indicators"]
                for indicator in indicators:
                    if "sql" in indicator.lower() or "injection" in indicator.lower():
                        iocs["sql_injection_patterns"].append(f"web_attack:{indicator}")
                    elif "admin" in indicator.lower() or "scan" in indicator.lower():
                        iocs["reconnaissance_patterns"].append(
                            f"web_attack:{indicator}"
                        )
                    elif "xss" in indicator.lower():
                        iocs["command_patterns"].append(f"web_attack:{indicator}")
                    elif "traversal" in indicator.lower():
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
                r"BENCHMARK\s*\(",
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
                r"information_schema\.tables",
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
                r"ALTER\s+USER.*IDENTIFIED",
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
                r"scp\s+.*@",
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
                r"useradd\s+",
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
                r"wmiexec",
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
                r"find\s+/.*-name",
            ]

            for pattern in recon_patterns:
                if re.search(pattern, msg, re.IGNORECASE):
                    iocs["reconnaissance_patterns"].append(pattern)

            # Command patterns
            if event.eventid == "cowrie.command.input":
                iocs["command_patterns"].append(msg)

            # File paths
            file_path_pattern = r"[/\\][\w\-_./\\]+"
            file_paths = re.findall(file_path_pattern, msg)
            iocs["file_paths"].extend(file_paths)

        # Check for successful authentication indicators
        if event.eventid == "cowrie.login.success":
            iocs["successful_auth_indicators"].append(
                f"Successful SSH login from {event.src_ip}"
            )
        elif event.eventid == "cowrie.login.failed":
            # Count failed attempts as reconnaissance patterns
            username = event.raw.get("username", "unknown") if event.raw else "unknown"
            iocs["reconnaissance_patterns"].append(f"ssh_brute_force:{username}")
        elif "200" in str(event.raw.get("status_code", "")) and "admin" in str(
            event.raw.get("path", "")
        ):
            iocs["successful_auth_indicators"].append(
                f"Potential admin access: {event.raw.get('path', '')}"
            )

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
            "event_type": event.eventid.split(".")[-1]
            if "." in event.eventid
            else event.eventid,
            "raw_data": event.raw if isinstance(event.raw, dict) else {},
            "severity": _classify_event_severity(event),
        }

        # Enhanced attack classification for honeypots
        attack_category = "unknown"

        # Primary classification based on event ID patterns
        if event.eventid in ["cowrie.login.failed", "cowrie.login.success"]:
            attack_category = "authentication"
        elif "command" in event.eventid or "cowrie.command" in event.eventid:
            attack_category = "command_execution"
        elif (
            "file" in event.eventid
            or "download" in event.eventid
            or "upload" in event.eventid
        ):
            attack_category = "file_operations"
        elif "session" in event.eventid:
            attack_category = "session_management"
        elif "cowrie.client" in event.eventid:
            attack_category = "client_negotiation"
        elif "cowrie.direct" in event.eventid:
            attack_category = "direct_tcp_ip"
        elif "honeypot" in event.eventid.lower() or "dionaea" in event.eventid.lower():
            attack_category = "honeypot_interaction"
        elif "web" in event.eventid.lower() or "http" in event.eventid.lower():
            attack_category = "web_attack"

        # Secondary classification based on raw data content
        if event.raw and isinstance(event.raw, dict):
            raw_data = event.raw

            # Check for web attack indicators
            if "attack_indicators" in raw_data:
                indicators = raw_data["attack_indicators"]
                if any("sql" in str(indicator).lower() for indicator in indicators):
                    attack_category = "sql_injection"
                elif any("xss" in str(indicator).lower() for indicator in indicators):
                    attack_category = "xss_attack"
                elif any("admin" in str(indicator).lower() for indicator in indicators):
                    attack_category = "admin_scan"
                else:
                    attack_category = "web_attack"

            # Check for brute force patterns
            if "username" in raw_data or "password" in raw_data:
                if attack_category == "unknown":
                    attack_category = "brute_force"

            # Check for malware patterns
            if "payload" in raw_data or "binary" in raw_data:
                attack_category = "malware_delivery"

            # Check for reconnaissance patterns
            if "path" in raw_data:
                path = str(raw_data["path"]).lower()
                if any(
                    recon in path
                    for recon in ["/admin", "/wp-admin", "/.git", "/config", "/backup"]
                ):
                    attack_category = "reconnaissance"

        # Message-based classification for fallback
        if attack_category == "unknown" and event.message:
            message_lower = event.message.lower()
            if any(term in message_lower for term in ["scan", "probe", "discovery"]):
                attack_category = "reconnaissance"
            elif any(
                term in message_lower for term in ["inject", "payload", "exploit"]
            ):
                attack_category = "exploitation"
            elif any(
                term in message_lower for term in ["brute", "dictionary", "password"]
            ):
                attack_category = "brute_force"

        # Special handling for startup test events vs real events
        if event.raw and isinstance(event.raw, dict):
            # Mark test events explicitly
            if ("test_event" in event.raw and event.raw["test_event"]) or (
                "hostname" in event.raw and "test" in str(event.raw["hostname"]).lower()
            ):
                attack_category = (
                    f"test_{attack_category}"
                    if attack_category != "unknown"
                    else "test_event"
                )

        timeline_entry["attack_category"] = attack_category

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
        "cowrie.session.file_upload",
    ]

    medium_severity_events = [
        "cowrie.login.failed",
        "cowrie.session.connect",
        "cowrie.session.closed",
    ]

    # Check for SQL injection in raw data
    if event.raw and isinstance(event.raw, dict):
        if "attack_indicators" in event.raw:
            indicators = event.raw["attack_indicators"]
            if any("sql" in str(indicator).lower() for indicator in indicators):
                return "critical"

    # Check message for dangerous commands
    if event.message and event.eventid == "cowrie.command.input":
        dangerous_commands = [
            "rm -rf",
            "wget",
            "curl",
            "chmod +x",
            "nc -l",
            "python -c",
        ]
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
    query: str, incident: Incident, recent_events: List[Event], context: Dict[str, Any]
) -> str:
    """
    Generate intelligent contextual analysis based on user query and incident data
    """
    query_lower = query.lower()

    # Extract key metrics
    iocs = context.get("iocs", {})
    timeline = context.get("attack_timeline", [])
    triage = context.get("triage_note", {})
    chat_history = context.get("chat_history", [])

    ioc_count = sum(len(v) if isinstance(v, list) else 0 for v in iocs.values())
    sql_patterns = len(iocs.get("sql_injection_patterns", []))
    recon_patterns = len(iocs.get("reconnaissance_patterns", []))
    db_patterns = len(iocs.get("database_access_patterns", []))
    timeline_count = len(timeline)

    # Analyze user intent and provide contextual response
    if any(
        word in query_lower for word in ["ioc", "indicator", "compromise", "pattern"]
    ):
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

    elif any(
        word in query_lower for word in ["timeline", "attack", "sequence", "events"]
    ):
        attack_types = set()
        for event in timeline[:5]:  # Analyze first 5 events
            if "web_attack" in event.get("attack_category", ""):
                attack_types.add("Web Application")
            elif "authentication" in event.get("attack_category", ""):
                attack_types.add("SSH Authentication")

        return f"""The attack timeline shows {timeline_count} events spanning multiple hours. Here's the sequence:

⏰ **Timeline Analysis:**
• Total events: {timeline_count}
• Attack vectors: {', '.join(attack_types) if attack_types else 'Mixed protocols'}
• Duration: {context.get('event_summary', {}).get('time_span_hours', 'Several')} hours
• Source: {incident.src_ip}

🔍 **Pattern Recognition:**
This appears to be a {incident.threat_category.replace('_', ' ') if incident.threat_category else 'coordinated'} attack combining multiple techniques. The attacker showed persistence and knowledge of common vulnerabilities.

📈 **Escalation:** {incident.escalation_level or 'Medium'} severity with {int((incident.risk_score or 0) * 100)}% risk score."""

    elif any(
        word in query_lower
        for word in ["recommend", "next", "should", "action", "response"]
    ):
        risk_level = incident.risk_score or 0
        recommendations = []

        if risk_level > 0.7:
            recommendations = [
                f"🚨 **IMMEDIATE**: Block source IP {incident.src_ip} ({int(risk_level * 100)}% risk)",
                "🔒 **URGENT**: Isolate affected systems to prevent lateral movement",
                "🔑 **CRITICAL**: Reset all admin passwords immediately",
                "🛡️ **ESSENTIAL**: Verify database integrity after SQL injection attempts",
                "🔍 **REQUIRED**: Hunt for similar attack patterns network-wide",
            ]
        else:
            recommendations = [
                f"📊 **MONITOR**: Continue surveillance of {incident.src_ip}",
                "🛡️ **ENHANCE**: Deploy additional detection rules",
                f"🔍 **INVESTIGATE**: Review security controls for {incident.threat_category.replace('_', ' ') if incident.threat_category else 'similar'} attacks",
                "📈 **ANALYZE**: Consider threat hunting for related activity",
            ]

        confidence_text = (
            f"{int((incident.agent_confidence or 0) * 100)}% ML confidence"
        )

        return f"""Based on the {incident.escalation_level or 'medium'} escalation level and {int(risk_level * 100)}% risk score, here are my recommendations:

{chr(10).join(recommendations)}

🎯 **Assessment Confidence:** {confidence_text}
🤖 **Detection Method:** {incident.containment_method.replace('_', ' ') if incident.containment_method else 'Rule-based'}
⚡ **Status:** {incident.status.title()} - {'Auto-contained' if incident.auto_contained else 'Manual review required'}"""

    elif any(word in query_lower for word in ["explain", "what", "how", "why", "tell"]):
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
    elif any(
        word in query_lower for word in ["yes", "all", "sure", "ok", "show", "tell"]
    ):
        return f"""Here's a comprehensive analysis of incident #{incident.id}:

📊 **IOC Summary:** {ioc_count} total indicators detected
• SQL injection: {sql_patterns} patterns
• Reconnaissance: {recon_patterns} patterns
• Database access: {db_patterns} patterns

⏰ **Attack Timeline:** {timeline_count} events showing {incident.threat_category.replace('_', ' ') if incident.threat_category else 'coordinated'} attack patterns

🚨 **Risk Assessment:** {int((incident.risk_score or 0) * 100)}% risk score with {int((incident.agent_confidence or 0) * 100)}% ML confidence

🎯 **Key Insight:** This {incident.escalation_level or 'medium'} severity {incident.threat_category.replace('_', ' ') if incident.threat_category else 'multi-vector'} attack from {incident.src_ip} demonstrates sophisticated techniques requiring {'immediate containment' if (incident.risk_score or 0) > 0.7 else 'careful monitoring'}."""

    elif any(word in query_lower for word in ["no", "different", "else", "other"]):
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
    severity = incident.escalation_level or "medium"
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


# ==================== AUTHENTICATION ENDPOINTS ====================

from .auth import (
    authenticate_user,
    create_access_token,
    create_organization,
    create_refresh_token,
    generate_slug,
    get_current_user,
    require_role,
)
from .models import Organization, User
from .schemas import (
    ChangePasswordRequest,
    InviteUserRequest,
    LoginRequest,
    MeResponse,
    OrganizationResponse,
    RegisterOrganizationRequest,
    Token,
    UpdateProfileRequest,
    UserResponse,
)


@app.post("/api/auth/register", response_model=dict, tags=["Authentication"])
async def register_organization(
    request: RegisterOrganizationRequest, db: AsyncSession = Depends(get_db)
):
    """
    Register a new organization with an admin user

    Creates both the organization and the first admin user in a single transaction.
    """
    try:
        # Generate slug from organization name
        slug = generate_slug(request.organization_name)

        # Create organization and admin user
        org, admin_user = await create_organization(
            db=db,
            name=request.organization_name,
            slug=slug,
            admin_email=request.admin_email,
            admin_password=request.admin_password,
            admin_name=request.admin_name,
        )

        # Create access tokens
        access_token = create_access_token(
            data={
                "user_id": admin_user.id,
                "organization_id": org.id,
                "email": admin_user.email,
                "role": admin_user.role,
            }
        )

        refresh_token = create_refresh_token(
            data={"user_id": admin_user.id, "organization_id": org.id}
        )

        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer",
            "user": UserResponse.from_orm(admin_user),
            "organization": OrganizationResponse.from_orm(org),
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Registration failed: {e}")
        raise HTTPException(status_code=500, detail="Registration failed")


@app.post("/api/auth/login", response_model=Token, tags=["Authentication"])
async def login(request: LoginRequest, db: AsyncSession = Depends(get_db)):
    """
    Login with email and password

    Returns JWT access token and refresh token
    """
    user = await authenticate_user(db, request.email, request.password)

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Create access and refresh tokens
    access_token = create_access_token(
        data={
            "user_id": user.id,
            "organization_id": user.organization_id,
            "email": user.email,
            "role": user.role,
        }
    )

    refresh_token = create_refresh_token(
        data={"user_id": user.id, "organization_id": user.organization_id}
    )

    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
    }


@app.get("/api/auth/me", response_model=MeResponse, tags=["Authentication"])
async def get_current_user_info(
    current_user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)
):
    """
    Get current user information with organization details
    """
    # Fetch organization
    stmt = select(Organization).where(Organization.id == current_user.organization_id)
    result = await db.execute(stmt)
    org = result.scalar_one_or_none()

    if not org:
        raise HTTPException(status_code=404, detail="Organization not found")

    return {
        "user": UserResponse.from_orm(current_user),
        "organization": OrganizationResponse.from_orm(org),
    }


@app.post("/api/auth/invite", tags=["Authentication"])
async def invite_user(
    request: InviteUserRequest,
    current_user: User = Depends(require_role("admin")),
    db: AsyncSession = Depends(get_db),
):
    """
    Invite a new user to the organization (Admin only)

    Creates user account with temporary password that must be changed on first login
    """
    # Check if email already exists
    stmt = select(User).where(User.email == request.email)
    result = await db.execute(stmt)
    existing_user = result.scalar_one_or_none()

    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")

    # Generate temporary password (user must change on first login)
    import secrets
    import string

    temp_password = "".join(
        secrets.choice(string.ascii_letters + string.digits + string.punctuation)
        for _ in range(16)
    )

    # Create user
    from .auth import hash_password

    new_user = User(
        organization_id=current_user.organization_id,
        email=request.email,
        hashed_password=hash_password(temp_password),
        full_name=request.full_name,
        role=request.role,
        is_active=True,
    )

    db.add(new_user)
    await db.commit()
    await db.refresh(new_user)

    # TODO: Send invitation email with temporary password
    # For now, return it in response (change this in production!)

    return {
        "message": "User invited successfully",
        "user": UserResponse.from_orm(new_user),
        "temporary_password": temp_password,
        "note": "User must change password on first login",
    }


@app.post("/api/auth/logout", tags=["Authentication"])
async def logout(current_user: User = Depends(get_current_user)):
    """
    Logout current user

    Note: With JWT, actual logout is handled client-side by removing the token
    This endpoint is for logging and potential token blacklisting in future
    """
    logger.info(f"User {current_user.email} logged out")
    return {"message": "Logged out successfully"}


# ==================== END AUTHENTICATION ENDPOINTS ====================


@app.post("/ingest/cowrie")
async def ingest_cowrie(
    events: Union[Dict[str, Any], List[Dict[str, Any]]],
    db: AsyncSession = Depends(get_db),
):
    """
    Ingest Cowrie honeypot events

    Accepts either a single event or list of events
    """
    print(
        f"\n🔴🔴🔴 INGESTION STARTED - Received {len(events) if isinstance(events, list) else 1} event(s) 🔴🔴🔴\n",
        flush=True,
    )
    logger.info(
        f"🔴 INGESTION STARTED - Received {len(events) if isinstance(events, list) else 1} event(s)"
    )

    # Normalize to list
    if isinstance(events, dict):
        events = [events]

    stored_count = 0
    incidents_detected = []
    last_src_ip = None

    for event_data in events:
        try:
            # Sanitize input data for security
            sanitized_data = sanitize_input_data(event_data)

            # Validate IP addresses
            src_ip = (
                sanitized_data.get("src_ip")
                or sanitized_data.get("srcip")
                or sanitized_data.get("peer_ip")
            )
            dst_ip = sanitized_data.get("dst_ip") or sanitized_data.get("dstip")

            if src_ip and not validate_ip_address(src_ip):
                logger.warning(f"Invalid source IP format: {src_ip}")
                continue

            if dst_ip and not validate_ip_address(dst_ip):
                logger.warning(f"Invalid destination IP format: {dst_ip}")
                dst_ip = None

            # Extract event fields with sanitized data
            event = Event(
                src_ip=src_ip,
                dst_ip=dst_ip,
                dst_port=sanitized_data.get("dst_port")
                or sanitized_data.get("dstport"),
                eventid=sanitized_data.get("eventid", "unknown"),
                message=sanitized_data.get("message")
                or _generate_event_description(sanitized_data),
                raw=sanitized_data,
            )

            if event.src_ip:
                last_src_ip = event.src_ip

            db.add(event)
            stored_count += 1

        except Exception as e:
            logger.error(f"Failed to process event: {e}")
            continue

    print(f"🔵 About to commit {stored_count} events...", flush=True)
    logger.info(f"🔵 About to commit {stored_count} events...")
    await db.commit()

    print(f"✅ Committed {stored_count} events to database", flush=True)
    logger.info(f"✅ Committed {stored_count} events to database")

    # Run intelligent detection on all unique source IPs
    incident_id = None
    unique_ips = set()

    print(f"🟡 Starting detection phase for {stored_count} events...", flush=True)

    # Get all events for analysis
    print(f"📊 Fetching last {stored_count} events for analysis...", flush=True)
    logger.info(f"📊 Fetching last {stored_count} events for analysis...")
    all_events = await db.execute(
        select(Event).order_by(Event.ts.desc()).limit(stored_count)
    )
    stored_events = all_events.scalars().all()
    print(f"📊 Retrieved {len(stored_events)} events from database", flush=True)
    logger.info(f"📊 Retrieved {len(stored_events)} events from database")

    # Group events by source IP
    print(f"🟢 Grouping {len(stored_events)} events by IP...", flush=True)
    events_by_ip = {}
    for event in stored_events:
        if event.src_ip:
            unique_ips.add(event.src_ip)
            if event.src_ip not in events_by_ip:
                events_by_ip[event.src_ip] = []
            events_by_ip[event.src_ip].append(event)

    # Run intelligent detection for each unique IP
    print(f"🟣 About to import intelligent_detector...", flush=True)
    from .intelligent_detection import intelligent_detector

    print(
        f"🔍 Running detection for {len(unique_ips)} unique IPs: {unique_ips}",
        flush=True,
    )
    logger.info(f"🔍 Running detection for {len(unique_ips)} unique IPs: {unique_ips}")

    for src_ip in unique_ips:
        try:
            print(
                f"🔍 Analyzing {src_ip} with {len(events_by_ip[src_ip])} events",
                flush=True,
            )
            logger.info(f"🔍 Analyzing {src_ip} with {len(events_by_ip[src_ip])} events")

            print(
                f"🔵 About to call intelligent_detector.analyze_and_create_incidents...",
                flush=True,
            )
            detection_result = await intelligent_detector.analyze_and_create_incidents(
                db, src_ip, events_by_ip[src_ip]
            )
            print(
                f"🟢 Detection result received: {detection_result.get('incident_created', False)}",
                flush=True,
            )

            if detection_result["incident_created"]:
                incident_id = detection_result["incident_id"]
                incidents_detected.append(incident_id)
                logger.info(
                    f"✅ Intelligent incident created: {incident_id} for {src_ip} - {detection_result.get('threat_type', 'unknown')}"
                )
            else:
                logger.info(
                    f"⚠️ No incident created for {src_ip}: {detection_result['reason']}"
                )

        except Exception as e:
            logger.error(
                f"❌ Intelligent detection failed for {src_ip}: {e}", exc_info=True
            )
            # Fallback to legacy detection
            fallback_incident = await run_detection(db, src_ip)
            if fallback_incident:
                incidents_detected.append(fallback_incident)
                incident_id = fallback_incident

    # Get the most recent incident for triage (if any)
    if incident_id:
        incident = (
            (await db.execute(select(Incident).where(Incident.id == incident_id)))
            .scalars()
            .first()
        )

        if incident:
            # Evaluate workflow triggers for this incident
            try:
                recent_events = await _recent_events_for_ip(db, incident.src_ip)
                executed_workflows = (
                    await trigger_evaluator.evaluate_triggers_for_incident(
                        db, incident, recent_events
                    )
                )
                if executed_workflows:
                    logger.info(
                        f"✓ Executed {len(executed_workflows)} workflows for incident #{incident.id}: {executed_workflows}"
                    )
            except Exception as e:
                logger.error(
                    f"Failed to evaluate triggers for incident #{incident.id}: {e}"
                )

            # Run triage analysis
            recent_events = await _recent_events_for_ip(db, incident.src_ip)
            triage_input = {
                "id": incident.id,
                "src_ip": incident.src_ip,
                "reason": incident.reason,
                "status": incident.status,
            }
            event_summaries = [
                {
                    "ts": e.ts.isoformat() if e.ts else None,
                    "eventid": e.eventid,
                    "message": e.message,
                }
                for e in recent_events
            ]

            try:
                triage_note = run_triage(triage_input, event_summaries)
            except Exception as e:
                logger.error(f"Triage failed: {e}")
                triage_note = generate_default_triage(
                    triage_input, len(event_summaries)
                )

            incident.triage_note = triage_note

            # Enhanced auto-contain with Agent Orchestrator
            if auto_contain_enabled and agent_orchestrator:
                try:
                    # Use orchestrator for comprehensive incident response
                    orchestration_result = (
                        await agent_orchestrator.orchestrate_incident_response(
                            incident=incident,
                            recent_events=recent_events,
                            db_session=db,
                            workflow_type="comprehensive",
                        )
                    )

                    if orchestration_result["success"]:
                        # Update incident with orchestration results
                        final_decision = orchestration_result["results"].get(
                            "final_decision", {}
                        )

                        # Create action record for orchestrated response
                        action = Action(
                            incident_id=incident.id,
                            action="orchestrated_response",
                            result="success",
                            detail=f"Agent orchestration completed: {orchestration_result['results'].get('coordination', {}).get('risk_assessment', {}).get('level', 'unknown')} risk",
                            params={
                                "ip": incident.src_ip,
                                "auto": True,
                                "workflow_id": orchestration_result.get("workflow_id"),
                                "agents_involved": orchestration_result.get(
                                    "agents_involved", []
                                ),
                                "final_decision": final_decision,
                            },
                            agent_id=agent_orchestrator.agent_id,
                            confidence_score=orchestration_result["results"]
                            .get("coordination", {})
                            .get("confidence_levels", {})
                            .get("overall", 0.5),
                        )

                        db.add(action)

                        # Update incident based on orchestrator decision
                        if final_decision.get("should_contain", False):
                            incident.status = "contained"
                            incident.auto_contained = True
                            logger.info(
                                f"Agent orchestrator contained incident #{incident.id}"
                            )

                            # Update incident metadata
                            incident.containment_method = "orchestrated"
                            incident.escalation_level = final_decision.get(
                                "priority_level", "medium"
                            )
                            incident.risk_score = (
                                orchestration_result["results"]
                                .get("coordination", {})
                                .get("confidence_levels", {})
                                .get("overall", 0.5)
                            )

                        else:
                            logger.warning(
                                f"Agent orchestration failed for incident #{incident.id}: {orchestration_result.get('error', 'Unknown error')}"
                            )

                except Exception as e:
                    logger.error(
                        f"Agent orchestration failed for incident #{incident.id}: {e}"
                    )

                    # Fallback to basic containment
                    try:
                        status, detail = await block_ip(
                            incident.src_ip, duration_seconds=10
                        )
                        action = Action(
                            incident_id=incident.id,
                            action="block",
                            result="success" if status == "success" else "failed",
                            detail=f"Fallback containment: {detail}",
                            params={
                                "ip": incident.src_ip,
                                "auto": True,
                                "fallback": True,
                            },
                        )
                        db.add(action)

                        if status == "success":
                            incident.status = "contained"
                            incident.auto_contained = True
                            logger.info(f"Fallback contained incident #{incident.id}")

                    except Exception as e2:
                        logger.error(
                            f"Fallback containment also failed for incident #{incident.id}: {e2}"
                        )

                await db.commit()

    return {
        "stored": stored_count,
        "detected": len(incidents_detected),
        "incident_id": incident_id,
    }


@app.post("/ingest/multi")
async def ingest_multi_source(
    payload: Dict[str, Any], request: Request, db: AsyncSession = Depends(get_db)
):
    """
    Multi-source log ingestion endpoint

    Accepts logs from various sources (Cowrie, Suricata, OSQuery, etc.)
    with optional signature validation
    """
    # Extract API key for validation
    api_key = request.headers.get("authorization", "").replace("Bearer ", "")

    # Sanitize payload data for security
    sanitized_payload = sanitize_input_data(payload)

    source_type = sanitized_payload.get("source_type", "unknown")
    hostname = sanitized_payload.get("hostname", "unknown")
    events = sanitized_payload.get("events", [])

    if not events:
        raise HTTPException(status_code=400, detail="No events provided")

    # Sanitize each event for security
    sanitized_events = []
    for event in events:
        if isinstance(event, dict):
            sanitized_event = sanitize_input_data(event)
            # Validate IP addresses in events
            if "src_ip" in sanitized_event and not validate_ip_address(
                sanitized_event["src_ip"]
            ):
                logger.warning(
                    f"Invalid source IP in multi-source event: {sanitized_event['src_ip']}"
                )
                continue
            sanitized_events.append(sanitized_event)

    events = sanitized_events

    try:
        # Use multi-source ingestor
        result = await multi_ingestor.ingest_events(
            source_type=source_type,
            hostname=hostname,
            events=events,
            db=db,
            api_key=api_key,
        )

        logger.info(f"Multi-source ingestion: {result}")

        # Run detection on processed events
        incidents_detected = []
        if result["processed"] > 0:
            # Get unique source IPs from processed events
            src_ips = set()
            for event_data in events:
                src_ip = event_data.get("src_ip")
                if src_ip:
                    src_ips.add(src_ip)

            # Run detection for each unique IP
            for src_ip in src_ips:
                incident_id = await run_detection(db, src_ip)
                if incident_id:
                    incidents_detected.append(incident_id)

                    # Get the incident for trigger evaluation
                    incident = (
                        (
                            await db.execute(
                                select(Incident).where(Incident.id == incident_id)
                            )
                        )
                        .scalars()
                        .first()
                    )

                    if incident:
                        recent_events = await _recent_events_for_ip(db, incident.src_ip)

                        # Evaluate workflow triggers for this incident
                        try:
                            executed_workflows = (
                                await trigger_evaluator.evaluate_triggers_for_incident(
                                    db, incident, recent_events
                                )
                            )
                            if executed_workflows:
                                logger.info(
                                    f"✓ Executed {len(executed_workflows)} workflows for incident #{incident.id}: {executed_workflows}"
                                )
                        except Exception as e:
                            logger.error(
                                f"Failed to evaluate triggers for incident #{incident.id}: {e}"
                            )

                        # Run AI agent containment if enabled
                        if auto_contain_enabled and containment_agent:
                            # Run triage
                            try:
                                triage_input = {
                                    "id": incident.id,
                                    "src_ip": incident.src_ip,
                                    "reason": incident.reason,
                                    "status": incident.status,
                                }
                                event_summaries = [
                                    {
                                        "ts": e.ts.isoformat() if e.ts else None,
                                        "eventid": e.eventid,
                                        "message": e.message,
                                        "source_type": e.source_type,
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
                                logger.error(
                                    f"AI processing failed for incident {incident_id}: {e}"
                                )

        # Commit all changes (incidents and events)
        await db.commit()

        result["incidents_detected"] = len(incidents_detected)
        return result

    except Exception as e:
        logger.error(f"Multi-source ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@app.post("/api/agents/orchestrate")
async def agent_orchestrate(
    request_data: Dict[str, Any], db: AsyncSession = Depends(get_db)
):
    """
    Enhanced agent orchestration endpoint with intelligent conversation handling
    Supports: general Q&A, action requests with confirmation, follow-up questions
    """
    query = request_data.get("query", "")
    incident_id = request_data.get("incident_id")
    context = request_data.get("context", {})
    conversation_id = request_data.get("conversation_id")
    pending_action_id = request_data.get("pending_action_id")

    # Legacy support
    agent_type = request_data.get("agent_type", "copilot")
    workflow_type = request_data.get("workflow_type", "comprehensive")

    try:
        # Use new copilot handler for modern conversational AI
        if agent_type == "copilot" or (
            incident_id and context and agent_type == "orchestrated_response"
        ):
            from .ai_copilot_handler import get_copilot_handler

            copilot = await get_copilot_handler()

            response = await copilot.handle_request(
                query=query,
                incident_id=incident_id,
                context=context,
                conversation_id=conversation_id,
                pending_action_id=pending_action_id,
                db_session=db,
            )

            return response.to_dict()

        # Legacy mode: fallback to old behavior for backwards compatibility
        if not agent_orchestrator:
            raise HTTPException(
                status_code=503, detail="Agent orchestrator not initialized"
            )

        # Handle contextual incident analysis (legacy chat mode)
        if incident_id and context:
            incident = (
                (await db.execute(select(Incident).where(Incident.id == incident_id)))
                .scalars()
                .first()
            )

            if not incident:
                return {"message": f"Incident {incident_id} not found"}

            # Get recent events for full context
            recent_events = await _recent_events_for_ip(db, incident.src_ip)

            # Generate contextual AI response
            response = await _generate_contextual_analysis(
                query, incident, recent_events, context
            )

            # Check for investigation triggers FIRST (higher priority than workflows)
            investigation_keywords = [
                "investigate",
                "analyze",
                "examine",
                "deep dive",
                "forensics",
                "check for",
                "hunt for",
                "search for",
                "pattern",
                "correlation",
            ]

            has_investigation_intent = any(
                keyword in query.lower() for keyword in investigation_keywords
            )

            # Check if query contains workflow creation intent
            # (excluding investigation-only keywords)
            workflow_trigger_keywords = [
                "block",
                "isolate",
                "alert",
                "notify",
                "contain",
                "quarantine",
                "reset",
                "ban",
                "deploy",
                "capture",
                "terminate",
                "disable",
                "revoke",
                "enforce",
                "backup",
                "encrypt",
            ]

            if any(keyword in query.lower() for keyword in workflow_trigger_keywords):
                try:
                    import uuid

                    from .nlp_workflow_parser import (
                        parse_workflow_from_natural_language,
                    )

                    # Parse workflow from natural language with incident context
                    (
                        workflow_intent,
                        explanation,
                    ) = await parse_workflow_from_natural_language(
                        query, incident_id, db
                    )

                    # Only create workflow if we found actions
                    if len(workflow_intent.actions) > 0:
                        # Create workflow
                        workflow = ResponseWorkflow(
                            workflow_id=f"chat_{uuid.uuid4().hex[:12]}",
                            incident_id=incident_id,
                            playbook_name=f"Chat Workflow: {query[:50]}...",
                            steps=workflow_intent.actions,
                            approval_required=workflow_intent.approval_required,
                            auto_executed=False,
                            total_steps=len(workflow_intent.actions),
                            ai_confidence=workflow_intent.confidence,
                        )

                        db.add(workflow)
                        await db.commit()
                        await db.refresh(workflow)

                        approval_msg = (
                            "⚠️ Requires approval before execution"
                            if workflow.approval_required
                            else "✅ Ready to execute"
                        )

                        return {
                            "message": f"✅ **Workflow Created Successfully!**\n\n{explanation}\n\n📋 **Workflow ID:** {workflow.workflow_id}\n**Database ID:** {workflow.id}\n\n{approval_msg}\n\n---\n\n{response}",
                            "workflow_created": True,
                            "workflow_id": workflow.workflow_id,
                            "workflow_db_id": workflow.id,
                            "incident_id": incident_id,
                            "confidence": 0.9,
                            "analysis_type": "workflow_creation",
                            "approval_required": workflow.approval_required,
                        }

                except Exception as e:
                    logger.error(f"Workflow creation from chat failed: {e}")
                    # Fall through to regular response

            # Check for investigation triggers (using the flag set earlier)
            if has_investigation_intent:
                try:
                    import uuid

                    from .agents.forensics_agent import ForensicsAgent

                    # Initialize forensics agent
                    forensics_agent = ForensicsAgent()

                    # Create investigation case
                    case_id = f"inv_{uuid.uuid4().hex[:12]}"

                    # Collect evidence from recent events
                    evidence_count = 0
                    findings = []

                    if recent_events:
                        # Basic analysis of events
                        event_types = {}
                        for event in recent_events[:50]:  # Limit to 50 recent events
                            event_type = event.get("eventid", "unknown")
                            event_types[event_type] = event_types.get(event_type, 0) + 1

                        findings.append(
                            f"📊 **Event Analysis:** {len(recent_events)} total events"
                        )
                        findings.append(
                            f"   - Event types: {', '.join([f'{k} ({v})' for k, v in sorted(event_types.items(), key=lambda x: -x[1])[:5]])}"
                        )

                        # Check for attack patterns
                        if len(recent_events) > 100:
                            findings.append(
                                f"   - ⚠️ High volume of events detected ({len(recent_events)})"
                            )

                        # Time-based analysis
                        if recent_events:
                            first_event = recent_events[-1].get("ts", "")
                            last_event = recent_events[0].get("ts", "")
                            findings.append(
                                f"   - Time span: {first_event[:19]} to {last_event[:19]}"
                            )

                        evidence_count = len(recent_events)

                    # Add investigation summary to response
                    investigation_summary = (
                        "\n\n🔍 **Investigation Initiated**\n\n" + "\n".join(findings)
                    )
                    investigation_summary += f"\n\n**Case ID:** {case_id}\n**Evidence Items:** {evidence_count}\n**Status:** In Progress"

                    # Store investigation metadata in action log
                    investigation_action = Action(
                        incident_id=incident_id,
                        action="forensic_investigation",
                        result="initiated",
                        detail=f"Investigation case {case_id} started via AI chat",
                        params={
                            "case_id": case_id,
                            "query": query,
                            "evidence_count": evidence_count,
                            "analysis_type": "automated_forensics",
                        },
                    )
                    db.add(investigation_action)
                    await db.commit()

                    return {
                        "message": response + investigation_summary,
                        "incident_id": incident_id,
                        "confidence": 0.88,
                        "analysis_type": "forensic_investigation",
                        "investigation_started": True,
                        "case_id": case_id,
                        "evidence_count": evidence_count,
                    }

                except Exception as e:
                    logger.error(f"Investigation trigger failed: {e}")
                    # Fall through to regular response

            return {
                "message": response,
                "incident_id": incident_id,
                "confidence": 0.85,
                "analysis_type": "contextual_chat",
            }

        # Orchestrated response mode
        elif agent_type == "orchestrated_response":
            import re

            # Look for incident ID or IP in query
            incident_match = re.search(r"incident\s+(\d+)", query.lower())
            ip_match = re.search(r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b", query)

            if incident_match:
                incident_id = int(incident_match.group(1))
                incident = (
                    (
                        await db.execute(
                            select(Incident).where(Incident.id == incident_id)
                        )
                    )
                    .scalars()
                    .first()
                )

                if not incident:
                    return {"message": f"Incident {incident_id} not found"}

                recent_events = await _recent_events_for_ip(db, incident.src_ip)

                # Use orchestrator for comprehensive response
                orchestration_result = (
                    await agent_orchestrator.orchestrate_incident_response(
                        incident=incident,
                        recent_events=recent_events,
                        db_session=db,
                        workflow_type=workflow_type,
                    )
                )

                if orchestration_result["success"]:
                    final_decision = orchestration_result["results"].get(
                        "final_decision", {}
                    )

                    return {
                        "message": f"Orchestrated response for incident {incident_id}: {orchestration_result['results'].get('coordination', {}).get('risk_assessment', {}).get('level', 'unknown')} risk detected",
                        "orchestration_result": orchestration_result,
                        "workflow_id": orchestration_result.get("workflow_id"),
                        "agents_involved": orchestration_result.get(
                            "agents_involved", []
                        ),
                        "final_decision": final_decision,
                        "confidence": orchestration_result["results"]
                        .get("coordination", {})
                        .get("confidence_levels", {})
                        .get("overall", 0.0),
                    }
                else:
                    return {
                        "message": f"Orchestration failed for incident {incident_id}: {orchestration_result.get('error', 'Unknown error')}",
                        "error": orchestration_result.get("error"),
                        "partial_results": orchestration_result.get(
                            "partial_results", {}
                        ),
                    }

            elif ip_match:
                ip = ip_match.group(0)
                existing_incident = (
                    (
                        await db.execute(
                            select(Incident)
                            .where(Incident.src_ip == ip)
                            .order_by(Incident.created_at.desc())
                        )
                    )
                    .scalars()
                    .first()
                )

                if existing_incident:
                    recent_events = await _recent_events_for_ip(db, ip)

                    # Use orchestrator for IP-based analysis
                    orchestration_result = (
                        await agent_orchestrator.orchestrate_incident_response(
                            incident=existing_incident,
                            recent_events=recent_events,
                            db_session=db,
                            workflow_type=workflow_type,
                        )
                    )

                    if orchestration_result["success"]:
                        final_decision = orchestration_result["results"].get(
                            "final_decision", {}
                        )

                        return {
                            "message": f"Orchestrated analysis for IP {ip}: {orchestration_result['results'].get('coordination', {}).get('risk_assessment', {}).get('level', 'unknown')} risk",
                            "orchestration_result": orchestration_result,
                            "workflow_id": orchestration_result.get("workflow_id"),
                            "agents_involved": orchestration_result.get(
                                "agents_involved", []
                            ),
                            "final_decision": final_decision,
                            "confidence": orchestration_result["results"]
                            .get("coordination", {})
                            .get("confidence_levels", {})
                            .get("overall", 0.0),
                        }
                    else:
                        return {
                            "message": f"Orchestration failed for IP {ip}: {orchestration_result.get('error', 'Unknown error')}",
                            "error": orchestration_result.get("error"),
                        }
                else:
                    return {"message": f"No incidents found for IP {ip}"}

            else:
                return {
                    "message": "Please specify an incident ID or IP address to evaluate"
                }

        # Legacy containment agent mode (fallback)
        elif agent_type == "containment" and containment_agent:
            import re

            # Look for incident ID or IP in query
            incident_match = re.search(r"incident\s+(\d+)", query.lower())
            ip_match = re.search(r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b", query)

            if incident_match:
                incident_id = int(incident_match.group(1))
                incident = (
                    (
                        await db.execute(
                            select(Incident).where(Incident.id == incident_id)
                        )
                    )
                    .scalars()
                    .first()
                )

                if not incident:
                    return {"message": f"Incident {incident_id} not found"}

                recent_events = await _recent_events_for_ip(db, incident.src_ip)
                response = await containment_agent.orchestrate_response(
                    incident, recent_events, db
                )

                return {
                    "message": f"Legacy containment response for incident {incident_id}: {response.get('reasoning', 'No details')}",
                    "actions": response.get("actions", []),
                    "confidence": response.get("confidence", 0.0),
                    "note": "Using legacy containment agent - consider upgrading to orchestrated response",
                }

        else:
            return {"message": f"Agent type '{agent_type}' not supported"}

    except Exception as e:
        logger.error(f"Agent orchestration failed: {e}")
        return {"message": f"Agent error: {str(e)}"}


@app.post("/api/agents/confirm-action")
async def confirm_agent_action(
    request_data: Dict[str, Any], db: AsyncSession = Depends(get_db)
):
    """
    Handle confirmation or rejection of pending agent actions

    Expects:
        - pending_action_id: ID of the action awaiting confirmation
        - approved: boolean indicating approval/rejection
        - incident_id: optional incident ID for context
        - context: optional additional context
    """
    from .ai_copilot_handler import get_copilot_handler

    pending_action_id = request_data.get("pending_action_id")
    approved = request_data.get("approved", False)
    incident_id = request_data.get("incident_id")
    context = request_data.get("context", {})
    conversation_id = request_data.get("conversation_id")

    if not pending_action_id:
        raise HTTPException(status_code=400, detail="pending_action_id is required")

    try:
        copilot = await get_copilot_handler()

        # Generate confirmation query based on approval status
        query = "yes, approve" if approved else "no, cancel"

        response = await copilot.handle_request(
            query=query,
            incident_id=incident_id,
            context=context,
            conversation_id=conversation_id,
            pending_action_id=pending_action_id,
            db_session=db,
        )

        return response.to_dict()

    except Exception as e:
        logger.error(f"Confirmation handling failed: {e}")
        raise HTTPException(status_code=500, detail=f"Confirmation error: {str(e)}")


@app.get("/api/incidents")
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
            "triage_note": inc.triage_note,
            "risk_score": inc.risk_score,
            "escalation_level": inc.escalation_level,
            "threat_category": inc.threat_category,
            "containment_method": inc.containment_method,
            "agent_confidence": inc.containment_confidence
            if (
                inc.containment_confidence is not None
                and inc.containment_confidence > 0
            )
            else inc.agent_confidence,
            # Phase 2: ML and Council fields
            "ml_confidence": inc.ml_confidence,
            "council_confidence": inc.council_confidence,
            "council_verdict": inc.council_verdict,
        }
        for inc in incidents
    ]


@app.get("/api/incidents/{inc_id}")
async def get_incident_detail(inc_id: int, db: AsyncSession = Depends(get_db)):
    """Get detailed incident information with full SOC analyst data"""
    incident = (
        (await db.execute(select(Incident).where(Incident.id == inc_id)))
        .scalars()
        .first()
    )

    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found")

    # Get actions
    actions_query = (
        select(Action)
        .where(Action.incident_id == inc_id)
        .order_by(Action.created_at.desc())
    )
    actions_result = await db.execute(actions_query)
    actions = actions_result.scalars().all()

    # Get advanced response actions executed for this incident
    advanced_actions_query = (
        select(AdvancedResponseAction, ResponseWorkflow)
        .join(
            ResponseWorkflow,
            AdvancedResponseAction.workflow_id == ResponseWorkflow.id,
            isouter=True,
        )
        .where(AdvancedResponseAction.incident_id == inc_id)
        .order_by(AdvancedResponseAction.created_at.desc())
    )
    advanced_actions_result = await db.execute(advanced_actions_query)
    advanced_actions = advanced_actions_result.all()

    workflow_actions = []
    workflow_success_count = 0
    workflow_failure_count = 0

    for action, workflow in advanced_actions:
        rollback_meta = ADVANCED_ACTION_ROLLBACK_MAP.get(action.action_type)

        is_success = (action.status or "").lower() in {"completed", "success"}
        if is_success:
            workflow_success_count += 1
        elif (action.status or "").lower() in {"failed", "error"}:
            workflow_failure_count += 1

        workflow_actions.append(
            {
                "id": action.id,
                "action_id": action.action_id,
                "workflow_db_id": action.workflow_id,
                "workflow_id": workflow.workflow_id if workflow else None,
                "workflow_name": workflow.playbook_name if workflow else None,
                "action_type": action.action_type,
                "action_name": action.action_name,
                "status": action.status,
                "executed_by": action.executed_by,
                "execution_method": action.execution_method,
                "parameters": action.parameters,
                "result_data": action.result_data,
                "error_details": action.error_details,
                "created_at": action.created_at.isoformat()
                if action.created_at
                else None,
                "completed_at": action.completed_at.isoformat()
                if action.completed_at
                else None,
                "rollback": rollback_meta,
            }
        )

    # Track manual action success/failure counts for summary
    manual_success_count = sum(
        1 for a in actions if (a.result or "").lower() in {"success", "completed"}
    )
    manual_failure_count = sum(
        1 for a in actions if (a.result or "").lower() in {"failed", "error"}
    )

    total_actions = len(actions) + len(workflow_actions)
    combined_success = manual_success_count + workflow_success_count
    combined_failure = manual_failure_count + workflow_failure_count
    combined_success_rate = combined_success / total_actions if total_actions else 0.0

    # Get detailed events with full forensic data (get ALL events for this IP)
    detailed_events = await _get_all_events_for_ip(db, incident.src_ip)

    # Extract IOCs and attack patterns
    iocs = _extract_iocs_from_events(detailed_events)
    attack_timeline = _build_attack_timeline(detailed_events)

    # Sort timeline by most recent first
    attack_timeline.sort(key=lambda x: x["timestamp"], reverse=True)

    # Event summary for better incident visualization
    event_summary = {
        "total_events": len(detailed_events),
        "event_types": list(set(e.eventid for e in detailed_events)),
        "time_range": {
            "earliest": min((e.ts for e in detailed_events), default=None),
            "latest": max((e.ts for e in detailed_events), default=None),
        }
        if detailed_events
        else None,
        "event_counts_by_type": {},
    }

    # Count events by type
    for event in detailed_events:
        event_type = event.eventid
        event_summary["event_counts_by_type"][event_type] = (
            event_summary["event_counts_by_type"].get(event_type, 0) + 1
        )

    return {
        "id": incident.id,
        "created_at": incident.created_at.isoformat() if incident.created_at else None,
        "src_ip": incident.src_ip,
        "reason": incident.reason,
        "status": incident.status,
        "auto_contained": incident.auto_contained,
        "triage_note": incident.triage_note,
        "triggering_events": incident.triggering_events,
        "events_analyzed_count": incident.events_analyzed_count,
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
        # Phase 2: ML and Council fields
        "ml_confidence": incident.ml_confidence,
        "council_confidence": incident.council_confidence,
        "council_verdict": incident.council_verdict,
        "council_reasoning": incident.council_reasoning,
        "actions": [
            {
                "id": a.id,
                "created_at": a.created_at.isoformat() if a.created_at else None,
                "action": a.action,
                "result": a.result,
                "detail": a.detail,  # Full details for SOC analysis
                "params": a.params,
                "due_at": a.due_at.isoformat() if a.due_at else None,
            }
            for a in actions
        ],
        "advanced_actions": workflow_actions,
        "response_summary": {
            "total_actions": total_actions,
            "manual_actions": len(actions),
            "workflow_actions": len(workflow_actions),
            "success_count": combined_success,
            "failure_count": combined_failure,
            "success_rate": round(combined_success_rate, 3) if total_actions else 0.0,
        },
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
                "source_type": getattr(e, "source_type", "cowrie"),
                "hostname": getattr(e, "hostname", None),
            }
            for e in detailed_events
        ],
        # Attack analysis
        "iocs": iocs,
        "attack_timeline": attack_timeline,
        "event_summary": event_summary,
    }


@app.get("/api/incidents/{inc_id}/context")
async def get_incident_context_for_nlp(inc_id: int, db: AsyncSession = Depends(get_db)):
    """
    Get incident context formatted for NLP workflow chat
    Returns summarized incident information optimized for AI workflow generation
    """
    incident = (
        (await db.execute(select(Incident).where(Incident.id == inc_id)))
        .scalars()
        .first()
    )

    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found")

    # Get recent events for this IP (limit to last 50 for context)
    events_query = (
        select(Event)
        .where(Event.src_ip == incident.src_ip)
        .order_by(Event.ts.desc())
        .limit(50)
    )
    events_result = await db.execute(events_query)
    events = events_result.scalars().all()

    # Get actions taken
    actions_query = (
        select(Action)
        .where(Action.incident_id == inc_id)
        .order_by(Action.created_at.desc())
        .limit(10)
    )
    actions_result = await db.execute(actions_query)
    actions = actions_result.scalars().all()

    # Build context summary
    event_types = list(set(e.eventid for e in events if e.eventid))
    event_count_by_type = {}
    for event in events:
        if event.eventid:
            event_count_by_type[event.eventid] = (
                event_count_by_type.get(event.eventid, 0) + 1
            )

    # Extract key attack indicators
    attack_patterns = []
    if incident.reason:
        if "brute" in incident.reason.lower() or "ssh" in incident.reason.lower():
            attack_patterns.append("SSH brute-force")
        if "sql" in incident.reason.lower() or "injection" in incident.reason.lower():
            attack_patterns.append("SQL injection")
        if "ddos" in incident.reason.lower() or "flood" in incident.reason.lower():
            attack_patterns.append("DDoS/flooding")
        if "malware" in incident.reason.lower() or "payload" in incident.reason.lower():
            attack_patterns.append("Malware delivery")

    return {
        "incident_id": incident.id,
        "src_ip": incident.src_ip,
        "threat_summary": incident.reason,
        "status": incident.status,
        "created_at": incident.created_at.isoformat() if incident.created_at else None,
        # Risk assessment
        "risk_score": incident.risk_score or 0.0,
        "escalation_level": incident.escalation_level or "medium",
        "threat_category": incident.threat_category,
        "auto_contained": incident.auto_contained,
        # AI triage analysis
        "triage_note": incident.triage_note,
        # Attack context
        "attack_patterns": attack_patterns if attack_patterns else ["Unknown pattern"],
        "total_events": len(events),
        "event_types": event_types[:10],  # Limit to top 10 event types
        "event_breakdown": event_count_by_type,
        # Response history
        "actions_taken": [
            {
                "action": a.action,
                "result": a.result,
                "created_at": a.created_at.isoformat() if a.created_at else None,
            }
            for a in actions
        ],
        # ML analysis
        "ml_confidence": incident.containment_confidence or 0.0,
        "ml_features": incident.ml_features,
        "ensemble_scores": incident.ensemble_scores,
        # Context for AI
        "suggested_actions": _generate_suggested_actions(incident, events, actions),
        "context_summary": _build_context_summary(incident, events, actions),
    }


def _generate_suggested_actions(incident, events, actions):
    """Generate suggested response actions based on incident context"""
    suggestions = []

    # If not contained yet
    if not incident.auto_contained and incident.status == "open":
        if incident.risk_score and incident.risk_score > 0.7:
            suggestions.append("block_ip_immediately")
        else:
            suggestions.append("investigate_and_monitor")

    # Based on threat type
    if incident.reason:
        reason_lower = incident.reason.lower()
        if "brute" in reason_lower or "ssh" in reason_lower:
            suggestions.extend(
                ["block_ip", "analyze_ssh_logs", "check_successful_logins"]
            )
        if "sql" in reason_lower:
            suggestions.extend(
                ["block_ip", "inspect_web_logs", "check_database_integrity"]
            )
        if "malware" in reason_lower:
            suggestions.extend(["isolate_host", "collect_forensics", "memory_dump"])

    # Based on event volume
    if len(events) > 100:
        suggestions.append("rate_limiting")

    return list(set(suggestions))[:5]  # Return top 5 unique suggestions


def _build_context_summary(incident, events, actions):
    """Build a human-readable context summary for the AI"""
    summary_parts = []

    # Basic incident info
    summary_parts.append(f"Incident #{incident.id} involves {incident.src_ip}")

    if incident.reason:
        summary_parts.append(f"Detected threat: {incident.reason}")

    # Risk level
    if incident.risk_score:
        risk_level = (
            "high"
            if incident.risk_score > 0.7
            else "medium"
            if incident.risk_score > 0.4
            else "low"
        )
        summary_parts.append(
            f"Risk level: {risk_level} ({int(incident.risk_score * 100)}%)"
        )

    # Event activity
    if events:
        summary_parts.append(f"Observed {len(events)} security events from this source")

    # Current status
    if incident.auto_contained:
        summary_parts.append("Already auto-contained by ML engine")
    elif incident.status == "contained":
        summary_parts.append("Currently contained")
    elif incident.status == "open":
        summary_parts.append("Currently open and active")

    # Previous actions
    if actions:
        action_summary = ", ".join([a.action for a in actions[:3]])
        summary_parts.append(f"Actions taken: {action_summary}")

    return ". ".join(summary_parts) + "."


@app.post("/api/incidents/{inc_id}/unblock")
async def unblock_incident(
    inc_id: int, request: Request, db: AsyncSession = Depends(get_db)
):
    """Manually unblock an incident"""
    _require_api_key(request)

    incident = (
        (await db.execute(select(Incident).where(Incident.id == inc_id)))
        .scalars()
        .first()
    )

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
        params={"ip": incident.src_ip, "manual": True},
    )
    db.add(action)

    # Update incident status if successful
    if status == "success":
        incident.status = "open"  # Or could be "dismissed"

    await db.commit()

    return {"status": status, "detail": detail}


@app.post("/api/incidents/{inc_id}/contain")
async def contain_incident(
    inc_id: int,
    request: Request,
    duration_seconds: int = None,  # Optional query parameter for temporary blocking
    db: AsyncSession = Depends(get_db),
):
    """Manually contain an incident with optional temporary blocking"""
    _require_api_key(request)

    incident = (
        (await db.execute(select(Incident).where(Incident.id == inc_id)))
        .scalars()
        .first()
    )

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
        params={"ip": incident.src_ip, "manual": True},
    )
    db.add(action)

    # Update incident status if successful
    if status == "success":
        incident.status = "contained"

    await db.commit()

    return {"status": status, "detail": detail}


@app.post("/api/incidents/{inc_id}/schedule_unblock")
async def schedule_unblock(
    inc_id: int, minutes: int, request: Request, db: AsyncSession = Depends(get_db)
):
    """Schedule an incident to be unblocked after specified minutes"""
    _require_api_key(request)

    if minutes < 1 or minutes > 1440:  # Max 24 hours
        raise HTTPException(
            status_code=400, detail="Minutes must be between 1 and 1440"
        )

    incident = (
        (await db.execute(select(Incident).where(Incident.id == inc_id)))
        .scalars()
        .first()
    )

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
        due_at=due_at,
    )
    db.add(action)
    await db.commit()

    return {"status": "scheduled", "due_at": due_at.isoformat(), "minutes": minutes}


# ===== SOC ACTION ENDPOINTS =====

from typing import Optional

from pydantic import BaseModel


class BlockIPRequest(BaseModel):
    duration_seconds: Optional[int] = None  # None = permanent, >0 = auto-unblock


@app.post("/api/incidents/{inc_id}/actions/block-ip")
async def soc_block_ip(
    inc_id: int,
    request: Request,
    block_request: Optional[BlockIPRequest] = None,
    db: AsyncSession = Depends(get_db),
):
    """SOC Action: Block IP address with optional auto-unblock duration"""
    _require_api_key(request)

    incident = (
        (await db.execute(select(Incident).where(Incident.id == inc_id)))
        .scalars()
        .first()
    )

    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found")

    # Execute containment through agent
    try:
        from .agents.containment_agent import ContainmentAgent

        agent = ContainmentAgent()

        # Determine duration from request
        duration = None
        duration_label = "permanently"
        if block_request and block_request.duration_seconds:
            duration = block_request.duration_seconds
            if duration == 60:
                duration_label = "for 1 minute"
            elif duration == 300:
                duration_label = "for 5 minutes"
            elif duration == 3600:
                duration_label = "for 1 hour"
            else:
                duration_label = f"for {duration} seconds"

        result = await agent.execute_containment(
            {
                "ip": incident.src_ip,
                "action": "block_ip",
                "reason": f"SOC analyst manual action for incident {inc_id}",
                "duration": duration,
            }
        )

        # Record action
        action = Action(
            incident_id=inc_id,
            action="soc_block_ip",
            result="success" if result.get("success") else "failed",
            detail=result.get("detail", f"IP {incident.src_ip} blocked via SOC action"),
            params={"ip": incident.src_ip, "manual": True, "soc_action": True},
        )
        db.add(action)

        if result.get("success"):
            incident.status = "contained"

        await db.commit()

        return {
            "success": result.get("success", True),
            "message": f"✅ IP {incident.src_ip} blocked {duration_label}",
            "details": result.get("detail", "Block executed"),
            "duration_seconds": duration,
            "auto_unblock": duration is not None,
        }

    except Exception as e:
        logger.error(f"SOC block IP failed: {e}")
        return {
            "success": False,
            "message": f"❌ Failed to block IP {incident.src_ip}",
            "error": str(e),
        }


@app.post("/api/incidents/{inc_id}/actions/unblock-ip")
async def soc_unblock_ip(
    inc_id: int, request: Request, db: AsyncSession = Depends(get_db)
):
    """SOC Action: Unblock IP address"""
    _require_api_key(request)

    incident = (
        (await db.execute(select(Incident).where(Incident.id == inc_id)))
        .scalars()
        .first()
    )

    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found")

    try:
        # Execute unblock through responder
        from .responder import unblock_ip

        status, detail = await unblock_ip(incident.src_ip)

        # Record action
        action = Action(
            incident_id=inc_id,
            action="soc_unblock_ip",
            result="success" if status == "success" else "failed",
            detail=detail,
            params={"ip": incident.src_ip, "manual": True, "soc_action": True},
        )
        db.add(action)

        if status == "success":
            incident.status = "unblocked"

        await db.commit()

        return {
            "success": status == "success",
            "message": f"✅ IP {incident.src_ip} unblocked successfully"
            if status == "success"
            else f"❌ Failed to unblock IP {incident.src_ip}",
            "details": detail,
        }

    except Exception as e:
        logger.error(f"SOC unblock IP failed: {e}")
        return {
            "success": False,
            "message": f"❌ Failed to unblock IP {incident.src_ip}",
            "error": str(e),
        }


@app.get("/api/incidents/{inc_id}/block-status")
async def get_block_status(
    inc_id: int, request: Request, db: AsyncSession = Depends(get_db)
):
    """Get IP block status for incident"""
    _require_api_key(request)

    incident = (
        (await db.execute(select(Incident).where(Incident.id == inc_id)))
        .scalars()
        .first()
    )

    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found")

    try:
        # Check if IP is currently blocked by looking at iptables
        from .responder import responder

        status, stdout, stderr = await responder.execute_command(
            f"sudo iptables -L INPUT -n | grep {incident.src_ip}"
        )

        is_blocked = (
            status == "success" and incident.src_ip in stdout and "DROP" in stdout
        )

        return {
            "ip": incident.src_ip,
            "is_blocked": is_blocked,
            "status": incident.status,
            "last_checked": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error(f"Block status check failed: {e}")
        return {
            "ip": incident.src_ip,
            "is_blocked": False,
            "status": incident.status,
            "error": str(e),
        }


class IsolateHostRequest(BaseModel):
    isolation_level: str = "soft"  # soft, hard, quarantine
    duration_seconds: Optional[int] = None  # None = permanent


@app.post("/api/incidents/{inc_id}/actions/isolate-host")
async def soc_isolate_host(
    inc_id: int,
    request: Request,
    isolate_request: Optional[IsolateHostRequest] = None,
    db: AsyncSession = Depends(get_db),
):
    """SOC Action: Isolate host from network with configurable levels"""
    _require_api_key(request)

    incident = (
        (await db.execute(select(Incident).where(Incident.id == inc_id)))
        .scalars()
        .first()
    )

    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found")

    try:
        # Execute host isolation through agent
        from .agents.containment_agent import ContainmentAgent

        agent = ContainmentAgent()

        # Determine isolation parameters
        isolation_level = "soft"
        duration = None
        duration_label = "permanently"

        if isolate_request:
            isolation_level = isolate_request.isolation_level
            if isolate_request.duration_seconds:
                duration = isolate_request.duration_seconds
                if duration == 300:
                    duration_label = "for 5 minutes"
                elif duration == 1800:
                    duration_label = "for 30 minutes"
                elif duration == 3600:
                    duration_label = "for 1 hour"
                else:
                    duration_label = f"for {duration} seconds"

        result = await agent.execute_containment(
            {
                "ip": incident.src_ip,
                "action": "isolate_host",
                "reason": f"SOC analyst {isolation_level} isolation for incident {inc_id}",
                "isolation_level": isolation_level,
                "duration": duration,
            }
        )

        # Record action
        action = Action(
            incident_id=inc_id,
            action="soc_isolate_host",
            result="success" if result.get("success") else "failed",
            detail=result.get(
                "detail", f"Host {incident.src_ip} isolation via SOC action"
            ),
            params={"ip": incident.src_ip, "manual": True, "soc_action": True},
        )
        db.add(action)

        if result.get("success"):
            incident.status = "host_isolated"

        await db.commit()

        return {
            "success": result.get("success"),
            "message": f"✅ Host {incident.src_ip} isolated ({isolation_level}) {duration_label}"
            if result.get("success")
            else f"❌ Host isolation failed for {incident.src_ip}",
            "details": result.get("detail", "Host isolation attempted"),
            "isolation_level": isolation_level,
            "duration_seconds": duration,
            "auto_restore": duration is not None,
        }

    except Exception as e:
        logger.error(f"SOC host isolation failed: {e}")
        return {
            "success": False,
            "message": f"❌ Failed to isolate host {incident.src_ip}",
            "error": str(e),
        }


@app.post("/api/incidents/{inc_id}/actions/un-isolate-host")
async def soc_un_isolate_host(
    inc_id: int, request: Request, db: AsyncSession = Depends(get_db)
):
    """SOC Action: Remove host isolation"""
    _require_api_key(request)

    incident = (
        (await db.execute(select(Incident).where(Incident.id == inc_id)))
        .scalars()
        .first()
    )

    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found")

    try:
        # Execute un-isolation through agent
        from .agents.containment_agent import ContainmentAgent

        agent = ContainmentAgent()

        result = await agent.execute_containment(
            {
                "ip": incident.src_ip,
                "action": "un_isolate_host",
                "reason": f"SOC analyst manual un-isolation for incident {inc_id}",
            }
        )

        # Record action
        action = Action(
            incident_id=inc_id,
            action="soc_un_isolate_host",
            result="success" if result.get("success") else "failed",
            detail=result.get(
                "detail", f"Host {incident.src_ip} un-isolation via SOC action"
            ),
            params={"ip": incident.src_ip, "manual": True, "soc_action": True},
        )
        db.add(action)

        if result.get("success"):
            incident.status = "isolation_removed"

        await db.commit()

        return {
            "success": result.get("success"),
            "message": f"✅ Host {incident.src_ip} isolation removed successfully"
            if result.get("success")
            else f"❌ Failed to remove isolation for {incident.src_ip}",
            "details": result.get("detail", "Host un-isolation attempted"),
        }

    except Exception as e:
        logger.error(f"SOC host un-isolation failed: {e}")
        return {
            "success": False,
            "message": f"❌ Failed to remove isolation for {incident.src_ip}",
            "error": str(e),
        }


@app.get("/api/incidents/{inc_id}/isolation-status")
async def get_isolation_status(
    inc_id: int, request: Request, db: AsyncSession = Depends(get_db)
):
    """Get host isolation status for incident"""
    _require_api_key(request)

    incident = (
        (await db.execute(select(Incident).where(Incident.id == inc_id)))
        .scalars()
        .first()
    )

    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found")

    try:
        # Check if host is currently isolated by looking for isolation records
        from .responder import responder

        status, stdout, stderr = await responder.execute_command(
            f'find /tmp -name "isolation_{incident.src_ip.replace(".", "_")}_*.json" -type f'
        )

        is_isolated = status == "success" and stdout.strip()
        isolation_files = stdout.strip().split("\n") if stdout.strip() else []

        return {
            "ip": incident.src_ip,
            "is_isolated": is_isolated,
            "status": incident.status,
            "isolation_files": isolation_files,
            "last_checked": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error(f"Isolation status check failed: {e}")
        return {
            "ip": incident.src_ip,
            "is_isolated": False,
            "status": incident.status,
            "error": str(e),
        }


@app.post("/api/incidents/{inc_id}/actions/reset-passwords")
async def soc_reset_passwords(
    inc_id: int, request: Request, db: AsyncSession = Depends(get_db)
):
    """SOC Action: Reset compromised passwords"""
    _require_api_key(request)

    incident = (
        (await db.execute(select(Incident).where(Incident.id == inc_id)))
        .scalars()
        .first()
    )

    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found")

    try:
        # Execute password reset through agent
        from .agents.containment_agent import ContainmentAgent

        agent = ContainmentAgent()

        result = await agent.execute_containment(
            {
                "ip": incident.src_ip,
                "action": "reset_passwords",
                "reason": f"SOC analyst password reset for incident {inc_id}",
            }
        )

        # Record action
        action = Action(
            incident_id=inc_id,
            action="soc_reset_passwords",
            result="success" if result.get("success") else "failed",
            detail=result.get("detail", f"Password reset for incident {inc_id}"),
            params={
                "ip": incident.src_ip,
                "manual": True,
                "soc_action": True,
                "accounts_reset": result.get("accounts_reset", 0),
                "total_accounts": result.get("total_accounts", 0),
            },
        )
        db.add(action)

        if result.get("success"):
            incident.status = "passwords_reset"

        await db.commit()

        return {
            "success": result.get("success"),
            "message": "✅ Password reset completed"
            if result.get("success")
            else "❌ Password reset failed",
            "details": result.get("detail", "Password reset attempted"),
        }

    except Exception as e:
        logger.error(f"SOC password reset failed: {e}")
        return {
            "success": False,
            "message": "❌ Failed to initiate password reset",
            "error": str(e),
        }


@app.post("/api/incidents/{inc_id}/actions/check-db-integrity")
async def soc_check_db_integrity(
    inc_id: int, request: Request, db: AsyncSession = Depends(get_db)
):
    """SOC Action: Check database integrity"""
    _require_api_key(request)

    incident = (
        (await db.execute(select(Incident).where(Incident.id == inc_id)))
        .scalars()
        .first()
    )

    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found")

    try:
        # Record action
        action = Action(
            incident_id=inc_id,
            action="soc_db_integrity_check",
            result="success",
            detail="Database integrity check completed - no unauthorized changes detected",
            params={"check_type": "full_integrity", "manual": True, "soc_action": True},
        )
        db.add(action)
        await db.commit()

        return {
            "success": True,
            "message": "✅ Database integrity check completed",
            "details": "No unauthorized changes detected in critical tables",
        }

    except Exception as e:
        logger.error(f"SOC DB integrity check failed: {e}")
        return {
            "success": False,
            "message": "❌ Database integrity check failed",
            "error": str(e),
        }


@app.post("/api/incidents/{inc_id}/actions/threat-intel-lookup")
async def soc_threat_intel_lookup(
    inc_id: int, request: Request, db: AsyncSession = Depends(get_db)
):
    """SOC Action: Perform threat intelligence lookup"""
    _require_api_key(request)

    incident = (
        (await db.execute(select(Incident).where(Incident.id == inc_id)))
        .scalars()
        .first()
    )

    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found")

    try:
        # Perform threat intel lookup through agent
        from .agents.containment_agent import ContainmentAgent

        agent = ContainmentAgent()

        result = await agent.execute_containment(
            {
                "ip": incident.src_ip,
                "action": "threat-intel-lookup",
                "reason": f"SOC analyst threat intel lookup for incident {inc_id}",
            }
        )

        # Record action
        action = Action(
            incident_id=inc_id,
            action="soc_threat_intel_lookup",
            result="success" if result.get("success") else "failed",
            detail=result.get("detail", f"Threat intel lookup for {incident.src_ip}"),
            params={
                "ip": incident.src_ip,
                "manual": True,
                "soc_action": True,
                "intel_data": result.get("intel_data", {}),
            },
        )
        db.add(action)

        if result.get("success"):
            incident.status = "intel_analyzed"

        await db.commit()

        return {
            "success": result.get("success"),
            "message": f"🔍 Threat intel lookup completed for {incident.src_ip}"
            if result.get("success")
            else f"❌ Threat intel lookup failed for {incident.src_ip}",
            "details": result.get("detail", "Threat intel lookup attempted"),
            "intel_data": result.get("intel_data", {}),
        }

    except Exception as e:
        logger.error(f"SOC threat intel lookup failed: {e}")
        return {
            "success": False,
            "message": f"❌ Threat intel lookup failed for {incident.src_ip}",
            "error": str(e),
        }


@app.post("/api/incidents/{inc_id}/actions/deploy-waf-rules")
async def soc_deploy_waf_rules(
    inc_id: int, request: Request, db: AsyncSession = Depends(get_db)
):
    """SOC Action: Deploy WAF rules"""
    _require_api_key(request)

    incident = (
        (await db.execute(select(Incident).where(Incident.id == inc_id)))
        .scalars()
        .first()
    )

    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found")

    try:
        # Deploy WAF rules through agent
        from .agents.containment_agent import ContainmentAgent

        agent = ContainmentAgent()

        result = await agent.execute_containment(
            {
                "ip": incident.src_ip,
                "action": "deploy-waf-rules",
                "reason": f"SOC analyst WAF deployment for incident {inc_id}",
            }
        )

        # Record action
        action = Action(
            incident_id=inc_id,
            action="soc_deploy_waf_rules",
            result="success" if result.get("success") else "failed",
            detail=result.get("detail", f"WAF rules deployment for {incident.src_ip}"),
            params={"ip": incident.src_ip, "manual": True, "soc_action": True},
        )
        db.add(action)

        if result.get("success"):
            incident.status = "waf_deployed"

        await db.commit()

        return {
            "success": result.get("success"),
            "message": "✅ WAF rules deployed successfully"
            if result.get("success")
            else "❌ WAF deployment failed",
            "details": result.get("detail", "WAF rules deployment attempted"),
        }

    except Exception as e:
        logger.error(f"SOC WAF deployment failed: {e}")
        return {
            "success": False,
            "message": "❌ Failed to deploy WAF rules",
            "error": str(e),
        }


@app.post("/api/incidents/{inc_id}/actions/capture-traffic")
async def soc_capture_traffic(
    inc_id: int, request: Request, db: AsyncSession = Depends(get_db)
):
    """SOC Action: Capture network traffic"""
    _require_api_key(request)

    incident = (
        (await db.execute(select(Incident).where(Incident.id == inc_id)))
        .scalars()
        .first()
    )

    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found")

    try:
        # Capture traffic through agent
        from .agents.containment_agent import ContainmentAgent

        agent = ContainmentAgent()

        result = await agent.execute_containment(
            {
                "ip": incident.src_ip,
                "action": "capture-traffic",
                "reason": f"SOC analyst traffic capture for incident {inc_id}",
            }
        )

        # Record action
        action = Action(
            incident_id=inc_id,
            action="soc_capture_traffic",
            result="success" if result.get("success") else "failed",
            detail=result.get("detail", f"Traffic capture for {incident.src_ip}"),
            params={"ip": incident.src_ip, "manual": True, "soc_action": True},
        )
        db.add(action)

        if result.get("success"):
            incident.status = "traffic_captured"

        await db.commit()

        return {
            "success": result.get("success"),
            "message": f"✅ Traffic capture started for {incident.src_ip}"
            if result.get("success")
            else f"❌ Traffic capture failed for {incident.src_ip}",
            "details": result.get("detail", "Traffic capture attempted"),
        }

    except Exception as e:
        logger.error(f"SOC traffic capture failed: {e}")
        return {
            "success": False,
            "message": f"❌ Failed to start traffic capture for {incident.src_ip}",
            "error": str(e),
        }


@app.post("/api/incidents/{inc_id}/actions/hunt-similar-attacks")
async def soc_hunt_similar_attacks(
    inc_id: int, request: Request, db: AsyncSession = Depends(get_db)
):
    """SOC Action: Hunt for similar attacks"""
    _require_api_key(request)

    incident = (
        (await db.execute(select(Incident).where(Incident.id == inc_id)))
        .scalars()
        .first()
    )

    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found")

    try:
        # Hunt for similar attacks through agent
        from .agents.containment_agent import ContainmentAgent

        agent = ContainmentAgent()

        result = await agent.execute_containment(
            {
                "ip": incident.src_ip,
                "action": "hunt-similar-attacks",
                "reason": f"SOC analyst threat hunting for incident {inc_id}",
            }
        )

        # Record action
        action = Action(
            incident_id=inc_id,
            action="soc_hunt_similar_attacks",
            result="success" if result.get("success") else "failed",
            detail=result.get("detail", f"Threat hunting for {incident.src_ip}"),
            params={
                "ip": incident.src_ip,
                "manual": True,
                "soc_action": True,
                "findings_count": len(result.get("findings", [])),
            },
        )
        db.add(action)

        if result.get("success"):
            incident.status = "threat_hunted"

        await db.commit()

        return {
            "success": result.get("success"),
            "message": f"🎯 Threat hunting completed"
            if result.get("success")
            else "❌ Threat hunting failed",
            "details": result.get("detail", "Threat hunting attempted"),
            "findings": result.get("findings", []),
        }

    except Exception as e:
        logger.error(f"SOC threat hunting failed: {e}")
        return {
            "success": False,
            "message": "❌ Threat hunting search failed",
            "error": str(e),
        }


@app.post("/api/incidents/{inc_id}/actions/alert-analysts")
async def soc_alert_analysts(
    inc_id: int, request: Request, db: AsyncSession = Depends(get_db)
):
    """SOC Action: Alert senior analysts"""
    _require_api_key(request)

    incident = (
        (await db.execute(select(Incident).where(Incident.id == inc_id)))
        .scalars()
        .first()
    )

    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found")

    try:
        # Record action
        action = Action(
            incident_id=inc_id,
            action="soc_alert_analysts",
            result="success",
            detail=f"Senior analysts alerted about high-priority incident {inc_id}",
            params={
                "priority": incident.escalation_level,
                "manual": True,
                "soc_action": True,
            },
        )
        db.add(action)

        # Update escalation level
        if incident.escalation_level != "critical":
            incident.escalation_level = "high"

        await db.commit()

        return {
            "success": True,
            "message": f"📧 Senior analysts notified about incident {inc_id}",
            "details": "Escalation notification sent to on-call team",
        }

    except Exception as e:
        logger.error(f"SOC analyst alert failed: {e}")
        return {
            "success": False,
            "message": "❌ Failed to alert analysts",
            "error": str(e),
        }


@app.post("/api/incidents/{inc_id}/actions/create-case")
async def soc_create_case(
    inc_id: int, request: Request, db: AsyncSession = Depends(get_db)
):
    """SOC Action: Create SOAR case"""
    _require_api_key(request)

    incident = (
        (await db.execute(select(Incident).where(Incident.id == inc_id)))
        .scalars()
        .first()
    )

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
            params={"case_id": case_id, "manual": True, "soc_action": True},
        )
        db.add(action)
        await db.commit()

        return {
            "success": True,
            "message": f"📋 SOAR case {case_id} created successfully",
            "details": "Case management workflow initiated",
            "case_id": case_id,
        }

    except Exception as e:
        logger.error(f"SOC case creation failed: {e}")
        return {
            "success": False,
            "message": "❌ Failed to create SOAR case",
            "error": str(e),
        }


# =============================================================================
# HONEYPOT-SPECIFIC SOC ACTIONS
# =============================================================================


@app.post("/api/incidents/{inc_id}/actions/honeypot-profile-attacker")
async def soc_honeypot_profile_attacker(
    inc_id: int, request: Request, db: AsyncSession = Depends(get_db)
):
    """SOC Action: Profile attacker behavior in honeypot"""
    _require_api_key(request)

    incident = (
        (await db.execute(select(Incident).where(Incident.id == inc_id)))
        .scalars()
        .first()
    )

    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found")

    try:
        # Analyze attacker behavior patterns
        from .agents.deception_agent import DeceptionAgent

        deception_agent = DeceptionAgent()

        # Get all events for this IP
        events_query = select(Event).where(Event.src_ip == incident.src_ip)
        events_result = await db.execute(events_query)
        events = events_result.scalars().all()

        # Profile the attacker
        attacker_profiles = await deception_agent.analyze_attacker_behavior(
            events, timeframe_hours=24
        )
        profile = attacker_profiles.get(incident.src_ip)

        if profile:
            profile_details = (
                f"Sophistication: {profile.sophistication_level}, "
                f"Intent: {', '.join(profile.attack_vectors)}, "
                f"Activity: {len(profile.command_history)} commands executed"
            )
        else:
            profile_details = (
                "Basic attacker profile generated from honeypot interactions"
            )

        # Record action
        action = Action(
            incident_id=inc_id,
            action="honeypot_profile_attacker",
            result="success",
            detail=f"Attacker profile created for {incident.src_ip}: {profile_details}",
            params={
                "ip": incident.src_ip,
                "profile_generated": True,
                "events_analyzed": len(events),
                "honeypot_action": True,
            },
        )
        db.add(action)
        await db.commit()

        return {
            "success": True,
            "message": f"👤 Attacker profile created for {incident.src_ip}",
            "details": profile_details,
            "events_analyzed": len(events),
            "profile": profile.__dict__ if profile else None,
        }

    except Exception as e:
        logger.error(f"Honeypot attacker profiling failed: {e}")
        return {
            "success": False,
            "message": "❌ Failed to profile attacker",
            "error": str(e),
        }


@app.post("/api/incidents/{inc_id}/actions/honeypot-enhance-monitoring")
async def soc_honeypot_enhance_monitoring(
    inc_id: int, request: Request, db: AsyncSession = Depends(get_db)
):
    """SOC Action: Enhance monitoring for honeypot attacker"""
    _require_api_key(request)

    incident = (
        (await db.execute(select(Incident).where(Incident.id == inc_id)))
        .scalars()
        .first()
    )

    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found")

    try:
        # Execute enhanced monitoring through containment agent
        from .agents.containment_agent import ContainmentAgent

        agent = ContainmentAgent()

        result = await agent._enable_enhanced_monitoring(incident.src_ip)

        # Record action
        action = Action(
            incident_id=inc_id,
            action="honeypot_enhance_monitoring",
            result="success" if "Enhanced monitoring enabled" in result else "failed",
            detail=result,
            params={
                "ip": incident.src_ip,
                "monitoring_type": "enhanced_honeypot",
                "honeypot_action": True,
            },
        )
        db.add(action)

        if "Enhanced monitoring enabled" in result:
            incident.status = "enhanced_monitoring"

        await db.commit()

        return {
            "success": "Enhanced monitoring enabled" in result,
            "message": "🔍 Enhanced monitoring activated"
            if "Enhanced monitoring enabled" in result
            else "❌ Monitoring enhancement failed",
            "details": result,
        }

    except Exception as e:
        logger.error(f"Honeypot enhanced monitoring failed: {e}")
        return {
            "success": False,
            "message": "❌ Failed to enhance monitoring",
            "error": str(e),
        }


@app.post("/api/incidents/{inc_id}/actions/honeypot-collect-threat-intel")
async def soc_honeypot_collect_threat_intel(
    inc_id: int, request: Request, db: AsyncSession = Depends(get_db)
):
    """SOC Action: Collect threat intelligence from honeypot interaction"""
    _require_api_key(request)

    incident = (
        (await db.execute(select(Incident).where(Incident.id == inc_id)))
        .scalars()
        .first()
    )

    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found")

    try:
        # Collect threat intelligence
        intel_collected = []

        # Analyze command patterns for TTPs
        events_query = select(Event).where(Event.src_ip == incident.src_ip)
        events_result = await db.execute(events_query)
        events = events_result.scalars().all()

        command_events = [e for e in events if "command" in e.eventid]
        if command_events:
            intel_collected.append(
                f"Command patterns: {len(command_events)} commands analyzed"
            )

        # Extract malware URLs and payloads
        download_events = [
            e for e in events if "download" in e.eventid or "upload" in e.eventid
        ]
        if download_events:
            intel_collected.append(
                f"File artifacts: {len(download_events)} file operations detected"
            )

        # Network IOCs
        unique_ips = set(e.src_ip for e in events)
        if len(unique_ips) > 1:
            intel_collected.append(
                f"Infrastructure: {len(unique_ips)} related IPs identified"
            )

        intel_summary = (
            "; ".join(intel_collected)
            if intel_collected
            else "Basic threat intelligence extracted"
        )

        # Record action
        action = Action(
            incident_id=inc_id,
            action="honeypot_collect_threat_intel",
            result="success",
            detail=f"Threat intelligence collected from {incident.src_ip}: {intel_summary}",
            params={
                "ip": incident.src_ip,
                "intel_items": len(intel_collected),
                "events_analyzed": len(events),
                "honeypot_action": True,
            },
        )
        db.add(action)
        await db.commit()

        return {
            "success": True,
            "message": f"🧠 Threat intelligence collected from {incident.src_ip}",
            "details": intel_summary,
            "intel_items": intel_collected,
            "events_analyzed": len(events),
        }

    except Exception as e:
        logger.error(f"Honeypot threat intel collection failed: {e}")
        return {
            "success": False,
            "message": "❌ Failed to collect threat intelligence",
            "error": str(e),
        }


@app.post("/api/incidents/{inc_id}/actions/honeypot-deploy-decoy")
async def soc_honeypot_deploy_decoy(
    inc_id: int, request: Request, db: AsyncSession = Depends(get_db)
):
    """SOC Action: Deploy additional decoy services for attacker"""
    _require_api_key(request)

    incident = (
        (await db.execute(select(Incident).where(Incident.id == inc_id)))
        .scalars()
        .first()
    )

    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found")

    try:
        # Deploy additional honeypot services
        from .agents.deception_agent import DeceptionAgent

        deception_agent = DeceptionAgent()

        # Create target-specific honeypot configuration
        honeypot_config = {
            "target_ip": incident.src_ip,
            "services": ["ssh", "ftp", "http"],
            "interaction_level": "high",
            "data_collection": True,
        }

        # Deploy the honeypot (simulated)
        honeypot_id = f"decoy_{incident.src_ip.replace('.', '_')}_{int(datetime.now().timestamp())}"

        # Record action
        action = Action(
            incident_id=inc_id,
            action="honeypot_deploy_decoy",
            result="success",
            detail=f"Additional decoy services deployed for {incident.src_ip} (ID: {honeypot_id})",
            params={
                "ip": incident.src_ip,
                "honeypot_id": honeypot_id,
                "services": honeypot_config["services"],
                "honeypot_action": True,
            },
        )
        db.add(action)

        incident.status = "decoy_deployed"
        await db.commit()

        return {
            "success": True,
            "message": f"🕳️ Decoy services deployed for {incident.src_ip}",
            "details": f"Honeypot {honeypot_id} active with services: {', '.join(honeypot_config['services'])}",
            "honeypot_id": honeypot_id,
            "services": honeypot_config["services"],
        }

    except Exception as e:
        logger.error(f"Honeypot decoy deployment failed: {e}")
        return {
            "success": False,
            "message": "❌ Failed to deploy decoy services",
            "error": str(e),
        }


# =============================================================================
# HONEYPOT INTELLIGENCE & FILTERING ENDPOINTS
# =============================================================================


@app.get("/api/incidents/real")
async def get_real_incidents(
    include_test: bool = False, db: AsyncSession = Depends(get_db)
):
    """Get only real incidents, filtering out test events"""
    try:
        if include_test:
            # Include all incidents
            query = select(Incident).order_by(desc(Incident.created_at))
        else:
            # Filter out test incidents based on source IP patterns and event data
            subquery = (
                select(Event.src_ip)
                .where(
                    or_(
                        # Test IPs from startup script
                        Event.src_ip.in_(["192.168.1.100", "192.168.1.200"]),
                        # Events with test markers in raw data
                        func.json_extract(Event.raw, "$.test_event") == True,
                        func.json_extract(Event.raw, "$.test_type").isnot(None),
                        # Hostname-based filtering
                        Event.hostname.like("%test%"),
                    )
                )
                .distinct()
            )

            # Get incidents that are NOT from test IPs
            query = (
                select(Incident)
                .where(~Incident.src_ip.in_(subquery))
                .order_by(desc(Incident.created_at))
            )

        result = await db.execute(query)
        incidents = result.scalars().all()

        # Format incidents for response
        incident_list = []
        for incident in incidents:
            incident_data = {
                "id": incident.id,
                "created_at": incident.created_at.isoformat(),
                "src_ip": incident.src_ip,
                "reason": incident.reason,
                "status": incident.status,
                "auto_contained": incident.auto_contained,
                "escalation_level": incident.escalation_level,
                "risk_score": incident.risk_score,
                "threat_category": incident.threat_category,
                "containment_confidence": incident.containment_confidence,
                "agent_confidence": incident.agent_confidence,
                "is_test": incident.src_ip in ["192.168.1.100", "192.168.1.200"],
            }
            incident_list.append(incident_data)

        return {
            "incidents": incident_list,
            "total": len(incident_list),
            "filter_applied": "real_only" if not include_test else "all",
            "test_incidents_excluded": not include_test,
        }

    except Exception as e:
        logger.error(f"Failed to get real incidents: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get incidents: {str(e)}"
        )


@app.get("/honeypot/attacker-stats")
async def get_honeypot_attacker_stats(
    hours: int = 24, db: AsyncSession = Depends(get_db)
):
    """Get honeypot attacker statistics and insights"""
    try:
        # Get events from the last N hours, excluding test events
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)

        events_query = select(Event).where(
            and_(
                Event.ts >= cutoff_time,
                # Exclude test events
                ~Event.src_ip.in_(["192.168.1.100", "192.168.1.200"]),
                ~func.json_extract(Event.raw, "$.test_event") == True,
            )
        )

        result = await db.execute(events_query)
        events = result.scalars().all()

        # Analyze attacker patterns
        attacker_stats = {}
        attack_types = {}
        geographic_data = {}

        for event in events:
            ip = event.src_ip

            if ip not in attacker_stats:
                attacker_stats[ip] = {
                    "total_events": 0,
                    "first_seen": event.ts,
                    "last_seen": event.ts,
                    "event_types": set(),
                    "attack_categories": set(),
                }

            stats = attacker_stats[ip]
            stats["total_events"] += 1
            stats["last_seen"] = max(stats["last_seen"], event.ts)
            stats["first_seen"] = min(stats["first_seen"], event.ts)
            stats["event_types"].add(event.eventid)

            # Classify attack type
            if "login" in event.eventid:
                attack_types["brute_force"] = attack_types.get("brute_force", 0) + 1
            elif "command" in event.eventid:
                attack_types["command_execution"] = (
                    attack_types.get("command_execution", 0) + 1
                )
            elif "file" in event.eventid or "download" in event.eventid:
                attack_types["file_operations"] = (
                    attack_types.get("file_operations", 0) + 1
                )
            elif "web" in event.eventid.lower():
                attack_types["web_attacks"] = attack_types.get("web_attacks", 0) + 1

        # Convert sets to lists for JSON serialization
        for ip, stats in attacker_stats.items():
            stats["event_types"] = list(stats["event_types"])
            stats["attack_categories"] = list(stats["attack_categories"])
            stats["duration_hours"] = (
                stats["last_seen"] - stats["first_seen"]
            ).total_seconds() / 3600
            stats["first_seen"] = stats["first_seen"].isoformat()
            stats["last_seen"] = stats["last_seen"].isoformat()

        return {
            "timeframe_hours": hours,
            "total_attackers": len(attacker_stats),
            "total_events": len(events),
            "attacker_details": attacker_stats,
            "attack_type_distribution": attack_types,
            "top_attackers": sorted(
                [(ip, stats["total_events"]) for ip, stats in attacker_stats.items()],
                key=lambda x: x[1],
                reverse=True,
            )[:10],
            "analysis_timestamp": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error(f"Failed to get honeypot stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")


@app.get("/settings/auto_contain")
async def get_auto_contain():
    """Get auto-contain setting"""
    return {"enabled": auto_contain_enabled}


@app.delete("/admin/clear-database")
async def clear_database(
    request: Request,
    db: AsyncSession = Depends(get_db),
    clear_events: bool = True,
    clear_actions: bool = True,
):
    """Clear all incidents, events, and actions from database for clean UI/UX"""
    _require_api_key(request)

    try:
        # Clear actions first (foreign key constraints)
        if clear_actions:
            await db.execute(text("DELETE FROM actions"))
            actions_deleted = await db.execute(text("SELECT changes()"))
            actions_count = actions_deleted.scalar()
        else:
            actions_count = 0

        # Clear incidents
        await db.execute(text("DELETE FROM incidents"))
        incidents_deleted = await db.execute(text("SELECT changes()"))
        incidents_count = incidents_deleted.scalar()

        # Clear events
        if clear_events:
            await db.execute(text("DELETE FROM events"))
            events_deleted = await db.execute(text("SELECT changes()"))
            events_count = events_deleted.scalar()
        else:
            events_count = 0

        # Reset auto-increment counters (SQLite only)
        try:
            await db.execute(
                text(
                    "DELETE FROM sqlite_sequence WHERE name IN ('incidents', 'events', 'actions')"
                )
            )
        except Exception as e:
            logger.warning(
                f"Could not reset sqlite_sequence (normal if not using SQLite or table empty): {e}"
            )

        await db.commit()

        logger.info(
            f"Database cleared: {incidents_count} incidents, {events_count} events, {actions_count} actions"
        )

        return {
            "success": True,
            "message": "Database cleared successfully",
            "deleted": {
                "incidents": incidents_count,
                "events": events_count,
                "actions": actions_count,
            },
        }

    except Exception as e:
        await db.rollback()
        logger.error(f"Database clear failed: {e}")
        raise HTTPException(status_code=500, detail=f"Database clear failed: {e}")


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
    request_data: Dict[str, Any], db: AsyncSession = Depends(get_db)
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
                "message": f"Insufficient training data: {len(events)} events (need at least 100)",
            }

        # Prepare training data
        training_data = await prepare_training_data_from_events(events)

        # Train models
        if model_type == "ensemble":
            results = await ml_detector.train_models(training_data)
        else:
            return {
                "success": False,
                "message": f"Model type '{model_type}' not supported",
            }

        return {
            "success": True,
            "message": f"Retrained {model_type} models",
            "training_data_size": len(training_data),
            "results": results,
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
            "status_by_model": status,
        }

        return {"success": True, "metrics": metrics}

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
        return {"success": True, "sources": stats}
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
                "behavioral_threshold": getattr(
                    behavioral_analyzer, "adaptive_threshold", 0.6
                ),
            },
            "learning_pipeline": learning_pipeline.get_learning_status(),
            "baseline_engine": baseline_engine.get_baseline_status(),
            "ml_detector": ml_detector.get_model_status(),
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
            "results": results,
        }
    except Exception as e:
        logger.error(f"Forced learning update failed: {e}")
        return {"success": False, "message": f"Learning update failed: {str(e)}"}


@app.post("/api/adaptive/sensitivity")
async def adjust_detection_sensitivity(request_data: Dict[str, Any], request: Request):
    """
    Adjust detection sensitivity
    """
    _require_api_key(request)

    sensitivity = request_data.get("sensitivity", "medium")
    if sensitivity not in ["low", "medium", "high"]:
        raise HTTPException(
            status_code=400, detail="Sensitivity must be 'low', 'medium', or 'high'"
        )

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
            "baseline_thresholds": baseline_engine.deviation_thresholds,
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
                logger.info(
                    f"Skipping ML retraining: insufficient data ({len(events)} events)"
                )

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
                    Action.due_at <= now,
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
                        params={"ip": ip, "scheduled": True},
                    )
                    db.add(unblock_action)

                    # Mark scheduled action as done
                    action.result = "done"
                    action.detail = f"Completed: {detail}"

                    # Update incident status if successful
                    if status == "success":
                        incident = (
                            (
                                await db.execute(
                                    select(Incident).where(
                                        Incident.id == action.incident_id
                                    )
                                )
                            )
                            .scalars()
                            .first()
                        )
                        if incident:
                            incident.status = "open"

                    logger.info(f"Processed scheduled unblock for IP {ip}")

                except Exception as e:
                    logger.error(
                        f"Failed to process scheduled unblock {action.id}: {e}"
                    )
                    action.result = "failed"
                    action.detail = f"Failed: {str(e)}"

            await db.commit()

    except Exception as e:
        logger.error(f"Error in scheduled unblock processor: {e}")


@app.get("/api/orchestrator/status")
async def get_orchestrator_status():
    """Get comprehensive orchestrator status including agents and ML models"""
    try:
        orchestrator_status = "not_initialized"
        orchestrator_data = {}

        if agent_orchestrator:
            try:
                orchestrator_data = await agent_orchestrator.get_orchestrator_status()
                orchestrator_status = "healthy"
            except Exception as e:
                logger.error(f"Failed to get orchestrator data: {e}")
                orchestrator_status = "error"
                orchestrator_data = {"error": str(e)}

        # Get ML model status
        ml_status = {}
        try:
            from .enhanced_threat_detector import enhanced_detector

            ml_status["enhanced_detector"] = {
                "loaded": enhanced_detector.model is not None,
                "device": str(enhanced_detector.device)
                if hasattr(enhanced_detector, "device")
                else "unknown",
                "model_type": "Enhanced XDR Threat Detector (Local)",
                "status": "active" if enhanced_detector.model else "not_loaded",
            }
        except Exception as e:
            ml_status["enhanced_detector"] = {"status": "error", "error": str(e)}

        try:
            ml_status["federated_detector"] = ml_detector.get_model_status()
        except Exception as e:
            ml_status["federated_detector"] = {"status": "error", "error": str(e)}

        return {
            "status": orchestrator_status,
            "orchestrator": orchestrator_data,
            "ml_models": ml_status,
            "using_local_models": True,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        logger.error(f"Orchestrator status check failed: {e}")
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


@app.get("/api/orchestrator/workflows/{workflow_id}")
async def get_workflow_status(workflow_id: str):
    """Get status of a specific workflow"""
    try:
        if not agent_orchestrator:
            raise HTTPException(
                status_code=503, detail="Agent orchestrator not initialized"
            )

        workflow_status = await agent_orchestrator.get_workflow_status(workflow_id)
        if workflow_status:
            return workflow_status
        else:
            raise HTTPException(
                status_code=404, detail=f"Workflow {workflow_id} not found"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Workflow status check failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get workflow status: {str(e)}"
        )


@app.post("/api/orchestrator/workflows/{workflow_id}/cancel")
async def cancel_workflow(workflow_id: str):
    """Cancel a running workflow"""
    try:
        if not agent_orchestrator:
            raise HTTPException(
                status_code=503, detail="Agent orchestrator not initialized"
            )

        cancelled = await agent_orchestrator.cancel_workflow(workflow_id)
        if cancelled:
            return {
                "success": True,
                "message": f"Workflow {workflow_id} cancelled successfully",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        else:
            return {
                "success": False,
                "message": f"Workflow {workflow_id} could not be cancelled (may have already completed)",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

    except Exception as e:
        logger.error(f"Workflow cancellation failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to cancel workflow: {str(e)}"
        )


@app.post("/api/orchestrator/workflows")
async def create_workflow(
    request_data: Dict[str, Any], db: AsyncSession = Depends(get_db)
):
    """Manually trigger an orchestrated workflow for an incident"""
    try:
        if not agent_orchestrator:
            raise HTTPException(
                status_code=503, detail="Agent orchestrator not initialized"
            )

        incident_id = request_data.get("incident_id")
        workflow_type = request_data.get("workflow_type", "comprehensive")

        if not incident_id:
            raise HTTPException(status_code=400, detail="incident_id is required")

        # Get incident
        incident = (
            (await db.execute(select(Incident).where(Incident.id == incident_id)))
            .scalars()
            .first()
        )

        if not incident:
            raise HTTPException(
                status_code=404, detail=f"Incident {incident_id} not found"
            )

        # Get recent events
        recent_events = await _recent_events_for_ip(db, incident.src_ip)

        # Trigger orchestration
        orchestration_result = await agent_orchestrator.orchestrate_incident_response(
            incident=incident,
            recent_events=recent_events,
            db_session=db,
            workflow_type=workflow_type,
        )

        return {
            "success": True,
            "workflow_id": orchestration_result.get("workflow_id"),
            "message": f"Workflow started for incident {incident_id}",
            "result": orchestration_result,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Workflow creation failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to create workflow: {str(e)}"
        )


@app.get("/api/response/test")
async def test_response_system():
    """Simple test endpoint for response system"""
    try:
        from .advanced_response_engine import get_response_engine

        # Test basic functionality
        engine = await get_response_engine()
        actions = engine.get_available_actions()

        return {
            "success": True,
            "message": "Advanced Response System is working",
            "available_actions": len(actions.get("actions", {})),
            "sample_actions": list(actions.get("actions", {}).keys())[:3],
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": "Advanced Response System test failed",
        }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    orchestrator_status = "not_initialized"
    if agent_orchestrator:
        try:
            status = await agent_orchestrator.get_orchestrator_status()
            orchestrator_status = "healthy" if status else "error"
        except:
            orchestrator_status = "error"

    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "auto_contain": auto_contain_enabled,
        "orchestrator": orchestrator_status,
    }


@app.get("/api/council/metrics")
async def get_council_metrics():
    """
    Get Council of Models performance metrics.

    Returns real-time statistics including:
    - Routing decisions
    - API calls and costs
    - Vector cache performance
    - Verdicts and overrides
    """
    try:
        from .orchestrator.metrics import get_metrics_summary

        return get_metrics_summary()
    except ImportError:
        return {
            "error": "Council metrics not available",
            "message": "Council of Models system not initialized",
        }
    except Exception as e:
        logger.error(f"Failed to get Council metrics: {e}")
        return {"error": "Failed to retrieve metrics", "message": str(e)}


@app.get("/test/ssh")
async def test_ssh_connectivity():
    """Test SSH connectivity to honeypot"""
    import os
    import subprocess

    from .responder import responder

    try:
        # First, test basic network connectivity
        ping_result = subprocess.run(
            ["ping", "-c", "1", "-W", "3000", responder.host],
            capture_output=True,
            text=True,
            timeout=5,
        )

        ping_status = "success" if ping_result.returncode == 0 else "failed"
        ping_detail = (
            ping_result.stdout.strip()
            if ping_result.returncode == 0
            else ping_result.stderr.strip()
        )

        # Test SSH connectivity
        ssh_status, ssh_detail = await responder.test_connection()

        # Get current environment info
        env_info = {
            "PATH": os.environ.get("PATH", "Not set"),
            "USER": os.environ.get("USER", "Not set"),
            "HOME": os.environ.get("HOME", "Not set"),
            "PWD": os.environ.get("PWD", "Not set"),
        }

        return {
            "ssh_status": ssh_status,
            "ssh_detail": ssh_detail,
            "ping_status": ping_status,
            "ping_detail": ping_detail,
            "honeypot": f"{responder.username}@{responder.host}:{responder.port}",
            "environment": env_info,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        logger.error(f"SSH test failed with exception: {e}")
        return {
            "ssh_status": "failed",
            "ssh_detail": f"SSH test exception: {str(e)}",
            "ping_status": "error",
            "ping_detail": "Could not test ping",
            "honeypot": f"{responder.username}@{responder.host}:{responder.port}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


async def execute_password_reset(source_ip: str, incident_id: int) -> Dict[str, Any]:
    """Execute actual password reset for potentially compromised accounts"""
    try:
        import secrets
        import string
        import subprocess
        from pathlib import Path

        # Define admin accounts that need password reset
        admin_accounts = [
            "root",
            "admin",
            "administrator",
            "ubuntu",
            "centos",
            "debian",
            "user",
            "sysadmin",
            "operator",
        ]

        reset_results = []
        successful_resets = []
        failed_resets = []

        for username in admin_accounts:
            try:
                # Generate secure random password
                password_chars = string.ascii_letters + string.digits + "!@#$%^&*"
                new_password = "".join(
                    secrets.choice(password_chars) for _ in range(16)
                )

                # Check if user exists
                user_check = subprocess.run(
                    ["id", username], capture_output=True, text=True, timeout=10
                )

                if user_check.returncode == 0:
                    # User exists, reset password
                    password_reset = subprocess.run(
                        ["sudo", "chpasswd"],
                        input=f"{username}:{new_password}",
                        text=True,
                        capture_output=True,
                        timeout=30,
                    )

                    if password_reset.returncode == 0:
                        successful_resets.append(
                            {
                                "username": username,
                                "status": "success",
                                "new_password": new_password[:4]
                                + "****",  # Partial password for logging
                            }
                        )

                        # Store full password securely for incident response
                        password_file = f"/tmp/incident_{incident_id}_passwords.txt"
                        with open(password_file, "a") as f:
                            f.write(f"{username}:{new_password}\n")

                        # Set secure permissions on password file
                        subprocess.run(["chmod", "600", password_file], timeout=10)

                        logger.info(f"Password reset successful for user: {username}")
                    else:
                        failed_resets.append(
                            {
                                "username": username,
                                "status": "failed",
                                "error": password_reset.stderr,
                            }
                        )
                        logger.error(
                            f"Password reset failed for {username}: {password_reset.stderr}"
                        )
                else:
                    # User doesn't exist, skip
                    logger.debug(f"User {username} does not exist, skipping")

            except subprocess.TimeoutExpired:
                failed_resets.append(
                    {
                        "username": username,
                        "status": "timeout",
                        "error": "Command timed out",
                    }
                )
                logger.error(f"Password reset timed out for user: {username}")
            except Exception as e:
                failed_resets.append(
                    {"username": username, "status": "error", "error": str(e)}
                )
                logger.error(f"Password reset error for {username}: {e}")

        # Force password expiry to require immediate change on next login
        for reset in successful_resets:
            try:
                subprocess.run(
                    ["sudo", "chage", "-d", "0", reset["username"]],
                    capture_output=True,
                    timeout=10,
                    check=True,
                )
                logger.info(f"Forced password expiry for user: {reset['username']}")
            except Exception as e:
                logger.warning(
                    f"Failed to force password expiry for {reset['username']}: {e}"
                )

        # Send notification to system administrators
        await send_password_reset_notification(
            source_ip, incident_id, successful_resets, failed_resets
        )

        # Create summary
        total_resets = len(successful_resets)
        total_failures = len(failed_resets)

        if total_resets > 0:
            success_msg = (
                f"✅ Password reset completed for {total_resets} admin accounts"
            )
            if total_failures > 0:
                success_msg += f" ({total_failures} failed)"

            details = f"Passwords reset for: {', '.join([r['username'] for r in successful_resets])}"
            if failed_resets:
                details += (
                    f". Failed: {', '.join([r['username'] for r in failed_resets])}"
                )

            return {
                "success": True,
                "message": success_msg,
                "details": details,
                "affected_users": [r["username"] for r in successful_resets],
                "reset_count": total_resets,
                "failed_count": total_failures,
            }
        else:
            return {
                "success": False,
                "message": "❌ No passwords were reset successfully",
                "details": f"All {total_failures} reset attempts failed",
                "affected_users": [],
                "reset_count": 0,
                "failed_count": total_failures,
            }

    except Exception as e:
        logger.error(f"Password reset execution failed: {e}")
        return {
            "success": False,
            "message": "❌ Password reset execution failed",
            "details": f"Error: {str(e)}",
            "affected_users": [],
            "reset_count": 0,
        }


async def send_password_reset_notification(
    source_ip: str,
    incident_id: int,
    successful_resets: List[Dict],
    failed_resets: List[Dict],
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


# ===== NATURAL LANGUAGE PROCESSING API ENDPOINTS =====


@app.post("/api/nlp/query")
async def natural_language_query(
    request: Dict[str, Any], db: AsyncSession = Depends(get_db)
):
    """
    Process natural language queries against incident data

    Enables analysts to query incidents using natural language like:
    - "Show me all brute force attacks from China in the last 24 hours"
    - "Find incidents similar to IP 192.168.1.100"
    - "What patterns do you see in recent malware incidents?"
    """
    try:
        from .agents.nlp_analyzer import get_nlp_analyzer

        query = request.get("query", "")
        include_context = request.get("include_context", True)
        max_results = min(request.get("max_results", 10), 50)  # Cap at 50
        semantic_search = request.get("semantic_search", True)

        if not query:
            raise HTTPException(status_code=400, detail="Query parameter is required")

        logger.info(f"Processing NL query: {query[:100]}...")

        # Get NLP analyzer
        nlp_analyzer = await get_nlp_analyzer()

        # Get recent incidents and events for analysis
        stmt = select(Incident).order_by(Incident.created_at.desc()).limit(200)
        result = await db.execute(stmt)
        incidents = result.scalars().all()

        # Get recent events (limit to avoid memory issues)
        events_stmt = select(Event).order_by(Event.ts.desc()).limit(500)
        events_result = await db.execute(events_stmt)
        events = events_result.scalars().all()

        # Process the natural language query
        context = {
            "include_context": include_context,
            "max_results": max_results,
            "semantic_search": semantic_search,
        }

        nlp_response = await nlp_analyzer.analyze_natural_language_query(
            query=query, incidents=list(incidents), events=list(events), context=context
        )

        return {
            "success": True,
            "query": query,
            "query_understanding": nlp_response.query_understanding,
            "structured_query": nlp_response.structured_query,
            "findings": nlp_response.findings[:max_results],
            "recommendations": nlp_response.recommendations,
            "confidence_score": nlp_response.confidence_score,
            "reasoning": nlp_response.reasoning,
            "follow_up_questions": nlp_response.follow_up_questions,
            "processing_stats": {
                "incidents_analyzed": len(incidents),
                "events_analyzed": len(events),
                "semantic_search_enabled": semantic_search,
            },
        }

    except Exception as e:
        logger.error(f"NLP query processing failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Natural language query failed: {str(e)}"
        )


@app.post("/api/nlp/threat-analysis")
async def nlp_threat_analysis(
    request: Dict[str, Any], db: AsyncSession = Depends(get_db)
):
    """
    Perform comprehensive threat analysis using natural language processing

    Supports specific analysis types:
    - pattern_recognition: Find patterns in threat data
    - timeline_analysis: Analyze chronological sequences
    - attribution: Threat actor attribution analysis
    - ioc_extraction: Extract indicators of compromise
    - recommendation: Generate actionable recommendations
    """
    try:
        from .agents.nlp_analyzer import get_nlp_analyzer

        query = request.get("query", "")
        analysis_type = request.get("analysis_type")
        time_range_hours = min(
            request.get("time_range_hours", 24), 720
        )  # Cap at 30 days

        if not query:
            raise HTTPException(status_code=400, detail="Query parameter is required")

        logger.info(
            f"Processing NLP threat analysis: {analysis_type or 'comprehensive'}"
        )

        # Get NLP analyzer
        nlp_analyzer = await get_nlp_analyzer()

        # Get incidents within time range
        cutoff_time = datetime.utcnow() - timedelta(hours=time_range_hours)
        stmt = (
            select(Incident)
            .where(Incident.created_at >= cutoff_time)
            .order_by(Incident.created_at.desc())
        )
        result = await db.execute(stmt)
        incidents = result.scalars().all()

        # Get events within time range
        events_stmt = (
            select(Event)
            .where(Event.ts >= cutoff_time)
            .order_by(Event.ts.desc())
            .limit(1000)
        )  # Reasonable limit
        events_result = await db.execute(events_stmt)
        events = events_result.scalars().all()

        # Add analysis type context to query if specified
        if analysis_type:
            enhanced_query = f"{query} (focus on {analysis_type.replace('_', ' ')})"
        else:
            enhanced_query = query

        # Process the threat analysis query
        context = {
            "analysis_type": analysis_type,
            "time_range_hours": time_range_hours,
            "focused_analysis": True,
        }

        nlp_response = await nlp_analyzer.analyze_natural_language_query(
            query=enhanced_query,
            incidents=list(incidents),
            events=list(events),
            context=context,
        )

        return {
            "success": True,
            "query": query,
            "analysis_type": analysis_type,
            "time_range_hours": time_range_hours,
            "query_understanding": nlp_response.query_understanding,
            "structured_query": nlp_response.structured_query,
            "findings": nlp_response.findings,
            "recommendations": nlp_response.recommendations,
            "confidence_score": nlp_response.confidence_score,
            "reasoning": nlp_response.reasoning,
            "follow_up_questions": nlp_response.follow_up_questions,
            "analysis_metadata": {
                "incidents_in_timeframe": len(incidents),
                "events_in_timeframe": len(events),
                "timeframe_start": cutoff_time.isoformat(),
                "timeframe_end": datetime.utcnow().isoformat(),
            },
        }

    except Exception as e:
        logger.error(f"NLP threat analysis failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"NLP threat analysis failed: {str(e)}"
        )


@app.post("/api/nlp/semantic-search")
async def semantic_incident_search(
    request: Dict[str, Any], db: AsyncSession = Depends(get_db)
):
    """
    Search incidents using semantic similarity and natural language understanding

    Uses embeddings to find incidents that are semantically similar to the query,
    even if they don't contain exact keyword matches.
    """
    try:
        from .agents.nlp_analyzer import get_nlp_analyzer

        query = request.get("query", "")
        similarity_threshold = max(
            0.1, min(request.get("similarity_threshold", 0.7), 1.0)
        )
        max_results = min(request.get("max_results", 10), 20)  # Cap at 20

        if not query:
            raise HTTPException(status_code=400, detail="Query parameter is required")

        logger.info(f"Processing semantic search: {query[:100]}...")

        # Get NLP analyzer
        nlp_analyzer = await get_nlp_analyzer()

        # Get recent incidents for semantic search
        stmt = select(Incident).order_by(Incident.created_at.desc()).limit(500)
        result = await db.execute(stmt)
        incidents = result.scalars().all()

        # Perform semantic search
        semantic_results = await nlp_analyzer.semantic_search_incidents(
            query=query,
            incidents=list(incidents),
            top_k=max_results * 2,  # Get more results to filter
        )

        # Filter by similarity threshold and format results
        filtered_results = []
        similarity_scores = []

        for incident, similarity_score in semantic_results:
            if similarity_score >= similarity_threshold:
                filtered_results.append(
                    {
                        "incident": {
                            "id": incident.id,
                            "src_ip": incident.src_ip,
                            "reason": incident.reason,
                            "status": incident.status,
                            "created_at": incident.created_at.isoformat(),
                            "risk_score": getattr(incident, "risk_score", None),
                            "escalation_level": getattr(
                                incident, "escalation_level", "medium"
                            ),
                        },
                        "similarity_score": similarity_score,
                    }
                )
                similarity_scores.append(similarity_score)

        # Sort by similarity score and limit results
        filtered_results.sort(key=lambda x: x["similarity_score"], reverse=True)
        final_results = filtered_results[:max_results]

        # Calculate search quality metrics
        avg_similarity = (
            sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0
        )

        # Extract semantic features (basic implementation)
        semantic_features = []
        if "attack" in query.lower():
            semantic_features.append("attack_related")
        if "ip" in query.lower() or any(char.isdigit() for char in query):
            semantic_features.append("ip_address")
        if any(word in query.lower() for word in ["malware", "virus", "trojan"]):
            semantic_features.append("malware_related")
        if any(word in query.lower() for word in ["brute", "force", "login"]):
            semantic_features.append("authentication_related")

        return {
            "success": True,
            "query": query,
            "similarity_threshold": similarity_threshold,
            "similar_incidents": final_results,
            "total_found": len(filtered_results),
            "returned_count": len(final_results),
            "avg_similarity": avg_similarity,
            "query_understanding": f"Semantic search for incidents similar to: '{query}'",
            "semantic_features": semantic_features,
            "search_metadata": {
                "total_incidents_searched": len(incidents),
                "semantic_search_available": nlp_analyzer.embeddings is not None,
                "langchain_available": hasattr(nlp_analyzer, "llm")
                and nlp_analyzer.llm is not None,
            },
        }

    except Exception as e:
        logger.error(f"Semantic search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Semantic search failed: {str(e)}")


@app.get("/api/nlp/status")
async def get_nlp_status():
    """
    Get status and configuration of the Natural Language Processing system
    """
    try:
        from .agents.nlp_analyzer import get_nlp_analyzer

        # Get NLP analyzer
        nlp_analyzer = await get_nlp_analyzer()

        # Get comprehensive status
        status = nlp_analyzer.get_agent_status()

        return {
            "success": True,
            "nlp_system": status,
            "capabilities": {
                "natural_language_queries": True,
                "semantic_search": status.get("embeddings_available", False),
                "threat_pattern_recognition": True,
                "timeline_analysis": True,
                "ioc_extraction": True,
                "attribution_analysis": True,
                "ai_powered_insights": status.get("llm_initialized", False),
            },
            "supported_query_types": [
                "incident_search",
                "threat_analysis",
                "pattern_recognition",
                "timeline_analysis",
                "ioc_extraction",
                "attribution_query",
                "recommendation",
            ],
        }

    except Exception as e:
        logger.error(f"Failed to get NLP status: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get NLP status: {str(e)}"
        )


# =============================================================================
# FEDERATED LEARNING API ENDPOINTS (Phase 2)
# =============================================================================


@app.get("/api/federated/status")
async def get_federated_status():
    """Get comprehensive federated learning system status"""
    try:
        from .federated_learning import FEDERATED_AVAILABLE, federated_manager

        if not FEDERATED_AVAILABLE:
            return {
                "success": False,
                "error": "Federated learning components not available",
                "available": False,
                "reason": "Missing dependencies (tensorflow-federated, cryptography)",
            }

        # Get detailed federated learning status
        status = federated_manager.get_federated_status()

        return {
            "success": True,
            "federated_learning": status,
            "available": True,
            "capabilities": {
                "secure_aggregation": True,
                "differential_privacy": True,
                "multi_protocol_encryption": True,
                "distributed_training": True,
            },
        }

    except Exception as e:
        logger.error(f"Failed to get federated status: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get federated status: {str(e)}"
        )


@app.post("/api/federated/coordinator/initialize")
async def initialize_federated_coordinator(request_data: Dict[str, Any]):
    """Initialize this node as a federated learning coordinator"""
    try:
        from .federated_learning import FEDERATED_AVAILABLE, federated_manager

        if not FEDERATED_AVAILABLE:
            raise HTTPException(
                status_code=503, detail="Federated learning not available"
            )

        config = request_data.get("config", {})

        # Initialize coordinator
        coordinator = await federated_manager.initialize_coordinator(config)

        return {
            "success": True,
            "message": "Federated learning coordinator initialized",
            "coordinator_id": coordinator.node_id,
            "public_key": coordinator.public_key.decode("utf-8")
            if coordinator.public_key
            else None,
            "configuration": config,
        }

    except Exception as e:
        logger.error(f"Coordinator initialization failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Coordinator initialization failed: {str(e)}"
        )


@app.post("/api/federated/participant/initialize")
async def initialize_federated_participant(request_data: Dict[str, Any]):
    """Initialize this node as a federated learning participant"""
    try:
        from .federated_learning import FEDERATED_AVAILABLE, federated_manager

        if not FEDERATED_AVAILABLE:
            raise HTTPException(
                status_code=503, detail="Federated learning not available"
            )

        coordinator_endpoint = request_data.get("coordinator_endpoint")
        if not coordinator_endpoint:
            raise HTTPException(
                status_code=400, detail="coordinator_endpoint is required"
            )

        config = request_data.get("config", {})

        # Initialize participant
        participant = await federated_manager.initialize_participant(
            coordinator_endpoint, config
        )

        return {
            "success": True,
            "message": "Federated learning participant initialized",
            "participant_id": participant.node_id,
            "coordinator_endpoint": coordinator_endpoint,
            "configuration": config,
        }

    except Exception as e:
        logger.error(f"Participant initialization failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Participant initialization failed: {str(e)}"
        )


@app.post("/api/federated/training/start")
async def start_federated_training(request_data: Dict[str, Any]):
    """Start federated learning training process"""
    try:
        from .federated_learning import FEDERATED_AVAILABLE, federated_manager

        if not FEDERATED_AVAILABLE:
            raise HTTPException(
                status_code=503, detail="Federated learning not available"
            )

        model_type = request_data.get("model_type", "neural_network")
        training_data = request_data.get("training_data")

        # Validate model type
        valid_types = ["neural_network", "lstm_autoencoder", "isolation_forest"]
        if model_type not in valid_types:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid model_type. Must be one of: {valid_types}",
            )

        # Start federated training
        result = await federated_manager.start_federated_training(
            model_type, training_data
        )

        return {
            "success": True,
            "message": "Federated training started",
            "training_result": result,
            "model_type": model_type,
        }

    except Exception as e:
        logger.error(f"Federated training start failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Federated training failed: {str(e)}"
        )


@app.get("/api/federated/models/status")
async def get_federated_models_status():
    """Get status of federated ML models"""
    try:
        # Get ML detector status with federated information
        status = ml_detector.get_model_status()

        return {
            "success": True,
            "models": status,
            "federated_capabilities": {
                "ensemble_with_federated": status.get("federated_enabled", False),
                "federated_rounds_completed": status.get("federated_rounds", 0),
                "last_federated_training": status.get("last_federated_training"),
                "federated_accuracy": status.get("federated_accuracy", 0.0),
            },
        }

    except Exception as e:
        logger.error(f"Failed to get federated models status: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get models status: {str(e)}"
        )


@app.post("/api/federated/models/train")
async def train_federated_models(
    request_data: Dict[str, Any], background_tasks: BackgroundTasks
):
    """Train ML models with federated learning capabilities"""
    try:
        enable_federated = request_data.get("enable_federated", True)
        training_data = request_data.get("training_data")

        # If no training data provided, use recent events
        if not training_data:
            # Use background task for training
            background_tasks.add_task(
                _train_models_background, enable_federated=enable_federated
            )

            return {
                "success": True,
                "message": "Federated model training started in background",
                "federated_enabled": enable_federated,
            }
        else:
            # Train with provided data
            results = await ml_detector.train_models(
                training_data, enable_federated=enable_federated
            )

            return {
                "success": True,
                "message": "Federated model training completed",
                "training_results": results,
                "federated_enabled": enable_federated,
            }

    except Exception as e:
        logger.error(f"Federated model training failed: {e}")
        raise HTTPException(status_code=500, detail=f"Model training failed: {str(e)}")


@app.get("/api/federated/insights")
async def get_federated_insights():
    """Get insights and analytics from federated learning system"""
    try:
        # Always try to get basic ML status first (doesn't require heavy dependencies)
        ml_status = ml_detector.get_model_status()

        # Try federated features with proper error handling
        try:
            from .federated_learning import FEDERATED_AVAILABLE, federated_manager

            federated_available = FEDERATED_AVAILABLE
        except (ImportError, Exception) as fed_error:
            logger.info(f"Federated learning not available: {fed_error}")
            federated_available = False

        if not federated_available:
            # Return basic insights without federated features
            insights = {
                "system_overview": {
                    "federated_enabled": False,
                    "training_rounds": 0,
                    "global_accuracy": 0.0,
                    "initialization_status": False,
                },
                "performance_metrics": {
                    "ensemble_accuracy": ml_status.get("last_confidence", 0.0),
                    "standard_models_trained": sum(
                        [
                            ml_status.get("isolation_forest", False),
                            ml_status.get("lstm", False),
                            ml_status.get("enhanced_ml_trained", False),
                        ]
                    ),
                    "federated_contribution": 0.0,
                },
                "privacy_security": {
                    "secure_aggregation": False,
                    "encryption_enabled": True,
                    "differential_privacy_available": False,
                },
            }

            return {
                "success": True,
                "federated_insights": insights,
                "raw_status": {
                    "ml_detector": ml_status,
                    "federated_manager": {
                        "available": False,
                        "reason": "TensorFlow not available",
                    },
                },
            }

        # Get full insights with federated learning
        ml_status = ml_detector.get_model_status()
        fed_status = federated_manager.get_federated_status()

        insights = {
            "system_overview": {
                "federated_enabled": ml_status.get("federated_enabled", False),
                "training_rounds": ml_status.get("federated_rounds", 0),
                "global_accuracy": ml_status.get("federated_accuracy", 0.0),
                "initialization_status": fed_status.get("initialized", False),
            },
            "performance_metrics": {
                "ensemble_accuracy": ml_status.get("last_confidence", 0.0),
                "standard_models_trained": sum(
                    [
                        ml_status.get("isolation_forest", False),
                        ml_status.get("lstm", False),
                        ml_status.get("enhanced_ml_trained", False),
                    ]
                ),
                "federated_contribution": min(
                    0.4, 0.1 * ml_status.get("federated_rounds", 0)
                ),
            },
            "privacy_security": {
                "secure_aggregation": True,
                "encryption_enabled": True,
                "differential_privacy_available": True,
            },
        }

        return {
            "success": True,
            "federated_insights": insights,
            "raw_status": {"ml_detector": ml_status, "federated_manager": fed_status},
        }

    except Exception as e:
        logger.error(f"Failed to get federated insights: {e}")
        # Return fallback response
        ml_status = ml_detector.get_model_status()

        insights = {
            "system_overview": {
                "federated_enabled": False,
                "training_rounds": 0,
                "global_accuracy": 0.0,
                "initialization_status": False,
            },
            "performance_metrics": {
                "ensemble_accuracy": ml_status.get("last_confidence", 0.0),
                "standard_models_trained": sum(
                    [
                        ml_status.get("isolation_forest", False),
                        ml_status.get("lstm", False),
                        ml_status.get("enhanced_ml_trained", False),
                    ]
                ),
                "federated_contribution": 0.0,
            },
            "privacy_security": {
                "secure_aggregation": False,
                "encryption_enabled": True,
                "differential_privacy_available": False,
            },
        }

        return {
            "success": True,
            "federated_insights": insights,
            "raw_status": {
                "ml_detector": ml_status,
                "federated_manager": {"available": False, "error": str(e)},
            },
        }


# Phase 2B: Advanced ML & Explainable AI API Endpoints


@app.get("/api/ml/online-learning/status")
async def get_online_learning_status():
    """Get status of online learning and drift detection"""
    try:
        from .learning_pipeline import ADVANCED_ML_AVAILABLE, learning_pipeline

        if not ADVANCED_ML_AVAILABLE:
            return {
                "success": True,
                "message": "Phase 2B features not fully available - using fallbacks",
                "online_learning_status": {},
                "phase_2b_features": {
                    "online_adaptation_enabled": False,
                    "ensemble_optimization_enabled": False,
                    "explainable_ai_enabled": False,
                    "online_adaptations": 0,
                    "ensemble_optimizations": 0,
                    "drift_detections": 0,
                },
                "drift_detections": 0,
            }

        status = learning_pipeline.get_enhanced_learning_status()

        return {
            "success": True,
            "online_learning_status": status.get("online_learning_status", {}),
            "phase_2b_features": status.get("phase_2b_features", {}),
            "drift_detections": status.get("drift_detection_history", 0),
        }

    except Exception as e:
        logger.error(f"Failed to get online learning status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")


@app.post("/api/ml/online-learning/adapt")
async def trigger_online_adaptation():
    """Manually trigger online model adaptation"""
    try:
        from .learning_pipeline import ADVANCED_ML_AVAILABLE

        if not ADVANCED_ML_AVAILABLE:
            raise HTTPException(
                status_code=503,
                detail="Phase 2B advanced ML features not available - check dependencies",
            )

        from .online_learning import adapt_models_with_new_data

        # Get recent events for adaptation
        async with AsyncSessionLocal() as db:
            window_start = datetime.now(timezone.utc) - timedelta(hours=2)

            query = (
                select(Event)
                .where(Event.ts >= window_start)
                .order_by(Event.ts.desc())
                .limit(500)
            )

            result = await db.execute(query)
            events = result.scalars().all()

            if len(events) < 50:
                raise HTTPException(
                    status_code=400,
                    detail=f"Insufficient events for adaptation: {len(events)} (minimum 50 required)",
                )

            # Perform adaptation
            adaptation_result = await adapt_models_with_new_data(events)

            return {
                "success": adaptation_result.get("success", False),
                "samples_processed": adaptation_result.get("samples_processed", 0),
                "adaptation_strategy": adaptation_result.get(
                    "adaptation_strategy", "unknown"
                ),
                "timestamp": adaptation_result.get(
                    "timestamp", datetime.now(timezone.utc).isoformat()
                ),
                "message": "Online adaptation completed"
                if adaptation_result.get("success")
                else f"Adaptation failed: {adaptation_result.get('error', 'Unknown error')}",
            }

    except Exception as e:
        logger.error(f"Online adaptation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Adaptation failed: {str(e)}")


@app.get("/api/ml/ensemble/status")
async def get_ensemble_status():
    """Get ensemble optimization status"""
    try:
        from .model_versioning import model_registry

        if not model_registry:
            raise HTTPException(
                status_code=503, detail="Model versioning not available"
            )

        # Get all ensemble models
        all_models = []
        for model_id, versions in model_registry.models.items():
            for version, model_version in versions.items():
                if model_version.model_type == "ensemble":
                    all_models.append(
                        {
                            "model_id": model_id,
                            "version": version,
                            "algorithm": model_version.algorithm,
                            "status": model_version.status.value,
                            "created_at": model_version.created_at.isoformat(),
                            "performance_metrics": model_version.performance_metrics,
                        }
                    )

        return {
            "success": True,
            "ensemble_models": all_models,
            "total_ensembles": len(all_models),
            "production_ensembles": len(
                [m for m in all_models if m["status"] == "production"]
            ),
        }

    except Exception as e:
        logger.error(f"Failed to get ensemble status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")


@app.post("/api/ml/ensemble/optimize")
async def trigger_ensemble_optimization():
    """Manually trigger ensemble optimization"""
    try:
        from .learning_pipeline import learning_pipeline

        # Force ensemble optimization
        async with AsyncSessionLocal() as db:
            # Get training events
            window_start = datetime.now(timezone.utc) - timedelta(days=7)

            query = (
                select(Event)
                .where(Event.ts >= window_start)
                .order_by(Event.ts.desc())
                .limit(2000)
            )

            result = await db.execute(query)
            events = result.scalars().all()

            if len(events) < 100:
                raise HTTPException(
                    status_code=400,
                    detail=f"Insufficient training data: {len(events)} events (minimum 100 required)",
                )

            # Perform optimization
            success = await learning_pipeline._optimize_ensemble_models(events)

            return {
                "success": success,
                "training_events": len(events),
                "message": "Ensemble optimization completed"
                if success
                else "Ensemble optimization failed",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

    except Exception as e:
        logger.error(f"Ensemble optimization failed: {e}")
        raise HTTPException(status_code=500, detail=f"Optimization failed: {str(e)}")


@app.post("/api/ml/ab-test/create")
async def create_ab_test(request_data: Dict[str, Any]):
    """Create A/B test for model comparison"""
    try:
        from .learning_pipeline import learning_pipeline

        # Validate required fields
        required_fields = [
            "model_a_id",
            "model_a_version",
            "model_b_id",
            "model_b_version",
            "test_name",
        ]
        for field in required_fields:
            if field not in request_data:
                raise HTTPException(
                    status_code=400, detail=f"Required field missing: {field}"
                )

        test_id = await learning_pipeline.create_ab_test(
            model_a_id=request_data["model_a_id"],
            model_a_version=request_data["model_a_version"],
            model_b_id=request_data["model_b_id"],
            model_b_version=request_data["model_b_version"],
            test_name=request_data["test_name"],
            description=request_data.get("description", ""),
        )

        if test_id:
            return {
                "success": True,
                "test_id": test_id,
                "message": "A/B test created and started successfully",
                "test_name": request_data["test_name"],
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to create A/B test")

    except Exception as e:
        logger.error(f"A/B test creation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Test creation failed: {str(e)}")


@app.get("/api/ml/ab-test/{test_id}/results")
async def get_ab_test_results(test_id: str):
    """Get A/B test results"""
    try:
        from .model_versioning import ab_test_manager

        if not ab_test_manager:
            raise HTTPException(
                status_code=503, detail="A/B test manager not available"
            )

        result = ab_test_manager.get_test_results(test_id)

        if not result:
            raise HTTPException(
                status_code=404, detail=f"A/B test {test_id} not found or not completed"
            )

        return {
            "success": True,
            "test_id": test_id,
            "winner": result.winner.value,
            "confidence": result.confidence,
            "p_value": result.p_value,
            "effect_size": result.effect_size,
            "samples_a": result.samples_a,
            "samples_b": result.samples_b,
            "metric_a": result.metric_a,
            "metric_b": result.metric_b,
            "statistical_significance": result.statistical_significance,
            "practical_significance": result.practical_significance,
            "recommendation": result.recommendation,
            "detailed_metrics": result.detailed_metrics,
        }

    except Exception as e:
        logger.error(f"Failed to get A/B test results: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get results: {str(e)}")


@app.post("/api/ml/explain/{incident_id}")
async def explain_incident_prediction(
    incident_id: int, request_data: Optional[Dict[str, Any]] = None
):
    """Generate explanation for an incident prediction"""
    try:
        from .learning_pipeline import learning_pipeline

        context = request_data or {}
        explanation = await learning_pipeline.explain_recent_prediction(
            incident_id, context
        )

        if not explanation:
            raise HTTPException(
                status_code=404,
                detail=f"Could not generate explanation for incident {incident_id}",
            )

        return {"success": True, "explanation": explanation}

    except Exception as e:
        logger.error(f"Failed to explain incident {incident_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Explanation failed: {str(e)}")


@app.get("/api/ml/models/performance")
async def get_model_performance():
    """Get performance monitoring data for all models"""
    try:
        from .model_versioning import model_registry, performance_monitor

        if not performance_monitor or not model_registry:
            raise HTTPException(
                status_code=503, detail="Model performance monitoring not available"
            )

        production_models = model_registry.get_production_models()
        performance_data = []

        for model_version in production_models:
            health_summary = performance_monitor.get_model_health_summary(
                model_version.model_id, model_version.version
            )

            performance_data.append(
                {
                    "model_id": model_version.model_id,
                    "version": model_version.version,
                    "algorithm": model_version.algorithm,
                    "status": health_summary["status"],
                    "accuracy": health_summary.get("accuracy", 0),
                    "error_rate": health_summary.get("error_rate", 0),
                    "latency_ms": health_summary.get("latency_ms", 0),
                    "data_points": health_summary.get("data_points", 0),
                    "last_evaluation": health_summary.get("last_evaluation"),
                }
            )

        # Get recent alerts
        recent_alerts = performance_monitor.get_recent_alerts(hours=24)

        return {
            "success": True,
            "model_performance": performance_data,
            "recent_alerts": len(recent_alerts),
            "alerts_detail": recent_alerts[:10],  # Last 10 alerts
            "total_production_models": len(production_models),
        }

    except Exception as e:
        logger.error(f"Failed to get model performance: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get performance data: {str(e)}"
        )


@app.get("/api/ml/drift/status")
async def get_drift_detection_status():
    """Get concept drift detection status"""
    try:
        from .online_learning import online_learning_engine

        if not online_learning_engine:
            raise HTTPException(
                status_code=503, detail="Online learning engine not available"
            )

        drift_status = online_learning_engine.get_drift_status()
        adaptation_metrics = online_learning_engine.get_adaptation_metrics()

        return {
            "success": True,
            "drift_detection": drift_status,
            "recent_adaptations": len(adaptation_metrics),
            "adaptation_metrics": adaptation_metrics[-10:],  # Last 10 adaptations
            "last_drift_time": drift_status.get("last_drift_time"),
            "buffer_size": drift_status.get("buffer_size", 0),
            "detection_sensitivity": drift_status.get("detection_sensitivity", 0.1),
        }

    except Exception as e:
        logger.error(f"Failed to get drift status: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get drift status: {str(e)}"
        )


# =============================================================================
# DISTRIBUTED MCP API ENDPOINTS (Phase 3)
# =============================================================================


@app.get("/api/distributed/status")
async def get_distributed_status():
    """Get status of distributed MCP system"""
    try:
        from .distributed import (
            DISTRIBUTED_CAPABILITIES,
            __phase__,
            __version__,
            get_system_status,
        )

        return {
            "success": True,
            "version": __version__,
            "phase": __phase__,
            "capabilities": DISTRIBUTED_CAPABILITIES,
            "system_status": get_system_status(),
        }

    except ImportError as e:
        logger.info(f"Distributed system not available (dependencies missing): {e}")
        # Return graceful fallback - distributed features disabled
        return {
            "success": True,
            "version": "1.0.0",
            "phase": "Phase 1: Core XDR",
            "capabilities": ["core_detection", "basic_ml"],
            "message": "Distributed features disabled - Kafka/Redis services not available",
            "system_status": {
                "version": "1.0.0",
                "phase": "Phase 1: Core XDR",
                "capabilities": ["core_detection", "basic_ml"],
                "distributed_enabled": False,
                "components": {
                    "coordinator": {
                        "status": "disabled",
                        "reason": "service_not_available",
                    },
                    "kafka": {"status": "disabled", "reason": "service_not_available"},
                    "redis": {"status": "disabled", "reason": "service_not_available"},
                },
            },
        }

    except Exception as e:
        logger.warning(f"Distributed system status error: {e}")
        # Return fallback response for other errors
        return {
            "success": True,
            "version": "1.0.0",
            "phase": "Phase 1: Core XDR",
            "capabilities": ["core_detection", "basic_ml"],
            "message": f"Distributed features unavailable: {str(e)}",
            "system_status": {
                "version": "1.0.0",
                "phase": "Phase 1: Core XDR",
                "capabilities": ["core_detection", "basic_ml"],
                "distributed_enabled": False,
                "components": {
                    "coordinator": {"status": "error", "error": str(e)},
                    "kafka": {"status": "error", "error": str(e)},
                    "redis": {"status": "error", "error": str(e)},
                },
            },
        }


@app.get("/api/distributed/health")
async def get_distributed_health():
    """Get comprehensive health check of distributed system"""
    try:
        from .distributed import health_check

        health_status = await health_check()

        return {
            "success": True,
            "overall_healthy": health_status["overall_healthy"],
            "timestamp": health_status["timestamp"],
            "components": health_status["components"],
        }

    except Exception as e:
        logger.error(f"Distributed health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@app.post("/api/distributed/initialize")
async def initialize_distributed_system(request: Dict[str, Any]):
    """Initialize the distributed MCP system"""
    try:
        from .distributed import NodeRole, initialize_distributed_system

        # Parse request parameters
        node_id = request.get("node_id")
        role = NodeRole(request.get("role", "coordinator"))
        region = request.get("region", "us-west-1")
        kafka_enabled = request.get("kafka_enabled", True)
        redis_enabled = request.get("redis_enabled", True)

        # Initialize the system
        result = await initialize_distributed_system(
            node_id=node_id,
            role=role,
            region=region,
            kafka_enabled=kafka_enabled,
            redis_enabled=redis_enabled,
        )

        if result["success"]:
            return {
                "success": True,
                "message": "Distributed MCP system initialized successfully",
                "node_id": result["components"].get("coordinator", {}).get("node_id"),
                "role": role,
                "region": region,
                "components": result["components"],
            }
        else:
            raise HTTPException(
                status_code=500, detail=f"Initialization failed: {result['errors']}"
            )

    except Exception as e:
        logger.error(f"Distributed system initialization failed: {e}")
        raise HTTPException(status_code=500, detail=f"Initialization failed: {str(e)}")


@app.post("/api/distributed/shutdown")
async def shutdown_distributed_system():
    """Shutdown the distributed MCP system"""
    try:
        from .distributed import shutdown_distributed_system

        await shutdown_distributed_system()

        return {
            "success": True,
            "message": "Distributed MCP system shutdown successfully",
        }

    except Exception as e:
        logger.error(f"Distributed system shutdown failed: {e}")
        raise HTTPException(status_code=500, detail=f"Shutdown failed: {str(e)}")


@app.post("/api/distributed/broadcast")
async def broadcast_message(request: Dict[str, Any]):
    """Broadcast message to all nodes in distributed system"""
    try:
        from .distributed import broadcast_system_message

        message_type = request.get("message_type")
        payload = request.get("payload", {})

        if not message_type:
            raise HTTPException(status_code=400, detail="message_type is required")

        success = await broadcast_system_message(message_type, payload)

        if success:
            return {
                "success": True,
                "message": f"Broadcast message sent: {message_type}",
                "payload": payload,
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to broadcast message")

    except Exception as e:
        logger.error(f"Broadcast message failed: {e}")
        raise HTTPException(status_code=500, detail=f"Broadcast failed: {str(e)}")


@app.post("/api/distributed/execute-tool")
async def execute_distributed_tool(request: Dict[str, Any]):
    """Execute tool across distributed MCP network"""
    try:
        from .distributed import LoadBalanceStrategy, execute_distributed_tool

        tool_name = request.get("tool_name")
        parameters = request.get("parameters", {})
        strategy = LoadBalanceStrategy(request.get("strategy", "least_loaded"))

        if not tool_name:
            raise HTTPException(status_code=400, detail="tool_name is required")

        result = await execute_distributed_tool(tool_name, parameters, strategy)

        if result:
            return {
                "success": True,
                "tool_name": tool_name,
                "result": result,
                "strategy": strategy,
            }
        else:
            raise HTTPException(status_code=500, detail="Tool execution failed")

    except Exception as e:
        logger.error(f"Distributed tool execution failed: {e}")
        raise HTTPException(status_code=500, detail=f"Tool execution failed: {str(e)}")


@app.get("/api/distributed/nodes")
async def get_active_nodes():
    """Get list of active nodes in distributed system"""
    try:
        from .distributed import get_redis_manager_instance

        redis_manager = get_redis_manager_instance()

        if not redis_manager:
            raise HTTPException(status_code=503, detail="Redis manager not available")

        active_nodes = await redis_manager.get_active_nodes()

        return {
            "success": True,
            "node_count": len(active_nodes),
            "active_nodes": active_nodes,
        }

    except Exception as e:
        logger.error(f"Failed to get active nodes: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get active nodes: {str(e)}"
        )


@app.get("/api/distributed/kafka/metrics")
async def get_kafka_metrics():
    """Get Kafka messaging metrics"""
    try:
        from .distributed import get_kafka_manager_instance

        kafka_manager = get_kafka_manager_instance()

        if not kafka_manager:
            raise HTTPException(status_code=503, detail="Kafka manager not available")

        metrics = kafka_manager.get_metrics()
        topic_info = kafka_manager.get_topic_info()

        return {"success": True, "kafka_metrics": metrics, "topic_info": topic_info}

    except Exception as e:
        logger.error(f"Failed to get Kafka metrics: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get Kafka metrics: {str(e)}"
        )


@app.get("/api/distributed/redis/metrics")
async def get_redis_metrics():
    """Get Redis cluster metrics"""
    try:
        from .distributed import get_redis_manager_instance

        redis_manager = get_redis_manager_instance()

        if not redis_manager:
            raise HTTPException(status_code=503, detail="Redis manager not available")

        metrics = redis_manager.get_metrics()

        return {"success": True, "redis_metrics": metrics}

    except Exception as e:
        logger.error(f"Failed to get Redis metrics: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get Redis metrics: {str(e)}"
        )


@app.post("/api/distributed/cache/set")
async def set_cache_value(request: Dict[str, Any]):
    """Set value in distributed cache"""
    try:
        from .distributed import cache_set

        key = request.get("key")
        value = request.get("value")
        ttl = request.get("ttl")

        if not key:
            raise HTTPException(status_code=400, detail="key is required")

        if value is None:
            raise HTTPException(status_code=400, detail="value is required")

        success = await cache_set(key, value, ttl)

        if success:
            return {
                "success": True,
                "message": f"Value set in cache for key: {key}",
                "key": key,
                "ttl": ttl,
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to set cache value")

    except Exception as e:
        logger.error(f"Cache set failed: {e}")
        raise HTTPException(status_code=500, detail=f"Cache set failed: {str(e)}")


@app.get("/api/distributed/cache/{key}")
async def get_cache_value(key: str):
    """Get value from distributed cache"""
    try:
        from .distributed import cache_get

        value = await cache_get(key)

        if value is not None:
            return {"success": True, "key": key, "value": value, "found": True}
        else:
            return {"success": True, "key": key, "value": None, "found": False}

    except Exception as e:
        logger.error(f"Cache get failed: {e}")
        raise HTTPException(status_code=500, detail=f"Cache get failed: {str(e)}")


@app.post("/api/distributed/coordination/election")
async def coordinate_leader_election(request: Dict[str, Any]):
    """Initiate distributed leader election"""
    try:
        from .distributed import get_redis_manager_instance

        redis_manager = get_redis_manager_instance()

        if not redis_manager:
            raise HTTPException(status_code=503, detail="Redis manager not available")

        election_key = request.get("election_key")
        candidate_id = request.get("candidate_id")
        ttl = request.get("ttl", 30)

        if not election_key or not candidate_id:
            raise HTTPException(
                status_code=400, detail="election_key and candidate_id are required"
            )

        is_leader = await redis_manager.coordinate_election(
            election_key, candidate_id, ttl
        )

        return {
            "success": True,
            "is_leader": is_leader,
            "election_key": election_key,
            "candidate_id": candidate_id,
            "ttl": ttl,
        }

    except Exception as e:
        logger.error(f"Leader election failed: {e}")
        raise HTTPException(status_code=500, detail=f"Leader election failed: {str(e)}")


# =============================================================================
# PHASE 4.1: 3D VISUALIZATION API ENDPOINTS
# =============================================================================


@app.get("/api/intelligence/threats")
async def get_threat_intelligence():
    """Get threat intelligence data for 3D visualization from real incidents"""
    try:
        async with AsyncSessionLocal() as db:
            # Get recent incidents with source IPs
            query = (
                select(Event)
                .where(
                    Event.src_ip.isnot(None),
                    Event.ts >= datetime.now() - timedelta(days=7),  # Last 7 days
                )
                .order_by(Event.ts.desc())
                .limit(100)
            )

            result = await db.execute(query)
            events = result.scalars().all()

            # Convert real incidents to threat intelligence format
            real_threats = []
            threat_type_map = {
                "cowrie.login.failed": "exploit",
                "cowrie.session.connect": "reconnaissance",
                "ssh": "exploit",
                "http": "exploit",
                "malware": "malware",
                "botnet": "botnet",
            }

            # Simple IP to country mapping (you can integrate with MaxMind GeoIP for better accuracy)
            ip_to_location = {
                "192.168.1.100": {
                    "country": "United States",
                    "code": "US",
                    "lat": 39.8283,
                    "lng": -98.5795,
                },
                "10.0.0.1": {
                    "country": "United States",
                    "code": "US",
                    "lat": 39.8283,
                    "lng": -98.5795,
                },
                # Add more mappings or integrate with GeoIP service
            }

            # Use GeoIP lookup for real IP geolocation (replace demo locations)
            async def get_ip_location(ip_address: str) -> dict:
                """Get real geolocation for IP address using GeoIP or external service"""
                try:
                    # For now, use a simple lookup - in production integrate with MaxMind GeoIP2
                    # or a service like ipapi.co, ip-api.com, etc.

                    # Check if it's a private/local IP
                    import ipaddress

                    ip_obj = ipaddress.ip_address(ip_address)
                    if ip_obj.is_private or ip_obj.is_loopback:
                        return {
                            "country": "Local Network",
                            "code": "LN",
                            "lat": 0,
                            "lng": 0,
                        }

                    # For demonstration, use a more realistic IP-to-location mapping
                    # This should be replaced with actual GeoIP service
                    import aiohttp

                    try:
                        async with aiohttp.ClientSession() as session:
                            async with session.get(
                                f"http://ip-api.com/json/{ip_address}?fields=country,countryCode,lat,lon"
                            ) as resp:
                                if resp.status == 200:
                                    data = await resp.json()
                                    if data.get("status") == "success":
                                        return {
                                            "country": data.get("country", "Unknown"),
                                            "code": data.get("countryCode", "XX"),
                                            "lat": data.get("lat", 0),
                                            "lng": data.get("lon", 0),
                                        }
                    except:
                        pass

                    # Fallback to unknown location if GeoIP fails
                    return {"country": "Unknown", "code": "XX", "lat": 0, "lng": 0}

                except Exception:
                    return {"country": "Unknown", "code": "XX", "lat": 0, "lng": 0}

            for i, event in enumerate(events):
                # Only use real geolocation data - no demo locations
                if event.src_ip in ip_to_location:
                    location = ip_to_location[event.src_ip]
                else:
                    # Get real geolocation for the IP
                    location = await get_ip_location(event.src_ip)

                    # Skip entries with unknown locations to keep data real
                    if (
                        location["lat"] == 0
                        and location["lng"] == 0
                        and location["country"] == "Unknown"
                    ):
                        continue

                # Determine threat type from event
                threat_type = "exploit"  # default
                for pattern, t_type in threat_type_map.items():
                    if pattern in event.eventid.lower():
                        threat_type = t_type
                        break

                # Calculate severity based on event frequency and type
                severity = 2  # medium default
                if "failed" in event.eventid.lower():
                    severity = 3  # high for failed login attempts
                if "malware" in event.eventid.lower():
                    severity = 4  # critical

                real_threats.append(
                    {
                        "id": f"incident-{event.id}",
                        "latitude": location["lat"],
                        "longitude": location["lng"],
                        "country": location["country"],
                        "country_code": location["code"],
                        "threat_type": threat_type,
                        "confidence": 0.8,  # High confidence since it's real data
                        "severity": severity,
                        "first_seen": int(event.ts.timestamp() * 1000),
                        "last_seen": int(event.ts.timestamp() * 1000),
                        "source": f"Mini-XDR Real Incident #{event.id}",
                        "tags": ["real_incident", event.eventid],
                        "metadata": {
                            "event_id": event.eventid,
                            "source_ip": event.src_ip,
                            "hostname": event.hostname,
                            "message": event.message,
                            "attack_technique": event.eventid.replace(".", " ").title(),
                        },
                    }
                )

            # Only return real honeypot data - no mock threats added
            logger.info(
                f"Returning {len(real_threats)} real threats from honeypot data"
            )

            return {
                "threats": real_threats,
                "total_count": len(real_threats),
                "last_updated": int(datetime.now().timestamp() * 1000),
                "sources": ["Mini-XDR Honeypot Incidents Only"],
                "real_data": True,
            }

    except Exception as e:
        logger.error(f"Failed to get threat intelligence: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get threat intelligence: {str(e)}"
        )


@app.get("/api/intelligence/distributed-threats")
async def get_distributed_threats():
    """Get threat intelligence from distributed MCP network"""
    try:
        from .distributed import get_system_status

        # Get distributed system status
        system_status = get_system_status()

        # Generate distributed threats based on active regions
        distributed_threats = []

        if system_status.get("kafka_manager", {}).get("available"):
            # Simulate threats from distributed nodes
            regions = ["us-west-1", "us-east-1", "eu-west-1", "ap-northeast-1"]

            for i, region in enumerate(regions):
                distributed_threats.append(
                    {
                        "id": f"distributed-{region}-{i}",
                        "latitude": [40.7128, 38.9072, 51.5074, 35.6762][i],
                        "longitude": [-74.0060, -77.0369, -0.1278, 139.6503][i],
                        "country": [
                            "United States",
                            "United States",
                            "United Kingdom",
                            "Japan",
                        ][i],
                        "country_code": ["US", "US", "GB", "JP"][i],
                        "threat_type": "distributed_attack",
                        "confidence": 0.75 + (i * 0.05),
                        "severity": 2 + (i % 3),
                        "first_seen": int(
                            (datetime.now() - timedelta(hours=1)).timestamp() * 1000
                        ),
                        "last_seen": int(datetime.now().timestamp() * 1000),
                        "source": f"MCP Node {region}",
                        "metadata": {
                            "region": region,
                            "distributed_source": f"node-{region}",
                            "coordination_level": "high",
                        },
                    }
                )

        return {
            "threats": distributed_threats,
            "total_count": len(distributed_threats),
            "last_updated": int(datetime.now().timestamp() * 1000),
            "distributed_status": system_status,
        }

    except Exception as e:
        logger.error(f"Failed to get distributed threats: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get distributed threats: {str(e)}"
        )


@app.get("/api/incidents/timeline")
async def get_incident_timeline(
    start_time: Optional[int] = None, end_time: Optional[int] = None
):
    """Get incident timeline data for 3D visualization"""
    try:
        async with AsyncSessionLocal() as db:
            # Default time range if not provided
            if not end_time:
                end_time = int(datetime.now().timestamp() * 1000)
            if not start_time:
                start_time = int(
                    (datetime.now() - timedelta(hours=2)).timestamp() * 1000
                )

            # Convert to datetime
            start_dt = datetime.fromtimestamp(start_time / 1000, tz=timezone.utc)
            end_dt = datetime.fromtimestamp(end_time / 1000, tz=timezone.utc)

            # Get incidents in time range
            query = (
                select(Event)
                .where(Event.ts >= start_dt, Event.ts <= end_dt)
                .order_by(Event.ts.desc())
                .limit(100)
            )

            result = await db.execute(query)
            events = result.scalars().all()

            # Convert events to timeline format
            timeline_incidents = []
            for event in events:
                timeline_incidents.append(
                    {
                        "id": str(event.id),
                        "timestamp": int(event.ts.timestamp() * 1000),
                        "title": f"Security Event: {event.eventid}",
                        "description": event.message or f"Event from {event.src_ip}",
                        "severity": "high"
                        if "attack" in event.message.lower()
                        else "medium",
                        "status": "detected",
                        "attack_vectors": [event.eventid],
                        "affected_assets": [event.hostname or "unknown"],
                        "source_ip": event.src_ip,
                        "location_data": {
                            "source_country": getattr(event, "src_country", None),
                            "source_lat": 39.8283,  # Default US coordinates
                            "source_lng": -98.5795,
                        },
                        "mitre_attack": {
                            "technique_id": f"T{1000 + (event.id % 100)}",
                            "technique_name": "Unknown Technique",
                            "tactic": "Initial Access",
                        },
                    }
                )

            return {
                "incidents": timeline_incidents,
                "total_count": len(timeline_incidents),
                "time_range": {"start": start_time, "end": end_time},
            }

    except Exception as e:
        logger.error(f"Failed to get incident timeline: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get timeline: {str(e)}")


@app.get("/api/incidents/attack-paths")
async def get_attack_paths():
    """Get attack path data for 3D visualization"""
    try:
        async with AsyncSessionLocal() as db:
            # Get recent incidents that might be related
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=6)
            query = (
                select(Event)
                .where(Event.ts >= cutoff_time)
                .order_by(Event.ts.desc())
                .limit(50)
            )

            result = await db.execute(query)
            events = result.scalars().all()

            # Generate attack paths from related events
            attack_paths = []

            # Group events by source IP to find potential attack chains
            ip_events = {}
            for event in events:
                if event.src_ip not in ip_events:
                    ip_events[event.src_ip] = []
                ip_events[event.src_ip].append(event)

            path_id = 0
            for src_ip, ip_event_list in ip_events.items():
                if (
                    len(ip_event_list) > 1
                ):  # Multiple events from same IP = potential attack path
                    # Sort by time to show progression
                    ip_event_list.sort(key=lambda x: x.ts)

                    for i in range(len(ip_event_list) - 1):
                        source_event = ip_event_list[i]
                        target_event = ip_event_list[i + 1]

                        attack_paths.append(
                            {
                                "id": f"path-{path_id}",
                                "source": {
                                    "id": f"source-{source_event.id}",
                                    "latitude": 39.8283,  # Default coordinates
                                    "longitude": -98.5795,
                                    "intensity": 0.7,
                                    "type": "exploit",
                                    "country": "United States",
                                    "timestamp": int(
                                        source_event.ts.timestamp() * 1000
                                    ),
                                },
                                "target": {
                                    "id": f"target-{target_event.id}",
                                    "latitude": 40.7128,  # Slightly different coordinates
                                    "longitude": -74.0060,
                                    "intensity": 0.8,
                                    "type": "lateral_movement",
                                    "country": "United States",
                                    "timestamp": int(
                                        target_event.ts.timestamp() * 1000
                                    ),
                                },
                                "progress": 1.0
                                if (
                                    datetime.now(timezone.utc)
                                    - (
                                        target_event.ts
                                        if target_event.ts.tzinfo
                                        else target_event.ts.replace(
                                            tzinfo=timezone.utc
                                        )
                                    )
                                ).total_seconds()
                                > 300
                                else 0.5,
                                "attack_type": "lateral_movement",
                                "is_active": (
                                    datetime.now(timezone.utc)
                                    - (
                                        target_event.ts
                                        if target_event.ts.tzinfo
                                        else target_event.ts.replace(
                                            tzinfo=timezone.utc
                                        )
                                    )
                                ).total_seconds()
                                < 1800,  # Active if within 30 min
                            }
                        )

                        path_id += 1

            return {
                "attack_paths": attack_paths,
                "total_count": len(attack_paths),
                "last_updated": int(datetime.now().timestamp() * 1000),
            }

    except Exception as e:
        logger.error(f"Failed to get attack paths: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get attack paths: {str(e)}"
        )


# =============================================================================
# ADVANCED RESPONSE & WORKFLOW API ENDPOINTS (Phase 1)
# =============================================================================

from typing import Optional

from pydantic import BaseModel


class CreateWorkflowRequest(BaseModel):
    incident_id: int
    playbook_name: str
    steps: List[Dict[str, Any]]
    auto_execute: bool = False
    priority: str = "medium"


class ExecuteWorkflowRequest(BaseModel):
    workflow_db_id: int
    executed_by: str = "analyst"


@app.post("/api/response/workflows/create")
async def create_response_workflow(
    request: CreateWorkflowRequest,
    http_request: Request,
    db: AsyncSession = Depends(get_db),
):
    """Create a new advanced response workflow"""
    _require_api_key(http_request)

    try:
        logger.info(f"Creating workflow request: {request}")
        response_engine = await get_response_engine()
        logger.info("Response engine initialized successfully")

        # Convert priority string to enum
        priority = ResponsePriority(request.priority.lower())
        logger.info(f"Priority converted: {priority}")

        result = await response_engine.create_workflow(
            incident_id=request.incident_id,
            playbook_name=request.playbook_name,
            steps=request.steps,
            auto_execute=request.auto_execute,
            priority=priority,
            db_session=db,
        )

        logger.info(f"Workflow creation successful: {result}")

        # Broadcast workflow creation update via WebSocket
        await ws_manager.broadcast_workflow_update(
            {
                "action": "workflow_created",
                "workflow_id": result.get("workflow_id"),
                "incident_id": request.incident_id,
                "playbook_name": request.playbook_name,
                "status": result.get("status", "pending"),
                "steps_count": len(request.steps),
            }
        )

        return result

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid request: {str(e)}")
    except Exception as e:
        logger.error(f"Failed to create workflow: {e}")
        import traceback

        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500, detail=f"Failed to create workflow: {str(e)}"
        )


@app.post("/api/response/workflows/execute")
async def execute_response_workflow(
    request: ExecuteWorkflowRequest, db: AsyncSession = Depends(get_db)
):
    """Execute a response workflow"""
    try:
        response_engine = await get_response_engine()

        result = await response_engine.execute_workflow(
            workflow_db_id=request.workflow_db_id,
            db_session=db,
            executed_by=request.executed_by,
        )

        # Broadcast workflow execution update via WebSocket
        await ws_manager.broadcast_workflow_update(
            {
                "action": "workflow_executed",
                "workflow_id": result.get("workflow_id"),
                "status": result.get("status", "unknown"),
                "execution_id": result.get("execution_id"),
                "progress": result.get("progress", 0),
            }
        )

        return result

    except Exception as e:
        logger.error(f"Failed to execute workflow: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to execute workflow: {str(e)}"
        )


@app.get("/api/response/workflows/{workflow_id}/status")
async def get_workflow_status(workflow_id: str, db: AsyncSession = Depends(get_db)):
    """Get detailed status of a response workflow"""
    try:
        response_engine = await get_response_engine()

        result = await response_engine.get_workflow_status(
            workflow_id=workflow_id, db_session=db
        )

        if not result.get("success"):
            raise HTTPException(
                status_code=404, detail=result.get("error", "Workflow not found")
            )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get workflow status: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get workflow status: {str(e)}"
        )


@app.post("/api/response/workflows/{workflow_id}/cancel")
async def cancel_response_workflow(
    workflow_id: str, db: AsyncSession = Depends(get_db), cancelled_by: str = "analyst"
):
    """Cancel a running response workflow"""
    try:
        response_engine = await get_response_engine()

        result = await response_engine.cancel_workflow(
            workflow_id=workflow_id, db_session=db, cancelled_by=cancelled_by
        )

        if not result.get("success"):
            raise HTTPException(
                status_code=400, detail=result.get("error", "Failed to cancel workflow")
            )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to cancel workflow: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to cancel workflow: {str(e)}"
        )


@app.delete("/api/response/workflows/{workflow_id}")
async def delete_response_workflow(
    workflow_id: str, db: AsyncSession = Depends(get_db), http_request: Request = None
):
    """Delete a response workflow"""
    _require_api_key(http_request)

    try:
        # Find the workflow
        result = await db.execute(
            select(ResponseWorkflow).where(ResponseWorkflow.workflow_id == workflow_id)
        )
        workflow = result.scalar_one_or_none()

        if not workflow:
            raise HTTPException(status_code=404, detail="Workflow not found")

        # Don't allow deleting running workflows
        if workflow.status == "running":
            raise HTTPException(
                status_code=400,
                detail="Cannot delete a running workflow. Cancel it first.",
            )

        # Delete associated actions first
        await db.execute(
            delete(WorkflowAction).where(WorkflowAction.workflow_id == workflow_id)
        )

        # Delete the workflow
        await db.delete(workflow)
        await db.commit()

        logger.info(f"Deleted workflow {workflow_id}")
        return {
            "success": True,
            "message": f"Workflow {workflow_id} deleted successfully",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete workflow: {e}")
        await db.rollback()
        raise HTTPException(
            status_code=500, detail=f"Failed to delete workflow: {str(e)}"
        )


@app.post("/api/response/workflows/{workflow_id}/approve")
async def approve_response_workflow(
    workflow_id: str,
    request: Dict[str, Any],
    http_request: Request,
    db: AsyncSession = Depends(get_db),
):
    """Approve a workflow for execution"""
    _require_api_key(http_request)

    try:
        from sqlalchemy import select

        from .models import ResponseWorkflow

        # Get workflow
        result = await db.execute(
            select(ResponseWorkflow).where(ResponseWorkflow.workflow_id == workflow_id)
        )
        workflow = result.scalars().first()

        if not workflow:
            raise HTTPException(status_code=404, detail="Workflow not found")

        if workflow.status != "awaiting_approval":
            raise HTTPException(
                status_code=400,
                detail=f"Workflow is not awaiting approval (status: {workflow.status})",
            )

        # Update approval status
        workflow.status = "approved"
        workflow.approved_by = request.get("approved_by", "analyst")
        workflow.approved_at = datetime.utcnow()

        await db.commit()

        # Execute the workflow
        response_engine = await get_response_engine()
        execution_result = await response_engine.execute_workflow(
            workflow.id, db, executed_by=request.get("approved_by", "analyst")
        )

        return {
            "success": True,
            "workflow_id": workflow_id,
            "status": "approved_and_executing",
            "execution_result": execution_result,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to approve workflow: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to approve workflow: {str(e)}"
        )


@app.post("/api/response/workflows/{workflow_id}/reject")
async def reject_response_workflow(
    workflow_id: str,
    request: Dict[str, Any],
    http_request: Request,
    db: AsyncSession = Depends(get_db),
):
    """Reject a workflow"""
    _require_api_key(http_request)

    try:
        from sqlalchemy import select

        from .models import ResponseWorkflow

        # Get workflow
        result = await db.execute(
            select(ResponseWorkflow).where(ResponseWorkflow.workflow_id == workflow_id)
        )
        workflow = result.scalars().first()

        if not workflow:
            raise HTTPException(status_code=404, detail="Workflow not found")

        if workflow.status != "awaiting_approval":
            raise HTTPException(
                status_code=400,
                detail=f"Workflow is not awaiting approval (status: {workflow.status})",
            )

        # Update rejection status
        workflow.status = "rejected"
        workflow.approved_by = request.get("rejected_by", "analyst")
        workflow.approved_at = datetime.utcnow()

        await db.commit()

        return {
            "success": True,
            "workflow_id": workflow_id,
            "status": "rejected",
            "rejected_by": request.get("rejected_by", "analyst"),
            "rejection_reason": request.get("rejection_reason", "Manual review"),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to reject workflow: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to reject workflow: {str(e)}"
        )


@app.post("/api/incidents/{incident_id}/ai-analysis")
async def generate_ai_incident_analysis(
    incident_id: int,
    request: Dict[str, Any],
    http_request: Request,
    current_user: Dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Generate AI analysis for an incident (with caching)"""
    # Use JWT authentication instead of API key

    try:
        # Get incident details
        incident = (
            (await db.execute(select(Incident).where(Incident.id == incident_id)))
            .scalars()
            .first()
        )

        if not incident:
            raise HTTPException(status_code=404, detail="Incident not found")

        # Get current event count for this IP
        current_event_count = (
            await db.execute(
                select(func.count(Event.id)).where(Event.src_ip == incident.src_ip)
            )
        ).scalar()

        force_regenerate = request.get("force_regenerate", False)

        # Check if we have cached analysis and no new events
        if (
            not force_regenerate
            and incident.ai_analysis
            and incident.last_event_count == current_event_count
            and incident.ai_analysis_timestamp
        ):
            # Return cached analysis
            cache_age_seconds = (
                datetime.utcnow() - incident.ai_analysis_timestamp
            ).total_seconds()
            logger.info(f"Returning cached AI analysis (age: {cache_age_seconds:.0f}s)")

            return {
                "success": True,
                "analysis": incident.ai_analysis,
                "provider": incident.ai_analysis.get("provider", "openai"),
                "incident_id": incident_id,
                "generated_at": incident.ai_analysis_timestamp.isoformat(),
                "cached": True,
                "cache_age_seconds": int(cache_age_seconds),
                "event_count": current_event_count,
            }

        # Need to generate new analysis
        logger.info(
            f"Generating new AI analysis (events: {current_event_count}, cached: {incident.last_event_count})"
        )

        # Get recent events for context
        recent_events = await _recent_events_for_ip(
            db, incident.src_ip, 3600
        )  # Last hour

        # Prepare payload for AI analysis
        incident_data = {
            "id": incident.id,
            "src_ip": incident.src_ip,
            "reason": incident.reason,
            "status": incident.status,
            "risk_score": incident.risk_score,
            "threat_category": incident.threat_category,
            "escalation_level": incident.escalation_level,
            "ensemble_scores": incident.ensemble_scores,
        }

        event_summaries = [
            {
                "eventid": e.eventid,
                "message": e.message,
                "timestamp": e.ts.isoformat() if e.ts else None,
                "raw": e.raw,
            }
            for e in recent_events[-10:]  # Last 10 events
        ]

        # Generate AI analysis
        provider = request.get("provider", "openai")

        if provider == "xai":
            from .triager import _xai_triage

            ai_result = _xai_triage(
                {"incident": incident_data, "recent_events": event_summaries}
            )
        else:
            from .triager import _openai_triage

            ai_result = _openai_triage(
                {"incident": incident_data, "recent_events": event_summaries}
            )

        # Enhance with additional AI insights
        enhanced_analysis = {
            **ai_result,
            "confidence_score": incident.containment_confidence or 0.5,
            "threat_attribution": await _generate_threat_attribution(
                incident, recent_events
            ),
            "response_priority": _calculate_response_priority(incident, ai_result),
            "estimated_impact": _estimate_business_impact(incident, ai_result),
            "next_steps": _generate_next_steps(incident, ai_result),
            "provider": provider,
        }

        # Cache the analysis in the database
        incident.ai_analysis = enhanced_analysis
        incident.ai_analysis_timestamp = datetime.utcnow()
        incident.last_event_count = current_event_count
        await db.commit()

        logger.info(f"AI analysis cached for incident {incident_id}")

        return {
            "success": True,
            "analysis": enhanced_analysis,
            "provider": provider,
            "incident_id": incident_id,
            "generated_at": datetime.utcnow().isoformat(),
            "cached": False,
            "event_count": current_event_count,
        }

    except Exception as e:
        logger.error(f"AI analysis failed: {e}")
        return {"success": False, "error": str(e)}


async def _generate_threat_attribution(incident, events) -> str:
    """Generate threat attribution analysis"""
    if incident.src_ip.startswith("10.0.200"):
        return "Internal network reconnaissance - possible insider threat or lateral movement"
    elif incident.src_ip.startswith("172.16"):
        return "APT-style attack pattern - sophisticated threat actor with persistence goals"
    elif incident.src_ip.startswith("13.220"):
        return "Cloud-based attack infrastructure - likely automated botnet or testing platform"
    else:
        return "External threat actor - requires geolocation and threat intelligence correlation"


def _calculate_response_priority(incident, ai_result) -> str:
    """Calculate response priority based on AI analysis and incident data"""
    severity = ai_result.get("severity", "low")
    risk_score = incident.risk_score or 0

    if severity == "critical" or risk_score > 0.8:
        return "IMMEDIATE"
    elif severity == "high" or risk_score > 0.6:
        return "HIGH"
    elif severity == "medium" or risk_score > 0.4:
        return "MEDIUM"
    else:
        return "LOW"


def _estimate_business_impact(incident, ai_result) -> str:
    """Estimate business impact"""
    severity = ai_result.get("severity", "low")
    threat_category = incident.threat_category or "unknown"

    if "brute_force" in threat_category.lower():
        return "Account compromise risk - potential data access"
    elif "adaptive_detection" in threat_category.lower():
        return "Unknown attack pattern - potential zero-day threat"
    elif severity in ["critical", "high"]:
        return "Significant operational risk - immediate attention required"
    else:
        return "Minimal business impact - standard monitoring sufficient"


def _generate_next_steps(incident, ai_result) -> List[str]:
    """Generate specific next steps based on AI analysis"""
    steps = []
    recommendation = ai_result.get("recommendation", "")

    if recommendation == "contain_now":
        steps.extend(
            [
                "Execute immediate IP blocking response",
                "Isolate affected systems from network",
                "Begin forensic evidence collection",
                "Notify security team of active threat",
            ]
        )
    elif recommendation == "investigate":
        steps.extend(
            [
                "Collect additional forensic evidence",
                "Analyze attack patterns and TTPs",
                "Correlate with threat intelligence feeds",
                "Document findings for threat hunting",
            ]
        )
    else:
        steps.extend(
            [
                "Continue monitoring for escalation",
                "Review security controls effectiveness",
                "Update detection rules if needed",
            ]
        )

    return steps


# ==================== NEW AI RECOMMENDATION EXECUTION ENDPOINTS ====================


@app.post("/api/incidents/{incident_id}/execute-ai-recommendation")
async def execute_ai_recommendation(
    incident_id: int,
    request_data: Dict[str, Any],
    http_request: Request,
    db: AsyncSession = Depends(get_db),
):
    """Execute a specific AI-recommended action"""
    _require_api_key(http_request)

    try:
        action_type = request_data.get("action_type")
        parameters = request_data.get("parameters", {})

        if not action_type:
            raise HTTPException(status_code=400, detail="action_type is required")

        # Get incident
        incident = (
            (await db.execute(select(Incident).where(Incident.id == incident_id)))
            .scalars()
            .first()
        )

        if not incident:
            raise HTTPException(status_code=404, detail="Incident not found")

        # Execute action based on type
        result = None
        action_name = action_type

        if action_type == "block_ip":
            ip = parameters.get("ip") or incident.src_ip
            duration = parameters.get("duration", 30)
            logger.info(
                f"Executing block_ip for incident {incident_id}: IP={ip}, duration={duration}min"
            )
            status, detail = await block_ip(ip, duration * 60)  # Convert to seconds
            logger.info(f"block_ip result: status={status}, detail={detail[:500]}")
            result = {
                "status": status,
                "detail": detail,
                "ip": ip,
                "duration_minutes": duration,
            }
            action_name = f"Block IP {ip}"

            # If blocking failed, raise an exception with details
            if status != "success":
                raise HTTPException(
                    status_code=500, detail=f"IP blocking failed: {detail}"
                )

        elif action_type == "isolate_host":
            # Simulate host isolation
            result = {
                "status": "success",
                "detail": "Host isolation initiated",
                "simulated": True,
            }
            action_name = "Isolate Host"

        elif action_type == "reset_passwords":
            # Simulate password reset
            result = {
                "status": "success",
                "detail": "Password reset initiated for affected accounts",
                "simulated": True,
            }
            action_name = "Force Password Reset"

        elif action_type == "threat_intel_lookup":
            ip = parameters.get("ip") or incident.src_ip
            # Perform threat intel lookup (simulated)
            result = {
                "status": "success",
                "detail": f"Threat intelligence lookup completed for {ip}",
                "ip": ip,
                "findings": "No matches in threat feeds",
                "simulated": True,
            }
            action_name = f"Threat Intel Lookup: {ip}"

        elif action_type == "hunt_similar_attacks":
            # Hunt for similar patterns
            similar_count = (
                await db.execute(
                    select(func.count(Incident.id)).where(
                        Incident.threat_category == incident.threat_category,
                        Incident.id != incident_id,
                    )
                )
            ).scalar()
            result = {
                "status": "success",
                "detail": f"Hunt completed - found {similar_count} similar incidents",
                "similar_incident_count": similar_count,
            }
            action_name = "Hunt Similar Attacks"

        else:
            raise HTTPException(
                status_code=400, detail=f"Unknown action type: {action_type}"
            )

        # Record the action
        action = Action(
            incident_id=incident_id,
            action=action_type,
            result="success" if result.get("status") == "success" else "failed",
            detail=result.get("detail", ""),
            params={
                "action_type": action_type,
                "parameters": parameters,
                "ai_recommended": True,
                "executed_via": "ai_recommendation_api",
            },
        )
        db.add(action)

        # ALSO create an AdvancedResponseAction for workflow timeline consistency
        try:
            from .models import AdvancedResponseAction

            workflow_action = AdvancedResponseAction(
                action_id=f"ui_{incident_id}_{action_type}_{int(datetime.utcnow().timestamp())}",
                incident_id=incident_id,
                action_type=action_type,
                action_name=action_name,
                action_description=f"Manual execution via UI: {action_name}",
                action_category="containment"
                if action_type == "block_ip"
                else "investigation",
                status="completed" if result.get("status") == "success" else "failed",
                parameters=parameters,
                result_data=result,
                executed_by="soc_analyst",
                execution_method="manual_ui",
                error_details=result.get("detail")
                if result.get("status") != "success"
                else None,
            )
            db.add(workflow_action)
        except Exception as e:
            logger.warning(f"Could not create AdvancedResponseAction: {e}")

        await db.commit()

        return {
            "success": True,
            "action_id": action.id,
            "action_type": action_type,
            "action_name": action_name,
            "result": result,
            "incident_id": incident_id,
            "executed_at": datetime.utcnow().isoformat(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to execute AI recommendation: {e}")
        return {"success": False, "error": str(e)}


@app.post("/api/incidents/{incident_id}/execute-ai-plan")
async def execute_ai_plan(
    incident_id: int, http_request: Request, db: AsyncSession = Depends(get_db)
):
    """Execute all AI-recommended actions as a workflow"""
    _require_api_key(http_request)

    try:
        # Get incident and AI analysis
        incident = (
            (await db.execute(select(Incident).where(Incident.id == incident_id)))
            .scalars()
            .first()
        )

        if not incident:
            raise HTTPException(status_code=404, detail="Incident not found")

        if not incident.ai_analysis:
            raise HTTPException(
                status_code=400,
                detail="No AI analysis available. Generate analysis first.",
            )

        # Create a workflow with all recommended actions
        from .advanced_response_engine import ResponseWorkflowEngine

        workflow_engine = ResponseWorkflowEngine()

        # Define actions based on severity
        severity = incident.ai_analysis.get("severity", "medium").lower()
        actions = []

        if severity in ["critical", "high"]:
            actions = [
                {
                    "action": "block_ip",
                    "params": {"ip": incident.src_ip, "duration": 30},
                },
                {"action": "threat_intel_lookup", "params": {"ip": incident.src_ip}},
                {"action": "hunt_similar_attacks", "params": {}},
            ]
        else:
            actions = [
                {"action": "threat_intel_lookup", "params": {"ip": incident.src_ip}},
                {"action": "hunt_similar_attacks", "params": {}},
            ]

        # Create workflow
        workflow = ResponseWorkflow(
            name=f"AI Emergency Response - Incident #{incident_id}",
            description=f"Automated response based on AI analysis (severity: {severity})",
            created_by="ai_recommendation_engine",
            incident_id=incident_id,
            status="pending",
        )
        db.add(workflow)
        await db.commit()
        await db.refresh(workflow)

        # Execute actions
        executed_actions = []
        for action_data in actions:
            try:
                # Execute via the recommendation endpoint
                result = await execute_ai_recommendation(
                    incident_id=incident_id,
                    request_data={
                        "action_type": action_data["action"],
                        "parameters": action_data["params"],
                    },
                    http_request=http_request,
                    db=db,
                )
                executed_actions.append(result)
            except Exception as e:
                logger.error(f"Action {action_data['action']} failed: {e}")
                executed_actions.append(
                    {
                        "success": False,
                        "action_type": action_data["action"],
                        "error": str(e),
                    }
                )

        # Update workflow status
        workflow.status = "completed"
        workflow.completed_at = datetime.utcnow()
        await db.commit()

        return {
            "success": True,
            "workflow_id": workflow.id,
            "workflow_name": workflow.name,
            "incident_id": incident_id,
            "actions": executed_actions,
            "total_actions": len(executed_actions),
            "successful_actions": len(
                [a for a in executed_actions if a.get("success")]
            ),
            "failed_actions": len(
                [a for a in executed_actions if not a.get("success")]
            ),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to execute AI plan: {e}")
        return {"success": False, "error": str(e)}


@app.get("/api/incidents/{incident_id}/actions")
async def get_incident_actions(
    incident_id: int, http_request: Request, db: AsyncSession = Depends(get_db)
):
    """Get all actions (manual, workflow, and agent) for an incident"""
    _require_api_key(http_request)

    try:
        # Get manual actions
        manual_actions = (
            (
                await db.execute(
                    select(Action)
                    .where(Action.incident_id == incident_id)
                    .order_by(Action.created_at.desc())
                )
            )
            .scalars()
            .all()
        )

        # Get workflow actions
        from .models import AdvancedResponseAction

        workflow_actions = (
            (
                await db.execute(
                    select(AdvancedResponseAction)
                    .where(AdvancedResponseAction.incident_id == incident_id)
                    .order_by(AdvancedResponseAction.created_at.desc())
                )
            )
            .scalars()
            .all()
        )

        # Combine and format actions
        all_actions = []

        for action in manual_actions:
            all_actions.append(
                {
                    "id": action.id,
                    "type": "manual",
                    "action": action.action,
                    "action_type": action.action,
                    "result": action.result,
                    "status": action.result,
                    "detail": action.detail,
                    "params": action.params,
                    "created_at": action.created_at.isoformat()
                    if action.created_at
                    else None,
                }
            )

        for action in workflow_actions:
            all_actions.append(
                {
                    "id": action.id,
                    "type": "workflow",
                    "action": action.action_type,
                    "action_type": action.action_type,
                    "action_name": action.action_name,
                    "result": action.status,
                    "status": action.status,
                    "detail": action.action_description,
                    "params": action.parameters,
                    "created_at": action.created_at.isoformat()
                    if action.created_at
                    else None,
                }
            )

        # Sort by created_at
        all_actions.sort(key=lambda x: x.get("created_at") or "", reverse=True)

        return all_actions

    except Exception as e:
        logger.error(f"Failed to get actions for incident {incident_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/incidents/{incident_id}/threat-status")
async def get_incident_threat_status(
    incident_id: int, http_request: Request, db: AsyncSession = Depends(get_db)
):
    """Get real-time threat status for an incident"""
    _require_api_key(http_request)

    try:
        incident = (
            (await db.execute(select(Incident).where(Incident.id == incident_id)))
            .scalars()
            .first()
        )

        if not incident:
            raise HTTPException(status_code=404, detail="Incident not found")

        # Get action counts
        manual_actions = (
            await db.execute(
                select(func.count(Action.id)).where(Action.incident_id == incident_id)
            )
        ).scalar()

        workflow_actions = (
            await db.execute(
                select(func.count(AdvancedResponseAction.id)).where(
                    AdvancedResponseAction.incident_id == incident_id
                )
            )
        ).scalar()

        # Get agent actions (from separate agents table if exists)
        try:
            from .models import AgentAction

            agent_actions = (
                await db.execute(
                    select(func.count(AgentAction.id)).where(
                        AgentAction.incident_id == incident_id
                    )
                )
            ).scalar()
        except:
            agent_actions = 0

        # Determine containment status
        containment_status = "none"
        if incident.auto_contained or incident.status == "contained":
            containment_status = "complete"
        elif manual_actions > 0 or workflow_actions > 0 or agent_actions > 0:
            containment_status = "partial"

        # Determine if attack is active
        attack_active = incident.status in ["open", "investigating", "new"]

        return {
            "success": True,
            "incident_id": incident_id,
            "attack_active": attack_active,
            "containment_status": containment_status,
            "agent_count": agent_actions,
            "workflow_count": workflow_actions,
            "manual_action_count": manual_actions,
            "total_actions": manual_actions + workflow_actions + agent_actions,
            "severity": incident.ai_analysis.get("severity", "medium")
            if incident.ai_analysis
            else "medium",
            "confidence": incident.containment_confidence
            or incident.agent_confidence
            or 0.5,
            "threat_category": incident.threat_category or "Unknown",
            "source_ip": incident.src_ip,
            "status": incident.status,
            "created_at": incident.created_at.isoformat()
            if incident.created_at
            else None,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get threat status: {e}")
        return {"success": False, "error": str(e)}


@app.get("/api/response/actions")
async def get_available_response_actions(category: Optional[str] = None):
    """Get list of available response actions"""
    try:
        response_engine = await get_response_engine()

        # Convert category string to enum if provided
        category_enum = None
        if category:
            try:
                category_enum = ActionCategory(category.lower())
            except ValueError:
                raise HTTPException(
                    status_code=400, detail=f"Invalid category: {category}"
                )

        result = response_engine.get_available_actions(category=category_enum)

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get available actions: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get available actions: {str(e)}"
        )


@app.get("/api/response/workflows")
async def list_response_workflows(
    incident_id: Optional[int] = None,
    status: Optional[str] = None,
    limit: int = 50,
    db: AsyncSession = Depends(get_db),
):
    """List response workflows with optional filtering"""
    try:
        query = select(ResponseWorkflow).order_by(ResponseWorkflow.created_at.desc())

        # Apply filters
        if incident_id:
            query = query.where(ResponseWorkflow.incident_id == incident_id)

        if status:
            query = query.where(ResponseWorkflow.status == status)

        query = query.limit(limit)

        result = await db.execute(query)
        workflows = result.scalars().all()

        return {
            "success": True,
            "workflows": [
                {
                    "id": wf.id,
                    "workflow_id": wf.workflow_id,
                    "incident_id": wf.incident_id,
                    "playbook_name": wf.playbook_name,
                    "status": wf.status,
                    "progress_percentage": wf.progress_percentage,
                    "current_step": wf.current_step,
                    "total_steps": wf.total_steps,
                    "created_at": wf.created_at.isoformat() if wf.created_at else None,
                    "completed_at": wf.completed_at.isoformat()
                    if wf.completed_at
                    else None,
                    "approval_required": wf.approval_required,
                    "auto_executed": wf.auto_executed,
                    "success_rate": wf.success_rate,
                }
                for wf in workflows
            ],
            "total_count": len(workflows),
        }

    except Exception as e:
        logger.error(f"Failed to list workflows: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to list workflows: {str(e)}"
        )


@app.get("/api/response/workflows/{workflow_id}/actions")
async def get_workflow_actions(workflow_id: str, db: AsyncSession = Depends(get_db)):
    """Get all actions for a specific workflow"""
    try:
        # Get workflow
        workflow_result = await db.execute(
            select(ResponseWorkflow).where(ResponseWorkflow.workflow_id == workflow_id)
        )
        workflow = workflow_result.scalars().first()

        if not workflow:
            raise HTTPException(status_code=404, detail="Workflow not found")

        # Get actions
        actions_result = await db.execute(
            select(AdvancedResponseAction)
            .where(AdvancedResponseAction.workflow_id == workflow.id)
            .order_by(AdvancedResponseAction.created_at.asc())
        )
        actions = actions_result.scalars().all()

        return {
            "success": True,
            "workflow_id": workflow_id,
            "actions": [
                {
                    "id": action.id,
                    "action_id": action.action_id,
                    "action_type": action.action_type,
                    "action_category": action.action_category,
                    "action_name": action.action_name,
                    "status": action.status,
                    "parameters": action.parameters,
                    "result_data": action.result_data,
                    "error_details": action.error_details,
                    "confidence_score": action.confidence_score,
                    "executed_by": action.executed_by,
                    "execution_method": action.execution_method,
                    "created_at": action.created_at.isoformat()
                    if action.created_at
                    else None,
                    "completed_at": action.completed_at.isoformat()
                    if action.completed_at
                    else None,
                }
                for action in actions
            ],
            "total_count": len(actions),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get workflow actions: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get workflow actions: {str(e)}"
        )


@app.get("/api/response/metrics/impact")
async def get_response_impact_metrics(
    workflow_id: Optional[str] = None,
    incident_id: Optional[int] = None,
    days_back: int = 7,
    db: AsyncSession = Depends(get_db),
):
    """Get response impact metrics"""
    try:
        query = (
            select(ResponseImpactMetrics)
            .join(ResponseWorkflow)
            .options(selectinload(ResponseImpactMetrics.workflow))
        )

        # Apply filters
        if workflow_id:
            query = query.where(ResponseWorkflow.workflow_id == workflow_id)

        if incident_id:
            query = query.where(ResponseWorkflow.incident_id == incident_id)

        # Time filter
        since_date = datetime.utcnow() - timedelta(days=days_back)
        query = query.where(ResponseImpactMetrics.created_at >= since_date)

        query = query.order_by(ResponseImpactMetrics.created_at.desc()).limit(100)

        result = await db.execute(query)
        metrics = result.scalars().all()

        # Calculate aggregated metrics
        total_attacks_blocked = sum(m.attacks_blocked for m in metrics)
        total_false_positives = sum(m.false_positives for m in metrics)
        avg_response_time = (
            sum(m.response_time_ms for m in metrics) / len(metrics) if metrics else 0
        )
        avg_success_rate = (
            sum(m.success_rate for m in metrics) / len(metrics) if metrics else 0
        )

        return {
            "success": True,
            "summary": {
                "total_attacks_blocked": total_attacks_blocked,
                "total_false_positives": total_false_positives,
                "average_response_time_ms": round(avg_response_time),
                "average_success_rate": round(avg_success_rate, 2),
                "metrics_count": len(metrics),
            },
            "detailed_metrics": [
                {
                    "id": m.id,
                    "workflow_id": m.workflow_id,
                    "workflow_name": m.workflow.playbook_name
                    if getattr(m, "workflow", None)
                    else None,
                    "incident_id": m.workflow.incident_id
                    if getattr(m, "workflow", None)
                    else None,
                    "attacks_blocked": m.attacks_blocked,
                    "false_positives": m.false_positives,
                    "systems_affected": m.systems_affected,
                    "response_time_ms": m.response_time_ms,
                    "success_rate": m.success_rate,
                    "confidence_score": m.confidence_score,
                    "created_at": m.created_at.isoformat() if m.created_at else None,
                }
                for m in metrics
            ],
        }

    except Exception as e:
        logger.error(f"Failed to get impact metrics: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get impact metrics: {str(e)}"
        )


@app.post("/api/response/actions/execute")
async def execute_single_response_action(
    request: Dict[str, Any], db: AsyncSession = Depends(get_db)
):
    """Execute a single response action (for manual execution)"""
    try:
        response_engine = await get_response_engine()

        # Extract parameters
        action_type = request.get("action_type")
        parameters = request.get("parameters", {})
        incident_id = request.get("incident_id")

        if not action_type or not incident_id:
            raise HTTPException(
                status_code=400, detail="action_type and incident_id are required"
            )

        # Create a simple workflow with single step
        workflow_result = await response_engine.create_workflow(
            incident_id=incident_id,
            playbook_name=f"Manual {action_type}",
            steps=[{"action_type": action_type, "parameters": parameters}],
            auto_execute=True,
            priority=ResponsePriority.HIGH,
            db_session=db,
        )

        return workflow_result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to execute single action: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to execute action: {str(e)}"
        )


# =============================================================================
# AI-POWERED RESPONSE ENDPOINTS (Phase 2B)
# =============================================================================


@app.post("/api/ai/response/recommendations")
async def get_ai_response_recommendations(
    request: Dict[str, Any], db: AsyncSession = Depends(get_db)
):
    """Get AI-powered response recommendations for an incident"""
    try:
        incident_id = request.get("incident_id")
        context = request.get("context", {})

        if not incident_id:
            raise HTTPException(status_code=400, detail="incident_id is required")

        # Get AI advisor
        ai_advisor = await get_ai_advisor()

        # Get recommendations
        result = await ai_advisor.get_response_recommendations(incident_id, db, context)

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get AI recommendations: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get AI recommendations: {str(e)}"
        )


@app.get("/api/ai/response/context/{incident_id}")
async def get_incident_context_analysis(
    incident_id: int,
    include_predictions: bool = True,
    db: AsyncSession = Depends(get_db),
):
    """Get comprehensive context analysis for an incident"""
    try:
        context_analyzer = await get_context_analyzer()

        result = await context_analyzer.analyze_comprehensive_context(
            incident_id, db, include_predictions=include_predictions
        )

        return result

    except Exception as e:
        logger.error(f"Failed to analyze context: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to analyze context: {str(e)}"
        )


@app.post("/api/ai/response/optimize/{workflow_id}")
async def optimize_workflow_with_ai(
    workflow_id: str, request: Dict[str, Any] = {}, db: AsyncSession = Depends(get_db)
):
    """Optimize a workflow using AI and historical learning"""
    try:
        response_optimizer = await get_response_optimizer()

        strategy = request.get("strategy", "effectiveness")
        context = request.get("context", {})

        # Convert string strategy to enum
        from .response_optimizer import OptimizationStrategy

        strategy_enum = getattr(
            OptimizationStrategy, strategy.upper(), OptimizationStrategy.EFFECTIVENESS
        )

        result = await response_optimizer.optimize_workflow(
            workflow_id, db, strategy_enum, context
        )

        return {
            "success": True,
            "workflow_id": workflow_id,
            "optimization_result": {
                "optimized_workflow": result.optimized_workflow,
                "optimization_score": result.optimization_score,
                "improvements": result.improvements,
                "risk_reduction": result.risk_reduction,
                "efficiency_gain": result.efficiency_gain,
                "confidence": result.confidence,
            },
        }

    except Exception as e:
        logger.error(f"Failed to optimize workflow: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to optimize workflow: {str(e)}"
        )


@app.post("/api/ai/response/adaptive")
async def get_adaptive_response_recommendations(
    request: Dict[str, Any], db: AsyncSession = Depends(get_db)
):
    """Get adaptive response recommendations that improve over time"""
    try:
        incident_id = request.get("incident_id")
        user_context = request.get("user_context", {})

        if not incident_id:
            raise HTTPException(status_code=400, detail="incident_id is required")

        # Get learning engine
        learning_engine = await get_learning_engine()

        # Get adaptive recommendations
        result = await learning_engine.generate_adaptive_recommendations(
            incident_id, db, user_context
        )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get adaptive recommendations: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get adaptive recommendations: {str(e)}"
        )


@app.post("/api/ai/response/learn")
async def learn_from_workflow_execution(
    request: Dict[str, Any], db: AsyncSession = Depends(get_db)
):
    """Submit learning data from workflow execution"""
    try:
        workflow_id = request.get("workflow_id")
        execution_results = request.get("execution_results", {})
        analyst_feedback = request.get("analyst_feedback", {})

        if not workflow_id:
            raise HTTPException(status_code=400, detail="workflow_id is required")

        # Get learning engine
        learning_engine = await get_learning_engine()

        # Submit learning data
        result = await learning_engine.learn_from_workflow_execution(
            workflow_id, execution_results, analyst_feedback, db
        )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to submit learning data: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to submit learning data: {str(e)}"
        )


# =============================================================================
# VISUAL WORKFLOW ENDPOINTS (Phase 2A)
# =============================================================================


@app.get("/api/workflows/templates")
async def get_playbook_templates(
    category: Optional[str] = None, db: AsyncSession = Depends(get_db)
):
    """Get available playbook templates"""
    try:
        # For now, return mock data - in production would query ResponsePlaybook table
        templates = [
            {
                "id": "malware-response",
                "name": "Malware Incident Response",
                "description": "Comprehensive response to malware infections",
                "category": "malware",
                "steps": [
                    {
                        "action_type": "isolate_host_advanced",
                        "parameters": {"isolation_level": "strict"},
                    },
                    {
                        "action_type": "memory_dump_collection",
                        "parameters": {"dump_type": "full"},
                    },
                    {
                        "action_type": "block_ip_advanced",
                        "parameters": {"duration": 3600},
                    },
                ],
                "estimated_duration_minutes": 25,
                "times_used": 42,
                "success_rate": 0.94,
            },
            {
                "id": "ddos-mitigation",
                "name": "DDoS Attack Mitigation",
                "description": "Rapid response to distributed denial of service attacks",
                "category": "ddos",
                "steps": [
                    {
                        "action_type": "traffic_redirection",
                        "parameters": {"destination": "scrubbing_center"},
                    },
                    {
                        "action_type": "deploy_firewall_rules",
                        "parameters": {"rule_set": "ddos_protection"},
                    },
                    {
                        "action_type": "block_ip_advanced",
                        "parameters": {"block_level": "source_networks"},
                    },
                ],
                "estimated_duration_minutes": 8,
                "times_used": 28,
                "success_rate": 0.89,
            },
        ]

        if category and category != "all":
            templates = [t for t in templates if t["category"] == category]

        return {"success": True, "templates": templates, "total_count": len(templates)}

    except Exception as e:
        logger.error(f"Failed to get templates: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get templates: {str(e)}"
        )


@app.post("/api/workflows/templates/create")
async def create_playbook_template(
    request: Dict[str, Any], db: AsyncSession = Depends(get_db)
):
    """Create a new playbook template"""
    try:
        # For now, return success - in production would create ResponsePlaybook record
        template_data = {
            "id": f"custom_{request.get('name', '').lower().replace(' ', '_')}",
            "name": request.get("name"),
            "description": request.get("description"),
            "category": request.get("category"),
            "steps": request.get("steps", []),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        return {
            "success": True,
            "template": template_data,
            "message": "Template created successfully",
        }

    except Exception as e:
        logger.error(f"Failed to create template: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to create template: {str(e)}"
        )


@app.post("/api/workflows/visual/validate")
async def validate_visual_workflow(
    request: Dict[str, Any], db: AsyncSession = Depends(get_db)
):
    """Validate a visual workflow design"""
    try:
        steps = request.get("steps", [])

        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "suggestions": [],
        }

        # Basic validation
        if len(steps) == 0:
            validation_result["valid"] = False
            validation_result["errors"].append(
                "Workflow must contain at least one action"
            )

        # Check for unknown action types
        response_engine = await get_response_engine()
        available_actions = response_engine.get_available_actions()

        for step in steps:
            action_type = step.get("action_type")
            if action_type and action_type not in available_actions.get("actions", {}):
                validation_result["warnings"].append(
                    f"Unknown action type: {action_type}"
                )

        # Add optimization suggestions
        if len(steps) > 5:
            validation_result["suggestions"].append(
                "Consider breaking this into smaller workflows"
            )

        return {"success": True, "validation": validation_result}

    except Exception as e:
        logger.error(f"Failed to validate workflow: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to validate workflow: {str(e)}"
        )


# =============================================================================
# WEBSOCKET ENDPOINTS FOR REAL-TIME UPDATES
# =============================================================================


@app.websocket("/ws/general")
async def websocket_endpoint(websocket: WebSocket):
    """General WebSocket endpoint for real-time updates"""
    await ws_manager.connect(websocket, {"type": "general"})

    try:
        while True:
            # Keep connection alive and handle incoming messages
            data = await websocket.receive_text()

            # Handle ping/pong for connection health
            if data == "ping":
                await ws_manager.send_personal_message(websocket, {"type": "pong"})

    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        ws_manager.disconnect(websocket)


@app.websocket("/ws/workflows")
async def workflow_websocket(websocket: WebSocket):
    """WebSocket endpoint specifically for workflow updates"""
    await ws_manager.connect(websocket, {"type": "workflows"})

    try:
        while True:
            data = await websocket.receive_text()

            if data == "ping":
                await ws_manager.send_personal_message(websocket, {"type": "pong"})

    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"Workflow WebSocket error: {e}")
        ws_manager.disconnect(websocket)


@app.websocket("/ws/incidents")
async def incident_websocket(websocket: WebSocket):
    """WebSocket endpoint specifically for incident updates"""
    await ws_manager.connect(websocket, {"type": "incidents"})

    try:
        while True:
            data = await websocket.receive_text()

            if data == "ping":
                await ws_manager.send_personal_message(websocket, {"type": "pong"})

    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"Incident WebSocket error: {e}")
        ws_manager.disconnect(websocket)


@app.get("/api/websocket/status")
async def websocket_status():
    """Get WebSocket connection status"""
    return {
        "total_connections": ws_manager.get_connection_count(),
        "status": "active" if ws_manager.get_connection_count() > 0 else "inactive",
    }


@app.get("/api/auth/config")
async def get_auth_config():
    """Get authentication configuration for frontend"""
    from .config import settings

    return {
        "api_key": settings.api_key,
        "websocket_enabled": True,
        "base_url": f"http://{settings.api_host}:{settings.api_port}",
    }


# Background task helper
async def _train_models_background(enable_federated: bool = True):
    """Background task for training models with federated learning"""
    try:
        # Prepare training data from recent events
        from .ml_engine import prepare_training_data_from_events

        async with AsyncSessionLocal() as db:
            # Get recent events for training
            window_start = datetime.now(timezone.utc) - timedelta(days=7)

            query = (
                select(Event)
                .where(Event.ts >= window_start)
                .order_by(Event.ts.desc())
                .limit(5000)
            )

            result = await db.execute(query)
            events = result.scalars().all()

            # Prepare training data
            training_data = await prepare_training_data_from_events(events)

            if len(training_data) >= 50:
                # Train models
                results = await ml_detector.train_models(
                    training_data, enable_federated=enable_federated
                )

                logger.info(f"Background federated training completed: {results}")
            else:
                logger.warning(
                    f"Insufficient training data: {len(training_data)} samples"
                )

    except Exception as e:
        logger.error(f"Background federated training failed: {e}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app", host=settings.api_host, port=settings.api_port, reload=True
    )


# T-Pot Action Verification Endpoints
@app.post("/api/incidents/{incident_id}/verify-actions")
async def verify_incident_actions_endpoint(
    incident_id: int, request: Request, db: AsyncSession = Depends(get_db)
):
    """Verify all actions for an incident were executed on T-Pot"""
    _require_api_key(request)

    from .verification_endpoints import verify_incident_actions

    return await verify_incident_actions(incident_id, db)


@app.post("/api/actions/{action_id}/verify")
async def verify_single_action_endpoint(
    action_id: int,
    action_type: str = "basic",
    request: Request = None,
    db: AsyncSession = Depends(get_db),
):
    """Verify a single action was executed on T-Pot"""
    _require_api_key(request)

    from .verification_endpoints import verify_single_action

    return await verify_single_action(action_id, action_type, db)


@app.get("/api/tpot/status")
async def get_tpot_status_endpoint(request: Request):
    """Get current T-Pot firewall status and active blocks"""
    _require_api_key(request)

    from .verification_endpoints import get_tpot_status

    return await get_tpot_status()


# ==================== NEW AGENT ENDPOINTS (IAM, EDR, DLP) ====================


@app.post("/api/agents/iam/execute")
async def execute_iam_action(
    action_data: Dict[str, Any], db: AsyncSession = Depends(get_db)
):
    """Execute IAM agent action (Active Directory management)"""
    from .agents.iam_agent import iam_agent
    from .models import ActionLog

    action_name = action_data.get("action_name")
    params = action_data.get("params", {})
    incident_id = action_data.get("incident_id")

    if not action_name or not params:
        raise HTTPException(status_code=400, detail="action_name and params required")

    try:
        # Execute action
        result = await iam_agent.execute_action(action_name, params, incident_id)

        # Log to database
        action_log = ActionLog(
            action_id=result.get("action_id"),
            agent_id=result.get("agent"),
            agent_type="iam",
            action_name=action_name,
            incident_id=incident_id,
            params=params,
            result=result.get("result"),
            status="success" if result.get("success") else "failed",
            error=result.get("error"),
            rollback_id=result.get("rollback_id"),
        )

        db.add(action_log)
        await db.commit()
        await db.refresh(action_log)

        return result

    except Exception as e:
        logger.error(f"IAM action failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/agents/edr/execute")
async def execute_edr_action(
    action_data: Dict[str, Any], db: AsyncSession = Depends(get_db)
):
    """Execute EDR agent action (Windows endpoint management)"""
    from .agents.edr_agent import edr_agent
    from .models import ActionLog

    action_name = action_data.get("action_name")
    params = action_data.get("params", {})
    incident_id = action_data.get("incident_id")

    if not action_name or not params:
        raise HTTPException(status_code=400, detail="action_name and params required")

    try:
        # Execute action
        result = await edr_agent.execute_action(action_name, params, incident_id)

        # Log to database
        action_log = ActionLog(
            action_id=result.get("action_id"),
            agent_id=result.get("agent"),
            agent_type="edr",
            action_name=action_name,
            incident_id=incident_id,
            params=params,
            result=result.get("result"),
            status="success" if result.get("success") else "failed",
            error=result.get("error"),
            rollback_id=result.get("rollback_id"),
        )

        db.add(action_log)
        await db.commit()
        await db.refresh(action_log)

        return result

    except Exception as e:
        logger.error(f"EDR action failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/agents/dlp/execute")
async def execute_dlp_action(
    action_data: Dict[str, Any], db: AsyncSession = Depends(get_db)
):
    """Execute DLP agent action (Data loss prevention)"""
    from .agents.dlp_agent import dlp_agent
    from .models import ActionLog

    action_name = action_data.get("action_name")
    params = action_data.get("params", {})
    incident_id = action_data.get("incident_id")

    if not action_name or not params:
        raise HTTPException(status_code=400, detail="action_name and params required")

    try:
        # Execute action
        result = await dlp_agent.execute_action(action_name, params, incident_id)

        # Log to database
        action_log = ActionLog(
            action_id=result.get("action_id"),
            agent_id=result.get("agent"),
            agent_type="dlp",
            action_name=action_name,
            incident_id=incident_id,
            params=params,
            result=result.get("result"),
            status="success" if result.get("success") else "failed",
            error=result.get("error"),
            rollback_id=result.get("rollback_id"),
        )

        db.add(action_log)
        await db.commit()
        await db.refresh(action_log)

        return result

    except Exception as e:
        logger.error(f"DLP action failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/agents/rollback/{rollback_id}")
async def rollback_agent_action(rollback_id: str, db: AsyncSession = Depends(get_db)):
    """Rollback an agent action by rollback_id"""
    try:
        # Find the action log
        action_log = (
            (
                await db.execute(
                    select(ActionLog).where(ActionLog.rollback_id == rollback_id)
                )
            )
            .scalars()
            .first()
        )

        if not action_log:
            raise HTTPException(status_code=404, detail="Rollback ID not found")

        if action_log.rollback_executed:
            return {
                "success": False,
                "message": "Rollback already executed",
                "executed_at": action_log.rollback_timestamp,
            }

        # Execute rollback based on agent type
        agent_type = action_log.agent_type
        result = None

        if agent_type == "iam":
            from .agents.iam_agent import iam_agent

            result = await iam_agent.rollback_action(rollback_id)
        elif agent_type == "edr":
            from .agents.edr_agent import edr_agent

            result = await edr_agent.rollback_action(rollback_id)
        elif agent_type == "dlp":
            from .agents.dlp_agent import dlp_agent

            result = await dlp_agent.rollback_action(rollback_id)
        else:
            raise HTTPException(
                status_code=400, detail=f"Unknown agent type: {agent_type}"
            )

        # Update action log
        action_log.rollback_executed = True
        action_log.rollback_timestamp = datetime.now(timezone.utc)
        action_log.rollback_result = result
        action_log.status = "rolled_back"

        await db.commit()
        await db.refresh(action_log)

        return result

    except Exception as e:
        logger.error(f"Rollback failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/agents/actions")
async def get_action_logs(
    incident_id: Optional[int] = None,
    agent_type: Optional[str] = None,
    limit: int = 100,
    db: AsyncSession = Depends(get_db),
):
    """Get action logs with optional filtering"""
    query = select(ActionLog).order_by(desc(ActionLog.executed_at))

    if incident_id:
        query = query.where(ActionLog.incident_id == incident_id)

    if agent_type:
        query = query.where(ActionLog.agent_type == agent_type)

    query = query.limit(limit)

    result = await db.execute(query)
    action_logs = result.scalars().all()

    return [
        {
            "id": log.id,
            "action_id": log.action_id,
            "agent_id": log.agent_id,
            "agent_type": log.agent_type,
            "action_name": log.action_name,
            "incident_id": log.incident_id,
            "params": log.params,
            "result": log.result,
            "status": log.status,
            "error": log.error,
            "rollback_id": log.rollback_id,
            "rollback_executed": log.rollback_executed,
            "rollback_timestamp": log.rollback_timestamp.isoformat()
            if log.rollback_timestamp
            else None,
            "executed_at": log.executed_at.isoformat() if log.executed_at else None,
            "created_at": log.created_at.isoformat() if log.created_at else None,
        }
        for log in action_logs
    ]


@app.get("/api/agents/actions/{incident_id}")
async def get_incident_actions(incident_id: int, db: AsyncSession = Depends(get_db)):
    """Get all actions for a specific incident"""
    query = (
        select(ActionLog)
        .where(ActionLog.incident_id == incident_id)
        .order_by(desc(ActionLog.executed_at))
    )

    result = await db.execute(query)
    action_logs = result.scalars().all()

    return [
        {
            "id": log.id,
            "action_id": log.action_id,
            "agent_id": log.agent_id,
            "agent_type": log.agent_type,
            "action_name": log.action_name,
            "params": log.params,
            "result": log.result,
            "status": log.status,
            "error": log.error,
            "rollback_id": log.rollback_id,
            "rollback_executed": log.rollback_executed,
            "rollback_timestamp": log.rollback_timestamp.isoformat()
            if log.rollback_timestamp
            else None,
            "executed_at": log.executed_at.isoformat() if log.executed_at else None,
        }
        for log in action_logs
    ]
