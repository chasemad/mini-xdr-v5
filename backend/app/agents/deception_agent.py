"""
Deception & Honeypot Management Agent
Dynamic honeypot and deception technology management
"""
import asyncio
import json
import logging
import yaml
import subprocess
import docker
import time
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict, field
from pathlib import Path
import tempfile
import shutil
import random
import string

try:
    from langchain_openai import ChatOpenAI
except ImportError:
    ChatOpenAI = None

from ..models import Event, Incident
from ..config import settings

logger = logging.getLogger(__name__)


@dataclass
class HoneypotConfig:
    """Honeypot configuration"""
    honeypot_id: str
    name: str
    type: str  # cowrie, dionaea, conpot, elastichoney, etc.
    status: str  # running, stopped, error, deploying
    created_at: datetime
    updated_at: datetime
    config: Dict[str, Any]
    network_config: Dict[str, str]
    container_id: Optional[str] = None
    ports: List[int] = field(default_factory=list)
    volumes: Dict[str, str] = field(default_factory=dict)
    environment: Dict[str, str] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DeceptionScenario:
    """Deception scenario definition"""
    scenario_id: str
    name: str
    description: str
    type: str  # honeypot, decoy_service, fake_vulnerability, etc.
    target_attacks: List[str]
    honeypots: List[str]
    effectiveness_metrics: Dict[str, float]
    created_at: datetime
    active: bool = True


@dataclass
class AttackerProfile:
    """Attacker behavioral profile"""
    profile_id: str
    src_ip: str
    first_seen: datetime
    last_seen: datetime
    total_interactions: int
    attack_patterns: List[str]
    sophistication_level: str  # low, medium, high, advanced
    persistence_score: float
    stealth_score: float
    technique_diversity: int
    target_preferences: List[str]
    behavioral_signature: Dict[str, Any]


class DeceptionAgent:
    """AI Agent for dynamic deception and honeypot management"""
    
    def __init__(self, llm_client=None):
        self.llm_client = llm_client or self._init_llm_client()
        self.agent_id = "deception_manager_v1"
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Docker client for container management
        try:
            self.docker_client = docker.from_env()
        except Exception as e:
            self.logger.warning(f"Docker client initialization failed: {e}")
            self.docker_client = None
        
        # Honeypot management
        self.honeypots: Dict[str, HoneypotConfig] = {}
        self.deception_scenarios: Dict[str, DeceptionScenario] = {}
        self.attacker_profiles: Dict[str, AttackerProfile] = {}
        
        # Configuration paths
        self.config_dir = Path("./honeypot_configs")
        self.config_dir.mkdir(exist_ok=True, parents=True)
        
        # Honeypot templates
        self.honeypot_templates = {
            "cowrie_ssh": {
                "image": "cowrie/cowrie:latest",
                "ports": {"2222": "22"},
                "environment": {
                    "COWRIE_HOSTNAME": "production-server",
                    "COWRIE_LOG_LEVEL": "INFO"
                },
                "config_template": self._get_cowrie_config_template(),
                "volumes": {
                    "/var/log/cowrie": "/cowrie/var/log/cowrie",
                    "/var/lib/cowrie": "/cowrie/var/lib/cowrie"
                }
            },
            "dionaea_malware": {
                "image": "dinotools/dionaea:latest",
                "ports": {"21": "21", "80": "80", "443": "443", "135": "135"},
                "environment": {
                    "DIONAEA_LOG_LEVEL": "info"
                },
                "config_template": self._get_dionaea_config_template(),
                "volumes": {
                    "/var/log/dionaea": "/opt/dionaea/var/log"
                }
            },
            "conpot_industrial": {
                "image": "honeynet/conpot:latest",
                "ports": {"502": "502", "161": "161"},
                "environment": {
                    "CONPOT_CONFIG": "default"
                },
                "config_template": self._get_conpot_config_template(),
                "volumes": {
                    "/var/log/conpot": "/var/log/conpot"
                }
            },
            "elastichoney_elasticsearch": {
                "image": "jordan/elastichoney:latest",
                "ports": {"9200": "9200"},
                "environment": {
                    "ELASTICHONEY_LOGFILE": "/var/log/elastichoney/elastichoney.log"
                },
                "config_template": self._get_elastichoney_config_template(),
                "volumes": {
                    "/var/log/elastichoney": "/var/log/elastichoney"
                }
            },
            "web_honeypot": {
                "image": "nginx:alpine",
                "ports": {"8080": "80"},
                "environment": {},
                "config_template": self._get_web_honeypot_config_template(),
                "volumes": {
                    "/var/log/nginx": "/var/log/nginx"
                }
            }
        }
        
        # Attack pattern recognition
        self.attack_patterns = {
            "brute_force": {
                "indicators": ["rapid_login_attempts", "credential_enumeration"],
                "countermeasures": ["rate_limiting", "credential_diversity"]
            },
            "malware_deployment": {
                "indicators": ["file_download", "executable_execution"],
                "countermeasures": ["sandbox_analysis", "behavioral_monitoring"]
            },
            "reconnaissance": {
                "indicators": ["port_scanning", "service_enumeration"],
                "countermeasures": ["decoy_services", "false_banners"]
            },
            "lateral_movement": {
                "indicators": ["multi_host_scanning", "credential_reuse"],
                "countermeasures": ["network_segmentation", "credential_honeytokens"]
            },
            "data_exfiltration": {
                "indicators": ["large_downloads", "database_queries"],
                "countermeasures": ["data_honeypots", "access_monitoring"]
            }
        }
        
        # Load existing configurations - will be called when needed
        # asyncio.create_task(self._load_existing_honeypots())
    
    def _init_llm_client(self):
        """Initialize LLM client for intelligent deception strategies"""
        try:
            if settings.llm_provider.lower() == "openai" and settings.openai_api_key:
                if ChatOpenAI:
                    return ChatOpenAI(
                        openai_api_key=settings.openai_api_key,
                        model_name=settings.openai_model,
                        temperature=0.3  # Some creativity for deception
                    )
        except Exception as e:
            self.logger.error(f"Failed to initialize LLM client: {e}")
        return None
    
    async def _load_existing_honeypots(self):
        """Load existing honeypot configurations"""
        
        try:
            if not self.docker_client:
                return
            
            # Find existing honeypot containers
            containers = self.docker_client.containers.list(all=True, filters={"label": "honeypot=true"})
            
            for container in containers:
                honeypot_id = container.labels.get("honeypot_id")
                if honeypot_id:
                    honeypot_config = self._container_to_config(container)
                    self.honeypots[honeypot_id] = honeypot_config
                    
                    self.logger.info(f"Loaded existing honeypot: {honeypot_id}")
        
        except Exception as e:
            self.logger.error(f"Failed to load existing honeypots: {e}")
    
    def _container_to_config(self, container) -> HoneypotConfig:
        """Convert Docker container to honeypot config"""
        
        return HoneypotConfig(
            honeypot_id=container.labels.get("honeypot_id", container.id[:12]),
            name=container.labels.get("honeypot_name", container.name),
            type=container.labels.get("honeypot_type", "unknown"),
            status="running" if container.status == "running" else "stopped",
            created_at=datetime.fromisoformat(container.labels.get("created_at", datetime.utcnow().isoformat())),
            updated_at=datetime.utcnow(),
            config={},
            network_config={},
            container_id=container.id,
            ports=[],
            volumes={},
            environment={}
        )
    
    async def deploy_honeypot(
        self, 
        honeypot_type: str, 
        name: str = None,
        custom_config: Dict[str, Any] = None,
        network_segment: str = "default"
    ) -> str:
        """
        Deploy a new honeypot
        
        Args:
            honeypot_type: Type of honeypot to deploy
            name: Custom name for the honeypot
            custom_config: Custom configuration overrides
            network_segment: Network segment to deploy to
            
        Returns:
            Honeypot ID
        """
        try:
            if honeypot_type not in self.honeypot_templates:
                raise ValueError(f"Unknown honeypot type: {honeypot_type}")
            
            if not self.docker_client:
                raise RuntimeError("Docker client not available")
            
            template = self.honeypot_templates[honeypot_type]
            honeypot_id = f"{honeypot_type}_{int(time.time())}"
            honeypot_name = name or f"honeypot_{honeypot_id}"
            
            # Prepare configuration
            config = template.copy()
            if custom_config:
                config.update(custom_config)
            
            # Generate configuration files
            config_path = await self._generate_honeypot_config(honeypot_id, honeypot_type, config)
            
            # Prepare container configuration
            container_config = {
                "image": config["image"],
                "name": honeypot_name,
                "detach": True,
                "ports": config.get("ports", {}),
                "environment": config.get("environment", {}),
                "volumes": config.get("volumes", {}),
                "labels": {
                    "honeypot": "true",
                    "honeypot_id": honeypot_id,
                    "honeypot_name": honeypot_name,
                    "honeypot_type": honeypot_type,
                    "created_at": datetime.utcnow().isoformat(),
                    "managed_by": "deception_agent"
                },
                "restart_policy": {"Name": "unless-stopped"}
            }
            
            # Add config volume if config was generated
            if config_path:
                container_config["volumes"][str(config_path)] = f"/config/{honeypot_type}.conf"
            
            # Deploy container
            container = self.docker_client.containers.run(**container_config)
            
            # Create honeypot configuration record
            honeypot_config = HoneypotConfig(
                honeypot_id=honeypot_id,
                name=honeypot_name,
                type=honeypot_type,
                status="running",
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                config=config,
                network_config={"segment": network_segment},
                container_id=container.id,
                ports=list(config.get("ports", {}).keys()),
                volumes=config.get("volumes", {}),
                environment=config.get("environment", {}),
                metrics={}
            )
            
            self.honeypots[honeypot_id] = honeypot_config
            
            self.logger.info(f"Deployed honeypot {honeypot_id} ({honeypot_type}) as container {container.id[:12]}")
            
            return honeypot_id
            
        except Exception as e:
            self.logger.error(f"Honeypot deployment failed: {e}")
            raise
    
    async def _generate_honeypot_config(
        self, 
        honeypot_id: str, 
        honeypot_type: str, 
        config: Dict[str, Any]
    ) -> Optional[Path]:
        """Generate configuration file for honeypot"""
        
        try:
            template = config.get("config_template")
            if not template:
                return None
            
            config_file = self.config_dir / f"{honeypot_id}.conf"
            
            # Render template with configuration
            rendered_config = template.format(**config.get("environment", {}))
            
            with open(config_file, 'w') as f:
                f.write(rendered_config)
            
            return config_file
            
        except Exception as e:
            self.logger.error(f"Configuration generation failed for {honeypot_id}: {e}")
            return None
    
    def _get_cowrie_config_template(self) -> str:
        """Get Cowrie configuration template"""
        return """
[honeypot]
hostname = {COWRIE_HOSTNAME}
log_path = /cowrie/var/log/cowrie
download_path = /cowrie/var/lib/cowrie/downloads
contents_path = /cowrie/honeyfs
txtcmds_path = /cowrie/txtcmds
data_path = /cowrie/var/lib/cowrie

[ssh]
enabled = true
listen_port = 2222
forwarded_ports = 22

[telnet]
enabled = false

[output_jsonlog]
enabled = true
logfile = /cowrie/var/log/cowrie/cowrie.json

[output_mysql]
enabled = false
        """
    
    def _get_dionaea_config_template(self) -> str:
        """Get Dionaea configuration template"""
        return """
[logging]
default.filename = /opt/dionaea/var/log/dionaea.log
default.levels = info,warning,error
default.domain = *

[modules]
python.ftpd = enabled
python.httpd = enabled
python.smbd = enabled

[processors]
filter_streamdumper = enabled
filter_emu = enabled
        """
    
    def _get_conpot_config_template(self) -> str:
        """Get Conpot configuration template"""
        return """
[device]
device_name = "Siemens S7-1200"
device_type = "PLC"

[modbus]
enabled = true
port = 502

[snmp]
enabled = true
port = 161
        """
    
    def _get_elastichoney_config_template(self) -> str:
        """Get ElasticHoney configuration template"""
        return """
{
    "logfile": "/var/log/elastichoney/elastichoney.log",
    "use_remote_syslog": false,
    "syslog_server": "",
    "syslog_port": 514,
    "use_syslog": true,
    "syslog_facility": "LOG_INFO",
    "instance_name": "elastichoney"
}
        """
    
    def _get_web_honeypot_config_template(self) -> str:
        """Get web honeypot configuration template"""
        return """
server {
    listen 80;
    server_name _;
    
    access_log /var/log/nginx/access.log;
    error_log /var/log/nginx/error.log;
    
    location / {
        return 200 "Welcome to Production Server";
        add_header Content-Type text/plain;
    }
    
    location /admin {
        return 401 "Unauthorized";
        add_header Content-Type text/plain;
    }
    
    location /api {
        return 200 '{"status": "ok", "version": "1.0"}';
        add_header Content-Type application/json;
    }
}
        """
    
    async def stop_honeypot(self, honeypot_id: str) -> bool:
        """Stop a running honeypot"""
        
        try:
            if honeypot_id not in self.honeypots:
                raise ValueError(f"Honeypot {honeypot_id} not found")
            
            honeypot = self.honeypots[honeypot_id]
            
            if honeypot.container_id and self.docker_client:
                container = self.docker_client.containers.get(honeypot.container_id)
                container.stop()
                
                honeypot.status = "stopped"
                honeypot.updated_at = datetime.utcnow()
                
                self.logger.info(f"Stopped honeypot {honeypot_id}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to stop honeypot {honeypot_id}: {e}")
            return False
    
    async def start_honeypot(self, honeypot_id: str) -> bool:
        """Start a stopped honeypot"""
        
        try:
            if honeypot_id not in self.honeypots:
                raise ValueError(f"Honeypot {honeypot_id} not found")
            
            honeypot = self.honeypots[honeypot_id]
            
            if honeypot.container_id and self.docker_client:
                container = self.docker_client.containers.get(honeypot.container_id)
                container.start()
                
                honeypot.status = "running"
                honeypot.updated_at = datetime.utcnow()
                
                self.logger.info(f"Started honeypot {honeypot_id}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to start honeypot {honeypot_id}: {e}")
            return False
    
    async def remove_honeypot(self, honeypot_id: str) -> bool:
        """Remove a honeypot completely"""
        
        try:
            if honeypot_id not in self.honeypots:
                raise ValueError(f"Honeypot {honeypot_id} not found")
            
            honeypot = self.honeypots[honeypot_id]
            
            # Stop and remove container
            if honeypot.container_id and self.docker_client:
                try:
                    container = self.docker_client.containers.get(honeypot.container_id)
                    container.stop()
                    container.remove()
                except Exception as e:
                    self.logger.warning(f"Container removal failed: {e}")
            
            # Remove configuration file
            config_file = self.config_dir / f"{honeypot_id}.conf"
            if config_file.exists():
                config_file.unlink()
            
            # Remove from tracking
            del self.honeypots[honeypot_id]
            
            self.logger.info(f"Removed honeypot {honeypot_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to remove honeypot {honeypot_id}: {e}")
            return False
    
    async def analyze_attacker_behavior(
        self, 
        events: List[Event], 
        timeframe_hours: int = 24
    ) -> Dict[str, AttackerProfile]:
        """
        Analyze attacker behavior patterns from honeypot events
        
        Args:
            events: List of honeypot events
            timeframe_hours: Analysis timeframe
            
        Returns:
            Dictionary of attacker profiles by IP
        """
        try:
            # Group events by source IP
            ip_events = {}
            cutoff_time = datetime.utcnow() - timedelta(hours=timeframe_hours)
            
            for event in events:
                if event.ts >= cutoff_time:
                    ip = event.src_ip
                    if ip not in ip_events:
                        ip_events[ip] = []
                    ip_events[ip].append(event)
            
            # Analyze each attacker
            attacker_profiles = {}
            
            for src_ip, attacker_events in ip_events.items():
                profile = await self._analyze_single_attacker(src_ip, attacker_events)
                attacker_profiles[src_ip] = profile
                
                # Update persistent profiles
                self.attacker_profiles[src_ip] = profile
            
            self.logger.info(f"Analyzed {len(attacker_profiles)} attacker profiles")
            
            return attacker_profiles
            
        except Exception as e:
            self.logger.error(f"Attacker behavior analysis failed: {e}")
            return {}
    
    async def _analyze_single_attacker(self, src_ip: str, events: List[Event]) -> AttackerProfile:
        """Analyze behavior pattern for a single attacker"""
        
        # Basic metrics
        first_seen = min(event.ts for event in events)
        last_seen = max(event.ts for event in events)
        total_interactions = len(events)
        
        # Analyze attack patterns
        attack_patterns = []
        event_types = [event.eventid for event in events]
        
        # Detect brute force
        failed_logins = len([e for e in events if e.eventid == "cowrie.login.failed"])
        if failed_logins > 10:
            attack_patterns.append("brute_force")
        
        # Detect malware deployment
        downloads = len([e for e in events if e.eventid == "cowrie.session.file_download"])
        if downloads > 0:
            attack_patterns.append("malware_deployment")
        
        # Detect reconnaissance
        commands = len([e for e in events if e.eventid == "cowrie.command.input"])
        if commands > 5:
            attack_patterns.append("reconnaissance")
        
        # Calculate sophistication level
        sophistication_level = self._calculate_sophistication_level(events, attack_patterns)
        
        # Calculate behavioral scores
        persistence_score = self._calculate_persistence_score(events, first_seen, last_seen)
        stealth_score = self._calculate_stealth_score(events)
        technique_diversity = len(set(event_types))
        
        # Identify target preferences
        target_preferences = self._identify_target_preferences(events)
        
        # Create behavioral signature
        behavioral_signature = {
            "session_duration_avg": self._calculate_avg_session_duration(events),
            "command_frequency": commands / max(total_interactions, 1),
            "download_frequency": downloads / max(total_interactions, 1),
            "login_failure_rate": failed_logins / max(total_interactions, 1),
            "event_types": list(set(event_types)),
            "time_pattern": self._analyze_time_pattern(events)
        }
        
        return AttackerProfile(
            profile_id=f"attacker_{src_ip}_{int(time.time())}",
            src_ip=src_ip,
            first_seen=first_seen,
            last_seen=last_seen,
            total_interactions=total_interactions,
            attack_patterns=attack_patterns,
            sophistication_level=sophistication_level,
            persistence_score=persistence_score,
            stealth_score=stealth_score,
            technique_diversity=technique_diversity,
            target_preferences=target_preferences,
            behavioral_signature=behavioral_signature
        )
    
    def _calculate_sophistication_level(self, events: List[Event], attack_patterns: List[str]) -> str:
        """Calculate attacker sophistication level"""
        
        score = 0
        
        # Pattern sophistication
        if "malware_deployment" in attack_patterns:
            score += 3
        if "reconnaissance" in attack_patterns:
            score += 2
        if "brute_force" in attack_patterns:
            score += 1
        
        # Command sophistication
        commands = [e.raw.get("input", "") for e in events if e.eventid == "cowrie.command.input" and e.raw]
        advanced_commands = ["wget", "curl", "nc", "nmap", "python", "perl", "base64"]
        
        for command in commands:
            if any(adv_cmd in command.lower() for adv_cmd in advanced_commands):
                score += 1
        
        # Time-based analysis
        if len(events) > 0:
            time_span = (max(e.ts for e in events) - min(e.ts for e in events)).total_seconds()
            if time_span > 3600:  # More than 1 hour of activity
                score += 2
        
        # Determine level
        if score >= 8:
            return "advanced"
        elif score >= 5:
            return "high"
        elif score >= 3:
            return "medium"
        else:
            return "low"
    
    def _calculate_persistence_score(self, events: List[Event], first_seen: datetime, last_seen: datetime) -> float:
        """Calculate attacker persistence score"""
        
        # Time span factor
        time_span_hours = (last_seen - first_seen).total_seconds() / 3600
        persistence_score = min(time_span_hours / 24, 1.0)  # Normalize to 0-1
        
        # Multiple session factor
        sessions = set()
        for event in events:
            if event.raw and "session" in event.raw:
                sessions.add(event.raw["session"])
        
        if len(sessions) > 1:
            persistence_score += 0.3
        
        # Retry factor
        failed_attempts = len([e for e in events if "failed" in e.eventid])
        if failed_attempts > 20:
            persistence_score += 0.2
        
        return min(persistence_score, 1.0)
    
    def _calculate_stealth_score(self, events: List[Event]) -> float:
        """Calculate attacker stealth score"""
        
        stealth_score = 1.0  # Start with maximum stealth
        
        # Reduce for high-volume activity
        if len(events) > 100:
            stealth_score -= 0.3
        
        # Reduce for obvious brute force
        failed_logins = len([e for e in events if e.eventid == "cowrie.login.failed"])
        if failed_logins > 50:
            stealth_score -= 0.4
        
        # Reduce for malware downloads
        downloads = len([e for e in events if e.eventid == "cowrie.session.file_download"])
        if downloads > 0:
            stealth_score -= 0.2
        
        return max(stealth_score, 0.0)
    
    def _identify_target_preferences(self, events: List[Event]) -> List[str]:
        """Identify attacker target preferences"""
        
        preferences = []
        
        # Service preferences
        ports = [e.dst_port for e in events if e.dst_port]
        if 22 in ports:
            preferences.append("ssh_services")
        if 80 in ports or 443 in ports:
            preferences.append("web_services")
        if 21 in ports:
            preferences.append("ftp_services")
        
        # Operating system preferences
        commands = [e.raw.get("input", "") for e in events if e.eventid == "cowrie.command.input" and e.raw]
        linux_commands = ["ls", "cat", "ps", "netstat", "uname"]
        windows_commands = ["dir", "type", "tasklist", "netstat", "systeminfo"]
        
        linux_score = sum(1 for cmd in commands if any(lc in cmd.lower() for lc in linux_commands))
        windows_score = sum(1 for cmd in commands if any(wc in cmd.lower() for wc in windows_commands))
        
        if linux_score > windows_score:
            preferences.append("linux_systems")
        elif windows_score > linux_score:
            preferences.append("windows_systems")
        
        return preferences
    
    def _calculate_avg_session_duration(self, events: List[Event]) -> float:
        """Calculate average session duration"""
        
        # Group events by session
        sessions = {}
        for event in events:
            if event.raw and "session" in event.raw:
                session_id = event.raw["session"]
                if session_id not in sessions:
                    sessions[session_id] = []
                sessions[session_id].append(event.ts)
        
        # Calculate durations
        durations = []
        for session_events in sessions.values():
            if len(session_events) > 1:
                duration = (max(session_events) - min(session_events)).total_seconds()
                durations.append(duration)
        
        return sum(durations) / len(durations) if durations else 0.0
    
    def _analyze_time_pattern(self, events: List[Event]) -> Dict[str, Any]:
        """Analyze temporal patterns in attacks"""
        
        hours = [event.ts.hour for event in events]
        days = [event.ts.weekday() for event in events]
        
        # Peak activity hour
        hour_counts = {}
        for hour in hours:
            hour_counts[hour] = hour_counts.get(hour, 0) + 1
        
        peak_hour = max(hour_counts.items(), key=lambda x: x[1])[0] if hour_counts else 0
        
        # Weekend activity
        weekend_events = len([d for d in days if d >= 5])
        weekend_ratio = weekend_events / len(events) if events else 0
        
        return {
            "peak_hour": peak_hour,
            "weekend_ratio": weekend_ratio,
            "time_span_hours": (max(event.ts for event in events) - min(event.ts for event in events)).total_seconds() / 3600 if events else 0
        }
    
    async def adaptive_honeypot_deployment(
        self, 
        attacker_profiles: Dict[str, AttackerProfile],
        threat_landscape: Dict[str, Any] = None
    ) -> List[str]:
        """
        Deploy honeypots adaptively based on attacker behavior and threat landscape
        
        Args:
            attacker_profiles: Current attacker profiles
            threat_landscape: Current threat landscape data
            
        Returns:
            List of deployed honeypot IDs
        """
        try:
            deployment_plan = await self._generate_deployment_plan(attacker_profiles, threat_landscape)
            
            deployed_honeypots = []
            
            for deployment in deployment_plan:
                try:
                    honeypot_id = await self.deploy_honeypot(
                        honeypot_type=deployment["type"],
                        name=deployment["name"],
                        custom_config=deployment.get("config", {}),
                        network_segment=deployment.get("network_segment", "default")
                    )
                    
                    deployed_honeypots.append(honeypot_id)
                    
                    self.logger.info(f"Adaptively deployed {deployment['type']} honeypot: {honeypot_id}")
                    
                except Exception as e:
                    self.logger.error(f"Failed to deploy adaptive honeypot {deployment['type']}: {e}")
            
            return deployed_honeypots
            
        except Exception as e:
            self.logger.error(f"Adaptive deployment failed: {e}")
            return []
    
    async def _generate_deployment_plan(
        self, 
        attacker_profiles: Dict[str, AttackerProfile],
        threat_landscape: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """Generate adaptive deployment plan"""
        
        deployment_plan = []
        
        # Analyze attacker preferences
        target_preferences = {}
        attack_patterns = {}
        sophistication_levels = {}
        
        for profile in attacker_profiles.values():
            # Count target preferences
            for pref in profile.target_preferences:
                target_preferences[pref] = target_preferences.get(pref, 0) + 1
            
            # Count attack patterns
            for pattern in profile.attack_patterns:
                attack_patterns[pattern] = attack_patterns.get(pattern, 0) + 1
            
            # Count sophistication levels
            level = profile.sophistication_level
            sophistication_levels[level] = sophistication_levels.get(level, 0) + 1
        
        # Generate deployments based on analysis
        
        # SSH honeypots for brute force attacks
        if attack_patterns.get("brute_force", 0) > 0 or target_preferences.get("ssh_services", 0) > 0:
            deployment_plan.append({
                "type": "cowrie_ssh",
                "name": f"ssh_adaptive_{int(time.time())}",
                "config": {
                    "environment": {
                        "COWRIE_HOSTNAME": "production-ssh-gateway"
                    }
                },
                "network_segment": "dmz"
            })
        
        # Web honeypots for web-focused attacks
        if target_preferences.get("web_services", 0) > 0:
            deployment_plan.append({
                "type": "web_honeypot",
                "name": f"web_adaptive_{int(time.time())}",
                "config": {},
                "network_segment": "web"
            })
        
        # Malware honeypots for sophisticated attackers
        if sophistication_levels.get("high", 0) > 0 or sophistication_levels.get("advanced", 0) > 0:
            deployment_plan.append({
                "type": "dionaea_malware",
                "name": f"malware_adaptive_{int(time.time())}",
                "config": {},
                "network_segment": "internal"
            })
        
        # Industrial honeypots if industrial targets detected
        if threat_landscape and threat_landscape.get("industrial_threats", False):
            deployment_plan.append({
                "type": "conpot_industrial",
                "name": f"industrial_adaptive_{int(time.time())}",
                "config": {},
                "network_segment": "ot"
            })
        
        return deployment_plan
    
    async def create_deception_scenario(
        self, 
        scenario_name: str,
        scenario_type: str,
        target_attacks: List[str],
        configuration: Dict[str, Any] = None
    ) -> str:
        """
        Create a comprehensive deception scenario
        
        Args:
            scenario_name: Name of the scenario
            scenario_type: Type of deception scenario
            target_attacks: List of attacks this scenario targets
            configuration: Scenario-specific configuration
            
        Returns:
            Scenario ID
        """
        try:
            scenario_id = f"{scenario_type}_{int(time.time())}"
            
            # Deploy honeypots for the scenario
            deployed_honeypots = []
            
            if scenario_type == "enterprise_network":
                # Deploy multiple honeypots to simulate enterprise network
                honeypot_types = ["cowrie_ssh", "web_honeypot", "dionaea_malware"]
                
                for hp_type in honeypot_types:
                    honeypot_id = await self.deploy_honeypot(
                        honeypot_type=hp_type,
                        name=f"{scenario_name}_{hp_type}",
                        custom_config=configuration.get(hp_type, {})
                    )
                    deployed_honeypots.append(honeypot_id)
            
            elif scenario_type == "industrial_control":
                # Deploy industrial control system honeypots
                honeypot_id = await self.deploy_honeypot(
                    honeypot_type="conpot_industrial",
                    name=f"{scenario_name}_scada",
                    custom_config=configuration.get("conpot_industrial", {})
                )
                deployed_honeypots.append(honeypot_id)
            
            elif scenario_type == "web_application":
                # Deploy web application honeypots with vulnerabilities
                honeypot_id = await self.deploy_honeypot(
                    honeypot_type="web_honeypot",
                    name=f"{scenario_name}_webapp",
                    custom_config=configuration.get("web_honeypot", {})
                )
                deployed_honeypots.append(honeypot_id)
            
            # Create scenario record
            scenario = DeceptionScenario(
                scenario_id=scenario_id,
                name=scenario_name,
                description=f"Deception scenario targeting: {', '.join(target_attacks)}",
                type=scenario_type,
                target_attacks=target_attacks,
                honeypots=deployed_honeypots,
                effectiveness_metrics={},
                created_at=datetime.utcnow(),
                active=True
            )
            
            self.deception_scenarios[scenario_id] = scenario
            
            self.logger.info(f"Created deception scenario {scenario_id} with {len(deployed_honeypots)} honeypots")
            
            return scenario_id
            
        except Exception as e:
            self.logger.error(f"Deception scenario creation failed: {e}")
            raise
    
    async def evaluate_honeypot_effectiveness(
        self, 
        honeypot_id: str, 
        evaluation_period_hours: int = 24
    ) -> Dict[str, Any]:
        """
        Evaluate the effectiveness of a honeypot
        
        Args:
            honeypot_id: ID of the honeypot to evaluate
            evaluation_period_hours: Evaluation period
            
        Returns:
            Effectiveness metrics
        """
        try:
            if honeypot_id not in self.honeypots:
                raise ValueError(f"Honeypot {honeypot_id} not found")
            
            honeypot = self.honeypots[honeypot_id]
            
            # Mock effectiveness evaluation (in production, analyze actual logs)
            effectiveness = {
                "honeypot_id": honeypot_id,
                "evaluation_period_hours": evaluation_period_hours,
                "interactions_count": random.randint(0, 50),
                "unique_attackers": random.randint(0, 10),
                "attack_types_detected": ["brute_force", "reconnaissance"],
                "data_collected_mb": random.uniform(0.1, 10.0),
                "false_positive_rate": random.uniform(0.0, 0.1),
                "detection_rate": random.uniform(0.7, 0.95),
                "threat_intelligence_value": random.uniform(0.5, 0.9),
                "recommendations": []
            }
            
            # Generate recommendations based on metrics
            if effectiveness["interactions_count"] < 5:
                effectiveness["recommendations"].append("Consider relocating honeypot to more visible network segment")
            
            if effectiveness["detection_rate"] < 0.8:
                effectiveness["recommendations"].append("Review honeypot configuration for improved realism")
            
            if effectiveness["false_positive_rate"] > 0.05:
                effectiveness["recommendations"].append("Tune honeypot to reduce false positives")
            
            # Update honeypot metrics
            honeypot.metrics.update({
                "last_evaluation": datetime.utcnow().isoformat(),
                "effectiveness_score": effectiveness["detection_rate"],
                "interaction_count": effectiveness["interactions_count"]
            })
            
            return effectiveness
            
        except Exception as e:
            self.logger.error(f"Honeypot effectiveness evaluation failed: {e}")
            return {}
    
    async def ai_powered_deception_strategy(
        self, 
        threat_intelligence: Dict[str, Any],
        current_attacks: List[str],
        organizational_profile: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Generate AI-powered deception strategy
        
        Args:
            threat_intelligence: Current threat intelligence
            current_attacks: List of current attack types
            organizational_profile: Organization-specific information
            
        Returns:
            Deception strategy recommendations
        """
        if not self.llm_client:
            return self._fallback_deception_strategy(current_attacks)
        
        # Prepare context for AI analysis
        context = {
            "threat_intelligence": threat_intelligence,
            "current_attacks": current_attacks,
            "organizational_profile": organizational_profile or {},
            "available_honeypot_types": list(self.honeypot_templates.keys()),
            "current_honeypots": len(self.honeypots),
            "attack_patterns": self.attack_patterns
        }
        
        prompt = f"""
        You are a cybersecurity deception expert. Design an adaptive deception strategy based on:
        
        THREAT CONTEXT:
        {json.dumps(context, indent=2)}
        
        Provide recommendations for:
        1. Honeypot deployment strategy
        2. Deception scenarios to implement
        3. Adaptive responses to current threats
        4. Resource allocation priorities
        
        Format response as JSON:
        {{
            "strategy_summary": "brief strategy overview",
            "honeypot_recommendations": [
                {{
                    "type": "honeypot_type",
                    "count": 2,
                    "placement": "network_segment",
                    "purpose": "specific_threat_targeting",
                    "priority": "high|medium|low"
                }}
            ],
            "deception_scenarios": [
                {{
                    "name": "scenario_name",
                    "type": "scenario_type",
                    "target_threats": ["threat1", "threat2"],
                    "complexity": "simple|moderate|complex"
                }}
            ],
            "adaptive_responses": [
                {{
                    "trigger": "threat_condition",
                    "response": "deception_action",
                    "automation_level": "manual|semi_auto|full_auto"
                }}
            ],
            "success_metrics": ["metric1", "metric2"],
            "timeline": "implementation_timeline"
        }}
        """
        
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.llm_client.invoke(prompt)
            )
            
            # Parse AI response
            import re
            json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if json_match:
                ai_strategy = json.loads(json_match.group())
                
                # Add AI metadata
                ai_strategy["ai_generated"] = True
                ai_strategy["generated_at"] = datetime.utcnow().isoformat()
                ai_strategy["confidence"] = 0.8
                
                return ai_strategy
        
        except Exception as e:
            self.logger.error(f"AI deception strategy generation failed: {e}")
        
        return self._fallback_deception_strategy(current_attacks)
    
    def _fallback_deception_strategy(self, current_attacks: List[str]) -> Dict[str, Any]:
        """Fallback deception strategy when AI is not available"""
        
        strategy = {
            "strategy_summary": "Rule-based deception strategy",
            "honeypot_recommendations": [],
            "deception_scenarios": [],
            "adaptive_responses": [],
            "success_metrics": ["interaction_count", "threat_detection_rate"],
            "timeline": "immediate_deployment",
            "ai_generated": False
        }
        
        # Basic rule-based recommendations
        if "brute_force" in current_attacks:
            strategy["honeypot_recommendations"].append({
                "type": "cowrie_ssh",
                "count": 2,
                "placement": "dmz",
                "purpose": "brute_force_detection",
                "priority": "high"
            })
        
        if "malware" in current_attacks:
            strategy["honeypot_recommendations"].append({
                "type": "dionaea_malware",
                "count": 1,
                "placement": "internal",
                "purpose": "malware_collection",
                "priority": "medium"
            })
        
        return strategy
    
    async def get_honeypot_status(self) -> Dict[str, Any]:
        """Get comprehensive status of all honeypots"""
        
        status = {
            "summary": {
                "total_honeypots": len(self.honeypots),
                "running_honeypots": 0,
                "stopped_honeypots": 0,
                "error_honeypots": 0
            },
            "honeypots": [],
            "scenarios": len(self.deception_scenarios),
            "attacker_profiles": len(self.attacker_profiles),
            "last_updated": datetime.utcnow().isoformat()
        }
        
        for honeypot_id, honeypot in self.honeypots.items():
            # Update container status if available
            if honeypot.container_id and self.docker_client:
                try:
                    container = self.docker_client.containers.get(honeypot.container_id)
                    honeypot.status = "running" if container.status == "running" else "stopped"
                except Exception:
                    honeypot.status = "error"
            
            # Count by status
            if honeypot.status == "running":
                status["summary"]["running_honeypots"] += 1
            elif honeypot.status == "stopped":
                status["summary"]["stopped_honeypots"] += 1
            else:
                status["summary"]["error_honeypots"] += 1
            
            # Add to detailed list
            status["honeypots"].append({
                "honeypot_id": honeypot_id,
                "name": honeypot.name,
                "type": honeypot.type,
                "status": honeypot.status,
                "created_at": honeypot.created_at.isoformat(),
                "ports": honeypot.ports,
                "metrics": honeypot.metrics
            })
        
        return status
    
    async def cleanup_inactive_honeypots(self, inactive_threshold_hours: int = 168) -> List[str]:
        """
        Clean up honeypots that have been inactive for too long
        
        Args:
            inactive_threshold_hours: Hours of inactivity before cleanup
            
        Returns:
            List of cleaned up honeypot IDs
        """
        try:
            cleaned_up = []
            cutoff_time = datetime.utcnow() - timedelta(hours=inactive_threshold_hours)
            
            for honeypot_id, honeypot in list(self.honeypots.items()):
                # Check if honeypot has been inactive
                if (honeypot.updated_at < cutoff_time and 
                    honeypot.metrics.get("interaction_count", 0) == 0):
                    
                    self.logger.info(f"Cleaning up inactive honeypot: {honeypot_id}")
                    
                    if await self.remove_honeypot(honeypot_id):
                        cleaned_up.append(honeypot_id)
            
            return cleaned_up
            
        except Exception as e:
            self.logger.error(f"Honeypot cleanup failed: {e}")
            return []
