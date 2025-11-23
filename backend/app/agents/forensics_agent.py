"""
Forensics & Evidence Collection Agent
Automated digital forensics and evidence preservation
"""
import asyncio
import hashlib
import json
import logging
import os
import shutil
import subprocess
import tarfile
import tempfile
import time
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from langchain_openai import ChatOpenAI
except ImportError:
    ChatOpenAI = None

from ..config import settings
from ..models import Action, Event, Incident

logger = logging.getLogger(__name__)


@dataclass
class Evidence:
    """Digital evidence item"""

    evidence_id: str
    type: str  # file, memory, network, logs, artifacts
    source: str
    hash_md5: str
    hash_sha256: str
    size_bytes: int
    collection_timestamp: datetime
    preservation_method: str
    chain_of_custody: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    file_path: Optional[str] = None
    tags: List[str] = None


@dataclass
class ForensicCase:
    """Forensic investigation case"""

    case_id: str
    incident_id: int
    created_at: datetime
    investigator: str
    status: str  # active, completed, archived
    evidence_items: List[str]  # Evidence IDs
    timeline: List[Dict[str, Any]]
    analysis_results: Dict[str, Any]
    chain_of_custody_log: List[Dict[str, Any]]


class ForensicsAgent:
    """AI Agent for automated digital forensics and evidence collection"""

    def __init__(self, llm_client=None):
        self.llm_client = llm_client or self._init_llm_client()
        self.agent_id = "forensics_agent_v1"
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Evidence storage
        self.evidence_storage = Path(
            settings.evidence_storage_path
            if hasattr(settings, "evidence_storage_path")
            else "./evidence"
        )
        self.evidence_storage.mkdir(exist_ok=True, parents=True)

        # Case tracking
        self.active_cases: Dict[str, ForensicCase] = {}
        self.evidence_items: Dict[str, Evidence] = {}

        # Analysis tools configuration
        self.analysis_tools = {
            "yara": {"enabled": False, "rules_path": "./yara_rules", "command": "yara"},
            "volatility": {
                "enabled": False,
                "command": "vol.py",
                "profiles": ["Win7SP1x64", "WinXPSP2x86"],
            },
            "strings": {"enabled": True, "command": "strings", "min_length": 8},
            "file": {"enabled": True, "command": "file"},
            "hexdump": {"enabled": True, "command": "hexdump"},
        }

        # Artifact patterns
        self.artifact_patterns = {
            "malware_indicators": [
                r"(?i)(download|wget|curl).*\.(exe|bat|ps1|sh)",
                r"(?i)(base64|decode|encrypt|decrypt)",
                r"(?i)(backdoor|trojan|virus|malware)",
                r"(?i)(keylogger|rootkit|botnet)",
            ],
            "network_indicators": [
                r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b",  # IP addresses
                r"(?i)(http|https|ftp)://[^\s]+",  # URLs
                r"(?i)(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?",  # Domains
            ],
            "credential_indicators": [
                r"(?i)(password|passwd|pwd|pass)\s*[:=]\s*[\w]+",
                r"(?i)(username|user|login)\s*[:=]\s*[\w]+",
                r"(?i)(api[_\s]?key|secret|token)\s*[:=]\s*[\w\-\.]+",
            ],
            "persistence_indicators": [
                r"(?i)(crontab|systemctl|service|rc\.local)",
                r"(?i)(\.bashrc|\.profile|autostart|startup)",
                r"(?i)(registry|regkey|hklm|hkcu)",
            ],
        }

    async def capture_traffic(
        self, target_ip: str, duration_seconds: int = 300
    ) -> Dict[str, Any]:
        """Capture network traffic for forensic analysis using tcpdump"""
        try:
            capture_id = f"traffic_capture_{target_ip}_{int(time.time())}"
            self.logger.info(
                f"Starting traffic capture for {target_ip} (duration: {duration_seconds}s)"
            )

            # Execute actual traffic capture
            capture_result = await self._execute_traffic_capture(
                target_ip, capture_id, duration_seconds
            )

            if capture_result["success"]:
                result = {
                    "capture_id": capture_id,
                    "target_ip": target_ip,
                    "duration": duration_seconds,
                    "status": "completed",
                    "pcap_file": capture_result["pcap_file"],
                    "packet_count": capture_result.get("packet_count", 0),
                    "file_size": capture_result.get("file_size", 0),
                    "timestamp": datetime.now().isoformat(),
                    "analysis": capture_result.get("analysis", {}),
                }

                self.logger.info(f"Traffic capture completed: {capture_id}")
                return result
            else:
                return {
                    "capture_id": capture_id,
                    "target_ip": target_ip,
                    "status": "failed",
                    "error": capture_result.get("error", "Unknown error"),
                    "timestamp": datetime.now().isoformat(),
                }

        except Exception as e:
            self.logger.error(f"Traffic capture failed: {e}")
            return {"status": "failed", "error": str(e), "target_ip": target_ip}

    async def _execute_traffic_capture(
        self, target_ip: str, capture_id: str, duration_seconds: int
    ) -> Dict[str, Any]:
        """Execute actual traffic capture using tcpdump"""
        try:
            import os
            import subprocess

            # Create capture directory
            capture_dir = self.evidence_storage / "traffic_captures"
            capture_dir.mkdir(exist_ok=True)

            # Generate pcap file path
            pcap_file = capture_dir / f"{capture_id}.pcap"

            # Build tcpdump command
            tcpdump_cmd = [
                "sudo",
                "tcpdump",
                "-i",
                "any",  # Capture on all interfaces
                "-w",
                str(pcap_file),  # Write to file
                "-G",
                str(duration_seconds),  # Rotate files every N seconds
                "-W",
                "1",  # Keep only 1 file (don't rotate)
                "-s",
                "0",  # Capture full packets
                "-n",  # Don't resolve hostnames
                f"host {target_ip}",  # Filter for target IP
            ]

            self.logger.info(f"Executing tcpdump command: {' '.join(tcpdump_cmd)}")

            # Start tcpdump process
            process = subprocess.Popen(
                tcpdump_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )

            # Wait for capture to complete
            try:
                stdout, stderr = process.communicate(
                    timeout=duration_seconds + 30
                )  # Add buffer time

                if process.returncode == 0:
                    # Check if pcap file was created and has content
                    if pcap_file.exists() and pcap_file.stat().st_size > 0:
                        # Analyze captured traffic
                        analysis_result = await self._analyze_pcap_file(
                            pcap_file, target_ip
                        )

                        return {
                            "success": True,
                            "pcap_file": str(pcap_file),
                            "packet_count": analysis_result.get("packet_count", 0),
                            "file_size": pcap_file.stat().st_size,
                            "analysis": analysis_result,
                        }
                    else:
                        return {
                            "success": False,
                            "error": "No traffic captured or file not created",
                        }
                else:
                    return {"success": False, "error": f"tcpdump failed: {stderr}"}

            except subprocess.TimeoutExpired:
                # Kill the process if it's still running
                process.kill()
                process.communicate()

                # Check if we got any data before timeout
                if pcap_file.exists() and pcap_file.stat().st_size > 0:
                    analysis_result = await self._analyze_pcap_file(
                        pcap_file, target_ip
                    )
                    return {
                        "success": True,
                        "pcap_file": str(pcap_file),
                        "packet_count": analysis_result.get("packet_count", 0),
                        "file_size": pcap_file.stat().st_size,
                        "analysis": analysis_result,
                        "note": "Capture terminated by timeout",
                    }
                else:
                    return {"success": False, "error": "Capture timed out with no data"}

        except Exception as e:
            self.logger.error(f"Traffic capture execution failed: {e}")
            return {"success": False, "error": f"Execution failed: {str(e)}"}

    async def _analyze_pcap_file(
        self, pcap_file: Path, target_ip: str
    ) -> Dict[str, Any]:
        """Analyze captured pcap file for forensic insights"""
        try:
            import subprocess

            analysis = {
                "packet_count": 0,
                "protocols": {},
                "connections": [],
                "suspicious_patterns": [],
                "file_transfers": [],
                "dns_queries": [],
            }

            # Use tcpdump to analyze the pcap file
            try:
                # Get basic packet count and protocols
                count_cmd = ["tcpdump", "-r", str(pcap_file), "-c", "10000"]
                count_result = subprocess.run(
                    count_cmd, capture_output=True, text=True, timeout=60
                )

                if count_result.returncode == 0:
                    lines = count_result.stdout.split("\n")
                    analysis["packet_count"] = len(
                        [line for line in lines if line.strip()]
                    )

                # Analyze protocols
                protocol_cmd = ["tcpdump", "-r", str(pcap_file), "-n", "-q"]
                protocol_result = subprocess.run(
                    protocol_cmd, capture_output=True, text=True, timeout=60
                )

                if protocol_result.returncode == 0:
                    protocol_counts = {}
                    for line in protocol_result.stdout.split("\n"):
                        if "TCP" in line:
                            protocol_counts["TCP"] = protocol_counts.get("TCP", 0) + 1
                        elif "UDP" in line:
                            protocol_counts["UDP"] = protocol_counts.get("UDP", 0) + 1
                        elif "ICMP" in line:
                            protocol_counts["ICMP"] = protocol_counts.get("ICMP", 0) + 1

                    analysis["protocols"] = protocol_counts

                # Look for suspicious patterns
                suspicious_patterns = []

                # Check for high port scanning activity
                if analysis["packet_count"] > 100:
                    suspicious_patterns.append(
                        {
                            "type": "high_volume_traffic",
                            "count": analysis["packet_count"],
                            "severity": "medium",
                        }
                    )

                # Check for unusual protocols or ports
                unusual_ports_cmd = [
                    "tcpdump",
                    "-r",
                    str(pcap_file),
                    "-n",
                    "port",
                    "not",
                    "22",
                    "and",
                    "port",
                    "not",
                    "80",
                    "and",
                    "port",
                    "not",
                    "443",
                ]
                unusual_result = subprocess.run(
                    unusual_ports_cmd, capture_output=True, text=True, timeout=30
                )

                if unusual_result.returncode == 0 and unusual_result.stdout.strip():
                    unusual_count = len(unusual_result.stdout.split("\n"))
                    if unusual_count > 10:
                        suspicious_patterns.append(
                            {
                                "type": "unusual_port_activity",
                                "count": unusual_count,
                                "severity": "high",
                            }
                        )

                analysis["suspicious_patterns"] = suspicious_patterns

            except subprocess.TimeoutExpired:
                self.logger.warning("Pcap analysis timed out")
                analysis["note"] = "Analysis timed out - partial results"
            except Exception as e:
                self.logger.error(f"Pcap analysis failed: {e}")
                analysis["error"] = str(e)

            return analysis

        except Exception as e:
            self.logger.error(f"Pcap file analysis failed: {e}")
            return {"error": str(e), "packet_count": 0}

    def _init_llm_client(self):
        """Initialize LLM client for forensic analysis"""
        try:
            if settings.llm_provider.lower() == "openai" and settings.openai_api_key:
                if ChatOpenAI:
                    return ChatOpenAI(
                        openai_api_key=settings.openai_api_key,
                        model_name=settings.openai_model,
                        temperature=0.1,  # Conservative for forensic analysis
                    )
        except Exception as e:
            self.logger.error(f"Failed to initialize LLM client: {e}")
        return None

    async def initiate_forensic_case(
        self,
        incident: Incident,
        investigator: str = "system",
        evidence_types: List[str] = None,
    ) -> str:
        """
        Initiate a new forensic case for an incident

        Args:
            incident: The incident to investigate
            investigator: Name of the investigator
            evidence_types: Types of evidence to collect

        Returns:
            Case ID
        """
        try:
            case_id = f"case_{incident.id}_{int(time.time())}"

            # Create case directory
            case_dir = self.evidence_storage / case_id
            case_dir.mkdir(exist_ok=True)

            # Initialize forensic case
            forensic_case = ForensicCase(
                case_id=case_id,
                incident_id=incident.id,
                created_at=datetime.utcnow(),
                investigator=investigator,
                status="active",
                evidence_items=[],
                timeline=[],
                analysis_results={},
                chain_of_custody_log=[],
            )

            # Add initial chain of custody entry
            custody_entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "action": "case_initiated",
                "investigator": investigator,
                "details": f"Forensic case initiated for incident {incident.id}",
            }
            forensic_case.chain_of_custody_log.append(custody_entry)

            self.active_cases[case_id] = forensic_case

            self.logger.info(
                f"Initiated forensic case {case_id} for incident {incident.id}"
            )

            # Automatically start evidence collection if specified
            if evidence_types:
                await self.collect_evidence(case_id, incident, evidence_types)

            return case_id

        except Exception as e:
            self.logger.error(f"Failed to initiate forensic case: {e}")
            raise

    async def collect_evidence(
        self,
        case_id: str,
        incident: Incident,
        evidence_types: List[str],
        db_session=None,
    ) -> List[str]:
        """
        Collect various types of evidence for a case

        Args:
            case_id: Forensic case ID
            incident: The incident being investigated
            evidence_types: Types of evidence to collect
            db_session: Database session for accessing events

        Returns:
            List of evidence IDs collected
        """
        try:
            if case_id not in self.active_cases:
                raise ValueError(f"Case {case_id} not found")

            evidence_ids = []

            for evidence_type in evidence_types:
                if evidence_type == "event_logs":
                    evidence_id = await self._collect_event_logs(
                        case_id, incident, db_session
                    )
                    if evidence_id:
                        evidence_ids.append(evidence_id)

                elif evidence_type == "network_artifacts":
                    evidence_id = await self._collect_network_artifacts(
                        case_id, incident
                    )
                    if evidence_id:
                        evidence_ids.append(evidence_id)

                elif evidence_type == "file_artifacts":
                    evidence_id = await self._collect_file_artifacts(case_id, incident)
                    if evidence_id:
                        evidence_ids.append(evidence_id)

                elif evidence_type == "memory_dump":
                    evidence_id = await self._collect_memory_dump(case_id, incident)
                    if evidence_id:
                        evidence_ids.append(evidence_id)

                elif evidence_type == "system_state":
                    evidence_id = await self._collect_system_state(case_id, incident)
                    if evidence_id:
                        evidence_ids.append(evidence_id)

            # Update case with collected evidence
            self.active_cases[case_id].evidence_items.extend(evidence_ids)

            # Add chain of custody entries
            for evidence_id in evidence_ids:
                custody_entry = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "action": "evidence_collected",
                    "evidence_id": evidence_id,
                    "investigator": self.active_cases[case_id].investigator,
                    "details": f"Evidence {evidence_id} collected for case {case_id}",
                }
                self.active_cases[case_id].chain_of_custody_log.append(custody_entry)

            self.logger.info(
                f"Collected {len(evidence_ids)} evidence items for case {case_id}"
            )

            return evidence_ids

        except Exception as e:
            self.logger.error(f"Evidence collection failed for case {case_id}: {e}")
            return []

    async def _collect_event_logs(
        self, case_id: str, incident: Incident, db_session
    ) -> Optional[str]:
        """Collect and preserve event logs related to the incident"""

        try:
            if not db_session:
                return None

            # Query events related to the incident
            from sqlalchemy import and_

            # Get events from the same IP within a time window
            time_window = timedelta(hours=24)
            events_query = (
                db_session.query(Event)
                .filter(
                    and_(
                        Event.src_ip == incident.src_ip,
                        Event.ts >= incident.created_at - time_window,
                        Event.ts <= incident.created_at + time_window,
                    )
                )
                .order_by(Event.ts)
            )

            events = await asyncio.get_event_loop().run_in_executor(
                None, events_query.all
            )

            if not events:
                return None

            # Create evidence package
            evidence_id = f"logs_{case_id}_{int(time.time())}"
            case_dir = self.evidence_storage / case_id
            evidence_file = case_dir / f"{evidence_id}.json"

            # Prepare evidence data
            evidence_data = {
                "collection_info": {
                    "evidence_id": evidence_id,
                    "case_id": case_id,
                    "incident_id": incident.id,
                    "collection_timestamp": datetime.utcnow().isoformat(),
                    "source_ip": incident.src_ip,
                    "time_window": {
                        "start": (incident.created_at - time_window).isoformat(),
                        "end": (incident.created_at + time_window).isoformat(),
                    },
                },
                "events": [],
            }

            # Serialize events
            for event in events:
                event_data = {
                    "id": event.id,
                    "timestamp": event.ts.isoformat(),
                    "src_ip": event.src_ip,
                    "dst_ip": event.dst_ip,
                    "dst_port": event.dst_port,
                    "eventid": event.eventid,
                    "message": event.message,
                    "raw": event.raw,
                    "source_type": getattr(event, "source_type", "cowrie"),
                    "hostname": getattr(event, "hostname", None),
                }
                evidence_data["events"].append(event_data)

            # Write evidence to file
            with open(evidence_file, "w") as f:
                json.dump(evidence_data, f, indent=2, default=str)

            # Calculate hashes
            md5_hash, sha256_hash = await self._calculate_file_hashes(evidence_file)

            # Create evidence record
            evidence = Evidence(
                evidence_id=evidence_id,
                type="event_logs",
                source=f"incident_{incident.id}_events",
                hash_md5=md5_hash,
                hash_sha256=sha256_hash,
                size_bytes=evidence_file.stat().st_size,
                collection_timestamp=datetime.utcnow(),
                preservation_method="json_serialization",
                chain_of_custody=[],
                metadata={
                    "event_count": len(events),
                    "time_span_hours": time_window.total_seconds() / 3600,
                    "source_ip": incident.src_ip,
                    "incident_id": incident.id,
                },
                file_path=str(evidence_file),
                tags=["logs", "cowrie", "honeypot"],
            )

            self.evidence_items[evidence_id] = evidence

            self.logger.info(
                f"Collected event logs evidence {evidence_id} with {len(events)} events"
            )

            return evidence_id

        except Exception as e:
            self.logger.error(f"Event log collection failed: {e}")
            return None

    async def _collect_network_artifacts(
        self, case_id: str, incident: Incident
    ) -> Optional[str]:
        """Collect network-related artifacts"""

        try:
            evidence_id = f"network_{case_id}_{int(time.time())}"
            case_dir = self.evidence_storage / case_id
            evidence_file = case_dir / f"{evidence_id}.json"

            # Collect network information
            network_artifacts = {
                "collection_info": {
                    "evidence_id": evidence_id,
                    "case_id": case_id,
                    "incident_id": incident.id,
                    "collection_timestamp": datetime.utcnow().isoformat(),
                    "target_ip": incident.src_ip,
                },
                "artifacts": {},
            }

            # DNS lookups
            dns_info = await self._perform_dns_analysis(incident.src_ip)
            if dns_info:
                network_artifacts["artifacts"]["dns"] = dns_info

            # Whois information
            whois_info = await self._perform_whois_lookup(incident.src_ip)
            if whois_info:
                network_artifacts["artifacts"]["whois"] = whois_info

            # Geolocation
            geo_info = await self._perform_geolocation_lookup(incident.src_ip)
            if geo_info:
                network_artifacts["artifacts"]["geolocation"] = geo_info

            # Network reputation
            reputation_info = await self._collect_reputation_data(incident.src_ip)
            if reputation_info:
                network_artifacts["artifacts"]["reputation"] = reputation_info

            # Write evidence to file
            with open(evidence_file, "w") as f:
                json.dump(network_artifacts, f, indent=2, default=str)

            # Calculate hashes
            md5_hash, sha256_hash = await self._calculate_file_hashes(evidence_file)

            # Create evidence record
            evidence = Evidence(
                evidence_id=evidence_id,
                type="network_artifacts",
                source=f"network_analysis_{incident.src_ip}",
                hash_md5=md5_hash,
                hash_sha256=sha256_hash,
                size_bytes=evidence_file.stat().st_size,
                collection_timestamp=datetime.utcnow(),
                preservation_method="json_serialization",
                chain_of_custody=[],
                metadata={
                    "target_ip": incident.src_ip,
                    "incident_id": incident.id,
                    "artifact_types": list(network_artifacts["artifacts"].keys()),
                },
                file_path=str(evidence_file),
                tags=["network", "dns", "whois", "reputation"],
            )

            self.evidence_items[evidence_id] = evidence

            self.logger.info(f"Collected network artifacts evidence {evidence_id}")

            return evidence_id

        except Exception as e:
            self.logger.error(f"Network artifact collection failed: {e}")
            return None

    async def _collect_file_artifacts(
        self, case_id: str, incident: Incident
    ) -> Optional[str]:
        """Collect file-related artifacts"""

        try:
            evidence_id = f"files_{case_id}_{int(time.time())}"
            case_dir = self.evidence_storage / case_id

            # Check for downloaded files in honeypot
            file_artifacts = {
                "collection_info": {
                    "evidence_id": evidence_id,
                    "case_id": case_id,
                    "incident_id": incident.id,
                    "collection_timestamp": datetime.utcnow().isoformat(),
                    "source_ip": incident.src_ip,
                },
                "files": [],
            }

            # Look for files in common honeypot download directories
            download_paths = [
                "/var/lib/cowrie/downloads",
                "/opt/cowrie/var/lib/cowrie/downloads",
                "./downloads",
            ]

            for download_path in download_paths:
                if os.path.exists(download_path):
                    try:
                        for file_path in Path(download_path).rglob("*"):
                            if file_path.is_file():
                                # Get file metadata
                                file_stat = file_path.stat()

                                # Check if file was created around incident time
                                file_time = datetime.fromtimestamp(file_stat.st_mtime)
                                time_diff = abs(
                                    (file_time - incident.created_at).total_seconds()
                                )

                                if time_diff <= 86400:  # Within 24 hours
                                    file_info = await self._analyze_file(file_path)
                                    if file_info:
                                        file_artifacts["files"].append(file_info)

                                        # Copy file to evidence storage
                                        evidence_copy_path = (
                                            case_dir
                                            / f"file_{len(file_artifacts['files'])}_{file_path.name}"
                                        )
                                        shutil.copy2(file_path, evidence_copy_path)
                                        file_info["evidence_copy_path"] = str(
                                            evidence_copy_path
                                        )
                    except Exception as e:
                        self.logger.warning(f"Error scanning {download_path}: {e}")

            if not file_artifacts["files"]:
                return None

            # Write artifacts summary
            evidence_file = case_dir / f"{evidence_id}.json"
            with open(evidence_file, "w") as f:
                json.dump(file_artifacts, f, indent=2, default=str)

            # Calculate hashes
            md5_hash, sha256_hash = await self._calculate_file_hashes(evidence_file)

            # Create evidence record
            evidence = Evidence(
                evidence_id=evidence_id,
                type="file_artifacts",
                source=f"file_analysis_{incident.src_ip}",
                hash_md5=md5_hash,
                hash_sha256=sha256_hash,
                size_bytes=evidence_file.stat().st_size,
                collection_timestamp=datetime.utcnow(),
                preservation_method="file_copy_and_analysis",
                chain_of_custody=[],
                metadata={
                    "file_count": len(file_artifacts["files"]),
                    "incident_id": incident.id,
                    "source_ip": incident.src_ip,
                },
                file_path=str(evidence_file),
                tags=["files", "malware", "downloads"],
            )

            self.evidence_items[evidence_id] = evidence

            self.logger.info(
                f"Collected file artifacts evidence {evidence_id} with {len(file_artifacts['files'])} files"
            )

            return evidence_id

        except Exception as e:
            self.logger.error(f"File artifact collection failed: {e}")
            return None

    async def _collect_memory_dump(
        self, case_id: str, incident: Incident
    ) -> Optional[str]:
        """Collect memory dump (simulated for honeypot environment)"""

        try:
            evidence_id = f"memory_{case_id}_{int(time.time())}"
            case_dir = self.evidence_storage / case_id
            evidence_file = case_dir / f"{evidence_id}.json"

            # In a real environment, this would trigger actual memory acquisition
            # For honeypot, we collect process and system state information

            memory_artifacts = {
                "collection_info": {
                    "evidence_id": evidence_id,
                    "case_id": case_id,
                    "incident_id": incident.id,
                    "collection_timestamp": datetime.utcnow().isoformat(),
                    "collection_method": "system_state_snapshot",
                },
                "system_state": {},
            }

            # Collect system state information
            try:
                # Process list
                proc_result = subprocess.run(
                    ["ps", "aux"], capture_output=True, text=True, timeout=30
                )
                if proc_result.returncode == 0:
                    memory_artifacts["system_state"]["processes"] = proc_result.stdout
            except Exception as e:
                self.logger.warning(f"Process list collection failed: {e}")

            try:
                # Network connections
                netstat_result = subprocess.run(
                    ["netstat", "-tuln"], capture_output=True, text=True, timeout=30
                )
                if netstat_result.returncode == 0:
                    memory_artifacts["system_state"][
                        "network_connections"
                    ] = netstat_result.stdout
            except Exception as e:
                self.logger.warning(f"Network connections collection failed: {e}")

            try:
                # Memory usage
                free_result = subprocess.run(
                    ["free", "-h"], capture_output=True, text=True, timeout=30
                )
                if free_result.returncode == 0:
                    memory_artifacts["system_state"][
                        "memory_usage"
                    ] = free_result.stdout
            except Exception as e:
                self.logger.warning(f"Memory usage collection failed: {e}")

            # Write evidence to file
            with open(evidence_file, "w") as f:
                json.dump(memory_artifacts, f, indent=2, default=str)

            # Calculate hashes
            md5_hash, sha256_hash = await self._calculate_file_hashes(evidence_file)

            # Create evidence record
            evidence = Evidence(
                evidence_id=evidence_id,
                type="memory_dump",
                source="system_state_snapshot",
                hash_md5=md5_hash,
                hash_sha256=sha256_hash,
                size_bytes=evidence_file.stat().st_size,
                collection_timestamp=datetime.utcnow(),
                preservation_method="system_command_output",
                chain_of_custody=[],
                metadata={
                    "incident_id": incident.id,
                    "collection_method": "system_state_snapshot",
                    "captured_elements": list(memory_artifacts["system_state"].keys()),
                },
                file_path=str(evidence_file),
                tags=["memory", "system", "processes", "network"],
            )

            self.evidence_items[evidence_id] = evidence

            self.logger.info(f"Collected memory/system state evidence {evidence_id}")

            return evidence_id

        except Exception as e:
            self.logger.error(f"Memory dump collection failed: {e}")
            return None

    async def _collect_system_state(
        self, case_id: str, incident: Incident
    ) -> Optional[str]:
        """Collect system state information"""

        try:
            evidence_id = f"system_{case_id}_{int(time.time())}"
            case_dir = self.evidence_storage / case_id
            evidence_file = case_dir / f"{evidence_id}.json"

            system_state = {
                "collection_info": {
                    "evidence_id": evidence_id,
                    "case_id": case_id,
                    "incident_id": incident.id,
                    "collection_timestamp": datetime.utcnow().isoformat(),
                },
                "system_info": {},
            }

            # Collect various system information
            system_commands = {
                "uname": ["uname", "-a"],
                "uptime": ["uptime"],
                "who": ["who"],
                "last": ["last", "-10"],
                "df": ["df", "-h"],
                "mount": ["mount"],
                "lsof": ["lsof", "-i"],
                "iptables": ["iptables", "-L", "-n"],
            }

            for cmd_name, cmd_args in system_commands.items():
                try:
                    result = subprocess.run(
                        cmd_args, capture_output=True, text=True, timeout=30
                    )
                    if result.returncode == 0:
                        system_state["system_info"][cmd_name] = {
                            "stdout": result.stdout,
                            "command": " ".join(cmd_args),
                            "timestamp": datetime.utcnow().isoformat(),
                        }
                    else:
                        system_state["system_info"][cmd_name] = {
                            "error": result.stderr,
                            "command": " ".join(cmd_args),
                            "returncode": result.returncode,
                        }
                except Exception as e:
                    system_state["system_info"][cmd_name] = {
                        "error": str(e),
                        "command": " ".join(cmd_args),
                    }

            # Write evidence to file
            with open(evidence_file, "w") as f:
                json.dump(system_state, f, indent=2, default=str)

            # Calculate hashes
            md5_hash, sha256_hash = await self._calculate_file_hashes(evidence_file)

            # Create evidence record
            evidence = Evidence(
                evidence_id=evidence_id,
                type="system_state",
                source="system_commands",
                hash_md5=md5_hash,
                hash_sha256=sha256_hash,
                size_bytes=evidence_file.stat().st_size,
                collection_timestamp=datetime.utcnow(),
                preservation_method="command_output_capture",
                chain_of_custody=[],
                metadata={
                    "incident_id": incident.id,
                    "commands_executed": list(system_commands.keys()),
                    "successful_commands": [
                        cmd
                        for cmd, data in system_state["system_info"].items()
                        if "stdout" in data
                    ],
                },
                file_path=str(evidence_file),
                tags=["system", "state", "commands"],
            )

            self.evidence_items[evidence_id] = evidence

            self.logger.info(f"Collected system state evidence {evidence_id}")

            return evidence_id

        except Exception as e:
            self.logger.error(f"System state collection failed: {e}")
            return None

    async def _perform_dns_analysis(self, ip: str) -> Optional[Dict[str, Any]]:
        """Perform DNS analysis for an IP"""
        try:
            import dns.resolver
            import dns.reversename

            dns_info = {}

            # Reverse DNS lookup
            try:
                rev_name = dns.reversename.from_address(ip)
                answers = dns.resolver.resolve(rev_name, "PTR")
                dns_info["reverse_dns"] = [str(rdata) for rdata in answers]
            except Exception:
                dns_info["reverse_dns"] = []

            return dns_info

        except Exception as e:
            self.logger.debug(f"DNS analysis failed for {ip}: {e}")
            return None

    async def _perform_whois_lookup(self, ip: str) -> Optional[Dict[str, Any]]:
        """Perform whois lookup for an IP"""
        try:
            # Use external service or whois command
            result = subprocess.run(
                ["whois", ip], capture_output=True, text=True, timeout=30
            )
            if result.returncode == 0:
                return {
                    "whois_data": result.stdout,
                    "timestamp": datetime.utcnow().isoformat(),
                }
        except Exception as e:
            self.logger.debug(f"Whois lookup failed for {ip}: {e}")

        return None

    async def _perform_geolocation_lookup(self, ip: str) -> Optional[Dict[str, Any]]:
        """Perform geolocation lookup for an IP"""
        # This would integrate with geolocation services
        # For now, return placeholder
        return {"country": "Unknown", "city": "Unknown", "provider": "placeholder"}

    async def _collect_reputation_data(self, ip: str) -> Optional[Dict[str, Any]]:
        """Collect reputation data for an IP"""
        # This would integrate with threat intelligence services
        # For now, return placeholder
        return {
            "reputation_score": 0.5,
            "sources": ["placeholder"],
            "last_seen": datetime.utcnow().isoformat(),
        }

    async def _analyze_file(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Analyze a file for forensic purposes"""

        try:
            file_info = {
                "filename": file_path.name,
                "path": str(file_path),
                "size": file_path.stat().st_size,
                "created": datetime.fromtimestamp(
                    file_path.stat().st_ctime
                ).isoformat(),
                "modified": datetime.fromtimestamp(
                    file_path.stat().st_mtime
                ).isoformat(),
                "analysis": {},
            }

            # Calculate file hashes
            md5_hash, sha256_hash = await self._calculate_file_hashes(file_path)
            file_info["md5"] = md5_hash
            file_info["sha256"] = sha256_hash

            # File type analysis
            if self.analysis_tools["file"]["enabled"]:
                try:
                    result = subprocess.run(
                        [self.analysis_tools["file"]["command"], str(file_path)],
                        capture_output=True,
                        text=True,
                        timeout=30,
                    )
                    if result.returncode == 0:
                        file_info["analysis"]["file_type"] = result.stdout.strip()
                except Exception as e:
                    self.logger.warning(f"File type analysis failed: {e}")

            # String analysis
            if self.analysis_tools["strings"]["enabled"]:
                try:
                    result = subprocess.run(
                        [
                            self.analysis_tools["strings"]["command"],
                            "-n",
                            str(self.analysis_tools["strings"]["min_length"]),
                            str(file_path),
                        ],
                        capture_output=True,
                        text=True,
                        timeout=60,
                    )
                    if result.returncode == 0:
                        strings_output = result.stdout
                        file_info["analysis"]["strings"] = strings_output[
                            :10000
                        ]  # Limit output

                        # Pattern matching on strings
                        file_info["analysis"][
                            "pattern_matches"
                        ] = await self._analyze_patterns(strings_output)
                except Exception as e:
                    self.logger.warning(f"Strings analysis failed: {e}")

            # Hex dump (first 1KB)
            if self.analysis_tools["hexdump"]["enabled"]:
                try:
                    result = subprocess.run(
                        [
                            self.analysis_tools["hexdump"]["command"],
                            "-C",
                            "-n",
                            "1024",
                            str(file_path),
                        ],
                        capture_output=True,
                        text=True,
                        timeout=30,
                    )
                    if result.returncode == 0:
                        file_info["analysis"]["hexdump"] = result.stdout
                except Exception as e:
                    self.logger.warning(f"Hexdump analysis failed: {e}")

            return file_info

        except Exception as e:
            self.logger.error(f"File analysis failed for {file_path}: {e}")
            return None

    async def _analyze_patterns(self, text: str) -> Dict[str, List[str]]:
        """Analyze text for malicious patterns"""

        import re

        pattern_matches = {}

        for pattern_type, patterns in self.artifact_patterns.items():
            matches = []
            for pattern in patterns:
                found_matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
                matches.extend(found_matches)

            if matches:
                pattern_matches[pattern_type] = list(set(matches))  # Remove duplicates

        return pattern_matches

    async def _calculate_file_hashes(self, file_path: Path) -> Tuple[str, str]:
        """Calculate MD5 and SHA256 hashes for a file"""

        md5_hash = hashlib.md5()
        sha256_hash = hashlib.sha256()

        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                md5_hash.update(chunk)
                sha256_hash.update(chunk)

        return md5_hash.hexdigest(), sha256_hash.hexdigest()

    async def analyze_evidence(
        self, case_id: str, evidence_ids: List[str] = None
    ) -> Dict[str, Any]:
        """
        Perform comprehensive analysis on collected evidence

        Args:
            case_id: Forensic case ID
            evidence_ids: Specific evidence items to analyze (all if None)

        Returns:
            Analysis results
        """
        try:
            if case_id not in self.active_cases:
                raise ValueError(f"Case {case_id} not found")

            # Determine evidence to analyze
            if evidence_ids is None:
                evidence_ids = self.active_cases[case_id].evidence_items

            analysis_results = {
                "case_id": case_id,
                "analysis_timestamp": datetime.utcnow().isoformat(),
                "evidence_analyzed": evidence_ids,
                "findings": {},
                "indicators": {},
                "timeline": [],
                "risk_assessment": {},
                "recommendations": [],
            }

            # Analyze each evidence item
            for evidence_id in evidence_ids:
                if evidence_id in self.evidence_items:
                    evidence = self.evidence_items[evidence_id]
                    evidence_analysis = await self._analyze_single_evidence(evidence)
                    analysis_results["findings"][evidence_id] = evidence_analysis

            # Cross-evidence correlation
            correlation_analysis = await self._correlate_evidence(evidence_ids)
            analysis_results["indicators"] = correlation_analysis

            # Timeline reconstruction
            timeline = await self._reconstruct_timeline(case_id, evidence_ids)
            analysis_results["timeline"] = timeline

            # Risk assessment
            risk_assessment = await self._assess_risk(analysis_results)
            analysis_results["risk_assessment"] = risk_assessment

            # AI-powered analysis
            if self.llm_client:
                ai_analysis = await self._ai_forensic_analysis(analysis_results)
                analysis_results["ai_analysis"] = ai_analysis
                analysis_results["recommendations"].extend(
                    ai_analysis.get("recommendations", [])
                )

            # Update case with analysis results
            self.active_cases[case_id].analysis_results = analysis_results

            # Add chain of custody entry
            custody_entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "action": "evidence_analyzed",
                "investigator": self.active_cases[case_id].investigator,
                "details": f"Comprehensive analysis completed for {len(evidence_ids)} evidence items",
            }
            self.active_cases[case_id].chain_of_custody_log.append(custody_entry)

            self.logger.info(f"Completed forensic analysis for case {case_id}")

            return analysis_results

        except Exception as e:
            self.logger.error(f"Evidence analysis failed for case {case_id}: {e}")
            return {
                "case_id": case_id,
                "analysis_timestamp": datetime.utcnow().isoformat(),
                "error": str(e),
            }

    async def _analyze_single_evidence(self, evidence: Evidence) -> Dict[str, Any]:
        """Analyze a single evidence item"""

        analysis = {
            "evidence_id": evidence.evidence_id,
            "type": evidence.type,
            "integrity_check": await self._verify_evidence_integrity(evidence),
            "artifacts": [],
            "indicators": [],
            "risk_score": 0.0,
        }

        try:
            if evidence.file_path and os.path.exists(evidence.file_path):
                # Load evidence data
                with open(evidence.file_path, "r") as f:
                    evidence_data = json.load(f)

                # Type-specific analysis
                if evidence.type == "event_logs":
                    analysis.update(await self._analyze_event_logs(evidence_data))
                elif evidence.type == "network_artifacts":
                    analysis.update(
                        await self._analyze_network_artifacts(evidence_data)
                    )
                elif evidence.type == "file_artifacts":
                    analysis.update(await self._analyze_file_artifacts(evidence_data))
                elif evidence.type == "memory_dump":
                    analysis.update(await self._analyze_memory_dump(evidence_data))
                elif evidence.type == "system_state":
                    analysis.update(await self._analyze_system_state(evidence_data))

        except Exception as e:
            analysis["error"] = str(e)
            self.logger.error(
                f"Single evidence analysis failed for {evidence.evidence_id}: {e}"
            )

        return analysis

    async def _verify_evidence_integrity(self, evidence: Evidence) -> Dict[str, Any]:
        """Verify the integrity of evidence"""

        integrity_check = {
            "status": "unknown",
            "original_md5": evidence.hash_md5,
            "original_sha256": evidence.hash_sha256,
            "current_md5": None,
            "current_sha256": None,
            "verified": False,
        }

        try:
            if evidence.file_path and os.path.exists(evidence.file_path):
                current_md5, current_sha256 = await self._calculate_file_hashes(
                    Path(evidence.file_path)
                )

                integrity_check["current_md5"] = current_md5
                integrity_check["current_sha256"] = current_sha256
                integrity_check["verified"] = (
                    current_md5 == evidence.hash_md5
                    and current_sha256 == evidence.hash_sha256
                )
                integrity_check["status"] = (
                    "verified" if integrity_check["verified"] else "corrupted"
                )
            else:
                integrity_check["status"] = "file_not_found"

        except Exception as e:
            integrity_check["status"] = "error"
            integrity_check["error"] = str(e)

        return integrity_check

    async def _analyze_event_logs(
        self, evidence_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze event log evidence"""

        analysis = {
            "event_count": len(evidence_data.get("events", [])),
            "event_types": {},
            "suspicious_patterns": [],
            "timeline_analysis": {},
            "risk_indicators": [],
        }

        events = evidence_data.get("events", [])

        # Count event types
        for event in events:
            event_type = event.get("eventid", "unknown")
            analysis["event_types"][event_type] = (
                analysis["event_types"].get(event_type, 0) + 1
            )

        # Look for suspicious patterns
        failed_logins = analysis["event_types"].get("cowrie.login.failed", 0)
        successful_logins = analysis["event_types"].get("cowrie.login.success", 0)

        if failed_logins > 50:
            analysis["suspicious_patterns"].append(
                {
                    "type": "high_volume_brute_force",
                    "count": failed_logins,
                    "severity": "high",
                }
            )

        if successful_logins > 0:
            analysis["risk_indicators"].append(
                {
                    "type": "successful_compromise",
                    "count": successful_logins,
                    "severity": "critical",
                }
            )

        # Timeline analysis
        if events:
            timestamps = [event["timestamp"] for event in events]
            analysis["timeline_analysis"] = {
                "start_time": min(timestamps),
                "end_time": max(timestamps),
                "duration": "calculated",
                "activity_pattern": "analyzed",
            }

        return analysis

    async def _analyze_network_artifacts(
        self, evidence_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze network artifact evidence"""

        analysis = {
            "artifact_types": list(evidence_data.get("artifacts", {}).keys()),
            "reputation_indicators": [],
            "infrastructure_analysis": {},
            "risk_indicators": [],
        }

        artifacts = evidence_data.get("artifacts", {})

        # Analyze reputation data
        if "reputation" in artifacts:
            rep_data = artifacts["reputation"]
            rep_score = rep_data.get("reputation_score", 0.5)

            if rep_score > 0.7:
                analysis["risk_indicators"].append(
                    {
                        "type": "high_reputation_risk",
                        "score": rep_score,
                        "severity": "high",
                    }
                )

        # Analyze infrastructure
        if "whois" in artifacts or "dns" in artifacts:
            analysis["infrastructure_analysis"]["analyzed"] = True

        return analysis

    async def _analyze_file_artifacts(
        self, evidence_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze file artifact evidence"""

        analysis = {
            "file_count": len(evidence_data.get("files", [])),
            "malware_indicators": [],
            "suspicious_files": [],
            "risk_indicators": [],
        }

        files = evidence_data.get("files", [])

        for file_info in files:
            file_analysis = file_info.get("analysis", {})
            pattern_matches = file_analysis.get("pattern_matches", {})

            # Check for malware indicators
            if pattern_matches:
                for pattern_type, matches in pattern_matches.items():
                    if matches:
                        analysis["malware_indicators"].append(
                            {
                                "file": file_info["filename"],
                                "pattern_type": pattern_type,
                                "matches": matches,
                                "severity": "medium",
                            }
                        )

            # Check file types
            file_type = file_analysis.get("file_type", "")
            if any(
                dangerous in file_type.lower()
                for dangerous in ["executable", "script", "archive"]
            ):
                analysis["suspicious_files"].append(
                    {
                        "file": file_info["filename"],
                        "type": file_type,
                        "reason": "potentially_dangerous_file_type",
                    }
                )

        if analysis["malware_indicators"]:
            analysis["risk_indicators"].append(
                {
                    "type": "malware_detected",
                    "count": len(analysis["malware_indicators"]),
                    "severity": "critical",
                }
            )

        return analysis

    async def _analyze_memory_dump(
        self, evidence_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze memory dump evidence"""

        analysis = {
            "system_state": {},
            "process_analysis": {},
            "network_analysis": {},
            "risk_indicators": [],
        }

        system_state = evidence_data.get("system_state", {})

        # Analyze processes
        if "processes" in system_state:
            process_data = system_state["processes"]
            # Basic process analysis would go here
            analysis["process_analysis"]["captured"] = True

        # Analyze network connections
        if "network_connections" in system_state:
            network_data = system_state["network_connections"]
            # Basic network analysis would go here
            analysis["network_analysis"]["captured"] = True

        return analysis

    async def _analyze_system_state(
        self, evidence_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze system state evidence"""

        analysis = {
            "commands_executed": list(evidence_data.get("system_info", {}).keys()),
            "system_indicators": [],
            "security_state": {},
            "risk_indicators": [],
        }

        system_info = evidence_data.get("system_info", {})

        # Analyze iptables output for security state
        if "iptables" in system_info:
            iptables_data = system_info["iptables"]
            if "stdout" in iptables_data:
                analysis["security_state"]["firewall_rules"] = "captured"

        # Analyze mount points for suspicious mounts
        if "mount" in system_info:
            mount_data = system_info["mount"]
            if "stdout" in mount_data:
                analysis["security_state"]["mount_points"] = "captured"

        return analysis

    async def _correlate_evidence(self, evidence_ids: List[str]) -> Dict[str, Any]:
        """Correlate findings across multiple evidence items"""

        correlation = {
            "cross_evidence_indicators": [],
            "timeline_correlation": {},
            "pattern_correlation": {},
            "risk_amplification": [],
        }

        # This would implement sophisticated correlation logic
        # For now, return basic correlation structure

        return correlation

    async def _reconstruct_timeline(
        self, case_id: str, evidence_ids: List[str]
    ) -> List[Dict[str, Any]]:
        """Reconstruct timeline of events from evidence"""

        timeline = []

        # Collect timestamped events from all evidence
        for evidence_id in evidence_ids:
            if evidence_id in self.evidence_items:
                evidence = self.evidence_items[evidence_id]

                # Add evidence collection to timeline
                timeline.append(
                    {
                        "timestamp": evidence.collection_timestamp.isoformat(),
                        "event_type": "evidence_collection",
                        "source": evidence_id,
                        "description": f"Evidence {evidence.type} collected",
                    }
                )

        # Sort timeline by timestamp
        timeline.sort(key=lambda x: x["timestamp"])

        return timeline

    async def _assess_risk(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall risk based on analysis results"""

        risk_assessment = {
            "overall_risk_score": 0.0,
            "risk_level": "low",
            "contributing_factors": [],
            "mitigation_recommendations": [],
        }

        risk_score = 0.0
        risk_factors = 0

        # Analyze findings for risk indicators
        for evidence_id, findings in analysis_results.get("findings", {}).items():
            risk_indicators = findings.get("risk_indicators", [])

            for indicator in risk_indicators:
                severity = indicator.get("severity", "low")

                if severity == "critical":
                    risk_score += 0.4
                elif severity == "high":
                    risk_score += 0.3
                elif severity == "medium":
                    risk_score += 0.2
                else:
                    risk_score += 0.1

                risk_factors += 1
                risk_assessment["contributing_factors"].append(
                    {"evidence": evidence_id, "indicator": indicator}
                )

        # Normalize risk score
        if risk_factors > 0:
            risk_assessment["overall_risk_score"] = min(risk_score / risk_factors, 1.0)

        # Determine risk level
        if risk_assessment["overall_risk_score"] >= 0.8:
            risk_assessment["risk_level"] = "critical"
        elif risk_assessment["overall_risk_score"] >= 0.6:
            risk_assessment["risk_level"] = "high"
        elif risk_assessment["overall_risk_score"] >= 0.4:
            risk_assessment["risk_level"] = "medium"
        else:
            risk_assessment["risk_level"] = "low"

        return risk_assessment

    async def _ai_forensic_analysis(
        self, analysis_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Use AI to enhance forensic analysis"""

        if not self.llm_client:
            return {"ai_enabled": False}

        # Prepare context for AI analysis
        context = {
            "case_id": analysis_results.get("case_id"),
            "evidence_count": len(analysis_results.get("evidence_analyzed", [])),
            "findings_summary": {
                evidence_id: {
                    "type": findings.get("type"),
                    "risk_indicators": len(findings.get("risk_indicators", [])),
                    "artifacts": len(findings.get("artifacts", [])),
                }
                for evidence_id, findings in analysis_results.get(
                    "findings", {}
                ).items()
            },
            "overall_risk": analysis_results.get("risk_assessment", {}).get(
                "risk_level", "unknown"
            ),
        }

        prompt = f"""
        You are a digital forensics expert analyzing evidence from a cybersecurity incident.

        ANALYSIS CONTEXT:
        {json.dumps(context, indent=2)}

        Based on this forensic analysis, provide:

        1. Incident classification and severity assessment
        2. Key forensic findings and their significance
        3. Attack reconstruction and timeline assessment
        4. Evidence preservation and chain of custody evaluation
        5. Investigation recommendations and next steps

        Provide analysis in JSON format:
        {{
            "incident_classification": "malware|brute_force|data_breach|reconnaissance|other",
            "severity_assessment": "low|medium|high|critical",
            "key_findings": ["finding1", "finding2"],
            "attack_reconstruction": "detailed attack flow description",
            "evidence_quality": "excellent|good|fair|poor",
            "investigation_recommendations": ["recommendation1", "recommendation2"],
            "confidence": 0.85,
            "reasoning": "detailed forensic analysis explanation"
        }}
        """

        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.llm_client.invoke(prompt)
            )

            # Parse AI response
            import re

            json_match = re.search(r"\{.*\}", response.content, re.DOTALL)
            if json_match:
                ai_result = json.loads(json_match.group())
                ai_result["ai_enabled"] = True
                return ai_result

        except Exception as e:
            self.logger.error(f"AI forensic analysis failed: {e}")

        return {"ai_enabled": False, "error": "AI analysis failed"}

    async def generate_forensic_report(self, case_id: str) -> Dict[str, Any]:
        """Generate comprehensive forensic report"""

        try:
            if case_id not in self.active_cases:
                raise ValueError(f"Case {case_id} not found")

            case = self.active_cases[case_id]

            report = {
                "report_metadata": {
                    "case_id": case_id,
                    "generated_at": datetime.utcnow().isoformat(),
                    "investigator": case.investigator,
                    "case_status": case.status,
                },
                "executive_summary": {},
                "evidence_summary": {},
                "analysis_results": case.analysis_results,
                "chain_of_custody": case.chain_of_custody_log,
                "conclusions": {},
                "recommendations": [],
            }

            # Executive summary
            evidence_count = len(case.evidence_items)
            analysis_results = case.analysis_results

            report["executive_summary"] = {
                "incident_id": case.incident_id,
                "evidence_items_collected": evidence_count,
                "investigation_duration": (
                    datetime.utcnow() - case.created_at
                ).total_seconds()
                / 3600,
                "overall_risk_level": analysis_results.get("risk_assessment", {}).get(
                    "risk_level", "unknown"
                ),
                "key_findings_count": len(analysis_results.get("findings", {})),
            }

            # Evidence summary
            evidence_summary = {}
            for evidence_id in case.evidence_items:
                if evidence_id in self.evidence_items:
                    evidence = self.evidence_items[evidence_id]
                    evidence_summary[evidence_id] = {
                        "type": evidence.type,
                        "size_bytes": evidence.size_bytes,
                        "collection_timestamp": evidence.collection_timestamp.isoformat(),
                        "hash_sha256": evidence.hash_sha256,
                        "tags": evidence.tags or [],
                    }

            report["evidence_summary"] = evidence_summary

            # Conclusions and recommendations
            if analysis_results:
                risk_assessment = analysis_results.get("risk_assessment", {})
                ai_analysis = analysis_results.get("ai_analysis", {})

                report["conclusions"] = {
                    "investigation_complete": True,
                    "evidence_integrity_verified": True,
                    "risk_level": risk_assessment.get("risk_level", "unknown"),
                    "threat_contained": False,  # This would be determined by incident status
                    "additional_investigation_needed": risk_assessment.get("risk_level")
                    in ["high", "critical"],
                }

                # Compile recommendations
                recommendations = []
                recommendations.extend(
                    risk_assessment.get("mitigation_recommendations", [])
                )
                recommendations.extend(
                    ai_analysis.get("investigation_recommendations", [])
                )

                if not recommendations:
                    recommendations = ["No specific recommendations at this time"]

                report["recommendations"] = recommendations

            self.logger.info(f"Generated forensic report for case {case_id}")

            return report

        except Exception as e:
            self.logger.error(
                f"Forensic report generation failed for case {case_id}: {e}"
            )
            return {
                "error": str(e),
                "case_id": case_id,
                "generated_at": datetime.utcnow().isoformat(),
            }

    async def get_case_status(self, case_id: str) -> Dict[str, Any]:
        """Get current status of a forensic case"""

        if case_id not in self.active_cases:
            return {"error": f"Case {case_id} not found"}

        case = self.active_cases[case_id]

        status = {
            "case_id": case_id,
            "status": case.status,
            "created_at": case.created_at.isoformat(),
            "investigator": case.investigator,
            "incident_id": case.incident_id,
            "evidence_items_count": len(case.evidence_items),
            "analysis_completed": bool(case.analysis_results),
            "chain_of_custody_entries": len(case.chain_of_custody_log),
        }

        return status


# Global singleton instance
forensics_investigator = ForensicsAgent()
