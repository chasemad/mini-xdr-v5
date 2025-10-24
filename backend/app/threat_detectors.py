"""Specialized threat detectors for non-SSH attack types.

These detectors complement the adaptive detection engine by providing
heuristics for threats that are hard to capture with the existing
rule-based or ML pipelines (e.g., cryptomining, ransomware).
"""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .models import Event

logger = logging.getLogger(__name__)


def _normalize_raw(event: Event) -> Dict[str, Any]:
    """Ensure event.raw is a dictionary we can safely inspect."""
    if isinstance(event.raw, dict):
        return event.raw
    if isinstance(event.raw, str):
        try:
            return json.loads(event.raw)
        except json.JSONDecodeError:
            return {}
    return {}


@dataclass
class SpecializedDetectionResult:
    """Result returned by a specialized detector."""

    category: str
    severity: str
    confidence: float
    description: str
    indicators: Dict[str, Any]


class CryptominingDetector:
    """Detects signs of illicit cryptomining activity."""

    command_markers = [
        "xmrig",
        "minerd",
        "cpuminer",
        "cryptonight",
        "stratum+tcp",
        "ethminer",
        "nicehash",
    ]

    suspicious_ports = {3333, 4444, 5555, 7777}

    async def detect(self, events: List[Event]) -> Optional[SpecializedDetectionResult]:
        command_hits: List[str] = []
        pool_connections: List[Dict[str, Any]] = []

        for event in events:
            raw = _normalize_raw(event)
            message = (event.message or "").lower()

            if event.eventid == "cowrie.command.input":
                candidate = "".join(str(raw.get("input", "")).lower())
                if any(marker in candidate for marker in self.command_markers):
                    command_hits.append(candidate)

            if raw:
                process_name = str(raw.get("process_name", "")).lower()
                if any(marker in process_name for marker in self.command_markers):
                    command_hits.append(process_name)

            if event.dst_port in self.suspicious_ports or raw.get("remote_port") in self.suspicious_ports:
                pool_connections.append({
                    "dst_ip": raw.get("remote_address") or event.dst_ip,
                    "dst_port": raw.get("remote_port") or event.dst_port,
                })

            if any(marker in message for marker in self.command_markers):
                command_hits.append(message)

        if command_hits or len(pool_connections) >= 3:
            indicators: Dict[str, Any] = {
                "command_hits": command_hits[:10],
                "pool_connections": pool_connections[:10],
                "evidence_count": len(command_hits) + len(pool_connections),
            }
            return SpecializedDetectionResult(
                category="cryptomining",
                severity="high" if len(command_hits) >= 2 else "medium",
                confidence=min(1.0, 0.4 + 0.1 * len(command_hits) + 0.15 * len(pool_connections)),
                description="Indicators of unauthorized cryptomining activity detected",
                indicators=indicators,
            )

        return None


class DataExfiltrationDetector:
    """Detects potential data exfiltration behaviour."""

    volume_threshold_bytes = 25 * 1024 * 1024  # 25 MB in aggregate
    burst_threshold_bytes = 75 * 1024 * 1024  # 75 MB indicates aggressive exfil

    async def detect(self, events: List[Event]) -> Optional[SpecializedDetectionResult]:
        total_bytes = 0
        upload_events: List[Dict[str, Any]] = []
        suspicious_commands: List[str] = []
        exfil_domains: Dict[str, int] = {}

        for event in events:
            raw = _normalize_raw(event)

            bytes_sent = raw.get("bytes_sent") or raw.get("size") or raw.get("data_length")
            if isinstance(bytes_sent, (int, float)):
                total_bytes += bytes_sent

            if event.eventid in {"cowrie.session.file_upload", "http.file_upload"}:
                upload_events.append({
                    "filename": raw.get("filename"),
                    "size": bytes_sent,
                })
                if raw.get("remote_address"):
                    exfil_domains[str(raw.get("remote_address"))] = exfil_domains.get(str(raw.get("remote_address")), 0) + 1

            if event.eventid == "cowrie.session.file_download":
                url = str(raw.get("url", ""))
                if "paste" in url or "dropbox" in url or "file.io" in url:
                    suspicious_commands.append(f"remote_download:{url}")

            if event.eventid == "cowrie.command.input":
                command = str(raw.get("input", "")).lower()
                if any(marker in command for marker in ["scp", "rsync", "curl -t", "ftp", "invoke-webrequest", "powershell -enc", "curl --upload"]):
                    suspicious_commands.append(command)
                if "base64" in command and "decode" in command:
                    suspicious_commands.append(f"encoded_transfer:{command[:80]}")

        if total_bytes >= self.volume_threshold_bytes or upload_events or suspicious_commands:
            indicators: Dict[str, Any] = {
                "approx_bytes_exfiltrated": total_bytes,
                "upload_events": upload_events[:10],
                "suspicious_commands": suspicious_commands[:10],
                "unique_destinations": len(exfil_domains),
            }
            severity = "high" if total_bytes >= self.volume_threshold_bytes else "medium"
            confidence = 0.55
            if total_bytes >= self.volume_threshold_bytes:
                confidence += 0.2
            if total_bytes >= self.burst_threshold_bytes:
                confidence += 0.15
                severity = "critical"
            if upload_events:
                confidence += min(0.2, 0.05 * len(upload_events))
            if suspicious_commands:
                confidence += min(0.2, 0.05 * len(suspicious_commands))
            if len(exfil_domains) >= 3:
                confidence += 0.1

            return SpecializedDetectionResult(
                category="data_exfiltration",
                severity=severity,
                confidence=min(confidence, 1.0),
                description="Potential data exfiltration activity detected",
                indicators=indicators,
            )

        return None


class RansomwareDetector:
    """Detects ransomware preparation and execution behaviours."""

    command_markers = [
        "vssadmin delete",
        "wbadmin delete",
        "bcdedit",
        "cipher /w",
        "shadowcopy",
        "reg add",
        "schtasks",
        "icacls",
        "takeown",
        "wmic shadowcopy delete",
        "powershell -enc",
    ]

    ransom_keywords = [
        "how_to_decrypt",
        "decrypt",
        "ransom",
        "help_restore",
        "recover_files",
        "bitcoin",
        "pay_ransom",
    ]
    ransom_extensions = (".lock", ".crypted", ".ransom", ".payme", ".encrypted")

    async def detect(self, events: List[Event]) -> Optional[SpecializedDetectionResult]:
        destructive_commands: List[str] = []
        ransom_files: List[str] = []
        process_hits: List[str] = []
        dropped_notes: List[str] = []

        for event in events:
            raw = _normalize_raw(event)
            message = (event.message or "").lower()

            if event.eventid == "cowrie.command.input":
                command = str(raw.get("input", "")).lower()
                if any(marker in command for marker in self.command_markers):
                    destructive_commands.append(command)
                if "wallpaper" in command or "set-mp" in command:
                    process_hits.append(command)

            filename = str(raw.get("filename", "")).lower()
            if any(keyword in filename for keyword in self.ransom_keywords):
                ransom_files.append(filename)
            if filename.endswith(self.ransom_extensions):
                dropped_notes.append(filename)

            if any(keyword in message for keyword in self.ransom_keywords):
                ransom_files.append(message)
            if "note" in filename and "readme" in filename:
                dropped_notes.append(filename)

        if destructive_commands or ransom_files or process_hits or dropped_notes:
            indicators = {
                "destructive_commands": destructive_commands[:10],
                "ransom_artifacts": ransom_files[:10],
                "process_artifacts": process_hits[:10],
                "dropped_notes": dropped_notes[:10],
            }
            base_confidence = 0.6 if destructive_commands else 0.55
            confidence = base_confidence
            confidence += 0.12 * min(len(destructive_commands), 3)
            confidence += 0.08 * min(len(ransom_files), 4)
            confidence += 0.05 * len(process_hits)
            confidence += 0.07 * len(dropped_notes)

            severity = "critical" if len(destructive_commands) >= 2 or dropped_notes else "high"
            if len(destructive_commands) >= 3 or (destructive_commands and dropped_notes):
                severity = "critical"
                confidence = max(confidence, 0.85)

            return SpecializedDetectionResult(
                category="ransomware",
                severity=severity,
                confidence=min(confidence, 1.0),
                description="Ransomware behaviour detected (shadow copy deletion or ransom notes)",
                indicators=indicators,
            )

        return None


class IoTBotnetDetector:
    """Detects IoT botnet activity (e.g., Mirai)."""

    default_creds = {
        ("root", "xc3511"),
        ("admin", "admin"),
        ("admin", "smcadmin"),
        ("root", "vizxv"),
        ("root", "antslq"),
    }

    command_markers = [
        "/bin/busybox", "tftp", "wget", "ftpget", "mirror", "/bin/mirai",
        "/tmp/mirai", "ECCHI", "lolol", "*hellcat*",
    ]

    async def detect(self, events: List[Event]) -> Optional[SpecializedDetectionResult]:
        default_credential_hits = 0
        botnet_commands: List[str] = []

        for event in events:
            raw = _normalize_raw(event)

            if event.eventid == "cowrie.login.failed":
                username = str(raw.get("username", "")).lower()
                password = str(raw.get("password", "")).lower()
                if (username, password) in self.default_creds:
                    default_credential_hits += 1

            if event.eventid == "cowrie.command.input":
                command = str(raw.get("input", "")).lower()
                if any(marker in command for marker in self.command_markers):
                    botnet_commands.append(command)

        if default_credential_hits >= 3 or botnet_commands:
            indicators = {
                "default_credentials_attempts": default_credential_hits,
                "botnet_commands": botnet_commands[:10],
            }
            confidence = min(1.0, 0.4 + 0.1 * default_credential_hits + 0.15 * len(botnet_commands))
            severity = "high" if default_credential_hits >= 3 else "medium"

            return SpecializedDetectionResult(
                category="iot_botnet",
                severity=severity,
                confidence=confidence,
                description="Indicators of IoT botnet recruitment detected",
                indicators=indicators,
            )

        return None


class DDoSMitigator:
    """Suggests mitigation actions for suspected DDoS attacks."""

    def suggest_actions(self, events: List[Event]) -> List[str]:
        unique_dst_ports = {event.dst_port for event in events if event.dst_port}
        unique_dst_ports.discard(None)

        actions = [
            "Apply temporary rate limiting on offending IP",
            "Enable SYN cookies / connection limiting on perimeter firewall",
            "Engage upstream provider to filter volumetric traffic",
        ]

        if unique_dst_ports and len(unique_dst_ports) > 1:
            actions.append("Block high-risk service ports targeted during the flood")
        else:
            actions.append("Sinkhole traffic destined for affected service")

        actions.append("Capture sample PCAP for forensic analysis")
        return actions


class SpecializedThreatDetector:
    """Coordinator that aggregates specialized heuristics."""

    def __init__(self):
        self.cryptomining = CryptominingDetector()
        self.data_exfil = DataExfiltrationDetector()
        self.ransomware = RansomwareDetector()
        self.iot_botnet = IoTBotnetDetector()
        self.ddos_mitigator = DDoSMitigator()

    async def evaluate(self, events: List[Event]) -> List[SpecializedDetectionResult]:
        if not events:
            return []

        detectors = [
            self.cryptomining.detect,
            self.data_exfil.detect,
            self.ransomware.detect,
            self.iot_botnet.detect,
        ]

        results: List[SpecializedDetectionResult] = []

        detection_tasks = [detector(events) for detector in detectors]
        detection_outputs = await asyncio.gather(*detection_tasks, return_exceptions=True)

        for output in detection_outputs:
            if isinstance(output, SpecializedDetectionResult):
                results.append(output)
            elif isinstance(output, Exception):
                logger.error("Specialized detector failed: %s", output)

        return results

    def get_ddos_actions(self, events: List[Event]) -> List[str]:
        return self.ddos_mitigator.suggest_actions(events)


# Global instance used by the detection pipeline
specialized_detector = SpecializedThreatDetector()
