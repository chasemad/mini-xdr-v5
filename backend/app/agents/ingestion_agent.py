"""
Ingestion Agent for Edge Deployment
Collects and forwards logs from remote honeypots and endpoints
"""
import argparse
import asyncio
import json
import logging
import os
import signal
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiofiles
import aiohttp

from .hmac_signer import build_hmac_headers, canonicalize_payload, sign_event

# This can be run as a standalone script on edge devices


@dataclass
class AgentConfig:
    """Configuration for the ingestion agent"""

    # Required fields (no defaults) - must come first
    backend_url: str
    source_type: str
    hostname: str
    log_paths: Dict[str, str]
    # Optional fields (with defaults) - must come after required fields
    api_key: Optional[str] = None
    batch_size: int = 50
    flush_interval: int = 30
    max_retries: int = 3
    retry_delay: int = 5
    validate_ssl: bool = True
    compress_data: bool = True
    device_id: Optional[str] = None
    hmac_key: Optional[str] = None
    device_id_env: Optional[str] = None
    hmac_key_env: Optional[str] = None


class LogTailer:
    """Asynchronous log file tailer"""

    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        self.position = 0
        self.last_inode = None
        self.logger = logging.getLogger(f"{__name__}.LogTailer")

    async def tail(self):
        """Async generator that yields new log lines"""
        while True:
            try:
                if not self.file_path.exists():
                    await asyncio.sleep(1)
                    continue

                # Check if file was rotated
                current_inode = self.file_path.stat().st_ino
                if self.last_inode and current_inode != self.last_inode:
                    self.position = 0  # File was rotated, start from beginning
                self.last_inode = current_inode

                # Read new lines
                async with aiofiles.open(self.file_path, "r") as f:
                    await f.seek(self.position)

                    while True:
                        line = await f.readline()
                        if not line:
                            break

                        self.position = await f.tell()
                        yield line.strip()

                # Wait before checking for more lines
                await asyncio.sleep(0.1)

            except Exception as e:
                self.logger.error(f"Error tailing {self.file_path}: {e}")
                await asyncio.sleep(5)


class IngestionAgent:
    """Edge agent for collecting and pushing logs to Mini-XDR backend"""

    def __init__(self, config: AgentConfig):
        self.config = config
        self.session = None
        self.logger = logging.getLogger(__name__)
        self.event_buffer = []
        self.running = False
        self.stats = {
            "events_collected": 0,
            "events_sent": 0,
            "events_failed": 0,
            "last_flush": None,
            "start_time": time.time(),
        }

        self.device_id = self._resolve_secret(
            self.config.device_id,
            self.config.device_id_env,
            "device ID",
            default_env="MINIXDR_AGENT_DEVICE_ID",
        )
        self.hmac_key = self._resolve_secret(
            self.config.hmac_key,
            self.config.hmac_key_env,
            "HMAC key",
            default_env="MINIXDR_AGENT_HMAC_KEY",
        )

        if not self.config.api_key:
            self.config.api_key = os.getenv("MINIXDR_AGENT_BEARER") or os.getenv(
                "API_KEY"
            )

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _resolve_secret(
        self,
        value: Optional[str],
        env_var: Optional[str],
        label: str,
        *,
        default_env: Optional[str] = None,
    ) -> str:
        """Resolve sensitive config either from direct value or environment."""
        if value:
            return value
        candidates = []
        if env_var:
            candidates.append(env_var)
        if default_env:
            candidates.append(default_env)
        for candidate in candidates:
            env_value = os.getenv(candidate)
            if env_value:
                return env_value
        raise ValueError(f"Agent {label} is required for authenticated requests")

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.info(f"Received signal {signum}, shutting down...")
        self.running = False

    async def start(self):
        """Start the ingestion agent"""
        self.running = True
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            connector=aiohttp.TCPConnector(verify_ssl=self.config.validate_ssl),
        )

        self.logger.info(
            f"Starting ingestion agent for {self.config.source_type}:{self.config.hostname}"
        )

        # Start collection tasks for each log path
        tasks = []

        for source_name, log_path in self.config.log_paths.items():
            task = asyncio.create_task(
                self._collect_logs(source_name, log_path), name=f"collect_{source_name}"
            )
            tasks.append(task)

        # Start periodic flush task
        flush_task = asyncio.create_task(self._periodic_flush(), name="flush")
        tasks.append(flush_task)

        # Start stats reporting task
        stats_task = asyncio.create_task(self._periodic_stats(), name="stats")
        tasks.append(stats_task)

        try:
            # Wait for all tasks
            await asyncio.gather(*tasks, return_exceptions=True)
        finally:
            await self._cleanup()

    async def _collect_logs(self, source_name: str, log_path: str):
        """Collect logs from a specific file"""
        self.logger.info(f"Starting collection from {log_path}")

        tailer = LogTailer(log_path)

        async for line in tailer.tail():
            if not self.running:
                break

            try:
                # Parse log line based on source type
                event = await self._parse_log_line(line, source_name)
                if event:
                    await self._buffer_event(event)

            except Exception as e:
                self.logger.error(f"Error processing log line from {log_path}: {e}")

    async def _parse_log_line(
        self, line: str, source_name: str
    ) -> Optional[Dict[str, Any]]:
        """Parse a log line based on source type"""
        if not line.strip():
            return None

        try:
            if self.config.source_type == "cowrie":
                return self._parse_cowrie_line(line)
            elif self.config.source_type == "suricata":
                return self._parse_suricata_line(line)
            elif self.config.source_type == "syslog":
                return self._parse_syslog_line(line)
            else:
                # Try to parse as JSON
                return json.loads(line)

        except json.JSONDecodeError:
            # If not JSON, create a basic event
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "message": line,
                "source": source_name,
                "raw_line": line,
            }
        except Exception as e:
            self.logger.error(f"Failed to parse line: {e}")
            return None

    def _parse_cowrie_line(self, line: str) -> Dict[str, Any]:
        """Parse Cowrie JSON log line"""
        event = json.loads(line)

        # Add agent metadata
        event["agent_hostname"] = self.config.hostname
        event["agent_timestamp"] = time.time()

        return event

    def _parse_suricata_line(self, line: str) -> Dict[str, Any]:
        """Parse Suricata EVE JSON log line"""
        event = json.loads(line)

        # Add agent metadata
        event["agent_hostname"] = self.config.hostname
        event["agent_timestamp"] = time.time()

        return event

    def _parse_syslog_line(self, line: str) -> Dict[str, Any]:
        """Parse syslog line (basic parsing)"""
        # Basic syslog parsing - in production you'd use proper syslog parser
        parts = line.split(" ", 5)

        event = {
            "timestamp": " ".join(parts[:3])
            if len(parts) >= 3
            else datetime.utcnow().isoformat(),
            "hostname": parts[3] if len(parts) >= 4 else "unknown",
            "service": parts[4] if len(parts) >= 5 else "unknown",
            "message": parts[5] if len(parts) >= 6 else line,
            "raw_line": line,
            "agent_hostname": self.config.hostname,
            "agent_timestamp": time.time(),
        }

        return event

    async def _buffer_event(self, event: Dict[str, Any]):
        """Add event to buffer and flush if needed"""
        # Add signature for integrity validation
        event = await self._sign_event(event)

        self.event_buffer.append(event)
        self.stats["events_collected"] += 1

        # Flush buffer if it's full
        if len(self.event_buffer) >= self.config.batch_size:
            await self._flush_buffer()

    async def _sign_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Sign event for integrity validation"""
        # Create signature
        event_json = json.dumps(event, sort_keys=True)
        signature = sign_event(self.hmac_key, event)

        event["signature"] = signature
        return event

    async def _periodic_flush(self):
        """Periodically flush the event buffer"""
        while self.running:
            await asyncio.sleep(self.config.flush_interval)

            if self.event_buffer:
                await self._flush_buffer()

    async def _flush_buffer(self):
        """Flush events to the backend"""
        if not self.event_buffer:
            return

        events_to_send = self.event_buffer.copy()
        self.event_buffer.clear()

        payload = {
            "source_type": self.config.source_type,
            "hostname": self.config.hostname,
            "events": events_to_send,
        }

        # Try to send with retries
        for attempt in range(self.config.max_retries):
            try:
                success = await self._send_events(payload)
                if success:
                    self.stats["events_sent"] += len(events_to_send)
                    self.stats["last_flush"] = datetime.utcnow().isoformat()
                    self.logger.debug(f"Sent {len(events_to_send)} events to backend")
                    return
                else:
                    self.logger.warning(f"Send attempt {attempt + 1} failed")

            except Exception as e:
                self.logger.error(f"Send attempt {attempt + 1} error: {e}")

            if attempt < self.config.max_retries - 1:
                await asyncio.sleep(self.config.retry_delay * (attempt + 1))

        # All retries failed
        self.stats["events_failed"] += len(events_to_send)
        self.logger.error(
            f"Failed to send {len(events_to_send)} events after {self.config.max_retries} attempts"
        )

    async def _send_events(self, payload: Dict[str, Any]) -> bool:
        """Send events to the backend"""
        headers = {
            "Content-Type": "application/json",
            "User-Agent": f"Mini-XDR-Agent/{self.config.hostname}",
        }

        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        # Compress if enabled
        body_text = canonicalize_payload(payload)
        data = body_text.encode()

        if self.config.compress_data:
            headers["Content-Encoding"] = "gzip"
            import gzip

            data = gzip.compress(data)

        signed_headers, _, _ = build_hmac_headers(
            self.device_id, self.hmac_key, "POST", "/ingest/multi", body_text
        )
        headers.update(signed_headers)

        try:
            async with self.session.post(
                f"{self.config.backend_url}/ingest/multi", headers=headers, data=data
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    self.logger.debug(f"Backend response: {result}")
                    return True
                elif response.status == 429:
                    self.logger.warning("Backend rate limited")
                    await asyncio.sleep(60)  # Wait before retry
                    return False
                else:
                    error_text = await response.text()
                    self.logger.error(f"Backend error {response.status}: {error_text}")
                    return False

        except Exception as e:
            self.logger.error(f"Network error sending events: {e}")
            return False

    async def _periodic_stats(self):
        """Periodically log statistics"""
        while self.running:
            await asyncio.sleep(300)  # Every 5 minutes

            uptime = time.time() - self.stats["start_time"]
            rate = (
                self.stats["events_collected"] / max(uptime, 1) * 60
            )  # events per minute

            self.logger.info(
                f"Agent stats - Collected: {self.stats['events_collected']}, "
                f"Sent: {self.stats['events_sent']}, Failed: {self.stats['events_failed']}, "
                f"Rate: {rate:.1f}/min, Buffer: {len(self.event_buffer)}"
            )

    async def _cleanup(self):
        """Cleanup resources"""
        self.logger.info("Shutting down ingestion agent...")

        # Flush remaining events
        if self.event_buffer:
            await self._flush_buffer()

        # Close HTTP session
        if self.session:
            await self.session.close()

        self.logger.info("Ingestion agent shutdown complete")


def load_config_from_file(config_path: str) -> AgentConfig:
    """Load agent configuration from file"""
    with open(config_path, "r") as f:
        config_data = json.load(f)

    return AgentConfig(**config_data)


def create_default_config(output_path: str):
    """Create a default configuration file"""
    default_config = {
        "backend_url": "http://mini-xdr-backend:8000",
        "api_key": "your-api-key-here",
        "source_type": "cowrie",
        "hostname": "honeypot-01",
        "log_paths": {"cowrie": "/var/log/cowrie/cowrie.json"},
        "batch_size": 50,
        "flush_interval": 30,
        "max_retries": 3,
        "retry_delay": 5,
        "validate_ssl": True,
        "compress_data": True,
        "device_id_env": "MINIXDR_AGENT_DEVICE_ID",
        "hmac_key_env": "MINIXDR_AGENT_HMAC_KEY",
    }

    with open(output_path, "w") as f:
        json.dump(default_config, f, indent=2)

    print(f"Default configuration written to {output_path}")


async def main():
    """Main entry point for the agent"""
    parser = argparse.ArgumentParser(description="Mini-XDR Ingestion Agent")
    parser.add_argument("--config", "-c", required=True, help="Configuration file path")
    parser.add_argument("--create-config", help="Create default config file and exit")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    if args.create_config:
        create_default_config(args.create_config)
        return

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    try:
        # Load configuration
        config = load_config_from_file(args.config)

        # Create and start agent
        agent = IngestionAgent(config)
        await agent.start()

    except KeyboardInterrupt:
        logging.info("Agent interrupted by user")
    except Exception as e:
        logging.error(f"Agent failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
