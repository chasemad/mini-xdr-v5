"""
T-Pot Honeypot Connector
Monitors T-Pot honeypot and ingests security events in real-time
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiohttp
import asyncssh
from sqlalchemy.ext.asyncio import AsyncSession

from .config import settings
from .multi_ingestion import multi_ingestor

logger = logging.getLogger(__name__)


class TPotConnector:
    """T-Pot honeypot connector for real-time monitoring and defensive actions"""

    # T-Pot log file paths
    LOG_PATHS = {
        "cowrie": "/home/luxieum/tpotce/data/cowrie/log/cowrie.json",
        "dionaea": "/home/luxieum/tpotce/data/dionaea/log/dionaea.json",
        "suricata": "/home/luxieum/tpotce/data/suricata/log/eve.json",
        "wordpot": "/home/luxieum/tpotce/data/wordpot/logs/wordpot.json",
        "elasticpot": "/home/luxieum/tpotce/data/elasticpot/log/elasticpot.json",
        "redishoneypot": "/home/luxieum/tpotce/data/redishoneypot/log/redishoneypot.log",
        "mailoney": "/home/luxieum/tpotce/data/mailoney/log/commands.log",
        "sentrypeer": "/home/luxieum/tpotce/data/sentrypeer/log/sentrypeer.json",
    }

    def __init__(self):
        self.host = settings.tpot_host or "24.11.0.176"
        self.ssh_port = settings.tpot_ssh_port or 64295
        self.web_port = settings.tpot_web_port or 64297
        self.ssh_user = settings.honeypot_user or "luxieum"
        self.ssh_password = settings.tpot_api_key  # Using API key field for password
        self.ssh_key_path = settings.expanded_ssh_key_path

        self.elasticsearch_port = 64298
        self.kibana_port = 64296

        self.ssh_conn: Optional[asyncssh.SSHClientConnection] = None
        self.tunnels: Dict[str, Any] = {}
        self.monitoring_tasks: Dict[str, asyncio.Task] = {}
        self.health_task: Optional[asyncio.Task] = None
        self.is_connected = False

        logger.info(f"T-Pot connector initialized for {self.host}:{self.ssh_port}")

    async def connect(self) -> bool:
        """Establish SSH connection to T-Pot"""
        try:
            logger.info(f"Connecting to T-Pot at {self.host}:{self.ssh_port}")

            # Determine authentication method
            auth_kwargs = {}
            if self.ssh_password:
                auth_kwargs["password"] = self.ssh_password
                logger.info("Using password authentication")
            elif Path(self.ssh_key_path).exists():
                auth_kwargs["client_keys"] = [self.ssh_key_path]
                logger.info(f"Using key authentication: {self.ssh_key_path}")
            else:
                logger.warning(
                    "No authentication method configured - T-Pot will be unavailable"
                )
                return False

            # Connect with timeout
            self.ssh_conn = await asyncio.wait_for(
                asyncssh.connect(
                    host=self.host,
                    port=self.ssh_port,
                    username=self.ssh_user,
                    known_hosts=None,  # Accept any host key
                    **auth_kwargs,
                ),
                timeout=10.0,
            )

            self.is_connected = True
            logger.info(f"✅ Successfully connected to T-Pot at {self.host}")

            # Test connection with a simple command
            result = await self.execute_command("echo 'Connected'")
            if result["success"]:
                logger.info("✅ Connection test successful")

            return True

        except asyncio.TimeoutError:
            logger.warning(
                f"⚠️  Connection timeout to T-Pot at {self.host}:{self.ssh_port} (this is expected if not at allowed IP 172.16.110.1)"
            )
            return False
        except ConnectionRefusedError as e:
            logger.warning(
                f"⚠️  Connection refused to T-Pot: {e} (check if T-Pot is running and firewall allows connection)"
            )
            return False
        except Exception as e:
            logger.warning(
                f"⚠️  Failed to connect to T-Pot: {e} (T-Pot monitoring will be unavailable)"
            )
            return False

    async def disconnect(self):
        """Close SSH connection and tunnels"""
        try:
            # Stop health monitor
            if self.health_task and not self.health_task.done():
                self.health_task.cancel()
                try:
                    await self.health_task
                except asyncio.CancelledError:
                    pass
            self.health_task = None

            # Stop all monitoring tasks
            for task_name, task in self.monitoring_tasks.items():
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
                logger.info(f"Stopped monitoring task: {task_name}")

            self.monitoring_tasks.clear()

            # Close SSH tunnels
            for tunnel_name, tunnel in self.tunnels.items():
                try:
                    tunnel.close()
                    await tunnel.wait_closed()
                    logger.info(f"Closed tunnel: {tunnel_name}")
                except Exception as e:
                    logger.warning(f"Error closing tunnel {tunnel_name}: {e}")

            self.tunnels.clear()

            # Close SSH connection
            if self.ssh_conn:
                self.ssh_conn.close()
                await self.ssh_conn.wait_closed()
                logger.info("SSH connection closed")

            self.is_connected = False

        except Exception as e:
            logger.error(f"Error during disconnect: {e}")
            self.health_task = None

    async def setup_tunnels(self) -> bool:
        """Set up SSH tunnels for internal services"""
        if not self.is_connected:
            logger.error("Not connected to T-Pot")
            return False

        try:
            # Create tunnel for Elasticsearch
            logger.info("Setting up Elasticsearch tunnel...")
            es_listener = await self.ssh_conn.forward_local_port(
                "",  # Listen on all interfaces
                self.elasticsearch_port,
                "localhost",
                self.elasticsearch_port,
            )
            self.tunnels["elasticsearch"] = es_listener
            logger.info(f"✅ Elasticsearch tunnel: localhost:{self.elasticsearch_port}")

            # Create tunnel for Kibana
            logger.info("Setting up Kibana tunnel...")
            kibana_listener = await self.ssh_conn.forward_local_port(
                "", self.kibana_port, "localhost", self.kibana_port
            )
            self.tunnels["kibana"] = kibana_listener
            logger.info(f"✅ Kibana tunnel: localhost:{self.kibana_port}")

            return True

        except Exception as e:
            logger.error(f"Failed to set up tunnels: {e}")
            return False

    async def execute_command(self, command: str, timeout: int = 30) -> Dict[str, Any]:
        """Execute command on T-Pot via SSH"""
        if not self.is_connected:
            return {
                "success": False,
                "error": "Not connected to T-Pot",
                "output": "",
                "stderr": "",
            }

        try:
            result = await asyncio.wait_for(
                self.ssh_conn.run(command, check=False), timeout=timeout
            )

            # asyncssh SSHCompletedProcess returns result as a string-like object
            # Direct access to the result gives us stdout
            exit_code = result.exit_status if hasattr(result, "exit_status") else 0

            # Get stdout and stderr correctly
            stdout_output = ""
            stderr_output = ""

            if hasattr(result, "stdout") and result.stdout:
                stdout_output = result.stdout
            elif isinstance(result, str):
                stdout_output = result
            else:
                # Try to convert to string
                stdout_output = str(result) if result else ""

            if hasattr(result, "stderr") and result.stderr:
                stderr_output = result.stderr

            return {
                "success": exit_code == 0,
                "output": stdout_output,
                "stderr": stderr_output,
                "exit_status": exit_code,
            }

        except asyncio.TimeoutError:
            logger.error(f"Command timeout: {command}")
            return {
                "success": False,
                "error": "Command timeout",
                "output": "",
                "stderr": "",
            }
        except Exception as e:
            logger.error(f"Command execution failed: {e}")
            return {"success": False, "error": str(e), "output": "", "stderr": ""}

    async def tail_log_file(
        self, log_path: str, callback, lines: int = 10
    ) -> asyncio.Task:
        """Tail a log file and call callback for each new line"""

        async def _tail():
            try:
                # Get initial file size
                cmd = f"sudo wc -c < {log_path} 2>/dev/null || echo 0"
                result = await self.execute_command(cmd)
                last_size = int(result["output"].strip() or 0)

                logger.info(f"Starting to tail {log_path} from size {last_size}")

                while True:
                    try:
                        # Check current file size
                        cmd = f"sudo wc -c < {log_path} 2>/dev/null || echo 0"
                        result = await self.execute_command(cmd)
                        current_size = int(result["output"].strip() or 0)

                        # If file has new content
                        if current_size > last_size:
                            # Read new content
                            bytes_to_read = current_size - last_size
                            cmd = f"sudo tail -c {bytes_to_read} {log_path}"
                            result = await self.execute_command(cmd)

                            if result["success"] and result["output"]:
                                # Process each line
                                for line in result["output"].strip().split("\n"):
                                    if line.strip():
                                        await callback(line, log_path)

                            last_size = current_size

                        # Wait before checking again
                        await asyncio.sleep(2)

                    except asyncio.CancelledError:
                        logger.info(f"Tailing stopped for {log_path}")
                        break
                    except Exception as e:
                        logger.error(f"Error tailing {log_path}: {e}")
                        await asyncio.sleep(5)

            except Exception as e:
                logger.error(f"Fatal error in tail task for {log_path}: {e}")

        task = asyncio.create_task(_tail())
        return task

    async def start_monitoring(
        self, db_session_factory, honeypot_types: List[str] = None
    ):
        """Start monitoring T-Pot honeypot logs"""
        if not self.is_connected:
            logger.error("Cannot start monitoring - not connected to T-Pot")
            return False

        # Default to monitoring all honeypot types
        if honeypot_types is None:
            honeypot_types = list(self.LOG_PATHS.keys())

        logger.info(f"Starting monitoring for: {', '.join(honeypot_types)}")

        async def process_log_line(line: str, log_path: str):
            """Process a single log line"""
            try:
                # Determine honeypot type from path
                honeypot_type = None
                for hp_type, path in self.LOG_PATHS.items():
                    if path == log_path:
                        honeypot_type = hp_type
                        break

                if not honeypot_type:
                    return

                # Parse JSON log line
                try:
                    event_data = json.loads(line)
                except json.JSONDecodeError:
                    # Not JSON, skip
                    return

                # Ingest event into XDR
                async with db_session_factory() as db:
                    await multi_ingestor.ingest_events(
                        source_type=honeypot_type,
                        hostname=self.host,
                        events=[event_data],
                        db=db,
                    )

                logger.debug(f"Ingested {honeypot_type} event from T-Pot")

            except Exception as e:
                logger.error(f"Failed to process log line: {e}")

        # Start tailing each log file
        for honeypot_type in honeypot_types:
            if honeypot_type not in self.LOG_PATHS:
                logger.warning(f"Unknown honeypot type: {honeypot_type}")
                continue

            log_path = self.LOG_PATHS[honeypot_type]

            # Check if log file exists
            check_cmd = f"sudo test -f {log_path} && echo 'exists' || echo 'missing'"
            result = await self.execute_command(check_cmd)

            if "missing" in result["output"]:
                logger.warning(f"Log file not found: {log_path}")
                continue

            # Start tailing
            task = await self.tail_log_file(log_path, process_log_line)
            self.monitoring_tasks[honeypot_type] = task
            logger.info(f"✅ Started monitoring: {honeypot_type}")

        return True

    async def stop_monitoring(self, honeypot_type: str = None):
        """Stop monitoring specific or all honeypots"""
        if honeypot_type:
            # Stop specific honeypot monitoring
            if honeypot_type in self.monitoring_tasks:
                task = self.monitoring_tasks[honeypot_type]
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
                del self.monitoring_tasks[honeypot_type]
                logger.info(f"Stopped monitoring: {honeypot_type}")
        else:
            # Stop all monitoring
            for task_name, task in list(self.monitoring_tasks.items()):
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
                logger.info(f"Stopped monitoring: {task_name}")

            self.monitoring_tasks.clear()

    async def query_elasticsearch(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """Query T-Pot Elasticsearch via tunnel"""
        if "elasticsearch" not in self.tunnels:
            return {"success": False, "error": "Elasticsearch tunnel not established"}

        try:
            async with aiohttp.ClientSession() as session:
                url = f"http://localhost:{self.elasticsearch_port}/_search"
                async with session.post(url, json=query, timeout=30) as response:
                    if response.status == 200:
                        data = await response.json()
                        return {"success": True, "data": data}
                    else:
                        error_text = await response.text()
                        return {
                            "success": False,
                            "error": f"Elasticsearch query failed: {error_text}",
                        }

        except Exception as e:
            logger.error(f"Elasticsearch query error: {e}")
            return {"success": False, "error": str(e)}

    async def get_recent_attacks(self, minutes: int = 5) -> List[Dict[str, Any]]:
        """Get recent attacks from Elasticsearch"""
        query = {
            "query": {"range": {"@timestamp": {"gte": f"now-{minutes}m"}}},
            "size": 100,
            "sort": [{"@timestamp": "desc"}],
        }

        result = await self.query_elasticsearch(query)
        if result["success"]:
            hits = result["data"].get("hits", {}).get("hits", [])
            return [hit["_source"] for hit in hits]

        return []

    # Defensive Actions

    async def block_ip(self, ip_address: str) -> Dict[str, Any]:
        """Block an IP address using UFW with password authentication"""
        logger.info(f"Blocking IP on T-Pot: {ip_address}")

        # Use password-based sudo for security
        # echo password | sudo -S command
        if self.ssh_password:
            logger.info("Using password-based sudo authentication")
            cmd = f"echo '{self.ssh_password}' | sudo -S ufw deny from {ip_address}"
        else:
            logger.warning(
                "No SSH password configured - trying without sudo authentication"
            )
            cmd = f"sudo ufw deny from {ip_address}"

        logger.debug(f"Executing UFW block command for {ip_address}")
        result = await self.execute_command(cmd, timeout=15)

        logger.info(
            f"UFW block result - success: {result.get('success')}, exit_status: {result.get('exit_status')}, output: {result.get('output')[:200]}, stderr: {result.get('stderr')[:200]}"
        )

        if result["success"]:
            logger.info(f"✅ Successfully blocked {ip_address} on T-Pot firewall")
            return {
                "success": True,
                "action": "block_ip",
                "ip_address": ip_address,
                "message": f"IP {ip_address} blocked on T-Pot firewall using authenticated sudo",
                "method": "password-authenticated"
                if self.ssh_password
                else "passwordless-sudo",
            }

        # Check if it partially succeeded (UFW sometimes returns non-zero but works)
        output = result.get("output", "")
        stderr = result.get("stderr", "")
        combined_output = f"{output} {stderr}"

        if any(
            keyword in combined_output
            for keyword in [
                "Rule added",
                "Rule updated",
                "added",
                "Skipping adding existing rule",
            ]
        ):
            logger.info(
                f"✅ IP {ip_address} blocked (UFW confirmed in output despite non-zero exit)"
            )
            return {
                "success": True,
                "action": "block_ip",
                "ip_address": ip_address,
                "message": f"IP {ip_address} blocked successfully",
                "method": "password-authenticated"
                if self.ssh_password
                else "passwordless-sudo",
                "note": "Rule added despite non-zero exit code",
            }

        # Failed - return detailed error
        error_details = result.get("stderr", result.get("error", "Unknown error"))
        logger.error(f"Failed to block {ip_address}: {error_details}")
        logger.error(f"Full output: {output}")

        return {
            "success": False,
            "action": "block_ip",
            "ip_address": ip_address,
            "error": error_details,
            "output": output,
            "stderr": stderr,
            "note": "UFW command failed - check T-Pot firewall configuration and SSH password",
        }

    async def unblock_ip(self, ip_address: str) -> Dict[str, Any]:
        """Unblock an IP address using password-authenticated sudo"""
        logger.info(f"Unblocking IP on T-Pot: {ip_address}")

        # Get rule number for the IP (with password)
        if self.ssh_password:
            cmd = f"echo '{self.ssh_password}' | sudo -S ufw status numbered | grep {ip_address} | head -1 | awk '{{print $1}}' | tr -d '[]'"
        else:
            cmd = f"sudo ufw status numbered | grep {ip_address} | head -1 | awk '{{print $1}}' | tr -d '[]'"

        result = await self.execute_command(cmd)

        if result["success"] and result["output"].strip():
            rule_num = result["output"].strip()
            logger.info(f"Found UFW rule #{rule_num} for {ip_address}")

            # Delete the rule with password
            if self.ssh_password:
                cmd = f"echo 'y\n{self.ssh_password}' | sudo -S ufw delete {rule_num}"
            else:
                cmd = f"echo 'y' | sudo ufw delete {rule_num}"

            result = await self.execute_command(cmd)

            if result["success"] or "Deleting" in result.get("output", ""):
                logger.info(f"✅ Successfully unblocked {ip_address}")
                return {
                    "success": True,
                    "action": "unblock_ip",
                    "ip_address": ip_address,
                    "message": f"IP {ip_address} unblocked",
                    "method": "password-authenticated",
                }

        return {
            "success": False,
            "action": "unblock_ip",
            "ip_address": ip_address,
            "error": "Failed to find or unblock IP",
        }

    async def get_active_blocks(self) -> Dict[str, Any]:
        """Get list of currently blocked IPs using password-authenticated sudo"""
        if self.ssh_password:
            cmd = f"echo '{self.ssh_password}' | sudo -S ufw status | grep DENY | awk '{{print $3}}'"
        else:
            cmd = "sudo ufw status | grep DENY | awk '{print $3}'"

        result = await self.execute_command(cmd)

        if result["success"]:
            blocked_ips = [
                ip.strip() for ip in result["output"].split("\n") if ip.strip()
            ]
            return {
                "success": True,
                "blocked_ips": blocked_ips,
                "count": len(blocked_ips),
            }

        return {"success": False, "error": result["stderr"]}

    async def _heartbeat(self, interval: int = 30):
        """Periodic connectivity check with auto-reconnect"""
        while True:
            try:
                if not self.is_connected:
                    await self.connect()
                else:
                    # lightweight ping
                    ping_result = await self.execute_command("true", timeout=5)
                    if not ping_result.get("success", False):
                        self.is_connected = False
                        logger.warning("T-Pot heartbeat failed, attempting reconnect")
                        await self.connect()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"T-Pot heartbeat error: {e}")
            await asyncio.sleep(interval)

    def start_health_monitor(self, interval: int = 30):
        """Start background heartbeat if not already running"""
        if self.health_task and not self.health_task.done():
            return
        loop = asyncio.get_event_loop()
        self.health_task = loop.create_task(self._heartbeat(interval))

    async def stop_honeypot_container(self, container_name: str) -> Dict[str, Any]:
        """Stop a specific honeypot container using password-authenticated sudo"""
        logger.info(f"Stopping container: {container_name}")

        if self.ssh_password:
            cmd = f"echo '{self.ssh_password}' | sudo -S docker stop {container_name}"
        else:
            cmd = f"docker stop {container_name}"

        result = await self.execute_command(cmd)

        if result["success"]:
            return {
                "success": True,
                "action": "stop_container",
                "container": container_name,
                "message": f"Container {container_name} stopped",
            }

        return {
            "success": False,
            "action": "stop_container",
            "container": container_name,
            "error": result["stderr"],
        }

    async def start_honeypot_container(self, container_name: str) -> Dict[str, Any]:
        """Start a specific honeypot container using password-authenticated sudo"""
        logger.info(f"Starting container: {container_name}")

        if self.ssh_password:
            cmd = f"echo '{self.ssh_password}' | sudo -S docker start {container_name}"
        else:
            cmd = f"docker start {container_name}"

        result = await self.execute_command(cmd)

        if result["success"]:
            return {
                "success": True,
                "action": "start_container",
                "container": container_name,
                "message": f"Container {container_name} started",
            }

        return {
            "success": False,
            "action": "start_container",
            "container": container_name,
            "error": result["stderr"],
        }

    async def get_container_status(self) -> Dict[str, Any]:
        """Get status of all T-Pot containers"""
        cmd = "docker ps --format '{{.Names}}\t{{.Status}}\t{{.Ports}}'"
        result = await self.execute_command(cmd)

        if result["success"]:
            containers = []
            for line in result["output"].split("\n"):
                if line.strip():
                    parts = line.split("\t")
                    if len(parts) >= 2:
                        containers.append(
                            {
                                "name": parts[0],
                                "status": parts[1],
                                "ports": parts[2] if len(parts) > 2 else "",
                            }
                        )

            return {"success": True, "containers": containers, "count": len(containers)}

        return {"success": False, "error": result["stderr"]}

    async def get_honeypot_stats(self) -> Dict[str, Any]:
        """Get T-Pot honeypot statistics"""
        stats = {
            "connected": self.is_connected,
            "host": self.host,
            "monitoring": list(self.monitoring_tasks.keys()),
            "tunnels": list(self.tunnels.keys()),
        }

        if self.is_connected:
            # Get container stats
            container_result = await self.get_container_status()
            if container_result["success"]:
                stats["containers"] = container_result["containers"]

            # Get active blocks
            blocks_result = await self.get_active_blocks()
            if blocks_result["success"]:
                stats["blocked_ips"] = blocks_result["blocked_ips"]
                stats["blocked_count"] = blocks_result["count"]

        return stats


# Global connector instance
_tpot_connector: Optional[TPotConnector] = None


def get_tpot_connector() -> TPotConnector:
    """Get or create global T-Pot connector instance"""
    global _tpot_connector
    if _tpot_connector is None:
        _tpot_connector = TPotConnector()
    return _tpot_connector


async def initialize_tpot_monitoring(db_session_factory):
    """Initialize T-Pot monitoring on startup"""
    connector = get_tpot_connector()

    try:
        # Connect to T-Pot
        if await connector.connect():
            logger.info("T-Pot connection established")

            # Set up tunnels
            if await connector.setup_tunnels():
                logger.info("T-Pot tunnels established")

            # Start monitoring
            if await connector.start_monitoring(db_session_factory):
                logger.info("T-Pot monitoring started")

            return True
        else:
            logger.error("Failed to connect to T-Pot")
            return False

    except Exception as e:
        logger.error(f"T-Pot initialization failed: {e}")
        return False


async def shutdown_tpot_monitoring():
    """Shutdown T-Pot monitoring"""
    connector = get_tpot_connector()
    await connector.disconnect()
    logger.info("T-Pot monitoring shutdown complete")
