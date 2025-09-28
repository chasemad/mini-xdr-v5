import asyncio
import paramiko
import logging
import os
from typing import Tuple, Optional
from datetime import datetime, timedelta
from .config import settings

logger = logging.getLogger(__name__)

# Simulation mode for when SSH connectivity is unavailable
SIMULATION_MODE = os.getenv("XDR_SIMULATION_MODE", "false").lower() == "true"


class ResponderAgent:
    """Handles remote execution of containment actions via SSH"""
    
    def __init__(self):
        self.host = settings.honeypot_host
        self.port = settings.honeypot_ssh_port
        self.username = settings.honeypot_user
        self.key_path = settings.expanded_ssh_key_path
        logger.info(f"ResponderAgent initialized: {self.username}@{self.host}:{self.port} using key {self.key_path}")
        
    async def test_connection(self) -> Tuple[str, str]:
        """Test SSH connectivity to the honeypot"""
        logger.info("Testing SSH connection to honeypot...")
        status, stdout, stderr = await self.execute_command("echo 'connection_test'", timeout=10)
        if status == "success" and "connection_test" in stdout:
            return "success", "SSH connection test successful"
        else:
            return "failed", f"SSH connection test failed: {stderr}"
    
    async def execute_command(self, command: str, timeout: int = 30) -> Tuple[str, str, str]:
        """
        Execute a command via SSH with fallback to subprocess
        
        Returns:
            Tuple of (status, stdout, stderr)
        """
        # Try paramiko first
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                self._sync_execute_command,
                command,
                timeout
            )
            # If paramiko succeeds, return the result
            if result[0] == "success":
                return result
            else:
                logger.warning(f"Paramiko failed: {result[2]}, trying subprocess SSH...")
        except Exception as e:
            logger.warning(f"Paramiko execution failed: {e}, trying subprocess SSH...")
        
        # Fallback to subprocess SSH
        try:
            return await self._subprocess_ssh_command(command, timeout)
        except Exception as e:
            logger.error(f"Both SSH methods failed: {e}")
            return "failed", "", str(e)
    
    def _sync_execute_command(self, command: str, timeout: int) -> Tuple[str, str, str]:
        """Synchronous SSH command execution"""
        client = None
        try:
            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            
            # Load the private key with better error handling
            try:
                private_key = paramiko.Ed25519Key.from_private_key_file(self.key_path)
            except Exception as key_error:
                logger.error(f"Failed to load SSH key from {self.key_path}: {key_error}")
                return "failed", "", f"SSH key error: {key_error}"
            
            # Connect to the honeypot with enhanced parameters for subprocess environment
            connect_params = {
                'hostname': self.host,
                'port': self.port,
                'username': self.username,
                'pkey': private_key,
                'timeout': timeout,
                'look_for_keys': False,
                'allow_agent': False,
                'banner_timeout': 30,
                'auth_timeout': 30,
                'channel_timeout': 30,
                'compress': False,
                'gss_auth': False,
                'gss_kex': False,
                'gss_deleg_creds': False,
                'disabled_algorithms': {'pubkeys': ['ssh-rsa-cert-v01@openssh.com']},
            }
            
            logger.info(f"Attempting SSH connection to {self.host}:{self.port} as {self.username}")
            client.connect(**connect_params)
            
            # Execute command
            stdin, stdout, stderr = client.exec_command(command, timeout=timeout)
            
            # Read output
            stdout_text = stdout.read().decode('utf-8').strip()
            stderr_text = stderr.read().decode('utf-8').strip()
            exit_status = stdout.channel.recv_exit_status()
            
            status = "success" if exit_status == 0 else "failed"
            
            logger.info(f"SSH command executed: {command} -> status={status}")
            if stderr_text:
                logger.warning(f"SSH stderr: {stderr_text}")
            
            return status, stdout_text, stderr_text
            
        except Exception as e:
            logger.error(f"SSH connection error: {e}")
            return "failed", "", str(e)
        finally:
            if client:
                client.close()
    
    async def _subprocess_ssh_command(self, command: str, timeout: int) -> Tuple[str, str, str]:
        """Execute SSH command using subprocess (fallback method)"""
        import subprocess
        
        # Build the SSH command
        ssh_cmd = [
            'ssh',
            '-p', str(self.port),
            '-i', self.key_path,
            '-o', 'StrictHostKeyChecking=yes',
            '-o', f'UserKnownHostsFile={os.path.expanduser("~/.ssh/known_hosts")}',
            '-o', 'ConnectTimeout=10',
            '-o', 'ServerAliveInterval=5',
            '-o', 'ServerAliveCountMax=2',
            f"{self.username}@{self.host}",
            command
        ]
        
        logger.info(f"Executing subprocess SSH: {' '.join(ssh_cmd[:-1])} '<command>'")
        
        try:
            # Run SSH command with subprocess
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                self._run_subprocess,
                ssh_cmd,
                timeout
            )
            return result
        except Exception as e:
            logger.error(f"Subprocess SSH failed: {e}")
            return "failed", "", str(e)
    
    def _run_subprocess(self, cmd: list, timeout: int) -> Tuple[str, str, str]:
        """Run subprocess synchronously"""
        import subprocess
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=os.environ.copy()  # Use current environment
            )
            
            status = "success" if result.returncode == 0 else "failed"
            logger.info(f"Subprocess SSH result: status={status}, returncode={result.returncode}")
            
            return status, result.stdout.strip(), result.stderr.strip()
            
        except subprocess.TimeoutExpired:
            logger.error(f"SSH command timed out after {timeout} seconds")
            return "failed", "", f"Command timed out after {timeout} seconds"
        except Exception as e:
            logger.error(f"Subprocess execution error: {e}")
            return "failed", "", str(e)


# Global responder instance
responder = ResponderAgent()


async def block_ip(ip: str, duration_seconds: int = None) -> Tuple[str, str]:
    """
    Block an IP address using UFW with optional temporary blocking
    
    Args:
        ip: IP address to block
        duration_seconds: If provided, automatically unblock after this many seconds
    
    Returns:
        Tuple of (status, detail_message)
    """
    if not _is_valid_ip(ip):
        return "failed", f"Invalid IP address: {ip}"
    
    if _is_private_ip(ip) and not settings.allow_private_ip_blocking:
        logger.warning(f"Refusing to block private IP: {ip} (allow_private_ip_blocking=False)")
        return "failed", f"Refusing to block private IP: {ip} - Enable allow_private_ip_blocking for testing"
    
    # Check if UFW is available, otherwise use iptables
    ufw_check_status, _, _ = await responder.execute_command("which ufw", timeout=5)
    
    if ufw_check_status == "success":
        command = f"sudo ufw deny from {ip} to any"
    else:
        # Use iptables for T-Pot systems
        command = f"sudo iptables -I INPUT -s {ip} -j DROP"
    
    status, stdout, stderr = await responder.execute_command(command)
    
    detail = f"Command: {command}\nStatus: {status}\nStdout: {stdout}\nStderr: {stderr}"
    
    if status == "success":
        logger.info(f"Successfully blocked IP: {ip}")
        
        # Schedule automatic unblock if duration specified
        if duration_seconds:
            logger.info(f"Scheduling auto-unblock for {ip} in {duration_seconds} seconds")
            asyncio.create_task(_auto_unblock_after_delay(ip, duration_seconds))
            detail += f"\nAuto-unblock scheduled in {duration_seconds} seconds"
    else:
        logger.error(f"Failed to block IP {ip}: {stderr}")
    
    return status, detail


async def _auto_unblock_after_delay(ip: str, delay_seconds: int):
    """Automatically unblock an IP after a delay"""
    try:
        await asyncio.sleep(delay_seconds)
        status, detail = await unblock_ip(ip)
        if status == "success":
            logger.info(f"Auto-unblocked IP {ip} after {delay_seconds} seconds")
        else:
            logger.error(f"Failed to auto-unblock IP {ip}: {detail}")
    except Exception as e:
        logger.error(f"Error during auto-unblock of {ip}: {e}")


async def unblock_ip(ip: str) -> Tuple[str, str]:
    """
    Unblock an IP address using UFW
    
    Returns:
        Tuple of (status, detail_message)
    """
    if not _is_valid_ip(ip):
        return "failed", f"Invalid IP address: {ip}"
    
    # Check if UFW is available, otherwise use iptables
    ufw_check_status, _, _ = await responder.execute_command("which ufw", timeout=5)
    
    if ufw_check_status == "success":
        command = f"sudo ufw delete deny from {ip} to any"
    else:
        # Use iptables for T-Pot systems
        command = f"sudo iptables -D INPUT -s {ip} -j DROP"
    
    status, stdout, stderr = await responder.execute_command(command)
    
    detail = f"Command: {command}\nStatus: {status}\nStdout: {stdout}\nStderr: {stderr}"
    
    if status == "success":
        logger.info(f"Successfully unblocked IP: {ip}")
    else:
        logger.error(f"Failed to unblock IP {ip}: {stderr}")
    
    return status, detail


def _is_valid_ip(ip: str) -> bool:
    """Basic IP validation"""
    try:
        parts = ip.split('.')
        if len(parts) != 4:
            return False
        for part in parts:
            if not (0 <= int(part) <= 255):
                return False
        return True
    except (ValueError, AttributeError):
        return False


def _is_private_ip(ip: str) -> bool:
    """Check if IP is in private ranges (RFC 1918)"""
    try:
        parts = [int(p) for p in ip.split('.')]
        
        # 10.0.0.0/8
        if parts[0] == 10:
            return True
        
        # 172.16.0.0/12
        if parts[0] == 172 and 16 <= parts[1] <= 31:
            return True
        
        # 192.168.0.0/16
        if parts[0] == 192 and parts[1] == 168:
            return True
        
        # 127.0.0.0/8 (loopback)
        if parts[0] == 127:
            return True
        
        return False
    except (ValueError, IndexError):
        return False
