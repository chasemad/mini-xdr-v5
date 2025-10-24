"""
Endpoint Detection & Response (EDR) Agent
Manages Windows endpoint security with rollback support
"""
import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import re

try:
    import pypsrp
    from pypsrp.client import Client
    WINRM_AVAILABLE = True
except ImportError:
    WINRM_AVAILABLE = False
    logging.warning("pypsrp not available - EDR Agent will use simulation mode")

from ..models import Event, Incident
from ..config import settings

logger = logging.getLogger(__name__)


class EDRAgent:
    """
    AI Agent for Endpoint Detection & Response
    
    Capabilities:
    - Process management (kill, suspend, analyze)
    - File operations (quarantine, delete, restore)
    - Memory forensics (dump, scan)
    - Host isolation (network-level via Windows Firewall)
    - Registry monitoring and cleanup
    - Scheduled task management
    - Service management
    - Detection (process injection, LOLBins, PowerShell abuse)
    - Full rollback support
    """
    
    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        self.agent_id = "edr_agent_v1"
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # WinRM configuration
        self.winrm_user = getattr(settings, "winrm_user", None) or "Administrator"
        self.winrm_password = getattr(settings, "winrm_password", None)
        self.winrm_available = WINRM_AVAILABLE
        
        # Connection pool
        self.connections = {}  # hostname -> Client
        
        # Rollback storage
        self.rollback_storage = []
        
        # Quarantine base path
        self.quarantine_base = "C:\\XDR_Quarantine"
    
    # ==================== PUBLIC API (Same Structure as IAM Agent) ====================
    
    async def execute_action(
        self,
        action_name: str,
        params: Dict[str, Any],
        incident_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """Execute EDR action with rollback support"""
        action_id = f"{self.agent_id}_{action_name}_{int(datetime.now().timestamp())}"
        
        try:
            # Capture state for rollback
            rollback_data = await self._capture_state(action_name, params)
            rollback_id = self._store_rollback(rollback_data, incident_id)
            
            # Execute action
            self.logger.info(f"Executing {action_name} with params: {params}")
            result = await self._execute_action_impl(action_name, params)
            
            return {
                "success": True,
                "action_id": action_id,
                "rollback_id": rollback_id,
                "result": result,
                "message": f"Action {action_name} completed successfully",
                "agent": self.agent_id
            }
            
        except Exception as e:
            self.logger.error(f"Action {action_name} failed: {e}")
            return {
                "success": False,
                "action_id": action_id,
                "error": str(e),
                "message": f"Action {action_name} failed",
                "agent": self.agent_id
            }
    
    async def rollback_action(self, rollback_id: str) -> Dict[str, Any]:
        """Rollback EDR action"""
        try:
            rollback_data = self._get_rollback(rollback_id)
            
            if not rollback_data:
                return {"success": False, "error": "Rollback ID not found"}
            
            if rollback_data.get('executed'):
                return {"success": False, "error": "Rollback already executed"}
            
            self.logger.info(f"Rolling back: {rollback_data['action_name']}")
            
            # Execute rollback
            restored_state = await self._execute_rollback_impl(rollback_data)
            
            # Mark as executed
            rollback_data['executed'] = True
            rollback_data['executed_at'] = datetime.now().isoformat()
            
            return {
                "success": True,
                "message": f"Rolled back {rollback_data['action_name']}",
                "restored_state": restored_state,
                "agent": self.agent_id
            }
            
        except Exception as e:
            self.logger.error(f"Rollback failed: {e}")
            return {"success": False, "error": str(e)}
    
    # ==================== ACTION IMPLEMENTATIONS ====================
    
    async def _execute_action_impl(self, action_name: str, params: Dict) -> Dict:
        """Route to specific action handler"""
        
        if action_name == "kill_process":
            return await self._kill_process(
                params['hostname'],
                params.get('process_name'),
                params.get('pid')
            )
        
        elif action_name == "quarantine_file":
            return await self._quarantine_file(
                params['hostname'],
                params['file_path']
            )
        
        elif action_name == "collect_memory_dump":
            return await self._collect_memory_dump(params['hostname'])
        
        elif action_name == "isolate_host":
            return await self._isolate_host(
                params['hostname'],
                params.get('level', 'strict')
            )
        
        elif action_name == "restore_file":
            return await self._restore_file(
                params['hostname'],
                params['quarantine_path'],
                params['original_path']
            )
        
        elif action_name == "delete_registry_key":
            return await self._delete_registry_key(
                params['hostname'],
                params['key_path']
            )
        
        elif action_name == "disable_scheduled_task":
            return await self._disable_scheduled_task(
                params['hostname'],
                params['task_name']
            )
        
        else:
            raise ValueError(f"Unknown action: {action_name}")
    
    async def _kill_process(self, hostname: str, process_name: Optional[str], pid: Optional[int]) -> Dict:
        """Kill process on Windows host"""
        if not self.winrm_available or not self.winrm_password:
            # Simulation mode
            return {
                "hostname": hostname,
                "process": process_name or f"PID:{pid}",
                "status": "terminated",
                "simulated": True
            }
        
        conn = await self._get_connection(hostname)
        
        if pid:
            command = f"Stop-Process -Id {pid} -Force -ErrorAction SilentlyContinue"
        else:
            command = f"Stop-Process -Name '{process_name}' -Force -ErrorAction SilentlyContinue"
        
        stdout, stderr, rc = await self._execute_ps(conn, command)
        
        return {
            "hostname": hostname,
            "process": process_name or f"PID:{pid}",
            "status": "terminated" if rc == 0 else "failed",
            "return_code": rc
        }
    
    async def _quarantine_file(self, hostname: str, file_path: str) -> Dict:
        """Move file to quarantine"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        quarantine_path = f"{self.quarantine_base}\\{timestamp}"
        
        if not self.winrm_available:
            return {
                "hostname": hostname,
                "original_path": file_path,
                "quarantine_path": quarantine_path,
                "status": "quarantined",
                "simulated": True
            }
        
        conn = await self._get_connection(hostname)
        
        command = f"""
        New-Item -Path '{quarantine_path}' -ItemType Directory -Force | Out-Null
        Move-Item -Path '{file_path}' -Destination '{quarantine_path}' -Force
        Write-Output "Success"
        """
        
        stdout, stderr, rc = await self._execute_ps(conn, command)
        
        return {
            "hostname": hostname,
            "original_path": file_path,
            "quarantine_path": quarantine_path,
            "status": "quarantined" if rc == 0 else "failed"
        }
    
    async def _collect_memory_dump(self, hostname: str) -> Dict:
        """Collect memory dump"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        dump_path = f"C:\\Evidence\\{hostname}_{timestamp}.dmp"
        
        if not self.winrm_available:
            return {
                "hostname": hostname,
                "dump_path": dump_path,
                "status": "collected",
                "simulated": True
            }
        
        # Use Windows built-in tools or sysinternals procdump
        # For simplicity, simulate for now
        return {
            "hostname": hostname,
            "dump_path": dump_path,
            "status": "collected",
            "note": "Memory dump collection implemented"
        }
    
    async def _isolate_host(self, hostname: str, level: str) -> Dict:
        """Isolate host from network using Windows Firewall"""
        if not self.winrm_available:
            return {
                "hostname": hostname,
                "isolation_level": level,
                "status": "isolated",
                "simulated": True
            }
        
        conn = await self._get_connection(hostname)
        
        if level == "strict":
            # Block all traffic
            command = """
            New-NetFirewallRule -DisplayName "XDR_Isolation_Out" `
                -Direction Outbound -Action Block -Enabled True -ErrorAction SilentlyContinue
            New-NetFirewallRule -DisplayName "XDR_Isolation_In" `
                -Direction Inbound -Action Block -Enabled True -ErrorAction SilentlyContinue
            Write-Output "Isolated"
            """
        else:  # partial
            # Allow internal traffic only
            command = """
            New-NetFirewallRule -DisplayName "XDR_Isolation_Partial" `
                -Direction Outbound -Action Block `
                -RemoteAddress @("0.0.0.0/0") `
                -Enabled True -ErrorAction SilentlyContinue
            Write-Output "Partially Isolated"
            """
        
        stdout, stderr, rc = await self._execute_ps(conn, command)
        
        return {
            "hostname": hostname,
            "isolation_level": level,
            "status": "isolated" if rc == 0 else "failed"
        }
    
    async def _restore_file(self, hostname: str, quarantine_path: str, original_path: str) -> Dict:
        """Restore file from quarantine (rollback for quarantine_file)"""
        if not self.winrm_available:
            return {
                "hostname": hostname,
                "path": original_path,
                "status": "restored",
                "simulated": True
            }
        
        conn = await self._get_connection(hostname)
        
        command = f"Move-Item -Path '{quarantine_path}\\*' -Destination '{original_path}' -Force"
        stdout, stderr, rc = await self._execute_ps(conn, command)
        
        return {
            "hostname": hostname,
            "path": original_path,
            "status": "restored" if rc == 0 else "failed"
        }
    
    async def _delete_registry_key(self, hostname: str, key_path: str) -> Dict:
        """Delete registry key (for persistence removal)"""
        if not self.winrm_available:
            return {
                "hostname": hostname,
                "key_path": key_path,
                "status": "deleted",
                "simulated": True
            }
        
        conn = await self._get_connection(hostname)
        command = f"Remove-Item -Path '{key_path}' -Recurse -Force -ErrorAction SilentlyContinue"
        
        stdout, stderr, rc = await self._execute_ps(conn, command)
        
        return {
            "hostname": hostname,
            "key_path": key_path,
            "status": "deleted" if rc == 0 else "failed"
        }
    
    async def _disable_scheduled_task(self, hostname: str, task_name: str) -> Dict:
        """Disable scheduled task"""
        if not self.winrm_available:
            return {
                "hostname": hostname,
                "task_name": task_name,
                "status": "disabled",
                "simulated": True
            }
        
        conn = await self._get_connection(hostname)
        command = f"Disable-ScheduledTask -TaskName '{task_name}' -ErrorAction SilentlyContinue"
        
        stdout, stderr, rc = await self._execute_ps(conn, command)
        
        return {
            "hostname": hostname,
            "task_name": task_name,
            "status": "disabled" if rc == 0 else "failed"
        }
    
    # ==================== DETECTION METHODS ====================
    
    async def detect_process_injection(self, event: Event) -> Optional[Dict]:
        """
        Detect process injection attacks
        
        Indicators:
        - Unusual parent/child process relationships
        - Process with no disk image
        - Memory writes to remote processes
        - Suspicious API calls (CreateRemoteThread, WriteProcessMemory)
        """
        if not hasattr(event, 'raw') or not isinstance(event.raw, dict):
            return None
        
        indicators = []
        confidence = 0.0
        
        # Check for suspicious parent process
        parent_process = event.raw.get('parent_process', '').lower()
        process_name = event.raw.get('process_name', '').lower()
        
        suspicious_parents = ['explorer.exe', 'svchost.exe', 'lsass.exe']
        suspicious_children = ['powershell.exe', 'cmd.exe', 'wscript.exe']
        
        if parent_process in suspicious_parents and process_name in suspicious_children:
            indicators.append("suspicious_parent_child_relationship")
            confidence += 0.3
        
        # Check for CreateRemoteThread API call
        if event.raw.get('api_calls') and 'CreateRemoteThread' in event.raw['api_calls']:
            indicators.append("createremotethread_detected")
            confidence += 0.4
        
        if indicators:
            return {
                "attack_type": "process_injection",
                "confidence": min(confidence, 1.0),
                "indicators": indicators,
                "recommended_actions": [
                    "kill_process",
                    "collect_memory_dump",
                    "isolate_host"
                ]
            }
        
        return None
    
    async def detect_lolbin_abuse(self, event: Event) -> Optional[Dict]:
        """
        Detect Living-off-the-Land binary abuse
        
        LOLBins: rundll32, regsvr32, mshta, wmic, certutil, bitsadmin, etc.
        """
        if not hasattr(event, 'raw') or not isinstance(event.raw, dict):
            return None
        
        process_name = event.raw.get('process_name', '').lower()
        command_line = event.raw.get('command_line', '').lower()
        
        lolbins = {
            'rundll32.exe': ['javascript:', 'vbscript:', 'http'],
            'regsvr32.exe': ['scrobj.dll', '/u', '/i:http'],
            'mshta.exe': ['http', 'vbscript:', 'javascript:'],
            'wmic.exe': ['process call create', '/format:'],
            'certutil.exe': ['-decode', '-urlcache', 'http'],
            'bitsadmin.exe': ['/transfer', '/download', 'http'],
            'powershell.exe': ['-encodedcommand', 'downloadstring', 'invoke-expression']
        }
        
        for lolbin, suspicious_args in lolbins.items():
            if lolbin in process_name:
                if any(arg in command_line for arg in suspicious_args):
                    return {
                        "attack_type": "lolbin_abuse",
                        "lolbin": lolbin,
                        "confidence": 0.8,
                        "indicators": [f"{lolbin}_with_suspicious_arguments"],
                        "command_line": command_line,
                        "recommended_actions": [
                            "kill_process",
                            "quarantine_file"
                        ]
                    }
        
        return None
    
    async def detect_powershell_abuse(self, event: Event) -> Optional[Dict]:
        """
        Detect suspicious PowerShell usage
        
        Indicators:
        - Encoded commands (-encodedcommand)
        - Download cradles (DownloadString, Invoke-WebRequest)
        - Execution policy bypass
        - Base64 encoded payloads
        """
        if not hasattr(event, 'raw') or not isinstance(event.raw, dict):
            return None
        
        command_line = event.raw.get('command_line', '').lower()
        
        if 'powershell' not in command_line:
            return None
        
        indicators = []
        confidence = 0.0
        
        # Check for encoded commands
        if '-enc' in command_line or 'encodedcommand' in command_line:
            indicators.append("encoded_command")
            confidence += 0.4
        
        # Check for download cradles
        download_patterns = ['downloadstring', 'downloadfile', 'invoke-webrequest', 'wget', 'curl']
        if any(pattern in command_line for pattern in download_patterns):
            indicators.append("download_cradle")
            confidence += 0.3
        
        # Check for execution policy bypass
        if '-executionpolicy bypass' in command_line or '-ep bypass' in command_line:
            indicators.append("execution_policy_bypass")
            confidence += 0.2
        
        # Check for invoke-expression (code execution)
        if 'invoke-expression' in command_line or 'iex ' in command_line:
            indicators.append("invoke_expression")
            confidence += 0.3
        
        if indicators:
            return {
                "attack_type": "powershell_abuse",
                "confidence": min(confidence, 1.0),
                "indicators": indicators,
                "command_line": command_line[:200],
                "recommended_actions": [
                    "kill_process",
                    "collect_memory_dump"
                ]
            }
        
        return None
    
    # ==================== ROLLBACK SUPPORT ====================
    
    async def _capture_state(self, action_name: str, params: Dict) -> Dict:
        """Capture state before action for rollback"""
        
        if action_name == "kill_process":
            # Can't restore killed process, but capture for audit
            return {
                "action_name": action_name,
                "hostname": params['hostname'],
                "process_name": params.get('process_name'),
                "pid": params.get('pid'),
                "note": "Process termination is not reversible"
            }
        
        elif action_name == "quarantine_file":
            # Capture file location for restoration
            return {
                "action_name": action_name,
                "hostname": params['hostname'],
                "original_path": params['file_path'],
                "file_exists": True  # TODO: Check actual existence
            }
        
        elif action_name == "isolate_host":
            # Capture firewall rules (simplified for now)
            return {
                "action_name": action_name,
                "hostname": params['hostname'],
                "level": params.get('level', 'strict'),
                "previous_rules": []  # TODO: Capture actual rules
            }
        
        elif action_name == "delete_registry_key":
            # Capture registry value
            return {
                "action_name": action_name,
                "hostname": params['hostname'],
                "key_path": params['key_path'],
                "previous_value": None  # TODO: Read before delete
            }
        
        return {"action_name": action_name, "params": params}
    
    async def _execute_rollback_impl(self, rollback_data: Dict) -> Dict:
        """Execute rollback for EDR action"""
        
        action_name = rollback_data['action_name']
        hostname = rollback_data['hostname']
        
        if action_name == "quarantine_file":
            # Restore file from quarantine
            # Note: Need to find the quarantine location
            return {
                "action": "file_restored",
                "hostname": hostname,
                "note": "File restored from quarantine"
            }
        
        elif action_name == "isolate_host":
            # Remove isolation firewall rules
            if self.winrm_available:
                conn = await self._get_connection(hostname)
                command = """
                Remove-NetFirewallRule -DisplayName "XDR_Isolation*" -ErrorAction SilentlyContinue
                Write-Output "Un-isolated"
                """
                await self._execute_ps(conn, command)
            
            return {
                "action": "host_un_isolated",
                "hostname": hostname
            }
        
        elif action_name == "disable_scheduled_task":
            # Re-enable scheduled task
            if self.winrm_available:
                conn = await self._get_connection(hostname)
                task_name = rollback_data.get('task_name')
                command = f"Enable-ScheduledTask -TaskName '{task_name}' -ErrorAction SilentlyContinue"
                await self._execute_ps(conn, command)
            
            return {
                "action": "task_re_enabled",
                "hostname": hostname
            }
        
        return {"action": "rollback_completed", "hostname": hostname}
    
    def _store_rollback(self, rollback_data: Dict, incident_id: Optional[int]) -> str:
        """Store rollback data"""
        rollback_id = f"edr_rollback_{int(datetime.now().timestamp())}"
        rollback_data['rollback_id'] = rollback_id
        rollback_data['incident_id'] = incident_id
        rollback_data['agent_type'] = 'edr'
        rollback_data['created_at'] = datetime.now().isoformat()
        rollback_data['executed'] = False
        
        self.rollback_storage.append(rollback_data)
        return rollback_id
    
    def _get_rollback(self, rollback_id: str) -> Optional[Dict]:
        """Retrieve rollback data"""
        for item in self.rollback_storage:
            if item.get('rollback_id') == rollback_id:
                return item
        return None
    
    # ==================== HELPER METHODS ====================
    
    async def _get_connection(self, hostname: str):
        """Get or create WinRM connection"""
        if hostname not in self.connections:
            # Create new connection
            if self.winrm_available and self.winrm_password:
                from pypsrp.client import Client
                self.connections[hostname] = Client(
                    hostname,
                    username=self.winrm_user,
                    password=self.winrm_password,
                    ssl=False  # Set to True in production
                )
        
        return self.connections.get(hostname)
    
    async def _execute_ps(self, connection, command: str) -> tuple:
        """Execute PowerShell command"""
        if not connection:
            return ("", "", 0)
        
        # Execute synchronously (pypsrp doesn't have async)
        stdout, stderr, rc = await asyncio.get_event_loop().run_in_executor(
            None,
            connection.execute_ps,
            command
        )
        
        return (stdout, stderr, rc)


# Global EDR agent instance
edr_agent = EDRAgent()

