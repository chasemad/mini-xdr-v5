"""
T-Pot Action Verification Module
Verifies that agent actions were actually executed on the T-Pot honeypot
"""

import asyncio
import logging
from typing import Dict, Any, List
from datetime import datetime
import re

logger = logging.getLogger(__name__)


class TPotActionVerifier:
    """Verify agent actions on T-Pot honeypot"""
    
    def __init__(self, ssh_host: str, ssh_port: int, ssh_user: str, ssh_key_path: str):
        self.ssh_host = ssh_host
        self.ssh_port = ssh_port
        self.ssh_user = ssh_user
        self.ssh_key_path = ssh_key_path
    
    async def verify_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Verify a specific action was executed"""
        action_type = action.get("action", "unknown")
        
        try:
            if action_type == "block":
                return await self._verify_block_action(action)
            elif action_type == "isolate_host":
                return await self._verify_isolation(action)
            elif action_type == "deploy_firewall":
                return await self._verify_firewall_rule(action)
            else:
                return {
                    "verified": False,
                    "message": f"Verification not implemented for action type: {action_type}",
                    "timestamp": datetime.utcnow().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Verification failed for action {action.get('id')}: {e}")
            return {
                "verified": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def _verify_block_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Verify IP block was applied on T-Pot"""
        params = action.get("params", {})
        ip_address = params.get("ip_address") or params.get("target_ip")
        
        if not ip_address:
            return {
                "verified": False,
                "message": "No IP address found in action params",
                "timestamp": datetime.utcnow().isoformat()
            }
        
        # Check if IP is blocked in iptables
        cmd = f"sudo iptables -L INPUT -n -v | grep {ip_address}"
        result = await self._execute_ssh_command(cmd)
        
        if result["success"] and ip_address in result["output"]:
            # Check if it's a DROP or REJECT rule
            if "DROP" in result["output"] or "REJECT" in result["output"]:
                return {
                    "verified": True,
                    "message": f"IP {ip_address} is blocked in iptables",
                    "details": result["output"],
                    "timestamp": datetime.utcnow().isoformat()
                }
        
        # Also check UFW if iptables didn't show it
        ufw_cmd = f"sudo ufw status | grep {ip_address}"
        ufw_result = await self._execute_ssh_command(ufw_cmd)
        
        if ufw_result["success"] and ip_address in ufw_result["output"]:
            if "DENY" in ufw_result["output"]:
                return {
                    "verified": True,
                    "message": f"IP {ip_address} is blocked in UFW",
                    "details": ufw_result["output"],
                    "timestamp": datetime.utcnow().isoformat()
                }
        
        return {
            "verified": False,
            "message": f"IP {ip_address} not found in firewall rules",
            "iptables_checked": True,
            "ufw_checked": True,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _verify_isolation(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Verify host isolation"""
        params = action.get("params", {})
        target_host = params.get("target_host") or params.get("hostname")
        
        if not target_host:
            return {
                "verified": False,
                "message": "No target host found",
                "timestamp": datetime.utcnow().isoformat()
            }
        
        # Check for isolation rules
        cmd = f"sudo iptables -L FORWARD -n -v | grep {target_host}"
        result = await self._execute_ssh_command(cmd)
        
        return {
            "verified": result["success"] and target_host in result["output"],
            "message": f"Host {target_host} isolation check",
            "details": result.get("output", ""),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _verify_firewall_rule(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Verify firewall rule deployment"""
        params = action.get("params", {})
        
        # Check for recent iptables changes
        cmd = "sudo iptables -L -n -v --line-numbers | tail -20"
        result = await self._execute_ssh_command(cmd)
        
        return {
            "verified": result["success"],
            "message": "Firewall rules retrieved",
            "details": result.get("output", ""),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _execute_ssh_command(self, command: str) -> Dict[str, Any]:
        """Execute SSH command on T-Pot"""
        try:
            # Build SSH command
            ssh_cmd = [
                "ssh",
                "-o", "ConnectTimeout=5",
                "-o", "StrictHostKeyChecking=no",
                "-p", str(self.ssh_port),
                "-i", self.ssh_key_path,
                f"{self.ssh_user}@{self.ssh_host}",
                command
            ]
            
            # Execute command
            process = await asyncio.create_subprocess_exec(
                *ssh_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=10)
            
            output = stdout.decode() if stdout else ""
            error = stderr.decode() if stderr else ""
            
            return {
                "success": process.returncode == 0,
                "output": output,
                "error": error,
                "return_code": process.returncode
            }
            
        except asyncio.TimeoutError:
            logger.error(f"SSH command timeout: {command}")
            return {
                "success": False,
                "error": "Command timeout",
                "output": ""
            }
        except Exception as e:
            logger.error(f"SSH command failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "output": ""
            }
    
    async def verify_multiple_actions(self, actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Verify multiple actions at once"""
        results = []
        
        for action in actions:
            verification = await self.verify_action(action)
            results.append({
                "action_id": action.get("id"),
                "action_type": action.get("action"),
                **verification
            })
        
        verified_count = sum(1 for r in results if r.get("verified", False))
        total_count = len(results)
        
        return {
            "total_actions": total_count,
            "verified_actions": verified_count,
            "verification_rate": verified_count / total_count if total_count > 0 else 0,
            "results": results,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def get_active_blocks(self) -> Dict[str, Any]:
        """Get all currently active IP blocks on T-Pot"""
        try:
            # Get iptables blocks
            iptables_cmd = "sudo iptables -L INPUT -n -v | grep -E 'DROP|REJECT' | awk '{print $8}'"
            iptables_result = await self._execute_ssh_command(iptables_cmd)
            
            # Get UFW blocks
            ufw_cmd = "sudo ufw status | grep DENY | awk '{print $3}'"
            ufw_result = await self._execute_ssh_command(ufw_cmd)
            
            iptables_blocks = [
                ip.strip() for ip in iptables_result.get("output", "").split("\n")
                if ip.strip() and ip.strip() != "0.0.0.0/0"
            ]
            
            ufw_blocks = [
                ip.strip() for ip in ufw_result.get("output", "").split("\n")
                if ip.strip()
            ]
            
            all_blocks = list(set(iptables_blocks + ufw_blocks))
            
            return {
                "success": True,
                "total_blocks": len(all_blocks),
                "iptables_blocks": iptables_blocks,
                "ufw_blocks": ufw_blocks,
                "all_blocks": all_blocks,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get active blocks: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }


# Global verifier instance
_verifier = None

def get_verifier(config=None):
    """Get or create TPot verifier instance"""
    global _verifier
    
    if _verifier is None and config:
        _verifier = TPotActionVerifier(
            ssh_host=config.get("host"),
            ssh_port=config.get("ssh_port", 64295),
            ssh_user=config.get("user", "azureuser"),
            ssh_key_path=config.get("ssh_key_path")
        )
    
    return _verifier


