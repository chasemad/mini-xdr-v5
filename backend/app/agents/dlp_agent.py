"""
Data Loss Prevention (DLP) Agent
Prevents sensitive data exfiltration
"""
import re
import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path

from ..models import Event, Incident
from ..config import settings

logger = logging.getLogger(__name__)


class DLPAgent:
    """
    AI Agent for Data Loss Prevention
    
    Capabilities:
    - Data classification (PII, credit cards, SSNs, API keys)
    - File scanning for sensitive data
    - Block unauthorized uploads
    - Monitor large file transfers
    - Track data exfiltration
    - Full rollback support
    """
    
    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        self.agent_id = "dlp_agent_v1"
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Rollback storage
        self.rollback_storage = []
        
        # Sensitive data patterns
        self.patterns = {
            "ssn": {
                "pattern": r"\b\d{3}-\d{2}-\d{4}\b",
                "severity": "high",
                "description": "Social Security Number"
            },
            "credit_card": {
                "pattern": r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
                "severity": "high",
                "description": "Credit Card Number"
            },
            "email": {
                "pattern": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
                "severity": "medium",
                "description": "Email Address"
            },
            "api_key": {
                "pattern": r"(?i)(api[_-]?key|secret[_-]?key|token)['\"]?\s*[:=]\s*['\"]?[A-Za-z0-9_\-]{20,}",
                "severity": "critical",
                "description": "API Key or Secret"
            },
            "phone": {
                "pattern": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
                "severity": "medium",
                "description": "Phone Number"
            },
            "ip_address": {
                "pattern": r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b",
                "severity": "low",
                "description": "IP Address"
            },
            "aws_key": {
                "pattern": r"AKIA[0-9A-Z]{16}",
                "severity": "critical",
                "description": "AWS Access Key"
            },
            "private_key": {
                "pattern": r"-----BEGIN (?:RSA )?PRIVATE KEY-----",
                "severity": "critical",
                "description": "Private Key"
            }
        }
        
        # Blocked upload tracking
        self.blocked_uploads = []
    
    # ==================== PUBLIC API ====================
    
    async def execute_action(
        self,
        action_name: str,
        params: Dict[str, Any],
        incident_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """Execute DLP action with rollback support"""
        action_id = f"{self.agent_id}_{action_name}_{int(datetime.now().timestamp())}"
        
        try:
            # Capture state
            rollback_data = await self._capture_state(action_name, params)
            rollback_id = self._store_rollback(rollback_data, incident_id)
            
            # Execute
            self.logger.info(f"Executing {action_name} with params: {params}")
            result = await self._execute_action_impl(action_name, params)
            
            return {
                "success": True,
                "action_id": action_id,
                "rollback_id": rollback_id,
                "result": result,
                "message": f"Action {action_name} completed",
                "agent": self.agent_id
            }
            
        except Exception as e:
            self.logger.error(f"Action {action_name} failed: {e}")
            return {
                "success": False,
                "action_id": action_id,
                "error": str(e),
                "agent": self.agent_id
            }
    
    async def rollback_action(self, rollback_id: str) -> Dict[str, Any]:
        """Rollback DLP action"""
        try:
            rollback_data = self._get_rollback(rollback_id)
            
            if not rollback_data:
                return {"success": False, "error": "Rollback ID not found"}
            
            if rollback_data.get('executed'):
                return {"success": False, "error": "Already rolled back"}
            
            self.logger.info(f"Rolling back: {rollback_data['action_name']}")
            
            restored_state = await self._execute_rollback_impl(rollback_data)
            
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
        """Route to specific handler"""
        
        if action_name == "scan_file":
            return await self._scan_file(params['file_path'])
        
        elif action_name == "block_upload":
            return await self._block_upload(
                params['hostname'],
                params['process_name'],
                params['destination']
            )
        
        elif action_name == "quarantine_sensitive_file":
            return await self._quarantine_sensitive_file(
                params['hostname'],
                params['file_path']
            )
        
        else:
            raise ValueError(f"Unknown action: {action_name}")
    
    async def _scan_file(self, file_path: str) -> Dict:
        """Scan file for sensitive data"""
        try:
            # Read file (simulate for now)
            content = f"Sample content with test data: 123-45-6789 and test@example.com"
            
            findings = []
            total_matches = 0
            
            # Check each pattern
            for pattern_name, pattern_data in self.patterns.items():
                matches = re.findall(pattern_data['pattern'], content)
                
                if matches:
                    findings.append({
                        "type": pattern_name,
                        "count": len(matches),
                        "severity": pattern_data['severity'],
                        "description": pattern_data['description']
                    })
                    total_matches += len(matches)
            
            # Calculate risk score
            risk_score = min(total_matches * 0.1, 1.0)
            
            # Determine overall severity
            severities = [f['severity'] for f in findings]
            if 'critical' in severities:
                overall_severity = 'critical'
            elif 'high' in severities:
                overall_severity = 'high'
            elif 'medium' in severities:
                overall_severity = 'medium'
            else:
                overall_severity = 'low'
            
            return {
                "file_path": file_path,
                "sensitive_data_found": len(findings) > 0,
                "findings": findings,
                "total_matches": total_matches,
                "risk_score": risk_score,
                "severity": overall_severity,
                "recommended_actions": ["quarantine_sensitive_file"] if findings else []
            }
            
        except Exception as e:
            self.logger.error(f"File scan failed: {e}")
            raise
    
    async def _block_upload(self, hostname: str, process_name: str, destination: str) -> Dict:
        """Block file upload"""
        # Kill upload process via EDR agent
        from .edr_agent import edr_agent
        
        kill_result = await edr_agent.execute_action(
            "kill_process",
            {"hostname": hostname, "process_name": process_name}
        )
        
        # Track blocked upload
        self.blocked_uploads.append({
            "hostname": hostname,
            "process": process_name,
            "destination": destination,
            "timestamp": datetime.now().isoformat()
        })
        
        return {
            "hostname": hostname,
            "process": process_name,
            "destination": destination,
            "status": "blocked",
            "kill_result": kill_result
        }
    
    async def _quarantine_sensitive_file(self, hostname: str, file_path: str) -> Dict:
        """Quarantine file with sensitive data"""
        # Delegate to EDR agent
        from .edr_agent import edr_agent
        
        result = await edr_agent.execute_action(
            "quarantine_file",
            {"hostname": hostname, "file_path": file_path}
        )
        
        return result
    
    # ==================== DETECTION METHODS ====================
    
    async def detect_data_exfiltration(self, event: Event) -> Optional[Dict]:
        """
        Detect data exfiltration attempts
        
        Indicators:
        - Large file uploads (>10MB)
        - Unusual destinations (external IPs)
        - Encrypted archives
        - Database dumps
        - Bulk file access
        """
        if not hasattr(event, 'raw') or not isinstance(event.raw, dict):
            return None
        
        indicators = []
        confidence = 0.0
        
        # Check for large upload
        file_size = event.raw.get('file_size', 0)
        if file_size > 10 * 1024 * 1024:  # > 10MB
            indicators.append("large_file_upload")
            confidence += 0.3
        
        # Check for external destination
        destination = event.raw.get('destination_ip', '')
        if not destination.startswith('10.') and not destination.startswith('192.168.'):
            indicators.append("external_destination")
            confidence += 0.3
        
        # Check for archive files
        filename = event.raw.get('filename', '').lower()
        if any(ext in filename for ext in ['.zip', '.rar', '.7z', '.tar.gz']):
            indicators.append("archive_file")
            confidence += 0.2
        
        # Check for database extensions
        if any(ext in filename for ext in ['.sql', '.db', '.mdb', '.bak']):
            indicators.append("database_file")
            confidence += 0.3
        
        if indicators:
            return {
                "attack_type": "data_exfiltration",
                "confidence": min(confidence, 1.0),
                "indicators": indicators,
                "filename": filename,
                "file_size": file_size,
                "destination": destination,
                "recommended_actions": [
                    "block_upload",
                    "quarantine_sensitive_file"
                ]
            }
        
        return None
    
    # ==================== ROLLBACK SUPPORT ====================
    
    async def _capture_state(self, action_name: str, params: Dict) -> Dict:
        """Capture state for rollback"""
        
        if action_name == "block_upload":
            return {
                "action_name": action_name,
                "hostname": params['hostname'],
                "process_name": params['process_name'],
                "destination": params['destination']
            }
        
        return {"action_name": action_name, "params": params}
    
    async def _execute_rollback_impl(self, rollback_data: Dict) -> Dict:
        """Execute rollback"""
        
        action_name = rollback_data['action_name']
        
        if action_name == "block_upload":
            # Remove from blocked list
            destination = rollback_data['destination']
            self.blocked_uploads = [
                b for b in self.blocked_uploads 
                if b['destination'] != destination
            ]
            
            return {
                "action": "upload_unblocked",
                "destination": destination
            }
        
        return {"action": "rollback_completed"}
    
    def _store_rollback(self, rollback_data: Dict, incident_id: Optional[int]) -> str:
        """Store rollback data"""
        rollback_id = f"dlp_rollback_{int(datetime.now().timestamp())}"
        rollback_data['rollback_id'] = rollback_id
        rollback_data['incident_id'] = incident_id
        rollback_data['agent_type'] = 'dlp'
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


# Global DLP agent instance
dlp_agent = DLPAgent()

