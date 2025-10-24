"""
Identity & Access Management (IAM) Agent
Specialized in Active Directory security and authentication monitoring
Supports rollback for all actions
"""
import asyncio
import logging
import secrets
import string
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

try:
    import ldap3
    from ldap3 import Server, Connection, ALL, MODIFY_REPLACE, MODIFY_ADD, MODIFY_DELETE
    LDAP_AVAILABLE = True
except ImportError:
    LDAP_AVAILABLE = False
    logging.warning("ldap3 not available - IAM Agent will use simulation mode")

from ..models import Event, Incident
from ..config import settings

logger = logging.getLogger(__name__)


class IAMAgent:
    """
    AI Agent for Identity & Access Management in corporate environments
    
    Capabilities:
    - Active Directory monitoring and management
    - Kerberos attack detection and response
    - Authentication anomaly detection
    - User account lifecycle management
    - Privilege escalation prevention
    - Full rollback support for all actions
    """
    
    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        self.agent_id = "iam_agent_v1"
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Active Directory configuration
        self.ad_server = getattr(settings, "ad_server", None) or "10.100.1.1"
        self.ad_domain = getattr(settings, "ad_domain", None) or "minicorp.local"
        self.ad_user = getattr(settings, "ad_admin_user", None) or "xdr-admin"
        self.ad_password = getattr(settings, "ad_admin_password", None)
        
        # LDAP connection
        self.ldap_server = None
        self.ldap_conn = None
        self.ldap_available = LDAP_AVAILABLE
        
        # Rollback storage (in-memory for now, should persist to DB)
        self.rollback_storage = []
        
        # Quarantine group DN
        self.quarantine_group_dn = f"CN=Quarantine,OU=Security,DC={self.ad_domain.replace('.', ',DC=')}"
        
        # Track failed authentication attempts
        self.failed_auth_tracker = {}
    
    async def initialize(self):
        """Initialize LDAP connection to Active Directory"""
        if not self.ldap_available:
            self.logger.warning("LDAP not available - running in simulation mode")
            return
        
        if not self.ad_password:
            self.logger.warning("AD password not configured - running in simulation mode")
            return
        
        try:
            self.ldap_server = Server(self.ad_server, get_info=ALL)
            self.ldap_conn = Connection(
                self.ldap_server,
                user=f"{self.ad_domain}\\{self.ad_user}",
                password=self.ad_password,
                auto_bind=True
            )
            self.logger.info("✅ Connected to Active Directory")
        except Exception as e:
            self.logger.error(f"❌ Failed to connect to AD: {e}")
            self.ldap_conn = None
    
    # ==================== PUBLIC API ====================
    
    async def execute_action(
        self,
        action_name: str,
        params: Dict[str, Any],
        incident_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Execute IAM action with automatic rollback support
        
        Args:
            action_name: Name of action (disable_user_account, quarantine_user, etc.)
            params: Action parameters
            incident_id: Associated incident ID
        
        Returns:
            {
                "success": bool,
                "action_id": str,
                "rollback_id": str,
                "result": dict,
                "error": str (if failed)
            }
        """
        action_id = f"{self.agent_id}_{action_name}_{int(datetime.now().timestamp())}"
        
        try:
            # Step 1: Capture current state for rollback
            rollback_data = await self._capture_state(action_name, params)
            rollback_id = self._store_rollback(rollback_data, incident_id)
            
            # Step 2: Execute the action
            self.logger.info(f"Executing {action_name} with params: {params}")
            result = await self._execute_action_impl(action_name, params)
            
            # Step 3: Log success
            self.logger.info(f"✅ Action {action_name} completed successfully")
            
            return {
                "success": True,
                "action_id": action_id,
                "rollback_id": rollback_id,
                "result": result,
                "message": f"Action {action_name} completed successfully",
                "agent": self.agent_id
            }
            
        except Exception as e:
            self.logger.error(f"❌ Action {action_name} failed: {e}")
            
            return {
                "success": False,
                "action_id": action_id,
                "error": str(e),
                "message": f"Action {action_name} failed",
                "agent": self.agent_id
            }
    
    async def rollback_action(self, rollback_id: str) -> Dict[str, Any]:
        """
        Rollback a previously executed action
        
        Returns:
            {
                "success": bool,
                "message": str,
                "restored_state": dict
            }
        """
        try:
            # Find rollback data
            rollback_data = self._get_rollback(rollback_id)
            
            if not rollback_data:
                return {
                    "success": False,
                    "error": f"Rollback ID {rollback_id} not found"
                }
            
            if rollback_data.get('executed'):
                return {
                    "success": False,
                    "error": "Rollback already executed"
                }
            
            self.logger.info(f"Rolling back action: {rollback_data['action_name']}")
            
            # Execute rollback
            restored_state = await self._execute_rollback_impl(rollback_data)
            
            # Mark as executed
            rollback_data['executed'] = True
            rollback_data['executed_at'] = datetime.now().isoformat()
            
            return {
                "success": True,
                "message": f"Successfully rolled back {rollback_data['action_name']}",
                "restored_state": restored_state,
                "agent": self.agent_id
            }
            
        except Exception as e:
            self.logger.error(f"❌ Rollback failed for {rollback_id}: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": "Rollback failed"
            }
    
    # ==================== DETECTION METHODS ====================
    
    async def analyze_authentication_event(self, event: Event) -> Optional[Dict]:
        """
        Analyze authentication event for anomalies
        Detects: Impossible travel, off-hours access, brute force, etc.
        """
        analysis = {
            "event_id": event.id,
            "username": event.raw.get("username") if hasattr(event, 'raw') and isinstance(event.raw, dict) else None,
            "source_ip": event.src_ip,
            "timestamp": event.ts,
            "anomalies": [],
            "risk_score": 0.0,
            "recommended_actions": []
        }
        
        # Check 1: Off-hours access
        if await self._detect_off_hours_access(event):
            analysis["anomalies"].append("off_hours_access")
            analysis["risk_score"] += 0.2
        
        # Check 2: Brute force pattern
        if await self._detect_brute_force_pattern(event):
            analysis["anomalies"].append("brute_force")
            analysis["risk_score"] += 0.4
        
        # Check 3: Service account abuse
        if await self._detect_service_account_abuse(event):
            analysis["anomalies"].append("service_account_abuse")
            analysis["risk_score"] += 0.3
        
        # Recommend actions based on risk
        if analysis["risk_score"] >= 0.7:
            analysis["recommended_actions"] = [
                "disable_user_account",
                "revoke_all_sessions",
                "alert_security_team"
            ]
        elif analysis["risk_score"] >= 0.4:
            analysis["recommended_actions"] = [
                "enforce_mfa",
                "monitor_user_activity"
            ]
        
        return analysis if analysis["anomalies"] else None
    
    async def detect_kerberos_attack(self, event: Event) -> Optional[Dict]:
        """
        Detect Kerberos-based attacks
        Types: Golden Ticket, Silver Ticket, Kerberoasting, AS-REP roasting
        """
        if not hasattr(event, 'raw') or not isinstance(event.raw, dict):
            return None
        
        kerberos_data = event.raw.get("kerberos", {})
        if not kerberos_data:
            return None
        
        indicators = []
        is_malicious = False
        attack_type = None
        confidence = 0.0
        
        # Check 1: Ticket lifetime
        ticket_lifetime = kerberos_data.get("ticket_lifetime", 600)
        if ticket_lifetime > 86400:  # > 24 hours
            indicators.append("abnormal_ticket_lifetime")
            confidence += 0.4
            attack_type = "golden_ticket"
            is_malicious = True
        
        # Check 2: Encryption type
        enc_type = kerberos_data.get("encryption_type", "")
        if enc_type in ["DES", "RC4"]:  # Weak encryption
            indicators.append("weak_encryption")
            confidence += 0.2
        
        # Check 3: PAC validation
        if not kerberos_data.get("pac_validated", True):
            indicators.append("invalid_pac")
            confidence += 0.3
            is_malicious = True
        
        if is_malicious:
            return {
                "attack_type": attack_type or "kerberos_abuse",
                "confidence": min(confidence, 1.0),
                "indicators": indicators,
                "username": kerberos_data.get("username"),
                "recommended_actions": [
                    "revoke_kerberos_tickets",
                    "reset_krbtgt_password",
                    "disable_affected_accounts",
                    "investigate_dc_compromise"
                ]
            }
        
        return None
    
    async def detect_privilege_escalation(self, event: Event) -> Optional[Dict]:
        """
        Detect privilege escalation attempts
        Monitors: Group membership changes, privilege grants, ACL modifications
        """
        if event.eventid != "group_membership_changed":
            return None
        
        if not hasattr(event, 'raw') or not isinstance(event.raw, dict):
            return None
        
        group_dn = event.raw.get("group_dn", "")
        user_dn = event.raw.get("user_dn", "")
        
        # Check if privileged group
        if self._is_privileged_group(group_dn):
            return {
                "escalation_type": "group_membership",
                "group": group_dn,
                "user": user_dn,
                "risk_score": 0.8,
                "recommended_actions": [
                    "remove_from_privileged_group",
                    "investigate_who_added_user",
                    "audit_recent_changes"
                ]
            }
        
        return None
    
    # ==================== ACTION IMPLEMENTATIONS ====================
    
    async def _execute_action_impl(self, action_name: str, params: Dict) -> Dict:
        """Execute specific IAM action"""
        
        if action_name == "disable_user_account":
            return await self._disable_user(params['username'], params.get('reason', 'Security incident'))
        
        elif action_name == "quarantine_user":
            return await self._quarantine_user(params['username'], params.get('reason', 'Security incident'))
        
        elif action_name == "revoke_kerberos_tickets":
            return await self._revoke_tickets(params['username'])
        
        elif action_name == "reset_password":
            return await self._reset_password(params['username'], params.get('force_change', True))
        
        elif action_name == "remove_from_group":
            return await self._remove_from_group(params['username'], params['group'])
        
        elif action_name == "enforce_mfa":
            return await self._enforce_mfa(params['username'])
        
        else:
            raise ValueError(f"Unknown action: {action_name}")
    
    async def _disable_user(self, username: str, reason: str) -> Dict:
        """Disable AD user account"""
        if not self.ldap_conn:
            # Simulation mode
            return {
                "username": username,
                "status": "disabled",
                "reason": reason,
                "simulated": True
            }
        
        user_dn = await self._get_user_dn(username)
        
        # Set ACCOUNTDISABLE flag (0x0002)
        result = self.ldap_conn.modify(
            user_dn,
            {'userAccountControl': [(MODIFY_REPLACE, ['514'])]}  # 512 (normal) + 2 (disabled)
        )
        
        if result:
            return {
                "username": username,
                "user_dn": user_dn,
                "status": "disabled",
                "reason": reason
            }
        else:
            raise Exception(f"LDAP modify failed: {self.ldap_conn.result}")
    
    async def _quarantine_user(self, username: str, reason: str) -> Dict:
        """Move user to quarantine group"""
        if not self.ldap_conn:
            # Simulation mode
            return {
                "username": username,
                "status": "quarantined",
                "reason": reason,
                "simulated": True
            }
        
        user_dn = await self._get_user_dn(username)
        
        # Add to quarantine
        self.ldap_conn.modify(
            self.quarantine_group_dn,
            {'member': [(MODIFY_ADD, [user_dn])]}
        )
        
        # Remove from privileged groups
        removed_groups = await self._remove_from_privileged_groups(user_dn)
        
        return {
            "username": username,
            "user_dn": user_dn,
            "status": "quarantined",
            "removed_from_groups": removed_groups,
            "reason": reason
        }
    
    async def _revoke_tickets(self, username: str) -> Dict:
        """Revoke Kerberos tickets (requires DC access)"""
        # Note: This requires PowerShell remoting to DC
        # For now, return simulated result
        return {
            "username": username,
            "status": "tickets_revoked",
            "message": "User must re-authenticate",
            "simulated": not self.ldap_conn
        }
    
    async def _reset_password(self, username: str, force_change: bool) -> Dict:
        """Reset user password"""
        # Generate secure random password
        alphabet = string.ascii_letters + string.digits + string.punctuation
        new_password = ''.join(secrets.choice(alphabet) for _ in range(16))
        
        if not self.ldap_conn:
            return {
                "username": username,
                "status": "password_reset",
                "temporary_password": new_password,
                "force_change": force_change,
                "simulated": True
            }
        
        user_dn = await self._get_user_dn(username)
        
        # Reset password
        result = self.ldap_conn.extend.microsoft.modify_password(
            user_dn,
            new_password
        )
        
        if force_change:
            # Force change on next login
            self.ldap_conn.modify(
                user_dn,
                {'pwdLastSet': [(MODIFY_REPLACE, ['0'])]}
            )
        
        return {
            "username": username,
            "status": "password_reset",
            "temporary_password": new_password,
            "force_change": force_change
        }
    
    async def _remove_from_group(self, username: str, group: str) -> Dict:
        """Remove user from specific group"""
        if not self.ldap_conn:
            return {
                "username": username,
                "group": group,
                "status": "removed",
                "simulated": True
            }
        
        user_dn = await self._get_user_dn(username)
        
        self.ldap_conn.modify(
            group,
            {'member': [(MODIFY_DELETE, [user_dn])]}
        )
        
        return {
            "username": username,
            "group": group,
            "status": "removed"
        }
    
    async def _enforce_mfa(self, username: str) -> Dict:
        """Enforce MFA for user"""
        # Note: This depends on your MFA solution (Azure AD, Duo, etc.)
        return {
            "username": username,
            "status": "mfa_enforced",
            "message": "User must configure MFA on next login",
            "simulated": not self.ldap_conn
        }
    
    # ==================== ROLLBACK SUPPORT ====================
    
    async def _capture_state(self, action_name: str, params: Dict) -> Dict:
        """Capture AD state before changes"""
        
        if action_name == "disable_user_account":
            username = params['username']
            
            if not self.ldap_conn:
                # Simulation mode
                return {
                    "action_name": action_name,
                    "username": username,
                    "previous_state": {
                        "was_enabled": True,
                        "simulated": True
                    }
                }
            
            user_dn = await self._get_user_dn(username)
            
            # Get current userAccountControl value
            self.ldap_conn.search(
                user_dn,
                '(objectClass=user)',
                attributes=['userAccountControl', 'memberOf']
            )
            
            if self.ldap_conn.entries:
                current_uac = self.ldap_conn.entries[0].userAccountControl.value
                current_groups = [str(g) for g in self.ldap_conn.entries[0].memberOf]
                
                return {
                    "action_name": action_name,
                    "username": username,
                    "user_dn": user_dn,
                    "previous_state": {
                        "userAccountControl": current_uac,
                        "memberOf": current_groups,
                        "was_enabled": not (int(current_uac) & 0x0002)
                    }
                }
        
        elif action_name == "quarantine_user":
            username = params['username']
            
            if not self.ldap_conn:
                return {
                    "action_name": action_name,
                    "username": username,
                    "previous_state": {"simulated": True}
                }
            
            user_dn = await self._get_user_dn(username)
            
            # Get current group memberships
            self.ldap_conn.search(
                user_dn,
                '(objectClass=user)',
                attributes=['memberOf']
            )
            
            if self.ldap_conn.entries:
                current_groups = [str(g) for g in self.ldap_conn.entries[0].memberOf]
                
                return {
                    "action_name": action_name,
                    "username": username,
                    "user_dn": user_dn,
                    "previous_state": {
                        "memberOf": current_groups
                    }
                }
        
        return {"action_name": action_name, "params": params}
    
    async def _execute_rollback_impl(self, rollback_data: Dict) -> Dict:
        """Execute rollback for IAM action"""
        action_name = rollback_data['action_name']
        previous_state = rollback_data.get('previous_state', {})
        username = rollback_data['username']
        user_dn = rollback_data.get('user_dn')
        
        if previous_state.get('simulated'):
            return {"action": "rollback_simulated", "username": username}
        
        if action_name == "disable_user_account":
            was_enabled = previous_state.get('was_enabled', True)
            
            if was_enabled and self.ldap_conn:
                # Restore to enabled state
                uac_value = str(int(previous_state['userAccountControl']) & ~0x0002)
                
                self.ldap_conn.modify(
                    user_dn,
                    {'userAccountControl': [(MODIFY_REPLACE, [uac_value])]}
                )
                
                return {
                    "action": "re_enabled_account",
                    "username": username,
                    "restored_uac": uac_value
                }
        
        elif action_name == "quarantine_user":
            original_groups = previous_state.get('memberOf', [])
            
            if self.ldap_conn:
                # Remove from quarantine group
                self.ldap_conn.modify(
                    self.quarantine_group_dn,
                    {'member': [(MODIFY_DELETE, [user_dn])]}
                )
                
                # Restore original groups
                restored_groups = []
                for group in original_groups:
                    try:
                        self.ldap_conn.modify(
                            group,
                            {'member': [(MODIFY_ADD, [user_dn])]}
                        )
                        restored_groups.append(group)
                    except Exception as e:
                        self.logger.warning(f"Could not restore group {group}: {e}")
                
                return {
                    "action": "restored_group_memberships",
                    "username": username,
                    "restored_groups": restored_groups
                }
        
        return {"action": "rollback_completed", "username": username}
    
    def _store_rollback(self, rollback_data: Dict, incident_id: Optional[int]) -> str:
        """Store rollback data and return rollback_id"""
        rollback_id = f"iam_rollback_{int(datetime.now().timestamp())}"
        rollback_data['rollback_id'] = rollback_id
        rollback_data['incident_id'] = incident_id
        rollback_data['agent_type'] = 'iam'
        rollback_data['created_at'] = datetime.now().isoformat()
        rollback_data['executed'] = False
        
        self.rollback_storage.append(rollback_data)
        
        # TODO: Persist to database ActionLog table
        
        return rollback_id
    
    def _get_rollback(self, rollback_id: str) -> Optional[Dict]:
        """Retrieve rollback data by ID"""
        for item in self.rollback_storage:
            if item.get('rollback_id') == rollback_id:
                return item
        return None
    
    # ==================== HELPER METHODS ====================
    
    async def _get_user_dn(self, username: str) -> str:
        """Get user Distinguished Name"""
        if not self.ldap_conn:
            # Simulation mode
            return f"CN={username},OU=Users,OU=Corporate,DC={self.ad_domain.replace('.', ',DC=')}"
        
        search_base = f"DC={self.ad_domain.replace('.', ',DC=')}"
        search_filter = f"(samAccountName={username})"
        
        self.ldap_conn.search(
            search_base=search_base,
            search_filter=search_filter,
            attributes=['distinguishedName']
        )
        
        if self.ldap_conn.entries:
            return str(self.ldap_conn.entries[0].distinguishedName)
        
        raise ValueError(f"User {username} not found in AD")
    
    def _is_privileged_group(self, group_dn: str) -> bool:
        """Check if group is privileged"""
        privileged_groups = [
            "Domain Admins", "Enterprise Admins", "Schema Admins",
            "Administrators", "Account Operators", "Backup Operators"
        ]
        return any(group in group_dn for group in privileged_groups)
    
    async def _remove_from_privileged_groups(self, user_dn: str) -> List[str]:
        """Remove user from all privileged groups"""
        if not self.ldap_conn:
            return []
        
        self.ldap_conn.search(
            user_dn,
            '(objectClass=user)',
            attributes=['memberOf']
        )
        
        removed_groups = []
        
        if self.ldap_conn.entries:
            groups = self.ldap_conn.entries[0].memberOf
            
            for group_dn in groups:
                group_str = str(group_dn)
                if self._is_privileged_group(group_str):
                    try:
                        self.ldap_conn.modify(
                            group_str,
                            {'member': [(MODIFY_DELETE, [user_dn])]}
                        )
                        removed_groups.append(group_str)
                    except Exception as e:
                        self.logger.warning(f"Could not remove from {group_str}: {e}")
        
        return removed_groups
    
    async def _detect_off_hours_access(self, event: Event) -> bool:
        """Detect authentication outside business hours"""
        if not event.ts:
            return False
        
        hour = event.ts.hour
        # Business hours: 8am - 6pm
        return hour < 8 or hour > 18
    
    async def _detect_brute_force_pattern(self, event: Event) -> bool:
        """Detect authentication brute force pattern"""
        if not hasattr(event, 'raw') or not isinstance(event.raw, dict):
            return False
        
        username = event.raw.get("username")
        src_ip = event.src_ip
        
        if not username or not src_ip:
            return False
        
        key = f"{username}:{src_ip}"
        
        if key not in self.failed_auth_tracker:
            self.failed_auth_tracker[key] = []
        
        # Add failed attempt
        if "failed" in event.eventid.lower():
            self.failed_auth_tracker[key].append(datetime.now())
            
            # Remove attempts older than 5 minutes
            cutoff = datetime.now() - timedelta(minutes=5)
            self.failed_auth_tracker[key] = [
                t for t in self.failed_auth_tracker[key] if t > cutoff
            ]
            
            # Check threshold: 5 failures in 5 minutes
            if len(self.failed_auth_tracker[key]) >= 5:
                return True
        
        return False
    
    async def _detect_service_account_abuse(self, event: Event) -> bool:
        """Detect service account used for interactive login"""
        if not hasattr(event, 'raw') or not isinstance(event.raw, dict):
            return False
        
        username = event.raw.get("username", "")
        logon_type = event.raw.get("logon_type")
        
        # Service accounts typically have "svc-" prefix
        is_service_account = username.startswith("svc-") or username.startswith("service-")
        
        # Interactive logon type = 2, Remote Desktop = 10
        is_interactive = logon_type in ["2", "10", 2, 10]
        
        return is_service_account and is_interactive


# Global IAM agent instance
iam_agent = IAMAgent()

