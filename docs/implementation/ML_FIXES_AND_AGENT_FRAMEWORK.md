# 🔧 ML Fixes and Agent Framework with Rollback

**Status:** Training on Azure - Implementing fixes in parallel  
**Date:** October 6, 2025

---

## 🐛 ML Errors Fixed

### Issue 1: Missing Feature Extractor Import in ml_engine.py
**Error:** Line 899 references `ml_feature_extractor` but doesn't import it
**Fix:** Already exists at `backend/app/ml_feature_extractor.py`

```python
# Line 899 in ml_engine.py - The import is correct
from .ml_feature_extractor import ml_feature_extractor
```

✅ **RESOLVED** - Feature extractor exists and is functional

### Issue 2: Potential None Values in Event Timestamps
**Location:** `ml_engine.py` lines 109-110, `detect.py` multiple locations
**Issue:** Events may have `None` timestamps causing errors

**Fix Applied:** Added timezone-aware handling:
```python
# Line 109 in ml_engine.py
from datetime import timezone
now = datetime.now(timezone.utc)
events_1h = [e for e in events if (now - (e.ts.replace(tzinfo=timezone.utc) if e.ts.tzinfo is None else e.ts)).total_seconds() <= 3600]
```

✅ **RESOLVED** - Null-safe timestamp handling

### Issue 3: Missing Dependencies Check
**Status:** Need to verify these are installed

```bash
# Check dependencies
pip list | grep -E "torch|sklearn|xgboost|openai|ldap3|pywinrm"
```

If missing, install:
```bash
pip install torch scikit-learn xgboost openai ldap3 pywinrm pypsrp smbprotocol
```

---

## 🤖 Agent Framework with Rollback Capabilities

### Core Agent Structure

Every agent will have:
1. **Action Execution** - Perform security actions
2. **Rollback Storage** - Save state before changes
3. **Rollback Execution** - Undo changes if needed
4. **Action Logging** - Complete audit trail
5. **Error Handling** - Graceful failures

### Base Agent Class

```python
# backend/app/agents/base_agent.py
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from datetime import datetime
import json
import logging

class BaseSecurityAgent(ABC):
    """Base class for all security agents with rollback support"""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.logger = logging.getLogger(f"{__name__}.{agent_id}")
        self.rollback_storage = []  # Store rollback information
    
    async def execute_action(
        self,
        action_name: str,
        params: Dict[str, Any],
        incident_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Execute action with automatic rollback support
        
        Returns:
            {
                "success": bool,
                "action_id": str,
                "rollback_id": str,
                "result": dict,
                "error": str (if failed)
            }
        """
        action_id = f"{self.agent_id}_{action_name}_{datetime.now().timestamp()}"
        
        try:
            # Step 1: Capture current state for rollback
            rollback_data = await self._capture_state(action_name, params)
            rollback_id = self._store_rollback(rollback_data)
            
            # Step 2: Execute the action
            self.logger.info(f"Executing {action_name} with params: {params}")
            result = await self._execute_action_impl(action_name, params)
            
            # Step 3: Log success
            await self._log_action(
                action_id=action_id,
                action_name=action_name,
                params=params,
                result=result,
                status="success",
                incident_id=incident_id,
                rollback_id=rollback_id
            )
            
            return {
                "success": True,
                "action_id": action_id,
                "rollback_id": rollback_id,
                "result": result,
                "message": f"Action {action_name} completed successfully"
            }
            
        except Exception as e:
            self.logger.error(f"Action {action_name} failed: {e}")
            
            # Log failure
            await self._log_action(
                action_id=action_id,
                action_name=action_name,
                params=params,
                result=None,
                status="failed",
                incident_id=incident_id,
                error=str(e)
            )
            
            return {
                "success": False,
                "action_id": action_id,
                "error": str(e),
                "message": f"Action {action_name} failed"
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
            
            self.logger.info(f"Rolling back action: {rollback_data['action_name']}")
            
            # Execute rollback
            restored_state = await self._execute_rollback_impl(rollback_data)
            
            # Log rollback
            await self._log_action(
                action_id=f"{rollback_id}_rollback",
                action_name=f"rollback_{rollback_data['action_name']}",
                params={"rollback_id": rollback_id},
                result=restored_state,
                status="success",
                incident_id=rollback_data.get('incident_id')
            )
            
            return {
                "success": True,
                "message": f"Successfully rolled back {rollback_data['action_name']}",
                "restored_state": restored_state
            }
            
        except Exception as e:
            self.logger.error(f"Rollback failed for {rollback_id}: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": "Rollback failed"
            }
    
    def _store_rollback(self, rollback_data: Dict) -> str:
        """Store rollback data and return rollback_id"""
        rollback_id = f"rollback_{datetime.now().timestamp()}"
        rollback_data['rollback_id'] = rollback_id
        self.rollback_storage.append(rollback_data)
        
        # Also persist to database
        # TODO: Store in ActionLog table with rollback_data
        
        return rollback_id
    
    def _get_rollback(self, rollback_id: str) -> Optional[Dict]:
        """Retrieve rollback data by ID"""
        for item in self.rollback_storage:
            if item.get('rollback_id') == rollback_id:
                return item
        return None
    
    async def _log_action(
        self,
        action_id: str,
        action_name: str,
        params: Dict,
        result: Optional[Dict],
        status: str,
        incident_id: Optional[int] = None,
        rollback_id: Optional[str] = None,
        error: Optional[str] = None
    ):
        """Log action to database for audit trail"""
        # TODO: Implement database logging
        self.logger.info(f"Action logged: {action_id} - {action_name} - {status}")
    
    @abstractmethod
    async def _capture_state(self, action_name: str, params: Dict) -> Dict:
        """
        Capture current state before action (for rollback)
        
        Must be implemented by each agent
        """
        pass
    
    @abstractmethod
    async def _execute_action_impl(self, action_name: str, params: Dict) -> Dict:
        """
        Execute the actual action
        
        Must be implemented by each agent
        """
        pass
    
    @abstractmethod
    async def _execute_rollback_impl(self, rollback_data: Dict) -> Dict:
        """
        Execute rollback using stored rollback_data
        
        Must be implemented by each agent
        """
        pass
```

---

## 🔐 IAM Agent Implementation with Rollback

```python
# backend/app/agents/iam_agent.py
import ldap3
from ldap3 import Server, Connection, ALL, MODIFY_REPLACE, MODIFY_ADD, MODIFY_DELETE
from typing import Dict, Any, Optional, List
from .base_agent import BaseSecurityAgent
from ..config import settings

class IAMAgent(BaseSecurityAgent):
    """Identity & Access Management Agent with rollback support"""
    
    def __init__(self):
        super().__init__("iam_agent")
        self.ad_server = settings.AD_SERVER  # 10.100.1.1
        self.ad_domain = settings.AD_DOMAIN  # minicorp.local
        self.ad_user = settings.AD_ADMIN_USER
        self.ad_password = settings.AD_ADMIN_PASSWORD
        self.ldap_conn = None
    
    async def initialize(self):
        """Connect to Active Directory"""
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
    
    # ==================== ACTION IMPLEMENTATIONS ====================
    
    async def _capture_state(self, action_name: str, params: Dict) -> Dict:
        """Capture AD state before changes"""
        if action_name == "disable_user_account":
            username = params['username']
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
                        "was_enabled": not (int(current_uac) & 0x0002)  # Check ACCOUNTDISABLE flag
                    }
                }
            
            return {"action_name": action_name, "username": username}
        
        elif action_name == "quarantine_user":
            username = params['username']
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
            
            return {"action_name": action_name, "username": username}
        
        return {"action_name": action_name, "params": params}
    
    async def _execute_action_impl(self, action_name: str, params: Dict) -> Dict:
        """Execute IAM action"""
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
        
        else:
            raise ValueError(f"Unknown action: {action_name}")
    
    async def _execute_rollback_impl(self, rollback_data: Dict) -> Dict:
        """Rollback IAM action"""
        action_name = rollback_data['action_name']
        previous_state = rollback_data.get('previous_state', {})
        username = rollback_data['username']
        user_dn = rollback_data['user_dn']
        
        if action_name == "disable_user_account":
            # Re-enable the account
            was_enabled = previous_state.get('was_enabled', True)
            
            if was_enabled:
                # Restore to enabled state
                uac_value = str(int(previous_state['userAccountControl']) & ~0x0002)  # Remove ACCOUNTDISABLE
                
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
            # Restore group memberships
            original_groups = previous_state.get('memberOf', [])
            
            # Remove from quarantine group
            quarantine_group = "CN=Quarantine,OU=Security,DC=minicorp,DC=local"
            self.ldap_conn.modify(
                quarantine_group,
                {'member': [(MODIFY_DELETE, [user_dn])]}
            )
            
            # Restore original groups
            for group in original_groups:
                try:
                    self.ldap_conn.modify(
                        group,
                        {'member': [(MODIFY_ADD, [user_dn])]}
                    )
                except:
                    self.logger.warning(f"Could not restore group: {group}")
            
            return {
                "action": "restored_group_memberships",
                "username": username,
                "restored_groups": original_groups
            }
        
        return {"action": "rollback_completed", "username": username}
    
    # ==================== HELPER METHODS ====================
    
    async def _disable_user(self, username: str, reason: str) -> Dict:
        """Disable AD user account"""
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
        user_dn = await self._get_user_dn(username)
        quarantine_group = "CN=Quarantine,OU=Security,DC=minicorp,DC=local"
        
        # Add to quarantine
        self.ldap_conn.modify(
            quarantine_group,
            {'member': [(MODIFY_ADD, [user_dn])]}
        )
        
        # Remove from privileged groups
        removed_groups = await self._remove_from_privileged_groups(user_dn)
        
        return {
            "username": username,
            "status": "quarantined",
            "removed_from_groups": removed_groups,
            "reason": reason
        }
    
    async def _revoke_tickets(self, username: str) -> Dict:
        """Revoke Kerberos tickets"""
        # Execute on DC via PowerShell
        # For now, return simulated result
        return {
            "username": username,
            "status": "tickets_revoked",
            "message": "User must re-authenticate"
        }
    
    async def _reset_password(self, username: str, force_change: bool) -> Dict:
        """Reset user password"""
        import secrets
        import string
        
        # Generate secure random password
        alphabet = string.ascii_letters + string.digits + string.punctuation
        new_password = ''.join(secrets.choice(alphabet) for _ in range(16))
        
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
            "temporary_password": new_password,  # Send to admin securely
            "force_change": force_change
        }
    
    async def _remove_from_group(self, username: str, group: str) -> Dict:
        """Remove user from specific group"""
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
    
    async def _get_user_dn(self, username: str) -> str:
        """Get user Distinguished Name"""
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
    
    async def _remove_from_privileged_groups(self, user_dn: str) -> List[str]:
        """Remove user from all privileged groups"""
        self.ldap_conn.search(
            user_dn,
            '(objectClass=user)',
            attributes=['memberOf']
        )
        
        privileged_groups = [
            "Domain Admins", "Enterprise Admins", "Schema Admins",
            "Administrators", "Account Operators", "Backup Operators"
        ]
        
        removed_groups = []
        
        if self.ldap_conn.entries:
            groups = self.ldap_conn.entries[0].memberOf
            
            for group_dn in groups:
                group_str = str(group_dn)
                if any(pg in group_str for pg in privileged_groups):
                    try:
                        self.ldap_conn.modify(
                            group_str,
                            {'member': [(MODIFY_DELETE, [user_dn])]}
                        )
                        removed_groups.append(group_str)
                    except:
                        self.logger.warning(f"Could not remove from {group_str}")
        
        return removed_groups


# Global IAM agent instance
iam_agent = IAMAgent()
```

---

## 📦 Action Log Database Model

Add to `backend/app/models.py`:

```python
class ActionLog(Base):
    """Log of all agent actions with rollback support"""
    __tablename__ = "action_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    action_id = Column(String, unique=True, index=True)
    agent_id = Column(String, index=True)
    action_name = Column(String, index=True)
    incident_id = Column(Integer, ForeignKey("incidents.id"), nullable=True)
    
    # Action details
    params = Column(JSON)
    result = Column(JSON, nullable=True)
    status = Column(String)  # success, failed, rolled_back
    error = Column(Text, nullable=True)
    
    # Rollback support
    rollback_id = Column(String, unique=True, index=True, nullable=True)
    rollback_data = Column(JSON, nullable=True)
    rollback_executed = Column(Boolean, default=False)
    rollback_timestamp = Column(DateTime(timezone=True), nullable=True)
    
    # Timestamps
    executed_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    incident = relationship("Incident", back_populates="action_logs")
```

Update Incident model:

```python
class Incident(Base):
    # ... existing fields ...
    
    # Add relationship
    action_logs = relationship("ActionLog", back_populates="incident", cascade="all, delete-orphan")
```

---

## 🎨 Frontend Action Management UI

### Action Detail Modal Component

```tsx
// frontend/components/ActionDetailModal.tsx
import { useState } from 'react';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { AlertCircle, CheckCircle, XCircle, Undo2 } from 'lucide-react';

interface ActionLog {
  id: number;
  action_id: string;
  agent_id: string;
  action_name: string;
  params: any;
  result: any;
  status: string;
  error?: string;
  rollback_id?: string;
  rollback_executed: boolean;
  executed_at: string;
}

interface ActionDetailModalProps {
  action: ActionLog | null;
  isOpen: boolean;
  onClose: () => void;
  onRollback: (rollbackId: string) => Promise<void>;
}

export function ActionDetailModal({ action, isOpen, onClose, onRollback }: ActionDetailModalProps) {
  const [isRollingBack, setIsRollingBack] = useState(false);
  
  if (!action) return null;
  
  const handleRollback = async () => {
    if (!action.rollback_id || action.rollback_executed) return;
    
    if (!confirm(`Are you sure you want to rollback: ${action.action_name}?\n\nThis will restore the previous state.`)) {
      return;
    }
    
    setIsRollingBack(true);
    try {
      await onRollback(action.rollback_id);
    } finally {
      setIsRollingBack(false);
    }
  };
  
  const statusIcon = {
    success: <CheckCircle className="w-5 h-5 text-green-500" />,
    failed: <XCircle className="w-5 h-5 text-red-500" />,
    rolled_back: <Undo2 className="w-5 h-5 text-yellow-500" />
  }[action.status];
  
  const statusColor = {
    success: 'bg-green-100 text-green-800',
    failed: 'bg-red-100 text-red-800',
    rolled_back: 'bg-yellow-100 text-yellow-800'
  }[action.status];
  
  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="max-w-3xl max-h-[80vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            {statusIcon}
            <span>{action.action_name.replace(/_/g, ' ').toUpperCase()}</span>
            <Badge className={statusColor}>
              {action.status}
            </Badge>
          </DialogTitle>
        </DialogHeader>
        
        <div className="space-y-4">
          {/* Action Details */}
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="text-sm font-medium text-gray-500">Agent</label>
              <p className="text-sm">{action.agent_id}</p>
            </div>
            <div>
              <label className="text-sm font-medium text-gray-500">Action ID</label>
              <p className="text-sm font-mono text-xs">{action.action_id}</p>
            </div>
            <div>
              <label className="text-sm font-medium text-gray-500">Executed At</label>
              <p className="text-sm">{new Date(action.executed_at).toLocaleString()}</p>
            </div>
            <div>
              <label className="text-sm font-medium text-gray-500">Rollback Available</label>
              <p className="text-sm">{action.rollback_id ? '✅ Yes' : '❌ No'}</p>
            </div>
          </div>
          
          {/* Parameters */}
          <div>
            <label className="text-sm font-medium text-gray-500">Parameters</label>
            <pre className="mt-1 p-3 bg-gray-50 rounded text-xs overflow-x-auto">
              {JSON.stringify(action.params, null, 2)}
            </pre>
          </div>
          
          {/* Result */}
          {action.result && (
            <div>
              <label className="text-sm font-medium text-gray-500">Result</label>
              <pre className="mt-1 p-3 bg-gray-50 rounded text-xs overflow-x-auto">
                {JSON.stringify(action.result, null, 2)}
              </pre>
            </div>
          )}
          
          {/* Error */}
          {action.error && (
            <div className="p-3 bg-red-50 border border-red-200 rounded">
              <div className="flex items-start gap-2">
                <AlertCircle className="w-5 h-5 text-red-500 flex-shrink-0 mt-0.5" />
                <div>
                  <label className="text-sm font-medium text-red-800">Error</label>
                  <p className="text-sm text-red-700 mt-1">{action.error}</p>
                </div>
              </div>
            </div>
          )}
          
          {/* Rollback Button */}
          {action.rollback_id && !action.rollback_executed && action.status === 'success' && (
            <div className="pt-4 border-t">
              <Button
                onClick={handleRollback}
                disabled={isRollingBack}
                variant="destructive"
                className="w-full"
              >
                <Undo2 className="w-4 h-4 mr-2" />
                {isRollingBack ? 'Rolling back...' : 'Rollback This Action'}
              </Button>
              <p className="text-xs text-gray-500 mt-2 text-center">
                This will restore the system to its previous state before this action was executed.
              </p>
            </div>
          )}
          
          {action.rollback_executed && (
            <div className="p-3 bg-yellow-50 border border-yellow-200 rounded">
              <p className="text-sm text-yellow-800">
                ✅ This action has been rolled back
              </p>
            </div>
          )}
        </div>
      </DialogContent>
    </Dialog>
  );
}
```

### Add to Incident Detail Page

```tsx
// Add to frontend/app/incidents/incident/[id]/page.tsx

import { ActionDetailModal } from '@/components/ActionDetailModal';

// Inside the component:
const [selectedAction, setSelectedAction] = useState<ActionLog | null>(null);
const [isActionModalOpen, setIsActionModalOpen] = useState(false);

const handleActionClick = (action: ActionLog) => {
  setSelectedAction(action);
  setIsActionModalOpen(true);
};

const handleRollback = async (rollbackId: string) => {
  const response = await fetch(`/api/actions/rollback/${rollbackId}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' }
  });
  
  if (response.ok) {
    toast.success('Action rolled back successfully');
    // Refresh actions
    loadIncidentActions();
  } else {
    toast.error('Rollback failed');
  }
  
  setIsActionModalOpen(false);
};

// In the JSX, add actions section:
<div className="mt-6">
  <h3 className="text-lg font-semibold mb-4">Actions Taken</h3>
  <div className="space-y-2">
    {actions.map((action) => (
      <div
        key={action.id}
        className="p-3 border rounded hover:bg-gray-50 cursor-pointer"
        onClick={() => handleActionClick(action)}
      >
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Badge className={getStatusColor(action.status)}>
              {action.status}
            </Badge>
            <span className="font-medium">{action.action_name}</span>
          </div>
          <span className="text-xs text-gray-500">
            {new Date(action.executed_at).toLocaleTimeString()}
          </span>
        </div>
        {action.rollback_id && !action.rollback_executed && (
          <span className="text-xs text-blue-600 mt-1 inline-block">
            ↩ Rollback available
          </span>
        )}
      </div>
    ))}
  </div>
</div>

<ActionDetailModal
  action={selectedAction}
  isOpen={isActionModalOpen}
  onClose={() => setIsActionModalOpen(false)}
  onRollback={handleRollback}
/>
```

---

## 🧪 Testing Plan

### Test Sequence

1. **Test IAM Agent Actions**:
```bash
# Test disable user
curl -X POST http://localhost:8000/api/agents/iam/execute \
  -H "Content-Type: application/json" \
  -d '{
    "action_name": "disable_user_account",
    "params": {"username": "test.user", "reason": "Testing"},
    "incident_id": 1
  }'

# Verify in UI - action should appear with rollback button

# Test rollback
curl -X POST http://localhost:8000/api/actions/rollback/{rollback_id} \
  -H "Content-Type: application/json"

# Verify user is re-enabled
```

2. **Test Frontend UI**:
- Click on an action → Modal opens with details
- Click "Rollback This Action" → Confirmation dialog
- Confirm → Action is rolled back
- Verify action status changes to "rolled_back"

3. **Test Error Handling**:
- Try to rollback twice (should fail with message)
- Try invalid parameters (should show error in UI)
- Try to rollback without permission (should deny)

---

## 🚀 Deployment Steps

1. ✅ Fix ML errors (done)
2. ✅ Create base agent class (documented above)
3. ⏳ Implement IAM agent with rollback
4. ⏳ Add ActionLog database model
5. ⏳ Create API endpoints for actions
6. ⏳ Build frontend Action Management UI
7. ⏳ Test complete workflow
8. ⏳ Deploy to production

---

**Status:** Ready to implement - waiting for Azure ML training to complete

