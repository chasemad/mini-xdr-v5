# 03: Enterprise Integrations - Production Implementation

**Status:** Critical Gap - Competitive Differentiator  
**Current State:** 3 basic integrations (Cowrie, Suricata, OSQuery)  
**Target State:** 20+ enterprise integrations with standardized framework  
**Priority:** P0 (Customers will ask "does it integrate with our stack?")

---

## Current State Analysis

### What EXISTS Now

**File:** `/backend/app/multi_ingestion.py`
```python
✅ Multi-source ingestion framework
✅ Cowrie honeypot integration
✅ Suricata IDS integration  
✅ OSQuery endpoint integration
✅ Custom JSON/syslog support
⚠️ No enterprise SIEM/EDR connectors
⚠️ No bidirectional communication (events in only, no actions out)
```

**File:** `/backend/app/external_intel.py`
```python
✅ AbuseIPDB threat intelligence
✅ VirusTotal integration
⚠️ Basic implementation, not production-grade
⚠️ No rate limiting or caching
```

### What's MISSING (Top 20 Customer Requests)

**SIEM Integrations:**
- ❌ Splunk (90% of enterprises use it)
- ❌ Elastic Security (SIEM)
- ❌ IBM QRadar
- ❌ Google Chronicle
- ❌ Microsoft Sentinel

**EDR/Endpoint:**
- ❌ CrowdStrike Falcon
- ❌ SentinelOne
- ❌ Microsoft Defender for Endpoint
- ❌ Carbon Black
- ❌ Palo Alto Cortex XDR

**Cloud Security:**
- ❌ AWS GuardDuty
- ❌ AWS Security Hub
- ❌ Azure Security Center
- ❌ GCP Security Command Center

**Network Security:**
- ❌ Palo Alto Firewall
- ❌ Cisco Firepower
- ❌ Fortinet FortiGate

**Ticketing/SOAR:**
- ❌ ServiceNow
- ❌ Jira
- ❌ PagerDuty

**Email Security:**
- ❌ Proofpoint
- ❌ Mimecast

---

## Implementation Strategy: Integration Framework First

Rather than building 20 one-off integrations, build a **reusable framework** that makes adding new integrations fast.

### Task 1: Build Integration Framework

#### 1.1: Create Base Integration Class
**New File:** `/backend/app/integrations/base.py`

```python
"""Base integration framework for all external systems"""
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone
from enum import Enum
import logging
import asyncio
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class IntegrationType(str, Enum):
    """Types of integrations"""
    SIEM = "siem"
    EDR = "edr"
    FIREWALL = "firewall"
    CLOUD_SECURITY = "cloud_security"
    TICKETING = "ticketing"
    EMAIL_SECURITY = "email_security"
    THREAT_INTEL = "threat_intel"


class IntegrationCapability(str, Enum):
    """What the integration can do"""
    INGEST_EVENTS = "ingest_events"
    SEND_ALERTS = "send_alerts"
    EXECUTE_ACTIONS = "execute_actions"
    QUERY_DATA = "query_data"
    BIDIRECTIONAL = "bidirectional"


class IntegrationConfig(BaseModel):
    """Configuration for an integration instance"""
    enabled: bool = True
    api_endpoint: str
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    organization_id: int
    
    # Rate limiting
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    
    # Retry policy
    max_retries: int = 3
    retry_delay_seconds: int = 5
    
    # Data mapping
    field_mapping: Dict[str, str] = Field(default_factory=dict)
    
    # Filters
    severity_filter: Optional[List[str]] = None
    event_type_filter: Optional[List[str]] = None


class IntegrationHealth(BaseModel):
    """Health status of an integration"""
    is_healthy: bool
    last_success: Optional[datetime] = None
    last_failure: Optional[datetime] = None
    consecutive_failures: int = 0
    error_message: Optional[str] = None
    events_processed_24h: int = 0
    average_latency_ms: float = 0


class BaseIntegration(ABC):
    """Abstract base class for all integrations"""
    
    def __init__(self, config: IntegrationConfig):
        self.config = config
        self.health = IntegrationHealth(is_healthy=True)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Integration name (e.g., 'splunk', 'crowdstrike')"""
        pass
    
    @property
    @abstractmethod
    def integration_type(self) -> IntegrationType:
        """Type of integration"""
        pass
    
    @property
    @abstractmethod
    def capabilities(self) -> List[IntegrationCapability]:
        """What this integration can do"""
        pass
    
    @abstractmethod
    async def test_connection(self) -> bool:
        """Test if connection to external system works"""
        pass
    
    @abstractmethod
    async def ingest_events(
        self,
        since: datetime,
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        Pull events from external system
        Returns: List of normalized events
        """
        pass
    
    @abstractmethod
    async def send_alert(
        self,
        incident_id: int,
        severity: str,
        title: str,
        description: str,
        metadata: Dict[str, Any]
    ) -> bool:
        """
        Send alert to external system
        Returns: True if successful
        """
        pass
    
    @abstractmethod
    async def execute_action(
        self,
        action_type: str,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute action on external system (e.g., block IP on firewall)
        Returns: Action result
        """
        pass
    
    async def normalize_event(self, raw_event: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize external event to Mini-XDR format
        Override for custom normalization
        """
        normalized = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source_integration": self.name,
            "raw": raw_event
        }
        
        # Apply field mapping
        for mini_xdr_field, external_field in self.config.field_mapping.items():
            if external_field in raw_event:
                normalized[mini_xdr_field] = raw_event[external_field]
        
        return normalized
    
    async def update_health(self, success: bool, error: Optional[str] = None):
        """Update integration health status"""
        now = datetime.now(timezone.utc)
        
        if success:
            self.health.is_healthy = True
            self.health.last_success = now
            self.health.consecutive_failures = 0
            self.health.error_message = None
        else:
            self.health.last_failure = now
            self.health.consecutive_failures += 1
            self.health.error_message = error
            
            # Mark unhealthy after 3 consecutive failures
            if self.health.consecutive_failures >= 3:
                self.health.is_healthy = False
                self.logger.error(f"Integration {self.name} marked unhealthy: {error}")
    
    async def with_retry(self, func, *args, **kwargs):
        """Execute function with retry logic"""
        for attempt in range(self.config.max_retries):
            try:
                result = await func(*args, **kwargs)
                await self.update_health(success=True)
                return result
            except Exception as e:
                self.logger.warning(
                    f"Attempt {attempt + 1}/{self.config.max_retries} failed: {e}"
                )
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(self.config.retry_delay_seconds)
                else:
                    await self.update_health(success=False, error=str(e))
                    raise
```

**Checklist:**
- [ ] Create `/backend/app/integrations/` directory
- [ ] Create `base.py` with BaseIntegration class
- [ ] Add to requirements.txt: `aiohttp==3.9.5` (already exists)
- [ ] Create `__init__.py` to export classes

#### 1.2: Create Integration Registry
**New File:** `/backend/app/integrations/registry.py`

```python
"""Registry for managing all integrations"""
from typing import Dict, Type, List, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from .base import BaseIntegration, IntegrationConfig
from ..models import Integration as IntegrationModel


class IntegrationRegistry:
    """Central registry for all integrations"""
    
    def __init__(self):
        self._integrations: Dict[str, Type[BaseIntegration]] = {}
        self._instances: Dict[int, BaseIntegration] = {}  # integration_id -> instance
    
    def register(self, integration_class: Type[BaseIntegration]):
        """Register an integration class"""
        instance = integration_class(IntegrationConfig(
            api_endpoint="placeholder",
            organization_id=0
        ))
        self._integrations[instance.name] = integration_class
    
    def get_integration_class(self, name: str) -> Optional[Type[BaseIntegration]]:
        """Get integration class by name"""
        return self._integrations.get(name)
    
    def list_available_integrations(self) -> List[str]:
        """List all registered integration names"""
        return list(self._integrations.keys())
    
    async def load_instance(
        self,
        integration_id: int,
        db: AsyncSession
    ) -> Optional[BaseIntegration]:
        """Load integration instance from database"""
        if integration_id in self._instances:
            return self._instances[integration_id]
        
        result = await db.execute(
            select(IntegrationModel).where(IntegrationModel.id == integration_id)
        )
        integration_model = result.scalars().first()
        
        if not integration_model or not integration_model.enabled:
            return None
        
        integration_class = self.get_integration_class(integration_model.integration_type)
        if not integration_class:
            return None
        
        config = IntegrationConfig(
            enabled=integration_model.enabled,
            api_endpoint=integration_model.config.get("api_endpoint", ""),
            api_key=integration_model.config.get("api_key"),
            api_secret=integration_model.config.get("api_secret"),
            organization_id=integration_model.organization_id,
            field_mapping=integration_model.config.get("field_mapping", {})
        )
        
        instance = integration_class(config)
        self._instances[integration_id] = instance
        
        return instance
    
    async def get_all_instances_for_org(
        self,
        organization_id: int,
        db: AsyncSession
    ) -> List[BaseIntegration]:
        """Get all enabled integration instances for an organization"""
        result = await db.execute(
            select(IntegrationModel)
            .where(IntegrationModel.organization_id == organization_id)
            .where(IntegrationModel.enabled == True)
        )
        
        instances = []
        for integration_model in result.scalars().all():
            instance = await self.load_instance(integration_model.id, db)
            if instance:
                instances.append(instance)
        
        return instances


# Global registry
integration_registry = IntegrationRegistry()
```

**Checklist:**
- [ ] Create registry.py
- [ ] Add Integration model to models.py (see Task 1.3)
- [ ] Test registry registration

#### 1.3: Add Integration Database Model
**File:** `/backend/app/models.py` - Add after existing models

```python
class Integration(Base):
    """External system integrations"""
    __tablename__ = "integrations"
    
    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    organization_id = Column(Integer, ForeignKey("organizations.id"), nullable=False, index=True)
    
    # Integration details
    integration_type = Column(String(64), nullable=False, index=True)  # splunk, crowdstrike, etc.
    display_name = Column(String(128), nullable=False)
    description = Column(Text, nullable=True)
    
    # Configuration (stored as JSON)
    config = Column(JSON, nullable=False)  # API keys, endpoints, field mappings
    
    # Status
    enabled = Column(Boolean, default=True, index=True)
    health_status = Column(String(32), default="unknown")  # healthy|degraded|unhealthy|unknown
    last_health_check = Column(DateTime(timezone=True), nullable=True)
    
    # Usage statistics
    events_ingested_total = Column(Integer, default=0)
    alerts_sent_total = Column(Integer, default=0)
    actions_executed_total = Column(Integer, default=0)
    last_sync_at = Column(DateTime(timezone=True), nullable=True)
    
    # Error tracking
    last_error = Column(Text, nullable=True)
    consecutive_failures = Column(Integer, default=0)
    
    # Relationships
    organization = relationship("Organization", backref="integrations")


class IntegrationLog(Base):
    """Log of integration actions for debugging"""
    __tablename__ = "integration_logs"
    
    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    
    integration_id = Column(Integer, ForeignKey("integrations.id"), nullable=False, index=True)
    
    action_type = Column(String(64), nullable=False)  # ingest|send_alert|execute_action|health_check
    status = Column(String(32), nullable=False)  # success|failed
    
    request_data = Column(JSON, nullable=True)
    response_data = Column(JSON, nullable=True)
    error_message = Column(Text, nullable=True)
    
    latency_ms = Column(Integer, nullable=True)
    
    integration = relationship("Integration", backref="logs")
```

**Checklist:**
- [ ] Add Integration and IntegrationLog models
- [ ] Create migration: `alembic revision --autogenerate -m "add_integrations"`
- [ ] Run migration: `alembic upgrade head`

---

### Task 2: Implement Priority Integrations

#### 2.1: Splunk Integration (Highest Priority - 90% of enterprises)
**New File:** `/backend/app/integrations/splunk.py`

```python
"""Splunk SIEM integration"""
from typing import Dict, List, Any
from datetime import datetime, timezone
import aiohttp
from .base import (
    BaseIntegration, IntegrationConfig, IntegrationType,
    IntegrationCapability
)


class SplunkIntegration(BaseIntegration):
    """Splunk Enterprise/Cloud integration"""
    
    @property
    def name(self) -> str:
        return "splunk"
    
    @property
    def integration_type(self) -> IntegrationType:
        return IntegrationType.SIEM
    
    @property
    def capabilities(self) -> List[IntegrationCapability]:
        return [
            IntegrationCapability.INGEST_EVENTS,
            IntegrationCapability.SEND_ALERTS,
            IntegrationCapability.QUERY_DATA
        ]
    
    async def test_connection(self) -> bool:
        """Test Splunk connection"""
        async with aiohttp.ClientSession() as session:
            headers = {
                "Authorization": f"Bearer {self.config.api_key}"
            }
            url = f"{self.config.api_endpoint}/services/server/info"
            
            try:
                async with session.get(url, headers=headers, ssl=False) as response:
                    return response.status == 200
            except Exception as e:
                self.logger.error(f"Splunk connection test failed: {e}")
                return False
    
    async def ingest_events(
        self,
        since: datetime,
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        Pull events from Splunk using REST API search
        SPL Query: Search for security events since timestamp
        """
        async with aiohttp.ClientSession() as session:
            headers = {
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json"
            }
            
            # Splunk Search Processing Language (SPL) query
            earliest_time = since.strftime("%Y-%m-%dT%H:%M:%S")
            search_query = f"""
            search index=security earliest={earliest_time}
            | head {limit}
            | table _time, source, sourcetype, host, _raw
            """
            
            # Create search job
            url = f"{self.config.api_endpoint}/services/search/jobs"
            data = {
                "search": search_query,
                "output_mode": "json"
            }
            
            try:
                # Submit search
                async with session.post(url, headers=headers, json=data) as response:
                    result = await response.json()
                    search_id = result.get("sid")
                
                # Poll for results
                results_url = f"{url}/{search_id}/results"
                await asyncio.sleep(2)  # Give search time to complete
                
                async with session.get(
                    results_url,
                    headers=headers,
                    params={"output_mode": "json"}
                ) as response:
                    data = await response.json()
                    events = data.get("results", [])
                
                # Normalize events
                normalized = [await self.normalize_event(event) for event in events]
                
                await self.update_health(success=True)
                return normalized
                
            except Exception as e:
                self.logger.error(f"Failed to ingest from Splunk: {e}")
                await self.update_health(success=False, error=str(e))
                return []
    
    async def send_alert(
        self,
        incident_id: int,
        severity: str,
        title: str,
        description: str,
        metadata: Dict[str, Any]
    ) -> bool:
        """
        Send alert to Splunk as a notable event
        Uses Splunk Notable Events API
        """
        async with aiohttp.ClientSession() as session:
            headers = {
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json"
            }
            
            url = f"{self.config.api_endpoint}/services/notable_events"
            
            payload = {
                "time": datetime.now(timezone.utc).isoformat(),
                "severity": severity,
                "title": title,
                "description": description,
                "source": "Mini-XDR",
                "incident_id": incident_id,
                **metadata
            }
            
            try:
                async with session.post(url, headers=headers, json=payload) as response:
                    success = response.status == 200
                    await self.update_health(success=success)
                    return success
            except Exception as e:
                self.logger.error(f"Failed to send alert to Splunk: {e}")
                await self.update_health(success=False, error=str(e))
                return False
    
    async def execute_action(
        self,
        action_type: str,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute action via Splunk (limited - mostly read-only)
        """
        return {
            "success": False,
            "error": "Splunk integration does not support direct actions"
        }
    
    async def normalize_event(self, raw_event: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize Splunk event to Mini-XDR format"""
        return {
            "timestamp": raw_event.get("_time"),
            "source_ip": raw_event.get("src_ip") or raw_event.get("clientip"),
            "dest_ip": raw_event.get("dest_ip") or raw_event.get("destip"),
            "event_type": raw_event.get("sourcetype"),
            "severity": raw_event.get("severity", "info"),
            "message": raw_event.get("_raw"),
            "source_integration": "splunk",
            "hostname": raw_event.get("host"),
            "raw": raw_event
        }


# Register integration
from .registry import integration_registry
integration_registry.register(SplunkIntegration)
```

**Checklist:**
- [ ] Create splunk.py
- [ ] Test Splunk connection
- [ ] Test event ingestion from Splunk
- [ ] Test sending alerts to Splunk
- [ ] Add Splunk setup guide to docs
- [ ] Create Splunk field mapping UI

#### 2.2: CrowdStrike Falcon Integration
**New File:** `/backend/app/integrations/crowdstrike.py`

```python
"""CrowdStrike Falcon EDR integration"""
from typing import Dict, List, Any
from datetime import datetime, timezone
import aiohttp
from .base import (
    BaseIntegration, IntegrationConfig, IntegrationType,
    IntegrationCapability
)


class CrowdStrikeIntegration(BaseIntegration):
    """CrowdStrike Falcon integration"""
    
    @property
    def name(self) -> str:
        return "crowdstrike"
    
    @property
    def integration_type(self) -> IntegrationType:
        return IntegrationType.EDR
    
    @property
    def capabilities(self) -> List[IntegrationCapability]:
        return [
            IntegrationCapability.INGEST_EVENTS,
            IntegrationCapability.SEND_ALERTS,
            IntegrationCapability.EXECUTE_ACTIONS,
            IntegrationCapability.BIDIRECTIONAL
        ]
    
    async def get_access_token(self) -> str:
        """Get OAuth2 access token from CrowdStrike"""
        async with aiohttp.ClientSession() as session:
            url = f"{self.config.api_endpoint}/oauth2/token"
            data = {
                "client_id": self.config.api_key,
                "client_secret": self.config.api_secret
            }
            
            async with session.post(url, data=data) as response:
                result = await response.json()
                return result["access_token"]
    
    async def test_connection(self) -> bool:
        """Test CrowdStrike API connection"""
        try:
            token = await self.get_access_token()
            return bool(token)
        except:
            return False
    
    async def ingest_events(
        self,
        since: datetime,
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        Pull detection events from CrowdStrike
        Uses /detects/queries/detects/v1 API
        """
        token = await self.get_access_token()
        
        async with aiohttp.ClientSession() as session:
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json"
            }
            
            # Query for detection IDs
            query_url = f"{self.config.api_endpoint}/detects/queries/detects/v1"
            params = {
                "filter": f"created_timestamp:>'{since.isoformat()}'",
                "limit": limit
            }
            
            try:
                async with session.get(query_url, headers=headers, params=params) as response:
                    data = await response.json()
                    detection_ids = data.get("resources", [])
                
                if not detection_ids:
                    return []
                
                # Get detection details
                details_url = f"{self.config.api_endpoint}/detects/entities/summaries/GET/v1"
                payload = {"ids": detection_ids}
                
                async with session.post(details_url, headers=headers, json=payload) as response:
                    data = await response.json()
                    detections = data.get("resources", [])
                
                normalized = [await self.normalize_event(d) for d in detections]
                await self.update_health(success=True)
                
                return normalized
                
            except Exception as e:
                self.logger.error(f"Failed to ingest from CrowdStrike: {e}")
                await self.update_health(success=False, error=str(e))
                return []
    
    async def send_alert(
        self,
        incident_id: int,
        severity: str,
        title: str,
        description: str,
        metadata: Dict[str, Any]
    ) -> bool:
        """Send alert to CrowdStrike as a custom IOA"""
        # CrowdStrike doesn't have a direct "send alert" API
        # Alternative: Create custom IOA or use Spotlight API
        return True
    
    async def execute_action(
        self,
        action_type: str,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute containment action on CrowdStrike
        Actions: contain_host, release_host, quarantine_file
        """
        token = await self.get_access_token()
        
        async with aiohttp.ClientSession() as session:
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json"
            }
            
            if action_type == "contain_host":
                # Network containment
                url = f"{self.config.api_endpoint}/devices/entities/devices-actions/v2"
                params = {"action_name": "contain"}
                payload = {"ids": [parameters["device_id"]]}
                
                try:
                    async with session.post(url, headers=headers, params=params, json=payload) as response:
                        result = await response.json()
                        return {
                            "success": response.status == 202,
                            "action_id": result.get("resources", [{}])[0].get("id"),
                            "result": result
                        }
                except Exception as e:
                    return {"success": False, "error": str(e)}
            
            elif action_type == "release_host":
                # Release network containment
                url = f"{self.config.api_endpoint}/devices/entities/devices-actions/v2"
                params = {"action_name": "lift_containment"}
                payload = {"ids": [parameters["device_id"]]}
                
                try:
                    async with session.post(url, headers=headers, params=params, json=payload) as response:
                        result = await response.json()
                        return {
                            "success": response.status == 202,
                            "result": result
                        }
                except Exception as e:
                    return {"success": False, "error": str(e)}
            
            return {"success": False, "error": f"Unknown action: {action_type}"}
    
    async def normalize_event(self, raw_event: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize CrowdStrike detection to Mini-XDR format"""
        behaviors = raw_event.get("behaviors", [])
        first_behavior = behaviors[0] if behaviors else {}
        
        return {
            "timestamp": raw_event.get("created_timestamp"),
            "event_id": raw_event.get("detection_id"),
            "severity": raw_event.get("max_severity_displayname", "medium"),
            "event_type": first_behavior.get("tactic"),
            "hostname": raw_event.get("device", {}).get("hostname"),
            "source_ip": raw_event.get("device", {}).get("local_ip"),
            "process_name": first_behavior.get("filename"),
            "command_line": first_behavior.get("cmdline"),
            "user": first_behavior.get("user_name"),
            "description": first_behavior.get("description"),
            "confidence": raw_event.get("max_confidence"),
            "source_integration": "crowdstrike",
            "raw": raw_event
        }


# Register integration
from .registry import integration_registry
integration_registry.register(CrowdStrikeIntegration)
```

**Checklist:**
- [ ] Create crowdstrike.py
- [ ] Get CrowdStrike API credentials (OAuth2)
- [ ] Test connection and token retrieval
- [ ] Test event ingestion
- [ ] Test host containment action
- [ ] Add CrowdStrike setup guide

#### 2.3: AWS GuardDuty Integration
**New File:** `/backend/app/integrations/aws_guardduty.py`

```python
"""AWS GuardDuty integration"""
import boto3
from typing import Dict, List, Any
from datetime import datetime, timezone, timedelta
from .base import (
    BaseIntegration, IntegrationConfig, IntegrationType,
    IntegrationCapability
)


class AWSGuardDutyIntegration(BaseIntegration):
    """AWS GuardDuty cloud security integration"""
    
    def __init__(self, config: IntegrationConfig):
        super().__init__(config)
        self.guardduty_client = boto3.client(
            'guardduty',
            aws_access_key_id=config.api_key,
            aws_secret_access_key=config.api_secret,
            region_name=config.config.get("region", "us-east-1")
        )
        self.detector_id = config.config.get("detector_id")
    
    @property
    def name(self) -> str:
        return "aws_guardduty"
    
    @property
    def integration_type(self) -> IntegrationType:
        return IntegrationType.CLOUD_SECURITY
    
    @property
    def capabilities(self) -> List[IntegrationCapability]:
        return [
            IntegrationCapability.INGEST_EVENTS,
            IntegrationCapability.QUERY_DATA
        ]
    
    async def test_connection(self) -> bool:
        """Test AWS GuardDuty connection"""
        try:
            response = self.guardduty_client.list_detectors()
            return len(response.get("DetectorIds", [])) > 0
        except Exception as e:
            self.logger.error(f"GuardDuty connection test failed: {e}")
            return False
    
    async def ingest_events(
        self,
        since: datetime,
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """Pull findings from AWS GuardDuty"""
        try:
            # Find findings updated since timestamp
            response = self.guardduty_client.list_findings(
                DetectorId=self.detector_id,
                FindingCriteria={
                    "Criterion": {
                        "updatedAt": {
                            "Gte": int(since.timestamp() * 1000)  # GuardDuty uses milliseconds
                        }
                    }
                },
                MaxResults=limit
            )
            
            finding_ids = response.get("FindingIds", [])
            
            if not finding_ids:
                return []
            
            # Get finding details
            findings_response = self.guardduty_client.get_findings(
                DetectorId=self.detector_id,
                FindingIds=finding_ids
            )
            
            findings = findings_response.get("Findings", [])
            normalized = [await self.normalize_event(f) for f in findings]
            
            await self.update_health(success=True)
            return normalized
            
        except Exception as e:
            self.logger.error(f"Failed to ingest from GuardDuty: {e}")
            await self.update_health(success=False, error=str(e))
            return []
    
    async def send_alert(
        self,
        incident_id: int,
        severity: str,
        title: str,
        description: str,
        metadata: Dict[str, Any]
    ) -> bool:
        """GuardDuty is read-only, can't send alerts back"""
        return False
    
    async def execute_action(
        self,
        action_type: str,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """GuardDuty doesn't support direct actions"""
        return {"success": False, "error": "GuardDuty is read-only"}
    
    async def normalize_event(self, raw_finding: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize GuardDuty finding to Mini-XDR format"""
        resource = raw_finding.get("Resource", {})
        instance_details = resource.get("InstanceDetails", {})
        
        return {
            "timestamp": raw_finding.get("UpdatedAt"),
            "event_id": raw_finding.get("Id"),
            "severity": raw_finding.get("Severity"),
            "event_type": raw_finding.get("Type"),
            "title": raw_finding.get("Title"),
            "description": raw_finding.get("Description"),
            "source_ip": instance_details.get("NetworkInterfaces", [{}])[0].get("PrivateIpAddress"),
            "instance_id": instance_details.get("InstanceId"),
            "region": raw_finding.get("Region"),
            "account_id": raw_finding.get("AccountId"),
            "confidence": raw_finding.get("Confidence"),
            "source_integration": "aws_guardduty",
            "raw": raw_finding
        }


# Register integration
from .registry import integration_registry
integration_registry.register(AWSGuardDutyIntegration)
```

**Checklist:**
- [ ] Create aws_guardduty.py
- [ ] Add boto3 to requirements.txt (already exists)
- [ ] Test with AWS credentials
- [ ] Test finding ingestion
- [ ] Add AWS IAM policy template
- [ ] Document setup process

---

### Task 3: Integration Management API

#### 3.1: Add Integration Endpoints
**File:** `/backend/app/main.py` - Add around line 500

```python
from .integrations.registry import integration_registry
from .integrations.base import IntegrationConfig

@app.get("/api/integrations/available")
async def list_available_integrations(
    current_user: User = Depends(get_current_user)
):
    """List all available integration types"""
    integrations = integration_registry.list_available_integrations()
    
    details = []
    for name in integrations:
        integration_class = integration_registry.get_integration_class(name)
        temp_instance = integration_class(IntegrationConfig(
            api_endpoint="placeholder",
            organization_id=0
        ))
        
        details.append({
            "name": temp_instance.name,
            "type": temp_instance.integration_type.value,
            "capabilities": [c.value for c in temp_instance.capabilities],
            "description": integration_class.__doc__ or ""
        })
    
    return {"integrations": details}


@app.post("/api/integrations")
async def create_integration(
    integration_data: dict,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Create new integration"""
    # Verify user has permission
    # ... permission check ...
    
    integration = Integration(
        organization_id=current_user.organization_id,
        integration_type=integration_data["integration_type"],
        display_name=integration_data["display_name"],
        config=integration_data["config"],
        enabled=True
    )
    
    db.add(integration)
    await db.commit()
    
    # Test connection
    instance = await integration_registry.load_instance(integration.id, db)
    if instance:
        is_healthy = await instance.test_connection()
        integration.health_status = "healthy" if is_healthy else "unhealthy"
        await db.commit()
    
    return {"id": integration.id, "status": integration.health_status}


@app.get("/api/integrations/{integration_id}/test")
async def test_integration(
    integration_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Test integration connection"""
    instance = await integration_registry.load_instance(integration_id, db)
    
    if not instance:
        raise HTTPException(status_code=404, detail="Integration not found")
    
    is_healthy = await instance.test_connection()
    
    return {
        "healthy": is_healthy,
        "health": instance.health.dict()
    }


@app.post("/api/integrations/{integration_id}/sync")
async def sync_integration(
    integration_id: int,
    since_minutes: int = 60,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Manually trigger integration sync"""
    instance = await integration_registry.load_instance(integration_id, db)
    
    if not instance:
        raise HTTPException(status_code=404, detail="Integration not found")
    
    since = datetime.now(timezone.utc) - timedelta(minutes=since_minutes)
    events = await instance.ingest_events(since=since)
    
    # Store events in database
    for event_data in events:
        event = Event(
            ts=datetime.fromisoformat(event_data["timestamp"]),
            src_ip=event_data.get("source_ip"),
            dst_ip=event_data.get("dest_ip"),
            eventid=event_data.get("event_type", "unknown"),
            message=event_data.get("description") or event_data.get("message"),
            raw=event_data.get("raw", {}),
            source_type=event_data["source_integration"],
            hostname=event_data.get("hostname")
        )
        db.add(event)
    
    await db.commit()
    
    return {"events_synced": len(events)}
```

**Checklist:**
- [ ] Add integration endpoints
- [ ] Test create integration
- [ ] Test connection test
- [ ] Test manual sync
- [ ] Add to API documentation

---

## Solo Developer Quickstart

Since you're working solo, prioritize this order:

### Week 1: Framework
- [ ] Build base integration framework (Task 1.1-1.3)
- [ ] Add Integration model and migration
- [ ] Create registry system

### Week 2: First Integration
- [ ] Implement Splunk integration (most common)
- [ ] Test end-to-end event flow
- [ ] Document setup process

### Week 3: Second Integration
- [ ] Implement CrowdStrike OR AWS GuardDuty
- [ ] Choose based on your target customers
- [ ] Test bidirectional communication

### Week 4: UI & Documentation
- [ ] Build integration management UI
- [ ] Create setup guides for each integration
- [ ] Add demo mode with sample data

---

## Testing Checklist

### Framework Tests
- [ ] Test integration registration
- [ ] Test health monitoring
- [ ] Test retry logic
- [ ] Test rate limiting

### Integration Tests (per integration)
- [ ] Test connection/authentication
- [ ] Test event ingestion (at least 100 events)
- [ ] Test event normalization
- [ ] Test alert sending
- [ ] Test action execution
- [ ] Test error handling

---

## Estimated Effort (Solo Developer)

| Task | Effort | Can Skip For MVP? |
|------|--------|-------------------|
| Integration framework | 3 days | No - foundation |
| Splunk integration | 2 days | No - most requested |
| CrowdStrike integration | 2 days | Yes - but valuable |
| AWS GuardDuty | 1 day | Yes - nice to have |
| Azure Sentinel | 2 days | Yes |
| Elastic SIEM | 1.5 days | Yes |
| API endpoints | 1 day | No |
| UI for management | 2 days | Yes - use API directly |
| Documentation | 2 days | No - customers need this |

**Total for MVP:** ~10-12 days (framework + 2 integrations + docs)

---

## Alternative: Use Pre-built Integration Platforms

If building custom integrations is too much work solo:

**Option A: Use Zapier/Make.com**
- Pros: 1000+ integrations ready
- Cons: Limited customization, monthly cost

**Option B: Use Tray.io**
- Pros: Enterprise-grade, drag-and-drop
- Cons: Expensive ($600+/month)

**Option C: Use N8N (Open Source)**
- Pros: Self-hosted, 200+ integrations
- Cons: Still requires setup per integration

**Recommendation:** Build the framework + 2-3 critical integrations yourself for differentiation, then add pre-built platform for long tail.

---

**Status:** Ready for implementation  
**Next Document:** `04_SCALABILITY_PERFORMANCE.md`


