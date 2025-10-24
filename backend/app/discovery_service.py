"""
Discovery Service - Tenant-aware network discovery wrapper

Wraps NetworkDiscoveryEngine with organization isolation and persistence.
"""
import logging
import uuid
from datetime import datetime, timezone
from typing import List, Dict, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from .discovery.network_scanner import NetworkDiscoveryEngine
from .discovery.asset_classifier import AssetClassifier
from .models import DiscoveredAsset, Organization

logger = logging.getLogger(__name__)


class DiscoveryService:
    """
    Tenant-scoped network discovery service
    
    Manages network scanning, asset classification, and result persistence
    per organization.
    """
    
    def __init__(self, organization_id: int, db: AsyncSession):
        self.organization_id = organization_id
        self.db = db
        self.scanner = NetworkDiscoveryEngine(timeout=2, max_workers=50)
        self.classifier = AssetClassifier()
        
    async def start_network_scan(
        self,
        network_ranges: List[str],
        port_ranges: Optional[List[int]] = None,
        scan_type: str = "quick"  # quick|full
    ) -> Dict:
        """
        Start a network discovery scan
        
        Args:
            network_ranges: List of CIDR ranges to scan
            port_ranges: Optional list of ports (uses defaults if None)
            scan_type: "quick" (common ports) or "full" (all ports)
            
        Returns:
            Scan summary with scan_id and initial status
        """
        scan_id = str(uuid.uuid4())
        
        logger.info(
            f"Starting {scan_type} network scan for org {self.organization_id}: "
            f"{len(network_ranges)} range(s), scan_id={scan_id}"
        )
        
        try:
            # Perform scan
            discovered_hosts = await self.scanner.comprehensive_scan(
                network_ranges=network_ranges,
                port_ranges=port_ranges
            )
            
            # Classify assets
            classified_hosts = self.classifier.classify_and_profile(discovered_hosts)
            
            # Persist to database
            asset_count = await self._persist_scan_results(scan_id, classified_hosts)
            
            logger.info(
                f"Scan {scan_id} complete: {asset_count} assets discovered for org {self.organization_id}"
            )
            
            return {
                "scan_id": scan_id,
                "status": "completed",
                "assets_discovered": asset_count,
                "network_ranges": network_ranges,
                "scan_type": scan_type,
                "completed_at": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Scan {scan_id} failed for org {self.organization_id}: {e}")
            return {
                "scan_id": scan_id,
                "status": "failed",
                "error": str(e),
                "assets_discovered": 0
            }
    
    async def _persist_scan_results(
        self,
        scan_id: str,
        classified_hosts: List[Dict]
    ) -> int:
        """
        Persist discovered assets to database
        
        Args:
            scan_id: Unique scan identifier
            classified_hosts: List of classified host dictionaries
            
        Returns:
            Number of assets persisted
        """
        asset_count = 0
        
        for host in classified_hosts:
            # Check if asset already exists for this org
            stmt = select(DiscoveredAsset).where(
                DiscoveredAsset.organization_id == self.organization_id,
                DiscoveredAsset.ip == host["ip"]
            )
            result = await self.db.execute(stmt)
            existing_asset = result.scalar_one_or_none()
            
            if existing_asset:
                # Update existing asset
                existing_asset.hostname = host.get("hostname")
                existing_asset.os_type = host.get("os_type", "unknown")
                existing_asset.os_role = host.get("os_role", "unknown")
                existing_asset.classification = host.get("classification", "Unknown")
                existing_asset.classification_confidence = host.get("classification_confidence", 0.0)
                existing_asset.open_ports = host.get("ports", [])
                existing_asset.services = host.get("services", {})
                existing_asset.deployment_profile = host.get("deployment_profile", {})
                existing_asset.deployment_priority = host.get("deployment_profile", {}).get("priority", "medium")
                existing_asset.agent_compatible = host.get("deployment_profile", {}).get("agent_compatible", True)
                existing_asset.last_seen = datetime.now(timezone.utc)
                existing_asset.scan_id = scan_id
            else:
                # Create new asset
                asset = DiscoveredAsset(
                    organization_id=self.organization_id,
                    ip=host["ip"],
                    hostname=host.get("hostname"),
                    os_type=host.get("os_type", "unknown"),
                    os_role=host.get("os_role", "unknown"),
                    classification=host.get("classification", "Unknown"),
                    classification_confidence=host.get("classification_confidence", 0.0),
                    open_ports=host.get("ports", []),
                    services=host.get("services", {}),
                    deployment_profile=host.get("deployment_profile", {}),
                    deployment_priority=host.get("deployment_profile", {}).get("priority", "medium"),
                    agent_compatible=host.get("deployment_profile", {}).get("agent_compatible", True),
                    scan_id=scan_id
                )
                self.db.add(asset)
            
            asset_count += 1
        
        await self.db.commit()
        return asset_count
    
    async def get_scan_results(
        self,
        scan_id: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict]:
        """
        Retrieve discovered assets for this organization
        
        Args:
            scan_id: Optional scan ID to filter by
            limit: Maximum number of results
            
        Returns:
            List of asset dictionaries
        """
        query = select(DiscoveredAsset).where(
            DiscoveredAsset.organization_id == self.organization_id
        )
        
        if scan_id:
            query = query.where(DiscoveredAsset.scan_id == scan_id)
        
        query = query.order_by(DiscoveredAsset.discovered_at.desc()).limit(limit)
        
        result = await self.db.execute(query)
        assets = result.scalars().all()
        
        return [
            {
                "id": asset.id,
                "ip": asset.ip,
                "hostname": asset.hostname,
                "os_type": asset.os_type,
                "os_role": asset.os_role,
                "classification": asset.classification,
                "classification_confidence": asset.classification_confidence,
                "open_ports": asset.open_ports,
                "services": asset.services,
                "deployment_profile": asset.deployment_profile,
                "deployment_priority": asset.deployment_priority,
                "agent_compatible": asset.agent_compatible,
                "discovered_at": asset.discovered_at.isoformat() if asset.discovered_at else None,
                "last_seen": asset.last_seen.isoformat() if asset.last_seen else None,
                "scan_id": asset.scan_id
            }
            for asset in assets
        ]
    
    async def generate_deployment_matrix(self) -> Dict:
        """
        Generate agent deployment recommendations
        
        Returns:
            Deployment matrix with priority groups and methods
        """
        # Get all assets for this org
        assets = await self.get_scan_results(limit=1000)
        
        return await self.scanner.generate_deployment_matrix(assets)



