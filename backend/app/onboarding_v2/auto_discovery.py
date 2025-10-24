"""
Auto-discovery engine for cloud assets
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update

from ..models import CloudAsset, Organization
from ..integrations.manager import IntegrationManager

logger = logging.getLogger(__name__)


class AutoDiscoveryEngine:
    """Automated cloud asset discovery"""

    def __init__(self, organization_id: int, db: AsyncSession):
        self.organization_id = organization_id
        self.db = db
        self.integration_manager = IntegrationManager(organization_id, db)
        self.discovery_status = {
            'status': 'not_started',  # not_started, discovering, completed, failed
            'progress': 0,
            'assets_found': 0,
            'regions_scanned': 0,
            'total_regions': 0,
            'current_region': None,
            'errors': []
        }
        logger.info(f"Initialized AutoDiscoveryEngine for org {organization_id}")

    async def discover_cloud_assets(self, provider: str) -> List[Dict[str, Any]]:
        """
        Discover assets from a cloud provider

        Args:
            provider: Cloud provider name (aws, azure, gcp)

        Returns:
            List of discovered assets
        """
        logger.info(f"Starting cloud asset discovery for {provider}")
        self.discovery_status['status'] = 'discovering'
        self.discovery_status['progress'] = 10

        try:
            # Get integration for the provider
            integration = await self.integration_manager.get_integration(provider)
            if not integration:
                raise Exception(f"No integration configured for {provider}")

            # Update last used timestamp
            await self.integration_manager.update_last_used(provider)

            # Get regions
            regions = await integration.get_regions()
            self.discovery_status['total_regions'] = len(regions)
            self.discovery_status['progress'] = 20
            logger.info(f"Scanning {len(regions)} regions")

            # Discover assets
            assets = await integration.discover_assets()
            self.discovery_status['assets_found'] = len(assets)
            self.discovery_status['regions_scanned'] = len(regions)
            self.discovery_status['progress'] = 80
            logger.info(f"Discovered {len(assets)} assets")

            # Store discovered assets in database
            await self._store_discovered_assets(assets)
            self.discovery_status['progress'] = 100
            self.discovery_status['status'] = 'completed'

            logger.info(f"Asset discovery completed: {len(assets)} assets")
            return assets

        except Exception as e:
            logger.error(f"Asset discovery failed: {e}")
            self.discovery_status['status'] = 'failed'
            self.discovery_status['errors'].append(str(e))
            raise

    async def _store_discovered_assets(self, assets: List[Dict[str, Any]]):
        """Store or update discovered assets in database"""

        for asset_data in assets:
            # Check if asset already exists
            stmt = select(CloudAsset).where(
                CloudAsset.organization_id == self.organization_id,
                CloudAsset.provider == asset_data['provider'],
                CloudAsset.asset_id == asset_data['asset_id']
            )
            result = await self.db.execute(stmt)
            existing_asset = result.scalars().first()

            if existing_asset:
                # Update existing asset
                existing_asset.asset_data = asset_data['asset_data']
                existing_asset.region = asset_data.get('region')
                existing_asset.last_seen_at = datetime.now(timezone.utc)
                existing_asset.tags = asset_data['asset_data'].get('tags', {})

                # Update agent deployment eligibility
                if asset_data.get('agent_compatible') and not existing_asset.agent_deployed:
                    existing_asset.agent_status = 'pending'

            else:
                # Create new asset
                new_asset = CloudAsset(
                    organization_id=self.organization_id,
                    provider=asset_data['provider'],
                    asset_type=asset_data['asset_type'],
                    asset_id=asset_data['asset_id'],
                    region=asset_data.get('region'),
                    asset_data=asset_data['asset_data'],
                    tags=asset_data['asset_data'].get('tags', {}),
                    agent_deployed=False,
                    agent_status='pending' if asset_data.get('agent_compatible') else 'incompatible'
                )
                self.db.add(new_asset)

        await self.db.commit()
        logger.info(f"Stored {len(assets)} assets in database")

    async def get_status(self) -> Dict[str, Any]:
        """Get current discovery status"""
        return {
            'status': self.discovery_status['status'],
            'progress': self.discovery_status['progress'],
            'assets_found': self.discovery_status['assets_found'],
            'regions_scanned': self.discovery_status['regions_scanned'],
            'total_regions': self.discovery_status['total_regions'],
            'current_region': self.discovery_status['current_region'],
            'errors': self.discovery_status['errors']
        }

    async def get_discovered_assets(self, provider: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get discovered assets from database

        Args:
            provider: Filter by provider (optional)

        Returns:
            List of discovered assets
        """
        stmt = select(CloudAsset).where(CloudAsset.organization_id == self.organization_id)

        if provider:
            stmt = stmt.where(CloudAsset.provider == provider)

        result = await self.db.execute(stmt)
        assets = result.scalars().all()

        return [
            {
                'id': asset.id,
                'provider': asset.provider,
                'asset_type': asset.asset_type,
                'asset_id': asset.asset_id,
                'region': asset.region,
                'asset_data': asset.asset_data,
                'tags': asset.tags,
                'discovered_at': asset.discovered_at.isoformat() if asset.discovered_at else None,
                'last_seen_at': asset.last_seen_at.isoformat() if asset.last_seen_at else None,
                'agent_deployed': asset.agent_deployed,
                'agent_status': asset.agent_status
            }
            for asset in assets
        ]

    async def refresh_discovery(self, provider: str) -> List[Dict[str, Any]]:
        """
        Refresh asset discovery for a provider

        Args:
            provider: Cloud provider to refresh

        Returns:
            Updated list of assets
        """
        logger.info(f"Refreshing asset discovery for {provider}")

        # Reset status
        self.discovery_status = {
            'status': 'not_started',
            'progress': 0,
            'assets_found': 0,
            'regions_scanned': 0,
            'total_regions': 0,
            'current_region': None,
            'errors': []
        }

        # Run discovery again
        return await self.discover_cloud_assets(provider)
