"""
Base class for cloud provider integrations
"""

import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class CloudIntegration(ABC):
    """Base class for cloud provider integrations"""

    def __init__(self, organization_id: int, credentials: Dict[str, Any]):
        """
        Initialize cloud integration

        Args:
            organization_id: Organization ID for multi-tenancy
            credentials: Decrypted credentials for cloud provider access
        """
        self.organization_id = organization_id
        self.credentials = credentials
        self.client = None
        logger.info(f"Initialized {self.__class__.__name__} for org {organization_id}")

    @abstractmethod
    async def authenticate(self) -> bool:
        """
        Authenticate with the cloud provider

        Returns:
            bool: True if authentication successful, False otherwise
        """
        pass

    @abstractmethod
    async def discover_assets(self) -> List[Dict[str, Any]]:
        """
        Discover all assets in the cloud environment

        Returns:
            List of discovered assets with metadata
        """
        pass

    @abstractmethod
    async def deploy_agents(self, assets: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Deploy agents to discovered assets

        Args:
            assets: List of assets to deploy agents to

        Returns:
            Deployment results with success/failure counts
        """
        pass

    @abstractmethod
    async def get_regions(self) -> List[str]:
        """
        Get available regions/zones

        Returns:
            List of region/zone identifiers
        """
        pass

    async def validate_permissions(self) -> Dict[str, bool]:
        """
        Validate required permissions for integration

        Returns:
            Dict of permission checks with boolean results
        """
        # Default implementation - override in subclasses for specific checks
        return {
            "read_compute": False,
            "read_network": False,
            "read_storage": False,
            "deploy_agents": False,
        }

    async def test_connection(self) -> bool:
        """
        Test connection to cloud provider

        Returns:
            bool: True if connection successful
        """
        try:
            return await self.authenticate()
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False

    def _get_asset_priority(self, asset: Dict[str, Any]) -> str:
        """
        Determine deployment priority for an asset

        Args:
            asset: Asset metadata

        Returns:
            Priority level: critical, high, medium, low
        """
        asset_type = asset.get("asset_type", "").lower()

        # Critical: Database servers, domain controllers
        if any(
            keyword in asset_type
            for keyword in [
                "db",
                "database",
                "rds",
                "sql",
                "postgres",
                "mysql",
                "oracle",
            ]
        ):
            return "critical"
        if "dc" in asset.get("data", {}).get("tags", {}).get("Name", "").lower():
            return "critical"

        # High: Web servers, application servers
        if any(keyword in asset_type for keyword in ["web", "app", "api"]):
            return "high"

        # Medium: Worker instances, batch processing
        if any(keyword in asset_type for keyword in ["worker", "batch", "job"]):
            return "medium"

        # Low: Everything else
        return "low"
