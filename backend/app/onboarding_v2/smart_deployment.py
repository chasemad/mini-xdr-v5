"""
Smart agent deployment engine with priority-based rollout
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update

from ..models import CloudAsset, AgentEnrollment, Organization
from ..integrations.manager import IntegrationManager
from ..agent_enrollment_service import AgentEnrollmentService

logger = logging.getLogger(__name__)


class SmartDeploymentEngine:
    """Intelligent agent deployment with priority-based rollout"""

    def __init__(self, organization_id: int, db: AsyncSession):
        self.organization_id = organization_id
        self.db = db
        self.integration_manager = IntegrationManager(organization_id, db)
        self.enrollment_service = AgentEnrollmentService(db)
        self.deployment_status = {
            'status': 'not_started',  # not_started, deploying, completed, failed
            'progress': 0,
            'agents_deployed': 0,
            'total_assets': 0,
            'current_asset': None,
            'errors': []
        }
        logger.info(f"Initialized SmartDeploymentEngine for org {organization_id}")

    async def deploy_to_assets(self, assets: List[Dict[str, Any]], provider: str) -> Dict[str, Any]:
        """
        Deploy agents to discovered assets with intelligent priority-based rollout

        Args:
            assets: List of discovered assets from auto_discovery
            provider: Cloud provider name

        Returns:
            Deployment results
        """
        logger.info(f"Starting smart agent deployment to {len(assets)} assets")
        self.deployment_status['status'] = 'deploying'
        self.deployment_status['total_assets'] = len(assets)
        self.deployment_status['progress'] = 10

        try:
            # Get integration for deployment
            integration = await self.integration_manager.get_integration(provider)
            if not integration:
                raise Exception(f"No integration configured for {provider}")

            # Filter assets that are agent-compatible and not yet deployed
            deployable_assets = [
                a for a in assets
                if a.get('agent_compatible', False) and not a.get('agent_deployed', False)
            ]

            # Sort by priority: critical > high > medium > low
            priority_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
            deployable_assets.sort(key=lambda x: priority_order.get(x.get('priority', 'low'), 3))

            logger.info(f"Deploying to {len(deployable_assets)} compatible assets")
            self.deployment_status['progress'] = 20

            # Deploy agents
            deployment_results = await integration.deploy_agents(deployable_assets)
            self.deployment_status['agents_deployed'] = deployment_results.get('success', 0)
            self.deployment_status['progress'] = 80

            # Update database with deployment results
            await self._update_deployment_status(deployment_results)
            self.deployment_status['progress'] = 100
            self.deployment_status['status'] = 'completed'

            logger.info(f"Smart deployment completed: {deployment_results['success']} succeeded, {deployment_results['failed']} failed")
            return deployment_results

        except Exception as e:
            logger.error(f"Smart deployment failed: {e}")
            self.deployment_status['status'] = 'failed'
            self.deployment_status['errors'].append(str(e))
            raise

    async def _update_deployment_status(self, deployment_results: Dict[str, Any]):
        """Update database with deployment results"""

        for detail in deployment_results.get('details', []):
            asset_id = detail['asset_id']
            status = detail['status']

            # Update CloudAsset record
            stmt = (
                update(CloudAsset)
                .where(
                    CloudAsset.organization_id == self.organization_id,
                    CloudAsset.asset_id == asset_id
                )
                .values(
                    agent_status='deploying' if status == 'success' else 'failed',
                    deployment_method='ssm',
                    deployment_error=detail.get('error') if status != 'success' else None
                )
            )
            await self.db.execute(stmt)

        await self.db.commit()
        logger.info("Updated deployment status in database")

    async def get_status(self) -> Dict[str, Any]:
        """Get current deployment status"""
        return {
            'status': self.deployment_status['status'],
            'progress': self.deployment_status['progress'],
            'agents_deployed': self.deployment_status['agents_deployed'],
            'total_assets': self.deployment_status['total_assets'],
            'current_asset': self.deployment_status['current_asset'],
            'errors': self.deployment_status['errors']
        }

    async def get_deployment_summary(self) -> Dict[str, Any]:
        """Get deployment summary for the organization"""

        # Query CloudAsset for deployment stats
        stmt = select(CloudAsset).where(CloudAsset.organization_id == self.organization_id)
        result = await self.db.execute(stmt)
        assets = result.scalars().all()

        summary = {
            'total_assets': len(assets),
            'agent_deployed': sum(1 for a in assets if a.agent_deployed),
            'deployment_pending': sum(1 for a in assets if a.agent_status == 'pending'),
            'deployment_in_progress': sum(1 for a in assets if a.agent_status == 'deploying'),
            'deployment_failed': sum(1 for a in assets if a.agent_status == 'failed'),
            'incompatible': sum(1 for a in assets if a.agent_status == 'incompatible'),
            'by_priority': {
                'critical': 0,
                'high': 0,
                'medium': 0,
                'low': 0
            },
            'by_provider': {},
            'by_asset_type': {}
        }

        # Calculate breakdowns
        for asset in assets:
            # Priority breakdown
            priority = asset.asset_data.get('priority', 'low')
            if priority in summary['by_priority']:
                summary['by_priority'][priority] += 1

            # Provider breakdown
            provider = asset.provider
            if provider not in summary['by_provider']:
                summary['by_provider'][provider] = 0
            summary['by_provider'][provider] += 1

            # Asset type breakdown
            asset_type = asset.asset_type
            if asset_type not in summary['by_asset_type']:
                summary['by_asset_type'][asset_type] = 0
            summary['by_asset_type'][asset_type] += 1

        return summary

    async def retry_failed_deployments(self, provider: str) -> Dict[str, Any]:
        """
        Retry deployment for failed assets

        Args:
            provider: Cloud provider

        Returns:
            Retry results
        """
        logger.info(f"Retrying failed deployments for {provider}")

        # Get assets with failed deployment
        stmt = select(CloudAsset).where(
            CloudAsset.organization_id == self.organization_id,
            CloudAsset.provider == provider,
            CloudAsset.agent_status == 'failed'
        )
        result = await self.db.execute(stmt)
        failed_assets = result.scalars().all()

        if not failed_assets:
            logger.info("No failed deployments to retry")
            return {'success': 0, 'failed': 0, 'details': []}

        # Convert to asset format for deployment
        assets_to_retry = [
            {
                'provider': asset.provider,
                'asset_type': asset.asset_type,
                'asset_id': asset.asset_id,
                'region': asset.region,
                'asset_data': asset.asset_data,
                'agent_compatible': True,
                'priority': asset.asset_data.get('priority', 'medium')
            }
            for asset in failed_assets
        ]

        # Attempt redeployment
        return await self.deploy_to_assets(assets_to_retry, provider)

    async def check_deployment_health(self) -> Dict[str, Any]:
        """
        Check health of deployed agents

        Returns:
            Health check results
        """
        logger.info("Checking deployment health")

        # Query AgentEnrollment for health status
        stmt = select(AgentEnrollment).where(
            AgentEnrollment.organization_id == self.organization_id
        )
        result = await self.db.execute(stmt)
        enrollments = result.scalars().all()

        health = {
            'total_agents': len(enrollments),
            'active': sum(1 for e in enrollments if e.status == 'active'),
            'inactive': sum(1 for e in enrollments if e.status == 'inactive'),
            'pending': sum(1 for e in enrollments if e.status == 'pending'),
            'revoked': sum(1 for e in enrollments if e.status == 'revoked'),
            'last_heartbeat_summary': {
                'recent': 0,  # < 5 minutes
                'stale': 0,   # 5-30 minutes
                'old': 0      # > 30 minutes
            }
        }

        # Analyze heartbeat freshness
        now = datetime.now(timezone.utc)
        for enrollment in enrollments:
            if enrollment.last_heartbeat:
                age_minutes = (now - enrollment.last_heartbeat).total_seconds() / 60
                if age_minutes < 5:
                    health['last_heartbeat_summary']['recent'] += 1
                elif age_minutes < 30:
                    health['last_heartbeat_summary']['stale'] += 1
                else:
                    health['last_heartbeat_summary']['old'] += 1

        return health
