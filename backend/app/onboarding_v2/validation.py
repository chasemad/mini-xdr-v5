"""
Onboarding validation service
"""

import logging
from typing import Dict, List, Any
from datetime import datetime, timezone, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from ..models import CloudAsset, AgentEnrollment, Organization, Event

logger = logging.getLogger(__name__)


class OnboardingValidator:
    """Validates successful onboarding completion"""

    def __init__(self, organization_id: int, db: AsyncSession):
        self.organization_id = organization_id
        self.db = db
        self.validation_status = {
            'status': 'not_started',  # not_started, validating, completed, failed
            'progress': 0,
            'checks_passed': 0,
            'total_checks': 5,
            'checks': {}
        }
        logger.info(f"Initialized OnboardingValidator for org {organization_id}")

    async def validate_deployment(self) -> Dict[str, Any]:
        """
        Run all validation checks

        Returns:
            Validation results
        """
        logger.info("Starting onboarding validation")
        self.validation_status['status'] = 'validating'
        self.validation_status['progress'] = 10

        checks = [
            self._check_assets_discovered(),
            self._check_agents_enrolled(),
            self._check_agents_active(),
            self._check_telemetry_flowing(),
            self._check_integration_healthy()
        ]

        results = {}
        for i, check_coro in enumerate(checks):
            try:
                check_result = await check_coro
                check_name = check_result['name']
                results[check_name] = check_result

                if check_result['passed']:
                    self.validation_status['checks_passed'] += 1

                # Update progress
                self.validation_status['progress'] = 20 + (i + 1) * 16  # 20-100%

            except Exception as e:
                logger.error(f"Validation check failed: {e}")
                results[f"check_{i}"] = {
                    'name': f"check_{i}",
                    'passed': False,
                    'error': str(e)
                }

        # Determine overall status
        all_passed = all(r.get('passed', False) for r in results.values())
        self.validation_status['status'] = 'completed' if all_passed else 'failed'
        self.validation_status['progress'] = 100
        self.validation_status['checks'] = results

        logger.info(f"Validation completed: {self.validation_status['checks_passed']}/{self.validation_status['total_checks']} checks passed")
        return results

    async def _check_assets_discovered(self) -> Dict[str, Any]:
        """Check if assets were discovered"""
        logger.info("Checking if assets were discovered")

        stmt = select(CloudAsset).where(CloudAsset.organization_id == self.organization_id)
        result = await self.db.execute(stmt)
        assets = result.scalars().all()

        passed = len(assets) > 0

        return {
            'name': 'assets_discovered',
            'passed': passed,
            'message': f"Found {len(assets)} cloud assets" if passed else "No cloud assets discovered",
            'details': {
                'total_assets': len(assets),
                'by_provider': {},
                'by_type': {}
            }
        }

    async def _check_agents_enrolled(self) -> Dict[str, Any]:
        """Check if agents have been enrolled"""
        logger.info("Checking if agents are enrolled")

        stmt = select(AgentEnrollment).where(
            AgentEnrollment.organization_id == self.organization_id,
            AgentEnrollment.status != 'revoked'
        )
        result = await self.db.execute(stmt)
        enrollments = result.scalars().all()

        passed = len(enrollments) > 0

        return {
            'name': 'agents_enrolled',
            'passed': passed,
            'message': f"{len(enrollments)} agents enrolled" if passed else "No agents enrolled yet",
            'details': {
                'total_enrolled': len(enrollments),
                'active': sum(1 for e in enrollments if e.status == 'active'),
                'pending': sum(1 for e in enrollments if e.status == 'pending'),
                'inactive': sum(1 for e in enrollments if e.status == 'inactive')
            }
        }

    async def _check_agents_active(self) -> Dict[str, Any]:
        """Check if enrolled agents are reporting heartbeats"""
        logger.info("Checking if agents are active")

        stmt = select(AgentEnrollment).where(
            AgentEnrollment.organization_id == self.organization_id,
            AgentEnrollment.status == 'active'
        )
        result = await self.db.execute(stmt)
        active_agents = result.scalars().all()

        # Check for recent heartbeats (within last 10 minutes)
        recent_threshold = datetime.now(timezone.utc) - timedelta(minutes=10)
        agents_with_recent_heartbeat = [
            a for a in active_agents
            if a.last_heartbeat and a.last_heartbeat > recent_threshold
        ]

        passed = len(agents_with_recent_heartbeat) > 0

        return {
            'name': 'agents_active',
            'passed': passed,
            'message': f"{len(agents_with_recent_heartbeat)} agents actively reporting" if passed else "No active agents with recent heartbeats",
            'details': {
                'active_agents': len(active_agents),
                'recent_heartbeats': len(agents_with_recent_heartbeat),
                'threshold_minutes': 10
            }
        }

    async def _check_telemetry_flowing(self) -> Dict[str, Any]:
        """Check if telemetry data is being received"""
        logger.info("Checking if telemetry is flowing")

        # Check for recent events (within last 30 minutes)
        recent_threshold = datetime.now(timezone.utc) - timedelta(minutes=30)

        stmt = select(Event).where(
            Event.organization_id == self.organization_id,
            Event.ts > recent_threshold
        )
        result = await self.db.execute(stmt)
        recent_events = result.scalars().all()

        passed = len(recent_events) > 0

        return {
            'name': 'telemetry_flowing',
            'passed': passed,
            'message': f"Receiving telemetry: {len(recent_events)} events in last 30 minutes" if passed else "No recent telemetry data",
            'details': {
                'recent_events': len(recent_events),
                'threshold_minutes': 30,
                'by_source_type': {}
            }
        }

    async def _check_integration_healthy(self) -> Dict[str, Any]:
        """Check if cloud integrations are healthy"""
        logger.info("Checking integration health")

        # Query for integration credentials
        from ..models import IntegrationCredentials

        stmt = select(IntegrationCredentials).where(
            IntegrationCredentials.organization_id == self.organization_id,
            IntegrationCredentials.status == 'active'
        )
        result = await self.db.execute(stmt)
        integrations = result.scalars().all()

        passed = len(integrations) > 0

        integration_details = {
            'total_integrations': len(integrations),
            'providers': [i.provider for i in integrations],
            'all_active': all(i.status == 'active' for i in integrations)
        }

        return {
            'name': 'integration_healthy',
            'passed': passed,
            'message': f"{len(integrations)} cloud integrations active" if passed else "No active cloud integrations",
            'details': integration_details
        }

    async def get_status(self) -> Dict[str, Any]:
        """Get current validation status"""
        return {
            'status': self.validation_status['status'],
            'progress': self.validation_status['progress'],
            'checks_passed': self.validation_status['checks_passed'],
            'total_checks': self.validation_status['total_checks'],
            'checks': self.validation_status['checks']
        }

    async def get_validation_summary(self) -> Dict[str, Any]:
        """Get a complete validation summary"""

        # Run validation if not already done
        if self.validation_status['status'] == 'not_started':
            await self.validate_deployment()

        # Get organization status
        stmt = select(Organization).where(Organization.id == self.organization_id)
        result = await self.db.execute(stmt)
        org = result.scalars().first()

        summary = {
            'organization_id': self.organization_id,
            'organization_name': org.name if org else None,
            'onboarding_status': org.onboarding_status if org else None,
            'validation_status': self.validation_status['status'],
            'checks_passed': self.validation_status['checks_passed'],
            'total_checks': self.validation_status['total_checks'],
            'pass_rate': f"{(self.validation_status['checks_passed'] / self.validation_status['total_checks'] * 100):.1f}%",
            'checks': self.validation_status['checks'],
            'timestamp': datetime.now(timezone.utc).isoformat()
        }

        return summary
