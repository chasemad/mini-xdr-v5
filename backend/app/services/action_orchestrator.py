"""
AI Action Orchestrator Service

Intelligently selects and executes vendor implementations based on high-level
security intents. Provides fallback handling, context-aware scoring, and
graceful degradation when integrations fail.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from app.database_models import IntegrationConfig
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


class IntentImplementation:
    """Represents a single vendor implementation of an intent"""

    def __init__(self, integration: IntegrationConfig, score: float = 0.0):
        self.integration = integration
        self.score = score
        self.vendor = integration.vendor
        self.category = integration.category
        self.health_status = integration.health_status
        self.priority = integration.priority


class ActionOrchestrator:
    """
    Intelligently selects and executes vendor implementations
    based on intent, available integrations, and context.
    """

    def __init__(self, db: AsyncSession):
        self.db = db

    async def execute_intent(
        self,
        intent: str,
        params: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Execute a high-level security intent using the best available vendor.

        Args:
            intent: High-level action intent (e.g., "block_ip", "isolate_host")
            params: Action parameters (e.g., {"ip": "1.2.3.4"})
            context: Optional execution context (incident details, urgency, etc.)

        Returns:
            Execution result with vendor info and status
        """
        context = context or {}

        logger.info(f"Orchestrating intent '{intent}' with params: {params}")

        # 1. Find available implementations
        implementations = await self.get_available_implementations(intent)

        if not implementations:
            logger.error(f"No implementations found for intent '{intent}'")
            return {
                "success": False,
                "error": f"No integrations available for intent '{intent}'",
                "intent": intent,
                "params": params,
            }

        # 2. Score and sort implementations
        scored = await self.score_implementations(implementations, params, context)
        sorted_impls = sorted(scored, key=lambda x: x.score, reverse=True)

        logger.info(
            f"Found {len(sorted_impls)} implementations, trying in priority order"
        )

        # 3. Try implementations with fallback
        errors = []
        for impl in sorted_impls:
            try:
                logger.info(f"Attempting {impl.vendor} (score: {impl.score})")
                result = await self.execute_implementation(
                    impl, intent, params, context
                )

                if result.get("success"):
                    logger.info(f"Successfully executed via {impl.vendor}")
                    return {
                        **result,
                        "intent": intent,
                        "vendor_used": impl.vendor,
                        "vendor_display_name": impl.integration.vendor_display_name,
                        "fallbacks_attempted": len(errors),
                    }
                else:
                    error_msg = result.get("error", "Unknown error")
                    errors.append({"vendor": impl.vendor, "error": error_msg})
                    logger.warning(f"{impl.vendor} failed: {error_msg}")

            except Exception as e:
                error_msg = str(e)
                errors.append({"vendor": impl.vendor, "error": error_msg})
                logger.error(f"Exception executing {impl.vendor}: {error_msg}")
                continue

        # All implementations failed
        logger.error(f"All implementations failed for intent '{intent}'")
        return {
            "success": False,
            "error": "All vendor implementations failed",
            "intent": intent,
            "params": params,
            "failures": errors,
        }

    async def get_available_implementations(
        self, intent: str
    ) -> List[IntegrationConfig]:
        """Find all enabled integrations that support this intent"""
        query = select(IntegrationConfig).where(IntegrationConfig.enabled == True)

        result = await self.db.execute(query)
        all_integrations = result.scalars().all()

        # Filter by capability
        capable = []
        for integration in all_integrations:
            if integration.capabilities and intent in integration.capabilities:
                capable.append(integration)

        return capable

    async def score_implementations(
        self,
        implementations: List[IntegrationConfig],
        params: Dict[str, Any],
        context: Dict[str, Any],
    ) -> List[IntentImplementation]:
        """
        Score each implementation based on:
        - Priority (from config)
        - Health status
        - Historical performance
        - Context awareness
        """
        scored = []

        for integration in implementations:
            score = 0.0

            # Base score from priority (1-10 range, lower is better)
            # Convert to positive score where higher priority = higher score
            priority_score = (10 - integration.priority) * 10
            score += priority_score

            # Health status bonus
            if integration.health_status == "healthy":
                score += 50
            elif integration.health_status == "degraded":
                score += 20
            elif integration.health_status == "offline":
                score -= 100  # Deprioritize but don't eliminate

            # Success rate bonus
            if integration.total_executions > 0:
                success_rate = (
                    integration.successful_executions / integration.total_executions
                )
                score += success_rate * 30

            # Performance bonus (fast response times)
            if integration.average_response_time_ms:
                if integration.average_response_time_ms < 1000:  # Under 1s
                    score += 20
                elif integration.average_response_time_ms < 5000:  # Under 5s
                    score += 10

            # Context-aware scoring (example: prefer certain vendors for specific scenarios)
            if context.get("urgency") == "critical":
                # Prefer faster tools for critical incidents
                if (
                    integration.average_response_time_ms
                    and integration.average_response_time_ms < 2000
                ):
                    score += 15

            scored.append(IntentImplementation(integration, score))

        return scored

    async def execute_implementation(
        self,
        impl: IntentImplementation,
        intent: str,
        params: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Execute a specific vendor implementation.

        For now, this is a placeholder that will be extended with actual
        vendor API calls. Each vendor will have its own executor module.
        """
        start_time = datetime.now()

        try:
            # TODO: Route to actual vendor-specific executors
            # For now, simulate execution
            logger.info(f"Executing {intent} via {impl.vendor} with params {params}")

            # Placeholder: In real implementation, this would call:
            # - PaloAltoExecutor.block_ip(params) for Palo Alto
            # - CiscoExecutor.block_ip(params) for Cisco
            # - CrowdStrikeExecutor.isolate_host(params) for CrowdStrike
            # etc.

            # For demo purposes, return success
            execution_time = (datetime.now() - start_time).total_seconds() * 1000

            # Update integration statistics
            impl.integration.total_executions += 1
            impl.integration.successful_executions += 1

            # Update average response time
            if impl.integration.average_response_time_ms:
                impl.integration.average_response_time_ms = (
                    impl.integration.average_response_time_ms * 0.8
                    + execution_time * 0.2
                )
            else:
                impl.integration.average_response_time_ms = execution_time

            await self.db.commit()

            return {
                "success": True,
                "message": f"Successfully executed {intent} via {impl.vendor}",
                "execution_time_ms": execution_time,
                "result_data": {
                    "intent": intent,
                    "params": params,
                    "vendor": impl.vendor,
                },
            }

        except Exception as e:
            # Update failure statistics
            impl.integration.total_executions += 1
            impl.integration.failed_executions += 1
            await self.db.commit()

            return {"success": False, "error": str(e)}

    async def get_vendors_for_intent(self, intent: str) -> List[Dict[str, Any]]:
        """
        Get list of vendors that can handle a specific intent.
        Useful for UI display.
        """
        implementations = await self.get_available_implementations(intent)

        return [
            {
                "vendor": impl.vendor,
                "vendor_display_name": impl.vendor_display_name,
                "category": impl.category,
                "health_status": impl.health_status,
                "priority": impl.priority,
                "enabled": impl.enabled,
            }
            for impl in implementations
        ]
