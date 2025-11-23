"""
Cloud provider integrations for seamless onboarding
"""

from .azure import AzureIntegration
from .base import CloudIntegration
from .gcp import GCPIntegration
from .manager import IntegrationManager

try:
    from .aws import (  # Optional, available when AWS integration is present
        AWSIntegration,
    )
except Exception:  # pragma: no cover - best-effort optional import
    AWSIntegration = None

__all__ = [
    "CloudIntegration",
    "IntegrationManager",
    "AzureIntegration",
    "GCPIntegration",
    "AWSIntegration",
]
