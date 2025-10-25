"""
Integration Manager for cloud provider credentials and lifecycle management
"""

import base64
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from ..models import IntegrationCredentials
from ..secrets_manager import secrets_manager
from .aws import AWSIntegration
from .azure import AzureIntegration
from .base import CloudIntegration
from .gcp import GCPIntegration

logger = logging.getLogger(__name__)


class IntegrationManager:
    """Manages cloud provider integrations and credential lifecycle"""

    def __init__(self, organization_id: int, db: AsyncSession):
        self.organization_id = organization_id
        self.db = db
        self.integrations = {
            "aws": AWSIntegration,
            "azure": AzureIntegration,
            "gcp": GCPIntegration,
        }
        logger.info(f"Initialized IntegrationManager for org {organization_id}")

    async def setup_integration(
        self, provider: str, credentials: Dict[str, Any]
    ) -> bool:
        """
        Setup and validate integration credentials

        Args:
            provider: Cloud provider name (aws, azure, gcp)
            credentials: Unencrypted credentials dict

        Returns:
            bool: True if setup successful
        """
        if provider not in self.integrations:
            logger.error(f"Unknown provider: {provider}")
            return False

        logger.info(f"Setting up {provider} integration for org {self.organization_id}")

        # Create integration instance
        integration_class = self.integrations[provider]
        integration = integration_class(self.organization_id, credentials)

        # Test authentication
        try:
            if not await integration.authenticate():
                logger.error(f"Authentication failed for {provider}")
                return False
        except Exception as e:
            logger.error(f"Authentication error for {provider}: {e}")
            return False

        # Validate permissions
        permissions = await integration.validate_permissions()
        required_permissions = ["read_compute", "read_network", "deploy_agents"]
        missing_permissions = [
            p for p in required_permissions if not permissions.get(p, False)
        ]

        if missing_permissions:
            logger.warning(f"Missing permissions for {provider}: {missing_permissions}")
            # We'll allow setup even with missing permissions, but log the warning
            # Some permissions (like deploy_agents) might not be available until later

        # Detect credential type
        credential_type = self._detect_credential_type(provider, credentials)

        # Store encrypted credentials
        await self._store_credentials(provider, credential_type, credentials)

        logger.info(
            f"Successfully set up {provider} integration for org {self.organization_id}"
        )
        return True

    async def get_integration(self, provider: str) -> Optional[CloudIntegration]:
        """
        Get configured integration instance with decrypted credentials

        Args:
            provider: Cloud provider name

        Returns:
            CloudIntegration instance or None if not configured
        """
        if provider not in self.integrations:
            logger.error(f"Unknown provider: {provider}")
            return None

        # Retrieve and decrypt credentials
        credentials = await self._get_credentials(provider)
        if not credentials:
            logger.warning(
                f"No credentials found for {provider} in org {self.organization_id}"
            )
            return None

        # Create and authenticate integration instance
        integration_class = self.integrations[provider]
        integration = integration_class(self.organization_id, credentials)

        # Authenticate the integration
        try:
            if not await integration.authenticate():
                logger.error(f"Failed to authenticate {provider} integration")
                return None
        except Exception as e:
            logger.error(f"Authentication error for {provider}: {e}")
            return None

        return integration

    async def list_integrations(self) -> List[Dict[str, Any]]:
        """
        List all configured integrations for this organization

        Returns:
            List of integration metadata
        """
        integrations = []

        for provider in self.integrations.keys():
            # Query database for this provider's credentials
            stmt = select(IntegrationCredentials).where(
                IntegrationCredentials.organization_id == self.organization_id,
                IntegrationCredentials.provider == provider,
            )
            result = await self.db.execute(stmt)
            cred_record = result.scalars().first()

            if cred_record:
                # Test the integration to get current status
                integration = await self.get_integration(provider)
                if integration:
                    status = (
                        "connected" if await integration.test_connection() else "error"
                    )
                else:
                    status = "error"

                integrations.append(
                    {
                        "provider": provider,
                        "status": status,
                        "credential_type": cred_record.credential_type,
                        "configured_at": cred_record.created_at.isoformat()
                        if cred_record.created_at
                        else None,
                        "last_used_at": cred_record.last_used_at.isoformat()
                        if cred_record.last_used_at
                        else None,
                        "expires_at": cred_record.expires_at.isoformat()
                        if cred_record.expires_at
                        else None,
                    }
                )

        return integrations

    async def remove_integration(self, provider: str) -> bool:
        """
        Remove an integration and its credentials

        Args:
            provider: Cloud provider name

        Returns:
            bool: True if removed successfully
        """
        logger.info(f"Removing {provider} integration for org {self.organization_id}")

        # Delete credentials from database
        stmt = select(IntegrationCredentials).where(
            IntegrationCredentials.organization_id == self.organization_id,
            IntegrationCredentials.provider == provider,
        )
        result = await self.db.execute(stmt)
        cred_record = result.scalars().first()

        if cred_record:
            await self.db.delete(cred_record)
            await self.db.commit()
            logger.info(f"Successfully removed {provider} integration")
            return True
        else:
            logger.warning(f"No integration found for {provider}")
            return False

    async def update_last_used(self, provider: str):
        """Update last_used_at timestamp for an integration"""
        stmt = (
            update(IntegrationCredentials)
            .where(
                IntegrationCredentials.organization_id == self.organization_id,
                IntegrationCredentials.provider == provider,
            )
            .values(last_used_at=datetime.now(timezone.utc))
        )
        await self.db.execute(stmt)
        await self.db.commit()

    # =========================================================================
    # PRIVATE METHODS - Credential Encryption/Decryption
    # =========================================================================

    async def _store_credentials(
        self, provider: str, credential_type: str, credentials: Dict[str, Any]
    ):
        """Store encrypted credentials in database"""

        # Encrypt the credentials
        encrypted_data = self._encrypt_credentials(credentials)

        # Check if credentials already exist
        stmt = select(IntegrationCredentials).where(
            IntegrationCredentials.organization_id == self.organization_id,
            IntegrationCredentials.provider == provider,
        )
        result = await self.db.execute(stmt)
        existing_record = result.scalars().first()

        if existing_record:
            # Update existing record
            existing_record.credential_data = encrypted_data
            existing_record.credential_type = credential_type
            existing_record.updated_at = datetime.now(timezone.utc)
            existing_record.status = "active"
        else:
            # Create new record
            cred_record = IntegrationCredentials(
                organization_id=self.organization_id,
                provider=provider,
                credential_type=credential_type,
                credential_data=encrypted_data,
                status="active",
            )
            self.db.add(cred_record)

        await self.db.commit()
        logger.info(f"Stored credentials for {provider}")

    async def _get_credentials(self, provider: str) -> Optional[Dict[str, Any]]:
        """Retrieve and decrypt credentials from database"""

        stmt = select(IntegrationCredentials).where(
            IntegrationCredentials.organization_id == self.organization_id,
            IntegrationCredentials.provider == provider,
        )
        result = await self.db.execute(stmt)
        cred_record = result.scalars().first()

        if not cred_record:
            return None

        # Check if credentials are expired
        if cred_record.expires_at and cred_record.expires_at < datetime.now(
            timezone.utc
        ):
            logger.warning(f"Credentials for {provider} have expired")
            cred_record.status = "expired"
            await self.db.commit()
            return None

        # Check if credentials are active
        if cred_record.status != "active":
            logger.warning(
                f"Credentials for {provider} are not active (status: {cred_record.status})"
            )
            return None

        # Decrypt and return credentials
        return self._decrypt_credentials(cred_record.credential_data)

    def _encrypt_credentials(self, credentials: Dict[str, Any]) -> Dict[str, Any]:
        """
        Encrypt sensitive credential data

        In production, this should use:
        - AWS KMS for key management
        - Organization-specific encryption keys
        - Proper key rotation

        For MVP, we'll use a simple approach with base64 encoding
        (Note: This is NOT production-ready encryption!)
        """
        # Convert credentials to JSON
        credentials_json = json.dumps(credentials)

        # In production, use proper encryption:
        # from cryptography.fernet import Fernet
        # encryption_key = self._get_org_encryption_key()
        # f = Fernet(encryption_key)
        # encrypted = f.encrypt(credentials_json.encode())

        # For MVP: Simple base64 encoding (NOT secure for production!)
        encrypted = base64.b64encode(credentials_json.encode()).decode()

        return {
            "encrypted": True,
            "data": encrypted,
            "encryption_method": "base64_mvp",  # In production: 'fernet' or 'kms'
        }

    def _decrypt_credentials(
        self, encrypted_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Decrypt credential data

        Matches the encryption method used in _encrypt_credentials
        """
        try:
            if not encrypted_data.get("encrypted"):
                # Data is not encrypted (legacy?)
                return encrypted_data

            encryption_method = encrypted_data.get("encryption_method", "base64_mvp")
            encrypted_payload = encrypted_data["data"]

            if encryption_method == "base64_mvp":
                # Simple base64 decoding (MVP only!)
                decrypted = base64.b64decode(encrypted_payload.encode()).decode()
                return json.loads(decrypted)
            elif encryption_method == "fernet":
                # Production Fernet decryption
                # from cryptography.fernet import Fernet
                # encryption_key = self._get_org_encryption_key()
                # f = Fernet(encryption_key)
                # decrypted = f.decrypt(encrypted_payload.encode())
                # return json.loads(decrypted)
                raise NotImplementedError("Fernet decryption not yet implemented")
            else:
                logger.error(f"Unknown encryption method: {encryption_method}")
                return None

        except Exception as e:
            logger.error(f"Failed to decrypt credentials: {e}")
            return None

    def _detect_credential_type(
        self, provider: str, credentials: Dict[str, Any]
    ) -> str:
        """Detect the type of credentials being used"""

        if provider == "aws":
            if "role_arn" in credentials:
                return "assume_role"
            elif "aws_access_key_id" in credentials:
                return "access_key"
            else:
                return "unknown"

        elif provider == "azure":
            if "client_id" in credentials and "client_secret" in credentials:
                return "service_principal"
            else:
                return "unknown"

        elif provider == "gcp":
            if "service_account_key" in credentials:
                return "service_account"
            else:
                return "unknown"

        return "unknown"

    def _get_org_encryption_key(self) -> bytes:
        """
        Get organization-specific encryption key

        In production, this should:
        - Use AWS KMS or similar
        - Generate unique keys per organization
        - Support key rotation
        - Store keys securely

        For MVP, we'll use a simple master key from environment
        """
        # Production approach:
        # key_id = f"org-{self.organization_id}-encryption-key"
        # return kms_client.get_data_key(KeyId=key_id)

        # MVP approach: Use master key from environment or Secrets Manager
        master_key = os.getenv("INTEGRATION_ENCRYPTION_KEY")
        if not master_key:
            # Try to get from Secrets Manager
            master_key = secrets_manager.get_secret(
                "mini-xdr/integration-encryption-key"
            )

        if not master_key:
            # Generate a default key (NOT for production!)
            logger.warning("Using default encryption key - NOT for production!")
            master_key = "mini-xdr-default-key-change-in-production-" + str(
                self.organization_id
            )

        # In production, use proper key derivation
        # from cryptography.hazmat.primitives import hashes
        # from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
        # ...

        return master_key.encode()[:32]  # Simplified for MVP
