from pydantic_settings import BaseSettings
from typing import Optional, List
import os
from pathlib import Path


def _load_secrets_on_init():
    """Load secrets from AWS Secrets Manager before Settings initialization"""
    # Only load secrets if not already done and if enabled
    if (os.getenv('SECRETS_MANAGER_ENABLED', 'false').lower() == 'true' and
        not os.getenv('_SECRETS_LOADED')):
        try:
            # Import here to avoid circular imports
            from .secrets_manager import load_common_secrets
            secrets = load_common_secrets()

            # Set environment variables for Settings to pick up
            secrets_loaded = 0
            for key, value in secrets.items():
                if value:
                    os.environ[key] = value
                    secrets_loaded += 1

            if secrets_loaded > 0:
                os.environ['_SECRETS_LOADED'] = 'true'
                print(f"✅ Loaded {secrets_loaded} secrets from AWS Secrets Manager for Settings")

        except ImportError:
            pass
        except Exception as e:
            print(f"Warning: Could not load secrets during Settings init: {e}")

# Load secrets before Settings class is instantiated
_load_secrets_on_init()


class Settings(BaseSettings):
    # API Configuration
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    ui_origin: str = "http://localhost:3000"
    api_key: Optional[str] = None
    
    # Authentication
    JWT_SECRET_KEY: Optional[str] = None
    ENCRYPTION_KEY: Optional[str] = None

    # Database
    database_url: str = "sqlite+aiosqlite:///./xdr.db"

    # Detection Configuration
    fail_window_seconds: int = 60
    fail_threshold: int = 6
    auto_contain: bool = False
    
    # Containment Configuration
    allow_private_ip_blocking: bool = True  # Enable for testing simulated attacks

    # Honeypot Configuration - UPDATED FOR TPOT
    honeypot_host: str = "34.193.101.171"
    honeypot_user: str = "admin"
    honeypot_ssh_key: str = "~/.ssh/mini-xdr-tpot-key.pem"
    honeypot_ssh_port: int = 64295

    # LLM Configuration
    llm_provider: str = "openai"
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4"
    xai_api_key: Optional[str] = None
    xai_model: str = "grok-beta"

    # Enhanced ML and AI Agent Configuration
    abuseipdb_api_key: Optional[str] = None
    virustotal_api_key: Optional[str] = None
    ml_models_path: str = "./models"
    policies_path: str = "../policies"
    auto_retrain_enabled: bool = True
    agent_api_key: Optional[str] = None
    
    # T-Pot Honeypot Integration
    tpot_api_key: Optional[str] = None
    tpot_host: Optional[str] = None
    tpot_ssh_port: Optional[int] = None
    tpot_web_port: Optional[int] = None

    # Agent HMAC Credentials
    containment_agent_device_id: Optional[str] = None
    containment_agent_public_id: Optional[str] = None
    containment_agent_hmac_key: Optional[str] = None
    containment_agent_secret: Optional[str] = None
    
    attribution_agent_device_id: Optional[str] = None
    attribution_agent_public_id: Optional[str] = None
    attribution_agent_hmac_key: Optional[str] = None
    attribution_agent_secret: Optional[str] = None
    
    forensics_agent_device_id: Optional[str] = None
    forensics_agent_public_id: Optional[str] = None
    forensics_agent_hmac_key: Optional[str] = None
    forensics_agent_secret: Optional[str] = None
    
    deception_agent_device_id: Optional[str] = None
    deception_agent_public_id: Optional[str] = None
    deception_agent_hmac_key: Optional[str] = None
    deception_agent_secret: Optional[str] = None
    
    hunter_agent_device_id: Optional[str] = None
    hunter_agent_public_id: Optional[str] = None
    hunter_agent_hmac_key: Optional[str] = None
    hunter_agent_secret: Optional[str] = None
    
    rollback_agent_device_id: Optional[str] = None
    rollback_agent_public_id: Optional[str] = None
    rollback_agent_hmac_key: Optional[str] = None
    rollback_agent_secret: Optional[str] = None

    model_config = {"env_file": ".env", "case_sensitive": False, "extra": "allow"}

    @property
    def expanded_ssh_key_path(self) -> str:
        """Expand the SSH key path to handle ~ notation"""
        return os.path.expanduser(self.honeypot_ssh_key)

    @property
    def cors_origins(self) -> List[str]:
        """Return list of allowed CORS origins from comma-separated env"""
        origins = [origin.strip() for origin in self.ui_origin.split(",") if origin.strip()]
        return origins or ["http://localhost:3000"]


settings = Settings()
