import os
from pathlib import Path
from typing import List, Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # API Configuration
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    ui_origin: str = "http://localhost:3000"
    api_key: Optional[str] = None

    # Authentication
    JWT_SECRET_KEY: Optional[str] = None
    ENCRYPTION_KEY: Optional[str] = None

    # Database - PostgreSQL for production-like local development
    database_url: str = (
        "postgresql+asyncpg://xdr_user:local_dev_password@localhost:5432/mini_xdr"
    )

    # Detection Configuration
    fail_window_seconds: int = 60
    fail_threshold: int = 6
    auto_contain: bool = (
        True  # Enable auto-containment to trigger LangChain orchestration
    )

    # Containment Configuration
    allow_private_ip_blocking: bool = True  # Enable for testing simulated attacks

    # Honeypot Configuration - Local T-Pot
    honeypot_host: str = "localhost"
    honeypot_user: str = "admin"
    honeypot_ssh_key: str = "~/.ssh/id_rsa"
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
    tpot_api_key: Optional[str] = None  # SSH password for T-Pot
    tpot_host: Optional[str] = "24.11.0.176"
    tpot_ssh_port: Optional[int] = 64295
    tpot_web_port: Optional[int] = 64297
    tpot_elasticsearch_port: int = 64298
    tpot_kibana_port: int = 64296

    # Redis Cache Configuration (for feature caching)
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    feature_cache_ttl: int = 300  # 5 minutes - balance freshness vs performance
    feature_cache_enabled: bool = True

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

    model_config = {
        "env_file": (".env", ".env.local"),
        "case_sensitive": False,
        "extra": "allow",
    }

    @property
    def expanded_ssh_key_path(self) -> str:
        """Expand the SSH key path to handle ~ notation"""
        return os.path.expanduser(self.honeypot_ssh_key)

    @property
    def cors_origins(self) -> List[str]:
        """Return list of allowed CORS origins from comma-separated env"""
        origins = [
            origin.strip() for origin in self.ui_origin.split(",") if origin.strip()
        ]
        return origins or ["http://localhost:3000"]


settings = Settings()
