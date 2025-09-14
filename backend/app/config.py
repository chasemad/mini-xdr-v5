from pydantic_settings import BaseSettings
from typing import Optional
import os
from pathlib import Path


class Settings(BaseSettings):
    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_key: Optional[str] = None

    # Database
    database_url: str = "sqlite+aiosqlite:///./xdr.db"

    # Detection Configuration
    fail_window_seconds: int = 60
    fail_threshold: int = 6
    auto_contain: bool = False
    
    # Containment Configuration
    allow_private_ip_blocking: bool = True  # Enable for testing simulated attacks

    # Honeypot Configuration
    honeypot_host: str = "10.0.0.23"
    honeypot_user: str = "xdrops"
    honeypot_ssh_key: str = "~/.ssh/xdrops_id_ed25519"
    honeypot_ssh_port: int = 22022

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

    class Config:
        env_file = ".env"
        case_sensitive = False

    @property
    def expanded_ssh_key_path(self) -> str:
        """Expand the SSH key path to handle ~ notation"""
        return os.path.expanduser(self.honeypot_ssh_key)


settings = Settings()
