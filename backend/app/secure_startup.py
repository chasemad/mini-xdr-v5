"""
Secure Startup Module for Mini-XDR
Integrates AWS Secrets Manager with the application startup process
"""

import os
import logging
from .secrets_manager import load_common_secrets

logger = logging.getLogger(__name__)

def initialize_secure_environment():
    """
    Initialize secure environment by loading secrets from AWS Secrets Manager
    Call this function early in your application startup (main.py)
    """
    
    # Check if Secrets Manager is enabled
    secrets_enabled = os.getenv('SECRETS_MANAGER_ENABLED', 'false').lower() == 'true'
    
    if not secrets_enabled:
        logger.info("AWS Secrets Manager disabled, using environment variables")
        return
    
    logger.info("Loading secure credentials from AWS Secrets Manager...")
    
    try:
        # Load all common secrets
        secrets = load_common_secrets()
        
        # Update environment with retrieved secrets
        secrets_loaded = 0
        for key, value in secrets.items():
            if value:
                os.environ[key] = value
                secrets_loaded += 1
                logger.debug(f"Loaded {key} from Secrets Manager")
        
        if secrets_loaded > 0:
            logger.info(f"✅ Successfully loaded {secrets_loaded} secrets from AWS Secrets Manager")
            logger.info("🔒 All API keys are now securely available to the application")

            # Reinitialize threat intelligence providers with loaded secrets
            try:
                from .external_intel import threat_intel
                threat_intel.reinitialize_providers()
                logger.info("🔍 Threat intelligence providers reinitialized with secure API keys")
            except Exception as e:
                logger.error(f"Failed to reinitialize threat intelligence providers: {e}")
        else:
            logger.warning("No secrets were loaded from AWS Secrets Manager")
            
    except Exception as e:
        logger.error(f"Failed to load secrets from AWS Secrets Manager: {e}")
        logger.info("Continuing with environment variables as fallback")

def get_startup_info():
    """Get information about the secure startup configuration"""
    
    secrets_enabled = os.getenv('SECRETS_MANAGER_ENABLED', 'false').lower() == 'true'
    
    info = {
        'secrets_manager_enabled': secrets_enabled,
        'aws_region': os.getenv('AWS_REGION', 'us-east-1'),
        'credentials_secured': secrets_enabled
    }
    
    if secrets_enabled:
        # Check which secret names are configured
        secret_names = []
        for env_var in ['API_KEY', 'OPENAI_API_KEY', 'XAI_API_KEY', 'ABUSEIPDB_API_KEY', 'VIRUSTOTAL_API_KEY']:
            secret_name_var = f"{env_var}_SECRET_NAME"
            if os.getenv(secret_name_var):
                secret_names.append(os.getenv(secret_name_var))
        
        info['configured_secrets'] = secret_names
        info['security_level'] = 'enterprise' if secret_names else 'basic'
    else:
        info['security_level'] = 'development'
    
    return info