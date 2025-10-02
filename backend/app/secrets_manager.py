"""
AWS Secrets Manager Integration for Mini-XDR
Securely loads API keys and secrets from AWS Secrets Manager
"""

import os
import boto3
import logging
from typing import Optional, Dict
from functools import lru_cache

logger = logging.getLogger(__name__)

class SecretsManager:
    """Secure secrets retrieval from AWS Secrets Manager"""
    
    def __init__(self, region: str = None):
        self.region = region or os.getenv('AWS_REGION', 'us-east-1')
        self.secrets_client = None
        self.enabled = os.getenv('SECRETS_MANAGER_ENABLED', 'false').lower() == 'true'
        
        if self.enabled:
            try:
                self.secrets_client = boto3.client('secretsmanager', region_name=self.region)
                logger.info("Secrets Manager client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Secrets Manager client: {e}")
                self.enabled = False
    
    @lru_cache(maxsize=128)
    def get_secret(self, secret_name: str) -> Optional[str]:
        """Retrieve secret value from AWS Secrets Manager with caching"""
        
        if not self.enabled or not self.secrets_client:
            logger.debug(f"Secrets Manager disabled, skipping secret: {secret_name}")
            return None
            
        try:
            response = self.secrets_client.get_secret_value(SecretId=secret_name)
            secret_value = response['SecretString']
            
            logger.debug(f"Successfully retrieved secret: {secret_name}")
            return secret_value
            
        except self.secrets_client.exceptions.ResourceNotFoundException:
            logger.warning(f"Secret not found: {secret_name}")
            return None
        except Exception as e:
            logger.error(f"Failed to retrieve secret '{secret_name}': {e}")
            return None
    
    def get_env_or_secret(self, env_var: str, secret_name: str = None, default: str = None) -> str:
        """Get value from environment variable or Secrets Manager fallback"""
        
        # First, try direct environment variable
        value = os.getenv(env_var)
        if value and not value.endswith('_SECRET_NAME') and value != "changeme" and not value.startswith("YOUR_"):
            return value
        
        # Try secret name from environment variable ending with _SECRET_NAME
        secret_name_env = f"{env_var}_SECRET_NAME"
        secret_name_from_env = os.getenv(secret_name_env)
        if secret_name_from_env:
            secret_value = self.get_secret(secret_name_from_env)
            if secret_value:
                return secret_value
        
        # Try provided secret name
        if secret_name:
            secret_value = self.get_secret(secret_name)
            if secret_value:
                return secret_value
        
        # Return default if nothing found
        if default:
            logger.warning(f"Using default value for {env_var}")
            return default
        
        logger.warning(f"Could not find value for {env_var} in environment or Secrets Manager")
        return ""

# Global instance
secrets_manager = SecretsManager()

def get_secure_env(env_var: str, secret_name: str = None, default: str = None) -> str:
    """Convenience function to get secure environment variable"""
    return secrets_manager.get_env_or_secret(env_var, secret_name, default)

def load_common_secrets() -> Dict[str, str]:
    """Pre-load commonly used secrets for better performance"""
    import os
    
    secrets = {}
    
    # Define secret mappings based on our configuration
    secret_mappings = {
        'API_KEY': 'mini-xdr/api-key',
        'OPENAI_API_KEY': 'mini-xdr/openai-api-key',
        'XAI_API_KEY': 'mini-xdr/xai-api-key',
        'ABUSEIPDB_API_KEY': 'mini-xdr/abuseipdb-api-key',
        'VIRUSTOTAL_API_KEY': 'mini-xdr/virustotal-api-key',
        'DATABASE_PASSWORD': 'mini-xdr/database-password'
    }
    
    for env_var, secret_name in secret_mappings.items():
        value = get_secure_env(env_var, secret_name)
        if value:
            secrets[env_var] = value
            # CRITICAL FIX: Actually set the environment variable
            os.environ[env_var] = value
            logger.debug(f"Loaded secret for {env_var} and set in environment")
    
    return secrets

# Test function to validate secrets integration
def test_secrets_integration():
    """Test function to validate secrets are working"""
    
    print("🔐 Testing AWS Secrets Manager Integration")
    print("=" * 50)
    
    # Test basic connection
    if not secrets_manager.enabled:
        print("❌ Secrets Manager is disabled")
        return False
    
    print("✅ Secrets Manager is enabled")
    
    # Test secret retrieval
    test_secret = secrets_manager.get_secret('mini-xdr/api-key')
    if test_secret:
        print(f"✅ API Key retrieved: {test_secret[:10]}...")
    else:
        print("❌ Failed to retrieve API key")
        return False
    
    # Test common secrets loading
    secrets = load_common_secrets()
    print(f"✅ Loaded {len(secrets)} secrets successfully")
    
    for key in secrets:
        print(f"   - {key}: {'✅ Loaded' if secrets[key] else '❌ Missing'}")
    
    return True

if __name__ == "__main__":
    test_secrets_integration()