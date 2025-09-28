"""
Secure Configuration Loader for Mini-XDR
Drop-in replacement for os.getenv that uses AWS Secrets Manager
"""

import os
from .secrets_manager import get_secure_env

class SecureConfig:
    """Drop-in replacement for os.getenv with Secrets Manager integration"""
    
    def __init__(self):
        # Map environment variables to secret names
        self.secret_mappings = {
            'API_KEY': 'mini-xdr/api-key',
            'OPENAI_API_KEY': 'mini-xdr/openai-api-key', 
            'XAI_API_KEY': 'mini-xdr/xai-api-key',
            'ABUSEIPDB_API_KEY': 'mini-xdr/abuseipdb-api-key',
            'VIRUSTOTAL_API_KEY': 'mini-xdr/virustotal-api-key',
            'DATABASE_PASSWORD': 'mini-xdr/database-password'
        }
    
    def getenv(self, key: str, default: str = None) -> str:
        """Secure getenv replacement - tries Secrets Manager for sensitive keys"""
        
        if key in self.secret_mappings:
            # Use Secrets Manager for sensitive keys
            return get_secure_env(key, self.secret_mappings[key], default)
        else:
            # Use regular environment variable for non-sensitive config
            return os.getenv(key, default)

# Create global instance for easy importing
secure_config = SecureConfig()

# Convenience function that can replace os.getenv
def getenv(key: str, default: str = None) -> str:
    """Secure version of os.getenv"""
    return secure_config.getenv(key, default)

# For easy integration - you can now replace:
# os.getenv("API_KEY") 
# with:
# from .secure_config_loader import getenv
# getenv("API_KEY")