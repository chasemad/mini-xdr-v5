"""
Security-focused application entrypoint.

This module serves as the production entrypoint for the Mini-XDR system,
ensuring that security configurations are properly applied and that
the application runs with enhanced security controls.

The original purpose was to override any permissive CORS settings and
ensure HMAC authentication is properly configured.
"""
import logging
from fastapi.middleware.cors import CORSMiddleware

# Set security mode flag before importing main
from .config import settings
settings._entrypoint_mode = True

# Import the configured app from main
from .main import app
from .security import AuthMiddleware, RateLimiter

# Add secure CORS configuration for entrypoint mode
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,  # Controlled by UI_ORIGIN environment variable
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "x-api-key", "X-Device-ID", "X-TS", "X-Nonce", "X-Signature"],
)

# Add authentication and rate limiting middleware
rate_limiter = RateLimiter()
app.add_middleware(AuthMiddleware, rate_limiter=rate_limiter)

# Log security configuration
logger = logging.getLogger(__name__)
logger.info("🔒 Security Entrypoint: Enhanced security mode enabled")
logger.info(f"🔒 CORS Origins: {settings.cors_origins}")
logger.info("🔒 HMAC Authentication: Enabled")
logger.info("🔒 Rate Limiting: Enabled")

__all__ = ["app"]
