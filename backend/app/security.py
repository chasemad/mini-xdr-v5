"""Security middleware and utilities for HMAC authentication and rate limiting."""
import asyncio
import gzip
import hashlib
import hmac
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from typing import Deque, Dict, Tuple

from fastapi import HTTPException
from sqlalchemy import delete, select
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from .config import settings
from .db import AsyncSessionLocal
from .models import AgentCredential, RequestNonce

MAX_CLOCK_SKEW_SECONDS = 300  # +/- 5 minutes
NONCE_TTL_SECONDS = 600
SECURED_PREFIXES = ("/api",)  # Removed /ingest for demo - no auth required

# Paths that bypass HMAC authentication (use simple API key instead or JWT)
SIMPLE_AUTH_PREFIXES = [
    "/api/auth",  # Authentication endpoints use JWT
    "/api/onboarding",  # Onboarding wizard endpoints use JWT
    "/api/response",  # All response system endpoints use simple API key
    "/api/intelligence",  # Visualization endpoints
    "/api/incidents",  # Incident endpoints including AI analysis
    "/api/ml",  # ML endpoints
    "/api/workflows",  # Workflow and NLP endpoints
    "/api/nlp-suggestions",  # NLP workflow suggestions
    "/api/triggers",  # Workflow trigger management endpoints
    "/api/agents",  # Agent orchestration and chat endpoints
    "/api/telemetry",  # Telemetry status endpoint uses JWT
    "/api/tpot",  # T-Pot honeypot monitoring endpoints
    "/api/health",  # System health check endpoints
    "/api/ingest",  # Event ingestion endpoints (for demo/testing)
    "/ingest/multi",  # Multi-source ingestion (for testing - use HMAC in production)
]

ALLOWED_PATHS = {
    "/health",
    "/api/health",  # Health check endpoint (Kubernetes probes)
    "/docs",
    "/openapi.json",
    "/api/auth/config",
}


def build_canonical_message(
    method: str, path: str, body: str, timestamp: str, nonce: str
) -> str:
    return "|".join([method.upper(), path, body, timestamp, nonce])


def compute_signature(secret_hash: str, canonical_message: str) -> str:
    return hmac.new(
        secret_hash.encode("utf-8"), canonical_message.encode("utf-8"), hashlib.sha256
    ).hexdigest()


def require_api_key(request: Request) -> bool:
    """Simple API key authentication for trigger endpoints"""
    if not settings.api_key:
        # If no API key is configured, allow access (for development)
        return True

    api_key = request.headers.get("x-api-key")
    if not api_key:
        raise HTTPException(status_code=401, detail="Missing API key header")

    if not hmac.compare_digest(api_key, settings.api_key):
        raise HTTPException(status_code=401, detail="Invalid API key")

    return True


def is_timestamp_valid(
    timestamp: int, now: datetime | None = None, max_skew: int = MAX_CLOCK_SKEW_SECONDS
) -> bool:
    if now is None:
        now = datetime.now(timezone.utc)
    return abs(int(now.timestamp()) - int(timestamp)) <= max_skew


class RateLimiter:
    """Simple in-memory sliding window rate limiter per device/endpoint."""

    def __init__(
        self,
        burst_limit: int = 10,
        burst_window: int = 60,
        sustained_limit: int = 100,
        sustained_window: int = 3600,
    ) -> None:
        self.burst_limit = burst_limit
        self.burst_window = burst_window
        self.sustained_limit = sustained_limit
        self.sustained_window = sustained_window
        self._events: Dict[Tuple[str, str], Deque[float]] = defaultdict(deque)
        self._lock = asyncio.Lock()

    async def check(self, device_id: str, endpoint: str) -> None:
        key = (device_id, endpoint)
        now = time.monotonic()

        async with self._lock:
            events = self._events[key]
            # Remove events outside sustained window
            while events and now - events[0] > self.sustained_window:
                events.popleft()

            # Count events within windows
            burst_count = sum(1 for ts in events if now - ts <= self.burst_window)
            sustained_count = len(events)

            if (
                burst_count >= self.burst_limit
                or sustained_count >= self.sustained_limit
            ):
                raise HTTPException(status_code=429, detail="Rate limit exceeded")

            events.append(now)


class AuthMiddleware(BaseHTTPMiddleware):
    """Middleware enforcing HMAC authentication with replay protection."""

    def __init__(self, app, rate_limiter: RateLimiter | None = None):
        super().__init__(app)
        self.rate_limiter = rate_limiter or RateLimiter()

    async def dispatch(self, request: Request, call_next):
        if not self._requires_auth(request):
            return await call_next(request)

        if request.method.upper() == "OPTIONS":
            return await call_next(request)

        body_bytes = await request.body()
        encoding = request.headers.get("content-encoding", "").lower()
        if encoding == "gzip" and body_bytes:
            try:
                body_bytes = gzip.decompress(body_bytes)
            except OSError as exc:
                raise HTTPException(
                    status_code=400, detail="Invalid gzip payload"
                ) from exc
        body_text = body_bytes.decode("utf-8") if body_bytes else ""

        device_id = request.headers.get("X-Device-ID")
        timestamp_header = request.headers.get("X-TS")
        nonce = request.headers.get("X-Nonce")
        signature = request.headers.get("X-Signature")

        if not all([device_id, timestamp_header, nonce, signature]):
            raise HTTPException(
                status_code=401, detail="Missing authentication headers"
            )

        try:
            timestamp = int(timestamp_header)
        except ValueError:
            raise HTTPException(
                status_code=400, detail="Invalid timestamp format"
            ) from None

        if not is_timestamp_valid(timestamp):
            raise HTTPException(
                status_code=401, detail="Timestamp outside allowed window"
            )

        canonical = build_canonical_message(
            request.method,
            request.url.path,
            body_text,
            timestamp_header,
            nonce,
        )

        async with AsyncSessionLocal() as session:
            credential = await self._get_active_credential(session, device_id)

            expected_signature = compute_signature(credential.secret_hash, canonical)

            if not hmac.compare_digest(expected_signature, signature):
                raise HTTPException(status_code=401, detail="Invalid signature")

            await self._enforce_nonce(session, device_id, nonce, request.url.path)
            await session.commit()

        await self.rate_limiter.check(device_id, request.url.path)

        return await call_next(request)

    def _requires_auth(self, request: Request) -> bool:
        path = request.url.path

        # Check for paths that don't require any auth
        if path in ALLOWED_PATHS or path.startswith("/static"):
            return False

        # Check for paths that use simple API key auth (bypass HMAC)
        if any(path.startswith(prefix) for prefix in SIMPLE_AUTH_PREFIXES):
            return False

        # Check if path requires HMAC authentication
        return any(path.startswith(prefix) for prefix in SECURED_PREFIXES)

    async def _get_active_credential(self, session, device_id: str) -> AgentCredential:
        result = await session.execute(
            select(AgentCredential).where(AgentCredential.device_id == device_id)
        )
        credential = result.scalars().first()
        if not credential:
            raise HTTPException(status_code=401, detail="Unknown device")

        now = datetime.now(timezone.utc)

        # Handle timezone-aware/naive comparison
        if credential.revoked_at:
            revoked_at = (
                credential.revoked_at.replace(tzinfo=timezone.utc)
                if credential.revoked_at.tzinfo is None
                else credential.revoked_at
            )
            if revoked_at <= now:
                raise HTTPException(status_code=401, detail="Device revoked")

        if credential.expires_at:
            expires_at = (
                credential.expires_at.replace(tzinfo=timezone.utc)
                if credential.expires_at.tzinfo is None
                else credential.expires_at
            )
            if expires_at <= now:
                raise HTTPException(status_code=401, detail="Device credential expired")
        return credential

    async def _enforce_nonce(
        self, session, device_id: str, nonce: str, endpoint: str
    ) -> None:
        # Check reuse
        existing = await session.execute(
            select(RequestNonce)
            .where(RequestNonce.device_id == device_id)
            .where(RequestNonce.nonce == nonce)
        )
        if existing.scalars().first() is not None:
            raise HTTPException(status_code=401, detail="Nonce already used")

        # Prune old nonces
        expiry_cutoff = datetime.now(timezone.utc) - timedelta(
            seconds=NONCE_TTL_SECONDS
        )
        await session.execute(
            delete(RequestNonce).where(RequestNonce.created_at < expiry_cutoff)
        )

        session.add(
            RequestNonce(
                device_id=device_id,
                nonce=nonce,
                endpoint=endpoint,
                created_at=datetime.now(timezone.utc),
            )
        )
