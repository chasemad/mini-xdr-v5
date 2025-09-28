import hashlib
import json
import os
import time
import importlib
from pathlib import Path

import pytest
from fastapi import HTTPException
from starlette.requests import Request
from starlette.responses import Response

BACKEND_PATH = Path(__file__).resolve().parents[1] / "backend"
if str(BACKEND_PATH) not in os.sys.path:
    os.sys.path.insert(0, str(BACKEND_PATH))


@pytest.fixture()
def middleware_fixture(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "sqlite+aiosqlite:///./test.db")
    for module in ["app.config", "app.db", "app.security"]:
        importlib.invalidate_caches()
        importlib.reload(importlib.import_module(module))

    import app.security as security_module
    from app.security import AuthMiddleware

    secret = "test-device-secret"
    hmac_key = hashlib.sha256(secret.encode("utf-8")).hexdigest()

    class DummySession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def commit(self):
            return None

    security_module.AsyncSessionLocal = lambda: DummySession()

    class StubCredential:
        def __init__(self, secret_hash: str):
            self.secret_hash = secret_hash
            self.revoked_at = None
            self.expires_at = None

    class StubbedAuthMiddleware(AuthMiddleware):
        def __init__(self, app):
            super().__init__(app)
            self.credentials = {"device-test-1": StubCredential(hmac_key)}
            self.seen_nonces = set()

        async def _get_active_credential(self, session, device_id: str):
            credential = self.credentials.get(device_id)
            if credential is None:
                raise HTTPException(status_code=401, detail="Unknown device")
            return credential

        async def _enforce_nonce(self, session, device_id: str, nonce: str, endpoint: str):
            key = (device_id, nonce)
            if key in self.seen_nonces:
                raise HTTPException(status_code=401, detail="Nonce already used")
            self.seen_nonces.add(key)

    async def app(scope, receive, send):
        response = Response(status_code=200)
        await response(scope, receive, send)

    return StubbedAuthMiddleware(app), hmac_key


def _make_request_headers(secret: str, body: str, nonce: str):
    from app.security import build_canonical_message, compute_signature

    timestamp = str(int(time.time()))
    canonical = build_canonical_message("POST", "/ingest/multi", body, timestamp, nonce)
    signature = compute_signature(secret, canonical)
    return [
        (b"content-type", b"application/json"),
        (b"x-device-id", b"device-test-1"),
        (b"x-ts", timestamp.encode()),
        (b"x-nonce", nonce.encode()),
        (b"x-signature", signature.encode()),
    ]


def _build_request(body: str, headers):
    scope = {
        "type": "http",
        "http_version": "1.1",
        "method": "POST",
        "path": "/ingest/multi",
        "headers": headers,
        "scheme": "http",
        "server": ("testserver", 80),
        "client": ("testclient", 12345),
    }

    async def receive():
        return {"type": "http.request", "body": body.encode("utf-8"), "more_body": False}

    return Request(scope, receive)


@pytest.mark.asyncio
async def test_signed_request_succeeds(middleware_fixture):
    middleware, hmac_key = middleware_fixture
    payload = {
        "source_type": "cowrie",
        "hostname": "unit-test",
        "events": [
            {"eventid": "cowrie.login.failed", "src_ip": "192.0.2.10", "message": "test"}
        ],
    }
    body = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    headers = _make_request_headers(hmac_key, body, nonce="nonce-1")
    request = _build_request(body, headers)

    called = False

    async def call_next(req):
        nonlocal called
        called = True
        return Response(status_code=200)

    response = await middleware.dispatch(request, call_next)
    assert response.status_code == 200
    assert called is True


@pytest.mark.asyncio
async def test_unsigned_request_rejected(middleware_fixture):
    middleware, _ = middleware_fixture
    payload = {
        "source_type": "cowrie",
        "hostname": "unit-test",
        "events": [
            {"eventid": "cowrie.login.failed", "src_ip": "192.0.2.11", "message": "unsigned"}
        ],
    }
    body = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    request = _build_request(body, headers=[(b"content-type", b"application/json")])

    async def call_next(req):
        return Response(status_code=200)

    with pytest.raises(HTTPException) as exc_info:
        await middleware.dispatch(request, call_next)

    assert exc_info.value.status_code == 401
    assert "Missing authentication" in exc_info.value.detail
