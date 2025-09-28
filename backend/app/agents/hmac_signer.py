"""Utilities for generating HMAC-authenticated request headers for agents."""
from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import time
from typing import Any, Dict, Tuple


def canonicalize_payload(payload: Any) -> str:
    """Return a deterministic JSON string for signing."""
    if isinstance(payload, bytes):
        return payload.decode("utf-8")
    if isinstance(payload, str):
        return payload
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def build_canonical_message(method: str, path: str, body: str, timestamp: str, nonce: str) -> str:
    return "|".join([method.upper(), path, body, timestamp, nonce])


def compute_signature(secret: str, canonical_message: str) -> str:
    return hmac.new(secret.encode("utf-8"), canonical_message.encode("utf-8"), hashlib.sha256).hexdigest()


def build_hmac_headers(
    device_id: str,
    secret: str,
    method: str,
    path: str,
    body: str,
    *,
    timestamp: str | None = None,
    nonce: str | None = None,
) -> Tuple[Dict[str, str], str, str]:
    """Return headers plus the timestamp and nonce used for signing."""
    timestamp = timestamp or str(int(time.time()))
    nonce = nonce or secrets.token_hex(16)
    canonical = build_canonical_message(method, path, body, timestamp, nonce)
    signature = compute_signature(secret, canonical)
    headers = {
        "X-Device-ID": device_id,
        "X-TS": timestamp,
        "X-Nonce": nonce,
        "X-Signature": signature,
    }
    return headers, timestamp, nonce


def sign_event(secret: str, event: Dict[str, Any]) -> str:
    """Create deterministic event signature for tamper checks."""
    event_json = json.dumps(event, sort_keys=True, separators=(",", ":"))
    return hmac.new(secret.encode("utf-8"), event_json.encode("utf-8"), hashlib.sha256).hexdigest()
