"""Utilities for loading agent credentials and building signed request headers."""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

BACKEND_PATH = Path(__file__).resolve().parents[1] / "backend"
if str(BACKEND_PATH) not in sys.path:
    sys.path.insert(0, str(BACKEND_PATH))

from app.agents.hmac_signer import canonicalize_payload, build_hmac_headers  # noqa: E402

_ENV_PATH = BACKEND_PATH / ".env"


@dataclass
class AgentCredentials:
    profile: str
    device_id: str
    hmac_key: str
    api_key: Optional[str] = None


def _read_env_file(key: str) -> Optional[str]:
    if not _ENV_PATH.exists():
        return None
    try:
        with _ENV_PATH.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                if k.strip() == key:
                    return v.strip().strip('"').strip("'")
    except OSError:
        return None
    return None


def _resolve_value(primary: str, *fallbacks: str, required: bool = True) -> Optional[str]:
    for key in (primary, *fallbacks):
        if not key:
            continue
        value = os.getenv(key)
        if value:
            return value
        file_value = _read_env_file(key)
        if file_value:
            return file_value
    if required:
        raise RuntimeError(f"Missing required credential value for {primary}")
    return None


def load_agent_credentials(profile: Optional[str] = None) -> AgentCredentials:
    profile = (profile or os.getenv("MINIXDR_AGENT_PROFILE", "HUNTER")).upper()
    bearer_env = os.getenv("MINIXDR_AGENT_BEARER_ENV", "API_KEY")

    device_id = _resolve_value(
        "MINIXDR_AGENT_DEVICE_ID",
        f"{profile}_AGENT_DEVICE_ID",
    )
    hmac_key = _resolve_value(
        "MINIXDR_AGENT_HMAC_KEY",
        f"{profile}_AGENT_HMAC_KEY",
    )
    api_key = _resolve_value(
        "MINIXDR_AGENT_BEARER",
        bearer_env,
        required=False,
    )

    return AgentCredentials(
        profile=profile,
        device_id=device_id,
        hmac_key=hmac_key,
        api_key=api_key,
    )


def build_signed_headers(
    credentials: AgentCredentials,
    method: str,
    path: str,
    payload: Any,
) -> Tuple[Dict[str, str], str]:
    body_text = canonicalize_payload(payload)
    headers, _, _ = build_hmac_headers(
        credentials.device_id,
        credentials.hmac_key,
        method,
        path,
        body_text,
    )
    if credentials.api_key:
        headers["Authorization"] = f"Bearer {credentials.api_key}"
    headers["Content-Type"] = "application/json"
    return headers, body_text
