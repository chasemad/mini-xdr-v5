#!/usr/bin/env python3
"""Send an HTTP request with Mini-XDR HMAC authentication."""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.join(SCRIPT_DIR, "..", "backend")
for entry in (SCRIPT_DIR, BACKEND_DIR):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import requests  # noqa: E402
from agent_auth import load_agent_credentials, build_signed_headers  # noqa: E402


def parse_body(body: str | None, body_file: str | None) -> Any:
    if body_file:
        try:
            with open(body_file, "r", encoding="utf-8") as handle:
                body = handle.read()
        except OSError as exc:
            raise SystemExit(f"Failed to read body file: {exc}")
    if body is None or body == "":
        return ""
    data = body.strip()
    try:
        return json.loads(data)
    except json.JSONDecodeError:
        return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Send a signed Mini-XDR request")
    parser.add_argument("--base-url", default=os.getenv("MINIXDR_API_BASE_URL", "http://localhost:8000"))
    parser.add_argument("--path", required=True, help="Request path, e.g. /ingest/multi")
    parser.add_argument("--method", default="POST", help="HTTP method (default: POST)")
    parser.add_argument("--body", help="Raw JSON string to send")
    parser.add_argument("--body-file", help="Path to file containing JSON payload")
    parser.add_argument("--timeout", type=float, default=15.0)

    args = parser.parse_args()

    body = parse_body(args.body, args.body_file)

    credentials = load_agent_credentials()
    headers, body_text = build_signed_headers(credentials, args.method, args.path, body)
    headers.setdefault("Content-Type", "application/json")

    data = body_text if body_text != "" else None

    response = requests.request(
        args.method.upper(),
        f"{args.base_url}{args.path}",
        headers=headers,
        data=data,
        timeout=args.timeout,
    )

    print(response.text)
    if response.status_code >= 400:
        sys.exit(response.status_code)


if __name__ == "__main__":
    main()
