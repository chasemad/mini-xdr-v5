#!/usr/bin/env bash
set -euo pipefail

ALB_URL=${ALB_URL:-${1:-}}
ADMIN_EMAIL=${ADMIN_EMAIL:-"chasemadrian@protonmail.com"}
ADMIN_PASSWORD=${ADMIN_PASSWORD:-"demo-tpot-api-key"}
ORG_NAME=${ORG_NAME:-"Mini Corp"}

if [[ -z "$ALB_URL" ]]; then
  echo "Usage: ALB_URL=http://<alb-dns> $0" >&2
  exit 1
fi

echo "[*] Checking login for $ADMIN_EMAIL at $ALB_URL"
LOGIN_STATUS=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$ALB_URL/api/auth/login" \
  -H 'Content-Type: application/json' \
  -d "{\"email\":\"$ADMIN_EMAIL\",\"password\":\"$ADMIN_PASSWORD\"}") || true

if [[ "$LOGIN_STATUS" == "200" ]]; then
  echo "[✓] Account already exists. No action taken."
  exit 0
fi

echo "[*] Registering organization and admin user"
curl -s -X POST "$ALB_URL/api/auth/register" \
  -H 'Content-Type: application/json' \
  -d "{\"organization_name\":\"$ORG_NAME\",\"admin_email\":\"$ADMIN_EMAIL\",\"admin_password\":\"$ADMIN_PASSWORD\",\"admin_name\":\"Chase\"}" | jq .

echo "[✓] Seed complete"


