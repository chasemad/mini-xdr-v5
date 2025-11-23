#!/usr/bin/env bash

# Inject synthetic events with a chosen source IP directly into the backend ingest endpoint.
# This does NOT send real network traffic; it just seeds events so Mini-XDR will create an incident
# and can block that IP on T-Pot.
#
# Usage:
#   chmod +x scripts/fake_attack_ingest.sh
#   FAKE_IP=203.0.113.250 ./scripts/fake_attack_ingest.sh
#
# Optional env vars:
#   FAKE_IP (default 203.0.113.250)
#   BACKEND (default http://localhost:8000)
#   COUNT (number of events, default 6)

set -eo pipefail

FAKE_IP=${FAKE_IP:-203.0.113.250}
BACKEND=${BACKEND:-http://localhost:8000}
COUNT=${COUNT:-6}

echo "Injecting ${COUNT} fake events for IP ${FAKE_IP} into ${BACKEND}/ingest/cowrie"

now_iso() {
  date -u +"%Y-%m-%dT%H:%M:%S.000Z"
}

for i in $(seq 1 "$COUNT"); do
  payload=$(cat <<EOF
{
  "src_ip": "${FAKE_IP}",
  "dst_port": 22,
  "eventid": "cowrie.login.failed",
  "username": "admin",
  "password": "wrongpass${i}",
  "session": "fake-${FAKE_IP//./-}-${i}",
  "timestamp": "$(now_iso)"
}
EOF
)

  echo "[${i}] POST /ingest/cowrie for ${FAKE_IP}"
  curl -s -o /dev/null -w "  -> %{http_code}\n" \
    -H "Content-Type: application/json" \
    -X POST "${BACKEND}/ingest/cowrie" \
    -d "$payload" || echo "  -> request failed"
done

echo "Done. Check incidents: curl -s ${BACKEND}/api/incidents | jq '.[0] | {src_ip, reason, ml_confidence, auto_contained}'"
