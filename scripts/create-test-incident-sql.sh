#!/bin/bash

# EMERGENCY VIDEO DEMO FIX
# Creates incident directly in database for immediate demo

ATTACKER_IP="45.$((RANDOM % 255)).$((RANDOM % 255)).$((RANDOM % 255))"

echo "═══════════════════════════════════════════════════════════════"
echo "🚨 Creating Test SSH Brute Force Incident (Direct DB)"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "Attacker IP: $ATTACKER_IP"
echo ""

sqlite3 $(cd "$(dirname "$0")/../.." .. pwd)/backend/xdr.db << SQL
-- Insert events
INSERT INTO events (src_ip, dst_port, eventid, message, raw, source_type, hostname, ts)
VALUES
  ('$ATTACKER_IP', 22, 'cowrie.login.failed', 'SSH login failed: root/admin', '{}', 'cowrie', 'tpot', datetime('now')),
  ('$ATTACKER_IP', 22, 'cowrie.login.failed', 'SSH login failed: root/password', '{}', 'cowrie', 'tpot', datetime('now')),
  ('$ATTACKER_IP', 22, 'cowrie.login.failed', 'SSH login failed: root/123456', '{}', 'cowrie', 'tpot', datetime('now')),
  ('$ATTACKER_IP', 22, 'cowrie.login.failed', 'SSH login failed: admin/admin', '{}', 'cowrie', 'tpot', datetime('now')),
  ('$ATTACKER_IP', 22, 'cowrie.login.failed', 'SSH login failed: admin/password', '{}', 'cowrie', 'tpot', datetime('now')),
  ('$ATTACKER_IP', 22, 'cowrie.login.failed', 'SSH login failed: ubuntu/ubuntu', '{}', 'cowrie', 'tpot', datetime('now')),
  ('$ATTACKER_IP', 22, 'cowrie.login.failed', 'SSH login failed: root/root', '{}', 'cowrie', 'tpot', datetime('now'));

-- Insert incident
INSERT INTO incidents (src_ip, severity, status, reason, created_at, updated_at)
VALUES (
  '$ATTACKER_IP',
  'high',
  'active',
  'SSH brute-force: 7 failed login attempts in 60s (T-Pot Cowrie)',
  datetime('now'),
  datetime('now')
);

SELECT 'Created incident ID: ' || last_insert_rowid();
SQL

echo ""
echo "✅ Test incident created!"
echo ""
echo "Refresh your dashboard: http://localhost:3000"
echo "You should now see the incident!"
echo ""
echo "═══════════════════════════════════════════════════════════════"
