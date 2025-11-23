#!/bin/bash
# Quick system verification
cd "$(dirname "$0")/backend"
echo "=== Quick System Check ==="
echo "Incidents: $(sqlite3 xdr.db 'SELECT COUNT(*) FROM incidents;')"
echo "Events: $(sqlite3 xdr.db 'SELECT COUNT(*) FROM events;')"
echo "Actions: $(sqlite3 xdr.db 'SELECT COUNT(*) FROM actions;')"
echo "DB Size: $(du -h xdr.db | cut -f1)"
echo ""
if [ $(sqlite3 xdr.db 'SELECT COUNT(*) FROM incidents;') -eq 0 ]; then
  echo "✓ System is CLEAN - Ready for T-Pot attacks"
else
  echo "ℹ System contains $(sqlite3 xdr.db 'SELECT COUNT(*) FROM incidents;') incident(s)"
  echo "Latest incidents:"
  sqlite3 xdr.db "SELECT id, src_ip, reason, created_at FROM incidents ORDER BY created_at DESC LIMIT 3;"
fi
