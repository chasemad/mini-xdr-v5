#!/usr/bin/env bash
# Mini-XDR Homelab Lockdown Script
# This script prints the commands required to restrict inbound traffic to the
# loopback interface. It does not execute any firewall changes unless you
# pass the --apply flag explicitly.

set -euo pipefail

usage() {
  cat <<USAGE
Usage: $0 [--apply]

Without --apply the script performs a dry run and displays the recommended
firewall rules for macOS (pf), Ubuntu/Debian (ufw), and generic iptables.

Examples:
  $0            # dry run, shows commands
  $0 --apply    # executes commands for the detected platform

USAGE
}

APPLY=false
if [[ $# -gt 0 ]]; then
  case "$1" in
    --apply) APPLY=true ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
fi

print_heading() {
  echo "========================================="
  echo "🔐 Homelab Network Lockdown Instructions"
  echo "========================================="
  echo
}

apply_macos_pf() {
  local rule_file="/etc/pf.anchors/mini-xdr-loopback"
  sudo sh -c "cat > $rule_file" <<'RULES'
block in all
pass quick on lo0 all keep state
RULES
  sudo sh -c 'echo "anchor \"mini-xdr-loopback\"" > /etc/pf-minixdr.conf'
  sudo sh -c 'echo "load anchor \"mini-xdr-loopback\" from \"/etc/pf.anchors/mini-xdr-loopback\"" >> /etc/pf-minixdr.conf'
  sudo pfctl -f /etc/pf-minixdr.conf
  sudo pfctl -e
}

show_macos_pf() {
  cat <<'CMDS'
# macOS pf (dry run)
sudo sh -c 'cat <<"RULES" > /etc/pf.anchors/mini-xdr-loopback
block in all
pass quick on lo0 all keep state
RULES'
sudo sh -c 'echo "anchor \"mini-xdr-loopback\"" > /etc/pf-minixdr.conf'
sudo sh -c 'echo "load anchor \"mini-xdr-loopback\" from \"/etc/pf.anchors/mini-xdr-loopback\"" >> /etc/pf-minixdr.conf'
sudo pfctl -f /etc/pf-minixdr.conf
sudo pfctl -e
CMDS
}

apply_ufw() {
  sudo ufw --force reset
  sudo ufw default deny incoming
  sudo ufw default allow outgoing
  sudo ufw allow in on lo
  sudo ufw enable
}

show_ufw() {
  cat <<'CMDS'
# Ubuntu/Debian with UFW (dry run)
sudo ufw --force reset
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow in on lo
sudo ufw enable
CMDS
}

apply_iptables() {
  sudo iptables -F
  sudo iptables -P INPUT DROP
  sudo iptables -P FORWARD DROP
  sudo iptables -P OUTPUT ACCEPT
  sudo iptables -A INPUT -i lo -j ACCEPT
}

show_iptables() {
  cat <<'CMDS'
# Generic iptables (dry run)
sudo iptables -F
sudo iptables -P INPUT DROP
sudo iptables -P FORWARD DROP
sudo iptables -P OUTPUT ACCEPT
sudo iptables -A INPUT -i lo -j ACCEPT
CMDS
}

print_heading

if $APPLY; then
  echo "Applying firewall rules for loopback-only access..."
  if [[ "$OSTYPE" == "darwin"* ]]; then
    apply_macos_pf
  elif command -v ufw >/dev/null 2>&1; then
    apply_ufw
  else
    apply_iptables
  fi
  echo
  echo "✅ Network restricted to 127.0.0.1. Verify with:"
  echo "   sudo lsof -iTCP -sTCP:LISTEN"
else
  echo "Dry run mode. Add --apply to execute." && echo
  if [[ "$OSTYPE" == "darwin"* ]]; then
    show_macos_pf
  elif command -v ufw >/dev/null 2>&1; then
    show_ufw
  else
    show_iptables
  fi
  echo
  echo "⚠️  Review these commands carefully before running with --apply."
fi
