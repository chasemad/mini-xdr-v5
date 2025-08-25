#!/bin/bash
# Containment proxy that works around network connectivity issues
# This runs as a background service and executes containment commands

LOG_FILE="/tmp/xdr-containment.log"
FIFO="/tmp/xdr-commands"

# Create named pipe for communication
mkfifo "$FIFO" 2>/dev/null

echo "🚀 XDR Containment Proxy starting..." >> "$LOG_FILE"
echo "Listening for containment commands..." >> "$LOG_FILE"

while true; do
    if read -r command < "$FIFO"; then
        echo "[$(date)] Received command: $command" >> "$LOG_FILE"
        
        case "$command" in
            "block:"*)
                IP="${command#block:}"
                echo "Blocking IP: $IP" >> "$LOG_FILE"
                ssh -p 22022 -i ~/.ssh/xdrops_id_ed25519 xdrops@10.0.0.23 "sudo ufw deny from $IP to any" >> "$LOG_FILE" 2>&1
                echo "Block result: $?" >> "$LOG_FILE"
                ;;
            "unblock:"*)
                IP="${command#unblock:}"
                echo "Unblocking IP: $IP" >> "$LOG_FILE"
                ssh -p 22022 -i ~/.ssh/xdrops_id_ed25519 xdrops@10.0.0.23 "sudo ufw delete deny from $IP to any" >> "$LOG_FILE" 2>&1
                echo "Unblock result: $?" >> "$LOG_FILE"
                ;;
            "status")
                echo "Getting UFW status" >> "$LOG_FILE"
                ssh -p 22022 -i ~/.ssh/xdrops_id_ed25519 xdrops@10.0.0.23 "sudo ufw status numbered" >> "$LOG_FILE" 2>&1
                ;;
        esac
    fi
done
