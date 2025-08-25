#!/bin/bash
# Install and configure Fluent Bit on honeypot VM
# Run this ON the honeypot VM after Cowrie is installed

echo "=== Installing Fluent Bit for XDR Log Forwarding ==="

# 1. Install Fluent Bit (Ubuntu/Debian)
curl https://raw.githubusercontent.com/fluent/fluent-bit/master/install.sh | sh

# 2. Create Fluent Bit configuration directory
sudo mkdir -p /etc/fluent-bit

# 3. Copy the configuration file
echo "Creating Fluent Bit configuration..."
sudo tee /etc/fluent-bit/fluent-bit.conf << 'EOF'
[SERVICE]
    Flush        1
    Log_Level    info
    Daemon       off
    HTTP_Server  On
    HTTP_Listen  0.0.0.0
    HTTP_Port    2020

[INPUT]
    Name              tail
    Path              /home/cowrie/cowrie/var/log/cowrie/cowrie.json*
    Parser            json
    Tag               cowrie
    Refresh_Interval  1
    Read_from_Head    false

[OUTPUT]
    Name  http
    Match cowrie
    Host  10.0.0.123
    Port  8000
    URI   /ingest/cowrie
    Format json
    Retry_Limit 5
EOF

# 4. Create systemd service
sudo tee /etc/systemd/system/fluent-bit-xdr.service << 'EOF'
[Unit]
Description=Fluent Bit XDR Log Forwarder
Documentation=https://fluentbit.io/
Requires=network.target
After=network.target

[Service]
Type=simple
ExecStart=/opt/fluent-bit/bin/fluent-bit -c /etc/fluent-bit/fluent-bit.conf
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF

# 5. Enable and start the service
sudo systemctl daemon-reload
sudo systemctl enable fluent-bit-xdr
sudo systemctl start fluent-bit-xdr

# 6. Check status
sudo systemctl status fluent-bit-xdr

echo "=== Fluent Bit installation complete! ==="
echo "Logs are being forwarded to XDR system at 10.0.0.123:8000/ingest/cowrie"
echo "Check status with: sudo systemctl status fluent-bit-xdr"
echo "View logs with: sudo journalctl -u fluent-bit-xdr -f"
