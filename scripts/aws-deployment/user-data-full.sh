#!/bin/bash
# Full Mini-XDR deployment user data script
set -e

# Update system
apt-get update -y
apt-get upgrade -y

# Install essential packages
apt-get install -y \
    python3 python3-pip python3-venv python3-dev \
    nodejs npm git curl wget unzip \
    build-essential pkg-config \
    libffi-dev libssl-dev \
    python3-numpy python3-scipy python3-pandas python3-sklearn \
    awscli

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh
usermod -aG docker ubuntu

# Create application directory
mkdir -p /opt/mini-xdr
cd /opt/mini-xdr

# Clone/setup Mini-XDR (this will be done via SSH after instance is ready)
chown -R ubuntu:ubuntu /opt/mini-xdr

# Setup system limits for production
cat >> /etc/security/limits.conf << EOF
ubuntu soft nofile 65536
ubuntu hard nofile 65536
EOF

# Create log directory
mkdir -p /var/log/mini-xdr
chown -R ubuntu:ubuntu /var/log/mini-xdr

# Signal completion
echo "Mini-XDR instance setup complete at $(date)" > /opt/mini-xdr-setup-complete.log