#!/bin/bash
# Advanced TPOT Installation Script

exec > >(tee /var/log/user-data.log) 2>&1

echo "Starting TPOT installation..."

# Update system
apt-get update
apt-get upgrade -y

# Install prerequisites
apt-get install -y git curl docker.io docker-compose-plugin

# Enable Docker
systemctl enable docker
systemctl start docker

# Create admin user with proper SSH setup
useradd -m -s /bin/bash admin
usermod -aG sudo admin
usermod -aG docker admin

# Set up SSH directory for admin
mkdir -p /home/admin/.ssh
chmod 700 /home/admin/.ssh

# Copy ubuntu user's authorized_keys to admin user
if [ -f /home/ubuntu/.ssh/authorized_keys ]; then
    cp /home/ubuntu/.ssh/authorized_keys /home/admin/.ssh/authorized_keys
    chown admin:admin /home/admin/.ssh/authorized_keys
    chmod 600 /home/admin/.ssh/authorized_keys
fi

# Clone TPOT
cd /home/admin
sudo -u admin git clone https://github.com/telekom-security/tpotce.git
chown -R admin:admin tpotce

# Create unattended install configuration
cat > /home/admin/tpot-install-config.txt << 'EOF'
HIVE
tpotadmin
SecurePassword123!
SecurePassword123!
y
EOF

# Run TPOT installation unattended
cd /home/admin/tpotce
sudo -u admin bash -c 'cat /home/admin/tpot-install-config.txt | ./install.sh --type=user --conf=HIVE'

# Set up logging permissions
chown -R admin:admin /home/admin/tpotce

echo "TPOT installation completed" > /tmp/tpot-install-complete

# Reboot to start TPOT services
shutdown -r +2