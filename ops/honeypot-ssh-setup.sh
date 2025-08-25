#!/bin/bash
# Honeypot SSH Setup for XDR Remote Access
# Run this script ON the honeypot VM (10.0.0.23)

echo "=== Setting up SSH access for XDR system ==="

# 1. Create xdrops user if it doesn't exist
if ! id "xdrops" &>/dev/null; then
    echo "Creating xdrops user..."
    sudo useradd -m -s /bin/bash xdrops
fi

# 2. Create SSH directory
sudo mkdir -p /home/xdrops/.ssh
sudo chmod 700 /home/xdrops/.ssh

# 3. Add the public key (you'll need to paste your public key here)
echo "Adding SSH public key..."
cat << 'EOF' | sudo tee /home/xdrops/.ssh/authorized_keys
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIPvpS9tZDSnYx9WZyymXagulQLnxIdxXtwOTzAYgwWUL chasemad@Chases-MacBook-Pro.local
EOF

sudo chmod 600 /home/xdrops/.ssh/authorized_keys
sudo chown -R xdrops:xdrops /home/xdrops/.ssh

# 4. Add sudo permission for UFW only
echo "Configuring sudo permissions for UFW..."
echo "xdrops ALL=(ALL) NOPASSWD: /usr/sbin/ufw" | sudo tee /etc/sudoers.d/xdrops-ufw

# 5. Configure SSH daemon to allow connections
echo "Configuring SSH daemon..."
sudo sed -i 's/#Port 22/Port 22/' /etc/ssh/sshd_config
sudo sed -i 's/#PermitRootLogin yes/PermitRootLogin no/' /etc/ssh/sshd_config
sudo sed -i 's/#PubkeyAuthentication yes/PubkeyAuthentication yes/' /etc/ssh/sshd_config

# 6. Enable UFW if not already enabled
echo "Enabling UFW firewall..."
sudo ufw --force enable

# 7. Allow SSH from XDR system
echo "Allowing SSH from XDR system..."
sudo ufw allow from 10.0.0.123 to any port 22
sudo ufw allow from 10.0.0.123 to any port 22022

# 8. Restart SSH service
echo "Restarting SSH service..."
sudo systemctl restart sshd

# 9. Test UFW commands
echo "Testing UFW commands..."
sudo ufw status numbered

echo "=== SSH setup complete! ==="
echo "Test from XDR system with:"
echo "ssh -i ~/.ssh/xdrops_id_ed25519 xdrops@10.0.0.23 'sudo ufw status'"
