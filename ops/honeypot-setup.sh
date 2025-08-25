#!/bin/bash
# Honeypot VM setup script for Mini-XDR integration

set -e

echo "=== Mini-XDR Honeypot Setup ==="

# Install required packages
sudo apt update
sudo apt install -y ufw nftables

# Configure UFW
sudo ufw --force enable
sudo ufw allow 22022/tcp comment "SSH management"
sudo ufw allow 2222/tcp comment "Cowrie honeypot"
sudo ufw allow 22/tcp comment "Redirected SSH"

# Setup nftables for port redirection (22 -> 2222)
cat << EOF | sudo tee /etc/nftables.conf
#!/usr/sbin/nft -f

flush ruleset

table inet nat {
    chain prerouting {
        type nat hook prerouting priority 0; policy accept;
        tcp dport 22 redirect to :2222
    }
}
EOF

# Enable nftables
sudo systemctl enable nftables
sudo systemctl start nftables

# Create XDR operations user
sudo useradd -m -s /bin/bash xdrops || echo "User xdrops already exists"

# Setup SSH key directory
sudo mkdir -p /home/xdrops/.ssh
sudo chmod 700 /home/xdrops/.ssh

# Generate SSH key pair if not exists
if [ ! -f /home/xdrops/.ssh/authorized_keys ]; then
    echo "Please add the public key for xdrops user to /home/xdrops/.ssh/authorized_keys"
    echo "You can generate a key pair on the XDR host with:"
    echo "ssh-keygen -t ed25519 -f ~/.ssh/xdrops_id_ed25519"
fi

# Configure sudoers for limited UFW access
cat << EOF | sudo tee /etc/sudoers.d/xdrops
# Allow xdrops user to manage UFW rules only
xdrops ALL=(ALL) NOPASSWD: /usr/sbin/ufw
EOF

# Set ownership
sudo chown -R xdrops:xdrops /home/xdrops/.ssh
sudo chmod 600 /home/xdrops/.ssh/authorized_keys 2>/dev/null || true

echo ""
echo "=== Setup Complete ==="
echo "Next steps:"
echo "1. Add public key to /home/xdrops/.ssh/authorized_keys"
echo "2. Configure Cowrie with JSON logging"
echo "3. Setup Fluent Bit to forward logs to Mini-XDR"
echo "4. Test SSH access: ssh -p 22022 -i ~/.ssh/xdrops_id_ed25519 xdrops@<honeypot-ip>"
echo "5. Test UFW access: ssh -p 22022 -i ~/.ssh/xdrops_id_ed25519 xdrops@<honeypot-ip> sudo ufw status"
