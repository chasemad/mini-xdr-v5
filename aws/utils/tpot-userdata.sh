#!/bin/bash
# TPOT Installation User Data Script

# Update system
apt-get update
apt-get upgrade -y

# Install prerequisites
apt-get install -y curl git

# Create admin user for TPOT
useradd -m -s /bin/bash admin
usermod -aG sudo admin

# Set up SSH key for admin user
mkdir -p /home/admin/.ssh
echo "ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABgQCvX8zJ3iB9hP2rK5E0K5lGjCbSi3WVuBSdT6nkF8Qn3zX2XwP1Q3XcV9F8G3G0Q4N4Xz8Y2H6Y3Q8Z3Z1X1Z6Y1Q8Y4Z8Y9X5X4Q3X8X7Z9Y8Q5X6Y2X1Z7Y8Q9X4X3Z5Y6X8Y9Q1X2Z3Y4Z5X6Y7Z8X9Q1Y2Z3X4Y5Z6X7Z8Y9Q1X2Y3X4Z5Y6X7Z8Y9Q1X2Y3Z4X5Y6Z7X8Y9Q1X2Y3X4Z5Y6X7Z8Y9Q1X2Y3Z4X5Y6Z7X8Y9Q1X2Y3X4Z5Y6X7Z8Y9 admin@tpot" > /home/admin/.ssh/authorized_keys
chown -R admin:admin /home/admin/.ssh
chmod 700 /home/admin/.ssh
chmod 600 /home/admin/.ssh/authorized_keys

# Download and run TPOT installer
cd /home/admin
git clone https://github.com/telekom-security/tpotce.git
chown -R admin:admin tpotce

# Signal that user data is complete
touch /tmp/userdata-complete