#!/bin/bash

# This script installs the birdweb service

# Make sure we're running as root
if [ "$EUID" -ne 0 ]; then
  echo "Please run as root (use sudo)"
  exit 1
fi

# Copy the service file to the systemd directory
cp birdweb.service /etc/systemd/system/

# Reload systemd
systemctl daemon-reload

# Enable the service to start on boot
systemctl enable birdweb

# Start the service
systemctl start birdweb

# Check the status
systemctl status birdweb

echo "Service installed and started. You can access the web interface at http://$(hostname -I | awk '{print $1}'):5000" 