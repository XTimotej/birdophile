#!/bin/bash
# Installation script for picamera2 and dependencies

echo "Installing picamera2 and dependencies for Birdophile..."

# Update package lists
echo "Updating package lists..."
sudo apt-get update

# Install picamera2 and dependencies
echo "Installing picamera2..."
sudo apt-get install -y python3-picamera2

# Install other system dependencies
echo "Installing system dependencies..."
sudo apt-get install -y libatlas-base-dev libhdf5-dev libhdf5-serial-dev libopenjp2-7 libtiff5

# Set up Python virtual environment
echo "Setting up Python virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

# Activate virtual environment and install Python dependencies
echo "Installing Python dependencies..."
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo "Installation complete!"
echo "To start the application, run: source venv/bin/activate && python app.py" 