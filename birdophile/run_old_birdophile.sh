#!/bin/bash

# Create images and videos directories if they don't exist
mkdir -p "$(dirname "$0")/images"
mkdir -p "$(dirname "$0")/videos"

# List video devices for debugging
echo "Available video devices:"
ls -la /dev/video*

# Check if ffmpeg is installed and working
echo "Testing ffmpeg availability:"
ffmpeg -version | head -n 1

# Check if picamera2 is available in system Python
if ! /usr/bin/python3 -c "import picamera2" 2>/dev/null; then
    echo "ERROR: picamera2 is not installed in system Python."
    echo "Please install it with: sudo apt-get update && sudo apt-get install -y python3-picamera2"
    exit 1
fi

# Run the camera service with system Python (which has picamera2 installed)
echo "Running camera service with system Python..."
/usr/bin/python3 "$(dirname "$0")/camera_service.py" 