#!/bin/bash

# This script attempts to reset camera resources by killing any processes that might be using it
# and reloading the camera modules

echo "Attempting to reset camera resources..."

# Find and kill any picamera2/libcamera processes
PIDS=$(ps -ef | grep -E 'picamera|libcamera|v4l2' | grep -v grep | awk '{print $2}')

if [ -n "$PIDS" ]; then
    echo "Found camera-related processes: $PIDS"
    echo "Killing processes..."
    for PID in $PIDS; do
        echo "Killing process $PID"
        sudo kill -9 $PID 2>/dev/null
    done
    echo "Camera processes killed."
else
    echo "No camera processes found."
fi

# Try to restart the camera module (only works if you have the right permissions)
echo "Attempting to reload camera modules..."
if [ -d "/sys/class/video4linux/" ]; then
    # Unload and reload the camera module
    sudo modprobe -r bcm2835-v4l2 2>/dev/null
    sudo modprobe -r v4l2_common 2>/dev/null
    sleep 1
    sudo modprobe v4l2_common 2>/dev/null
    sudo modprobe bcm2835-v4l2 2>/dev/null
    echo "Camera modules reloaded."
else
    echo "No video4linux directory found, skipping module reload."
fi

# Try to reset the libcamera state
echo "Clearing libcamera cache..."
if [ -d "$HOME/.cache/libcamera" ]; then
    rm -rf $HOME/.cache/libcamera/*
    echo "Libcamera cache cleared."
else
    echo "No libcamera cache found."
fi

# Wait a moment for resources to be released
sleep 2

echo "Camera reset completed. You can now run the camera service." 