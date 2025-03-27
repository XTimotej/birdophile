#!/bin/bash
# Enhanced script to forcefully clean up camera resources and start the server

# Function to log messages with timestamps
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Kill any existing Python processes running app.py
log "Forcefully stopping any existing Python processes running app.py..."
sudo pkill -9 -f "python app.py" || true

# Wait for processes to terminate
sleep 3

# Check for any remaining camera processes
log "Checking for remaining camera processes..."
ps aux | grep -E "python|camera|v4l2|libcamera" | grep -v grep

# Try to release camera resources more aggressively
log "Forcefully releasing camera resources..."
sudo rmmod -f bcm2835-v4l2 2>/dev/null || true
sleep 1
sudo modprobe bcm2835-v4l2 2>/dev/null || true
sudo systemctl restart udev

# Kill any processes that might be using the camera
log "Killing any processes that might be using the camera..."
for pid in $(sudo lsof /dev/video0 2>/dev/null | awk 'NR>1 {print $2}'); do
    log "Killing process $pid using camera..."
    sudo kill -9 $pid || true
done

# Reset the camera device more aggressively
log "Resetting camera device..."
if [ -e /dev/video0 ]; then
    sudo chmod 666 /dev/video0
    log "Camera device permissions reset"
else
    log "Camera device not found"
fi

# Additional steps to reset the camera hardware
log "Performing additional camera reset steps..."
sudo v4l2-ctl --list-devices || true
sudo v4l2-ctl --all || true
sudo v4l2-ctl --set-ctrl=power_line_frequency=0 || true
sudo v4l2-ctl --set-ctrl=exposure_auto=1 || true

# Completely unload and reload camera modules
log "Unloading and reloading camera modules..."
sudo modprobe -r v4l2_common || true
sudo modprobe -r videobuf2_common || true
sudo modprobe -r videobuf2_v4l2 || true
sudo modprobe -r videobuf2_memops || true
sudo modprobe -r videobuf2_vmalloc || true
sleep 2
sudo modprobe videobuf2_common || true
sudo modprobe videobuf2_v4l2 || true
sudo modprobe videobuf2_memops || true
sudo modprobe videobuf2_vmalloc || true
sudo modprobe v4l2_common || true
sleep 2

# Make sure the uploads directory exists
log "Ensuring uploads directory exists..."
mkdir -p uploads

# Wait for camera to be available
log "Waiting for camera to be available..."
sleep 5

# Activate virtual environment and start server
log "Starting server..."
source venv/bin/activate
python app.py

# If we get here, the server has stopped
log "Server has stopped" 