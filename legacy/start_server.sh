#!/bin/bash
# Script to properly kill any existing camera processes and start the server

# Function to log messages with timestamps
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Stop any existing Python processes running app.py
log "Stopping any existing Python processes running app.py..."
pkill -f "python app.py" || true

# Wait for processes to terminate
sleep 2

# Check for any remaining camera processes
log "Checking for remaining camera processes..."
ps aux | grep -E "python|picamera" | grep -v grep

# Try to release camera resources
log "Attempting to release camera resources..."
python3 << EOF
try:
    from picamera2 import Picamera2
    print("Initializing temporary camera to release resources...")
    camera = Picamera2()
    camera.close()
    print("Camera resources released successfully")
except Exception as e:
    print(f"Error releasing camera resources: {e}")
EOF

# Make sure the uploads directory exists
log "Ensuring uploads directory exists..."
mkdir -p /home/timotej/birdshere/uploads

# Make sure the static directory and placeholder image exist
log "Checking for placeholder image..."
if [ ! -f "static/no_camera.jpg" ]; then
    log "Creating placeholder image..."
    source venv/bin/activate
    python3 static/create_placeholder.py
fi

# Kill any processes that might be using the camera
log "Killing any processes that might be using the camera..."
for pid in $(lsof -t /dev/video0 2>/dev/null); do
    log "Killing process $pid using camera..."
    kill -9 $pid || true
done

# Activate virtual environment and start server
log "Starting server..."
source venv/bin/activate
python app.py

# If we get here, the server has stopped
log "Server has stopped" 