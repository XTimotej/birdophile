#!/bin/bash

# Restart script for the bird detection system
echo "Restarting the bird detection system..."

# Stop any running instances
echo "Stopping any running instances..."
pkill -f "python3 app.py" || true
pkill -f "python3 camera_handler.py" || true
sleep 2

# Check if any processes are still running
if pgrep -f "python3 app.py" > /dev/null || pgrep -f "python3 camera_handler.py" > /dev/null; then
    echo "Forcefully killing remaining processes..."
    pkill -9 -f "python3 app.py" || true
    pkill -9 -f "python3 camera_handler.py" || true
    sleep 1
fi

# Clear any temporary files
echo "Cleaning up temporary files..."
rm -f /tmp/picamera*

# Start the application
echo "Starting the application..."
cd /home/timotej/birdweb
source venv/bin/activate
nohup python3 app.py > birdweb.log 2>&1 &

echo "Waiting for the application to start..."
sleep 5

# Check if the application is running
if pgrep -f "python3 app.py" > /dev/null; then
    echo "Bird detection system restarted successfully!"
    echo "You can view the log with: tail -f birdweb.log"
else
    echo "Failed to restart the bird detection system. Check the logs for errors."
fi 