#!/bin/bash

# This script is called by the motion detection system when motion is detected
# It will send the event to the Flask application

# Configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PYTHON_ENV="$SCRIPT_DIR/venv/bin/python"  # Use the virtual environment Python

# Log file
LOG_FILE="/home/timotej/birdshere/motion_events.log"

# Function to log messages
log_message() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" >> "$LOG_FILE"
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1"
}

# Check if we have the required arguments
if [ $# -lt 1 ]; then
    log_message "Error: Missing arguments. Usage: $0 <image_path> [video_path]"
    exit 1
fi

# Get arguments
IMAGE_PATH="$1"
VIDEO_PATH=""

if [ $# -ge 2 ]; then
    VIDEO_PATH="$2"
fi

# Log the event
log_message "Motion detected: Image=$IMAGE_PATH, Video=$VIDEO_PATH"

# Call the Python script to send the event to the Flask application
if [ -n "$VIDEO_PATH" ]; then
    $PYTHON_ENV "$SCRIPT_DIR/send_event.py" "$IMAGE_PATH" --video "$VIDEO_PATH"
else
    $PYTHON_ENV "$SCRIPT_DIR/send_event.py" "$IMAGE_PATH"
fi

# Check if the event was sent successfully
if [ $? -eq 0 ]; then
    log_message "Event sent successfully"
else
    log_message "Failed to send event"
fi

exit 0 