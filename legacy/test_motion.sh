#!/bin/bash

# This script simulates a motion detection event

# Configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
TEST_IMAGE="/tmp/test_bird.jpg"
VENV_PYTHON="$SCRIPT_DIR/venv/bin/python"

# Download a test image if it doesn't exist
if [ ! -f "$TEST_IMAGE" ]; then
    echo "Downloading test image..."
    wget -q "https://upload.wikimedia.org/wikipedia/commons/thumb/4/45/Eopsaltria_australis_-_Mogo_Campground.jpg/640px-Eopsaltria_australis_-_Mogo_Campground.jpg" -O "$TEST_IMAGE"
    
    if [ $? -ne 0 ]; then
        echo "Failed to download test image. Using a different URL..."
        wget -q "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a9/European_robin_%28Erithacus_rubecula%29.jpg/640px-European_robin_%28Erithacus_rubecula%29.jpg" -O "$TEST_IMAGE"
    fi
fi

# Check if the image exists
if [ ! -f "$TEST_IMAGE" ]; then
    echo "Error: Test image not found and could not be downloaded."
    exit 1
fi

echo "Using test image: $TEST_IMAGE"

# Call the motion handler script
echo "Simulating motion detection event..."
$SCRIPT_DIR/motion_handler.sh "$TEST_IMAGE"

echo "Done. Check the web interface to see if the event was processed." 