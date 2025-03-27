#!/bin/bash

# Simple script to capture video using libcamera-vid
# Usage: ./capture_video.sh <output_video> <duration_seconds> <framerate> <width> <height> <rotation>

# Get parameters
OUTPUT_VIDEO=$1
DURATION=$2
FRAMERATE=${3:-5}  # Default to 5 fps
WIDTH=${4:-1296}  # Default width if not provided
HEIGHT=${5:-972}  # Default height if not provided
ROTATION=${6:-0}  # Default rotation if not provided
TEMP_H264="${OUTPUT_VIDEO%.mp4}.h264"

# Make sure the output directory exists
mkdir -p "$(dirname "$OUTPUT_VIDEO")"

echo "Capturing video for $DURATION seconds at $FRAMERATE fps with resolution ${WIDTH}x${HEIGHT} and rotation ${ROTATION}°"

# Apply rotation based on ROTATION parameter
if [ "$ROTATION" == "180" ]; then
    # For 180 degrees rotation, use both vflip and hflip
    libcamera-vid -t $((DURATION * 1000)) \
      --width "$WIDTH" \
      --height "$HEIGHT" \
      --framerate $FRAMERATE \
      --codec h264 \
      --inline \
      --vflip \
      --hflip \
      -o "$TEMP_H264"
    echo "Applied 180° rotation (vflip and hflip) during capture"
else
    # No rotation
    libcamera-vid -t $((DURATION * 1000)) \
      --width "$WIDTH" \
      --height "$HEIGHT" \
      --framerate $FRAMERATE \
      --codec h264 \
      --inline \
      -o "$TEMP_H264"
fi

# Wait a moment to ensure file is properly written
sleep 0.5

# Check if the file was created and has content
if [ -f "$TEMP_H264" ] && [ -s "$TEMP_H264" ]; then
    echo "Video captured successfully to $TEMP_H264"
    
    # Convert to MP4 using ffmpeg - no additional rotation needed here
    # since we already applied it during capture
    echo "Converting to browser-compatible MP4 format..."
    
    # Use libx264 for better quality and compatibility
    ffmpeg -y -r $FRAMERATE -i "$TEMP_H264" \
      -c:v libx264 \
      -r $FRAMERATE \
      "$OUTPUT_VIDEO"
    
    # Check if conversion was successful
    if [ -f "$OUTPUT_VIDEO" ] && [ -s "$OUTPUT_VIDEO" ]; then
        echo "Successfully converted to MP4: $OUTPUT_VIDEO"
        # Remove temporary file
        rm -f "$TEMP_H264"
        
        # Print duration info
        duration=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$OUTPUT_VIDEO")
        echo "Video duration: $duration seconds"
        exit 0
    else
        echo "Conversion to MP4 failed, keeping H264 file"
        # Rename h264 to mp4 so we return something usable
        mv "$TEMP_H264" "$OUTPUT_VIDEO"
        exit 1
    fi
else
    echo "Error: libcamera-vid failed to capture video or created empty file"
    exit 1
fi 