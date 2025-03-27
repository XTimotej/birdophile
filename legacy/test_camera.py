#!/usr/bin/env python3
import time
import cv2
import os
import signal
import sys

def signal_handler(sig, frame):
    print("Received signal, shutting down...")
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

print("Starting camera test...")

# Try to release any existing camera resources
try:
    os.system("sudo v4l2-ctl --list-devices")
    print("Listed devices")
except Exception as e:
    print(f"Error listing devices: {e}")

# Initialize the camera
cap = None
try:
    print("Attempting to open camera...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Failed to open camera")
        sys.exit(1)
    
    print("Camera opened successfully!")
    
    # Get camera properties
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Camera properties: {width}x{height} @ {fps}fps")
    
    # Capture a few frames
    for i in range(10):
        ret, frame = cap.read()
        if not ret:
            print(f"Failed to capture frame {i}")
        else:
            print(f"Captured frame {i}: {frame.shape}")
        time.sleep(0.5)
    
    print("Camera test completed successfully!")
    
except Exception as e:
    print(f"Error during camera test: {e}")
finally:
    if cap is not None:
        cap.release()
    print("Camera resources released") 