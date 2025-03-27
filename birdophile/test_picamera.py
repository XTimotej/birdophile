#!/usr/bin/env python3

import sys
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")

try:
    import picamera2
    print(f"picamera2 imported successfully: {picamera2.__file__}")
    
    # Try to initialize camera
    from picamera2 import Picamera2
    try:
        camera = Picamera2()
        print("Picamera2 instance created successfully")
        
        # Try to get camera info
        camera_info = camera.camera_properties
        print(f"Camera info: {camera_info}")
        
    except Exception as e:
        print(f"Error initializing Picamera2: {e}")
    
except ImportError as e:
    print(f"Error importing picamera2: {e}")
    
    # Print paths where Python looks for modules
    print("\nPython path:")
    for path in sys.path:
        print(f"  {path}") 