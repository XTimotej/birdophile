#!/usr/bin/python3
import sys
print("Using Python at:", sys.executable)
print("Checking for picamera2:")
try:
    import picamera2
    print("picamera2 is available")
except ImportError:
    print("picamera2 is NOT available")
