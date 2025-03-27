#!/usr/bin/env python3
import sys
import os
import requests
import argparse
from pathlib import Path

# Import the bird recognition function from the existing script
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from bird_recognition import recognize_bird
except ImportError:
    # Fallback function if the module is not available
    def recognize_bird(image_path):
        print(f"Using fallback recognition for {image_path}")
        return "Unknown Bird"

# Configuration
FLASK_SERVER = "http://localhost:5000/api/event"

def send_event(image_path, video_path=None):
    """
    Send a motion detection event to the Flask server
    
    Args:
        image_path (str): Path to the captured image
        video_path (str, optional): Path to the captured video
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Recognize bird species
        species = recognize_bird(image_path)
        print(f"Recognized species: {species}")
        
        # Prepare files for upload
        files = {'image': open(image_path, 'rb')}
        data = {'species': species}
        
        if video_path and os.path.exists(video_path):
            files['video'] = open(video_path, 'rb')
        
        # Send to Flask server
        response = requests.post(FLASK_SERVER, files=files, data=data)
        
        # Close file handles
        for f in files.values():
            f.close()
        
        if response.status_code == 201:
            print(f"Successfully sent event to server: {response.json()}")
            return True
        else:
            print(f"Failed to send event: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"Error sending event: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Send motion detection event to Flask server')
    parser.add_argument('image_path', help='Path to the captured image')
    parser.add_argument('--video', help='Path to the captured video (optional)')
    
    args = parser.parse_args()
    
    success = send_event(args.image_path, args.video)
    sys.exit(0 if success else 1) 