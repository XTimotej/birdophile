#!/usr/bin/env python3
import os
import time
import threading
import requests
import traceback  # Import traceback for better error reporting
from flask import Response, Blueprint, current_app, request, jsonify
import json
import cv2
import numpy as np
from datetime import datetime

# Import our camera handler
from camera_handler import get_camera_instance
from bird_recognition import recognize_bird

# Create a Flask blueprint for camera routes
camera_bp = Blueprint('camera', __name__)

# Configuration
DATA_DIR = "/home/timotej/birdshere"
UPLOAD_FOLDER = os.path.join(DATA_DIR, "uploads")
DATABASE_FILE = os.path.join(DATA_DIR, "bird_sightings.json")

# Initialize camera with bird recognition function
camera = None
# Lock for video recording to prevent multiple recordings at once
recording_lock = threading.Lock()
# Flag to track if we're currently processing a capture request
capture_in_progress = False

def init_camera():
    """Initialize the camera with bird recognition"""
    global camera
    camera = get_camera_instance(bird_recognizer=recognize_bird)
    if camera.start():
        print("Camera system initialized successfully")
        
        # Set up event callbacks - make sure these are set as attributes
        camera._notify_event_callback = notify_event
        camera._update_event_with_video_callback = update_event_with_video
        
        # For backward compatibility, also set the old names
        camera._notify_event = notify_event
        camera._update_event_with_video = update_event_with_video
        
        print("Event callbacks registered successfully")
        
        # Verify the camera is running and callbacks are set
        if camera.is_running:
            print("Camera is running and ready to detect birds")
            
            # Verify callbacks are set correctly
            if hasattr(camera, '_notify_event_callback') and callable(camera._notify_event_callback):
                print("Event notification callback is properly set")
            else:
                print("WARNING: Event notification callback is not properly set!")
                
            if hasattr(camera, '_update_event_with_video_callback') and callable(camera._update_event_with_video_callback):
                print("Video update callback is properly set")
            else:
                print("WARNING: Video update callback is not properly set!")
        else:
            print("WARNING: Camera is not running after initialization!")
            
        return True
    else:
        print("Failed to initialize camera system")
        return False

def gen_frames():
    """Generate camera frames for streaming"""
    global camera
    consecutive_errors = 0
    last_error_time = time.time()
    last_frame_time = time.time()
    last_successful_frame = None
    health_check_interval = 20  # Check camera health every 20 frames
    frame_count = 0
    
    # Register this connection as a viewer
    if camera:
        camera.increment_stream_viewers()
    
    try:
        while True:
            try:
                # Track current time for timeout detection
                current_time = time.time()
                
                # Health check for the camera - periodically verify it's still working
                frame_count += 1
                if frame_count % health_check_interval == 0:
                    if camera:
                        if not camera.is_running:
                            print("Camera is stopped during streaming, attempting to restart")
                            try:
                                camera.start()
                            except Exception as restart_error:
                                print(f"Error restarting camera during health check: {restart_error}")
                    else:
                        print("Camera object is None during health check, attempting to reinitialize")
                        init_camera()
                        # Register as viewer with the new camera instance if needed
                        if camera:
                            camera.increment_stream_viewers()
                
                # If we're having consistent errors or no frame for too long, try to restart the camera
                if (consecutive_errors > 5 or (current_time - last_frame_time > 10)):
                    print(f"Detected camera issue: {consecutive_errors} consecutive errors, " + 
                          f"last frame {current_time - last_frame_time:.1f}s ago")
                    
                    # If there's a camera instance, try to restart it
                    if camera:
                        print("Attempting to restart camera due to persistent errors")
                        try:
                            # Try to cleanly stop and restart
                            camera.stop()
                            time.sleep(1)  # Wait for resources to release
                            if camera.start():
                                print("Successfully restarted camera")
                                consecutive_errors = 0
                                last_frame_time = time.time()
                            else:
                                print("Failed to restart camera - will reinitialize")
                                camera = None  # Will trigger complete reinitialization below
                        except Exception as restart_error:
                            print(f"Error during camera restart: {restart_error}")
                            camera = None  # Force reinitialization
                    
                    # If camera is None at this point, reinitialize completely
                    if camera is None:
                        print("Reinitializing camera completely")
                        if init_camera():
                            print("Camera successfully reinitialized")
                            consecutive_errors = 0
                            last_frame_time = time.time()
                        else:
                            print("Failed to reinitialize camera")
                            # Use the fallback blank frame since reinitialization failed
                            if consecutive_errors <= 10:  # Don't flood the logs with the same message
                                blank_frame = create_blank_frame("Camera initialization failed - retrying...")
                                yield (b'--frame\r\n'
                                      b'Content-Type: image/jpeg\r\n\r\n' + blank_frame + b'\r\n')
                            
                            # Wait before trying again to prevent rapid retries
                            time.sleep(2)
                            continue
                
                # Get frame from camera
                if camera and camera.is_running:
                    try:
                        # Get the frame
                        frame = camera.get_frame()
                        
                        if frame is not None:
                            # Update timers on success
                            last_frame_time = time.time()
                            last_successful_frame = frame
                            consecutive_errors = 0
                            
                            # Yield the frame in MJPEG format
                            yield (b'--frame\r\n'
                                  b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                        else:
                            # Frame is None
                            print("Camera returned None frame")
                            consecutive_errors += 1
                            
                            # Use last successful frame if available, otherwise show error
                            if last_successful_frame is not None and consecutive_errors <= 3:
                                yield (b'--frame\r\n'
                                       b'Content-Type: image/jpeg\r\n\r\n' + last_successful_frame + b'\r\n')
                            else:
                                blank_frame = create_blank_frame("Camera returned empty frame")
                                yield (b'--frame\r\n'
                                       b'Content-Type: image/jpeg\r\n\r\n' + blank_frame + b'\r\n')
                    except Exception as e:
                        print(f"Error capturing frame: {e}")
                        consecutive_errors += 1
                        last_error_time = time.time()
                        
                        # Use last successful frame if available for minor glitches
                        if last_successful_frame is not None and consecutive_errors <= 3:
                            yield (b'--frame\r\n'
                                   b'Content-Type: image/jpeg\r\n\r\n' + last_successful_frame + b'\r\n')
                        else:
                            blank_frame = create_blank_frame(f"Error: {str(e)}")
                            yield (b'--frame\r\n'
                                   b'Content-Type: image/jpeg\r\n\r\n' + blank_frame + b'\r\n')
                else:
                    # Camera not available
                    consecutive_errors += 1
                    
                    # Try to initialize the camera if it doesn't exist
                    if camera is None:
                        print("Camera object is None, attempting to initialize")
                        init_camera()
                    # Try to start the camera if it's not running
                    elif not camera.is_running:
                        print("Camera is not running, attempting to start")
                        camera.start()
                    
                    # If we have a last successful frame and only a few errors, use it instead of blank frame
                    if last_successful_frame is not None and consecutive_errors <= 3:
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + last_successful_frame + b'\r\n')
                    else:
                        # Yield a blank frame with message
                        blank_frame = create_blank_frame("Camera not available")
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + blank_frame + b'\r\n')
                
            except Exception as e:
                print(f"Error in gen_frames: {e}")
                consecutive_errors += 1
                last_error_time = time.time()
                
                # Yield a blank frame with error message
                blank_frame = create_blank_frame(f"Stream error: {str(e)}")
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + blank_frame + b'\r\n')
            
            # Slight delay to control frame rate and prevent CPU overuse
            # Use a shorter delay for better responsiveness
            time.sleep(0.05)
    finally:
        # Make sure to decrement the viewer count when the stream ends
        if camera:
            camera.decrement_stream_viewers()
            print("Stream connection closed, decremented viewer count")

def create_blank_frame(message):
    """Create a blank frame with text message"""
    # Create a black image
    img = cv2.imread(os.path.join(os.path.dirname(__file__), 'static/no_camera.jpg')) if os.path.exists(os.path.join(os.path.dirname(__file__), 'static/no_camera.jpg')) else None
    
    if img is None:
        img = np.zeros((480, 640, 3), np.uint8)
    
    # Add text
    cv2.putText(img, message, (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Encode as JPEG
    _, buffer = cv2.imencode('.jpg', img)
    return buffer.tobytes()

def notify_event(timestamp, image_path, species):
    """Send a bird detection event to the Flask app"""
    try:
        # Extract filename from path
        image_filename = os.path.basename(image_path)
        
        # Ensure timestamp is in ISO format for the web app
        if isinstance(timestamp, str) and '_' in timestamp:
            # Convert YYYYMMDD_HHMMSS format to ISO format
            try:
                dt = datetime.strptime(timestamp, '%Y%m%d_%H%M%S')
                timestamp_iso = dt.isoformat()
            except ValueError:
                timestamp_iso = datetime.now().isoformat()
        else:
            timestamp_iso = datetime.now().isoformat()
        
        print(f"Processing event with timestamp: {timestamp}, ISO: {timestamp_iso}")
        
        # Create sighting record
        sighting = {
            'id': timestamp,
            'timestamp': timestamp_iso,
            'species': species,
            'image_path': f"uploads/{image_filename}",
            'video_path': None,  # Will be updated when video is ready
            'is_manual': False   # All events are automated now
        }
        
        # Add to database - this will check for duplicates
        add_sighting_to_db(sighting)
        
        print(f"Event notification sent: {species}")
        
        # Verify the sighting was added
        sightings = get_sightings_from_db()
        found = False
        for s in sightings:
            if s.get('id') == timestamp:
                found = True
                print(f"Verified sighting was added to database: {timestamp}")
                break
        
        if not found:
            print(f"WARNING: Sighting {timestamp} was not found in database after adding!")
    except Exception as e:
        print(f"Error sending event notification: {e}")
        traceback.print_exc()

def update_event_with_video(timestamp, video_path):
    """Update an existing event with video information"""
    try:
        # Verify the video file exists and has content
        if not os.path.exists(video_path):
            print(f"Warning: Video file does not exist: {video_path}")
            return
            
        # Check if the file has content (more than 1KB)
        if os.path.getsize(video_path) < 1000:
            print(f"Warning: Video file is too small (likely empty): {video_path}")
            return
            
        # Extract filename from path
        video_filename = os.path.basename(video_path)
        
        print(f"Updating event with video: {timestamp} -> {video_filename}")
        
        # Update database entry
        sightings = get_sightings_from_db()
        updated = False
        
        for sighting in sightings:
            # Match by timestamp
            if sighting.get('id') == timestamp:
                sighting['video_path'] = f"uploads/{video_filename}"
                updated = True
                print(f"Updated sighting {timestamp} with video {video_filename}")
                break
        
        if updated:
            save_sightings_to_db(sightings)
            print(f"Saved updated sightings to database with video {video_filename}")
        else:
            print(f"Warning: Could not find sighting with timestamp {timestamp} to update with video")
            
            # Create a new entry if no matching entry was found
            # This ensures videos are always added to the feed even if the initial detection was missed
            
            # Convert timestamp to ISO format
            if isinstance(timestamp, str) and '_' in timestamp:
                try:
                    dt = datetime.strptime(timestamp, '%Y%m%d_%H%M%S')
                    timestamp_iso = dt.isoformat()
                except ValueError:
                    timestamp_iso = datetime.now().isoformat()
            else:
                timestamp_iso = datetime.now().isoformat()
                
            sighting = {
                'id': timestamp,
                'timestamp': timestamp_iso,
                'species': "Unknown",  # We don't know the species at this point
                'image_path': None,  # We don't have an image
                'video_path': f"uploads/{video_filename}",
                'is_manual': False
            }
            
            # Add to database
            add_sighting_to_db(sighting)
            print(f"Created new sighting with video {video_filename} since no matching entry was found")
    except Exception as e:
        print(f"Error updating event with video: {e}")
        traceback.print_exc()

def get_sightings_from_db():
    """Load bird sightings from the JSON database"""
    try:
        if os.path.exists(DATABASE_FILE):
            with open(DATABASE_FILE, 'r') as f:
                return json.load(f)
        else:
            print(f"Database file {DATABASE_FILE} does not exist, creating empty database")
            # Create the directory if it doesn't exist
            os.makedirs(os.path.dirname(DATABASE_FILE), exist_ok=True)
            return []
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Error loading database: {e}")
        return []

def save_sightings_to_db(sightings):
    """Save bird sightings to the JSON database"""
    try:
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(DATABASE_FILE), exist_ok=True)
        
        with open(DATABASE_FILE, 'w') as f:
            json.dump(sightings, f, indent=2)
    except Exception as e:
        print(f"Error saving to database: {e}")

def add_sighting_to_db(sighting):
    """Add a new sighting to the database"""
    sightings = get_sightings_from_db()
    
    # Check if a sighting with this ID already exists
    for existing in sightings:
        if existing.get('id') == sighting['id']:
            print(f"Sighting with ID {sighting['id']} already exists, not adding duplicate")
            # If the existing sighting doesn't have a video path but the new one does, update it
            if existing.get('video_path') is None and sighting.get('video_path') is not None:
                existing['video_path'] = sighting['video_path']
                print(f"Updated existing sighting with video path: {sighting['video_path']}")
                save_sightings_to_db(sightings)
            return
    
    # Check for sightings that are very close in time (within 5 seconds)
    # This helps prevent duplicates from rapid detections
    try:
        # Parse complete timestamp (including time)
        if '_' in sighting['id']:
            current_dt = datetime.strptime(sighting['id'], '%Y%m%d_%H%M%S')
            
            for existing in sightings:
                if '_' in existing.get('id', ''):
                    try:
                        existing_dt = datetime.strptime(existing.get('id'), '%Y%m%d_%H%M%S')
                        time_diff = abs((current_dt - existing_dt).total_seconds())
                        
                        # If timestamps are within 5 seconds and it's not a manual capture, consider it a duplicate
                        if time_diff < 5 and not sighting.get('is_manual', False):
                            print(f"Sighting at {sighting['id']} is too close to existing sighting at {existing['id']} ({time_diff} seconds), not adding")
                            return
                    except (ValueError, TypeError):
                        # Skip if we can't parse the timestamp
                        continue
    except (ValueError, TypeError, IndexError):
        # Continue with adding if we can't parse timestamps
        print(f"Couldn't parse timestamp for duplicate check, adding anyway: {sighting['id']}")
    
    # Add the new sighting
    sightings.append(sighting)
    save_sightings_to_db(sightings)
    print(f"Successfully added new sighting to database: {sighting['id']}")

# Flask routes
@camera_bp.route('/stream')
def video_feed():
    """Video streaming route for live camera feed"""
    try:
        # Check if camera is available first
        if not camera or not camera.is_running:
            print("Camera not available or not running when stream requested")
            # Attempt to initialize/restart if needed
            if camera is None:
                if not init_camera():
                    blank_frame = create_blank_frame("Camera initialization failed")
                    return Response(blank_frame, mimetype='image/jpeg')
            elif not camera.is_running:
                if not camera.start():
                    blank_frame = create_blank_frame("Failed to start camera")
                    return Response(blank_frame, mimetype='image/jpeg')
        
        # Create a generator function that correctly handles viewer counting
        def stream_with_viewer_tracking():
            # Register viewer when stream starts
            if camera:
                camera.increment_stream_viewers()
            
            try:
                yield from gen_frames()
            finally:
                # Make sure to decrement when stream ends (connection closed)
                if camera:
                    camera.decrement_stream_viewers()
                    print("Stream connection closed via finally block")
        
        # Set response headers for better streaming performance
        response = Response(
            stream_with_viewer_tracking(),
            mimetype='multipart/x-mixed-replace; boundary=frame'
        )
        
        # Add response headers (keep existing code)
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
        response.headers['Connection'] = 'close'  # Don't keep connections open
        response.headers['Accept-Ranges'] = 'none'
        response.timeout = 30  # 30 seconds
        
        @response.call_on_close
        def on_close():
            # Additional safety check to ensure viewer count is decremented
            if camera:
                camera.decrement_stream_viewers()
                print("Stream connection closed via call_on_close")
        
        return response
    except Exception as e:
        print(f"Error in video_feed route: {e}")
        traceback.print_exc()
        blank_frame = create_blank_frame(f"Stream error: {str(e)}")
        return Response(blank_frame, mimetype='image/jpeg')

@camera_bp.route('/status', methods=['GET'])
def camera_status():
    """Get camera status"""
    global camera
    
    if not camera:
        return jsonify({
            'status': 'not_initialized',
            'active_viewers': 0,
            'detection_active': False
        })
    
    return jsonify({
        'status': 'running' if camera.is_running else 'stopped',
        'recording': camera.recording,
        'active_viewers': camera.stream_viewers,
        'detection_active': camera.detection_active
    })

@camera_bp.route('/control', methods=['POST'])
def camera_control():
    """Control the camera (start, stop)"""
    global camera
    
    if camera is None:
        return jsonify({'error': 'Camera not initialized'}), 500
    
    action = request.json.get('action')
    
    if action == 'start':
        # If camera is already running, don't try to start it again
        if camera.is_running:
            return jsonify({'status': 'already_running'})
            
        # Try to restart the camera if it was stopped
        if camera.start():
            return jsonify({'status': 'started'})
        else:
            # If we can't start it, try to reinitialize it completely
            camera.stop()  # Make sure it's fully stopped
            time.sleep(1)  # Give it time to release resources
            
            # Create a new camera instance
            camera = get_camera_instance(bird_recognizer=recognize_bird)
            if camera.start():
                # Set up event callback
                camera._notify_event = notify_event
                camera._update_event_with_video = update_event_with_video
                return jsonify({'status': 'reinitialized'})
            else:
                return jsonify({'error': 'Failed to start camera after reinitialization'}), 500
    
    elif action == 'stop':
        camera.stop()
        return jsonify({'status': 'stopped'})
    
    else:
        return jsonify({'error': 'Invalid action. Supported actions: start, stop'}), 400

def shutdown_camera():
    """Shutdown the camera when the application exits"""
    global camera
    
    # If no camera, nothing to do
    if not camera:
        print("No camera to shutdown")
        return
    
    try:
        # Use a timeout thread to prevent hanging
        import threading
        import time
        
        shutdown_completed = False
        
        def stop_with_timeout():
            nonlocal shutdown_completed
            try:
                if camera and camera.is_running:
                    # First try to stop any active recording
                    try:
                        with camera.recording_lock:
                            if camera.recording:
                                print("Stopping active recording during shutdown")
                                try:
                                    camera.camera.stop_recording()
                                except Exception as e:
                                    print(f"Error stopping recording during shutdown: {e}")
                    except Exception as e:
                        print(f"Error accessing recording lock: {e}")
                    
                    # Now stop the camera
                    print("Stopping camera...")
                    camera.stop()
                    print("Camera system shutdown successfully")
            except Exception as e:
                print(f"Error during camera shutdown: {e}")
            finally:
                shutdown_completed = True
        
        # Start the shutdown in a separate thread
        shutdown_thread = threading.Thread(target=stop_with_timeout)
        shutdown_thread.daemon = True
        shutdown_thread.start()
        
        # Wait for shutdown with timeout
        timeout = 5  # 5 seconds timeout
        start_time = time.time()
        while not shutdown_completed and time.time() - start_time < timeout:
            time.sleep(0.1)
        
        if not shutdown_completed:
            print(f"Camera shutdown timed out after {timeout} seconds. Forcing exit.")
    except Exception as e:
        print(f"Error in shutdown_camera: {e}")
        traceback.print_exc() 