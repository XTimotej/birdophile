#!/usr/bin/env python3
import os
import time
import threading
import queue
import numpy as np
from datetime import datetime
from pathlib import Path
import cv2
import sys
import traceback

# Import picamera2
try:
    from picamera2 import Picamera2
    from picamera2.encoders import H264Encoder, Quality
    from picamera2.outputs import FileOutput
    PICAMERA2_AVAILABLE = True
except ImportError:
    PICAMERA2_AVAILABLE = False
    print("Warning: picamera2 not available. Camera functionality will be limited.")

# Configuration
DATA_DIR = "/home/timotej/birdshere"
UPLOAD_FOLDER = os.path.join(DATA_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Bird detection model settings
DETECTION_INTERVAL = 5.0  # seconds between detection attempts (increased for birds that stay longer)
MOTION_THRESHOLD = 500    # minimum contour area to trigger detection (reduced to be much more sensitive)
RECORD_SECONDS = 10       # seconds to record after detection

# Check if we're in the Flask reloader process
IN_RELOADER = os.environ.get('WERKZEUG_RUN_MAIN') == 'true'

class CameraHandler:
    def __init__(self, bird_recognizer=None):
        """Initialize the camera handler with optional bird recognition"""
        self.bird_recognizer = bird_recognizer
        self.camera = None
        self.is_running = False
        self.detection_thread = None
        self.frame_queue = queue.Queue(maxsize=10)
        self.last_frame = None
        self.last_detection_time = 0
        self.recording = False
        self.recording_lock = threading.Lock()  # Add lock for recording state
        self.camera_lock = threading.Lock()  # Add lock for camera access
        self.restart_needed = False
        self.last_error = None
        self.stream_viewers = 0  # Track number of active stream viewers
        self.stream_viewers_lock = threading.Lock()  # Lock for thread-safe viewer counting
        self.detection_active = True  # Flag to control detection activity
        self._background_thread = None
        self._monitor_thread = None
        self._notify_event_callback = None
        self._update_event_with_video_callback = None
        
        # Initialize camera if available and not in reloader process
        if PICAMERA2_AVAILABLE and (not IN_RELOADER or os.environ.get('FLASK_DEBUG') != '1'):
            self.setup_camera()
    
    def setup_camera(self):
        """Set up the picamera2 with appropriate configuration"""
        with self.camera_lock:
            # Reset error state
            self.last_error = None
            
            # Try multiple times to initialize the camera
            for attempt in range(3):  # Try 3 times
                try:
                    print(f"Camera initialization attempt {attempt+1}/3")
                    
                    # First check if camera is already in use
                    try:
                        # Try to create a temporary camera to check availability
                        temp_camera = Picamera2()
                        temp_camera.close()
                        del temp_camera
                    except Exception as e:
                        if "Pipeline handler in use by another process" in str(e):
                            print("Camera is already in use by another process. Cannot initialize.")
                            self.last_error = "Camera is already in use by another process"
                            time.sleep(1)  # Wait before retry
                            continue
                        # Other errors can be ignored here as we'll try the full setup anyway
                    
                    # Now try the full setup
                    self.camera = Picamera2()
                    
                    # Use simpler configuration for stream stability
                    # Avoid complex multi-stream setups that can cause issues
                    try:
                        # First try a reliable configuration with only main stream
                        preview_config = self.camera.create_preview_configuration(
                            main={"size": (640, 480), "format": "BGR888"},  # Use BGR888 for direct compatibility with OpenCV
                        )
                        self.camera.configure(preview_config)
                        
                        # Minimal camera controls to prevent timeouts
                        self.camera.set_controls({
                            "AwbEnable": True,  # Auto white balance as boolean
                            "FrameDurationLimits": (33333, 33333),  # Lock to ~30fps
                            "NoiseReductionMode": 1,  # Minimal noise reduction (faster)
                        })
                    except Exception as e:
                        print(f"Error with preferred configuration: {e}, trying fallback")
                        # If that fails, try an even simpler configuration
                        try:
                            fallback_config = self.camera.create_still_configuration()
                            self.camera.configure(fallback_config)
                        except Exception as e2:
                            print(f"Error with fallback configuration: {e2}, using default")
                            # Just use whatever default we can get
                            
                    print("Starting camera with simplified configuration for stability")
                    self.camera.start()
                    
                    # Wait a moment for camera to initialize properly
                    time.sleep(0.5)
                    
                    # Test capture to ensure camera is working, with error handling
                    try:
                        test_frame = self.camera.capture_array()
                        if test_frame is None or test_frame.size == 0:
                            raise Exception("Camera returned empty frame during test")
                        
                        # Log frame info for debugging
                        print(f"Test frame shape: {test_frame.shape}, dtype: {test_frame.dtype}")
                    except Exception as frame_error:
                        print(f"Error during test capture: {frame_error}")
                        raise  # Re-raise to be caught by outer exception handler
                    
                    print("Camera initialized successfully")
                    return True
                    
                except Exception as e:
                    self.last_error = str(e)
                    print(f"Error setting up camera (attempt {attempt+1}/3): {e}")
                    
                    # Try to clean up before retry
                    if self.camera:
                        try:
                            self.camera.close()
                        except:
                            pass
                    self.camera = None
                    
                    # Wait before retry - increase delay for later attempts
                    time.sleep(2 + attempt)
            
            # If we get here, all attempts failed
            print("Failed to initialize camera after multiple attempts")
            return False
    
    def start(self):
        """Start the camera and detection thread"""
        # If we're in Flask debug mode's reloader process, don't start the camera
        if IN_RELOADER and os.environ.get('FLASK_DEBUG') == '1':
            print("Not starting camera in reloader process")
            return False
            
        try:
            print("Starting camera handler...")
            
            # Make sure we're stopped first
            if self.is_running or self.camera:
                self.stop()
                time.sleep(0.5)  # Wait for cleanup to complete
            
            # If camera is not initialized, initialize it
            if not self.camera:
                if not self.setup_camera():
                    print("Failed to setup camera, cannot start")
                    return False
            
            # Start the camera
            with self.camera_lock:
                if not self.camera:
                    print("Camera not initialized")
                    return False
                    
                try:
                    self.camera.start()
                    print("Camera started successfully")
                except Exception as e:
                    print(f"Error starting camera: {e}")
                    try:
                        self.camera.close()
                    except:
                        pass
                    self.camera = None
                    return False
            
            # Set running flag
            self.is_running = True
            
            # Start detection thread
            self.detection_thread = threading.Thread(target=self._detection_loop)
            self.detection_thread.daemon = True
            self.detection_thread.start()
            print("Detection thread started")
            
            # Start monitoring thread
            self.monitor_thread = threading.Thread(target=self._monitor_thread)
            self.monitor_thread.daemon = True
            self.monitor_thread.start()
            print("Camera monitor thread started")
            
            # Reset error state
            self.restart_needed = False
            self.last_error = None
            if not hasattr(self, '_frame_failure_count'):
                self._frame_failure_count = 0
            else:
                self._frame_failure_count = 0
            
            # Try to get a test frame to verify the camera is working
            try:
                test_frame = self.get_frame()
                if test_frame:
                    print("Verified camera is producing frames")
                else:
                    print("Warning: Camera did not produce a test frame")
            except Exception as e:
                print(f"Warning: Error getting test frame: {e}")
            
            return True
        except Exception as e:
            print(f"Error starting camera: {e}")
            self.is_running = False
            
            # Try to clean up
            try:
                if self.camera:
                    self.camera.close()
                    self.camera = None
            except:
                pass
                
            return False
    
    def stop(self):
        """Stop the camera and detection thread"""
        try:
            print("Stopping camera handler...")
            
            # Stop the detection thread
            self.is_running = False
            if self.detection_thread and self.detection_thread.is_alive():
                try:
                    self.detection_thread.join(timeout=2.0)
                    if self.detection_thread.is_alive():
                        print("Warning: Detection thread did not terminate cleanly")
                except Exception as e:
                    print(f"Error stopping detection thread: {e}")
            
            # Stop the camera
            with self.camera_lock:
                if self.camera:
                    try:
                        self.camera.close()
                        print("Camera closed successfully")
                    except Exception as e:
                        print(f"Error closing camera: {e}")
                    self.camera = None
            
            # Reset internal state
            self.recording = False
            self.restart_needed = False
            
            # Force release any recording locks if they're still held
            try:
                if hasattr(self.recording_lock, '_owner') and self.recording_lock._owner:
                    print("Warning: Recording lock was still owned, forcing release")
                    self.recording_lock._owner = None
                    self.recording_lock = threading.Lock()
            except Exception as e:
                print(f"Error resetting recording lock: {e}")
            
            return True
        except Exception as e:
            print(f"Error stopping camera handler: {e}")
            return False
    
    def restart(self):
        """Restart the camera completely with low-level reset"""
        print("Performing thorough camera restart...")
        
        # First, ensure we've stopped properly
        self.stop()
        
        # Add a small delay to ensure resources are fully released
        time.sleep(2.0)
        
        # Check if there are any lingering camera processes and try to kill them
        try:
            import subprocess
            import os
            
            # Try to find and kill any stuck camera processes
            try:
                # Find Picamera2-related processes
                cmd = "ps -ef | grep -i picamera | grep -v grep | awk '{print $2}'"
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                
                if result.returncode == 0 and result.stdout.strip():
                    pids = result.stdout.strip().split('\n')
                    print(f"Found lingering camera processes: {pids}")
                    
                    # Kill each process
                    for pid in pids:
                        if pid.strip():
                            kill_cmd = f"kill -9 {pid.strip()}"
                            print(f"Killing process {pid.strip()}")
                            subprocess.run(kill_cmd, shell=True)
            except Exception as e:
                print(f"Error trying to kill lingering processes: {e}")
                
            # Reset the camera subsystem if possible (Pi-specific)
            try:
                if os.path.exists('/sys/class/video4linux/'):
                    print("Attempting to reset camera module...")
                    os.system("sudo modprobe -r v4l2_common")
                    time.sleep(0.5)
                    os.system("sudo modprobe v4l2_common")
                    time.sleep(1.0)
            except Exception as e:
                print(f"Error trying to reset camera module: {e}")
                
        except Exception as e:
            print(f"Error during low-level camera reset: {e}")
        
        # Clear instance variables
        self.camera = None
        self.is_running = False
        self.recording = False
        self.restart_needed = False
        self._frame_failure_count = 0
        
        # Create a fresh camera instance
        success = self.setup_camera()
        if success:
            # Start the camera
            if self.start():
                print("Camera successfully restarted with thorough reset")
                return True
            else:
                print("Camera failed to start after thorough reset")
        else:
            print("Failed to setup camera after thorough reset")
        
        return False
    
    def get_frame(self):
        """Get the latest camera frame as JPEG bytes"""
        if not self.camera or not self.is_running:
            # Try to restart the camera if it's not running
            if self.restart_needed:
                print("Attempting to restart camera since it's not available")
                self.restart_needed = False
                if self.stop():
                    time.sleep(1)  # Wait for resources to be released
                    if self.start():
                        print("Camera successfully restarted")
                    else:
                        print("Failed to restart camera")

            # Return a blank frame if camera still not available
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            # Add text to the blank frame
            message = "Camera not available"
            if self.last_error:
                message += f": {self.last_error}"
            cv2.putText(blank, message, (50, 240), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            _, jpeg = cv2.imencode('.jpg', blank)
            return jpeg.tobytes()
        
        try:
            # Get the latest frame with a timeout mechanism
            frame = None
            frame_captured = False
            error_message = None
            
            # Check if we need to use an emergency restart approach for persistent failures
            if hasattr(self, '_frame_failure_count') and self._frame_failure_count > 3:
                print(f"Emergency camera reset after {self._frame_failure_count} consecutive failures")
                # Reset the camera object completely
                try:
                    with self.camera_lock:
                        # Close the current camera object
                        if self.camera:
                            try:
                                self.camera.close()
                            except Exception as e:
                                print(f"Error closing camera during emergency reset: {e}")
                        
                        # Create a fresh camera instance
                        try:
                            self.camera = Picamera2()
                            
                            # Configure camera with simpler settings to ensure stability
                            preview_config = self.camera.create_preview_configuration(
                                main={"size": (640, 480), "format": "RGB888"}
                            )
                            self.camera.configure(preview_config)
                            self.camera.start()
                            
                            print("Emergency camera reset successful")
                            # Reset the failure counter
                            self._frame_failure_count = 0
                        except Exception as e:
                            print(f"Failed to create new camera during emergency reset: {e}")
                            self._frame_failure_count += 1
                except Exception as e:
                    print(f"Error during emergency camera reset: {e}")
                    self._frame_failure_count += 1
            
            # Define a function to capture frame with timeout
            def capture_with_timeout():
                nonlocal frame, frame_captured, error_message
                try:
                    # Set a flag to show we've entered the function
                    # If this flag is set but frame_captured is not, 
                    # we know the capture is hanging
                    entered_function = True
                    
                    with self.camera_lock:
                        if not self.camera or not self.is_running:
                            error_message = "Camera not available"
                            return
                        
                        # Try to verify the camera is responding before attempting capture
                        try:
                            # Check camera is ready by querying a control value
                            # (This won't capture an image but checks camera is responsive)
                            self.camera.camera_controls
                        except Exception as e:
                            error_message = f"Camera not responsive: {e}"
                            return
                        
                        # Now try to capture with extra error information
                        try:
                            frame = self.camera.capture_array()
                            if frame is None:
                                error_message = "Camera returned None frame"
                                return
                            if frame.size == 0:
                                error_message = "Camera returned empty frame"
                                return
                                
                            # Successfully captured a frame
                            frame_captured = True
                        except Exception as capture_error:
                            error_message = f"Capture error: {str(capture_error)}"
                            # Don't set frame_captured flag since we failed
                            return
                except Exception as e:
                    error_message = f"Error in capture thread: {e}"
            
            # Track start time for diagnostics
            start_time = time.time()
            
            # Start capture thread
            capture_thread = threading.Thread(target=capture_with_timeout)
            capture_thread.daemon = True
            capture_thread.start()
            
            # Wait for thread with timeout (increased timeout for slow hardware)
            capture_thread.join(timeout=1.5)  # 1.5 second timeout (increased for reliability)
            
            # Calculate how long the capture took
            capture_time = time.time() - start_time
            
            if not frame_captured:
                # If capture timed out or failed
                if not hasattr(self, '_frame_failure_count'):
                    self._frame_failure_count = 1
                else:
                    self._frame_failure_count += 1
                
                self.restart_needed = True  # Mark for restart on consistent failures
                
                # Check if the thread is still running and try to stop it
                if capture_thread.is_alive():
                    print(f"Warning: Frame capture thread is still running after {capture_time:.2f}s timeout (failure #{self._frame_failure_count})")
                    # We can't directly stop the thread, but we can prepare for restart
                
                # Log more detailed error information
                print(f"Frame capture failed after {capture_time:.2f}s: {error_message or 'Timeout'}")
                
                # If we have a previous frame, use it instead of a blank frame
                if self.last_frame is not None and self._frame_failure_count <= 2:
                    # Add "delayed" text to the frame
                    frame = self.last_frame.copy()
                    cv2.putText(frame, "Delayed Feed", (50, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    _, jpeg = cv2.imencode('.jpg', frame)
                    return jpeg.tobytes()
                else:
                    # Create a blank frame with error message
                    blank = np.zeros((480, 640, 3), dtype=np.uint8)
                    message = f"Capture timed out (attempt #{self._frame_failure_count})"
                    if error_message:
                        message += f": {error_message}"
                    cv2.putText(blank, message, (50, 240), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    _, jpeg = cv2.imencode('.jpg', blank)
                    return jpeg.tobytes()
            else:
                # Reset failure counter on success
                self._frame_failure_count = 0
            
            # Log if capture was slow but successful
            if capture_time > 0.2:  # Log if capture took more than 200ms
                print(f"Warning: Frame capture was slow but successful ({capture_time:.2f}s)")
            
            # Ensure frame is in RGB format
            if len(frame.shape) == 2:  # If grayscale
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            elif frame.shape[2] == 1:  # If single channel
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
                
            # Store as last frame for backup in case of future timeouts
            self.last_frame = frame.copy()
            
            # Convert to JPEG
            _, jpeg = cv2.imencode('.jpg', frame)
            return jpeg.tobytes()
        except Exception as e:
            self.last_error = str(e)
            print(f"Error getting frame: {e}")
            
            # Track failure count
            if not hasattr(self, '_frame_failure_count'):
                self._frame_failure_count = 1
            else:
                self._frame_failure_count += 1
            
            # Mark for restart if needed
            if "Camera not running" in str(e) or "not available" in str(e):
                self.restart_needed = True
            
            # Return a blank frame on error
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(blank, f"Error: {str(e)}", (50, 240), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            _, jpeg = cv2.imencode('.jpg', blank)
            return jpeg.tobytes()
    
    def _detection_loop(self):
        """Background thread for motion detection and bird recognition"""
        prev_frame = None
        error_count = 0
        last_debug_time = 0
        
        while self.is_running:
            try:
                # Check if detection is suspended due to active streaming
                if not self.detection_active:
                    time.sleep(0.5)  # Sleep longer when detection is suspended to reduce CPU usage
                    continue
                    
                # Periodic debug output
                current_time = time.time()
                if current_time - last_debug_time > 10:  # Every 10 seconds
                    status = "active" if self.detection_active else "suspended (stream active)"
                    print(f"Detection loop {status}. Motion threshold: {MOTION_THRESHOLD}, Detection interval: {DETECTION_INTERVAL}s")
                    last_debug_time = current_time
                
                # Check if restart is needed
                if self.restart_needed:
                    print("Detection loop detected restart needed")
                    with self.camera_lock:
                        self.restart_needed = False
                        self.restart()
                    time.sleep(1)  # Give time for restart to complete
                    continue
                
                # Get current frame
                if not self.camera:
                    time.sleep(0.1)
                    continue
                
                # Skip if recording is in progress
                with self.recording_lock:
                    if self.recording:
                        time.sleep(0.1)
                        continue
                
                # Capture frame from lores stream
                with self.camera_lock:
                    if not self.camera or not self.is_running:
                        time.sleep(0.1)
                        continue
                    try:
                        # Try lores stream first (for performance)
                        try:
                            frame = self.camera.capture_array("lores")
                        except Exception as e:
                            # If lores stream is not available, use main stream instead
                            if "Stream 'lores' is not defined" in str(e):
                                print("Lores stream not available, using main stream for motion detection")
                                frame = self.camera.capture_array("main")
                            else:
                                raise
                    except Exception as e:
                        print(f"Error capturing frame for motion detection: {e}")
                        time.sleep(0.1)
                        continue
                
                # Check if frame is valid and has the right format
                if frame is None or len(frame.shape) < 2:
                    time.sleep(0.1)
                    continue
                
                # Convert to grayscale for motion detection
                # Handle different color formats that might come from picamera2
                if len(frame.shape) == 2:
                    # Already grayscale
                    gray = frame
                elif len(frame.shape) == 3:
                    if frame.shape[2] == 1:
                        # Single channel but 3D array
                        gray = frame[:, :, 0]
                    else:
                        # RGB or other multi-channel format
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                else:
                    # Unexpected format, skip this frame
                    time.sleep(0.1)
                    continue
                
                # Apply Gaussian blur
                gray = cv2.GaussianBlur(gray, (21, 21), 0)
                
                # Initialize prev_frame if needed
                if prev_frame is None:
                    prev_frame = gray
                    time.sleep(0.1)
                    continue
                
                # Calculate frame difference for motion detection
                frame_diff = cv2.absdiff(prev_frame, gray)
                thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)[1]
                thresh = cv2.dilate(thresh, None, iterations=2)
                
                # Find contours
                contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Check for significant motion
                motion_detected = False
                max_contour_area = 0
                for contour in contours:
                    area = cv2.contourArea(contour)
                    max_contour_area = max(max_contour_area, area)
                    if area > MOTION_THRESHOLD:
                        motion_detected = True
                        break
                
                # Update previous frame
                prev_frame = gray
                
                # Handle motion detection
                current_time = time.time()
                if motion_detected and (current_time - self.last_detection_time) > DETECTION_INTERVAL:
                    print(f"Motion detected! Contour area: {max_contour_area} (threshold: {MOTION_THRESHOLD})")
                    self.last_detection_time = current_time
                    # Don't handle detection if already recording
                    with self.recording_lock:
                        if not self.recording:
                            self._handle_detection()
                elif max_contour_area > 0 and max_contour_area < MOTION_THRESHOLD and (current_time - last_debug_time > 10):
                    # Debug output for near-threshold motion
                    print(f"Motion detected but below threshold. Area: {max_contour_area} (threshold: {MOTION_THRESHOLD})")
                
                # Reset error count on successful iteration
                error_count = 0
                
                # Sleep to reduce CPU usage
                time.sleep(0.05)
                
            except Exception as e:
                error_count += 1
                self.last_error = str(e)
                print(f"Error in detection loop: {e}")
                traceback.print_exc()
                
                # If we get too many errors, try to restart the camera
                if error_count > 5:
                    print("Too many errors in detection loop, marking for restart")
                    self.restart_needed = True
                    error_count = 0
                
                time.sleep(0.1)
    
    def _handle_detection(self):
        """Handle a potential bird detection"""
        print("Motion detected - capturing image and video")
        
        try:
            # Capture high-resolution image
            with self.camera_lock:
                if not self.camera or not self.is_running:
                    print("Camera not available for detection handling")
                    return
                main_frame = self.camera.capture_array()
            
            # Ensure frame is in RGB format for saving
            if len(main_frame.shape) == 2:  # If grayscale
                main_frame = cv2.cvtColor(main_frame, cv2.COLOR_GRAY2BGR)
            elif main_frame.shape[2] == 1:  # If single channel
                main_frame = cv2.cvtColor(main_frame[:, :, 0], cv2.COLOR_GRAY2BGR)
                
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Ensure upload directory exists
            os.makedirs(UPLOAD_FOLDER, exist_ok=True)
            
            # Save image
            image_path = os.path.join(UPLOAD_FOLDER, f"{timestamp}_bird.jpg")
            cv2.imwrite(image_path, main_frame)
            
            # Start video recording in a separate thread
            threading.Thread(
                target=self._record_video,
                args=(timestamp,)
            ).start()
            
            # Identify bird species if recognizer is available
            species = "Unknown"
            if self.bird_recognizer:
                species = self.bird_recognizer(image_path)
            
            # Trigger event notification
            self._notify_event(timestamp, image_path, species)
            
        except Exception as e:
            self.last_error = str(e)
            print(f"Error handling detection: {e}")
            traceback.print_exc()
    
    def _record_video(self, timestamp):
        """Record a video clip after detection"""
        # Try to acquire the recording lock
        acquired = False
        try:
            # Use a timeout to prevent deadlocks (5 seconds)
            acquired = self.recording_lock.acquire(blocking=True, timeout=5)
            if not acquired:
                print("Already recording for too long, force resetting lock")
                # Force reset the lock (dangerous but prevents deadlock)
                try:
                    self.recording_lock._owner = None
                    self.recording_lock = threading.Lock()
                    acquired = self.recording_lock.acquire()
                    print("Lock was forcibly reset")
                except:
                    print("Could not reset lock, skipping this recording request")
                    return
            
            # If we still can't acquire the lock, give up
            if not acquired:
                print("Still cannot acquire recording lock, skipping this recording request")
                return
            
            print(f"Starting video recording for event: {timestamp}")
            self.recording = True
            
            # Use .mp4 extension for better compatibility
            video_path = os.path.join(UPLOAD_FOLDER, f"{timestamp}_bird.mp4")
            
            # Ensure upload directory exists
            os.makedirs(UPLOAD_FOLDER, exist_ok=True)
            
            # Set up encoder with Safari-compatible settings
            # Use lower bitrate, baseline profile, and more compatible settings
            try:
                # First try with the most compatible settings
                encoder = H264Encoder(
                    bitrate=2000000,  # Lower bitrate (2Mbps) for better compatibility
                    repeat=True,
                    iperiod=15,  # Keyframe every 15 frames
                    # inline_headers parameter removed as it's not supported
                    profile="baseline",  # Use baseline profile for maximum compatibility
                    level="3.1"  # Common compatibility level
                )
            except TypeError as e:
                print(f"Error with encoder parameters: {e}, trying simplified parameters")
                # If the above fails, try with minimal parameters
                try:
                    encoder = H264Encoder(
                        bitrate=2000000,
                        repeat=True
                    )
                except Exception as e2:
                    print(f"Error with simplified encoder parameters: {e2}, using default encoder")
                    # If all else fails, use default constructor
                    encoder = H264Encoder()
            
            output = FileOutput(video_path)
            
            # Start recording
            with self.camera_lock:
                if not self.camera or not self.is_running:
                    print("Camera not available for video recording")
                    return
                self.camera.start_recording(encoder, output)
            
            # Record for specified duration
            time.sleep(RECORD_SECONDS)
            
            # Stop recording
            with self.camera_lock:
                if self.camera and self.is_running:
                    self.camera.stop_recording()
                    print(f"Video recorded to {video_path}")
                    
                    # Always convert video to a more compatible format using ffmpeg
                    try:
                        import subprocess
                        compatible_video_path = os.path.join(UPLOAD_FOLDER, f"{timestamp}_bird_compatible.mp4")
                        
                        # Use ffmpeg to convert the video to a Safari-compatible format
                        cmd = [
                            'ffmpeg', '-y', '-i', video_path, 
                            '-c:v', 'libx264', '-profile:v', 'baseline', '-level', '3.0',
                            '-pix_fmt', 'yuv420p', '-preset', 'fast', '-crf', '23',
                            '-movflags', '+faststart',  # Add faststart flag for better streaming
                            compatible_video_path
                        ]
                        
                        print(f"Converting video to Safari compatible format: {' '.join(cmd)}")
                        result = subprocess.run(cmd, capture_output=True, text=True)
                        
                        if result.returncode == 0 and os.path.exists(compatible_video_path) and os.path.getsize(compatible_video_path) > 1000:
                            print(f"Video successfully converted to compatible format")
                            # Update event with the compatible video path
                            self._update_event_with_video(timestamp, compatible_video_path)
                            
                            # Optionally remove the original video to save space
                            try:
                                os.remove(video_path)
                                print(f"Removed original video file to save space: {video_path}")
                            except Exception as e:
                                print(f"Error removing original video file: {e}")
                        else:
                            print(f"Video conversion failed: {result.stderr}")
                            # Use the original video if conversion fails
                            self._update_event_with_video(timestamp, video_path)
                    except Exception as e:
                        print(f"Error converting video to compatible format: {e}")
                        # Use the original video if conversion fails
                        self._update_event_with_video(timestamp, video_path)
                else:
                    print("Camera not available to stop recording")
            
        except Exception as e:
            self.last_error = str(e)
            print(f"Error recording video: {e}")
            traceback.print_exc()
            
            # Mark for restart if needed
            if "Camera not running" in str(e) or "not available" in str(e):
                self.restart_needed = True
        finally:
            self.recording = False
            if acquired:
                try:
                    self.recording_lock.release()
                    print(f"Recording lock released for event: {timestamp}")
                except Exception as e:
                    print(f"Error releasing recording lock: {e}")
                    # Force reset the lock if it can't be released
                    try:
                        self.recording_lock._owner = None
                        self.recording_lock = threading.Lock()
                        print("Lock was forcibly reset due to release error")
                    except:
                        print("Could not reset lock after release error")
    
    def _notify_event(self, timestamp, image_path, species):
        """Send detection event to the Flask app"""
        # This is a placeholder that will be replaced by the actual implementation
        # from the camera_server.py module
        print(f"Bird detected! Species: {species}, Image: {image_path}")
        
        # Convert timestamp to ISO format if it's not already
        if isinstance(timestamp, str) and '_' in timestamp:
            # Convert YYYYMMDD_HHMMSS format to ISO format
            try:
                dt = datetime.strptime(timestamp, '%Y%m%d_%H%M%S')
                timestamp_iso = dt.isoformat()
            except ValueError:
                timestamp_iso = datetime.now().isoformat()
        else:
            timestamp_iso = datetime.now().isoformat()
        
        # Check for callback using the attribute name
        if hasattr(self, '_notify_event_callback') and callable(self._notify_event_callback):
            try:
                self._notify_event_callback(timestamp, image_path, species)
                print(f"Event notification sent via callback: {species}")
                return
            except Exception as e:
                print(f"Error in event notification callback: {e}")
                # Continue with direct method if callback fails
        
        # If the callback is not set or failed, try to send the event directly
        try:
            # Create a direct database entry
            from camera_server import add_sighting_to_db
            
            # Extract filename from path
            image_filename = os.path.basename(image_path)
            
            # Create sighting record
            sighting = {
                'id': timestamp,
                'timestamp': timestamp_iso,  # Use ISO format for timestamp
                'species': species,
                'image_path': f"uploads/{image_filename}",
                'video_path': None,  # Will be updated when video is ready
                'is_manual': False
            }
            
            # Add to database
            add_sighting_to_db(sighting)
            print(f"Added sighting directly to database: {species}")
            
            # Verify the sighting was added by checking the database
            from camera_server import get_sightings_from_db
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
            print(f"Error adding sighting directly to database: {e}")
            traceback.print_exc()
    
    def _update_event_with_video(self, timestamp, video_path):
        """Update the event with the video path"""
        # This is a placeholder that will be replaced by the actual implementation
        # from the camera_server.py module
        print(f"Updating event with video: {video_path}")
        
        # Verify the video file exists and has content
        if not os.path.exists(video_path):
            print(f"WARNING: Video file does not exist: {video_path}")
            return
            
        # Check if the file has content (more than 1KB)
        if os.path.getsize(video_path) < 1000:
            print(f"WARNING: Video file is too small (likely empty): {video_path}")
            return
        
        # Check for callback using the attribute name
        if hasattr(self, '_update_event_with_video_callback') and callable(self._update_event_with_video_callback):
            try:
                self._update_event_with_video_callback(timestamp, video_path)
                print(f"Event updated with video via callback: {video_path}")
                return
            except Exception as e:
                print(f"Error in update event with video callback: {e}")
                # Continue with direct method if callback fails
        
        # If the callback is not set or failed, try to update the event directly
        try:
            # Update database directly
            from camera_server import get_sightings_from_db, save_sightings_to_db
            
            # Extract filename from path
            video_filename = os.path.basename(video_path)
            
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
                print(f"WARNING: Could not find sighting with timestamp {timestamp} to update with video")
                # Create a new entry if no matching entry was found
                sighting = {
                    'id': timestamp,
                    'timestamp': datetime.now().isoformat(),
                    'species': "Unknown", # We don't know the species at this point
                    'image_path': None,  # We don't have an image
                    'video_path': f"uploads/{video_filename}",
                    'is_manual': False
                }
                
                # Add to database
                from camera_server import add_sighting_to_db
                add_sighting_to_db(sighting)
                print(f"Created new sighting with video {video_filename} since no matching entry was found")
        except Exception as e:
            print(f"Error updating event with video directly: {e}")
            traceback.print_exc()

    def _monitor_thread(self):
        """Background thread to monitor camera health and restart if needed"""
        last_restart_time = 0
        consecutive_restarts = 0
        
        while self.is_running:
            try:
                current_time = time.time()
                
                # Only attempt a restart if we haven't tried too recently
                if self.restart_needed and (current_time - last_restart_time > 30):
                    print("Monitor thread detected restart needed flag")
                    
                    # Limit consecutive restarts to prevent rapid restart loops
                    consecutive_restarts += 1
                    if consecutive_restarts > 5:
                        print("Too many consecutive restarts, waiting longer...")
                        time.sleep(60)  # Wait a minute before trying again
                        consecutive_restarts = 0
                    
                    # Perform the restart
                    last_restart_time = current_time
                    self.restart()
                    
                    # Reset restart flag
                    self.restart_needed = False
                elif self.restart_needed:
                    print(f"Restart needed but waiting ({int(current_time - last_restart_time)} seconds since last restart)")
                
                # Reset the consecutive restart counter if we've been stable for a while
                if (current_time - last_restart_time > 300) and consecutive_restarts > 0:  # 5 minutes
                    consecutive_restarts = 0
            except Exception as e:
                print(f"Error in monitor thread: {e}")
            
            # Sleep to prevent CPU overuse
            time.sleep(5)  # Check every 5 seconds

    def increment_stream_viewers(self):
        """Increment the count of active stream viewers"""
        with self.stream_viewers_lock:
            self.stream_viewers += 1
            if self.stream_viewers == 1:  # First viewer
                print("First stream viewer connected, suspending bird detection")
                self.detection_active = False
        
    def decrement_stream_viewers(self):
        """Decrement the count of active stream viewers"""
        with self.stream_viewers_lock:
            if self.stream_viewers > 0:
                self.stream_viewers -= 1
                if self.stream_viewers == 0:  # No more viewers
                    print("No more stream viewers, resuming bird detection")
                    self.detection_active = True
            else:
                print("Warning: stream_viewers count already at 0")
        
    def has_active_viewers(self):
        """Check if there are any active stream viewers"""
        with self.stream_viewers_lock:
            return self.stream_viewers > 0

# Singleton camera instance
_camera_instance = None

def get_camera_instance(bird_recognizer=None):
    """Get or create the singleton camera instance"""
    global _camera_instance
    if _camera_instance is None:
        _camera_instance = CameraHandler(bird_recognizer)
    return _camera_instance

if __name__ == "__main__":
    # Test the camera handler
    camera = get_camera_instance()
    if camera.start():
        print("Camera started, press Ctrl+C to stop")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            camera.stop()
    else:
        print("Failed to start camera") 