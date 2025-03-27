#!/usr/bin/env python3
import os
import time
import signal
import sys
import numpy as np
import threading
import queue
import traceback
import json
import shutil
from datetime import datetime
from pathlib import Path
import cv2
import requests

# Import picamera2 for camera control
from picamera2 import Picamera2
from picamera2.encoders import H264Encoder, Quality
from picamera2.outputs import FileOutput, CircularOutput

# Configuration
IMAGES_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")
VIDEOS_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "videos")
PUBLIC_FOLDER = os.path.join("/home/timotej/dev/web/public")
SIGHTINGS_JSON = os.path.join(PUBLIC_FOLDER, "sightings.json")
os.makedirs(IMAGES_FOLDER, exist_ok=True)
os.makedirs(VIDEOS_FOLDER, exist_ok=True)
os.makedirs(PUBLIC_FOLDER, exist_ok=True)

# Webhook configuration
WEBHOOK_ENABLED = True
WEBHOOK_URL = "http://localhost:3000/api/webhook"
WEBHOOK_SECRET = "bird-camera-webhook-secret"

# Camera configuration
CAMERA_ROTATION = 180
CAMERA_WIDTH = 1920
CAMERA_HEIGHT = 1080

# Motion detection settings
MOTION_THRESHOLD = 25           # Default 25
MOTION_PIXELS_PERCENTAGE = 0.01
FRAME_INTERVAL = 0.2            # Faster frame sampling for quicker detection
COOLDOWN_PERIOD = 15.0
CONSECUTIVE_FRAMES = 2          # Keep at 2 for quick response
VIDEO_DURATION = 10             # CONFIRMED: Total video duration should be 10 seconds (buffer + additional)
PRE_BUFFER_SECONDS = 5          # CONFIRMED: Should have 5 seconds of buffer
BUFFER_SIZE = 5 * 1024 * 1024   # POTENTIAL ISSUE: Buffer size might be too small
POST_MOTION_SECONDS = 5         # NEW: Explicitly define post-motion seconds (buffer + post = duration)
BUFFER_SIZE = 10 * 1024 * 1024  # INCREASED: Double buffer size to ensure it can hold 5 seconds
MAX_FRAME_PROCESSING_TIME = 3.0
BACKGROUND_LEARNING_RATE = 0.03
MOTION_HISTORY_FRAMES = 4
STATUE_DETECTION_MODE = False   # Special mode for detecting statues/test objects
USE_ABSOLUTE_DIFF = True
MOVING_OBJECT_DETECTION = False # CHANGED: Set to bird detection mode
CHECK_MOTION_DIRECTION = False
DEBUG_LOGGING = True
IGNORE_JITTER = False
JITTER_PATTERN_THRESHOLD = 0.7
PRINT_FRAME_COUNT = 30
SELECT_FIRST_QUARTER_FRAME = False  # Changed to FALSE - don't look in first quarter (buffer part)
FOCUS_ON_LATE_FRAMES = True     # NEW: Focus on frames near the end of the video (seconds 7-9)
DEFAULT_FRAME_POSITION = 0.7    # NEW: Default frame position at 70% of the video if no motion found

# ROI and blob detection settings
ROI_ENABLED = True
ROI_X = 0.15
ROI_Y = 0.15
ROI_WIDTH = 0.7
ROI_HEIGHT = 0.7
MOTION_MIN_BLOB_SIZE = 6500     # INCREASED: Reduce false positives (was 5000)
MOTION_MAX_BLOB_SIZE = 500000
MIN_BLOB_ASPECT_RATIO = 0.3
MAX_BLOB_ASPECT_RATIO = 3.0
MIN_BLOB_SOLIDITY = 0.3
DETECT_TEST_OBJECTS = True
MIN_MOVEMENT_SIZE = 950         # INCREASED: Reduce false positives (was 800)

# Contour-based detection settings
CONTOUR_DILATE_ITERATIONS = 1
MIN_CONTOUR_SCORE = 3.0         # INCREASED: Higher threshold to reduce false positives (was 2.0)

class CameraServiceBuffer:
    def __init__(self):
        """Initialize the camera service with buffer approach"""
        self.camera = None
        self.encoder = None
        self.circular_output = None
        self.file_output = None
        self.initialized = False
        self.prev_frame = None
        self.running = False
        self.frame_width = CAMERA_WIDTH
        self.frame_height = CAMERA_HEIGHT
        self.total_pixels = self.frame_width * self.frame_height
        
        # Recording state
        self.recording_lock = threading.Lock()
        self.recording = False
        self.buffer_active = False
        
        # Motion detection history
        self.background_model = None
        self.motion_history = []
        self.frame_timestamps = []
        self.motion_scores = []
        
        # Initialize the camera with buffer
        self.initialized = self.setup_camera()
    
    def setup_camera(self):
        """Set up the picamera2 with circular buffer for continuous recording"""
        try:
            # Initialize camera
            self.camera = Picamera2()
            
            # Configure camera for video recording
            preview_config = self.camera.create_video_configuration(
                main={"size": (CAMERA_WIDTH, CAMERA_HEIGHT), "format": "RGB888"}
            )
            self.camera.configure(preview_config)
            
            # Apply 180 degree rotation if needed
            if CAMERA_ROTATION == 180:
                try:
                    self.camera.set_controls({
                        "ScalerFlipY": 1,  # Vertical flip
                        "ScalerFlipX": 1   # Horizontal flip
                    })
                    print("Applied 180° rotation using ScalerFlip")
                except Exception as e:
                    print(f"ScalerFlip failed: {e}, trying RotationDegrees")
                    try:
                        self.camera.set_controls({"RotationDegrees": CAMERA_ROTATION})
                        print("Applied rotation using RotationDegrees")
                    except Exception as e2:
                        print(f"RotationDegrees failed: {e2}")
            
            # Disable autofocus to prevent jitter
            try:
                self.camera.set_controls({
                    "AfMode": 0,  # Manual focus (0=manual, 1=auto, 2=continuous)
                    "LensPosition": 0.5,  # Fixed middle position
                    "AwbEnable": True,  # Keep auto white balance
                    "FrameDurationLimits": (33333, 33333),  # Lock to ~30fps
                    "NoiseReductionMode": 1  # Minimal noise reduction
                })
                print("Autofocus disabled")
            except Exception as e:
                print(f"Could not disable autofocus: {e}, camera may jitter")
            
            print(f"Starting camera with {CAMERA_WIDTH}x{CAMERA_HEIGHT} resolution and {CAMERA_ROTATION}° rotation")
            self.camera.start()
            
            # Wait for camera to initialize
            time.sleep(0.5)
            
            # Test capture to ensure camera is working
            test_frame = self.camera.capture_array()
            if test_frame is None or test_frame.size == 0:
                raise Exception("Camera returned empty frame during test")
            
            print(f"Test frame shape: {test_frame.shape}, dtype: {test_frame.dtype}")
            print("Camera initialized successfully")
            
            return True
            
        except Exception as e:
            print(f"Error setting up camera: {e}")
            traceback.print_exc()
            if self.camera:
                try:
                    self.camera.close()
                except:
                    pass
                self.camera = None
            return False
    
    def start_buffer_recording(self):
        """Start recording to a circular buffer"""
        if not self.initialized or not self.camera:
            print("Camera not initialized. Cannot start buffer recording.")
            return False
        
        if self.buffer_active:
            print("Buffer recording already active")
            return True
        
        try:
            # Clean up any existing encoder/output
            if hasattr(self, 'encoder') and self.encoder:
                try:
                    self.camera.stop_encoder()
                except:
                    pass
                self.encoder = None
                self.circular_output = None
            
            # Calculate appropriate buffer size to ensure 5 seconds
            # For 1920x1080 at moderate compression, about 2-3 MB per second is needed
            # So for 5 seconds, we want at least 10-15 MB
            bitrate = 3000000  # 3 Mbps for good quality
            buffer_seconds = PRE_BUFFER_SECONDS
            calculated_buffer_size = int(bitrate * buffer_seconds / 8)  # Convert bits to bytes
            print(f"Using buffer size of {calculated_buffer_size/1024/1024:.1f}MB for {buffer_seconds} seconds at {bitrate/1000000:.1f}Mbps")
            
            # Initialize H264 encoder with appropriate bitrate
            self.encoder = H264Encoder(bitrate=bitrate)
            
            # Create a circular buffer output
            try:
                # First, try with newer API
                self.circular_output = CircularOutput(self.encoder, size=calculated_buffer_size)
                print("Using newer CircularOutput API with encoder parameter")
            except TypeError:
                # Fall back to older API
                try:
                    self.circular_output = CircularOutput(buffersize=calculated_buffer_size)
                    self.encoder.output = self.circular_output
                    print("Using older CircularOutput API with buffersize parameter")
                except Exception as e:
                    print(f"Error creating CircularOutput: {e}")
                    raise
            
            # Start the encoder with circular buffer
            try:
                self.camera.start_encoder(self.encoder)
            except Exception as e:
                # If this fails, try the alternative method
                print(f"Error with start_encoder: {e}, trying alternative method")
                try:
                    # Alternative method if CircularOutput is used differently
                    self.camera.start_recording(encoder=self.encoder, output=self.circular_output)
                    print("Started recording using start_recording method")
                except Exception as e2:
                    print(f"Alternative method also failed: {e2}")
                    raise
            
            # Successfully started buffer
            self.buffer_active = True
            print(f"Started circular buffer recording ({calculated_buffer_size/1024/1024:.1f}MB buffer for {buffer_seconds}s)")
            return True
            
        except Exception as e:
            print(f"Error starting buffer recording: {e}")
            traceback.print_exc()
            # Clean up if there was an error
            if hasattr(self, 'encoder') and self.encoder:
                try:
                    self.camera.stop_encoder()
                except:
                    pass
                self.encoder = None
            self.circular_output = None
            self.buffer_active = False
            return False
    
    def save_buffer_and_continue(self, duration=VIDEO_DURATION):
        """Save current buffer content and continue recording for additional time"""
        if not self.initialized or not self.camera or not self.buffer_active:
            print("Buffer recording not active. Cannot save buffer.")
            return None
        
        # Try to acquire recording lock
        acquired = False
        try:
            acquired = self.recording_lock.acquire(blocking=True, timeout=5)
            if not acquired:
                print("Already recording, skipping this recording request")
                return None
            
            self.recording = True
            print("Motion detected! Saving buffer and continuing recording...")
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_type = "_movement" if MOVING_OBJECT_DETECTION else "_statue" if STATUE_DETECTION_MODE else "_bird"
            mp4_path = os.path.join(VIDEOS_FOLDER, f"video_{timestamp}.mp4")
            temp_buffer_path = os.path.join(VIDEOS_FOLDER, f"buffer_{timestamp}.h264")
            
            try:
                # Method 1: Save buffer to temporary file
                print("Saving circular buffer to temporary file...")
                self.circular_output.stop()
                if hasattr(self.circular_output, 'save'):
                    self.circular_output.save(temp_buffer_path)
                    has_buffer = os.path.exists(temp_buffer_path) and os.path.getsize(temp_buffer_path) > 0
                    print(f"Buffer saved to {temp_buffer_path}, size: {os.path.getsize(temp_buffer_path) if has_buffer else 0} bytes")
                else:
                    print("CircularOutput doesn't have save method, trying alternatives")
                    has_buffer = False
                
                # Stop the current encoder and clean up
                try:
                    self.camera.stop_encoder()
                except:
                    pass
                
                # Create new encoder and file output for the continued recording
                additional_video_path = os.path.join(VIDEOS_FOLDER, f"additional_{timestamp}.h264")
                file_output = FileOutput(additional_video_path)
                encoder = H264Encoder(bitrate=4000000)  # Higher bitrate for better quality
                encoder.output = file_output
                
                # Start recording the additional footage - IMPORTANT: Use POST_MOTION_SECONDS not (duration - PRE_BUFFER_SECONDS)
                post_duration = POST_MOTION_SECONDS
                print(f"Recording additional {post_duration} seconds after buffer...")
                self.camera.start_encoder(encoder)
                time.sleep(post_duration)
                self.camera.stop_encoder()
                
                # Check sizes of both files to ensure we got data
                buffer_size = os.path.getsize(temp_buffer_path) if os.path.exists(temp_buffer_path) else 0
                additional_size = os.path.getsize(additional_video_path) if os.path.exists(additional_video_path) else 0
                print(f"Buffer file size: {buffer_size/1024:.1f}KB, Additional file size: {additional_size/1024:.1f}KB")
                
                # Now use ffmpeg to combine the buffer and additional recording
                import subprocess
                
                # Apply proper rotation using ffmpeg
                rotation_filter = ""
                if CAMERA_ROTATION == 180:
                    rotation_filter = "-vf 'transpose=2,transpose=2'"  # 180 degree rotation
                
                if has_buffer and buffer_size > 0:
                    print(f"Combining buffer and additional footage into {mp4_path}")
                    
                    # Verify buffer length using ffprobe
                    try:
                        cmd = f"ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 {temp_buffer_path}"
                        buffer_duration = float(subprocess.check_output(cmd, shell=True).decode('utf-8').strip())
                        print(f"Buffer duration from ffprobe: {buffer_duration:.2f} seconds")
                        
                        cmd = f"ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 {additional_video_path}"
                        additional_duration = float(subprocess.check_output(cmd, shell=True).decode('utf-8').strip())
                        print(f"Additional recording duration from ffprobe: {additional_duration:.2f} seconds")
                        
                        total_duration = buffer_duration + additional_duration
                        print(f"Total expected video duration: {total_duration:.2f} seconds")
                    except:
                        print("Could not verify durations with ffprobe")
                    
                    # Create a file list for ffmpeg
                    file_list = os.path.join(VIDEOS_FOLDER, f"filelist_{timestamp}.txt")
                    with open(file_list, 'w') as f:
                        f.write(f"file '{temp_buffer_path}'\n")
                        f.write(f"file '{additional_video_path}'\n")
                    
                    # Apply rotation if needed
                    cmd_str = f"ffmpeg -y -f concat -safe 0 -i {file_list} "
                    if CAMERA_ROTATION == 180:
                        cmd_str += f"-vf 'transpose=2,transpose=2' "
                    cmd_str += f"-c:v libx264 -preset fast -crf 22 {mp4_path}"
                    
                    print(f"Running ffmpeg command: {cmd_str}")
                    subprocess.run(cmd_str, shell=True, check=True)
                    
                    # Verify final video duration
                    try:
                        cmd = f"ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 {mp4_path}"
                        final_duration = float(subprocess.check_output(cmd, shell=True).decode('utf-8').strip())
                        print(f"FINAL VIDEO DURATION: {final_duration:.2f} seconds (expected around {VIDEO_DURATION})")
                    except:
                        print("Could not verify final duration with ffprobe")
                    
                    # Clean up temporary files
                    os.remove(file_list)
                    os.remove(temp_buffer_path)
                    os.remove(additional_video_path)
                else:
                    # If buffer saving failed, just use the additional recording
                    print("Buffer saving failed or empty, using only the additional recording")
                    
                    cmd_str = f"ffmpeg -y -i {additional_video_path} "
                    if CAMERA_ROTATION == 180:
                        cmd_str += f"-vf 'transpose=2,transpose=2' "
                    cmd_str += f"-c:v libx264 -preset fast -crf 22 {mp4_path}"
                    
                    print(f"Running ffmpeg command: {cmd_str}")
                    subprocess.run(cmd_str, shell=True, check=True)
                    os.remove(additional_video_path)
                
                print(f"Video saved to: {mp4_path}")
                
            except Exception as inner_e:
                print(f"Error processing video: {inner_e}")
                traceback.print_exc()
                # Use just the additional recording if there was an error
                if os.path.exists(additional_video_path):
                    import subprocess
                    try:
                        cmd_str = f"ffmpeg -y -i {additional_video_path} -c:v copy {mp4_path}"
                        subprocess.run(cmd_str, shell=True, check=True)
                        print(f"Saved additional recording to {mp4_path}")
                    except:
                        print("Failed to save even the additional recording")
                
            # Clean up any temporary files that might be left
            for path in [temp_buffer_path, additional_video_path, os.path.join(VIDEOS_FOLDER, f"filelist_{timestamp}.txt")]:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except:
                        pass
            
            # Extract a still frame from the video for the image
            image_path = self.extract_image_from_video(mp4_path, timestamp, output_type)
            
            # Restart the buffer recording for future events
            self.buffer_active = False
            self.start_buffer_recording()
            
            return mp4_path, image_path, timestamp
            
        except Exception as e:
            print(f"Error saving buffer: {e}")
            traceback.print_exc()
            # Try to restart buffer recording if it failed
            self.buffer_active = False
            self.start_buffer_recording()
            return None
        finally:
            # Reset recording state
            self.recording = False
            if acquired:
                try:
                    self.recording_lock.release()
                except:
                    pass
    
    def extract_image_from_video(self, video_path, timestamp, suffix):
        """Extract a still frame from the recorded video to use as the detection image"""
        try:
            # Generate image filename
            image_path = os.path.join(IMAGES_FOLDER, f"image_{timestamp}{suffix}.jpg")
            temp_image_path = os.path.join(IMAGES_FOLDER, f"temp_{timestamp}{suffix}.jpg")
            
            # Open the video file
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Error: Could not open video file {video_path}")
                return None
                
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_duration = total_frames / fps if fps > 0 else 10.0
            
            print(f"Video has {total_frames} frames at {fps:.1f} FPS (duration: {video_duration:.1f}s)")
            
            # SIMPLIFIED: Just grab a frame at 70% of the video
            # This is around the 7-second mark in a 10-second video
            target_frame = int(total_frames * DEFAULT_FRAME_POSITION)
            print(f"Extracting frame at position {target_frame} (approx. {DEFAULT_FRAME_POSITION*video_duration:.1f}s)")
            
            # Set position and read frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            ret, frame = cap.read()
            
            if not ret:
                print("Failed to read frame at target position, trying frame 0")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
                if not ret:
                    print("Could not extract any frame from video")
                    return None
            
            # Save the unrotated frame first
            if cv2.imwrite(temp_image_path, frame):
                print(f"Extracted unrotated image saved to: {temp_image_path}")
            else:
                print(f"Error saving unrotated image to {temp_image_path}")
                return None
            
            # Release the video capture
            cap.release()
            
            # SUPER SIMPLE ROTATION: Just create a Python script and run it
            if CAMERA_ROTATION == 180:
                print("Rotating image using direct Python script approach")
                
                # Create a temporary Python script to do the rotation
                script_path = os.path.join(IMAGES_FOLDER, f"rotate_{timestamp}.py")
                with open(script_path, 'w') as f:
                    f.write("""#!/usr/bin/env python3
import cv2
import sys

# Get input and output filenames from command line arguments
input_file = sys.argv[1]
output_file = sys.argv[2]

# Read the image
img = cv2.imread(input_file)
if img is None:
    print(f"Error: Could not read {input_file}")
    sys.exit(1)

# Rotate 180 degrees (flip both horizontally and vertically)
rotated = cv2.flip(img, -1)

# Save the rotated image
if cv2.imwrite(output_file, rotated):
    print(f"Successfully rotated and saved: {output_file}")
else:
    print(f"Error saving rotated image to {output_file}")
    sys.exit(1)
""")
                
                # Make the script executable
                os.chmod(script_path, 0o755)
                
                # Run the script
                try:
                    import subprocess
                    cmd = [sys.executable, script_path, temp_image_path, image_path]
                    print(f"Running: {' '.join(cmd)}")
                    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                    print(f"Rotation result: {result.stdout}")
                    
                    # Check if the rotated file exists
                    if os.path.exists(image_path):
                        print(f"Successfully rotated image to {image_path}")
                        os.remove(temp_image_path)  # Clean up temporary file
                    else:
                        print(f"Rotated file not found at {image_path}, using unrotated image")
                        os.rename(temp_image_path, image_path)
                except Exception as e:
                    print(f"Error running rotation script: {e}")
                    if os.path.exists(temp_image_path):
                        os.rename(temp_image_path, image_path)
                        print(f"Using unrotated image as fallback")
                
                # Clean up the script
                try:
                    os.remove(script_path)
                except:
                    pass
            else:
                # No rotation needed
                os.rename(temp_image_path, image_path)
            
            return image_path
            
        except Exception as e:
            print(f"Error extracting image from video: {e}")
            traceback.print_exc()
            return None
    
    def detect_motion(self):
        """Capture a frame and detect motion using contour-based algorithm"""
        if not self.initialized or not self.camera:
            return False
        
        try:
            # Skip if recording is in progress
            if self.recording:
                return False
            
            # Capture current frame
            current_frame = self.camera.capture_array()
            current_time = time.time()
            
            # Print frame number occasionally
            if hasattr(self, 'frame_counter'):
                self.frame_counter += 1
                if self.frame_counter % PRINT_FRAME_COUNT == 0:
                    print(f"\n--- Frame {self.frame_counter} ---")
            else:
                self.frame_counter = 0
            
            # Convert to grayscale for motion detection
            curr_gray = np.mean(current_frame, axis=2).astype(np.uint8)
            
            # Apply region of interest if enabled
            if ROI_ENABLED:
                # Calculate ROI boundaries
                h, w = curr_gray.shape
                roi_x_start = int(ROI_X * w)
                roi_y_start = int(ROI_Y * h)
                roi_width = int(ROI_WIDTH * w)
                roi_height = int(ROI_HEIGHT * h)
                
                # Extract ROI from current frame
                curr_roi = curr_gray[roi_y_start:roi_y_start+roi_height, 
                                    roi_x_start:roi_x_start+roi_width]
                roi_total_pixels = roi_width * roi_height
            else:
                curr_roi = curr_gray
                roi_total_pixels = self.total_pixels
            
            # Initialize background model if needed
            if self.background_model is None:
                print("Initializing background model")
                self.background_model = curr_roi.copy().astype(np.float32)
                self.motion_history = []
                self.frame_timestamps = []
                self.motion_scores = []
                # Initialize jitter detection attributes
                self.contour_areas_history = []
                self.contour_centers_history = []
                return False
            
            # Update motion history with current frame
            self.motion_history.append(curr_roi.copy())
            self.frame_timestamps.append(current_time)
            
            # Keep history limited to MOTION_HISTORY_FRAMES
            if len(self.motion_history) > MOTION_HISTORY_FRAMES:
                self.motion_history.pop(0)
                self.frame_timestamps.pop(0)
                
            # If we don't have enough frames yet, just update background and return
            if len(self.motion_history) < 2:  # Need at least 2 frames
                # Update background model with slow adaptation
                cv_alpha = BACKGROUND_LEARNING_RATE
                self.background_model = cv_alpha * curr_roi + (1 - cv_alpha) * self.background_model
                return False
                
            # Get previous frame for comparison
            prev_roi = self.motion_history[-2]
            
            # LEGACY MOTION DETECTION ALGORITHM
            # Apply Gaussian blur to reduce noise
            blurred_curr = cv2.GaussianBlur(curr_roi, (25, 25), 0)
            blurred_prev = cv2.GaussianBlur(prev_roi, (25, 25), 0)
            
            # Calculate frame difference
            frame_diff = cv2.absdiff(blurred_prev, blurred_curr)
            
            # Apply binary threshold
            thresh = cv2.threshold(frame_diff, MOTION_THRESHOLD, 255, cv2.THRESH_BINARY)[1]
            
            # Dilate thresholded image to fill in holes
            thresh = cv2.dilate(thresh, None, iterations=CONTOUR_DILATE_ITERATIONS)
            
            # Find contours in thresholded image
            try:
                contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            except ValueError:
                # OpenCV 3.x returns 3 values, OpenCV 4.x returns 2 values
                _, contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Count significant contours and get largest contour area
            significant_contours = 0
            largest_contour_area = 0
            contour_centers = []
            contour_areas = []
            
            for contour in contours:
                area = cv2.contourArea(contour)
                
                # Skip very small contours immediately
                if area < MIN_MOVEMENT_SIZE / 2:
                    continue
                    
                largest_contour_area = max(largest_contour_area, area)
                
                # Calculate contour center for tracking
                M = cv2.moments(contour)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    contour_centers.append((cx, cy))
                    contour_areas.append(area)
                
                # Check if contour is significant
                if area > MIN_MOVEMENT_SIZE:
                    significant_contours += 1
            
            # Print info periodically
            if self.frame_counter % PRINT_FRAME_COUNT == 0:
                print(f"Found {len(contours)} contours, {significant_contours} significant")
                if significant_contours > 0:
                    print(f"Largest contour area: {largest_contour_area} (threshold: {MIN_MOVEMENT_SIZE})")
            
            # Store contour data for jitter detection
            if not hasattr(self, 'contour_areas_history'):
                self.contour_areas_history = []
                self.contour_centers_history = []
                
            self.contour_areas_history.append(contour_areas)
            self.contour_centers_history.append(contour_centers)
            
            # Keep limited history
            if len(self.contour_areas_history) > MOTION_HISTORY_FRAMES:
                self.contour_areas_history.pop(0)
                self.contour_centers_history.pop(0)
            
            # JITTER DETECTION - Check for autofocus-like oscillation patterns
            is_jitter = False
            if IGNORE_JITTER == False and len(self.contour_areas_history) >= 3 and significant_contours > 0:
                # Check if the contour areas are oscillating (a sign of focus jitter)
                area_oscillation = False
                position_oscillation = False
                
                # Analyze area history for oscillation pattern
                if self.contour_areas_history[-3] and self.contour_areas_history[-2] and self.contour_areas_history[-1]:
                    area_diffs = []
                    for i in range(1, len(self.contour_areas_history)):
                        # Compare current frame areas with previous frame areas
                        prev_areas = self.contour_areas_history[i-1]
                        curr_areas = self.contour_areas_history[i]
                        
                        # If we have areas in both frames, calculate differences
                        if prev_areas and curr_areas:
                            # Use the first area as representative
                            if len(prev_areas) > 0 and len(curr_areas) > 0:
                                area_diff = abs(prev_areas[0] - curr_areas[0]) / max(prev_areas[0], 1)
                                area_diffs.append(area_diff)
                    
                    # Check for alternating pattern in area differences
                    if len(area_diffs) >= 2:
                        # Jitter typically shows as small, regular oscillations
                        avg_diff = sum(area_diffs) / len(area_diffs)
                        if avg_diff < JITTER_PATTERN_THRESHOLD:  # Using configurable threshold
                            area_oscillation = True
                            print("Detected area oscillation pattern typical of focus jitter")
                
                # Also check for position oscillation (jitter often moves back and forth)
                if len(self.contour_centers_history) >= 3:
                    # Check centers from consecutive frames
                    center_movement = []
                    
                    for i in range(1, len(self.contour_centers_history)):
                        prev_centers = self.contour_centers_history[i-1]
                        curr_centers = self.contour_centers_history[i]
                        
                        # If we have centers in both frames
                        if prev_centers and curr_centers:
                            # Use the first center as representative
                            if len(prev_centers) > 0 and len(curr_centers) > 0:
                                dx = prev_centers[0][0] - curr_centers[0][0]
                                dy = prev_centers[0][1] - curr_centers[0][1]
                                dist = np.sqrt(dx*dx + dy*dy)
                                center_movement.append(dist)
                    
                    # Check if movement is small and consistent (jitter characteristic)
                    if len(center_movement) >= 2:
                        avg_movement = sum(center_movement) / len(center_movement)
                        if 1 < avg_movement < 15:  # INCREASED: Allow more movement before calling it jitter (was < 10)
                            position_oscillation = True
                            print("Detected position oscillation pattern typical of focus jitter")
                
                # Combine criteria to identify jitter - CHANGED: Now needs both criteria to trigger
                is_jitter = area_oscillation and position_oscillation  # Changed from OR to AND for stricter filtering
                if is_jitter:
                    print("FOCUS JITTER DETECTED - Ignoring false motion")
            
            # Track consecutive motion frames
            if largest_contour_area > MIN_MOVEMENT_SIZE and significant_contours > 0 and not is_jitter:
                # If we have motion scores, add a new score
                if hasattr(self, 'motion_scores'):
                    # Calculate score based on contour size and count
                    score = largest_contour_area / MIN_MOVEMENT_SIZE
                    self.motion_scores.append(score)
                    
                    # Keep limited history
                    if len(self.motion_scores) > CONSECUTIVE_FRAMES:
                        self.motion_scores.pop(0)
                    
                    # Check if we have enough consecutive frames with motion
                    if len(self.motion_scores) >= CONSECUTIVE_FRAMES:
                        avg_score = sum(self.motion_scores) / len(self.motion_scores)
                        print(f"Motion scores: {', '.join([f'{score:.2f}' for score in self.motion_scores])}")
                        print(f"Average score: {avg_score:.2f}")
                        
                        # If average score is significant, trigger detection
                        if avg_score > MIN_CONTOUR_SCORE:
                            print(f"MOTION DETECTED! Average score: {avg_score:.2f}")
                            return True
                        else:
                            print(f"Motion detected but score too low: {avg_score:.2f} < {MIN_CONTOUR_SCORE}")
                    else:
                        # Print partial scores
                        if DEBUG_LOGGING and len(self.motion_scores) > 0:
                            print(f"Partial motion scores ({len(self.motion_scores)}/{CONSECUTIVE_FRAMES}): {', '.join([f'{score:.2f}' for score in self.motion_scores])}")
                else:
                    # Initialize motion scores if they don't exist
                    self.motion_scores = []
            else:
                # Reset motion scores when no significant motion or when jitter is detected
                if hasattr(self, 'motion_scores') and len(self.motion_scores) > 0:
                    if self.frame_counter % PRINT_FRAME_COUNT == 0:
                        print(f"Resetting motion scores")
                    self.motion_scores = []
            
            # Update background model with slow adaptation
            cv_alpha = BACKGROUND_LEARNING_RATE
            self.background_model = cv_alpha * curr_roi + (1 - cv_alpha) * self.background_model
            
            return False
                
        except Exception as e:
            print(f"Error detecting motion: {e}")
            traceback.print_exc()
            return False
    
    def send_webhook_notification(self):
        """Send a webhook notification to the Next.js app to trigger revalidation"""
        if not WEBHOOK_ENABLED:
            return False
        
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {WEBHOOK_SECRET}"
            }
            
            response = requests.post(WEBHOOK_URL, headers=headers, json={"event": "sightings_updated"})
            
            if response.status_code == 200:
                print(f"Webhook notification sent successfully: {response.json()}")
                return True
            else:
                print(f"Failed to send webhook notification: {response.status_code} {response.text}")
                return False
        except Exception as e:
            print(f"Error sending webhook notification: {e}")
            return False
    
    def update_sightings_json(self, image_path, video_path, timestamp):
        """Update the sightings JSON file with new detection information"""
        try:
            # Create a timestamp in the desired format for the JSON
            formatted_timestamp = datetime.strptime(timestamp, "%Y%m%d_%H%M%S").strftime("%Y-%m-%d %H:%M:%S")
            
            # Get relative paths for web display
            rel_image_path = os.path.basename(image_path)
            rel_video_path = os.path.basename(video_path)
            
            # Create entry for the new sighting
            new_sighting = {
                "id": timestamp,
                "timestamp": formatted_timestamp,
                "date": datetime.strptime(timestamp, "%Y%m%d_%H%M%S").strftime("%Y-%m-%d"),
                "time": datetime.strptime(timestamp, "%Y%m%d_%H%M%S").strftime("%H:%M:%S"),
                "image": rel_image_path,
                "video": rel_video_path,
                "type": "bird" if not STATUE_DETECTION_MODE and not MOVING_OBJECT_DETECTION else
                       "statue" if STATUE_DETECTION_MODE else "moving_object"
            }
            
            # Load existing sightings or create new structure
            sightings = []
            if os.path.exists(SIGHTINGS_JSON):
                try:
                    with open(SIGHTINGS_JSON, 'r') as f:
                        sightings = json.load(f)
                except json.JSONDecodeError:
                    print(f"Error reading existing JSON, creating new file")
                    sightings = []
            
            # Add new sighting
            sightings.append(new_sighting)
            
            # Sort by timestamp (newest first)
            sightings.sort(key=lambda x: x["timestamp"], reverse=True)
            
            # Write updated sightings back to file
            with open(SIGHTINGS_JSON, 'w') as f:
                json.dump(sightings, f, indent=2)
            
            # Copy image and video to web public folder
            web_images_folder = os.path.join(PUBLIC_FOLDER, "images")
            web_videos_folder = os.path.join(PUBLIC_FOLDER, "videos")
            
            os.makedirs(web_images_folder, exist_ok=True)
            os.makedirs(web_videos_folder, exist_ok=True)
            
            # Copy image if it exists
            if image_path and os.path.exists(image_path):
                web_image_path = os.path.join(web_images_folder, os.path.basename(image_path))
                shutil.copy2(image_path, web_image_path)
            
            # Copy video if it exists
            if video_path and os.path.exists(video_path):
                web_video_path = os.path.join(web_videos_folder, os.path.basename(video_path))
                shutil.copy2(video_path, web_video_path)
            
            print(f"Updated sightings JSON at {SIGHTINGS_JSON}")
            
            # Send webhook notification to Next.js
            self.send_webhook_notification()
            
            return True
        except Exception as e:
            print(f"Error updating sightings JSON: {e}")
            traceback.print_exc()
            return False
    
    def monitor_motion(self):
        """Continuously monitor for motion, using buffer-based recording"""
        if not self.initialized or not self.camera:
            print("Camera not initialized. Cannot monitor for motion.")
            return
        
        self.running = True
        last_motion_time = 0
        frame_count = 0
        consecutive_motion_frames = 0
        
        try:
            # Set appropriate detection mode description
            if MOVING_OBJECT_DETECTION:
                mode_desc = "MOVING OBJECT DETECTION"
            elif STATUE_DETECTION_MODE:
                mode_desc = "STATUE DETECTION"
            else:
                mode_desc = "BIRD DETECTION"
                
            print(f"\n*** Starting buffered motion monitoring with {mode_desc} mode ***")
            print(f"Motion threshold: {MOTION_THRESHOLD}")
            print(f"Required consecutive frames: {CONSECUTIVE_FRAMES}")
            print(f"Pre-recording buffer: {PRE_BUFFER_SECONDS}s, total duration: {VIDEO_DURATION}s")
            print(f"Checking every {FRAME_INTERVAL}s, cooldown: {COOLDOWN_PERIOD}s")
            if ROI_ENABLED:
                print(f"ROI enabled: using {ROI_WIDTH*100:.0f}% x {ROI_HEIGHT*100:.0f}% of frame")
            
            # Learning phase for background
            print("\nStarting background learning...")
            
            # First frame
            first_frame = self.camera.capture_array()
            first_gray = np.mean(first_frame, axis=2).astype(np.uint8)
            
            if ROI_ENABLED:
                h, w = first_gray.shape
                roi_x_start = int(ROI_X * w)
                roi_y_start = int(ROI_Y * h)
                roi_width = int(ROI_WIDTH * w)
                roi_height = int(ROI_HEIGHT * h)
                first_roi = first_gray[roi_y_start:roi_y_start+roi_height, 
                                      roi_x_start:roi_x_start+roi_width]
            else:
                first_roi = first_gray
            
            self.background_model = first_roi.copy().astype(np.float32)
            
            # Gather more frames for better initial background
            num_frames = 6 if MOVING_OBJECT_DETECTION else 4
            print(f"Capturing {num_frames} frames to establish baseline...")
            for i in range(num_frames):
                time.sleep(0.3)
                curr_frame = self.camera.capture_array()
                curr_gray = np.mean(curr_frame, axis=2).astype(np.uint8)
                
                if ROI_ENABLED:
                    curr_roi = curr_gray[roi_y_start:roi_y_start+roi_height, 
                                        roi_x_start:roi_x_start+roi_width]
                else:
                    curr_roi = curr_gray
                
                # Update background model with higher weight for initial learning
                learn_rate = 0.4 if i < 2 else 0.2  # Higher rate for first frames
                self.background_model = learn_rate * curr_roi + (1 - learn_rate) * self.background_model
                print(f"Learning frame {i+1}/{num_frames}...")
            
            # Start circular buffer recording
            print("\nStarting circular buffer recording...")
            if not self.start_buffer_recording():
                print("Failed to start buffer recording, exiting")
                return
            
            print(f"\nBackground learned and buffer active. Monitoring for motion...")
            print("Ready to detect birds!")
            
            # Register signal handlers
            def signal_handler(sig, frame):
                print("\nReceived signal, shutting down...")
                self.running = False
            
            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)
            
            # Clear motion history
            self.motion_history = []
            self.frame_timestamps = []
            self.motion_scores = []
            
            last_frame_time = time.time()
            buffer_check_time = time.time()
            
            while self.running:
                # Periodically check buffer status
                if time.time() - buffer_check_time > 60:  # Every minute
                    if not self.buffer_active:
                        print("Buffer not active, restarting...")
                        self.start_buffer_recording()
                    buffer_check_time = time.time()
                
                frame_count += 1
                current_time = time.time()
                frame_delta = current_time - last_frame_time
                
                # Check if we need to sleep to maintain our desired frame rate
                if frame_delta < FRAME_INTERVAL:
                    sleep_time = FRAME_INTERVAL - frame_delta
                    time.sleep(sleep_time)
                
                # Time the motion detection processing
                process_start_time = time.time()
                
                # Refresh background model periodically
                refresh_interval = 300
                if frame_count % refresh_interval == 0:
                    print(f"Periodic background refresh (frame {frame_count})")
                    # Don't reset completely, just increase adaptation rate temporarily
                    curr_frame = self.camera.capture_array()
                    curr_gray = np.mean(curr_frame, axis=2).astype(np.uint8)
                    
                    if ROI_ENABLED:
                        h, w = curr_gray.shape
                        roi_x_start = int(ROI_X * w)
                        roi_y_start = int(ROI_Y * h)
                        roi_width = int(ROI_WIDTH * w)
                        roi_height = int(ROI_HEIGHT * h)
                        curr_roi = curr_gray[roi_y_start:roi_y_start+roi_height, 
                                            roi_x_start:roi_x_start+roi_width]
                    else:
                        curr_roi = curr_gray
                    
                    # Use moderate learning rate for periodic refresh
                    self.background_model = 0.3 * curr_roi + 0.7 * self.background_model
                    self.motion_history = []
                    self.frame_timestamps = []
                    self.motion_scores = []
                    
                # Check for motion
                try:
                    motion_detected = self.detect_motion()
                except Exception as e:
                    print(f"Error in motion detection: {e}")
                    motion_detected = False
                    # If we get consistent errors, check if we need to restart the camera
                    if not self.camera or not self.initialized:
                        print("Camera error detected, attempting to reinitialize...")
                        try:
                            self.setup_camera()
                            if self.initialized:
                                print("Camera reinitialized successfully")
                                # Reset background model
                                self.background_model = None
                        except:
                            print("Failed to reinitialize camera")
                            time.sleep(5)  # Wait before trying again
                
                # Calculate how long the processing took
                process_time = time.time() - process_start_time
                
                # Update timing
                last_frame_time = time.time()
                
                if motion_detected:
                    # Track consecutive motion frames
                    consecutive_motion_frames += 1
                    print(f"Motion detected in {consecutive_motion_frames} consecutive frames")
                    
                    # Only trigger if we're past cooldown period AND have consistent motion
                    if (time.time() - last_motion_time > COOLDOWN_PERIOD and 
                        consecutive_motion_frames >= CONSECUTIVE_FRAMES):
                        if MOVING_OBJECT_DETECTION:
                            object_type = "MOVING OBJECT"
                        elif STATUE_DETECTION_MODE:
                            object_type = "STATUE"
                        else:
                            object_type = "BIRD"
                            
                        print(f"\n*** {object_type} DETECTED! SAVING BUFFER... ***\n")
                        
                        # Save the buffer and continue recording
                        result = self.save_buffer_and_continue(VIDEO_DURATION)
                        
                        if result:
                            video_path, image_path, timestamp = result
                            print(f"Video saved to: {video_path}")
                            print(f"Image extracted to: {image_path}")
                            last_motion_time = time.time()
                            
                            # Update sightings JSON file with new detection
                            try:
                                self.update_sightings_json(image_path, video_path, timestamp)
                            except Exception as e:
                                print(f"Error updating sightings JSON: {e}")
                        
                        # Reset background and motion history
                        self.background_model = None
                        self.motion_history = []
                        self.frame_timestamps = []
                        self.motion_scores = []
                        
                        # Re-learn background
                        print("Re-learning background...")
                        first_frame = self.camera.capture_array()
                        first_gray = np.mean(first_frame, axis=2).astype(np.uint8)
                        
                        if ROI_ENABLED:
                            h, w = first_gray.shape
                            roi_x_start = int(ROI_X * w)
                            roi_y_start = int(ROI_Y * h)
                            roi_width = int(ROI_WIDTH * w)
                            roi_height = int(ROI_HEIGHT * h)
                            first_roi = first_gray[roi_y_start:roi_y_start+roi_height, 
                                                roi_x_start:roi_x_start+roi_width]
                        else:
                            first_roi = first_gray
                            
                        self.background_model = first_roi.copy()
                        
                        # Add a few more frames to stabilize
                        for i in range(3):
                            time.sleep(0.3)
                            curr_frame = self.camera.capture_array()
                            curr_gray = np.mean(curr_frame, axis=2).astype(np.uint8)
                            
                            if ROI_ENABLED:
                                curr_roi = curr_gray[roi_y_start:roi_y_start+roi_height, 
                                                   roi_x_start:roi_x_start+roi_width]
                            else:
                                curr_roi = curr_gray
                            
                            # Higher learning rate for quick re-learning
                            self.background_model = 0.4 * curr_roi + 0.6 * self.background_model
                            
                        print("\nBackground re-learned. Resuming monitoring...")
                        print("Ready to detect more birds!")
                        last_frame_time = time.time()  # Reset timing after recording
                        
                        # Reset consecutive motion count
                        consecutive_motion_frames = 0
                else:
                    # Reset consecutive motion count on any non-motion frame
                    consecutive_motion_frames = 0
        
        except Exception as e:
            print(f"Error in buffered motion monitoring: {e}")
            traceback.print_exc()
        finally:
            # Try to stop any ongoing recording
            if self.buffer_active:
                try:
                    self.camera.stop_encoder()
                    self.encoder = None
                    self.circular_output = None
                    self.file_output = None
                    self.buffer_active = False
                except:
                    pass
            print("Buffered motion monitoring stopped")
    
    def close(self):
        """Close the camera resources"""
        self.running = False
        
        # Stop any ongoing recording
        if self.buffer_active:
            try:
                self.camera.stop_encoder()
                self.encoder = None
                self.circular_output = None
                self.file_output = None
                self.buffer_active = False
            except:
                pass
                
        if self.camera:
            try:
                self.camera.close()
                print("Camera closed")
            except Exception as e:
                print(f"Error closing camera: {e}")
            finally:
                self.camera = None
                self.initialized = False

def main():
    """Main function to test the buffered camera service"""
    print("Starting buffered camera service...")
    
    camera_service = CameraServiceBuffer()
    
    if camera_service.initialized:
        # Start motion monitoring with buffer
        try:
            camera_service.monitor_motion()
        except KeyboardInterrupt:
            print("\nKeyboard interrupt received, shutting down...")
    else:
        print("Camera could not be initialized.")
    
    # Clean up
    camera_service.close()
    print("Camera service completed.")

if __name__ == "__main__":
    main()
