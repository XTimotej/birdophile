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
import requests  # Add import for webhook functionality

# Try to import picamera2, which might be installed system-wide on Raspberry Pi
try:
    from picamera2 import Picamera2
    from picamera2.encoders import H264Encoder
    from picamera2.outputs import FileOutput
    PICAMERA2_AVAILABLE = True
except ImportError:
    PICAMERA2_AVAILABLE = False
    print("Warning: picamera2 not available. Will try fallback methods if needed.")

# Try to import scipy for advanced filtering (optional)
try:
    from scipy import ndimage
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available, using simplified motion detection.")

# Initialize tool availability (will be updated in check_system_requirements)
RASPIVID_AVAILABLE = False
LIBCAMERA_VID_AVAILABLE = False
FFMPEG_AVAILABLE = False

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
CAMERA_ROTATION = 180          # Rotate camera 180 degrees (fix upside down)
CAMERA_WIDTH = 1920            # Higher resolution width
CAMERA_HEIGHT = 1080            # Higher resolution height

# Motion detection settings
MOTION_THRESHOLD = 30           # Increased threshold to reduce sensitivity (was 25)
MOTION_PIXELS_PERCENTAGE = 0.01  # Back to more moderate values
FRAME_INTERVAL = 0.4             # Slightly longer interval to reduce jitter detection (was 0.3)
COOLDOWN_PERIOD = 15.0           # Extended cooldown period (was 10.0)
CONSECUTIVE_FRAMES = 3           # Require more consecutive frames (was 2)
VIDEO_DURATION = 5              # Keep video duration
MAX_FRAME_PROCESSING_TIME = 3.0  # Keep higher allowance for processing time
CONSECUTIVE_FRAME_MAX_TIME = 3.0 # Keep for consistency
BACKGROUND_LEARNING_RATE = 0.03  # Slower background adaptation to reduce jitter impact (was 0.05)
MOTION_HISTORY_FRAMES = 4        # Increased frame history (was 3)
STATUE_DETECTION_MODE = False     # Special mode for detecting statues/test objects
USE_ABSOLUTE_DIFF = True         # Use absolute difference for stable detection
MOVING_OBJECT_DETECTION = True   # Focus on detecting purposeful movement
CHECK_MOTION_DIRECTION = False   # Keep direction checking disabled
DEBUG_LOGGING = False            # Disable detailed debug logging

# Disable all the remaining special debug settings
PRINT_PIXEL_VALUES = False
SHOW_MOTION_THRESHOLDS = False
FORCE_DETECTION_DEBUG = False
FORCE_DETECTION_INTERVAL = 100
IGNORE_JITTER = True
JITTER_PATTERN_THRESHOLD = 0.5   # Higher jitter threshold (was 0.3)
CONSISTENT_MOTION_FRAMES = 5     # Require more consistent frames (was 4)

# Disable global motion detection
FILTER_GLOBAL_MOTION = False
GLOBAL_MOTION_THRESHOLD = 0.7
OPTICAL_FLOW_ENABLED = False

# Bird detection specific settings
ROI_ENABLED = True              # Enable Region of Interest to focus on entry area
ROI_X = 0.15                    # Increase margin to focus on center (was 0.1)
ROI_Y = 0.15                    # Increase margin to focus on center (was 0.1)
ROI_WIDTH = 0.7                # Reduce width to avoid edges (was 0.8)
ROI_HEIGHT = 0.7               # Reduce height to avoid edges (was 0.8)
MOTION_MIN_BLOB_SIZE = 8000     # INCREASED to better filter out focus jitter (was 5000)
MOTION_MAX_BLOB_SIZE = 500000   # Keep high max
MIN_BLOB_ASPECT_RATIO = 0.3     # Stricter ratio to filter out lens aberrations (was 0.2)
MAX_BLOB_ASPECT_RATIO = 3.0     # Stricter ratio to filter out lens aberrations (was 5.0)
MIN_BLOB_SOLIDITY = 0.3         # Increased to require more solid objects (was 0.2)
DETECT_TEST_OBJECTS = True      # Enable test object detection
EXTREME_SENSITIVITY = False     # DISABLED extreme sensitivity

# Add camera jitter filter settings
MIN_MOVEMENT_SIZE = 1000          # DOUBLED to require larger movement (was 500)

# NEW GLOBAL MOTION FILTER SETTINGS

# Legacy-style contour-based detection settings
CONTOUR_DILATE_ITERATIONS = 1    # Reduced dilation to prevent merging noise (was 2)
MIN_CONTOUR_SCORE = 2.5          # Increased minimum score for higher confidence (was 1.5)

class CameraService:
    def __init__(self):
        """Initialize the camera service"""
        self.camera = None
        self.initialized = False
        self.prev_frame = None
        self.running = False
        self.frame_width = CAMERA_WIDTH
        self.frame_height = CAMERA_HEIGHT
        self.total_pixels = self.frame_width * self.frame_height
        
        # Add recording lock to prevent multiple recordings at once
        self.recording_lock = threading.Lock()
        self.recording = False
        
        # Motion detection history
        self.background_model = None
        self.motion_history = []
        self.frame_timestamps = []
        self.motion_scores = []
        
        # Clean up any diagnostic folders from previous runs
        try:
            for folder in ['diagnostics', 'masks', 'heatmaps']:
                diag_dir = os.path.join(IMAGES_FOLDER, folder)
                if os.path.exists(diag_dir):
                    shutil.rmtree(diag_dir)
                    print(f"Cleaned up diagnostic folder: {diag_dir}")
        except Exception as e:
            print(f"Error cleaning up diagnostic folders: {e}")
        
        # Only try to initialize camera if picamera2 is available
        if PICAMERA2_AVAILABLE:
            self.initialized = self.setup_camera()
        else:
            print("Picamera2 is not available. Cannot initialize camera.")
    
    def setup_camera(self):
        """Set up the picamera2 with appropriate configuration from legacy code"""
        try:
            # Now try the full setup
            self.camera = Picamera2()
            
            # Use simpler configuration for stream stability
            try:
                # First try a reliable configuration with only main stream
                preview_config = self.camera.create_preview_configuration(
                    main={"size": (CAMERA_WIDTH, CAMERA_HEIGHT), "format": "BGR888"}  # Use higher resolution
                )
                self.camera.configure(preview_config)
                
                # Apply rotation to fix upside-down image
                if CAMERA_ROTATION == 180:
                    # For 180 degrees, we need both horizontal and vertical flips
                    try:
                        # Try using ScalerFlip for Pi Camera Module 3
                        self.camera.set_controls({
                            "ScalerFlipY": 1,  # Vertical flip
                            "ScalerFlipX": 1   # Horizontal flip
                        })
                        print("Applied 180° rotation using ScalerFlip")
                    except Exception as e:
                        print(f"ScalerFlip failed: {e}, trying RotationDegrees")
                        try:
                            # Try standard rotation
                            self.camera.set_controls({"RotationDegrees": CAMERA_ROTATION})
                            print("Applied rotation using RotationDegrees")
                        except Exception as e2:
                            print(f"RotationDegrees failed: {e2}")
                
                # Try multiple methods to disable autofocus
                try:
                    print("Attempting to disable autofocus using all available methods...")
                    
                    # Method 1: Try libcamera direct controls
                    try:
                        # Completely disable auto-focus
                        control_dict = {
                            "AfMode": 0,  # Manual focus (0=manual, 1=auto, 2=continuous)
                            "AfTrigger": 0,  # Don't trigger AF
                            "LensPosition": 0.5,  # Fixed middle position
                            "AfSpeed": 0,  # Slowest AF speed
                            "AwbEnable": True,  # Keep auto white balance 
                            "FrameDurationLimits": (33333, 33333),  # Lock to ~30fps
                            "NoiseReductionMode": 1  # Minimal noise reduction
                        }
                        
                        # Add rotation settings again to ensure they're applied
                        if CAMERA_ROTATION == 180:
                            try:
                                control_dict["ScalerFlipY"] = 1
                                control_dict["ScalerFlipX"] = 1
                            except:
                                control_dict["RotationDegrees"] = CAMERA_ROTATION
                                
                        self.camera.set_controls(control_dict)
                        print("Autofocus disabled using primary method")
                    except Exception as e1:
                        print(f"Primary autofocus disable method failed: {e1}")
                        
                        # Method 2: Try simpler set of controls
                        try:
                            self.camera.set_controls({
                                "AfMode": 0,  # Manual focus mode
                                "LensPosition": 0.5  # Fixed middle position
                            })
                            print("Autofocus disabled using simplified method")
                        except Exception as e2:
                            print(f"Simplified autofocus disable method failed: {e2}")
                            
                            # Method 3: Try legacy method with tuples
                            try:
                                # Some picamera2 versions use different format
                                self.camera.set_controls({"AfMode": (0,)})
                                self.camera.set_controls({"LensPosition": (0.5,)})
                                print("Autofocus disabled using legacy tuple method")
                            except Exception as e3:
                                print(f"Legacy autofocus disable method failed: {e3}")
                                
                                # Method 4: Try libcamera-still command first to set state persistently
                                try:
                                    import subprocess
                                    subprocess.run(["libcamera-still", "--autofocus", "off"], 
                                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=2)
                                    print("Disabled autofocus using external libcamera-still command")
                                except Exception as e4:
                                    print(f"External autofocus disable failed: {e4}")
                    
                    # Verify if any method worked by checking the control value
                    try:
                        controls = self.camera.camera_controls
                        if "AfMode" in controls:
                            af_mode = controls["AfMode"]
                            print(f"Current AfMode: {af_mode} (0=manual, 1=auto, 2=continuous)")
                            if af_mode == 0:
                                print("Verified autofocus is disabled!")
                            else:
                                print("Warning: Autofocus still appears to be enabled!")
                    except Exception as e:
                        print(f"Could not verify autofocus state: {e}")
                
                except Exception as e:
                    print(f"Could not disable autofocus: {e}, camera may jitter")
                    # Fall back to minimum controls if disabling autofocus fails
                    control_dict = {
                        "AwbEnable": True,  # Auto white balance as boolean
                        "FrameDurationLimits": (33333, 33333),  # Lock to ~30fps
                        "NoiseReductionMode": 1  # Minimal noise reduction (faster)
                    }
                    
                    # Add rotation settings
                    if CAMERA_ROTATION == 180:
                        try:
                            control_dict["ScalerFlipY"] = 1
                            control_dict["ScalerFlipX"] = 1
                        except:
                            control_dict["RotationDegrees"] = CAMERA_ROTATION
                    
                    self.camera.set_controls(control_dict)
            except Exception as e:
                print(f"Error with preferred configuration: {e}, trying fallback")
                # If that fails, try an even simpler configuration
                try:
                    fallback_config = self.camera.create_still_configuration()
                    self.camera.configure(fallback_config)
                    # Still try to apply rotation
                    try:
                        if CAMERA_ROTATION == 180:
                            try:
                                self.camera.set_controls({
                                    "ScalerFlipY": 1,
                                    "ScalerFlipX": 1
                                })
                            except:
                                self.camera.set_controls({"RotationDegrees": CAMERA_ROTATION})
                    except:
                        print("Could not apply rotation in fallback mode")
                except Exception as e2:
                    print(f"Error with fallback configuration: {e2}, using default")
                    # Just use whatever default we can get
                    
            print(f"Starting camera with {CAMERA_WIDTH}x{CAMERA_HEIGHT} resolution and {CAMERA_ROTATION}° rotation")
            self.camera.start()
            
            # Wait a moment for camera to initialize properly
            time.sleep(0.5)
            
            # Test capture to ensure camera is working
            test_frame = self.camera.capture_array()
            if test_frame is None or test_frame.size == 0:
                raise Exception("Camera returned empty frame during test")
            
            # Log frame info for debugging
            print(f"Test frame shape: {test_frame.shape}, dtype: {test_frame.dtype}")
            
            print("Camera initialized successfully")
            return True
            
        except Exception as e:
            print(f"Error setting up camera: {e}")
            if self.camera:
                try:
                    self.camera.close()
                except:
                    pass
                self.camera = None
            return False
    
    def capture_image(self, file_suffix=""):
        """Capture an image and save it to disk"""
        if not self.initialized or not self.camera:
            print("Camera not initialized. Cannot capture image.")
            return None
        
        try:
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_path = os.path.join(IMAGES_FOLDER, f"image_{timestamp}{file_suffix}.jpg")
            
            # Capture high-resolution image
            print(f"Capturing {CAMERA_WIDTH}x{CAMERA_HEIGHT} image to {image_path}")
            
            # For higher quality image, use a dedicated capture configuration
            try:
                # Import libcamera Transform if available
                try:
                    from libcamera import Transform
                    transform_available = True
                except ImportError:
                    transform_available = False
                
                # Set the appropriate transform based on rotation
                if transform_available and CAMERA_ROTATION == 180:
                    # If rotation is 180, both flip horizontal and vertical
                    transform = Transform(hflip=1, vflip=1)
                    capture_config = self.camera.create_still_configuration(
                        main={"size": (CAMERA_WIDTH, CAMERA_HEIGHT), "format": "BGR888"},
                        transform=transform
                    )
                    self.camera.switch_mode_and_capture_file(capture_config, image_path, format='jpg', quality=95)
                else:
                    # Fallback to standard capture with rotation handled using set_controls
                    # Keep any existing rotation setting
                    if hasattr(self.camera, 'controls') and "RotationDegrees" in self.camera.controls:
                        current_rotation = self.camera.controls["RotationDegrees"]
                        print(f"Using current rotation of {current_rotation}°")
                    
                    # For 180-degree rotation, we'll manually perform vflip and hflip when capturing
                    # First save the current configuration
                    current_config = self.camera.camera_config
                    
                    # Create still configuration
                    still_config = self.camera.create_still_configuration(
                        main={"size": (CAMERA_WIDTH, CAMERA_HEIGHT), "format": "BGR888"}
                    )
                    
                    # Switch to still mode, capture with flips if needed, and switch back
                    self.camera.switch_mode(still_config)
                    
                    # Apply appropriate flips for 180-degree rotation
                    if CAMERA_ROTATION == 180:
                        self.camera.set_controls({"ScalerFlipY": 1, "ScalerFlipX": 1})
                    
                    # Capture the image
                    self.camera.capture_file(image_path, format='jpg')
                    
                    # Restore original configuration and rotation
                    self.camera.switch_mode(current_config)
            except Exception as e:
                print(f"Error capturing with transform: {e}")
                # Fallback to simplest capture method
                try:
                    # For 180-degree rotation, we need to flip the image after capture
                    # First, capture the image
                    self.camera.capture_file(image_path)
                    
                    # If rotation is 180, read the image, flip it, and save it back
                    if CAMERA_ROTATION == 180:
                        # Use OpenCV to flip the image 180 degrees
                        img = cv2.imread(image_path)
                        if img is not None:
                            flipped = cv2.flip(img, -1)  # -1 flips both horizontally and vertically (180 rotation)
                            cv2.imwrite(image_path, flipped)
                            print("Applied 180° rotation using OpenCV post-processing")
                except Exception as e2:
                    print(f"Error in fallback image capture: {e2}")
                    return None
            
            print(f"Image captured and saved to {image_path}")
            
            return image_path
            
        except Exception as e:
            print(f"Error capturing image: {e}")
            return None

    def record_video(self, duration=VIDEO_DURATION, timestamp=None):
        """Record a video using external tools for reliability"""
        # Try to acquire the recording lock
        acquired = False
        try:
            # Use a timeout to prevent deadlocks (5 seconds)
            acquired = self.recording_lock.acquire(blocking=True, timeout=5)
            if not acquired:
                print("Already recording, skipping this recording request")
                return None
                
            print(f"Starting video recording at {CAMERA_WIDTH}x{CAMERA_HEIGHT} resolution...")
            self.recording = True
            
            # Generate timestamp if not provided
            if timestamp is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
            # Generate video filename
            mp4_path = os.path.join(VIDEOS_FOLDER, f"video_{timestamp}.mp4")
            
            # Close the camera to free up resources
            if self.camera:
                self.camera.close()
                self.camera = None
                time.sleep(1)
            
            # Record video using fallback shell script
            try:
                import subprocess
                import stat
                
                # Make sure the shell script exists and is executable
                script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "capture_video.sh")
                if not os.path.exists(script_path):
                    print(f"Error: Script not found: {script_path}")
                    raise Exception("Capture script not found")
                
                # Set execute permission if needed
                current_mode = os.stat(script_path).st_mode
                if not (current_mode & stat.S_IXUSR):
                    print("Setting execute permission on script")
                    os.chmod(script_path, current_mode | stat.S_IXUSR)
                
                # Use the same resolution as images for consistency
                # This ensures the field of view matches between images and videos
                video_width = CAMERA_WIDTH
                video_height = CAMERA_HEIGHT
                    
                print(f"Using video resolution {video_width}x{video_height} with rotation {CAMERA_ROTATION}°")
                
                # Run the capture script, which will:
                # 1. Capture a video using libcamera-vid with appropriate flags
                # 2. Convert it to an MP4 using ffmpeg
                print(f"Capturing video using script for {duration} seconds...")
                framerate = 5  # 5 fps is a good balance for motion vs file size
                cmd = [script_path, mp4_path, str(duration), str(framerate), 
                       str(video_width), str(video_height), str(CAMERA_ROTATION)]
                subprocess.run(cmd, check=True)
                
                # Check if the file was created
                if not os.path.exists(mp4_path) or os.path.getsize(mp4_path) < 1000:
                    print(f"Error: Video file is missing or too small: {mp4_path}")
                    raise Exception("Video recording failed: Empty or missing file")
                
                print(f"Successfully recorded video to {mp4_path}")
                return mp4_path
                
            except Exception as e:
                print(f"Error recording video using fallback script: {e}")
                traceback.print_exc()
                return None
                
        except Exception as e:
            print(f"Error in record_video: {e}")
            traceback.print_exc()
            return None
        finally:
            # Reinitialize camera for motion detection
            self.setup_camera()
            
            # Ensure recording flag is reset
            self.recording = False
            
            # Release recording lock
            if acquired:
                try:
                    self.recording_lock.release()
                except Exception as e:
                    print(f"Error releasing recording lock: {e}")
                    # Force reset the lock if it can't be released
                    try:
                        self.recording_lock._owner = None
                        self.recording_lock = threading.Lock()
                    except:
                        pass
    
    def filter_small_blobs(self, binary_image):
        """Filter out small blobs of motion that might be leaves or noise"""
        try:
            from scipy import ndimage
            labeled, num_features = ndimage.label(binary_image)
            
            # Calculate sizes of all labeled regions
            sizes = ndimage.sum(binary_image, labeled, range(1, num_features + 1))
            
            # Create a mask for regions that are within our target size range
            # For in-feeder camera, we expect larger blobs (birds) and filter out smaller ones
            mask = np.zeros_like(binary_image, dtype=bool)
            for i, size in enumerate(sizes):
                if MOTION_MIN_BLOB_SIZE <= size <= MOTION_MAX_BLOB_SIZE:
                    mask[labeled == i + 1] = True
            
            # Apply the mask to the original binary image
            filtered = binary_image.copy()
            filtered[~mask] = 0
            
            return filtered
        except (ImportError, NameError):
            # Simple filtering for when scipy is not available
            # For in-feeder camera, we're looking for larger contiguous regions
            eroded = binary_image.copy()
            dilated = np.zeros_like(binary_image)
            
            # Remove very small areas
            for i in range(1, binary_image.shape[0]-1):
                for j in range(1, binary_image.shape[1]-1):
                    if np.sum(binary_image[i-1:i+2, j-1:j+2]) < 6:  # Require more filled pixels
                        eroded[i, j] = 0
            
            # Connect and expand larger regions
            for i in range(1, eroded.shape[0]-1):
                for j in range(1, eroded.shape[1]-1):
                    if np.sum(eroded[i-1:i+2, j-1:j+2]) > 1:  # Less strict to connect regions
                        dilated[i, j] = 1
                        
            return dilated
    
    def detect_motion(self):
        """Capture a frame and compare with previous frame to detect motion using contour-based algorithm
        from the legacy implementation which doesn't suffer from false positives due to autofocus."""
        if not self.initialized or not self.camera:
            return False
        
        try:
            # Skip if recording is in progress
            if self.recording:
                return False
                
            # Capture current frame
            current_frame = self.camera.capture_array()
            current_time = time.time()
            
            # Simplified log to reduce console spam
            print("\n==== MOTION DETECTION ====")
            
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
            # This uses contour-based detection instead of pixel differences
            
            # Apply Gaussian blur to reduce noise - increased kernel size for stronger smoothing
            blurred_curr = cv2.GaussianBlur(curr_roi, (25, 25), 0)  # Was (21, 21)
            blurred_prev = cv2.GaussianBlur(prev_roi, (25, 25), 0)  # Was (21, 21)
            
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
            
            # Debug information
            if contours:
                print(f"Contours detected: {len(contours)}")
                print(f"Largest contour area: {largest_contour_area} (min threshold: {MIN_MOVEMENT_SIZE})")
                print(f"Significant contours: {significant_contours}")
            
            # JITTER DETECTION - Check for autofocus-like oscillation patterns
            is_jitter = False
            if len(self.contour_areas_history) >= 3 and significant_contours > 0:
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
                        if avg_diff < 0.4:  # Small regular changes in area
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
                        if 1 < avg_movement < 10:  # Small pixel movements typical of jitter
                            position_oscillation = True
                            print("Detected position oscillation pattern typical of focus jitter")
                
                # Combine criteria to identify jitter
                is_jitter = area_oscillation or position_oscillation
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
                    # Initialize motion scores if they don't exist
                    self.motion_scores = []
            else:
                # Reset motion scores when no significant motion or when jitter is detected
                if hasattr(self, 'motion_scores'):
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
    
    def update_sightings_json(self, image_path, video_path, timestamp=None):
        """Update the sightings JSON file with new detection information"""
        try:
            if timestamp is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
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
            
            # Copy image and video to web public folder if needed
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
        """Continuously monitor for motion, record video when detected"""
        if not self.initialized or not self.camera:
            print("Camera not initialized. Cannot monitor for motion.")
            return
        
        self.running = True
        last_motion_time = 0
        frame_count = 0
        debug_frame_counter = 0
        consecutive_motion_frames = 0
        
        try:
            # Set appropriate detection mode description
            if MOVING_OBJECT_DETECTION:
                mode_desc = "MOVING OBJECT DETECTION"
            elif STATUE_DETECTION_MODE:
                mode_desc = "STATUE DETECTION"
            else:
                mode_desc = "BIRD DETECTION"
                
            print(f"\n*** Starting motion monitoring with {mode_desc} mode ***")
            print(f"Motion threshold: {MOTION_THRESHOLD}")
            print(f"Required consecutive frames: {CONSECUTIVE_FRAMES}")
            print(f"Checking every {FRAME_INTERVAL}s, cooldown: {COOLDOWN_PERIOD}s")
            print(f"Video duration: {VIDEO_DURATION}s when motion detected")
            if ROI_ENABLED:
                print(f"ROI enabled: using {ROI_WIDTH*100:.0f}% x {ROI_HEIGHT*100:.0f}% of frame")
            print("\nStarting background learning...")
            
            # Learning phase
            # First frame
            first_frame = self.camera.capture_array()
            self.background_model = np.mean(first_frame, axis=2).astype(np.float32)
            
            if ROI_ENABLED:
                h, w = self.background_model.shape
                roi_x_start = int(ROI_X * w)
                roi_y_start = int(ROI_Y * h)
                roi_width = int(ROI_WIDTH * w)
                roi_height = int(ROI_HEIGHT * h)
                self.background_model = self.background_model[roi_y_start:roi_y_start+roi_height, 
                                                           roi_x_start:roi_x_start+roi_width]
            
            # Gather more frames for better initial background
            # For hand-moved objects, need a solid baseline
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
            
            print(f"\nBackground learned. Monitoring for motion...")
            print("Ready to detect! Move the statue to simulate bird activity.")
            
            # Initialize motion tracking
            self.prev_motion_centroids = []
            
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
            
            while self.running:
                frame_count += 1
                debug_frame_counter += 1
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
                    self.prev_motion_centroids = []  # Reset motion tracking
                    
                # Check for motion
                motion_detected = self.detect_motion()
                
                # Calculate how long the processing took - only log if taking too long
                process_time = time.time() - process_start_time
                if process_time > MAX_FRAME_PROCESSING_TIME:
                    print(f"Warning: Frame processing took {process_time:.2f}s (exceeds target {MAX_FRAME_PROCESSING_TIME}s)")
                
                # Update timing
                last_frame_time = time.time()
                
                if motion_detected:
                    # Track consecutive motion frames
                    consecutive_motion_frames += 1
                    print(f"Motion detected in {consecutive_motion_frames} consecutive frames (need {CONSISTENT_MOTION_FRAMES})")
                    
                    # Only trigger if we're past cooldown period AND have consistent motion
                    if (time.time() - last_motion_time > COOLDOWN_PERIOD and 
                        consecutive_motion_frames >= CONSISTENT_MOTION_FRAMES):
                        if MOVING_OBJECT_DETECTION:
                            object_type = "MOVING OBJECT"
                        elif STATUE_DETECTION_MODE:
                            object_type = "STATUE"
                        else:
                            object_type = "BIRD"
                            
                        print(f"\n*** {object_type} DETECTED! CAPTURING IMAGE AND VIDEO... ***\n")
                        
                        # First capture a still image
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        suffix = "_movement" if MOVING_OBJECT_DETECTION else "_statue" if STATUE_DETECTION_MODE else "_bird"
                        motion_image = self.capture_image(suffix)
                        
                        # Then record video
                        video_path = self.record_video(VIDEO_DURATION, timestamp)
                        if video_path:
                            print(f"Video saved to: {video_path}")
                            last_motion_time = time.time()
                            
                            # Update sightings JSON file with new detection
                            self.update_sightings_json(motion_image, video_path, timestamp)
                        
                        # Re-initialize after recording
                        print("Re-initializing camera...")
                        if not self.camera or not self.initialized:
                            self.setup_camera()
                        
                        # Reset background and motion history
                        self.background_model = None
                        self.motion_history = []
                        self.frame_timestamps = []
                        self.motion_scores = []
                        self.prev_motion_centroids = []  # Reset motion tracking
                        
                        # Re-learn background
                        print("Re-learning background...")
                        first_frame = self.camera.capture_array()
                        first_gray = np.mean(first_frame, axis=2).astype(np.float32)
                        
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
                        print("Ready to detect more movement!")
                        last_frame_time = time.time()  # Reset timing after recording
                        
                        # Reset consecutive motion count
                        consecutive_motion_frames = 0
                else:
                    # Reset consecutive motion count on any non-motion frame
                    consecutive_motion_frames = 0
        
        except Exception as e:
            print(f"Error in motion monitoring: {e}")
            traceback.print_exc()
        finally:
            print("Motion monitoring stopped")
    
    def close(self):
        """Close the camera resources"""
        self.running = False
        if self.camera:
            try:
                self.camera.close()
                print("Camera closed")
            except Exception as e:
                print(f"Error closing camera: {e}")
            finally:
                self.camera = None
                self.initialized = False

def check_system_requirements():
    """Check if system requirements are met"""
    requirements_met = True
    global FFMPEG_AVAILABLE
    
    # Check for ffmpeg
    try:
        import subprocess
        result = subprocess.run(['which', 'ffmpeg'], capture_output=True, text=True)
        if result.returncode != 0:
            print("WARNING: ffmpeg not found. Video conversion will not work.")
            requirements_met = False
        else:
            print(f"Found ffmpeg at: {result.stdout.strip()}")
            FFMPEG_AVAILABLE = True
            
        # Check for raspistill (used in fallback script)
        result = subprocess.run(['which', 'raspistill'], capture_output=True, text=True)
        if result.returncode != 0:
            print("WARNING: raspistill not found. Fallback video recording will not work.")
            requirements_met = False
        else:
            print(f"Found raspistill at: {result.stdout.strip()} (will use for fallback video recording)")
            
    except Exception as e:
        print(f"Error checking for system tools: {e}")
        requirements_met = False
    
    # Check for picamera2
    if not PICAMERA2_AVAILABLE:
        print("WARNING: picamera2 not available. Camera functionality will be limited.")
        requirements_met = False
    
    return requirements_met

def main():
    """Main function to test the camera service"""
    print("Starting camera service...")
    
    # Check system requirements
    check_system_requirements()
    
    camera_service = CameraService()
    
    if camera_service.initialized:
        # Capture an initial image
        image_path = camera_service.capture_image()
        if image_path:
            print(f"Successfully captured initial image: {image_path}")
        
        # Generate sample JSON with existing files if it doesn't exist yet
        if not os.path.exists(SIGHTINGS_JSON):
            print("Generating sample sightings JSON from existing files...")
            sample_sightings = []
            
            # Scan existing images
            images = [f for f in os.listdir(IMAGES_FOLDER) if f.startswith("image_")]
            for img in images:
                try:
                    # Extract timestamp from filename
                    timestamp = img.split("_")[1].split(".")[0]
                    
                    # Find matching video, if any
                    matching_video = None
                    videos = [v for v in os.listdir(VIDEOS_FOLDER) if v.startswith(f"video_{timestamp}")]
                    if videos:
                        matching_video = videos[0]
                    
                    # Create sighting entry
                    formatted_timestamp = datetime.strptime(timestamp, "%Y%m%d_%H%M%S").strftime("%Y-%m-%d %H:%M:%S")
                    sighting = {
                        "id": timestamp,
                        "timestamp": formatted_timestamp,
                        "date": datetime.strptime(timestamp, "%Y%m%d_%H%M%S").strftime("%Y-%m-%d"),
                        "time": datetime.strptime(timestamp, "%Y%m%d_%H%M%S").strftime("%H:%M:%S"),
                        "image": img,
                        "video": matching_video,
                        "type": "bird"  # Default to bird for existing files
                    }
                    sample_sightings.append(sighting)
                except Exception as e:
                    print(f"Error processing file {img}: {e}")
            
            # Sort sightings by timestamp (newest first)
            sample_sightings.sort(key=lambda x: x["timestamp"], reverse=True)
            
            # Write to JSON file
            with open(SIGHTINGS_JSON, 'w') as f:
                json.dump(sample_sightings, f, indent=2)
            
            # Copy files to web public directories
            web_images_folder = os.path.join(PUBLIC_FOLDER, "images")
            web_videos_folder = os.path.join(PUBLIC_FOLDER, "videos")
            
            os.makedirs(web_images_folder, exist_ok=True)
            os.makedirs(web_videos_folder, exist_ok=True)
            
            for sighting in sample_sightings:
                # Copy image
                if sighting["image"]:
                    src_img = os.path.join(IMAGES_FOLDER, sighting["image"])
                    dst_img = os.path.join(web_images_folder, sighting["image"])
                    if os.path.exists(src_img) and not os.path.exists(dst_img):
                        shutil.copy2(src_img, dst_img)
                
                # Copy video
                if sighting["video"]:
                    src_vid = os.path.join(VIDEOS_FOLDER, sighting["video"])
                    dst_vid = os.path.join(web_videos_folder, sighting["video"])
                    if os.path.exists(src_vid) and not os.path.exists(dst_vid):
                        shutil.copy2(src_vid, dst_vid)
            
            print(f"Created sample sightings JSON with {len(sample_sightings)} entries")
        
        # Monitor for motion
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