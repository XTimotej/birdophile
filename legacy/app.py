import os
import json
import atexit
import signal
import sys
from datetime import datetime
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename

# Import our camera modules
from camera_server import camera_bp, init_camera, shutdown_camera

# Configuration
DATA_DIR = "/home/timotej/birdshere"
UPLOAD_FOLDER = os.path.join(DATA_DIR, "uploads")
DATABASE_FILE = os.path.join(DATA_DIR, "bird_sightings.json")

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max upload
app.config['APPLICATION_NAME'] = "Birdophile"

# Register the camera blueprint
app.register_blueprint(camera_bp, url_prefix='/camera')

# Initialize database if it doesn't exist
if not os.path.exists(DATABASE_FILE):
    try:
        os.makedirs(os.path.dirname(DATABASE_FILE), exist_ok=True)
        with open(DATABASE_FILE, 'w') as f:
            json.dump([], f)
    except Exception as e:
        print(f"Error initializing database: {e}")

def get_sightings():
    """Load bird sightings from the JSON database"""
    try:
        if os.path.exists(DATABASE_FILE):
            with open(DATABASE_FILE, 'r') as f:
                sightings = json.load(f)
                # Sort sightings by timestamp (newest first)
                sightings.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
                return sightings
        else:
            print(f"Database file {DATABASE_FILE} does not exist, creating empty database")
            with open(DATABASE_FILE, 'w') as f:
                json.dump([], f)
            return []
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Error loading database: {e}")
        return []

def save_sightings(sightings):
    """Save bird sightings to the JSON database"""
    try:
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(DATABASE_FILE), exist_ok=True)
        
        with open(DATABASE_FILE, 'w') as f:
            json.dump(sightings, f, indent=2)
    except Exception as e:
        print(f"Error saving to database: {e}")

@app.route('/')
def index():
    """Render the main page with bird sightings"""
    sightings = get_sightings()
    return render_template('index.html', sightings=sightings, app_name=app.config['APPLICATION_NAME'])

@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    """Serve uploaded files"""
    try:
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
    except Exception as e:
        print(f"Error serving file {filename}: {e}")
        return "File not found", 404

@app.route('/api/event', methods=['POST'])
def receive_event():
    """Endpoint to receive events from the motion detection script"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({'error': 'Empty filename'}), 400
        
        # Get bird species from request or set as unknown
        species = request.form.get('species', 'Unknown')
        
        # Generate unique filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{secure_filename(image_file.filename)}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Ensure upload directory exists
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        
        # Save the image
        image_file.save(filepath)
        
        # Create sighting record
        sighting = {
            'id': timestamp,
            'timestamp': datetime.now().isoformat(),
            'species': species,
            'image_path': f"uploads/{filename}",
            'video_path': None  # Will be updated if video is provided
        }
        
        # Save video if provided
        if 'video' in request.files:
            video_file = request.files['video']
            if video_file.filename != '':
                video_filename = f"{timestamp}_{secure_filename(video_file.filename)}"
                video_filepath = os.path.join(app.config['UPLOAD_FOLDER'], video_filename)
                video_file.save(video_filepath)
                sighting['video_path'] = f"uploads/{video_filename}"
        
        # Add to database
        sightings = get_sightings()
        
        # Check for duplicates
        for existing in sightings:
            if existing.get('id') == sighting['id']:
                print(f"Sighting with ID {sighting['id']} already exists, not adding duplicate")
                return jsonify({'success': True, 'sighting': existing}), 200
        
        sightings.append(sighting)
        save_sightings(sightings)
        
        return jsonify({'success': True, 'sighting': sighting}), 201
    except Exception as e:
        print(f"Error processing event: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/sightings', methods=['GET'])
def get_all_sightings():
    """API endpoint to get all sightings"""
    try:
        sightings = get_sightings()
        print(f"Returning {len(sightings)} sightings from database")
        
        # Log the first few sightings for debugging
        if sightings:
            print(f"First sighting: {sightings[0]}")
            
        return jsonify(sightings)
    except Exception as e:
        print(f"Error retrieving sightings: {e}")
        return jsonify({'error': str(e)}), 500

def cleanup_resources():
    """Clean up resources when the application exits"""
    import threading
    import time
    
    print("Cleaning up resources...")
    
    # Use a timeout mechanism to prevent hanging
    cleanup_completed = False
    
    def do_cleanup_with_timeout():
        nonlocal cleanup_completed
        try:
            # Call shutdown_camera which now has its own timeout
            shutdown_camera()
        except Exception as e:
            print(f"Error during resource cleanup: {e}")
        finally:
            cleanup_completed = True
            print("Cleanup process completed")
    
    # Start cleanup in a separate thread
    cleanup_thread = threading.Thread(target=do_cleanup_with_timeout)
    cleanup_thread.daemon = True
    cleanup_thread.start()
    
    # Wait for cleanup with timeout
    timeout = 5  # 5 seconds timeout
    start_time = time.time()
    while not cleanup_completed and time.time() - start_time < timeout:
        time.sleep(0.1)
    
    if not cleanup_completed:
        print(f"Resource cleanup timed out after {timeout} seconds")
    
    print("Application shutdown complete")

# Register signal handlers for graceful shutdown
def signal_handler(sig, frame):
    print(f"Received signal {sig}, shutting down...")
    
    # Set a timeout for the entire shutdown process
    import threading
    import time
    import os
    
    # Start cleanup in a separate thread
    cleanup_thread = threading.Thread(target=cleanup_resources)
    cleanup_thread.daemon = True
    cleanup_thread.start()
    
    # Wait for cleanup with timeout
    timeout = 10  # 10 seconds total timeout for shutdown
    start_time = time.time()
    while cleanup_thread.is_alive() and time.time() - start_time < timeout:
        time.sleep(0.1)
    
    if cleanup_thread.is_alive():
        print(f"Shutdown process timed out after {timeout} seconds. Forcing exit.")
    
    # Force exit after timeout or when cleanup is done
    os._exit(0)  # Use os._exit to force exit without further cleanup

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Initialize camera system
with app.app_context():
    camera_initialized = init_camera()
    if not camera_initialized:
        print("Warning: Camera initialization failed. The application will run without camera functionality.")

# Register shutdown function
atexit.register(cleanup_resources)

if __name__ == '__main__':
    # Run in non-debug mode to avoid reloader issues with the camera
    app.run(host='0.0.0.0', port=5000, debug=False) 