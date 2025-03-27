# Birdophile - Smart Bird Feeder Web Application

A web-based application for a Raspberry Pi 5 smart bird feeder that detects birds using picamera2, recognizes bird species using an on-device model, and displays the results on a modern web interface styled with Tailwind CSS.

## System Overview

This system consists of several modules:

1. **Camera and Motion Detection**: Uses picamera2 for live video streaming and built-in motion detection to capture images and videos when birds are detected.
2. **Bird Recognition**: Uses TensorFlow Lite to recognize bird species from captured images.
3. **Web Interface**: A Flask application that displays bird sightings with images and videos, styled with Tailwind CSS.

## Requirements

- Raspberry Pi 5 with Raspberry Pi Camera Module 3
- Python 3.7+
- picamera2 library (installed via apt)
- Internet connection for initial model download

## Installation

1. Clone this repository to your Raspberry Pi:

```bash
git clone https://github.com/yourusername/birdophile.git
cd birdophile
```

2. Run the installation script to set up picamera2 and other dependencies:

```bash
./install_picamera.sh
```

This script will:
- Install picamera2 and required system dependencies
- Create a Python virtual environment
- Install all required Python packages

3. Create necessary directories (if not already created by the installation script):

```bash
mkdir -p /home/timotej/birdshere/uploads
mkdir -p /home/timotej/birdshere/model_cache
```

4. Install the systemd service:

```bash
sudo cp birdophile.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable birdophile
sudo systemctl start birdophile
```

## Configuration

### Camera Configuration

The camera settings can be adjusted in the `camera_handler.py` file:

- `DETECTION_INTERVAL`: Time between detection attempts (seconds)
- `MOTION_THRESHOLD`: Minimum contour area to trigger detection
- `RECORD_SECONDS`: Duration to record video after detection

### Bird Recognition

The bird recognition module uses TensorFlow Lite with a pre-trained model for bird species recognition. The model will be downloaded automatically on first run.

## Usage

1. Access the web interface by navigating to `http://<raspberry-pi-ip>:5000` in your web browser.

2. The web interface will display:
   - Live camera feed from the Raspberry Pi camera
   - Camera controls to start/stop the camera and manually capture images
   - All bird sightings with images, videos, timestamps, and recognized species

3. The interface automatically refreshes every 30 seconds to show new sightings.

4. Camera features:
   - **Live Stream**: View the camera feed in real-time
   - **Automatic Detection**: The system automatically detects motion and captures images/videos of birds
   - **Manual Capture**: Capture an image manually using the "Capture Now" button
   - **Camera Control**: Start or stop the camera using the toggle button

## Troubleshooting

- If the camera is not working, check if picamera2 is installed correctly with `python3 -c "from picamera2 import Picamera2; print('Picamera2 installed')"`.
- If the bird recognition model is not working, make sure TensorFlow Lite is installed correctly.
- If the web interface is not accessible, check if the Flask application is running with `sudo systemctl status birdophile`.
- Check the system logs with `journalctl -u birdophile` for any errors.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 