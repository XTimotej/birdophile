#!/usr/bin/env python3
import sys
import os
import numpy as np
from PIL import Image
import requests
import json
from pathlib import Path

# Configuration
DATA_DIR = "/home/timotej/birdshere"
CACHE_DIR = os.path.join(DATA_DIR, "model_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# Fallback to iNaturalist API if specified
USE_INAT_API = False
INAT_API = "https://api.inaturalist.org/v1/identifications"
API_TOKEN = "YOUR_INATURALIST_API_TOKEN"  # Replace after signup

# Try to import TensorFlow Lite for on-device inference
try:
    import tflite_runtime.interpreter as tflite
    TFLITE_AVAILABLE = True
except ImportError:
    TFLITE_AVAILABLE = False
    print("TensorFlow Lite not available. Will use fallback methods.")

# Model paths - these will be downloaded on first run
MODEL_URL = "https://storage.googleapis.com/download.tensorflow.org/models/tflite/task_library/image_classification/android/mobilenet_v1_1.0_224_quantized_1_metadata_1.tflite"
MODEL_PATH = os.path.join(CACHE_DIR, "bird_model.tflite")
LABELS_URL = "https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/lite/java/demo/app/src/main/assets/labels_mobilenet_quant_v1_224.txt"
LABELS_PATH = os.path.join(CACHE_DIR, "bird_labels.txt")

def download_file(url, local_path):
    """Download a file from a URL to a local path"""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False

def load_labels():
    """Load bird species labels"""
    if not os.path.exists(LABELS_PATH):
        print(f"Downloading bird labels to {LABELS_PATH}...")
        if not download_file(LABELS_URL, LABELS_PATH):
            return []
    
    with open(LABELS_PATH, 'r') as f:
        lines = f.readlines()
    
    return [line.strip() for line in lines]

def load_model():
    """Load the TFLite model"""
    if not os.path.exists(MODEL_PATH):
        print(f"Downloading bird model to {MODEL_PATH}...")
        if not download_file(MODEL_URL, MODEL_PATH):
            return None
    
    try:
        interpreter = tflite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def preprocess_image(image_path, input_size=(224, 224)):
    """Preprocess image for model input"""
    try:
        image = Image.open(image_path).convert('RGB')
        image = image.resize(input_size)
        image_array = np.array(image, dtype=np.uint8)
        return np.expand_dims(image_array, axis=0)
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

def recognize_with_tflite(image_path):
    """Recognize bird species using TensorFlow Lite model"""
    # Load model and labels
    interpreter = load_model()
    labels = load_labels()
    
    if not interpreter or not labels:
        return "Unknown (Model loading failed)"
    
    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Preprocess image
    input_data = preprocess_image(image_path)
    if input_data is None:
        return "Unknown (Image preprocessing failed)"
    
    # Run inference
    try:
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        # Get top prediction
        top_index = np.argmax(output_data[0])
        confidence = output_data[0][top_index] / 255.0  # Quantized model outputs 0-255
        
        if confidence < 0.3:  # Confidence threshold
            return "Unknown (Low confidence)"
        
        return labels[top_index]
    except Exception as e:
        print(f"Error during inference: {e}")
        return "Unknown (Inference error)"

def recognize_with_inat(image_path):
    """Recognize bird using iNaturalist API"""
    try:
        with open(image_path, "rb") as f:
            response = requests.post(
                INAT_API,
                files={"image": f},
                headers={"Authorization": f"Bearer {API_TOKEN}"}
            )
        
        if response.status_code == 200 and response.json().get("results"):
            return response.json()["results"][0]["taxon"]["name"]
    except Exception as e:
        print(f"iNaturalist API error: {e}")
    
    return "Unknown (API failed)"

def recognize_bird(image_path):
    """
    Recognize bird species from an image
    
    Args:
        image_path (str): Path to the bird image
    
    Returns:
        str: Bird species name or "Unknown"
    """
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return "Unknown (Image not found)"
    
    # Try TensorFlow Lite first if available
    if TFLITE_AVAILABLE:
        try:
            species = recognize_with_tflite(image_path)
            if "Unknown" not in species:
                return species
        except Exception as e:
            print(f"TensorFlow Lite error: {e}")
    
    # Fall back to iNaturalist API if enabled
    if USE_INAT_API:
        return recognize_with_inat(image_path)
    
    # Simple fallback - just return a generic bird name
    return "Bird (Species unknown)"

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python bird_recognition.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    species = recognize_bird(image_path)
    print(f"Recognized bird species: {species}")
    
    # Save result to a text file if requested
    if len(sys.argv) > 2 and sys.argv[2] == "--save":
        with open(f"{image_path}.txt", "w") as f:
            f.write(species) 