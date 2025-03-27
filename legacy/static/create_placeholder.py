#!/usr/bin/env python3
import cv2
import numpy as np
import os

# Create a directory for static files
os.makedirs('static', exist_ok=True)

# Create a black image
img = np.zeros((480, 640, 3), dtype=np.uint8)

# Add a dark gray background rectangle
cv2.rectangle(img, (0, 0), (640, 480), (50, 50, 50), -1)

# Add a camera icon (simplified)
# Camera body
cv2.rectangle(img, (240, 180), (400, 300), (100, 100, 100), -1)
cv2.rectangle(img, (240, 180), (400, 300), (150, 150, 150), 2)

# Camera lens
cv2.circle(img, (320, 240), 40, (80, 80, 80), -1)
cv2.circle(img, (320, 240), 40, (150, 150, 150), 2)
cv2.circle(img, (320, 240), 30, (60, 60, 60), -1)
cv2.circle(img, (320, 240), 20, (40, 40, 40), -1)

# Camera flash
cv2.rectangle(img, (370, 190), (390, 210), (200, 200, 200), -1)

# Add text
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img, "Camera Not Available", (150, 350), font, 1, (255, 255, 255), 2)
cv2.putText(img, "Please check connection", (170, 380), font, 0.7, (200, 200, 200), 1)

# Save the image
cv2.imwrite('static/no_camera.jpg', img)
print("Placeholder image created at static/no_camera.jpg") 