"""
FPS Tracker Utility

This script measures the actual frame rate (FPS) capability of the connected webcam.
It processes a set number of frames and calculates the average FPS based on the
total elapsed time.
"""

import cv2
import time

# Initialize webcam
cap = cv2.VideoCapture(0)

num_frames = 100 # Number of frames to process for the test
start = time.time()

print(f"Capturing {num_frames} frames to calculate FPS...")

# Loop to read frames
for _ in range(num_frames):
    ret, frame = cap.read()
    if not ret:
        break

end = time.time()
cap.release()

# Calculate and print FPS
fps = num_frames / (end - start)
print(f"Actual Webcam FPS: {fps:.2f}")


