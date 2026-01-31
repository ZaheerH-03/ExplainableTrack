"""
ByteTrack Live Tracking Script

This script demonstrates real-time object tracking using YOLO for detection and ByteTrack for tracking.
It captures video from the webcam, tracks objects, calculates FPS, and displays the annotated output.

Prerequisites:
- 'weights/yolov8m.pt' model file.
- 'byte_track.yaml' configuration file.
"""

import cv2
import time
from ultralytics import YOLO

# ----------------------------
# Load YOLO model
# ----------------------------
# Load the pretrained YOLOv8 model
model = YOLO("weights/yolov8m.pt")  # <-- your trained model

# ----------------------------
# Open webcam (0 = default)
# ----------------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam")

print("[INFO] Press 'q' to quit")

prev_time = time.time()

# Main Loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ----------------------------
    # YOLO + ByteTrack Inference
    # ----------------------------
    # Run tracking on the current frame
    results = model.track(
        frame,
        tracker="byte_track.yaml", # Configuration for ByteTrack
        persist=True,   # Maintain track history
        device="cuda",  # Use GPU
        conf=0.25,      # Detection confidence threshold
        iou=0.5         # NMS IoU threshold
    )

    # ----------------------------
    # FPS calculation
    # ----------------------------
    curr_time = time.time()
    fps = 1.0 / (curr_time - prev_time)
    prev_time = curr_time

    # ----------------------------
    # Draw results + FPS
    # ----------------------------
    # Generate frame with bounding boxes and labels
    annotated_frame = results[0].plot()

    # Overlay FPS on the frame
    cv2.putText(
        annotated_frame,
        f"FPS: {fps:.2f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2
    )

    # Show the frame
    cv2.imshow("YOLO + ByteTrack (LIVE)", annotated_frame)

    # Exit condition
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ----------------------------
# Cleanup
# ----------------------------
cap.release()
cv2.destroyAllWindows()