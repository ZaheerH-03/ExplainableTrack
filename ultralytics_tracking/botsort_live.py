"""
BoT-SORT Live Tracking Script

This script performs real-time object tracking using the YOLOv8 model and BoT-SORT algorithm.
It captures video from the default webcam, runs tracking, and displays the annotated frames.

Prerequisites:
- 'weights/yolov8m.pt' model file must exist.
- 'bot_sort.yaml' configuration file must be available.
"""

import cv2
from ultralytics import YOLO

# Initialize YOLO model with pretrained weights
# 'yolov8m.pt' should be in the 'weights' directory
model = YOLO("weights/yolov8m.pt")

# Open connection to the default webcam (index 0)
cap = cv2.VideoCapture(0)

# Main Processing Loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform Object Tracking
    # persist=True ensures tracks are maintained across frames
    results = model.track(
        frame,
        tracker="bot_sort.yaml", # Configuration for BoT-SORT
        persist=True,
        conf=0.25,      # Confidence threshold
        device="cuda"   # Use GPU acceleration
    )

    # Visualize the results
    annotated_frame = results[0].plot()

    # Display the frame
    cv2.imshow("YOLOv11 + BoT-SORT (LIVE)", annotated_frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup resources
cap.release()
cv2.destroyAllWindows()


