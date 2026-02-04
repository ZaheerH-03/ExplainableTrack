"""
YOLOv11 Tracking Script

This script utilizes the YOLOv11 object detection model combined with the BoT-SORT tracker
to perform real-time multiple object tracking on video inputs.

Key Functions:
- Loads YOLOv11 model.
- Processes video frames.
- Detects objects and passes them to the tracker.
- Visualizes tracking output.
"""

import sys
from pathlib import Path
import cv2
import numpy as np
import torch
from ultralytics import YOLO

# --- Path Setup ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "BoT-SORT"))

from tracker.mc_bot_sort import BoTSORT
from tracker.tracking_utils.timer import Timer

# --- YOLOv11 Model Initialization ---
# Assuming weights are in the standard weights directory relative to project root
model_path = PROJECT_ROOT / "weights/yolov11_best.pt"
if not model_path.exists():
    # Fallback or auto-download by YOLO class if strictly needed, 
    # but let's point to where it should be.
    pass

yolo = YOLO(model_path, task="detect")
device = "cuda" if torch.cuda.is_available() else "cpu"
yolo.to(device)

class Opt:
    """
    Configuration class for BoT-SORT options.
    """
    pass

# --- Tracker Configuration ---
opt = Opt()
opt.name = "yolov11"
opt.ablation = False
opt.track_high_thresh = 0.5
opt.track_low_thresh = 0.1
opt.new_track_thresh = 0.6
opt.track_buffer = 3600
opt.match_thresh = 0.7
opt.aspect_ratio_thresh = 1.6
opt.min_box_area = 10
opt.mot20 = False

# ReID (Re-Identification) settings
opt.with_reid = True
# Clean paths using PROJECT_ROOT
opt.fast_reid_config = str(PROJECT_ROOT / "BoT-SORT/fast_reid/configs/MOT17/sbs_S50.yml")
opt.fast_reid_weights = str(PROJECT_ROOT / "weights/mot17_sbs_S50.pth")
opt.proximity_thresh = 2.0
opt.appearance_thresh = 0.30
opt.cmc_method = "sparseOptFlow"
opt.device = device
opt.fps = 30

# Initialize Tracker
tracker = BoTSORT(opt, frame_rate = opt.fps)
tracker_timer = Timer()
frame_timer = Timer()

# COCO Class Names for display
# COCO_CLASSES = [
#     "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
#     "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
#     "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
#     "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
#     "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
#     "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
#     "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
#     "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
#     "hair drier", "toothbrush"
# ]
CUSTOM_CLASSES = ["apc",
    "army_truck",
    "artillery_gun",
    "bmp",
    "camouflage_soldier",
    "civilian",
    "civilian_vehicle",
    "command_vehicle",
    "engineer_vehicle",
    "fkill",
    "imv",
    "kkill",
    "military_aircraft",
    "military_artillery",
    "military_truck",
    "military_vehicle",
    "military_warship",
    "missile",
    "mkill",
    "mt_lb",
    "reconnaissance_vehicle",
    "rocket",
    "rocket_artillery",
    "soldier",
    "tank",
    "trench",
    "weapon"
]
# --- Main Processing Loop ---
# cap = cv2.VideoCapture(0) # Use 0 for webcam
cap = cv2.VideoCapture(str(PROJECT_ROOT / "media/input.mp4"))
assert cap.isOpened(), "Error opening video stream or file"

frame_id = 0
cv2.namedWindow("Tracking", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Tracking", 1280, 720)
with torch.no_grad():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_timer.tic()

        # Run YOLO inference
        results = yolo(frame, conf=0.25, iou=0.7, device=device)[0]
        detections = []

        # Parse results
        if results.boxes is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            scores = results.boxes.conf.cpu().numpy()
            clss = results.boxes.cls.cpu().numpy()

            for box, score, cls in zip(boxes, scores, clss):
                x1, y1, x2, y2 = box
                detections.append([x1, y1, x2, y2, score, cls])

        # Prepare detections for BoT-SORT
        if len(detections) == 0:
            detections = np.empty((0, 6), dtype=np.float32)
        else:
            detections = np.asarray(detections, dtype=np.float32)

        # Update Tracker
        tracker_timer.tic()
        online_targets = tracker.update(detections, frame)
        tracker_timer.toc()

        # Visualize Tracks
        for t in online_targets:
            x1, y1, x2, y2 = map(int, t.tlbr)
            tid = t.track_id

            # Filter small boxes
            if (x2 - x1) * (y2 - y1) < opt.min_box_area:
                continue

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Get class name
            try:
                # Use t.cls (BoT-SORT track class)
                cls_id = int(t.cls) if hasattr(t, 'cls') else 0
                cls_name = CUSTOM_CLASSES[cls_id] if 0 <= cls_id < len(CUSTOM_CLASSES) else str(cls_id)
            except Exception:
                cls_name = "Unknown"

            # Draw label
            cv2.putText(
                frame,
                f"ID {tid} | {cls_name}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

        cv2.imshow("Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        frame_timer.toc()

cap.release()
cv2.destroyAllWindows()
print("Avg FPS:",1.0/frame_timer.average_time)
print("Tracker FPS:",1.0/tracker_timer.average_time)

