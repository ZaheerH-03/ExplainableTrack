import sys
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "BoT-SORT"))

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from tracker.mc_bot_sort import BoTSORT
from tracker.tracking_utils.timer import Timer

# YOLOv11 Model
# Assuming weights are in the standard weights directory relative to project root
model_path = PROJECT_ROOT / "weights/yolo11m.pt"
if not model_path.exists():
    # Fallback or auto-download by YOLO class if strictly needed, 
    # but let's point to where it should be.
    pass

yolo = YOLO(model_path, task="detect")
device = "cuda" if torch.cuda.is_available() else "cpu"
yolo.to(device)

class Opt:
    pass

opt = Opt()
opt.name = "yolov11"
opt.ablation = False
opt.track_high_thresh = 0.5
opt.track_low_thresh = 0.1
opt.new_track_thresh = 0.6
opt.track_buffer = 1200
opt.match_thresh = 0.7
opt.aspect_ratio_thresh = 1.6
opt.min_box_area = 10
opt.mot20 = False

opt.with_reid = True
# Clean paths using PROJECT_ROOT
opt.fast_reid_config = str(PROJECT_ROOT / "BoT-SORT/fast_reid/configs/MOT17/sbs_S50.yml")
opt.fast_reid_weights = str(PROJECT_ROOT / "weights/mot17_sbs_S50.pth")
opt.proximity_thresh = 2.0
opt.appearance_thresh = 0.45
opt.cmc_method = "sparseOptFlow"
opt.device = device
opt.fps = 30

tracker = BoTSORT(opt, frame_rate = opt.fps)
tracker_timer = Timer()
frame_timer = Timer()

# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture(str(PROJECT_ROOT / "input3.mp4"))
assert cap.isOpened()

frame_id = 0
with torch.no_grad():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_timer.tic()

        results = yolo(frame, conf=0.25, iou=0.7, device=device)[0]
        detections = []

        if results.boxes is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            scores = results.boxes.conf.cpu().numpy()
            clss = results.boxes.cls.cpu().numpy()

            for box, score, cls in zip(boxes, scores, clss):
                x1, y1, x2, y2 = box
                detections.append([x1, y1, x2, y2, score, cls])

        if len(detections) == 0:
            detections = np.empty((0, 6), dtype=np.float32)
        else:
            detections = np.asarray(detections, dtype=np.float32)

        tracker_timer.tic()
        online_targets = tracker.update(detections, frame)
        tracker_timer.toc()

        for t in online_targets:
            x1, y1, x2, y2 = map(int, t.tlbr)
            tid = t.track_id

            if (x2 - x1) * (y2 - y1) < opt.min_box_area:
                continue

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"ID {tid}",
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
