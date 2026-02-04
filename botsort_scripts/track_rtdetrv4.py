"""
RT-DETRv4 Tracking Script

This script demonstrates object tracking using the RT-DETRv4 object detection model 
and BoT-SORT tracker. It processes video input, detects objects, tracks them across 
frames, and visualizes the results with bounding boxes and unique IDs.

Key Features:
- Integates RT-DETRv4 for object detection.
- Uses BoT-SORT for multi-object tracking.
- Supports FastReID for re-identification features.
- Visualizes tracking results in real-time.
"""

import sys
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from pathlib import Path

# --- Path Setup ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# RT-DETRv4 Path Setup
RT_DETR_ROOT = PROJECT_ROOT / "RT-DETRv4-main"
if not RT_DETR_ROOT.exists():
    print(f"Error: Could not find RT-DETRv4-main at {RT_DETR_ROOT}")
    sys.exit(1)

# Import RT-DETRv4 modules (adding to sys.path)
sys.path.append(str(RT_DETR_ROOT))
from engine.core import YAMLConfig

# Import BoT-SORT modules
sys.path.append(str(PROJECT_ROOT / "BoT-SORT"))
from tracker.mc_bot_sort import BoTSORT
from tracker.tracking_utils.timer import Timer

# --- Configuration & Constants ---
WEIGHTS_PATH = PROJECT_ROOT / "weights/best_stg1.pth"
CONFIG_PATH = RT_DETR_ROOT / "configs/rtv4/rtv4_x_custom.yml"

if not WEIGHTS_PATH.exists():
    print(f"Error: Weights not found at {WEIGHTS_PATH}")
    # sys.exit(1) # Warning only, let user fix

class Opt:
    """
    Configuration class for BoT-SORT tracker settings.
    Acts as a namespace for passing arguments to the tracker.
    """
    pass

opt = Opt()
opt.name = "rtdetr_v4"
opt.ablation = False
opt.track_high_thresh = 0.5  # High detection threshold for tracking
opt.track_low_thresh = 0.1   # Low detection threshold for tracking
opt.new_track_thresh = 0.6   # Threshold for creating a new track
opt.track_buffer = 3600      # Buffer size for tracking history
opt.match_thresh = 0.7       # IoU matching threshold
opt.aspect_ratio_thresh = 1.6 # Aspect ratio threshold for filtering
opt.min_box_area = 10        # Minimum box area for valid detections
opt.mot20 = False            # MOT20 dataset setting
opt.with_reid = True         # Enable ReID module
opt.fast_reid_config = str(PROJECT_ROOT / "BoT-SORT/fast_reid/configs/MOT17/sbs_S50.yml")
opt.fast_reid_weights = str(PROJECT_ROOT / "weights/mot17_sbs_S50.pth")
opt.proximity_thresh = 2.0   # Proximity threshold for ReID
opt.appearance_thresh = 0.30 # Appearance similarity threshold
opt.cmc_method = "sparseOptFlow" # Camera motion compensation method
opt.device = "cuda" if torch.cuda.is_available() else "cpu"
opt.fps = 30

# COCO Class Names for visualization
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
    "weapon"]

def load_rtdetr_model(config_path, resume_path, device):
    """
    Loads the RT-DETR model from the specified configuration and checkpoint.

    Args:
        config_path (str or Path): Path to the YAML configuration file.
        resume_path (str or Path): Path to the model checkpoint (.pth).
        device (str): Device to load the model on ('cuda' or 'cpu').

    Returns:
        model (nn.Module): Loaded and ready-to-use RT-DETR model.
    """
    cfg = YAMLConfig(str(config_path), resume=str(resume_path))
    
    # Disable pretrained backbone if HGNetv2 is used (prevents unnecessary downloads/warnings if local)
    if 'HGNetv2' in cfg.yaml_cfg:
        cfg.yaml_cfg['HGNetv2']['pretrained'] = False
        
    checkpoint = torch.load(resume_path, map_location='cpu')
    if 'ema' in checkpoint:
        state = checkpoint['ema']['module']
    else:
        state = checkpoint['model']
        
    cfg.model.load_state_dict(state)
    
    class Model(nn.Module):
        """Wrapper class to bundle model and postprocessor."""
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()
            
        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs
            
    model = Model().to(device)
    model.eval()
    return model

def main():
    """
    Main execution loop for video object tracking.
    """
    device = opt.device
    print(f"Loading RT-DETR model from {WEIGHTS_PATH}...")
    try:
        model = load_rtdetr_model(CONFIG_PATH, WEIGHTS_PATH, device)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # Initialize Tracker
    tracker = BoTSORT(opt, frame_rate=opt.fps)
    tracker_timer = Timer()
    frame_timer = Timer()

    # Video Source Input
    # cap = cv2.VideoCapture(0) # Uncomment for webcam
    cap = cv2.VideoCapture("/media/aid-pc/My1TB/Zaheer/Explainable_Object_tacking/media/input.mp4")
    
    # Preprocessing transforms
    transforms = T.Compose([
        T.Resize((640, 640)), # Adjust based on config if needed
        T.ToTensor(),
    ])

    print("Starting tracking...")
    cv2.namedWindow("RT-DETRv4 Tracking", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("RT-DETRv4 Tracking", 1280, 720)
    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_timer.tic()
            
            # --- Preprocessing ---
            frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            w, h = frame_pil.size
            orig_size = torch.tensor([[w, h]]).to(device)
            im_data = transforms(frame_pil).unsqueeze(0).to(device)
            
            # --- Inference ---
            output = model(im_data, orig_size)
            labels, boxes, scores = output
            
            # --- Format Detections for BoT-SORT ---
            # output structure: labels (1, N), boxes (1, N, 4), scores (1, N)
            
            # Unpack batch (assuming batch size 1)
            labels = labels[0]
            boxes = boxes[0]
            scores = scores[0]
            
            # Filter low confidence detections
            mask = scores > 0.4 # basic confidence filter
            labels = labels[mask]
            boxes = boxes[mask]
            scores = scores[mask]
            
            detections = []
            if len(scores) > 0:
                boxes = boxes.cpu().numpy()
                scores = scores.cpu().numpy()
                labels = labels.cpu().numpy()
                
                for box, score, label in zip(boxes, scores, labels):
                    x1, y1, x2, y2 = box
                    detections.append([x1, y1, x2, y2, score, label])
            
            if len(detections) == 0:
                detections = np.empty((0, 6), dtype=np.float32)
            else:
                # [x1, y1, x2, y2, score, cls]
                detections = np.asarray(detections, dtype=np.float32)

            # --- Tracking Update ---
            tracker_timer.tic()
            online_targets = tracker.update(detections, frame)
            tracker_timer.toc()

            # --- Visualization ---
            for t in online_targets:
                x1, y1, x2, y2 = map(int, t.tlbr) # Get bounding box coordinates
                tid = t.track_id
                
                # Draw Bounding Box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Get class name
                try:
                    # Retrieve class ID from tracker object (if available)
                    cls_id = int(t.cls) if hasattr(t, 'cls') else 0
                    cls_name = CUSTOM_CLASSES[cls_id] if 0 <= cls_id < len(CUSTOM_CLASSES) else str(cls_id)
                except Exception:
                    cls_name = "Unknown"

                # Draw Label
                label_text = f"ID {tid} | {cls_name}"
                cv2.putText(frame, label_text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Show Frame
            cv2.imshow('RT-DETRv4 Tracking', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            frame_timer.toc()
            
    cap.release()
    cv2.destroyAllWindows()
    print("Avg FPS:", 1.0/frame_timer.average_time)

if __name__ == "__main__":
    main()

