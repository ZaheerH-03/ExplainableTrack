import sys
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image

from pathlib import Path

# Paths
# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# RT-DETRv4 Path
RT_DETR_ROOT = PROJECT_ROOT / "RT-DETRv4-main"
if not RT_DETR_ROOT.exists():
    print(f"Error: Could not find RT-DETRv4-main at {RT_DETR_ROOT}")
    sys.exit(1)

# Import RT-DETRv4 modules (PYTHONPATH handled by run.sh)
sys.path.append(str(RT_DETR_ROOT))
from engine.core import YAMLConfig

sys.path.append(str(PROJECT_ROOT / "BoT-SORT"))
from tracker.mc_bot_sort import BoTSORT
from tracker.tracking_utils.timer import Timer

# Configuration
WEIGHTS_PATH = PROJECT_ROOT / "weights/RTv4-L-hgnet.pth"
CONFIG_PATH = RT_DETR_ROOT / "configs/rtv4/rtv4_hgnetv2_l_coco.yml"

if not WEIGHTS_PATH.exists():
    print(f"Error: Weights not found at {WEIGHTS_PATH}")
    # sys.exit(1) # Warning only, let user fix

class Opt:
    pass

opt = Opt()
opt.name = "rtdetr_v4"
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
opt.fast_reid_config = str(PROJECT_ROOT / "BoT-SORT/fast_reid/configs/MOT17/sbs_S50.yml")
opt.fast_reid_weights = str(PROJECT_ROOT / "weights/mot17_sbs_S50.pth")
opt.proximity_thresh = 2.0
opt.appearance_thresh = 0.45
opt.cmc_method = "sparseOptFlow"
opt.device = "cuda" if torch.cuda.is_available() else "cpu"
opt.fps = 30

def load_rtdetr_model(config_path, resume_path, device):
    cfg = YAMLConfig(str(config_path), resume=str(resume_path))
    
    if 'HGNetv2' in cfg.yaml_cfg:
        cfg.yaml_cfg['HGNetv2']['pretrained'] = False
        
    checkpoint = torch.load(resume_path, map_location='cpu')
    if 'ema' in checkpoint:
        state = checkpoint['ema']['module']
    else:
        state = checkpoint['model']
        
    cfg.model.load_state_dict(state)
    
    class Model(nn.Module):
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
    device = opt.device
    print(f"Loading RT-DETR model from {WEIGHTS_PATH}...")
    try:
        model = load_rtdetr_model(CONFIG_PATH, WEIGHTS_PATH, device)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    tracker = BoTSORT(opt, frame_rate=opt.fps)
    tracker_timer = Timer()
    frame_timer = Timer()

    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture("/media/aid-pc/My1TB/Zaheer/botsort/input3.mp4")
    
    transforms = T.Compose([
        T.Resize((640, 640)), # Adjust based on config if needed
        T.ToTensor(),
    ])

    print("Starting tracking...")
    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_timer.tic()
            
            # Preprocess
            frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            w, h = frame_pil.size
            orig_size = torch.tensor([[w, h]]).to(device)
            im_data = transforms(frame_pil).unsqueeze(0).to(device)
            
            # Inference
            output = model(im_data, orig_size)
            labels, boxes, scores = output
            
            # Format detections for BoT-SORT [x1, y1, x2, y2, score, cls]
            # output content: labels (1, N), boxes (1, N, 4), scores (1, N)
            
            # Assuming batch size 1
            labels = labels[0]
            boxes = boxes[0]
            scores = scores[0]
            
            # Filter low confidence
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
                detections = np.asarray(detections, dtype=np.float32)

            tracker_timer.tic()
            online_targets = tracker.update(detections, frame)
            tracker_timer.toc()

            for t in online_targets:
                x1, y1, x2, y2 = map(int, t.tlbr)
                tid = t.track_id
                
                # Draw
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID {tid}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            cv2.imshow('RT-DETRv4 Tracking', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            frame_timer.toc()
            
    cap.release()
    cv2.destroyAllWindows()
    print("Avg FPS:", 1.0/frame_timer.average_time)

if __name__ == "__main__":
    main()
