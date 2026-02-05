import sys
import os
import torch
import numpy as np
import cv2
import torchvision.transforms as T
from PIL import Image
from pathlib import Path

# --- Path Setup ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

# RT-DETRv4 Path Setup (Check if exists)
RT_DETR_ROOT = PROJECT_ROOT / "RT-DETRv4"
if RT_DETR_ROOT.exists():
    sys.path.append(str(RT_DETR_ROOT))
    try:
        from engine.core import YAMLConfig
    except ImportError:
        print("Warning: Could not import RT-DETR engine.")

# BoT-SORT Path Setup
BOT_SORT_ROOT = PROJECT_ROOT / "BoT-SORT"
if BOT_SORT_ROOT.exists():
    sys.path.append(str(BOT_SORT_ROOT))
    try:
        from tracker.mc_bot_sort import BoTSORT
    except ImportError:
        print("Warning: Could not import BoT-SORT.")

# YOLO
try:
    from ultralytics import YOLO
except ImportError:
    print("Warning: Could not import Ultralytics YOLO.")


class Opt:
    """Mock options for BoT-SORT"""
    def __init__(self):
        self.track_high_thresh = 0.5
        self.track_low_thresh = 0.1
        self.new_track_thresh = 0.6
        self.track_buffer = 30
        self.match_thresh = 0.8
        self.aspect_ratio_thresh = 1.6
        self.min_box_area = 10
        self.mot20 = False
        self.cmc_method = "sparseOptFlow"
        self.name = "rtdetr_tracker"
        self.ablation = False
        self.with_reid = True # Enable ReID for better tracking
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.fps = 30
        
        # ReID specific
        self.proximity_thresh = 2.0
        self.appearance_thresh = 0.25
        self.fast_reid_config = str(PROJECT_ROOT / "BoT-SORT/fast_reid/configs/MOT17/sbs_S50.yml")
        self.fast_reid_weights = str(PROJECT_ROOT / "weights/mot17_sbs_S50.pth")

class ObjectTracker:
    def __init__(self, model_type='yolo', weights_path=None, config_path=None):
        self.model_type = model_type.lower()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tracker = None
        
        if self.model_type == 'yolo':
            self.model = YOLO(weights_path)
            # YOLO has internal tracker
        elif self.model_type == 'rtdetr':
            self._init_rtdetr(weights_path, config_path)
            # Initialize BoT-SORT
            opt = Opt()
            self.tracker = BoTSORT(opt, frame_rate=30)
            
            # Transforms for RT-DETR
            self.transforms = T.Compose([
                T.Resize((640, 640)),
                T.ToTensor(),
            ])

    def _init_rtdetr(self, weights, config):
        print(f"Loading RT-DETR: {weights} | Config: {config}")
        cfg = YAMLConfig(str(config), resume=str(weights))
        if 'HGNetv2' in cfg.yaml_cfg:
            cfg.yaml_cfg['HGNetv2']['pretrained'] = False
        
        checkpoint = torch.load(weights, map_location='cpu')
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
        else:
            state = checkpoint['model']
        
        cfg.model.load_state_dict(state)
        
        class Model(torch.nn.Module):
            def __init__(self, cfg):
                super().__init__()
                self.model = cfg.model.deploy()
                self.postprocessor = cfg.postprocessor.deploy()
            def forward(self, images, orig_target_sizes):
                outputs = self.model(images)
                return self.postprocessor(outputs, orig_target_sizes)
        
        self.model = Model(cfg).to(self.device)
        self.model.eval()

    def update(self, frame):
        """
        Run detection + tracking on the frame.
        Returns:
            results: List of tracked objects. 
                     Format varies, but we aim to standardize or return raw object that main script parses.
                     Actually, let's standardize output to: list of [x1, y1, x2, y2, id, cls_id, conf]
        """
        detections = []
        
        if self.model_type == 'yolo':
            # Ultralytics track
            # persist=True is important for tracking
            results = self.model.track(frame, persist=True, verbose=False)[0]
            
            # Parse to standard format
            boxes = results.boxes
            if boxes.id is not None:
                track_ids = boxes.id.int().cpu().tolist()
                cls = boxes.cls.int().cpu().tolist()
                conf = boxes.conf.cpu().tolist()
                xyxy = boxes.xyxy.cpu().tolist()
                
                for box, tid, c, cf in zip(xyxy, track_ids, cls, conf):
                    detections.append([*box, tid, c, cf])
                    
        elif self.model_type == 'rtdetr':
             # 1. Preprocess
            frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            w, h = frame_pil.size
            orig_size = torch.tensor([[w, h]]).to(self.device)
            im_data = self.transforms(frame_pil).unsqueeze(0).to(self.device)
            
            # 2. Inference
            with torch.no_grad():
                output = self.model(im_data, orig_size)
            
            labels, boxes, scores = output
            
            # 3. Format
            labels, boxes, scores = labels[0], boxes[0], scores[0]
            mask = scores > 0.4
            labels, boxes, scores = labels[mask], boxes[mask], scores[mask]
            
            botsort_dets = []
            if len(scores) > 0:
                boxes_np = boxes.cpu().numpy()
                scores_np = scores.cpu().numpy()
                labels_np = labels.cpu().numpy()
                for box, score, label in zip(boxes_np, scores_np, labels_np):
                     # [x1, y1, x2, y2, score, label]
                     botsort_dets.append([*box, score, label])
            
            if len(botsort_dets) == 0:
                botsort_dets = np.empty((0, 6), dtype=np.float32)
            else:
                botsort_dets = np.asarray(botsort_dets, dtype=np.float32)
                
            # 4. Track
            online_targets = self.tracker.update(botsort_dets, frame)
            
            # 5. Format Output
            for t in online_targets:
                # BoT-SORT returns tlbr (x1, y1, x2, y2)
                x1, y1, x2, y2 = t.tlbr
                tid = t.track_id
                cls_id = int(t.cls) if hasattr(t, 'cls') else 0
                score = t.score if hasattr(t, 'score') else 1.0 # BoT-SORT track score
                
                detections.append([x1, y1, x2, y2, tid, cls_id, score])

        return detections

    def get_model_object(self):
        """Returns the underlying model for XAI."""
        # For XAI, we usually need the core model
        if self.model_type == 'rtdetr':
            return self.model # Full Model instance (has .model Deploy and .postprocessor)
        return self.model.model # The YOLO detection model
