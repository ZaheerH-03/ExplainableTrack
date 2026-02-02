"""
GradCAM for RT-DETR Object Detection
This script generates Gradient-weighted Class Activation Maps (GradCAM) for RT-DETR models.

Key Logic:
1. RT-DETR uses fixed queries (e.g. 300).
2. We identify which query corresponds to our target detection.
3. We maximize the score of that specific query to find which image regions contributed to it.
"""

import sys
import argparse
import cv2
import numpy as np
import torch
import torch.nn as nn
from pytorch_grad_cam import GradCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pathlib import Path

# --- Path Setup (Same as other scripts) ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RT_DETR_ROOT = PROJECT_ROOT / "RT-DETRv4-main"
if str(RT_DETR_ROOT) not in sys.path:
    sys.path.append(str(RT_DETR_ROOT))

try:
    from engine.core import YAMLConfig
except ImportError:
    print(f"Error: Could not import YAMLConfig from {RT_DETR_ROOT}")
    sys.exit(1)

def load_rtdetr_model(config_path, weights_path, device='cuda'):
    cfg = YAMLConfig(str(config_path), resume=str(weights_path))
    if 'HGNetv2' in cfg.yaml_cfg:
        cfg.yaml_cfg['HGNetv2']['pretrained'] = False
        
    checkpoint = torch.load(weights_path, map_location='cpu')
    if 'ema' in checkpoint:
        state = checkpoint['ema']['module']
    else:
        state = checkpoint['model']
    
    cfg.model.load_state_dict(state)
    
    # We need the base model, not the deploy wrapper, to ensure gradients flow easily?
    # Actually deploy wrapper is fine if it keeps the graph.
    # But standard deploy() usually sets eval() and might detach.
    # Let's check: Config.model.deploy() typically allows forward.
    
    model = cfg.model.deploy()
    # Ensure gradients
    for param in model.parameters():
        if param.dtype.is_floating_point or param.dtype.is_complex:
            param.requires_grad = True
        
    model.to(device)
    model.eval() # but gradients enabled
    return model, cfg

class RTDETRGradWrapper(nn.Module):
    """
    Wraps RT-DETR to return the specific score of the target query.
    """
    def __init__(self, model, target_query_idx, target_class):
        super().__init__()
        self.model = model
        self.target_query_idx = target_query_idx
        self.target_class = target_class
        
    def forward(self, x):
        # RT-DETR forward (no post-processor needed for raw logits)
        # We need raw logits, not processed boxes
        # model(x) returns dictionary usually {pred_logits, pred_boxes, ...}
        
        # NOTE: deploy() model might return tuple or whatever.
        # Let's inspect carefully.
        # In RT-DETRv4, model forward returns dict with 'pred_logits' and 'pred_boxes'
        
        outputs = self.model(x)
        
        # If it's the deploy wrapper from track_rtdetrv4, it calls postprocessor
        # We want RAW outputs for GradCAM
        if isinstance(outputs, (tuple, list)):
            # If wrapped by postprocessor, it might be too late (no gradients?)
            # We assume 'model' passed here is the CORE model
            pass
            
        # Extract Logits: [batch, queries, classes]
        logits = outputs['pred_logits']
        
        # Return the score of the specific query and class
        # logits are usually unnormalized? GradCAM works fine with logits.
        # [0, query_idx, class_idx]
        return logits[0, self.target_query_idx, self.target_class].reshape(1, 1)

def run_gradcam_rtdetr(image_path, config_path, weights_path, box_idx=0, target_layer_name='backbone', output_path="media/gradcam_rtdetr.jpg"):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Loading RT-DETR...")
    base_model, cfg = load_rtdetr_model(config_path, weights_path, device)
    
    # 1. Identify Target Layer
    target_layer = None
    for name, module in base_model.named_modules():
        if name == target_layer_name:
            target_layer = module
            # Heuristic for backbone last layer
            if len(list(target_layer.children())) > 0:
                # Drill down (simplified for brevity, assume last child)
                 target_layer = list(target_layer.children())[-1]
                 if len(list(target_layer.children())) > 0:
                     target_layer = list(target_layer.children())[-1]
            break
            
    if target_layer is None:
        print(f"Could not find target layer: {target_layer_name}")
        # Default fallback
        pass

    print(f"Target Layer: {target_layer.__class__.__name__}")

    # 2. Preprocess Image
    img = cv2.imread(image_path)
    img_rs = cv2.resize(img, (640, 640))
    img_rgb = cv2.cvtColor(img_rs, cv2.COLOR_BGR2RGB)
    
    transforms = nltk = args = None # cleanup namespace
    import torchvision.transforms as T
    t_transform = T.Compose([
        T.ToTensor(),
    ])
    tensor_img = t_transform(img_rgb).unsqueeze(0).to(device)
    
    # 3. Find Best Query (Forward Pass)
    # We need to run standard inference to find the best box index
    # But RT-DETR outputs 300 queries. We need to match detections to queries.
    # Usually they are sorted by score if post-processed.
    # But raw outputs are NOT sorted.
    
    # Run Raw Forward
    with torch.no_grad():
        raw_out = base_model(tensor_img)
        # pred_logits: [1, 300, 80]
        # pred_boxes: [1, 300, 4]
        prob = raw_out['pred_logits'].sigmoid()
        topk_values, topk_indexes = torch.topk(prob.view(1, -1), 300)
        
    # We emulate the post-processor to find which query is our "box_idx" (e.g. 0 = highest confidence)
    # The 'topk_indexes' gives us the flattened index.
    # We need to unravel it.
    scores = prob.squeeze() # [300, 80]
    # Max score per query
    max_scores, max_classes = scores.max(dim=1)
    
    # Sort queries by their max score
    sorted_indices = torch.argsort(max_scores, descending=True)
    
    # Select the query corresponding to the requested rank (box_idx)
    target_query_idx = sorted_indices[box_idx].item()
    target_class_idx = max_classes[target_query_idx].item()
    target_score = max_scores[target_query_idx].item()
    
    print(f"Targeting Query #{target_query_idx}")
    print(f"  Class: {target_class_idx}")
    print(f"  Score: {target_score:.4f}")
    
    # 4. Wrap Model for GradCAM
    # This wrapper returns the scalar score of that specific query
    grad_wrapper = RTDETRGradWrapper(base_model, target_query_idx, target_class_idx)
    
    # 5. Run GradCAM
    cam = GradCAM(model=grad_wrapper, target_layers=[target_layer])
    
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    
    # Targets=[ClassifierOutputTarget(0)] means "maximize the 0-th element of the output vector"
    # Our wrapper returns a vector of size 1 (the specific score), so index 0 is correct.
    grayscale_cam = cam(input_tensor=tensor_img, targets=[ClassifierOutputTarget(0)])[0, :]
    
    # 6. Visualize
    visualization = show_cam_on_image(img_rgb.astype(float)/255.0, grayscale_cam, use_rgb=True)
    
    cv2.imwrite(output_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
    print(f"Saved {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default="media/3.png")
    parser.add_argument("--box_idx", type=int, default=0)
    parser.add_argument("--output", type=str, default="media/gradcam_rtdetr.jpg")
    args = parser.parse_args()
    
    # Hardcoded paths matching environment
    WEIGHTS = PROJECT_ROOT / "weights/RTv4-L-hgnet.pth"
    CONFIG = RT_DETR_ROOT / "configs/rtv4/rtv4_hgnetv2_l_coco.yml"
    
    run_gradcam_rtdetr(args.image, CONFIG, WEIGHTS, args.box_idx, output_path=args.output)
