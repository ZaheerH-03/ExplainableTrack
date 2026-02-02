"""
LIME Explanation for RT-DETR Object Detection

This script produces Local Interpretable Model-agnostic Explanations (LIME) for RT-DETR 
object detection predictions. It uses the custom RT-DETR v4 implementation found in this repository.

Structure:
1.  **Model Loading**: Custom logic to load RT-DETRv4 from `RT-DETRv4-main`.
2.  **LIME Image Explainer**: Generates perturbed samples.
3.  **Predict Function**: Wrapper that adapts LIME's input (images) to RT-DETR's expected input 
    (tensors + orig_sizes) and returns a probability distribution based on Class Confidence * IoU.
4.  **Visualization**: Displays Original, LIME Explanation, and Heatmap.
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
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries
from lime import lime_image

# --- Path Setup ---
# Assuming this script is in botsort/xai/lime/
# We need to reach botsort/ (parents[2])
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# RT-DETRv4 Path Setup
RT_DETR_ROOT = PROJECT_ROOT / "RT-DETRv4-main"
if not RT_DETR_ROOT.exists():
    raise FileNotFoundError(f"Could not find RT-DETRv4-main at {RT_DETR_ROOT}")

# Import RT-DETRv4 modules
sys.path.append(str(RT_DETR_ROOT))
from engine.core import YAMLConfig

# --- Configuration ---
DEFAULT_WEIGHTS = PROJECT_ROOT / "weights/RTv4-L-hgnet.pth"
DEFAULT_CONFIG = RT_DETR_ROOT / "configs/rtv4/rtv4_hgnetv2_l_coco.yml"

# COCO Class Names
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]

def load_rtdetr_model(config_path, resume_path, device='cuda'):
    """
    Loads the RT-DETR model using the valid config and checkpoint.
    Replicated from botsort_scripts/track_rtdetrv4.py for consistency.
    """
    print(f"Loading RT-DETR config from {config_path}")
    print(f"Loading RT-DETR weights from {resume_path}")
    
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

def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) of two bounding boxes.
    Boxes are [x1, y1, x2, y2].
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0

def lime_explain_rtdetr(
    image_path,
    config_path=DEFAULT_CONFIG,
    model_path=DEFAULT_WEIGHTS,
    target_box_idx=0,
    num_samples=1000,
    num_features=10,
    output_path='./media/lime_rtdetr_explanation.jpg'
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Load Model
    try:
        model = load_rtdetr_model(config_path, model_path, device)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 2. Load Image
    print(f"Loading image from {image_path}...")
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not find image at {image_path}")
    
    # LIME works with RGB
    image_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w, _ = image_rgb.shape
    
    # Transform for RT-DETR (Resize + ToTensor)
    transforms = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor(),
    ])

    # 3. Initial Detection to find Target
    print("Running initial detection...")
    
    frame_pil = Image.fromarray(image_rgb)
    orig_size = torch.tensor([[w, h]]).to(device)
    im_data = transforms(frame_pil).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(im_data, orig_size)
    
    labels, boxes, scores = output
    # unpack batch
    labels = labels[0].cpu().numpy()
    boxes = boxes[0].cpu().numpy()
    scores = scores[0].cpu().numpy()
    
    # Filter detections (threshold 0.3 for example)
    mask = scores > 0.3
    labels = labels[mask]
    boxes = boxes[mask]
    scores = scores[mask]
    
    if len(scores) == 0:
        print("No objects detected!")
        return

    # Sort by confidence
    sorted_idx = np.argsort(scores)[::-1]
    if target_box_idx >= len(sorted_idx):
        print(f"Warning: Requested index {target_box_idx} out of range, using 0.")
        target_box_idx = 0
        
    target_idx = sorted_idx[target_box_idx]
    target_box = boxes[target_idx]
    target_score = scores[target_idx]
    target_label = int(labels[target_idx])
    
    class_name = COCO_CLASSES[target_label] if target_label < len(COCO_CLASSES) else str(target_label)
    
    print(f"\nExplaining detection #{target_box_idx}:")
    print(f"  Class: {class_name} ({target_label})")
    print(f"  Confidence: {target_score:.3f}")
    print(f"  Box: {target_box}")

    # 4. Define Predict Function
    def predict_fn(images):
        """
        Args:
            images: List or array of RGB images (H, W, 3).
                    LIME generates these with the same size as the input image.
        """
        # Batch preparation
        batch_tensors = []
        batch_sizes = []
        
        for img in images:
            # Check if img is uint8 or float. LIME usually passes float [0,1] or uint8.
            # But the 'images' from LIME are typically numpy arrays.
            if img.dtype == np.float64 or img.dtype == np.float32:
                # If float, valid range [0,1] for PIL?
                # PIL usually expects uint8 for Image.fromarray if 0-255 or separate mode.
                # Usually best to convert to uint8 for consistency
                img_uint8 = (img * 255).astype(np.uint8)
            else:
                img_uint8 = img
            
            pil_img = Image.fromarray(img_uint8)
            tensor_img = transforms(pil_img)
            batch_tensors.append(tensor_img)
            batch_sizes.append([pil_img.size[0], pil_img.size[1]]) # w, h
            
        # Stack
        batch_input = torch.stack(batch_tensors).to(device)
        orig_sizes = torch.tensor(batch_sizes).to(device)
        
        with torch.no_grad():
            outputs = model(batch_input, orig_sizes)
        
        # unpack
        # batch_labels: (B, N)
        # batch_boxes: (B, N, 4)
        # batch_scores: (B, N)
        batch_labels, batch_boxes, batch_scores = outputs
        
        results_list = []
        
        batch_labels = batch_labels.cpu().numpy()
        batch_boxes = batch_boxes.cpu().numpy()
        batch_scores = batch_scores.cpu().numpy()
        
        for i in range(len(images)):
            # Per image results
            i_labels = batch_labels[i]
            i_boxes = batch_boxes[i]
            i_scores = batch_scores[i]
            
            best_score = 0.0
            
            # Use lower threshold for explanation stability
            mask = i_scores > 0.1
            i_labels = i_labels[mask]
            i_boxes = i_boxes[mask]
            i_scores = i_scores[mask]
            
            for lbl, box, score in zip(i_labels, i_boxes, i_scores):
                if int(lbl) == target_label:
                    iou = calculate_iou(box, target_box)
                    # Combined metric
                    combined = score * iou
                    if combined > best_score:
                        best_score = combined
            
            results_list.append([best_score, 1.0 - best_score])
            
        return np.array(results_list)

    # 5. Run LIME
    print(f"\nGenerating LIME explanation ({num_samples} samples)...")
    
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        image_rgb,
        predict_fn,
        top_labels=2,
        hide_color=0,
        num_samples=num_samples,
        batch_size=4 # Reduce batch size if OOM
    )
    
    # 6. Visualize
    print("Creating visualization...")
    temp, mask = explanation.get_image_and_mask(
        0,
        positive_only=True,
        num_features=num_features,
        hide_rest=False
    )
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original
    axes[0].imshow(image_rgb)
    x1, y1, x2, y2 = target_box
    rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, color='red', linewidth=3)
    axes[0].add_patch(rect)
    axes[0].text(x1, y1-10, f"{class_name}: {target_score:.2f}",
                 color='red', fontsize=12, weight='bold',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    axes[0].set_title('Original RT-DETR Detection', fontsize=14, weight='bold')
    axes[0].axis('off')
    
    # Explanation
    axes[1].imshow(mark_boundaries(temp, mask))
    axes[1].set_title(f'LIME Explanation\n(Top {num_features} Superpixels)', fontsize=14, weight='bold')
    axes[1].axis('off')
    
    # Heatmap
    # Extract weights
    segments = explanation.segments
    weights = np.zeros(image_rgb.shape[:2])
    for seg_id, weight in explanation.local_exp[0]:
        if weight > 0:
            weights[segments == seg_id] = weight
            
    if weights.max() > 0:
        weights = weights / weights.max()
        
    axes[2].imshow(image_rgb)
    axes[2].imshow(weights, cmap='jet', alpha=0.5)
    axes[2].set_title('Importance Heatmap', fontsize=14, weight='bold')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nExplanation saved to: {output_path}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="LIME Explanation for RT-DETR")
    parser.add_argument("--image", type=str, default=str(PROJECT_ROOT / "media/test1.png"), help="Path to input image")
    parser.add_argument("--box_idx", type=int, default=0, help="Index of detection to explain")
    parser.add_argument("--samples", type=int, default=1000, help="Number of LIME samples")
    parser.add_argument("--output", type=str, default="media/lime_rtdetr_result.jpg", help="Output path")
    
    args = parser.parse_args()
    
    lime_explain_rtdetr(
        image_path=args.image,
        target_box_idx=args.box_idx,
        num_samples=args.samples,
        output_path=args.output
    )
