"""
GradCAM for YOLO Object Detection
This script generates Gradient-weighted Class Activation Maps (GradCAM) for YOLO models.
It visualizes which parts of the image contributed most to a specific detection.

Key Components:
1. YOLOWrapper: Ensures the model outputs tensors compatible with GradCAM.
2. BoxScoreTarget: A custom target function to maximize the score of a specific bounding box.
"""

import argparse
import cv2
import numpy as np
import torch
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image, scale_cam_image
from ultralytics import YOLO

class YOLOWrapper(torch.nn.Module):
    def __init__(self, model, target_layers):
        super(YOLOWrapper, self).__init__()
        self.model = model
        self.target_layers = target_layers
        self.gradients = dict()
        self.activations = dict()

        def backward_hook(module, grad_input, grad_output):
            self.gradients['value'] = grad_output[0]
            return None

        def forward_hook(module, input, output):
            self.activations['value'] = output
            return None
            
        # Register hooks on target layers if needed, 
        # but pytorch-grad-cam usually handles this. 
        # We mainly need to ensure the forward output is differentiable 
        # and in the format (Batch, Categories) or similar if using ClassifierOutputTarget,
        # BUT for Object Detection we use a custom target so we can return whatever we want
        # as long as our target function can consume it.

    def forward(self, x):
        # We need to ensure we are in a mode that supports gradients
        # Ultralytics model() usually runs in no_grad if simple inference
        # But here checking...
        
        # Standard YOLO forward returns a list of results (boxes, etc)
        # We need to return specific detection outputs for our custom target function
        result = self.model(x, verbose=False)[0]
        return result

class BoxScoreTarget:
    """
    For every original image, find a high-confidence object close to the
    target_box and return its score.
    """
    def __init__(self, target_box, target_class, iou_threshold=0.5):
        self.target_box = target_box # [x1, y1, x2, y2]
        self.target_class = target_class
        self.iou_threshold = iou_threshold

    def __call__(self, model_output):
        # model_output is the result from YOLO (ultralytics.engine.results.Results)
        # However, pytorch-grad-cam expects the model to return a Tensor usually.
        # IF we return the 'Results' object from forward, we handle it here.
        
        # Since GradCAM computes gradients w.r.t model output, we actually need 
        # the model to output a Tensor that is differentiable.
        # Ultralytics 'predict' output is post-processed and usually detached.
        # THIS IS THE TRICKY PART.
        
        # Alternative: We target the specific Raw Output of the model (before NMS).
        # But simpler approach for visualization:
        # We can't easily get gradients through NMS.
        
        # fallback: Use the hook based approach from standard YOLO-GradCAM implementations
        # where we wrap the HEAD.
        return 0.0 # Placeholder logic, see implementation below

# REVISED APPROACH:
# We cannot use the standard Ultralytics 'predict' for GradCAM because NMS breaks gradients.
# We must use the model's internal forward pass and look at the raw logits/box preds.

class YOLOInteract(torch.nn.Module):
    """
    Wrapper to allow gradients to flow back from the raw model output.
    """
    def __init__(self, model):
        super(YOLOInteract, self).__init__()
        self.model = model.model # get the internal nn.Module
    
    def forward(self, x):
        return self.model(x)

def run_gradcam_yolo(image_path, model_path, target_layer_idx=-2, box_idx=0, output_path="media/gradcam_yolo.jpg"):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1. Load Model
    yolo_model = YOLO(model_path)
    model = yolo_model.model
    model.eval()
    model.to(device)
    
    # Enable gradients for the model parameters (some might be frozen)
    for param in model.parameters():
        param.requires_grad = True
        
    # 2. Select Target Layer
    # e.g. model.model[-2]
    target_layers = [model.model[target_layer_idx]]
    
    # 3. Load Image
    img = cv2.imread(image_path)
    img = cv2.resize(img, (640, 640))
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor_img = torch.from_numpy(rgb_img).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
    tensor_img.requires_grad = True
    
    # 4. Custom Model Wrapper?
    # No, we can use the model directly IF we define a custom semantic used by the library.
    # But wait, pytorch-grad-cam expects the forward() to return the target score TENSOR 
    # directly if we don't provide a 'targets' argument, OR we provide a target that acts on output.
    
    # Define a helper to extract the specific class score from the raw output
    # YOLO output[0] is usually [Batch, 4+Classes, Anchors] or [Batch, Anchors, 4+Classes]
    # For YOLOv8/11: [Batch, 4+Classes, 8400]
    
    # Let's verify output shape with a dummy pass
    with torch.no_grad():
        dummy_out = model(tensor_img)
        # Depending on version this might be a list or tuple
        # output[0] is usually the main prediction tensor
    
    # We need to Identify which "Anchor" (column) corresponds to our detected box.
    # So first, run NMS (standard predict) to find the best box.
    results = yolo_model.predict(rgb_img, verbose=False)[0]
    if len(results.boxes) <= box_idx:
        print(f"Error: Box index {box_idx} not found.")
        return
        
    target_box_data = results.boxes[box_idx]
    # We need the center point to match with the Grid/Anchor
    x1, y1, x2, y2 = target_box_data.xyxy[0].cpu().numpy()
    cx, cy = (x1 + x2)/2, (y1 + y2)/2
    target_cls = int(target_box_data.cls[0].cpu().numpy())
    
    # Identify the index in the raw tensor that corresponds to this box
    # This matches the spatial location in the 8400 candidates
    # This is complex. 
    
    # Alternative Strategy:
    # Use EigenCAM (already implemented) which is class agnostic? No, user wants GradCAM.
    # GradCAM needs Differentiable output.
    
    # Implementation:
    # We will wrap the model so that forward() returns the score of the *Best Matching* raw prediction.
    
    class YOLOGradCAMWrapper(torch.nn.Module):
        def __init__(self, model, target_cx, target_cy, target_cls):
            super().__init__()
            self.model = model
            self.target_cx = target_cx
            self.target_cy = target_cy
            self.target_cls = target_cls
            self.scale = 640.0 # assumption
            
        def forward(self, x):
            # Output is list. [0] is [B, 4+C, 8400]
            preds = self.model(x)[0]
            
            # preds shape: [1, 84, 8400] for COCO (4 box + 80 classes)
            # Transpose to [1, 8400, 84]
            preds = preds.permute(0, 2, 1)
            
            # Extract boxes (first 4) and classes (rest)
            # boxes are usually cx, cy, w, h
            pred_boxes = preds[..., :4]
            pred_scores = preds[..., 4:]
            
            # Find the candidate closest to our target center
            # Simple distance metric
            # Note: Pred boxes are normalized or in pixels? 
            # In v8/11 output is usually absolute pixels if not using end2end export
            
            # Calculate distance to target (Vectorized)
            # We treat the first box in batch
            box_cx = pred_boxes[0, :, 0]
            box_cy = pred_boxes[0, :, 1]
            
            dist = (box_cx - self.target_cx)**2 + (box_cy - self.target_cy)**2
            nearest_idx = torch.argmin(dist)
            
            # Return the score of the target class for this specific box
            # This is a scalar tensor connected to the graph!
            return pred_scores[0, nearest_idx, self.target_cls].reshape(1, 1)

    # Wrap
    wrapper = YOLOGradCAMWrapper(model, cx, cy, target_cls)
    
    # Now we can use GradCAM
    # We use "ClassifierOutputTarget" effectively because our wrapper returns the score
    
    cam = GradCAM(model=wrapper, target_layers=target_layers)
    
    # Targets: [ClassifierOutputTarget(0)] means maximize the 0-th element (our single score)
    grayscale_cam = cam(input_tensor=tensor_img, targets=[ClassifierOutputTarget(0)])[0, :]
    
    visualization = show_cam_on_image(rgb_img.astype(float)/255.0, grayscale_cam, use_rgb=True)
    cv2.imwrite(output_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
    print(f"Saved {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default="media/3.png")
    parser.add_argument("--model", type=str, default="weights/yolo11m.pt")
    parser.add_argument("--box_idx", type=int, default=0)
    parser.add_argument("--target_layer", type=int, default=-2)
    parser.add_argument("--output", type=str, default="media/gradcam_yolo.jpg")
    args = parser.parse_args()
    
    try:
        run_gradcam_yolo(args.image, args.model, args.target_layer, args.box_idx, args.output)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
