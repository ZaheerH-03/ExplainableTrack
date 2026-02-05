import os
import torch
import cv2
import numpy as np
import sys
from PIL import Image

# Import pytorch-grad-cam
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# --- Wrapper Classes for Compatibility ---

class YOLOWrapper(torch.nn.Module):
    """Features wrapper for YOLO"""
    def __init__(self, yolo_model, target_layers):
        super().__init__()
        self.model = yolo_model.model
        self.target_layers = target_layers
        self.features = None
        
        def hook_fn(module, input, output):
            if isinstance(output, (list, tuple)):
                for item in output:
                    if isinstance(item, torch.Tensor):
                        self.features = item
                        break
                if self.features is None and isinstance(input, (list, tuple)):
                     for item in input:
                        if isinstance(item, torch.Tensor):
                            self.features = item
                            break
            else:
                self.features = output

        for layer in target_layers:
            layer.register_forward_hook(hook_fn)

    def forward(self, x):
        _ = self.model(x)
        return self.features

class RTDETRWrapper(torch.nn.Module):
    """
    Wrapper for RT-DETR model compatible with pytorch-grad-cam.
    Returns logits for EigenCAM processing.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, x):
        # RT-DETR expects images and orig_target_sizes
        b, c, h, w = x.shape
        orig_target_sizes = torch.tensor([[w, h]]).to(x.device)
        
        try:
            outputs = self.model(x, orig_target_sizes)
        except Exception:
            # Fallback for models not needing target sizes
            outputs = self.model(x)

        # EigenCAM expects tensor output (logits)
        # RT-DETR returns (logits, boxes) tuple or dict
        if isinstance(outputs, (tuple, list)):
            return outputs[0]
        elif isinstance(outputs, dict):
            if 'pred_logits' in outputs:
                return outputs['pred_logits']
        
        return outputs

# --- Explainer Class ---

class Explainer:
    def __init__(self, model, model_type='yolo'):
        """
        Args:
            model: The loaded torch model (YOLO or RT-DETR)
            model_type: 'yolo' or 'rtdetr'
        """
        self.model = model
        self.model_type = model_type.lower()
        self.cam = None
        
        self._setup_eigencam()

    def _setup_eigencam(self):
        if self.model_type == 'yolo':
            # Target layer -3 (Neck) usually
            # YOLO model structure: model (DetectionModel) -> model (Sequential) -> layers
            # So we want self.model.model[-3]
            target_layer = self.model.model[-3]
            self.wrapped_model = YOLOWrapper(self.model, [target_layer])
            self.target_layers = [target_layer]
            
        elif 'rtdetr' in self.model_type:
            # Target backbone last stage usually
            # RT-DETR structure varies, assuming 'backbone' exists
            # We will search for 'backbone' logic similar to standalone script
            # For simplicity, we'll try to find the last sequential block of backbone
            try:
                # For RT-DETR, self.model is Model instance, backbone is in self.model.model (Deploy)
                rtdetr_deploy = self.model.model if hasattr(self.model, 'model') else self.model
                
                target_layer = None
                if hasattr(rtdetr_deploy, 'backbone'):
                    target_layer = rtdetr_deploy.backbone
                    print(f"Found backbone: {target_layer.__class__.__name__}")
                    
                    # Check for 'body' (TIMM backbones)
                    if hasattr(target_layer, 'body'):
                        target_layer = target_layer.body
                        print(f"Descended into .body ({target_layer.__class__.__name__})")
                    
                    # If ModuleList, take last
                    if isinstance(target_layer, torch.nn.ModuleList):
                        if len(target_layer) > 0:
                            target_layer = target_layer[-1]
                            print(f"Selected last item in ModuleList ({target_layer.__class__.__name__})")
                    
                    # Check for 'stages' (HGNet)
                    if hasattr(target_layer, 'stages'):
                        target_layer = target_layer.stages
                        print(f"Descended into .stages ({target_layer.__class__.__name__})")
                        if isinstance(target_layer, torch.nn.ModuleList) and len(target_layer) > 0:
                            target_layer = target_layer[-1]
                            print(f"Selected last stage ({target_layer.__class__.__name__})")
                else:
                    print("Explainer Warning: Could not find 'backbone' in RT-DETR. XAI may fail.")
                    return
                
                # Final drill: get last child (matching standalone)
                children_final = list(target_layer.children())
                if len(children_final) > 0:
                     child_name = children_final[-1].__class__.__name__
                     print(f"Drilling down to last child: {child_name}")
                     target_layer = children_final[-1]

                print(f"Final Target Layer for XAI: {target_layer.__class__.__name__}")

                # For RT-DETR, wrap the full model (like standalone script)
                self.wrapped_model = RTDETRWrapper(self.model)
                self.target_layers = [target_layer]
            except Exception as e:
                print(f"Error setting up RT-DETR XAI: {e}")
                import traceback
                traceback.print_exc()
                return

        self.cam = EigenCAM(model=self.wrapped_model, target_layers=self.target_layers)

    def generate_eigencam(self, img_bgr, detections=None):
        """
        Fast generation of EigenCAM heatmap.
        Args:
            img_bgr: Full frame or crop (numpy BGR)
            detections: Optional list of [x1, y1, x2, y2, ...] to check/correct sign flip
        """
        if self.cam is None:
            return None

        # Preprocess: Match Standalone script exactly
        # 1. Resize BGR first (cv2 default)
        target_shape = (640, 640)
        img_resized_bgr = cv2.resize(img_bgr, target_shape)
        
        # 2. Convert to RGB
        img_rgb = cv2.cvtColor(img_resized_bgr, cv2.COLOR_BGR2RGB)
        
        # 3. Create Tensor from RGB (uint8) then float/norm
        input_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        
        # 4. Create Normalized Image for Visualization
        img_normalized = img_rgb.astype(np.float32) / 255.0
        
        # Device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        try:
            p = next(self.model.parameters())
            device = p.device
        except:
            pass
        input_tensor = input_tensor.to(device)
        
        grayscale_cam = self.cam(input_tensor=input_tensor, targets=None)[0, :]
        
        # --- Sign Canonicalization ---
        # PCA direction is arbitrary (v and -v are both eigenvectors).
        # We enforce a canonical orientation where objects (detections) are "hotter" than background.
        # This resolves the "flip" issue fundamental to EigenCAM on video streams.
        if detections is not None and len(detections) > 0:
            h_orig, w_orig = img_bgr.shape[:2]
            # Heatmap is 640x640 usually (output of cam), or size of input_tensor
            # grayscale_cam is normalized 0-1.
            h_cam, w_cam = grayscale_cam.shape
            
            scale_x = w_cam / w_orig
            scale_y = h_cam / h_orig
            
            mask_fg = np.zeros_like(grayscale_cam, dtype=bool)
            
            for det in detections:
                # det is [x1, y1, x2, y2, ...] or object with .tlbr
                if hasattr(det, 'tlbr'):
                    x1, y1, x2, y2 = det.tlbr
                elif isinstance(det, (list, np.ndarray, tuple)):
                     x1, y1, x2, y2 = det[:4]
                else:
                    continue
                    
                x1 = int(x1 * scale_x)
                y1 = int(y1 * scale_y)
                x2 = int(x2 * scale_x)
                y2 = int(y2 * scale_y)
                
                # Clip
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w_cam, x2), min(h_cam, y2)
                
                mask_fg[y1:y2, x1:x2] = True
                
            # Compare mean intensities
            if np.any(mask_fg) and np.any(~mask_fg):
                mean_fg = np.mean(grayscale_cam[mask_fg])
                mean_bg = np.mean(grayscale_cam[~mask_fg])
                
                # If background is hotter, the eigenvector is inverted. Flip it.
                if mean_bg > mean_fg:
                    # print(f"  [XAI] Canonicalizing sign: BG({mean_bg:.2f}) > FG({mean_fg:.2f}). Inverting.")
                    grayscale_cam = 1.0 - grayscale_cam

        # Overlay
        visualization = show_cam_on_image(img_normalized, grayscale_cam, use_rgb=True)
        visualization_bgr = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
        
        return visualization_bgr

    def generate_lime(self, img_bgr, box_fn):
        """
        Placeholder for expensive LIME.
        box_fn: Function that runs inference and returns boxes
        """
        pass
