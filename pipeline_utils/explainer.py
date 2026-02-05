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
            return outputs[0]  # Usually logits [batch, queries, classes]
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
                
                if hasattr(rtdetr_deploy, 'backbone'):
                    target_layer = rtdetr_deploy.backbone
                    
                    # Check for 'body' (TIMM backbones)
                    if hasattr(target_layer, 'body'):
                        target_layer = target_layer.body
                    
                    # If ModuleList, take last
                    if isinstance(target_layer, torch.nn.ModuleList):
                        if len(target_layer) > 0:
                            target_layer = target_layer[-1]
                    
                    # Check for 'stages' (HGNet)
                    if hasattr(target_layer, 'stages'):
                        target_layer = target_layer.stages
                        if isinstance(target_layer, torch.nn.ModuleList) and len(target_layer) > 0:
                            target_layer = target_layer[-1]
                else:
                    print("Explainer Warning: Could not find 'backbone' in RT-DETR. XAI may fail.")
                    return
                
                # Final drill: get last child (matching standalone)
                children_final = list(target_layer.children())
                if len(children_final) > 0:
                    target_layer = children_final[-1]

                # For RT-DETR, wrap the full model (like standalone script)
                self.wrapped_model = RTDETRWrapper(self.model)
                self.target_layers = [target_layer]
            except Exception as e:
                print(f"Error setting up RT-DETR XAI: {e}")
                return

        self.cam = EigenCAM(model=self.wrapped_model, target_layers=self.target_layers)

    def generate_eigencam(self, img_bgr):
        """
        Fast generation of EigenCAM heatmap.
        Args:
            img_bgr: Full frame or crop (numpy BGR)
        """
        if self.cam is None:
            return None

        # Preprocess: BGR to RGB first
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # Resize to standard size (speed/consistency)
        input_size = (640, 640)
        img_resized = cv2.resize(img_rgb, input_size)
        
        # Normalize [0, 1]
        img_normalized = img_resized.astype(np.float32) / 255.0
        
        # Tensor
        input_tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).unsqueeze(0).float()
        
        # Device
        # Robustly ensure we match the model's device
        # Default to CUDA if available, as model is likely on GPU
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Verify if model is weirdly on CPU despite CUDA being available
        try:
            # Check first parameter
            p = next(self.model.parameters())
            device = p.device
        except:
            pass # Keep default
        input_tensor = input_tensor.to(device)
        # Generate
        grayscale_cam = self.cam(input_tensor=input_tensor, targets=None)[0, :]
        # Overlay
        visualization = show_cam_on_image(img_normalized, grayscale_cam, use_rgb=True)
        visualization_bgr = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
        
        # Resize back to original if input wasn't 640x640 (preserving aspect? No, cam usually distorts, ignore for thumbnail)
        return visualization_bgr

    def generate_lime(self, img_bgr, box_fn):
        """
        Placeholder for expensive LIME.
        box_fn: Function that runs inference and returns boxes
        """
        pass
