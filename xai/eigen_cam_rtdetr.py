import sys
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pathlib import Path

# Add RT-DETR root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RT_DETR_ROOT = PROJECT_ROOT / "RT-DETRv4-main"

if str(RT_DETR_ROOT) not in sys.path:
    sys.path.append(str(RT_DETR_ROOT))

try:
    from engine.core import YAMLConfig
except ImportError as e:
    print(f"Error importing RT-DETR modules: {e}")
    print(f"Please ensure {RT_DETR_ROOT} exists and contains the RT-DETR codebase.")
    sys.exit(1)


class RTDETRWrapper(nn.Module):
    """
    Wrapper for RT-DETR model to make it compatible with pytorch-grad-cam.
    """
    def __init__(self, model, target_layer):
        super(RTDETRWrapper, self).__init__()
        self.model = model
        self.target_layer = target_layer
        self.features = None
        
        # Register hook to capture features
        def hook_fn(module, input, output):
            # RT-DETR layers might return various structures
            # We want the tensor feature map
            if isinstance(output, torch.Tensor):
                self.features = output
            elif isinstance(output, (list, tuple)):
                for item in output:
                    if isinstance(item, torch.Tensor):
                        self.features = item
                        break
            
            # If still nothing, check input
            if self.features is None and isinstance(input, (list, tuple)):
                for item in input:
                    if isinstance(item, torch.Tensor):
                        self.features = item
                        break
                        
        self.handle = self.target_layer.register_forward_hook(hook_fn)

    def forward(self, x):
        # RT-DETR expects images and orig_target_sizes
        # We dummy the sizes for visualization as we don't need post-processed boxes
        b, c, h, w = x.shape
        orig_target_sizes = torch.tensor([[w, h]]).to(x.device)
        
        # We manually forward through the underlying model components if needed, 
        # but since we wrapped the whole loaded model (which has a forward method),
        # we can just call it.
        # However, the loaded model's forward expects (images, orig_target_sizes)
        try:
            _ = self.model(x, orig_target_sizes)
        except Exception:
            # Fallback for models that might just take x (e.g. backbone only)
            _ = self.model(x)
            
        if self.features is None:
            raise RuntimeError("Failed to capture features from target layer")
            
        return self.features
    
    def __del__(self):
        if hasattr(self, 'handle'):
            self.handle.remove()


def load_model(config_path, weights_path, device='cuda'):
    """
    Load RT-DETR model using the same logic as track_rtdetrv4.py
    """
    print(f"Loading RT-DETR configuration from {config_path}")
    cfg = YAMLConfig(str(config_path), resume=str(weights_path))
    
    if 'HGNetv2' in cfg.yaml_cfg:
        cfg.yaml_cfg['HGNetv2']['pretrained'] = False
        
    checkpoint = torch.load(weights_path, map_location='cpu')
    if 'ema' in checkpoint:
        state = checkpoint['ema']['module']
    else:
        state = checkpoint['model']
        
    cfg.model.load_state_dict(state)
    
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            # We don't strictly need postprocessor for EigenCAM, but keep it for consistency
            self.postprocessor = cfg.postprocessor.deploy()
            
        def forward(self, images, orig_target_sizes=None):
            outputs = self.model(images)
            if orig_target_sizes is not None:
                outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs
            
    model = Model().to(device)
    model.eval()
    return model


def print_model_layers(model):
    """
    Traverse and print model layers to help select targets.
    """
    print("\nAvailable Model Layers:")
    print("-" * 80)
    
    # We want to show the structure of the underlying RT-DETR model
    # The 'Model' wrapper has .model which is the deployed RT-DETR
    rtdetr = model.model
    
    for name, module in rtdetr.named_modules():
        # Filter for likely interesting layers
        if isinstance(module, (nn.Conv2d, nn.BatchNorm2d, nn.Linear)):
            continue
            
        # specifically look for encoder/decoder/backbone parts
        if any(x in name for x in ['backbone', 'encoder', 'decoder', 'transformer', 'neck']):
            print(f"Layer: {name} | Type: {module.__class__.__name__}")
            
    print("-" * 80)
    print("Recommendation: Look for the last stages of the backbone or encoder output.")
    print("Examples likely to work:")
    print(" - 'backbone.body.layer3' (or similar for HGNet)")
    print(" - 'encoder.layers.5' (Transformer encoder layers)")
    print("-" * 80)


def generate_eigencam_visualization(
    config_path,
    weights_path,
    image_path,
    output_path='rtdetr_eigencam.jpg',
    target_layer_name='backbone' # heuristics will be needed
):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1. Load Model
    model = load_model(config_path, weights_path, device)
    
    # 2. Identify Target Layer
    # Access the internal model structure
    # model.model is the deployed RT-DETR
    rtdetr_base = model.model
    
    target_layer = None
    
    # Try to find the requested layer by name
    for name, module in rtdetr_base.named_modules():
        if name == target_layer_name:
            target_layer = module
            break
            
    # Auto-resolve common targets if not found or generic name given
    if target_layer is None:
        print(f"Target '{target_layer_name}' not found by exact match. Searching heuristics...")
        
        candidates = []
        if 'backbone' in target_layer_name:
            # Try to grab the last stage of backbone
            # RT-DETR backbones often end stored in .body or just list of layers
            # For HGNet/ResNet, usually layer3 or layer4
            candidates = ['backbone.body.layer3', 'backbone.body.layer4', 'backbone.3', 'backbone']
            
        elif 'encoder' in target_layer_name:
             candidates = ['encoder.layers.-1', 'encoder.layers.5']
        
        for cand in candidates:
            try:
                # Naive attribute access for dot notation
                curr = rtdetr_base
                found = True
                for part in cand.split('.'):
                    if part.startswith('-') or part.isdigit():
                        idx = int(part)
                        if hasattr(curr, '__getitem__'):
                            curr = curr[idx]
                        else:
                            found = False
                            break
                    elif hasattr(curr, part):
                        curr = getattr(curr, part)
                    else:
                        found = False
                        break
                if found:
                    target_layer = curr
                    print(f"Auto-selected layer: {cand}")
                    break
            except Exception:
                continue
    
    if target_layer is None:
        # Fallback to recursively printing and asking user, or just erroring
        # For now, let's try to grab the last module of the backbone if possible
        try:
             # This is highly specific to structure; valid for many TIMM backbones
             target_layer = rtdetr_base.backbone
             print("Falling back to entire backbone as target.")
        except:
             raise ValueError(f"Could not find valid target layer for '{target_layer_name}'. Run with --show-layers to see options.")

    print(f"Targeting layer: {target_layer.__class__.__name__}")

    # 3. Setup Wrapper
    wrapped_model = RTDETRWrapper(model, target_layer)
    
    # 4. Initialize EigenCAM
    cam = EigenCAM(model=wrapped_model, target_layers=[target_layer])
    
    # 5. Process Image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
        
    img = cv2.resize(img, (640, 640))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_normalized = img_rgb.astype(np.float32) / 255.0
    
    input_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    input_tensor = input_tensor.to(device)
    
    # 6. Generate CAM
    print("Generating EigenCAM...")
    grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0, :]
    
    visualization = show_cam_on_image(img_normalized, grayscale_cam, use_rgb=True)
    visualization_bgr = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
    
    cv2.imwrite(output_path, visualization_bgr)
    print(f"Visualization saved to {output_path}")


if __name__ == "__main__":
    # Hardcoded paths from previous context, adjustable via args in real app
    PROJECT_ROOT_PATH = Path("/media/aid-pc/My1TB/Zaheer/botsort")
    WEIGHTS = PROJECT_ROOT_PATH / "weights/RTv4-L-hgnet.pth"
    # Note: Config path depends on where RT-DETR root is actually mounted/symlinked
    # Based on track_rtdetrv4.py:
    RT_DETR_MAIN = PROJECT_ROOT_PATH / "RT-DETRv4-main"
    CONFIG = RT_DETR_MAIN / "configs/rtv4/rtv4_hgnetv2_l_coco.yml"
    
    IMAGE = "devendar.jpg" # Using same test image as YOLO script
    OUTPUT = "rtdetr_eigencam.jpg"
    
    if not CONFIG.exists():
        print(f"Config not found: {CONFIG}")
        sys.exit(1)
        
    if not WEIGHTS.exists():
        print(f"Weights not found: {WEIGHTS}")
        # warning only for dev
        
    print_layers_flag = False
    if len(sys.argv) > 1 and sys.argv[1] == '--show-layers':
        print_layers_flag = True
        
    try:
        if print_layers_flag:
            model = load_model(CONFIG, WEIGHTS)
            print_model_layers(model)
        else:
            # You can change target_layer_name here to experiment
            # 'backbone' or specific layers like 'backbone.body.layer3'
            generate_eigencam_visualization(
                CONFIG, 
                WEIGHTS, 
                IMAGE, 
                OUTPUT, 
                target_layer_name='backbone' 
            )
            
            # Try another one common for detection (high level features)
            generate_eigencam_visualization(
                CONFIG, 
                WEIGHTS, 
                IMAGE, 
                "rtdetr_eigencam_encoder.jpg", 
                target_layer_name='encoder' 
            )
            
    except Exception as e:
        print(f"Context: {e}")
        import traceback
        traceback.print_exc()
