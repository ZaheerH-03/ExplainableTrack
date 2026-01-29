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
    def __init__(self, model):
        super(RTDETRWrapper, self).__init__()
        self.model = model
        
    def forward(self, x):
        # RT-DETR expects images and orig_target_sizes
        b, c, h, w = x.shape
        orig_target_sizes = torch.tensor([[w, h]]).to(x.device)
        
        try:
            outputs = self.model(x, orig_target_sizes)
        except Exception:
            outputs = self.model(x)

        # PyTorch GradCAM expects a single tensor output (logits) to determine target categories
        # RT-DETR returns (logits, boxes) or similar structure
        if isinstance(outputs, (tuple, list)):
            # Usually the first element is logits [batch, queries, classes]
            return outputs[0]
        elif isinstance(outputs, dict):
            if 'pred_logits' in outputs:
                return outputs['pred_logits']
        
        return outputs

def load_model(config_path, weights_path, device='cuda'):
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
    print("\nAvailable Model Layers:")
    print("-" * 80)
    rtdetr = model.model
    for name, module in rtdetr.named_modules():
        if isinstance(module, (nn.Conv2d, nn.BatchNorm2d, nn.Linear)):
            continue
        if any(x in name for x in ['backbone', 'encoder', 'decoder', 'transformer', 'neck']):
            print(f"Layer: {name} | Type: {module.__class__.__name__}")     
    print("-" * 80)


def generate_eigencam_visualization(
    config_path,
    weights_path,
    image_path,
    output_path='rtdetr_eigencam.jpg',
    target_layer_name='backbone'
):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1. Load Model
    model = load_model(config_path, weights_path, device)
    rtdetr_base = model.model
    
    # 2. Identify Target Layer
    target_layer = None
    
    # Try naive search first
    found_layer = None
    for name, module in rtdetr_base.named_modules():
        if name == target_layer_name:
            found_layer = module
            break
            
    if found_layer:
        print(f"Found layer '{target_layer_name}' ({found_layer.__class__.__name__}).")
        target_layer = found_layer
        
        # Heuristic: If it's the backbone container, we need to dig deeper
        # because the backbone container returns a list/tuple.
        # We want the LAST computational block that returns a Tensor.
        if len(list(target_layer.children())) > 0:
            print("Inspecting children to find suitable Tensor-outputting layer...")
            
            # Check for 'body' (common in TIMM)
            if hasattr(target_layer, 'body'):
                target_layer = target_layer.body
                print(f"Decended into .body ({target_layer.__class__.__name__})")

            # If it's a ModuleList (like 'stages'), pick the last one
            if isinstance(target_layer, nn.ModuleList):
                if len(target_layer) > 0:
                    target_layer = target_layer[-1]
                    print(f"Selected last item in ModuleList ({target_layer.__class__.__name__})")

            # Check for 'stages' child (common in HGNet)
            if hasattr(target_layer, 'stages'):
                target_layer = target_layer.stages
                print(f"Decended into .stages ({target_layer.__class__.__name__})")
                if isinstance(target_layer, nn.ModuleList) and len(target_layer) > 0:
                    target_layer = target_layer[-1]
                    print(f"Selected last stage ({target_layer.__class__.__name__})")

            # General fallback: if still a container with children, verify if it's the one we want
            # If we are effectively at a Stage/Block (Sequential), that is usually fine.
            # But if we are at a generic container that iterates and returns list, we must be careful.
            
            # Check if current target has children
            children = list(target_layer.children())
            if len(children) > 0:
                 # If it's likely a container of stages, take the last one
                 child_name = children[-1].__class__.__name__
                 print(f"Drilling down to last child: {child_name}")
                 target_layer = children[-1]

    # Check fallbacks if specific targeting failed
    if target_layer is None:
        print(f"Layer '{target_layer_name}' not found exact match. Using heuristics...")
        # (Existing fallbacks can be re-added if needed, but the drill-down above is robust)
                     
    if target_layer is None:
        raise ValueError(f"Could not resolve target layer for '{target_layer_name}'")
        
    print(f"Final Target Layer: {target_layer.__class__.__name__}")

    # 3. Setup Wrapper (Model level)
    wrapped_model = RTDETRWrapper(model)
    
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
    PROJECT_ROOT_PATH = Path("/media/aid-pc/My1TB/Zaheer/botsort")
    WEIGHTS = PROJECT_ROOT_PATH / "weights/RTv4-L-hgnet.pth"
    RT_DETR_MAIN = PROJECT_ROOT_PATH / "RT-DETRv4-main"
    CONFIG = RT_DETR_MAIN / "configs/rtv4/rtv4_hgnetv2_l_coco.yml"
    
    IMAGE = "media/devendar.jpg"
    OUTPUT = "media/rtdetr_eigencam.jpg"
    
    if not CONFIG.exists():
        print(f"Config not found: {CONFIG}")
        sys.exit(1)
        
    print_layers_flag = False
    if len(sys.argv) > 1 and sys.argv[1] == '--show-layers':
        print_layers_flag = True
        
    try:
        if print_layers_flag:
            model = load_model(CONFIG, WEIGHTS)
            print_model_layers(model)
        else:
            generate_eigencam_visualization(
                CONFIG, 
                WEIGHTS, 
                IMAGE, 
                OUTPUT, 
                target_layer_name='backbone' 
            )
            
    except Exception as e:
        print(f"Context: {e}")
        import traceback
        traceback.print_exc()
