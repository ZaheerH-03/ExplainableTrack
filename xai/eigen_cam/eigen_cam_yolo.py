"""
EigenCAM Visualization for YOLOv11

This script computes and visualizes EigenCAM heatmaps for YOLO models. It allows 
interpreting which regions of the image contribute most to the model's feature representations.

Features:
- Custom YOLO wrapper for feature extraction hooks.
- Layer selection suggestions to avoid incompatible Detect layers.
- Supports visualizing feature maps from backbone and neck layers.
"""

import cv2
import numpy as np
import torch
import os
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from ultralytics import YOLO


class YOLOWrapper(torch.nn.Module):
    """
    Wrapper for YOLO model to make it compatible with pytorch-grad-cam.
    
    Problem: Standard YOLO forward pass returns predictions (boxes, scores).
    EigenCAM needs access to intermediate 4D feature maps (B, C, H, W).
    
    Solution: This wrapper registers a forward hook on the target layer to intercept 
    the feature maps during interference and return them instead of the final predictions.
    """
    def __init__(self, yolo_model, target_layer):
        super(YOLOWrapper, self).__init__()
        self.model = yolo_model.model
        self.target_layer = target_layer
        self.features = None
        
        # Register hook to capture features from target layer
        def hook_fn(module, input, output):
            # Handle both tensor and tuple/list outputs
            if isinstance(output, (list, tuple)):
                # For YOLO Detect layer, output is a tuple
                # We want the feature maps, not the final predictions
                # Try to get the first element that is a tensor
                for item in output:
                    if isinstance(item, torch.Tensor):
                        self.features = item
                        break
                # If no tensor found in tuple, try the input instead
                if self.features is None and isinstance(input, (list, tuple)):
                    for item in input:
                        if isinstance(item, torch.Tensor):
                            self.features = item
                            break
            else:
                self.features = output
        
        self.target_layer.register_forward_hook(hook_fn)
        
    def forward(self, x):
        # Run forward pass to trigger hooks
        _ = self.model(x)
        # Return the captured features from the target layer
        if self.features is None:
            raise RuntimeError("Failed to capture features from target layer")
        return self.features


def print_model_layers(model_path):
    """
    Print all layers in the YOLO model to help select appropriate target layers.
    
    Args:
        model_path: Path to YOLO weights
    """
    yolo_model = YOLO(model_path)
    print("\nAvailable layers in YOLO model:")
    print("-" * 60)
    for idx, layer in enumerate(yolo_model.model.model):
        print(f"Index: {idx:3d} | {idx - len(yolo_model.model.model):3d} | {layer.__class__.__name__}")
    print("-" * 60)
    print("\nRecommended layers for GradCAM:")
    print("  - Use layers from the backbone (indices 0-9)")
    print("  - Avoid the last layer (Detect) as it returns complex outputs")
    print("  - Good choices: -3, -4, -5 (neck layers)")
    print()


def generate_eigencam_visualization(
    model_path='/media/aid-pc/My1TB/Zaheer/Explainable_Object_tacking/weights/yolov11_best.pt',
    image_path='media/input_images_xai/tank.jpg',
    output_path='media/eigen_cam_outputs/yolo_eigencam_explanation1.jpg',
    target_layer_idx=-2
):
    """
    Generate EigenCAM visualization for YOLO model.
    
    Args:
        model_path: Path to YOLO weights
        image_path: Path to input image
        output_path: Path to save the visualization
        target_layer_idx: Index of target layer (default: -2). 
                          Negative indices count from the end.
                          NOTE: Avoid the final 'Detect' layer.
    """
    
    # Check if image exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Check if model exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    print(f"Loading YOLO model from: {model_path}")
    yolo_model = YOLO(model_path)
    
    # Select the target layer first
    # For YOLO models, the backbone's last layers are usually good choices
    target_layer = yolo_model.model.model[target_layer_idx]
    
    # Check if it's the Detect layer and warn the user
    if target_layer.__class__.__name__ == 'Detect':
        print(f"\n⚠️  WARNING: Layer {target_layer_idx} is a Detect layer, which may not work well with GradCAM.")
        print("    The Detect layer outputs complex tuples that are incompatible with GradCAM.")
        print("    Recommended: Use layer -3, -4, or -5 instead for better results.")
        print("    Automatically switching to layer -3...\n")
        target_layer_idx = -3
        target_layer = yolo_model.model.model[target_layer_idx]
    
    print(f"Using target layer: {target_layer_idx} - {target_layer.__class__.__name__}")
    
    # Wrap the model with the target layer
    wrapped_model = YOLOWrapper(yolo_model, target_layer)
    
    # Pass target layer as a list to EigenCAM
    target_layers = [target_layer]
    
    # Initialize EigenCAM
    cam = EigenCAM(model=wrapped_model, target_layers=target_layers)
    
    # Prepare the image
    print(f"Loading image from: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    # Resize to model input size (standard YOLO size)
    img_resized = cv2.resize(img, (640, 640))
    
    # Convert to RGB (OpenCV loads as BGR)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    
    # Normalize to [0, 1]
    img_normalized = img_rgb.astype(np.float32) / 255.0
    
    # Prepare tensor for model (CHW format)
    input_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    
    # Move to same device as model
    device = next(wrapped_model.parameters()).device
    input_tensor = input_tensor.to(device)
    
    print("Generating EigenCAM heatmap...")
    # Generate heatmap
    grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0, :]
    
    # Overlay heatmap on image
    visualization = show_cam_on_image(img_normalized, grayscale_cam, use_rgb=True)
    
    # Convert back to BGR for saving with OpenCV
    visualization_bgr = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
    
    # Save the result
    cv2.imwrite(output_path, visualization_bgr)
    print(f"Visualization saved to: {output_path}")
    
    return visualization_bgr


if __name__ == "__main__":
    # Uncomment the line below to see all available layers
    print_model_layers('/media/aid-pc/My1TB/Zaheer/Explainable_Object_tacking/weights/yolov11_best.pt')
    
    # Example usage
    try:
        generate_eigencam_visualization(
            model_path='/media/aid-pc/My1TB/Zaheer/Explainable_Object_tacking/weights/yolov11_best.pt',
            image_path='media/input_images_xai/tank.jpg',
            output_path='media/eigen_cam_outputs/yolo_eigencam_explanation1.jpg',
            target_layer_idx=-3  # Changed from -1 to -3 to avoid Detect layer
        )
        # generate_eigencam_visualization(
        #     model_path='/media/aid-pc/My1TB/Zaheer/botsort/weights/yolo11m.pt',
        #     image_path='devendar.jpg',
        #     output_path='yolo_eigencam_explanation4.jpg',
        #     target_layer_idx=-4  # Changed from -1 to -3 to avoid Detect layer
        # )
        # generate_eigencam_visualization(
        #     model_path='/media/aid-pc/My1TB/Zaheer/botsort/weights/yolo11m.pt',
        #     image_path='devendar.jpg',
        #     output_path='yolo_eigencam_explanation5.jpg',
        #     target_layer_idx=-5  # Changed from -1 to -3 to avoid Detect layer
        # )
    except Exception as e:
        print(f"Error: {e}")
        print("\nPlease ensure:")
        print("1. The image file exists in the current directory")
        print("2. The model file exists at the specified path")
        print("3. Required packages are installed: pip install opencv-python pytorch-grad-cam ultralytics")
        print("\nTo see available layers, uncomment the print_model_layers() line in the code.")
