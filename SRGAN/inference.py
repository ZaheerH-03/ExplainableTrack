import argparse
import time
import cv2
import torch
import numpy as np
import os
from model import Generator
import torchvision.transforms as transforms
from PIL import Image

def run_inference(image_path, model_path, upscale_factor=4, output_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Model
    print(f"Loading Generator from {model_path}...")
    # Initialize Generator structure
    # Note: num_of_resblocks must match what was used in training (default 16 or 8? check model.py)
    # train.py uses 8 blocks.
    model = Generator(scaling_factor=upscale_factor, num_of_resblocks=8).to(device)
    
    # Load weights
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e:
        print(f"Error loading weights: {e}")
        print("Ensure 'num_of_resblocks' matches the training configuration.")
        return

    model.eval()
    
    # 2. Load Image
    if not os.path.exists(image_path):
        print(f"Error: Image {image_path} not found.")
        return
        
    img = Image.open(image_path).convert('RGB')
    
    # 3. Preprocess
    # SRGAN expects [0, 1] tensor
    to_tensor = transforms.ToTensor()
    img_tensor = to_tensor(img).unsqueeze(0).to(device)
    
    # 4. Inference
    print(f"Running inference on {image_path}...")
    start = time.time()
    with torch.no_grad():
        out = model(img_tensor)
    end = time.time()
    print(f"Done in {end - start:.4f}s")
    
    # 5. Save Output
    # Convert back to image
    out_img = out.squeeze(0).cpu().detach()
    # SRGAN output is likely in [-1, 1] due to Tanh.
    # However, since training data was [0, 1] (not normalized to [-1, 1]), 
    # the model tries to output in [0, 1] range despite Tanh (learning to be positive).
    # Denormalizing (x+1)/2 shifts everything up, causing the white layer.
    # Instead, we just clamp to valid range.
    out_img = torch.clamp(out_img, 0.0, 1.0)
    
    out_pil = transforms.ToPILImage()(out_img)
    
    if output_path is None:
        base, ext = os.path.splitext(image_path)
        output_path = f"{base}_sr_{upscale_factor}x.png"
        
    out_pil.save(output_path)
    print(f"Result saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SRGAN Inference')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--model', type=str, required=True, help='Path to generator weights (.pth)')
    parser.add_argument('--scale', type=int, default=4, help='Upscale factor (default: 4)')
    parser.add_argument('--output', type=str, default=None, help='Output path')
    
    args = parser.parse_args()
    
    run_inference(args.image, args.model, args.scale, args.output)
