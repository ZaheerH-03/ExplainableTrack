import os
import sys
import torch
import cv2
import numpy as np
from PIL import Image

# Add DeblurGAN and SRGAN to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

DEBLUR_ROOT = os.path.join(PROJECT_ROOT, "DeblurGAN")
SRGAN_ROOT = os.path.join(PROJECT_ROOT, "SRGAN")
sys.path.append(DEBLUR_ROOT)
sys.path.append(SRGAN_ROOT)

# Imports for DeblurGAN
from DeblurGAN.options.test_options import TestOptions
from DeblurGAN.models.models import create_model
from DeblurGAN.util.util import tensor2im

# Imports for SRGAN
# Note: SRGAN/model.py contains Generator
from SRGAN.model import Generator as SRGenerator

class ImageEnhancer:
    def __init__(self, deblur_weights=None, sr_weights=None, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.deblur_model = None
        self.sr_model = None
        self.deblur_opt = None

        if deblur_weights:
            self._load_deblur_model(deblur_weights)
        
        if sr_weights:
            self._load_sr_model(sr_weights)

    def _load_deblur_model(self, weights_path):
        print(f"Loading DeblurGAN from {weights_path}...")
        # Mock options
        opt = TestOptions().parser.parse_args([])
        opt.model = 'test'
        opt.dataset_mode = 'single'
        opt.learn_residual = True
        opt.resize_or_crop = 'none'
        opt.nThreads = 1
        opt.batchSize = 1
        opt.serial_batches = True
        opt.no_flip = True
        opt.gpu_ids = [0] if self.device.type == 'cuda' else []
        opt.isTrain = False
        opt.model_path = weights_path
        # Missing required args for BaseModel
        opt.checkpoints_dir = './checkpoints' 
        opt.name = 'experiment_name'
        # Additional Model Param Defaults (from TestOptions/BaseOptions)
        opt.input_nc = 3
        opt.output_nc = 3
        opt.ngf = 64
        opt.norm = 'instance'
        opt.which_model_netG = 'resnet_9blocks'
        opt.no_dropout = True
        opt.which_epoch = 'latest'
        opt.fineSize = 256 # Not used for 'none' resize but required by Tensor alloc
        
        self.deblur_model = create_model(opt)
        self.deblur_model.netG.eval()
        self.deblur_opt = opt

    def _load_sr_model(self, weights_path):
        print(f"Loading SRGAN from {weights_path}...")
        self.sr_model = SRGenerator(img_feat=3, n_feats=64, kernel_size=3, num_res_blocks=16)
        self.sr_model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.sr_model.to(self.device)
        self.sr_model.eval()

    def deblur(self, img_bgr):
        """
        Deblurs a BGR image (numpy array).
        """
        if self.deblur_model is None:
            return img_bgr

        # Convert to PIL RGB
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        
        # Resize to multiple of 4
        orig_w, orig_h = img_pil.size
        new_w = (orig_w + 3) // 4 * 4
        new_h = (orig_h + 3) // 4 * 4
        if new_w != orig_w or new_h != orig_h:
            img_pil = img_pil.resize((new_w, new_h), Image.BICUBIC)
            
        # Transform for DeblurGAN (ToTensor + Normalize)
        # Using the standard transform logic from DeblurGAN data loader
        # (Usually ToTensor is enough for test_model if not using full Dataset class, 
        # but let's replicate standard behavior: [0,1] -> [-1, 1])
        t_img = torch.from_numpy(np.array(img_pil)).float() / 255.0
        t_img = t_img.permute(2, 0, 1).unsqueeze(0) # CHW -> BCHW
        t_img = (t_img - 0.5) / 0.5 # Normalize to [-1, 1]
        
        # Inference
        self.deblur_model.set_input({'A': t_img, 'A_paths': ''})
        self.deblur_model.test()
        visuals = self.deblur_model.get_current_visuals()
        result = visuals['fake_B'] # HWC numpy array
        
        # Crop back
        if new_w != orig_w or new_h != orig_h:
            result = result[:orig_h, :orig_w, :]
            
        # Convert back to BGR
        # tensor2im returns uint8 [0,255] RGB
        result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        return result_bgr

    def upscale(self, img_bgr):
        """
        Upscales BGR image using SRGAN (4x).
        """
        if self.sr_model is None:
            return img_bgr
            
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb).float() / 255.0
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.sr_model(img_tensor)
            
        output = output.squeeze(0).cpu().permute(1, 2, 0).numpy()
        output = (output * 255.0).clip(0, 255).astype(np.uint8)
        
        return cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
