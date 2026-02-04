import os
import torch
import shutil
import tempfile
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.util import tensor2im
from PIL import Image

def run_deblur():
    # 1. Setup options
    test_opt = TestOptions()
    test_opt.initialize()
    
    # Custom args
    test_opt.parser.add_argument('--image', type=str, required=True, help='Path to input image')
    test_opt.parser.add_argument('--output', type=str, help='Output path')
    test_opt.parser.add_argument('--model_path', type=str, help='Direct path to net_G.pth weights')
    
    # Parse
    opt = test_opt.parser.parse_args()
    opt.isTrain = False
    
    # Check if user passed model path via --model
    if opt.model and opt.model.endswith('.pth') and os.path.exists(opt.model):
        if not opt.model_path:
             print(f"Note: usage of --model for weight path detected. Moving '{opt.model}' to model_path.")
             opt.model_path = opt.model
    
    # Defaults for high quality inference (from README and test.py)
    opt.model = 'test'
    opt.dataset_mode = 'single'
    opt.learn_residual = True # Crucial for DeblurGAN
    opt.serial_batches = True
    opt.no_flip = True
    opt.nThreads = 1
    opt.batchSize = 1
    
    # Disable the default cropping/resizing in the data loader
    # to prevent "cutting the images"
    opt.resize_or_crop = 'none'
    
    # Standard GPU parsing
    str_ids = opt.gpu_ids.split(',')
    opt.gpu_ids = []
    for str_id in str_ids:
        try:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        except ValueError:
            pass
    
    if len(opt.gpu_ids) > 0 and torch.cuda.is_available():
        torch.cuda.set_device(opt.gpu_ids[0])
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # 2. Pre-process Image to handle multiples of 4
    if not os.path.exists(opt.image):
        print(f"Error: Image {opt.image} not found.")
        return
        
    img = Image.open(opt.image).convert('RGB')
    orig_w, orig_h = img.size
    
    # The ResNet architecture requires width and height to be multiples of 4
    new_w = (orig_w + 3) // 4 * 4
    new_h = (orig_h + 3) // 4 * 4
    
    if new_w != orig_w or new_h != orig_h:
        print(f"Resizing input from {orig_w}x{orig_h} to {new_w}x{new_h} (multiple of 4)...")
        img = img.resize((new_w, new_h), Image.BICUBIC)

    # Prepare Temp Data Root
    with tempfile.TemporaryDirectory() as tmp_dir:
        img_name = os.path.basename(opt.image)
        # Save the correctly-sized image to the temp folder
        img.save(os.path.join(tmp_dir, img_name))
        opt.dataroot = tmp_dir

        # 3. Initialize Model
        model = create_model(opt)

        # 4. Use Official DataLoader
        data_loader = CreateDataLoader(opt)
        dataset = data_loader.load_data()
        
        # 5. Process
        for i, data in enumerate(dataset):
            model.set_input(data)
            model.test()
            
            visuals = model.get_current_visuals()
            deblurred_img_numpy = visuals['fake_B']
            
            # Crop back to original size if we resized/padded
            if new_w != orig_w or new_h != orig_h:
                # deblurred_img_numpy is HWC
                deblurred_img_numpy = deblurred_img_numpy[:orig_h, :orig_w, :]
            
            # Save
            if opt.output:
                # If path is a directory, ends with separator, or has no extension, treat as directory
                if os.path.isdir(opt.output) or opt.output.endswith(os.sep) or not os.path.splitext(opt.output)[1]:
                    os.makedirs(opt.output, exist_ok=True)
                    save_path = os.path.join(opt.output, f"{os.path.splitext(img_name)[0]}_deblurred.png")
                else:
                    save_path = opt.output
            else:
                save_path = f"{os.path.splitext(img_name)[0]}_deblurred.png"
            
            Image.fromarray(deblurred_img_numpy).save(save_path)
            print(f"Result saved to {save_path}")
            break 

if __name__ == "__main__":
    run_deblur()
