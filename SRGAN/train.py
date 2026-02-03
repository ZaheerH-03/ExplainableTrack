from math import log10
import os
import pandas as pd
import torch.optim as optim
import torch.utils.data
import torchvision.utils as utils
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
from dataloader import TrainDatasetFromFolder, ValDatasetFromFolder, display_transform
from model import Generator, Discriminator
from generator_loss_functions import GeneratorLoss

ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).cuda()



import argparse

parser = argparse.ArgumentParser(description='Train SRGAN')
parser.add_argument('--crop_size', default=88, type=int, help='training images crop size')
parser.add_argument('--upscale_factor', default=4, type=int, choices=[2, 4, 8], help='super resolution upscale factor')
parser.add_argument('--num_epochs', default=100, type=int, help='train epoch number')
parser.add_argument('--train_dir', type=str, required=True, help='Path to training dataset (HR)')
parser.add_argument('--val_dir', type=str, required=True, help='Path to validation dataset (HR)')

opt = parser.parse_args()

CROP_SIZE = opt.crop_size
UPSCALE_FACTOR = opt.upscale_factor
NUM_EPOCHS = opt.num_epochs

train_set = TrainDatasetFromFolder(opt.train_dir,
                                   crop_size=CROP_SIZE,
                                   upscale_factor=UPSCALE_FACTOR)
val_set = ValDatasetFromFolder(opt.val_dir,
                               upscale_factor=UPSCALE_FACTOR)

train_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=64, shuffle=True)
val_loader = DataLoader(dataset=val_set, num_workers=4, batch_size=1, shuffle=False)

# ==========================
# Model Setup
# ==========================
netG = Generator(scaling_factor=UPSCALE_FACTOR, num_of_resblocks=8)
netD = Discriminator()
generator_criterion = GeneratorLoss()

if torch.cuda.is_available():
    netG.cuda()
    netD.cuda()
    generator_criterion.cuda()

# ==========================
# Optimizers
# ==========================
optimizerG = optim.Adam(netG.parameters(), lr=1e-4, betas=(0.9, 0.999))
optimizerD = optim.Adam(netD.parameters(), lr=1e-4, betas=(0.9, 0.999))

# ==========================
# Logging setup
# ==========================
results = {'d_loss': [], 'g_loss': [], 'd_score': [], 'g_score': [], 'psnr': [], 'ssim': []}
os.makedirs("epochs", exist_ok=True)

# ==========================
# Training Loop
# ==========================
for epoch in range(1, NUM_EPOCHS + 1):
    train_bar = tqdm(train_loader)
    running_results = {'batch_sizes': 0, 'd_loss': 0, 'g_loss': 0, 'd_score': 0, 'g_score': 0}

    netG.train()
    netD.train()

    for data, target in train_bar:
        batch_size = data.size(0)
        running_results['batch_sizes'] += batch_size

        real_img = target
        if torch.cuda.is_available():
            real_img = real_img.float().cuda()
            data = data.float().cuda()

        # Generate fake image
        fake_img = netG(data)

        # ---------------------------
        # Train Generator
        # ---------------------------
        optimizerG.zero_grad()
        g_loss = generator_criterion(fake_img, fake_img, real_img)
        g_loss.backward()
        optimizerG.step()

        # ---------------------------
        # Train Discriminator
        # ---------------------------
        real_out = netD(real_img).mean()
        fake_out = netD(fake_img.detach()).mean()
        optimizerD.zero_grad()
        d_loss = 1 - real_out + fake_out
        d_loss.backward()
        optimizerD.step()

        # Metrics
        running_results['g_loss'] += g_loss.item() * batch_size
        running_results['d_loss'] += d_loss.item() * batch_size
        running_results['d_score'] += real_out.item() * batch_size
        running_results['g_score'] += fake_out.item() * batch_size

        train_bar.set_description(
            desc='[%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f' % (
                epoch, NUM_EPOCHS,
                running_results['d_loss'] / running_results['batch_sizes'],
                running_results['g_loss'] / running_results['batch_sizes'],
                running_results['d_score'] / running_results['batch_sizes'],
                running_results['g_score'] / running_results['batch_sizes']
            ))

    # ==========================
    # Validation
    # ==========================
    netG.eval()
    out_path = 'training_results/SRF_' + str(UPSCALE_FACTOR)
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    with torch.no_grad():
        val_bar = tqdm(val_loader)
        valing_results = {'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'batch_sizes': 0}
        val_images = []

        for val_lr, val_hr_restore, val_hr in val_bar:
            batch_size = val_lr.size(0)
            valing_results['batch_sizes'] += batch_size

            if torch.cuda.is_available():
                val_lr = val_lr.float().cuda()
                val_hr = val_hr.float().cuda()

            sr = netG(val_lr)

            batch_mse = ((sr - val_hr) ** 2).data.mean()
            valing_results['mse'] += batch_mse * batch_size
            batch_ssim = ssim_metric(sr, val_hr).item()
            valing_results['ssims'] += batch_ssim * batch_size
            valing_results['psnr'] = 10 * log10((val_hr.max() ** 2) / (valing_results['mse'] / valing_results['batch_sizes']))
            valing_results['ssim'] = valing_results['ssims'] / valing_results['batch_sizes']

            val_bar.set_description(
                desc='[converting LR→SR] PSNR: %.4f dB SSIM: %.4f' %
                (valing_results['psnr'], valing_results['ssim'])
            )

            val_images.extend([
                display_transform()(val_hr_restore.squeeze(0)),
                display_transform()(val_hr.data.cpu().squeeze(0)),
                display_transform()(sr.data.cpu().squeeze(0))
            ])

        # Save validation results
        val_images = torch.stack(val_images)
        val_images = torch.chunk(val_images, val_images.size(0) // 15)
        val_save_bar = tqdm(val_images, desc='[saving training results]')
        index = 1
        for image in val_save_bar:
            image = utils.make_grid(image, nrow=3, padding=5)
            utils.save_image(image, f'{out_path}/epoch_{epoch}_index_{index}.png', padding=5)
            index += 1

    # ==========================
    # Save Models + Metrics
    # ==========================
    torch.save(netG.state_dict(), f'epochs/netG_epoch_{UPSCALE_FACTOR}_{epoch}.pth')
    torch.save(netD.state_dict(), f'epochs/netD_epoch_{UPSCALE_FACTOR}_{epoch}.pth')

    results['d_loss'].append(running_results['d_loss'] / running_results['batch_sizes'])
    results['g_loss'].append(running_results['g_loss'] / running_results['batch_sizes'])
    results['d_score'].append(running_results['d_score'] / running_results['batch_sizes'])
    results['g_score'].append(running_results['g_score'] / running_results['batch_sizes'])
    results['psnr'].append(valing_results['psnr'])
    results['ssim'].append(valing_results['ssim'])

    if epoch % 10 == 0 and epoch != 0:
        stats_path = 'statistics/'
        if not os.path.exists(stats_path):
            os.makedirs(stats_path)
        data_frame = pd.DataFrame(
            data={
                'Loss_D': results['d_loss'],
                'Loss_G': results['g_loss'],
                'Score_D': results['d_score'],
                'Score_G': results['g_score'],
                'PSNR': results['psnr'],
                'SSIM': results['ssim']
            },
            index=range(1, epoch + 1)
        )
        data_frame.to_csv(f'{stats_path}/srf_{UPSCALE_FACTOR}_train_results.csv', index_label='Epoch')
