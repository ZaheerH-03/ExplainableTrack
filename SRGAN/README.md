# Super-Resolution GAN (SRGAN) — PyTorch Implementation

A **clean, research-aligned implementation of SRGAN** for single-image super-resolution using PyTorch. This project reproduces the core ideas from the original SRGAN paper while keeping the code modular, readable, and extensible for experimentation and research.

The implementation focuses on **perceptual quality**, combining adversarial learning with VGG-based perceptual loss and pixel-level reconstruction objectives.

---

## 🚀 Project Overview

Single Image Super-Resolution (SISR) aims to reconstruct a high-resolution (HR) image from its low-resolution (LR) counterpart. Traditional pixel-wise losses (e.g., MSE) often yield over-smoothed outputs. **SRGAN** addresses this by optimizing for perceptual similarity using adversarial training and deep feature losses.

This repository provides:

* A **residual-based Generator** with PixelShuffle upsampling
* A **CNN Discriminator** trained adversarially
* **VGG16-based perceptual loss** for high-frequency detail preservation
* A complete **training + validation pipeline** using the DIV2K dataset

---

## 🧠 Model Architecture

This section provides a **clear, layer-by-layer architectural visualization** of both the Generator and Discriminator, closely aligned with the actual implementation in this repository and the original SRGAN design philosophy.

---

## 🧠 Generator Network (G)

### Very Deep Residual Generator with Sub-Pixel Upsampling

**Input:** Low-resolution RGB image
**Example:** `3 × 64 × 64`
**Output:** Super-resolved image
**Example (×4):** `3 × 256 × 256`

---

### 🌱 Input Feature Extraction

```text
Conv2d   : 3 → 64, kernel=9×9, stride=1, padding=4
Activation: PReLU
Output   : 64 × H × W
```

Large receptive fields capture global context early in the network.

---

### 🔁 Residual Learning Trunk (B Residual Blocks)

Each residual block:

```text
Conv2d   : 64 → 64, kernel=3×3, stride=1, padding=1
BatchNorm
PReLU
Conv2d   : 64 → 64, kernel=3×3, stride=1, padding=1
BatchNorm
Skip     : Input + Output
```

* Preserves spatial resolution
* Enables stable deep training
* Refines mid/high-frequency features

---

### 🔄 Post-Residual Fusion

```text
Conv2d   : 64 → 64, kernel=3×3
BatchNorm
Skip     : Added to features from input block
```

This global skip connection improves gradient flow and stabilizes convergence.

---

### 📈 Upsampling via Sub-Pixel Convolution

To achieve ×4 super-resolution, **two upsampling blocks (×2 each)** are used.

Each block:

```text
Conv2d      : 64 → 256
PixelShuffle: scale=2
PReLU
```

PixelShuffle rearranges channel information into spatial resolution, avoiding checkerboard artifacts.

---

### 🎯 Output Reconstruction

```text
Conv2d   : 64 → 3, kernel=9×9, stride=1, padding=4
Activation: Tanh
```

* Produces final RGB image
* Output normalized to **[-1, 1]**

---

### ✅ Generator Summary

| Stage          | Operation          | Key Details  |
| -------------- | ------------------ | ------------ |
| Input          | Conv + PReLU       | 3 → 64 (9×9) |
| Residual Trunk | Residual Blocks ×B | 64 channels  |
| Fusion         | Conv + BN + Skip   | Global skip  |
| Upsampling     | PixelShuffle ×2    | ×4 SR        |
| Output         | Conv + Tanh        | 64 → 3       |

---

## ⚡ Discriminator Network (D)

### VGG-Style CNN Binary Classifier

**Input:** HR or SR image patch
**Example:** `3 × 128 × 128`

**Output:** Scalar probability → *Real vs Fake*

---

### 📊 Convolutional Feature Extractor

| Layer | Filters | Kernel | Stride | Activation     |
| ----- | ------- | ------ | ------ | -------------- |
| Conv1 | 64      | 3×3    | 1      | LeakyReLU(0.2) |
| Conv2 | 64      | 3×3    | 2      | LeakyReLU(0.2) |
| Conv3 | 128     | 3×3    | 1      | LeakyReLU(0.2) |
| Conv4 | 128     | 3×3    | 2      | LeakyReLU(0.2) |
| Conv5 | 256     | 3×3    | 1      | LeakyReLU(0.2) |
| Conv6 | 256     | 3×3    | 2      | LeakyReLU(0.2) |
| Conv7 | 512     | 3×3    | 1      | LeakyReLU(0.2) |
| Conv8 | 512     | 3×3    | 2      | LeakyReLU(0.2) |

* No pooling layers
* Downsampling via strided convolutions
* Progressive channel expansion

---

### 🔹 Classification Head

```text
AdaptiveAvgPool2d(1×1)
Conv2d: 512 → 1024, kernel=1×1
LeakyReLU(0.2)
Conv2d: 1024 → 1, kernel=1×1
Sigmoid
```

Produces a scalar realism score per image.

---

### ✅ Discriminator Summary

| Component    | Description           |
| ------------ | --------------------- |
| Backbone     | 8-layer VGG-style CNN |
| Downsampling | Strided convolutions  |
| Activations  | LeakyReLU (α=0.2)     |
| Output       | Sigmoid probability   |

---

### 🔍 Key Architectural Insights

* Generator uses **PReLU** for adaptive non-linearity
* Discriminator uses **LeakyReLU** for stable gradients
* **Residual learning** enables deep feature refinement
* **PixelShuffle** avoids interpolation artifacts
* No pooling layers — preserves spatial detail

---

### Discriminator (D)

* Fully convolutional CNN
* Progressive channel expansion: 64 → 512
* Strided convolutions for downsampling
* Global average pooling
* Binary real/fake prediction

The discriminator guides the generator to produce **photo-realistic textures** rather than pixel-perfect averages.

---

## 🎯 Loss Functions

The generator is trained using a **weighted combination of four losses**:

1. **Pixel (Content) Loss**
   Mean Squared Error (MSE) between SR and HR images

2. **Adversarial Loss**
   Encourages realism via discriminator feedback

3. **Perceptual Loss**
   MSE between VGG16 feature maps (up to layer 31)

4. **Total Variation (TV) Loss**
   Regularizes spatial smoothness

**Overall Generator Objective:**

```
L_G = L_pixel + 0.001·L_adv + 0.006·L_perceptual + 2e−8·L_TV
```

This balance prioritizes visual fidelity over raw PSNR.

---

## 📂 Project Structure

```
SRGAN/
│
├── model.py
│   ├── ResBlock
│   ├── UpsampleBlock
│   ├── Generator
│   └── Discriminator
│
├── dataloader.py
│   ├── TrainDatasetFromFolder
│   ├── ValDatasetFromFolder
│   └── TestDatasetFromFolder
│
├── generator_loss_functions.py
│   ├── GeneratorLoss (VGG + GAN + TV)
│   └── TVLoss
│
├── train.py
│   └── Full training & validation pipeline
│
└── requirements.txt
```

Each module is intentionally decoupled to allow easy experimentation and replacement.

---

## 📊 Dataset

* **DIV2K** high-resolution image dataset
* Random cropping for training
* Bicubic downsampling for LR generation

Expected directory structure:

```
/data/
├── DIV2K_train_HR/
│   └── DIV2K_train_HR/
└── DIV2K_valid_HR/
    └── DIV2K_valid_HR/
```

---

## 🏋️ Training Details

**Hyperparameters**

* Upscale factor: ×4
* Crop size: 88×88
* Residual blocks: 8
* Batch size: 64
* Optimizer: Adam (β₁=0.9, β₂=0.999)
* Learning rate: 1e−4
* Epochs: 100

**Metrics Tracked**

* Generator loss
* Discriminator loss
* PSNR
* SSIM

Validation images are periodically saved for qualitative inspection.

---

## 📈 Evaluation Metrics

* **PSNR (Peak Signal-to-Noise Ratio)** — pixel fidelity
* **SSIM (Structural Similarity Index)** — perceptual structure

Note: SRGAN optimizes perceptual quality, so PSNR may be lower than purely MSE-trained models, but visual realism is significantly improved.

---

## 🖼️ Visual Results

During validation, results are saved as triplets:

```
[ Bicubic Upscaled                 | Ground Truth |                    SRGAN Output ]
```
![Alt text](results/epoch_100_index_1.png)


These qualitative comparisons are critical for assessing GAN-based super-resolution.

---

## 📈 Training Logs & Convergence Analysis

### Final Quantitative Results (×4 SR)

* **PSNR:** ≈ **24.23 dB**
* **SSIM:** ≈ **0.713**

These values fall squarely within the **expected and realistic performance range** for SRGAN-style models trained on common super-resolution benchmarks (e.g., DIV2K). The results are *strong but not artificially inflated*, indicating healthy training dynamics.

---

### Metric Improvements Over Training

* **PSNR gain:** +13.7 dB (Epoch 1 → 100)
* **SSIM gain:** +0.41 (Epoch 1 → 100)

Both metrics show **rapid early improvement** followed by **smooth saturation**, suggesting effective learning of low-frequency structure early on and gradual refinement of perceptual details.

---

### Convergence Behavior

* PSNR and SSIM **plateau smoothly** in the final training phase
* SSIM variation in the last 20 epochs ≈ **0.0055**, indicating stability
* Marginal gains beyond epoch ~90 are minimal

This behavior strongly suggests the model has **fully converged** by the end of training.

---

### Discriminator–Generator Dynamics

**Discriminator Confidence**

* **D(x) ≈ 0.96** → high confidence on real HR images
* **D(G(z)) ≈ 0.09** → generator outputs usually detected as fake

This is a **typical and expected regime for BCE-based GAN training** in SRGAN, where the discriminator remains strong while the generator focuses on perceptual realism rather than fooling the discriminator completely.

---

### Loss Trends Interpretation

* **Discriminator Loss:** Starts high (~0.8) and decreases steadily, showing rapid discrimination learning
* **Generator Loss:** Drops early (~0.1 → ~0.005) and remains stable

The absence of oscillations or divergence indicates:

* Stable adversarial training
* No mode collapse
* No exploding gradients

---

### Overall Training Assessment

* The SRGAN reaches a **stable equilibrium** between Generator and Discriminator
* Image quality metrics improve consistently without instability
* The training duration (100 epochs) is sufficient; further training would yield diminishing returns

Overall, the observed trends confirm a **well-balanced, correctly implemented SRGAN training pipeline** producing perceptually meaningful super-resolution results.

---

## 🔬 Research Alignment

This implementation closely follows:

**Ledig et al., 2017**
*Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network*

Key design choices such as residual learning, perceptual loss, and adversarial training are faithful to the original paper.

**Code Reference & Inspiration**

* SRGAN PyTorch Reference Implementation by Donghee Han

---

## 🛠️ Future Improvements

* Add RRDB-based generator (ESRGAN)
* Mixed-precision training (AMP)
* Multi-scale discriminators
* LPIPS perceptual metric
* Inference-only script for deployment

---

## 📜 License

This project is intended for **research and educational use**. Please ensure proper attribution if used in academic or commercial work.

---

## ⭐ Acknowledgements

* PyTorch
* torchvision
* DIV2K Dataset
* SRGAN authors
* Donghee Han — SRGAN PyTorch reference implementation

If u find this repo helpful pls consider giving it a ⭐ on github.