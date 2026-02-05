# Project Documentation: Object Tracking & Explainable AI (XAI) Frameowrk

This document provides a comprehensive overview of the project's codebase, focusing on the implementation of object tracking and Explainable AI (XAI) models. It details the models used, the logic behind the implementation, and how the different components interact.

---

## 1. Project Overview

The goal of this project is to implement robust **Multi-Object Tracking (MOT)** pipelines and provide **interpretability** for detecting objects within video streams. The system integrates state-of-the-art object detection models with advanced tracking algorithms and visualization tools to "explain" why models make certain predictions.

### Key Components
1.  **Botsort Scripts**: Custom tracking implementations using BoT-SORT interactively with different detectors (RT-DETR, YOLO).
2.  **Ultralytics Tracking**: Live webcam tracking demos demonstrating real-time capabilities.
3.  **XAI (Explainable AI)**: Modules to visualize model attention (EigenCAM) and feature importance (LIME).

---

## 2. Models & Algorithms

### A. Object Detection Models

#### 1. YOLOv11 (You Only Look Once v11)
*   **Role**: Primary object detector for real-time applications.
*   **Description**: The latest iteration in the YOLO family, known for its speed and accuracy balance. It treats object detection as a single regression problem, predicting bounding boxes and class probabilities directly from full images in one evaluation.
*   **Implementation**: 
    *   **Tracking**: Used in `botsort_scripts/track_yolov11.py`. configured to use **custom weights** (`weights/yolov11_best.pt`) and a specific list of **27 Military/Vehicle Classes**.
    *   **Inference**: A standalone script `Yolo_detection/yolo_infer.py` allows for direct inference on images/videos using the custom model.
    *   **Live Tracking**: The `ultralytics_tracking` examples demonstrate usage with standard models.

#### 2. RT-DETRv4 (Real-Time DEtection TRansformer)
*   **Role**: High-accuracy detector using Transformer architecture.
*   **Description**: An efficient version of the DETR (Detection Transformer) architecture optimized for real-time performance. Unlike YOLO (CNN-based), it uses attention mechanisms to capture global context.
*   **Implementation**: Used in `botsort_scripts/track_rtdetrv4.py`.
*   **Custom Configuration**: 
    *   We utilize a **custom configuration chain** (`configs/rtv4/rtv4_x_custom.yml` -> `configs/dfine/dfine_x_custom.yml` -> `configs/dataset/custom_detection.yml`) to support our 27-class dataset on the **X-Large (X)** model variant. 
    *   **Crucially**, distillation and teacher model parameters are **disabled** for inference to prevent errors when lacking the teacher model weights.

### B. Multi-Object Tracking (MOT) Algorithms

#### 1. BoT-SORT (Bag of Tricks for SORT)
*   **Role**: High-performance tracker.
*   **Description**: An improvement over ByteTrack and DeepSORT. It combines motion cues (Kalman Filter) with appearance information (Re-Identification or ReID features) and camera motion compensation (CMC) to robustly track objects even during occlusions.
*   **Implementation**:
    *   **Offline/Scripted**: In `botsort_scripts`, we explicitly instantiate the `BoT-SORT` tracker class, pass detections (boxes, scores, classes) frame-by-frame, and update the tracks.
    *   **Integrated**: In `ultralytics_tracking`, we use the built-in `model.track` method which wraps BoT-SORT via a configuration file (`bot_sort.yaml`).

#### 2. ByteTrack
*   **Role**: Efficient, association-based tracker.
*   **Description**: Focuses on associating every detection box, not just high-confidence ones. It matches high-confidence detections first, then tries to match remaining low-confidence detections to unmatched tracks, reducing "missing" tracks.
*   **Implementation**: Used in `bytetrack_live.py` via the Ultralytics tracking API (`byte_track.yaml`).

### C. Explainable AI (XAI) Techniques

#### 1. EigenCAM (Eigen Class Activation Mapping)
*   **Role**: Visualization of learned features.
*   **Description**: Computes the principal components (PCA) of the feature maps from a convolutional layer. It visualizes *where* the model is looking without needing class-specific gradients.
*   **Implementation**: Applied to both YOLO and RT-DETR. We assume the backbone/neck layers contain the most relevant spatial features.

#### 2. LIME (Local Interpretable Model-agnostic Explanations)
*   **Role**: Feature importance explanation.
*   **Description**: Perturbs the input image (by masking superpixels) and observes changes in prediction confidence. It fits a simple linear model to these local perturbations to determine which superpixels (regions) contributed most to the detection.
*   **Implementation**: 
    *   **YOLO**: `lime_yolo_detection.py`
    *   **RT-DETR**: `lime_rtdetr_detection.py`

#### 3. DeblurGAN (Image Deblurring)
*   **Role**: Improving image quality before detection/tracking.
*   **Description**: Uses a Generative Adversarial Network (GAN) to remove motion blur from images. It works by having a generator create sharp images and a discriminator trying to distinguish them from real sharp images.
*   **Implementation**: `DeblurGAN/` directory. Optimized to use RESNET-based generator.

---

## 3. Implementation Details & Workflow

### A. Tracking Pipelines (`botsort_scripts/`)

#### 1. `track_rtdetrv4.py` (RT-DETR + BoT-SORT)
*   **Workflow**:
    1.  **Setup**: Loads the RT-DETR model configuration (`configs/rtv4/rtv4_x_custom.yml`) and weights (`weights/best_stg2.pth` or `best_stg1.pth`). Initializes the `BoT-SORT` tracker with specific parameters (ReID enabled, thresholds).
    2.  **Preprocessing**: Reads video frames, resizes them to 640x640, and converts them to tensors.
    3.  **Inference**: Passes the tensor to the RT-DETR model to get raw detections (boxes, scores, labels).
    4.  **Formatting**: The raw output is converted to the format expected by BoT-SORT: `[x1, y1, x2, y2, score, class_id]`.
    5.  **Class Mapping**: Detections are mapped to our **Custom 27-Class List** (e.g., `tank`, `soldier`, `military_truck`...) instead of standard COCO classes.
    6.  **Tracking Update**: `tracker.update(detections, frame)` is called. The tracker matches new detections to existing tracks using ReID features and motion prediction.
    7.  **Visualization**: We iterate through `online_targets` (active tracks), drawing bounding boxes and ID labels on the frame.

#### 2. `track_yolov11.py` (YOLOv11 + BoT-SORT)
*   **Workflow**:
    1.  **Setup**: Loads `weights/yolov11_best.pt` using the `ultralytics` library. Initializes `BoT-SORT` similarly to the RT-DETR script.
    2.  **Inference**: Uses `yolo(frame)` to get detections.
    3.  **Class Mapping**: Uses `CUSTOM_CLASSES` list (27 classes) for proper label display.
    4.  **Integration**: Unlike the "Live" scripts which use the internal tracker, this script *manually* extracts boxes/scores from YOLO results and feeds them into the standalone `BoT-SORT` instance. This allows for finer control over the tracking parameters defined in the `Opt` class.

### B. Live Tracking Demos (`ultralytics_tracking/`)

These scripts differ from `botsort_scripts` by prioritizing simplicity and real-time webcam use.

*   **`botsort_live.py` & `bytetrack_live.py`**:
    *   Instead of manually managing the tracker, these use the high-level `model.track()` API provided by Ultralytics.
    *   **How it works**: You pass `tracker="bot_sort.yaml"` or `tracker="byte_track.yaml"`. The library handles detection, association, and drawing internally.
    *   `persist=True` is crucial here; it tells the model to remember tracks between frames.

### C. XAI Implementation (`xai/`)

#### 1. EigenCAM Implementation
*   **Challenge**: Standard XAI libraries (`pytorch-grad-cam`) expect models to output a simple tensor (logits). Object detectors output complex tuples (boxes, scores, various feature maps).
*   **Solution**: We implemented **Wrapper Classes** (`RTDETRWrapper`, `YOLOWrapper`).
    *   **For RT-DETR**: The wrapper intercepts the output and extracts just the logits (classification scores) if the library requests gradients, or ensures the forward pass completes without error.
    *   **For YOLO**: Since YOLO's detection layer is complex/non-differentiable in a standard way, the wrapper uses **pyTorch hooks** (`register_forward_hook`) to intercept the 4D feature maps directly from an intermediate layer (e.g., the last block of the backbone or neck).
*   **Layer Selection**: The scripts effectively "guess" the best layer to visualize (usually the last layer of the backbone) to show high-level semantic features.

#### 2. LIME Implementation
*   **Logic**:
    1.  **Segmentation**: The image is divided into "superpixels" (clusters of similar pixels).
    2.  **Perturbation**: We generate $N$ samples (e.g., 2000). In each sample, some superpixels are "turned off" (blacked out).
    3.  **Probing**: We run the YOLO detector on all $N$ perturbed images.
    4.  **scoring**: We define a custom `predict_fn`. For each perturbed image, it checks: *Is the original object still detected?* If yes, and the IoU is high, we return a high probability. If the object disappears, low probability.
    5.  **Fitting**: LIME fits a linear model to see which superpixels' presence correlates most with the object being detected. These are highlighted in the final output.

#### 3. LIME for RT-DETR (`lime_rtdetr_detection.py`)
*   **Overview**: Adapts the LIME methodology for the RT-DETR architecture.
*   **Key Differences**:
    *   **Model Loading**: Uses the custom `RT-DETRv4-main` codebase's `YAMLConfig` and `model.deploy()` to load the Transformer-based model properly, rather than Ultralytics' standard loader.
    *   **Prediction Wrapper**: The `predict_fn` handles RT-DETR's specific input requirements (tensors + `orig_target_sizes`) and returns a probability distribution based on **Confidence $\times$ IoU** to quantify detection quality for LIME.
    *   **Visualization**: Generates a 3-panel explanation: Original Detection, LIME Superpixel Overlay, and Importance Heatmap.

#### 4. GradCAM (Gradient-weighted Class Activation Mapping)
*   **Role**: Visualizing specific feature importance via gradients.
*   **Difference from EigenCAM**: EigenCAM is class-agnostic (shows "what objects are here"), whereas GradCAM is class-specific (shows "what features make this a *Car*?").
*   **Implementation**:
    *   **YOLO** (`gradcam_yolo.py`): We assume the regression head is differentiable and maximize the class confidence score for the specific detected box.
    *   **RT-DETR** (`gradcam_rtdetr.py`): We identify the specific **Transformer Query** responsible for the detection and propagate gradients from that query's output back to the backbone.

#### 5. Unified Pipeline logic (`full_pipeline.py`)
*   **Role**: Integrates all components (Tracking, Enhancement, XAI) into a single executable pipeline.
*   **Workflow**:
    1.  **Enhancement (Optional)**: Checks frame blurriness. If blurry, applies DeblurGAN. If enabled and frame is small, applies SRGAN.
    2.  **Tracking**: Runs the selected tracker (RT-DETR or YOLO + BoT-SORT) on the processed frame.
    3.  **Visualization**: Draws bounding boxes and labels on the display frame.
    4.  **XAI (Optional)**: If enabled, triggers EigenCAM explanation for *newly* appeared tracks to explain the detection.
    5.  **Output**: Displays the annotated stream with FPS.

---

## 4. How to Run

### A. Environment Setup

To recreate the local environment, follow these steps:

1.  **Create a new Conda environment** (Python 3.10 recommended):
    ```bash
    conda create -n yolo_bot python=3.10 -y
    conda activate yolo_bot
    ```

2.  **Install Base Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Install Project-Specific Libraries** (YOLO, LIME, etc.):
    ```bash
    pip install ultralytics lime pytorch-grad-cam
    ```

### B. Tracking Scripts

Run these from the project root (`botsort/`):

**RT-DETR Tracking**
```bash
python botsort_scripts/track_rtdetrv4.py
```

**YOLOv11 Tracking**
```bash
python botsort_scripts/track_yolov11.py
```

### B. XAI Scripts

**LIME for YOLO**
```bash
python xai/lime/lime_yolo_detection.py --image <path_to_image> --box_idx 0
```

**LIME for RT-DETR**
*Note: If your path contains wildcards (like `*`), enclose the paths in quotes to avoid shell expansion errors.*

```bash
python xai/lime/lime_rtdetr_detection.py --image "media/test1.png" --box_idx 0
```

**GradCAM for YOLO**
```bash
python xai/gradcam/gradcam_yolo.py --image media/test1.png --box_idx 0
# Outputs to media/gradcam_yolo.jpg by default
```

**GradCAM for RT-DETR**
```bash
python xai/gradcam/gradcam_rtdetr.py --image media/test1.png --box_idx 0
# Outputs to media/gradcam_rtdetr.jpg by default
```

### C. Image Enhancement (SRGAN & DeblurGAN)

**SRGAN (Super Resolution)**
```bash
python SRGAN/inference.py --image media/lowres.jpg --model weights/generator_sr.pth --output media/highres.jpg
```

**DeblurGAN (Deblurring)**
```bash
python DeblurGAN/inference_deblur.py --image media/blurred.jpg --output media/sharp.jpg
```


**EigenCAM**
*Note: These scripts currently use hardcoded paths. You may need to edit the `if __name__ == "__main__":` block in the files to change the input image or model path.*

```bash
# YOLO
python xai/eigen_cam/eigen_cam_yolo.py

# RT-DETR
python xai/eigen_cam/eigen_cam_rtdetr.py
# Optional: View available layers
python xai/eigen_cam/eigen_cam_rtdetr.py --show-layers
```

---

## 5. Repository Modifications & Customizations

### A. RT-DETRv4 Adaptations
To support our custom 27-class Military/Vehicle dataset on the RT-DETRv4-X model without requiring the original teacher model during inference, we implemented a custom configuration hierarchy:

1.  **`configs/dataset/custom_detection.yml`**: Defines the dataset properties, specifically `num_classes: 27` and disabling COCO mapping.
2.  **`configs/dfine/dfine_x_custom.yml`**: Inherits the custom dataset and defines the `HGNetv2` backbone properties for the X-model.
3.  **`configs/rtv4/rtv4_x_custom.yml`**: The top-level inference config. It inherits the `dfine` config but **explicitly disables all distillation and teacher model parameters** (`distill_teacher_dim: 0`, `loss_distill: 0`, `distill_adaptive_params: enabled: False`). This prevents runtime errors when the teacher weights are missing.

We also enhanced `tools/inference/torch_inf.py` to support a custom `--output` directory argument.

### B. BoT-SORT Integration
We utilize the standalone BoT-SORT repository (as a submodule or integrated code) but have modified the `botsort_scripts/` drivers (`track_yolov11.py`, `track_rtdetrv4.py`) to:
1.  Accept our specific model paths.
2.  Map class IDs to our 27-class `CUSTOM_CLASSES` list instead of the standard 80 COCO classes.
3.  Visualize outputs with these custom labels.
