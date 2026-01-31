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
*   **Implementation**: Used in `track_yolov11.py`, `botsort_live.py`, and the XAI modules. We utilize the `ultralytics` library to load pretrained weights (`yolov11m.pt`, `yolov8m.pt`) and perform inference.

#### 2. RT-DETRv4 (Real-Time DEtection TRansformer)
*   **Role**: High-accuracy detector using Transformer architecture.
*   **Description**: An efficient version of the DETR (Detection Transformer) architecture optimized for real-time performance. Unlike YOLO (CNN-based), it uses attention mechanisms to capture global context.
*   **Implementation**: Used in `track_rtdetrv4.py`. We integrate the `RT-DETRv4-main` codebase, manually handling configuration loading (`YAMLConfig`) and model deployment.

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
*   **Implementation**: Applied to YOLO in `lime_yolo_detection.py`.

---

## 3. Implementation Details & Workflow

### A. Tracking Pipelines (`botsort_scripts/`)

#### 1. `track_rtdetrv4.py` (RT-DETR + BoT-SORT)
*   **Workflow**:
    1.  **Setup**: Loads the RT-DETR model configuration (`rtv4_hgnetv2_l_coco.yml`) and weights. Initializes the `BoT-SORT` tracker with specific parameters (ReID enabled, thresholds).
    2.  **Preprocessing**: Reads video frames, resizes them to 640x640, and converts them to tensors.
    3.  **Inference**: Passes the tensor to the RT-DETR model to get raw detections (boxes, scores, labels).
    4.  **Formatting**: The raw output is converted to the format expected by BoT-SORT: `[x1, y1, x2, y2, score, class_id]`.
    5.  **Tracking Update**: `tracker.update(detections, frame)` is called. The tracker matches new detections to existing tracks using ReID features and motion prediction.
    6.  **Visualization**: We iterate through `online_targets` (active tracks), drawing bounding boxes and ID labels on the frame.

#### 2. `track_yolov11.py` (YOLOv11 + BoT-SORT)
*   **Workflow**:
    1.  **Setup**: Loads `yolo11m.pt` using the `ultralytics` library. Initializes `BoT-SORT` similarly to the RT-DETR script.
    2.  **Inference**: Uses `yolo(frame)` to get detections.
    3.  **Integration**: Unlike the "Live" scripts which use the internal tracker, this script *manually* extracts boxes/scores from YOLO results and feeds them into the standalone `BoT-SORT` instance. This allows for finer control over the tracking parameters defined in the `Opt` class.

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
