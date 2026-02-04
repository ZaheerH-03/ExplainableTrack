# ExplainableTrack: Explainable Multi-Object Tracking

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive repository for state-of-the-art Multi-Object Tracking (MOT) integrated with Explainable AI (XAI) techniques. This project combines **BoT-SORT** with the latest **YOLOv11** and **RT-DETRv4** detectors, providing tools to not only track objects but also understand *why* the models make their decisions.

---

## 🚀 Key Features

-   **State-of-the-Art Tracking**: Integration of [BoT-SORT](https://github.com/NirAharon/BoT-SORT) for robust multi-object tracking.
-   **Modern Detectors**: Support for **YOLOv11** via [Ultralytics](https://github.com/ultralytics/ultralytics) and **RT-DETRv4** via the included custom `RT-DETRv4-main` codebase.
-   **Explainable AI (XAI)**:
    -   **LIME (Local Interpretable Model-agnostic Explanations)**: Highlight superpixels most responsible for a specific detection.
    -   **Eigen-CAM**: Visualize class-activation maps to see where the model is looking in the image or video.
-   **Re-Identification (ReID)**: High-performance ReID using `fast-reid` models.
-   **Live & Batch Processing**: Scripts for real-time tracking from webcams and batch processing of video files.
-   **Custom Military/Vehicle Dataset Support**: Configured for 27 specific classes (e.g., Tank, APC, Soldier) using custom trained models.

---

## 📁 Project Structure

```text
├── BoT-SORT               # Original BoT-SORT repository with patches and fixes as a submodule/dependency
├── RT-DETRv4         # RT-DETRv4 source code and utilities
├── botsort_scripts        # Main entry points for YOLOv11 and RT-DETR tracking
│   ├── track_yolov11.py   # Tracker using YOLOv11
│   └── track_rtdetrv4.py  # Tracker using RT-DETRv4
├── ultralytics_tracking   # Configuration and scripts for Ultralytics-based tracking
│   ├── bot_sort.yaml      # BoT-SORT hyperparameter configuration
│   └── byte_track.yaml    # ByteTrack hyperparameter configuration
├── xai                    # Explainable AI tools
│   ├── lime               # LIME-based detection explanations
│   └── eigen_cam          # Eigen-CAM visualization scripts for YOLO/RT-DETR
├── media                  # Sample videos, images, and output visualizations
├── weights                # Directory for model weights (YOLO, RT-DETR, ReID)
└── requirements.txt       # Project dependencies
```

---

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/ZaheerH-03/ExplainableTrack.git
cd ExplainableTrack
```

### 2. Setup Environment
We recommend using a conda environment:
```bash
conda create -n botsort_xai python=3.9
conda activate botsort_xai
pip install -r requirements.txt
```

### 3. Install BoT-SORT dependencies
Ensure the BoT-SORT requirements are also met:
```bash
cd BoT-SORT
pip install -r requirements.txt
pip install -v -e .
cd ..
```

---

## 📈 Usage

### 🔍 Multi-Object Tracking

To run the tracker using **YOLOv11**:
```bash
python botsort_scripts/track_yolov11.py
```

To run the tracker using **RT-DETRv4** (configured for X-Large model with custom classes):
```bash
python botsort_scripts/track_rtdetrv4.py
```

To run **RT-DETR Inference Only** (on images/videos):
```bash
python RT-DETRv4-main/tools/inference/torch_inf.py \
    -c RT-DETRv4-main/configs/rtv4/rtv4_x_custom.yml \
    -r weights/best_stg2.pth \
    -i media/input_images_xai/tank.jpg \
    -o media/output_result.jpg \
    -d cuda:0
```

### 💡 Explainable AI (XAI)

#### LIME Explanation
Analyze a specific detection to understand which image regions influenced the prediction:
```bash
python xai/lime/lime_yolo_detection.py
```
*Outputs will be saved in `media/lime_explanation.jpg`.*

#### Eigen-CAM Visualization
Generate class-activation maps for YOLOv11:
```bash
python xai/eigen_cam/eigen_cam_yolo.py
```

---

## 🧠 Models & Weights

Please ensure the following weights are placed in the `weights/` directory:
-   `yolo11m.pt` (YOLOv11 Medium weights)
-   `mot17_sbs_S50.pth` (Fast-ReID weights for BoT-SORT)
-   Any custom RT-DETR weights used in scripts.

---

## 🧪 Results & Visualizations

### Tracking Performance
The system achieves high FPS on modern GPUs, balancing tracking accuracy (MOTA) and speed.

### XAI Insights
LIME and Eigen-CAM provide intuitive heatmaps, helping researchers identify if models are correctly focusing on object features or "cheating" based on background context.

---

## 🤝 Acknowledgements

-   [BoT-SORT](https://github.com/NirAharon/BoT-SORT) for the tracking logic.
-   [Ultralytics](https://github.com/ultralytics/ultralytics) for the YOLO framework.
-   [RT-DETRv4](https://github.com/RT-DETRs/RT-DETRv4) for the transformer-based detection architecture.
-   [LIME](https://github.com/marcotcr/lime) for the explanation framework.
-   [pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam) for CAM implementations.

---

Will Soon be updated with a RT-DETRv4 repo and also a RTDETRv4 custom repo with required changes needed to run on 
personal datasets with and without teacher model.

---
## 📜 License

This project is licensed under the MIT License - see the `LICENSE` file for details.
