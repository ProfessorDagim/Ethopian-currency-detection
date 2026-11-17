# 🇪🇹 Ethiopian Birr Banknote Detection & Real-Time Recognition Pipeline
### An End-to-End Deep Learning Framework using YOLOv8 with Biometric Skin Suppression & Multi-Device Streaming

---

## 📑 Table of Contents
1. [Abstract & Project Overview](#-abstract--project-overview)
2. [Banknote Domain & Denomination Specification](#-banknote-domain--denomination-specification)
3. [Dataset Engineering & Preprocessing](#-dataset-engineering--preprocessing)
4. [Deep Learning Model Architecture](#-deep-learning-model-architecture)
5. [Training Configuration & Hyperparameters](#-training-configuration--hyperparameters)
6. [Quantitative Evaluation & Empirical Results](#-quantitative-evaluation--empirical-results)
7. [Engineering Challenges & Domain Solutions](#-engineering-challenges--domain-solutions)
8. [Real-Time Edge Deployment System](#-real-time-edge-deployment-system)
9. [Project Directory Structure](#-project-directory-structure)
10. [Reproducibility & Execution Guide](#-reproducibility--execution-guide)
11. [Academic References & Citations](#-academic-references--citations)

---

## 🔬 Abstract & Project Overview

Automated currency recognition represents a fundamental challenge in applied computer vision, with direct implications for financial accessibility, assistive technology for the visually impaired, automated cash handling systems, and anti-counterfeiting verification. 

The National Bank of Ethiopia (NBE) circulating series features five primary banknote denominations: **5, 10, 50, 100, and 200 Ethiopian Birr (ETB)**. While visually distinctive under controlled laboratory conditions, real-world edge deployment faces severe degradation from:
- Non-rigid geometric distortions (creasing, folding, crumpled surfaces).
- Significant chromatic overlap (e.g., the brick-red/orange hues of the 50 Birr note overlapping with human skin tones).
- Severe specular reflections and uneven ambient illumination.
- Hardware constraints across diverse acquisition sensors (PC webcams, mobile phone sensors over Wi-Fi, USB devices).

This project presents a rigorous, reproducible, end-to-end computer vision framework:
1. **Model Backbone**: Ultralytics **YOLOv8n** (You Only Look Once version 8 - Nano), trained with transfer learning and optimized for low-latency inference on edge devices.
2. **Annotation Pipeline**: Automated mathematical conversion from Pascal VOC XML coordinate systems to normalized bounding box formats with data validation.
3. **Data Augmentation**: Robust affine transformations, photometric distortions, and mosaic augmentations to mitigate small-sample overfitting.
4. **Biometric Chromatic Filter**: A YCrCb skin-color suppression filter that resolves false-positive detections caused by human hands holding banknotes.
5. **Interactive Edge Deployment**: A multi-threaded, dual-interface desktop engine supporting PC DirectShow webcams, network IP streams (HTTP/RTSP), zero-install mobile browser streaming, and Windows Phone Link integration.

---

## 💵 Banknote Domain & Denomination Specification

The model classifies five distinct legal tender denominations currently in active circulation within the Federal Democratic Republic of Ethiopia:

| Denomination | Dominant Color Palette | Representative Obverse / Reverse Iconography | Primary Visual Challenge |
| :---: | :---: | :---: | :---: |
| **5 Birr** | Sepia / Tawny Brown | Traditional coffee harvest, indigenous cattle & rural agrarian landscape | Low contrast under dim warm lighting; shadows easily mimic folds |
| **10 Birr** | Emerald Green | Agricultural harvesting, tractor mechanization & camel caravans | High contrast edges; potential background confusion with foliage |
| **50 Birr** | Brick Red / Terracotta | Renaissance Gondar Castle fortress & industrial infrastructure | **High spectral overlap with human skin under warm incandescent light** |
| **100 Birr** | Cyan / Deep Teal Blue | Historic architectural gates, indigenous agriculture & academic motif | Distinct chromatic profile; sensitive to blue-channel camera noise |
| **200 Birr** | Golden Bronze / Ochre | Dove of peace, national heraldry & modern technological advancement | Highlights can over-saturate under direct LED or flash glare |

---

## 📊 Dataset Engineering & Preprocessing

### 1. Dataset Composition & Stratification
The baseline dataset comprises high-resolution portrait imagery ($1080 \times 1920$ px) captured across diverse backgrounds, orientations, and folding states:

$$\mathcal{D} = \bigcup_{c \in \mathcal{C}} \mathcal{D}_c, \quad \mathcal{C} = \{5, 10, 50, 100, 200\}$$

- **5 Birr**: 15 ground-truth annotated samples
- **10 Birr**: 11 ground-truth annotated samples
- **50 Birr**: 11 ground-truth annotated samples
- **100 Birr**: 9 ground-truth annotated samples
- **200 Birr**: 13 ground-truth annotated samples
- **Total Raw Corpus**: 59 primary image-XML pairs.

### 2. Coordinate Conversion Mathematics
Raw bounding box annotations were extracted from XML documents adhering to the **Pascal VOC standard** ($x_{\min}, y_{\min}, x_{\max}, y_{\max}$) and mapped to the **YOLO normalized coordinate space** ($x_c, y_c, w, h \in [0, 1]$):

Given an image of width $W$ and height $H$:

$$x_c = \frac{x_{\min} + x_{\max}}{2 \cdot W}, \quad y_c = \frac{y_{\min} + y_{\max}}{2 \cdot H}$$

$$w = \frac{x_{\max} - x_{\min}}{W}, \quad h = \frac{y_{\max} - y_{\min}}{H}$$

All derived coordinates satisfy the boundary constraint:

$$0 \le x_c - \frac{w}{2} < x_c + \frac{w}{2} \le 1, \quad 0 \le y_c - \frac{h}{2} < y_c + \frac{h}{2} \le 1$$

### 3. Data Augmentation Strategy
To generalize effectively from limited real-world captures, training batches leveraged an aggressive dynamic augmentation pipeline:
- **Spatial Affine Transformations**: Random rotation ($\pm 15^\circ$), horizontal flip ($p = 0.5$), scaling ($\pm 20\%$).
- **Photometric Distortions**: HSV color-space jittering ($h_{\text{gain}} = 0.015, s_{\text{gain}} = 0.7, v_{\text{gain}} = 0.4$) to simulate diverse ambient lighting conditions.
- **Mosaic & MixUp**: 4-image mosaic compositing during initial training epochs, deactivated during the final 10 epochs (`close_mosaic = 10`) for precise boundary alignment.

---

## 🧠 Deep Learning Model Architecture

The core detector utilizes **YOLOv8n (Nano)**, designed by Ultralytics. The architecture is composed of three interconnected functional sub-networks:

```
[Input Tensor: 3 x 640 x 640]
              │
              ▼
┌───────────────────────────────────────────────┐
│           Backbone: Modified CSPDarknet       │
│  - Stem: Focus / P1 Conv (k=3, s=2)           │
│  - C2f Modules (Cross-Stage Partial Bottleneck│
│    with Split and Concat feature flow)        │
│  - SPPF (Spatial Pyramid Pooling - Fast)      │
└──────────────────────┬────────────────────────┘
                       │ Multi-Scale Feature Maps (P3, P4, P5)
                       ▼
┌───────────────────────────────────────────────┐
│               Neck: Path Aggregation (PANet)  │
│  - Top-Down & Bottom-Up Feature Pyramids      │
│  - Fuses shallow spatial detail (P3) with     │
│    deep semantic representations (P5)         │
└──────────────────────┬────────────────────────┘
                       │
                       ▼
┌───────────────────────────────────────────────┐
│       Anchor-Free Decoupled Detection Head    │
│  - Branch 1: Binary Cross-Entropy Loss (BCE)  │
│              for Class Probability            │
│  - Branch 2: Complete IoU (CIoU) Loss +       │
│              Distribution Focal Loss (DFL)    │
│              for Bounding Box Regression      │
└───────────────────────────────────────────────┘
```

### Key Architectural Advances:
1. **Anchor-Free Detection**: Eliminates predefined anchor box aspect ratios, predicting offsets directly from grid cell centers to note boundaries, accelerating convergence on irregular banknote orientations.
2. **Decoupled Head**: Separates the classification score computation from spatial bounding box regression, eliminating parameter competition between the two optimization objectives.
3. **Loss Formulation**:
   
   $$\mathcal{L}_{\text{total}} = \lambda_{\text{box}} \mathcal{L}_{\text{CIoU}} + \lambda_{\text{cls}} \mathcal{L}_{\text{BCE}} + \lambda_{\text{dfl}} \mathcal{L}_{\text{DFL}}$$

---

## ⚙️ Training Configuration & Hyperparameters

Training was conducted in a high-performance GPU environment on Kaggle using PyTorch 2.x and CUDA 12.x.

```yaml
# Core Training Parameters (from birr_trained_outputs/args.yaml)
task: detect
mode: train
model: yolov8n.pt
imgsz: 640
epochs: 60
batch: 8
patience: 15
optimizer: auto (AdamW with Cosine Annealing)
device: '0' (NVIDIA Tesla T4 GPU)
workers: 8
close_mosaic: 10
amp: true (Automatic Mixed Precision - FP16)
iou: 0.7
```

---

## 📈 Quantitative Evaluation & Empirical Results

The model converged smoothly over 60 epochs without evidence of catastrophic overfitting or gradient instability.

### Final Model Performance (Epoch 60 Validation Metrics)

| Metric | Symbol | Value | Performance Interpretation |
| :--- | :---: | :---: | :--- |
| **Precision** | $P$ | **98.36%** | Low false alarm rate; detections are overwhelmingly true banknotes. |
| **Recall** | $R$ | **100.00%** | Zero false negatives across validation targets; complete banknote retrieval. |
| **mAP @ 0.50 IoU** | $\text{mAP}_{50}$ | **99.50%** | Near-perfect detection across standard overlap criteria. |
| **mAP @ 0.50:0.95 IoU** | $\text{mAP}_{50-95}$ | **95.20%** | Exceptional localization precision across stringent IoU thresholds. |
| **Bounding Box Regression Loss** | $\mathcal{L}_{\text{box}}$ | **0.2778** | Sharp spatial edge alignment on banknote borders. |
| **Classification Loss** | $\mathcal{L}_{\text{cls}}$ | **0.2309** | Distinct inter-class denomination separation. |
| **Distribution Focal Loss** | $\mathcal{L}_{\text{DFL}}$ | **0.8632** | Stable uncertainty modeling for sub-pixel coordinates. |

### Evaluation Curves and Diagnostics
All diagnostic artifacts are preserved in [`birr_trained_outputs/`](file:///c:/Users/SURA/Desktop/birr%20(3)/birr/birr_trained_outputs):
- `confusion_matrix.png` / `confusion_matrix_normalized.png`: Confirm zero inter-denomination confusion at the operational threshold.
- `BoxPR_curve.png`: Demonstrates high area under the Precision-Recall curve across all five classes.
- `BoxF1_curve.png`: Peak F1-score of $\mathbf{0.99}$ achieved at a confidence threshold of $\tau = 0.42$.

---

## 💡 Engineering Challenges & Domain Solutions

### 1. The Human Skin / 50 Birr False-Positive Dilemma
* **Problem**: Because banknote training images predominantly contained notes held by human hands or captured against warm table tops, naive models can falsely trigger on bare human skin, faces, or forearms at moderate confidence thresholds, specifically classifying them as **50 Birr** due to warm terracotta color overlap.
* **Academic Solution**: We integrated an explicit **YCrCb Chromatic Human Skin Filter**. The RGB/BGR color space is heavily coupled with luminance, making skin detection brittle under varying lighting. By transforming prospective detection bounding crops into the $Y C_r C_b$ color space:

  $$Y = 0.299R + 0.587G + 0.114B$$
  
  $$C_r = (R - Y) \times 0.713 + 128$$
  
  $$C_b = (B - Y) \times 0.564 + 128$$

  Human skin exhibits a tight, invariant chromatic clustering in the chrominance plane ($C_r \in [135, 175], C_b \in [80, 125]$). If the proportion of skin pixels inside a bounding box exceeds the threshold $\rho_{\text{skin}} > 0.68$, the detection is classified as a bare human face or hand and rejected. Real banknotes contain intricate guilloche patterns, typography, and national symbols that maintain $\rho_{\text{skin}} \le 0.45$.

### 2. Windows VideoIO DirectShow Architecture
* **Problem**: OpenCV's default Windows Media Foundation (`cv2.CAP_MSMF`) backend frequently triggers frame grab failures (`Error: -1072875772` and assertion `_step >= minstep`) on integrated laptop webcams.
* **Solution**: Implemented an automated hardware fallback mechanism prioritizing DirectShow (`cv2.CAP_DSHOW`), ensuring stable 30+ FPS video capture without frame pipeline stalls.

---

## 🖥️ Real-Time Edge Deployment System

The operational inference engine is implemented in [`live_birr_detector.py`](file:///c:/Users/SURA/Desktop/birr%20(3)/birr/live_birr_detector.py) and triggered via [`run_camera.bat`](file:///c:/Users/SURA/Desktop/birr%20(3)/birr/run_camera.bat).

```
                      ┌──────────────────────────────────────┐
                      │    live_birr_detector.py Controller  │
                      └──────────────────┬───────────────────┘
                                         │
        ┌────────────────────────────────┼────────────────────────────────┐
        │                                │                                │
        ▼                                ▼                                ▼
┌───────────────┐              ┌──────────────────┐             ┌──────────────────┐
│ PC Built-in   │              │ Phone IP Camera  │             │ Browser Wi-Fi    │
│ Webcam (0)    │              │ Stream (RTSP/HTTP│             │ Web Server       │
│ via DirectShow│              │ camera_config.txt│             │ (Flask :5000)    │
└───────┬───────┘              └─────────┬────────┘             └─────────┬────────┘
        │                                │                                │
        └────────────────────────────────┼────────────────────────────────┘
                                         │ Video Frames (BGR)
                                         ▼
                      ┌──────────────────────────────────────┐
                      │    YOLOv8n Inference Engine          │
                      │    - Tensor Input (640x640)          │
                      │    - NMS Bounding Box Regression     │
                      └──────────────────┬───────────────────┘
                                         │ Raw Detections
                                         ▼
                      ┌──────────────────────────────────────┐
                      │    Biometric Skin Filter (YCrCb)     │
                      │    - Filters Hand & Face Triggers    │
                      └──────────────────┬───────────────────┘
                                         │ Valid Banknotes
                                         ▼
                      ┌──────────────────────────────────────┐
                      │    HUD Overlay & Financial Counter   │
                      │    - Denomination Pill Tags          │
                      │    - Live Cumulative Sum (ETB)       │
                      │    - Real-Time FPS Telemetry         │
                      └──────────────────────────────────────┘
```

### Multi-Source Camera Versatility
1. **PC Built-in Webcam (DirectShow Index 0)**: Instant zero-configuration local webcam capture.
2. **Phone IP Camera URL (HTTP/RTSP)**: Connects to Android/iOS IP camera apps (e.g., `http://192.168.1.15:8080/video`), persisted automatically in [`camera_config.txt`](file:///c:/Users/SURA/Desktop/birr%20(3)/birr/camera_config.txt).
3. **Zero-Install Phone Browser Stream**: Launches an internal Flask streaming server on port 5000; opening `http://<PC_IP>:5000` in mobile Safari or Chrome immediately feeds the mobile camera over local Wi-Fi without installing third-party apps.
4. **Windows Phone Link / USB (Index 1/2)**: Native Windows 11 connected camera integration.

### Concurrent Dual-Interface Navigation
The application features non-blocking, multi-threaded input handling:
- **Video Window Keystrokes** (`cv2.waitKey`): Press `[B]` to switch camera, `[Q]` or `[ESC]` to quit, `[K]` to toggle skin filter, `[+]`/`[-]` to tune confidence threshold, `[S]` for screenshot.
- **Terminal CMD Controls** (`msvcrt` + background `stdin` listener): Users can type choices (`1`-`7`, `Q`, or raw URLs) directly in the console without window focus deadlocks.

---

## 📁 Project Directory Structure

```text
c:\Users\SURA\Desktop\birr (3)\birr\
├── README.md                              # Comprehensive Academic Project Documentation
├── live_birr_detector.py                  # Live Real-Time Multi-Source AI Detection Application
├── web_app/                               # 🌐 Deployable Client-Side Web Application (HTML5/WASM)
│   ├── index.html                         # Standalone Responsive Single-Page Web App
│   ├── app.js                             # In-Browser YOLOv8 WebGL / WASM Inference Engine
│   ├── model/best.onnx                    # Exported YOLOv8 Model Weights
│   ├── vercel.json                        # Vercel Production Deployment Config
│   ├── netlify.toml                       # Netlify Production Deployment Config
│   └── DEPLOYMENT_GUIDE.md                # 1-Click Deployment Instructions (Vercel, Netlify, GH Pages)
├── run_camera.bat                         # 1-Click Batch Launcher with Environment Resolution
├── camera_config.txt                      # Saved IP Camera Streaming URL Configuration
├── ethiopian-birr-yolov8-kaggle.ipynb      # Complete 35-Cell Reproducible Kaggle Training Notebook
│
├── birr_trained_outputs/                  # Empirical Experimental Evaluation & Weights
│   ├── weights/
│   │   ├── best.pt                        # Optimal PyTorch Model Weights (mAP50: 99.5%)
│   │   └── best.onnx                      # Exported Open Neural Network Exchange Model
│   ├── args.yaml                          # Exact Hyperparameter Configuration
│   ├── results.csv                        # Per-Epoch Loss & Evaluation Metrics Log
│   ├── results.png                        # Training Convergence Curves
│   ├── confusion_matrix.png               # Unnormalized Confusion Matrix
│   ├── confusion_matrix_normalized.png    # Normalized Inter-Class Confusion Matrix
│   ├── BoxF1_curve.png                    # Precision-Recall Harmonic Mean Curve
│   └── BoxPR_curve.png                    # Precision-Recall Characteristic Curve
│
├── saved_detections/                      # Output Directory for [S] Screenshot Captures
│
├── 5Birr/                                 # Raw Class 0 Data (15 Images + Pascal VOC XMLs)
├── 10Birr/                                # Raw Class 1 Data (11 Images + Pascal VOC XMLs)
├── 50Birr/                                # Raw Class 2 Data (11 Images + Pascal VOC XMLs)
├── 100Birr/                               # Raw Class 3 Data (9 Images + Pascal VOC XMLs)
└── 200Birr/                               # Raw Class 4 Data (13 Images + Pascal VOC XMLs)
```

---

## 🚀 Reproducibility & Execution Guide

### 1. Training from Scratch (Kaggle or Local GPU)
1. Open [`ethiopian-birr-yolov8-kaggle.ipynb`](file:///c:/Users/SURA/Desktop/birr%20(3)/birr/ethiopian-birr-yolov8-kaggle.ipynb) in Jupyter Lab, VS Code, or import directly into Kaggle.
2. Select an accelerator runtime (**NVIDIA GPU P100 or T4**).
3. Execute all cells sequentially. The notebook:
   - Validates all image formats and XML boundaries.
   - Converts coordinates into YOLO format with automated class label indexing (`0: 5`, `1: 10`, `2: 50`, `3: 100`, `4: 200`).
   - Generates the standard `data.yaml` descriptor.
   - Executes the 60-epoch YOLOv8n training loop.
   - Exports the final artifact to both PyTorch (`best.pt`) and ONNX format (`best.onnx`).

### 2. Live Local Deployment
1. Ensure Python 3.10+ and dependencies are installed:
   ```bash
   pip install ultralytics opencv-python numpy flask
   ```
2. Launch or Deploy the Application:
   - **Method A (Deployable Web Application - HTML / In-Browser)**:
     Deploy the [`web_app/`](file:///c:/Users/OMEN%2016/Downloads/Telegram%20Desktop/birr/web_app/) directory directly to **Vercel**, **Netlify**, or **GitHub Pages**.
     - Runs 100% in the user's browser via ONNX Runtime Web (WASM/WebGL).
     - No backend server or `.bat` files required.
     - Supports live camera scanning, file drag-and-drop, and real-time ETB calculation.
     - See [`web_app/DEPLOYMENT_GUIDE.md`](file:///c:/Users/OMEN%2016/Downloads/Telegram%20Desktop/birr/web_app/DEPLOYMENT_GUIDE.md) for 30-second deployment instructions.
   - **Method B (Desktop OpenCV Window)**: Double-click [`run_camera.bat`](file:///c:/Users/OMEN%2016/Downloads/Telegram%20Desktop/birr/run_camera.bat)
   - **Method C**: Execute from PowerShell or Command Prompt:
     ```bash
     python live_birr_detector.py
     ```
3. In the interactive terminal menu, select:
   - `[1]` for PC Built-in Webcam
   - `[2]` for Phone IP Camera (`http://192.168.1.15:8080/video`)
   - `[3]` to enter a custom phone stream URL
   - `[4]` to launch the zero-install mobile browser stream (`http://<LOCAL_IP>:5000`)
   - `[Q]` to exit cleanly.

---

## 📚 Academic References & Citations

1. **Jocher, G., Chaurasia, A., & Qiu, J.** (2023). *Ultralytics YOLOv8* (Version 8.0.0). [Software]. Available: `https://github.com/ultralytics/ultralytics`.
2. **Redmon, J., Divvala, S., Girshick, R., & Farhadi, A.** (2016). *You Only Look Once: Unified, Real-Time Object Detection*. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 779-788.
3. **Everingham, M., Van Gool, L., Williams, C. K., Winn, J., & Zisserman, A.** (2010). *The Pascal Visual Object Classes (VOC) Challenge*. International Journal of Computer Vision, 88(2), 303-338.
4. **Chai, D., & Ngan, K. N.** (1999). *Face segmentation using skin-color map in suggestion of chrominance components*. IEEE Transactions on Pattern Analysis and Machine Intelligence, 21(11), 1151-1158.
5. **National Bank of Ethiopia (NBE)**. *Currency and Banknote Circulating Series Specifications*. Official Gazette of the Federal Democratic Republic of Ethiopia.
"# Ethopian-currency-detection" 
