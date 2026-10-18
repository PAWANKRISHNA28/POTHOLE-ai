# POTHOLE AI — Road Pothole Detection Using Edge Enhancement Techniques

**Academic Project Title:** *Apply Edge Enhancement Techniques to Highlight Potholes in Road Images*

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-34d399.svg?style=flat&logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-ff4b4b.svg?style=flat&logo=streamlit)](https://streamlit.io/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Headless%204.8%2B-5c3ee8.svg?style=flat&logo=opencv)](https://opencv.org/)
[![Pytest](https://img.shields.io/badge/Tests-27%20Passed-10b981.svg?style=flat&logo=pytest)](https://pytest.org/)
[![License: Academic](https://img.shields.io/badge/License-Academic%20Use-blue.svg)](#credits--academic-use-notice)

---

## 📌 Executive Overview

**POTHOLE AI** is an academic, interpretable computer vision dashboard designed to analyze road surface imagery, suppress pavement noise, and highlight potential pothole distress boundaries using classical edge enhancement and morphological analysis.

Rather than operating as an uninterpretable, high-latency deep learning black box that requires dedicated cloud GPUs, this project investigates fundamental mathematical edge operators:
* **Canny Edge Detector**: Multi-stage optimal edge detection utilizing Gaussian smoothing, directional gradient orientation, non-maximum suppression (NMS), and dual-threshold hysteresis.
* **Sobel Operator**: Discrete first-order spatial differentiation using separable $3 \times 3$ kernels measuring horizontal ($G_x$) and vertical ($G_y$) luminance transitions.
* **Laplacian Operator**: Second-order isotropic spatial derivative operator ($\nabla^2 f$) isolating rapid inflection zero-crossings.

Combined with **Contrast Limited Adaptive Histogram Equalization (CLAHE)**, elliptical morphological closing, and geometric contour filtering, the system acts as a high-recall **Region of Interest (RoI) candidate proposer** for highway maintenance crews and smart city infrastructure monitoring.

---

## 🛑 Problem Statement

Potholes and asphalt surface depressions represent primary triggers of vehicular structural damage, traffic congestion, and fatal roadway accidents globally. Traditional municipal road inspections rely on:
1. **Manual Visual Surveys**: Extremely slow, labor-intensive, subjective, and dangerous for field inspectors on high-speed thoroughfares.
2. **Dedicated Laser Profilometer Vans**: Prohibitively expensive hardware ($100,000+ per vehicle) inaccessible to municipal budgets.
3. **Pure Deep Learning Classifiers**: High computational requirements, cloud dependency, susceptibility to illumination distribution shifts, and lack of mathematical interpretability.

There is a significant engineering need for an **interpretable, CPU-efficient, edge-deployable computer vision solution** that isolates pavement distress contours in real-time ($< 55\text{ ms}$) without external cloud dependencies.

---

## 🎯 Project Objectives

1. **Investigate Classical Edge Operators**: Compare the mathematical localization, noise rejection, and execution runtimes of Canny, Sobel, and Laplacian operators on asphalt pavement imagery.
2. **Optimize Illumination Variance**: Implement CLAHE and Gaussian filtering to normalize shadows and suppress high-frequency aggregate gravel texture noise.
3. **Isolate Candidate Distress Contours**: Bridge fragmented edge discontinuities via elliptical morphological closing and retrieve closed candidate contours.
4. **Transparent Geometric Filtering**: Apply explicit, physical geometric rules (area bounds, aspect ratio limits, convex hull solidity) to reject lane stripes and chaotic aggregate noise.
5. **Interactive Analytics Dashboard**: Deliver a production-grade Streamlit interface with live parameter calibration, high-resolution zoom, side-by-side operator comparisons, and standardized research report exports.

---

## 🌟 Key Features

* **Tri-Detector Benchmarking Suite**: Side-by-side comparative inspection of Canny, Sobel, and Laplacian operators with live latency profiling (ms) and edge pixel density calculations.
* **Directional Gradient Analysis**: Dedicated inspection of orthogonal Sobel $G_x$ (vertical fissures) and $G_y$ (horizontal cracks) alongside combined Euclidean gradient magnitude.
* **Adaptive Contrast Optimization**: Configurable CLAHE tile grid size and clip limit, alongside Global Histogram Equalization and Linear Percentile Contrast Stretching.
* **Morphological Candidate Extraction**: Elliptical closing kernels bridging disjoint boundary contours, followed by topological contour retrieval (`cv2.findContours`).
* **Interactive Candidate Inspector (ROI Magnifier)**: Detailed per-candidate metric table (bounding box, contour area, perimeter, aspect ratio, convex hull solidity, interior mean intensity) and magnified visual zoom.
* **Standardized Research Artifact Exports**: One-click download of processed candidate overlay PNGs, 6-stage pipeline comparison montages, CSV audit logs, and standardized JSON research reports.
* **Pre-Loaded Benchmark Scenarios**: Realistic asphalt test scenarios (Single Deep Pothole, Multiple Depressions & Cracks, Wet Specular Pavement) ready for demonstration without manual file upload.

---

## 🛠️ Technology Stack

| Component | Library / Tool | Purpose in Project |
| :--- | :--- | :--- |
| **Language** | Python 3.10+ / 3.12 / 3.14 | Core programming language for algorithm implementation. |
| **Computer Vision** | `opencv-python-headless` | Image decoding, spatial convolutions, gradients, morphology, and contours. |
| **Numerical Processing** | `numpy` | Multi-dimensional matrix operations, masking, and statistics. |
| **Tabular Data** | `pandas` | Telemetry formatting, candidate dataframes, and CSV history exports. |
| **Presentation Layer** | `streamlit` | Interactive web dashboard, reactive sliders, multi-page routing, state management. |
| **Visualization** | `matplotlib` & `Pillow` | Intensity distribution histograms, montage compilation, and byte serialization. |
| **Testing & QA** | `pytest` | 27 automated unit tests validating edge cases, kernels, and report schemas. |

---

## 🔬 Computer Vision Processing Pipeline

The computational pipeline executes across 5 sequential stages:

```
[ Input Road Image (PNG/JPG/WEBP) ]
               │
               ▼
   [ Stage 1: Validation & Ingestion ] ── Checks channels, dimensions (16px to 8192px), RGB decoding
               │
               ▼
   [ Stage 2: Preprocessing & Contrast ] ── Grayscale conversion + CLAHE contrast + Gaussian noise smoothing
               │
               ▼
   [ Stage 3: Edge Enhancement ] ── Canny (NMS + Hysteresis) / Sobel (Gx, Gy) / Laplacian (∇²f)
               │
               ▼
   [ Stage 4: Morphology & Contours ] ── Elliptical kernel closing + Topological contour retrieval
               │
               ▼
   [ Stage 5: Filtering & Telemetry ] ── Geometric constraints (Area, Aspect Ratio, Solidity)
               │
               ▼
[ Visual Overlay HUD + JSON Research Report + CSV Audit Trail ]
```

---

## 📂 Project Folder Structure

```
ROAD POTHOLE DETECTION/
├── app.py                      # Root entrypoint delegating to Road-Pothole-Detection
├── requirements.txt            # Root Python dependencies
├── README.md                   # Repository documentation & quick start guide
├── Road-Pothole-Detection/
│   ├── app.py                  # Primary Streamlit Dashboard (7 cohesive multi-page views)
│   ├── requirements.txt        # Pinned dependency specifications
│   ├── README.md               # Detailed project guide
│   ├── src/                    # Core Computer Vision Engine
│   │   ├── __init__.py         # Package exports
│   │   ├── preprocessing.py    # Grayscale, CLAHE, Histogram Eq, Linear Stretch, Gaussian blur
│   │   ├── edge_detection.py   # Canny (auto-sorting), Sobel (Gx/Gy), Laplacian, density calculations
│   │   ├── pothole_detection.py# Morphology, candidate contour extraction, overlay rendering
│   │   ├── image_utils.py      # IO helpers, PNG encoding, comparison montage, synthetic road generator
│   │   └── evaluation.py       # Candidate summaries, comparison matrix, JSON report builder
│   ├── assets/
│   │   └── style.css           # Refined dark charcoal (#11191B) & emerald (#32D6A0) design system
│   ├── data/
│   │   ├── input/              # Staging directory for image uploads
│   │   └── sample_images/      # Synthetic and real asphalt benchmark samples
│   ├── outputs/                # Generated research artifacts and exports
│   ├── tests/                  # Pytest automated test suite (27 passing tests)
│   │   ├── test_preprocessing.py
│   │   ├── test_edge_detection.py
│   │   ├── test_pothole_detection.py
│   │   └── test_image_utils_and_evaluation.py
│   └── docs/                   # Formal Academic Documentation Suite
│       ├── project_report.md   # Formal College Project Report (16 comprehensive sections)
│       ├── methodology.md      # Detailed 12-step pipeline documentation with Mermaid diagrams
│       ├── results.md          # Real experimental benchmarks & observation template
│       ├── demo_script.md      # 5–7 minute live presentation and viva demonstration script
│       └── presentation_outline.md # 10-slide academic project presentation deck outline
```

---

## 🚀 Installation & Setup Guide (Windows)

### 1. Prerequisites
* **Operating System**: Windows 10 or 11 (also runs on Linux and macOS).
* **Python**: Python 3.10, 3.11, or 3.12 (Python 3.14 compatible).

### 2. Clone or Open the Workspace
Open PowerShell in the project root directory:
```powershell
cd "c:\Users\mpawa\OneDrive\Desktop\ROAD POTHOLE DETECTION"
```

### 3. Create or Activate Virtual Environment
If `.venv` already exists in your workspace, activate it directly without re-creating:
```powershell
.\.venv\Scripts\Activate.ps1
```
*(If setting up on a new machine for the first time)*:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 4. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 5. Launch the POTHOLE AI Dashboard
Launch the application directly from the workspace root:
```powershell
python -m streamlit run app.py
```
Or launch from within the `Road-Pothole-Detection` folder:
```powershell
cd Road-Pothole-Detection
python -m streamlit run app.py
```

The application will start immediately and display:
```
Local URL: http://localhost:8501
Network URL: http://192.168.x.x:8501
```
Open **`http://localhost:8501`** in your web browser.

---

## 🧪 Automated Testing & Verification

Run the full automated test suite using `pytest`:

```powershell
pytest -v
```

### Verified Test Coverage (27 / 27 Passed)
* `tests/test_preprocessing.py`: Image validation (NumPy, PIL, bytes), corrupted file detection, 2D grayscale, 3-channel RGB, 4-channel RGBA, CLAHE, Histogram Equalization, and Gaussian/Bilateral/Median noise filters.
* `tests/test_edge_detection.py`: Canny hysteresis, Sobel directional gradients ($G_x, G_y$), Laplacian zero-crossings, edge density formulas, auto-sorted reversed Canny thresholds, and invalid kernel size fallbacks.
* `tests/test_pothole_detection.py`: Morphological closing, contour extraction, bounding box calculations, empty image handling, candidate overlay rendering, and inverted area/aspect ratio range auto-recovery.
* `tests/test_image_utils_and_evaluation.py`: Synthetic road generator, 6-stage montage composition, JSON serialization, and comparison tables.

---

## 🖼️ Supported Image Formats & Specifications

* **File Types**: `PNG`, `JPG`, `JPEG`, `WEBP`.
* **Color Channels**:
  * 3-channel RGB / BGR (standard color camera imagery).
  * 4-channel RGBA (alpha transparency automatically blended to RGB).
  * 1-channel / 2D Grayscale (automatically expanded to RGB).
* **Dimensions**:
  * Minimum supported: $16 \times 16$ px (prevents kernel convolution underflow).
  * Maximum supported: $8192 \times 8192$ px (prevents CPU memory exhaustion).
  * Recommended resolution: $640 \times 480$ px up to $1920 \times 1080$ px.

---

## ⚠️ Academic Limitations & Scope

1. **Distress Candidates vs. Confirmed Potholes**: Edge detection identifies sudden spatial intensity gradients. It **does not semantic classify** physical potholes. Shadows cast by roadside trees, dark oil stains, tire skid marks, bitumen expansion joints, and puddles can trigger false candidate boundaries.
2. **Threshold Sensitivity**: Global thresholding parameters require tuning when moving between brightly lit open highways and shadowed urban streets.
3. **Depth Blindness**: 2D monocular images lack physical depth measurement; true cavity depth cannot be confirmed without stereo disparity or LiDAR sensors.
4. **Session Memory**: Inspection history is tracked in browser session memory during the active session. If the user refreshes or closes the browser tab, session history resets.

---

## 🔮 Future Enhancements

1. **Cascaded Deep Learning Verification**: Use edge enhancement as a lightweight pre-filter and forward candidate bounding box crops to a trained YOLOv8/v10 or MobileNet classifier for confirmed semantic verification.
2. **Stereo Disparity Mapping**: Compute 3D point clouds from calibrated stereo camera rigs to measure physical depth in centimeters.
3. **Multi-Modal Sensor Fusion**: Integrate vehicle-mounted accelerometer (IMU) z-axis acceleration spikes to cross-validate visual candidate detections with physical impact telemetry.

---

## 📚 Academic Documentation Suite

Comprehensive documentation is provided in the [`docs/`](file:///c:/Users/mpawa/OneDrive/Desktop/ROAD%20POTHOLE%20DETECTION/Road-Pothole-Detection/docs/) directory:
* [`docs/project_report.md`](file:///c:/Users/mpawa/OneDrive/Desktop/ROAD%20POTHOLE%20DETECTION/Road-Pothole-Detection/docs/project_report.md): Formal 16-chapter college project report.
* [`docs/methodology.md`](file:///c:/Users/mpawa/OneDrive/Desktop/ROAD%20POTHOLE%20DETECTION/Road-Pothole-Detection/docs/methodology.md): Complete 12-stage image processing pipeline with mathematical formulations and Mermaid architecture diagrams.
* [`docs/results.md`](file:///c:/Users/mpawa/OneDrive/Desktop/ROAD%20POTHOLE%20DETECTION/Road-Pothole-Detection/docs/results.md): Experimental benchmarks and results recording template.
* [`docs/demo_script.md`](file:///c:/Users/mpawa/OneDrive/Desktop/ROAD%20POTHOLE%20DETECTION/Road-Pothole-Detection/docs/demo_script.md): Practical 5–7 minute live presentation and viva demo script.
* [`docs/presentation_outline.md`](file:///c:/Users/mpawa/OneDrive/Desktop/ROAD%20POTHOLE%20DETECTION/Road-Pothole-Detection/docs/presentation_outline.md): 10-slide academic project presentation deck structure with real screenshot capture instructions.

---

## 📄 Credits & Academic-Use Notice

Developed as an undergraduate research project in Applied Computer Vision and Intelligent Transportation Systems:
* **Project Title**: *Apply Edge Enhancement Techniques to Highlight Potholes in Road Images*
* **Product Name**: *POTHOLE AI — Road Surface Intelligence*
* **Intended Use**: Academic research, educational demonstration, laboratory benchmarking, and presentation expo.
