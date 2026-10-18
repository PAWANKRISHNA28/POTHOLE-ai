# LIVE DEMONSTRATION & PROJECT EXPO CHECKLIST

## PROJECT: Apply Edge Enhancement Techniques to Highlight Potholes in Road Images
**Platform:** POTHOLE AI — Road Surface Intelligence  
**Launch Command:** `python -m streamlit run app.py`  
**Target Duration:** 5 to 7 Minutes  

---

## 📋 PRE-DEMO SYSTEM CHECKLIST (30 MINUTES BEFORE PRESENTATION)

- [ ] **1. Laptop Power & Screen Settings:**
  - Connect laptop to AC power (avoid battery saver throttling CPU frequency).
  - Set screen display scaling to 100% or 125% and resolution to $1920 \times 1080$.
  - Disable sleep mode and notifications (`Windows Focus Assist: On`).
- [ ] **2. Virtual Environment Verification:**
  - Open Windows PowerShell in workspace root:
    ```powershell
    cd "c:\Users\mpawa\OneDrive\Desktop\ROAD POTHOLE DETECTION"
    ```
  - Activate the verified virtual environment:
    ```powershell
    .\.venv\Scripts\Activate.ps1
    ```
  - Verify terminal prompt indicates `(.venv)`.
- [ ] **3. Dependency & Test Verification:**
  - Verify installed packages match requirements:
    ```powershell
    pip list
    ```
  - Run the automated test suite to confirm zero regressions:
    ```powershell
    pytest -v
    ```
  - Confirm output: **`27 passed in ~0.5s`**.
- [ ] **4. Launch Streamlit Application:**
  - Start the application from the workspace root:
    ```powershell
    python -m streamlit run app.py
    ```
  - Verify local terminal displays:
    ```
    Local URL: http://localhost:8501
    ```
- [ ] **5. Browser Staging:**
  - Open Chrome or Edge and navigate to `http://localhost:8501`.
  - Press `F11` (or maximize window) for a clean presentation screen without messy desktop icons.
  - Confirm the dark charcoal and emerald theme (`#11191B` / `#32D6A0`) loads cleanly.

---

## 🎬 STEP-BY-STEP LIVE DEMONSTRATION SCRIPT (5–7 MINUTES)

### STEP 1: Introduction & Problem Context (0:00 – 0:45)
* [ ] Display **Dashboard** view (`◈  Dashboard`).
* [ ] State project title: *"Apply Edge Enhancement Techniques to Highlight Potholes in Road Images"*.
* [ ] State product identity: **POTHOLE AI — Road Surface Intelligence**.
* [ ] Point out the 4 high-level metric cards: *Images Analyzed*, *Candidate Regions*, *Processing Time*, and *Edge Technique*.
* [ ] Highlight core motivation: Accessible, interpretable, CPU-efficient screening that avoids the cost of $100,000 laser profiler vans and the computational overhead of cloud GPUs.

### STEP 2: Ingest Road Image & Baseline Verification (0:45 – 1:30)
* [ ] Click **Load Scenario A (Single Deep Pothole)** or drag-and-drop a road inspection image.
* [ ] Point to the **2-by-2 Inspection Grid**:
  * Top-Left: **Input Road Surface** (validated RGB).
  * Top-Right: **Enhanced Contrast** (CLAHE local contrast equalization).
  * Bottom-Left: **Active Edge Map** (Canny binary edge contours).
  * Bottom-Right: **Annotated Candidate Overlay** (candidate bounding box HUD).
* [ ] Point to the **Results Summary** table:
  * Dimensions: `640 × 480 px`
  * Execution Time: `~47.3 ms`
  * Edge Operator: `Canny`
  * Candidate Regions: `1 region`

### STEP 3: Compare Implemented Edge Detectors (1:30 – 3:00)
* [ ] Navigate to **`∿  Edge Analysis`** in the sidebar.
* [ ] Demonstrate the **Compact Method Selector**:
  1. **Canny Operator**:
     * Point out the crisp, single-pixel thin perimeter contour.
     * Explain: Gaussian smoothing + gradient orientation + non-maximum suppression (NMS) + hysteresis thresholding.
     * Highlight low edge density (~$4.8\%$) and noise rejection.
  2. **Sobel Operator**:
     * Switch to **Sobel**.
     * Point out the thicker gradient band outlining the cavity.
     * Click the **Directional Gradients** tab: show $G_x$ (vertical fissures) vs. $G_y$ (horizontal cracks).
     * Explain: First-order derivative convolution measuring directional crack alignment.
  3. **Laplacian Operator**:
     * Switch to **Laplacian**.
     * Point out second-order zero-crossings detecting rapid concave intensity drops.
     * Explain why Laplacian is sensitive to texture noise and produces double-line contours.
* [ ] Click on the **Tri-Operator Side-by-Side Suite** tab to show all three operators simultaneously.

### STEP 4: Inspect Potential Pothole Candidates (3:00 – 4:15)
* [ ] Navigate to **`⊙  Pothole Candidates`** in the sidebar.
* [ ] Point out the **Operational Advisory Banner**:
  * Emphasize: Detected regions represent *potential candidate hypotheses* for engineering review, not guaranteed ground-truth potholes.
* [ ] Point out the **Candidate Properties Table**:
  * Candidate ID `#1`
  * Bounding Box: `(x, y, w, h) px`
  * Contour Area: `4,320 px`
  * Solidity: `0.78` (filters chaotic noise)
  * Aspect Ratio: `1.24` (filters linear lane stripes)
* [ ] Demonstrate the **ROI Magnifier**:
  * Select Candidate `#1` in the dropdown.
  * Show the zoomed RGB cavity crop side-by-side with the localized contrast depression crop.

### STEP 5: Safe Error Handling Demonstration (4:15 – 4:45)
* [ ] Return to **`⇡  Upload & Analyze`**.
* [ ] Upload an invalid file (e.g., a `.txt` file renamed or a non-image file).
* [ ] Point out the application behavior:
  * The app **does not crash** or show a Python traceback.
  * It displays a clean, user-friendly error card:  
    `"Image validation failed: Failed to decode image bytes. Unsupported or corrupted file format."`
* [ ] Re-load a valid road image to show instantaneous recovery.

### STEP 6: Multi-Format Research Exports (4:45 – 5:30)
* [ ] Navigate to **`▤  Reports & Metrics`**.
* [ ] Point out the exact end-to-end latency breakdown:
  * Preprocessing: `~12.4 ms`
  * Edge Detection: `~3.8 ms`
  * Candidate Extraction: `~5.9 ms`
  * Total Pipeline: **`~47.3 ms`** (~20 FPS throughput on CPU).
* [ ] Test the download actions:
  * Click **Download JSON Research Report**: Open the JSON file to show standardized image shape, parameter dictionary, per-detector latencies, and candidate bounding box coordinates.
  * Click **Download Processed Overlay (PNG)**.
* [ ] Navigate to **`◷  Analysis History`**:
  * Show that inspections executed during the session are logged with timestamps, dimensions, methods, and candidate counts.
  * Click **Download History (CSV)**.

### STEP 7: Conclude & Address Academic Limitations (5:30 – 6:30)
* [ ] Navigate to **`ⓘ  About Project`**.
* [ ] Discuss verified academic limitations openly:
  * *Candidates vs. Semantic Truth*: Shadows, painted markings, and oil stains produce gradients that can trigger candidate alerts.
  * *2D Monocular Depth Blindness*: 2D cameras cannot measure true cavity depth in centimeters without 3D stereo disparity.
* [ ] State future roadmap:
  * Cascading POTHOLE AI as an ultra-fast candidate proposer to a lightweight YOLOv8 classifier.
  * Multi-modal sensor fusion with vehicle IMU accelerometers and stereo disparity cameras.
* [ ] Close presentation and welcome examiner questions.

---

## 🛑 EMERGENCY TROUBLESHOOTING GUIDE

| Issue Encountered | Root Cause | Instant Fix During Live Demo |
| :--- | :--- | :--- |
| **Port 8501 is busy / App running on 8502** | Previous session did not release port. | Open `http://localhost:8502` directly in browser, or run `taskkill /F /IM python.exe` in PowerShell and re-run. |
| **Browser shows blank screen** | Browser cache or disconnected WebSocket. | Press `Ctrl + F5` to force hard reload. |
| **User accidentally inputs reversed Canny thresholds** | $T_{\text{low}} > T_{\text{high}}$ (e.g., 180 and 40). | Don't panic! The codebase automatically executes `low, high = min(t1, t2), max(t1, t2)`, auto-sorting them without errors. |
| **Uploaded image appears completely black** | Unsupported 16-bit or HDR format. | System auto-clamps to 8-bit uint8. Click **Load Scenario A** to return to benchmark imagery. |
| **Candidate count is zero** | Image has no closed high-contrast depressions. | Explain that this proves the geometric filter rejects clean road surfaces. Adjust minimum area slider or reload Scenario B. |
