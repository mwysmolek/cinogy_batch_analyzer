# Beam M² / BPP Analyzer (GUI + Batch)

**Dedicated for Cinogy _CinSquare M² Tool_.**

A fast, cross-platform Python application for **laser beam quality analysis** from CinSquare-style M² measurement exports (`.m2`), with optional image-based verification from associated **TIFF** frames.

This project is built for lab + production workflows:
- **Batch first** (metadata-only, fast)
- **Deep-dive on demand** (load images only when you actually need them)
- **Exportable results** (Excel/CSV)
- **Interactive plots** (caustics, histograms, component comparisons)

---

## What this is for

Use this tool when you want to:
- Process **large batches** of M² measurements quickly
- Compute and trend **M²**, **BPP**, **waist**, **divergence**, and caustic fit parameters
- Compare results across **components/ports**, grouped by folder structure
- Validate vendor results using alternate width methods (moments, 2σ, Gaussian 1/e²)

### What it is *not*
This is **not** a certified metrology instrument. Results depend on measurement conditions (background, saturation, clipping, calibration, truncation, etc.) and should be validated in your setup.

---

## Key features

### Analysis methods
- **Metadata-only analysis from `.m2`** (recommended default for batches)
- Optional **image-based widths** (loads images only when requested):
  - **Second moments (2σ)** from intensity distribution
  - **2D rotated Gaussian fit (1/e²)**  
    _Note: can under-report beam size for non-Gaussian beams or clipped/saturated images._

### Batch workflow
- **Recursive scan**: select a root folder and it finds all `.m2` files in subdirectories
- **Component grouping**: treat a directory of `.m2` files as characterization of a component/port
- Batch summary and stats:
  - Caustic plots for selected measurements
  - **Histograms**, boxplots, and correlation scatter plots
  - Component comparison plots with selectable metrics (dynamic subplots)
- Plot X-axis can be shown as **row index (0..N-1)** for clarity, while file names remain in the table

### GUI usability
- Responsive, modern **PyQt GUI** (compatible with **PyQt5 and PyQt6**)
- Batch processing runs in a **background worker** with progress + cancel
- **Double-click** a batch row to open that measurement in the Single analysis view

### Images (optional)
- **Lazy loading**: loads one frame at a time (not all TIFFs)
- Downsampled preview for speed + lower memory use
- False-color display (Viridis/Inferno/Magma/Plasma/Turbo + grayscale), gamma controls

### Export
- **Excel (.xlsx)**: summary + stats (+ component-level stats)
- **CSV**: summary tables
- Optional plot/image export (depending on configuration)

---

## Input expectations

### `.m2` files
The parser reads typical M² export fields such as:
- Per-frame: `z`, `XX`, `YY`, `XY`, `X0`, `Y0`, `facX`, `facY`, and file references
- Global: wavelength (if present)

It computes beam widths from moments and fits the caustic to derive **M²** and **BPP**.

### TIFF frames (optional)
Only required for image-based analysis or image preview. TIFFs can be large; the default workflow avoids loading them in batch mode.

---

## Installation

### Requirements
- Python **3.9+** recommended
- One Qt binding:
  - **PyQt6** (recommended), or
  - **PyQt5**

### Create a virtual environment (recommended)

**Windows (PowerShell)**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**Linux**
```bash
python -m venv .venv
source .venv/bin/activate
```

### Install dependencies

**PyQt6**
```bash
pip install pyqt6 numpy scipy pandas matplotlib pillow openpyxl
```

**PyQt5**
```bash
pip install pyqt5 numpy scipy pandas matplotlib pillow openpyxl
```

---

## Quick start

### Run the GUI
```bash
python run_gui.py
```

### Recommended workflow (fast + scalable)
1. Go to **Batch**
2. Select a root folder
3. Use method **“M2 file moments (fast)”** for large batches (no image I/O)
4. Review histograms/correlation plots/component plots
5. Double-click any interesting measurement to open it in **Single**
6. Only then enable image preview or image-based analysis (2σ / Gaussian)

---

## Using the GUI

### Batch analysis
1. Open the **Batch** tab
2. Choose a root directory (recursive scan for `.m2`)
3. Select grouping mode (component inference) and metrics to visualize
4. Use the plots and stats to identify drift/outliers
5. Double-click a row to open it in **Single** view

#### Component grouping convention
A common layout looks like this:
```
/maindir/
  DeviceA/
    Port_001.m2
    Port_002.m2
    ...
  DeviceB/
    Port_001.m2
    Port_002.m2
    ...
```

The tool can treat `DeviceA` and `DeviceB` as **components**, allowing direct per-component comparisons.

### Single analysis
1. Open a single `.m2`
2. Inspect:
   - w(z) vs z (per-axis)
   - fitted caustic curve
   - derived parameters: M², BPP, waist, divergence
3. If needed:
   - enable image preview
   - select colormap and preview scaling
   - switch to image-based widths for verification

---

## CLI usage (automation)

Run a single file:
```bash
python -m beam_m2_app.cli /path/to/file.m2 --out results.xlsx
```

Run batch recursively:
```bash
python -m beam_m2_app.cli /path/to/root --batch --recursive --out batch_summary.xlsx
```

Export CSV instead:
```bash
python -m beam_m2_app.cli /path/to/root --batch --recursive --out batch_summary.csv
```

Use `--help` for full options:
```bash
python -m beam_m2_app.cli --help
```

---

## Notes on interpretation

### Moment-based (from `.m2`) is usually the right default
- Very fast
- No TIFF loading
- Aligns with vendor/internal calculations

### Gaussian 1/e² can produce “too small” widths
A Gaussian fit emphasizes the core. It may under-report width for:
- non-Gaussian beams (wings, multimode)
- saturated/clipped frames
- poor background subtraction
- hot pixels/speckle artifacts

Use 2σ (moments) as a robust cross-check and inspect residuals/fit quality.

---

## Performance and image I/O

This tool is designed to avoid unnecessary TIFF reads:
- Batch defaults to **metadata-only** analysis
- Images are loaded **only when you request image-based methods or previews**
- Previews are typically **downsampled** for speed and memory safety

---

## Contributing

Contributions are welcome, especially:
- additional width methods / robust fitting options
- better reporting exports (PDF summaries, plot bundles)
- test fixtures and regression tests

Please include:
- minimal sample `.m2` files (and a few frames, if needed)
- expected outputs (M²/BPP) for validation

---

## License

Released under the **MIT License**. See `LICENSE`.

---

## Disclaimer

Provided “as is” without warranty. Intended for engineering use and not a certified metrology instrument. Validate results in your measurement setup.
