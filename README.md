# Beam M² / BPP Analyzer

A small, vendor-agnostic tool to parse `.m2` text exports and compute beam quality metrics:

- per-frame beam widths (camera X/Y + principal axes)
- caustic fitting (ISO-11146 style) to get waist, divergence, BPP and M²
- GUI (PyQt5 or PyQt6) + CLI batch mode

## Install

Pick **one** Qt binding:

### Option A: PyQt6

```bash
pip install pyqt6 matplotlib numpy scipy pandas openpyxl pillow
```

### Option B: PyQt5

```bash
pip install pyqt5 matplotlib numpy scipy pandas openpyxl pillow
```

## Run the GUI

From this folder:

```bash
python run_gui.py
```

## Run the CLI

Single file -> workbook:

```bash
python -m beam_m2_app.cli /path/to/file.m2 --out results.xlsx
```

Batch (folder of `.m2`) -> summary workbook:

```bash
python -m beam_m2_app.cli /path/to/folder --batch --out summary.xlsx
```

## Notes on methods

- **M2 file moments (fast)** uses `XX/YY/XY` from the `.m2` file. This does not require the referenced TIFFs to be present.
- **Image 2nd moments (2σ)** computes centroid + second moments from the TIFF intensity (after robust background subtraction). Optional `drop_down` masks low intensity.
- **Image Gaussian fit (1/e²)** fits a rotated 2D Gaussian in physical coordinates (uses `facX/facY`).

The caustic fit is performed on `w(z)^2` via a quadratic fit:

```
w(z)^2 = A z^2 + B z + C
z0 = -B/(2A)
w0^2 = C - B^2/(4A)
theta = sqrt(A)
```

Then:

- `BPP = w0 * theta`
- `M² = pi * w0 * theta / lambda`

Units:
- `w` and `z` are in **mm**
- `theta` is in **rad** (reported as mrad in the UI)

