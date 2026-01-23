"""Export utilities (CSV, Excel, images)."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd

from .analysis import FrameWidths, M2Results
from .image_io import read_tiff_preview, robust_background
from .models import M2Measurement


def widths_to_dataframe(widths: List[FrameWidths]) -> pd.DataFrame:
    rows = []
    for w in widths:
        rows.append(
            {
                'index': w.index,
                'z': w.z,
                'w_x_mm': w.w_x,
                'w_y_mm': w.w_y,
                'w_major_mm': w.w_major,
                'w_minor_mm': w.w_minor,
                'angle_deg': w.angle_deg,
                'cx_px': w.cx,
                'cy_px': w.cy,
                'snr': w.snr,
                'filename': w.filename,
                'image_path': w.image_path,
            }
        )
    return pd.DataFrame(rows)


def results_to_dataframe(results: M2Results) -> pd.DataFrame:
    # Flatten key results for convenient batch tables
    r = {
        'wavelength_nm': results.wavelength_nm,
        'm2_x': results.fit_x.m2,
        'm2_y': results.fit_y.m2,
        'm2_geo_mean': results.m2_geo_mean,
        'bpp_x_mm_rad': results.fit_x.bpp,
        'bpp_y_mm_rad': results.fit_y.bpp,
        'bpp_geo_mean_mm_rad': results.bpp_geo_mean,
        # Human-friendly units
        'bpp_x_mm_mrad': results.fit_x.bpp * 1e3,
        'bpp_y_mm_mrad': results.fit_y.bpp * 1e3,
        'bpp_geo_mean_mm_mrad': results.bpp_geo_mean * 1e3,
        'w0_x_mm': results.fit_x.w0,
        'w0_y_mm': results.fit_y.w0,
        'theta_x_rad': results.fit_x.theta,
        'theta_y_rad': results.fit_y.theta,
        'theta_x_mrad': results.fit_x.theta * 1e3,
        'theta_y_mrad': results.fit_y.theta * 1e3,
        'z0_x': results.fit_x.z0,
        'z0_y': results.fit_y.z0,
        'zR_x': results.fit_x.zR,
        'zR_y': results.fit_y.zR,
        'fit_method': results.fit_x.method.value,
        'axis_mode': results.fit_x.axis_mode.value,
    }
    return pd.DataFrame([r])


def fit_summary_dataframe(results: M2Results) -> pd.DataFrame:
    rows = []
    for fit in (results.fit_x, results.fit_y):
        rows.append(
            {
                'axis': fit.axis,
                'method': fit.method.value,
                'axis_mode': fit.axis_mode.value,
                'w0_mm': fit.w0,
                'theta_rad': fit.theta,
                'theta_mrad': fit.theta * 1e3,
                'z0': fit.z0,
                'zR': fit.zR,
                'bpp_mm_rad': fit.bpp,
                'bpp_mm_mrad': fit.bpp * 1e3,
                'm2': fit.m2,
                'A': fit.A,
                'B': fit.B,
                'C': fit.C,
            }
        )
    return pd.DataFrame(rows)


def export_widths_csv(widths: List[FrameWidths], out_path: Union[str, Path]) -> Path:
    out = Path(out_path).expanduser().resolve()
    df = widths_to_dataframe(widths)
    df.to_csv(out, index=False)
    return out


def export_widths_excel(widths: List[FrameWidths], out_path: Union[str, Path], *, sheet: str = 'frames') -> Path:
    out = Path(out_path).expanduser().resolve()
    df = widths_to_dataframe(widths)
    df.to_excel(out, index=False, sheet_name=sheet)
    return out


def export_results_excel(
    results: M2Results,
    widths: Optional[List[FrameWidths]],
    out_path: Union[str, Path],
) -> Path:
    """Write a single workbook containing summary + per-frame widths."""
    out = Path(out_path).expanduser().resolve()

    with pd.ExcelWriter(out, engine='openpyxl') as writer:
        results_to_dataframe(results).to_excel(writer, index=False, sheet_name='summary')
        if widths is not None:
            widths_to_dataframe(widths).to_excel(writer, index=False, sheet_name='frames')

    return out


def export_single_report_excel(
    results: M2Results,
    widths: Optional[List[FrameWidths]],
    out_path: Union[str, Path],
    *,
    meas: Optional[M2Measurement] = None,
    image_max_dim: int = 256,
    colormap: str = 'viridis',
    gamma: float = 1.0,
) -> Path:
    """Write a single workbook containing summary + fit details + per-frame widths + images.

    Parameters
    ----------
    results : M2Results
        M² analysis results
    widths : List[FrameWidths], optional
        Per-frame beam width measurements (used for images sheet and frames sheet)
    out_path : str or Path
        Output Excel file path
    meas : M2Measurement, optional
        Measurement data for image embedding
    image_max_dim : int
        Maximum dimension for embedded images (default: 256)
    colormap : str
        Colormap for beam profile images (default: 'viridis')
    gamma : float
        Gamma correction for image display (default: 1.0)

    Returns
    -------
    Path
        Path to the created Excel file
    """
    out = Path(out_path).expanduser().resolve()

    with pd.ExcelWriter(out, engine='openpyxl') as writer:
        results_to_dataframe(results).to_excel(writer, index=False, sheet_name='summary')
        fit_summary_dataframe(results).to_excel(writer, index=False, sheet_name='fit')
        if widths is not None:
            widths_to_dataframe(widths).to_excel(writer, index=False, sheet_name='frames')

        if meas is not None:
            _add_images_sheet(
                writer,
                meas,
                widths=widths,
                image_max_dim=image_max_dim,
                colormap=colormap,
                gamma=gamma,
            )

    return out


def _normalize_image(image: np.ndarray) -> np.ndarray:
    img = np.asarray(image, dtype=float)
    if img.size == 0:
        return img
    bg = robust_background(img)
    img = img - float(bg)
    img[img < 0] = 0.0
    vmax = float(np.max(img)) if img.size else 0.0
    if vmax > 0:
        img = img / vmax
    return img


def _png_bytes_from_image(image: np.ndarray) -> "BytesIO":
    from io import BytesIO
    from PIL import Image

    img_u8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    pil = Image.fromarray(img_u8, mode='L')
    buf = BytesIO()
    pil.save(buf, format='PNG', optimize=True, compress_level=9)
    buf.seek(0)
    return buf


def _render_beam_overlay_image(
    image: np.ndarray,
    cx_px: float,
    cy_px: float,
    w_x_px: float,
    w_y_px: float,
    angle_deg: float = 0.0,
    colormap: str = 'viridis',
    gamma: float = 1.0,
) -> "BytesIO":
    """Render a beam profile image with overlay (ellipse + crosshair) as PNG bytes.

    Parameters
    ----------
    image : np.ndarray
        2D grayscale image (normalized 0-1 or will be normalized)
    cx_px, cy_px : float
        Centroid position in pixels
    w_x_px, w_y_px : float
        Beam radii in pixels (1/e² radii)
    angle_deg : float
        Rotation angle for the ellipse
    colormap : str
        Matplotlib colormap name (default: 'viridis')
    gamma : float
        Gamma correction (default: 1.0)

    Returns
    -------
    BytesIO
        PNG image bytes ready for embedding in Excel
    """
    from io import BytesIO
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    from matplotlib.transforms import Affine2D

    # Normalize image to 0-1 range
    img = np.asarray(image, dtype=float)
    if img.size == 0:
        # Return empty image
        buf = BytesIO()
        return buf

    vmin = float(np.min(img))
    vmax = float(np.max(img))
    if vmax > vmin:
        img = (img - vmin) / (vmax - vmin)
    else:
        img = np.zeros_like(img)

    # Apply gamma correction
    if gamma != 1.0 and gamma > 0:
        img = np.power(img, 1.0 / gamma)

    # Create figure with no margins
    h, w = img.shape[:2]
    dpi = 100
    fig_w = w / dpi
    fig_h = h / dpi

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    ax.set_position([0, 0, 1, 1])

    # Show image with colormap
    ax.imshow(img, cmap=colormap, aspect='equal', origin='upper')
    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(h - 0.5, -0.5)  # Invert y-axis to match image coordinates
    ax.axis('off')

    # Draw ellipse (red, 2px width)
    ellipse = Ellipse(
        (cx_px, cy_px),
        width=2 * w_x_px,
        height=2 * w_y_px,
        angle=angle_deg,
        fill=False,
        edgecolor='red',
        linewidth=2,
    )
    ax.add_patch(ellipse)

    # Draw crosshair (green)
    ch_size = max(w_x_px, w_y_px, 20)
    ax.plot([cx_px - ch_size, cx_px + ch_size], [cy_px, cy_px],
            color='lime', linewidth=1)
    ax.plot([cx_px, cx_px], [cy_px - ch_size, cy_px + ch_size],
            color='lime', linewidth=1)

    # Save to bytes
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, pad_inches=0,
                bbox_inches='tight', facecolor='black')
    plt.close(fig)
    buf.seek(0)
    return buf


def _add_images_sheet(
    writer: pd.ExcelWriter,
    meas: M2Measurement,
    *,
    widths: Optional[List["FrameWidths"]] = None,
    image_max_dim: int = 256,
    colormap: str = 'viridis',
    gamma: float = 1.0,
) -> None:
    """Add images sheet with beam profile overlays.

    Parameters
    ----------
    writer : pd.ExcelWriter
        Excel writer with openpyxl engine
    meas : M2Measurement
        Measurement data containing frame info
    widths : List[FrameWidths], optional
        Frame widths for overlay rendering. If provided, images will include
        ellipse and crosshair overlays showing beam measurements.
    image_max_dim : int
        Maximum dimension for preview images
    colormap : str
        Matplotlib colormap for image rendering
    gamma : float
        Gamma correction for display
    """
    from openpyxl.drawing.image import Image as XlImage

    wb = writer.book
    sheet = wb.create_sheet(title='images')
    writer.sheets['images'] = sheet
    sheet['A1'] = 'Frame'
    sheet['B1'] = 'z'

    frames = meas.active_frames()
    if not frames:
        sheet['A2'] = 'No frames'
        return

    # Build lookup for widths by frame index
    widths_by_index = {}
    if widths:
        for w in widths:
            widths_by_index[w.index] = w

    row = 2
    for frame in frames:
        sheet[f'A{row}'] = frame.index
        sheet[f'B{row}'] = float(frame.z)

        try:
            img, scale_x, scale_y = read_tiff_preview(
                meas.resolve_image_path(frame),
                max_dim=image_max_dim,
                return_scale=True,
            )
        except Exception:
            img = None
            scale_x = scale_y = 1.0

        if img is not None:
            # Normalize image
            norm = _normalize_image(img)

            # Check if we have width data for overlay
            fw = widths_by_index.get(frame.index)
            if fw is not None:
                # Convert centroid to preview coordinates
                cx_px = fw.cx * scale_x
                cy_px = fw.cy * scale_y

                # Convert radii from mm to pixels, then scale for preview
                # facX/facY are mm per pixel
                facX = frame.facX if frame.facX else 1.0
                facY = frame.facY if frame.facY else 1.0
                w_x_px = (fw.w_x / facX) * scale_x
                w_y_px = (fw.w_y / facY) * scale_y

                # Render with overlay
                img_bytes = _render_beam_overlay_image(
                    norm,
                    cx_px, cy_px,
                    w_x_px, w_y_px,
                    angle_deg=0.0,  # Use camera XY for Excel export
                    colormap=colormap,
                    gamma=gamma,
                )
            else:
                # Fallback to simple grayscale
                img_bytes = _png_bytes_from_image(norm)

            xl_img = XlImage(img_bytes)
            xl_img.anchor = f'C{row}'
            sheet.add_image(xl_img)

            sheet.row_dimensions[row].height = max(26, image_max_dim * 0.75)
            sheet.column_dimensions['C'].width = max(16, image_max_dim * 0.15)

        row += 1

