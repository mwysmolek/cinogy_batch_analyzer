"""Single-measurement report generation (PDF + Excel)."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union

import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure

from .analysis import AxisMode, FrameWidths, M2Results, WidthMethod
from .export import export_single_report_excel
from .image_io import read_tiff_preview, robust_background
from .models import M2Frame, M2Measurement


def _axis_width_arrays(widths: List[FrameWidths], axis_mode: AxisMode) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    z = np.array([w.z for w in widths], dtype=float)
    if axis_mode == AxisMode.CAMERA_XY:
        wx = np.array([w.w_x for w in widths], dtype=float)
        wy = np.array([w.w_y for w in widths], dtype=float)
    else:
        wx = np.array([w.w_major for w in widths], dtype=float)
        wy = np.array([w.w_minor for w in widths], dtype=float)
    return z, wx, wy


def _caustic_curve(z: np.ndarray, w0: float, theta: float, z0: float) -> np.ndarray:
    return np.sqrt((w0 ** 2) + (theta * (z - z0)) ** 2)


def _metric_rows(results: M2Results, method: WidthMethod, axis_mode: AxisMode) -> List[Tuple[str, str]]:
    rows: List[Tuple[str, str]] = [
        ("Wavelength (nm)", f"{results.wavelength_nm:.6g}"),
        ("Method", method.value),
        ("Axis mode", axis_mode.value),
        ("M² X", f"{results.fit_x.m2:.6g}"),
        ("M² Y", f"{results.fit_y.m2:.6g}"),
        ("M² geo-mean", f"{results.m2_geo_mean:.6g}"),
        ("BPP X (mm·mrad)", f"{results.fit_x.bpp * 1e3:.6g}"),
        ("BPP Y (mm·mrad)", f"{results.fit_y.bpp * 1e3:.6g}"),
        ("BPP geo-mean (mm·mrad)", f"{results.bpp_geo_mean * 1e3:.6g}"),
        ("w0 X (mm)", f"{results.fit_x.w0:.6g}"),
        ("w0 Y (mm)", f"{results.fit_y.w0:.6g}"),
        ("θ X (mrad)", f"{results.fit_x.theta * 1e3:.6g}"),
        ("θ Y (mrad)", f"{results.fit_y.theta * 1e3:.6g}"),
        ("z0 X", f"{results.fit_x.z0:.6g}"),
        ("z0 Y", f"{results.fit_y.z0:.6g}"),
        ("zR X", f"{results.fit_x.zR:.6g}"),
        ("zR Y", f"{results.fit_y.zR:.6g}"),
    ]
    return rows


def _add_summary_page(
    pdf: PdfPages,
    *,
    meas: M2Measurement,
    widths: List[FrameWidths],
    results: M2Results,
    method: WidthMethod,
    axis_mode: AxisMode,
) -> None:
    fig = Figure(figsize=(8.27, 11.69))
    ax = fig.add_subplot(111)
    ax.axis("off")

    fig.text(0.05, 0.95, "Beam M² Report (Single Measurement)", fontsize=16, weight="bold")
    fig.text(0.05, 0.92, f"Measurement: {meas.m2_path.name}", fontsize=11)
    fig.text(0.05, 0.90, f"Source path: {meas.m2_path}", fontsize=9)

    z_vals = np.array([w.z for w in widths], dtype=float)
    if z_vals.size:
        z_range = f"{np.nanmin(z_vals):.6g} .. {np.nanmax(z_vals):.6g}"
    else:
        z_range = "n/a"
    fig.text(0.05, 0.87, f"Frames: {len(widths)} (z range: {z_range})", fontsize=10)

    rows = _metric_rows(results, method, axis_mode)
    table_data = [[label, value] for label, value in rows]

    table = ax.table(
        cellText=table_data,
        colLabels=["Metric", "Value"],
        cellLoc="left",
        colLoc="left",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.5)

    pdf.savefig(fig)


def _add_caustic_page(
    pdf: PdfPages,
    *,
    widths: List[FrameWidths],
    results: M2Results,
    axis_mode: AxisMode,
) -> None:
    fig = Figure(figsize=(8.27, 11.69))
    ax = fig.add_subplot(111)

    z, wx, wy = _axis_width_arrays(widths, axis_mode)
    if z.size == 0:
        ax.text(0.5, 0.5, "No frame data for caustic plot", ha="center", va="center")
    else:
        order = np.argsort(z)
        z = z[order]
        wx = wx[order]
        wy = wy[order]

        z_dense = np.linspace(float(np.nanmin(z)), float(np.nanmax(z)), 200)
        wx_fit = _caustic_curve(z_dense, results.fit_x.w0, results.fit_x.theta, results.fit_x.z0)
        wy_fit = _caustic_curve(z_dense, results.fit_y.w0, results.fit_y.theta, results.fit_y.z0)

        ax.plot(z, wx, "o", label="X data")
        ax.plot(z_dense, wx_fit, "-", label="X fit")
        ax.plot(z, wy, "o", label="Y data")
        ax.plot(z_dense, wy_fit, "-", label="Y fit")
        ax.set_xlabel("z")
        ax.set_ylabel("Beam radius w (mm)")
        ax.grid(True)
        ax.legend()

    ax.set_title("Caustic Fit")
    fig.tight_layout()
    pdf.savefig(fig)


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


def _pad_to_shape(image: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    h, w = image.shape[:2]
    target_h, target_w = shape
    pad_h = max(target_h - h, 0)
    pad_w = max(target_w - w, 0)
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    return np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right)), mode="constant")


def _iter_image_pages(
    meas: M2Measurement,
    frames: Iterable[M2Frame],
    *,
    max_dim: int,
    per_page: int,
) -> Iterable[List[Tuple[M2Frame, Optional[np.ndarray]]]]:
    batch: List[Tuple[M2Frame, Optional[np.ndarray]]] = []
    for frame in frames:
        img = None
        try:
            img = read_tiff_preview(meas.resolve_image_path(frame), max_dim=max_dim)
        except Exception:
            img = None
        batch.append((frame, img))
        if len(batch) >= per_page:
            yield batch
            batch = []
    if batch:
        yield batch


def _add_image_pages(
    pdf: PdfPages,
    *,
    meas: M2Measurement,
    frames: List[M2Frame],
    max_dim: int,
    per_page: int,
) -> None:
    if not frames:
        fig = Figure(figsize=(8.27, 11.69))
        ax = fig.add_subplot(111)
        ax.axis("off")
        ax.text(0.5, 0.5, "No image frames found for this measurement", ha="center", va="center")
        pdf.savefig(fig)
        return

    for page in _iter_image_pages(meas, frames, max_dim=max_dim, per_page=per_page):
        images = [img for _, img in page if img is not None]
        if images:
            target_h = max(img.shape[0] for img in images)
            target_w = max(img.shape[1] for img in images)
            target_shape = (target_h, target_w)
        else:
            target_shape = (max_dim, max_dim)

        cols = 4
        rows = int(np.ceil(len(page) / float(cols)))
        fig = Figure(figsize=(8.27, 11.69))

        for i, (frame, img) in enumerate(page):
            r = i // cols
            c = i % cols
            ax = fig.add_subplot(rows, cols, i + 1)
            ax.axis("off")
            if img is None:
                ax.text(0.5, 0.5, "missing image", ha="center", va="center", fontsize=8)
            else:
                norm = _normalize_image(img)
                norm = _pad_to_shape(norm, target_shape)
                ax.imshow(norm, cmap="gray", origin="lower")
            ax.set_title(f"#{frame.index} z={frame.z:.3g}", fontsize=8)

        fig.suptitle("Reduced Beam Images (Normalized, Same Size)", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        pdf.savefig(fig)


def generate_single_report(
    meas: M2Measurement,
    widths: List[FrameWidths],
    results: M2Results,
    *,
    method: WidthMethod,
    axis_mode: AxisMode,
    pdf_path: Union[str, Path],
    excel_path: Union[str, Path],
    image_max_dim: int = 220,
    images_per_page: int = 12,
) -> Tuple[Path, Path]:
    pdf_out = Path(pdf_path).expanduser().resolve()
    excel_out = Path(excel_path).expanduser().resolve()

    excel_out.parent.mkdir(parents=True, exist_ok=True)
    pdf_out.parent.mkdir(parents=True, exist_ok=True)

    export_single_report_excel(results, widths, excel_out, meas=meas, image_max_dim=image_max_dim)

    frames = meas.active_frames()
    with PdfPages(pdf_out) as pdf:
        _add_summary_page(
            pdf,
            meas=meas,
            widths=widths,
            results=results,
            method=method,
            axis_mode=axis_mode,
        )
        _add_caustic_page(
            pdf,
            widths=widths,
            results=results,
            axis_mode=axis_mode,
        )
        _add_image_pages(
            pdf,
            meas=meas,
            frames=frames,
            max_dim=image_max_dim,
            per_page=images_per_page,
        )

    return pdf_out, excel_out
