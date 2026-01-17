"""Beam width and M^2 analysis.

This module focuses on practical ISO-11146 style caustic fitting:

    w(z)^2 = w0^2 + (theta * (z - z0))^2

Where:
- w is 1/e^2 radius (mm)
- theta is far-field half-angle divergence (rad, small-angle approx)

Then:
- BPP = w0 * theta
- M^2 = pi * w0 * theta / lambda

We expose multiple width extraction methods:
- From the .m2 file second moments (fast, does not need image files)
- From image second moments (2nd moment / 2σ method)
- From image gaussian fit (1/e^2 parameters)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from scipy.optimize import least_squares

from .image_io import read_tiff, robust_background
from .models import M2Frame, M2Measurement


class WidthMethod(str, Enum):
    M2_FILE_MOMENTS = "m2_file_moments"
    IMAGE_2ND_MOMENTS = "image_2nd_moments"
    IMAGE_GAUSS_FIT = "image_gauss_fit"


class AxisMode(str, Enum):
    CAMERA_XY = "camera_xy"
    PRINCIPAL_AXES = "principal_axes"


@dataclass
class FrameWidths:
    index: int
    z: float

    # Beam radii (mm)
    w_x: float
    w_y: float

    # Principal axes radii (mm)
    w_major: float
    w_minor: float
    angle_deg: float

    # Centroid (px)
    cx: float
    cy: float

    # For debugging / UI
    snr: Optional[float] = None


@dataclass
class CausticFit:
    axis: str
    method: WidthMethod
    axis_mode: AxisMode

    z0: float
    w0: float
    theta: float

    bpp: float
    m2: float
    zR: float

    # Quadratic params for w^2 = A z^2 + B z + C
    A: float
    B: float
    C: float


@dataclass
class M2Results:
    """Full M2 fit results for a sequence."""

    wavelength_nm: float

    fit_x: CausticFit
    fit_y: CausticFit

    # Convenience combined metrics
    m2_geo_mean: float
    bpp_geo_mean: float


# ----------------------- Width extraction -----------------------

def _cov_from_frame_mm(frame: M2Frame) -> np.ndarray:
    """2x2 covariance matrix in mm^2 from .m2 per-frame XX/YY/XY."""
    cxx = frame.XX * (frame.facX ** 2)
    cyy = frame.YY * (frame.facY ** 2)
    cxy = frame.XY * (frame.facX * frame.facY)
    return np.array([[cxx, cxy], [cxy, cyy]], dtype=float)


def widths_from_m2_frame(frame: M2Frame) -> FrameWidths:
    """Compute beam widths from the vendor-provided second moments."""
    C = _cov_from_frame_mm(frame)

    # Camera axes (1/e^2 radius for Gaussian = 2σ)
    w_x = 2.0 * np.sqrt(max(C[0, 0], 0.0))
    w_y = 2.0 * np.sqrt(max(C[1, 1], 0.0))

    # Principal axes (eigenvalues)
    vals, vecs = np.linalg.eigh(C)
    # ascending order
    vmin = float(max(vals[0], 0.0))
    vmax = float(max(vals[1], 0.0))
    w_minor = 2.0 * np.sqrt(vmin)
    w_major = 2.0 * np.sqrt(vmax)

    major_vec = vecs[:, 1]
    angle = np.degrees(np.arctan2(major_vec[1], major_vec[0]))

    cx = frame.X0
    cy = frame.Y0

    return FrameWidths(
        index=frame.index,
        z=float(frame.z),
        w_x=float(w_x),
        w_y=float(w_y),
        w_major=float(w_major),
        w_minor=float(w_minor),
        angle_deg=float(angle),
        cx=float(cx),
        cy=float(cy),
        snr=frame.snr,
    )


def moments_from_image(
    image: np.ndarray,
    *,
    drop_down: Optional[float] = None,
    background: Optional[float] = None,
    background_border: int = 10,
) -> Tuple[float, float, float, float, float]:
    """Compute centroid and covariance from a 2D image.

    Parameters
    ----------
    drop_down:
        If set (0..1), ignore pixels below drop_down*max(image).
        e.g., 0.1353 ~ exp(-2) is common.

    Returns
    -------
    cx, cy, cxx, cyy, cxy
        Centroid (pixels) and covariance (pixels^2).
    """
    I = np.asarray(image, dtype=np.float64)

    if background is None:
        background = robust_background(I, border=background_border)

    I = I - float(background)
    I[I < 0] = 0.0

    if drop_down is not None:
        thr = float(drop_down) * float(I.max() if I.size else 0.0)
        mask = I > thr
        I = I * mask

    total = I.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError("Image contains no positive signal after preprocessing")

    ys, xs = np.indices(I.shape)
    cx = float((xs * I).sum() / total)
    cy = float((ys * I).sum() / total)

    dx = xs - cx
    dy = ys - cy

    cxx = float(((dx ** 2) * I).sum() / total)
    cyy = float(((dy ** 2) * I).sum() / total)
    cxy = float(((dx * dy) * I).sum() / total)

    return cx, cy, cxx, cyy, cxy


def widths_from_image_moments(
    image: np.ndarray,
    facX: float,
    facY: float,
    *,
    drop_down: Optional[float] = None,
) -> FrameWidths:
    """Compute widths from image 2nd moments (2σ / second moment method)."""
    cx, cy, xx, yy, xy = moments_from_image(image, drop_down=drop_down)

    # Convert to mm^2
    C = np.array(
        [[xx * (facX ** 2), xy * (facX * facY)], [xy * (facX * facY), yy * (facY ** 2)]],
        dtype=float,
    )

    w_x = 2.0 * np.sqrt(max(C[0, 0], 0.0))
    w_y = 2.0 * np.sqrt(max(C[1, 1], 0.0))

    vals, vecs = np.linalg.eigh(C)
    vmin = float(max(vals[0], 0.0))
    vmax = float(max(vals[1], 0.0))
    w_minor = 2.0 * np.sqrt(vmin)
    w_major = 2.0 * np.sqrt(vmax)
    major_vec = vecs[:, 1]
    angle = np.degrees(np.arctan2(major_vec[1], major_vec[0]))

    return FrameWidths(
        index=-1,
        z=float("nan"),
        w_x=float(w_x),
        w_y=float(w_y),
        w_major=float(w_major),
        w_minor=float(w_minor),
        angle_deg=float(angle),
        cx=float(cx),
        cy=float(cy),
        snr=None,
    )


def _gauss_model(params: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Elliptical rotated Gaussian with 1/e^2 radii."""
    amp, x0, y0, wx, wy, theta, offset = params
    # Keep widths positive
    wx = np.abs(wx) + 1e-12
    wy = np.abs(wy) + 1e-12

    ct = np.cos(theta)
    st = np.sin(theta)

    dx = xs - x0
    dy = ys - y0

    x_p = ct * dx + st * dy
    y_p = -st * dx + ct * dy

    expo = -2.0 * ((x_p ** 2) / (wx ** 2) + (y_p ** 2) / (wy ** 2))
    return offset + amp * np.exp(expo)


def gauss_fit_2d(
    image: np.ndarray,
    *,
    facX: float = 1.0,
    facY: float = 1.0,
    max_pixels: int = 250_000,
    drop_down: Optional[float] = 0.01,
) -> Dict[str, float]:
    """Fit a rotated 2D Gaussian and return parameters.

    Parameters
    ----------
    facX, facY:
        Physical pixel sizes. If provided, the fit is performed in physical
        coordinates (e.g., mm), which makes wx/wy directly comparable even if
        pixels are not square.

    Returns
    -------
    dict
        Keys include: x0, y0, wx, wy (all in *physical units*), theta (rad),
        theta_deg, amp, offset.

    Notes
    -----
    This is not meant to be a perfect metrology-grade implementation.
    It's a pragmatic fitting routine that works well for clean camera data.
    """
    I = np.asarray(image, dtype=np.float64)

    bg = robust_background(I)
    I = I - bg
    I[I < 0] = 0.0

    if not np.isfinite(I.max()) or I.max() <= 0:
        raise ValueError("Image has no usable signal")

    # Optionally limit to ROI to speed up fitting
    if drop_down is not None:
        thr = float(drop_down) * float(I.max())
        mask = I > thr
    else:
        mask = np.ones_like(I, dtype=bool)

    ys_full, xs_full = np.indices(I.shape)
    xs = xs_full[mask].astype(np.float64)
    ys = ys_full[mask].astype(np.float64)
    vals = I[mask].astype(np.float64)

    if vals.size > max_pixels:
        # Uniform random subsample
        rng = np.random.default_rng(0)
        idx = rng.choice(vals.size, size=max_pixels, replace=False)
        xs = xs[idx]
        ys = ys[idx]
        vals = vals[idx]

    # Convert to physical coordinates for fitting
    xs_p = xs * float(facX)
    ys_p = ys * float(facY)

    # Initial guess from moments (pixels -> physical)
    cx, cy, xx, yy, xy = moments_from_image(I, drop_down=drop_down)

    cx_p = float(cx) * float(facX)
    cy_p = float(cy) * float(facY)

    C_p = np.array(
        [[xx * (facX ** 2), xy * (facX * facY)], [xy * (facX * facY), yy * (facY ** 2)]],
        dtype=float,
    )
    eigvals, eigvecs = np.linalg.eigh(C_p)
    vmin, vmax = float(max(eigvals[0], 1e-18)), float(max(eigvals[1], 1e-18))

    # For a Gaussian, variance = w^2 / 4 => w = 2 sqrt(variance)
    w_major = 2.0 * np.sqrt(vmax)
    w_minor = 2.0 * np.sqrt(vmin)

    major_vec = eigvecs[:, 1]
    theta0 = float(np.arctan2(major_vec[1], major_vec[0]))

    amp0 = float(np.percentile(vals, 99))
    off0 = 0.0

    p0 = np.array([amp0, cx_p, cy_p, w_major, w_minor, theta0, off0], dtype=float)

    def residuals(p):
        return (_gauss_model(p, xs_p, ys_p) - vals).ravel()

    # Bounds: keep widths sensible
    w_phys = float(I.shape[1]) * float(facX)
    h_phys = float(I.shape[0]) * float(facY)
    lower = np.array([0.0, 0.0, 0.0, 1e-12, 1e-12, -np.pi, -np.inf])
    upper = np.array([np.inf, w_phys, h_phys, np.inf, np.inf, np.pi, np.inf])

    # Robust loss helps with hot pixels / mild clipping / speckle outliers.
    # It shouldn't magically fix a non-Gaussian beam, but it makes the fit
    # less fragile and reduces "tiny width" failures.
    try:
        f_scale = float(max(1.0, 0.1 * amp0))
    except Exception:
        f_scale = 1.0

    res = least_squares(
        residuals,
        p0,
        bounds=(lower, upper),
        method='trf',
        loss='soft_l1',
        f_scale=f_scale,
    )
    amp, x0, y0, wx, wy, theta, offset = res.x

    # Sort so wx >= wy (major/minor)
    wx = float(abs(wx))
    wy = float(abs(wy))
    if wy > wx:
        wx, wy = wy, wx
        theta = float(theta + np.pi / 2.0)

    return {
        'amp': float(amp),
        'x0': float(x0),
        'y0': float(y0),
        'wx': wx,
        'wy': wy,
        'theta': float(theta),
        'theta_deg': float(np.degrees(theta)),
        'offset': float(offset),
        'bg': float(bg),
        'success': bool(res.success),
        'cost': float(res.cost),
        'facX': float(facX),
        'facY': float(facY),
    }


def widths_from_image_gauss_fit(
    image: np.ndarray,
    facX: float,
    facY: float,
    *,
    drop_down: Optional[float] = None,
) -> FrameWidths:
    # For the Gaussian-fit method, a too-high threshold biases widths low.
    # If the user didn't specify one, use a gentle default.
    dd = 0.01 if drop_down is None else float(drop_down)
    fit = gauss_fit_2d(image, facX=facX, facY=facY, drop_down=dd)

    # Fit already returns 1/e^2 radii in physical units
    w_major = float(fit['wx'])
    w_minor = float(fit['wy'])

    # Camera-axis widths from covariance of the fitted Gaussian
    # variance = w^2 / 4
    var_major = (w_major ** 2) / 4.0
    var_minor = (w_minor ** 2) / 4.0
    ct = float(np.cos(fit['theta']))
    st = float(np.sin(fit['theta']))

    C_mm = np.array(
        [
            [ct ** 2 * var_major + st ** 2 * var_minor, ct * st * (var_major - var_minor)],
            [ct * st * (var_major - var_minor), st ** 2 * var_major + ct ** 2 * var_minor],
        ],
        dtype=float,
    )

    w_x = 2.0 * np.sqrt(max(C_mm[0, 0], 0.0))
    w_y = 2.0 * np.sqrt(max(C_mm[1, 1], 0.0))

    # Convert centroid back to pixels for overlay convenience
    cx_px = float(fit['x0']) / float(facX) if facX else float('nan')
    cy_px = float(fit['y0']) / float(facY) if facY else float('nan')

    return FrameWidths(
        index=-1,
        z=float("nan"),
        w_x=float(w_x),
        w_y=float(w_y),
        w_major=float(w_major),
        w_minor=float(w_minor),
        angle_deg=float(fit['theta_deg']),
        cx=float(cx_px),
        cy=float(cy_px),
        snr=None,
    )


def compute_frame_widths(
    meas: M2Measurement,
    *,
    method: WidthMethod = WidthMethod.M2_FILE_MOMENTS,
    drop_down: Optional[float] = None,
) -> List[FrameWidths]:
    """Compute widths for every active frame."""
    out: List[FrameWidths] = []

    for fr in meas.active_frames():
        if method == WidthMethod.M2_FILE_MOMENTS:
            fw = widths_from_m2_frame(fr)
            out.append(fw)
            continue

        # Image-based methods
        img_path = meas.resolve_image_path(fr)
        image = read_tiff(img_path)

        if method == WidthMethod.IMAGE_2ND_MOMENTS:
            fw = widths_from_image_moments(image, fr.facX, fr.facY, drop_down=drop_down)
        elif method == WidthMethod.IMAGE_GAUSS_FIT:
            fw = widths_from_image_gauss_fit(image, fr.facX, fr.facY, drop_down=drop_down)
        else:
            raise ValueError(f"Unsupported method: {method}")

        fw.index = fr.index
        fw.z = fr.z
        fw.snr = fr.snr
        out.append(fw)

    # Sort by z for fitting
    out.sort(key=lambda r: r.z)
    return out


# ----------------------- Caustic fitting -----------------------

def fit_caustic_quadratic(z: np.ndarray, w: np.ndarray) -> Tuple[float, float, float, float, float]:
    """Fit w(z)^2 = A z^2 + B z + C and derive (z0, w0, theta).

    Returns
    -------
    A, B, C, z0, w0, theta
    """
    z = np.asarray(z, dtype=float)
    w = np.asarray(w, dtype=float)

    if z.size < 3:
        raise ValueError("Need at least 3 points for quadratic caustic fit")

    y = w ** 2
    A, B, C = np.polyfit(z, y, 2)

    # Sanity: A should be positive (theta^2)
    if A <= 0:
        # Fall back to absolute (rare, but noisy data can do this)
        A = abs(A)

    z0 = -B / (2 * A)
    w0_sq = C - (B ** 2) / (4 * A)
    w0 = float(np.sqrt(max(w0_sq, 0.0)))
    theta = float(np.sqrt(max(A, 0.0)))

    return float(A), float(B), float(C), float(z0), float(w0), float(theta)


def compute_m2_results(
    meas: M2Measurement,
    widths: List[FrameWidths],
    *,
    method: WidthMethod,
    axis_mode: AxisMode,
    wavelength_nm: Optional[float] = None,
) -> M2Results:
    if wavelength_nm is None:
        if meas.wavelength_nm is None:
            raise ValueError("Wavelength not found in .m2 file; please provide wavelength_nm")
        wavelength_nm = float(meas.wavelength_nm)

    lam_mm = float(wavelength_nm) * 1e-6  # nm -> mm

    z = np.array([w.z for w in widths], dtype=float)

    if axis_mode == AxisMode.CAMERA_XY:
        wx = np.array([w.w_x for w in widths], dtype=float)
        wy = np.array([w.w_y for w in widths], dtype=float)
    elif axis_mode == AxisMode.PRINCIPAL_AXES:
        wx = np.array([w.w_major for w in widths], dtype=float)
        wy = np.array([w.w_minor for w in widths], dtype=float)
    else:
        raise ValueError(f"Unsupported axis_mode: {axis_mode}")

    Ax, Bx, Cx, z0x, w0x, thetax = fit_caustic_quadratic(z, wx)
    Ay, By, Cy, z0y, w0y, thetay = fit_caustic_quadratic(z, wy)

    bpp_x = w0x * thetax
    bpp_y = w0y * thetay

    m2_x = float(np.pi * bpp_x / lam_mm)
    m2_y = float(np.pi * bpp_y / lam_mm)

    zR_x = float(w0x / thetax) if thetax > 0 else float('nan')
    zR_y = float(w0y / thetay) if thetay > 0 else float('nan')

    fit_x = CausticFit(
        axis='X',
        method=method,
        axis_mode=axis_mode,
        z0=float(z0x),
        w0=float(w0x),
        theta=float(thetax),
        bpp=float(bpp_x),
        m2=m2_x,
        zR=zR_x,
        A=float(Ax),
        B=float(Bx),
        C=float(Cx),
    )

    fit_y = CausticFit(
        axis='Y',
        method=method,
        axis_mode=axis_mode,
        z0=float(z0y),
        w0=float(w0y),
        theta=float(thetay),
        bpp=float(bpp_y),
        m2=m2_y,
        zR=zR_y,
        A=float(Ay),
        B=float(By),
        C=float(Cy),
    )

    m2_geo = float(np.sqrt(m2_x * m2_y))
    bpp_geo = float(np.sqrt(bpp_x * bpp_y))

    return M2Results(
        wavelength_nm=float(wavelength_nm),
        fit_x=fit_x,
        fit_y=fit_y,
        m2_geo_mean=m2_geo,
        bpp_geo_mean=bpp_geo,
    )
