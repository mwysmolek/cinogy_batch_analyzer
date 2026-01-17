"""Image I/O utilities.

We prefer Pillow for TIFF because it handles common compressions without
extra native dependencies (e.g., LZW). If Pillow fails, we fall back to
`tifffile`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np


def downsample_max_dim(image: np.ndarray, max_dim: int) -> np.ndarray:
    """Downsample an image by simple stride so max(h, w) <= max_dim.

    This is intended for *preview / UI rendering*, not metrology.
    It's fast, uses little memory, and avoids pulling multi-megapixel
    images into the GUI at full resolution.
    """
    try:
        max_dim = int(max_dim)
    except Exception:
        return image

    if max_dim <= 0:
        return image

    if image is None:
        return image

    h, w = image.shape[:2]
    m = max(h, w)
    if m <= max_dim:
        return image

    # Stride factor (ceil)
    step = int(np.ceil(m / float(max_dim)))
    step = max(step, 1)
    return image[::step, ::step]


def read_tiff_preview(
    path: Union[str, Path],
    *,
    max_dim: int = 1024,
    return_scale: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, float, float]]:
    """Read a TIFF file into a 2D array, optionally downsampled for preview.

    Parameters
    ----------
    max_dim:
        Target maximum dimension of the preview (pixels).
    return_scale:
        If True, also return (scale_x, scale_y) mapping original pixel
        coordinates to preview coordinates.

    We *try* to downsample early using Pillow's resize. If that fails,
    we fall back to full read + stride downsample.
    """

    p = Path(path).expanduser().resolve()

    # Pillow path with early resize
    try:
        from PIL import Image

        im = Image.open(p)
        if getattr(im, 'n_frames', 1) > 1:
            im.seek(0)

        w, h = im.size
        m = max(w, h)
        sx = 1.0
        sy = 1.0
        if max_dim and m > int(max_dim):
            scale = float(max_dim) / float(m)
            nw = max(1, int(round(w * scale)))
            nh = max(1, int(round(h * scale)))

            sx = float(nw) / float(w) if w else 1.0
            sy = float(nh) / float(h) if h else 1.0

            # Resize in a mode that Pillow is comfortable with.
            # Paletted ('P') can be resized directly (keeps indices).
            # 16-bit and float modes also generally work.
            im = im.resize((nw, nh), resample=Image.BILINEAR)

        # Convert depending on mode
        if im.mode in ('I;16', 'I;16B', 'I;16L', 'I', 'F', 'L', 'P'):
            arr = np.array(im, dtype=np.float64)
        else:
            arr = np.array(im.convert('L'), dtype=np.float64)

        # Safety: enforce 2D
        if arr.ndim == 3:
            arr = arr[..., 0]
        if return_scale:
            return arr, sx, sy
        return arr
    except Exception:
        # Fallback: robust full read
        arr = read_tiff(p)
        h0, w0 = arr.shape[:2]
        arr2 = downsample_max_dim(arr, int(max_dim))
        h1, w1 = arr2.shape[:2]
        sx = float(w1) / float(w0) if w0 else 1.0
        sy = float(h1) / float(h0) if h0 else 1.0
        if return_scale:
            return arr2, sx, sy
        return arr2


def read_tiff(path: Union[str, Path]) -> np.ndarray:
    """Read a TIFF file into a 2D numpy array.

    Returns a float64 array. If the TIFF is RGB, it is converted to luminance.
    If the TIFF is paletted (mode 'P'), the index values are returned (the
    palette is display-only).
    """

    p = Path(path).expanduser().resolve()

    # Pillow first
    try:
        from PIL import Image

        im = Image.open(p)
        # Multi-page TIFF: use first page for now
        if getattr(im, 'n_frames', 1) > 1:
            im.seek(0)

        # Convert depending on mode
        if im.mode in ('I;16', 'I;16B', 'I;16L', 'I'):
            arr = np.array(im, dtype=np.float64)
        elif im.mode in ('F',):
            arr = np.array(im, dtype=np.float64)
        elif im.mode == 'P':
            # Palette indices, 0-255
            arr = np.array(im, dtype=np.float64)
        elif im.mode in ('L',):
            arr = np.array(im, dtype=np.float64)
        else:
            # RGB / RGBA etc -> convert to grayscale
            im_g = im.convert('L')
            arr = np.array(im_g, dtype=np.float64)

        return arr

    except Exception:
        pass

    # Fallback: tifffile (may require imagecodecs for some compressions)
    try:
        import tifffile

        arr = tifffile.imread(str(p))
        if arr.ndim == 3:
            # Simple RGB -> luminance
            arr = arr[..., :3]
            arr = 0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]
        return arr.astype(np.float64)
    except Exception as e:
        raise RuntimeError(f"Failed to read TIFF '{p}': {e}")


def robust_background(image: np.ndarray, border: int = 10) -> float:
    """Estimate background from border pixels (median)."""
    h, w = image.shape[:2]
    b = int(max(1, min(border, h // 2, w // 2)))
    top = image[:b, :]
    bottom = image[-b:, :]
    left = image[:, :b]
    right = image[:, -b:]
    samp = np.concatenate([top.ravel(), bottom.ravel(), left.ravel(), right.ravel()])
    return float(np.median(samp))
