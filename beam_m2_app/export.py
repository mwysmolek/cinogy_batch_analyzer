"""Export utilities (CSV, Excel, images)."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Iterable, List, Optional, Union

import numpy as np
import pandas as pd

from .analysis import FrameWidths, M2Results


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
) -> Path:
    """Write a single workbook containing summary + fit details + per-frame widths."""
    out = Path(out_path).expanduser().resolve()

    with pd.ExcelWriter(out, engine='openpyxl') as writer:
        results_to_dataframe(results).to_excel(writer, index=False, sheet_name='summary')
        fit_summary_dataframe(results).to_excel(writer, index=False, sheet_name='fit')
        if widths is not None:
            widths_to_dataframe(widths).to_excel(writer, index=False, sheet_name='frames')

    return out
