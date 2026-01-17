"""Beam quality (M^2 / BPP) analysis utilities and GUI.

This project is intentionally lightweight and avoids vendor lock-in.
"""

from .m2_parser import parse_m2_file, M2Measurement, M2Frame
from .analysis import (
    WidthMethod,
    AxisMode,
    compute_frame_widths,
    fit_caustic_quadratic,
    compute_m2_results,
)
