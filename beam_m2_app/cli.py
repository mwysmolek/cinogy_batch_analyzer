"""Command line entry points.

Useful for batch automation and testing without the GUI.

Examples
--------
Single file:
    python -m beam_m2_app.cli /path/to/file.m2 --out results.xlsx

Folder (batch):
    python -m beam_m2_app.cli /path/to/folder --batch --out summary.xlsx
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import pandas as pd

from .analysis import AxisMode, WidthMethod, compute_frame_widths, compute_m2_results
from .export import export_results_excel, results_to_dataframe
from .m2_parser import parse_m2_file
from .report import generate_single_report


def _parse_method(s: str) -> WidthMethod:
    try:
        return WidthMethod(s)
    except Exception:
        raise argparse.ArgumentTypeError(f"Unknown width method: {s}")


def _parse_axis(s: str) -> AxisMode:
    try:
        return AxisMode(s)
    except Exception:
        raise argparse.ArgumentTypeError(f"Unknown axis mode: {s}")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description='Beam quality analysis from .m2 files')
    ap.add_argument('path', help='Path to .m2 file or folder')
    ap.add_argument('--batch', action='store_true', help='Treat path as folder and analyze all *.m2 inside')
    ap.add_argument('--method', type=_parse_method, default=WidthMethod.M2_FILE_MOMENTS,
                    choices=list(WidthMethod),
                    help='Width extraction method')
    ap.add_argument('--axis', type=_parse_axis, default=AxisMode.CAMERA_XY,
                    choices=list(AxisMode),
                    help='Axis mode for M2 fitting')
    ap.add_argument('--drop', type=float, default=None,
                    help='Drop-down threshold fraction for image moments (e.g. 0.1353)')
    ap.add_argument('--wavelength', type=float, default=None, help='Wavelength override (nm)')
    ap.add_argument('--out', type=str, default='results.xlsx', help='Output .xlsx path')
    ap.add_argument('--report', type=str, default=None,
                    help='Generate a single-measurement report (PDF + Excel) using this base path')

    args = ap.parse_args(argv)

    p = Path(args.path).expanduser().resolve()

    if not args.batch:
        meas = parse_m2_file(p)
        widths = compute_frame_widths(meas, method=args.method, drop_down=args.drop)
        results = compute_m2_results(meas, widths, method=args.method, axis_mode=args.axis, wavelength_nm=args.wavelength)
        export_results_excel(results, widths, args.out)
        print(f"Wrote {args.out}")
        if args.report:
            report_base = Path(args.report).expanduser()
            if report_base.suffix.lower() == '.pdf':
                pdf_path = report_base
                excel_path = report_base.with_suffix('.xlsx')
            elif report_base.suffix.lower() == '.xlsx':
                excel_path = report_base
                pdf_path = report_base.with_suffix('.pdf')
            else:
                pdf_path = report_base.with_suffix('.pdf')
                excel_path = report_base.with_suffix('.xlsx')
            pdf_out, excel_out = generate_single_report(
                meas,
                widths,
                results,
                method=args.method,
                axis_mode=args.axis,
                pdf_path=pdf_path,
                excel_path=excel_path,
            )
            print(f"Wrote {pdf_out}")
            print(f"Wrote {excel_out}")
        return 0

    # Batch
    files = sorted([x for x in p.rglob('*') if x.is_file() and x.suffix.lower() == '.m2'])
    if not files:
        raise SystemExit(f"No .m2 files found in {p}")

    rows = []
    for f in files:
        try:
            meas = parse_m2_file(f)
            widths = compute_frame_widths(meas, method=args.method, drop_down=args.drop)
            results = compute_m2_results(meas, widths, method=args.method, axis_mode=args.axis, wavelength_nm=args.wavelength)
            df = results_to_dataframe(results)
            df.insert(0, 'file', str(f.relative_to(p)))
            df.insert(1, 'status', 'OK')
            df.insert(2, 'error', '')
            rows.append(df)
        except Exception as e:
            err = pd.DataFrame([{'file': str(f.relative_to(p)), 'status': 'ERR', 'error': str(e)}])
            rows.append(err)

    out_df = pd.concat(rows, ignore_index=True)
    out_path = Path(args.out).expanduser().resolve()

    # Basic stats sheet
    def _stats(df: pd.DataFrame) -> pd.DataFrame:
        metrics = [
            ('m2_x', 'M² X'),
            ('m2_y', 'M² Y'),
            ('m2_geo_mean', 'M² geo-mean'),
            ('bpp_x_mm_mrad', 'BPP X (mm·mrad)'),
            ('bpp_y_mm_mrad', 'BPP Y (mm·mrad)'),
            ('bpp_geo_mean_mm_mrad', 'BPP geo-mean (mm·mrad)'),
        ]
        rows = []
        for col, label in metrics:
            if col not in df.columns:
                continue
            s = pd.to_numeric(df[col], errors='coerce').dropna()
            if s.empty:
                continue
            rows.append(
                {
                    'metric': label,
                    'count': int(s.count()),
                    'mean': float(s.mean()),
                    'std': float(s.std(ddof=1)) if len(s) > 1 else 0.0,
                    'min': float(s.min()),
                    'median': float(s.median()),
                    'max': float(s.max()),
                }
            )
        return pd.DataFrame(rows)

    stats_df = _stats(out_df)

    with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
        out_df.to_excel(writer, index=False, sheet_name='summary')
        if not stats_df.empty:
            stats_df.to_excel(writer, index=False, sheet_name='stats')
    print(f"Wrote {out_path}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
