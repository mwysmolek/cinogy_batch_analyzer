"""PyQt GUI for beam quality analysis.

This is a practical front-end for:
- Parsing vendor .m2 files
- Showing per-frame beam widths
- Fitting caustics and reporting M^2 / BPP
- Batch processing many .m2 files

The GUI aims to be PyQt5/PyQt6 compatible.
"""

from __future__ import annotations

import traceback
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

# Matplotlib backend changed names over time. Prefer the modern QtAgg backend,
# but fall back to Qt5Agg for older Matplotlib installs.
try:  # pragma: no cover
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
except Exception:  # pragma: no cover
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from .qt_compat import (
    QtCore,
    QtGui,
    QtWidgets,
    Qt,
    QAction,
    qexec,
    qsignal,
    AlignCenter,
    KeepAspectRatio,
    DisplayRole,
    Horizontal,
    Vertical,
    ItemIsEnabled,
    ItemIsSelectable,
    SelectRows,
    SingleSelection,
    ScrollHandDrag,
    AnchorUnderMouse,
    Antialiasing,
    QImage_Grayscale8,
    QImage_ARGB32,
    QImage_RGB888,
)
from .analysis import (
    AxisMode,
    WidthMethod,
    FrameWidths,
    M2Results,
    compute_frame_widths,
    compute_m2_results,
    widths_from_m2_frame,
    widths_from_image_moments,
    widths_from_image_gauss_fit,
)
from .export import export_results_excel, export_single_report_excel, export_widths_csv, export_widths_excel, results_to_dataframe, widths_to_dataframe
from .image_io import read_tiff, read_tiff_preview
from .m2_parser import parse_m2_file


class DataFrameModel(QtCore.QAbstractTableModel):
    """Minimal QAbstractTableModel wrapper for pandas DataFrames."""

    def __init__(self, df: Optional[pd.DataFrame] = None, parent=None):
        super().__init__(parent)
        self._df = df if df is not None else pd.DataFrame()

    def set_dataframe(self, df: pd.DataFrame) -> None:
        self.beginResetModel()
        self._df = df.copy()
        self.endResetModel()

    def dataframe(self) -> pd.DataFrame:
        """Return the model's current dataframe (sorted order included)."""
        return self._df

    def rowCount(self, parent=QtCore.QModelIndex()):
        return 0 if parent.isValid() else len(self._df)

    def columnCount(self, parent=QtCore.QModelIndex()):
        return 0 if parent.isValid() else len(self._df.columns)

    def data(self, index: QtCore.QModelIndex, role=DisplayRole):
        if not index.isValid():
            return None
        if role != DisplayRole:
            return None
        val = self._df.iat[index.row(), index.column()]
        if isinstance(val, float):
            if np.isnan(val):
                return ''
            return f"{val:.6g}"
        return str(val)

    def headerData(self, section: int, orientation, role=DisplayRole):
        if role != DisplayRole:
            return None
        if orientation == Horizontal:
            return str(self._df.columns[section])
        else:
            return str(section)

    def flags(self, index: QtCore.QModelIndex):
        if not index.isValid():
            return QtCore.Qt.ItemFlag.NoItemFlags if hasattr(QtCore.Qt, 'ItemFlag') else QtCore.Qt.NoItemFlags
        return ItemIsEnabled | ItemIsSelectable

    def sort(self, column: int, order) -> None:
        """Enable clickable header sorting.

        Qt hands us an order enum that differs slightly between Qt5/Qt6.
        We keep it simple and rely on pandas for the actual sorting.
        """
        if self._df is None or self._df.empty:
            return
        if column < 0 or column >= len(self._df.columns):
            return

        colname = self._df.columns[column]
        # Robust check for ascending/descending across Qt versions.
        asc = True
        try:
            asc = bool(int(order) == int(getattr(Qt, 'AscendingOrder', 0)))
        except Exception:
            # Fallback: string contains 'Ascending'
            asc = 'Ascending' in str(order)

        self.layoutAboutToBeChanged.emit()
        try:
            self._df = self._df.sort_values(colname, ascending=asc, kind='mergesort').reset_index(drop=True)
        except Exception:
            # If a column contains unorderable mixed types, pandas can throw.
            pass
        self.layoutChanged.emit()


@dataclass
class BatchEntry:
    """One analyzed .m2 file in batch mode."""

    path: Path
    rel_path: str
    results: Optional[M2Results] = None
    widths: Optional[List[FrameWidths]] = None
    error: Optional[str] = None


# ----------------------------
# Metric registry (batch plots)
# ----------------------------

# (Human label, dataframe column)
METRICS: List[Tuple[str, str]] = [
    ('M² geo-mean', 'm2_geo_mean'),
    ('M² X', 'm2_x'),
    ('M² Y', 'm2_y'),
    ('BPP geo-mean (mm·mrad)', 'bpp_geo_mean_mm_mrad'),
    ('BPP X (mm·mrad)', 'bpp_x_mm_mrad'),
    ('BPP Y (mm·mrad)', 'bpp_y_mm_mrad'),
    ('w0 X (mm)', 'w0_x_mm'),
    ('w0 Y (mm)', 'w0_y_mm'),
    ('θ X (mrad)', 'theta_x_mrad'),
    ('θ Y (mrad)', 'theta_y_mrad'),
    ('z0 X', 'z0_x'),
    ('z0 Y', 'z0_y'),
    ('zR X', 'zR_x'),
    ('zR Y', 'zR_y'),
]

METRIC_LABEL: Dict[str, str] = {col: label for (label, col) in METRICS}


def _metric_label(col: str) -> str:
    return METRIC_LABEL.get(col, col)


# ----------------------------
# Component grouping
# ----------------------------

GROUP_MODES: List[Tuple[str, str]] = [
    ('First subdir under root', 'root_child'),
    ('Parent folder', 'parent'),
    ('None', 'none'),
]


def _component_from_rel(rel_path: str, group_mode: str) -> str:
    """Map a relative path string to a component/group name.

    Typical lab layout:
        /root/componentA/001.m2
        /root/componentB/001.m2
    """
    p = Path(rel_path)
    if group_mode == 'none':
        return '(all)'

    if group_mode == 'parent':
        name = p.parent.name
        return name if name else '(root)'

    # Default: first child folder under the chosen root
    parts = p.parts
    if len(parts) >= 2:
        return parts[0]
    return '(root)'


def _format_results_text(res: M2Results) -> str:
    fx = res.fit_x
    fy = res.fit_y
    txt = []
    txt.append(f"Wavelength: {res.wavelength_nm:g} nm")
    txt.append('')
    txt.append(f"M² X: {fx.m2:.4g}   M² Y: {fy.m2:.4g}   geo-mean: {res.m2_geo_mean:.4g}")
    txt.append(
        f"BPP X: {fx.bpp*1e3:.4g} mm·mrad   BPP Y: {fy.bpp*1e3:.4g} mm·mrad   geo-mean: {res.bpp_geo_mean*1e3:.4g}"
    )
    txt.append('')
    txt.append('Note: w is 1/e² *radius* (mm). Diameter D = 2w. Full-angle divergence = 2θ.')
    txt.append('')
    txt.append(
        f"waist w0 X: {fx.w0:.4g} mm (D0: {2*fx.w0:.4g} mm)   "
        f"w0 Y: {fy.w0:.4g} mm (D0: {2*fy.w0:.4g} mm)"
    )
    txt.append(
        f"divergence θ X: {fx.theta*1e3:.4g} mrad (full: {2*fx.theta*1e3:.4g} mrad)   "
        f"θ Y: {fy.theta*1e3:.4g} mrad (full: {2*fy.theta*1e3:.4g} mrad)"
    )
    txt.append(f"z0 X: {fx.z0:.4g}   z0 Y: {fy.z0:.4g}")
    txt.append(f"zR X: {fx.zR:.4g}   zR Y: {fy.zR:.4g}")
    return '\n'.join(txt)


def _compute_batch_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute basic statistics for common batch metrics."""

    metrics = [(col, label) for (label, col) in METRICS]

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


def _compute_component_stats(df: pd.DataFrame, *, group_col: str = 'component') -> pd.DataFrame:
    """Aggregate results per component/group.

    Returns a *wide* table (one row per component) with count and mean/std
    for common metrics, suitable for Excel export and quick QA.
    """
    if df is None or df.empty or group_col not in df.columns:
        return pd.DataFrame()

    work = df.copy()
    if 'status' in work.columns:
        work = work[work['status'] == 'OK']
    if work.empty:
        return pd.DataFrame()

    g = work.groupby(group_col, dropna=False)

    sizes = g.size()
    out = pd.DataFrame({group_col: list(sizes.index), 'n': sizes.to_numpy(dtype=int)})

    for _label, col in METRICS:
        if col not in work.columns:
            continue

        vals = pd.to_numeric(work[col], errors='coerce')
        tmp = work[[group_col]].copy()
        tmp['_v'] = vals

        means = tmp.groupby(group_col, dropna=False)['_v'].mean()
        stds = tmp.groupby(group_col, dropna=False)['_v'].std(ddof=1).fillna(0.0)

        out[f'{col}_mean'] = out[group_col].map(means).to_numpy(dtype=float)
        out[f'{col}_std'] = out[group_col].map(stds).to_numpy(dtype=float)

    # Sort: most common components first (useful when you have a billion)
    try:
        out = out.sort_values(['n', group_col], ascending=[False, True]).reset_index(drop=True)
    except Exception:
        out = out.reset_index(drop=True)
    return out


class ImageView(QtWidgets.QGraphicsView):
    """Image viewer with a simple ellipse + centroid overlay."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QtWidgets.QGraphicsScene(self))
        self._pix_item = None
        self._ellipse_item = None
        self._crosshair_items = []

        self.setRenderHint(Antialiasing, True)
        self.setDragMode(ScrollHandDrag)
        self.setTransformationAnchor(AnchorUnderMouse)

    def wheelEvent(self, event: QtGui.QWheelEvent):
        # Zoom on wheel
        delta = event.angleDelta().y() if hasattr(event, 'angleDelta') else event.delta()
        if delta == 0:
            return
        factor = 1.25 if delta > 0 else 0.8
        self.scale(factor, factor)

    def set_image(
        self,
        image: np.ndarray,
        *,
        colormap: str = 'viridis',
        gamma: float = 1.0,
    ) -> None:
        """Display an image.

        Parameters
        ----------
        colormap:
            'gray' for grayscale, or a Matplotlib colormap name like 'viridis'.
        gamma:
            Gamma applied after autoscale (>=0.1). 1.0 means no gamma.
        """
        img = np.asarray(image)
        if img.ndim != 2:
            raise ValueError('Expected 2D image')

        # Robust autoscale
        lo, hi = np.percentile(img, [1, 99])
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = float(np.min(img)), float(np.max(img))
            if hi <= lo:
                hi = lo + 1

        scaled = (img.astype(np.float64) - lo) / (hi - lo)
        scaled = np.clip(scaled, 0, 1)

        try:
            g = float(gamma)
        except Exception:
            g = 1.0
        if g <= 0:
            g = 1.0
        if abs(g - 1.0) > 1e-6:
            # Perceptual boost for dim beams
            scaled = np.power(scaled, 1.0 / g)

        u8 = (scaled * 255).astype(np.uint8)

        h, w = u8.shape

        cmap = (colormap or 'gray').strip().lower()
        if cmap in ('gray', 'grey', 'grayscale'):
            qimg = QtGui.QImage(u8.data, w, h, w, QImage_Grayscale8).copy()
        else:
            # False color using Matplotlib colormap
            try:
                import matplotlib.cm as cm

                lut = cm.get_cmap(cmap)
                rgba = (lut(u8.astype(np.float32) / 255.0) * 255).astype(np.uint8)
                rgb = np.ascontiguousarray(rgba[:, :, :3])
                qimg = QtGui.QImage(rgb.data, w, h, 3 * w, QImage_RGB888).copy()
            except Exception:
                # Fallback: grayscale
                qimg = QtGui.QImage(u8.data, w, h, w, QImage_Grayscale8).copy()

        pix = QtGui.QPixmap.fromImage(qimg)

        scene = self.scene()
        # Clearing the scene *also* deletes any previous overlay items.
        # If we keep stale Python references and later try to remove them again,
        # Qt will complain with:
        #   QGraphicsScene::removeItem: item's scene (0x0) is different...
        scene.clear()
        self._ellipse_item = None
        self._crosshair_items = []
        self._pix_item = scene.addPixmap(pix)
        self._pix_item.setZValue(0)
        scene.setSceneRect(0, 0, w, h)

        self.resetTransform()
        self.fitInView(scene.sceneRect(), KeepAspectRatio)

    def set_overlay(self, cx_px: float, cy_px: float, w_x_px: float, w_y_px: float, angle_deg: float = 0.0) -> None:
        """Overlay an ellipse defined by radii in pixels."""
        scene = self.scene()
        if self._pix_item is None:
            return

        def _safe_remove(item):
            if item is None:
                return
            try:
                if item.scene() is scene:
                    scene.removeItem(item)
            except Exception:
                pass

        # Remove old items safely.
        _safe_remove(self._ellipse_item)
        self._ellipse_item = None
        for it in list(self._crosshair_items):
            _safe_remove(it)
        self._crosshair_items = []

        # Ellipse centered at (cx, cy) with radii w_x, w_y
        rect = QtCore.QRectF(cx_px - w_x_px, cy_px - w_y_px, 2 * w_x_px, 2 * w_y_px)
        pen = QtGui.QPen(QtGui.QColor(255, 0, 0))
        pen.setWidth(2)

        self._ellipse_item = scene.addEllipse(rect, pen)
        self._ellipse_item.setZValue(10)
        self._ellipse_item.setTransformOriginPoint(cx_px, cy_px)
        self._ellipse_item.setRotation(angle_deg)

        # Crosshair
        ch_pen = QtGui.QPen(QtGui.QColor(0, 255, 0))
        ch_pen.setWidth(1)
        size = max(w_x_px, w_y_px, 20)
        hline = scene.addLine(cx_px - size, cy_px, cx_px + size, cy_px, ch_pen)
        vline = scene.addLine(cx_px, cy_px - size, cx_px, cy_px + size, ch_pen)
        hline.setZValue(11)
        vline.setZValue(11)
        self._crosshair_items = [hline, vline]


    def save_png(self, path: Union[str, Path]) -> None:
        'Save the current scene (image + overlays) as a PNG.'
        from pathlib import Path as _Path

        p = _Path(path).expanduser().resolve()
        scene = self.scene()
        rect = scene.sceneRect()
        w = int(max(1, rect.width()))
        h = int(max(1, rect.height()))

        img = QtGui.QImage(w, h, QImage_ARGB32)
        img.fill(QtGui.QColor(0, 0, 0, 255))

        painter = QtGui.QPainter(img)
        scene.render(painter)
        painter.end()

        img.save(str(p))


class FitPlot(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.fig = Figure(figsize=(5, 3), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

    def plot(self, widths_df: pd.DataFrame, results=None, axis_mode: AxisMode = AxisMode.CAMERA_XY):
        self.fig.clear()
        ax = self.fig.add_subplot(111)

        if widths_df is None or widths_df.empty:
            ax.set_title('No data')
            self.canvas.draw()
            return

        z = widths_df['z'].to_numpy(dtype=float)
        if axis_mode == AxisMode.CAMERA_XY:
            wx = widths_df['w_x_mm'].to_numpy(dtype=float)
            wy = widths_df['w_y_mm'].to_numpy(dtype=float)
            labelx, labely = 'w_x', 'w_y'
        else:
            wx = widths_df['w_major_mm'].to_numpy(dtype=float)
            wy = widths_df['w_minor_mm'].to_numpy(dtype=float)
            labelx, labely = 'w_major', 'w_minor'

        ax.plot(z, wx, 'o', label=labelx)
        ax.plot(z, wy, 'o', label=labely)

        if results is not None:
            # Plot fitted curves: w(z) = sqrt(A z^2 + B z + C)
            z_fit = np.linspace(np.nanmin(z), np.nanmax(z), 250)
            fx = results.fit_x
            fy = results.fit_y
            wx_fit = np.sqrt(np.maximum(fx.A * z_fit**2 + fx.B * z_fit + fx.C, 0))
            wy_fit = np.sqrt(np.maximum(fy.A * z_fit**2 + fy.B * z_fit + fy.C, 0))
            ax.plot(z_fit, wx_fit, '-', label=f'fit {labelx}')
            ax.plot(z_fit, wy_fit, '-', label=f'fit {labely}')

        ax.set_xlabel('z')
        ax.set_ylabel('1/e² radius w (mm)')
        ax.grid(True)
        ax.legend()

        self.canvas.draw()


class BatchSummaryPlot(QtWidgets.QWidget):
    """Simple batch plot: one metric across many files.

    Design choice: x-axis is always the *table row index* (0, 1, 2, ...).
    File names stay in the table so the plot stays readable.

    Bonus: click a point to select the corresponding row in the batch table.
    """

    point_clicked = qsignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.fig = Figure(figsize=(5, 3), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

        self._line = None
        # One-time MPL event hook (safe to keep across redraws)
        try:
            self.canvas.mpl_connect('pick_event', self._on_pick)
        except Exception:
            pass

    def _on_pick(self, event) -> None:
        try:
            if self._line is None:
                return
            if getattr(event, 'artist', None) is not self._line:
                return
            ind = getattr(event, 'ind', None)
            if ind is None or len(ind) == 0:
                return
            idx = int(ind[0])
            self.point_clicked.emit(idx)
        except Exception:
            return

    def plot(self, df: Optional[pd.DataFrame], metric_col: str, *, selected_row: Optional[int] = None) -> None:
        self.fig.clear()
        ax = self.fig.add_subplot(111)

        if df is None or df.empty:
            ax.set_title('No batch data')
            self.canvas.draw()
            return

        if metric_col not in df.columns:
            ax.set_title(f"Missing column: {metric_col}")
            self.canvas.draw()
            return

        # x axis is ALWAYS the table row index
        x = np.arange(len(df), dtype=float)
        y = pd.to_numeric(df[metric_col], errors='coerce').to_numpy(dtype=float)

        # Use a pickable line so users can click points to select rows.
        (line,) = ax.plot(x, y, 'o-', picker=6)
        self._line = line

        # Highlight selection (if any and numeric)
        if selected_row is not None and 0 <= selected_row < len(df):
            ysel = y[selected_row]
            if np.isfinite(ysel):
                ax.plot([selected_row], [ysel], 'o', markersize=12, markerfacecolor='none', markeredgewidth=2)

        ax.set_xlabel('row #')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        # If the batch is small enough, show every integer tick.
        if len(df) <= 60:
            ax.set_xticks(np.arange(len(df)))

        ax.grid(True)
        ax.set_title(_metric_label(metric_col))
        self.fig.tight_layout()
        self.canvas.draw()


class BatchDistributionsPlot(QtWidgets.QWidget):
    """Histogram + boxplot view for one metric.

    This is meant for quick QA when running large batches:
    - overall distribution (hist)
    - per-component spread (boxplot), if a group column exists
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

    def plot(
        self,
        df: Optional[pd.DataFrame],
        metric_col: str,
        *,
        bins: int = 30,
        group_col: str = 'component',
    ) -> None:
        self.fig.clear()
        ax_hist = self.fig.add_subplot(211)
        ax_box = self.fig.add_subplot(212)

        if df is None or df.empty:
            ax_hist.set_title('No batch data')
            self.canvas.draw()
            return

        work = df.copy()
        if 'status' in work.columns:
            work = work[work['status'] == 'OK']
        if work.empty or metric_col not in work.columns:
            ax_hist.set_title('No numeric data')
            self.canvas.draw()
            return

        y = pd.to_numeric(work[metric_col], errors='coerce').dropna()
        if y.empty:
            ax_hist.set_title('No numeric data')
            self.canvas.draw()
            return

        bins = int(max(5, min(500, bins)))
        ax_hist.hist(y.to_numpy(dtype=float), bins=bins)
        mu = float(y.mean())
        med = float(y.median())
        ax_hist.axvline(mu, label='mean')
        ax_hist.axvline(med, label='median')
        ax_hist.set_title(_metric_label(metric_col))
        ax_hist.grid(True)
        ax_hist.legend(fontsize=8)

        # Boxplot: per component if possible, else single box
        if group_col in work.columns and work[group_col].nunique(dropna=False) > 1:
            groups = [str(g) for g in work[group_col].fillna('').unique().tolist()]
            # Stable ordering: alphabetical, but keep '(root)' up front
            groups = sorted(groups, key=lambda s: (s != '(root)', s.lower()))
            data = []
            labels = []
            for gname in groups:
                sub = work[work[group_col].fillna('') == gname]
                vals = pd.to_numeric(sub[metric_col], errors='coerce').dropna().to_numpy(dtype=float)
                if len(vals) == 0:
                    continue
                data.append(vals)
                labels.append(gname)

            if data:
                ax_box.boxplot(data, labels=labels, showfliers=True)
                ax_box.set_ylabel(_metric_label(metric_col))
                ax_box.grid(True, axis='y')
                for tick in ax_box.get_xticklabels():
                    tick.set_rotation(90)
                    tick.set_fontsize(8)
            else:
                ax_box.boxplot([y.to_numpy(dtype=float)], labels=['all'])
                ax_box.grid(True, axis='y')
        else:
            ax_box.boxplot([y.to_numpy(dtype=float)], labels=['all'])
            ax_box.set_ylabel(_metric_label(metric_col))
            ax_box.grid(True, axis='y')

        self.fig.tight_layout()
        self.canvas.draw()


class BatchCorrelationPlot(QtWidgets.QWidget):
    """Scatter plot between two metrics, optionally grouped."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.fig = Figure(figsize=(5, 3), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

    def plot(
        self,
        df: Optional[pd.DataFrame],
        x_col: str,
        y_col: str,
        *,
        group_col: str = 'component',
        color_by_group: bool = True,
    ) -> None:
        self.fig.clear()
        ax = self.fig.add_subplot(111)

        if df is None or df.empty:
            ax.set_title('No batch data')
            self.canvas.draw()
            return

        work = df.copy()
        if 'status' in work.columns:
            work = work[work['status'] == 'OK']
        if work.empty or x_col not in work.columns or y_col not in work.columns:
            ax.set_title('No numeric data')
            self.canvas.draw()
            return

        x = pd.to_numeric(work[x_col], errors='coerce')
        y = pd.to_numeric(work[y_col], errors='coerce')
        mask = np.isfinite(x.to_numpy(dtype=float)) & np.isfinite(y.to_numpy(dtype=float))
        work = work.loc[mask]
        if work.empty:
            ax.set_title('No numeric data')
            self.canvas.draw()
            return

        x = pd.to_numeric(work[x_col], errors='coerce').to_numpy(dtype=float)
        y = pd.to_numeric(work[y_col], errors='coerce').to_numpy(dtype=float)

        # Correlation coefficient (useful, fast, and occasionally sobering)
        r = np.corrcoef(x, y)[0, 1] if len(x) > 1 else np.nan

        if color_by_group and group_col in work.columns and work[group_col].nunique() <= 12:
            # Legend only if number of groups is sane
            for gname, sub in work.groupby(group_col):
                xs = pd.to_numeric(sub[x_col], errors='coerce').to_numpy(dtype=float)
                ys = pd.to_numeric(sub[y_col], errors='coerce').to_numpy(dtype=float)
                ax.plot(xs, ys, 'o', label=str(gname))
            ax.legend(fontsize=8, loc='best')
        else:
            ax.plot(x, y, 'o')

        ax.set_xlabel(_metric_label(x_col))
        ax.set_ylabel(_metric_label(y_col))
        ax.grid(True)
        title = f"{_metric_label(y_col)} vs {_metric_label(x_col)}"
        if np.isfinite(r):
            title += f"   (r={r:.3g})"
        ax.set_title(title)
        self.fig.tight_layout()
        self.canvas.draw()


class CorrelationMatrixPlot(QtWidgets.QWidget):
    """Correlation matrix heatmap for the standard metric set."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

    def plot(self, df: Optional[pd.DataFrame], cols: Optional[Sequence[str]] = None) -> None:
        self.fig.clear()
        ax = self.fig.add_subplot(111)

        if df is None or df.empty:
            ax.set_title('No batch data')
            self.canvas.draw()
            return

        work = df.copy()
        if 'status' in work.columns:
            work = work[work['status'] == 'OK']
        if work.empty:
            ax.set_title('No numeric data')
            self.canvas.draw()
            return

        if cols is None:
            cols = [c for (_label, c) in METRICS if c in work.columns]
        cols = [c for c in cols if c in work.columns]
        if len(cols) < 2:
            ax.set_title('Not enough numeric columns')
            self.canvas.draw()
            return

        mat = work[cols].apply(pd.to_numeric, errors='coerce')
        corr = mat.corr(method='pearson')

        im = ax.imshow(corr.to_numpy(dtype=float), vmin=-1, vmax=1, cmap='coolwarm')
        self.fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        labels = [_metric_label(c) for c in cols]
        ax.set_xticks(np.arange(len(cols)))
        ax.set_yticks(np.arange(len(cols)))
        ax.set_xticklabels(labels, rotation=90, fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_title('Correlation matrix (Pearson)')

        self.fig.tight_layout()
        self.canvas.draw()


class ComponentScatterPlot(QtWidgets.QWidget):
    """Per-component scatter plots for one or more metrics.

    X axis: component name (categorical)
    Y axis: chosen metric

    Multiple checked metrics -> stacked subplots.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.fig = Figure(figsize=(6, 4), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

    def plot(
        self,
        df: Optional[pd.DataFrame],
        metric_cols: Sequence[str],
        *,
        group_col: str = 'component',
        show_mean_std: bool = True,
        jitter: float = 0.12,
    ) -> None:
        self.fig.clear()

        if df is None or df.empty:
            ax = self.fig.add_subplot(111)
            ax.set_title('No batch data')
            self.canvas.draw()
            return

        work = df.copy()
        if 'status' in work.columns:
            work = work[work['status'] == 'OK']
        if work.empty or group_col not in work.columns:
            ax = self.fig.add_subplot(111)
            ax.set_title('No grouped data')
            self.canvas.draw()
            return

        metric_cols = [c for c in metric_cols if c in work.columns]
        if not metric_cols:
            ax = self.fig.add_subplot(111)
            ax.set_title('Pick one or more metrics')
            self.canvas.draw()
            return

        comps = work[group_col].fillna('').astype(str).unique().tolist()
        # Stable ordering
        comps = sorted(comps, key=lambda s: (s != '(root)', s.lower()))
        comp_to_x = {c: i for i, c in enumerate(comps)}

        rng = np.random.default_rng(0)
        n = len(metric_cols)
        axes = self.fig.subplots(nrows=n, ncols=1, sharex=True) if n > 1 else [self.fig.add_subplot(111)]

        for ax, col in zip(axes, metric_cols):
            for comp, sub in work.groupby(group_col):
                comp = '' if comp is None else str(comp)
                if comp not in comp_to_x:
                    continue
                vals = pd.to_numeric(sub[col], errors='coerce').dropna().to_numpy(dtype=float)
                if len(vals) == 0:
                    continue
                x0 = comp_to_x[comp]
                xs = x0 + rng.uniform(-jitter, jitter, size=len(vals))
                ax.plot(xs, vals, 'o', markersize=4, alpha=0.85)

                if show_mean_std:
                    mu = float(np.mean(vals))
                    ax.plot([x0], [mu], 's', markersize=7, markerfacecolor='none', markeredgewidth=2)
                    if len(vals) > 1:
                        sd = float(np.std(vals, ddof=1))
                        ax.vlines([x0], [mu - sd], [mu + sd], linewidth=2)

            ax.set_ylabel(_metric_label(col))
            ax.grid(True, axis='y')

        axes[-1].set_xticks(np.arange(len(comps)))
        axes[-1].set_xticklabels(comps, rotation=90, fontsize=8)
        axes[-1].set_xlabel('component')

        self.fig.tight_layout()
        self.canvas.draw()


class SingleTab(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.meas = None
        self.widths = None
        self.results = None

        # Background compute state (Single recompute)
        self._thread: Optional[QtCore.QThread] = None
        self._worker: Optional['SingleWorker'] = None
        self._pending_recompute: bool = False
        self._pending_settings: Optional[Tuple[WidthMethod, AxisMode, Optional[float]]] = None

        # Controls
        self.path_label = QtWidgets.QLabel('No .m2 loaded')
        self.method_combo = QtWidgets.QComboBox()
        self.method_combo.addItem('M2 file moments (fast)', WidthMethod.M2_FILE_MOMENTS)
        self.method_combo.addItem('Image 2nd moments (2σ)', WidthMethod.IMAGE_2ND_MOMENTS)
        self.method_combo.addItem('Image Gaussian fit (1/e²)', WidthMethod.IMAGE_GAUSS_FIT)

        self.axis_combo = QtWidgets.QComboBox()
        self.axis_combo.addItem('Camera X/Y', AxisMode.CAMERA_XY)
        self.axis_combo.addItem('Principal axes (major/minor)', AxisMode.PRINCIPAL_AXES)

        self.drop_edit = QtWidgets.QLineEdit('')
        self.drop_edit.setPlaceholderText('drop_down (e.g. 0.1353)')
        self.recompute_btn = QtWidgets.QPushButton('Recompute')

        ctrl = QtWidgets.QHBoxLayout()
        ctrl.addWidget(self.path_label, 1)
        ctrl.addWidget(QtWidgets.QLabel('Method:'))
        ctrl.addWidget(self.method_combo)
        ctrl.addWidget(QtWidgets.QLabel('Axes:'))
        ctrl.addWidget(self.axis_combo)
        ctrl.addWidget(QtWidgets.QLabel('Drop:'))
        ctrl.addWidget(self.drop_edit)
        ctrl.addWidget(self.recompute_btn)

        # Progress / cancellation for heavy (image-based) computations
        self.progress = QtWidgets.QProgressBar()
        self.progress.setVisible(False)
        self.progress.setTextVisible(True)
        self.cancel_btn = QtWidgets.QPushButton('Cancel')
        self.cancel_btn.setEnabled(False)

        # Table and plots
        self.table = QtWidgets.QTableView()
        self.table_model = DataFrameModel()
        self.table.setModel(self.table_model)
        self.table.setSelectionBehavior(SelectRows)
        self.table.setSelectionMode(SingleSelection)
        self.table.setSortingEnabled(True)

        self.results_box = QtWidgets.QTextEdit()
        self.results_box.setReadOnly(True)
        self.results_box.setMinimumHeight(120)

        self.image_view = ImageView()
        # Image display controls (preview + false color)
        self.show_images_chk = QtWidgets.QCheckBox('Show images')
        self.show_images_chk.setChecked(True)

        self.cmap_combo = QtWidgets.QComboBox()
        # label, matplotlib name
        cmaps = [
            ('Viridis', 'viridis'),
            ('Grayscale', 'gray'),
            ('Inferno', 'inferno'),
            ('Magma', 'magma'),
            ('Plasma', 'plasma'),
            ('Turbo', 'turbo'),
        ]
        for label, name in cmaps:
            self.cmap_combo.addItem(label, name)
        self.cmap_combo.setCurrentIndex(0)

        self.gamma_spin = QtWidgets.QDoubleSpinBox()
        self.gamma_spin.setRange(0.2, 5.0)
        self.gamma_spin.setSingleStep(0.1)
        self.gamma_spin.setValue(1.0)
        self.gamma_spin.setToolTip('Gamma for display only (not analysis)')

        self.preview_chk = QtWidgets.QCheckBox('Downsample preview')
        self.preview_chk.setChecked(True)
        self.preview_spin = QtWidgets.QSpinBox()
        self.preview_spin.setRange(128, 8192)
        self.preview_spin.setValue(1024)
        self.preview_spin.setToolTip('Maximum preview dimension (pixels)')

        img_ctrl = QtWidgets.QHBoxLayout()
        img_ctrl.addWidget(self.show_images_chk)
        img_ctrl.addWidget(QtWidgets.QLabel('Colormap:'))
        img_ctrl.addWidget(self.cmap_combo)
        img_ctrl.addWidget(QtWidgets.QLabel('Gamma:'))
        img_ctrl.addWidget(self.gamma_spin)
        img_ctrl.addWidget(self.preview_chk)
        img_ctrl.addWidget(QtWidgets.QLabel('Max:'))
        img_ctrl.addWidget(self.preview_spin)
        img_ctrl.addStretch(1)
        self.plot = FitPlot()

        # Layout
        left = QtWidgets.QVBoxLayout()
        left.addWidget(self.table, 1)
        left.addWidget(QtWidgets.QLabel('Results'))
        left.addWidget(self.results_box)

        right = QtWidgets.QVBoxLayout()
        right.addLayout(img_ctrl)
        right.addWidget(self.image_view, 2)
        right.addWidget(self.plot, 1)

        splitter = QtWidgets.QSplitter()
        leftw = QtWidgets.QWidget(); leftw.setLayout(left)
        rightw = QtWidgets.QWidget(); rightw.setLayout(right)
        splitter.addWidget(leftw)
        splitter.addWidget(rightw)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)

        root = QtWidgets.QVBoxLayout(self)
        root.addLayout(ctrl)
        prog_row = QtWidgets.QHBoxLayout()
        prog_row.addWidget(self.progress, 1)
        prog_row.addWidget(self.cancel_btn)
        root.addLayout(prog_row)
        root.addWidget(splitter, 1)

        # Signals
        self.recompute_btn.clicked.connect(self.recompute)
        self.cancel_btn.clicked.connect(self.cancel_compute)
        self.method_combo.currentIndexChanged.connect(self.recompute)
        self.axis_combo.currentIndexChanged.connect(self.recompute)
        self.cancel_btn.clicked.connect(self.cancel_compute)
        self.table.selectionModel().selectionChanged.connect(self._on_row_selected)

        # Image display signals (display only, no recompute)
        self.show_images_chk.stateChanged.connect(self._refresh_current_image)
        self.cmap_combo.currentIndexChanged.connect(self._refresh_current_image)
        self.gamma_spin.valueChanged.connect(self._refresh_current_image)
        self.preview_chk.stateChanged.connect(self._refresh_current_image)
        self.preview_spin.valueChanged.connect(self._refresh_current_image)

        # Thread state
        self._thread: Optional[QtCore.QThread] = None
        self._worker: Optional['SingleWorker'] = None
        self._pending_recompute: bool = False

    def load_m2(self, path: Union[str, Path]) -> None:
        self.meas = parse_m2_file(path)
        self.path_label.setText(str(self.meas.m2_path))
        self.recompute()


    def save_current_overlay_png(self, out_path: Union[str, Path]) -> None:
        'Save the currently selected frame image with overlay as PNG.'
        if self.meas is None or self.widths is None:
            return

        # Pick row
        sel = self.table.selectionModel().selectedRows()
        if sel:
            row = sel[0].row()
        else:
            row = 0

        if row < 0 or row >= len(self.widths):
            return

        w = self.widths[row]
        fr = next((f for f in self.meas.active_frames() if f.index == w.index), None)
        if fr is None:
            return

        img_path = self.meas.resolve_image_path(fr)
        if not img_path.exists():
            return

        try:
            img = read_tiff(img_path)
            cmap = self.cmap_combo.currentData() if hasattr(self, 'cmap_combo') else 'viridis'
            gamma = float(self.gamma_spin.value()) if hasattr(self, 'gamma_spin') else 1.0
            self.image_view.set_image(img, colormap=str(cmap), gamma=gamma)

            axis_mode = self.axis_combo.currentData()
            if axis_mode == AxisMode.PRINCIPAL_AXES:
                wx_mm, wy_mm, ang = w.w_major, w.w_minor, w.angle_deg
            else:
                wx_mm, wy_mm, ang = w.w_x, w.w_y, 0.0

            wx_px = wx_mm / fr.facX if fr.facX else 0
            wy_px = wy_mm / fr.facY if fr.facY else 0
            self.image_view.set_overlay(w.cx, w.cy, wx_px, wy_px, ang)
            self.image_view.save_png(out_path)
        except Exception:
            pass

    def _get_drop(self) -> Optional[float]:
        s = self.drop_edit.text().strip()
        if not s:
            return None
        try:
            return float(s)
        except Exception:
            return None

    def recompute(self) -> None:
        """Recompute widths + M².

        Runs in a worker thread so the GUI doesn't freeze when you pick an
        image-based method (TIFFs are not small, because why would they be).
        """
        if self.meas is None:
            return

        method = self.method_combo.currentData()
        axis_mode = self.axis_combo.currentData()
        drop = self._get_drop()

        # If a compute is already running, cancel and queue a fresh run.
        if self._thread is not None and self._worker is not None:
            self._pending_recompute = True
            self._pending_settings = (method, axis_mode, drop)
            try:
                self._worker.cancel()
            except Exception:
                pass
            return

        # UI state
        self.recompute_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)  # indeterminate until we know n
        self.progress.setValue(0)

        # Worker thread
        self._thread = QtCore.QThread(self)
        self._worker = SingleWorker(
            meas=self.meas,
            method=method,
            axis_mode=axis_mode,
            drop=drop,
        )
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_compute_progress)
        self._worker.finished.connect(self._on_compute_finished)
        self._worker.finished.connect(self._thread.quit)
        self._worker.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)

        self._thread.start()

    def cancel_compute(self) -> None:
        if self._worker is not None:
            try:
                self._worker.cancel()
            except Exception:
                pass

    def _on_compute_progress(self, cur: int, total: int, msg: str) -> None:
        try:
            self.progress.setRange(0, int(total) if total else 0)
            self.progress.setValue(int(cur))
            if msg:
                self.progress.setFormat(msg)
        except Exception:
            pass

    def _on_compute_finished(self, payload: dict) -> None:
        # Tear down thread refs first
        self._thread = None
        self._worker = None

        # UI state reset
        self.recompute_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.progress.setVisible(False)

        if payload is None:
            return

        if payload.get('cancelled'):
            self.results_box.setPlainText('Cancelled.')
        elif payload.get('error'):
            self.results_box.setPlainText(payload.get('error') or 'Error')
        else:
            widths = payload.get('widths')
            res = payload.get('results')
            if widths is None or res is None:
                self.results_box.setPlainText('No results')
                return

            self.widths = widths
            self.results = res

            wdf = widths_to_dataframe(widths)
            self.table_model.set_dataframe(wdf)
            try:
                self.table.resizeColumnsToContents()
            except Exception:
                pass

            self._update_results_text(res)
            self.plot.plot(wdf, res, axis_mode=res.fit_x.axis_mode)

            if len(wdf) > 0:
                self.table.selectRow(0)

        # If the user changed settings mid-run, restart now.
        if self._pending_recompute and self._pending_settings is not None:
            method, axis_mode, drop = self._pending_settings
            self._pending_recompute = False
            self._pending_settings = None

            try:
                # Reflect the queued settings in the UI so the user isn't gaslit.
                idx = self.method_combo.findData(method)
                if idx >= 0:
                    self.method_combo.setCurrentIndex(idx)
                idx2 = self.axis_combo.findData(axis_mode)
                if idx2 >= 0:
                    self.axis_combo.setCurrentIndex(idx2)
                self.drop_edit.setText('' if drop is None else str(drop))
            except Exception:
                pass

            # Kick off the queued recompute.
            self.recompute()

    def _update_results_text(self, res) -> None:
        self.results_box.setPlainText(_format_results_text(res))

    def _refresh_current_image(self, *args) -> None:
        """Re-render the currently selected frame with new display settings."""
        self._show_selected_frame()

    def _show_selected_frame(self) -> None:
        """Load and display the currently selected frame image (preview).

        This is UI-only: it does not affect any computed widths/results.
        """
        if self.meas is None or self.widths is None:
            return

        # If the user disabled images, clear view and exit.
        try:
            if not bool(self.show_images_chk.isChecked()):
                try:
                    self.image_view.scene().clear()
                except Exception:
                    pass
                return
        except Exception:
            pass

        sel = self.table.selectionModel().selectedRows() if self.table.selectionModel() else []
        if not sel:
            return
        row = sel[0].row()
        if row < 0 or row >= len(self.widths):
            return

        w = self.widths[row]
        fr = next((f for f in self.meas.active_frames() if f.index == w.index), None)
        if fr is None:
            return

        img_path = self.meas.resolve_image_path(fr)
        if not img_path.exists():
            return

        cmap = self.cmap_combo.currentData() if hasattr(self, 'cmap_combo') else 'viridis'
        gamma = float(self.gamma_spin.value()) if hasattr(self, 'gamma_spin') else 1.0
        use_preview = bool(self.preview_chk.isChecked()) if hasattr(self, 'preview_chk') else True
        max_dim = int(self.preview_spin.value()) if hasattr(self, 'preview_spin') else 1024

        try:
            if use_preview:
                img, sx, sy = read_tiff_preview(img_path, max_dim=max_dim, return_scale=True)
            else:
                img = read_tiff(img_path)
                sx, sy = 1.0, 1.0

            # Display image with colormap
            self.image_view.set_image(img, colormap=str(cmap), gamma=gamma)

            axis_mode = self.axis_combo.currentData()
            if axis_mode == AxisMode.PRINCIPAL_AXES:
                wx_mm = w.w_major
                wy_mm = w.w_minor
                ang = w.angle_deg
            else:
                wx_mm = w.w_x
                wy_mm = w.w_y
                ang = 0.0

            # Overlay expects pixels, convert radii mm -> px in *original* image pixel units
            wx_px = wx_mm / fr.facX if fr.facX else 0.0
            wy_px = wy_mm / fr.facY if fr.facY else 0.0

            # If the preview is downsampled, scale overlay coordinates accordingly.
            cx_d = float(w.cx) * float(sx)
            cy_d = float(w.cy) * float(sy)
            wx_d = float(wx_px) * float(sx)
            wy_d = float(wy_px) * float(sy)

            self.image_view.set_overlay(cx_d, cy_d, wx_d, wy_d, ang)
        except Exception:
            # Visualization should never crash the app
            return

    def _on_row_selected(self, *args) -> None:
        self._show_selected_frame()


class _CancelledError(RuntimeError):
    pass


class SingleWorker(QtCore.QObject):
    """Worker for single-file recompute.

    This mirrors the batch worker style, but focuses on a single measurement.
    """

    progress = qsignal(int, int, str)  # current, total, message
    finished = qsignal(object)  # dict payload

    def __init__(self, *, meas, method: WidthMethod, axis_mode: AxisMode, drop: Optional[float]):
        super().__init__()
        self.meas = meas
        self.method = method
        self.axis_mode = axis_mode
        self.drop = drop
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def _check_cancel(self) -> None:
        if self._cancel:
            raise _CancelledError('Cancelled')

    def run(self) -> None:
        payload: Dict[str, object] = {}
        try:
            frames = list(self.meas.active_frames())
            total = len(frames)
            widths: List[FrameWidths] = []

            for i, fr in enumerate(frames, start=1):
                self._check_cancel()

                if self.method == WidthMethod.M2_FILE_MOMENTS:
                    fw = widths_from_m2_frame(fr)
                else:
                    img_path = self.meas.resolve_image_path(fr)
                    img = read_tiff(img_path)
                    if self.method == WidthMethod.IMAGE_2ND_MOMENTS:
                        fw = widths_from_image_moments(img, fr.facX, fr.facY, drop_down=self.drop)
                    elif self.method == WidthMethod.IMAGE_GAUSS_FIT:
                        fw = widths_from_image_gauss_fit(img, fr.facX, fr.facY, drop_down=self.drop)
                    else:
                        raise ValueError(f"Unsupported method: {self.method}")

                fw.index = fr.index
                fw.z = fr.z
                fw.snr = fr.snr
                widths.append(fw)

                try:
                    self.progress.emit(i, total, f"Frame {i}/{total}")
                except Exception:
                    pass

            widths.sort(key=lambda r: r.z)
            self._check_cancel()

            res = compute_m2_results(self.meas, widths, method=self.method, axis_mode=self.axis_mode)
            payload = {'widths': widths, 'results': res, 'cancelled': False}
        except _CancelledError:
            payload = {'cancelled': True}
        except Exception as e:
            payload = {'error': f"Error while computing: {e}\n\n{traceback.format_exc()}"}

        try:
            self.finished.emit(payload)
        except Exception:
            pass


class BatchWorker(QtCore.QObject):
    """Background worker for batch processing.

    This keeps the GUI responsive while chewing through large folder trees.
    Humans love clicking things while computations run. It's adorable.
    """

    progress = qsignal(int, int, str)  # current, total, message
    finished = qsignal(object)  # dict with results

    def __init__(
        self,
        *,
        root: Path,
        files: List[Path],
        method: WidthMethod,
        axis_mode: AxisMode,
        drop: Optional[float],
        group_mode: str,
    ):
        super().__init__()
        self.root = root
        self.files = files
        self.method = method
        self.axis_mode = axis_mode
        self.drop = drop
        self.group_mode = group_mode
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:
        rows = []
        entries: List[BatchEntry] = []
        log_lines: List[str] = []

        total = len(self.files)
        for i, f in enumerate(self.files, start=1):
            if self._cancel:
                log_lines.append('Cancelled by user.')
                break

            rel = str(f.relative_to(self.root))
            component = _component_from_rel(rel, self.group_mode)

            try:
                meas = parse_m2_file(f)
                widths = compute_frame_widths(meas, method=self.method, drop_down=self.drop)
                res = compute_m2_results(meas, widths, method=self.method, axis_mode=self.axis_mode)

                df = results_to_dataframe(res)
                df.insert(0, 'file', rel)
                df.insert(1, 'component', component)
                df.insert(2, 'status', 'OK')
                df.insert(3, 'error', '')
                rows.append(df)

                entries.append(BatchEntry(path=f, rel_path=rel, results=res, widths=widths, error=None))
                log_lines.append(f"OK: {rel}")
                msg = f"OK: {rel}"
            except Exception as e:
                rows.append(pd.DataFrame([{'file': rel, 'component': component, 'status': 'ERR', 'error': str(e)}]))
                entries.append(BatchEntry(path=f, rel_path=rel, results=None, widths=None, error=str(e)))
                log_lines.append(f"ERR: {rel}: {e}")
                msg = f"ERR: {rel}"

            try:
                self.progress.emit(i, total, msg)
            except Exception:
                pass

        out_df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
        stats_df = _compute_batch_stats(out_df) if not out_df.empty else pd.DataFrame()
        comp_stats_df = _compute_component_stats(out_df, group_col='component') if not out_df.empty else pd.DataFrame()

        payload = {
            'entries': entries,
            'df': out_df,
            'stats_df': stats_df,
            'component_stats_df': comp_stats_df,
            'log': '\n'.join(log_lines),
            'cancelled': bool(self._cancel),
        }
        self.finished.emit(payload)


class BatchTab(QtWidgets.QWidget):
    # Emitted when the user wants to open a batch entry in the Single tab.
    open_single_requested = qsignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)

        # -----------------
        # Controls (top bar)
        # -----------------
        self.folder_label = QtWidgets.QLabel('No folder selected')
        self.pick_btn = QtWidgets.QPushButton('Select Folder')
        self.run_btn = QtWidgets.QPushButton('Run Batch')
        self.cancel_btn = QtWidgets.QPushButton('Cancel')
        self.cancel_btn.setEnabled(False)

        self.export_csv_btn = QtWidgets.QPushButton('Export CSV')
        self.export_xlsx_btn = QtWidgets.QPushButton('Export Excel')

        self.progress = QtWidgets.QProgressBar()
        self.progress.setVisible(False)
        self.progress.setTextVisible(True)

        # Analysis options
        self.method_combo = QtWidgets.QComboBox()
        self.method_combo.addItem('M2 file moments (fast)', WidthMethod.M2_FILE_MOMENTS)
        self.method_combo.addItem('Image 2nd moments (2σ)', WidthMethod.IMAGE_2ND_MOMENTS)
        self.method_combo.addItem('Image Gaussian fit (1/e²)', WidthMethod.IMAGE_GAUSS_FIT)

        self.axis_combo = QtWidgets.QComboBox()
        self.axis_combo.addItem('Camera X/Y', AxisMode.CAMERA_XY)
        self.axis_combo.addItem('Principal axes (major/minor)', AxisMode.PRINCIPAL_AXES)

        self.drop_edit = QtWidgets.QLineEdit('')
        self.drop_edit.setPlaceholderText('drop_down (e.g. 0.1353)')

        self.group_combo = QtWidgets.QComboBox()
        for label, mode in GROUP_MODES:
            self.group_combo.addItem(label, mode)
        # Default: first child folder under root
        self.group_combo.setCurrentIndex(0)

        top1 = QtWidgets.QHBoxLayout()
        top1.addWidget(self.folder_label, 1)
        top1.addWidget(self.pick_btn)
        top1.addWidget(self.run_btn)
        top1.addWidget(self.cancel_btn)
        top1.addWidget(self.export_csv_btn)
        top1.addWidget(self.export_xlsx_btn)

        top2 = QtWidgets.QHBoxLayout()
        top2.addWidget(QtWidgets.QLabel('Method:'))
        top2.addWidget(self.method_combo)
        top2.addWidget(QtWidgets.QLabel('Axes:'))
        top2.addWidget(self.axis_combo)
        top2.addWidget(QtWidgets.QLabel('Drop:'))
        top2.addWidget(self.drop_edit)
        top2.addWidget(QtWidgets.QLabel('Group:'))
        top2.addWidget(self.group_combo)
        top2.addStretch(1)

        # -----------------
        # Table (left panel)
        # -----------------
        self.table = QtWidgets.QTableView()
        self.table_model = DataFrameModel()
        self.table.setModel(self.table_model)
        self.table.setSelectionBehavior(SelectRows)
        self.table.setSelectionMode(SingleSelection)
        self.table.setSortingEnabled(True)

        # -----------------
        # Right panel tabs
        # -----------------

        # Summary tab
        self.summary_metric_combo = QtWidgets.QComboBox()
        for label, col in METRICS:
            self.summary_metric_combo.addItem(label, col)
        self.summary_plot = BatchSummaryPlot()
        self.stats_box = QtWidgets.QTextEdit()
        self.stats_box.setReadOnly(True)
        self.stats_box.setMinimumHeight(120)

        metric_row = QtWidgets.QHBoxLayout()
        metric_row.addWidget(QtWidgets.QLabel('Metric:'))
        metric_row.addWidget(self.summary_metric_combo, 1)

        summary_w = QtWidgets.QWidget()
        summary_l = QtWidgets.QVBoxLayout(summary_w)
        summary_l.setContentsMargins(0, 0, 0, 0)
        summary_l.addLayout(metric_row)
        summary_l.addWidget(self.summary_plot, 1)
        summary_l.addWidget(QtWidgets.QLabel('Statistics'))
        summary_l.addWidget(self.stats_box)

        # Distributions tab
        self.dist_metric_combo = QtWidgets.QComboBox()
        for label, col in METRICS:
            self.dist_metric_combo.addItem(label, col)
        self.bins_spin = QtWidgets.QSpinBox()
        self.bins_spin.setRange(5, 500)
        self.bins_spin.setValue(30)

        dist_row = QtWidgets.QHBoxLayout()
        dist_row.addWidget(QtWidgets.QLabel('Metric:'))
        dist_row.addWidget(self.dist_metric_combo, 1)
        dist_row.addWidget(QtWidgets.QLabel('Bins:'))
        dist_row.addWidget(self.bins_spin)

        self.dist_plot = BatchDistributionsPlot()

        dist_w = QtWidgets.QWidget()
        dist_l = QtWidgets.QVBoxLayout(dist_w)
        dist_l.setContentsMargins(0, 0, 0, 0)
        dist_l.addLayout(dist_row)
        dist_l.addWidget(self.dist_plot, 1)

        # Correlation tab
        self.corr_x_combo = QtWidgets.QComboBox()
        self.corr_y_combo = QtWidgets.QComboBox()
        for label, col in METRICS:
            self.corr_x_combo.addItem(label, col)
            self.corr_y_combo.addItem(label, col)
        # Reasonable defaults
        self.corr_x_combo.setCurrentIndex(0)
        self.corr_y_combo.setCurrentIndex(3 if len(METRICS) > 3 else 0)

        self.corr_color_chk = QtWidgets.QCheckBox('Color by component (<=12)')
        self.corr_color_chk.setChecked(True)

        corr_row = QtWidgets.QHBoxLayout()
        corr_row.addWidget(QtWidgets.QLabel('X:'))
        corr_row.addWidget(self.corr_x_combo)
        corr_row.addWidget(QtWidgets.QLabel('Y:'))
        corr_row.addWidget(self.corr_y_combo)
        corr_row.addWidget(self.corr_color_chk)
        corr_row.addStretch(1)

        self.corr_plot = BatchCorrelationPlot()
        corr_scatter_w = QtWidgets.QWidget()
        corr_scatter_l = QtWidgets.QVBoxLayout(corr_scatter_w)
        corr_scatter_l.setContentsMargins(0, 0, 0, 0)
        corr_scatter_l.addLayout(corr_row)
        corr_scatter_l.addWidget(self.corr_plot, 1)

        self.corr_matrix_plot = CorrelationMatrixPlot()

        corr_tabs = QtWidgets.QTabWidget()
        corr_tabs.addTab(corr_scatter_w, 'Scatter')
        corr_tabs.addTab(self.corr_matrix_plot, 'Matrix')

        corr_w = QtWidgets.QWidget()
        corr_l = QtWidgets.QVBoxLayout(corr_w)
        corr_l.setContentsMargins(0, 0, 0, 0)
        corr_l.addWidget(corr_tabs, 1)

        # Components tab
        self.comp_metrics_list = QtWidgets.QListWidget()
        self.comp_metrics_list.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection if hasattr(QtWidgets.QAbstractItemView, 'SelectionMode') else QtWidgets.QAbstractItemView.NoSelection)
        for label, col in METRICS:
            it = QtWidgets.QListWidgetItem(label)
            it.setData(Qt.ItemDataRole.UserRole if hasattr(Qt, 'ItemDataRole') else Qt.UserRole, col)
            it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable if hasattr(Qt, 'ItemFlag') else it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.CheckState.Unchecked if hasattr(Qt, 'CheckState') else Qt.Unchecked)
            self.comp_metrics_list.addItem(it)

        # Default checks: M² geo-mean and BPP geo-mean
        for i in range(self.comp_metrics_list.count()):
            it = self.comp_metrics_list.item(i)
            col = it.data(Qt.ItemDataRole.UserRole if hasattr(Qt, 'ItemDataRole') else Qt.UserRole)
            if col in ('m2_geo_mean', 'bpp_geo_mean_mm_mrad'):
                it.setCheckState(Qt.CheckState.Checked if hasattr(Qt, 'CheckState') else Qt.Checked)

        self.comp_mean_chk = QtWidgets.QCheckBox('Show mean ± std')
        self.comp_mean_chk.setChecked(True)
        self.comp_jitter_spin = QtWidgets.QDoubleSpinBox()
        self.comp_jitter_spin.setRange(0.0, 0.5)
        self.comp_jitter_spin.setSingleStep(0.02)
        self.comp_jitter_spin.setValue(0.12)

        comp_opts = QtWidgets.QHBoxLayout()
        comp_opts.addWidget(self.comp_mean_chk)
        comp_opts.addWidget(QtWidgets.QLabel('Jitter:'))
        comp_opts.addWidget(self.comp_jitter_spin)
        comp_opts.addStretch(1)

        self.comp_plot = ComponentScatterPlot()
        self.comp_stats_table = QtWidgets.QTableView()
        self.comp_stats_model = DataFrameModel()
        self.comp_stats_table.setModel(self.comp_stats_model)
        self.comp_stats_table.setSortingEnabled(True)

        comp_split = QtWidgets.QSplitter()
        comp_left = QtWidgets.QWidget()
        comp_left_l = QtWidgets.QVBoxLayout(comp_left)
        comp_left_l.setContentsMargins(0, 0, 0, 0)
        comp_left_l.addWidget(QtWidgets.QLabel('Metrics (check to plot)'))
        comp_left_l.addWidget(self.comp_metrics_list, 1)
        comp_left_l.addLayout(comp_opts)
        comp_split.addWidget(comp_left)
        comp_split.addWidget(self.comp_plot)
        comp_split.setStretchFactor(0, 1)
        comp_split.setStretchFactor(1, 3)

        comp_w = QtWidgets.QWidget()
        comp_l = QtWidgets.QVBoxLayout(comp_w)
        comp_l.setContentsMargins(0, 0, 0, 0)
        comp_l.addWidget(comp_split, 2)
        comp_l.addWidget(QtWidgets.QLabel('Per-component summary (n, mean/std)'))
        comp_l.addWidget(self.comp_stats_table, 1)

        # Selected file tab
        self.selected_label = QtWidgets.QLabel('No file selected')
        self.file_results_box = QtWidgets.QTextEdit()
        self.file_results_box.setReadOnly(True)
        self.file_results_box.setMinimumHeight(120)
        self.fit_plot = FitPlot()

        selected_w = QtWidgets.QWidget()
        selected_l = QtWidgets.QVBoxLayout(selected_w)
        selected_l.setContentsMargins(0, 0, 0, 0)
        selected_l.addWidget(self.selected_label)
        selected_l.addWidget(self.file_results_box)
        selected_l.addWidget(self.fit_plot, 1)

        # Combine right-side tabs
        self.right_tabs = QtWidgets.QTabWidget()
        self.right_tabs.addTab(summary_w, 'Summary')
        self.right_tabs.addTab(dist_w, 'Distributions')
        self.right_tabs.addTab(corr_w, 'Correlation')
        self.right_tabs.addTab(comp_w, 'Components')
        self.right_tabs.addTab(selected_w, 'Selected file')

        splitter = QtWidgets.QSplitter()
        leftw = QtWidgets.QWidget()
        leftl = QtWidgets.QVBoxLayout(leftw)
        leftl.setContentsMargins(0, 0, 0, 0)
        leftl.addWidget(self.table)
        splitter.addWidget(leftw)
        splitter.addWidget(self.right_tabs)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)

        # Log
        self.log = QtWidgets.QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMinimumHeight(120)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(top1)
        layout.addLayout(top2)
        layout.addWidget(self.progress)
        layout.addWidget(splitter, 1)
        layout.addWidget(QtWidgets.QLabel('Log'))
        layout.addWidget(self.log)

        # -----------------
        # State
        # -----------------
        self._folder: Optional[Path] = None
        self._last_df: Optional[pd.DataFrame] = None
        self._stats_df: Optional[pd.DataFrame] = None
        self._component_stats_df: Optional[pd.DataFrame] = None
        self._entries: List[BatchEntry] = []
        self._entry_by_file: Dict[str, BatchEntry] = {}

        self._thread: Optional[QtCore.QThread] = None
        self._worker: Optional[BatchWorker] = None

        # -----------------
        # Signals
        # -----------------
        self.pick_btn.clicked.connect(self.pick_folder)
        self.run_btn.clicked.connect(self.run_batch)
        self.cancel_btn.clicked.connect(self.cancel_batch)
        self.export_csv_btn.clicked.connect(self.export_csv)
        self.export_xlsx_btn.clicked.connect(self.export_xlsx)

        self.summary_metric_combo.currentIndexChanged.connect(self._refresh_summary_plot)
        self.summary_plot.point_clicked.connect(self._on_summary_point_clicked)
        self.dist_metric_combo.currentIndexChanged.connect(self._refresh_distributions_plot)
        self.bins_spin.valueChanged.connect(self._refresh_distributions_plot)
        self.corr_x_combo.currentIndexChanged.connect(self._refresh_corr_scatter)
        self.corr_y_combo.currentIndexChanged.connect(self._refresh_corr_scatter)
        self.corr_color_chk.stateChanged.connect(self._refresh_corr_scatter)
        self.comp_metrics_list.itemChanged.connect(self._refresh_component_plot)
        self.comp_mean_chk.stateChanged.connect(self._refresh_component_plot)
        self.comp_jitter_spin.valueChanged.connect(self._refresh_component_plot)
        self.group_combo.currentIndexChanged.connect(self._on_group_mode_changed)

        self.table.selectionModel().selectionChanged.connect(self._on_row_selected)
        self.table.doubleClicked.connect(self._on_row_double_clicked)

    # ----------
    # Utilities
    # ----------
    def _get_drop(self) -> Optional[float]:
        s = self.drop_edit.text().strip()
        if not s:
            return None
        try:
            return float(s)
        except Exception:
            return None

    def pick_folder(self) -> None:
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, 'Select root folder (recursive scan for *.m2)')
        if not folder:
            return
        self._folder = Path(folder).resolve()
        self.folder_label.setText(str(self._folder))

    # ----------------
    # Batch execution
    # ----------------
    def run_batch(self) -> None:
        if self._folder is None:
            return
        if self._thread is not None:
            return  # already running

        method = self.method_combo.currentData()
        axis_mode = self.axis_combo.currentData()
        drop = self._get_drop()
        group_mode = self.group_combo.currentData()

        files = sorted([p for p in self._folder.rglob('*') if p.is_file() and p.suffix.lower() == '.m2'])
        if not files:
            self._entries = []
            self._last_df = None
            self._stats_df = None
            self._component_stats_df = None
            self.table_model.set_dataframe(pd.DataFrame())
            self.log.setPlainText('No .m2 files found (including subfolders).')
            self.stats_box.setPlainText('No .m2 files found (including subfolders).')
            self._refresh_all_plots()
            return

        # UI state
        self.run_btn.setEnabled(False)
        self.pick_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.log.setPlainText('')

        self.progress.setVisible(True)
        self.progress.setRange(0, len(files))
        self.progress.setValue(0)

        # Worker thread
        self._thread = QtCore.QThread(self)
        self._worker = BatchWorker(
            root=self._folder,
            files=files,
            method=method,
            axis_mode=axis_mode,
            drop=drop,
            group_mode=group_mode,
        )
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_worker_progress)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.finished.connect(self._thread.quit)
        self._worker.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)

        self._thread.start()

    def cancel_batch(self) -> None:
        if self._worker is not None:
            self._worker.cancel()

    def _on_worker_progress(self, cur: int, total: int, msg: str) -> None:
        try:
            self.progress.setRange(0, total)
            self.progress.setValue(cur)
        except Exception:
            pass

    def _on_worker_finished(self, payload: dict) -> None:
        # Tear down thread references first
        self._thread = None
        self._worker = None

        self.progress.setVisible(False)
        self.run_btn.setEnabled(True)
        self.pick_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)

        self._entries = payload.get('entries', []) or []
        self._entry_by_file = {e.rel_path: e for e in self._entries if e is not None}
        self._last_df = payload.get('df', None)
        self._stats_df = payload.get('stats_df', None)
        self._component_stats_df = payload.get('component_stats_df', None)

        log_text = payload.get('log', '')
        self.log.setPlainText(log_text)

        if self._last_df is None:
            self._last_df = pd.DataFrame()

        self.table_model.set_dataframe(self._last_df)
        # Resizing based on *all* rows can be sluggish on huge batches.
        try:
            if len(self._last_df) <= 500:
                self.table.resizeColumnsToContents()
        except Exception:
            pass

        self.stats_box.setPlainText(self._format_stats_text(self._last_df, self._stats_df))

        if self._component_stats_df is None:
            self._component_stats_df = pd.DataFrame()
        self.comp_stats_model.set_dataframe(self._component_stats_df)
        self.comp_stats_table.resizeColumnsToContents()

        self._refresh_all_plots()

        if len(self._last_df) > 0:
            self.table.selectRow(0)

    # ---------
    # Exports
    # ---------
    def export_csv(self) -> None:
        if self._last_df is None or self._last_df.empty:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Export batch CSV', '', 'CSV (*.csv)')
        if not path:
            return
        self._last_df.to_csv(path, index=False)

    def export_xlsx(self) -> None:
        if self._last_df is None or self._last_df.empty:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Export batch Excel', '', 'Excel (*.xlsx)')
        if not path:
            return

        with pd.ExcelWriter(path, engine='openpyxl') as writer:
            self._last_df.to_excel(writer, index=False, sheet_name='summary')
            if self._stats_df is not None and not self._stats_df.empty:
                self._stats_df.to_excel(writer, index=False, sheet_name='stats')
            if self._component_stats_df is not None and not self._component_stats_df.empty:
                self._component_stats_df.to_excel(writer, index=False, sheet_name='component_stats')

    # ---------
    # Display
    # ---------
    def _format_stats_text(self, df: pd.DataFrame, stats_df: Optional[pd.DataFrame]) -> str:
        n_total = int(len(df)) if df is not None else 0
        n_ok = 0
        if df is not None and not df.empty:
            if 'status' in df.columns:
                n_ok = int((df['status'] == 'OK').sum())
            elif 'error' in df.columns:
                n_ok = int(pd.isna(df['error']).sum())
        n_err = n_total - n_ok

        lines = []
        lines.append(f"Files found: {n_total}   OK: {n_ok}   ERR: {n_err}")

        if df is not None and 'component' in df.columns:
            try:
                n_comp = int(df['component'].astype(str).nunique())
                lines.append(f"Components: {n_comp}")
            except Exception:
                pass
        lines.append('')

        if stats_df is None or stats_df.empty:
            lines.append('No numeric statistics available.')
            return '\n'.join(lines)

        try:
            lines.append(stats_df.to_string(index=False, float_format=lambda x: f"{x:.6g}"))
        except Exception:
            lines.append(str(stats_df))
        return '\n'.join(lines)

    def _refresh_all_plots(self) -> None:
        self._refresh_summary_plot()
        self._refresh_distributions_plot()
        self._refresh_corr_scatter()
        self._refresh_corr_matrix()
        self._refresh_component_plot()

    def _refresh_summary_plot(self) -> None:
        metric = self.summary_metric_combo.currentData()
        sel = self.table.selectionModel().selectedRows() if self.table.selectionModel() else []
        sel_row = sel[0].row() if sel else None
        self.summary_plot.plot(self.table_model.dataframe(), metric, selected_row=sel_row)


    def _on_summary_point_clicked(self, row: int) -> None:
        """Select a table row when the user clicks a point on the summary plot."""
        try:
            df_view = self.table_model.dataframe()
            if df_view is None or df_view.empty:
                return
            if row < 0 or row >= len(df_view):
                return
            self.table.selectRow(int(row))
            try:
                self.table.scrollTo(self.table_model.index(int(row), 0))
            except Exception:
                pass
        except Exception:
            return

    def _refresh_distributions_plot(self) -> None:
        metric = self.dist_metric_combo.currentData()
        bins = int(self.bins_spin.value())
        self.dist_plot.plot(self._last_df, metric, bins=bins, group_col='component')

    def _refresh_corr_scatter(self) -> None:
        x = self.corr_x_combo.currentData()
        y = self.corr_y_combo.currentData()
        color_by = bool(self.corr_color_chk.isChecked())
        self.corr_plot.plot(self._last_df, x, y, group_col='component', color_by_group=color_by)

    def _refresh_corr_matrix(self) -> None:
        self.corr_matrix_plot.plot(self._last_df)

    def _checked_component_metrics(self) -> List[str]:
        cols: List[str] = []
        role = Qt.ItemDataRole.UserRole if hasattr(Qt, 'ItemDataRole') else Qt.UserRole
        checked = Qt.CheckState.Checked if hasattr(Qt, 'CheckState') else Qt.Checked
        for i in range(self.comp_metrics_list.count()):
            it = self.comp_metrics_list.item(i)
            if it.checkState() == checked:
                cols.append(str(it.data(role)))
        return cols

    def _refresh_component_plot(self) -> None:
        cols = self._checked_component_metrics()
        show_mean = bool(self.comp_mean_chk.isChecked())
        jitter = float(self.comp_jitter_spin.value())
        self.comp_plot.plot(self._last_df, cols, group_col='component', show_mean_std=show_mean, jitter=jitter)

        # component stats table is already updated on batch completion / group change

    def _on_group_mode_changed(self) -> None:
        # Grouping is purely derived from relative paths, so we can update it
        # without re-running any analysis.
        if self._last_df is None or self._last_df.empty or 'file' not in self._last_df.columns:
            self._refresh_all_plots()
            return

        group_mode = self.group_combo.currentData()
        try:
            self._last_df = self._last_df.copy()
            self._last_df['component'] = [
                _component_from_rel(str(rel), group_mode) for rel in self._last_df['file'].astype(str).tolist()
            ]
            self._component_stats_df = _compute_component_stats(self._last_df, group_col='component')
            self.table_model.set_dataframe(self._last_df)
            self.comp_stats_model.set_dataframe(self._component_stats_df if self._component_stats_df is not None else pd.DataFrame())
        except Exception:
            pass

        self._refresh_all_plots()

    def _on_row_selected(self, *args) -> None:
        sel = self.table.selectionModel().selectedRows() if self.table.selectionModel() else []
        if not sel:
            self.selected_label.setText('No file selected')
            self.file_results_box.setPlainText('')
            self.fit_plot.plot(pd.DataFrame(), None)
            self._refresh_summary_plot()
            return

        row = sel[0].row()
        self._refresh_summary_plot()

        df_view = self.table_model.dataframe()
        ent = None
        try:
            rel = str(df_view.iloc[row]['file']) if 'file' in df_view.columns else None
            if rel is not None:
                ent = self._entry_by_file.get(rel)
        except Exception:
            ent = None

        if ent is None:
            # Fallback: best-effort indexing if no 'file' column exists
            if row < 0 or row >= len(self._entries):
                return
            ent = self._entries[row]
        self.selected_label.setText(ent.rel_path)

        if ent.error or ent.results is None or ent.widths is None:
            self.file_results_box.setPlainText(ent.error or 'No results')
            self.fit_plot.plot(pd.DataFrame(), None)
            return

        self.file_results_box.setPlainText(_format_results_text(ent.results))
        wdf = widths_to_dataframe(ent.widths)
        self.fit_plot.plot(wdf, ent.results, axis_mode=ent.results.fit_x.axis_mode)

    def _on_row_double_clicked(self, index) -> None:
        """Open the selected measurement in the Single tab."""
        try:
            sel = self.table.selectionModel().selectedRows() if self.table.selectionModel() else []
            if not sel:
                return
            row = sel[0].row()

            df_view = self.table_model.dataframe()
            ent = None
            if df_view is not None and not df_view.empty and 'file' in df_view.columns:
                try:
                    rel = str(df_view.iloc[row]['file'])
                    ent = self._entry_by_file.get(rel)
                except Exception:
                    ent = None

            if ent is None:
                if row < 0 or row >= len(self._entries):
                    return
                ent = self._entries[row]

            if ent is None or ent.path is None:
                return
            self.open_single_requested.emit(str(ent.path))
        except Exception:
            return


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Beam M² / BPP Analyzer')

        self.tabs = QtWidgets.QTabWidget()
        self.single_tab = SingleTab()
        self.batch_tab = BatchTab()
        self.tabs.addTab(self.single_tab, 'Single')
        self.tabs.addTab(self.batch_tab, 'Batch')
        self.setCentralWidget(self.tabs)

        # Batch -> Single navigation
        try:
            self.batch_tab.open_single_requested.connect(self.open_m2_from_batch)
        except Exception:
            pass

        self._create_actions()
        self._create_menus()

    def _create_actions(self):
        self.act_open = QAction('Open .m2...', self)
        self.act_open.triggered.connect(self.open_m2)

        self.act_save_png = QAction('Save overlay PNG...', self)
        self.act_save_png.triggered.connect(self.save_overlay_png)

        self.act_export_csv = QAction('Export frames CSV...', self)
        self.act_export_csv.triggered.connect(self.export_csv)

        self.act_export_xlsx = QAction('Export workbook (summary+frames)...', self)
        self.act_export_xlsx.triggered.connect(self.export_xlsx)

        self.act_exit = QAction('Exit', self)
        self.act_exit.triggered.connect(self.close)

    def _create_menus(self):
        m = self.menuBar().addMenu('File')
        m.addAction(self.act_open)
        m.addAction(self.act_save_png)
        m.addSeparator()
        m.addAction(self.act_export_csv)
        m.addAction(self.act_export_xlsx)
        m.addSeparator()
        m.addAction(self.act_exit)

    def open_m2(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open .m2 file', '', 'M2 files (*.m2);;All files (*.*)')
        if not path:
            return
        self.tabs.setCurrentWidget(self.single_tab)
        self.single_tab.load_m2(path)

    def open_m2_from_batch(self, path: str) -> None:
        """Open a measurement selected in Batch.

        We also mirror the batch analysis settings into the Single tab so
        the numbers match what the user just saw.
        """
        try:
            # Mirror settings
            self.single_tab.method_combo.setCurrentIndex(self.batch_tab.method_combo.currentIndex())
            self.single_tab.axis_combo.setCurrentIndex(self.batch_tab.axis_combo.currentIndex())
            self.single_tab.drop_edit.setText(self.batch_tab.drop_edit.text())
        except Exception:
            pass

        self.tabs.setCurrentWidget(self.single_tab)
        self.single_tab.load_m2(path)

    def save_overlay_png(self):
        if self.single_tab.meas is None or self.single_tab.widths is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Save overlay PNG', '', 'PNG (*.png)')
        if not path:
            return
        self.single_tab.save_current_overlay_png(path)

    def export_csv(self):
        if self.single_tab.widths is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Export frames CSV', '', 'CSV (*.csv)')
        if not path:
            return
        export_widths_csv(self.single_tab.widths, path)

    def export_xlsx(self):
        if self.single_tab.widths is None or self.single_tab.results is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Export workbook', '', 'Excel (*.xlsx)')
        if not path:
            return
        export_single_report_excel(
            self.single_tab.results,
            self.single_tab.widths,
            path,
            meas=self.single_tab.meas,
            image_max_dim=128,
        )


def run():
    app = QtWidgets.QApplication([])
    w = MainWindow()
    w.resize(1400, 800)
    w.show()
    qexec(app)

