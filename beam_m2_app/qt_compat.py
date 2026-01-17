"""Qt compatibility shim for PyQt5 / PyQt6.

Goal: write the GUI once, run it on either binding.

This file avoids third-party abstraction layers (QtPy, etc.) on purpose.
"""

from __future__ import annotations

from dataclasses import dataclass

PYQT6 = False

try:
    from PyQt6 import QtCore, QtGui, QtWidgets  # type: ignore
    from PyQt6.QtCore import Qt  # type: ignore

    PYQT6 = True
except Exception:  # pragma: no cover
    from PyQt5 import QtCore, QtGui, QtWidgets  # type: ignore
    from PyQt5.QtCore import Qt  # type: ignore


# ---- Class aliases that moved between Qt5 and Qt6 ----
# QAction lives in QtWidgets in Qt5, but in QtGui in Qt6.
QAction = QtGui.QAction if PYQT6 else QtWidgets.QAction


def qexec(obj):
    """Call .exec() / .exec_() depending on the Qt binding.

    PyQt6 uses exec(); older PyQt5 code often exposes exec_().
    """
    if hasattr(obj, "exec"):
        return obj.exec()
    return obj.exec_()


# ---- Enum aliases with stable names ----
if PYQT6:  # pragma: no cover
    AlignCenter = Qt.AlignmentFlag.AlignCenter
    KeepAspectRatio = Qt.AspectRatioMode.KeepAspectRatio
    SmoothTransformation = Qt.TransformationMode.SmoothTransformation

    DisplayRole = Qt.ItemDataRole.DisplayRole
    EditRole = Qt.ItemDataRole.EditRole
    ToolTipRole = Qt.ItemDataRole.ToolTipRole

    Horizontal = Qt.Orientation.Horizontal
    Vertical = Qt.Orientation.Vertical

    ItemIsEnabled = Qt.ItemFlag.ItemIsEnabled
    ItemIsSelectable = Qt.ItemFlag.ItemIsSelectable

    # Common widget enum aliases
    SelectRows = QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
    SingleSelection = QtWidgets.QAbstractItemView.SelectionMode.SingleSelection
    ScrollHandDrag = QtWidgets.QGraphicsView.DragMode.ScrollHandDrag
    AnchorUnderMouse = QtWidgets.QGraphicsView.ViewportAnchor.AnchorUnderMouse
    Antialiasing = QtGui.QPainter.RenderHint.Antialiasing
    QImage_Grayscale8 = QtGui.QImage.Format.Format_Grayscale8
    QImage_ARGB32 = QtGui.QImage.Format.Format_ARGB32
    QImage_RGB888 = QtGui.QImage.Format.Format_RGB888

else:
    AlignCenter = Qt.AlignCenter
    KeepAspectRatio = Qt.KeepAspectRatio
    SmoothTransformation = Qt.SmoothTransformation

    DisplayRole = Qt.DisplayRole
    EditRole = Qt.EditRole
    ToolTipRole = Qt.ToolTipRole

    Horizontal = Qt.Horizontal
    Vertical = Qt.Vertical

    ItemIsEnabled = Qt.ItemIsEnabled
    ItemIsSelectable = Qt.ItemIsSelectable

    # Common widget enum aliases
    SelectRows = QtWidgets.QAbstractItemView.SelectRows
    SingleSelection = QtWidgets.QAbstractItemView.SingleSelection
    ScrollHandDrag = QtWidgets.QGraphicsView.ScrollHandDrag
    AnchorUnderMouse = QtWidgets.QGraphicsView.AnchorUnderMouse
    Antialiasing = QtGui.QPainter.Antialiasing
    QImage_Grayscale8 = QtGui.QImage.Format_Grayscale8
    QImage_ARGB32 = QtGui.QImage.Format_ARGB32
    QImage_RGB888 = QtGui.QImage.Format_RGB888


def qsignal(*args, **kwargs):
    """Return the correct signal class for the active Qt binding."""
    if PYQT6:  # pragma: no cover
        return QtCore.pyqtSignal(*args, **kwargs)
    return QtCore.pyqtSignal(*args, **kwargs)


@dataclass(frozen=True)
class QtBindingInfo:
    name: str
    version: str


def get_qt_binding_info() -> QtBindingInfo:
    if PYQT6:  # pragma: no cover
        return QtBindingInfo(name="PyQt6", version=getattr(QtCore, "PYQT_VERSION_STR", "?"))
    return QtBindingInfo(name="PyQt5", version=getattr(QtCore, "PYQT_VERSION_STR", "?"))
