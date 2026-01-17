from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class M2Frame:
    """One image/frame entry in an .m2 sequence."""

    index: int
    active: bool = True

    filename: str = ""
    path: str = ""  # path as stored in the file (often relative)

    # Stage/propagation position (vendor file uses 'z')
    z: float = float("nan")

    # Pixel size (physical unit per pixel, usually mm/px)
    facX: float = float("nan")
    facY: float = float("nan")

    # Centroid (pixels)
    X0: float = float("nan")
    Y0: float = float("nan")

    # Second central moments (pixels^2)
    XX: float = float("nan")
    YY: float = float("nan")
    XY: float = float("nan")

    # Optional fields
    center_x: Optional[float] = None
    center_y: Optional[float] = None
    range_min: Optional[float] = None
    range_max: Optional[float] = None
    snr: Optional[float] = None
    rel_diameter_error: Optional[float] = None
    comment: str = ""

    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class M2Measurement:
    """A full measurement sequence parsed from an .m2 file."""

    m2_path: Path
    frames: List[M2Frame] = field(default_factory=list)

    # Metadata
    wavelength_nm: Optional[float] = None

    # Arbitrary sections (for UI / debugging)
    sections: Dict[str, Dict[str, str]] = field(default_factory=dict)

    def active_frames(self) -> List[M2Frame]:
        return [f for f in self.frames if f.active]

    def image_root_dir(self) -> Path:
        """Best-effort guess for where the referenced images live."""
        # Many vendor exports use a relative folder stored in each frame's 'Path'.
        # Prefer the first frame, fall back to the .m2 file location.
        if not self.frames:
            return self.m2_path.parent
        p = self.frames[0].path.strip()
        if not p:
            return self.m2_path.parent

        # Normalize Windows-style .\folder\ paths.
        p = p.replace('\\', '/').replace('./', '').replace('.\\', '')
        p = p.strip('/')
        return (self.m2_path.parent / p).resolve()

    def resolve_image_path(self, frame: M2Frame) -> Path:
        """Resolve the image path for a frame.

        The .m2 file typically stores a relative folder (frame.Path).
        If that fails, we also try the .m2 directory itself.
        """
        root = self.image_root_dir()
        p1 = (root / frame.filename).resolve()
        if p1.exists():
            return p1

        p2 = (self.m2_path.parent / frame.filename).resolve()
        if p2.exists():
            return p2

        return p1
