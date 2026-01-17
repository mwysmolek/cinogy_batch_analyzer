"""Parser for vendor .m2 text files.

The file format behaves like an INI with lots of UI junk.
The only parts we *need* for analysis are:

- Per-frame: z, facX/facY, X0/Y0, XX/YY/XY, file name
- Global: wavelength (nm)

Everything else is preserved into `measurement.sections` so you can
inspect or extend later.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional, Union, Tuple

from .models import M2Frame, M2Measurement


def _parse_key_value(line: str) -> Optional[Tuple[str, str]]:
    """Parse lines like 'Key\t= 123' or 'Key = 123'."""
    if '=' not in line:
        return None
    left, right = line.split('=', 1)
    key = left.strip()
    value = right.strip()
    if not key:
        return None
    return key, value


def _to_float(s: str) -> Optional[float]:
    try:
        # Many values look like '3.402823466e+038'
        return float(s)
    except Exception:
        return None


def _to_int(s: str) -> Optional[int]:
    try:
        return int(float(s))
    except Exception:
        return None


def parse_m2_file(path: Union[str, Path]) -> M2Measurement:
    p = Path(path).expanduser().resolve()
    text = p.read_text(encoding='utf-8', errors='ignore').splitlines()

    meas = M2Measurement(m2_path=p)

    section_name: Optional[str] = None
    section: Dict[str, str] = {}

    def commit_section(name: Optional[str], data: Dict[str, str]) -> None:
        if not name:
            return
        if name.isdigit() and len(name) == 4:
            idx = int(name)
            frame = M2Frame(index=idx)
            frame.raw = dict(data)

            # Core fields (best-effort)
            frame.active = (_to_int(data.get('Active', '1')) or 1) != 0
            frame.filename = data.get('FileName', '')
            frame.path = data.get('Path', '')

            z = _to_float(data.get('z', ''))
            if z is not None:
                frame.z = z

            facx = _to_float(data.get('facX', ''))
            facy = _to_float(data.get('facY', ''))
            if facx is not None:
                frame.facX = facx
            if facy is not None:
                frame.facY = facy

            for k in ('X0', 'Y0', 'XX', 'YY', 'XY'):
                val = _to_float(data.get(k, ''))
                if val is not None:
                    setattr(frame, k, val)

            # Optional
            frame.center_x = _to_float(data.get('Center X', ''))
            frame.center_y = _to_float(data.get('Center Y', ''))
            frame.range_min = _to_float(data.get('RangeMin', ''))
            frame.range_max = _to_float(data.get('RangeMax', ''))
            frame.snr = _to_float(data.get('SNR', ''))
            frame.rel_diameter_error = _to_float(data.get('Rel Diameter Error', ''))
            frame.comment = data.get('Comment', '')

            meas.frames.append(frame)
        else:
            # Store generic sections
            meas.sections[name] = dict(data)

    for raw in text:
        line = raw.strip('\r\n')
        if not line:
            continue
        if line.strip() == '-':
            continue

        if line.startswith('[') and line.endswith(']'):
            # New section
            commit_section(section_name, section)
            section_name = line[1:-1].strip()
            section = {}
            continue

        kv = _parse_key_value(line)
        if kv is None:
            continue
        k, v = kv
        section[k] = v

    commit_section(section_name, section)

    # Global metadata: wavelength
    # It appears in multiple places: [Beam Quality] WaveLength, [Lens] Wavelength
    beamq = meas.sections.get('Beam Quality', {})
    wl = _to_float(beamq.get('WaveLength', ''))
    if wl is None:
        header = meas.sections.get('Header', {})
        wl = _to_float(header.get('WaveLength', ''))
    if wl is None:
        lens = meas.sections.get('Lens', {})
        wl = _to_float(lens.get('Wavelength', ''))

    if wl is not None:
        meas.wavelength_nm = wl

    # Sort frames by index (file may already be ordered)
    meas.frames.sort(key=lambda f: f.index)

    return meas


# Backwards-compatible re-export
parse = parse_m2_file
