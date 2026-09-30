"""Project files (.lofi, and the old .fbproj).

JSON with the image list, per-image settings and rotations, and the current
index. Image paths are stored relative to the project where possible, so a
project can be moved together with its images. In memory, paths are always
absolute.
"""
import json
import logging
import os
from pathlib import Path, PurePosixPath

log = logging.getLogger(__name__)

PROJECT_EXT = '.lofi'
# Opens, but new saves use PROJECT_EXT.
LEGACY_PROJECT_EXT = '.fbproj'
SCHEMA_VERSION = 2


def _to_portable(image_path: Path, project_dir: Path) -> str:
    """POSIX relative path if on the same drive, else absolute."""
    image_path = image_path.resolve()
    try:
        rel = image_path.relative_to(project_dir.resolve())
        return str(PurePosixPath(*rel.parts))
    except ValueError:
        try:
            rel = os.path.relpath(image_path, start=project_dir.resolve())
            if os.path.isabs(rel):
                return str(image_path)
            return str(PurePosixPath(*Path(rel).parts))
        except ValueError:  # different drive on Windows
            return str(image_path)


def _from_portable(stored: str, project_dir: Path) -> Path:
    """Inverse of _to_portable."""
    p = Path(stored)
    if p.is_absolute() or (len(stored) >= 2 and stored[1] == ':'):
        return p
    posix = PurePosixPath(stored)
    return (project_dir / Path(*posix.parts)).resolve()


def save_project(path, image_files, image_settings, image_rotations=None, current_index=0):
    """Write a project file. `image_files` may contain Path or str entries."""
    path = Path(path)
    if path.suffix.lower() != PROJECT_EXT:
        path = path.with_suffix(PROJECT_EXT)
    project_dir = path.parent

    # Map both the given and the resolved path, since they can differ
    # (/tmp -> /private/tmp on macOS).
    abs_to_portable = {}
    stored_files = []
    for p in image_files:
        raw = Path(p)
        resolved = raw.resolve()
        portable = _to_portable(resolved, project_dir)
        stored_files.append(portable)
        for key in (str(raw), str(resolved)):
            abs_to_portable[key] = portable

    def _rekey(d):
        out = {}
        for k, v in (d or {}).items():
            k_str = str(k)
            portable = (abs_to_portable.get(k_str)
                        or abs_to_portable.get(str(Path(k_str).resolve())
                                              if Path(k_str).exists() else k_str))
            if portable is None:
                portable = k_str
            out[portable] = v
        return out

    payload = {
        'schema': SCHEMA_VERSION,
        'app': 'flashback_editor',   # format marker, predates the rename
        'image_files': stored_files,
        'image_settings': _rekey(image_settings),
        'image_rotations': {k: int(v) for k, v in _rekey(image_rotations).items()},
        'current_index': int(current_index),
    }
    path.write_text(json.dumps(payload, indent=2))
    return path


def load_project(path):
    """Returns (image_files, image_settings, image_rotations, current_index).
    Missing images are dropped."""
    path = Path(path)
    project_dir = path.parent
    payload = json.loads(path.read_text())
    if payload.get('app') != 'flashback_editor':
        raise ValueError(f"Not a Flashback project file: {path}")

    raw_files = payload.get('image_files', [])
    image_files = []
    portable_to_abs = {}
    missing = []
    for s in raw_files:
        resolved = _from_portable(s, project_dir)
        if resolved.exists():
            image_files.append(resolved)
            portable_to_abs[s] = str(resolved)
        else:
            missing.append(s)
    if missing:
        log.warning("Project references %d missing file(s); skipped.", len(missing))

    def _rekey(d):
        out = {}
        for k, v in (d or {}).items():
            abs_key = portable_to_abs.get(k)
            if abs_key is None:
                # Schema 1 stored absolute keys.
                if Path(k).exists():
                    abs_key = str(Path(k).resolve())
                else:
                    continue
            out[abs_key] = v
        return out

    image_settings = _rekey(payload.get('image_settings', {}))
    image_rotations = {k: int(v) % 360
                       for k, v in _rekey(payload.get('image_rotations', {})).items()}

    current_index = int(payload.get('current_index', 0))
    if image_files:
        current_index = max(0, min(current_index, len(image_files) - 1))
    else:
        current_index = 0

    return image_files, image_settings, image_rotations, current_index
