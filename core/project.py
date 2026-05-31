"""Save/load Flashback project files (.fbproj).

A project file is JSON containing:
  - schema version
  - list of image file paths (stored relative to the project file when on the
    same drive; absolute otherwise — projects survive being moved alongside
    their image tree, and absolute paths still work for images outside it)
  - per-image settings dict (path -> ImageAdjustments.to_dict())
  - per-image rotation dict (path -> degrees)
  - current selected index

Paths in `image_settings` and `image_rotations` always use the original
absolute path of the image file at load time (the in-memory representation),
not the relative form stored on disk. Conversion happens at save/load.
"""
import json
import logging
import os
from pathlib import Path, PurePosixPath

log = logging.getLogger(__name__)

PROJECT_EXT = '.fbproj'
SCHEMA_VERSION = 2


def _to_portable(image_path: Path, project_dir: Path) -> str:
    """Return a string path suitable for serialising in the project file.

    Uses a POSIX-style relative path when the image lives on the same drive
    as the project (works across OSes). Falls back to the platform-native
    absolute path when relativisation isn't possible.
    """
    image_path = image_path.resolve()
    try:
        rel = image_path.relative_to(project_dir.resolve())
        return str(PurePosixPath(*rel.parts))
    except ValueError:
        # Not under project_dir; try a common-ancestor relative path so e.g.
        # a sibling folder still serialises portably.
        try:
            rel = os.path.relpath(image_path, start=project_dir.resolve())
            # Reject if the relative path escapes to a different drive on
            # Windows (relpath raises ValueError for that case).
            if os.path.isabs(rel):
                return str(image_path)
            return str(PurePosixPath(*Path(rel).parts))
        except ValueError:
            return str(image_path)


def _from_portable(stored: str, project_dir: Path) -> Path:
    """Inverse of _to_portable. Resolves stored value to an absolute Path."""
    # Absolute path on either OS → use as-is.
    p = Path(stored)
    if p.is_absolute() or (len(stored) >= 2 and stored[1] == ':'):
        return p
    # Treat as POSIX-style relative; convert separators for the host OS.
    posix = PurePosixPath(stored)
    return (project_dir / Path(*posix.parts)).resolve()


def save_project(path, image_files, image_settings, image_rotations=None, current_index=0):
    """Write a project file. `image_files` may contain Path or str entries."""
    path = Path(path)
    if path.suffix.lower() != PROJECT_EXT:
        path = path.with_suffix(PROJECT_EXT)
    project_dir = path.parent

    # Map every input form (pre- and post-resolve) to its portable form so
    # the dict-rekey below tolerates callers that pass keys as `/tmp/x` while
    # image_files holds Path('/tmp/x') (resolves to /private/tmp/x on macOS).
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
                # Unknown image (e.g. setting for a file no longer in the
                # project) — keep the original key so we don't silently drop.
                portable = k_str
            out[portable] = v
        return out

    payload = {
        'schema': SCHEMA_VERSION,
        'app': 'flashback_editor',
        'image_files': stored_files,
        'image_settings': _rekey(image_settings),
        'image_rotations': {k: int(v) for k, v in _rekey(image_rotations).items()},
        'current_index': int(current_index),
    }
    path.write_text(json.dumps(payload, indent=2))
    return path


def load_project(path):
    """Read a project file.

    Returns (image_files, image_settings, image_rotations, current_index).
    Image paths are returned as absolute Paths. Missing image files are
    dropped; their settings/rotations are dropped with them.
    """
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
                # Tolerate older v1 payloads that stored absolute keys directly.
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
