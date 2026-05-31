"""
Per-vibe state persistence.

Saved vibes live in `vibe_state.json` under the platform user-data
directory (Qt's QStandardPaths.AppDataLocation). The file maps
vibe_id → VibeConfig (serialized as a dict).

Three layers as seen by the editor:
  - factory  — VIBE_PRESETS (config.vibe_config_for)
  - saved    — what's in this file, if anything (load_all / save_one)
  - session  — the live VibeConfig instance the editor currently holds
"""
import json
import logging
from pathlib import Path

from PySide6.QtCore import QStandardPaths

from .config import VibeConfig

log = logging.getLogger(__name__)

_FILE_NAME = 'vibe_state.json'


def _state_path() -> Path:
    base = QStandardPaths.writableLocation(QStandardPaths.AppDataLocation)
    if not base:
        base = str(Path.home() / '.flashback')
    p = Path(base)
    p.mkdir(parents=True, exist_ok=True)
    return p / _FILE_NAME


def _read_raw() -> dict:
    path = _state_path()
    if not path.exists():
        return {}
    try:
        with path.open('r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, dict):
            return {k: v for k, v in data.items() if isinstance(v, dict)}
    except (OSError, json.JSONDecodeError) as e:
        log.warning("⚠ Could not read %s: %s", path, e)
    return {}


def load_all() -> dict:
    """Return {vibe_id: VibeConfig}. Empty dict if nothing saved or file is corrupt."""
    return {vibe_id: VibeConfig.from_dict(d) for vibe_id, d in _read_raw().items()}


def _write_all(data: dict) -> None:
    path = _state_path()
    tmp = path.with_suffix('.json.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, sort_keys=True)
    tmp.replace(path)


def save_one(vibe_id: str, vibe: VibeConfig) -> None:
    """Persist `vibe` as the saved defaults for `vibe_id`, replacing any prior entry."""
    data = _read_raw()
    data[vibe_id] = vibe.to_dict()
    _write_all(data)


def clear_one(vibe_id: str) -> None:
    """Remove saved defaults for `vibe_id`. No-op if no entry exists."""
    data = _read_raw()
    if vibe_id in data:
        del data[vibe_id]
        _write_all(data)


def has_saved(vibe_id: str) -> bool:
    return vibe_id in _read_raw()
