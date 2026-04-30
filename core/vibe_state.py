"""
Per-vibe state persistence.

Saved overrides live in `vibe_state.json` under the platform user-data
directory (Qt's QStandardPaths.AppDataLocation). The file maps vibe_id →
state dict (a subset or full snapshot of VIBE_FIELDS).

Three layers as seen by the editor:
  - factory  — VIBE_PRESETS + module defaults  (config.factory_state_for)
  - saved    — what's in this file, if anything (load_all / save_one)
  - session  — live DebugConfig
"""
import json
from pathlib import Path

from PySide6.QtCore import QStandardPaths

_FILE_NAME = 'vibe_state.json'


def _state_path() -> Path:
    base = QStandardPaths.writableLocation(QStandardPaths.AppDataLocation)
    if not base:
        base = str(Path.home() / '.flashback')
    p = Path(base)
    p.mkdir(parents=True, exist_ok=True)
    return p / _FILE_NAME


def load_all() -> dict:
    """Return {vibe_id: state_dict}. Empty dict if nothing saved or file is corrupt."""
    path = _state_path()
    if not path.exists():
        return {}
    try:
        with path.open('r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, dict):
            return {k: v for k, v in data.items() if isinstance(v, dict)}
    except (OSError, json.JSONDecodeError) as e:
        print(f"⚠ Could not read {path}: {e}")
    return {}


def _write_all(data: dict) -> None:
    path = _state_path()
    tmp = path.with_suffix('.json.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, sort_keys=True)
    tmp.replace(path)


def save_one(vibe_id: str, state: dict) -> None:
    """Persist `state` as the saved defaults for `vibe_id`, replacing any prior entry."""
    data = load_all()
    data[vibe_id] = dict(state)
    _write_all(data)


def clear_one(vibe_id: str) -> None:
    """Remove saved defaults for `vibe_id`. No-op if no entry exists."""
    data = load_all()
    if vibe_id in data:
        del data[vibe_id]
        _write_all(data)


def has_saved(vibe_id: str) -> bool:
    return vibe_id in load_all()
