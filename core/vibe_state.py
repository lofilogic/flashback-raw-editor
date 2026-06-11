"""
Per-vibe state persistence.

Saved vibes live in `vibe_state_1_5_0.json` under the platform user-data
directory (Qt's QStandardPaths.AppDataLocation). The filename encodes the
version that introduced the current schema (v2). Future schema breaks bump
both the integer `schema_version` in the envelope *and* the filename, so
older versions of the app keep reading their own file untouched.

File envelope:
    {
        "schema_version": 2,
        "migrated_from":  "vibe_state.json" | null,
        "vibes":          { "<id>": {…VibeConfig dict…}, … }
    }

Three layers as seen by the editor:
  - factory  — VIBE_PRESETS (config.vibe_config_for)
  - saved    — what's in this file, if anything (load_all / save_one)
  - session  — the live VibeConfig instance the editor currently holds

Legacy migration (pre-1.5):
  - The pre-1.5 file is `vibe_state.json` (no schema_version field).
  - On first launch with no v2 file present, `migrate_and_load()` reads
    the legacy file, runs each vibe through the bucket table below, and
    writes the v2 file. The legacy file is *never* modified, so a user
    rolling back to a pre-1.5 install finds their settings intact.
"""
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QStandardPaths

from .config import (
    VibeConfig, VIBE_PRESETS, vibe_config_for, FACTORY_LUTS,
    LUT_REF_FACTORY, LUT_REF_USER,
)

log = logging.getLogger(__name__)

# Bumping either of these is a schema break; bump them together. The
# filename uses underscores instead of dots so the file is safe on every
# platform and easy to grep for in an AppData listing.
SCHEMA_VERSION = 2
_FILE_NAME = 'vibe_state_1_5_0.json'
_FILE_NAME_LEGACY = 'vibe_state.json'


# =============================================================================
# MIGRATION TABLES — pre-1.5  →  1.5.0 normalized schema
# =============================================================================
# Bucket A (verbatim): not listed here — handled by carrying the field name
# through unchanged. Currently only the `enable_*` booleans qualify; the
# numeric fields all needed rescaling or reset.
#
# Bucket B (linear rescale): legacy_name → (new_name, multiplier). The
# multiplier turns the old 0..N internal scalar into the new 0..N×mult
# user-facing percent. `vignette_color_shift` is special-cased (its old
# range was 0..0.2, so the multiplier is 500).
#
# Bucket C (reset): old field is dropped; new field stays at the factory
# default for the vibe. The reason string ends up in MigrationReport so
# the user can read it in the post-migration summary.

_BUCKET_B_RESCALE = {
    'halation_strength':   ('halation_strength_pct',   100.0),
    'sharpen_strength':    ('sharpen_strength_pct',    100.0),
    'vignette_strength':   ('vignette_strength_pct',   100.0),
    'bloom_strength':      ('bloom_strength_pct',      100.0),
    'grain_strength':      ('grain_strength_pct',      100.0),
    'ca_zoom_blur':        ('ca_zoom_blur_pct',        100.0),
    'vignette_color_shift':('vignette_color_pct',      500.0),  # /0.2*100
}

_BUCKET_B_VERBATIM = {
    'halation_blur_radius', 'ca_steps', 'ca_blue_blur',
    'softness_sigma', 'sharpen_radius',
}

_BUCKET_C_RESET = {
    'halation_threshold':  ('halation_threshold_stops',
                            'threshold unit changed to EV above middle grey'),
    'bloom_threshold':     ('bloom_threshold_stops',
                            'threshold unit changed to EV above middle grey'),
    'cnr_sigma':           ('cnr_amount_pct',
                            'CNR algorithm and color space changed (Rec.2020 → Lab)'),
    'vignette_feather':    ('vignette_curve',
                            'sign convention inverted (now signed -100…+100)'),
    'ca_strength':         ('ca_pixels',
                            'CA geometry changed; legacy scale factor would mis-target'),
}

# Bucket A: explicit allowlist of legacy fields that survive verbatim.
_BUCKET_A_VERBATIM = {
    'enable_halation', 'enable_chromatic_aberration', 'enable_softness',
    'enable_grain', 'enable_sharpen', 'enable_cnr', 'enable_lut',
    'enable_vignette', 'enable_bloom',
    'enable_reverse_autoexposure', 'enable_post_ae_exposure_boost',
    'reverse_autoexposure_t_ref', 'post_ae_exposure_boost_ev',
    'reverse_ae_strength', 'base_exposure_offset_v2',
    'dng_profile_name',
}


@dataclass
class MigrationReport:
    """Summary of one legacy → v2 migration, for the post-migration UI."""
    legacy_file: Path
    migrated_vibe_ids: list = field(default_factory=list)
    rescaled_fields: list = field(default_factory=list)       # (vibe_id, legacy_name, new_name)
    reset_fields: list = field(default_factory=list)          # (vibe_id, new_name, reason)
    custom_luts_reset: list = field(default_factory=list)     # (vibe_id, legacy_path)

    @property
    def any_changes(self) -> bool:
        return bool(self.migrated_vibe_ids)


# =============================================================================
# PATHS
# =============================================================================

def _state_dir() -> Path:
    base = QStandardPaths.writableLocation(QStandardPaths.AppDataLocation)
    if not base:
        base = str(Path.home() / '.lofilogic')
    p = Path(base)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _state_path() -> Path:
    return _state_dir() / _FILE_NAME


def _legacy_path() -> Path:
    return _state_dir() / _FILE_NAME_LEGACY


# =============================================================================
# RAW I/O
# =============================================================================

def _read_envelope() -> Optional[dict]:
    """Read the v2 file. Returns the parsed envelope (dict) or None if
    the file does not exist or is unreadable / malformed."""
    path = _state_path()
    if not path.exists():
        return None
    try:
        with path.open('r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, dict) and data.get('schema_version') == SCHEMA_VERSION:
            return data
        log.warning("⚠ %s has unexpected schema (%r) — ignoring", path, data.get('schema_version'))
    except (OSError, json.JSONDecodeError) as e:
        log.warning("⚠ Could not read %s: %s", path, e)
    return None


def _read_legacy() -> Optional[dict]:
    """Read the pre-1.5 file. Returns the raw {vibe_id: dict} map, or
    None if the file is absent / unreadable. The file has no envelope —
    it's a flat dict keyed by vibe id."""
    path = _legacy_path()
    if not path.exists():
        return None
    try:
        with path.open('r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, dict):
            return {k: v for k, v in data.items() if isinstance(v, dict)}
    except (OSError, json.JSONDecodeError) as e:
        log.warning("⚠ Could not read legacy %s: %s", path, e)
    return None


def _write_envelope(vibes: dict, migrated_from: Optional[str] = None,
                    migration_acknowledged: bool = True) -> None:
    """Write the v2 envelope. `migrated_from` and `migration_acknowledged`
    track the post-migration notice state — both default to "no migration"
    / "nothing to acknowledge" so plain save_one calls don't disturb them
    once acknowledged."""
    existing = _read_envelope() or {}
    envelope = {
        'schema_version':         SCHEMA_VERSION,
        'migrated_from':          migrated_from if migrated_from is not None else existing.get('migrated_from'),
        'migration_acknowledged': bool(existing.get('migration_acknowledged', True))
                                  if migrated_from is None else migration_acknowledged,
        'vibes':                  vibes,
    }
    path = _state_path()
    tmp = path.with_suffix('.json.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        json.dump(envelope, f, indent=2, sort_keys=True)
    tmp.replace(path)


# =============================================================================
# LEGACY MIGRATION (pre-1.5 → 1.5.0)
# =============================================================================

def _migrate_lut_path(legacy_path: str, vibe_id: str, report: MigrationReport):
    """Map a pre-1.5 `lut_path` string onto a new tagged `lut_ref`.

    Bundled factory files (matched by filename) are rewritten to
    `factory:<id>`. User LUTs are reset to the vibe's factory LUT because
    a pre-1.5 .cube was built against the old color pipeline (linear
    Rec.2020-ish) and would look ~2 stops over + colour-shifted under the
    new ACEScg pipeline. The original path is stashed in
    `legacy_user_lut` so the user can re-import it manually once they've
    regenerated it.

    Returns (new_lut_ref, legacy_user_lut_to_store).
    """
    if not legacy_path:
        return '', ''

    # Factory match by filename — robust to install-dir relocation.
    factory_files = {Path(rel).name: fid for fid, rel in FACTORY_LUTS.items()}
    fname = Path(legacy_path).name
    if fname in factory_files:
        return f"{LUT_REF_FACTORY}{factory_files[fname]}", ''

    # Custom LUT — reset to this vibe's factory and stash the legacy path
    # so the summary can list it. If the vibe id isn't a known factory,
    # fall back to "no LUT".
    report.custom_luts_reset.append((vibe_id, legacy_path))
    if vibe_id in VIBE_PRESETS:
        return vibe_config_for(vibe_id).lut_ref, legacy_path
    return '', legacy_path


def _migrate_one_vibe(vibe_id: str, legacy: dict, report: MigrationReport) -> dict:
    """Turn one legacy vibe dict into a v2 VibeConfig dict.

    Strategy: start from the factory baseline for this vibe (so anything
    the user didn't have or that we're resetting falls back to a sane
    new-pipeline default), then overlay carried-forward and rescaled
    fields from the legacy entry.
    """
    if vibe_id in VIBE_PRESETS:
        base = vibe_config_for(vibe_id).to_dict()
    else:
        # Unknown vibe id — keep the entry so the user doesn't lose it,
        # but use the bare-defaults VibeConfig as the baseline.
        base = VibeConfig().to_dict()

    for legacy_name in _BUCKET_A_VERBATIM:
        if legacy_name in legacy and legacy_name in base:
            base[legacy_name] = legacy[legacy_name]

    for legacy_name in _BUCKET_B_VERBATIM:
        if legacy_name in legacy and legacy_name in base:
            base[legacy_name] = legacy[legacy_name]

    for legacy_name, (new_name, mult) in _BUCKET_B_RESCALE.items():
        if legacy_name in legacy:
            try:
                base[new_name] = float(legacy[legacy_name]) * mult
                report.rescaled_fields.append((vibe_id, legacy_name, new_name))
            except (TypeError, ValueError):
                pass

    for legacy_name, (new_name, reason) in _BUCKET_C_RESET.items():
        if legacy_name in legacy:
            # New field stays at the factory default; just record the
            # reset so the user knows what shifted.
            report.reset_fields.append((vibe_id, new_name, reason))

    new_ref, legacy_user_lut = _migrate_lut_path(
        legacy.get('lut_path', ''), vibe_id, report,
    )
    if new_ref:
        base['lut_ref'] = new_ref
    if legacy_user_lut:
        base['legacy_user_lut'] = legacy_user_lut

    return base


def _run_legacy_migration() -> Optional[MigrationReport]:
    """If a legacy file exists and no v2 file does, migrate and write v2.

    Returns the MigrationReport, or None if no migration ran (either no
    legacy file present, or v2 file already exists, in which case the
    legacy file is just ignored — we never re-migrate over an existing
    v2 file)."""
    legacy = _read_legacy()
    if legacy is None:
        return None

    report = MigrationReport(legacy_file=_legacy_path())
    migrated = {}
    for vibe_id, legacy_dict in legacy.items():
        migrated[vibe_id] = _migrate_one_vibe(vibe_id, legacy_dict, report)
        report.migrated_vibe_ids.append(vibe_id)

    _write_envelope(migrated, migrated_from=_FILE_NAME_LEGACY,
                    migration_acknowledged=False)
    log.info("✓ Migrated %d vibe(s) from %s → %s",
             len(migrated), _FILE_NAME_LEGACY, _FILE_NAME)
    return report


# =============================================================================
# PUBLIC API
# =============================================================================

def _read_raw_vibes() -> dict:
    """Return the {vibe_id: dict} map from the v2 file, or {} if absent."""
    env = _read_envelope()
    if env is None:
        return {}
    vibes = env.get('vibes', {})
    if not isinstance(vibes, dict):
        return {}
    return {k: v for k, v in vibes.items() if isinstance(v, dict)}


def load_all() -> dict:
    """Return {vibe_id: VibeConfig}. Empty dict if nothing saved.

    Does NOT trigger migration — call `migrate_and_load()` once at
    startup if you want the pre-1.5 file pulled in. This function is
    safe to call repeatedly (e.g. on every vibe switch) without
    re-running the migrator.
    """
    return {vid: VibeConfig.from_dict(d) for vid, d in _read_raw_vibes().items()}


def migrate_and_load() -> tuple:
    """Run legacy migration if needed, then return (vibes, pending_notice).

    Call this once at app startup. `vibes` is {vibe_id: VibeConfig} as
    `load_all` returns. `pending_notice` is True when there is a
    not-yet-acknowledged migration the editor should surface (either the
    migration just ran, or it ran previously but the app exited before
    the user dismissed the summary)."""
    env = _read_envelope()
    if env is None:
        # No v2 file — see if there's a legacy file to migrate.
        report = _run_legacy_migration()
        return load_all(), (report if report is not None and report.any_changes else None)
    # v2 file exists — was a previous migration left unacknowledged?
    if env.get('migrated_from') and not env.get('migration_acknowledged', True):
        # We have no MigrationReport on disk (it's transient by design);
        # surface a lightweight "migration happened previously" notice so
        # the user can dismiss it. The detailed report is only available
        # the run in which migration ran — that's a deliberate trade-off
        # (the legacy file is unchanged on disk for re-running if needed).
        legacy_marker = MigrationReport(legacy_file=_legacy_path())
        legacy_marker.migrated_vibe_ids = list(_read_raw_vibes().keys())
        return load_all(), legacy_marker
    return load_all(), None


def mark_migration_acknowledged() -> None:
    """Persist the user's dismissal of the post-migration notice so it
    doesn't reappear next launch. No-op if no acknowledgement is pending."""
    env = _read_envelope()
    if env is None or env.get('migration_acknowledged', True):
        return
    _write_envelope(env.get('vibes', {}),
                    migrated_from=env.get('migrated_from'),
                    migration_acknowledged=True)


def save_one(vibe_id: str, vibe: VibeConfig) -> None:
    """Persist `vibe` as the saved defaults for `vibe_id`, replacing any
    prior entry. Writes the full v2 envelope so the schema_version stays
    current on every save."""
    vibes = _read_raw_vibes()
    vibes[vibe_id] = vibe.to_dict()
    _write_envelope(vibes)


def clear_one(vibe_id: str) -> None:
    """Remove saved defaults for `vibe_id`. No-op if no entry exists."""
    vibes = _read_raw_vibes()
    if vibe_id in vibes:
        del vibes[vibe_id]
        _write_envelope(vibes)


def has_saved(vibe_id: str) -> bool:
    return vibe_id in _read_raw_vibes()
