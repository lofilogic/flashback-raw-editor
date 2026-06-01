"""
Tests for VibeConfig / ImageAdjustments and the vibe preset machinery.

These are the parts that will need to keep working when projects (saved
per-image state) get added — so an explicit safety net here is worth it.
"""
from dataclasses import fields

from core.config import (
    VibeConfig, ImageAdjustments, VIBE_PRESETS,
    vibe_config_for, VIBE_FIELD_NAMES,
)


# =============================================================================
# VibeConfig serialization
# =============================================================================

def test_vibeconfig_to_dict_from_dict_roundtrip():
    """to_dict / from_dict must roundtrip without information loss.

    This is the foundational invariant for Save Projects: if a config
    can't survive a JSON round-trip, projects will silently corrupt.
    """
    original = vibe_config_for('disposable')
    restored = VibeConfig.from_dict(original.to_dict())
    assert restored == original


def test_vibeconfig_from_dict_ignores_unknown_keys():
    """Loading a saved config with extra/renamed keys must not raise."""
    d = vibe_config_for('disposable').to_dict()
    d['this_key_does_not_exist'] = 42
    restored = VibeConfig.from_dict(d)
    assert restored == vibe_config_for('disposable')


def test_vibeconfig_from_dict_coerces_types():
    """Strings like '0.5' for a float field should be coerced, not silently dropped."""
    cfg = VibeConfig.from_dict({'grain_strength_pct': '42', 'ca_steps': '7'})
    assert cfg.grain_strength_pct == 42.0
    assert cfg.ca_steps == 7


def test_vibeconfig_copy_is_independent():
    """copy() must return a deep-enough copy that mutations don't leak."""
    a = vibe_config_for('disposable')
    b = a.copy()
    b.grain_strength_pct = 999.0
    assert a.grain_strength_pct != 999.0


# =============================================================================
# Vibe presets
# =============================================================================

def test_every_preset_yields_complete_vibeconfig():
    """Every preset must yield a VibeConfig containing all declared fields."""
    expected_names = {f.name for f in fields(VibeConfig)}
    for vibe_id in VIBE_PRESETS:
        cfg = vibe_config_for(vibe_id)
        # All fields are present (dataclass guarantees) and accessible
        for name in expected_names:
            assert hasattr(cfg, name), f"{vibe_id} missing field {name}"


def test_preset_field_types_match_dataclass():
    """Preset values must match the dataclass type, not just coerce."""
    type_by_name = {f.name: f.type for f in fields(VibeConfig)}
    for vibe_id in VIBE_PRESETS:
        cfg = vibe_config_for(vibe_id)
        for name, expected_type in type_by_name.items():
            value = getattr(cfg, name)
            if expected_type is bool:
                assert isinstance(value, bool), \
                    f"{vibe_id}.{name} should be bool, got {type(value).__name__}"
            elif expected_type in (int, float, str):
                assert isinstance(value, expected_type), \
                    f"{vibe_id}.{name} should be {expected_type.__name__}, " \
                    f"got {type(value).__name__}"


def test_default_vibeconfig_is_factory():
    """VibeConfig() with no args must equal the documented factory baseline.

    This anchors the dataclass defaults to the named module constants.
    """
    cfg = VibeConfig()
    # Spot-check fields against module constants so a future drift fails loudly
    from core.config import (
        HALATION_THRESHOLD_STOPS, GRAIN_STRENGTH_PCT, BLOOM_THRESHOLD_STOPS,
        BASE_EXPOSURE_OFFSET_V2,
    )
    assert cfg.halation_threshold_stops == HALATION_THRESHOLD_STOPS
    assert cfg.grain_strength_pct == GRAIN_STRENGTH_PCT
    assert cfg.bloom_threshold_stops == BLOOM_THRESHOLD_STOPS
    assert cfg.base_exposure_offset_v2 == BASE_EXPOSURE_OFFSET_V2


# =============================================================================
# ImageAdjustments
# =============================================================================

def test_image_adjustments_roundtrip():
    a = ImageAdjustments(exposure_ev=1.5, wb_temp=200.0, tint=-3.0,
                         push_pull_ev=0.5, rotation=90,
                         active_vibe_id='disposable')
    assert ImageAdjustments.from_dict(a.to_dict()) == a


def test_image_adjustments_default_has_all_fields():
    a = ImageAdjustments()
    assert a.exposure_ev == 0.0
    assert a.wb_temp == 0.0
    assert a.tint == 0.0
    assert a.push_pull_ev == 0.0
    assert a.rotation == 0
    assert a.active_vibe_id == ''


def test_image_adjustments_copy_is_independent():
    a = ImageAdjustments(exposure_ev=1.0)
    b = a.copy()
    b.exposure_ev = 99.0
    assert a.exposure_ev == 1.0


# =============================================================================
# Field schema
# =============================================================================

def test_vibe_field_names_matches_dataclass():
    """VIBE_FIELD_NAMES must stay in sync with the VibeConfig dataclass."""
    expected = tuple(f.name for f in fields(VibeConfig))
    assert VIBE_FIELD_NAMES == expected
