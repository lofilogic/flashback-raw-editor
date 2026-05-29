"""
Tests for DebugConfig serialization and vibe-state plumbing.

These are the parts that will need to keep working when projects (saved
per-image state) get added — so an explicit safety net here is worth it.
"""
import pytest

from core.config import (
    VIBE_FIELDS, VIBE_PRESETS, DebugConfig,
    factory_state_for, snapshot_debug_config, apply_state_to_debug_config,
)


def test_snapshot_apply_is_roundtrip():
    """snapshot → apply → snapshot must produce identical state.

    This is the foundational invariant for any Save-Project feature: if
    snapshot/apply ever drifted, projects could silently corrupt settings.
    """
    snap = snapshot_debug_config()
    apply_state_to_debug_config(snap)
    snap2 = snapshot_debug_config()
    assert snap == snap2


def test_every_vibe_produces_complete_factory_state():
    """Every preset must yield a state dict containing all VIBE_FIELDS."""
    for vibe_id in VIBE_PRESETS:
        state = factory_state_for(vibe_id)
        for name, _ in VIBE_FIELDS:
            assert name in state, f"vibe '{vibe_id}' missing field '{name}'"


def test_factory_state_field_types_match_schema():
    """Factory state values must match the type declared in VIBE_FIELDS."""
    for vibe_id in VIBE_PRESETS:
        state = factory_state_for(vibe_id)
        for name, expected_type in VIBE_FIELDS:
            value = state[name]
            # bools are ints in Python; check bool first
            if expected_type is bool:
                assert isinstance(value, bool), \
                    f"{vibe_id}.{name} should be bool, got {type(value).__name__}"
            else:
                assert isinstance(value, expected_type), \
                    f"{vibe_id}.{name} should be {expected_type.__name__}, " \
                    f"got {type(value).__name__}"


def test_apply_state_ignores_unknown_keys():
    """Loading a saved state with extra/renamed keys must not raise."""
    before = snapshot_debug_config()
    apply_state_to_debug_config({'this_key_does_not_exist': 42})
    after = snapshot_debug_config()
    assert before == after


def test_apply_state_coerces_types():
    """Strings like '0.5' for a float field should be coerced, not crash."""
    before = DebugConfig.grain_strength
    apply_state_to_debug_config({'grain_strength': '0.42'})
    assert DebugConfig.grain_strength == pytest.approx(0.42)
    # restore so other tests aren't affected
    DebugConfig.grain_strength = before


def test_reset_restores_all_vibe_fields():
    """DebugConfig.reset() must touch every field that VIBE_FIELDS exposes,
    otherwise reset() will silently leave per-vibe overrides behind.
    """
    # Mutate every field to a sentinel
    for name, t in VIBE_FIELDS:
        if t is bool:
            setattr(DebugConfig, name, not getattr(DebugConfig, name))
        elif t is int:
            setattr(DebugConfig, name, 999)
        elif t is float:
            setattr(DebugConfig, name, -999.0)
        elif t is str:
            setattr(DebugConfig, name, '__sentinel__')

    DebugConfig.reset()

    # After reset, no field should still hold the sentinel
    for name, t in VIBE_FIELDS:
        value = getattr(DebugConfig, name)
        if t is int:
            assert value != 999, f"{name} not reset"
        elif t is float:
            assert value != -999.0, f"{name} not reset"
        elif t is str:
            assert value != '__sentinel__', f"{name} not reset"
