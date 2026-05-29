"""
Sanity tests for the color pipeline math.

These guard against accidental edits to the hard-coded matrices and the
WB / tint / tone-curve helpers. They don't validate that the *output looks
good* — only that the math is internally consistent and that documented
invariants hold.
"""
import numpy as np
import pytest

from core.processor import (
    RAW_TO_ACESCG, XYZ_D50_TO_ACESCG, FM1_RAW_TO_XYZ_D50,
    ACESCG_TO_LINSRGB, LINSRGB_TO_ACESCG,
    _kelvin_to_acescg_gain, _tint_to_acescg_gain, BASE_KELVIN,
    _apply_tone_curve, _TONE_CURVE_LUT,
)


# =============================================================================
# Matrix consistency
# =============================================================================

def test_raw_to_acescg_matches_fused_chain():
    """RAW_TO_ACESCG must equal the two-step XYZ_D50_TO_ACESCG @ FM1_RAW_TO_XYZ_D50.

    If someone edits one matrix but forgets the fused version (or vice versa),
    Flashback raws will silently render with a hue shift. This catches it.
    """
    expected = XYZ_D50_TO_ACESCG @ FM1_RAW_TO_XYZ_D50
    assert np.allclose(RAW_TO_ACESCG, expected, atol=1e-6)


def test_acescg_linsrgb_matrices_are_inverses():
    """ACESCG_TO_LINSRGB and LINSRGB_TO_ACESCG must be matrix inverses."""
    identity = ACESCG_TO_LINSRGB @ LINSRGB_TO_ACESCG
    assert np.allclose(identity, np.eye(3), atol=1e-5)


# =============================================================================
# White balance
# =============================================================================

def test_wb_gain_at_base_kelvin_is_identity():
    """At slider zero (BASE_KELVIN = 5500K) the WB gain must be (1, 1, 1).

    This is what makes the WB slider feel neutral at its zero position.
    """
    gain = _kelvin_to_acescg_gain(BASE_KELVIN)
    assert np.allclose(gain, [1.0, 1.0, 1.0], atol=1e-5)


@pytest.mark.parametrize("kelvin", [2000, 3200, 4000, 5500, 6500, 7000, 10000])
def test_wb_gain_g_channel_always_normalised(kelvin):
    """G is the Bayer reference channel; gain[1] must always be exactly 1.0."""
    gain = _kelvin_to_acescg_gain(float(kelvin))
    assert gain[1] == pytest.approx(1.0, abs=1e-6)


def test_wb_lower_target_neutralises_warm_scene():
    """Target K = tungsten (3200K) describes "scene was shot under tungsten".
    The gain must neutralise that warmth: boost B, cut R relative to base.
    Catches the surprisingly easy mistake of reversing the formula direction.
    """
    base = _kelvin_to_acescg_gain(BASE_KELVIN)
    tungsten_target = _kelvin_to_acescg_gain(3200.0)
    assert tungsten_target[0] < base[0]   # red cut
    assert tungsten_target[2] > base[2]   # blue boosted


def test_wb_higher_target_neutralises_cool_scene():
    """Symmetric check: target K = shade (8000K) means scene was shot under
    cool light; gain must warm it up — boost R, cut B."""
    base = _kelvin_to_acescg_gain(BASE_KELVIN)
    shade_target = _kelvin_to_acescg_gain(8000.0)
    assert shade_target[0] > base[0]   # red boosted
    assert shade_target[2] < base[2]   # blue cut


# =============================================================================
# Tint
# =============================================================================

def test_tint_zero_is_identity():
    gain = _tint_to_acescg_gain(0.0)
    assert np.allclose(gain, [1.0, 1.0, 1.0])


def test_tint_positive_is_magenta():
    """Positive tint = magenta tone = decreases G; R and B untouched."""
    gain = _tint_to_acescg_gain(10.0)
    assert gain[0] == 1.0
    assert gain[1] < 1.0
    assert gain[2] == 1.0


# =============================================================================
# Tone curve
# =============================================================================

def test_tone_curve_endpoints():
    """Profile tone curve starts at 0 and ends at 1."""
    assert _apply_tone_curve(np.array([0.0]))[0] == pytest.approx(0.0, abs=1e-3)
    assert _apply_tone_curve(np.array([1.0]))[0] == pytest.approx(1.0, abs=1e-3)


def test_tone_curve_monotonic_nondecreasing():
    """A tone curve that dips would invert local contrast — never the intent."""
    assert np.all(np.diff(_TONE_CURVE_LUT) >= 0), \
        "tone curve must be monotonically non-decreasing"


def test_tone_curve_output_in_unit_range():
    """All output samples are clipped to [0, 1]."""
    assert _TONE_CURVE_LUT.min() >= 0.0
    assert _TONE_CURVE_LUT.max() <= 1.0
