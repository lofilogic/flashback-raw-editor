"""
Tests for image-effect functions.

These don't check that the result *looks* right — they check shape, dtype,
and the explicit "strength=0 means no-op" contracts that the render
pipeline relies on for short-circuiting.
"""
import numpy as np
import pytest

from core.effects import (
    apply_chromatic_aberration, apply_softness, apply_sharpen,
    apply_vignette, apply_bloom,
)


@pytest.fixture
def neutral_gray():
    """64x64 neutral grey float32 image."""
    return np.full((64, 64, 3), 0.5, dtype=np.float32)


# =============================================================================
# No-op contracts
# =============================================================================

def test_vignette_strength_zero_is_noop(neutral_gray):
    out = apply_vignette(neutral_gray, strength=0.0)
    assert np.array_equal(out, neutral_gray)


def test_bloom_strength_zero_is_noop(neutral_gray):
    out = apply_bloom(neutral_gray, strength=0.0)
    assert np.array_equal(out, neutral_gray)


# =============================================================================
# Shape / dtype preservation
# =============================================================================

@pytest.mark.parametrize("fn,args", [
    (apply_chromatic_aberration, (0.005,)),
    (apply_softness, (0.5,)),
    (apply_sharpen, (0.5, 1.0)),
    (apply_vignette, (0.5, 0.05, 1.0)),
    (apply_bloom, (0.3, 0.65)),
])
def test_effect_preserves_shape_and_dtype(neutral_gray, fn, args):
    out = fn(neutral_gray, *args)
    assert out.shape == neutral_gray.shape, f"{fn.__name__} changed shape"
    assert out.dtype == np.float32, f"{fn.__name__} changed dtype"


# =============================================================================
# Output range
# =============================================================================

def test_vignette_never_brightens_pixels(neutral_gray):
    """A vignette darkens edges; no output pixel should exceed input value."""
    out = apply_vignette(neutral_gray, strength=0.5, color_shift=0.0,
                          feather=1.0)
    assert (out <= neutral_gray + 1e-6).all()


def test_vignette_center_unchanged(neutral_gray):
    """The exact image center should be ~untouched by the vignette."""
    out = apply_vignette(neutral_gray, strength=0.5, color_shift=0.05,
                          feather=1.0)
    h, w = neutral_gray.shape[:2]
    center_in = neutral_gray[h // 2, w // 2]
    center_out = out[h // 2, w // 2]
    assert np.allclose(center_out, center_in, atol=1e-3)
