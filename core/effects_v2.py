"""
processor_v2 effects layer.

Most v1 effect implementations are space-agnostic (float array in/out,
no hardcoded primary assumptions), so we re-export them here rather
than duplicating code. The exception is reduce_color_noise_chroma_v2,
which performs chroma noise reduction in CIE Lab space for correct
luma/chroma separation without edge fringing.
"""
import numpy as np

from .effects import (  # noqa: F401 -- re-exported as v2 surface
    apply_lut_fast,
    apply_chromatic_aberration,
    apply_halation,
    apply_softness,
    apply_sharpen,
    apply_vignette,
    apply_bloom,
    add_blue_noise_dither,
)
from .kernels import gaussian_blur

# ACEScg (AP1, D60) -> XYZ_D60. Standard ACES AP1 primaries (inverse of the
# XYZ_D60_TO_ACESCG matrix in processor_v2.py; defined here to avoid a
# circular import).
_ACESCG_TO_XYZ_D60 = np.array([
    [ 0.6624541811,  0.1340042065,  0.1561876744],
    [ 0.2722287168,  0.6740817658,  0.0536895174],
    [-0.0055746495,  0.0040607335,  1.0103391685],
], dtype=np.float32)

# XYZ_D60 -> ACEScg (round-trip inverse).
_XYZ_D60_TO_ACESCG = np.array([
    [ 1.6410233797, -0.3248032942, -0.2364246952],
    [-0.6636628587,  1.6153315917,  0.0167563477],
    [ 0.0117218943, -0.0082844420,  0.9883948585],
], dtype=np.float32)

# ACES D60 white in XYZ (chromaticity x=0.32168, y=0.33767).
_D60_WHITE_XYZ = np.array([0.95265, 1.0, 1.00883], dtype=np.float32)

_LAB_DELTA3 = (6.0 / 29.0) ** 3   # ≈ 0.008856
_LAB_SLOPE  = (29.0 / 6.0) ** 2 / 3.0  # ≈ 7.787


def _f_lab(t):
    return np.where(t > _LAB_DELTA3,
                    np.cbrt(np.maximum(t, 0.0)),
                    _LAB_SLOPE * t + 4.0 / 29.0)


def _f_lab_inv(t):
    delta = 6.0 / 29.0
    return np.where(t > delta,
                    t ** 3,
                    (t - 4.0 / 29.0) / _LAB_SLOPE)


def _acescg_to_lab(img):
    """Linear ACEScg -> CIE Lab (D60 white)."""
    h, w = img.shape[:2]
    xyz = (img.reshape(-1, 3) @ _ACESCG_TO_XYZ_D60.T).reshape(h, w, 3)
    xyz = np.maximum(xyz, 0.0) / _D60_WHITE_XYZ
    fx, fy, fz = _f_lab(xyz[:, :, 0]), _f_lab(xyz[:, :, 1]), _f_lab(xyz[:, :, 2])
    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)
    return np.stack([L, a, b], axis=2)


def _lab_to_acescg(lab):
    """CIE Lab (D60 white) -> linear ACEScg."""
    L, a, b = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]
    fy = (L + 16.0) / 116.0
    fx = a / 500.0 + fy
    fz = fy - b / 200.0
    xyz = np.stack([_f_lab_inv(fx), _f_lab_inv(fy), _f_lab_inv(fz)], axis=2)
    xyz *= _D60_WHITE_XYZ
    h, w = lab.shape[:2]
    return (xyz.reshape(-1, 3) @ _XYZ_D60_TO_ACESCG.T).reshape(h, w, 3)


def reduce_color_noise_chroma_v2(image, sigma=0.7):
    """Chroma noise reduction in CIE Lab space.

    Converts linear ACEScg -> Lab, Gaussian-blurs a* and b* while leaving
    L* untouched, then converts back. Lab's perceptual luma/chroma separation
    avoids both the colour-shift that comes from using linear weights in log
    space and the highlight fringing that comes from blurring in linear light.
    """
    lab = _acescg_to_lab(image)
    lab[:, :, 1] = gaussian_blur(lab[:, :, 1], sigma)  # a*
    lab[:, :, 2] = gaussian_blur(lab[:, :, 2], sigma)  # b*
    return _lab_to_acescg(lab)
