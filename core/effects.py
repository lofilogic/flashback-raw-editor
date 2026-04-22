"""
Standalone image effect functions.

Each function takes and returns a float32 numpy array in [0, 1].
Effects are applied in the order defined in the render pipeline:
  chromatic aberration → ACEScct encode → dither → LUT → softness → grain → sharpen
Halation is baked into the intermediate during load (export path only).
"""
import numpy as np
import cv2
import time

from .kernels import (
    HAS_NUMBA,
    _trilinear_lut_numba,
    _screen_blend_numba,
    _unsharp_mask_numba,
)
from .config import (
    _timing_print,
    CHROMATIC_ABERRATION_STRENGTH, CHROMATIC_ABERRATION_STEPS,
    HALATION_THRESHOLD, HALATION_BLUR_RADIUS, HALATION_STRENGTH,
    SOFTNESS_SIGMA, SHARPEN_STRENGTH, SHARPEN_RADIUS, DITHER_STRENGTH,
)

# =============================================================================
# LUT APPLICATION
# =============================================================================

def apply_lut_fast(image, lut):
    """
    Fast 3D LUT application using Numba JIT or optimized numpy fallback.
    """
    total_start = time.time()

    _timing_print(f"    [LUT DEBUG] Using LUT size: {lut.table.shape}")

    if image.dtype != np.float32:
        image = np.float32(image)
    if not image.flags['C_CONTIGUOUS']:
        image = np.ascontiguousarray(image)

    lut_table = np.ascontiguousarray(lut.table.astype(np.float32))
    lut_size = lut_table.shape[0]

    if HAS_NUMBA:
        result = _trilinear_lut_numba(image, lut_table, lut_size)
        method = "Numba JIT"
    else:
        result = _apply_lut_numpy_vectorized(image, lut_table, lut_size)
        method = "Numpy vectorized"

    total_time = time.time() - total_start
    _timing_print(f"    [LUT] {method}: {total_time*1000:.2f} ms")

    return result


def _apply_lut_numpy_vectorized(image, lut_table, lut_size):
    """
    Optimized numpy fallback when Numba is unavailable.
    """
    img_scaled = np.clip(image, 0, 1) * (lut_size - 1)
    indices = img_scaled.astype(np.int32)
    fractions = img_scaled - indices
    np.clip(indices, 0, lut_size - 2, out=indices)

    r_idx = indices[:, :, 0]
    g_idx = indices[:, :, 1]
    b_idx = indices[:, :, 2]
    r_frac = fractions[:, :, 0, np.newaxis]
    g_frac = fractions[:, :, 1, np.newaxis]
    b_frac = fractions[:, :, 2, np.newaxis]

    c000 = lut_table[r_idx, g_idx, b_idx]
    c001 = lut_table[r_idx, g_idx, b_idx + 1]
    c010 = lut_table[r_idx, g_idx + 1, b_idx]
    c011 = lut_table[r_idx, g_idx + 1, b_idx + 1]
    c100 = lut_table[r_idx + 1, g_idx, b_idx]
    c101 = lut_table[r_idx + 1, g_idx, b_idx + 1]
    c110 = lut_table[r_idx + 1, g_idx + 1, b_idx]
    c111 = lut_table[r_idx + 1, g_idx + 1, b_idx + 1]

    c00 = c000 + (c001 - c000) * b_frac
    c01 = c010 + (c011 - c010) * b_frac
    c10 = c100 + (c101 - c100) * b_frac
    c11 = c110 + (c111 - c110) * b_frac

    c0 = c00 + (c01 - c00) * g_frac
    c1 = c10 + (c11 - c10) * g_frac

    return c0 + (c1 - c0) * r_frac

# =============================================================================
# EFFECT FUNCTIONS
# =============================================================================

def apply_chromatic_aberration(image, strength=CHROMATIC_ABERRATION_STRENGTH, steps=CHROMATIC_ABERRATION_STEPS):
    """
    Applies chromatic aberration in LINEAR Rec.2020 space.
    Called BEFORE ACEScct encoding to prevent log-space banding.
    """
    start_total = time.time()

    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)

    # --- PHASE 1: Color Splitting ---
    r_acc = np.zeros((h, w), dtype=np.float32)
    g_acc = np.zeros((h, w), dtype=np.float32)
    b_acc = np.zeros((h, w), dtype=np.float32)

    for i in range(steps):
        factor = i / max(1, steps - 1) if steps > 1 else 1.0

        scale_r = 1.0 - (strength/2 * factor)
        M_r = cv2.getRotationMatrix2D(center, 0, scale_r)
        r_acc += cv2.warpAffine(image[:, :, 0], M_r, (w, h),
                                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

        scale_g = 1.0 - (strength/8 * factor)
        M_g = cv2.getRotationMatrix2D(center, 0, scale_g)
        g_acc += cv2.warpAffine(image[:, :, 1], M_g, (w, h),
                                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

        scale_b = 1.0 + (strength * factor)
        M_b = cv2.getRotationMatrix2D(center, 0, scale_b)
        b_acc += cv2.warpAffine(image[:, :, 2], M_b, (w, h),
                                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    ca_result = np.empty_like(image, dtype=np.float32)
    ca_result[:, :, 0] = r_acc / steps
    ca_result[:, :, 1] = g_acc / steps
    ca_result[:, :, 2] = b_acc / steps

    # --- PHASE 2: Global Zoom Blur ---
    zoom_acc = np.zeros_like(ca_result, dtype=np.float32)

    for i in range(steps):
        factor = i / max(1, steps - 1) if steps > 1 else 1.0
        scale_z = 1.0 + (strength/2 * factor)
        M_z = cv2.getRotationMatrix2D(center, 0, scale_z)
        zoom_acc += cv2.warpAffine(ca_result, M_z, (w, h),
                                   flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    final_result = zoom_acc / steps

    total_time = time.time() - start_total
    _timing_print(f"    [Chromatic Aberration] Total: {total_time*1000:.2f}ms (linear Rec.2020)")

    return final_result


def reduce_color_noise_chroma(image, sigma=0.7):
    """Blur chroma channels while preserving luma detail."""
    r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]

    # Rec. 2020 coefficients
    kr = 0.2627
    kg = 0.6780
    kb = 0.0593

    luma = kr * r + kg * g + kb * b
    cb = b - luma
    cr = r - luma

    cb_blur = cv2.GaussianBlur(cb, (0, 0), sigmaX=sigma, sigmaY=sigma)
    cr_blur = cv2.GaussianBlur(cr, (0, 0), sigmaX=sigma, sigmaY=sigma)

    r_out = cr_blur + luma
    g_out = luma - (kr / kg) * cr_blur - (kb / kg) * cb_blur
    b_out = cb_blur + luma

    return np.stack([r_out, g_out, b_out], axis=2)


def _halation_glow(img_f, gray, threshold, blur_radius, k=20.0):
    """Compute one halation glow layer for a given threshold and radius."""
    mask = 1.0 / (1.0 + np.exp(-k * (gray - threshold)))
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=2.0, sigmaY=2.0)
    mask_3d = np.stack([mask, mask, mask], axis=2)
    highlights = img_f * mask_3d
    glow = np.zeros_like(highlights)
    glow[:, :, 0] = cv2.GaussianBlur(highlights[:, :, 0] * 1.0, (0, 0), sigmaX=blur_radius)
    glow[:, :, 1] = cv2.GaussianBlur(highlights[:, :, 1] * 0.6, (0, 0), sigmaX=blur_radius)
    glow[:, :, 2] = cv2.GaussianBlur(highlights[:, :, 2] * 0.0, (0, 0), sigmaX=blur_radius)
    return glow


def apply_halation(img, threshold=HALATION_THRESHOLD, blur_radius=HALATION_BLUR_RADIUS, strength=HALATION_STRENGTH):
    """
    Two-pass halation: regular highlights + extreme highlights with 3x radius.
    The second pass targets only the very brightest areas (threshold + 0.15)
    and spreads much wider, simulating the larger glow of intense light sources.
    Both passes use the same parameters so debug sliders control both naturally.
    """
    start_total = time.time()

    img_f = img.astype(np.float32)
    gray = np.max(img_f, axis=2)

    # Pass 1 — regular highlights
    glow1 = _halation_glow(img_f, gray, threshold, blur_radius)

    # Pass 2 — extreme highlights: higher threshold, 3x radius, 60% strength
    threshold2 = min(threshold + 0.15, 0.98)
    glow2 = _halation_glow(img_f, gray, threshold2, blur_radius * 3)

    glow_combined = (glow1 + glow2 * 0.6) * strength

    if HAS_NUMBA:
        result = _screen_blend_numba(img_f, glow_combined)
    else:
        result = 1.0 - (1.0 - img_f) * (1.0 - glow_combined)

    total_time = time.time() - start_total
    _timing_print(f"    [Halation] Total: {total_time*1000:.2f}ms")

    return np.maximum(result, 0)


def apply_softness(image, sigma=SOFTNESS_SIGMA):
    """Subtle Gaussian blur for film-like softness."""
    return cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma)


def apply_sharpen(image, strength=SHARPEN_STRENGTH, radius=SHARPEN_RADIUS):
    """Unsharp mask sharpening."""
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=radius, sigmaY=radius)

    if HAS_NUMBA:
        return _unsharp_mask_numba(image, blurred, strength)
    else:
        return image + (image - blurred) * strength


def add_blue_noise_dither(image, strength=DITHER_STRENGTH):
    """
    Add dithering noise to reduce LUT banding artifacts.
    Uses white noise with a slight blur as a blue noise approximation —
    less visually intrusive than pure white noise.
    """
    h, w = image.shape[:2]
    noise = np.random.normal(0, strength, (h, w, 3)).astype(np.float32)
    noise = cv2.GaussianBlur(noise, (0, 0), sigmaX=0.5)
    return np.clip(image + noise, 0, 1)
