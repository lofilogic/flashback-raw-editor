"""
Standalone image effect functions.

Each function takes and returns a float32 numpy array. Most stay in [0, 1],
but linear-light effects (apply_bloom with linear=True, apply_halation) can
return values >1 since they run before the display transform.

Render-pipeline ordering (see processor._render):
  bloom → vignette       (linear ACEScg)
  CNR → ACEScct → LUT    (display transform)
  CA → softness → grain → sharpen   (display sRGB, post-LUT)

Halation is baked into the cached intermediate at load time (see
processor.load_image) so it benefits both the live preview and export.
"""
import numpy as np
import cv2
import time

from .kernels import (
    apply_lut_gpu,
    apply_lut_cpu,
    screen_blend,
    unsharp_mask,
    gaussian_blur,
    acescct_encode,
)
from .config import (
    _timing_print,
    CHROMATIC_ABERRATION_STRENGTH, CHROMATIC_ABERRATION_STEPS,
    HALATION_THRESHOLD, HALATION_BLUR_RADIUS, HALATION_STRENGTH,
    SOFTNESS_SIGMA, SHARPEN_STRENGTH, SHARPEN_RADIUS,
)

# =============================================================================
# LUT APPLICATION
# =============================================================================

def apply_lut_fast(image, lut):
    """
    Fast 3D LUT application — GPU tetrahedral or CPU trilinear fallback.
    """
    total_start = time.time()

    if image.dtype != np.float32:
        image = image.astype(np.float32)
    if not image.flags['C_CONTIGUOUS']:
        image = np.ascontiguousarray(image)

    result = apply_lut_gpu(image)
    if result is not None:
        method = "GPU tetrahedral"
    else:
        lut_table = np.ascontiguousarray(lut.table.astype(np.float32))
        result = apply_lut_cpu(image, lut_table)
        method = "CPU trilinear"

    _timing_print(f"    [LUT] {method}: {(time.time()-total_start)*1000:.2f} ms")
    return result



# =============================================================================
# EFFECT FUNCTIONS
# =============================================================================

def apply_chromatic_aberration(image, strength=CHROMATIC_ABERRATION_STRENGTH, steps=CHROMATIC_ABERRATION_STEPS, blue_blur=0.0):
    """
    Radial chromatic aberration via per-channel rotation-matrix scaling.

    Runs on the post-LUT display-sRGB image (gamma-encoded), not linear.
    Working in display space keeps the fringing visually localised to bright
    edges the way a real lens behaves; doing it in linear would over-spread
    highlights once the display curve is reapplied.

    blue_blur: optional Gaussian sigma applied to the blue channel of the final result.
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

        scale_r = 1.0
        M_r = cv2.getRotationMatrix2D(center, 0, scale_r)
        r_acc += cv2.warpAffine(image[:, :, 0], M_r, (w, h),
                                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

        scale_g = 1.0 + (strength/2 * factor)
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
        scale_z = 1.0 + (strength/6 * factor)
        M_z = cv2.getRotationMatrix2D(center, 0, scale_z)
        zoom_acc += cv2.warpAffine(ca_result, M_z, (w, h),
                                   flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    final_result = zoom_acc / steps

    if blue_blur > 0:
        final_result[:, :, 2] = gaussian_blur(final_result[:, :, 2], blue_blur)

    total_time = time.time() - start_total
    _timing_print(f"    [Chromatic Aberration] Total: {total_time*1000:.2f}ms (display sRGB)")

    return final_result


# ACEScg (AP1, D60) <-> XYZ_D60 matrices for Lab CNR round-trip.
_ACESCG_TO_XYZ_D60 = np.array([
    [ 0.6624541811,  0.1340042065,  0.1561876744],
    [ 0.2722287168,  0.6740817658,  0.0536895174],
    [-0.0055746495,  0.0040607335,  1.0103391685],
], dtype=np.float32)
_XYZ_D60_TO_ACESCG = np.array([
    [ 1.6410233797, -0.3248032942, -0.2364246952],
    [-0.6636628587,  1.6153315917,  0.0167563477],
    [ 0.0117218943, -0.0082844420,  0.9883948585],
], dtype=np.float32)
_D60_WHITE_XYZ = np.array([0.95265, 1.0, 1.00883], dtype=np.float32)
_LAB_DELTA3 = (6.0 / 29.0) ** 3
_LAB_SLOPE  = (29.0 / 6.0) ** 2 / 3.0


def _f_lab(t):
    return np.where(t > _LAB_DELTA3, np.cbrt(np.maximum(t, 0.0)),
                    _LAB_SLOPE * t + 4.0 / 29.0)


def _f_lab_inv(t):
    delta = 6.0 / 29.0
    return np.where(t > delta, t ** 3, (t - 4.0 / 29.0) / _LAB_SLOPE)


def _acescg_to_lab(img):
    h, w = img.shape[:2]
    xyz = (img.reshape(-1, 3) @ _ACESCG_TO_XYZ_D60.T).reshape(h, w, 3)
    xyz = np.maximum(xyz, 0.0) / _D60_WHITE_XYZ
    fx, fy, fz = _f_lab(xyz[:,:,0]), _f_lab(xyz[:,:,1]), _f_lab(xyz[:,:,2])
    return np.stack([116.0*fy - 16.0, 500.0*(fx - fy), 200.0*(fy - fz)], axis=2)


def _lab_to_acescg(lab):
    L, a, b = lab[:,:,0], lab[:,:,1], lab[:,:,2]
    fy = (L + 16.0) / 116.0
    xyz = np.stack([_f_lab_inv(a/500.0 + fy), _f_lab_inv(fy),
                    _f_lab_inv(fy - b/200.0)], axis=2) * _D60_WHITE_XYZ
    h, w = lab.shape[:2]
    return (xyz.reshape(-1, 3) @ _XYZ_D60_TO_ACESCG.T).reshape(h, w, 3)


def reduce_color_noise_chroma(image, sigma=0.7):
    """Chroma noise reduction in CIE Lab space (linear ACEScg in/out).

    Blurs a* and b* with a bilateral filter while leaving L* untouched.
    Bilateral filtering stops at color edges (unlike Gaussian), preventing
    chroma from bleeding across sharp luma boundaries which causes moiré on
    fine periodic patterns like textiles.
    """
    lab = _acescg_to_lab(image)
    # d must be a positive odd integer; keep it small so the filter stays fast.
    # sigma=0.7→5, sigma=2→7, sigma=4→11
    d = max(5, int(sigma) * 2 + 3)
    if d % 2 == 0:
        d += 1
    # sigmaColor in Lab units (a*/b* range ~±80): smooths noise-level variation
    # (typically < 8 Lab units) while stopping at real colour edges (> 20 units).
    sigma_color = 15.0
    lab[:, :, 1] = cv2.bilateralFilter(lab[:, :, 1], d, sigma_color, sigma)
    lab[:, :, 2] = cv2.bilateralFilter(lab[:, :, 2], d, sigma_color, sigma)
    return _lab_to_acescg(lab)


def _halation_glow(img_f, gray, threshold, blur_radius, k=20.0):
    """Compute one halation glow layer for a given threshold and radius."""
    gray_log = acescct_encode(gray)
    mask = 1.0 / (1.0 + np.exp(-k * (gray_log - threshold)))
    mask = gaussian_blur(mask, 2.0)
    mask_3d = np.stack([mask, mask, mask], axis=2)
    highlights = img_f * mask_3d
    glow = np.zeros_like(highlights)
    glow[:, :, 0] = gaussian_blur(highlights[:, :, 0], blur_radius)
    glow[:, :, 1] = gaussian_blur(highlights[:, :, 1] * 0.2, blur_radius)
    # Blue stays at zero (orange/red glow only, as on real film halation).
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

    result = screen_blend(img_f, glow_combined)

    total_time = time.time() - start_total
    _timing_print(f"    [Halation] Total: {total_time*1000:.2f}ms")

    return np.maximum(result, 0)


def apply_softness(image, sigma=SOFTNESS_SIGMA):
    """Subtle Gaussian blur for film-like softness."""
    return gaussian_blur(image, sigma)


def apply_sharpen(image, strength=SHARPEN_STRENGTH, radius=SHARPEN_RADIUS):
    """Unsharp mask sharpening."""
    blurred = gaussian_blur(image, radius)
    return unsharp_mask(image, blurred, strength)


def apply_vignette(image, strength=0.5, color_shift=0.05, feather=1.0):
    """
    Smooth cosine vignette with a cool-edge tint.

    All channels share the same `dark` falloff; per-channel offsets shift the
    edges cooler — red darkens a bit more than `dark` while blue darkens
    slightly less (so blue can stay near its center value, or even gain a
    touch, relative to red). The net effect is a mild blue cast at the
    periphery rather than a pure neutral darkening.
    feather > 1: sharper falloff (bright center, darkness compressed to edges).
    feather < 1: softer, more gradual falloff.
    """
    if strength <= 0:
        return image
    h, w = image.shape[:2]
    y = np.linspace(-1.0, 1.0, h, dtype=np.float32)
    x = np.linspace(-1.0, 1.0, w, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    radius = np.sqrt(xx ** 2 + yy ** 2)
    r_norm = np.clip(radius / np.sqrt(2.0), 0.0, 1.0)
    falloff = (0.5 * (1.0 + np.cos(np.pi * r_norm))).astype(np.float32)
    if feather != 1.0:
        falloff = np.power(falloff, feather)
    dark = 1.0 - strength * (1.0 - falloff)
    edge = 1.0 - falloff  # 0 at center, 1 at corners
    result = np.empty_like(image)
    result[:, :, 0] = np.maximum(0.0, image[:, :, 0] * (dark - color_shift * edge))
    result[:, :, 1] = np.maximum(0.0, image[:, :, 1] * dark)
    result[:, :, 2] = np.maximum(0.0, image[:, :, 2] * (dark + color_shift * 0.4 * edge))
    return result


def apply_bloom(image, strength=0.3, threshold=0.6, linear=False):
    """
    Fast large-radius bloom via 4x downsample → heavy Gaussian → upsample → blend.
    threshold: ACEScct-space highlight cutoff (0–1); ~0.555 = scene white, 0.65 = overbright only.
    linear: when True, uses additive blend (correct for HDR linear-light space);
            when False, uses screen blend (correct for display/LUT-output [0,1] space).
    Luma, ACEScct encoding, masking, and blur all happen at 1/4 resolution.
    """
    if strength <= 0:
        return image
    h, w = image.shape[:2]
    scale = 4
    bh, bw = max(4, h // scale), max(4, w // scale)
    small = cv2.resize(image, (bw, bh), interpolation=cv2.INTER_AREA).astype(np.float32)
    luma_small = (0.2126 * small[:, :, 0] + 0.7152 * small[:, :, 1] + 0.0722 * small[:, :, 2])
    luma_log = acescct_encode(luma_small)
    soft_mask = np.clip((luma_log - threshold) / max(0.01, 1.0 - threshold), 0.0, 1.0)
    bloom_src = small * soft_mask[:, :, np.newaxis]
    sigma = max(2, bw // 5)
    blurred = gaussian_blur(bloom_src, sigma)
    bloom_layer = cv2.resize(blurred, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    if linear:
        result = image + bloom_layer * strength
        return np.maximum(0.0, result).astype(np.float32)
    else:
        result = 1.0 - (1.0 - image) * (1.0 - bloom_layer * strength)
        return np.clip(result, 0.0, 1.0).astype(np.float32)


