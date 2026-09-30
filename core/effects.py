"""
Effects on float32 numpy arrays.

These are the CPU fallbacks for the GPU stages in gpu.py, and the reference
the parity tests compare against. Order in processor._render:

  vignette → bloom → CNR                           linear ACEScg
  ACEScct encode → LUT
  CA → edge softness → softness → grain → sharpen  display sRGB

Halation is applied once at load, not per render.
"""
import numpy as np
import cv2
import time
import logging

log = logging.getLogger(__name__)
_resident_halation_warned = False

from .kernels import (
    apply_lut_gpu,
    apply_lut_cpu,
    unsharp_mask,
    gaussian_blur,
    exp_blur,
    disc_blur,
    acescct_encode,
)
from .gpu import gpu, HAS_GPU, Frame
from .config import (
    _timing_print,
    HALATION_BLUR_RADIUS,
    HALATION_SCALES, halation_scale_tint,
    SOFTNESS_SIGMA, SHARPEN_RADIUS,
    cnr_sigma_color,
)

# =============================================================================
# LUT APPLICATION
# =============================================================================

def apply_lut_fast(image, lut):
    """3D LUT: tetrahedral on the GPU, trilinear on the CPU."""
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

CA_SPECTRAL_SAMPLES = 16


def _ca_band_weights(samples: int) -> np.ndarray:
    """(samples, 3) RGB weights for t in [0, 1]: Gaussian bands centred on
    red (0), green (0.5) and blue (1). Matches band() in ca_tex.wgsl."""
    t = (np.linspace(0.0, 1.0, samples, dtype=np.float32) if samples > 1
         else np.zeros(1, dtype=np.float32))
    s2 = 2.0 * 0.25 * 0.25
    return np.stack([
        np.exp(-(t - 0.0) ** 2 / s2),
        np.exp(-(t - 0.5) ** 2 / s2),
        np.exp(-(t - 1.0) ** 2 / s2),
    ], axis=1).astype(np.float32)


def _bilinear_sample_edge(image, map_x, map_y):
    """Bilinear sample, clamp-to-edge. Not cv2.remap: it rounds to 1/32 px
    and then doesn't match the shader at hard edges."""
    h, w = image.shape[:2]
    x = np.clip(map_x, 0.0, w - 1.0)
    y = np.clip(map_y, 0.0, h - 1.0)
    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    fx = (x - x0)[..., None]
    fy = (y - y0)[..., None]
    c00 = image[y0, x0]; c10 = image[y0, x1]
    c01 = image[y1, x0]; c11 = image[y1, x1]
    cx0 = c00 + (c10 - c00) * fx
    cx1 = c01 + (c11 - c01) * fx
    return cx0 + (cx1 - cx0) * fy


def apply_chromatic_aberration(image, scale, samples=CA_SPECTRAL_SAMPLES):
    """Lateral CA summed over ``samples`` wavelengths, red unshifted, blue
    magnified by (1 + scale).

    Runs on the display-encoded image; in linear light the fringes spread
    too far from bright edges.
    """
    if scale <= 0:
        return image.astype(np.float32, copy=True)
    start_total = time.time()

    h, w = image.shape[:2]
    cx, cy = w * 0.5, h * 0.5
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    dx, dy = xs - cx, ys - cy

    weights = _ca_band_weights(samples)
    ts = (np.linspace(0.0, 1.0, samples, dtype=np.float32) if samples > 1
          else np.zeros(1, dtype=np.float32))

    acc = np.zeros((h, w, 3), dtype=np.float32)
    for t, wband in zip(ts, weights):
        # Sampling inward moves content outward.
        sc = 1.0 / (1.0 + scale * t)
        acc += _bilinear_sample_edge(image, cx + dx * sc, cy + dy * sc) * wband
    result = (acc / weights.sum(axis=0)).astype(np.float32)

    total_time = time.time() - start_total
    _timing_print(f"    [Chromatic Aberration] Total: {total_time*1000:.2f}ms (display sRGB)")
    return result


# ACEScg <-> XYZ_D60, for CNR in Lab.
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


def reduce_color_noise_chroma(image, sigma=0.7, despike=(0.0, 0.0)):
    """Chroma NR: bilateral on a*/b* in Lab, L* untouched. ACEScg in and out.

    A Gaussian would bleed colour across edges. ``despike`` comes from
    config.cnr_despike_thresholds.
    """
    lab = _acescg_to_lab(image)
    thr_green, thr_other = despike
    if thr_green > 0:
        # medianBlur replicates borders, like the shader's clamp-to-edge.
        med_a = cv2.medianBlur(lab[:, :, 1], 3)
        med_b = cv2.medianBlur(lab[:, :, 2], 3)
        lab[:, :, 1] = np.clip(lab[:, :, 1], med_a - thr_green, med_a + thr_other)
        lab[:, :, 2] = np.clip(lab[:, :, 2], med_b - thr_other, med_b + thr_other)
    if sigma > 0:
        # Odd window: sigma 0.7→5, 2→7, 4→11
        d = max(5, int(sigma) * 2 + 3)
        if d % 2 == 0:
            d += 1
        sigma_color = cnr_sigma_color(sigma)
        lab[:, :, 1] = cv2.bilateralFilter(lab[:, :, 1], d, sigma_color, sigma)
        lab[:, :, 2] = cv2.bilateralFilter(lab[:, :, 2], d, sigma_color, sigma)
    return _lab_to_acescg(lab)


def _halation_glow(img_f, gray, threshold, size, tint, kind, k=20.0):
    """One halation scale. See config.HALATION_SCALES."""
    gray_log = acescct_encode(gray)
    mask = 1.0 / (1.0 + np.exp(-k * (gray_log - threshold)))
    mask = gaussian_blur(mask, 2.0)
    mask_3d = np.stack([mask, mask, mask], axis=2)
    highlights = img_f * mask_3d * np.asarray(tint, dtype=np.float32)
    return disc_blur(highlights, size) if kind == 'disc' else exp_blur(highlights, size)


def apply_halation(img, threshold=0.65, blur_radius=HALATION_BLUR_RADIUS, strength=0.5,
                   warmth_pct=100.0):
    """Three-scale halation, see config.HALATION_SCALES."""
    start_total = time.time()

    img_f = img.astype(np.float32)

    # The GPU version blurs at half resolution, so it only approximates the
    # CPU path below (within a code value; the glow is low-frequency).
    if HAS_GPU:
        try:
            # Use the arena so repeated loads reuse textures. Read back before
            # end_render.
            gpu.begin_render()
            try:
                res = gpu.halation_frame(Frame.from_cpu(img_f), threshold, blur_radius, strength, warmth_pct)
                out = res.cpu() if res is not None else None
            finally:
                gpu.end_render()
            if out is not None:
                _timing_print(f"    [Halation] Total (resident): {(time.time()-start_total)*1000:.2f}ms")
                return out
        except Exception:
            # Fall back to the CPU; log once.
            global _resident_halation_warned
            if not _resident_halation_warned:
                _resident_halation_warned = True
                log.warning("⚠ resident halation failed; using per-op fallback", exc_info=True)

    gray = np.max(img_f, axis=2)

    glow_combined = np.zeros_like(img_f)
    for radius_mult, thresh_off, weight, gf, bf, kind in HALATION_SCALES:
        tint = halation_scale_tint(gf, bf, weight, warmth_pct)
        glow_combined += _halation_glow(img_f, gray, min(threshold + thresh_off, 0.98),
                                        blur_radius * radius_mult, tint, kind)
    glow_combined *= strength

    # Additive: screen blend breaks on linear values > 1.
    result = img_f + glow_combined

    total_time = time.time() - start_total
    _timing_print(f"    [Halation] Total: {total_time*1000:.2f}ms")

    return np.maximum(result, 0)


def apply_softness(image, sigma=SOFTNESS_SIGMA):
    """Subtle Gaussian blur for film-like softness."""
    return gaussian_blur(image, sigma)


def apply_edge_softness(image, sigma, strength, start):
    """Blend toward a blurred copy from ``start`` (fraction of the corner
    radius) to the corners, scaled by ``strength``."""
    if strength <= 0 or sigma <= 0:
        return image.astype(np.float32, copy=True)
    h, w = image.shape[:2]
    blurred = gaussian_blur(image, sigma)
    cx, cy = w * 0.5, h * 0.5
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    r_norm = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2) / max(np.hypot(cx, cy), 1e-6)
    t = np.clip((r_norm - start) / max(1.0 - start, 1e-6), 0.0, 1.0)
    w_blend = (t * t * (3.0 - 2.0 * t) * strength)[..., None]
    return (image * (1.0 - w_blend) + blurred * w_blend).astype(np.float32)


def apply_sharpen(image, strength=0.5, radius=SHARPEN_RADIUS):
    """Unsharp mask sharpening."""
    blurred = gaussian_blur(image, radius)
    return unsharp_mask(image, blurred, strength)


def apply_vignette(image, strength=0.5, color_shift=0.05, feather=1.0):
    """Cosine vignette. Red darkens a bit more and blue a bit less, so the
    edges go slightly cool. feather > 1 is harder, < 1 softer."""
    if strength <= 0:
        return image
    h, w = image.shape[:2]
    y = np.linspace(-1.0, 1.0, h, dtype=np.float32)
    x = np.linspace(-1.0, 1.0, w, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    radius = np.sqrt(xx ** 2 + yy ** 2)
    r_norm = np.clip(radius / np.sqrt(2.0), 0.0, 1.0)
    # Clamp before pow: a tiny negative value in the corner gives NaN.
    falloff = np.maximum(0.5 * (1.0 + np.cos(np.pi * r_norm)), 0.0).astype(np.float32)
    if feather != 1.0:
        falloff = np.power(falloff, feather)
    dark = 1.0 - strength * (1.0 - falloff)
    edge = 1.0 - falloff
    result = np.empty_like(image)
    result[:, :, 0] = np.maximum(0.0, image[:, :, 0] * (dark - color_shift * edge))
    result[:, :, 1] = np.maximum(0.0, image[:, :, 1] * dark)
    result[:, :, 2] = np.maximum(0.0, image[:, :, 2] * (dark + color_shift * 0.4 * edge))
    return result


def apply_bloom(image, strength=0.3, threshold=0.6, linear=False):
    """Bloom at 1/4 resolution: mask, blur, upsample, blend.

    threshold is in ACEScct (~0.555 is scene white). linear=True blends
    additively for linear input, which is what the render uses; otherwise it
    screen-blends.
    """
    if strength <= 0:
        return image
    h, w = image.shape[:2]
    scale = 4
    bh, bw = max(4, h // scale), max(4, w // scale)
    small = cv2.resize(image, (bw, bh), interpolation=cv2.INTER_AREA).astype(np.float32)
    if linear:
        luma_small = (0.2722 * small[:, :, 0] + 0.6741 * small[:, :, 1] + 0.0537 * small[:, :, 2])
    else:
        luma_small = (0.2126 * small[:, :, 0] + 0.7152 * small[:, :, 1] + 0.0722 * small[:, :, 2])
    luma_log = acescct_encode(luma_small)
    soft_mask = np.clip((luma_log - threshold) / max(0.01, 1.0 - threshold), 0.0, 1.0)
    bloom_src = small * soft_mask[:, :, np.newaxis]
    # Long edge, so rotation doesn't change the glow.
    sigma = max(2, max(bw, bh) // 5)
    blurred = gaussian_blur(bloom_src, sigma)
    bloom_layer = cv2.resize(blurred, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    if linear:
        result = image + bloom_layer * strength
        return np.maximum(0.0, result).astype(np.float32)
    else:
        result = 1.0 - (1.0 - image) * (1.0 - bloom_layer * strength)
        return np.clip(result, 0.0, 1.0).astype(np.float32)


