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
import logging

log = logging.getLogger(__name__)
_resident_halation_warned = False

from .kernels import (
    apply_lut_gpu,
    apply_lut_cpu,
    screen_blend,
    unsharp_mask,
    gaussian_blur,
    acescct_encode,
)
from .gpu import gpu, HAS_GPU, Frame
from .config import (
    _timing_print,
    HALATION_BLUR_RADIUS,
    SOFTNESS_SIGMA, SHARPEN_RADIUS,
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

CA_SPECTRAL_SAMPLES = 16


def _ca_band_weights(samples: int) -> np.ndarray:
    """Per-channel spectral sensitivity for each spectral sample t in [0, 1].

    Smooth Gaussian bands centred at t=0 (red), 0.5 (green), 1 (blue); the
    caller normalises per channel so a neutral input stays neutral. Matches the
    band() helper in ca_tex.wgsl. Returns an (samples, 3) float32 array.
    """
    t = (np.linspace(0.0, 1.0, samples, dtype=np.float32) if samples > 1
         else np.zeros(1, dtype=np.float32))
    s2 = 2.0 * 0.25 * 0.25
    return np.stack([
        np.exp(-(t - 0.0) ** 2 / s2),
        np.exp(-(t - 0.5) ** 2 / s2),
        np.exp(-(t - 1.0) ** 2 / s2),
    ], axis=1).astype(np.float32)


def _bilinear_sample_edge(image, map_x, map_y):
    """Bilinear sample with clamp-to-edge, matching ca_tex.wgsl's sample_edge.

    True float bilinear (numpy), unlike cv2.remap which quantises the sub-pixel
    fraction to 1/32 px and so drifts from the GPU at hard edges. Coords are in
    pixel space; integer coords are texel centres.
    """
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
    """Spectral chromatic aberration — numpy oracle / no-GPU fallback for
    gpu.ca_frame (the resident path the render pipeline normally takes).

    Models lateral CA by integrating ``samples`` points across the spectrum,
    each radially magnified from 1.0 (red, ~unshifted) to 1.0 + ``scale`` (blue)
    and weighted by that band's RGB sensitivity, giving a smooth purple->green
    fringe that grows with radius. Runs on the post-LUT display-sRGB image
    (gamma-encoded), not linear, so the fringing stays localised to bright edges
    the way real glass behaves. ``scale`` is ca_pixels_to_scale(ca_pixels, w).
    """
    if scale <= 0:
        return image.astype(np.float32, copy=True)
    start_total = time.time()

    h, w = image.shape[:2]
    cx, cy = w * 0.5, h * 0.5          # matches the GPU centre / cv2 convention
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    dx, dy = xs - cx, ys - cy

    weights = _ca_band_weights(samples)
    ts = (np.linspace(0.0, 1.0, samples, dtype=np.float32) if samples > 1
          else np.zeros(1, dtype=np.float32))

    acc = np.zeros((h, w, 3), dtype=np.float32)
    for t, wband in zip(ts, weights):
        sc = 1.0 + scale * t
        acc += _bilinear_sample_edge(image, cx + dx * sc, cy + dy * sc) * wband
    result = (acc / weights.sum(axis=0)).astype(np.float32)

    total_time = time.time() - start_total
    _timing_print(f"    [Chromatic Aberration] Total: {total_time*1000:.2f}ms (display sRGB)")
    return result


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


def apply_halation(img, threshold=0.65, blur_radius=HALATION_BLUR_RADIUS, strength=0.5):
    """
    Two-pass halation: regular highlights + extreme highlights with 3x radius.
    The second pass targets only the very brightest areas (threshold + 0.15)
    and spreads much wider, simulating the larger glow of intense light sources.
    Both passes use the same parameters so debug sliders control both naturally.
    """
    start_total = time.time()

    img_f = img.astype(np.float32)

    # Resident path: the whole two-pass halation runs on the GPU with one
    # upload/readback instead of ~9 CPU<->GPU round-trips. Bit-identical to the
    # per-op path below (validated max abs diff ~1e-7). Falls back on any GPU
    # issue so a bad driver can only slow a render, never break it.
    if HAS_GPU:
        try:
            res = gpu.halation_frame(Frame.from_cpu(img_f), threshold, blur_radius, strength)
            if res is not None:
                out = res.cpu()   # combine shader already clamps to >= 0
                _timing_print(f"    [Halation] Total (resident): {(time.time()-start_total)*1000:.2f}ms")
                return out
        except Exception:
            # Bit-exact per-op fallback below — a GPU/driver issue can only slow
            # a render, never break it. Log the cause once so backend-specific
            # failures (e.g. Vulkan vs Metal) are diagnosable instead of silent.
            global _resident_halation_warned
            if not _resident_halation_warned:
                _resident_halation_warned = True
                log.warning("⚠ resident halation failed; using per-op fallback", exc_info=True)

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


def apply_edge_softness(image, sigma, strength, start):
    """Radial edge (corner) softness — numpy oracle / no-GPU fallback for
    gpu.edge_softness_frame.

    Emulates lens field curvature: a sharp centre that softens toward the
    corners. Blends the image with a Gaussian-blurred copy by a weight that
    grows from ``start`` (fraction of the corner radius) to 1.0 at the corners,
    scaled by ``strength`` (0..1). Matches edge_softness_tex.wgsl.
    """
    if strength <= 0 or sigma <= 0:
        return image.astype(np.float32, copy=True)
    h, w = image.shape[:2]
    blurred = gaussian_blur(image, sigma)
    cx, cy = w * 0.5, h * 0.5
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    r_norm = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2) / max(np.hypot(cx, cy), 1e-6)
    t = np.clip((r_norm - start) / max(1.0 - start, 1e-6), 0.0, 1.0)
    w_blend = (t * t * (3.0 - 2.0 * t) * strength)[..., None]   # smoothstep * strength
    return (image * (1.0 - w_blend) + blurred * w_blend).astype(np.float32)


def apply_sharpen(image, strength=0.5, radius=SHARPEN_RADIUS):
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
    linear: when True, uses additive blend (correct for HDR linear-light space —
            this is the path the render pipeline calls with ACEScg input);
            when False, uses screen blend (correct for display/LUT-output [0,1] space).
    Luma, ACEScct encoding, masking, and blur all happen at 1/4 resolution.
    """
    if strength <= 0:
        return image
    h, w = image.shape[:2]
    scale = 4
    bh, bw = max(4, h // scale), max(4, w // scale)
    small = cv2.resize(image, (bw, bh), interpolation=cv2.INTER_AREA).astype(np.float32)
    # Luminance weights: ACEScg (AP1, D60) when called on linear scene-referred
    # input, Rec.709 otherwise. The render pipeline takes the linear branch, so
    # the AP1 weights are what matter in practice.
    if linear:
        luma_small = (0.2722 * small[:, :, 0] + 0.6741 * small[:, :, 1] + 0.0537 * small[:, :, 2])
    else:
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


