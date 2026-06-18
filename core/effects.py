"""
Standalone image effect functions.

Each function takes and returns a float32 numpy array. Most stay in [0, 1],
but linear-light effects (apply_bloom with linear=True, apply_halation) can
return values >1 since they run before the display transform.

Render-pipeline ordering (see processor._render):
  vignette → bloom → CNR                      (linear ACEScg, pre-LUT)
  ACEScct encode → LUT                         (display transform)
  CA → edge-softness → softness → grain → sharpen   (display sRGB, post-LUT)

At full res with a LUT active these all run as one GPU-resident chain (one
upload/readback); these numpy functions are the oracle + no-GPU fallback.
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
    each weighted by that band's RGB sensitivity and radially magnified by the
    reciprocal 1/(1+scale*t): red (t=0) ~unshifted, blue (t=1) magnified outward
    by (1+scale). This gives a smooth fringe that grows with radius and runs the
    physically-correct direction (light->dark edge => blue). Runs on the post-LUT
    display-sRGB image
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
        # Reciprocal magnification (matches ca_tex.wgsl and the legacy cv2
        # inverse-matrix warp): blue samples inward -> content outward, so the
        # fringe direction is physically correct (light->dark edge => blue).
        sc = 1.0 / (1.0 + scale * t)
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


def reduce_color_noise_chroma(image, sigma=0.7, despike=(0.0, 0.0)):
    """Chroma noise reduction in CIE Lab space (linear ACEScg in/out).

    Blurs a* and b* with a bilateral filter while leaving L* untouched.
    Bilateral filtering stops at color edges (unlike Gaussian), preventing
    chroma from bleeding across sharp luma boundaries which causes moiré on
    fine periodic patterns like textiles.

    ``despike`` = the (thr_green, thr_other) pair from
    config.cnr_despike_thresholds. When thr_green>0 a 3x3-median outlier clamp
    runs first, pulling isolated colour spikes (fireflies) back toward their
    neighbours — the bilateral is edge-preserving and protects such spikes, so
    this is what actually removes them without washing out colour detail.
    """
    lab = _acescg_to_lab(image)
    thr_green, thr_other = despike
    if thr_green > 0:
        # cv2.medianBlur supports float32 single-channel at ksize 3/5 and uses
        # BORDER_REPLICATE (= the GPU clamp-to-edge). Clamp each chroma channel
        # into [median +/- thr]: green (a* below median) by thr_green, magenta
        # and both b* directions by thr_other.
        med_a = cv2.medianBlur(lab[:, :, 1], 3)
        med_b = cv2.medianBlur(lab[:, :, 2], 3)
        lab[:, :, 1] = np.clip(lab[:, :, 1], med_a - thr_green, med_a + thr_other)
        lab[:, :, 2] = np.clip(lab[:, :, 2], med_b - thr_other, med_b + thr_other)
    if sigma > 0:
        # d must be a positive odd integer; keep it small so the filter stays fast.
        # sigma=0.7→5, sigma=2→7, sigma=4→11
        d = max(5, int(sigma) * 2 + 3)
        if d % 2 == 0:
            d += 1
        # sigmaColor in Lab units (a*/b* range ~±80), scaled with sigma so high
        # settings denoise harder (see config.cnr_sigma_color); floor 15 keeps low
        # settings edge-preserving.
        sigma_color = cnr_sigma_color(sigma)
        lab[:, :, 1] = cv2.bilateralFilter(lab[:, :, 1], d, sigma_color, sigma)
        lab[:, :, 2] = cv2.bilateralFilter(lab[:, :, 2], d, sigma_color, sigma)
    return _lab_to_acescg(lab)


def _halation_glow(img_f, gray, threshold, size, tint, kind, k=20.0):
    """Compute one halation glow scale for a given threshold, size, tint and kind.

    `tint` is (r, g, b) with the scale weight already folded in; it carries the
    scale's chroma so the summed glow reddens outward. `kind` selects the
    kernel: 'disc' = defined circle-of-confusion core, 'exp' = diffuse scatter
    tail. Mirrors halation_highlights.wgsl + gpu.halation_frame.
    """
    gray_log = acescct_encode(gray)
    mask = 1.0 / (1.0 + np.exp(-k * (gray_log - threshold)))
    mask = gaussian_blur(mask, 2.0)
    mask_3d = np.stack([mask, mask, mask], axis=2)
    highlights = img_f * mask_3d * np.asarray(tint, dtype=np.float32)
    return disc_blur(highlights, size) if kind == 'disc' else exp_blur(highlights, size)


def apply_halation(img, threshold=0.65, blur_radius=HALATION_BLUR_RADIUS, strength=0.5,
                   warmth_pct=100.0):
    """
    Three-scale halation: a defined disc core (circle-of-confusion — the film
    back-reflection is a defocused copy of the highlights) plus faint
    exponential scatter tails, each carrying its own chroma so the halo reddens
    outward. Wider scales target only the brightest areas (threshold offset).
    Warmth scales the green/blue falloff around the physical red-orange baseline
    (100% = physical, 0% = colourless, >100% = the saturated no-remjet /
    CineStill halo). See config.HALATION_SCALES — single source of truth shared
    with gpu.halation_frame.
    """
    start_total = time.time()

    img_f = img.astype(np.float32)

    # Resident path: the whole two-pass halation runs on the GPU with one
    # upload/readback instead of ~9 CPU<->GPU round-trips. It blurs the glow at
    # half resolution for speed, so it APPROXIMATES the full-res per-op path
    # below (perceptually identical — the glow is low-frequency; ~sub-code-value
    # end to end), rather than matching bit-for-bit. Falls back on any GPU issue
    # so a bad driver can only slow a render, never break it.
    if HAS_GPU:
        try:
            # Draw halation's ~13 full-res scratch textures from the per-render
            # arena (the same mechanism the live render uses) so repeat image
            # loads reuse them instead of allocating fresh each time — texture
            # allocation is pure CPU-side driver cost, not GPU compute. The
            # readback must happen inside the scope, before end_render releases
            # the arena (mirrors kernels.run_resident).
            gpu.begin_render()
            try:
                res = gpu.halation_frame(Frame.from_cpu(img_f), threshold, blur_radius, strength, warmth_pct)
                out = res.cpu() if res is not None else None   # combine clamps to >= 0
            finally:
                gpu.end_render()
            if out is not None:
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

    glow_combined = np.zeros_like(img_f)
    for radius_mult, thresh_off, weight, gf, bf, kind in HALATION_SCALES:
        tint = halation_scale_tint(gf, bf, weight, warmth_pct)
        glow_combined += _halation_glow(img_f, gray, min(threshold + thresh_off, 0.98),
                                        blur_radius * radius_mult, tint, kind)
    glow_combined *= strength

    # Additive in linear light — halation is added re-exposure, and screen blend
    # breaks on linear HDR values > 1 (see halation_combine.wgsl).
    result = img_f + glow_combined

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
    # clamp the cosine base to >=0 before pow (matches vignette_tex.wgsl; guards
    # against a fractional power of a tiny-negative corner value -> NaN).
    falloff = np.maximum(0.5 * (1.0 + np.cos(np.pi * r_norm)), 0.0).astype(np.float32)
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
    # Long downsampled edge so the glow size is orientation-invariant (matches
    # gpu.bloom_frame; rotation swaps bw/bh but not their max).
    sigma = max(2, max(bw, bh) // 5)
    blurred = gaussian_blur(bloom_src, sigma)
    bloom_layer = cv2.resize(blurred, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    if linear:
        result = image + bloom_layer * strength
        return np.maximum(0.0, result).astype(np.float32)
    else:
        result = 1.0 - (1.0 - image) * (1.0 - bloom_layer * strength)
        return np.clip(result, 0.0, 1.0).astype(np.float32)


