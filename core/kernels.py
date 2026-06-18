"""
GPU-accelerated kernels for performance-critical image operations.

Each function tries the GPU path first (via wgpu), then falls back to a
numpy implementation if GPU is unavailable. The fallbacks are mathematically
identical — they exist only for CI and systems without a usable GPU.

HAS_GPU reflects whether wgpu loaded. The GPU device itself is lazy-initialized
on first use (no startup cost, no pre-compilation delay).
"""
from __future__ import annotations
import numpy as np
import cv2

from .gpu import gpu, HAS_GPU, Frame


def run_resident(img: np.ndarray, stages) -> np.ndarray | None:
    """Compose GPU-resident ``Frame -> Frame`` stages with a single upload and a
    single readback.

    ``stages`` is a sequence of callables taking and returning a Frame; the
    image is uploaded once, each stage runs on the GPU without touching numpy,
    and the result is read back once at the end. Returns None — so the caller
    falls back to the CPU path — if the GPU is unavailable or any stage opts out
    (returns None). Any GPU error is swallowed into a None fallback so a bad
    driver can never break a render, only slow it.
    """
    if not HAS_GPU:
        return None
    # A render scope draws each stage's textures/uniforms from a reuse arena
    # instead of allocating fresh per frame. The readback (frame.cpu()) happens
    # inside the scope; end_render only flips the arena off afterwards, so the
    # final texture is still valid when it's read back.
    gpu.begin_render()
    try:
        frame = Frame.from_cpu(img)
        for stage in stages:
            frame = stage(frame)
            if frame is None:
                return None
        return frame.cpu()
    except Exception:
        return None
    finally:
        gpu.end_render()


# Below this the numpy BLAS matmul is as fast as the GPU path once the
# upload+readback round-trip is counted (measured break-even at ~3 MP on M3, and
# transfers cost more on discrete GPUs), so small/common frames skip the GPU to
# avoid any chance of a load regression. The GPU only wins once the matmul itself
# is large — i.e. high-megapixel (generic) raws.
_GPU_MATMUL_MIN_PIXELS = 8_000_000


def color_transform(img: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Per-pixel 3x3 colour-space transform, GPU or numpy. Equivalent to
    ``(img.reshape(-1,3) @ M.T).reshape(img.shape)``; used for the load-time
    raw -> ACEScg matmul. The GPU path is taken only for large frames (see
    _GPU_MATMUL_MIN_PIXELS), where it beats numpy despite the transfer.
    """
    if HAS_GPU and img.size // 3 >= _GPU_MATMUL_MIN_PIXELS:
        result = gpu.color_transform(img, M)
        if result is not None:
            return result
    return (img.reshape(-1, 3) @ M.T).reshape(img.shape).astype(np.float32)


def encode_then_lut(img: np.ndarray) -> np.ndarray | None:
    """ACEScct-encode then apply the LUT as one resident chain (one upload, one
    readback). Returns None if unavailable so the caller can use the CPU path."""
    return run_resident(img, [gpu.encode_frame, gpu.lut_frame])

# Rotation uses OpenCV — fast, correct, and rotation happens rarely
def rotate_90_clockwise(img: np.ndarray) -> np.ndarray:
    return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

def rotate_90_counterclockwise(img: np.ndarray) -> np.ndarray:
    return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

# =============================================================================
# ACEScct encode / decode
# =============================================================================

def acescct_decode(img: np.ndarray) -> np.ndarray:
    """ACEScct log → linear ACEScg (AP1, D60)."""
    if HAS_GPU:
        result = gpu.acescct_decode(img)
        if result is not None:
            return result
    # CPU fallback — both branches computed everywhere, np.where selects
    flat = img.ravel().astype(np.float32)
    out = np.where(
        flat < 0.155251141552511,
        (flat - 0.0729055341958355) / 10.5402377416545,
        (2.0 ** (flat * 17.52 - 9.72)),
    ).astype(np.float32)
    return out.reshape(img.shape)

def acescct_encode(img: np.ndarray) -> np.ndarray:
    """Linear ACEScg (AP1, D60) → ACEScct log."""
    if HAS_GPU:
        result = gpu.acescct_encode(img)
        if result is not None:
            return result
    # CPU fallback
    flat = np.maximum(img.ravel(), 1e-10).astype(np.float32)
    out = np.where(
        flat <= 0.0078125,
        10.5402377416545 * flat + 0.0729055341958355,
        (np.log2(flat) + 9.72) / 17.52,
    ).astype(np.float32)
    return out.reshape(img.shape)

# =============================================================================
# LUT application
# =============================================================================

def apply_lut_gpu(img: np.ndarray) -> np.ndarray | None:
    """Apply the LUT that's already uploaded to the GPU. Returns None if GPU unavailable."""
    if not HAS_GPU:
        return None
    return gpu.apply_lut(img)

def apply_lut_cpu(img: np.ndarray, lut_table: np.ndarray) -> np.ndarray:
    """Trilinear LUT fallback (numpy vectorized)."""
    lut_size = lut_table.shape[0]
    img_scaled = np.clip(img, 0, 1) * (lut_size - 1)
    idx = img_scaled.astype(np.int32)
    frac = img_scaled - idx
    np.clip(idx, 0, lut_size - 2, out=idx)

    r0 = idx[:, :, 0]; g0 = idx[:, :, 1]; b0 = idx[:, :, 2]
    rf = frac[:, :, 0, np.newaxis]
    gf = frac[:, :, 1, np.newaxis]
    bf = frac[:, :, 2, np.newaxis]

    c000 = lut_table[r0,     g0,     b0    ]
    c001 = lut_table[r0,     g0,     b0 + 1]
    c010 = lut_table[r0,     g0 + 1, b0    ]
    c011 = lut_table[r0,     g0 + 1, b0 + 1]
    c100 = lut_table[r0 + 1, g0,     b0    ]
    c101 = lut_table[r0 + 1, g0,     b0 + 1]
    c110 = lut_table[r0 + 1, g0 + 1, b0    ]
    c111 = lut_table[r0 + 1, g0 + 1, b0 + 1]

    c00 = c000 + (c001 - c000) * bf
    c01 = c010 + (c011 - c010) * bf
    c10 = c100 + (c101 - c100) * bf
    c11 = c110 + (c111 - c110) * bf
    c0  = c00  + (c01  - c00 ) * gf
    c1  = c10  + (c11  - c10 ) * gf
    return (c0 + (c1 - c0) * rf).astype(np.float32)

# =============================================================================
# Film grain blend
# =============================================================================

def apply_grain(image: np.ndarray, grain_layer: np.ndarray,
                intensity: float = 1.0, min_grain: float = 0.2,
                highlight_bias: float = 0.0) -> np.ndarray:
    """Grain blend with luma-based highlight bias."""
    if HAS_GPU:
        result = gpu.grain_blend(image, grain_layer, intensity, min_grain, highlight_bias)
        if result is not None:
            return result
    # CPU fallback
    grain_delta = (2.0 * grain_layer - 1.0) * intensity
    weight  = (1.0 - highlight_bias) * (1.0 - image) + highlight_bias * image
    falloff = min_grain + weight * (1.0 - min_grain)
    return np.clip(image + grain_delta * falloff, 0.0, 1.0).astype(np.float32)

# =============================================================================
# Screen blend (halation)
# =============================================================================

def screen_blend(base: np.ndarray, blend: np.ndarray) -> np.ndarray:
    """Screen blend: 1 - (1-base)*(1-blend)."""
    if HAS_GPU:
        result = gpu.screen_blend(base, blend)
        if result is not None:
            return result
    return (1.0 - (1.0 - base) * (1.0 - blend)).astype(np.float32)

# =============================================================================
# Unsharp mask
# =============================================================================

def unsharp_mask(image: np.ndarray, blurred: np.ndarray, strength: float) -> np.ndarray:
    """Unsharp mask: image + (image - blurred) * strength."""
    if HAS_GPU:
        result = gpu.unsharp_mask(image, blurred, strength)
        if result is not None:
            return result
    return (image + (image - blurred) * strength).astype(np.float32)

# =============================================================================
# Gaussian blur
# =============================================================================

def gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    """Separable Gaussian blur — GPU or cv2 fallback.

    Accepts (H, W) single-channel or (H, W, C) multi-channel float32 arrays.
    Matches cv2.GaussianBlur(img, (0,0), sigmaX=sigma) semantics.
    """
    if sigma <= 0:
        return img.copy()
    if HAS_GPU:
        result = gpu.gaussian_blur(img, sigma)
        if result is not None:
            return result
    # cv2 fallback
    return cv2.GaussianBlur(img, (0, 0), sigmaX=sigma, sigmaY=sigma)


def exp_blur(img: np.ndarray, lam: float) -> np.ndarray:
    """Separable exponential blur with a 1-D exp(-|x|/lam) kernel.

    The 2-D response is exp(-(|x|+|y|)/lam) — a sharp central cusp with a long
    tail, the halation falloff. numpy/cv2 oracle twin of gpu.blur_frame_exp;
    must match its _exp_kernel (radius = 4*lam). Runs only on the no-GPU path,
    so cv2.sepFilter2D is fine.
    """
    if lam <= 0:
        return img.copy()
    radius = max(1, int(round(lam * 4)))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    k = np.exp(-np.abs(x) / lam).astype(np.float32)
    k /= k.sum()
    return cv2.sepFilter2D(img, -1, k, k)


def disc_blur(img: np.ndarray, radius: float) -> np.ndarray:
    """Disc (circle-of-confusion) blur: average within `radius` px.

    The defined-edge halation core — numpy/cv2 oracle twin of gpu.disc_blur.
    A disc is not separable, so this is a single 2D filter2D with a circular
    kernel. Runs only on the no-GPU path.
    """
    if radius <= 0:
        return img.copy()
    r = max(1, int(round(radius)))
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    k = (x * x + y * y <= r * r).astype(np.float32)
    k /= k.sum()
    return cv2.filter2D(img, -1, k)
