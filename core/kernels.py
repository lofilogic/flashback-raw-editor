"""
Numba JIT-compiled kernels for performance-critical image operations.

The trilinear LUT kernel uses an explicit lazy-compilation wrapper to defer
the expensive first-time compile (~30-60s) until first slider interaction,
keeping startup instant. All other kernels compile on first call and are
cached on disk thereafter.

When Numba is unavailable, all names are set to None so imports always
succeed — callers check HAS_NUMBA before use.
"""
import numpy as np
import time

# =============================================================================
# NUMBA AVAILABILITY
# =============================================================================

try:
    from numba import njit, prange
    HAS_NUMBA = True
    print("✓ Numba loaded successfully - JIT compilation enabled")
except ImportError:
    HAS_NUMBA = False
    print("⚠ Numba not available - using numpy fallbacks")

# =============================================================================
# TRILINEAR LUT (LAZY COMPILATION)
# =============================================================================

if HAS_NUMBA:
    def _trilinear_lut_numba_raw(img, lut_table, lut_size):
        """
        Numba-accelerated trilinear interpolation.
        Parallel across rows, manual loops for cache efficiency.
        """
        h, w = img.shape[:2]
        result = np.empty((h, w, 3), dtype=np.float32)

        for i in prange(h):
            for j in range(w):
                r = img[i, j, 0] * (lut_size - 1)
                g = img[i, j, 1] * (lut_size - 1)
                b = img[i, j, 2] * (lut_size - 1)

                r0 = int(r)
                g0 = int(g)
                b0 = int(b)

                if r0 >= lut_size - 1: r0 = lut_size - 2
                if g0 >= lut_size - 1: g0 = lut_size - 2
                if b0 >= lut_size - 1: b0 = lut_size - 2
                if r0 < 0: r0 = 0
                if g0 < 0: g0 = 0
                if b0 < 0: b0 = 0

                rf = r - r0
                gf = g - g0
                bf = b - b0

                for c in range(3):
                    c000 = lut_table[r0, g0, b0, c]
                    c001 = lut_table[r0, g0, b0 + 1, c]
                    c010 = lut_table[r0, g0 + 1, b0, c]
                    c011 = lut_table[r0, g0 + 1, b0 + 1, c]
                    c100 = lut_table[r0 + 1, g0, b0, c]
                    c101 = lut_table[r0 + 1, g0, b0 + 1, c]
                    c110 = lut_table[r0 + 1, g0 + 1, b0, c]
                    c111 = lut_table[r0 + 1, g0 + 1, b0 + 1, c]

                    c00 = c000 + (c001 - c000) * bf
                    c01 = c010 + (c011 - c010) * bf
                    c10 = c100 + (c101 - c100) * bf
                    c11 = c110 + (c111 - c110) * bf

                    c0 = c00 + (c01 - c00) * gf
                    c1 = c10 + (c11 - c10) * gf

                    result[i, j, c] = c0 + (c1 - c0) * rf

        return result

    _trilinear_lut_compiled = None

    def _trilinear_lut_numba(img, lut_table, lut_size):
        """
        Lazy compilation wrapper — compiles on first use.
        Defers the ~30-60s first-time compile to first slider interaction
        instead of blocking app startup.
        """
        global _trilinear_lut_compiled
        if _trilinear_lut_compiled is None:
            print("")
            print("=" * 60)
            print("[Numba] First-time compilation starting...")
            print("        This takes ~30-60 seconds (one time only)")
            print("        App will be instant on next launch!")
            print("=" * 60)
            print("")
            start = time.time()
            _trilinear_lut_compiled = njit(
                parallel=True,
                cache=True,
                fastmath=True
            )(_trilinear_lut_numba_raw)
            elapsed = time.time() - start
            print(f"[Numba] ✓ Compilation complete! ({elapsed:.1f} seconds)")
            print("")
        return _trilinear_lut_compiled(img, lut_table, lut_size)

# =============================================================================
# EAGER KERNELS (compiled on first call, cached after)
# =============================================================================

if HAS_NUMBA:
    @njit(parallel=True, cache=True, fastmath=True)
    def _apply_grain_numba(image, grain_layer, min_grain=0.2):
        """
        Fast grain blend using linear light formula.
        Includes highlight attenuation with a defined minimum floor.
        """
        h, w = image.shape[:2]
        result = np.empty((h, w, 3), dtype=np.float32)

        for i in prange(h):
            for j in range(w):
                for c in range(3):
                    img_val = image[i, j, c]
                    grain_val = grain_layer[i, j, c]
                    grain_delta = (2.0 * grain_val) - 1.0
                    highlight_falloff = min_grain + ((1.0 - img_val) * (1.0 - min_grain))
                    blended = img_val + (grain_delta * highlight_falloff)
                    result[i, j, c] = max(0.0, min(1.0, blended))

        return result

    @njit(parallel=True, cache=True, fastmath=True)  # No parallel=True: prange on small inner loops gives no benefit here
    def _screen_blend_numba(base, blend):
        """Fast screen blend: 1 - (1-base)*(1-blend)."""
        h, w = base.shape[:2]
        result = np.empty((h, w, 3), dtype=np.float32)

        for i in prange(h):
            for j in range(w):
                for c in range(3):
                    b = base[i, j, c]
                    bl = blend[i, j, c]
                    result[i, j, c] = 1.0 - (1.0 - b) * (1.0 - bl)

        return result

    @njit(parallel=True, cache=True, fastmath=True)
    def _unsharp_mask_numba(image, blurred, strength):
        """Fast unsharp mask: image + (image - blurred) * strength."""
        h, w = image.shape[:2]
        result = np.empty((h, w, 3), dtype=np.float32)

        for i in prange(h):
            for j in range(w):
                for c in range(3):
                    diff = image[i, j, c] - blurred[i, j, c]
                    result[i, j, c] = image[i, j, c] + diff * strength

        return result

    @njit(cache=True, fastmath=True)
    def _rotate_90_clockwise_numba(img):
        """
        Rotate image 90 degrees clockwise.
        Mapping: pixel at (row i, col j) → (row j, col h-1-i)
        """
        h, w = img.shape[:2]
        result = np.empty((w, h, 3), dtype=np.float32)

        for i in prange(h):
            for j in range(w):
                for c in range(3):
                    result[j, h - 1 - i, c] = img[i, j, c]

        return result

    @njit(cache=True, fastmath=True)
    def _rotate_90_counterclockwise_numba(img):
        """
        Rotate image 90 degrees counter-clockwise.
        Mapping: pixel at (row i, col j) → (row w-1-j, col i)
        """
        h, w = img.shape[:2]
        result = np.empty((w, h, 3), dtype=np.float32)

        for i in prange(h):
            for j in range(w):
                for c in range(3):
                    result[w - 1 - j, i, c] = img[i, j, c]

        return result

    @njit(parallel=True, cache=True, fastmath=True)
    def _numba_acescct_decode_core(flat_img, out):
        """JIT core for ACEScct to Linear. Bypasses Numba reshape bugs."""
        for i in prange(flat_img.size):
            val = flat_img[i]
            if val < 0.155251141552511:
                out[i] = (val - 0.0729055341958355) / 10.5402377416545
            else:
                out[i] = 2.0 ** (val * 17.52 - 9.72)

    @njit(parallel=True, cache=True, fastmath=True)
    def _numba_acescct_encode_core(flat_img, out):
        """JIT core for Linear to ACEScct. Bypasses Numba reshape bugs."""
        for i in prange(flat_img.size):
            val = flat_img[i]
            if val <= 0.0078125:
                out[i] = 10.5402377416545 * val + 0.0729055341958355
            else:
                safe_val = val if val > 1e-10 else 1e-10
                out[i] = (np.log2(safe_val) + 9.72) / 17.52

# =============================================================================
# None stubs so imports always succeed when Numba is unavailable
# =============================================================================

if not HAS_NUMBA:
    _trilinear_lut_compiled = None
    _trilinear_lut_numba = None
    _apply_grain_numba = None
    _screen_blend_numba = None
    _unsharp_mask_numba = None
    _rotate_90_clockwise_numba = None
    _rotate_90_counterclockwise_numba = None
    _numba_acescct_decode_core = None
    _numba_acescct_encode_core = None
